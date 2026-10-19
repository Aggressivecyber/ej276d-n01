"""Independent saved-array audit of bounded 32-draw calibration and dual target reports."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch

started=time.monotonic()
root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
out=here/'terminal_01'
run=out/'collected/run_01'
p=json.loads((root/'configs/joint_response_01_c1_precision_pilot_20261020.json').read_bytes())
remote=json.loads((out/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
manifest=json.loads((out.parent/'SOURCE_MANIFEST.json').read_bytes())
result=json.loads((run/'RESULT.json').read_bytes())
complete=json.loads((run/'COMPLETE.json').read_bytes())
assert remote['native_exit']['exit_code']==0 and remote['processes']==[]
assert remote['complete_present'] and remote['result_present'] and not remote['failed_present']
assert complete['result_sha256']==hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest()
assert result['config_sha256']==hashlib.sha256((root/'configs/joint_response_01_c1_precision_pilot_20261020.json').read_bytes()).hexdigest()
raw=(out/'terminal.tar').read_bytes()
assert len(raw)==remote['archive_bytes'] and hashlib.sha256(raw).hexdigest()==remote['archive_sha256']
assert result['elapsed_seconds']<p['resources']['internal_seconds']
assert remote['native_exit']['peak_rss_bytes']<p['resources']['RSS_GiB']*1024**3
assert result['peak_GPU_allocated_bytes']<p['resources']['GPU_GiB']*1024**3
old_run=root/'receipts/joint_response_01_20261020/c1_fullpool_teacher_01/terminal_01/collected/run_01'
for filename in ('TEACHER_ROLES.npz','seed0_FEATURE_MAPS.npz','seed1_FEATURE_MAPS.npz'):
    a,b=np.load(run/filename),np.load(old_run/filename)
    assert a.files==b.files
    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
for seed in range(2):
    assert json.loads((run/f'seed{seed}_FEATURE_REPLAY.json').read_bytes())['pass_all']
local_sources, remote_only_sources = [], []
for name, item in manifest['files'].items():
    if (root / name).is_file():
        raw = (root / name).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], name
        local_sources.append(name)
    else:
        remote_only_sources.append(name)
assert remote['source_files_verified'] == len(manifest['files'])
for name, digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256'] == digest, name
for name, item in remote['outputs'].items():
    raw = (out / 'collected' / name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], name
roles = np.load(run / 'TEACHER_ROLES.npz', allow_pickle=False)
pool, heldout = roles['train_rows'], roles['heldout_rows']
assert len(pool) == 6465 and len(heldout) == 128
assert np.array_equal(np.sort(np.r_[pool, heldout]), np.arange(6593))
hypotheses = ('fast', 'slow', 'common_level')
counts = Counter()
track_reports, track_diagnosis = {}, {}
max_delta_error = max_risk_error = max_geometry_error = max_moment_error = max_FD_error = 0.
for track in ('seed0_joint','seed1_joint'):
    seed = int(track[4])
    dimension, iterations = 54, 4
    initial_endpoint=np.load(root/'receipts/joint_response_01_20261020/c1_matched_control_01/terminal_01/collected/run_01'/f'seed{seed}_joint_ITER11.npz',allow_pickle=False)['endpoint']
    trace = json.loads((run / (track + '_TRACE.json')).read_text())
    state = np.load(run / (track + '_STATE.npz'), allow_pickle=False)
    factors = np.load(run / f'seed{seed}_FEATURE_MAPS.npz', allow_pickle=False)
    slices = json.loads((run / f'seed{seed}_FEATURE_SLICES.json').read_text())
    assert not trace['pairing_used_for_fit'] and not trace['conditional_teacher_labels_used_for_fit']
    assert len(trace['iterations']) == iterations
    source_coverage, target_coverage = np.zeros(6593, dtype=int), np.zeros(6593, dtype=int)
    accepted_count, rejected_reasons, iteration_stats = 0, Counter(), []
    previous_endpoint = None
    for i, row in enumerate(trace['iterations']):
        a = np.load(run / f'{track}_ITER{i:02d}.npz', allow_pickle=False)
        theta, J, residual = a['theta'], a['jacobian'], a['residual']
        assert J.shape == (816, dimension) and theta.shape == (dimension,)
        assert all(np.isfinite(a[key]).all() for key in a.files)
        assert np.array_equal(theta, a['initial'] if i == 0 else previous_endpoint)
        assert np.all(theta >= a['lower']) and np.all(theta <= a['upper'])
        np.testing.assert_array_equal(a['bank_stream_base_seeds'], [p['bank_seed']+1000000*seed+10000*i+1000*role for role in range(4)])
        for role_index, field in enumerate(('fit_parents', 'target_parents', 'check_parents', 'check_target_parents')):
            assert np.array_equal(np.sort(a[field]), np.sort(pool))
            permutation = torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['order_seed']+1000*seed+100*i+role_index)).numpy()
            np.testing.assert_array_equal(a[field],pool[permutation])
        np.testing.assert_array_equal(a['initial'],initial_endpoint)
        span = np.minimum(a['upper'],theta+p['solver']['finite_difference_step']) - np.maximum(a['lower'],theta-p['solver']['finite_difference_step'])
        np.testing.assert_array_equal(span, a['FD_parameter_spans'])
        reconstructed_J = ((a['FD_high']-a['FD_low'])/span[:,None]).T
        max_FD_error = max(max_FD_error, float(np.max(np.abs(reconstructed_J-J))))
        np.testing.assert_allclose(reconstructed_J,J,atol=1e-14,rtol=1e-12)
        np.add.at(source_coverage, a['fit_parents'], 1)
        np.add.at(target_coverage, a['target_parents'], 1)
        gram = J.T @ J
        ridge = p['solver']['ridge_fraction'] * np.diag(gram).mean()
        unbounded = np.linalg.solve(gram + ridge * np.eye(dimension), -(J.T @ residual))
        delta = unbounded * min(1., p['solver']['radius'] / np.linalg.norm(unbounded))
        max_delta_error = max(max_delta_error, float(np.max(np.abs(delta - a['delta']))))
        np.testing.assert_allclose(delta, a['delta'], atol=2e-11, rtol=2e-9)
        singular = np.linalg.svd(J, compute_uv=False)
        np.testing.assert_allclose(singular, row['geometry']['singular_values'], atol=2e-11, rtol=2e-9)
        geometry = dict(ridge=ridge, proposed_step_norm=np.linalg.norm(delta),
            off_diagonal_gram_fraction=np.linalg.norm(gram - np.diag(np.diag(gram))) / np.linalg.norm(gram))
        for key, value in geometry.items():
            max_geometry_error = max(max_geometry_error, abs(value - row['geometry'][key]))
            np.testing.assert_allclose(value, row['geometry'][key], atol=2e-11, rtol=2e-9)
        assert row['geometry']['effective_rank_at_1pct'] == int((singular > .01 * singular[0]).sum())
        assert row['geometry']['full_cross_module_gram_used']
        all_risks = []
        for index, record in enumerate([row['old_CHECK_risks']] + [t['risks'] for t in row['attempts']]):
            whitened = a['old_CHECK_residual' if index == 0 else f'trial{index-1}_CHECK_residual'].reshape(3, 272) * np.sqrt(3.)
            reconstructed = {}
            for h, white in zip(hypotheses, whitened, strict=True):
                factor = factors['CHECK_' + h + '_cholesky']
                raw = factor @ white
                variance = np.square(factor).sum(1)
                value = dict(joint_whitened=float(np.square(white).mean()))
                for block, (left, right) in slices.items():
                    value[block] = float((np.square(raw[left:right]) / variance[left:right]).mean())
                for key in value:
                    max_risk_error = max(max_risk_error, abs(value[key] - record[h][key]))
                    np.testing.assert_allclose(value[key], record[h][key], atol=1e-14, rtol=1e-12)
                reconstructed[h] = value
            all_risks.append(reconstructed)
        endpoint = theta.copy()
        old_risk = all_risks[0]
        attempts_stats = []
        for j, trial in enumerate(row['attempts']):
            fraction = p['solver']['backtracking_fractions'][j]
            assert trial['fraction'] == fraction
            proposal = np.clip(theta + fraction * a['delta'], a['lower'], a['upper'])
            assert np.array_equal(proposal, a[f'trial{j}_proposal'])
            trust = trial['trust']
            assert trust['denominator'] == 206880
            relative = a[f'trial{j}_relative_RMS']
            assert relative.shape == (620640,) and (relative >= 0).all()
            for key, value in [('mean_relative_RMS',relative.mean()),('p99_relative_RMS',np.quantile(relative,.99)),('max_relative_RMS',relative.max())]:
                np.testing.assert_allclose(value,trust[key],atol=2e-14,rtol=2e-12)
            trust_pass = (trust['mean_relative_RMS'] <= p['solver']['output_TR_mean_RMS_fraction']
                and trust['p99_relative_RMS'] <= p['solver']['output_TR_p99_RMS_fraction']
                and max(v for d in trust['flag_count_changes'].values() for v in d.values()) <=
                    max(4, 206880 * p['solver']['flag_increase_fraction']))
            assert trust_pass == trust['pass']
            new_risk = all_risks[j+1]
            gain = np.mean([new_risk[h]['joint_whitened'] for h in hypotheses]) / np.mean([old_risk[h]['joint_whitened'] for h in hypotheses])
            gain_pass = gain <= 1. - p['solver']['minimum_CHECK_gain']
            block_pass = all(new_risk[h][block] <= p['solver']['macro_degradation_factor'] * old_risk[h][block] + 2./206880
                for h in hypotheses for block in ('wave', 'charge_morphology', 'peak_codes', 'baseline'))
            accepted = bool(trust_pass and gain_pass and block_pass)
            assert accepted == trial['accepted']
            if accepted:
                endpoint = proposal
                assert j == len(row['attempts']) - 1
            else:
                rejected_reasons.update(k for k, v in [('output_trust', trust_pass), ('joint_gain', gain_pass), ('macro_block', block_pass)] if not v)
            attempts_stats.append(dict(fraction=fraction, CHECK_joint_ratio=float(gain), accepted=accepted,
                joint_gain_pass=bool(gain_pass), macro_pass=bool(block_pass), trust_pass=bool(trust_pass)))
        assert np.array_equal(endpoint, a['endpoint'])
        assert row['accepted'] == row['attempts'][-1]['accepted']
        np.testing.assert_allclose(np.linalg.norm(endpoint), row['parameter_norm'], atol=1e-14, rtol=1e-12)
        accepted_count += int(row['accepted'])
        previous_endpoint = endpoint
        counts.update(joint_direction_solves=1, finite_difference_columns=dimension,
            joint_candidate_trials=len(row['attempts']), teacher_accepted_joint_updates=int(row['accepted']))
        iteration_stats.append(dict(iteration=i, FIT_linear_risk_ratio=float(np.square(residual + J @ a['delta']).sum()/np.square(residual).sum()),
            CHECK_before=float(np.mean([old_risk[h]['joint_whitened'] for h in hypotheses])), attempts=attempts_stats,
            endpoint_boundary_coordinates=int(((endpoint == a['lower']) | (endpoint == a['upper'])).sum()),
            mean_step_norm=float(np.linalg.norm(endpoint[:24]-theta[:24])),
            shared_random_step_norm=float(np.linalg.norm(endpoint[24:45]-theta[24:45]))))
    assert np.array_equal(previous_endpoint, state['theta'])
    assert np.array_equal(source_coverage, state['source_coverage']) and np.array_equal(target_coverage, state['target_coverage'])
    assert source_coverage.sum() == target_coverage.sum() == iterations * 6465
    assert not source_coverage[heldout].any() and not target_coverage[heldout].any()
    assert (source_coverage > 0).sum() == (target_coverage > 0).sum() == 6465
    assert (source_coverage[pool] == iterations).all() and (target_coverage[pool] == iterations).all()
    assert result['tracks'][track]['accepted_updates']==accepted_count
    track_reports[track] = dict(directions=iterations, accepted_updates=accepted_count,
        source_parents_visited=int((source_coverage > 0).sum()), target_parents_visited=int((target_coverage > 0).sum()),
        complete_parameter_chain_and_saved_matrix_solve=True)
    track_diagnosis[track] = dict(iterations=iteration_stats, rejection_reason_counts=dict(rejected_reasons),
        shared_random_frobenius=float(np.linalg.norm(previous_endpoint[24:45])),
        mean_norm=float(np.linalg.norm(previous_endpoint[:24])), electronics_coordinate=float(previous_endpoint[-1]))

assert counts['joint_direction_solves']==8 and counts['finite_difference_columns']==432
assert 8<=counts['joint_candidate_trials']<=24 and 0<=counts['teacher_accepted_joint_updates']<=8
expected=dict(p['operation_budget'])
unused_trials=expected['joint_candidate_trials']-counts['joint_candidate_trials']
expected.update(joint_candidate_trials=counts['joint_candidate_trials'],teacher_accepted_joint_updates=counts['teacher_accepted_joint_updates'])
for key,per_trial in (('joint_response_calls',102),('joint_response_rows',206880),('generated_observation_calls',306),('generated_observation_rows',620640)):
    expected[key]-=unused_trials*per_trial
for key,value in expected.items():assert result['execution_counts'].get(key,0)==value,key
assert set(result['execution_counts'])<=set(expected)
assert not result['scientific_acceptance'] and not complete['scientific_acceptance']
assert result['REAL_response_updates']==0 and not result['automatic_REAL_promotion']
assert result['original_cross_family_failure_preserved'] and result['positive_control_not_stage3_replacement']
assert not result['known_parameters_supplied_to_student'] and not result['report_parents_used_for_fitting'] and not result['paired_labels_used_for_fitting']
exact_dual_target_arrays=0
for seed in range(2):
    for library in range(2):
        a=np.load(run/'matched_C1'/f'seed{seed}_library{library}_PRECISION.npz')
        b=np.load(run/'original'/f'seed{seed}_library{library}_PRECISION.npz')
        for label in ('M0','candidate'):
            for metric in ('mean','covariance','Q_variance','wave_variance'):
                np.testing.assert_array_equal(a[label+'_'+metric],b[label+'_'+metric])
                exact_dual_target_arrays+=1
assert exact_dual_target_arrays==32
hard_gates, precision_gates, precision_reports = [], [], {}
for family in ('matched_C1','original'):
    for seed in range(2):
        evaluation = json.loads((run / family / f'seed{seed}_EVALUATION.json').read_text())
        decision = json.loads((run / family / f'seed{seed}_DECISION.json').read_text())
        assert decision == result['decisions'][str(seed)][family]
        reported = json.loads((run / family / f'seed{seed}_PRECISION.json').read_text())
        reconstructed_hard = []
        for library in range(2):
            baseline, candidate, oracle = (evaluation[str(library)][name] for name in ('M0', 'joint', 'teacher_oracle'))
            for h in hypotheses:
                for metric in ('joint_risk', 'native_wave', 'native_QPSD'):
                    values = [v['risks'][h]['joint_whitened'] if metric == 'joint_risk' else v['native'][h]['values'][metric[7:]]
                              for v in (baseline, candidate, oracle)]
                    b, c, t = values
                    limit = t + p['success']['remaining_excess_risk_fraction'] * max(b-t, 0.) if metric == 'joint_risk' else max(1.02*b, 1.20*t)
                    reconstructed_hard.append(dict(name=f'library{library}/{h}/{metric}', pass_gate=c <= limit, baseline=b, candidate=c, oracle=t))
        assert reconstructed_hard == decision['hard_observation_gates']
        hard_gates.extend(dict(target_family=family, seed=seed, **g) for g in reconstructed_hard)
        moments, calculated = {}, {}
        for library in range(2):
            a = np.load(run / family / f'seed{seed}_library{library}_PRECISION.npz', allow_pickle=False)
            assert np.array_equal(a['heldout_rows'], heldout)
            for label in ('teacher', 'M0', 'candidate'):
                moments[library, label] = tuple(a[label+'_'+name] for name in ('mean', 'covariance', 'Q_variance', 'wave_variance'))
                assert [x.shape for x in moments[library, label]] == [(128,384), (128,16,16), (128,), (128,)]
                assert all(np.isfinite(x).all() for x in moments[library, label])
            for label in ('M0', 'candidate'):
                mean, cov, qvar, wvar = moments[library, label]
                tm, tc, tq, tw = moments[library, 'teacher']
                sq = np.square(cov-tc).sum((1,2)); top = np.argsort(sq)[-13:][::-1]
                scalar = float((cov*tc).sum()/np.square(cov).sum())
                values = dict(conditional_mean_relative_RMS=float(np.linalg.norm(mean-tm)/np.linalg.norm(tm)),
                    conditional_covariance_relative_Frobenius=float(np.linalg.norm(cov-tc)/np.linalg.norm(tc)),
                    covariance_trace_ratio=float(np.trace(cov,axis1=1,axis2=2).sum()/np.trace(tc,axis1=1,axis2=2).sum()),
                    covariance_cosine=float((cov*tc).sum()/(np.linalg.norm(cov)*np.linalg.norm(tc))),
                    diagnostic_best_scalar=scalar, diagnostic_after_scalar_relative_error=float(np.linalg.norm(scalar*cov-tc)/np.linalg.norm(tc)),
                    Q_variance_ratio=float(qvar.mean()/tq.mean()), wave_variance_ratio=float(wvar.mean()/tw.mean()),
                    top13_error_fraction=float(sq[top].sum()/sq.sum()),
                    projection_diagonal_residual=np.diagonal(cov-tc,axis1=1,axis2=2).mean(0).tolist())
                reference = reported['reports'][f'seed{seed}/library{library}'][label]
                assert top.tolist() == reference['top13_of128_parent_positions']
                for key, value in values.items():
                    max_moment_error = max(max_moment_error, float(np.max(np.abs(np.asarray(value)-reference[key]))))
                    np.testing.assert_allclose(value, reference[key], atol=2e-14, rtol=2e-12)
                calculated[library, label] = values
                precision_reports[f'{family}/seed{seed}/library{library}/{label}'] = values
        reconstructed_precision = []
        for library in range(2):
            base, fitted = calculated[library, 'M0'], calculated[library, 'candidate']
            for metric, absolute, fraction in [('conditional_mean_relative_RMS','mean_absolute_relative_RMS','mean_error_fraction'),
                                              ('conditional_covariance_relative_Frobenius','covariance_absolute_relative_error','covariance_error_fraction')]:
                limit = min(p['success'][absolute], p['success'][fraction] * base[metric])
                reconstructed_precision.append(dict(name=f'seed{seed}/library{library}/{metric}',value=fitted[metric],upper=limit,pass_gate=fitted[metric]<=limit))
            ratio = fitted['Q_variance_ratio']; lower, upper = p['success']['variance_ratio_low'], p['success']['variance_ratio_high']
            reconstructed_precision.append(dict(name=f'seed{seed}/library{library}/Q_variance_ratio',value=ratio,lower=lower,upper=upper,pass_gate=lower<=ratio<=upper))
        for label in ('teacher','M0','candidate'):
            a,b = moments[0,label][1], moments[1,label][1]
            value = float(np.linalg.norm(a-b)/(.5*(np.linalg.norm(a)+np.linalg.norm(b))))
            limit = p['success']['MC_covariance_difference_max']
            reconstructed_precision.append(dict(name=f'seed{seed}/{label}/independent_bank_MC_difference',value=value,upper=limit,pass_gate=value<=limit))
        for calculated_gate, saved_gate in zip(reconstructed_precision, decision['conditional_precision_gates'], strict=True):
            assert calculated_gate.keys() == saved_gate.keys()
            for key,value in calculated_gate.items():
                if isinstance(value,float): np.testing.assert_allclose(value,saved_gate[key],atol=2e-14,rtol=2e-12)
                else: assert value == saved_gate[key]
        assert decision['pass_all'] == all(x['pass_gate'] for x in reconstructed_hard+reconstructed_precision)
        precision_gates.extend(dict(target_family=family,**g) for g in reconstructed_precision)

assert len(hard_gates)==72 and len(precision_gates)==36
assert len(manifest['files'])==780 and len(p['source_pins'])==120
assert sum(v['bytes'] for name,v in remote['outputs'].items() if name.startswith('run_01/'))<=p['resources']['output_MiB']*1024**2
training=[p['bank_seed']+1000000*h+10000*i+1000*r+3*c+o for h in range(2) for i in range(4) for r in range(4) for c in range(102) for o in range(3)]
evaluation_streams=[p['evaluation_bank_seed']+1000000*h+10000*l+1000*r+o for h in range(2) for l in range(2) for r in range(2) for o in range(3)]
assert len(training)==len(set(training))==9792 and len(set(training+evaluation_streams+sum(p['precision_bank_seeds'],[])))==9820
for seed in range(2):
    for iteration in range(4):
        memory=json.loads((run/f'seed{seed}_joint_MEMORY_ITER{iteration:02d}.json').read_bytes())
        assert memory['peak_RSS_bytes']<=12*1024**3 and memory['full_banks_live']==0 and memory['allocator_trim_requested']
matched_pass=all(g['pass_gate'] for g in hard_gates+precision_gates if g['target_family']=='matched_C1')
original_pass=all(g['pass_gate'] for g in hard_gates+precision_gates if g['target_family']=='original')
report=dict(status='PASS_SAVED_PRECISION_PILOT_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    config_sha256=manifest['config_sha256'],source_files_verified_remote=len(manifest['files']),source_files_rehashed_local=len(local_sources),
    remote_only_source_paths=remote_only_sources,source_pins_verified=len(p['source_pins']),output_files_verified=len(remote['outputs']),
    output_bytes=sum(v['bytes'] for v in remote['outputs'].values()),archive_verified=True,native_exit=remote['native_exit'],
    result_sha256=complete['result_sha256'],tracks=track_reports,verified_direction_counts=dict(counts),execution_counts=result['execution_counts'],
    feature_factor_replays=18,parent_permutation_replays=32,training_substreams_unique=9792,evaluation_substreams_unique=24,precision_streams_unique=4,matched_recovery_pass=matched_pass,original_target_report_pass=original_pass,dual_target_moment_array_equalities=32,
    hard_observation_gates=hard_gates,conditional_precision_gates=precision_gates,
    by_target={family:dict(hard_pass=sum(g['pass_gate'] for g in hard_gates if g['target_family']==family),hard_total=36,
        precision_pass=sum(g['pass_gate'] for g in precision_gates if g['target_family']==family),precision_total=18,
        failed_gates=[g for g in hard_gates+precision_gates if g['target_family']==family and not g['pass_gate']]) for family in ('matched_C1','original')},
    maximum_absolute_reconstruction_difference=dict(FD=max_FD_error,joint_delta=max_delta_error,risk=max_risk_error,geometry=max_geometry_error,moment=max_moment_error),
    audit_boundary='Independent saved FD pairs, full coupled solves, all actual candidate risks and 620640-entry trust vectors, all actual accepted parameter updates, full-parent coverage/permutations, and eight 2048-draw moment files. Flag counts and final native gates use saved summaries. No generator or raw hard waveform replay.',
    numerical_tolerances_do_not_change_scientific_gates=True,scientific_acceptance=False,automatic_REAL_promotion=False,
    original_cross_family_failure_preserved=True,new_model_calls=0,new_parameter_updates=0,elapsed_seconds=time.monotonic()-started)
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
(out/'SAVED_FIT_DIAGNOSIS.json').write_text(json.dumps(dict(tracks=track_diagnosis,precision=precision_reports,
    interpretation='Both target families are reported under unchanged gates. This positive-control pilot does not replace the original cross-family fitted recovery or automatically promote REAL calibration.',new_model_calls=0,new_parameter_updates=0),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:report[k] for k in ('status','UTC','source_files_rehashed_local','output_files_verified','output_bytes','verified_direction_counts','by_target','maximum_absolute_reconstruction_difference','elapsed_seconds')},indent=2))
