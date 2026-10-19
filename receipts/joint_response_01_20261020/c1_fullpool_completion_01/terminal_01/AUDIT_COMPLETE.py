"""Independent NumPy audit of the complete frozen full-parent comparison."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

started = time.monotonic()
root = Path('E:/GAN/research/N01')
out = Path(__file__).resolve().parent
new_run=out/'collected/run_01'
old_out=root/'receipts/joint_response_01_20261020/c1_fullpool_teacher_01/terminal_01'
old_run=old_out/'collected/run_01'
run=out/'combined_saved_view'
p = json.loads((root / 'configs/joint_response_01_c1_fullpool_completion_20261020.json').read_text())
remote = json.loads((out / 'REMOTE_TERMINAL_VERIFICATION.json').read_text())
manifest = json.loads((out.parent / 'SOURCE_MANIFEST.json').read_text())
result=json.loads((new_run/'RESULT.json').read_text())
complete=json.loads((new_run/'COMPLETE.json').read_text())
assert remote['native_exit']['exit_code']==0 and remote['processes']==[]
assert complete['result_sha256']==hashlib.sha256((new_run/'RESULT.json').read_bytes()).hexdigest()
assert remote['complete_present'] and not remote['failed_present'] and remote['result_present']
old_remote=json.loads((old_out/'REMOTE_TERMINAL_VERIFICATION.json').read_text())
old_review=json.loads((old_out/'TERMINAL_REVIEW.json').read_text())
for name,item in old_remote['outputs'].items():
    raw=(old_out/'collected'/name).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
# A local read-only assembled view selects new continuation outputs over original partial outputs.
run.mkdir()
selected={path.name:path for path in old_run.iterdir() if path.is_file()}
selected.update({path.name:path for path in new_run.iterdir() if path.is_file()})
for name,path in selected.items(): (run/name).hardlink_to(path)
for name in ('TEACHER_ROLES.npz','seed0_FEATURE_MAPS.npz','seed1_FEATURE_MAPS.npz'):
    a,b=np.load(old_run/name),np.load(new_run/name)
    assert a.files==b.files
    for key in a.files: np.testing.assert_array_equal(a[key],b[key])
old_trace=json.loads((old_run/'seed1_physical_TRACE.json').read_text())['iterations']
new_trace=json.loads((new_run/'seed1_physical_TRACE.json').read_text())['iterations']
assert new_trace[:7]==old_trace and len(new_trace)==12
partial=np.load(old_run/'seed1_physical_ITER07.npz');continued=np.load(new_run/'seed1_physical_ITER07.npz')
for key in partial.files:
    np.testing.assert_array_equal(partial[key],continued[key][:9] if key in ('FD_high','FD_low','FD_parameter_spans') else continued[key])
assert json.loads((new_run/'RESTART_REPLAY.json').read_text())['exact_array_checks']==37
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
new_counts = Counter()
track_reports, track_diagnosis = {}, {}
max_delta_error = max_risk_error = max_geometry_error = max_moment_error = max_FD_error = 0.
for track in ('seed0_joint','seed0_physical','seed1_joint','seed1_physical','population_high_h_fit_nominal','population_high_h_fit_high_h'):
    population = track.startswith('population_')
    seed = 0 if population else int(track[4])
    base_seed = 2150000 if population else 1950000 + seed * 100000
    physical = track.endswith('physical')
    dimension, iterations = (46 if physical else 54), (3 if population else 12)
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
        np.testing.assert_array_equal(a['bank_stream_base_seeds'], [base_seed*100000+i*1000+role*100 for role in range(4)])
        for field in ('fit_parents', 'target_parents', 'check_parents', 'check_target_parents'):
            assert np.array_equal(np.sort(a[field]), np.sort(pool))
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
            assert trust['denominator'] == 25860
            relative = a[f'trial{j}_relative_RMS']
            assert relative.shape == (77580,) and (relative >= 0).all()
            for key, value in [('mean_relative_RMS',relative.mean()),('p99_relative_RMS',np.quantile(relative,.99)),('max_relative_RMS',relative.max())]:
                np.testing.assert_allclose(value,trust[key],atol=2e-14,rtol=2e-12)
            trust_pass = (trust['mean_relative_RMS'] <= p['solver']['output_TR_mean_RMS_fraction']
                and trust['p99_relative_RMS'] <= p['solver']['output_TR_p99_RMS_fraction']
                and max(v for d in trust['flag_count_changes'].values() for v in d.values()) <=
                    max(4, 25860 * p['solver']['flag_increase_fraction']))
            assert trust_pass == trust['pass']
            new_risk = all_risks[j+1]
            gain = np.mean([new_risk[h]['joint_whitened'] for h in hypotheses]) / np.mean([old_risk[h]['joint_whitened'] for h in hypotheses])
            gain_pass = gain <= 1. - p['solver']['minimum_CHECK_gain']
            block_pass = all(new_risk[h][block] <= p['solver']['macro_degradation_factor'] * old_risk[h][block] + 2./25860
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
        if population or (track=='seed1_physical' and i>=7):
            new_counts.update(joint_direction_solves=1,finite_difference_columns=dimension,
                joint_candidate_trials=len(row['attempts']),teacher_accepted_joint_updates=int(row['accepted']))
        iteration_stats.append(dict(iteration=i, FIT_linear_risk_ratio=float(np.square(residual + J @ a['delta']).sum()/np.square(residual).sum()),
            CHECK_before=float(np.mean([old_risk[h]['joint_whitened'] for h in hypotheses])), attempts=attempts_stats,
            endpoint_boundary_coordinates=int(((endpoint == a['lower']) | (endpoint == a['upper'])).sum()),
            mean_step_norm=float(np.linalg.norm(endpoint[:24]-theta[:24])),
            shared_random_step_norm=float(np.linalg.norm(endpoint[24:45]-theta[24:45]))))
    assert np.array_equal(previous_endpoint, state['theta'])
    assert np.array_equal(source_coverage, state['source_coverage']) and np.array_equal(target_coverage, state['target_coverage'])
    assert source_coverage.sum() == target_coverage.sum() == iterations * 6465
    assert not source_coverage[heldout].any() and not target_coverage[heldout].any()
    if not population:
        assert (source_coverage > 0).sum() == (target_coverage > 0).sum() == 6465
        assert (source_coverage[pool] == iterations).all() and (target_coverage[pool] == iterations).all()
        assert result['tracks'][track]['accepted_updates']==accepted_count
    if population:
        profile=trace['response_population']
        assert result['population_stress'][profile]['accepted_updates']==accepted_count
        initial=np.load(run/f'{track}_ITER00.npz')['initial']
        np.testing.assert_allclose(np.linalg.norm(previous_endpoint-initial),result['population_stress'][profile]['parameter_shift_norm'],atol=1e-14,rtol=1e-12)
    track_reports[track] = dict(directions=iterations, accepted_updates=accepted_count,
        source_parents_visited=int((source_coverage > 0).sum()), target_parents_visited=int((target_coverage > 0).sum()),
        complete_parameter_chain_and_saved_matrix_solve=True)
    track_diagnosis[track] = dict(iterations=iteration_stats, rejection_reason_counts=dict(rejected_reasons),
        shared_random_frobenius=float(np.linalg.norm(previous_endpoint[24:45])),
        mean_norm=float(np.linalg.norm(previous_endpoint[:24])), electronics_coordinate=float(previous_endpoint[-1]))
assert dict(counts)==dict(joint_direction_solves=54,finite_difference_columns=2724,joint_candidate_trials=78,teacher_accepted_joint_updates=47)
new_counts['fullpool_FIT_chunk_columns']=new_counts['finite_difference_columns']*13
for key,value in new_counts.items(): assert result['execution_counts'][key]==value,key
for key,value in result['execution_counts'].items(): assert value<=p['operation_budget'][key],key
combined={}
for bound in ('lower','upper'):
    old=old_review['execution_counts_'+bound+'_bound']
    combined[bound]={key:old.get(key,0)+result['execution_counts'].get(key,0) for key in set(old)|set(result['execution_counts'])}
    assert combined[bound]==result['combined_execution_counts_'+bound+'_bound']
assert combined['lower']['finite_difference_columns']==2733 and combined['upper']['finite_difference_columns']==2741
assert result['new_direction_count']==11 and result['new_REAL_updates']==0
for field,file in [('recovery_decisions','DECISION'),('precision_runs','PRECISION')]:
    assert result[field]['0']==json.loads((old_run/f'seed0_{file}.json').read_text())

hard_gates, precision_gates, precision_reports = [], [], {}
for seed in range(2):
    evaluation = json.loads((run / f'seed{seed}_EVALUATION.json').read_text())
    decision = json.loads((run / f'seed{seed}_DECISION.json').read_text())
    reported = json.loads((run / f'seed{seed}_PRECISION.json').read_text())
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
    hard_gates.extend(dict(seed=seed, **g) for g in reconstructed_hard)
    moments, calculated = {}, {}
    for library in range(2):
        a = np.load(run / f'seed{seed}_library{library}_PRECISION.npz', allow_pickle=False)
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
            precision_reports[f'seed{seed}/library{library}/{label}'] = values
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
    precision_gates.extend(reconstructed_precision)
assert len(hard_gates)==36 and len(precision_gates)==18
assert sum(not g['pass_gate'] for g in hard_gates)==0
assert sum(not g['pass_gate'] for g in precision_gates)==2
assert not result['coupled_teacher_pass_all'] and not complete['coupled_teacher_pass_all']
report=dict(status='PASS_COMPLETE_SAVED_EVIDENCE_AUDIT_SCIENTIFIC_GATES_FAILED',UTC=datetime.now(timezone.utc).isoformat(),
    config_sha256=manifest['config_sha256'],source_files_verified_remote=len(manifest['files']),
    source_files_independently_verified_local=len(local_sources),remote_only_source_paths=remote_only_sources,
    source_pins_verified_against_manifest=len(p['source_pins']),output_files_verified=len(remote['outputs']),
    output_bytes=sum(x['bytes'] for x in remote['outputs'].values()),original_output_files_reverified=len(old_remote['outputs']),
    native_exit=remote['native_exit'],result_sha256=complete['result_sha256'],tracks=track_reports,
    original_and_completed_unique_counts=dict(counts),new_job_counts=result['execution_counts'],
    combined_execution_counts_lower_bound=combined['lower'],combined_execution_counts_upper_bound=combined['upper'],
    original_unpersisted_FD_columns_min=0,original_unpersisted_FD_columns_max=8,
    partial_direction_exact_replay_checks=37,old_trace_prefix_unchanged=True,feature_factor_array_replays=18,
    hard_observation_gates=hard_gates,conditional_precision_gates=precision_gates,
    hard_observation_gate_pass_count=36,conditional_precision_gate_pass_count=16,conditional_precision_gate_fail_count=2,
    maximum_absolute_reconstruction_difference=dict(FD=max_FD_error,joint_delta=max_delta_error,risk=max_risk_error,geometry=max_geometry_error,moment=max_moment_error),
    numerical_audit_tolerances=dict(FD_atol=1e-14,FD_rtol=1e-12,solve_atol=2e-11,solve_rtol=2e-9,risk_atol=1e-14,risk_rtol=1e-12,moments_atol=2e-14,moments_rtol=2e-12),
    numerical_tolerances_do_not_change_scientific_gates=True,
    audit_boundary='Reconstructed saved FD pairs, 54 complete unique matrix solves, 78 CHECK proposals and full relative-RMS trust vectors, 47 updates, parameter chains, full-parent coverage and both heads saved 2048-draw moments. Flag counts and final native gates use saved summaries; population parameter drift is reconstructed but conditional population reports are not raw-waveform replay. No generator or hard waveform was regenerated. Original missing cost remains an explicit interval.',
    scientific_acceptance=False,C1_REAL_use_supported=False,original_comparison_complete=True,
    new_model_calls=0,new_parameter_updates=0,elapsed_seconds=time.monotonic()-started)
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(out/'SAVED_FAILURE_DIAGNOSIS.json').write_text(json.dumps(dict(tracks=track_diagnosis,precision=precision_reports,population_stress=result['population_stress'],
    interpretation='The full-parent fit passes seed1 gates but seed0 Q variance still fails both precision banks. Late accepted steps alone do not distinguish sampling noise, local nonlinearity or identifiable training directions.',new_model_calls=0,new_parameter_updates=0),indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('hard_observation_gates','conditional_precision_gates','remote_only_source_paths','combined_execution_counts_lower_bound','combined_execution_counts_upper_bound')},indent=2))
print('failed_gates', [g for g in precision_gates if not g['pass_gate']])
