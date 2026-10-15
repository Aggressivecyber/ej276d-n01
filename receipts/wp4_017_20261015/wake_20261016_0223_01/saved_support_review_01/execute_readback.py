import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

out = Path(__file__).resolve().parent
n01 = Path('E:/GAN/research/N01')
selected = n01 / 'runs/wp4_017_20261015/selected_collected_01'
inventory_path = out.parent / 'collection_01/REMOTE_FULL_INVENTORY.json'
inventory = json.loads(inventory_path.read_text(encoding='utf-8'))['whole_remote_inventory']
hypotheses = ('fast', 'slow', 'common_level')
array_paths = ['runs/primary/artifacts/cache/PARENT_SPLIT.npz', 'runs/primary/artifacts/cache/Q.npy']
array_paths += [f'runs/primary/artifacts/T1/DERIVED64_{h}.npz' for h in hypotheses]
for seed in range(2):
    trial = f'runs/primary/artifacts/T1_paired_seed{seed}'
    array_paths += [f'{trial}/COMPARISON.npz']
    array_paths += [f'{trial}/evaluation/{kind}_{h}.npz' for h in hypotheses for kind in ('DERIVED64', 'PARENT_COMPARISON')]
report_paths = ['runs/primary/artifacts/T1/EVALUATION.json']
report_paths += [f'runs/primary/artifacts/T1_paired_seed{seed}/{name}' for seed in range(2) for name in ('COMPARISON.json', 'evaluation/EVALUATION.json')]
reports = {}
local_verified = {}
for rel in report_paths:
    path = selected / rel
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == inventory[rel]['bytes'] and digest == inventory[rel]['sha256'], rel
    report = json.loads(raw)
    local_verified[rel] = dict(bytes=len(raw), sha256=digest)
    if rel.endswith('/COMPARISON.json'):
        report = dict(trial=report['trial'], Q_group_cuts=report['Q_group_cuts'],
                      parameter_recovery_secondary=report['parameter_recovery_secondary'],
                      all_required_cells_pass=report['all_required_cells_pass'],
                      producer_status_counts={status: sum(c['status'] == status for c in report['cells']) for status in ('pass', 'pending', 'fail')},
                      target_cells=[dict(c, cell_index=i) for i, c in enumerate(report['cells']) if c['status'] == 'pending' or c['metric'] == 'analytic_charge_variance'])
    else:
        report = {k: report[k] for k in ('parents', 'banks', 'draws', 'a_Vns', 'b', 'teacher', 'student')}
    reports[rel] = report
payload = dict(root='/home/rylan/projects/ej276d-n01/wp4_017_20261015',
               expected={rel: inventory[rel] for rel in array_paths + report_paths},
               array_paths=array_paths, reports=reports, local_verified_reports=local_verified,
               inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
               source_commit='f7f720989584187adeb6fa7613a8ed287206ad08',
               config_sha256='deb7b08a4533421139c8c021608c56d3f06ec033d111f99da62cd4aba484d723')

remote = '''import base64, hashlib, json, os, resource, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2, 1536 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (170, 175))
import numpy as np
started = time.monotonic()
PAYLOAD_LINE
root = Path(payload['root'])
verified = {}
for rel, expected in payload['expected'].items():
    path = root / rel
    sha = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024**2):
            sha.update(block)
    digest = sha.hexdigest()
    assert path.stat().st_size == expected['bytes'] and digest == expected['sha256'], rel
    verified[rel] = dict(bytes=path.stat().st_size, sha256=digest)

features = ('Q', 'tail', 'PSD_raw', 'baseline_mean', 'baseline_RMS', 'peak_voltage', 'rise10_90', 'FWHM', 'peak_index')
coverage_features = (0, 1, 5, 8, 6, 7)
hypotheses = ('fast', 'slow', 'common_level')
with np.load(root / 'runs/primary/artifacts/cache/PARENT_SPLIT.npz', allow_pickle=False) as ar:
    fit = ar['fit_rows']; test = ar['test_rows']; ids = ar['source_ids']
Q = np.load(root / 'runs/primary/artifacts/cache/Q.npy', allow_pickle=False, mmap_mode='r')
assert np.isfinite(Q).all() and (Q > 0).all()
cuts = np.quantile(Q[fit], [1/3, 2/3], method='linear')
label = np.searchsorted(cuts, Q[test], side='right')
group_expected = {'all_heldout': np.ones(len(test), bool), **{f'fit_Q_tertile{k}_heldout': label == k for k in range(3)}}
rank_order = np.lexsort((ids[test].astype(str), Q[test]))
rank = np.empty(len(test), np.int64); rank[rank_order] = np.arange(len(test))
u = 2 * (rank + .5) / len(rank) - 1
weight_expected = {'nominal': np.ones(len(test)), 'low': np.exp(-np.log(2) * u), 'high': np.exp(np.log(2) * u)}
low_group = group_expected['fit_Q_tertile0_heldout']
T = {}
member_receipts = {}
for h in hypotheses:
    rel = f'runs/primary/artifacts/T1/DERIVED64_{h}.npz'
    with np.load(root / rel, allow_pickle=False) as ar:
        members = ['feature_support', 'pair_support', 'scalar_defined_draws', 'scalar_jointly_defined_draws', 'full_slot_undefined_fraction']
        members += [k for k in ar.files if k.startswith('fraction_flag__rise_') or k == 'fraction_flag__temporal_nonpositive_peak']
        T[h] = {k: ar[k] for k in members}
    member_receipts[rel] = members

result = dict(schema='n01/wp4-017-saved-support-numeric-readback/v1',
              job_id='N01_WP4_017_SUPPORT_ARRAY_READBACK_20261016',
              source_commit=payload['source_commit'], config_sha256=payload['config_sha256'],
              inventory_sha256=payload['inventory_sha256'], verified_files=verified,
              local_verified_reports=payload['local_verified_reports'],
              physical_parents=len(Q), fit_parents=len(fit), heldout_parents=len(test),
              original_low_Q_heldout_parents=int(low_group.sum()),
              actual_Q_all_original_parents=dict(min=float(Q.min()), max=float(Q.max()), finite_positive=int(np.sum(np.isfinite(Q) & (Q > 0)))),
              group_cuts=cuts.tolist(), group_original_counts={k: int(g.sum()) for k, g in group_expected.items()},
              scope=dict(new_G=0, new_AD=0, new_NN=0, new_random_draws=0, new_probes=0, original_SOURCE_REAL_DEV_CP_decodes=0,
                         hard_wave_members_decoded=0, energy_or_PIT_recomputed=False, per_native_bank_scalar_support_recomputed=False,
                         saved_derived_masks_and_saved_finite_counts_verified=True, original_run_mutated=False),
              trial_results=[])
teacher_eval = payload['reports']['runs/primary/artifacts/T1/EVALUATION.json']
ta = .01 * 3.7103998046755806; tb = .0025
assert teacher_eval['a_Vns'] == ta and teacher_eval['b'] == tb

for seed in range(2):
    trial_rel = f'runs/primary/artifacts/T1_paired_seed{seed}'
    report = payload['reports'][f'{trial_rel}/COMPARISON.json']
    evaluation = payload['reports'][f'{trial_rel}/evaluation/EVALUATION.json']
    a = report['parameter_recovery_secondary']['a']; b = report['parameter_recovery_secondary']['b']
    assert a == evaluation['a_Vns'] and b == evaluation['b']
    with np.load(root / f'{trial_rel}/COMPARISON.npz', allow_pickle=False) as cmp:
        np.testing.assert_array_equal(cuts, cmp[report['Q_group_cuts']['array']])
        np.testing.assert_array_equal(cmp['root/heldout_source_rows'], test)
        np.testing.assert_array_equal(cmp['root/heldout_parent_ids'], ids[test])
        groups = {k: cmp[f'root/groups/{k}'] for k in group_expected}
        weights = {k: cmp[f'root/weights/{k}'] for k in weight_expected}
        for k in groups: np.testing.assert_array_equal(groups[k], group_expected[k])
        for k in weights: np.testing.assert_array_equal(weights[k], weight_expected[k])
        S = {}; cov_support = {}; role_summaries = []; cov_overlap = []; coverage_support = {}
        for h in hypotheses:
            rel = f'{trial_rel}/evaluation/DERIVED64_{h}.npz'
            with np.load(root / rel, allow_pickle=False) as ar:
                members = member_receipts[f'runs/primary/artifacts/T1/DERIVED64_{h}.npz']
                S[h] = {k: ar[k] for k in members}
            member_receipts[rel] = members
            fs = S[h]['feature_support']; ft = T[h]['feature_support']
            ps = S[h]['pair_support']; pt = T[h]['pair_support']
            cov_support[h] = ((fs & ft).all(1) & (ps & pt).all((1, 2)))[test]
            coverage_support[h] = (fs & ft)[:, coverage_features]
            rel = f'{trial_rel}/evaluation/PARENT_COMPARISON_{h}.npz'
            with np.load(root / rel, allow_pickle=False) as ar:
                members = ['parent_ids', 'test_rows', 'coverage_supported', 'coverage_n_student_finite', 'coverage_n_teacher_defined', 'coverage_hits', 'coverage_rates']
                np.testing.assert_array_equal(ar['parent_ids'], ids)
                np.testing.assert_array_equal(ar['test_rows'], test)
                supported = ar['coverage_supported']
                np.testing.assert_array_equal(supported, coverage_support[h])
                np.testing.assert_array_equal(ar['coverage_n_student_finite'], S[h]['scalar_defined_draws'][:, coverage_features])
                np.testing.assert_array_equal(ar['coverage_n_teacher_defined'], T[h]['scalar_defined_draws'][:, coverage_features])
                assert np.all(ar['coverage_hits'][~supported] == -1)
                assert np.all(np.isnan(ar['coverage_rates'][~supported]))
            member_receipts[rel] = members
            for role, data in (('teacher', T[h]), (f'student_seed{seed}', S[h])):
                fmask = data['feature_support']; pmask = data['pair_support']
                cmask = fmask.all(1) & pmask.all((1, 2))
                selected_mask = cmask[test] & low_group
                rise = fmask[test, 6]
                bad_rows = test[low_group & ~cmask[test]]
                scalar_summary = {}
                for j, name in enumerate(features):
                    counts = data['scalar_defined_draws'][test[low_group], j]
                    scalar_summary[name] = dict(supported_parents=int(fmask[test[low_group], j].sum()), original_parents=int(low_group.sum()),
                                                combined_64_finite_draw_min=int(counts.min()), combined_64_finite_draw_max=int(counts.max()),
                                                combined_below48_parents=int(np.sum(counts < 48)),
                                                support_false_with_combined_at_least48=int(np.sum(~fmask[test[low_group], j] & (counts >= 48))))
                masses = {}
                for name, w in weights.items():
                    N = int(low_group.sum()); n = int(selected_mask.sum()); W = float(w[low_group].sum()); Wi = float(w[selected_mask].sum())
                    mass = Wi / W; ess = Wi**2 / float(np.square(w[selected_mask]).sum())
                    masses[name] = dict(supported_parents=n, original_parents=N, count_fraction=n/N, original_weight=W,
                                        supported_weight=Wi, supported_weight_fraction=mass, ESS=ess,
                                        status='supported' if n >= 64 and n/N >= .95 and mass >= .95 and ess >= .5*N else 'pending')
                role_summaries.append(dict(hypothesis=h, role=role, low_group_covariance=masses, scalar_features_low_group=scalar_summary,
                                           full_cache_feature_supported_counts={name:int(fmask[:, j].sum()) for j,name in enumerate(features)},
                                           heldout_feature_supported_counts={name:int(fmask[test,j].sum()) for j,name in enumerate(features)},
                                           low_group_cov_mask_equals_rise_mask=bool(np.array_equal(cmask[test][low_group], rise[low_group])),
                                           low_group_cov_failure_without_rise_failure=int(np.sum(low_group & ~cmask[test] & rise)),
                                           low_group_failed_cov_source_rows=bad_rows.tolist(),
                                           low_group_flag_summaries={k:dict(undefined_slots=int(round(float(v[test[low_group]].sum()) * 64)),
                                                                           original_slots=int(low_group.sum()) * 64,
                                                                           fraction=float(v[test[low_group]].mean()),
                                                                           affected_parents=int(np.sum(v[test[low_group]] > 0)))
                                                                    for k,v in data.items() if k.startswith('fraction_')}))
            tgood = (ft.all(1) & pt.all((1, 2)))[test]
            sgood = (fs.all(1) & ps.all((1, 2)))[test]
            cov_overlap.append(dict(hypothesis=h, original_low_group_parents=int(low_group.sum()),
                                    both_supported=int(np.sum(low_group & tgood & sgood)),
                                    teacher_unsupported_only=int(np.sum(low_group & ~tgood & sgood)),
                                    student_unsupported_only=int(np.sum(low_group & tgood & ~sgood)),
                                    both_unsupported=int(np.sum(low_group & ~tgood & ~sgood)),
                                    covariance_common_equals_rise_common_low_group=bool(np.array_equal(cov_support[h][low_group], coverage_support[h][test[low_group],4]))))

        pending_results = []; analytic_cells = []
        for cell in report['target_cells']:
            g = groups[cell['group']]; w = weights[cell['weights']]
            if cell['metric'] == 'analytic_charge_variance':
                ix = np.flatnonzero(g)
                np.testing.assert_array_equal(cmp[cell['supported_parent_indices']['array']], ix)
                v = a * Q[test] + b * Q[test]**2; t = ta * Q[test] + tb * Q[test]**2
                assert np.isfinite(v).all() and np.isfinite(t).all() and (v > 0).all() and (t > 0).all()
                alpha = w[ix] / w[ix].sum()
                rms = float(np.sqrt(np.sum(alpha * (v[ix] - t[ix])**2) / np.sum(alpha * t[ix]**2)))
                maximum = float(np.max(np.abs(v[ix] - t[ix]) / t[ix]))
                np.testing.assert_allclose([rms, maximum], [cell['relative_RMS'], cell['relative_max']], rtol=1e-12, atol=1e-14)
                assert cell['original_parents'] == cell['supported_parents'] == len(ix)
                analytic_cells.append(dict(cell_index=cell['cell_index'], group=cell['group'], weights=cell['weights'],
                                           original_parents=len(ix), all_original_group_indices_equal=True,
                                           relative_RMS=rms, relative_max=maximum,
                                           status='pass' if rms <= .10 and maximum <= .20 else 'fail'))
                continue
            h = cell['hypothesis']
            if cell['metric'] == 'conditional_cov9': support = cov_support[h]
            elif cell['metric'] == 'time_rise10_90': support = (S[h]['feature_support'] & T[h]['feature_support'])[test,6]
            else:
                assert cell['metric'] == 'PIT_coverage' and cell['feature'] == 'rise10_90'
                support = coverage_support[h][test,4]
            ix = np.flatnonzero(g & support)
            np.testing.assert_array_equal(cmp[cell['indices']['array']], ix)
            N = int(g.sum()); n = len(ix); W = float(w[g].sum()); Wi = float(w[ix].sum()); mass = Wi/W
            ess = Wi**2 / float(np.square(w[ix]).sum())
            reasons = []
            if n < 64: reasons.append('fewer_than64_common_parents')
            if N == 0 or n/N < .95: reasons.append('common_count_below95percent_original')
            if mass < .95: reasons.append('common_weight_mass_below95percent_original')
            if ess < .5*N: reasons.append('ESS_below_half_original_parent_count')
            assert N == cell['original_parents'] and n == cell['supported_parents'] and reasons == cell['reasons']
            np.testing.assert_allclose([W, Wi, mass, ess], [cell['original_weight'], cell['supported_weight'], cell['supported_weight_fraction'], cell['ESS']], rtol=1e-12, atol=1e-14)
            pending_results.append(dict(cell_index=cell['cell_index'], metric=cell['metric'], hypothesis=h,
                                        group=cell['group'], weights=cell['weights'], feature=cell.get('feature'), level=cell.get('level'),
                                        original_parents=N, supported_parents=n, count_fraction=n/N,
                                        original_weight=W, supported_weight=Wi, supported_weight_fraction=mass, ESS=ess,
                                        status='pending', reasons=reasons, exact_saved_indices_equal=True))

    all_curve = a*Q + b*Q**2; teacher_curve = ta*Q + tb*Q**2
    assert np.isfinite(all_curve).all() and (all_curve > 0).all()
    curve_roles = {}
    for name, ix in (('all_original_cache', np.arange(len(Q))), ('fit', fit), ('heldout', test)):
        signed = (all_curve[ix] - teacher_curve[ix]) / teacher_curve[ix]
        curve_roles[name] = dict(parents=len(ix), deleted_parents=0, invalid_parents=0,
                                Q_min=float(Q[ix].min()), Q_max=float(Q[ix].max()),
                                teacher_variance_min=float(teacher_curve[ix].min()), teacher_variance_max=float(teacher_curve[ix].max()),
                                student_variance_min=float(all_curve[ix].min()), student_variance_max=float(all_curve[ix].max()),
                                relative_RMS_unweighted=float(np.sqrt(np.sum((all_curve[ix]-teacher_curve[ix])**2)/np.sum(teacher_curve[ix]**2))),
                                signed_relative_min=float(signed.min()), signed_relative_max=float(signed.max()), relative_max=float(np.max(np.abs(signed))))
    result['trial_results'].append(dict(seed=seed, producer_status_counts=report['producer_status_counts'],
                                       producer_all_required_cells_pass=report['all_required_cells_pass'],
                                       pending_cells_verified=len(pending_results), pending_cell_results=pending_results,
                                       role_support_summaries=role_summaries, teacher_student_covariance_overlap=cov_overlap,
                                       a_b=dict(a=a,b=b,teacher_a=ta,teacher_b=tb,
                                                a_relative_error=abs(a-ta)/ta,b_relative_error=abs(b-tb)/tb,
                                                comparison_and_evaluation_reports_equal=True, checkpoint_decoded=False),
                                       analytic_variance_cells=analytic_cells, analytic_all_cached_Q_curves=curve_roles))

result['decoded_npz_files'] = [rel for rel in payload['array_paths'] if rel.endswith('.npz')]
result['decoded_npy_files'] = [rel for rel in payload['array_paths'] if rel.endswith('.npy')]
result['npz_member_reads'] = member_receipts
result['saved_support_masks_rederived_from_native_raw_scalars'] = False
result['coverage_values_recomputed'] = False
result['primary_overall_acceptance_recomputed'] = False
result['pending_is_pass'] = False
result['peak_RSS_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result['elapsed_seconds'] = time.monotonic() - started
result['CPU_threads_env'] = {k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
assert len(result['decoded_npz_files']) == 18 and len(result['decoded_npy_files']) == 1
assert result['peak_RSS_KiB'] <= 1536 * 1024 and result['elapsed_seconds'] < 180
output = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
assert len(output.encode()) < 8 * 1024**2
print(output)
'''
payload_line = "payload = json.loads(base64.b64decode('" + base64.b64encode(json.dumps(payload, ensure_ascii=True).encode()).decode() + "'))"
remote = remote.replace('PAYLOAD_LINE', payload_line)
(out / 'SSH_STDIN.py').write_text(remote, encoding='utf-8', newline='\n')
command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'rylan@100.82.231.102',
           'timeout --signal=TERM --kill-after=5s 180s env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -']
(out / 'EXECUTION_COMMAND.json').write_text(json.dumps(dict(command=command, stdin_file='SSH_STDIN.py', stdin_sha256=hashlib.sha256(remote.encode()).hexdigest()), indent=2), encoding='utf-8')
started = time.time()
with (out / 'NUMERIC_READBACK.json').open('wb') as stdout, (out / 'SSH_STDERR.txt').open('wb') as stderr:
    execution = subprocess.run(command, input=remote.encode(), stdout=stdout, stderr=stderr, timeout=205)
(out / 'EXECUTION_RECEIPT.json').write_text(json.dumps(dict(exit_code=execution.returncode, elapsed_seconds=time.time()-started,
                                                          stdout_bytes=(out/'NUMERIC_READBACK.json').stat().st_size,
                                                          stderr_bytes=(out/'SSH_STDERR.txt').stat().st_size,
                                                          stdin_sha256=hashlib.sha256(remote.encode()).hexdigest()), indent=2), encoding='utf-8')
print((out / 'EXECUTION_RECEIPT.json').read_text())
if execution.returncode != 0:
    print((out / 'SSH_STDERR.txt').read_text())
    raise SystemExit(execution.returncode)
receipt = json.loads((out / 'NUMERIC_READBACK.json').read_text(encoding='utf-8'))
print(json.dumps(dict(npz_files=len(receipt['decoded_npz_files']), npy_files=len(receipt['decoded_npy_files']),
                      peak_RSS_KiB=receipt['peak_RSS_KiB'], elapsed_seconds=receipt['elapsed_seconds'],
                      pending_verified=[t['pending_cells_verified'] for t in receipt['trial_results']]), indent=2))
