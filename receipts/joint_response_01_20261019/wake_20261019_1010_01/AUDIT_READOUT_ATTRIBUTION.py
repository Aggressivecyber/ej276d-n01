from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'collected/run_01'
config_path = ROOT / 'configs/joint_response_01_readout_attribution_20261019.json'
p = json.loads(config_path.read_text())
remote = json.loads((HERE / 'REMOTE_TERMINAL_REVIEW.json').read_text())
result = json.loads((OUT / 'RESULT.json').read_text())
assert remote['source_files_verified'] == 631 and remote['native_exit']['exit_code'] == 0
assert not (OUT / 'FAILED.json').exists()
for name, identity in remote['outputs'].items():
    raw = (HERE / 'collected' / name).read_bytes()
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256'], name
assert result['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
c = result['execution_counts']
for key, value in p['operation_budget'].items():
    assert c.get(key, 0) == value, (key, c.get(key, 0), value)
assert set(c) <= set(p['operation_budget'])
assert result['elapsed_seconds'] < p['resources']['internal_seconds']
assert result['peak_GPU_reserved_bytes'] < p['resources']['GPU_GiB'] * 1024**3
assert remote['native_exit']['peak_rss_bytes'] < p['resources']['RSS_GiB'] * 1024**3
assert sum(f.stat().st_size for f in OUT.iterdir()) < p['resources']['output_MiB'] * 1024**2
role_path = ROOT / 'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(role_path.read_bytes()).hexdigest() == p['source_pins'][p['roles_file']]
roles = dict(np.load(role_path, allow_pickle=False))
checked = Counter()
summaries = {}
joint_views = ('original', 'hard_fixed_peak')
native_views = (*joint_views, 'continuous_fixed_peak')
lower = np.full(54, -1.)
lower[-1] = 0.

for seed in range(2):
    report = json.loads((OUT / f'seed{seed}_READOUT_ATTRIBUTION.json').read_text())
    a = dict(np.load(OUT / f'seed{seed}_READOUT_ATTRIBUTION.npz', allow_pickle=False))
    old_path, old_report_path = [ROOT / p[key][seed] for key in ('fd_arrays', 'fd_reports')]
    for path in (old_path, old_report_path):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == p['source_pins'][path.relative_to(ROOT).as_posix()]
    old = dict(np.load(old_path, allow_pickle=False))
    old_report = json.loads(old_report_path.read_text())
    for key, value in a.items():
        assert np.isfinite(value).all(), key
        checked['finite_arrays'] += 1
    for key in ('theta', 'FIT_source_parents', 'FIT_real_rows', 'cholesky'):
        assert np.array_equal(a[key], old[key]), key
        checked['exact_array_replays'] += 1
    assert a['theta'].shape == (54,) and a['cholesky'].shape == (272, 272)
    theta = a['theta']
    assert ((theta >= lower) & (theta <= 1.)).all()
    for key, pool, holdout, number in (
        ('FIT_source_parents', 'source_train_rows', 'source_heldout_rows', 1024),
        ('FIT_real_rows', 'real_train_rows', 'real_heldout_rows', 4096)):
        rows = a[key]
        assert rows.shape == (number,) and len(np.unique(rows)) == number
        assert np.isin(rows, roles[pool]).all() and not np.isin(rows, roles[holdout]).any()
        checked['role_rows_checked'] += number
    assert a['fixed_peak_indices'].shape == (3, 4096)
    assert ((a['fixed_peak_indices'] >= 112) & (a['fixed_peak_indices'] <= 143)).all()
    for key, old_key in (('baseline_residual', 'residual'), ('baseline_native', 'native_values')):
        assert np.array_equal(a['original_' + key], a['hard_fixed_peak_' + key])
        assert np.array_equal(a['original_' + key], old[old_key])
        checked['exact_array_replays'] += 2
    for view in native_views:
        assert a[view + '_baseline_native'].shape == (9,)
        assert (a[view + '_baseline_native'] > 0).all()
        np.testing.assert_array_equal(a[view + '_baseline_native'], report['baseline_native'][view])
        checked['numeric_comparisons'] += 1
    assert report['baseline_diagnostics']['peak_changed_rows'] == [0, 0, 0]
    assert report['baseline_diagnostics']['frozen_window_changed_rows'] == [0, 0, 0]
    assert report['baseline_diagnostics']['alignment_window_RMS_V'] == [0., 0., 0.]
    cases = []
    for ci, case in enumerate(report['cases']):
        saved_index = p['saved_case_indices'][ci]
        step = p['difference_steps'][ci]
        assert case['difference_step'] == old_report['cases'][saved_index]['difference_step'] == step
        assert case['saved_case_index'] == saved_index
        denominator = np.minimum(1., theta + step) - np.maximum(lower, theta - step)
        assert (denominator > 0).all()
        for view in joint_views:
            pairs = a[f'case{ci}_{view}_pair_residuals']
            J = a[f'case{ci}_{view}_jacobian']
            assert pairs.shape == (54, 2, 816) and J.shape == (816, 54)
            np.testing.assert_allclose((pairs[:, 0] - pairs[:, 1]).T / denominator, J, rtol=2e-12, atol=2e-12)
            checked['joint_FD_columns_recomputed'] += 54
        for view in native_views:
            pairs = a[f'case{ci}_{view}_pair_native']
            N = a[f'case{ci}_{view}_native_jacobian']
            assert pairs.shape == (54, 2, 9) and N.shape == (9, 54)
            np.testing.assert_allclose((pairs[:, 0] - pairs[:, 1]).T / denominator, N, rtol=2e-12, atol=2e-12)
            checked['native_FD_columns_recomputed'] += 54
        for new_key, old_key in (('original_jacobian', 'jacobian'), ('original_native_jacobian', 'native_jacobian'), ('saved_direction', 'direction')):
            assert np.array_equal(a[f'case{ci}_{new_key}'], old[f'case{saved_index}_{old_key}'])
            checked['exact_array_replays'] += 1
        diagnostics = case['pair_diagnostics']
        assert len(diagnostics) == 54 and all(len(pair) == 2 for pair in diagnostics)
        counts = np.asarray([[d['peak_changed_rows'] for d in pair] for pair in diagnostics])
        windows = np.asarray([[d['frozen_window_changed_rows'] for d in pair] for pair in diagnostics])
        align = np.asarray([[d['alignment_window_RMS_V'] for d in pair] for pair in diagnostics])
        quant = np.asarray([[d['quantization_fixed_window_RMS_V'] for d in pair] for pair in diagnostics])
        assert counts.shape == windows.shape == align.shape == quant.shape == (54, 2, 3)
        assert ((counts >= 0) & (counts <= 4096)).all()
        assert ((windows >= 0) & (windows <= counts)).all()
        assert (align >= 0).all() and (quant >= 0).all()
        assert np.isfinite(align).all() and np.isfinite(quant).all()
        direction = a[f'case{ci}_saved_direction']
        probes = []
        for pi, probe in enumerate(case['probes']):
            fraction = p['directional_probe_fractions'][pi]
            assert probe['fraction'] == fraction
            prefix = f'case{ci}_probe{pi}_'
            proposal = np.clip(theta + fraction * direction, lower, 1.)
            np.testing.assert_array_equal(proposal, a[prefix + 'proposal'])
            delta = proposal - theta
            joint_summary, native_summary = {}, {}
            for view in joint_views:
                baseline = a[view + '_baseline_residual']
                actual = a[prefix + view + '_actual_residual']
                predicted = baseline + a[f'case{ci}_{view}_jacobian'] @ delta
                assert baseline.shape == actual.shape == predicted.shape == (816,)
                np.testing.assert_allclose(predicted, a[prefix + view + '_predicted_residual'], rtol=2e-11, atol=2e-12)
                metrics = dict(relative_vector_error=float(np.linalg.norm(actual - predicted) / np.linalg.norm(baseline)),
                    actual_whitened_risk_remaining=float(actual @ actual / (baseline @ baseline)),
                    predicted_whitened_risk_remaining=float(predicted @ predicted / (baseline @ baseline)))
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, probe['joint'][view][key], rtol=2e-11, atol=2e-12)
                checked['joint_probe_vectors_recomputed'] += 1
                joint_summary[view] = metrics
            for view in native_views:
                baseline = a[view + '_baseline_native']
                actual = a[prefix + view + '_actual_native']
                predicted = baseline + a[f'case{ci}_{view}_native_jacobian'] @ delta
                np.testing.assert_allclose(predicted, a[prefix + view + '_predicted_native'], rtol=2e-11, atol=2e-12)
                metrics = dict(actual_ratio=(actual / baseline).reshape(3, 3).tolist(),
                    predicted_ratio=(predicted / baseline).reshape(3, 3).tolist(),
                    relative_prediction_error=((actual - predicted) / baseline).reshape(3, 3).tolist())
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, probe['native'][view][key], rtol=2e-11, atol=2e-12)
                checked['native_probe_vectors_recomputed'] += 1
                native_summary[view] = metrics
            matched = [trial for trial in old_report['cases'][saved_index]['trials'] if trial['fraction'] == fraction]
            if matched:
                check = matched[0]['checks']['FIT']
                np.testing.assert_array_equal(a[prefix + 'original_actual_native'], check['native'])
                np.testing.assert_allclose(joint_summary['original']['relative_vector_error'], check['joint_linearization_relative_error'], rtol=0, atol=1e-12)
                checked['prior_directional_checks_reproduced'] += 1
            probes.append(dict(fraction=fraction, joint=joint_summary, native=native_summary))
        cases.append(dict(difference_step=step,
            reported_peak_changed_fraction=float(counts.sum() / (54 * 2 * 3 * 4096)),
            reported_alignment_RMS_range_V=[float(align.min()), float(align.max())],
            reported_quantization_RMS_range_V=[float(quant.min()), float(quant.max())], probes=probes))
    comparisons = {}
    for view in joint_views:
        u, v = [a[f'case{i}_{view}_jacobian'] for i in range(2)]
        metrics = dict(relative_Frobenius_difference=float(np.linalg.norm(u - v) / np.linalg.norm(u)),
            cosine=float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v))))
        for key, value in metrics.items():
            np.testing.assert_allclose(value, report['scale_comparisons'][view + '/joint'][key], rtol=2e-11, atol=2e-12)
            checked['matrix_comparisons_recomputed'] += 1
        comparisons[view + '/joint'] = metrics
    for view in native_views:
        u, v = [a[f'case{i}_{view}_native_jacobian'] for i in range(2)]
        x, y = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        metrics = dict(row_norms_left=x.tolist(), row_norms_right=y.tolist(),
            row_cosine=[None if m == 0 or n == 0 else float(s @ t / (m * n)) for s, t, m, n in zip(u, v, x, y)],
            row_relative_difference=[None if m == 0 else float(np.linalg.norm(s - t) / m) for s, t, m in zip(u, v, x)])
        for key, value in metrics.items():
            for actual, expected in zip(value, report['scale_comparisons'][view + '/native'][key]):
                if actual is None:
                    assert expected is None
                else:
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
            checked['matrix_comparisons_recomputed'] += 1
        comparisons[view + '/native'] = metrics
    assert report['committed_parameter_updates'] == 0
    assert report['fitting_role_only'] and report['original_readout_unchanged']
    assert report['counterfactual_views_are_not_scientific_acceptance']
    summaries[str(seed)] = dict(scale_comparisons=comparisons, cases=cases)

assert checked['joint_FD_columns_recomputed'] == 432
assert checked['native_FD_columns_recomputed'] == 648
assert checked['joint_probe_vectors_recomputed'] == 24
assert checked['native_probe_vectors_recomputed'] == 36
assert checked['prior_directional_checks_reproduced'] == c['saved_directional_checks_reproduced'] == 9
assert result['committed_parameter_updates'] == 0 and result['scientific_acceptance'] is False
audit = dict(status='PASS_EXECUTION_AND_INDEPENDENT_SAVED_VECTOR_AUDIT',
    source_files_verified=631, output_hashes_verified=len(remote['outputs']),
    native_exit=remote['native_exit'], result_sha256=remote['result_sha256'],
    elapsed_seconds=result['elapsed_seconds'], execution_counts=c, audit_counts=dict(checked),
    seeds=summaries, committed_parameter_updates=0, scientific_acceptance=False,
    limitations=['Peak-change counts and window RMS are bounded consistency checks of saved reports; individual generated waveforms were not saved.',
                 'Fixed-peak and continuous-voltage views are counterfactual diagnostic outputs and do not change original physical evaluation.',
                 'This uses the original FIT roles and frozen random bank only; it is neither a new holdout test nor a training endpoint.'],
    interpretation='Freezing peak selection stabilizes native wave derivatives, but not the full joint or QPSD derivatives. Removing ADC quantization stabilizes baseline-noise derivatives. Directional prediction errors remain with frozen continuous readout, so readout changes alone do not explain or fix the combined-step failure.',
    next_action='Inspect remaining model and metric nonlinearities using saved evidence, then prospectively register one bounded method diagnostic. Do not automatically select a scale, counterfactual endpoint, or new training run.')
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: audit[key] for key in ('status', 'source_files_verified', 'output_hashes_verified', 'elapsed_seconds', 'audit_counts', 'scientific_acceptance')}, indent=2))
