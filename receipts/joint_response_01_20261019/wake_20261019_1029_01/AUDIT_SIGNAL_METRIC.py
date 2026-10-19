from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'collected/run_01'
OLD = ROOT / 'receipts/joint_response_01_20261019/wake_20261019_1010_01/collected/run_01'
config_path = ROOT / 'configs/joint_response_01_signal_metric_attribution_20261019.json'
p = json.loads(config_path.read_text())
remote = json.loads((HERE / 'REMOTE_TERMINAL_REVIEW.json').read_text())
result = json.loads((OUT / 'RESULT.json').read_text())
assert remote['source_files_verified'] == 640 and remote['native_exit']['exit_code'] == 0
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
all_large_wave = []

for seed in range(2):
    report = json.loads((OUT / f'seed{seed}_SIGNAL_METRIC.json').read_text())
    a = dict(np.load(OUT / f'seed{seed}_SIGNAL_METRIC.npz', allow_pickle=False))
    old_path = OLD / f'seed{seed}_READOUT_ATTRIBUTION.npz'
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == p['source_pins'][p['readout_arrays'][seed]]
    old = dict(np.load(old_path, allow_pickle=False))
    for key, value in a.items():
        assert np.isfinite(value).all(), key
        checked['finite_arrays'] += 1
    for key in ('theta', 'FIT_source_parents', 'FIT_real_rows', 'cholesky', 'fixed_peak_indices'):
        np.testing.assert_array_equal(a[key], old[key])
        checked['exact_array_replays'] += 1
    assert a['theta'].shape == (54,) and a['cholesky'].shape == (272, 272)
    theta, baseline = a['theta'], a['baseline_signal']
    assert baseline.shape == a['target_carrier_adc'].shape == (4096, 384)
    assert a['shape_latents'].shape == (4096, 8)
    assert ((theta >= lower) & (theta <= 1.)).all()
    for key, pool, holdout, number in (
        ('FIT_source_parents', 'source_train_rows', 'source_heldout_rows', 1024),
        ('FIT_real_rows', 'real_train_rows', 'real_heldout_rows', 4096)):
        rows = a[key]
        assert rows.shape == (number,) and len(np.unique(rows)) == number
        assert np.isin(rows, roles[pool]).all() and not np.isin(rows, roles[holdout]).any()
        checked['role_rows_checked'] += number
    np.testing.assert_array_equal(a['repeated_source_rows'], np.repeat(a['FIT_source_parents'], 4))
    np.testing.assert_allclose(a['baseline_noise_scale'], np.exp(np.log(1.03) * theta[-1]), rtol=2e-14, atol=0.)
    np.testing.assert_array_equal(a['target_carrier_adc'], np.rint((a['target_carrier_adc'] + .5) * 16383) / 16383 - .5)
    for view in joint_views:
        np.testing.assert_array_equal(a[view + '_baseline_residual'], old[view + '_baseline_residual'])
        checked['exact_array_replays'] += 1
    for view in native_views:
        np.testing.assert_array_equal(a[view + '_baseline_native'], old[view + '_baseline_native'])
        assert (a[view + '_baseline_native'] > 0).all()
        checked['exact_array_replays'] += 1
    for hi, h in enumerate(('fast', 'slow', 'common_level')):
        noise = a['bank_noise_' + h]
        assert noise.shape == baseline.shape
        measured = np.rint((np.clip(baseline + a['baseline_noise_scale'] * noise, -.5, .5) + .5) * 16383) / 16383 - .5
        centered = measured - measured[:, :100].mean(1, keepdims=True)
        peaks = centered[:, 112:144].argmax(1) + 112
        np.testing.assert_array_equal(peaks, a['fixed_peak_indices'][hi])
        checked['independent_baseline_hard_readout_rows'] += len(peaks)
    cases = []
    for ci, case in enumerate(report['cases']):
        step = p['difference_steps'][ci]
        assert case['difference_step'] == step
        prefix = f'case{ci}_'
        direction, tangent = a[prefix + 'saved_direction'], a[prefix + 'signal_tangent']
        np.testing.assert_array_equal(direction, old[prefix + 'saved_direction'])
        assert tangent.shape == (4096, 384)
        plus, minus = np.minimum(1., theta + step), np.maximum(lower, theta - step)
        denominator = plus - minus
        np.testing.assert_array_equal(denominator, a[prefix + 'secant_denominators'])
        pairs = a[prefix + 'signal_pair_audit_first8']
        assert pairs.shape == (54, 2, 8, 384)
        reconstructed = np.zeros((8, 384))
        for column in range(54):
            reconstructed += direction[column] * (pairs[column, 0] - pairs[column, 1]) / denominator[column]
            checked['signal_secant_columns_first8_recomputed'] += 1
        np.testing.assert_allclose(reconstructed, tangent[:8], rtol=2e-12, atol=2e-13)
        noise_tangent = direction[-1] * (np.exp(np.log(1.03) * plus[-1]) - np.exp(np.log(1.03) * minus[-1])) / denominator[-1]
        np.testing.assert_allclose(noise_tangent, a[prefix + 'noise_scale_tangent'], rtol=2e-12, atol=2e-13)
        probes = []
        for pi, probe in enumerate(case['probes']):
            fraction = p['directional_probe_fractions'][pi]
            assert fraction == probe['fraction']
            pp = prefix + f'probe{pi}_'
            proposal = theta + fraction * direction
            assert ((proposal >= lower) & (proposal <= 1.)).all()
            np.testing.assert_array_equal(proposal, a[pp + 'proposal'])
            np.testing.assert_array_equal(proposal, old[pp + 'proposal'])
            actual, linear = a[pp + 'actual_signal'], baseline + fraction * tangent
            assert actual.shape == (4096, 384)
            actual_scale, linear_scale = a[pp + 'actual_noise_scale'], a[pp + 'linear_noise_scale']
            np.testing.assert_allclose(actual_scale, np.exp(np.log(1.03) * proposal[-1]), rtol=2e-14, atol=0.)
            np.testing.assert_allclose(linear_scale, a['baseline_noise_scale'] + fraction * noise_tangent, rtol=2e-14, atol=0.)
            error = np.linalg.norm(actual - linear)
            relative = float(error / np.linalg.norm(baseline))
            movement_norm = float(np.linalg.norm(actual - baseline))
            movement_error = None if movement_norm == 0 else float(error / movement_norm)
            np.testing.assert_allclose(relative, probe['signal_error_relative_to_baseline_RMS'], rtol=2e-11, atol=2e-12)
            if movement_error is None:
                assert probe['signal_error_relative_to_actual_movement'] is None
            else:
                np.testing.assert_allclose(movement_error, probe['signal_error_relative_to_actual_movement'], rtol=2e-11, atol=2e-12)
            checked['full_row_signal_errors_recomputed'] += 1
            knots = a[pp + 'changed_interpolation_knots_per_row']
            assert knots.shape == (4096,) and ((knots >= 0) & (knots <= 283)).all()
            np.testing.assert_allclose(knots.mean(), probe['mean_changed_interpolation_knots'], rtol=0, atol=0)
            for branch, signal, scale in (('actual', actual, actual_scale), ('linear_signal', linear, linear_scale)):
                diagnostics = probe['readout_diagnostics'][branch]
                recomputed = {key: [] for key in diagnostics}
                for hi, h in enumerate(('fast', 'slow', 'common_level')):
                    carrier = signal + scale * a['bank_noise_' + h]
                    measured = np.rint((np.clip(carrier, -.5, .5) + .5) * 16383) / 16383 - .5
                    centered = measured - measured[:, :100].mean(1, keepdims=True)
                    peaks = centered[:, 112:144].argmax(1) + 112
                    indices = peaks[:, None] - 16 + np.arange(256)[None]
                    fixed_indices = a['fixed_peak_indices'][hi, :, None] - 16 + np.arange(256)[None]
                    window = np.take_along_axis(centered, indices, 1)
                    fixed = np.take_along_axis(centered, fixed_indices, 1)
                    continuous = np.take_along_axis(carrier - carrier[:, :100].mean(1, keepdims=True), fixed_indices, 1)
                    recomputed['peak_changed_rows'].append(int((peaks != a['fixed_peak_indices'][hi]).sum()))
                    recomputed['frozen_window_changed_rows'].append(int((window != fixed).any(1).sum()))
                    recomputed['alignment_window_RMS_V'].append(float(np.sqrt(np.mean((window - fixed)**2))))
                    recomputed['quantization_fixed_window_RMS_V'].append(float(np.sqrt(np.mean((fixed - continuous)**2))))
                    checked['independent_probe_hard_readout_rows'] += len(peaks)
                for key, value in recomputed.items():
                    if key.endswith('_rows'):
                        np.testing.assert_array_equal(value, diagnostics[key])
                    else:
                        np.testing.assert_allclose(value, diagnostics[key], rtol=2e-10, atol=2e-13)
                    checked['readout_diagnostic_comparisons'] += 1
            joint, native = {}, {}
            for view in joint_views:
                A, L, P = [a[pp + view + '_' + part + '_residual'] for part in ('actual', 'linear_signal', 'scalar_prediction')]
                np.testing.assert_array_equal(A, old[pp + view + '_actual_residual'])
                np.testing.assert_array_equal(P, old[pp + view + '_predicted_residual'])
                checked['exact_array_replays'] += 2
                base = a[view + '_baseline_residual']
                norm = np.linalg.norm(base)
                metrics = dict(total_error=float(np.linalg.norm(A - P) / norm),
                    signal_approximation_effect=float(np.linalg.norm(A - L) / norm),
                    readout_metric_remainder=float(np.linalg.norm(L - P) / norm))
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, probe['joint'][view][key], rtol=2e-11, atol=2e-12)
                np.testing.assert_allclose(A - P, (A - L) + (L - P), rtol=2e-11, atol=2e-12)
                assert metrics['total_error'] <= metrics['signal_approximation_effect'] + metrics['readout_metric_remainder'] + 1e-12
                checked['joint_probe_vector_decompositions'] += 1
                joint[view] = metrics
            for view in native_views:
                A, L, P = [a[pp + view + '_' + part + '_native'] for part in ('actual', 'linear_signal', 'scalar_prediction')]
                np.testing.assert_array_equal(A, old[pp + view + '_actual_native'])
                np.testing.assert_array_equal(P, old[pp + view + '_predicted_native'])
                checked['exact_array_replays'] += 2
                base = a[view + '_baseline_native']
                metrics = dict(total_error=((A - P) / base).reshape(3, 3).tolist(),
                    signal_approximation_effect=((A - L) / base).reshape(3, 3).tolist(),
                    readout_metric_remainder=((L - P) / base).reshape(3, 3).tolist(),
                    actual_ratio=(A / base).reshape(3, 3).tolist(), linear_signal_ratio=(L / base).reshape(3, 3).tolist())
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, probe['native'][view][key], rtol=2e-11, atol=2e-12)
                np.testing.assert_allclose(np.asarray(metrics['total_error']),
                    np.asarray(metrics['signal_approximation_effect']) + metrics['readout_metric_remainder'], rtol=2e-11, atol=2e-12)
                checked['native_probe_vector_decompositions'] += 1
                native[view] = metrics
                if fraction == .25 and view == 'original':
                    for hi in range(3):
                        all_large_wave.append(dict(seed=seed, step=step, hypothesis=hi,
                            total=metrics['total_error'][hi][0], signal_effect=metrics['signal_approximation_effect'][hi][0],
                            remainder=metrics['readout_metric_remainder'][hi][0]))
            probes.append(dict(fraction=fraction, signal_error_relative_to_baseline_RMS=relative,
                signal_error_relative_to_actual_movement=movement_error, joint=joint, native=native))
        cases.append(dict(difference_step=step, probes=probes))
    summaries[str(seed)] = dict(cases=cases)

assert checked['signal_secant_columns_first8_recomputed'] == 216
assert checked['full_row_signal_errors_recomputed'] == 12
assert checked['joint_probe_vector_decompositions'] == 24
assert checked['native_probe_vector_decompositions'] == 36
assert result['committed_parameter_updates'] == 0 and result['scientific_acceptance'] is False
signal_dominates = sum(abs(row['signal_effect']) > abs(row['remainder']) for row in all_large_wave)
audit = dict(status='PASS_EXECUTION_FULL_ROW_SIGNAL_AND_INDEPENDENT_READOUT_AUDIT',
    source_files_verified=640, output_hashes_verified=len(remote['outputs']), native_exit=remote['native_exit'],
    result_sha256=remote['result_sha256'], elapsed_seconds=result['elapsed_seconds'], execution_counts=c,
    audit_counts=dict(checked), seeds=summaries, large_step_original_wave_decomposition=all_large_wave,
    large_step_original_wave_signal_effect_larger_than_remainder_count=signal_dominates,
    large_step_original_wave_comparisons=len(all_large_wave), committed_parameter_updates=0, scientific_acceptance=False,
    limitations=['The first eight rows, not all 4096, have stored per-coordinate signal pairs; all rows have stored aggregated tangents and actual signals.',
        'Interpolation-cell counts were checked for bounds and aggregation only; query locations are not stored.',
        'Native and joint scalar/vector values exactly replay the preceding diagnostic; their decompositions are recomputed, but the metrics were not reexecuted from raw waveforms in this local audit.',
        'This is finite-scale signed error accounting on FIT data, not isolated causal attribution or new scientific validation.'],
    interpretation='Large joint steps have substantial generator signal linearization error. The signal-approximation effect exceeds the remaining hard-readout/metric effect in all 12 large-step original-wave comparisons. At smaller steps, readout/metric fluctuations can dominate, so reducing a coordinate finite-difference scale alone is not a complete remedy.',
    next_action='Prospectively register one bounded nonlinear trust-region correction of combined directions, evaluated by the unchanged original hard objective and guards; retain the original generator and failed scientific results.')
assert signal_dominates == 12
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: audit[key] for key in ('status', 'source_files_verified', 'output_hashes_verified', 'elapsed_seconds',
    'audit_counts', 'large_step_original_wave_signal_effect_larger_than_remainder_count', 'scientific_acceptance')}, indent=2))
