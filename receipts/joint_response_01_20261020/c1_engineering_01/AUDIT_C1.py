from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
out = HERE / 'collected/run_01'
started = time.monotonic()
registration = json.loads((HERE / 'REGISTRATION.json').read_text())
p = json.loads((ROOT / registration['config']).read_bytes())
assert hashlib.sha256((ROOT / registration['config']).read_bytes()).hexdigest() == registration['config_sha256']
terminal = json.loads((HERE / 'REMOTE_TERMINAL_VERIFICATION.json').read_text())
manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
assert terminal['native_exit']['exit_code'] == 0 and terminal['processes'] == []
assert terminal['source_files_verified'] == len(manifest['files']) == 688
for name, digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256'] == digest
for name, record in terminal['outputs'].items():
    raw = (HERE / 'collected' / name).read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
result = json.loads((out / 'RESULT.json').read_text())
assert result['config_sha256'] == registration['config_sha256']
assert set(result['execution_counts']) <= set(p['operation_budget'])
for name, maximum in p['operation_budget'].items():
    assert result['execution_counts'].get(name, 0) == maximum, name
identities = json.loads((out / 'ZERO_IDENTITY.json').read_text())
assert len(identities['cases']) == 6 and identities['parameter_counts'] == [54, 54, 46]
assert sum(item['rows'] for item in identities['cases']) == 39558
assert all(item['exact'] and item['signal_sha256']['original'] == item['signal_sha256']['C1'] for item in identities['cases'])
checks = []
for seed in (0, 1):
    report = json.loads((out / f'seed{seed}_C1_ENGINEERING.json').read_text())
    with np.load(out / f'seed{seed}_C1_ENGINEERING.npz', allow_pickle=False) as z, np.load(ROOT / p['cases'][seed]['arrays'], allow_pickle=False) as old:
        assert np.array_equal(z['source_parents'], old['FIT0_source_parents'])
        assert np.array_equal(z['proposals'], np.stack((old['trial0_proposal'], old['trial1_proposal'])))
        assert np.array_equal(z['original_baseline_signal'], z['C1_baseline_signal'])
        assert int(z['bank_seed']) == p['training_seed'] + seed * 100000 + 1000
        proposals = z['proposals']
        for branch in ('original', 'C1'):
            baseline, tangents = z[branch + '_baseline_signal'], z[branch + '_signal_tangent']
            pairs = z[branch + '_FD_pairs_first8']
            plus, minus = z[branch + '_FD_plus_theta'], z[branch + '_FD_minus_theta']
            expected_plus = np.eye(54) * p['finite_difference_step']
            expected_minus = -expected_plus.copy(); expected_minus[-1, -1] = 0
            assert np.array_equal(plus, expected_plus) and np.array_equal(minus, expected_minus)
            reconstructed = np.einsum('pj,jnt->pnt', proposals, (pairs[:, 0] - pairs[:, 1]) / (plus - minus).diagonal()[:, None, None])
            secant_error = float(np.max(abs(reconstructed - tangents[:, :8])))
            assert secant_error < 1e-12
            for pi, point in enumerate(('trial0', 'trial1')):
                actual = z[branch + '_' + point + '_actual_signal']
                movement = actual - baseline
                error = actual - baseline - tangents[pi]
                numbers = dict(baseline_norm=np.linalg.norm(baseline), movement_norm=np.linalg.norm(movement),
                    error_norm=np.linalg.norm(error), relative_baseline_error=np.linalg.norm(error) / np.linalg.norm(baseline),
                    relative_movement_error=np.linalg.norm(error) / np.linalg.norm(movement))
                for name, value in numbers.items():
                    assert np.isclose(value, report[branch][pi][name], rtol=1e-12, atol=1e-12)
                assert np.isfinite(actual).all() and int((actual[:, 100:].sum(1) <= 0).sum()) == 0
                record = dict(seed=seed, branch=branch, point=point, full_signal_rows=len(actual),
                    first8_secant_max_difference=secant_error, **{k: float(v) for k, v in numbers.items()})
                if branch == 'C1':
                    base, queries, a = [z[branch + '_' + point + '_' + key] for key in ('base', 'queries', 'coordinates')]
                    interpolated = np.stack([PchipInterpolator(np.arange(384), values)(q) for values, q in zip(base, queries, strict=True)])
                    peak = z['peak'][:, None]
                    t = np.arange(384)[None]
                    multiplier = np.exp(a[:, 0:1] + a[:, 3:4] / (1 + np.exp(-(t - peak - 30) / 8))
                        + a[:, 5:6] * np.exp(-.5 * ((t - peak) / 4)**2))
                    reproduced = interpolated * multiplier
                    difference = float(np.max(abs(reproduced - actual)))
                    assert difference <= 1e-12
                    left = np.minimum(np.floor(queries).astype(int), 382)
                    v0, v1 = np.take_along_axis(base, left, axis=1), np.take_along_axis(base, left + 1, axis=1)
                    overshoot = max(0., float(np.max(np.minimum(v0, v1) - interpolated)), float(np.max(interpolated - np.maximum(v0, v1))))
                    assert overshoot <= 1e-12 and np.all(np.diff(queries, axis=1) > 0)
                    record.update(SciPy_PCHIP_full_signal_max_difference_V=difference, interpolation_max_overshoot_V=overshoot)
                checks.append(record)
        (HERE / 'AUDIT_PROGRESS.json').write_text(json.dumps(dict(seed=seed, elapsed_seconds=time.monotonic() - started)) + '\n')
    if time.monotonic() - started > 300:
        raise TimeoutError('Independent C1 audit budget exceeded')
for comparison in result['comparisons']:
    original, c1 = [next(x for x in checks if x['seed'] == comparison['seed'] and x['point'] == comparison['point'] and x['branch'] == branch) for branch in ('original', 'C1')]
    ratio = c1['relative_baseline_error'] / original['relative_baseline_error']
    movement = c1['movement_norm'] / original['movement_norm']
    assert np.isclose(ratio, comparison['signal_error_ratio'], rtol=1e-12, atol=1e-12)
    assert ratio <= p['success']['maximum_signal_error_ratio'] and c1['relative_baseline_error'] <= p['success']['maximum_baseline_error']
    assert p['success']['movement_ratio_low'] <= movement <= p['success']['movement_ratio_high']
    assert comparison['pass_for_teacher_validation']
receipt = dict(status='PASS_INDEPENDENT_C1_ENGINEERING_AUDIT', UTC=datetime.now(timezone.utc).isoformat(),
    config_sha256=registration['config_sha256'], source_files_verified=688, source_pins_verified=33,
    output_files_verified=len(terminal['outputs']), output_bytes=sum(x['bytes'] for x in terminal['outputs'].values()),
    result_sha256=terminal['result_sha256'], native_exit=terminal['native_exit'], execution_counts=result['execution_counts'],
    original_and_C1_zero_identity_rows=39558, saved_zero_identity_hash_cases=6,
    independently_audited_signal_rows=32768, SciPy_PCHIP_verified_rows=16384,
    first8_signal_secant_columns_recomputed=216, full_row_contrast_gates_verified=4,
    SciPy_PCHIP_maximum_absolute_signal_difference_V=max(x.get('SciPy_PCHIP_full_signal_max_difference_V', 0) for x in checks),
    checks=checks, comparisons=result['comparisons'], engineering_gate_pass=True, scientific_acceptance=False,
    parameter_updates=0, elapsed_scientific_seconds=result['elapsed_seconds'], audit_elapsed_seconds=time.monotonic() - started,
    boundary='Full-row saved C1 signals are independently reconstructed; full SOURCE zero identity is evidenced by native comparisons and matching hashes. This does not verify REAL matching or inherit the original teacher pass.',
    next_action='Register renewed coupled teacher validation using the unchanged original coupled teacher as the recovery target and C1 learners; require original hard-output gates plus fresh 2048-draw conditional-kernel and MC-precision gates before any REAL use.')
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({k: v for k, v in receipt.items() if k not in ('checks', 'comparisons', 'execution_counts')}))
