from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'collected/run_01'
p = json.loads((ROOT / 'configs/joint_response_01_fd_scale_20261019.json').read_text())
remote = json.loads((HERE / 'REMOTE_TERMINAL_REVIEW.json').read_text())
result = json.loads((OUT / 'RESULT.json').read_text())
assert remote['source_files_verified'] == 622 and remote['native_exit']['exit_code'] == 0
assert not (OUT / 'FAILED.json').exists()
for name, identity in remote['outputs'].items():
    raw = (HERE / 'collected' / name).read_bytes()
    assert len(raw) == identity['bytes'] and hashlib.sha256(raw).hexdigest() == identity['sha256'], name
assert result['config_sha256'] == hashlib.sha256((ROOT / 'configs/joint_response_01_fd_scale_20261019.json').read_bytes()).hexdigest()
for key, value in result['execution_counts'].items():
    assert value <= p['operation_budget'][key], key
assert result['elapsed_seconds'] < p['resources']['internal_seconds']
assert result['peak_GPU_reserved_bytes'] < p['resources']['GPU_GiB'] * 1024**3
assert remote['native_exit']['peak_rss_bytes'] < p['resources']['RSS_GiB'] * 1024**3
assert sum(f.stat().st_size for f in OUT.iterdir()) < p['resources']['output_MiB'] * 1024**2
role_path = ROOT / 'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(role_path.read_bytes()).hexdigest() == p['source_pins'][p['roles_file']]
roles = dict(np.load(role_path, allow_pickle=False))
hypotheses = ('fast', 'slow', 'common_level')
blocks = ('wave', 'charge_morphology', 'peak_codes', 'baseline')
checked = Counter()
summary = {}


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    checked['numeric_array_comparisons'] += 1


def gate(check, baseline, native_base, anchor):
    native = np.asarray(check['native'])
    ratios = (native / native_base).reshape(3, 3)
    anchored = (native / anchor).reshape(3, 3)
    close(ratios, check['native_ratios'])
    close(anchored, check['initial_anchor_ratios'])
    native_pass = all((x[:, :2] <= 1.01).all() and (x[:, 2] <= 1.1).all() for x in (ratios, anchored))
    trust = check['trust']
    flags = max(x for values in trust['flag_count_changes'].values() for x in values.values())
    trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and flags <= max(4, .005 * 4096)
    assert trust['denominator'] == 4096 and trust_pass == trust['pass']
    old = np.mean([baseline[h]['joint_whitened'] for h in hypotheses])
    new = np.mean([check['risks'][h]['joint_whitened'] for h in hypotheses])
    macro_pass = all(check['risks'][h][b] <= 1.15 * baseline[h][b] + 2. / 4096 for h in hypotheses for b in blocks)
    joint_pass = trust_pass and new <= .99 * old and macro_pass
    return bool(native_pass and joint_pass), bool(native_pass), bool(joint_pass), float(new / old)


for seed in range(2):
    report = json.loads((OUT / f'seed{seed}_FD_SCALE.json').read_text())
    arrays = dict(np.load(OUT / f'seed{seed}_FD_SCALE.npz', allow_pickle=False))
    assert hashlib.sha256((ROOT / p['endpoint_files'][seed]).read_bytes()).hexdigest() == p['source_pins'][p['endpoint_files'][seed]]
    with np.load(ROOT / p['endpoint_files'][seed], allow_pickle=False) as source:
        assert np.array_equal(source['theta'], arrays['theta'])
    sampled = {}
    for label in ('FIT', 'CHECK', 'FRESH'):
        for key, pool, number in (('source_parents', 'source_train_rows', 1024), ('real_rows', 'real_train_rows', 4096)):
            rows = arrays[label + '_' + key]
            assert rows.shape == (number,) and len(np.unique(rows)) == number
            assert np.isin(rows, roles[pool]).all()
            sampled[label + '_' + key] = number
    r, chol, v = arrays['residual'], arrays['cholesky'], arrays['native_values']
    close(v, report['baseline_native']['FIT'])
    close(r @ r / 272, np.mean([report['baseline_risks']['FIT'][h]['joint_whitened'] for h in hypotheses]))
    cases = []
    for index, case in enumerate(report['cases']):
        checked['FD_columns'] += 54
        checked['solves'] += 1
        J, N, direction = [arrays[f'case{index}_' + key] for key in ('jacobian', 'native_jacobian', 'direction')]
        assert J.shape == (816, 54) and N.shape == (9, 54) and direction.shape == (54,)
        theta = arrays['theta']; lower = np.full(54, -1.); lower[-1] = 0.
        assert np.linalg.norm(direction) <= .4 + 1e-8
        assert (theta + direction >= lower - 1e-8).all() and (theta + direction <= 1. + 1e-8).all()
        geometry = case['geometry']
        close(np.linalg.norm(direction), geometry['step_norm'])
        close(v + N @ direction, geometry['predicted_native_values'])
        close(np.square(r + J @ direction).sum() / (r @ r), geometry['predicted_joint_remaining_fraction'])
        close(.02 * np.diag(J.T @ J).mean(), geometry['ridge'])
        assert ((N @ direction) / v <= 1e-6).all()
        for hi, hypothesis in enumerate(hypotheses):
            take = slice(272 * hi, 272 * (hi + 1))
            raw = chol @ (np.sqrt(3) * r[take]) / np.sqrt(np.square(chol).sum(1))
            predicted = chol @ (np.sqrt(3) * (r[take] + J[take] @ direction)) / np.sqrt(np.square(chol).sum(1))
            for bi, block in enumerate(blocks):
                section = slice(56 * bi, 56 * (bi + 1))
                limit = 1.15 * np.square(raw[section]).mean() + 2. / 4096
                risk = np.square(predicted[section]).mean()
                close(limit, geometry['macro_limits'][hypothesis + '/' + block])
                close(risk, geometry['predicted_macro_risks'][hypothesis + '/' + block])
                assert risk <= limit * (1. + 1e-6)
        assert int((np.linalg.norm(J, axis=0) == 0).sum()) == case['zero_joint_columns']
        assert int((np.linalg.norm(N, axis=0) == 0).sum()) == case['zero_native_columns']
        peak, window = arrays[f'case{index}_peak_changes'], arrays[f'case{index}_window_changes']
        assert peak.shape == window.shape == (54, 3)
        assert ((peak >= 0) & (peak <= 4096)).all() and ((window >= 0) & (window <= 4096)).all()
        trial_summary = []
        for ti, trial in enumerate(case['trials']):
            checked['trials'] += 1
            assert trial['fraction'] == p['backtracking_fractions'][ti]
            move = np.clip(theta + trial['fraction'] * direction, lower, 1.) - theta
            passed = []
            for label, check in trial['checks'].items():
                passed_all, native_pass, joint_pass, remaining = gate(check, report['baseline_risks'][label],
                    np.asarray(report['baseline_native'][label]), np.asarray(report['initial_native'][label]))
                assert (passed_all, native_pass, joint_pass) == (check['accepted'], check['native_pass'], check['original_joint_accept'])
                passed.append(passed_all)
                checked['actual_gate_recomputations'] += 1
                if label == 'FIT':
                    prediction = v + N @ move
                    close((prediction / v).reshape(3, 3), check['predicted_native_ratios'])
                    close(((np.asarray(check['native']) - prediction) / v).reshape(3, 3), check['native_linearization_relative_error'])
                    actual_norm = np.sqrt(272 * np.mean([check['risks'][h]['joint_whitened'] for h in hypotheses]))
                    predicted_norm = np.linalg.norm(r + J @ move)
                    error = check['joint_linearization_relative_error']
                    assert abs(actual_norm - predicted_norm) / np.linalg.norm(r) - 1e-10 <= error
                    assert error <= (actual_norm + predicted_norm) / np.linalg.norm(r) + 1e-10
            assert trial['accepted'] == all(passed)
            assert not trial['accepted'] or ti == len(case['trials']) - 1
            fit = trial['checks']['FIT']
            trial_summary.append(dict(fraction=trial['fraction'], accepted=trial['accepted'],
                FIT_native_prediction_error_max=float(np.abs(fit['native_linearization_relative_error']).max()),
                FIT_joint_residual_error_reported=fit['joint_linearization_relative_error']))
        selected = case['selected_fraction']
        assert (selected is not None) == case['trials'][-1]['accepted']
        fresh = None
        if selected is not None:
            checked['fresh_calls'] += 1
            close(arrays[f'case{index}_uncommitted_candidate'], np.clip(theta + selected * direction, lower, 1.))
            check = case['fresh_confirmation']
            passed, _, _, remaining = gate(check, report['baseline_risks']['FRESH'],
                np.asarray(report['baseline_native']['FRESH']), np.asarray(report['initial_native']['FRESH']))
            assert passed == check['pass_all'] == case['fresh_pass']
            fresh = dict(pass_all=passed, joint_whitened_remaining_fraction=remaining,
                native_ratios=check['native_ratios'], M0_anchor_ratios=check['initial_anchor_ratios'])
        cases.append(dict(step=case['difference_step'], selected_fraction=selected, fresh=fresh,
            peak_changed_fraction=float(peak.sum() / (54 * 3 * 4096)),
            window_changed_fraction=float(window.sum() / (54 * 3 * 4096)), trials=trial_summary))
    for comparison, (left, right) in zip(report['scale_comparisons'], ((0, 1), (1, 2))):
        a, b = arrays[f'case{left}_jacobian'], arrays[f'case{right}_jacobian']
        na, nb = arrays[f'case{left}_native_jacobian'], arrays[f'case{right}_native_jacobian']
        close(np.linalg.norm(a - b) / np.linalg.norm(a), comparison['joint_relative_Frobenius_difference'])
        close((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)), comparison['joint_cosine'])
        norms_a, norms_b = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
        close(norms_a, comparison['native_row_norms_left']); close(norms_b, comparison['native_row_norms_right'])
        for u, w, x, y, cosine in zip(na, nb, norms_a, norms_b, comparison['native_row_cosine']):
            if x == 0 or y == 0:
                assert cosine is None
            else:
                close((u * w).sum() / (x * y), cosine)
    summary[str(seed)] = dict(sampled_fit_role_rows=sampled, fixed_theta_exact=True,
        baseline_to_M0={key: (np.asarray(report['baseline_native'][key]) / np.asarray(report['initial_native'][key])).reshape(3, 3).tolist()
                        for key in ('FIT', 'CHECK', 'FRESH')},
        scale_comparisons=report['scale_comparisons'], cases=cases)

calls = 12 + 2 * checked['FD_columns'] + 2 * checked['trials'] + checked['fresh_calls']
c = result['execution_counts']
assert calls == c['joint_response_calls'] == 747
assert calls * 4096 == c['joint_response_rows']
assert calls * 3 == c['generated_observation_calls']
assert calls * 3 * 4096 == c['generated_observation_rows']
assert calls * 3 * 8192 == c['native_component_rows']
assert checked['FD_columns'] == c['finite_difference_columns'] == 324
assert checked['solves'] == c['joint_direction_solves'] == 6
assert checked['trials'] == c['joint_candidate_trials'] == 42
assert result['committed_parameter_updates'] == 0 and result['scientific_acceptance'] is False
audit = dict(status='PASS_EXECUTION_AND_SAVED_MATRIX_AUDIT', native_exit=remote['native_exit'],
    result_sha256=remote['result_sha256'], source_files_verified=622, output_hashes_verified=len(remote['outputs']),
    elapsed_seconds=result['elapsed_seconds'], execution_counts=c, audit_counts=dict(checked),
    seeds=summary, committed_parameter_updates=0, scientific_acceptance=False,
    limitation='Actual trial joint residual vectors were not saved. Their reported vector error is checked only against norm bounds; native direction errors and matrix predictions are independently recomputed.',
    interpretation='Derivative matrices do not converge across the tested scales; no automatic best scale or new training is selected. Seed0 baseline already breaches the 1.01 M0 wave guard on some new FIT/CHECK hypotheses. Seed1 step .004 passes one local fresh guard, but all fresh wave ratios to M0 exceed 1. No original final acceptance is established.',
    next_action='Register zero-update readout-operator attribution on the same frozen inputs: exact hard replay, frozen peak indices, and continuous-voltage frozen-peak native diagnostics. Keep original evaluation untouched.')
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: audit[key] for key in ('status', 'source_files_verified', 'output_hashes_verified', 'elapsed_seconds', 'audit_counts', 'scientific_acceptance')}, indent=2))
