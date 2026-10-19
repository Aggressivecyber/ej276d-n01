from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'collected/run_01'
p = json.loads((ROOT / 'configs/joint_response_01_secant_trust_fix01_20261019.json').read_text())
result = json.loads((OUT / 'RESULT.json').read_text())
collection = json.loads((HERE / 'COLLECTION_REVIEW.json').read_text())
assert hashlib.sha256((OUT / 'RESULT.json').read_bytes()).hexdigest() == collection['result_sha256']
assert result['config_sha256'] == hashlib.sha256((ROOT / 'configs/joint_response_01_secant_trust_fix01_20261019.json').read_bytes()).hexdigest()
assert collection['native_exit']['exit_code'] == 0
assert result['elapsed_seconds'] <= p['resources']['internal_seconds']
assert collection['native_exit']['peak_rss_bytes'] <= p['resources']['RSS_GiB'] * 1024**3
assert result['peak_GPU_reserved_bytes'] <= p['resources']['GPU_GiB'] * 1024**3
assert collection['output_bytes'] <= p['resources']['output_MiB'] * 1024**2
for key, count in result['execution_counts'].items():
    assert count <= p['operation_budget'][key], key
roles_path = ROOT / 'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(roles_path.read_bytes()).hexdigest() == p['source_pins'][p['roles_file']]
with np.load(roles_path, allow_pickle=False) as archive:
    roles = {key: archive[key].copy() for key in ('source_train_rows', 'real_train_rows')}
counts = Counter()
comparisons, initial_ratios = [], {}
limits = np.array([1.01, 1.01, 1.10])[None, :]
hypotheses = ('fast', 'slow', 'common_level')
blocks = ('wave', 'charge_morphology', 'peak_codes', 'baseline', 'joint')
for seed in range(2):
    with np.load(OUT / f'seed{seed}_SECANT_TRUST.npz', allow_pickle=False) as archive:
        a = {key: archive[key].copy() for key in archive.files}
    report = json.loads((OUT / f'seed{seed}_SECANT_TRUST.json').read_text())
    for key, value in a.items():
        assert np.isfinite(value).all(), key
        counts['finite_arrays'] += 1
    with np.load(ROOT / f'receipts/joint_response_01_20261019/wake_20261019_1010_01/collected/run_01/seed{seed}_READOUT_ATTRIBUTION.npz', allow_pickle=False) as archive:
        old = {key: archive[key].copy() for key in archive.files}
    for key, old_key in (('theta', 'theta'), ('FIT_source_parents', 'FIT_source_parents'),
            ('FIT_real_rows', 'FIT_real_rows'), ('FIT_cholesky', 'cholesky'),
            ('FIT_baseline_residual', 'original_baseline_residual'), ('FIT_baseline_native', 'original_baseline_native'),
            ('original_jacobian', 'case0_original_jacobian'), ('original_native_jacobian', 'case0_original_native_jacobian'),
            ('initial_secant_actual_residual', 'case0_probe1_original_actual_residual'),
            ('initial_secant_actual_native', 'case0_probe1_original_actual_native')):
        np.testing.assert_array_equal(a[key], old[old_key])
        counts['exact_saved_inputs'] += 1
    np.testing.assert_array_equal(a['initial_secant_step'], old['case0_probe1_proposal'] - old['theta'])
    theta, r, n = a['theta'], a['FIT_baseline_residual'], a['FIT_baseline_native']
    assert a['lower'][-1] == 0 and (a['lower'][:-1] == -1).all() and (a['upper'] == 1).all()
    initial_ratios[str(seed)] = {}
    risk_records, checks = [], []
    for label in ('FIT', 'CHECK', 'FRESH'):
        for suffix, role_key, count, offset in (('source_parents', 'source_train_rows', 1024, 100),
                ('real_rows', 'real_train_rows', 4096, 200)):
            selected = a[label + '_' + suffix]
            assert selected.shape == (count,) and len(np.unique(selected)) == count
            assert np.isin(selected, roles[role_key]).all()
            counts['fitting_role_rows'] += count
            if label != 'FIT':
                rng_seed = p['new_role_seed_bases'][label] + seed * 100000 + offset
                order = torch.randperm(len(roles[role_key]), generator=torch.Generator().manual_seed(rng_seed))[:count].numpy()
                np.testing.assert_array_equal(selected, roles[role_key][order])
                counts['new_draws_exactly_replayed'] += 1
        if label != 'FIT':
            assert report['native_projection_contexts'][label]['seed'] == p['native_check_projection_seeds'][label]
        initial_ratios[str(seed)][label] = (a[label + '_baseline_native'] / a[label + '_M0_anchor_native']).reshape(3, 3).tolist()
        risk_records.append((label, a[label + '_baseline_residual'], report['baseline_risks'][label]))
        np.testing.assert_array_equal(report['baseline_native'][label], a[label + '_baseline_native'])
        np.testing.assert_array_equal(report['M0_anchor_native'][label], a[label + '_M0_anchor_native'])
    for arm, max_trials in p['arm_max_trials'].items():
        J, N = a['original_jacobian'].copy(), a['original_native_jacobian'].copy()
        if arm == 'secant_corrected':
            s = a['initial_secant_step']
            J += np.outer(a['initial_secant_actual_residual'] - r - J @ s, s) / (s @ s)
            N += np.outer(a['initial_secant_actual_native'] - n - N @ s, s) / (s @ s)
            np.testing.assert_allclose(J @ s, a['initial_secant_actual_residual'] - r, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(N @ s, a['initial_secant_actual_native'] - n, rtol=1e-11, atol=1e-13)
            counts['secant_corrections_recomputed'] += 1
        ar = report['arms'][arm]
        assert ar['current_solve'] is None
        assert 1 <= len(ar['trials']) <= max_trials
        accepted = []
        for i, trial in enumerate(ar['trials']):
            prefix = f'{arm}_trial{i}_'
            radius = p['initial_radius'] * p['rejection_radius_factor']**i
            assert trial['radius'] == radius
            np.testing.assert_allclose(a[prefix + 'jacobian'], J, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(a[prefix + 'native_jacobian'], N, rtol=1e-10, atol=1e-11)
            counts['matrices_recomputed'] += 2
            proposal = a[prefix + 'proposal']
            s = proposal - theta
            assert (proposal >= a['lower'] - 1e-12).all() and (proposal <= a['upper'] + 1e-12).all()
            assert np.linalg.norm(s) <= radius * np.sqrt(1 + 1e-6)
            predicted_r, predicted_n = r + J @ s, n + N @ s
            np.testing.assert_allclose(a[prefix + 'predicted_residual'], predicted_r, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(a[prefix + 'predicted_native'], predicted_n, rtol=1e-10, atol=1e-11)
            counts['predicted_vectors_recomputed'] += 2
            g = trial['geometry']
            assert g['variable_scale'] == radius
            np.testing.assert_allclose(g['step_norm'], np.linalg.norm(s), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(g['unit_step_norm'] * radius, g['step_norm'], rtol=1e-12)
            ridge = p['solver']['ridge_fraction'] * np.diag(J.T @ J).mean()
            np.testing.assert_allclose(g['ridge'], ridge, rtol=1e-10)
            np.testing.assert_allclose(g['unit_ridge'] / radius**2, ridge, rtol=1e-10)
            np.testing.assert_allclose(g['predicted_joint_remaining_fraction'], predicted_r @ predicted_r / (r @ r), rtol=1e-10)
            assert ((predicted_n - n) / n <= 1e-6).all()
            L = a['FIT_cholesky']
            for ih, h in enumerate(hypotheses):
                part = slice(ih * len(L), (ih + 1) * len(L))
                norm = np.square(L).sum(1)
                raw0, raw1 = L @ (r[part] * np.sqrt(3)), L @ (predicted_r[part] * np.sqrt(3))
                for ib, block in enumerate(blocks[:4]):
                    sl = slice(ib * 56, (ib + 1) * 56)
                    cap = 1.15 * np.mean(raw0[sl]**2 / norm[sl]) + 2. / 4096
                    risk = np.mean(raw1[sl]**2 / norm[sl])
                    np.testing.assert_allclose(g['macro_limits'][h + '/' + block], cap, rtol=1e-10)
                    np.testing.assert_allclose(g['predicted_macro_risks'][h + '/' + block], risk, rtol=1e-10)
                    assert risk <= cap * (1 + 1e-6)
            fit_r, fit_n = a[prefix + 'FIT_actual_residual'], a[prefix + 'FIT_actual_native']
            predicted_gain, actual_gain = r @ r - predicted_r @ predicted_r, r @ r - fit_r @ fit_r
            np.testing.assert_allclose(trial['predicted_joint_reduction'], predicted_gain, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(trial['actual_joint_reduction'], actual_gain, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(trial['FIT_joint_vector_relative_error'], np.linalg.norm(fit_r - predicted_r) / np.linalg.norm(r), rtol=1e-10)
            np.testing.assert_allclose(trial['FIT_native_relative_error'], ((fit_n - predicted_n) / n).reshape(3, 3), rtol=1e-9, atol=1e-11)
            if predicted_gain > 0:
                np.testing.assert_allclose(trial['reduction_ratio'], actual_gain / predicted_gain, rtol=1e-9, atol=1e-11)
            else:
                assert trial['reduction_ratio'] is None
            for label, check in trial['checks'].items():
                risk_records.append((label, a[prefix + label + '_actual_residual'], check['risks']))
                checks.append((arm, i, label, a[prefix + label + '_actual_native'], check))
            assert trial['accepted'] == all(c['accepted'] for c in trial['checks'].values())
            if trial['accepted']:
                accepted.append(i)
            if arm == 'secant_corrected' and not trial['accepted'] and i + 1 < max_trials:
                J += np.outer(fit_r - r - J @ s, s) / (s @ s)
                N += np.outer(fit_n - n - N @ s, s) / (s @ s)
                np.testing.assert_allclose(J @ s, fit_r - r, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(N @ s, fit_n - n, rtol=1e-10, atol=1e-12)
                counts['secant_corrections_recomputed'] += 1
            counts['joint_candidates'] += 1
        if accepted:
            assert accepted == [len(ar['trials']) - 1]
            assert ar['selected_trial'] == accepted[0] and ar['stop_reason'] == 'first_FIT_CHECK_pass'
            check = ar['fresh_confirmation']
            np.testing.assert_array_equal(a[arm + '_uncommitted_candidate'], a[f'{arm}_trial{accepted[0]}_proposal'])
            risk_records.append(('FRESH', a[arm + '_FRESH_actual_residual'], check['risks']))
            checks.append((arm, accepted[0], 'FRESH', a[arm + '_FRESH_actual_native'], check))
            assert ar['fresh_pass'] == check['accepted']
            counts['FRESH_checks'] += 1
        else:
            assert ar['selected_trial'] is None and not ar['fresh_pass']
            assert ar['stop_reason'] == 'trial_budget' and len(ar['trials']) == max_trials
        summary = result['summaries'][str(seed)][arm]
        assert summary == dict(selected_trial=ar['selected_trial'], fresh_pass=ar['fresh_pass'],
            trials=len(ar['trials']), stop_reason=ar['stop_reason'])
    for label, vector, recorded in risk_records:
        L = a[label + '_cholesky']
        variance = np.square(L).sum(1)
        for ih, h in enumerate(hypotheses):
            white = vector[ih * len(L):(ih + 1) * len(L)] * np.sqrt(3)
            raw = L @ white
            np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-11)
            for ib, block in enumerate(blocks):
                part = slice(ib * 56, (ib + 1) * 56) if ib < 4 else slice(224, 272)
                np.testing.assert_allclose(recorded[h][block], np.mean(raw[part]**2 / variance[part]), rtol=1e-10, atol=1e-12)
            counts['risk_views_recomputed'] += 1
    for arm, i, label, native, check in checks:
        np.testing.assert_array_equal(native, check['native'])
        ratios = (native / a[label + '_baseline_native']).reshape(3, 3)
        anchored = (native / a[label + '_M0_anchor_native']).reshape(3, 3)
        np.testing.assert_allclose(ratios, check['native_ratios'], rtol=1e-12)
        np.testing.assert_allclose(anchored, check['initial_anchor_ratios'], rtol=1e-12)
        native_pass = (ratios <= limits).all() and (anchored <= limits).all()
        assert bool(native_pass) == check['native_pass']
        old, new, trust = report['baseline_risks'][label], check['risks'], check['trust']
        old_mean, new_mean = (np.mean([risk[h]['joint_whitened'] for h in hypotheses]) for risk in (old, new))
        block_pass = all(new[h][block] <= 1.15 * old[h][block] + 2. / 4096 for h in hypotheses for block in blocks[:4])
        assert trust['denominator'] == 4096
        trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(
            value for flags in trust['flag_count_changes'].values() for value in flags.values()) <= max(4, .005 * 4096)
        assert trust['pass'] == bool(trust_pass)
        joint_pass = trust_pass and new_mean <= .99 * old_mean and block_pass
        assert bool(joint_pass) == check['original_joint_accept']
        assert bool(joint_pass and native_pass) == check['accepted']
        comparisons.append(dict(seed=seed, arm=arm, trial=i, label=label,
            joint_ratio=new_mean / old_mean, original_joint_accept=bool(joint_pass), native_pass=bool(native_pass),
            accepted=check['accepted'], current_fail_indices=np.argwhere(ratios > limits).tolist(),
            M0_anchor_fail_indices=np.argwhere(anchored > limits).tolist(),
            native_ratios=ratios.tolist(), M0_anchor_ratios=anchored.tolist()))
        counts['acceptance_checks_recomputed'] += 1
assert counts['joint_candidates'] == result['execution_counts']['joint_candidate_trials'] == 12
assert result['execution_counts']['joint_direction_solve_attempts'] == result['execution_counts']['joint_direction_solves'] == 12
assert counts['secant_corrections_recomputed'] == result['execution_counts']['secant_matrix_corrections'] == 10
assert counts['FRESH_checks'] == result['execution_counts']['FRESH_candidate_checks'] == 1
assert result['execution_counts']['joint_response_calls'] == 12 + 2 * counts['joint_candidates'] + counts['FRESH_checks']
assert result['execution_counts']['joint_response_rows'] == 4096 * result['execution_counts']['joint_response_calls']
assert result['execution_counts']['generated_observation_calls'] == 3 * result['execution_counts']['joint_response_calls']
assert result['execution_counts']['generated_observation_rows'] == 3 * result['execution_counts']['joint_response_rows']
assert result['execution_counts']['native_component_rows'] == 2 * result['execution_counts']['generated_observation_rows']
assert result['execution_counts']['hard_endpoint_replays_verified'] == 2
assert counts['fitting_role_rows'] == 30720
assert not result['corrected_two_seed_FRESH_pass'] and not result['scientific_acceptance'] and result['committed_parameter_updates'] == 0
review = dict(collection, status='PASS_EXECUTION_AND_FULL_SAVED_NUMERIC_AUDIT_LOCAL_METHOD_FAILED',
    array_numeric_audit_pending=False, audit_counts=dict(counts), initial_native_ratios_to_M0=initial_ratios,
    acceptance_checks=comparisons, scientific_acceptance=False, committed_parameter_updates=0,
    limitations=['Output RMS and flags are consistency checks of saved summaries; generated waveforms were not persisted.',
        'Native distances are saved values whose ratios and predictions were independently recomputed; native metric extraction was not rerun.',
        'CHECK/FRESH use fitting-role records and frozen full-TRAIN components; neither is full-pipeline unseen validation.'])
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(dict(status=review['status'], audit_counts=dict(counts), summaries=result['summaries'],
    initial_native_ratios_to_M0=initial_ratios, FRESH=[v for v in comparisons if v['label'] == 'FRESH'])))
