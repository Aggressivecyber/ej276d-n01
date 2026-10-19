from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'run_01'
p = json.loads((ROOT / 'configs/joint_response_01_multibank_m0_20261020.json').read_text())
result = json.loads((OUT / 'RESULT.json').read_text())
terminal = json.loads((HERE / 'REMOTE_TERMINAL_VERIFICATION.json').read_text())
assert result['config_sha256'] == hashlib.sha256((ROOT / 'configs/joint_response_01_multibank_m0_20261020.json').read_bytes()).hexdigest()
assert terminal['native_exit']['exit_code'] == 0
for name, item in terminal['outputs'].items():
    raw = (HERE / name).read_bytes()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], name
for name, digest in p['source_pins'].items():
    if not name.startswith('diagnostic_inputs/'):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
assert result['elapsed_seconds'] <= p['resources']['internal_seconds']
assert terminal['native_exit']['peak_rss_bytes'] <= p['resources']['RSS_GiB'] * 1024**3
assert result['peak_GPU_reserved_bytes'] <= p['resources']['GPU_GiB'] * 1024**3
assert sum(x['bytes'] for x in terminal['outputs'].values()) <= p['resources']['output_MiB'] * 1024**2
for key, value in result['execution_counts'].items():
    assert value <= p['operation_budget'][key], key
role_path = ROOT / 'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(role_path.read_bytes()).hexdigest() == p['source_pins'][p['roles_file']]
with np.load(role_path, allow_pickle=False) as archive:
    roles = {k: archive[k].copy() for k in archive.files}
with np.load(OUT / 'M0_ZERO_IDENTITY.npz', allow_pickle=False) as archive:
    zero = {k: archive[k].copy() for k in archive.files}
native_rng = json.loads((HERE / 'NATIVE_ZERO_RNG_REVIEW.json').read_text())
assert native_rng['status'] == 'PASS_NATIVE_RANDOM_STREAM_EXACT' and native_rng['values'] == 2048
assert native_rng['zero_archive_sha256'] == hashlib.sha256((OUT / 'M0_ZERO_IDENTITY.npz').read_bytes()).hexdigest()
counts = Counter()
comparisons = []
hypotheses = ('fast', 'slow', 'common_level')
blocks = ('wave', 'charge_morphology', 'peak_codes', 'baseline', 'joint')
fit_labels = ('FIT0', 'FIT1', 'FIT2')
limits = np.array([1.01, 1.01, 1.10])[None, :]
for seed in range(2):
    np.testing.assert_array_equal(zero[f'seed{seed}_rows'], roles['source_train_rows'][:128])
    assert zero[f'seed{seed}_original_signal'].tobytes() == zero[f'seed{seed}_joint_zero_signal'].tobytes()
    counts['zero_identity_rows'] += 128
    with np.load(OUT / f'seed{seed}_MULTIBANK_M0.npz', allow_pickle=False) as archive:
        a = {k: archive[k].copy() for k in archive.files}
    report = json.loads((OUT / f'seed{seed}_MULTIBANK_M0.json').read_text())
    for key, value in a.items():
        assert np.isfinite(value).all(), key
        counts['finite_arrays'] += 1
    theta = a['theta']
    assert theta.shape == (54,) and (theta == 0).all()
    assert (a['lower'][:-1] == -1).all() and a['lower'][-1] == 0 and (a['upper'] == 1).all()
    assert report['current_solve'] is None and report['M0_anchor_equals_center'] and report['FD_columns_completed'] == 54
    risk_records, checks = [], []
    for label, seed_base in p['role_seed_bases'].items():
        for suffix, role_key, number, offset in (('source_parents', 'source_train_rows', 1024, 100), ('real_rows', 'real_train_rows', 4096, 200)):
            selected = a[label + '_' + suffix]
            assert selected.shape == (number,) and len(np.unique(selected)) == number
            order = torch.randperm(len(roles[role_key]), generator=torch.Generator().manual_seed(seed_base + seed * 100000 + offset))[:number].numpy()
            np.testing.assert_array_equal(selected, roles[role_key][order])
            counts['fitting_role_entries'] += number
            counts['selection_streams_replayed'] += 1
        assert report['native_projection_contexts'][label]['seed'] == p['native_projection_seeds'][label]
        np.testing.assert_array_equal(report['baseline_native'][label], a[label + '_baseline_native'])
        risk_records.append((label, a[label + '_baseline_residual'], report['baseline_risks'][label]))
    r = np.concatenate([a[k + '_baseline_residual'] for k in fit_labels]) / np.sqrt(3)
    n = np.concatenate([a[k + '_baseline_native'] for k in fit_labels])
    np.testing.assert_allclose(r, a['stacked_baseline_residual'], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(n, a['stacked_baseline_native'])
    plus = np.minimum(a['upper'][:, None], theta[:, None] + p['finite_difference_step'] * np.eye(54))
    minus = np.maximum(a['lower'][:, None], theta[:, None] - p['finite_difference_step'] * np.eye(54))
    np.testing.assert_array_equal(a['FD_plus_theta'], plus)
    np.testing.assert_array_equal(a['FD_minus_theta'], minus)
    difference = np.diag(plus) - np.diag(minus)
    J = (a['FD_plus_residual'] - a['FD_minus_residual']) / difference
    N = (a['FD_plus_native'] - a['FD_minus_native']) / difference
    np.testing.assert_array_equal(J, a['original_jacobian'])
    np.testing.assert_array_equal(N, a['original_native_jacobian'])
    counts['FD_columns_recomputed'] += 54
    assert J.shape == (2448, 54) and N.shape == (27, 54)
    accepted = []
    for i, trial in enumerate(report['trials']):
        prefix = f'trial{i}_'
        radius = p['initial_radius'] * p['rejection_radius_factor']**i
        assert trial['radius'] == radius
        np.testing.assert_allclose(a[prefix + 'jacobian'], J, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(a[prefix + 'native_jacobian'], N, rtol=1e-10, atol=1e-11)
        proposal = a[prefix + 'proposal']; step = proposal - theta
        assert (proposal >= a['lower'] - 1e-12).all() and (proposal <= a['upper'] + 1e-12).all()
        assert np.linalg.norm(step) <= radius * np.sqrt(1 + 1e-6)
        pr, pn = r + J @ step, n + N @ step
        np.testing.assert_allclose(a[prefix + 'predicted_residual'], pr, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(a[prefix + 'predicted_native'], pn, rtol=1e-10, atol=1e-11)
        g = trial['geometry']; assert g['variable_scale'] == radius
        ridge = p['solver']['ridge_fraction'] * np.diag(J.T @ J).mean()
        np.testing.assert_allclose(g['ridge'], ridge, rtol=1e-10)
        np.testing.assert_allclose(g['unit_ridge'] / radius**2, ridge, rtol=1e-10)
        np.testing.assert_allclose(g['unit_step_norm'] * radius, np.linalg.norm(step), rtol=1e-10)
        np.testing.assert_allclose(g['predicted_joint_remaining_fraction'], pr @ pr / (r @ r), rtol=1e-10)
        assert ((pn - n) / n <= 1e-6).all()
        macro_slacks = []
        for il, label in enumerate(fit_labels):
            L = a[label + '_cholesky']; variance = np.square(L).sum(1)
            for ih, h in enumerate(hypotheses):
                sl = slice((il * 3 + ih) * len(L), (il * 3 + ih + 1) * len(L))
                raw0, raw1 = L @ (r[sl] * 3), L @ (pr[sl] * 3)
                for ib, block in enumerate(blocks[:4]):
                    take = slice(ib * 56, (ib + 1) * 56)
                    cap = 1.15 * np.mean(raw0[take]**2 / variance[take]) + 2 / 4096
                    risk = np.mean(raw1[take]**2 / variance[take]); key = label + '/' + h + '/' + block
                    np.testing.assert_allclose(g['macro_limits'][key], cap, rtol=1e-10)
                    np.testing.assert_allclose(g['predicted_macro_risks'][key], risk, rtol=1e-10)
                    macro_slacks.append(1 - risk / cap)
        slack = min(*macro_slacks, *(-(pn - n) / n), 1 - step @ step / radius**2)
        np.testing.assert_allclose(g['normalized_minimum_slack'], slack, atol=1e-12, rtol=1e-5)
        assert slack >= -1e-6
        fr = np.concatenate([a[prefix + label + '_actual_residual'] for label in fit_labels]) / np.sqrt(3)
        fn = np.concatenate([a[prefix + label + '_actual_native'] for label in fit_labels])
        predicted_gain, actual_gain = r @ r - pr @ pr, r @ r - fr @ fr
        np.testing.assert_allclose(trial['predicted_joint_reduction'], predicted_gain, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(trial['actual_joint_reduction'], actual_gain, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(trial['FIT_joint_vector_relative_error'], np.linalg.norm(fr - pr) / np.linalg.norm(r), rtol=1e-10)
        np.testing.assert_allclose(trial['FIT_native_relative_error'], ((fn - pn) / n).reshape(3, 3, 3), rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(trial['reduction_ratio'], actual_gain / predicted_gain, rtol=1e-9, atol=1e-11)
        for label, check in trial['checks'].items():
            risk_records.append((label, a[prefix + label + '_actual_residual'], check['risks']))
            checks.append((i, label, a[prefix + label + '_actual_native'], check))
        assert trial['accepted'] == all(c['accepted'] for c in trial['checks'].values())
        if trial['accepted']:
            accepted.append(i)
        elif i + 1 < len(report['trials']):
            J += np.outer(fr - r - J @ step, step) / (step @ step)
            N += np.outer(fn - n - N @ step, step) / (step @ step)
            np.testing.assert_allclose(J @ step, fr - r, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(N @ step, fn - n, rtol=1e-10, atol=1e-12)
            counts['secant_corrections_recomputed'] += 1
        counts['joint_candidates'] += 1
    assert accepted == [len(report['trials']) - 1] and report['selected_trial'] == accepted[0]
    assert report['stop_reason'] == 'first_all_FIT_CHECK_pass'
    np.testing.assert_array_equal(a['uncommitted_candidate'], a[f'trial{accepted[0]}_proposal'])
    check = report['fresh_confirmation']; assert report['fresh_pass'] == check['accepted']
    risk_records.append(('FRESH', a['FRESH_actual_residual'], check['risks']))
    checks.append((accepted[0], 'FRESH', a['FRESH_actual_native'], check))
    counts['FRESH_checks'] += 1
    for label, vector, recorded in risk_records:
        L = a[label + '_cholesky']; variance = np.square(L).sum(1)
        for ih, h in enumerate(hypotheses):
            white = vector[ih * len(L):(ih + 1) * len(L)] * np.sqrt(3); raw = L @ white
            np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-11)
            for ib, block in enumerate(blocks):
                take = slice(ib * 56, (ib + 1) * 56) if ib < 4 else slice(224, 272)
                np.testing.assert_allclose(recorded[h][block], np.mean(raw[take]**2 / variance[take]), rtol=1e-10, atol=1e-12)
            counts['risk_views_recomputed'] += 1
    for i, label, native, check in checks:
        np.testing.assert_array_equal(native, check['native'])
        ratio = (native / a[label + '_baseline_native']).reshape(3, 3)
        np.testing.assert_allclose(ratio, check['native_ratios'], rtol=1e-12)
        np.testing.assert_allclose(ratio, check['initial_anchor_ratios'], rtol=1e-12)
        native_pass = bool((ratio <= limits).all()); assert native_pass == check['native_pass']
        old, new, trust = report['baseline_risks'][label], check['risks'], check['trust']
        old_mean, new_mean = (np.mean([risk[h]['joint_whitened'] for h in hypotheses]) for risk in (old, new))
        block_pass = all(new[h][block] <= 1.15 * old[h][block] + 2 / 4096 for h in hypotheses for block in blocks[:4])
        assert trust['denominator'] == 4096
        trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(v for flags in trust['flag_count_changes'].values() for v in flags.values()) <= max(4, .005 * 4096)
        assert trust['pass'] == trust_pass
        joint_pass = bool(trust_pass and new_mean <= .99 * old_mean and block_pass)
        assert joint_pass == check['original_joint_accept'] and (joint_pass and native_pass) == check['accepted']
        comparisons.append(dict(seed=seed, trial=i, label=label, joint_ratio=new_mean/old_mean, native_ratios=ratio.tolist(), accepted=check['accepted']))
        counts['acceptance_checks_recomputed'] += 1
c = result['execution_counts']
assert counts['joint_candidates'] == c['joint_candidate_trials'] == c['joint_direction_solves'] == c['joint_direction_solve_attempts'] == 4
assert counts['secant_corrections_recomputed'] == c['secant_matrix_corrections'] == 2
assert counts['FRESH_checks'] == c['FRESH_candidate_checks'] == 2
assert counts['FD_columns_recomputed'] == c['finite_difference_columns'] == 108
assert c['FIT_bank_finite_difference_columns'] == 324
assert c['joint_response_calls'] == 2 + 10 + 2 * 3 * 108 + 4 * 4 + 2
assert c['joint_response_rows'] == 256 + (c['joint_response_calls'] - 2) * 4096
assert c['generated_observation_calls'] == 3 * (c['joint_response_calls'] - 2)
assert c['generated_observation_rows'] == c['generated_observation_calls'] * 4096
assert c['native_component_rows'] == 2 * c['generated_observation_rows']
assert result['two_seed_FRESH_pass'] and not result['scientific_acceptance'] and result['committed_parameter_updates'] == 0
review = dict(status='PASS_EXECUTION_AND_SAVED_NUMERIC_AUDIT_TWO_SEED_LOCAL_METHOD_PASS', UTC=datetime.now(timezone.utc).isoformat(), source_files_verified=terminal['source_files_verified'], outputs_rehashed=len(terminal['outputs']), output_bytes=sum(x['bytes'] for x in terminal['outputs'].values()), config_sha256=result['config_sha256'], result_sha256=terminal['result_sha256'], native_exit=terminal['native_exit'], elapsed_seconds=result['elapsed_seconds'], execution_counts=c, audit_counts=dict(counts), comparisons=comparisons, two_seed_FRESH_pass=True, committed_parameter_updates=0, scientific_acceptance=False, next_step='Prospectively register rotating three-bank REAL calibration from M0 using this fixed local method and unchanged final gates.', evidence_limit='Saved vectors verify matrix algebra, reported risks and gate decisions; trial waveforms were not saved, so feature extraction, RMS and flag summaries are not independently recomputed here. FRESH is within fitting roles and frozen full-TRAIN components.')
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'audit_counts':dict(counts),'fresh':[x for x in comparisons if x['label']=='FRESH']}))
