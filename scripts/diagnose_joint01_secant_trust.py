"""Bounded coupled secant correction at fixed endpoints; no committed updates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks
from n01.joint01_native_step import native_joint_step
from n01.observation008 import observe_existing_carrier
from diagnose_joint01_native_direction import native_values
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents, output_trust, accept_joint

ROOT = Path(__file__).resolve().parents[1]


def hard_check(p, model, proposal, bank, target, features, maps, baseline, old_risk,
               old_native, anchor, setup, counts):
    obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
    means = feature_means(obs, maps)
    residual = residual_vector(means, features, maps)
    risk = risks(means, features, maps)
    native = native_values(obs, target, setup, counts)
    trust = output_trust(baseline, obs, model, bank['rows'], p)
    ratios, anchored = (native / old_native).reshape(3, 3), (native / anchor).reshape(3, 3)
    native_pass = all(bool((value[:, :2] <= p['native_guard']['wave_QPSD_factor']).all()
        and (value[:, 2] <= p['native_guard']['noise_factor']).all()) for value in (ratios, anchored))
    joint_pass = accept_joint(old_risk, risk, trust, 4096, p)
    return residual, native, dict(risks=risk, native=native.tolist(), native_ratios=ratios.tolist(),
        initial_anchor_ratios=anchored.tolist(), native_pass=native_pass, trust=trust,
        original_joint_accept=joint_pass, accepted=native_pass and joint_pass)


def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        source_pool, real_pool = [torch.as_tensor(archive[key], device=device)
            for key in ('source_train_rows', 'real_train_rows')]
    cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
        q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
    for key in ('window_V', 'baseline100'):
        if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
            raise ValueError('REAL readout differs from frozen loader: ' + key)
    real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
        flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
    counts['REAL_observable_replay_calls'] += 1
    counts['REAL_observable_replay_rows'] += len(real['Q'])
    del setup['real'], cpu_real
    blocks = observable_blocks(select_real(real, real_pool))
    maps = {}
    for label, seed in p['feature_seeds'].items():
        fmap = JointFeatureMap01(blocks, seed=seed)
        maps[label] = {h: fmap for h in HYPOTHESES}
        counts['feature_maps_constructed'] += 1
    del blocks
    contexts = {'FIT': setup['context']}
    for label, seed in p['native_check_projection_seeds'].items():
        contexts[label] = setup['loss'].create_evaluation_context(seed=seed)
        counts['new_native_projection_contexts'] += 1
    role_setup = {label: dict(setup, context=context) for label, context in contexts.items()}
    summaries = {}
    for seed_index, model in enumerate(setup['joint_models']):
        with np.load(ROOT / p['readout_arrays'][seed_index], allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        theta = torch.as_tensor(saved['theta'], device=device)
        selections = {'FIT': (torch.as_tensor(saved['FIT_source_parents'], device=device),
            torch.as_tensor(saved['FIT_real_rows'], device=device), p['FIT_seed_base'] + seed_index * 100000 + 1000)}
        for label, seed_base in p['new_role_seed_bases'].items():
            seed = seed_base + seed_index * 100000
            selections[label] = (draw_parents(source_pool, 1024, seed + 100),
                draw_parents(real_pool, 4096, seed + 200), seed + 1000)
        banks, targets, features, baseline, old_risks, old_native, anchors, residuals = {}, {}, {}, {}, {}, {}, {}, {}
        arrays = dict(theta=theta.cpu().numpy())
        for label, (parents, rows, rng_seed) in selections.items():
            if not np.isin(parents.cpu().numpy(), source_pool.cpu().numpy()).all() or not np.isin(rows.cpu().numpy(), real_pool.cpu().numpy()).all():
                raise ValueError('Secant trust rows are outside fitting roles')
            arrays[label + '_source_parents'], arrays[label + '_real_rows'] = parents.cpu().numpy(), rows.cpu().numpy()
            arrays[label + '_cholesky'] = maps[label][HYPOTHESES[0]].cholesky.cpu().numpy()
            counts['REAL_fit_feature_rows' if label == 'FIT' else 'REAL_CHECK_feature_rows'] += len(rows)
            banks[label] = random_bank(setup, parents, 4, rng_seed, counts)
            targets[label] = select_real(real, rows)
            features[label] = feature_means({h: targets[label] for h in HYPOTHESES}, maps[label])
            baseline[label] = prediction(model, theta, banks[label], setup['loss'].q_floor, counts)
            means = feature_means(baseline[label], maps[label])
            old_risks[label] = risks(means, features[label], maps[label])
            residuals[label] = residual_vector(means, features[label], maps[label])
            old_native[label] = native_values(baseline[label], targets[label], role_setup[label], counts)
            initial = prediction(model, model.initial(), banks[label], setup['loss'].q_floor, counts)
            anchors[label] = native_values(initial, targets[label], role_setup[label], counts)
            arrays[label + '_baseline_residual'] = residuals[label].cpu().numpy()
            arrays[label + '_baseline_native'] = old_native[label].cpu().numpy()
            arrays[label + '_M0_anchor_native'] = anchors[label].cpu().numpy()
        r, n = residuals['FIT'], old_native['FIT']
        if not np.array_equal(r.cpu().numpy(), saved['original_baseline_residual']):
            raise ValueError('FIT residual does not exactly replay attribution')
        if not np.array_equal(n.cpu().numpy(), saved['original_baseline_native']):
            raise ValueError('FIT native does not exactly replay attribution')
        counts['hard_endpoint_replays_verified'] += 1
        original_J = torch.as_tensor(saved['case0_original_jacobian'], device=device)
        original_N = torch.as_tensor(saved['case0_original_native_jacobian'], device=device)
        seed_step = torch.as_tensor(saved['case0_probe1_proposal'], device=device) - theta
        seed_r = torch.as_tensor(saved['case0_probe1_original_actual_residual'], device=device)
        seed_n = torch.as_tensor(saved['case0_probe1_original_actual_native'], device=device)
        arrays.update(original_jacobian=original_J.cpu().numpy(), original_native_jacobian=original_N.cpu().numpy(),
            initial_secant_step=seed_step.cpu().numpy(), initial_secant_actual_residual=seed_r.cpu().numpy(),
            initial_secant_actual_native=seed_n.cpu().numpy())
        report = dict(seed=seed_index, endpoint_exact_replay=True, baseline_risks=old_risks,
            baseline_native={key: value.tolist() for key, value in old_native.items()},
            M0_anchor_native={key: value.tolist() for key, value in anchors.items()},
            native_projection_contexts={key: value.metadata() for key, value in contexts.items()}, arms={})
        for arm, max_trials in p['arm_max_trials'].items():
            J, N = original_J.clone(), original_N.clone()
            if arm == 'secant_corrected':
                J += torch.outer(seed_r - r - J @ seed_step, seed_step) / seed_step.square().sum()
                N += torch.outer(seed_n - n - N @ seed_step, seed_step) / seed_step.square().sum()
                counts['secant_matrix_corrections'] += 1
                if max(float((J @ seed_step - (seed_r - r)).abs().max()),
                       float((N @ seed_step - (seed_n - n)).abs().max())) > 1e-11:
                    raise ValueError('Initial coupled secant identity failed')
            radius = p['initial_radius']
            arm_report = dict(trials=[], selected_trial=None, fresh_pass=False, stop_reason='trial_budget')
            selected = None
            for trial_index in range(max_trials):
                check_budget(p, counts, started)
                delta, geometry = native_joint_step(r, J, maps['FIT'], n, N, theta, model.lower, model.upper,
                    ridge_fraction=p['solver']['ridge_fraction'], radius=radius,
                    macro_factor=p['solver']['macro_degradation_factor'], sample_count=4096,
                    max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
                counts['joint_direction_solves'] += 1
                prefix = f'{arm}_trial{trial_index}_'
                arrays[prefix + 'jacobian'], arrays[prefix + 'native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
                if float(delta.square().sum()) == 0:
                    arm_report['stop_reason'] = 'zero_joint_direction'
                    arm_report['zero_direction_geometry'] = geometry
                    break
                proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + delta))
                step = proposal - theta
                counts['joint_candidate_trials'] += 1
                arrays[prefix + 'proposal'] = proposal.cpu().numpy()
                predicted_r, predicted_n = r + J @ step, n + N @ step
                arrays[prefix + 'predicted_residual'] = predicted_r.cpu().numpy()
                arrays[prefix + 'predicted_native'] = predicted_n.cpu().numpy()
                trial = dict(radius=radius, geometry=geometry, checks={})
                for label in ('FIT', 'CHECK'):
                    actual_r, actual_n, check = hard_check(p, model, proposal, banks[label], targets[label],
                        features[label], maps[label], baseline[label], old_risks[label], old_native[label],
                        anchors[label], role_setup[label], counts)
                    arrays[prefix + label + '_actual_residual'] = actual_r.cpu().numpy()
                    arrays[prefix + label + '_actual_native'] = actual_n.cpu().numpy()
                    trial['checks'][label] = check
                    if label == 'FIT':
                        fit_r, fit_n = actual_r, actual_n
                trial['accepted'] = all(check['accepted'] for check in trial['checks'].values())
                predicted_gain = float(r.square().sum() - predicted_r.square().sum())
                actual_gain = float(r.square().sum() - fit_r.square().sum())
                trial.update(predicted_joint_reduction=predicted_gain, actual_joint_reduction=actual_gain,
                    reduction_ratio=None if predicted_gain <= 0 else actual_gain / predicted_gain,
                    FIT_joint_vector_relative_error=float((fit_r - predicted_r).norm() / r.norm()),
                    FIT_native_relative_error=((fit_n - predicted_n) / n).reshape(3, 3).tolist())
                arm_report['trials'].append(trial)
                progress_file(out, counts, 'coupled_secant_trust_candidate', seed=seed_index, arm=arm,
                    trial=trial_index + 1, radius=radius, accepted=trial['accepted'])
                if trial['accepted']:
                    selected = proposal
                    arm_report['selected_trial'] = trial_index
                    arm_report['stop_reason'] = 'first_FIT_CHECK_pass'
                    arrays[arm + '_uncommitted_candidate'] = proposal.cpu().numpy()
                    break
                if arm == 'secant_corrected' and trial_index + 1 < max_trials:
                    J += torch.outer(fit_r - r - J @ step, step) / step.square().sum()
                    N += torch.outer(fit_n - n - N @ step, step) / step.square().sum()
                    counts['secant_matrix_corrections'] += 1
                    if max(float((J @ step - (fit_r - r)).abs().max()),
                           float((N @ step - (fit_n - n)).abs().max())) > 1e-11:
                        raise ValueError('Adaptive coupled secant identity failed')
                    radius *= p['rejection_radius_factor']
            if selected is not None:
                actual_r, actual_n, check = hard_check(p, model, selected, banks['FRESH'], targets['FRESH'],
                    features['FRESH'], maps['FRESH'], baseline['FRESH'], old_risks['FRESH'], old_native['FRESH'],
                    anchors['FRESH'], role_setup['FRESH'], counts)
                counts['FRESH_candidate_checks'] += 1
                arrays[arm + '_FRESH_actual_residual'], arrays[arm + '_FRESH_actual_native'] = actual_r.cpu().numpy(), actual_n.cpu().numpy()
                arm_report['fresh_confirmation'] = check
                arm_report['fresh_pass'] = check['accepted']
            report['arms'][arm] = arm_report
            np.savez(out / f'seed{seed_index}_SECANT_TRUST.npz', **arrays)
            write_json(out / f'seed{seed_index}_SECANT_TRUST.json', report)
        summaries[str(seed_index)] = {arm: dict(selected_trial=value['selected_trial'], fresh_pass=value['fresh_pass'],
            trials=len(value['trials']), stop_reason=value['stop_reason']) for arm, value in report['arms'].items()}
    return dict(status='COMPLETE', summaries=summaries,
        corrected_two_seed_FRESH_pass=all(value['secant_corrected']['fresh_pass'] for value in summaries.values()),
        committed_parameter_updates=0, scientific_acceptance=False, automatic_training=False,
        original_final_acceptance_gates_unchanged=True, generator_readout_and_metric_code_unchanged=True,
        control_scope='One raw-radius-only first-step control per seed, not an equal-budget full optimization comparison.',
        FRESH_role='New random library, joint feature map and native projection on fitting-role records; no report holdout.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered secant-trust config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned secant-trust input changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            for key in ('encoding', 'references', 'models', 'baseline', 'heads'):
                del setup[key]
            progress_file(args.out, counts, 'secant_trust_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered secant-trust output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                committed_parameter_updates=0, scientific_acceptance=False, automatic_restart=False))
