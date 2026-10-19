"""Bounded REAL pilot with a common native-constrained method for both model families."""
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
from n01.model008 import canonical_charge008
from diagnose_joint01_native_direction import native_values
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, HYPOTHESES)
from run_joint_response01_real import select_real, evaluate, decide, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents, output_trust, accept_joint

ROOT = Path(__file__).resolve().parents[1]


def fit_track(p, setup, model, source_pool, real_pool, maps, check_maps, real, seed, out, counts, started, label):
    theta, initial = model.initial(), model.initial()
    batch, draws, real_batch = (p['batch'][key] for key in ('source_parents', 'draws_per_parent', 'target_rows'))
    source_coverage = torch.zeros(len(setup['source']), dtype=torch.long, device=source_pool.device)
    real_coverage = torch.zeros(len(real['Q']), dtype=torch.long, device=source_pool.device)
    order = {}
    for name, pool, count, offset in (('source', source_pool, batch, 100), ('real', real_pool, real_batch, 200)):
        epochs = (p['iterations'] * count + len(pool) - 1) // len(pool)
        order[name] = torch.cat([draw_parents(pool, len(pool), seed + offset + epoch) for epoch in range(epochs)])
    trace = []
    for iteration in range(p['iterations']):
        check_budget(p, counts, started)
        parents = order['source'][iteration * batch:(iteration + 1) * batch]
        target_rows = order['real'][iteration * real_batch:(iteration + 1) * real_batch]
        check_parents = draw_parents(source_pool, batch, seed + 3000 + iteration)
        check_rows = draw_parents(real_pool, real_batch, seed + 5000 + iteration)
        targets = dict(FIT=select_real(real, target_rows), CHECK=select_real(real, check_rows))
        counts['REAL_fit_feature_rows'] += len(target_rows)
        counts['REAL_CHECK_feature_rows'] += len(check_rows)
        feature_maps = dict(FIT=maps, CHECK=check_maps)
        target_features = {part: feature_means({h: obs for h in HYPOTHESES}, feature_maps[part])
            for part, obs in targets.items()}
        banks = dict(FIT=random_bank(setup, parents, draws, seed + 1000 + 20 * iteration, counts),
            CHECK=random_bank(setup, check_parents, draws, seed + 4000 + 20 * iteration, counts))
        current_obs, current_risks, current_native, anchor_native = {}, {}, {}, {}
        for part in ('FIT', 'CHECK'):
            obs = prediction(model, theta, banks[part], setup['loss'].q_floor, counts)
            current_obs[part] = obs
            current_risks[part] = risks(feature_means(obs, feature_maps[part]), target_features[part], feature_maps[part])
            current_native[part] = native_values(obs, targets[part], setup, counts)
            anchor = prediction(model, initial, banks[part], setup['loss'].q_floor, counts)
            anchor_native[part] = native_values(anchor, targets[part], setup, counts)
        residual = residual_vector(feature_means(current_obs['FIT'], maps), target_features['FIT'], maps)
        columns, native_columns = [], []
        for j in range(model.parameter_count):
            check_budget(p, counts, started)
            plus, minus = theta.clone(), theta.clone()
            plus[j] = min(float(model.upper[j]), float(theta[j]) + p['solver']['finite_difference_step'])
            minus[j] = max(float(model.lower[j]), float(theta[j]) - p['solver']['finite_difference_step'])
            pairs, native_pairs = [], []
            for proposal in (plus, minus):
                obs = prediction(model, proposal, banks['FIT'], setup['loss'].q_floor, counts)
                pairs.append(residual_vector(feature_means(obs, maps), target_features['FIT'], maps))
                native_pairs.append(native_values(obs, targets['FIT'], setup, counts))
            columns.append((pairs[0] - pairs[1]) / (plus[j] - minus[j]))
            native_columns.append((native_pairs[0] - native_pairs[1]) / (plus[j] - minus[j]))
            counts['finite_difference_columns'] += 1
            if j % 9 == 0:
                progress_file(out, counts, 'REAL_native_joint_Jacobian', track=label, iteration=iteration + 1, column=j + 1)
        J, native_J = torch.stack(columns, 1), torch.stack(native_columns, 1)
        delta, geometry = native_joint_step(residual, J, maps, current_native['FIT'], native_J,
            theta, model.lower, model.upper, ridge_fraction=p['solver']['ridge_fraction'],
            radius=p['solver']['radius'], macro_factor=p['solver']['macro_degradation_factor'], sample_count=4096,
            max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
        counts['joint_direction_solves'] += 1
        attempts, accepted = [], False
        for fraction in p['backtracking_fractions']:
            check_budget(p, counts, started)
            counts['joint_candidate_trials'] += 1
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
            attempt = dict(fraction=fraction, checks={})
            for part in ('FIT', 'CHECK'):
                obs = prediction(model, proposal, banks[part], setup['loss'].q_floor, counts)
                risk = risks(feature_means(obs, feature_maps[part]), target_features[part], feature_maps[part])
                native = native_values(obs, targets[part], setup, counts)
                trust = output_trust(current_obs[part], obs, model, banks[part]['rows'], p)
                ratios, anchor_ratios = (native / current_native[part]).reshape(3, 3), (native / anchor_native[part]).reshape(3, 3)
                native_pass = all(bool((value[:, :2] <= p['native_guard']['wave_QPSD_factor']).all()
                    and (value[:, 2] <= p['native_guard']['noise_factor']).all()) for value in (ratios, anchor_ratios))
                joint_pass = accept_joint(current_risks[part], risk, trust, 4096, p)
                attempt['checks'][part] = dict(risks=risk, native=native.tolist(), native_ratios=ratios.tolist(),
                    initial_anchor_ratios=anchor_ratios.tolist(), native_pass=native_pass, trust=trust,
                    original_joint_accept=joint_pass, accepted=joint_pass and native_pass)
            accepted = all(entry['accepted'] for entry in attempt['checks'].values())
            attempt['accepted'] = accepted
            attempts.append(attempt)
            if accepted:
                theta = proposal
                counts['REAL_response_updates'] += 1
                break
        source_coverage.index_add_(0, parents, torch.ones_like(parents))
        real_coverage.index_add_(0, target_rows, torch.ones_like(target_rows))
        trace.append(dict(iteration=iteration, current_risks=current_risks,
            current_native={part: value.tolist() for part, value in current_native.items()},
            initial_anchor_native={part: value.tolist() for part, value in anchor_native.items()},
            attempts=attempts, accepted=accepted, geometry=geometry, parameter_norm=float(theta.norm())))
        write_json(out / (label + '_TRACE.json'), dict(track=label, iterations=trace,
            paired_supervision=False, population_weights_fitted=False, native_projection_used_for_direction_fitting=True))
        np.savez(out / (label + '_STATE.npz'), theta=theta.cpu().numpy(),
            source_coverage=source_coverage.cpu().numpy(), real_coverage=real_coverage.cpu().numpy())
        progress_file(out, counts, 'REAL_native_joint_training', track=label, iteration=iteration + 1,
            iterations=p['iterations'], accepted_updates=sum(row['accepted'] for row in trace))
    return theta, dict(track=label, accepted_updates=sum(row['accepted'] for row in trace),
        source_parents_visited=int((source_coverage > 0).sum()), allowed_source_parents=len(source_pool),
        real_records_visited=int((real_coverage > 0).sum()), allowed_real_records=len(real_pool),
        source_visit_sum=int(source_coverage.sum()), real_visit_sum=int(real_coverage.sum()))


def kernel_control(p, candidates, parents, seed_index, out, counts, started):
    device = parents.device
    projection = torch.cos(torch.arange(384, device=device, dtype=torch.float64)[:, None]
        * torch.arange(1, 17, device=device)[None] * torch.pi / 384.) / 384. ** .5
    moments, summary = {}, {}
    for library in range(2):
        generator = torch.Generator(device='cpu').manual_seed(p['kernel_seed'] + seed_index * 10000 + library * 100)
        counts['allocated_banks'] += 1
        samples = {label: [] for label in candidates}
        draws = p['kernel_draws_per_parent']
        for position, parent in enumerate(parents):
            check_budget(p, counts, started)
            rows = torch.full((draws,), int(parent), device=device, dtype=torch.long)
            z = torch.randn(draws, 8, generator=generator, dtype=torch.float32).to(device)
            counts['shape_standard_normal_values'] += z.numel()
            for label, (model, theta) in candidates.items():
                counts['joint_response_calls'] += 1
                counts['joint_response_rows'] += draws
                counts['kernel_signal_calls'] += 1
                counts['kernel_signal_rows'] += draws
                signal = model(theta, rows, z)
                reduced = signal @ projection
                centered = reduced - reduced.mean(0)
                covariance = centered.T @ centered / (draws - 1)
                samples[label].append((signal.mean(0).cpu().numpy(), covariance.cpu().numpy(),
                    float(canonical_charge008(signal).var()), float(signal.var(0).mean())))
            if position % 32 == 0:
                progress_file(out, counts, 'REAL_same_parent_kernel_control', seed=seed_index,
                    library=library, parents_complete=position + 1)
        arrays = {}
        for label, entries in samples.items():
            packed = tuple(np.stack([entry[j] for entry in entries]) for j in range(4))
            moments[library, label] = packed
            for name, array in zip(('mean', 'covariance', 'Q_variance', 'wave_variance'), packed):
                arrays[label + '_' + name] = array
        np.savez(out / f'seed{seed_index}_library{library}_KERNEL.npz', parents=parents.cpu().numpy(), **arrays)
        summary[str(library)] = {}
        for candidate, reference in (('joint', 'M0'), ('physical', 'physical_initial')):
            mean, covariance, qvar, wavevar = moments[library, candidate]
            mean0, covariance0, qvar0, wavevar0 = moments[library, reference]
            summary[str(library)][candidate] = dict(reference=reference,
                actual_MC_mean_relative_change=float(np.linalg.norm(mean - mean0) / np.linalg.norm(mean0)),
                covariance_relative_change=float(np.linalg.norm(covariance - covariance0) / np.linalg.norm(covariance0)),
                covariance_trace_ratio=float(np.trace(covariance, axis1=1, axis2=2).sum() /
                    np.trace(covariance0, axis1=1, axis2=2).sum()),
                Q_variance_ratio=float(qvar.mean() / qvar0.mean()), wave_variance_ratio=float(wavevar.mean() / wavevar0.mean()),
                covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
                Q_variance_zero_parent_count=int((qvar == 0).sum()), parents=len(parents), draws_per_parent=draws)
    precision = {}
    for label in candidates:
        a, b = moments[0, label][1], moments[1, label][1]
        precision[label] = float(np.linalg.norm(a - b) / (.5 * (np.linalg.norm(a) + np.linalg.norm(b))))
    result = dict(comparisons=summary, between_library_covariance_relative_difference=precision,
        REAL_conditional_kernel_truth_available=False, no_gate_or_variance_target_fitted_from_this_control=True,
        mean_is_actual_MC_mean=True, all_parents_retained=True)
    write_json(out / f'seed{seed_index}_KERNEL.json', result)
    return result


def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        roles = {key: archive[key].copy() for key in archive.files}
    np.savez(out / 'REAL_CALIBRATION_ROLES.npz', **roles)
    source_pool, source_holdout, real_pool, real_holdout = [torch.as_tensor(roles[key], device=device) for key in
        ('source_train_rows', 'source_heldout_rows', 'real_train_rows', 'real_heldout_rows')]
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
    feature_maps = {}
    for label, seed in p['feature_seeds'].items():
        counts['feature_maps_constructed'] += 1
        fmap = JointFeatureMap01(blocks, seed=seed)
        feature_maps[label] = {h: fmap for h in HYPOTHESES}
    del blocks
    tracks, decisions, kernels = {}, {}, {}
    for seed_index, neural in enumerate(setup['joint_models']):
        candidates = dict(M0=(neural, neural.initial()),
            physical_initial=(setup['physical_model'], setup['physical_model'].initial()))
        for name, model in (('joint', neural), ('physical', setup['physical_model'])):
            label = f'seed{seed_index}_{name}'
            endpoint, summary = fit_track(p, setup, model, source_pool, real_pool, feature_maps['FIT'],
                feature_maps['CHECK'], real, p['training_seed'] + seed_index * 100000,
                out, counts, started, label)
            candidates[name], tracks[label] = (model, endpoint), summary
        evaluation = evaluate(p, setup, seed_index, candidates, source_holdout, real, real_holdout,
            feature_maps['FINAL'], out, counts)
        decisions[str(seed_index)] = decide(p, evaluation)
        write_json(out / f'seed{seed_index}_DECISION.json', decisions[str(seed_index)])
        kernels[str(seed_index)] = kernel_control(p, candidates, source_holdout, seed_index, out, counts, started)
    passed = all(entry['pass_all'] for entry in decisions.values())
    return dict(status='COMPLETE', REAL_calibration_candidate_pass_all=passed, decisions=decisions,
        tracks=tracks, kernel_controls=kernels, role_counts={key: len(value) for key, value in roles.items()},
        validation_scope='Frozen-system posthoc SOURCE/REAL group holdout, already exposed by prior REAL run; no full-pipeline unseen claim',
        native_metric_projection_used_for_fitting=True, physical_and_neural_solver_and_iteration_budgets_matched=True,
        final_all_TRAIN_refit_pending=True, stage5_validation_pending=True, population_weights_fitted=0,
        DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_stage5_response_validation' if passed else 'diagnose_saved_joint_REAL_failure_before_more_training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered native REAL pilot config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned native REAL input changed: ' + name)
    prerequisite = json.loads((ROOT / p['native_direction_result']).read_text())
    if not prerequisite['two_seed_fresh_diagnostic_pass']:
        raise ValueError('Two-seed native direction prerequisite did not pass')
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
            progress_file(args.out, counts, 'REAL_native_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered native REAL output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(), REAL_calibration_candidate_pass_all=result['REAL_calibration_candidate_pass_all']))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False,
                scientific_acceptance=False))


if __name__ == '__main__':
    main()
