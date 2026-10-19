"""Zero-update attribution of hard-readout derivative and directional errors."""
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
from n01.observation008 import observe_existing_carrier, crop_indices_torch
from diagnose_joint01_native_direction import native_values
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]
JOINT_VIEWS = ('original', 'hard_fixed_peak')
NATIVE_VIEWS = (*JOINT_VIEWS, 'continuous_fixed_peak')


def fixed_peak_observation(obs, peaks, q_floor, *, continuous):
    measured = obs['carrier_input'] if continuous else obs['carrier_adc']
    prefix = measured[:, :100]
    centered = measured - prefix.mean(1, keepdim=True)
    indices = crop_indices_torch(peaks)
    window = centered.gather(1, indices)
    if continuous:
        # This view has no ADC-code features or invented ADC flag semantics.
        return dict(window_V=window, baseline100=prefix)
    q, tail = window.sum(1), window[:, 46:].sum(1)
    nonzero, guarded = q != 0, q.abs() < q_floor
    protected = torch.where(q < 0, -torch.full_like(q, q_floor), torch.full_like(q, q_floor))
    flags = dict(obs['flags'])
    flags.update(peak_at_search_boundary=(peaks == 112) | (peaks == 143),
        Q_nonpositive=q <= 0, PSD_undefined=~nonzero, PSD_guarded=guarded)
    result = dict(obs)
    result.update(window_V=window, waveform256=window, peak_index=peaks, crop_indices=indices,
        indices=indices, crop_start=peaks - 16, crop_stop=peaks + 240, Q=q, tail=tail,
        PSD=tail / torch.where(guarded, protected, q),
        PSD_raw=torch.where(nonzero, tail / torch.where(nonzero, q, torch.ones_like(q)),
                            torch.full_like(q, float('nan'))), flags=flags)
    return result


def readout_values(observations, peaks, maps, target_features, target, setup, counts):
    joint, native = {}, {}
    joint['original'] = residual_vector(feature_means(observations, maps), target_features, maps)
    native['original'] = native_values(observations, target, setup, counts)
    diagnostics = dict(peak_changed_rows=[], frozen_window_changed_rows=[],
        alignment_window_RMS_V=[], quantization_fixed_window_RMS_V=[])
    frozen, continuous = {}, {}
    for h in HYPOTHESES:
        original = observations[h]
        frozen[h] = fixed_peak_observation(original, peaks[h], setup['loss'].q_floor, continuous=False)
        continuous[h] = fixed_peak_observation(original, peaks[h], setup['loss'].q_floor, continuous=True)
        counts['frozen_peak_observation_calls'] += 1
        counts['frozen_peak_observation_rows'] += len(original['window_V'])
        counts['continuous_fixed_peak_observation_calls'] += 1
        counts['continuous_fixed_peak_observation_rows'] += len(original['window_V'])
        diagnostics['peak_changed_rows'].append(int((original['peak_index'] != peaks[h]).sum()))
        diagnostics['frozen_window_changed_rows'].append(int((original['window_V'] != frozen[h]['window_V']).any(1).sum()))
        diagnostics['alignment_window_RMS_V'].append(float((original['window_V'] - frozen[h]['window_V']).square().mean().sqrt()))
        diagnostics['quantization_fixed_window_RMS_V'].append(float((frozen[h]['window_V'] - continuous[h]['window_V']).square().mean().sqrt()))
    joint['hard_fixed_peak'] = residual_vector(feature_means(frozen, maps), target_features, maps)
    native['hard_fixed_peak'] = native_values(frozen, target, setup, counts)
    native['continuous_fixed_peak'] = native_values(continuous, target, setup, counts)
    return joint, native, diagnostics


def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        source_pool, real_pool = [archive[key].copy() for key in ('source_train_rows', 'real_train_rows')]
    cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
        q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
    for key in ('window_V', 'baseline100'):
        if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
            raise ValueError('REAL replay differs from frozen loader: ' + key)
    real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
        flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
    counts['REAL_observable_replay_calls'] += 1
    counts['REAL_observable_replay_rows'] += len(real['Q'])
    del setup['real'], cpu_real
    fmap = JointFeatureMap01(observable_blocks(select_real(real, torch.as_tensor(real_pool, device=device))),
                            seed=p['feature_seed'])
    maps = {h: fmap for h in HYPOTHESES}
    counts['feature_maps_constructed'] += 1
    summaries = {}
    for seed_index, model in enumerate(setup['joint_models']):
        with np.load(ROOT / p['fd_arrays'][seed_index], allow_pickle=False) as archive:
            saved = {key: archive[key].copy() for key in archive.files}
        previous = json.loads((ROOT / p['fd_reports'][seed_index]).read_text())
        theta = torch.as_tensor(saved['theta'], device=device)
        parents, rows = [torch.as_tensor(saved[key], device=device)
                         for key in ('FIT_source_parents', 'FIT_real_rows')]
        if not np.isin(parents.cpu().numpy(), source_pool).all() or not np.isin(rows.cpu().numpy(), real_pool).all():
            raise ValueError('Frozen attribution rows are outside fitting roles')
        counts['REAL_fit_feature_rows'] += len(rows)
        bank = random_bank(setup, parents, 4, p['training_seed'] + seed_index * 100000 + 1000, counts)
        target = select_real(real, rows)
        target_features = feature_means({h: target for h in HYPOTHESES}, maps)
        observations = prediction(model, theta, bank, setup['loss'].q_floor, counts)
        peaks = {h: observations[h]['peak_index'].clone() for h in HYPOTHESES}
        baseline_joint, baseline_native, baseline_diagnostics = readout_values(observations, peaks,
            maps, target_features, target, setup, counts)
        del observations
        if not torch.equal(baseline_joint['original'], baseline_joint['hard_fixed_peak']):
            raise ValueError('Frozen-peak view fails exact original endpoint feature parity')
        if not torch.equal(baseline_native['original'], baseline_native['hard_fixed_peak']):
            raise ValueError('Frozen-peak view fails exact original endpoint native parity')
        if not np.array_equal(baseline_joint['original'].cpu().numpy(), saved['residual']):
            raise ValueError('Original endpoint residual does not exactly replay saved FD diagnostic')
        if not np.array_equal(baseline_native['original'].cpu().numpy(), saved['native_values']):
            raise ValueError('Original endpoint native values do not exactly replay saved FD diagnostic')
        counts['hard_endpoint_replays_verified'] += 1
        arrays = dict(theta=theta.cpu().numpy(), FIT_source_parents=parents.cpu().numpy(),
            FIT_real_rows=rows.cpu().numpy(), cholesky=fmap.cholesky.cpu().numpy(),
            fixed_peak_indices=torch.stack([peaks[h] for h in HYPOTHESES]).cpu().numpy())
        for view in JOINT_VIEWS:
            arrays[view + '_baseline_residual'] = baseline_joint[view].cpu().numpy()
        for view in NATIVE_VIEWS:
            arrays[view + '_baseline_native'] = baseline_native[view].cpu().numpy()
        report = dict(seed=seed_index, original_endpoint_exact_replay=True,
            original_and_frozen_peak_endpoint_exact_parity=True, baseline_diagnostics=baseline_diagnostics,
            baseline_native={k: v.tolist() for k, v in baseline_native.items()}, cases=[])
        for case_index, saved_index in enumerate(p['saved_case_indices']):
            step = previous['cases'][saved_index]['difference_step']
            if step != p['difference_steps'][case_index]:
                raise ValueError('Saved finite-difference scale differs from attribution registration')
            joint_pairs = {view: [] for view in JOINT_VIEWS}
            native_pairs = {view: [] for view in NATIVE_VIEWS}
            J_columns = {view: [] for view in JOINT_VIEWS}
            N_columns = {view: [] for view in NATIVE_VIEWS}
            readout = []
            for column in range(model.parameter_count):
                check_budget(p, counts, started)
                plus, minus = theta.clone(), theta.clone()
                plus[column] = min(float(model.upper[column]), float(theta[column]) + step)
                minus[column] = max(float(model.lower[column]), float(theta[column]) - step)
                joint_values, native_results, pair_diagnostics = [], [], []
                for proposal in (plus, minus):
                    obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
                    joint, native, diagnostic = readout_values(obs, peaks, maps, target_features, target, setup, counts)
                    joint_values.append(joint); native_results.append(native); pair_diagnostics.append(diagnostic)
                    del obs
                for view in JOINT_VIEWS:
                    pair = torch.stack([values[view] for values in joint_values])
                    derivative = (pair[0] - pair[1]) / (plus[column] - minus[column])
                    joint_pairs[view].append(pair.cpu().numpy())
                    J_columns[view].append(derivative)
                for view in NATIVE_VIEWS:
                    pair = torch.stack([values[view] for values in native_results])
                    derivative = (pair[0] - pair[1]) / (plus[column] - minus[column])
                    native_pairs[view].append(pair.cpu().numpy())
                    N_columns[view].append(derivative)
                if not np.array_equal(J_columns['original'][-1].cpu().numpy(), saved[f'case{saved_index}_jacobian'][:, column]):
                    raise ValueError(f'Original joint Jacobian replay mismatch: seed{seed_index}/case{saved_index}/column{column}')
                if not np.array_equal(N_columns['original'][-1].cpu().numpy(), saved[f'case{saved_index}_native_jacobian'][:, column]):
                    raise ValueError(f'Original native Jacobian replay mismatch: seed{seed_index}/case{saved_index}/column{column}')
                counts['finite_difference_columns'] += 1
                counts['hard_replay_columns_verified'] += 1
                readout.append(pair_diagnostics)
                if column % 9 == 0:
                    progress_file(out, counts, 'readout_attribution_columns', seed=seed_index,
                        difference_step=step, column=column + 1)
            jacobians = {view: torch.stack(values, 1) for view, values in J_columns.items()}
            native_jacobians = {view: torch.stack(values, 1) for view, values in N_columns.items()}
            prefix = f'case{case_index}_'
            for view in JOINT_VIEWS:
                arrays[prefix + view + '_jacobian'] = jacobians[view].cpu().numpy()
                arrays[prefix + view + '_pair_residuals'] = np.asarray(joint_pairs[view])
            for view in NATIVE_VIEWS:
                arrays[prefix + view + '_native_jacobian'] = native_jacobians[view].cpu().numpy()
                arrays[prefix + view + '_pair_native'] = np.asarray(native_pairs[view])
            case = dict(difference_step=step, saved_case_index=saved_index,
                original_joint_and_native_columns_exact_replay=True, pair_diagnostics=readout, probes=[])
            direction = torch.as_tensor(saved[f'case{saved_index}_direction'], device=device)
            arrays[prefix + 'saved_direction'] = direction.cpu().numpy()
            for probe_index, fraction in enumerate(p['directional_probe_fractions']):
                check_budget(p, counts, started)
                proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * direction))
                obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
                joint, native, diagnostic = readout_values(obs, peaks, maps, target_features, target, setup, counts)
                del obs
                counts['directional_probe_calls'] += 1
                probe = dict(fraction=fraction, readout_diagnostics=diagnostic, joint={}, native={})
                probe_prefix = prefix + f'probe{probe_index}_'
                arrays[probe_prefix + 'proposal'] = proposal.cpu().numpy()
                for view in JOINT_VIEWS:
                    predicted = baseline_joint[view] + jacobians[view] @ (proposal - theta)
                    error = float((joint[view] - predicted).norm() / baseline_joint[view].norm())
                    arrays[probe_prefix + view + '_actual_residual'] = joint[view].cpu().numpy()
                    arrays[probe_prefix + view + '_predicted_residual'] = predicted.cpu().numpy()
                    probe['joint'][view] = dict(relative_vector_error=error,
                        actual_whitened_risk_remaining=float(joint[view].square().sum() / baseline_joint[view].square().sum()),
                        predicted_whitened_risk_remaining=float(predicted.square().sum() / baseline_joint[view].square().sum()))
                for view in NATIVE_VIEWS:
                    predicted = baseline_native[view] + native_jacobians[view] @ (proposal - theta)
                    arrays[probe_prefix + view + '_actual_native'] = native[view].cpu().numpy()
                    arrays[probe_prefix + view + '_predicted_native'] = predicted.cpu().numpy()
                    probe['native'][view] = dict(actual_ratio=(native[view] / baseline_native[view]).reshape(3, 3).tolist(),
                        predicted_ratio=(predicted / baseline_native[view]).reshape(3, 3).tolist(),
                        relative_prediction_error=((native[view] - predicted) / baseline_native[view]).reshape(3, 3).tolist())
                matched = [trial for trial in previous['cases'][saved_index]['trials'] if trial['fraction'] == fraction]
                if matched:
                    old = matched[0]['checks']['FIT']
                    if not np.array_equal(native['original'].cpu().numpy(), np.asarray(old['native'])):
                        raise ValueError('Original saved directional native values do not replay exactly')
                    if abs(probe['joint']['original']['relative_vector_error'] - old['joint_linearization_relative_error']) > 1e-12:
                        raise ValueError('Original saved directional joint residual error does not replay')
                    counts['saved_directional_checks_reproduced'] += 1
                case['probes'].append(probe)
            report['cases'].append(case)
            np.savez(out / f'seed{seed_index}_READOUT_ATTRIBUTION.npz', **arrays)
            write_json(out / f'seed{seed_index}_READOUT_ATTRIBUTION.json', report)
        comparisons = {}
        for view in JOINT_VIEWS:
            a, b = [arrays[f'case{i}_{view}_jacobian'] for i in range(2)]
            comparisons[view + '/joint'] = dict(relative_Frobenius_difference=float(np.linalg.norm(a - b) / np.linalg.norm(a)),
                cosine=float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))))
        for view in NATIVE_VIEWS:
            a, b = [arrays[f'case{i}_{view}_native_jacobian'] for i in range(2)]
            norms_a, norms_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            comparisons[view + '/native'] = dict(row_norms_left=norms_a.tolist(), row_norms_right=norms_b.tolist(),
                row_cosine=[None if x == 0 or y == 0 else float((u * v).sum() / (x * y))
                            for u, v, x, y in zip(a, b, norms_a, norms_b)],
                row_relative_difference=[None if x == 0 else float(np.linalg.norm(u - v) / x)
                                         for u, v, x in zip(a, b, norms_a)])
        report.update(scale_comparisons=comparisons, committed_parameter_updates=0,
            fitting_role_only=True, original_readout_unchanged=True,
            counterfactual_views_are_not_scientific_acceptance=True)
        write_json(out / f'seed{seed_index}_READOUT_ATTRIBUTION.json', report)
        summaries[str(seed_index)] = dict(scale_comparisons=comparisons,
            directional_probes=[dict(step=c['difference_step'], probes=c['probes']) for c in report['cases']])
    return dict(status='COMPLETE', summaries=summaries, committed_parameter_updates=0,
        scientific_acceptance=False, automatic_training=False, original_final_gates_unchanged=True,
        next_action='Use exact-replayed attribution to choose one bounded method change; counterfactual readout gains are not physical validation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered readout-attribution config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned readout-attribution input changed: ' + name)
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
            progress_file(args.out, counts, 'readout_attribution_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered readout-attribution output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                committed_parameter_updates=0, scientific_acceptance=False, automatic_restart=False))


if __name__ == '__main__':
    main()
