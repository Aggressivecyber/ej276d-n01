"""Separate generator secant error from readout/metric remainder on saved directions."""
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
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
from diagnose_joint01_readout_attribution import readout_values, JOINT_VIEWS, NATIVE_VIEWS
from run_joint_response01 import load_joint_setup, random_bank, feature_means, progress_file, HYPOTHESES
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered signal/metric-attribution config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned attribution input changed: ' + name)
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
            progress_file(args.out, counts, 'signal_metric_attribution_setup_loaded')
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
                with np.load(ROOT / p['readout_arrays'][seed_index], allow_pickle=False) as archive:
                    saved = {key: archive[key].copy() for key in archive.files}
                theta = torch.as_tensor(saved['theta'], device=device)
                parents, rows = [torch.as_tensor(saved[key], device=device)
                    for key in ('FIT_source_parents', 'FIT_real_rows')]
                if not np.isin(parents.cpu().numpy(), source_pool).all() or not np.isin(rows.cpu().numpy(), real_pool).all():
                    raise ValueError('Frozen attribution rows are outside fitting roles')
                counts['REAL_fit_feature_rows'] += len(rows)
                bank = random_bank(setup, parents, 4, p['training_seed'] + seed_index * 100000 + 1000, counts)
                target = select_real(real, rows)
                target_features = feature_means({h: target for h in HYPOTHESES}, maps)
                counts['joint_response_calls'] += 1
                counts['joint_response_rows'] += len(bank['rows'])
                base_detail = model(theta, bank['rows'], bank['z'], return_details=True)
                baseline_signal, baseline_scale = base_detail['signal'], base_detail['noise_scale']
                observations = {}
                for h in HYPOTHESES:
                    observations[h] = observe_generated_carrier(baseline_signal + baseline_scale * bank['noise'][h],
                        q_floor=setup['loss'].q_floor, return_carrier=True)
                    counts['generated_observation_calls'] += 1
                    counts['generated_observation_rows'] += len(bank['rows'])
                peaks = {h: torch.as_tensor(saved['fixed_peak_indices'][i], device=device) for i, h in enumerate(HYPOTHESES)}
                if any(not torch.equal(observations[h]['peak_index'], peaks[h]) for h in HYPOTHESES):
                    raise ValueError('Endpoint peak indices differ from saved readout diagnostic')
                baseline_joint, baseline_native, baseline_diagnostic = readout_values(observations, peaks,
                    maps, target_features, target, setup, counts)
                for view in JOINT_VIEWS:
                    if not np.array_equal(baseline_joint[view].cpu().numpy(), saved[view + '_baseline_residual']):
                        raise ValueError('Endpoint joint residual replay failed: ' + view)
                for view in NATIVE_VIEWS:
                    if not np.array_equal(baseline_native[view].cpu().numpy(), saved[view + '_baseline_native']):
                        raise ValueError('Endpoint native replay failed: ' + view)
                counts['hard_endpoint_replays_verified'] += 1
                arrays = dict(theta=theta.cpu().numpy(), FIT_source_parents=parents.cpu().numpy(),
                    FIT_real_rows=rows.cpu().numpy(), repeated_source_rows=bank['rows'].cpu().numpy(),
                    shape_latents=bank['z'].cpu().numpy(), fixed_peak_indices=saved['fixed_peak_indices'],
                    cholesky=fmap.cholesky.cpu().numpy(), baseline_signal=baseline_signal.cpu().numpy(),
                    baseline_noise_scale=baseline_scale.cpu().numpy(), target_carrier_adc=target['carrier_adc'].cpu().numpy())
                for h in HYPOTHESES:
                    arrays['bank_noise_' + h] = bank['noise'][h].cpu().numpy()
                for view in JOINT_VIEWS:
                    arrays[view + '_baseline_residual'] = baseline_joint[view].cpu().numpy()
                for view in NATIVE_VIEWS:
                    arrays[view + '_baseline_native'] = baseline_native[view].cpu().numpy()
                report = dict(seed=seed_index, endpoint_exact_replay=True, cases=[])
                del observations
                for case_index, step in enumerate(p['difference_steps']):
                    direction = torch.as_tensor(saved[f'case{case_index}_saved_direction'], device=device)
                    tangent = torch.zeros_like(baseline_signal)
                    noise_tangent = torch.zeros_like(baseline_scale)
                    signal_pair_audit, denominators = [], []
                    for column in range(model.parameter_count):
                        check_budget(p, counts, started)
                        plus, minus = theta.clone(), theta.clone()
                        plus[column] = min(float(model.upper[column]), float(theta[column]) + step)
                        minus[column] = max(float(model.lower[column]), float(theta[column]) - step)
                        denominator = plus[column] - minus[column]
                        signals = []
                        for proposal in (plus, minus):
                            signals.append(model(proposal, bank['rows'], bank['z']))
                            counts['joint_response_calls'] += 1
                            counts['joint_response_rows'] += len(bank['rows'])
                        tangent += direction[column] * (signals[0] - signals[1]) / denominator
                        noise_tangent += direction[column] * (model.noise_scale(plus) - model.noise_scale(minus)) / denominator
                        signal_pair_audit.append(torch.stack([value[:8] for value in signals]).cpu().numpy())
                        denominators.append(float(denominator))
                        counts['signal_secant_columns'] += 1
                        del signals
                        if column % 9 == 0:
                            progress_file(args.out, counts, 'signal_secant_columns', seed=seed_index,
                                difference_step=step, column=column + 1)
                    prefix = f'case{case_index}_'
                    arrays[prefix + 'saved_direction'] = direction.cpu().numpy()
                    arrays[prefix + 'signal_tangent'] = tangent.cpu().numpy()
                    arrays[prefix + 'noise_scale_tangent'] = noise_tangent.cpu().numpy()
                    arrays[prefix + 'signal_pair_audit_first8'] = np.asarray(signal_pair_audit)
                    arrays[prefix + 'secant_denominators'] = np.asarray(denominators)
                    case = dict(difference_step=step, probes=[])
                    for probe_index, fraction in enumerate(p['directional_probe_fractions']):
                        check_budget(p, counts, started)
                        proposal = theta + fraction * direction
                        if bool(((proposal < model.lower) | (proposal > model.upper)).any()):
                            raise ValueError('Saved path is not straight within bounds')
                        if not np.array_equal(proposal.cpu().numpy(), saved[prefix + f'probe{probe_index}_proposal']):
                            raise ValueError('Saved directional proposal differs')
                        actual_detail = model(proposal, bank['rows'], bank['z'], return_details=True)
                        counts['joint_response_calls'] += 1
                        counts['joint_response_rows'] += len(bank['rows'])
                        counts['directional_probe_calls'] += 1
                        actual_signal, actual_scale = actual_detail['signal'], actual_detail['noise_scale']
                        linear_signal = baseline_signal + fraction * tangent
                        linear_scale = baseline_scale + fraction * noise_tangent
                        output = {}
                        for branch, signal, scale in (('actual', actual_signal, actual_scale),
                                ('linear_signal', linear_signal, linear_scale)):
                            obs = {}
                            for h in HYPOTHESES:
                                obs[h] = observe_generated_carrier(signal + scale * bank['noise'][h],
                                    q_floor=setup['loss'].q_floor, return_carrier=True)
                                count_prefix = 'generated' if branch == 'actual' else 'linear_signal'
                                counts[count_prefix + '_observation_calls'] += 1
                                counts[count_prefix + '_observation_rows'] += len(bank['rows'])
                            joint, native, diagnostic = readout_values(obs, peaks, maps, target_features, target, setup, counts)
                            output[branch] = dict(joint=joint, native=native, diagnostic=diagnostic)
                            del obs
                        probe_prefix = prefix + f'probe{probe_index}_'
                        arrays[probe_prefix + 'proposal'] = proposal.cpu().numpy()
                        arrays[probe_prefix + 'actual_signal'] = actual_signal.cpu().numpy()
                        arrays[probe_prefix + 'actual_noise_scale'] = actual_scale.cpu().numpy()
                        arrays[probe_prefix + 'linear_noise_scale'] = linear_scale.cpu().numpy()
                        difference = actual_signal - linear_signal
                        delta_signal = actual_signal - baseline_signal
                        relative_signal_error = float(difference.norm() / baseline_signal.norm())
                        delta_norm = float(delta_signal.norm())
                        movement_error = None if delta_norm == 0 else float(difference.norm() / delta_norm)
                        moved_knots = (actual_detail['queries'].floor() != base_detail['queries'].floor()).sum(1)
                        arrays[probe_prefix + 'changed_interpolation_knots_per_row'] = moved_knots.cpu().numpy()
                        probe = dict(fraction=fraction, signal_error_relative_to_baseline_RMS=relative_signal_error,
                            signal_error_relative_to_actual_movement=movement_error,
                            mean_changed_interpolation_knots=float(moved_knots.double().mean()),
                            readout_diagnostics={key: value['diagnostic'] for key, value in output.items()},
                            joint={}, native={})
                        for view in JOINT_VIEWS:
                            actual, linear = [output[branch]['joint'][view] for branch in ('actual', 'linear_signal')]
                            if not np.array_equal(actual.cpu().numpy(), saved[probe_prefix + view + '_actual_residual']):
                                raise ValueError('Saved actual joint probe replay failed: ' + view)
                            scalar_prediction = torch.as_tensor(saved[probe_prefix + view + '_predicted_residual'], device=device)
                            arrays[probe_prefix + view + '_actual_residual'] = actual.cpu().numpy()
                            arrays[probe_prefix + view + '_linear_signal_residual'] = linear.cpu().numpy()
                            arrays[probe_prefix + view + '_scalar_prediction_residual'] = scalar_prediction.cpu().numpy()
                            denominator = baseline_joint[view].norm()
                            probe['joint'][view] = dict(total_error=float((actual - scalar_prediction).norm() / denominator),
                                signal_approximation_effect=float((actual - linear).norm() / denominator),
                                readout_metric_remainder=float((linear - scalar_prediction).norm() / denominator))
                        for view in NATIVE_VIEWS:
                            actual, linear = [output[branch]['native'][view] for branch in ('actual', 'linear_signal')]
                            if not np.array_equal(actual.cpu().numpy(), saved[probe_prefix + view + '_actual_native']):
                                raise ValueError('Saved actual native probe replay failed: ' + view)
                            scalar_prediction = torch.as_tensor(saved[probe_prefix + view + '_predicted_native'], device=device)
                            arrays[probe_prefix + view + '_actual_native'] = actual.cpu().numpy()
                            arrays[probe_prefix + view + '_linear_signal_native'] = linear.cpu().numpy()
                            arrays[probe_prefix + view + '_scalar_prediction_native'] = scalar_prediction.cpu().numpy()
                            denominator = baseline_native[view]
                            probe['native'][view] = dict(total_error=((actual - scalar_prediction) / denominator).reshape(3, 3).tolist(),
                                signal_approximation_effect=((actual - linear) / denominator).reshape(3, 3).tolist(),
                                readout_metric_remainder=((linear - scalar_prediction) / denominator).reshape(3, 3).tolist(),
                                actual_ratio=(actual / denominator).reshape(3, 3).tolist(),
                                linear_signal_ratio=(linear / denominator).reshape(3, 3).tolist())
                        counts['saved_directional_checks_reproduced'] += 1
                        case['probes'].append(probe)
                        progress_file(args.out, counts, 'signal_metric_directional_probe', seed=seed_index,
                            difference_step=step, fraction=fraction, signal_relative_error=relative_signal_error)
                    report['cases'].append(case)
                    np.savez(args.out / f'seed{seed_index}_SIGNAL_METRIC.npz', **arrays)
                    write_json(args.out / f'seed{seed_index}_SIGNAL_METRIC.json', report)
                summaries[str(seed_index)] = report
            check_budget(p, counts, started)
            result = dict(status='COMPLETE', config_sha256=args.config_sha256, summaries=summaries,
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
                committed_parameter_updates=0, scientific_acceptance=False, automatic_training=False,
                counterfactual_linear_signals_for_attribution_only=True, original_final_gates_unchanged=True,
                attribution_scope='Actual minus saved scalar prediction equals actual minus linear-signal metric plus linear-signal metric minus saved scalar prediction. This is an exact signed accounting identity, not an isolated causal derivative or proof of physical validity.')
            write_json(args.out / 'RESULT.json', result)
            if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
                raise ValueError('Registered signal/metric-attribution output bound exceeded')
            write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
                (args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                committed_parameter_updates=0, scientific_acceptance=False, automatic_restart=False))
