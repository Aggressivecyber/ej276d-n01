"""Registered JOINT_RESPONSE_01 engineering and shared joint-solver operations."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_model import JointResponse01, response_cache
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.response008 import HYPOTHESES
from run_response036 import electronic_noise
from run_response038 import load_setup

ROOT = Path(__file__).resolve().parents[1]


def progress_file(out, counts, stage, **values):
    write_json(out / 'PROGRESS.json', dict(stage=stage, UTC=datetime.now(timezone.utc).isoformat(),
                                         execution_counts=dict(counts), **values))


def load_joint_setup(p, device, counts):
    original = json.loads((ROOT / p['frozen_setup_config']).read_text(encoding='utf8'))
    setup = load_setup(original, device, counts)
    source, encoding = setup['source'], setup['encoding']
    caches = []
    for model in setup['references']:
        bases = []
        for start in range(0, len(source), 128):
            rows = torch.arange(start, min(start + 128, len(source)), device=device)
            selected = model.select_encoding(encoding, rows)
            counts['M0_head_calls'] += 1
            counts['M0_head_rows'] += len(rows)
            result = model(selected['source'], torch.zeros(len(rows), 8, device=device), cache=selected,
                           mean_enabled=False, charge_enabled=False)
            bases.append(result['voltage_basis_V'])
        caches.append(response_cache(encoding['original']['signal_pre'], torch.cat(bases), encoding['condition']))
    setup['joint_models'] = [JointResponse01(cache) for cache in caches]
    setup['physical_model'] = JointResponse01(response_cache(source, None, encoding['condition']), physical=True)
    return setup


def random_bank(setup, parents, draws, seed, counts):
    device = setup['source'].device
    generator = torch.Generator(device='cpu').manual_seed(seed)
    rows = parents.repeat_interleave(draws)
    z = torch.randn(len(rows), 8, generator=generator, dtype=torch.float32).to(device)
    counts['allocated_banks'] += 1
    counts['shape_standard_normal_values'] += z.numel()
    counts['noise005_calls'] += 1
    noise = electronic_noise(setup['extension'], len(rows),
        torch.Generator(device='cpu').manual_seed(seed + 1),
        torch.Generator(device='cpu').manual_seed(seed + 2), device, counts)
    return dict(rows=rows, z=z, noise=noise, parent_count=len(parents), draws=draws, seed=seed)


def prediction(model, theta, bank, q_floor, counts):
    counts['joint_response_calls'] += 1
    counts['joint_response_rows'] += len(bank['rows'])
    signal = model(theta, bank['rows'], bank['z'])
    observations = {}
    for h in HYPOTHESES:
        counts['generated_observation_calls'] += 1
        counts['generated_observation_rows'] += len(signal)
        observations[h] = observe_generated_carrier(signal + model.noise_scale(theta) * bank['noise'][h],
                                                     q_floor=q_floor, return_carrier=True)
    return observations


def feature_means(observations, maps, weights=None):
    return {h: maps[h].mean(observable_blocks(observations[h]), weights) for h in HYPOTHESES}


def residual_vector(predicted, target, maps):
    return torch.cat([maps[h].whiten(predicted[h] - target[h]) for h in HYPOTHESES]) / (len(HYPOTHESES) ** .5)


def risks(predicted, target, maps):
    return {h: maps[h].risks(predicted[h] - target[h]) for h in HYPOTHESES}


def finite_jacobian(model, theta, bank, q_floor, maps, target, counts, step, out=None, weights=None):
    columns = []
    for j in range(model.parameter_count):
        plus, minus = theta.clone(), theta.clone()
        plus[j] = min(float(model.upper[j]), float(theta[j]) + step)
        minus[j] = max(float(model.lower[j]), float(theta[j]) - step)
        high = feature_means(prediction(model, plus, bank, q_floor, counts), maps, weights)
        low = feature_means(prediction(model, minus, bank, q_floor, counts), maps, weights)
        columns.append((residual_vector(high, target, maps) - residual_vector(low, target, maps)) / (plus[j] - minus[j]))
        counts['finite_difference_columns'] += 1
        if out is not None and j % 9 == 0:
            progress_file(out, counts, 'joint_jacobian', column=j, parameters=model.parameter_count)
    return torch.stack(columns, 1)


def teacher_parameters(model):
    theta = model.lower.new_zeros(model.parameter_count)
    theta[:24].reshape(6, 4)[:, 0] = theta.new_tensor([.18, .22, -.12, .25, .18, -.12])
    theta[:24].reshape(6, 4)[:, 1] = theta.new_tensor([.06, -.04, .05, .06, -.03, .04])
    theta[:24].reshape(6, 4)[:, 2] = theta.new_tensor([-.04, .05, .03, -.05, .05, .02])
    L = theta.new_tensor([[.18, 0, 0, 0, 0, 0], [.09, .13, 0, 0, 0, 0],
        [-.05, .06, .14, 0, 0, 0], [-.10, .04, .06, .18, 0, 0],
        [.05, -.04, .03, .08, .14, 0], [.03, .02, -.04, .03, .02, .12]])
    theta[24:45] = L[model.tril[0], model.tril[1]]
    if not model.physical:
        theta[45:53] = torch.linspace(-.15, .15, 8, device=theta.device, dtype=theta.dtype)
    theta[-1] = .60
    return theta


def conditional_report(model, theta, bank, truth_signal, counts):
    counts['conditional_signal_rows'] += len(bank['rows'])
    counts['joint_response_calls'] += 1
    counts['joint_response_rows'] += len(bank['rows'])
    sample = model(theta, bank['rows'], bank['z']).reshape(bank['parent_count'], bank['draws'], 384)
    truth = truth_signal.reshape_as(sample)
    mean, true_mean = sample.mean(1), truth.mean(1)
    q, true_q = canonical_charge008(sample), canonical_charge008(truth)
    projection = torch.cos(torch.arange(384, device=sample.device, dtype=torch.float64)[:, None]
                           * torch.arange(1, 17, device=sample.device)[None] * torch.pi / 384.) / 384. ** .5
    reduced, true_reduced = sample @ projection, truth @ projection
    centered, true_centered = reduced - reduced.mean(1, keepdim=True), true_reduced - true_reduced.mean(1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / (bank['draws'] - 1)
    true_covariance = true_centered.transpose(1, 2) @ true_centered / (bank['draws'] - 1)
    return dict(conditional_mean_relative_RMS=float((mean - true_mean).square().mean().sqrt() / true_mean.square().mean().sqrt()),
        conditional_covariance_relative_Frobenius=float((covariance - true_covariance).norm() / true_covariance.norm()),
        conditional_Q_variance_mean=float(q.var(1).mean()), teacher_Q_variance_mean=float(true_q.var(1).mean()),
        conditional_wave_variance_mean=float(sample.var(1).mean()), teacher_wave_variance_mean=float(truth.var(1).mean()),
        covariance_min_eigenvalue=float(torch.linalg.eigvalsh(covariance).min()), parents=bank['parent_count'],
        draws_per_parent=bank['draws'], mean_is_actual_MC_mean=True,
        paired_teacher_truth_used_for_training=False)


def engineering(p, setup, out, counts):
    device = setup['source'].device
    q_floor = float(setup['loss'].q_floor)
    source_count = len(setup['source'])
    zero_results = []
    for seed, model in enumerate(setup['joint_models']):
        theta = model.initial()
        checked = 0
        for start in range(0, source_count, 128):
            rows = torch.arange(start, min(start + 128, source_count), device=device)
            bank = random_bank(setup, rows, 1, 191001 + start, counts)
            selected = setup['references'][seed].select_encoding(setup['encoding'], rows)
            counts['M0_head_calls'] += 1
            counts['M0_head_rows'] += len(rows)
            original = setup['references'][seed](selected['source'], bank['z'], cache=selected,
                                                 mean_enabled=False, charge_enabled=False)['signal_pre']
            details = model(theta, rows, bank['z'], return_details=True)
            counts['joint_response_calls'] += 1
            counts['joint_response_rows'] += len(rows)
            assert torch.equal(details['signal'].view(torch.uint8), original.view(torch.uint8))
            for h in HYPOTHESES:
                a = observe_generated_carrier(original + bank['noise'][h], q_floor=q_floor, return_carrier=True)
                b = observe_generated_carrier(details['signal'] + bank['noise'][h], q_floor=q_floor, return_carrier=True)
                counts['generated_observation_calls'] += 2
                counts['generated_observation_rows'] += 2 * len(rows)
                for key in ('carrier_adc', 'baseline100', 'window_V', 'peak_index', 'Q', 'tail', 'PSD'):
                    assert torch.equal(a[key], b[key]), key
            checked += len(rows)
        zero_results.append(dict(seed=seed, rows=checked, all_three_hypotheses_bitwise_identical=True))
        progress_file(out, counts, 'M0_zero_identity_complete', seed=seed)
    model = setup['joint_models'][0]
    theta = teacher_parameters(model)
    parents = torch.arange(64, device=device)
    kernel_bank = random_bank(setup, parents, 128, 191020, counts)
    details = model(theta, kernel_bank['rows'], kernel_bank['z'], return_details=True)
    counts['joint_response_calls'] += 1
    counts['joint_response_rows'] += len(kernel_bank['rows'])
    assert bool(torch.isfinite(details['signal']).all())
    assert bool((details['signal'][:, :100] == 0).all())
    assert bool((details['queries'].diff(dim=1) > 0).all())
    assert bool((canonical_charge008(details['signal']) > 0).all())
    report = conditional_report(model, theta, kernel_bank, details['signal'], counts)
    z0 = model(theta, parents, torch.zeros(len(parents), 8, device=device))
    counts['joint_response_calls'] += 1
    counts['joint_response_rows'] += len(parents)
    report['MC_mean_minus_zero_latent_RMS_V'] = float((details['signal'].reshape(64, 128, 384).mean(1) - z0).square().mean().sqrt())
    assert bool((setup['source'][:, :100] == 0).all())
    physical = setup['physical_model']
    physical_output = physical(physical.initial(), kernel_bank['rows'], kernel_bank['z'])
    counts['joint_response_calls'] += 1
    counts['joint_response_rows'] += len(kernel_bank['rows'])
    assert bool(torch.isfinite(physical_output).all() & (physical_output[:, :100] == 0).all())
    report['physical_baseline_Q_variance_mean'] = float(canonical_charge008(physical_output.reshape(64, 128, 384)).var(1).mean())
    # Independently sampled target parents and latent/noise banks: no pair labels
    # enter the fit features, residual, or full-Jacobian direction.
    target_parents = torch.randperm(source_count, generator=torch.Generator().manual_seed(191030))[:1024].to(device)
    target_bank = random_bank(setup, target_parents, 2, 191040, counts)
    target_obs = prediction(model, theta, target_bank, q_floor, counts)
    maps = {h: JointFeatureMap01(observable_blocks(target_obs[h]), seed=191050) for h in HYPOTHESES}
    target = feature_means(target_obs, maps)
    parents = torch.randperm(source_count, generator=torch.Generator().manual_seed(191060))[:128].to(device)
    fit_bank = random_bank(setup, parents, 2, 191070, counts)
    start_theta = model.initial()
    start = time.monotonic()
    predicted = feature_means(prediction(model, start_theta, fit_bank, q_floor, counts), maps)
    residual = residual_vector(predicted, target, maps)
    J = finite_jacobian(model, start_theta, fit_bank, q_floor, maps, target, counts,
                        p['solver']['finite_difference_step'], out)
    delta, geometry = joint_step(residual, J, ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'])
    elapsed = time.monotonic() - start
    candidate = torch.maximum(model.lower, torch.minimum(model.upper, start_theta + delta))
    candidate_mean = feature_means(prediction(model, candidate, fit_bank, q_floor, counts), maps)
    assert bool(torch.isfinite(J).all() & torch.isfinite(candidate).all())
    geometry.update(jacobian_shape=list(J.shape), residual_feature_dimensions=272,
        three_hypotheses=HYPOTHESES, engineering_J_seconds=elapsed,
        original_risks=risks(predicted, target, maps), proposed_risks=risks(candidate_mean, target, maps),
        candidate_saved_as_engineering_only=True, all_parameter_names=model.parameter_names(),
        zero_column_names=[name for name, column in zip(model.parameter_names(), J.T) if float(column.norm()) == 0.],
        mean_random_cross_terms_kept=True, automatic_differentiation_queries=0)
    np.savez(out / 'JOINT_JACOBIAN_ENGINEERING.npz', jacobian=J.cpu().numpy(),
             residual=residual.cpu().numpy(), joint_proposal=candidate.cpu().numpy())
    return dict(status='PASS', M0_zero_identity=zero_results, conditional_kernel_engineering=report,
        physical_baseline_implemented=True, neural_response_parameters=53, electronic_parameters=1,
        physical_response_parameters=45, geometry=geometry,
        REAL_response_updates=0, scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered JOINT_RESPONSE_01 configuration changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned source changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    started = time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
            torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            progress_file(args.out, counts, 'setup_loaded')
            result = engineering(p, setup, args.out, counts)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
                      elapsed_seconds=time.monotonic() - started,
                      torch_version=torch.__version__, device=str(device))
        for name, maximum in p['operation_budget'].items():
            assert counts[name] <= maximum, (name, counts[name], maximum)
        assert result['elapsed_seconds'] < p['resources']['hard_wall_seconds']
        write_json(args.out / 'RESULT.json', result)
        write_json(args.out / 'COMPLETE.json', dict(status='PASS', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(), execution_counts=dict(counts)))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                automatic_restart=False, scientific_acceptance=False))


if __name__ == '__main__':
    main()
