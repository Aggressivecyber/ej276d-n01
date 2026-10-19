"""No-update joint direction test with native metric constraints and hard backtracking."""
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
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents, output_trust, accept_joint
from run_response036 import hard_metrics, WAVE_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
NATIVE_NAMES = ('wave', 'QPSD', 'noise')


def native_values(observations, target, setup, counts):
    loss, context = setup['loss'], setup['context']
    result = []
    for h in HYPOTHESES:
        obs = observations[h]
        counts['native_component_rows'] += len(obs['window_V']) + len(target['window_V'])
        parts = loss.wave_components(obs['window_V'], target['window_V'], None, context, mean_response=False)
        result.extend((sum(WAVE_WEIGHTS[key] * value for key, value in parts.items()),
            loss._qpsd_with_context(obs['window_V'], target['window_V'], None, context, mean_response=False),
            loss.noise_loss(obs['baseline100'], target['baseline100'], None, projection_context=context)))
    return torch.stack(result).double()


def native_report(observations, target, expected, setup, counts):
    result, parity = {}, 0.
    for hi, h in enumerate(HYPOTHESES):
        obs = observations[h]
        counts['native_metric_rows'] += len(obs['window_V']) + len(target['window_V'])
        result[h] = hard_metrics(obs['window_V'].cpu().numpy(), obs['baseline100'].cpu().numpy(),
            target['window_V'].cpu().numpy(), target['baseline100'].cpu().numpy(), setup['loss'], setup['context'])
        parity = max(parity, max(abs(result[h]['values'][name] - float(expected[hi * 3 + j]))
            for j, name in enumerate(NATIVE_NAMES)))
    if parity > 1e-12:
        raise ValueError('Native constraint values differ from original full report: ' + str(parity))
    return dict(report=result, full_report_parity_max_difference=parity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered native-direction config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned native-direction input changed: ' + name)
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
            with np.load(ROOT / p['roles_file'], allow_pickle=False) as z:
                source_pool, real_pool = [torch.as_tensor(z[key], device=device)
                    for key in ('source_train_rows', 'real_train_rows')]
            cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
                q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
            for key in ('window_V', 'baseline100'):
                if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
                    raise ValueError('REAL readout differs: ' + key)
            real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
                flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
            del setup['real'], cpu_real
            counts['REAL_observable_replay_calls'] += 1
            counts['REAL_observable_replay_rows'] += len(real['Q'])
            blocks = observable_blocks(select_real(real, real_pool))
            maps = {}
            for label in ('FIT', 'CHECK', 'FRESH'):
                fmap = JointFeatureMap01(blocks, seed=p['feature_seeds'][label])
                maps[label] = {h: fmap for h in HYPOTHESES}
                counts['feature_maps_constructed'] += 1
            del blocks
            summaries = {}
            for seed_index, model in enumerate(setup['joint_models']):
                check_budget(p, counts, started)
                seed, theta = p['training_seed'] + seed_index * 100000, model.initial()
                with np.load(ROOT / p['direction_files'][seed_index], allow_pickle=False) as z:
                    saved = {key: z[key].copy() for key in z.files}
                selections = dict(FIT=(torch.as_tensor(saved['fit_source_parents'], device=device),
                    torch.as_tensor(saved['fit_real_rows'], device=device), seed + 1000))
                for label, offsets in (('CHECK', (3000, 5000, 4000)), ('FRESH', (11000, 13000, 12000))):
                    selections[label] = (draw_parents(source_pool, 1024, seed + offsets[0]),
                        draw_parents(real_pool, 4096, seed + offsets[1]), seed + offsets[2])
                banks, targets, target_features, baseline_obs, baseline_risk, baseline_native = {}, {}, {}, {}, {}, {}
                arrays = {}
                for label, (parents, rows, rng_seed) in selections.items():
                    arrays[label + '_source_parents'], arrays[label + '_real_rows'] = parents.cpu().numpy(), rows.cpu().numpy()
                    counts['REAL_fit_feature_rows' if label == 'FIT' else 'REAL_CHECK_feature_rows'] += len(rows)
                    banks[label] = random_bank(setup, parents, 4, rng_seed, counts)
                    targets[label] = select_real(real, rows)
                    target_features[label] = feature_means({h: targets[label] for h in HYPOTHESES}, maps[label])
                    baseline_obs[label] = prediction(model, theta, banks[label], setup['loss'].q_floor, counts)
                    baseline_risk[label] = risks(feature_means(baseline_obs[label], maps[label]), target_features[label], maps[label])
                    baseline_native[label] = native_values(baseline_obs[label], targets[label], setup, counts)
                r = residual_vector(feature_means(baseline_obs['FIT'], maps['FIT']), target_features['FIT'], maps['FIT'])
                columns, native_columns = [], []
                for j in range(model.parameter_count):
                    check_budget(p, counts, started)
                    plus, minus = theta.clone(), theta.clone()
                    plus[j] = min(float(model.upper[j]), float(theta[j]) + p['solver']['finite_difference_step'])
                    minus[j] = max(float(model.lower[j]), float(theta[j]) - p['solver']['finite_difference_step'])
                    pairs, native_pairs = [], []
                    for proposal in (plus, minus):
                        obs = prediction(model, proposal, banks['FIT'], setup['loss'].q_floor, counts)
                        pairs.append(residual_vector(feature_means(obs, maps['FIT']), target_features['FIT'], maps['FIT']))
                        native_pairs.append(native_values(obs, targets['FIT'], setup, counts))
                    columns.append((pairs[0] - pairs[1]) / (plus[j] - minus[j]))
                    native_columns.append((native_pairs[0] - native_pairs[1]) / (plus[j] - minus[j]))
                    counts['finite_difference_columns'] += 1
                    if j % 9 == 0:
                        progress_file(args.out, counts, 'native_joint_Jacobian', seed=seed_index, column=j + 1)
                J, native_J = torch.stack(columns, 1), torch.stack(native_columns, 1)
                parity = max(float((r - torch.as_tensor(saved['residual'], device=device)).abs().max()),
                    float((J - torch.as_tensor(saved['jacobian'], device=device)).abs().max()))
                if parity > 1e-12:
                    raise ValueError('Original joint Jacobian/residual reproduction differs: ' + str(parity))
                delta, geometry = native_joint_step(r, J, maps['FIT'], baseline_native['FIT'], native_J,
                    theta, model.lower, model.upper, ridge_fraction=p['solver']['ridge_fraction'],
                    radius=p['solver']['radius'], macro_factor=p['solver']['macro_degradation_factor'],
                    sample_count=4096, max_iterations=p['constrained_solver']['max_iterations'],
                    ftol=p['constrained_solver']['ftol'])
                counts['joint_direction_solves'] += 1
                arrays.update(residual=r.cpu().numpy(), jacobian=J.cpu().numpy(),
                    native_values=baseline_native['FIT'].cpu().numpy(), native_jacobian=native_J.cpu().numpy(),
                    native_delta=delta.cpu().numpy(), cholesky=maps['FIT'][HYPOTHESES[0]].cholesky.cpu().numpy())
                report = dict(geometry=geometry, original_J_and_r_parity=parity, baseline_risks=baseline_risk,
                    baseline_native={label: value.tolist() for label, value in baseline_native.items()},
                    baseline_full_native={label: native_report(baseline_obs[label], targets[label], baseline_native[label], setup, counts)
                        for label in ('CHECK', 'FRESH')}, trials=[])
                selected = None
                for fraction in p['backtracking_fractions']:
                    check_budget(p, counts, started)
                    counts['joint_candidate_trials'] += 1
                    proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
                    trial = dict(fraction=fraction, checks={})
                    for label in ('FIT', 'CHECK'):
                        obs = prediction(model, proposal, banks[label], setup['loss'].q_floor, counts)
                        risk = risks(feature_means(obs, maps[label]), target_features[label], maps[label])
                        native = native_values(obs, targets[label], setup, counts)
                        trust = output_trust(baseline_obs[label], obs, model, banks[label]['rows'], p)
                        native_ratios = (native / baseline_native[label]).reshape(3, 3)
                        native_pass = bool((native_ratios[:, :2] <= p['native_guard']['wave_QPSD_factor']).all()
                            and (native_ratios[:, 2] <= p['native_guard']['noise_factor']).all())
                        accepted = accept_joint(baseline_risk[label], risk, trust, 4096, p) and native_pass
                        trial['checks'][label] = dict(risks=risk, native=native.tolist(), native_ratios=native_ratios.tolist(),
                            native_pass=native_pass, trust=trust, original_joint_accept=accept_joint(baseline_risk[label], risk, trust, 4096, p),
                            accepted=accepted)
                    report['trials'].append(trial)
                    if all(entry['accepted'] for entry in trial['checks'].values()):
                        selected = dict(fraction=fraction, theta=proposal)
                        report['selected_CHECK_full_native'] = native_report(obs, targets['CHECK'], native, setup, counts)
                        break
                fresh_pass = False
                if selected is not None:
                    obs = prediction(model, selected['theta'], banks['FRESH'], setup['loss'].q_floor, counts)
                    risk = risks(feature_means(obs, maps['FRESH']), target_features['FRESH'], maps['FRESH'])
                    native = native_values(obs, targets['FRESH'], setup, counts)
                    trust = output_trust(baseline_obs['FRESH'], obs, model, banks['FRESH']['rows'], p)
                    native_ratios = (native / baseline_native['FRESH']).reshape(3, 3)
                    fresh_pass = (accept_joint(baseline_risk['FRESH'], risk, trust, 4096, p)
                        and bool((native_ratios[:, :2] <= p['native_guard']['wave_QPSD_factor']).all())
                        and bool((native_ratios[:, 2] <= p['native_guard']['noise_factor']).all()))
                    report['fresh_confirmation'] = dict(fraction=selected['fraction'], risks=risk, trust=trust,
                        native=native.tolist(), native_ratios=native_ratios.tolist(), pass_all=fresh_pass,
                        full_native=native_report(obs, targets['FRESH'], native, setup, counts))
                report.update(selected_fraction=None if selected is None else selected['fraction'], fresh_pass=fresh_pass,
                    committed_parameter_updates=0, heldout_rows_used=0)
                np.savez(args.out / f'seed{seed_index}_NATIVE_DIRECTION.npz', **arrays)
                write_json(args.out / f'seed{seed_index}_NATIVE_DIRECTION.json', report)
                summaries[str(seed_index)] = dict(selected_fraction=report['selected_fraction'], fresh_pass=fresh_pass,
                    trials=len(report['trials']), original_J_and_r_parity=parity)
                progress_file(args.out, counts, 'native_joint_direction_completed', seed=seed_index)
        check_budget(p, counts, started)
        result = dict(status='COMPLETE', config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, summaries=summaries,
            two_seed_fresh_diagnostic_pass=all(entry['fresh_pass'] for entry in summaries.values()),
            committed_parameter_updates=0, scientific_acceptance=False,
            original_final_acceptance_gates_unchanged=True,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Native-direction output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE',
            result_sha256=hashlib.sha256((args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False))


if __name__ == '__main__':
    main()
