"""Fixed-endpoint finite-difference scale audit under the original hard readout."""
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
    summaries = {}
    for seed_index, model in enumerate(setup['joint_models']):
        with np.load(ROOT / p['endpoint_files'][seed_index], allow_pickle=False) as archive:
            theta = torch.as_tensor(archive['theta'].copy(), device=device)
        seed = p['training_seed'] + seed_index * 100000
        banks, targets, features, baseline, old_risks, old_native, initial_native = {}, {}, {}, {}, {}, {}, {}
        arrays = dict(theta=theta.cpu().numpy(), cholesky=maps['FIT'][HYPOTHESES[0]].cholesky.cpu().numpy())
        for label, offsets in (('FIT', (100, 200, 1000)), ('CHECK', (3000, 5000, 4000)),
                               ('FRESH', (11000, 13000, 12000))):
            parents, rows = draw_parents(source_pool, 1024, seed + offsets[0]), draw_parents(real_pool, 4096, seed + offsets[1])
            arrays[label + '_source_parents'], arrays[label + '_real_rows'] = parents.cpu().numpy(), rows.cpu().numpy()
            counts['REAL_fit_feature_rows' if label == 'FIT' else 'REAL_CHECK_feature_rows'] += len(rows)
            banks[label] = random_bank(setup, parents, 4, seed + offsets[2], counts)
            targets[label] = select_real(real, rows)
            features[label] = feature_means({h: targets[label] for h in HYPOTHESES}, maps[label])
            baseline[label] = prediction(model, theta, banks[label], setup['loss'].q_floor, counts)
            old_risks[label] = risks(feature_means(baseline[label], maps[label]), features[label], maps[label])
            old_native[label] = native_values(baseline[label], targets[label], setup, counts)
            initial = prediction(model, model.initial(), banks[label], setup['loss'].q_floor, counts)
            initial_native[label] = native_values(initial, targets[label], setup, counts)
        r = residual_vector(feature_means(baseline['FIT'], maps['FIT']), features['FIT'], maps['FIT'])
        arrays.update(residual=r.cpu().numpy(), native_values=old_native['FIT'].cpu().numpy())
        report = dict(endpoint_file=p['endpoint_files'][seed_index], baseline_risks=old_risks,
            baseline_native={k: v.tolist() for k, v in old_native.items()},
            initial_native={k: v.tolist() for k, v in initial_native.items()}, cases=[])
        for case_index, step in enumerate(p['difference_steps']):
            columns, native_columns, peak_changes, window_changes = [], [], [], []
            for column in range(model.parameter_count):
                check_budget(p, counts, started)
                plus, minus = theta.clone(), theta.clone()
                plus[column] = min(float(model.upper[column]), float(theta[column]) + step)
                minus[column] = max(float(model.lower[column]), float(theta[column]) - step)
                pairs, native_pairs, peaks, windows = [], [], [], []
                for proposal in (plus, minus):
                    obs = prediction(model, proposal, banks['FIT'], setup['loss'].q_floor, counts)
                    pairs.append(residual_vector(feature_means(obs, maps['FIT']), features['FIT'], maps['FIT']))
                    native_pairs.append(native_values(obs, targets['FIT'], setup, counts))
                    peaks.append([obs[h]['peak_index'] for h in HYPOTHESES])
                    windows.append([obs[h]['window_V'] for h in HYPOTHESES])
                columns.append((pairs[0] - pairs[1]) / (plus[column] - minus[column]))
                native_columns.append((native_pairs[0] - native_pairs[1]) / (plus[column] - minus[column]))
                peak_changes.append([int((a != b).sum()) for a, b in zip(*peaks)])
                window_changes.append([int((a != b).any(1).sum()) for a, b in zip(*windows)])
                counts['finite_difference_columns'] += 1
                if column % 9 == 0:
                    progress_file(out, counts, 'fixed_endpoint_FD_scale_Jacobian', seed=seed_index,
                        difference_step=step, column=column + 1)
            J, native_J = torch.stack(columns, 1), torch.stack(native_columns, 1)
            delta, geometry = native_joint_step(r, J, maps['FIT'], old_native['FIT'], native_J,
                theta, model.lower, model.upper, ridge_fraction=p['solver']['ridge_fraction'],
                radius=p['solver']['radius'], macro_factor=p['solver']['macro_degradation_factor'],
                sample_count=4096, max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
            counts['joint_direction_solves'] += 1
            prefix = f'case{case_index}_'
            arrays.update({prefix + 'jacobian': J.cpu().numpy(), prefix + 'native_jacobian': native_J.cpu().numpy(),
                prefix + 'direction': delta.cpu().numpy(), prefix + 'peak_changes': np.asarray(peak_changes),
                prefix + 'window_changes': np.asarray(window_changes)})
            case = dict(difference_step=step, geometry=geometry, trials=[], selected_fraction=None,
                zero_joint_columns=int((J.norm(dim=0) == 0).sum()),
                zero_native_columns=int((native_J.norm(dim=0) == 0).sum()), readout_rows_per_H=4096)
            selected = None
            for fraction in p['backtracking_fractions']:
                check_budget(p, counts, started)
                counts['joint_candidate_trials'] += 1
                proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
                trial = dict(fraction=fraction, checks={})
                for label in ('FIT', 'CHECK'):
                    obs = prediction(model, proposal, banks[label], setup['loss'].q_floor, counts)
                    mean = feature_means(obs, maps[label])
                    risk = risks(mean, features[label], maps[label])
                    native = native_values(obs, targets[label], setup, counts)
                    trust = output_trust(baseline[label], obs, model, banks[label]['rows'], p)
                    ratios, anchor = (native / old_native[label]).reshape(3, 3), (native / initial_native[label]).reshape(3, 3)
                    native_pass = all(bool((value[:, :2] <= p['native_guard']['wave_QPSD_factor']).all()
                        and (value[:, 2] <= p['native_guard']['noise_factor']).all()) for value in (ratios, anchor))
                    joint_pass = accept_joint(old_risks[label], risk, trust, 4096, p)
                    check = dict(risks=risk, native=native.tolist(), native_ratios=ratios.tolist(),
                        initial_anchor_ratios=anchor.tolist(), native_pass=native_pass, trust=trust,
                        original_joint_accept=joint_pass, accepted=native_pass and joint_pass)
                    if label == 'FIT':
                        actual_r = residual_vector(mean, features[label], maps[label])
                        predicted_r = r + J @ (proposal - theta)
                        predicted_native = old_native[label] + native_J @ (proposal - theta)
                        check.update(joint_linearization_relative_error=float((actual_r - predicted_r).norm() / r.norm()),
                            predicted_native_ratios=(predicted_native / old_native[label]).reshape(3, 3).tolist(),
                            native_linearization_relative_error=((native - predicted_native) / old_native[label]).reshape(3, 3).tolist())
                    trial['checks'][label] = check
                trial['accepted'] = all(check['accepted'] for check in trial['checks'].values())
                case['trials'].append(trial)
                if trial['accepted']:
                    selected = proposal
                    case['selected_fraction'] = fraction
                    arrays[prefix + 'uncommitted_candidate'] = proposal.cpu().numpy()
                    break
            case['fresh_pass'] = False
            if selected is not None:
                obs = prediction(model, selected, banks['FRESH'], setup['loss'].q_floor, counts)
                risk = risks(feature_means(obs, maps['FRESH']), features['FRESH'], maps['FRESH'])
                native = native_values(obs, targets['FRESH'], setup, counts)
                trust = output_trust(baseline['FRESH'], obs, model, banks['FRESH']['rows'], p)
                ratios, anchor = (native / old_native['FRESH']).reshape(3, 3), (native / initial_native['FRESH']).reshape(3, 3)
                native_pass = all(bool((value[:, :2] <= p['native_guard']['wave_QPSD_factor']).all()
                    and (value[:, 2] <= p['native_guard']['noise_factor']).all()) for value in (ratios, anchor))
                case['fresh_pass'] = native_pass and accept_joint(old_risks['FRESH'], risk, trust, 4096, p)
                case['fresh_confirmation'] = dict(risks=risk, native=native.tolist(), trust=trust,
                    native_ratios=ratios.tolist(), initial_anchor_ratios=anchor.tolist(), pass_all=case['fresh_pass'])
            report['cases'].append(case)
            np.savez(out / f'seed{seed_index}_FD_SCALE.npz', **arrays)
            write_json(out / f'seed{seed_index}_FD_SCALE.json', report)
        comparisons = []
        for left, right in ((0, 1), (1, 2)):
            a, b = arrays[f'case{left}_jacobian'], arrays[f'case{right}_jacobian']
            na, nb = arrays[f'case{left}_native_jacobian'], arrays[f'case{right}_native_jacobian']
            norms_a, norms_b = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
            comparisons.append(dict(steps=[p['difference_steps'][left], p['difference_steps'][right]],
                joint_relative_Frobenius_difference=float(np.linalg.norm(a - b) / np.linalg.norm(a)),
                joint_cosine=float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))),
                native_row_norms_left=norms_a.tolist(), native_row_norms_right=norms_b.tolist(),
                native_row_cosine=[None if x == 0 or y == 0 else float((u * v).sum() / (x * y))
                    for u, v, x, y in zip(na, nb, norms_a, norms_b)]))
        report.update(scale_comparisons=comparisons, committed_parameter_updates=0,
            all_cases_share_fixed_endpoint_and_random_banks=True, heldout_rows_used=0,
            FRESH_role='Independent random library and feature map on fitting-role records; no reporting holdout')
        write_json(out / f'seed{seed_index}_FD_SCALE.json', report)
        summaries[str(seed_index)] = dict(scale_comparisons=comparisons,
            cases=[{key: case[key] for key in ('difference_step', 'selected_fraction', 'fresh_pass',
                'zero_joint_columns', 'zero_native_columns')} for case in report['cases']])
    return dict(status='COMPLETE', summaries=summaries, committed_parameter_updates=0,
        scientific_acceptance=False, original_final_acceptance_gates_unchanged=True,
        automatic_training_or_best_scale_selection=False,
        next_action='Review derivative consistency, hard-readout changes and actual fresh native effects before any method change')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered FD-scale config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned FD-scale input changed: ' + name)
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
            progress_file(args.out, counts, 'FD_scale_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered FD-scale output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False,
                committed_parameter_updates=0, scientific_acceptance=False))


if __name__ == '__main__':
    main()
