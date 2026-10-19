"""One prospectively registered precision confirmation of frozen teacher fits."""
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
from n01.model008 import canonical_charge008
from diagnose_joint01_kernel import compare_moments
from run_joint_response01 import load_joint_setup, progress_file, teacher_parameters
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered precision configuration changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned precision input changed: ' + name)
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
            models = setup['joint_models']
            del setup
            with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
                heldout = archive['heldout_rows'].copy()
            projection = torch.cos(torch.arange(384, device=device, dtype=torch.float64)[:, None]
                * torch.arange(1, 17, device=device)[None] * torch.pi / 384.) / 384. ** .5
            all_moments, reports, gates = {}, {}, []
            draws, success = p['draws_per_parent'], p['success']
            for seed_index, model in enumerate(models):
                teacher = teacher_parameters(model)
                with np.load(ROOT / p['endpoint_files'][seed_index], allow_pickle=False) as archive:
                    candidate = torch.as_tensor(archive['theta'], device=device)
                for library in range(2):
                    generator = torch.Generator(device='cpu').manual_seed(p['bank_seeds'][seed_index][library])
                    counts['allocated_banks'] += 1
                    samples = {label: [] for label in ('teacher', 'M0', 'candidate')}
                    for position, parent in enumerate(heldout):
                        check_budget(p, counts, started)
                        rows = torch.full((draws,), int(parent), device=device, dtype=torch.long)
                        z = torch.randn(draws, 8, generator=generator, dtype=torch.float32).to(device)
                        counts['shape_standard_normal_values'] += z.numel()
                        for label, theta in (('teacher', teacher), ('M0', model.initial()), ('candidate', candidate)):
                            counts['joint_response_calls'] += 1
                            counts['joint_response_rows'] += draws
                            signal = model(theta, rows, z)
                            reduced = signal @ projection
                            centered = reduced - reduced.mean(0)
                            covariance = centered.T @ centered / (draws - 1)
                            samples[label].append((signal.mean(0).cpu().numpy(), covariance.cpu().numpy(),
                                float(canonical_charge008(signal).var()), float(signal.var(0).mean())))
                        if position % 32 == 0:
                            progress_file(args.out, counts, 'frozen_precision_confirmation', seed=seed_index,
                                          library=library, parents_complete=position + 1)
                    arrays = {}
                    for label, entries in samples.items():
                        packed = tuple(np.stack([entry[i] for entry in entries]) for i in range(4))
                        all_moments[seed_index, library, label] = packed
                        for name, array in zip(('mean', 'covariance', 'Q_variance', 'wave_variance'), packed):
                            arrays[f'{label}_{name}'] = array
                    np.savez(args.out / f'seed{seed_index}_library{library}_MOMENTS.npz', heldout_rows=heldout, **arrays)
                    truth = all_moments[seed_index, library, 'teacher']
                    baseline = compare_moments(all_moments[seed_index, library, 'M0'], truth)
                    fitted = compare_moments(all_moments[seed_index, library, 'candidate'], truth)
                    key = f'seed{seed_index}/library{library}'
                    reports[key] = dict(M0=baseline, candidate=fitted)
                    for metric, absolute, fraction in (
                        ('conditional_mean_relative_RMS', success['mean_absolute_relative_RMS'], success['mean_error_fraction']),
                        ('conditional_covariance_relative_Frobenius', success['covariance_absolute_relative_error'], success['covariance_error_fraction'])):
                        limit = min(absolute, fraction * baseline[metric])
                        gates.append(dict(name=key+'/'+metric, value=fitted[metric], upper=limit,
                                          pass_gate=fitted[metric] <= limit))
                    ratio = fitted['Q_variance_ratio']
                    gates.append(dict(name=key+'/Q_variance_ratio', value=ratio,
                        lower=success['variance_ratio_low'], upper=success['variance_ratio_high'],
                        pass_gate=success['variance_ratio_low'] <= ratio <= success['variance_ratio_high']))
            precision = {}
            for seed_index in range(2):
                for label in ('teacher', 'M0', 'candidate'):
                    a = all_moments[seed_index, 0, label][1]
                    b = all_moments[seed_index, 1, label][1]
                    value = float(np.linalg.norm(a - b) / (.5 * (np.linalg.norm(a) + np.linalg.norm(b))))
                    name = f'seed{seed_index}/{label}/independent_bank_MC_difference'
                    precision[name] = value
                    gates.append(dict(name=name, value=value, upper=success['MC_covariance_difference_max'],
                                      pass_gate=value <= success['MC_covariance_difference_max']))
        check_budget(p, counts, started)
        carried = {}
        for index, name in enumerate(p['original_decision_files']):
            original = json.loads((ROOT / name).read_text())
            carried[str(index)] = [gate for gate in original['gates'] if len(gate['name'].split('/')) == 3]
            if len(carried[str(index)]) != 18:
                raise ValueError('Original hard-observation gate set changed')
        confirmed = all(gate['pass_gate'] for gate in gates) and all(g['pass_gate'] for group in carried.values() for g in group)
        result = dict(status='COMPLETE', precision_confirmation_pass_all=confirmed,
            stage4_REAL_calibration_eligible_under_precision_amendment=confirmed, reports=reports,
            new_conditional_and_MC_precision_gates=gates, original_hard_observation_gates_carried_unchanged=carried,
            original_teacher_pass_all=False, original_teacher_gate_changed=False, parameter_updates=0,
            config_sha256=args.config_sha256, execution_counts=dict(counts), elapsed_seconds=time.monotonic()-started,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), scientific_acceptance=False,
            scope='Known synthetic teacher only; frozen-system posthoc heldout, not full-pipeline or REAL response acceptance.')
        write_json(args.out/'RESULT.json', result)
        output_bytes = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
        if output_bytes > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered precision output bound exceeded')
        write_json(args.out/'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out/'RESULT.json').read_bytes()).hexdigest(), precision_confirmation_pass_all=confirmed))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic()-started, automatic_restart=False))


if __name__ == '__main__':
    main()
