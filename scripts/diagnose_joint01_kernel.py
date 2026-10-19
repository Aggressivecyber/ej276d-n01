"""Frozen endpoint conditional-kernel convergence; no fitting or new acceptance."""
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
from run_joint_response01 import load_joint_setup, progress_file, teacher_parameters
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]


def compare_moments(candidate, truth):
    mean, covariance, qvar, wavevar = candidate
    tmean, tcovariance, tqvar, twavevar = truth
    squared = np.square(covariance - tcovariance).sum(axis=(1, 2))
    denominator = np.square(tcovariance).sum()
    scalar = float((covariance * tcovariance).sum() / np.square(covariance).sum())
    top = np.argsort(squared)[-13:][::-1]
    return dict(conditional_mean_relative_RMS=float(np.linalg.norm(mean - tmean) / np.linalg.norm(tmean)),
        conditional_covariance_relative_Frobenius=float(np.sqrt(squared.sum() / denominator)),
        covariance_trace_ratio=float(np.trace(covariance, axis1=1, axis2=2).sum() / np.trace(tcovariance, axis1=1, axis2=2).sum()),
        covariance_cosine=float((covariance * tcovariance).sum() / (np.linalg.norm(covariance) * np.linalg.norm(tcovariance))),
        diagnostic_best_scalar=scalar,
        diagnostic_after_scalar_relative_error=float(np.linalg.norm(scalar * covariance - tcovariance) / np.linalg.norm(tcovariance)),
        scalar_is_diagnostic_only_no_candidate_change=True,
        Q_variance_ratio=float(qvar.mean() / tqvar.mean()), wave_variance_ratio=float(wavevar.mean() / twavevar.mean()),
        top13_of128_parent_positions=top.tolist(), top13_error_fraction=float(squared[top].sum() / squared.sum()),
        projection_diagonal_residual=(np.diagonal(covariance - tcovariance, axis1=1, axis2=2).mean(0)).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered kernel diagnostic configuration changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned diagnostic input changed: ' + name)
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
            with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
                heldout = archive['heldout_rows'].copy()
            models = setup['joint_models']
            parent_data = dict(heldout_rows=heldout,
                source_Q=canonical_charge008(setup['source'])[heldout].cpu().numpy(),
                bounded_PC1=models[0].cache['condition'][heldout, 2].cpu().numpy(),
                h=setup['data']['h'][heldout].copy())
            del setup
            projection = torch.cos(torch.arange(384, device=device, dtype=torch.float64)[:, None]
                * torch.arange(1, 17, device=device)[None] * torch.pi / 384.) / 384. ** .5
            results, full_moments = {}, {}
            prefixes, draws = p['draw_prefixes'], max(p['draw_prefixes'])
            for seed_index, model in enumerate(models):
                truth = teacher_parameters(model)
                with np.load(ROOT / p['endpoint_files'][seed_index], allow_pickle=False) as archive:
                    endpoint = torch.as_tensor(archive['theta'], device=device)
                for library in range(2):
                    bank_seed = p['bank_seeds'][seed_index][library]
                    generator = torch.Generator(device='cpu').manual_seed(bank_seed)
                    counts['allocated_banks'] += 1
                    moments = {(label, n): [] for label in ('teacher', 'candidate') for n in prefixes}
                    for position, parent in enumerate(heldout):
                        check_budget(p, counts, started)
                        rows = torch.full((draws,), int(parent), device=device, dtype=torch.long)
                        z = torch.randn(draws, 8, generator=generator, dtype=torch.float32).to(device)
                        counts['shape_standard_normal_values'] += z.numel()
                        for label, theta in (('teacher', truth), ('candidate', endpoint)):
                            counts['joint_response_calls'] += 1
                            counts['joint_response_rows'] += draws
                            signal = model(theta, rows, z)
                            for n in prefixes:
                                sample = signal[:n]
                                reduced = sample @ projection
                                centered = reduced - reduced.mean(0)
                                covariance = centered.T @ centered / (n - 1)
                                moments[label, n].append((sample.mean(0).cpu().numpy(), covariance.cpu().numpy(),
                                    float(canonical_charge008(sample).var()), float(sample.var(0).mean())))
                        if position % 32 == 0:
                            progress_file(args.out, counts, 'frozen_kernel_diagnostic', seed=seed_index,
                                          library=library, parents_complete=position + 1)
                    arrays = {}
                    for (label, n), entries in moments.items():
                        packed = tuple(np.stack([entry[i] for entry in entries]) for i in range(4))
                        full_moments[seed_index, library, label, n] = packed
                        for name, array in zip(('mean', 'covariance', 'Q_variance', 'wave_variance'), packed):
                            arrays[f'{label}_{n}_{name}'] = array
                    np.savez(args.out / f'seed{seed_index}_library{library}_MOMENTS.npz', **parent_data, **arrays)
                    reports = {}
                    for n in prefixes:
                        candidate = full_moments[seed_index, library, 'candidate', n]
                        target = full_moments[seed_index, library, 'teacher', n]
                        report = compare_moments(candidate, target)
                        report['strata'] = {}
                        for field in ('source_Q', 'bounded_PC1', 'h'):
                            groups = np.array_split(np.argsort(parent_data[field], kind='stable'), 4)
                            report['strata'][field] = [dict(parent_count=len(group),
                                minimum=float(parent_data[field][group].min()), maximum=float(parent_data[field][group].max()),
                                covariance_relative_Frobenius=float(np.linalg.norm(candidate[1][group] - target[1][group]) /
                                                                  np.linalg.norm(target[1][group])),
                                squared_error_contribution=float(np.square(candidate[1][group] - target[1][group]).sum() /
                                                                 np.square(candidate[1] - target[1]).sum())) for group in groups]
                        reports[str(n)] = report
                    results[f'seed{seed_index}_library{library}'] = dict(seed=seed_index, bank_seed=bank_seed,
                        parents=len(heldout), prefix_reports=reports)
            between_banks = {}
            for seed_index in range(2):
                between_banks[str(seed_index)] = {}
                for label in ('teacher', 'candidate'):
                    between_banks[str(seed_index)][label] = {}
                    for n in prefixes:
                        a = full_moments[seed_index, 0, label, n][1]
                        b = full_moments[seed_index, 1, label, n][1]
                        between_banks[str(seed_index)][label][str(n)] = float(np.linalg.norm(a - b) / (.5 * (np.linalg.norm(a) + np.linalg.norm(b))))
        check_budget(p, counts, started)
        result = dict(status='COMPLETE', config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, banks=results, covariance_between_independent_banks=between_banks,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), original_teacher_gate_changed=False,
            new_fit_or_parameter_updates=0, REAL_response_updates=0, scientific_acceptance=False,
            scope='Same128 frozen-system posthoc heldout parents, preADC signal kernel only; paired truth is diagnostic only.')
        write_json(args.out / 'RESULT.json', result)
        output_bytes = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
        if output_bytes > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered diagnostic output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(), output_bytes_before_complete=output_bytes))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False))


if __name__ == '__main__':
    main()
