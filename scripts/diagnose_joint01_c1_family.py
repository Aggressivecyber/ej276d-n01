"""Frozen 2x2 interpolation-family/parameter contrast; paired truth is diagnostic only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_model_c1 import JointResponse01C1
from n01.model008 import canonical_charge008
from diagnose_joint01_kernel import compare_moments
from run_joint_response01 import load_joint_setup, progress_file, teacher_parameters
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
        raise ValueError('Registered C1 family contrast changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned family contrast input changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        models = setup['joint_models']
        del setup
        with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
            heldout = archive['heldout_rows'].copy()
        projection = torch.cos(torch.arange(384, device=device, dtype=torch.float64)[:, None] *
            torch.arange(1, 17, device=device)[None] * torch.pi / 384.) / 384.**.5
        reports, replay = {}, []
        for seed, original in enumerate(models):
            c1 = JointResponse01C1(original.cache)
            truth = teacher_parameters(original)
            with np.load(ROOT / p['endpoint_files'][seed], allow_pickle=False) as archive:
                endpoint = torch.as_tensor(archive['theta'], device=device)
            for library in range(2):
                generator = torch.Generator().manual_seed(p['bank_seeds'][seed][library])
                counts['allocated_banks'] += 1
                cells = {'original_truth': (original, truth), 'C1_truth': (c1, truth),
                         'original_endpoint': (original, endpoint), 'C1_endpoint': (c1, endpoint)}
                samples = {name: [] for name in cells}
                for position, parent in enumerate(heldout):
                    check_budget(p, counts, started)
                    rows = torch.full((p['draws'],), int(parent), device=device, dtype=torch.long)
                    z = torch.randn(p['draws'], 8, generator=generator, dtype=torch.float32).to(device)
                    counts['shape_standard_normal_values'] += z.numel()
                    for name, (model, theta) in cells.items():
                        signal = model(theta, rows, z)
                        counts['joint_response_calls'] += 1
                        counts['joint_response_rows'] += len(rows)
                        reduced = signal @ projection
                        centered = reduced - reduced.mean(0)
                        covariance = centered.T @ centered / (len(rows) - 1)
                        samples[name].append((signal.mean(0).cpu().numpy(), covariance.cpu().numpy(),
                            float(canonical_charge008(signal).var()), float(signal.var(0).mean())))
                    if (position + 1) % 64 == 0:
                        progress_file(args.out, counts, 'C1_family_contrast', seed=seed,library=library,parents_complete=position+1)
                arrays, moments = {}, {}
                for name, entries in samples.items():
                    moments[name] = tuple(np.stack([entry[i] for entry in entries]) for i in range(4))
                    for field, value in zip(('mean','covariance','Q_variance','wave_variance'),moments[name],strict=True):
                        arrays[name+'_'+field] = value
                with np.load(ROOT / p['reference_moment_files'][seed][library], allow_pickle=False) as reference:
                    if not np.array_equal(reference['heldout_rows'], heldout):
                        raise ValueError('Family contrast role ordering differs')
                    for cell, previous in (('original_truth','teacher'),('C1_endpoint','candidate')):
                        for field in ('mean','covariance','Q_variance','wave_variance'):
                            if not np.array_equal(arrays[cell+'_'+field], reference[previous+'_'+field]):
                                raise ValueError('Frozen moment byte replay differs: '+cell+'/'+field)
                            replay.append(dict(seed=seed,library=library,cell=cell,moment=field,byte_equal=True))
                np.savez(args.out / f'seed{seed}_library{library}_FAMILY_MOMENTS.npz',heldout_rows=heldout,**arrays)
                key = f'seed{seed}/library{library}'
                reports[key] = dict(comparisons={},decomposition={})
                for candidate, target in (('C1_truth','original_truth'),('original_endpoint','original_truth'),
                                          ('C1_endpoint','original_truth'),('C1_endpoint','C1_truth')):
                    statistics = compare_moments(moments[candidate],moments[target])
                    del statistics['top13_of128_parent_positions']
                    reports[key]['comparisons'][candidate+'_vs_'+target] = statistics
                for index, name in enumerate(('mean','covariance','Q_variance','wave_variance')):
                    a,b,c,d = (moments[cell][index] for cell in ('original_truth','C1_truth','original_endpoint','C1_endpoint'))
                    total, family, parameters = d-a,b-a,d-b
                    denominator = np.square(a).sum()
                    reports[key]['decomposition'][name] = dict(
                        total_squared_relative_error=float(np.square(total).sum()/denominator),
                        family_at_truth_squared_relative_error=float(np.square(family).sum()/denominator),
                        parameter_at_C1_squared_relative_error=float(np.square(parameters).sum()/denominator),
                        cross_term_relative=float(2*(family*parameters).sum()/denominator),
                        family_parameter_interaction_relative_norm=float(np.linalg.norm((d-c)-(b-a))/np.linalg.norm(a)))
    check_budget(p,counts,started)
    result=dict(status='COMPLETE',config_sha256=args.config_sha256,execution_counts=dict(counts),reports=reports,
        replay=replay,existing_moment_arrays_exactly_replayed=len(replay),elapsed_seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        parameter_updates=0,REAL_response_updates=0,scientific_acceptance=False,
        scope='Fixed 2x2 contrast of interpolation family and truth/fitted parameters on the same exposed 128 reporting parents and 2048-draw banks. The decomposition is algebraic, order-dependent and diagnostic; no conditional labels enter fitting or select a candidate.')
    write_json(args.out/'RESULT.json',result)
    output_bytes=sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
    if output_bytes>p['resources']['output_MiB']*1024**2:
        raise RuntimeError('Family contrast output budget exceeded')
    write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
