"""Frozen endpoint ranking under both teacher families; no parameter fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks
from n01.joint01_model_c1 import JointResponse01C1
from run_joint_response01 import load_joint_setup, progress_file, random_bank, prediction, residual_vector, teacher_parameters, HYPOTHESES
from run_joint_response01_teacher import check_budget, draw_parents, TeacherBudget01
from run_joint_response01_c1_fullpool import cpu_chunks
from run_joint_response01_c1_precision_pilot import libc
from diagnose_joint01_c1_estimator import pooled_three_maps

ROOT = Path(__file__).resolve().parents[1]
MAP_ROLES = ('FIT', 'CHECK', 'FINAL')
LABELS = ('original_oracle', 'C1_known_truth', 'C1_original_initial', 'C1_positive_endpoint', 'C1_original_endpoint')
TARGETS = ('original', 'matched_C1')


def run(p, setup, out, counts, started):
    device = setup['source'].device
    roles = np.load(ROOT/p['roles'], allow_pickle=False)
    pool = torch.as_tensor(roles['train_rows'], device=device)
    reports = {}
    for seed, teacher in enumerate(setup['joint_models']):
        truth = teacher_parameters(teacher)
        student = JointResponse01C1(teacher.cache)
        initial = torch.as_tensor(np.load(ROOT/p['endpoint_iterations'][seed], allow_pickle=False)['endpoint'], device=device)
        positive = torch.as_tensor(np.load(ROOT/p['positive_endpoints'][seed], allow_pickle=False)['theta'], device=device)
        endpoint = torch.as_tensor(np.load(ROOT/p['original_endpoints'][seed], allow_pickle=False)['theta'], device=device)
        stats_bank = random_bank(setup, pool, 1, 1940000+10000*seed, counts)
        stats_obs = prediction(teacher, truth, stats_bank, float(setup['loss'].q_floor), counts)
        blocks = {h: observable_blocks(stats_obs[h]) for h in HYPOTHESES}
        maps = {role: {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds'][role]) for h in HYPOTHESES} for role in MAP_ROLES}
        counts['feature_maps_constructed'] += 9
        factors = np.load(ROOT/p['feature_factors'][seed], allow_pickle=False)
        arrays = dict(pool=pool.cpu().numpy(), initial=initial.cpu().numpy(), positive=positive.cpu().numpy(), endpoint=endpoint.cpu().numpy(), truth=truth.cpu().numpy())
        for role in MAP_ROLES:
            for h in HYPOTHESES:
                key = role+'_'+h+'_cholesky'
                arrays[key] = maps[role][h].cholesky.cpu().numpy()
                np.testing.assert_array_equal(arrays[key], factors[key])
        write_json(out/f'seed{seed}_FEATURE_REPLAY.json', dict(pass_all=True, factors=9))
        del stats_bank, stats_obs, blocks, factors
        libc.malloc_trim(0)
        rows, targets, sources, residuals, streams = [], [], [], [], []
        for repeat in range(p['independent_repeats']):
            orders = [draw_parents(pool, len(pool), p['order_seed']+1000*seed+100*repeat+side) for side in range(2)]
            bases = [p['bank_seed']+1000000*seed+10000*repeat+1000*side for side in range(2)]
            target_bank = cpu_chunks(p, setup, orders[1], 32, bases[1], counts)
            target_means = []
            for model in (teacher, student):
                values = pooled_three_maps(p, model, truth, target_bank, maps, float(setup['loss'].q_floor), counts, started)
                target_means.append(values)
                counts['target_family_evaluations'] += 1
            del target_bank
            libc.malloc_trim(0)
            source_bank = cpu_chunks(p, setup, orders[0], 32, bases[0], counts)
            source_means, vectors = [], []
            for model, theta in zip((teacher, student, student, student, student), (truth, truth, initial, positive, endpoint), strict=True):
                values = pooled_three_maps(p, model, theta, source_bank, maps, float(setup['loss'].q_floor), counts, started)
                source_means.append(np.stack([[values[role][h].cpu().numpy() for h in HYPOTHESES] for role in MAP_ROLES]))
                vectors.append(np.stack([[residual_vector(values[role], target[role], maps[role]).cpu().numpy() for role in MAP_ROLES] for target in target_means]))
                counts['fixed_endpoint_evaluations'] += 1
            rows.append(np.stack([order.cpu().numpy() for order in orders]))
            targets.append(np.stack([[[target[role][h].cpu().numpy() for h in HYPOTHESES] for role in MAP_ROLES] for target in target_means]))
            sources.append(np.stack(source_means))
            residuals.append(np.stack(vectors).transpose(1, 0, 2, 3))
            streams.append(bases)
            counts['independent_bank_pairs'] += 1
            arrays.update(parent_orders=np.stack(rows), target_feature_means=np.stack(targets), source_feature_means=np.stack(sources),
                residuals=np.stack(residuals), bank_base_seeds=np.array(streams, dtype=np.int64))
            np.savez(out/f'seed{seed}_SAVED.npz', **arrays)
            progress_file(out, counts, 'C1_objective_alignment_frozen_repeats', seed=seed, repeat=repeat+1, total=p['independent_repeats'])
            del source_bank
            libc.malloc_trim(0)
        reports[str(seed)] = {}
        n, denominator = p['independent_repeats'], 272.
        for ti, target in enumerate(TARGETS):
            reports[str(seed)][target] = {}
            for ri, role in enumerate(MAP_ROLES):
                reports[str(seed)][target][role] = {}
                for li, label in enumerate(LABELS):
                    v = arrays['residuals'][:, ti, li, ri]
                    total, square = v.sum(0), np.square(v).sum(1)
                    u = (np.square(total).sum()-square.sum())/(n*(n-1)*denominator)
                    loo = np.array([(np.square(total-row).sum()-square.sum()+s)/((n-1)*(n-2)*denominator) for row, s in zip(v, square, strict=True)])
                    reports[str(seed)][target][role][label] = dict(single_32draw_risks=(square/denominator).tolist(),
                        pooled_128draw_risk=float(np.square(v.mean(0)).sum()/denominator), independent_cross_product_risk=float(u),
                        estimated_pooled_MC_noise=float(np.square(v-v.mean(0)).sum()/(n*(n-1)*denominator)),
                        jackknife_standard_error=float(np.sqrt((n-1)/n*np.square(loo-loo.mean()).sum())),
                        leave_one_bank_out_cross_product_risks=loo.tolist())
    return dict(status='COMPLETE', labels=list(LABELS), targets=list(TARGETS), map_roles=list(MAP_ROLES), reports=reports,
        report_parent_rows_used=False, known_truth_is_diagnostic_only=True, all_endpoints_frozen=True,
        paired_source_randomness_across_labels=True, paired_target_randomness_across_families=True,
        source_target_independent=True, parameter_updates=0, REAL_response_updates=0,
        four_bank_jackknife_descriptive_only=True, scientific_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered objective-alignment config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned input changed: '+name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            for key in ('encoding', 'references', 'models', 'baseline', 'heads', 'real'):
                del setup[key]
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
    except TeacherBudget01 as error:
        write_json(args.out/'FAILED.json', dict(status='BUDGET_STOP', error=str(error), execution_counts=dict(counts),
            elapsed_seconds=time.monotonic()-started, UTC=datetime.now(timezone.utc).isoformat()))
        raise
    result.update(config_sha256=args.config_sha256, execution_counts=dict(counts), elapsed_seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.out/'RESULT.json', result)
    if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()) > p['resources']['output_MiB']*1024**2:
        raise RuntimeError('Objective-alignment output budget exceeded')
    write_json(args.out/'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest(), scientific_acceptance=False))
