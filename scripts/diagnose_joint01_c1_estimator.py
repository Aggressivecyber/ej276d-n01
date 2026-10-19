"""Frozen full-parent teacher residuals versus independent-bank noise; no fitting."""
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
from run_joint_response01 import (load_joint_setup, progress_file, random_bank,
    prediction, feature_means, residual_vector, teacher_parameters, HYPOTHESES)
from run_joint_response01_teacher import check_budget, draw_parents, TeacherBudget01
from run_joint_response01_c1_fullpool import cpu_chunks

ROOT = Path(__file__).resolve().parents[1]
MAP_ROLES = ('FIT', 'CHECK', 'FINAL')
LABELS = ('original_oracle', 'C1_known_truth', 'C1_before_last', 'C1_endpoint')


def pooled_three_maps(p, model, theta, chunks, maps, q_floor, counts, started):
    sums = {role: {h: torch.zeros(272, dtype=torch.float64, device=theta.device)
        for h in HYPOTHESES} for role in MAP_ROLES}
    mass = 0
    for chunk in chunks:
        check_budget(p, counts, started)
        bank = dict(rows=chunk['rows'].to(theta.device), z=chunk['z'].to(theta.device),
            noise={h: chunk['noise'][h].to(theta.device) for h in HYPOTHESES})
        observations = prediction(model, theta, bank, q_floor, counts)
        blocks = {h: observable_blocks(observations[h]) for h in HYPOTHESES}
        for role in MAP_ROLES:
            for h in HYPOTHESES:
                sums[role][h] += len(bank['rows']) * maps[role][h].mean(blocks[h])
        mass += len(bank['rows'])
    return {role: {h: sums[role][h] / mass for h in HYPOTHESES} for role in MAP_ROLES}


def summarize_residuals(residuals):
    # r concatenates three whitened 272-vectors divided by sqrt(3).
    # Independent-bank cross-products estimate the squared population discrepancy.
    n, denominator = len(residuals), 272.
    total = residuals.sum(0)
    squared = np.square(residuals).sum(1)
    debiased = (np.square(total).sum() - squared.sum()) / (n * (n - 1) * denominator)
    leave_one_out = np.array([(np.square(total-r).sum() - squared.sum() + s) /
        ((n-1)*(n-2)*denominator) for r,s in zip(residuals,squared,strict=True)])
    return dict(single_four_draw_risks=(squared/denominator).tolist(),
        pooled_sixteen_draw_risks=[float(np.square(part.mean(0)).sum()/denominator)
            for part in np.split(residuals,2)],
        pooled_thirtytwo_draw_risk=float(np.square(residuals.mean(0)).sum()/denominator),
        estimated_pooled_MC_noise=float(np.square(residuals-residuals.mean(0)).sum()/(n*(n-1)*denominator)),
        independent_cross_product_risk=float(debiased),
        jackknife_standard_error=float(np.sqrt((n-1)/n*np.square(leave_one_out-leave_one_out.mean()).sum())),
        leave_one_bank_out_cross_product_risks=leave_one_out.tolist())


def run(p, setup, out, counts, started):
    device = setup['source'].device
    roles = np.load(ROOT/p['roles'], allow_pickle=False)
    pool = torch.as_tensor(roles['train_rows'], device=device)
    reports = {}
    for seed_index, teacher in enumerate(setup['joint_models']):
        truth = teacher_parameters(teacher)
        student = JointResponse01C1(teacher.cache)
        saved = np.load(ROOT/p['endpoint_iterations'][seed_index], allow_pickle=False)
        before = torch.as_tensor(saved['theta'], device=device)
        endpoint = torch.as_tensor(saved['endpoint'], device=device)
        stats_bank = random_bank(setup,pool,1,1940000+10000*seed_index,counts)
        stats_obs = prediction(teacher,truth,stats_bank,float(setup['loss'].q_floor),counts)
        blocks = {h: observable_blocks(stats_obs[h]) for h in HYPOTHESES}
        maps = {role: {h: JointFeatureMap01(blocks[h],seed=p['feature_seeds'][role])
            for h in HYPOTHESES} for role in MAP_ROLES}
        counts['feature_maps_constructed'] += 9
        factors = np.load(ROOT/p['feature_factors'][seed_index], allow_pickle=False)
        arrays = dict(pool=pool.cpu().numpy(),before=before.cpu().numpy(),endpoint=endpoint.cpu().numpy(),truth=truth.cpu().numpy())
        for role in MAP_ROLES:
            for h in HYPOTHESES:
                factor = maps[role][h].cholesky.cpu().numpy()
                np.testing.assert_array_equal(factor,factors[role+'_'+h+'_cholesky'])
                arrays[role+'_'+h+'_cholesky'] = factor
        write_json(out/f'seed{seed_index}_FEATURE_REPLAY.json',dict(pass_all=True,factors=9))
        rows, targets, sources, residuals, streams = [], [], [], [], []
        for repeat in range(p['independent_repeats']):
            orders = [draw_parents(pool,len(pool),p['order_seed']+seed_index*100+2*repeat+side) for side in range(2)]
            bases = [p['bank_seed']+seed_index*100000+repeat*1000+side*100 for side in range(2)]
            source_bank,target_bank = [cpu_chunks(p,setup,order,4,stream,counts)
                for order,stream in zip(orders,bases,strict=True)]
            target = pooled_three_maps(p,teacher,truth,target_bank,maps,float(setup['loss'].q_floor),counts,started)
            del target_bank
            source_means, vectors = [], []
            for label,model,theta in zip(LABELS,(teacher,student,student,student),(truth,truth,before,endpoint),strict=True):
                values = pooled_three_maps(p,model,theta,source_bank,maps,float(setup['loss'].q_floor),counts,started)
                source_means.append(np.stack([[values[role][h].cpu().numpy() for h in HYPOTHESES] for role in MAP_ROLES]))
                vectors.append(np.stack([residual_vector(values[role],target[role],maps[role]).cpu().numpy() for role in MAP_ROLES]))
                counts['fixed_endpoint_evaluations'] += 1
            rows.append(np.stack([order.cpu().numpy() for order in orders]))
            targets.append(np.stack([[target[role][h].cpu().numpy() for h in HYPOTHESES] for role in MAP_ROLES]))
            sources.append(np.stack(source_means));residuals.append(np.stack(vectors));streams.append(bases)
            counts['independent_bank_pairs'] += 1
            arrays.update(parent_orders=np.stack(rows),target_feature_means=np.stack(targets),
                source_feature_means=np.stack(sources),residuals=np.stack(residuals),bank_base_seeds=np.array(streams,dtype=np.int64))
            np.savez(out/f'seed{seed_index}_SAVED.npz',**arrays)
            progress_file(out,counts,'C1_estimator_frozen_repeats',seed=seed_index,repeat=repeat+1,total=p['independent_repeats'])
            del source_bank
        reports[str(seed_index)] = {role: {label: summarize_residuals(arrays['residuals'][:,li,ri])
            for li,label in enumerate(LABELS)} for ri,role in enumerate(MAP_ROLES)}
    return dict(status='COMPLETE',reports=reports,labels=list(LABELS),map_roles=list(MAP_ROLES),
        report_parent_rows_used=False,paired_conditional_labels_used_for_fitting=False,
        known_teacher_only_a_fixed_diagnostic_control=True,parameter_updates=0,REAL_response_updates=0,
        scientific_acceptance=False,diagnostic_not_a_replacement_scientific_gate=True,
        negative_cross_product_estimates_preserved=True,small_eight_bank_jackknife_descriptive_only=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256: raise ValueError('Registered estimator diagnostic config changed')
    p=json.loads(raw)
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Pinned input changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    counts,started=Counter(),time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    try:
        with torch.no_grad():
            setup=load_joint_setup(p,device,counts)
            for key in ('encoding','references','models','baseline','heads','real'): del setup[key]
            result=run(p,setup,args.out,counts,started)
        check_budget(p,counts,started)
    except TeacherBudget01 as error:
        write_json(args.out/'FAILED.json',dict(status='BUDGET_STOP',error=str(error),execution_counts=dict(counts),
            elapsed_seconds=time.monotonic()-started,UTC=datetime.now(timezone.utc).isoformat()))
        raise
    result.update(config_sha256=args.config_sha256,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.out/'RESULT.json',result)
    if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file())>p['resources']['output_MiB']*1024**2:
        raise RuntimeError('Estimator diagnostic output budget exceeded')
    write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
