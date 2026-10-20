"""Direct original-teacher recovery with the frozen 32-draw joint fitter."""
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
from run_joint_response01 import load_joint_setup, progress_file, random_bank, prediction, teacher_parameters, HYPOTHESES
from run_joint_response01_teacher import check_budget, TeacherBudget01, recovery_decision
from run_joint_response01_c1_teacher import precision_check
from run_joint_response01_c1_precision_pilot import fit_track_precision, evaluate_fresh_seed

ROOT = Path(__file__).resolve().parents[1]


def run(p, setup, out, counts, started):
    device = setup['source'].device
    roles = np.load(ROOT/p['roles'], allow_pickle=False)
    pool, heldout = (torch.as_tensor(roles[name], device=device) for name in ('train_rows', 'heldout_rows'))
    np.savez(out/'TEACHER_ROLES.npz', train_rows=pool.cpu().numpy(), heldout_rows=heldout.cpu().numpy())
    tracks, decisions = {}, {}
    for seed, teacher in enumerate(setup['joint_models']):
        truth = teacher_parameters(teacher)
        student = JointResponse01C1(teacher.cache)
        bank = random_bank(setup, pool, 1, 1940000+10000*seed, counts)
        obs = prediction(teacher, truth, bank, float(setup['loss'].q_floor), counts)
        blocks = {h: observable_blocks(obs[h]) for h in HYPOTHESES}
        maps = {role: {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds'][role]) for h in HYPOTHESES}
                for role in ('FIT', 'CHECK', 'FINAL')}
        counts['feature_maps_constructed'] += 9
        old_factors = np.load(ROOT/p['feature_factors'][seed], allow_pickle=False)
        factors = {}
        for role in maps:
            for h in HYPOTHESES:
                key = role+'_'+h+'_cholesky'
                factors[key] = maps[role][h].cholesky.cpu().numpy()
                np.testing.assert_array_equal(factors[key], old_factors[key])
        np.savez(out/f'seed{seed}_FEATURE_MAPS.npz', **factors)
        write_json(out/f'seed{seed}_FEATURE_SLICES.json', {name: [s.start, s.stop] for name, s in maps['FIT'][HYPOTHESES[0]].block_slices.items()})
        write_json(out/f'seed{seed}_FEATURE_REPLAY.json', dict(pass_all=True, factors=9, statistics_target='original_teacher'))
        del bank, obs, blocks, old_factors
        label = f'seed{seed}_joint'
        saved = np.load(ROOT/p['endpoint_iterations'][seed], allow_pickle=False)
        initial = torch.as_tensor(saved['endpoint'], device=device)
        endpoint, summary = fit_track_precision(p, setup, teacher, truth, student, initial, pool, maps['FIT'], maps['CHECK'],
            'nominal', 'nominal', p['iterations'], seed, out, counts, started, label)
        tracks[label] = summary
        candidates = {'M0': (student, student.initial()), 'joint': (student, endpoint)}
        target_out = out/'original'
        target_out.mkdir(exist_ok=True)
        evaluation = evaluate_fresh_seed(p, setup, seed, teacher, truth, candidates, heldout, maps['FINAL'], target_out, counts)
        low_draw = recovery_decision(p, evaluation)
        hard = [gate for gate in low_draw['gates'] if len(gate['name'].split('/')) == 3]
        precise = precision_check(p, teacher, truth, student, endpoint, heldout, target_out, counts, started, seed)
        decision = dict(pass_all=all(item['pass_gate'] for item in hard) and precise['pass_all'],
            hard_observation_gates=hard, conditional_precision_gates=precise['gates'], diagnostic_64draw_decision=low_draw,
            target_family='original', direct_original_target_fitted=True, automatic_REAL_promotion=False)
        decisions[str(seed)] = {'original': decision}
        write_json(target_out/f'seed{seed}_EVALUATION.json', evaluation)
        write_json(target_out/f'seed{seed}_PRECISION.json', precise)
        write_json(target_out/f'seed{seed}_DECISION.json', decision)
        progress_file(out, counts, 'C1_original_precision_head_complete', seed=seed)
    return dict(status='COMPLETE', tracks=tracks, decisions=decisions, estimator_draws_per_parent=32,
        additional_iterations_per_head=p['iterations'], warm_start_from_frozen_original_endpoints=True,
        feature_statistics_from_original_teacher=True, known_parameters_supplied_to_student=False,
        report_parents_used_for_fitting=False, paired_labels_used_for_fitting=False,
        original_cross_family_failure_preserved=True, direct_original_target_fitted=True,
        automatic_REAL_promotion=False, REAL_response_updates=0, scientific_acceptance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered original-teacher precision config changed')
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
    if sum(path.stat().st_size for path in args.out.rglob('*') if path.is_file()) > p['resources']['output_MiB']*1024**2:
        raise RuntimeError('Original-teacher precision output budget exceeded')
    write_json(args.out/'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest(), scientific_acceptance=False))
