"""Complete the one missing physical track and reports after the fixed wall stop."""
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
from n01.observation008 import observe_existing_carrier
from run_joint_response01 import load_joint_setup, progress_file, HYPOTHESES
from run_joint_response01_real import select_real, evaluate, decide, REAL_KEYS
from run_joint_response01_native_real import fit_track, kernel_control
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]


def run(p, setup, out, counts, started):
    prior = ROOT / p['completion_inputs']
    review = json.loads((prior / 'TERMINAL_REVIEW.json').read_text())
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        roles = {key: archive[key].copy() for key in archive.files}
    np.savez(out / 'REAL_CALIBRATION_ROLES.npz', **roles)
    device = setup['source'].device
    source_pool, source_holdout, real_pool, real_holdout = [torch.as_tensor(roles[key], device=device) for key in
        ('source_train_rows', 'source_heldout_rows', 'real_train_rows', 'real_heldout_rows')]
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
    feature_maps = {}
    for label, seed in p['feature_seeds'].items():
        counts['feature_maps_constructed'] += 1
        fmap = JointFeatureMap01(blocks, seed=seed)
        feature_maps[label] = {h: fmap for h in HYPOTHESES}
    del blocks

    neural, physical = setup['joint_models'][1], setup['physical_model']
    with np.load(prior / 'seed1_joint_STATE.npz', allow_pickle=False) as archive:
        theta_joint = torch.as_tensor(archive['theta'].copy(), device=device)
    candidates = dict(M0=(neural, neural.initial()), physical_initial=(physical, physical.initial()),
        joint=(neural, theta_joint))
    theta_physical, physical_summary = fit_track(p, setup, physical, source_pool, real_pool,
        feature_maps['FIT'], feature_maps['CHECK'], real, p['training_seed'] + 100000,
        out, counts, started, 'seed1_physical')
    candidates['physical'] = (physical, theta_physical)
    evaluation = evaluate(p, setup, 1, candidates, source_holdout, real, real_holdout,
        feature_maps['FINAL'], out, counts)
    decision = decide(p, evaluation)
    write_json(out / 'seed1_DECISION.json', decision)
    kernel = kernel_control(p, candidates, source_holdout, 1, out, counts, started)
    tracks = {label: {key: value for key, value in summary.items() if key != 'theta'}
        for label, summary in review['completed_tracks'].items()}
    tracks['seed1_physical'] = physical_summary
    decisions = {'0': json.loads((prior / 'seed0_DECISION.json').read_text()), '1': decision}
    kernels = {'0': json.loads((prior / 'seed0_KERNEL.json').read_text()), '1': kernel}
    passed = all(value['pass_all'] for value in decisions.values())
    return dict(status='COMPLETE', REAL_calibration_candidate_pass_all=passed, decisions=decisions,
        tracks=tracks, kernel_controls=kernels, role_counts={key: len(value) for key, value in roles.items()},
        prior_pilot_status='FAILED_INTERNAL_WALL', prior_native_exit=review['native_exit'],
        prior_execution_counts=review['failure']['execution_counts'],
        combined_execution_counts=dict(Counter(review['failure']['execution_counts']) + counts),
        reused_completed_tracks=['seed0_joint', 'seed0_physical', 'seed1_joint'],
        new_neural_updates=0, new_physical_updates=physical_summary['accepted_updates'],
        completion_inputs=p['completion_inputs'], repeated_aborted_finite_difference_columns=3,
        validation_scope='Frozen-system posthoc; reporting roles already exposed; no full-pipeline unseen claim',
        native_metric_projection_used_for_fitting=True, model_solver_gates_and_random_streams_unchanged=True,
        final_all_TRAIN_refit_pending=True, stage5_validation_pending=True, population_weights_fitted=0,
        DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_stage5_response_validation' if passed else 'diagnose_saved_joint_REAL_failure_before_more_training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered completion config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned completion input changed: ' + name)
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
            progress_file(args.out, counts, 'REAL_completion_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered completion output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(),
            REAL_calibration_candidate_pass_all=result['REAL_calibration_candidate_pass_all']))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False,
                scientific_acceptance=False))


if __name__ == '__main__':
    main()
