"""Two-seed no-update diagnostic of joint directions under existing macro bounds."""
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
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.joint01_constrained_step import constrained_joint_step
from n01.observation008 import observe_existing_carrier
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction, feature_means,
    residual_vector, risks, finite_jacobian, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents, output_trust, accept_joint
from run_response036 import hard_metrics

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
        raise ValueError('Registered direction diagnostic configuration changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned direction input changed: ' + name)
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
            with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
                source_pool, real_pool = [torch.as_tensor(archive[key], device=device) for key in
                    ('source_train_rows', 'real_train_rows')]
            cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
                q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
            for key in ('window_V', 'baseline100'):
                if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
                    raise ValueError('REAL diagnostic readout differs: ' + key)
            real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
                flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
            del setup['real'], cpu_real
            counts['REAL_observable_replay_calls'] += 1
            counts['REAL_observable_replay_rows'] += len(real['Q'])
            blocks = observable_blocks(select_real(real, real_pool))
            maps = {}
            for label in ('FIT', 'CHECK'):
                fmap = JointFeatureMap01(blocks, seed=p['feature_seeds'][label])
                maps[label] = {h: fmap for h in HYPOTHESES}
                counts['feature_maps_constructed'] += 1
            del blocks
            results = {}
            for seed_index, model in enumerate(setup['joint_models']):
                check_budget(p, counts, started)
                seed, theta = p['training_seed'] + seed_index * 100000, model.initial()
                expected = json.loads((ROOT / p['original_trace_files'][seed_index]).read_text())['iterations'][0]
                parents = draw_parents(source_pool, len(source_pool), seed + 100)[:1024]
                real_rows = draw_parents(real_pool, len(real_pool), seed + 200)[:4096]
                target_obs = select_real(real, real_rows)
                target = feature_means({h: target_obs for h in HYPOTHESES}, maps['FIT'])
                counts['REAL_fit_feature_rows'] += len(real_rows)
                fit = random_bank(setup, parents, 4, seed + 1000, counts)
                predicted = feature_means(prediction(model, theta, fit, setup['loss'].q_floor, counts), maps['FIT'])
                residual = residual_vector(predicted, target, maps['FIT'])
                J = finite_jacobian(model, theta, fit, setup['loss'].q_floor, maps['FIT'], target, counts,
                    p['solver']['finite_difference_step'])
                original, old_geometry = joint_step(residual, J, ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'])
                constrained, new_geometry = constrained_joint_step(residual, J, maps['FIT'], theta, model.lower, model.upper,
                    ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'],
                    macro_factor=p['solver']['macro_degradation_factor'], sample_count=4096,
                    max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
                counts['joint_direction_solves'] += 2
                arrays = dict(residual=residual.cpu().numpy(), jacobian=J.cpu().numpy(),
                    original_delta=original.cpu().numpy(), constrained_delta=constrained.cpu().numpy(),
                    fit_source_parents=parents.cpu().numpy(), fit_real_rows=real_rows.cpu().numpy(),
                    cholesky=maps['FIT'][HYPOTHESES[0]].cholesky.cpu().numpy())
                report = dict(original_geometry=old_geometry, constrained_geometry=new_geometry,
                              FIT_risks=risks(predicted, target, maps['FIT']), checks={})
                parity_error = 0.
                for library, offsets in (('original_CHECK', (3000, 4000, 5000)), ('fresh_CHECK', (7000, 8000, 9000))):
                    check_parents = draw_parents(source_pool, 1024, seed + offsets[0])
                    check_real_rows = draw_parents(real_pool, 4096, seed + offsets[2])
                    arrays[library + '_source_parents'] = check_parents.cpu().numpy()
                    arrays[library + '_real_rows'] = check_real_rows.cpu().numpy()
                    bank = random_bank(setup, check_parents, 4, seed + offsets[1], counts)
                    check_obs = select_real(real, check_real_rows)
                    counts['REAL_CHECK_feature_rows'] += len(check_real_rows)
                    target_check = feature_means({h: check_obs for h in HYPOTHESES}, maps['CHECK'])
                    baseline_obs = prediction(model, theta, bank, setup['loss'].q_floor, counts)
                    baseline_risk = risks(feature_means(baseline_obs, maps['CHECK']), target_check, maps['CHECK'])
                    target_wave, target_base = (check_obs[k].cpu().numpy() for k in ('window_V', 'baseline100'))
                    current = dict(baseline_risks=baseline_risk, candidates=[])
                    if library == 'fresh_CHECK':
                        current['baseline_native'] = {}
                        for h in HYPOTHESES:
                            counts['native_metric_rows'] += 8192
                            current['baseline_native'][h] = hard_metrics(baseline_obs[h]['window_V'].cpu().numpy(),
                                baseline_obs[h]['baseline100'].cpu().numpy(), target_wave, target_base, setup['loss'], setup['context'])
                    for direction, delta in (('original', original), ('constrained', constrained)):
                        for attempt_index, fraction in enumerate(p['solver']['backtracking_fractions']):
                            counts['joint_candidate_trials'] += 1
                            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
                            obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
                            risk = risks(feature_means(obs, maps['CHECK']), target_check, maps['CHECK'])
                            trust = output_trust(baseline_obs, obs, model, bank['rows'], p)
                            entry = dict(direction=direction, fraction=fraction, risks=risk, trust=trust,
                                would_accept=accept_joint(baseline_risk, risk, trust, 4096, p))
                            if library == 'original_CHECK' and direction == 'original':
                                parity_error = max(parity_error, max(abs(risk[h][metric] - expected['attempts'][attempt_index]['risks'][h][metric])
                                    for h in HYPOTHESES for metric in risk[h]))
                            if library == 'fresh_CHECK':
                                entry['native'] = {}
                                for h in HYPOTHESES:
                                    counts['native_metric_rows'] += 8192
                                    entry['native'][h] = hard_metrics(obs[h]['window_V'].cpu().numpy(), obs[h]['baseline100'].cpu().numpy(),
                                        target_wave, target_base, setup['loss'], setup['context'])
                            current['candidates'].append(entry)
                    report['checks'][library] = current
                    progress_file(args.out, counts, 'direction_diagnostic', seed=seed_index, check_library=library)
                if parity_error > 1e-8:
                    raise ValueError('Original rejected direction reproduction differs: ' + str(parity_error))
                report['original_rejected_direction_risk_max_difference'] = parity_error
                results[str(seed_index)] = report
                np.savez(args.out / f'seed{seed_index}_DIRECTION.npz', **arrays)
                write_json(args.out / f'seed{seed_index}_DIAGNOSTIC.json', report)
        check_budget(p, counts, started)
        result = dict(status='COMPLETE', config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic()-started, committed_parameter_updates=0, heldout_rows_used_for_fit_or_check=0,
            original_acceptance_gates_unchanged=True, scientific_acceptance=False,
            constrained_directions_would_pass_fresh_CHECK={seed: any(c['would_accept'] for c in report['checks']['fresh_CHECK']['candidates']
                if c['direction']=='constrained') for seed, report in results.items()},
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB']*1024**2:
            raise ValueError('Diagnostic output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic()-started, automatic_restart=False))


if __name__ == '__main__':
    main()
