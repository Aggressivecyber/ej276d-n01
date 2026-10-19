"""Complete the remaining registered multibank REAL work after a wall-budget stop."""
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
from diagnose_joint01_secant_trust import hard_check
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, HYPOTHESES)
from run_joint_response01_real import select_real, evaluate, decide, REAL_KEYS
from run_joint_response01_native_real import kernel_control
from run_joint_response01_multibank_real import fit_track as original_fit_track
from run_joint_response01_teacher import check_budget, draw_parents

ROOT = Path(__file__).resolve().parents[1]
FIT_LABELS = ('FIT0', 'FIT1', 'FIT2')


def fit_remaining_joint_track(p, setup, model, source_pool, real_pool, maps, contexts, real, seed, out, counts, started, label):
    with np.load(ROOT / p['completion_inputs']['seed1_state'], allow_pickle=False) as archive:
        saved_state = {k: archive[k].copy() for k in archive.files}
    theta = torch.as_tensor(saved_state['theta'], device=source_pool.device)
    initial = model.initial()
    source_coverage = torch.as_tensor(saved_state['source_coverage'], device=source_pool.device)
    real_coverage = torch.as_tensor(saved_state['real_coverage'], device=real_pool.device)
    with np.load(ROOT / p['completion_inputs']['seed1_partial_arrays'], allow_pickle=False) as archive:
        replay = {k: archive[k].copy() for k in archive.files}
    replay_report = json.loads((ROOT / p['completion_inputs']['seed1_partial_report']).read_text())
    batch, draws, real_batch = (p['batch'][k] for k in ('source_parents', 'draws_per_parent', 'target_rows'))
    order = {}
    for name, pool, number, offset in (('source', source_pool, batch, 100), ('real', real_pool, real_batch, 200)):
        visits = p['iterations'] * len(FIT_LABELS) * number
        epochs = (visits + len(pool) - 1) // len(pool)
        order[name] = torch.cat([draw_parents(pool, len(pool), seed + offset + epoch) for epoch in range(epochs)])
    role_setup = {part: dict(setup, context=contexts[part]) for part in (*FIT_LABELS, 'CHECK')}
    joint_maps = {part + '/' + h: maps[part][h] for part in FIT_LABELS for h in HYPOTHESES}
    trace = json.loads((ROOT / p['completion_inputs']['seed1_trace']).read_text())['iterations']
    for iteration in range(p['resume']['completed_iterations'], p['iterations']):
        check_budget(p, counts, started)
        stem = f'{label}_ITER{iteration:02d}'
        arrays = dict(theta=theta.cpu().numpy().copy(), initial=initial.cpu().numpy(),
            lower=model.lower.cpu().numpy(), upper=model.upper.cpu().numpy())
        banks, targets, features, baseline, old_risk, old_native, anchor_native, residuals = {}, {}, {}, {}, {}, {}, {}, {}
        for index, part in enumerate((*FIT_LABELS, 'CHECK')):
            if part in FIT_LABELS:
                position = iteration * len(FIT_LABELS) + index
                parents = order['source'][position * batch:(position + 1) * batch]
                real_rows = order['real'][position * real_batch:(position + 1) * real_batch]
                bank_seed = seed + 1000 + iteration * 100 + index * 10
                source_coverage.index_add_(0, parents, torch.ones_like(parents))
                real_coverage.index_add_(0, real_rows, torch.ones_like(real_rows))
                counts['REAL_fit_feature_rows'] += len(real_rows)
            else:
                parents = draw_parents(source_pool, batch, seed + 50000 + iteration)
                real_rows = draw_parents(real_pool, real_batch, seed + 60000 + iteration)
                bank_seed = seed + 70000 + iteration * 100
                counts['REAL_CHECK_feature_rows'] += len(real_rows)
            arrays[part + '_source_parents'], arrays[part + '_real_rows'] = parents.cpu().numpy(), real_rows.cpu().numpy()
            arrays[part + '_cholesky'] = maps[part][HYPOTHESES[0]].cholesky.cpu().numpy()
            banks[part] = random_bank(setup, parents, draws, bank_seed, counts)
            targets[part] = select_real(real, real_rows)
            features[part] = feature_means({h: targets[part] for h in HYPOTHESES}, maps[part])
            baseline[part] = prediction(model, theta, banks[part], setup['loss'].q_floor, counts)
            means = feature_means(baseline[part], maps[part])
            old_risk[part] = risks(means, features[part], maps[part])
            residuals[part] = residual_vector(means, features[part], maps[part])
            old_native[part] = native_values(baseline[part], targets[part], role_setup[part], counts)
            anchor = prediction(model, initial, banks[part], setup['loss'].q_floor, counts)
            anchor_native[part] = native_values(anchor, targets[part], role_setup[part], counts)
            arrays[part + '_baseline_residual'] = residuals[part].cpu().numpy()
            arrays[part + '_baseline_native'] = old_native[part].cpu().numpy()
            arrays[part + '_initial_native'] = anchor_native[part].cpu().numpy()
        del anchor
        r = torch.cat([residuals[part] for part in FIT_LABELS]) / np.sqrt(3)
        n = torch.cat([old_native[part] for part in FIT_LABELS])
        arrays['stacked_baseline_residual'], arrays['stacked_baseline_native'] = r.cpu().numpy(), n.cpu().numpy()
        report = dict(track=label, iteration=iteration, baseline_risks=old_risk,
            baseline_native={k: v.tolist() for k, v in old_native.items()},
            initial_native={k: v.tolist() for k, v in anchor_native.items()},
            native_projection_contexts={k: contexts[k].metadata() for k in (*FIT_LABELS, 'CHECK')},
            FD_columns_completed=0, trials=[], selected_trial=None, stop_reason='trial_budget', current_solve=None)
        for key, value in arrays.items():
            expected = replay[key]
            if value.dtype != expected.dtype or value.shape != expected.shape or value.tobytes() != expected.tobytes():
                raise ValueError('Resume baseline differs from saved input: ' + key)
        for key in ('baseline_risks', 'baseline_native', 'initial_native', 'native_projection_contexts'):
            if report[key] != replay_report[key]:
                raise ValueError('Resume baseline report differs: ' + key)
        plus_r, minus_r, plus_n, minus_n, plus_theta, minus_theta = [], [], [], [], [], []
        for coordinate in range(model.parameter_count):
            check_budget(p, counts, started)
            plus, minus = theta.clone(), theta.clone()
            plus[coordinate] = min(float(model.upper[coordinate]), float(theta[coordinate]) + p['finite_difference_step'])
            minus[coordinate] = max(float(model.lower[coordinate]), float(theta[coordinate]) - p['finite_difference_step'])
            pair_r, pair_n = [], []
            for proposal in (plus, minus):
                br, bn = [], []
                for part in FIT_LABELS:
                    obs = prediction(model, proposal, banks[part], setup['loss'].q_floor, counts)
                    means = feature_means(obs, maps[part])
                    br.append(residual_vector(means, features[part], maps[part]))
                    bn.append(native_values(obs, targets[part], role_setup[part], counts))
                pair_r.append(torch.cat(br) / np.sqrt(3)); pair_n.append(torch.cat(bn))
            plus_r.append(pair_r[0]); minus_r.append(pair_r[1]); plus_n.append(pair_n[0]); minus_n.append(pair_n[1])
            plus_theta.append(plus); minus_theta.append(minus)
            counts['finite_difference_columns'] += 1
            counts['FIT_bank_finite_difference_columns'] += 3
            if (coordinate + 1) % 9 == 0 or coordinate + 1 == model.parameter_count:
                for key, values in dict(FD_plus_residual=plus_r, FD_minus_residual=minus_r,
                        FD_plus_native=plus_n, FD_minus_native=minus_n, FD_plus_theta=plus_theta, FD_minus_theta=minus_theta).items():
                    arrays[key] = torch.stack(values, 1).cpu().numpy()
                if coordinate + 1 == p['resume']['saved_FD_columns']:
                    for key, expected in replay.items():
                        value = arrays[key]
                        if value.dtype != expected.dtype or value.shape != expected.shape or value.tobytes() != expected.tobytes():
                            raise ValueError('Resume saved finite difference differs: ' + key)
                    write_json(out / 'RESTART_REPLAY.json', dict(
                        status='PASS_NATIVE_BASELINE_AND_SAVED_FD_BYTE_REPLAY',
                        prior_completed_iterations=p['resume']['completed_iterations'],
                        saved_arrays_replayed=len(replay), saved_FD_columns_replayed=coordinate + 1,
                        prior_partial_FD_columns_consumed=p['resume']['partial_FD_columns_consumed'],
                        prior_partial_FD_columns_without_saved_vectors=p['resume']['partial_FD_columns_consumed'] - coordinate - 1,
                        execution_counts=dict(counts), new_solves=counts['joint_direction_solves'],
                        new_committed_updates=counts['REAL_response_updates']))
                report['FD_columns_completed'] = coordinate + 1
                np.savez(out / (stem + '.npz'), **arrays)
                write_json(out / (stem + '.json'), report)
                progress_file(out, counts, 'REAL_multibank_joint_Jacobian', track=label, iteration=iteration + 1, column=coordinate + 1)
        difference = torch.stack(plus_theta, 1).diag() - torch.stack(minus_theta, 1).diag()
        J = (torch.stack(plus_r, 1) - torch.stack(minus_r, 1)) / difference
        N = (torch.stack(plus_n, 1) - torch.stack(minus_n, 1)) / difference
        arrays['original_jacobian'], arrays['original_native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
        radius = p['initial_radius']
        for trial_index in range(p['max_trials_per_iteration']):
            check_budget(p, counts, started)
            prefix = f'trial{trial_index}_'
            arrays[prefix + 'jacobian'], arrays[prefix + 'native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
            counts['joint_direction_solve_attempts'] += 1
            report['current_solve'] = dict(trial=trial_index, radius=radius)
            np.savez(out / (stem + '.npz'), **arrays); write_json(out / (stem + '.json'), report)
            progress_file(out, counts, 'REAL_multibank_joint_solve', track=label, iteration=iteration + 1, trial=trial_index + 1)
            unit_delta, geometry = native_joint_step(r, J * radius, joint_maps, n, N * radius,
                theta / radius, model.lower / radius, model.upper / radius,
                ridge_fraction=p['solver']['ridge_fraction'], radius=1.,
                macro_factor=p['solver']['macro_degradation_factor'], sample_count=4096,
                max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
            delta = unit_delta * radius
            geometry.update(solver='SLSQP_multibank_full_joint_native_nonincrease_radius_scaled_variable',
                variable_scale=radius, unit_step_norm=geometry['step_norm'], unit_ridge=geometry['ridge'],
                step_norm=float(delta.norm()), ridge=geometry['ridge'] / radius**2)
            counts['joint_direction_solves'] += 1; report['current_solve'] = None
            if float(delta.square().sum()) == 0:
                report['stop_reason'], report['zero_direction_geometry'] = 'zero_joint_direction', geometry
                break
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + delta))
            step = proposal - theta
            counts['joint_candidate_trials'] += 1
            pr, pn = r + J @ step, n + N @ step
            arrays[prefix + 'proposal'] = proposal.cpu().numpy()
            arrays[prefix + 'predicted_residual'], arrays[prefix + 'predicted_native'] = pr.cpu().numpy(), pn.cpu().numpy()
            trial = dict(radius=radius, geometry=geometry, checks={})
            br, bn = [], []
            for part in (*FIT_LABELS, 'CHECK'):
                ar, an, check = hard_check(p, model, proposal, banks[part], targets[part], features[part], maps[part],
                    baseline[part], old_risk[part], old_native[part], anchor_native[part], role_setup[part], counts)
                arrays[prefix + part + '_actual_residual'], arrays[prefix + part + '_actual_native'] = ar.cpu().numpy(), an.cpu().numpy()
                trial['checks'][part] = check
                if part in FIT_LABELS: br.append(ar); bn.append(an)
            fr, fn = torch.cat(br) / np.sqrt(3), torch.cat(bn)
            trial['accepted'] = all(check['accepted'] for check in trial['checks'].values())
            predicted_gain, actual_gain = float(r.square().sum() - pr.square().sum()), float(r.square().sum() - fr.square().sum())
            trial.update(predicted_joint_reduction=predicted_gain, actual_joint_reduction=actual_gain,
                reduction_ratio=None if predicted_gain <= 0 else actual_gain / predicted_gain,
                FIT_joint_vector_relative_error=float((fr - pr).norm() / r.norm()),
                FIT_native_relative_error=((fn - pn) / n).reshape(3, 3, 3).tolist())
            report['trials'].append(trial)
            np.savez(out / (stem + '.npz'), **arrays); write_json(out / (stem + '.json'), report)
            if trial['accepted']:
                theta = proposal; counts['REAL_response_updates'] += 1
                report['selected_trial'], report['stop_reason'] = trial_index, 'first_all_FIT_CHECK_pass'
                break
            if trial_index + 1 < p['max_trials_per_iteration']:
                J += torch.outer(fr - r - J @ step, step) / step.square().sum()
                N += torch.outer(fn - n - N @ step, step) / step.square().sum()
                counts['secant_matrix_corrections'] += 1
                if max(float((J @ step - (fr - r)).abs().max()), float((N @ step - (fn - n)).abs().max())) > 1e-11:
                    raise ValueError('Multibank REAL FIT-only secant identity failed')
                radius *= p['rejection_radius_factor']
        arrays['endpoint'] = theta.cpu().numpy().copy()
        np.savez(out / (stem + '.npz'), **arrays); write_json(out / (stem + '.json'), report)
        trace.append(dict(iteration=iteration, accepted=report['selected_trial'] is not None,
            selected_trial=report['selected_trial'], stop_reason=report['stop_reason'], parameter_norm=float(theta.norm()),
            evidence=stem, trials=len(report['trials'])))
        write_json(out / (label + '_TRACE.json'), dict(track=label, iterations=trace, paired_supervision=False,
            population_weights_fitted=False, native_projection_used_for_direction_fitting=True))
        np.savez(out / (label + '_STATE.npz'), theta=theta.cpu().numpy(),
            source_coverage=source_coverage.cpu().numpy(), real_coverage=real_coverage.cpu().numpy())
        progress_file(out, counts, 'REAL_multibank_joint_training', track=label, iteration=iteration + 1,
            accepted_updates=sum(item['accepted'] for item in trace))
    return theta, dict(track=label, accepted_updates=sum(item['accepted'] for item in trace),
        source_parents_visited=int((source_coverage > 0).sum()), allowed_source_parents=len(source_pool),
        real_records_visited=int((real_coverage > 0).sum()), allowed_real_records=len(real_pool),
        source_visit_sum=int(source_coverage.sum()), real_visit_sum=int(real_coverage.sum()))


def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        roles = {k: archive[k].copy() for k in archive.files}
    np.savez(out / 'REAL_CALIBRATION_ROLES.npz', **roles)
    source_pool, source_holdout, real_pool, real_holdout = [torch.as_tensor(roles[k], device=device) for k in
        ('source_train_rows', 'source_heldout_rows', 'real_train_rows', 'real_heldout_rows')]
    cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
        q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
    for key in ('window_V', 'baseline100'):
        if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
            raise ValueError('REAL readout differs from frozen loader: ' + key)
    real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
        flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
    counts['REAL_observable_replay_calls'] += 1; counts['REAL_observable_replay_rows'] += len(real['Q'])
    del setup['real'], cpu_real
    blocks = observable_blocks(select_real(real, real_pool)); maps, contexts = {}, {}
    for part, seed in p['feature_seeds'].items():
        fmap = JointFeatureMap01(blocks, seed=seed); maps[part] = {h: fmap for h in HYPOTHESES}
        contexts[part] = setup['loss'].create_evaluation_context(seed=p['native_projection_seeds'][part])
        counts['feature_maps_constructed'] += 1; counts['new_native_projection_contexts'] += 1
    del blocks
    prior_review = json.loads((ROOT / p['completion_inputs']['prior_seed0_review']).read_text())
    tracks = {}
    for label, entry in prior_review['tracks'].items():
        tracks[label] = dict(track=label, accepted_updates=entry['accepted_updates'],
            source_parents_visited=entry['coverage']['source']['unique'], allowed_source_parents=len(source_pool),
            real_records_visited=entry['coverage']['real']['unique'], allowed_real_records=len(real_pool),
            source_visit_sum=entry['coverage']['source']['visits'], real_visit_sum=entry['coverage']['real']['visits'])
    for key, filename in (('seed0_evaluation', 'seed0_EVALUATION.json'),
            ('seed0_decision', 'seed0_DECISION.json'), ('seed0_kernel', 'seed0_KERNEL.json')):
        (out / filename).write_bytes((ROOT / p['completion_inputs'][key]).read_bytes())
    decisions = {'0': json.loads((out / 'seed0_DECISION.json').read_text())}
    kernels = {'0': json.loads((out / 'seed0_KERNEL.json').read_text())}
    neural, physical = setup['joint_models'][1], setup['physical_model']
    candidates = dict(M0=(neural, neural.initial()), physical_initial=(physical, physical.initial()))
    endpoint, summary = fit_remaining_joint_track(p, setup, neural, source_pool, real_pool, maps, contexts, real,
        p['training_seed'] + 100000, out, counts, started, 'seed1_joint')
    candidates['joint'], tracks['seed1_joint'] = (neural, endpoint), summary
    endpoint, summary = original_fit_track(p, setup, physical, source_pool, real_pool, maps, contexts, real,
        p['training_seed'] + 100000, out, counts, started, 'seed1_physical')
    candidates['physical'], tracks['seed1_physical'] = (physical, endpoint), summary
    evaluation = evaluate(p, dict(setup, context=contexts['FINAL']), 1, candidates, source_holdout,
        real, real_holdout, maps['FINAL'], out, counts)
    decisions['1'] = decide(p, evaluation)
    write_json(out / 'seed1_DECISION.json', decisions['1'])
    kernels['1'] = kernel_control(p, candidates, source_holdout, 1, out, counts, started)
    prior_failure = json.loads((ROOT / p['completion_inputs']['prior_failure']).read_text())
    combined = Counter(prior_failure['execution_counts']) + counts
    return dict(status='COMPLETE', REAL_calibration_candidate_pass_all=all(x['pass_all'] for x in decisions.values()),
        decisions=decisions, tracks=tracks, kernel_controls=kernels, role_counts={k: len(v) for k, v in roles.items()},
        prior_execution_counts=prior_failure['execution_counts'], combined_execution_counts=dict(combined),
        prior_committed_parameter_updates=prior_failure['execution_counts']['REAL_response_updates'],
        new_committed_parameter_updates=counts['REAL_response_updates'],
        combined_committed_parameter_updates=combined['REAL_response_updates'],
        duplicate_partial_FD_columns=p['resume']['partial_FD_columns_consumed'],
        original_iteration_limit_per_track=p['iterations'], full_four_track_comparison_complete=True,
        prior_job_id=p['prior_job_id'], prior_terminal_review=p['completion_inputs']['prior_terminal_review'],
        prior_seed0_evidence_reused_without_new_generation=True,
        validation_scope='Frozen-system posthoc reporting roles already exposed; not full-pipeline unseen validation',
        physical_and_neural_solver_and_iteration_budgets_matched=True,
        final_all_TRAIN_refit_pending=True, stage5_validation_pending=True, scientific_acceptance=False,
        population_weights_fitted=0, DEV_waveforms_decoded=0,
        next_action='register_stage5_response_validation' if all(x['pass_all'] for x in decisions.values()) else
            'diagnose_saved_joint_REAL_failure_before_more_training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256: raise ValueError('Registered multibank REAL config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest: raise ValueError('Pinned multibank REAL input changed: ' + name)
    prerequisite = json.loads((ROOT / p['multibank_method_review']).read_text())
    if not prerequisite['two_seed_FRESH_pass']: raise ValueError('Two-seed multibank method prerequisite did not pass')
    terminal = json.loads((ROOT / p['completion_inputs']['prior_terminal_review']).read_text())
    if terminal['status'] != 'PASS_TERMINAL_BUDGET_STOP_AND_SAVED_STATE_AUDIT':
        raise ValueError('Previous multibank budget stop has not passed terminal audit')
    args.out.mkdir(parents=True, exist_ok=False); counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads']); torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            for key in ('encoding', 'references', 'models', 'baseline', 'heads'): del setup[key]
            progress_file(args.out, counts, 'REAL_multibank_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2: raise ValueError('Registered multibank REAL output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256((args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error), execution_counts=dict(counts),
                elapsed_seconds=time.monotonic() - started, scientific_acceptance=False, automatic_restart=False))
