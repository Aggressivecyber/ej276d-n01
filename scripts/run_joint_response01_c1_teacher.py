"""C1 learners recover the unchanged original coupled teachers, with fresh MC precision."""
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
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.joint01_model_c1 import JointResponse01C1
from n01.model008 import canonical_charge008
from diagnose_joint01_kernel import compare_moments
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, finite_jacobian, teacher_parameters, conditional_report, HYPOTHESES)
from run_joint_response01_teacher import (check_budget, draw_parents, population_weights,
    output_trust, accept_joint, evaluate_seed, recovery_decision)

ROOT = Path(__file__).resolve().parents[1]


def fit_track_c1(p, setup, teacher, teacher_theta, model, initial, pool, maps, check_maps,
                fit_population, target_population, iterations, seed, out, counts, started, name):
    q_floor, theta = float(setup['loss'].q_floor), initial.clone()
    batch, draws = p['batch']['source_parents'], p['batch']['draws_per_parent']
    fit_weights, target_weights = population_weights(setup, fit_population), population_weights(setup, target_population)
    coverage = torch.zeros(len(setup['source']), dtype=torch.long, device=pool.device)
    target_coverage = torch.zeros_like(coverage)
    # Cycling independently permuted lists covers the full allowed teacher role.
    source_order, target_order = [], []
    for epoch in range((iterations * batch + len(pool) - 1) // len(pool)):
        source_order.append(draw_parents(pool, len(pool), seed + 100 + epoch))
        target_order.append(draw_parents(pool, len(pool), seed + 200 + epoch))
    source_order, target_order = torch.cat(source_order), torch.cat(target_order)
    trace = []
    for iteration in range(iterations):
        check_budget(p, counts, started)
        fit_parents = source_order[iteration * batch:(iteration + 1) * batch]
        target_parents = target_order[iteration * batch:(iteration + 1) * batch]
        fit = random_bank(setup, fit_parents, draws, seed + 1000 + 20 * iteration, counts)
        target_bank = random_bank(setup, target_parents, draws, seed + 1005 + 20 * iteration, counts)
        target = feature_means(prediction(teacher, teacher_theta, target_bank, q_floor, counts), maps,
                               target_weights[target_bank['rows']])
        predicted = feature_means(prediction(model, theta, fit, q_floor, counts), maps, fit_weights[fit['rows']])
        residual = residual_vector(predicted, target, maps)
        J = finite_jacobian(model, theta, fit, q_floor, maps, target, counts,
                             p['solver']['finite_difference_step'], weights=fit_weights[fit['rows']])
        arrays = dict(theta=theta.cpu().numpy().copy(), initial=initial.cpu().numpy(),
            lower=model.lower.cpu().numpy(), upper=model.upper.cpu().numpy(),
            fit_parents=fit_parents.cpu().numpy(), target_parents=target_parents.cpu().numpy(),
            fit_seed=np.array(fit['seed']), target_seed=np.array(target_bank['seed']),
            residual=residual.cpu().numpy(), jacobian=J.cpu().numpy())
        counts['joint_direction_solves'] += 1
        delta, geometry = joint_step(residual, J, ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'])
        arrays['delta'] = delta.cpu().numpy()
        np.savez(out / f'{name}_ITER{iteration:02d}.npz', **arrays)
        check = random_bank(setup, draw_parents(pool, batch, seed + 3000 + iteration), draws,
                            seed + 4000 + 20 * iteration, counts)
        check_target_bank = random_bank(setup, draw_parents(pool, batch, seed + 5000 + iteration), draws,
                                        seed + 6000 + 20 * iteration, counts)
        check_target = feature_means(prediction(teacher, teacher_theta, check_target_bank, q_floor, counts), check_maps,
                                    target_weights[check_target_bank['rows']])
        old_obs = prediction(model, theta, check, q_floor, counts)
        old_mean = feature_means(old_obs, check_maps, fit_weights[check['rows']])
        old_risk = risks(old_mean, check_target, check_maps)
        arrays.update(check_parents=check['rows'][::draws].cpu().numpy(),
            check_target_parents=check_target_bank['rows'][::draws].cpu().numpy(),
            check_seed=np.array(check['seed']), check_target_seed=np.array(check_target_bank['seed']),
            old_CHECK_residual=residual_vector(old_mean, check_target, check_maps).cpu().numpy())
        attempts, accepted = [], False
        for fraction in p['solver']['backtracking_fractions']:
            counts['joint_candidate_trials'] += 1
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
            candidate_obs = prediction(model, proposal, check, q_floor, counts)
            candidate_mean = feature_means(candidate_obs, check_maps, fit_weights[check['rows']])
            candidate_risk = risks(candidate_mean, check_target, check_maps)
            trust = output_trust(old_obs, candidate_obs, model, check['rows'], p)
            accepted = accept_joint(old_risk, candidate_risk, trust, len(check['rows']), p)
            index = len(attempts)
            arrays[f'trial{index}_proposal'] = proposal.cpu().numpy()
            arrays[f'trial{index}_CHECK_residual'] = residual_vector(candidate_mean, check_target, check_maps).cpu().numpy()
            attempts.append(dict(fraction=fraction, accepted=accepted, risks=candidate_risk, trust=trust))
            np.savez(out / f'{name}_ITER{iteration:02d}.npz', **arrays)
            if accepted:
                theta = proposal
                counts['teacher_accepted_joint_updates'] += 1
                break
        arrays['endpoint'] = theta.cpu().numpy().copy()
        np.savez(out / f'{name}_ITER{iteration:02d}.npz', **arrays)
        coverage.index_add_(0, fit_parents, torch.ones_like(fit_parents))
        target_coverage.index_add_(0, target_parents, torch.ones_like(target_parents))
        trace.append(dict(iteration=iteration, old_CHECK_risks=old_risk, attempts=attempts,
                          accepted=accepted, geometry=geometry, parameter_norm=float(theta.norm())))
        write_json(out / (name + '_TRACE.json'), dict(track=name, iterations=trace,
            response_population=fit_population, target_population=target_population,
            pairing_used_for_fit=False, conditional_teacher_labels_used_for_fit=False))
        np.savez(out / (name + '_STATE.npz'), theta=theta.cpu().numpy(),
                 source_coverage=coverage.cpu().numpy(), target_coverage=target_coverage.cpu().numpy())
        progress_file(out, counts, 'teacher_training', track=name, iteration=iteration + 1,
                      iterations=iterations, accepted_updates=sum(row['accepted'] for row in trace))
    return theta, dict(track=name, iterations=iterations, accepted_updates=sum(row['accepted'] for row in trace),
        source_parents_visited=int((coverage > 0).sum()), target_parents_visited=int((target_coverage > 0).sum()),
        allowed_parent_count=len(pool), fit_population=fit_population, target_population=target_population,
        source_ESS=float(fit_weights[pool].sum().square() / fit_weights[pool].square().sum()),
        conditional_pairing_for_fit=False)


def precision_check(p, teacher, truth_theta, student, endpoint, heldout, out, counts, started, seed_index):
    device = heldout.device
    projection = torch.cos(torch.arange(384, device=device, dtype=torch.float64)[:, None]
        * torch.arange(1, 17, device=device)[None] * torch.pi / 384.) / 384. ** .5
    moments, reports, gates = {}, {}, []
    draws, success = p['precision_draws_per_parent'], p['success']
    for library in range(2):
        generator = torch.Generator().manual_seed(p['precision_bank_seeds'][seed_index][library])
        counts['allocated_banks'] += 1
        samples = {label: [] for label in ('teacher', 'M0', 'candidate')}
        for position, parent in enumerate(heldout):
            check_budget(p, counts, started)
            rows = torch.full((draws,), int(parent), device=device, dtype=torch.long)
            z = torch.randn(draws, 8, generator=generator, dtype=torch.float32).to(device)
            counts['shape_standard_normal_values'] += z.numel()
            for label, model, theta in (('teacher', teacher, truth_theta), ('M0', student, student.initial()),
                    ('candidate', student, endpoint)):
                counts['joint_response_calls'] += 1
                counts['joint_response_rows'] += draws
                signal = model(theta, rows, z)
                reduced = signal @ projection
                centered = reduced - reduced.mean(0)
                covariance = centered.T @ centered / (draws - 1)
                samples[label].append((signal.mean(0).cpu().numpy(), covariance.cpu().numpy(),
                    float(canonical_charge008(signal).var()), float(signal.var(0).mean())))
            if (position + 1) % 32 == 0:
                progress_file(out, counts, 'C1_teacher_precision', seed=seed_index,
                    library=library, parents_complete=position + 1)
        arrays = {}
        for label, entries in samples.items():
            packed = tuple(np.stack([entry[i] for entry in entries]) for i in range(4))
            moments[library, label] = packed
            for name, values in zip(('mean', 'covariance', 'Q_variance', 'wave_variance'), packed, strict=True):
                arrays[label + '_' + name] = values
        np.savez(out / f'seed{seed_index}_library{library}_PRECISION.npz', heldout_rows=heldout.cpu().numpy(), **arrays)
        baseline = compare_moments(moments[library, 'M0'], moments[library, 'teacher'])
        fitted = compare_moments(moments[library, 'candidate'], moments[library, 'teacher'])
        key = f'seed{seed_index}/library{library}'
        reports[key] = dict(M0=baseline, candidate=fitted)
        for metric, absolute, fraction in (
                ('conditional_mean_relative_RMS', success['mean_absolute_relative_RMS'], success['mean_error_fraction']),
                ('conditional_covariance_relative_Frobenius', success['covariance_absolute_relative_error'], success['covariance_error_fraction'])):
            limit = min(absolute, fraction * baseline[metric])
            gates.append(dict(name=key + '/' + metric, value=fitted[metric], upper=limit, pass_gate=fitted[metric] <= limit))
        ratio = fitted['Q_variance_ratio']
        gates.append(dict(name=key + '/Q_variance_ratio', value=ratio, lower=success['variance_ratio_low'],
            upper=success['variance_ratio_high'], pass_gate=success['variance_ratio_low'] <= ratio <= success['variance_ratio_high']))
    for label in ('teacher', 'M0', 'candidate'):
        a, b = moments[0, label][1], moments[1, label][1]
        value = float(np.linalg.norm(a - b) / (.5 * (np.linalg.norm(a) + np.linalg.norm(b))))
        gates.append(dict(name=f'seed{seed_index}/{label}/independent_bank_MC_difference', value=value,
            upper=success['MC_covariance_difference_max'], pass_gate=value <= success['MC_covariance_difference_max']))
    return dict(reports=reports, gates=gates, pass_all=all(item['pass_gate'] for item in gates))




def run_teacher_c1(p, setup, out, counts, started):
    device, N = setup['source'].device, len(setup['source'])
    order = torch.randperm(N, generator=torch.Generator().manual_seed(p['role_seed'])).to(device)
    heldout, pool = order[:p['final']['parents']], order[p['final']['parents']:]
    np.savez(out / 'TEACHER_ROLES.npz', train_rows=pool.cpu().numpy(), heldout_rows=heldout.cpu().numpy())
    tracks, evaluations, decisions, saved, fixed_population, precision_runs = {}, {}, {}, {}, {}, {}
    students = [JointResponse01C1(model.cache) for model in setup['joint_models']]
    setup['physical_model'] = JointResponse01C1(setup['physical_model'].cache, physical=True)
    for seed_index, teacher in enumerate(setup['joint_models']):
        check_budget(p, counts, started)
        truth = teacher_parameters(teacher)
        student = students[seed_index]
        stats_bank = random_bank(setup, pool, 1, 1940000 + 10000 * seed_index, counts)
        stats_obs = prediction(teacher, truth, stats_bank, float(setup['loss'].q_floor), counts)
        blocks = {h: observable_blocks(stats_obs[h]) for h in HYPOTHESES}
        maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FIT']) for h in HYPOTHESES}
        check_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['CHECK']) for h in HYPOTHESES}
        final_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FINAL']) for h in HYPOTHESES}
        counts['feature_maps_constructed'] += 9
        np.savez(out / f'seed{seed_index}_FEATURE_MAPS.npz', **{
            role + '_' + h + '_cholesky': mapping[h].cholesky.cpu().numpy()
            for role, mapping in (('FIT', maps), ('CHECK', check_maps), ('FINAL', final_maps)) for h in HYPOTHESES})
        write_json(out / f'seed{seed_index}_FEATURE_SLICES.json', {
            block: [take.start, take.stop] for block, take in maps[HYPOTHESES[0]].block_slices.items()})
        fixed_population[str(seed_index)] = {}
        nominal = feature_means(stats_obs, maps)
        for scenario in ('nominal', 'low_h', 'high_h'):
            weights = population_weights(setup, scenario)[stats_bank['rows']]
            means = feature_means(stats_obs, maps, weights)
            fixed_population[str(seed_index)][scenario] = dict(
                ESS=float(weights.sum().square() / weights.square().sum()),
                weight_min=float(weights.min()), weight_max=float(weights.max()),
                mean_observed_Q={h: float((stats_obs[h]['Q'] * weights).sum() / weights.sum()) for h in HYPOTHESES},
                distance_from_nominal_same_fixed_response=risks(means, nominal, maps))
        candidates = {'M0': (student, student.initial()), 'physical_initial': (setup['physical_model'], setup['physical_model'].initial())}
        for name, learner in (('joint', student), ('physical', setup['physical_model'])):
            label = f'seed{seed_index}_{name}'
            endpoint, summary = fit_track_c1(p, setup, teacher, truth, learner, learner.initial(), pool, maps, check_maps,
                'nominal', 'nominal', p['iterations'], 1950000 + seed_index * 100000,
                out, counts, started, label)
            candidates[name] = (learner, endpoint)
            tracks[label] = summary
        evaluations[str(seed_index)] = evaluate_seed(p, setup, seed_index, teacher, truth, candidates, heldout,
                                                    final_maps, out, counts)
        diagnostic64 = recovery_decision(p, evaluations[str(seed_index)])
        hard_gates = [gate for gate in diagnostic64['gates'] if len(gate['name'].split('/')) == 3]
        if len(hard_gates) != 18:
            raise ValueError('C1 teacher hard-observation gate set differs')
        precise = precision_check(p, teacher, truth, student, candidates['joint'][1], heldout,
            out, counts, started, seed_index)
        precision_runs[str(seed_index)] = precise
        decisions[str(seed_index)] = dict(pass_all=all(item['pass_gate'] for item in hard_gates) and precise['pass_all'],
            hard_observation_gates=hard_gates, conditional_precision_gates=precise['gates'],
            diagnostic_64draw_decision=diagnostic64, conditional_acceptance_uses_prespecified_2048_draws=True)
        write_json(out / f'seed{seed_index}_PRECISION.json', precise)
        write_json(out / f'seed{seed_index}_EVALUATION.json', evaluations[str(seed_index)])
        write_json(out / f'seed{seed_index}_DECISION.json', decisions[str(seed_index)])
        saved[seed_index] = (teacher, truth, student, maps, check_maps, final_maps)
    # Deliberately warm-start at the known response. With a changed population,
    # compare correct fixed profile with a wrong nominal assumption; no free weights.
    original_teacher, truth, teacher, maps, check_maps, final_maps = saved[0]
    stress = {}
    for profile in ('nominal', 'high_h'):
        label = 'population_high_h_fit_' + profile
        endpoint, summary = fit_track_c1(p, setup, teacher, truth, teacher, truth, pool, maps, check_maps,
            profile, 'high_h', p['population_stress_iterations'], 2150000,
            out, counts, started, label)
        bank = random_bank(setup, heldout, p['final']['draws_per_parent'], 2160000, counts)
        counts['joint_response_calls'] += 1
        counts['joint_response_rows'] += len(bank['rows'])
        truth_signal = teacher(truth, bank['rows'], bank['z'])
        summary['conditional_drift_from_known_response'] = conditional_report(teacher, endpoint, bank, truth_signal, counts)
        summary['parameter_shift_norm'] = float((endpoint - truth).norm())
        summary['known_response_warm_start_not_recovery_evidence'] = True
        stress[profile] = summary
    return dict(status='COMPLETE', coupled_teacher_pass_all=all(d['pass_all'] for d in decisions.values()),
        tracks=tracks, recovery_decisions=decisions, population_stress=stress,
        fixed_response_population_scenarios=fixed_population, precision_runs=precision_runs,
        recovery_target='unchanged original joint01 coupled teachers', recovery_learners='C1 neural and C1 physical',
        population_stress_target_and_learner='same known C1 response, fixed high_h target population',
        all_scientific_teacher_gates_preregistered=True,
        teacher_training_parent_count=len(pool), teacher_holdout_parent_count=len(heldout),
        validation_scope='post-hoc heldout parents of frozen A0/M0; not full-pipeline unseen validation',
        true_teacher_parameter_labels_supplied_to_recovery_solver=False,
        three_noise_hypotheses_retained=True, population_parameters_fitted=0,
        REAL_response_updates=0, DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_REAL_joint_calibration' if all(d['pass_all'] for d in decisions.values()) else
                    'diagnose_teacher_failure_before_REAL_fit; do not enlarge network or open_REAL_calibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered C1 coupled-teacher config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned C1 teacher input changed: ' + name)
    review = json.loads((ROOT / p['engineering_review']).read_text())
    if not review['engineering_gate_pass'] or review['status'] != 'PASS_INDEPENDENT_C1_ENGINEERING_AUDIT':
        raise ValueError('C1 engineering prerequisite was not independently confirmed')
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        for key in ('encoding', 'references', 'models', 'baseline', 'heads', 'real'):
            del setup[key]
        progress_file(args.out, counts, 'C1_teacher_setup_loaded')
        result = run_teacher_c1(p, setup, args.out, counts, started)
    check_budget(p, counts, started)
    result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
        elapsed_seconds=time.monotonic() - started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.out / 'RESULT.json', result)
    if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()) > p['resources']['output_MiB'] * 1024**2:
        raise RuntimeError('C1 teacher output budget exceeded')
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE',
        result_sha256=hashlib.sha256((args.out / 'RESULT.json').read_bytes()).hexdigest(),
        coupled_teacher_pass_all=result['coupled_teacher_pass_all']))


