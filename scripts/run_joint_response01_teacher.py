"""Coupled unpaired teacher recovery, physical control, and population stress.

Teacher labels/conditional pairing are confined to final diagnostics. The two
population stress tracks explicitly start at the known response to isolate
compensation for a wrong population; they are not recovery demonstrations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from run_joint_response01 import (load_joint_setup, progress_file, random_bank,
    prediction, feature_means, residual_vector, risks, finite_jacobian,
    teacher_parameters, conditional_report, HYPOTHESES)
from run_response036 import hard_metrics

ROOT = Path(__file__).resolve().parents[1]


class TeacherBudget01(Exception):
    """Registered scientific operation or runtime boundary, never auto-restarted."""


def check_budget(p, counts, started):
    if time.monotonic() - started > p['resources']['internal_seconds']:
        raise TeacherBudget01('Teacher internal wall bound reached')
    for name, limit in p['operation_budget'].items():
        if counts[name] > limit:
            raise TeacherBudget01(f'{name}: {counts[name]} > {limit}')


def draw_parents(pool, count, seed):
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))[:count].to(pool.device)
    return pool[order]


def population_weights(setup, scenario):
    h = torch.as_tensor(setup['data']['h'], device=setup['source'].device, dtype=torch.float64)
    if scenario == 'nominal':
        return torch.ones_like(h)
    if scenario == 'high_h':
        return 1. + .20 * (2. * h - 1.)
    if scenario == 'low_h':
        return 1. - .20 * (2. * h - 1.)
    raise ValueError('Unregistered population scenario')


def output_trust(old, new, model, rows, p):
    scale = model.cache['A'][rows].abs().amax(1)
    relative = torch.cat([((new[h]['carrier_input'] - old[h]['carrier_input']).square().mean(1).sqrt() / scale)
                          for h in HYPOTHESES])
    changes = {}
    for h in HYPOTHESES:
        changes[h] = {name: int(new[h]['flags'][name].sum() - old[h]['flags'][name].sum())
                      for name in old[h]['flags']}
    maximum_flags = max(value for h in changes.values() for value in h.values())
    result = dict(mean_relative_RMS=float(relative.mean()), p99_relative_RMS=float(torch.quantile(relative, .99)),
                  max_relative_RMS=float(relative.max()), flag_count_changes=changes, denominator=len(rows))
    result['pass'] = (result['mean_relative_RMS'] <= p['solver']['output_TR_mean_RMS_fraction']
                      and result['p99_relative_RMS'] <= p['solver']['output_TR_p99_RMS_fraction']
                      and maximum_flags <= max(4, p['solver']['flag_increase_fraction'] * len(rows)))
    return result


def accept_joint(old_risk, new_risk, trust, count, p):
    old = np.mean([old_risk[h]['joint_whitened'] for h in HYPOTHESES])
    new = np.mean([new_risk[h]['joint_whitened'] for h in HYPOTHESES])
    # Mean-feature Monte Carlo scale, not a manufactured experimental tolerance.
    floor = 2. / count
    block_pass = all(new_risk[h][name] <= p['solver']['macro_degradation_factor'] * old_risk[h][name] + floor
                     for h in HYPOTHESES for name in ('wave', 'charge_morphology', 'peak_codes', 'baseline'))
    return bool(trust['pass'] and new <= (1. - p['solver']['minimum_CHECK_gain']) * old and block_pass)


def train_track(p, setup, teacher, teacher_theta, model, initial, pool, maps, check_maps,
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
        counts['joint_direction_solves'] += 1
        delta, geometry = joint_step(residual, J, ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'])
        check = random_bank(setup, draw_parents(pool, batch, seed + 3000 + iteration), draws,
                            seed + 4000 + 20 * iteration, counts)
        check_target_bank = random_bank(setup, draw_parents(pool, batch, seed + 5000 + iteration), draws,
                                        seed + 6000 + 20 * iteration, counts)
        check_target = feature_means(prediction(teacher, teacher_theta, check_target_bank, q_floor, counts), check_maps,
                                    target_weights[check_target_bank['rows']])
        old_obs = prediction(model, theta, check, q_floor, counts)
        old_mean = feature_means(old_obs, check_maps, fit_weights[check['rows']])
        old_risk = risks(old_mean, check_target, check_maps)
        attempts, accepted = [], False
        for fraction in p['solver']['backtracking_fractions']:
            counts['joint_candidate_trials'] += 1
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
            candidate_obs = prediction(model, proposal, check, q_floor, counts)
            candidate_mean = feature_means(candidate_obs, check_maps, fit_weights[check['rows']])
            candidate_risk = risks(candidate_mean, check_target, check_maps)
            trust = output_trust(old_obs, candidate_obs, model, check['rows'], p)
            accepted = accept_joint(old_risk, candidate_risk, trust, len(check['rows']), p)
            attempts.append(dict(fraction=fraction, accepted=accepted, risks=candidate_risk, trust=trust))
            if accepted:
                theta = proposal
                counts['teacher_accepted_joint_updates'] += 1
                break
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


def evaluate_seed(p, setup, seed_index, teacher, truth_theta, candidates, heldout, maps, out, counts):
    q_floor = float(setup['loss'].q_floor)
    result = {}
    for library in range(2):
        bank = random_bank(setup, heldout, p['final']['draws_per_parent'], 1920000 + seed_index * 10000 + library * 100, counts)
        target_bank = random_bank(setup, heldout, p['final']['draws_per_parent'], 1930000 + seed_index * 10000 + library * 100, counts)
        target_obs = prediction(teacher, truth_theta, target_bank, q_floor, counts)
        target = feature_means(target_obs, maps)
        counts['joint_response_calls'] += 1
        counts['joint_response_rows'] += len(bank['rows'])
        truth_signal = teacher(truth_theta, bank['rows'], bank['z'])
        comparisons = dict(candidates)
        comparisons['teacher_oracle'] = (teacher, truth_theta)
        result[str(library)] = {}
        for name, (model, theta) in comparisons.items():
            obs = prediction(model, theta, bank, q_floor, counts)
            means = feature_means(obs, maps)
            conditional = conditional_report(model, theta, bank, truth_signal, counts)
            native = {}
            # The original metrics remain separately defined and reported.
            for h in HYPOTHESES:
                counts['native_metric_rows'] += 2 * len(bank['rows'])
                native[h] = hard_metrics(obs[h]['window_V'].cpu().numpy(), obs[h]['baseline100'].cpu().numpy(),
                    target_obs[h]['window_V'].cpu().numpy(), target_obs[h]['baseline100'].cpu().numpy(),
                    setup['loss'], setup['context'])
            result[str(library)][name] = dict(risks=risks(means, target, maps), conditional=conditional,
                native=native, observation_flags={h: {key: dict(count=int(value.sum()), denominator=len(value))
                    for key, value in obs[h]['flags'].items()} for h in HYPOTHESES})
        progress_file(out, counts, 'teacher_new_library_evaluation', seed=seed_index, library=library)
    return result


def recovery_decision(p, evaluation):
    gates = []
    for library, results in evaluation.items():
        base, candidate, truth = (results[k] for k in ('M0', 'joint', 'teacher_oracle'))
        b, c = base['conditional'], candidate['conditional']
        for h in HYPOTHESES:
            br, cr, tr = (x['risks'][h]['joint_whitened'] for x in (base, candidate, truth))
            gates.append(dict(name=f'library{library}/{h}/joint_risk', pass_gate=cr <= tr +
                p['success']['remaining_excess_risk_fraction'] * max(br - tr, 0.), baseline=br, candidate=cr, oracle=tr))
            for metric in ('wave', 'QPSD'):
                bn, cn, tn = (x['native'][h]['values'][metric] for x in (base, candidate, truth))
                gates.append(dict(name=f'library{library}/{h}/native_{metric}',
                    pass_gate=cn <= max(1.02 * bn, 1.20 * tn), baseline=bn, candidate=cn, oracle=tn))
        gates += [dict(name=f'library{library}/conditional_mean', pass_gate=c['conditional_mean_relative_RMS'] <=
                       min(p['success']['mean_absolute_relative_RMS'], p['success']['mean_error_fraction'] * b['conditional_mean_relative_RMS']),
                       baseline=b['conditional_mean_relative_RMS'], candidate=c['conditional_mean_relative_RMS']),
                  dict(name=f'library{library}/conditional_covariance', pass_gate=c['conditional_covariance_relative_Frobenius'] <=
                       min(p['success']['covariance_absolute_relative_error'], p['success']['covariance_error_fraction'] * b['conditional_covariance_relative_Frobenius']),
                       baseline=b['conditional_covariance_relative_Frobenius'], candidate=c['conditional_covariance_relative_Frobenius']),
                  dict(name=f'library{library}/Q_variance', pass_gate=p['success']['variance_ratio_low'] <=
                       c['conditional_Q_variance_mean'] / c['teacher_Q_variance_mean'] <= p['success']['variance_ratio_high'],
                       ratio=c['conditional_Q_variance_mean'] / c['teacher_Q_variance_mean'])]
    return dict(pass_all=all(x['pass_gate'] for x in gates), gates=gates,
                physical_comparator_not_required_to_recover_neural_teacher_exactly=True)


def run_teacher(p, setup, out, counts, started):
    device, N = setup['source'].device, len(setup['source'])
    order = torch.randperm(N, generator=torch.Generator().manual_seed(p['role_seed'])).to(device)
    heldout, pool = order[:p['final']['parents']], order[p['final']['parents']:]
    np.savez(out / 'TEACHER_ROLES.npz', train_rows=pool.cpu().numpy(), heldout_rows=heldout.cpu().numpy())
    tracks, evaluations, decisions, saved, fixed_population = {}, {}, {}, {}, {}
    for seed_index, teacher in enumerate(setup['joint_models']):
        check_budget(p, counts, started)
        truth = teacher_parameters(teacher)
        stats_bank = random_bank(setup, pool, 1, 1940000 + 10000 * seed_index, counts)
        stats_obs = prediction(teacher, truth, stats_bank, float(setup['loss'].q_floor), counts)
        blocks = {h: observable_blocks(stats_obs[h]) for h in HYPOTHESES}
        maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FIT']) for h in HYPOTHESES}
        check_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['CHECK']) for h in HYPOTHESES}
        final_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FINAL']) for h in HYPOTHESES}
        counts['feature_maps_constructed'] += 9
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
        candidates = {'M0': (teacher, teacher.initial()), 'physical_initial': (setup['physical_model'], setup['physical_model'].initial())}
        for name, learner in (('joint', teacher), ('physical', setup['physical_model'])):
            label = f'seed{seed_index}_{name}'
            endpoint, summary = train_track(p, setup, teacher, truth, learner, learner.initial(), pool, maps, check_maps,
                'nominal', 'nominal', p['iterations'], 1950000 + seed_index * 100000,
                out, counts, started, label)
            candidates[name] = (learner, endpoint)
            tracks[label] = summary
        evaluations[str(seed_index)] = evaluate_seed(p, setup, seed_index, teacher, truth, candidates, heldout,
                                                    final_maps, out, counts)
        decisions[str(seed_index)] = recovery_decision(p, evaluations[str(seed_index)])
        write_json(out / f'seed{seed_index}_EVALUATION.json', evaluations[str(seed_index)])
        write_json(out / f'seed{seed_index}_DECISION.json', decisions[str(seed_index)])
        saved[seed_index] = (teacher, truth, maps, check_maps, final_maps)
    # Deliberately warm-start at the known response. With a changed population,
    # compare correct fixed profile with a wrong nominal assumption; no free weights.
    teacher, truth, maps, check_maps, final_maps = saved[0]
    stress = {}
    for profile in ('nominal', 'high_h'):
        label = 'population_high_h_fit_' + profile
        endpoint, summary = train_track(p, setup, teacher, truth, teacher, truth, pool, maps, check_maps,
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
        fixed_response_population_scenarios=fixed_population,
        teacher_training_parent_count=len(pool), teacher_holdout_parent_count=len(heldout),
        validation_scope='post-hoc heldout parents of frozen A0/M0; not full-pipeline unseen validation',
        true_teacher_parameter_labels_supplied_to_recovery_solver=False,
        three_noise_hypotheses_retained=True, population_parameters_fitted=0,
        REAL_response_updates=0, DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_REAL_joint_calibration' if all(d['pass_all'] for d in decisions.values()) else
                    'diagnose_teacher_failure_before_REAL_fit; do not enlarge network or open_REAL_calibration')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered teacher configuration changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned teacher source changed: ' + name)
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
            # Full-waveform A0/B caches now contain everything this solver uses.
            # Keep the engineering runner's direct-M0 identity path separate.
            for key in ('encoding', 'references', 'models', 'baseline', 'heads'):
                del setup[key]
            progress_file(args.out, counts, 'teacher_setup_loaded')
            result = run_teacher(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
                      elapsed_seconds=time.monotonic() - started,
                      peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(), coupled_teacher_pass_all=result['coupled_teacher_pass_all']))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                automatic_restart=False, scientific_acceptance=False))


if __name__ == '__main__':
    main()
