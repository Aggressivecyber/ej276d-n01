"""Registered REAL joint calibration and equal-budget physical comparison."""
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
from n01.observation008 import observe_existing_carrier
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, finite_jacobian, HYPOTHESES)
from run_joint_response01_teacher import check_budget, draw_parents, output_trust, accept_joint
from run_response036 import hard_metrics

ROOT = Path(__file__).resolve().parents[1]
REAL_KEYS = ('window_V', 'baseline100', 'Q', 'tail', 'PSD', 'carrier_adc', 'peak_index')


def select_real(observations, rows):
    return dict(**{key: observations[key][rows] for key in REAL_KEYS},
                flags={key: value[rows] for key, value in observations['flags'].items()})


def fit_track(p, setup, model, source_pool, real_pool, maps, check_maps, real, seed, out, counts, started, label):
    theta = model.initial()
    batch, draws, real_batch = (p['batch'][key] for key in ('source_parents', 'draws_per_parent', 'target_rows'))
    source_coverage = torch.zeros(len(setup['source']), dtype=torch.long, device=source_pool.device)
    real_coverage = torch.zeros(len(real['Q']), dtype=torch.long, device=source_pool.device)
    order = {}
    for name, pool, count, offset in (('source', source_pool, batch, 100), ('real', real_pool, real_batch, 200)):
        epochs = (p['iterations'] * count + len(pool) - 1) // len(pool)
        order[name] = torch.cat([draw_parents(pool, len(pool), seed + offset + epoch) for epoch in range(epochs)])
    trace = []
    for iteration in range(p['iterations']):
        check_budget(p, counts, started)
        parents = order['source'][iteration * batch:(iteration + 1) * batch]
        target_rows = order['real'][iteration * real_batch:(iteration + 1) * real_batch]
        target_obs = select_real(real, target_rows)
        counts['REAL_fit_feature_rows'] += len(target_rows)
        target = feature_means({h: target_obs for h in HYPOTHESES}, maps)
        fit = random_bank(setup, parents, draws, seed + 1000 + 20 * iteration, counts)
        current = feature_means(prediction(model, theta, fit, setup['loss'].q_floor, counts), maps)
        residual = residual_vector(current, target, maps)
        J = finite_jacobian(model, theta, fit, setup['loss'].q_floor, maps, target, counts,
                            p['solver']['finite_difference_step'])
        counts['joint_direction_solves'] += 1
        delta, geometry = joint_step(residual, J, ridge_fraction=p['solver']['ridge_fraction'], radius=p['solver']['radius'])
        check = random_bank(setup, draw_parents(source_pool, batch, seed + 3000 + iteration), draws,
                            seed + 4000 + 20 * iteration, counts)
        check_rows = draw_parents(real_pool, real_batch, seed + 5000 + iteration)
        check_target_obs = select_real(real, check_rows)
        counts['REAL_CHECK_feature_rows'] += len(check_rows)
        check_target = feature_means({h: check_target_obs for h in HYPOTHESES}, check_maps)
        old_obs = prediction(model, theta, check, setup['loss'].q_floor, counts)
        old_risk = risks(feature_means(old_obs, check_maps), check_target, check_maps)
        attempts, accepted = [], False
        for fraction in p['solver']['backtracking_fractions']:
            counts['joint_candidate_trials'] += 1
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + fraction * delta))
            candidate_obs = prediction(model, proposal, check, setup['loss'].q_floor, counts)
            new_risk = risks(feature_means(candidate_obs, check_maps), check_target, check_maps)
            trust = output_trust(old_obs, candidate_obs, model, check['rows'], p)
            accepted = accept_joint(old_risk, new_risk, trust, len(check['rows']), p)
            attempts.append(dict(fraction=fraction, accepted=accepted, risks=new_risk, trust=trust))
            if accepted:
                theta = proposal
                counts['REAL_response_updates'] += 1
                break
        source_coverage.index_add_(0, parents, torch.ones_like(parents))
        real_coverage.index_add_(0, target_rows, torch.ones_like(target_rows))
        trace.append(dict(iteration=iteration, old_CHECK_risks=old_risk, attempts=attempts, accepted=accepted,
                          geometry=geometry, parameter_norm=float(theta.norm())))
        write_json(out / (label + '_TRACE.json'), dict(track=label, iterations=trace,
            paired_supervision=False, population_weights_fitted=False))
        np.savez(out / (label + '_STATE.npz'), theta=theta.cpu().numpy(),
                 source_coverage=source_coverage.cpu().numpy(), real_coverage=real_coverage.cpu().numpy())
        progress_file(out, counts, 'REAL_joint_training', track=label, iteration=iteration + 1,
                      iterations=p['iterations'], accepted_updates=sum(row['accepted'] for row in trace))
    return theta, dict(track=label, accepted_updates=sum(row['accepted'] for row in trace),
        source_parents_visited=int((source_coverage > 0).sum()), allowed_source_parents=len(source_pool),
        real_records_visited=int((real_coverage > 0).sum()), allowed_real_records=len(real_pool),
        source_visit_sum=int(source_coverage.sum()), real_visit_sum=int(real_coverage.sum()))


def evaluate(p, setup, seed_index, candidates, heldout, real, real_rows, maps, out, counts):
    target_obs = select_real(real, real_rows)
    target = feature_means({h: target_obs for h in HYPOTHESES}, maps)
    counts['REAL_evaluation_feature_rows'] += len(real_rows)
    target_wave, target_baseline = (target_obs[key].cpu().numpy() for key in ('window_V', 'baseline100'))
    result = {}
    for library in range(p['final']['libraries']):
        bank = random_bank(setup, heldout, p['final']['draws_per_parent'],
                           p['evaluation_seed'] + seed_index * 10000 + library * 100, counts)
        reports = {}
        for name, (model, theta) in candidates.items():
            obs = prediction(model, theta, bank, setup['loss'].q_floor, counts)
            native = {}
            for h in HYPOTHESES:
                counts['native_metric_rows'] += len(bank['rows']) + len(real_rows)
                native[h] = hard_metrics(obs[h]['window_V'].cpu().numpy(), obs[h]['baseline100'].cpu().numpy(),
                    target_wave, target_baseline, setup['loss'], setup['context'])
            reports[name] = dict(risks=risks(feature_means(obs, maps), target, maps), native=native,
                flags={h: {flag: dict(count=int(value.sum()), denominator=len(value))
                    for flag, value in obs[h]['flags'].items()} for h in HYPOTHESES})
        result[str(library)] = reports
        progress_file(out, counts, 'REAL_new_library_holdout_evaluation', seed=seed_index, library=library)
    write_json(out / f'seed{seed_index}_EVALUATION.json', result)
    return result


def decide(p, evaluation):
    gates = []
    for library, reports in evaluation.items():
        baseline, candidate = reports['M0'], reports['joint']
        for h in HYPOTHESES:
            prefix = f'library{library}/{h}/'
            for metric, factor in (('joint_whitened', p['success']['joint_remaining_fraction']),
                                   ('peak_codes', p['success']['peak_remaining_fraction'])):
                b, c = baseline['risks'][h][metric], candidate['risks'][h][metric]
                gates.append(dict(name=prefix + metric, baseline=b, candidate=c, upper=factor * b, pass_gate=c <= factor * b))
            for metric in ('wave', 'QPSD', 'noise'):
                factor = p['success']['noise_max_factor'] if metric == 'noise' else p['success']['native_remaining_fraction']
                b, c = baseline['native'][h]['values'][metric], candidate['native'][h]['values'][metric]
                gates.append(dict(name=prefix + 'native_' + metric, baseline=b, candidate=c, upper=factor * b, pass_gate=c <= factor * b))
            for metric in ('wave', 'charge_morphology', 'peak_codes', 'baseline'):
                b, c = baseline['risks'][h][metric], candidate['risks'][h][metric]
                limit = p['success']['macro_max_factor'] * b + 2. / (p['final']['parents'] * p['final']['draws_per_parent'])
                gates.append(dict(name=prefix + 'macro_' + metric, baseline=b, candidate=c, upper=limit, pass_gate=c <= limit))
            for flag, entry in candidate['flags'][h].items():
                old = baseline['flags'][h][flag]
                limit = old['count'] + max(4, p['success']['flag_increase_fraction'] * entry['denominator'])
                gates.append(dict(name=prefix + 'flag_' + flag, baseline=old['count'], candidate=entry['count'],
                                  denominator=entry['denominator'], upper=limit, pass_gate=entry['count'] <= limit))
    return dict(pass_all=all(gate['pass_gate'] for gate in gates), gates=gates,
                final_scientific_acceptance=False, stage5_required=True)


def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        roles = {key: archive[key].copy() for key in archive.files}
    np.savez(out / 'REAL_CALIBRATION_ROLES.npz', **roles)
    source_pool, source_holdout, real_pool, real_holdout = [torch.as_tensor(roles[key], device=device) for key in
        ('source_train_rows', 'source_heldout_rows', 'real_train_rows', 'real_heldout_rows')]
    # Preserve the frozen loader's CPU readout of existing ADC codes. GPU
    # redigitization can differ in floating-point reconstruction of code values.
    cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
                                       q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
    for key in ('window_V', 'baseline100'):
        if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
            raise ValueError('REAL readout differs from frozen loader: ' + key)
    real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
                flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
    counts['REAL_observable_replay_calls'] += 1
    counts['REAL_observable_replay_rows'] += len(real['Q'])
    del setup['real']
    del cpu_real
    blocks = observable_blocks(select_real(real, real_pool))
    feature_maps = {}
    for label, seed in p['feature_seeds'].items():
        counts['feature_maps_constructed'] += 1
        fmap = JointFeatureMap01(blocks, seed=seed)
        feature_maps[label] = {h: fmap for h in HYPOTHESES}
    del blocks
    tracks, decisions = {}, {}
    for seed_index, neural in enumerate(setup['joint_models']):
        candidates = dict(M0=(neural, neural.initial()),
                          physical_initial=(setup['physical_model'], setup['physical_model'].initial()))
        for name, model in (('joint', neural), ('physical', setup['physical_model'])):
            label = f'seed{seed_index}_{name}'
            endpoint, summary = fit_track(p, setup, model, source_pool, real_pool, feature_maps['FIT'],
                feature_maps['CHECK'], real, p['training_seed'] + seed_index * 100000,
                out, counts, started, label)
            candidates[name] = (model, endpoint)
            tracks[label] = summary
        evaluation = evaluate(p, setup, seed_index, candidates, source_holdout, real, real_holdout,
                              feature_maps['FINAL'], out, counts)
        decisions[str(seed_index)] = decide(p, evaluation)
        write_json(out / f'seed{seed_index}_DECISION.json', decisions[str(seed_index)])
    return dict(status='COMPLETE', REAL_calibration_candidate_pass_all=all(v['pass_all'] for v in decisions.values()),
        decisions=decisions, tracks=tracks, role_counts={key: len(value) for key, value in roles.items()},
        feature_statistics='new joint maps fit only on research REAL train role; original A0/M0/005/PCA/native scales remain full-TRAIN frozen',
        validation_scope='Frozen-system posthoc SOURCE and REAL-group holdout; single acquisition, no full-pipeline unseen claim',
        final_all_TRAIN_refit_pending=True, stage5_validation_pending=True,
        population_weights_fitted=0, DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_stage5_response_validation' if all(v['pass_all'] for v in decisions.values()) else
                    'diagnose_saved_joint_REAL_failure_before_more_training')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered REAL calibration config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned REAL input changed: ' + name)
    prerequisite = json.loads((ROOT / p['teacher_precision_result']).read_text())
    if not prerequisite['stage4_REAL_calibration_eligible_under_precision_amendment']:
        raise ValueError('Teacher precision prerequisite did not pass')
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
            progress_file(args.out, counts, 'REAL_calibration_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        output_bytes = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
        if output_bytes > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered REAL output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest(), REAL_calibration_candidate_pass_all=result['REAL_calibration_candidate_pass_all']))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False,
                scientific_acceptance=False))


if __name__ == '__main__':
    main()
