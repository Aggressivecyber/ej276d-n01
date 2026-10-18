"""Bounded, hard-observation calibration of eight existing mean time biases.

All TRAIN parents and REAL rows participate. A0, M0 fluctuations, 005 and the
observation operator are reused unchanged. No AD, paired REAL labels or DEV.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import sha256, write_json
from n01.model008 import canonical_charge008
from n01.model015 import _ExactZeroMeanAddition015
from n01.model038 import MeanProposalOutsideDomain038
from n01.observation008 import observe_generated_carrier
from n01.response008 import HYPOTHESES, state_digest
from n01.stats018 import morphology018, scalar_features018, observable018
from run_response036 import column_W1, electronic_noise, hard_metrics
from run_response038 import load_setup
from wp1_evaluation import raw_distance

ROOT = Path(__file__).resolve().parents[1]
MORPH = ('peak_voltage_V', 'rise10_90_ns', 'FWHM_ns')
PROTECTED = ('QPSD', 'noise', 'Q_Vns', 'front_Vns', 'tail_Vns', 'PSD_guarded',
             'baseline_mean_V', 'baseline_RMS_V')


def seed039(seed, purpose, bank, stream):
    key = f'n01/response039/v1/{seed}/{purpose}/{bank}/{stream}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & (2**63-1)


def joint_embedding(wave, baseline, peaks, scales):
    """018 observable tuple + tail and guarded PSD; masks retain undefined rows."""
    scalar, morph = scalar_features018(wave, baseline, peaks)
    z = observable018(wave, scalar, morph, scales)['embedding']
    z[:, 9:15] /= scales['rise10_90_ns']
    q, tail = scalar[:, 0], scalar[:, 1]
    floor = scales['q_floor']
    denominator = np.where(q < 0, -1., 1.)*np.maximum(np.abs(q), floor)
    return np.column_stack((z, tail/scales['tail_Vns'],
                            (tail/denominator)/scales['PSD_guarded']))


def projection039(seed, device):
    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(21, 64, generator=generator, dtype=torch.float64)
    random /= torch.linalg.vector_norm(random, dim=0)
    return torch.cat((torch.eye(21, dtype=torch.float64), random), 1).to(device)


def objective039(report, reference, tolerances, contract):
    """Equal H/metric weights; constraints remain separate from the score."""
    ratios, violations = [], []
    for h in HYPOTHESES:
        a, b = report[h]['values'], reference[h]['values']
        for key in ('wave', 'joint_morphology'):
            ratio = a[key]/b[key]
            ratios.append(ratio)
            if ratio > 1.+contract['fit_primary_relative_worsening_max']:
                violations.append(f'{h}/{key}')
        for key in PROTECTED:
            if a[key]-b[key] > contract['protected_allowance_T']*tolerances[h][key]:
                violations.append(f'{h}/{key}')
        for key in MORPH:
            if a[key] is None or b[key] is None:
                violations.append(f'{h}/{key}/undefined_population')
            elif a[key]-b[key] > contract['morphology_individual_allowance_T']*tolerances[h][key]:
                violations.append(f'{h}/{key}')
        if a['negative_lobe_area_Vns'] > b['negative_lobe_area_Vns']:
            violations.append(f'{h}/negative_lobe_area')
        for family in ('fake_flags', 'observation_flags'):
            for name, row in report[h][family].items():
                if row['count'] > reference[h][family][name]['count']:
                    violations.append(f'{h}/{family}/{name}')
        if not report[h]['conditional_coverage']['conditional_coverage_complete']:
            violations.append(f'{h}/incomplete_output_Q_coverage')
    return dict(score=float(np.mean(ratios)), feasible=not violations, violations=violations)


def estimate039(values):
    x = np.asarray(values, np.float64)
    mean, se = float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean, MCSE=se, lower3MCSE=mean-3*se, upper3MCSE=mean+3*se,
                banks=len(x), uncertainty_scope='fresh noise banks; fixed TRAIN population and REAL sample')


def confirmation039(pairs, tolerances, contract):
    """Only unopened MC banks enter this stage decision; no endpoint reselection."""
    result, failures = {}, []
    for h in HYPOTHESES:
        metrics = {}
        for key in ('wave', 'joint_morphology'):
            metrics[key+'_relative_gain'] = estimate039([
                (b[h]['values'][key]-a[h]['values'][key])/b[h]['values'][key] for a, b in pairs])
        metrics['morphology_mean_T_gain'] = estimate039([np.mean([
            (b[h]['values'][k]-a[h]['values'][k])/tolerances[h][k] for k in MORPH]) for a, b in pairs])
        if metrics['wave_relative_gain']['lower3MCSE'] < contract['useful_wave_relative_gain']:
            failures.append(f'{h}/useful_wave_gain')
        if metrics['joint_morphology_relative_gain']['lower3MCSE'] < contract['useful_joint_relative_gain']:
            failures.append(f'{h}/useful_joint_gain')
        if metrics['morphology_mean_T_gain']['lower3MCSE'] < contract['useful_morphology_mean_T_gain']:
            failures.append(f'{h}/useful_morphology_T_gain')
        for key in PROTECTED+MORPH:
            metrics[key+'_delta_T'] = estimate039([
                (a[h]['values'][key]-b[h]['values'][key])/tolerances[h][key] for a, b in pairs])
            allowance = (contract['protected_allowance_T'] if key in PROTECTED
                         else contract['morphology_individual_allowance_T'])
            if metrics[key+'_delta_T']['upper3MCSE'] > allowance:
                failures.append(f'{h}/{key}/noninferiority')
        # Preserve all per-bank anomaly and negative-area checks, not their mean alone.
        for i, (a, b) in enumerate(pairs):
            checks = objective039({h0: a[h0] for h0 in HYPOTHESES},
                                  {h0: b[h0] for h0 in HYPOTHESES}, tolerances, contract)
            failures.extend(f'bank{i}/{v}' for v in checks['violations'] if v.startswith(h+'/'))
        result[h] = metrics
    return dict(useful_mean_correction=not failures, failures=failures, by_hypothesis=result,
                simultaneous_confidence_claim=False, physical_hardware_calibrated=False,
                full_random_response_scientific_acceptance=False)


@torch.no_grad()
def make_population_bank(setup, index, purpose, bank_id, p, progress):
    """One draw for every SOURCE parent; explicit independent MC bank namespaces."""
    model, encoding = setup['models'][index], setup['encoding']
    model.mean_adapter[-1].bias.zero_()
    device, count = setup['source'].device, len(setup['source'])
    generators = {key: torch.Generator().manual_seed(seed039(p['seeds'][index], purpose, bank_id, key))
                  for key in ('shape', 'noise005', 'extension')}
    parts, noise = [], {h: [] for h in HYPOTHESES}
    for start in range(0, count, p['chunk_rows']):
        ids = torch.arange(start, min(start+p['chunk_rows'], count), device=device)
        cache = model.select_encoding(encoding, ids)
        z = torch.randn(len(ids), 8, generator=generators['shape']).to(device)
        progress['response_calls'] += 1
        progress['response_rows'] += len(ids)
        output = model(cache['source'], z, cache=cache, charge_enabled=False)
        parts.append(output['signal_pre'])
        progress['noise005_calls'] += 1
        block = electronic_noise(setup['extension'], len(ids), generators['noise005'],
                                 generators['extension'], device, progress)
        for h in HYPOTHESES:
            noise[h].append(block[h])
    progress['full_population_banks'] += 1
    return dict(M0=torch.cat(parts), noise={h: torch.cat(noise[h]) for h in HYPOTHESES},
                purpose=purpose, bank_id=bank_id, seed=p['seeds'][index])


@torch.no_grad()
def evaluate039(setup, index, bias, bank, tolerances, projections, real_joint,
                p, out, progress, *, examples=None):
    model = setup['models'][index]
    model.mean_adapter[-1].bias.copy_(torch.as_tensor(bias, device=setup['source'].device,
                                                     dtype=model.mean_adapter[-1].bias.dtype))
    progress['mean_operator_calls'] += 1
    progress['mean_operator_rows'] += len(setup['source'])
    adapter = model._mean_response(setup['encoding'], enabled=True)
    signal = _ExactZeroMeanAddition015.apply(bank['M0'], adapter['delta_mean_pre_V'])
    geometry = dict(max_time_displacement_ns=float((adapter['time_queries']-
                       torch.arange(384, device=signal.device)).abs().max()),
                    minimum_warp_Q_ratio=float(adapter['warp_Q_ratio'].min()),
                    maximum_normalization_gain=float(adapter['normalization_gain'].max()),
                    max_canonical_Q_residual_Vns=float((canonical_charge008(adapter['mean_signal_pre_V'])-
                       canonical_charge008(setup['encoding']['original']['signal_pre'])).abs().max()),
                    prefix_bitwise_unchanged=torch.equal(signal[:, :100], bank['M0'][:, :100]))
    reports = {}
    for h in HYPOTHESES:
        obs = observe_generated_carrier(signal+bank['noise'][h], ste=False,
                   q_floor=float(setup['loss'].q_floor), return_indices=True)
        progress['generated_observation_rows'] += len(signal)
        wave, baseline, peaks = [obs[k].cpu().numpy() for k in ('window_V', 'baseline100', 'peak_index')]
        report = hard_metrics(wave, baseline, setup['data']['real'], setup['data']['real_baseline100'],
                              setup['loss'], setup['context'])
        z = joint_embedding(wave, baseline, peaks, dict(tolerances[h], q_floor=float(setup['loss'].q_floor)))
        projected = torch.as_tensor(z, device=signal.device)@projections
        report['values']['joint_morphology'] = float(column_W1(projected, real_joint[h], 16))
        report['values']['negative_lobe_area_Vns'] = raw_distance(np.maximum(-wave, 0).sum(1),
                                           np.maximum(-setup['data']['real'], 0).sum(1))['value']
        report['observation_flags'] = {k: dict(count=int(v.sum()), denominator=len(signal))
                                       for k, v in obs['flags'].items()}
        progress['full_native_hypothesis_evaluations'] += 1
        reports[h] = report
        if examples is not None:
            np.savez_compressed(out/f'{examples}_{h}.npz', wave=wave[:64], baseline=baseline[:64],
                                parent_row=np.arange(min(64, len(signal))), peak_index=peaks[:64])
    return reports, geometry


def compact039(reports):
    return {h: {k: r[k] for k in ('values', 'fake_flags', 'observation_flags', 'conditional_coverage')}
            for h, r in reports.items()}


def check_resources(p, out, started):
    if time.monotonic()-started >= p['resources']['internal_seconds']:
        raise TimeoutError('039 internal wall limit; no automatic restart')
    if sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) > p['resources']['output_MiB']*1024**2:
        raise OSError('039 output cap exceeded')


@torch.no_grad()
def run039(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), run_id=p['run_id'],
               config_sha256=p['_config_sha256'], pid=os.getpid(), resources=p['resources'], phase='setup'))
    setup_config = json.loads((ROOT/p['setup038']['path']).read_text(encoding='utf8'))
    setup = load_setup(setup_config, device, progress)
    for model in setup['models']:
        model.set_phase('frozen').eval()
    tolerances = json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    real_wave, real_baseline = setup['data']['real'], setup['data']['real_baseline100']
    fit_projection = projection039(p['joint_projection_fit_seed'], device)
    real_embedding = {h: torch.as_tensor(joint_embedding(real_wave, real_baseline, np.full(len(real_wave), 16),
                         dict(tolerances[h], q_floor=float(setup['loss'].q_floor))), device=device) for h in HYPOTHESES}
    fit_real = {h: value@fit_projection for h, value in real_embedding.items()}
    endpoints, fits = [], []
    for index in range(2):
        check_resources(p, out, started)
        bank = make_population_bank(setup, index, 'FIT', 0, p, progress)
        current = np.zeros(8)
        reference, geometry = evaluate039(setup, index, current, bank, tolerances, fit_projection,
                                         fit_real, p, out, progress)
        write_json(out/f'seed{index}_FIT_M0.json', reference)
        best, current_report = 1., reference
        trial, accepted, ended = 0, 0, 'step_schedule_completed'
        search_start = time.monotonic()
        write_json(out/'PROGRESS.json', dict(phase='calibration', seed_index=index, trials=trial,
                   retained_updates=accepted, score=best, elapsed_seconds=time.monotonic()-started,
                   execution_counts=dict(progress)))
        budget_hit = False
        for step in p['search']['step_sizes']:
            for sweep in range(p['search']['sweeps_per_step']):
                sweep_accepted = 0
                for coordinate in range(8):
                    # Compare both signs around the same current point, then retain the best.
                    winner, winner_report, winner_score = current.copy(), current_report, best
                    winner_geometry = geometry
                    for sign in (-1, 1):
                        if time.monotonic()-search_start >= p['resources']['search_seconds_per_seed']:
                            budget_hit, ended = True, 'search_wall_budget'
                            break
                        check_resources(p, out, started)
                        proposal = current.copy()
                        proposal[coordinate] += sign*step
                        if abs(proposal[coordinate]) > p['search']['absolute_log_increment_bound']:
                            continue
                        trial += 1
                        progress['parameter_proposals'] += 1
                        before = time.monotonic()
                        try:
                            report, trial_geometry = evaluate039(setup, index, proposal, bank, tolerances,
                                               fit_projection, fit_real, p, out, progress)
                            decision = objective039(report, reference, tolerances, p['contract'])
                            if decision['feasible'] and decision['score'] < winner_score:
                                winner, winner_report, winner_score = proposal, report, decision['score']
                                winner_geometry = trial_geometry
                            record = dict(trial=trial, bias=proposal, step=step, sweep=sweep,
                                          coordinate=coordinate, sign=sign, decision=decision,
                                          geometry=trial_geometry, metrics=compact039(report))
                        except MeanProposalOutsideDomain038 as exc:
                            record = dict(trial=trial, bias=proposal, step=step, sweep=sweep,
                                          coordinate=coordinate, sign=sign, rejected_domain=str(exc))
                        record['evaluation_seconds'] = time.monotonic()-before
                        record['execution_counts'] = dict(progress)
                        write_json(out/f'seed{index}_trials'/f'{trial:03d}.json', record)
                        setup['models'][index].mean_adapter[-1].bias.copy_(torch.as_tensor(
                            current, device=device, dtype=setup['models'][index].mean_adapter[-1].bias.dtype))
                    if winner_score < best:
                        current, current_report, best, geometry = winner, winner_report, winner_score, winner_geometry
                        accepted += 1
                        sweep_accepted += 1
                        progress['retained_parameter_updates'] += 1
                    setup['models'][index].mean_adapter[-1].bias.copy_(torch.as_tensor(
                        current, device=device, dtype=setup['models'][index].mean_adapter[-1].bias.dtype))
                    # Persist the retained endpoint, including completed work at the wall boundary.
                    write_json(out/f'seed{index}_CURRENT.json', dict(bias=current, score=best,
                               trials=trial, retained_updates=accepted, geometry=geometry))
                    write_json(out/'PROGRESS.json', dict(phase='calibration', seed_index=index, trials=trial,
                               retained_updates=accepted, score=best, elapsed_seconds=time.monotonic()-started,
                               execution_counts=dict(progress)))
                    if budget_hit:
                        break
                if budget_hit or sweep_accepted == 0:
                    break
            if budget_hit:
                break
        endpoint = dict(seed_index=index, seed=p['seeds'][index], bias=current.tolist(),
                        trials=trial, retained_updates=accepted, fit_score=best, stopped=ended,
                        search_seconds=time.monotonic()-search_start, geometry=geometry)
        endpoints.append(endpoint)
        fits.append(current_report)
        write_json(out/f'seed{index}_FIT_ENDPOINT.json', dict(endpoint=endpoint, report=current_report))
        del bank
    write_json(out/'ENDPOINTS_FROZEN.json', dict(UTC=datetime.now(timezone.utc).isoformat(), endpoints=endpoints,
               unopened_confirmation_banks=p['confirmation_banks'], no_endpoint_reselection=True))
    confirm_projection = projection039(p['joint_projection_confirmation_seed'], device)
    confirm_real = {h: value@confirm_projection for h, value in real_embedding.items()}
    results = []
    for index, endpoint in enumerate(endpoints):
        pairs = []
        for bank_id in range(p['confirmation_banks']):
            check_resources(p, out, started)
            bank = make_population_bank(setup, index, 'CONFIRM', bank_id, p, progress)
            reference, _ = evaluate039(setup, index, np.zeros(8), bank, tolerances, confirm_projection,
                                       confirm_real, p, out, progress,
                                       examples=f'seed{index}_confirm{bank_id}_M0' if bank_id == 0 else None)
            report, geometry = evaluate039(setup, index, endpoint['bias'], bank, tolerances, confirm_projection,
                                       confirm_real, p, out, progress,
                                       examples=f'seed{index}_confirm{bank_id}' if bank_id == 0 else None)
            pairs.append((report, reference))
            write_json(out/f'seed{index}_CONFIRM_{bank_id}.json', dict(candidate=report, M0=reference, geometry=geometry))
            write_json(out/'PROGRESS.json', dict(phase='fresh_confirmation', seed_index=index,
                       completed_banks=bank_id+1, elapsed_seconds=time.monotonic()-started,
                       execution_counts=dict(progress)))
            del bank
        decision = confirmation039(pairs, tolerances, p['contract'])
        results.append(dict(endpoint=endpoint, decision=decision))
    for index, model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if state_digest(model.basis_head.state_dict()) != state_digest(setup['heads'][index]):
            raise RuntimeError('039 frozen M0 basis changed')
    if state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest']:
        raise RuntimeError('039 frozen A0 changed')
    if state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']:
        raise RuntimeError('039 frozen005 changed')
    result = dict(schema='n01/response039-result/v1', run_id=p['run_id'], results=results,
                  useful_mean_correction_both_seeds=all(r['decision']['useful_mean_correction'] for r in results),
                  scientific_status='response_match_not_established', full_random_response_acceptance=False,
                  execution_counts=dict(progress), elapsed_seconds=time.monotonic()-started,
                  frozen_A0_M0_005_verified=True, all_TRAIN_SOURCE_parents=6593, all_TRAIN_REAL_rows=21047,
                  DEV_opened=False, all_parent_count_weights_preserved=True, hardware_calibrated=False,
                  final_uncertainty='4 fresh CRN bank contrasts; same TRAIN acquisition, not unseen-real validation',
                  next_step='Review physical gains and signed protected failures, then advance the necessary bounded next stage.')
    write_json(out/'RESULT.json', result)
    write_json(out/'COMPLETE.json', dict(UTC=datetime.now(timezone.utc).isoformat(), native_exit_expected=0,
               result_sha256=sha256(out/'RESULT.json'), elapsed_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256:
        raise ValueError('039 config hash mismatch')
    p = json.loads(args.config.read_text(encoding='utf8'))
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('039 source pin mismatch: '+name)
    if sha256(ROOT/p['setup038']['path']) != p['setup038']['sha256']:
        raise ValueError('039 setup038 pin mismatch')
    p['_config_sha256'] = args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/physical, device)
    progress = Counter(dict.fromkeys(('checkpoint_loads', 'TRAIN_archives_decoded', 'PCA_archives_decoded',
        'DEV_archives_decoded', 'AD_queries', 'optimizer_steps', 'old_A_calls', 'old_A_rows',
        'response_calls', 'response_rows', 'noise005_calls', 'noise005_rows', 'extension_rows',
        'full_population_banks', 'mean_operator_calls', 'mean_operator_rows', 'generated_observation_rows',
        'full_native_hypothesis_evaluations', 'parameter_proposals', 'retained_parameter_updates'), 0))
    try:
        run039(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                       execution_counts=dict(progress), automatic_restart=False,
                       scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
