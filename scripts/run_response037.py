"""One036 Adam direction audit with anchored ADC/crop controls; never train onward."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import sha256, write_json
from n01.diagnose_mean_random009 import read_train_members, validate_train
from n01.losses014 import N01Loss014
from n01.model import ADC_MIN, ADC_MAX
from n01.model029 import N01Response029
from n01.observation008 import (SAMPLE_INTERVAL_NS, crop_indices_torch,
    peak_indices_torch, observe_existing_carrier, observe_generated_carrier)
from n01.response008 import HYPOTHESES, loss_tensor_statistics, make_extension, saturation_penalty, state_digest
from run_response032 import check_limits
from run_response033 import NUMPY_STREAMS, TORCH_STREAMS, energy_parts, make_bank, summarize_tensor
from run_response035 import estimate035

ROOT = Path(__file__).resolve().parents[1]
POINTS = (('baseline', 0.), ('minus_1_64', -1/64), ('plus_1_64', 1/64),
          ('minus_1_16', -1/16), ('plus_1_16', 1/16), ('full1', 1.))
MODES = ('H', 'S', 'F')
ADC_STEP = (ADC_MAX-ADC_MIN)/(2**14-1)


def surrogate037(measured, frozen_peaks=None):
    """Diagnostic voltage readout only: no ADC/grid validation or new noise."""
    prefix = measured[:, :100].clone()
    baseline = prefix.mean(1)
    centered = measured-baseline[:, None]
    peaks = peak_indices_torch(centered) if frozen_peaks is None else frozen_peaks
    return dict(carrier_adc=measured, baseline_mean_V=baseline,
        window_V=centered.gather(1, crop_indices_torch(peaks)), peak_index=peaks)


def score037(outputs, observations, pre, anchor, bank, loss, carrier_scale, mode):
    parts = {h: energy_parts([observations[0][h], observations[1][h], *bank['targets']], loss, carrier_scale)
             for h in HYPOTHESES}
    regular = torch.stack([value['regular'] for value in outputs]).mean()
    saturation = torch.stack([saturation_penalty(value[h]) for value in pre for h in HYPOTHESES]).mean()
    data = torch.stack([value['wave']+2.*value['integral'] for value in parts.values()]).mean()
    total = data+.01*(regular+saturation)
    hypotheses = {}
    for h in HYPOTHESES:
        obs = [value[h] for value in observations]
        measured = torch.cat([value['carrier_adc'] for value in obs])
        centered = measured-torch.cat([value['baseline_mean_V'] for value in obs])[:, None]
        wave = torch.cat([value['window_V'] for value in obs])
        peaks = torch.cat([value['peak_index'] for value in obs])
        original_peaks = torch.cat([value['peaks'][h] for value in anchor])
        search = centered[:, 112:144]
        search_ties = (search == search.amax(1, keepdim=True)).sum(1)
        wave_ties = (wave == wave.amax(1, keepdim=True)).sum(1)
        q = wave.sum(1)*SAMPLE_INTERVAL_NS
        h_saturation = torch.stack([saturation_penalty(value[h]) for value in pre]).mean()
        record = dict(wave=float(parts[h]['wave'].detach()), integral=float(parts[h]['integral'].detach()),
            carrier_report_only=float(parts[h]['carrier'].detach()),
            data=float((parts[h]['wave']+2.*parts[h]['integral']).detach()),
            total=float((parts[h]['wave']+2.*parts[h]['integral']+.01*(regular+h_saturation)).detach()),
            Q_Vns=summarize_tensor(q), negative_area_Vns=summarize_tensor(torch.relu(-wave).sum(1)*SAMPLE_INTERVAL_NS),
            peak_changes_from_initial=dict(count=int((peaks != original_peaks).sum()), denominator=128),
            selected_peak_at_search_boundary=dict(count=int(((peaks == 112) | (peaks == 143)).sum()), denominator=128),
            global_peak_outside_search=dict(count=int(((centered.argmax(1) < 112) | (centered.argmax(1) >= 144)).sum()), denominator=128),
            Q_nonpositive=dict(count=int((q <= 0).sum()), denominator=128),
            search_amax_ties=dict(rows_gt1=int((search_ties > 1).sum()), rows_gt2=int((search_ties > 2).sum()), maximum=int(search_ties.max()), denominator=128),
            window_amax_ties=dict(rows_gt1=int((wave_ties > 1).sum()), rows_gt2=int((wave_ties > 2).sum()), maximum=int(wave_ties.max()), denominator=128),
            normalized_peak_floor_rows=dict(count=int((wave.amax(1) <= loss.peak_floor).sum()), denominator=128),
            selected_peak_below_crop_global_max=dict(count=int((wave[:, 16] < wave.amax(1)).sum()), denominator=128))
        if mode == 'H':
            hard0 = torch.cat([value['hard'][h] for value in anchor])
            different = measured != hard0
            record['hard_ADC_cell_changes'] = dict(cells=int(different.sum()), cells_denominator=128*384,
                rows=int(different.any(1).sum()), rows_denominator=128)
            record['hard_flags'] = {key: dict(count=int(torch.cat([value['flags'][key] for value in obs]).sum()), denominator=128)
                                    for key in obs[0]['flags']}
        hypotheses[h] = record
    report = dict(data=float(data.detach()), regular=float(regular.detach()), saturation=float(saturation.detach()),
        total=float(total.detach()), regular_parts={key: float(torch.stack([value['regular_'+key] for value in outputs]).mean().detach())
            for key in ('mean', 'basis', 'charge', 'local')}, hypotheses=hypotheses,
        physical_observation=(mode == 'H'), surrogate_is_not_physical_data=(mode != 'H'))
    report['scalars'] = {key: report[key] for key in ('data', 'regular', 'saturation', 'total')}
    report['scalars'].update({key: sum(value[key] for value in hypotheses.values())/3
                             for key in ('wave', 'integral', 'carrier_report_only')})
    report['scalars'].update({h+'.'+key: value[key] for h, value in hypotheses.items()
                             for key in ('wave', 'integral', 'data', 'total')})
    return total, report


def evaluate037(model, bank, loss, carrier_scale, progress, *, ste, anchor=None):
    outputs, pre, hard = [], [], []
    for item in bank['generated']:
        result = model(item['source'], item['response'])
        progress['NN_calls'] += 1; progress['NN_rows'] += 64
        outputs.append(result)
        values = {h: result['signal_pre']+item['noise'][h] for h in HYPOTHESES}
        pre.append(values)
        hard.append({h: observe_generated_carrier(value, ste=ste, q_floor=float(loss.q_floor),
            return_carrier=True, return_indices=True) for h, value in values.items()})
        progress['hard_generated_ADC_rows'] += 64*len(HYPOTHESES)
    if anchor is None:
        anchor = [dict(signal=value['signal_pre'].detach(),
            clamp={h: pre[i][h].detach().clamp(ADC_MIN, ADC_MAX) for h in HYPOTHESES},
            hard={h: hard[i][h]['carrier_adc'].detach() for h in HYPOTHESES},
            peaks={h: hard[i][h]['peak_index'].detach() for h in HYPOTHESES}) for i, value in enumerate(outputs)]
        is_baseline = True
    else:
        is_baseline = False
    objective, h_report = score037(outputs, hard, pre, anchor, bank, loss, carrier_scale, 'H')
    reports = dict(H=h_report)
    with torch.no_grad():
        anchored = [{h: anchor[i]['hard'][h]+(pre[i][h].clamp(ADC_MIN, ADC_MAX)-anchor[i]['clamp'][h])
                     for h in HYPOTHESES} for i in range(2)]
        for mode in ('S', 'F'):
            observations = [{h: surrogate037(value[h], None if mode == 'S' else anchor[i]['peaks'][h])
                             for h in HYPOTHESES} for i, value in enumerate(anchored)]
            progress['anchored_surrogate_crop_rows'] += 128*len(HYPOTHESES)
            if is_baseline and not all(torch.equal(observations[i][h][key], hard[i][h][key])
                for i in range(2) for h in HYPOTHESES for key in ('carrier_adc', 'baseline_mean_V', 'window_V', 'peak_index')):
                raise ValueError('037 anchored lambda0 differs from exact H baseline')
            unused, reports[mode] = score037(outputs, observations, pre, anchor, bank, loss, carrier_scale, mode)
        displacement = torch.cat([value['signal_pre'].detach()-anchor[i]['signal'] for i, value in enumerate(outputs)])
        precision = dict(signal_delta_RMS_V=float(displacement.double().square().mean().sqrt()),
            signal_delta_absmax_V=float(displacement.abs().max()),
            signal_delta_RMS_ADC_LSB=float(displacement.double().square().mean().sqrt())/ADC_STEP,
            signal_delta_absmax_ADC_LSB=float(displacement.abs().max())/ADC_STEP)
    return objective, dict(modes=reports, signal_displacement=precision,
        mode0_elementwise_equality_checked=is_baseline, ADC_STE_for_first_combined_AD=ste), anchor


def parameter_point037(model, initial, updated, delta, gradients, group_for_name, fraction):
    groups = {key: dict(coordinates=0, changed_coordinates=0, intended_nonzero_coordinates=0,
        delta_squared=0., rounding_error_squared=0., g_dot_effective_delta=0., max_abs_effective_delta=0.)
        for key in ('mean', 'shape', 'charge', 'local')}
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(updated[name] if fraction == 1 else initial[name] if fraction == 0 else initial[name]+fraction*delta[name])
            effective = value.double()-initial[name].double()
            intended = (updated[name].double()-initial[name].double())*fraction
            group = groups[group_for_name[name]]
            group['coordinates'] += value.numel()
            group['changed_coordinates'] += int(effective.ne(0).sum())
            group['intended_nonzero_coordinates'] += int(intended.ne(0).sum())
            group['delta_squared'] += float(effective.square().sum())
            group['rounding_error_squared'] += float((effective-intended).square().sum())
            group['g_dot_effective_delta'] += float((gradients[name].double()*effective).sum())
            group['max_abs_effective_delta'] = max(group['max_abs_effective_delta'], float(effective.abs().max()))
    for group in groups.values():
        group['L2_effective_delta'] = math.sqrt(group.pop('delta_squared'))
        group['L2_rounding_error_vs_scaled_actual_delta'] = math.sqrt(group.pop('rounding_error_squared'))
        group['intended_nonzero_but_rounded_to_zero'] = group['intended_nonzero_coordinates']-group['changed_coordinates']
    return dict(lambda_value=fraction, groups=groups,
        g_dot_effective_delta=sum(value['g_dot_effective_delta'] for value in groups.values()),
        floating_parameter_dtypes=sorted({str(value.dtype) for value in model.parameters()}),
        full1_copies_actual_updated_parameters=(fraction == 1), parameter_checkpoint_saved=False)


def run037(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True)
    write_json(out/'START.json', dict(config=p, UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), device=str(device)))
    paths = {}
    for name, record in p['inputs'].items():
        path = Path(record['path'])
        if sha256(path) != record['sha256']: raise ValueError('037 pinned input differs: '+name)
        paths[name] = path
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest: raise ValueError('037 pinned source differs: '+name)
    cp = torch.load(paths['initial036'], map_location='cpu', weights_only=False); progress['checkpoint_loads'] += 1
    if not (cp['schema'] == 'n01/response036-pilot-checkpoint/v1' and cp['G_updates'] == 0
            and cp['config']['seed'] == 2026101732 and cp['config']['training']['carrier_weight'] == 0
            and not cp['optimizer']['state'] and len(cp['optimizer']['param_groups']) == 3
            and set(cp['numpy_streams']) == set(NUMPY_STREAMS) and set(cp['torch_streams']) == set(TORCH_STREAMS)
            and not cp['source_visits'].any() and not cp['real_visits'].any()
            and cp['statistics']['statistics_split'] == cp['statistics']['loss']['fit_split'] == 'TRAIN'
            and cp['statistics']['DEV_waveforms_decoded'] is False
            and cp['statistics']['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']):
        raise ValueError('037 requires exact initial036 TRAIN state, statistics and fresh optimizer')
    for group, lr in zip(cp['optimizer']['param_groups'], (1e-4, 1.875e-5, 3e-4)):
        if not (group['lr'] == lr and group['weight_decay'] == 0 and group['eps'] == 1e-8 and tuple(group['betas']) == (.9, .999)):
            raise ValueError('037 native036 AdamW hyperparameters differ')
    noise_cp = torch.load(paths['noise005'], map_location='cpu', weights_only=False); progress['checkpoint_loads'] += 1
    if not (noise_cp['schema_version'] == 5 and noise_cp['update'] == 5000
            and noise_cp['model_type'] == 'ObservedBaselineNoiseV3' and noise_cp['engineering'] is False):
        raise ValueError('037 original005 metadata differs')
    model = N01Response029(cp['statistics']['model'], seed=2026101732)
    model.load_state_dict(cp['model'], strict=True)
    model.mean_model.set_stage('mean_response'); model.train(); model.to(device)
    extension = make_extension(noise_cp, device); del noise_cp
    frozen_digest = state_digest(extension.state_dict())
    if frozen_digest != cp['frozen005_state_sha256'] or state_digest(model.state_dict()) != p['expected_initial_model_state_sha256']:
        raise ValueError('037 initial model/electronics state binding differs')
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive); data['real_carrier384'] = archive['real_carrier384'].copy()
    progress['TRAIN_archives_decoded'] += 1
    validate_train(data)
    if not (data['real_carrier384'].shape == (21047, 384)
            and ('original_source_weights' not in data or np.array_equal(data['original_source_weights'], np.ones(6593)))):
        raise ValueError('037 original uniform-count TRAIN mixture differs')
    real_cpu = observe_existing_carrier(data['real_carrier384'], already_digitized=True,
        q_floor=float(cp['statistics']['loss']['q_floor']), return_carrier=True)
    if not (np.array_equal(real_cpu['window_V'], data['real']) and np.array_equal(real_cpu['baseline100'], data['real_baseline100'])):
        raise ValueError('037 existing REAL CPU observation differs from prepared target')
    real = {key: torch.as_tensor(data['real'] if key == 'window_V' else real_cpu[key], dtype=torch.float64, device=device)
            for key in ('window_V', 'carrier_adc', 'baseline_mean_V')}
    progress['REAL_existing_observation_rows'] += 21047; del real_cpu
    source = torch.as_tensor(data['source_carrier384'], dtype=torch.float64, device=device)
    ev = cp['config']['evaluation']
    loss = N01Loss014(loss_tensor_statistics(cp['statistics']['loss']), cp['integral_statistics'],
        seed=ev['projection_training_seed'], evaluation_seed=ev['projection_evaluation_seed'], integral_seed=ev['integral_seed']).to(device)
    carrier_scale = cp['carrier_scale'].to(device)
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([dict(params=groups['mean'], lr=1e-4), dict(params=groups['shape'], lr=1.875e-5),
        dict(params=groups['charge']+groups['local'], lr=3e-4)], weight_decay=0.)
    optimizer.load_state_dict(cp['optimizer'])
    nr = {name: np.random.default_rng(0) for name in NUMPY_STREAMS}
    tr = {name: torch.Generator(device='cpu') for name in TORCH_STREAMS}
    for name, rng in nr.items(): rng.bit_generator.state = copy.deepcopy(cp['numpy_streams'][name])
    for name, rng in tr.items(): rng.set_state(cp['torch_streams'][name])
    write_json(out/'INITIAL_BINDING.json', dict(initial_model_state_sha256=state_digest(model.state_dict()),
        frozen005_extension_state_sha256=frozen_digest, original_A_state_sha256=cp['original_A_state_sha256'],
        statistics_refitted=False, four_numpy_six_torch_streams_restored=True, model_mode='train',
        physical_weights='unknown', source_mixture='uniform_count', DEV_reads=0))
    del cp
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    first_bank = make_bank(model, extension, source, real, data, nr, tr, device, progress)
    optimizer.zero_grad(set_to_none=True)
    objective, baseline, first_anchor = evaluate037(model, first_bank, loss, carrier_scale, progress, ste=True)
    objective.backward(); progress['training_AD_queries'] += 1
    gradients = {name: torch.zeros_like(value) if value.grad is None else value.grad.detach().clone()
                 for name, value in model.named_parameters()}
    group_norms = {key: math.sqrt(sum(float(value.grad.double().square().sum()) for value in values if value.grad is not None))
                   for key, values in groups.items()}
    norm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), 5., error_if_nonfinite=True))
    observed = {key: baseline['modes']['H'][key] for key in ('data', 'regular', 'saturation')}
    observed.update(objective=baseline['modes']['H']['total'], preclip_gradient_norm=norm, group_gradient_norms=group_norms)
    checks = {}
    for key in ('data', 'objective', 'regular', 'saturation', 'preclip_gradient_norm'):
        absolute, relative = p['replay_tolerance']['norm' if key == 'preclip_gradient_norm' else 'loss']
        expected = p['expected_first_step'][key]
        checks[key] = dict(observed=observed[key], expected=expected, tolerance=absolute+relative*abs(expected),
            passed=abs(observed[key]-expected) <= absolute+relative*abs(expected))
    for key, value in group_norms.items():
        absolute, relative = p['replay_tolerance']['norm']; expected = p['expected_first_step']['group_gradient_norms'][key]
        checks['group_'+key] = dict(observed=value, expected=expected, tolerance=absolute+relative*abs(expected),
            passed=abs(value-expected) <= absolute+relative*abs(expected))
    write_json(out/'FIRST_STEP_REPLAY.json', dict(checks=checks, all_passed=all(value['passed'] for value in checks.values())))
    if not all(value['passed'] for value in checks.values()): raise ValueError('037 native036 first replay failed; no diagnostic update')
    optimizer.step(); progress['diagnostic_G_updates'] += 1
    updated = {name: value.detach().clone() for name, value in model.named_parameters()}
    updated_digest = state_digest(model.state_dict())
    delta = {name: updated[name]-value for name, value in initial.items()}
    group_ids = {id(value): key for key, values in groups.items() for value in values}
    group_for_name = {name: group_ids[id(value)] for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    direction = {name: parameter_point037(model, initial, updated, delta, gradients, group_for_name, fraction)
                 for name, fraction in POINTS}
    write_json(out/'PARAMETER_DIRECTION.json', dict(points=direction, gradient_scope='exact preclip combined036 objective',
        actual_updated_model_state_sha256=updated_digest, scalar_parameters_not_persisted=True))
    records = []
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as stream:
        for bank_index in range(9):
            if bank_index == 0:
                bank, anchor = first_bank, first_anchor
            else:
                seeds = p['risk_bank_seeds'][bank_index-1]
                nr = {name: np.random.default_rng(seeds['numpy'][name]) for name in NUMPY_STREAMS}
                tr = {name: torch.Generator(device='cpu').manual_seed(seeds['torch'][name]) for name in TORCH_STREAMS}
                bank = make_bank(model, extension, source, real, data, nr, tr, device, progress); anchor = None
            for name, fraction in POINTS:
                with torch.no_grad():
                    for parameter_name, value in model.named_parameters():
                        value.copy_(updated[parameter_name] if fraction == 1 else initial[parameter_name] if fraction == 0
                                    else initial[parameter_name]+fraction*delta[parameter_name])
                    if bank_index == 0 and fraction == 0:
                        report = copy.deepcopy(baseline); report['baseline_AD_forward_reused'] = True
                    else:
                        unused, report, anchor = evaluate037(model, bank, loss, carrier_scale, progress, ste=False, anchor=anchor)
                report.update(point=name, lambda_value=fraction, bank_index=bank_index, identity=bank['identity'],
                    bank_role='exact_first036_training' if bank_index == 0 else 'previously_examined033_TRAIN_diagnostic',
                    fresh_confirmation=False, g_dot_effective_parameter_delta=direction[name]['g_dot_effective_delta'])
                records.append(report); stream.write(json.dumps(report, allow_nan=False)+'\n')
                check_limits(p, out, started)
            print(json.dumps(dict(bank_index=bank_index, execution_counts=progress)), flush=True)
    lookup = {(r['bank_index'], r['point']): r for r in records}
    metrics = baseline['modes']['H']['scalars']
    paired = {name: {mode: {metric: estimate035([lookup[bank, name]['modes'][mode]['scalars'][metric]
        -lookup[bank, 'baseline']['modes'][mode]['scalars'][metric] for bank in range(1, 9)])
        for metric in metrics} for mode in MODES} for name, fraction in POINTS if fraction != 0}
    nested = {name: {label: {metric: estimate035([lookup[bank, name]['modes'][a]['scalars'][metric]
        -lookup[bank, name]['modes'][b]['scalars'][metric] for bank in range(1, 9)]) for metric in metrics}
        for label, a, b in (('H_minus_S', 'H', 'S'), ('S_minus_F', 'S', 'F'))} for name, fraction in POINTS}
    first_fd, risk_fd = {}, {}
    for suffix, epsilon in (('1_64', 1/64), ('1_16', 1/16)):
        plus, minus = 'plus_'+suffix, 'minus_'+suffix
        predicted = (direction[plus]['g_dot_effective_delta']-direction[minus]['g_dot_effective_delta'])/(2*epsilon)
        first_fd[suffix] = dict(epsilon=epsilon, effective_parameter_linear_prediction=predicted, modes={})
        risk_fd[suffix] = {}
        for mode in MODES:
            first_fd[suffix]['modes'][mode] = {metric: (lookup[0, plus]['modes'][mode]['scalars'][metric]
                -lookup[0, minus]['modes'][mode]['scalars'][metric])/(2*epsilon) for metric in metrics}
            risk_fd[suffix][mode] = {metric: estimate035([(lookup[bank, plus]['modes'][mode]['scalars'][metric]
                -lookup[bank, minus]['modes'][mode]['scalars'][metric])/(2*epsilon) for bank in range(1, 9)]) for metric in metrics}
        first_fd[suffix]['F_total_minus_linear_prediction'] = first_fd[suffix]['modes']['F']['total']-predicted
    if state_digest(extension.state_dict()) != frozen_digest or state_digest(model.state_dict()) != updated_digest:
        raise RuntimeError('037 frozen electronics or final full1 state differs')
    if progress != p['budget']: raise RuntimeError('037 fixed execution budget differs')
    write_json(out/'RESULT.json', dict(schema='n01/response037-direction-audit-result/v1', native_status='COMPLETE',
        execution_counts=progress, first_step_replay_passed=True, first_training_bank={r['point']: r for r in records if r['bank_index'] == 0},
        parameter_direction=direction, first_training_centered_FD=first_fd, reused_TRAIN_risk_centered_FD=risk_fd,
        F_total_FD_scale_difference=first_fd['1_64']['modes']['F']['total']-first_fd['1_16']['modes']['F']['total'],
        reused_TRAIN_paired_deltas=paired, reused_TRAIN_nested_mode_differences=nested,
        fresh_confirmation=False, conditional_identification=False, lambda_selected=False, continued_training=False,
        checkpoints_written=False, scientific_status='response_match_not_established',
        interpretation='Two scales diagnose numerical stability, not LR selection. F retains amax/abs/guards/clamps and float32 parameter rounding; unstableFD is not an AD bug. H-S andS-F are nested functional comparisons with interactions. Initial direction evidence does not prove a later training cause or detector-response fit.'))
    write_json(out/'COMPLETE.json', dict(native_status='COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'), execution_counts=progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256: raise ValueError('037 external config SHA differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if not (p['status'] == 'ready_for_direction_audit' and p['seed'] == 2026101732
            and p['lambdas'] == [value for name, value in POINTS] and p['modes'] == list(MODES)
            and len(p['risk_bank_seeds']) == 8 and set(p['inputs']) == {'initial036', 'noise005', 'dataset'}):
        raise ValueError('037 requires the fixed ROOT-reviewed audit, not its draft')
    if args.out.exists(): raise FileExistsError(args.out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']); torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory; limit = p['resources']['GPU_GiB']*1024**3
        if physical < limit: raise ValueError('037 GPU is smaller than registered allocator bound')
        torch.cuda.set_per_process_memory_fraction(limit/physical, device)
    progress = {key: 0 for key in p['budget']}
    try:
        run037(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error), execution_counts=progress,
                automatic_restart=False, scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
