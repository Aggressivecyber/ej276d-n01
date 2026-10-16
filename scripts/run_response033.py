"""Fixed first-Adam-step audit of response032; no continued training or DEV."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
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
from n01.model029 import N01Response029
from n01.observation008 import SAMPLE_INTERVAL_NS, observe_existing_carrier, observe_generated_carrier
from n01.response008 import HYPOTHESES, loss_tensor_statistics, make_extension, saturation_penalty, state_digest
from run_response031 import energy_four_banks, standardized_banks
from run_response032 import check_limits, electronic_noise, response_noise

ROOT = Path(__file__).resolve().parents[1]
STATES = ('baseline', 'full', 'quarter', 'sixteenth', 'mean_only', 'shape_only', 'charge_only', 'local_only')
NUMPY_STREAMS = ('source_A', 'source_B', 'real_T', 'real_U')
TORCH_STREAMS = ('response_A', 'response_B', 'noise_A', 'noise_B', 'extension_A', 'extension_B')


def energy_parts(observations, loss, carrier_scale):
    banks = standardized_banks(observations, loss, carrier_scale)
    wave = torch.stack([energy_four_banks(*(bank['views'][name] for bank in banks))
                        for name in banks[0]['views']]).mean()
    integral = energy_four_banks(*(bank['integrals'] for bank in banks))
    carrier = energy_four_banks(*(bank['carrier'] for bank in banks))
    return dict(wave=wave, integral=integral, carrier=carrier, data=wave+2.*integral+.25*carrier)


def make_bank(model, extension, source, real, data, numpy_rng, torch_rng, device, progress):
    generated, indices = [], {}
    for bank in ('A', 'B'):
        rows = numpy_rng['source_'+bank].integers(0, len(source), 64)
        indices['source_'+bank] = rows
        generated.append(dict(source=source[torch.as_tensor(rows, device=device)],
            response=response_noise(model, 64, torch_rng['response_'+bank], device),
            noise=electronic_noise(extension, 64, torch_rng['noise_'+bank], torch_rng['extension_'+bank], device, progress)))
    targets = []
    for bank in ('T', 'U'):
        rows = numpy_rng['real_'+bank].integers(0, len(real['window_V']), 128)
        indices['real_'+bank] = rows
        targets.append({key: value[torch.as_tensor(rows, device=device)] for key, value in real.items()})
    identity = {key: dict(rows=len(rows), unique_rows=len(np.unique(rows)),
        row_indices_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
        parent_ids_sha256=hashlib.sha256(data['source_ids' if key.startswith('source') else 'real_ids'][rows].tobytes()).hexdigest())
        for key, rows in indices.items()}
    progress['banks_cached'] += 1
    return dict(generated=generated, targets=targets, identity=identity)


def summarize_tensor(value):
    x = value.detach().double()
    return dict(mean=float(x.mean()), RMS=float(x.square().mean().sqrt()), minimum=float(x.min()), maximum=float(x.max()))


def evaluate_point(model, bank, loss, carrier_scale, *, ste, progress):
    outputs, observed = [], []
    for item in bank['generated']:
        result = model(item['source'], item['response'])
        progress['NN_calls'] += 1; progress['NN_rows'] += 64
        outputs.append(result)
        observed.append({h: observe_generated_carrier(result['signal_pre']+item['noise'][h],
            ste=ste, q_floor=float(loss.q_floor), return_carrier=True) for h in HYPOTHESES})
        progress['generated_ADC_rows'] += 64*len(HYPOTHESES)
    parts = {h: energy_parts([observed[0][h], observed[1][h], *bank['targets']], loss, carrier_scale)
             for h in HYPOTHESES}
    regular = torch.stack([result['regular'] for result in outputs]).mean()
    saturation = torch.stack([saturation_penalty(result['signal_pre']+item['noise'][h])
        for result, item in zip(outputs, bank['generated']) for h in HYPOTHESES]).mean()
    data_term = torch.stack([parts[h]['data'] for h in HYPOTHESES]).mean()
    total = data_term+.01*(regular+saturation)
    regular_parts = {name: float(torch.stack([result['regular_'+name] for result in outputs]).mean().detach())
                     for name in ('mean', 'basis', 'charge', 'local')}
    head_scales = dict(
        A_absolute_peak=summarize_tensor(torch.cat([r['base_signal_pre_V'].abs().amax(1) for r in outputs])),
        B_voltage_RMS=summarize_tensor(torch.cat([r['B'].square().mean((1, 2)).sqrt() for r in outputs])),
        charge_v_norm=summarize_tensor(torch.cat([r['charge_v'].square().sum(1).sqrt() for r in outputs])),
        alpha=summarize_tensor(torch.cat([r['alpha'] for r in outputs])),
        local_envelope_voltage=summarize_tensor(torch.cat([r['signal_envelope'] for r in outputs])),
        local_realization_voltage=summarize_tensor(torch.cat([r['signal_noise'] for r in outputs])),
        qbar=summarize_tensor(torch.cat([r['qbar'] for r in outputs])))
    hypothesis = {}
    for h in HYPOTHESES:
        wave = torch.cat([obs[h]['window_V'] for obs in observed])
        flags = {key: dict(count=int(torch.cat([obs[h]['flags'][key] for obs in observed]).sum()), denominator=128)
                 for key in observed[0][h]['flags']}
        h_saturation = torch.stack([saturation_penalty(result['signal_pre']+item['noise'][h])
            for result, item in zip(outputs, bank['generated'])]).mean()
        hypothesis[h] = dict(energy={key: float(value.detach()) for key, value in parts[h].items()},
            observed_Q=summarize_tensor(torch.cat([obs[h]['Q'] for obs in observed])),
            negative_area_Vns=summarize_tensor(torch.relu(-wave).sum(1)*SAMPLE_INTERVAL_NS), flags=flags,
            software_sample_interval_ns=SAMPLE_INTERVAL_NS,
            saturation=float(h_saturation.detach()), total=float((parts[h]['data']+.01*(regular+h_saturation)).detach()))
    report = dict(data=float(data_term.detach()), regular=float(regular.detach()), saturation=float(saturation.detach()),
        total=float(total.detach()), regular_parts=regular_parts, head_scales=head_scales, hypotheses=hypothesis,
        mean_residual_saturation_fraction=float(torch.stack([r['mean_residual_saturation_fraction'] for r in outputs]).mean().detach()),
        forward_ADC_is_hard=True, ADC_STE_backward_enabled=ste, generated_rows=128, REAL_rows_per_bank=128)
    report['risk_scalars'] = {key: report[key] for key in ('data', 'regular', 'saturation', 'total')}
    report['risk_scalars'].update({'regular.'+key: value for key, value in regular_parts.items()})
    report['risk_scalars'].update({h+'.'+key: value for h, record in hypothesis.items() for key, value in record['energy'].items()})
    report['risk_scalars'].update({h+'.total': record['total'] for h, record in hypothesis.items()})
    return total, report


def run033(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True)
    write_json(out/'START.json', dict(config=p, UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), device=str(device)))
    paths = {}
    for name, record in p['inputs'].items():
        path = Path(record['path'])
        if sha256(path) != record['sha256']:
            raise ValueError('033 pinned input differs: '+name)
        paths[name] = path
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('033 pinned source differs: '+name)
    cp = torch.load(paths['initial032'], map_location='cpu', weights_only=False)
    progress['checkpoint_loads'] += 1
    if not (cp['schema'] == 'n01/response032-pilot-checkpoint/v1' and cp['G_updates'] == 0
            and cp['config']['seed'] == 2026101732 and not cp['optimizer']['state'] and len(cp['optimizer']['param_groups']) == 2
            and set(cp['numpy_streams']) == set(NUMPY_STREAMS) and set(cp['torch_streams']) == set(TORCH_STREAMS)
            and not cp['source_visits'].any() and not cp['real_visits'].any()
            and cp['statistics']['statistics_split'] == cp['statistics']['loss']['fit_split'] == 'TRAIN'
            and cp['statistics']['DEV_waveforms_decoded'] is False
            and cp['statistics']['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']):
        raise ValueError('033 requires the exact initial032 TRAIN state and fresh optimizer')
    for group, lr in zip(cp['optimizer']['param_groups'], (1e-4, 3e-4)):
        if not (group['lr'] == lr and group['weight_decay'] == 0 and group['eps'] == 1e-8
                and tuple(group['betas']) == (.9, .999)):
            raise ValueError('033 original AdamW hyperparameters differ')
    noise_cp = torch.load(paths['noise005'], map_location='cpu', weights_only=False)
    progress['checkpoint_loads'] += 1
    if not (noise_cp['schema_version'] == 5 and noise_cp['update'] == 5000
            and noise_cp['model_type'] == 'ObservedBaselineNoiseV3' and noise_cp['engineering'] is False):
        raise ValueError('033 original005 metadata differs')
    model = N01Response029(cp['statistics']['model'], seed=2026101732)
    model.load_state_dict(cp['model'], strict=True)
    model.mean_model.set_stage('mean_response')
    model.train(); model.to(device)
    extension = make_extension(noise_cp, device)
    if state_digest(extension.state_dict()) != cp['frozen005_state_sha256']:
        raise ValueError('033 initial032 frozen electronics binding differs')
    frozen_digest = state_digest(extension.state_dict())
    del noise_cp
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
        data['real_carrier384'] = archive['real_carrier384'].copy()
    progress['TRAIN_archives_decoded'] += 1
    validate_train(data)
    if not (data['real_carrier384'].shape == (21047, 384)
            and ('original_source_weights' not in data or np.array_equal(data['original_source_weights'], np.ones(6593)))):
        raise ValueError('033 original TRAIN mixture differs')
    real_cpu = observe_existing_carrier(data['real_carrier384'], already_digitized=True,
        q_floor=float(cp['statistics']['loss']['q_floor']), return_carrier=True)
    if not (np.array_equal(real_cpu['window_V'], data['real'])
            and np.array_equal(real_cpu['baseline100'], data['real_baseline100'])):
        raise ValueError('033 existing REAL observation differs from prepared032 target')
    real = {key: torch.as_tensor(data['real'] if key == 'window_V' else real_cpu[key], dtype=torch.float64, device=device)
            for key in ('window_V', 'carrier_adc', 'baseline_mean_V')}
    progress['REAL_existing_observation_rows'] += 21047
    del real_cpu
    source = torch.as_tensor(data['source_carrier384'], dtype=torch.float64, device=device)
    ev = cp['config']['evaluation']
    loss = N01Loss014(loss_tensor_statistics(cp['statistics']['loss']), cp['integral_statistics'],
        seed=ev['projection_training_seed'], evaluation_seed=ev['projection_evaluation_seed'], integral_seed=ev['integral_seed']).to(device)
    carrier_scale = cp['carrier_scale'].to(device)
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([dict(params=groups['mean'], lr=1e-4),
        dict(params=groups['shape']+groups['charge']+groups['local'], lr=3e-4)], weight_decay=0.)
    optimizer.load_state_dict(cp['optimizer'])
    numpy_rng = {name: np.random.default_rng(0) for name in NUMPY_STREAMS}
    torch_rng = {name: torch.Generator(device='cpu') for name in TORCH_STREAMS}
    for name, rng in numpy_rng.items(): rng.bit_generator.state = copy.deepcopy(cp['numpy_streams'][name])
    for name, rng in torch_rng.items(): rng.set_state(cp['torch_streams'][name])
    write_json(out/'INITIAL_BINDING.json', dict(initial_model_state_sha256=state_digest(model.state_dict()),
        original_A_state_sha256=cp['original_A_state_sha256'], frozen005_extension_state_sha256=frozen_digest,
        checkpoint_streams_restored=True, model_mode='train', statistics_refitted=False,
        original_statistics_inputs=cp['config']['inputs']['statistics'], source_physical_weights='unknown',
        source_mixture='uniform_count', DEV_reads=0))
    del cp
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    first_bank = make_bank(model, extension, source, real, data, numpy_rng, torch_rng, device, progress)
    optimizer.zero_grad(set_to_none=True)
    objective, baseline = evaluate_point(model, first_bank, loss, carrier_scale, ste=True, progress=progress)
    objective.backward(); progress['AD_queries'] += 1
    group_norms = {key: math.sqrt(sum(float(value.grad.double().square().sum()) for value in values if value.grad is not None))
                   for key, values in groups.items()}
    norm = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()), 5., error_if_nonfinite=True))
    observed = dict(data=baseline['data'], objective=baseline['total'], regular=baseline['regular'], saturation=baseline['saturation'],
        preclip_gradient_norm=norm, group_gradient_norms=group_norms)
    checks = {}
    for key in ('data', 'objective', 'regular', 'saturation', 'preclip_gradient_norm'):
        absolute, relative = p['replay_tolerance']['norm' if key == 'preclip_gradient_norm' else 'loss']
        expected = p['expected_first_step'][key]
        checks[key] = dict(observed=observed[key], expected=expected,
            tolerance=absolute+relative*abs(expected), passed=abs(observed[key]-expected) <= absolute+relative*abs(expected))
    for key, value in group_norms.items():
        absolute, relative = p['replay_tolerance']['norm']
        expected = p['expected_first_step']['group_gradient_norms'][key]
        checks['group_'+key] = dict(observed=value, expected=expected,
            tolerance=absolute+relative*abs(expected), passed=abs(value-expected) <= absolute+relative*abs(expected))
    write_json(out/'FIRST_STEP_REPLAY.json', dict(checks=checks, observed=observed, all_passed=all(v['passed'] for v in checks.values())))
    if not all(value['passed'] for value in checks.values()):
        raise ValueError('033 first-step replay failed fixed tolerance; no diagnostic optimizer update')
    optimizer.step(); progress['diagnostic_G_updates'] += 1
    updated = {name: value.detach().clone() for name, value in model.named_parameters()}
    delta = {name: updated[name]-value for name, value in initial.items()}
    parameter_group = {id(value): key for key, values in groups.items() for value in values}
    group_for_name = {name: parameter_group[id(value)] for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    records = []
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as stream:
        for bank_index in range(9):
            if bank_index == 0:
                bank = first_bank
            else:
                seeds = p['risk_bank_seeds'][bank_index-1]
                nr = {name: np.random.default_rng(seeds['numpy'][name]) for name in NUMPY_STREAMS}
                tr = {name: torch.Generator(device='cpu').manual_seed(seeds['torch'][name]) for name in TORCH_STREAMS}
                bank = make_bank(model, extension, source, real, data, nr, tr, device, progress)
            for state in STATES:
                scale = {'baseline': 0., 'full': 1., 'quarter': .25, 'sixteenth': .0625}.get(state, 1.)
                active_group = state.removesuffix('_only') if state.endswith('_only') else None
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        fraction = scale if active_group is None or group_for_name[name] == active_group else 0.
                        value.copy_(updated[name] if fraction == 1 else initial[name] if fraction == 0 else initial[name]+fraction*delta[name])
                    applied_norms = {group: math.sqrt(sum(float((value-initial[name]).double().square().sum())
                        for name, value in model.named_parameters() if group_for_name[name] == group)) for group in groups}
                    if bank_index == 0 and state == 'baseline':
                        report = copy.deepcopy(baseline)
                        report['baseline_AD_forward_reused'] = True
                    else:
                        unused, report = evaluate_point(model, bank, loss, carrier_scale, ste=False, progress=progress)
                report.update(state=state, bank_index=bank_index, bank_role='exact_first_training' if bank_index == 0 else 'independent_TRAIN_risk',
                    identity=bank['identity'], parameter_delta_norms=applied_norms,
                    mean_only_is_parameter_group_not_pure_mean_signal=(state == 'mean_only'))
                stream.write(json.dumps(report, allow_nan=False)+'\n')
                records.append(report)
                check_limits(p, out, started)
            print(json.dumps(dict(bank_index=bank_index, execution_counts=dict(progress))), flush=True)
    paired = {}
    risks = {(r['bank_index'], r['state']): r['risk_scalars'] for r in records}
    for state in STATES[1:]:
        paired[state] = {}
        for metric in baseline['risk_scalars']:
            values = np.asarray([risks[(bank, state)][metric]-risks[(bank, 'baseline')][metric] for bank in range(1, 9)])
            mean, mcse = float(values.mean()), float(values.std(ddof=1)/math.sqrt(8))
            paired[state][metric] = dict(mean_delta=mean, MCSE=mcse, independent_banks=8,
                descriptive_3MCSE_direction='increase' if mean > 3*mcse else 'decrease' if mean < -3*mcse else 'unresolved',
                deltas_by_bank=values.tolist(), first_training_bank_excluded=True)
    if state_digest(extension.state_dict()) != frozen_digest:
        raise RuntimeError('033 frozen005/extension changed')
    expected_counts = dict(NN_calls=144, NN_rows=9216, noise005_rows=1152, extension_rows=3456,
        generated_ADC_rows=27648, checkpoint_loads=2, TRAIN_archives_decoded=1, AD_queries=1,
        diagnostic_G_updates=1, banks_cached=9, REAL_existing_observation_rows=21047)
    if progress != expected_counts:
        raise RuntimeError('033 execution counts differ from fixed budget')
    write_json(out/'RESULT.json', dict(schema='n01/response033-first-step-audit-result/v1', native_status='COMPLETE',
        execution_counts=progress, first_step_replay_passed=True, independent_risk_paired_deltas=paired,
        first_training_bank={r['state']: r for r in records if r['bank_index'] == 0},
        source_physical_weights='unknown', DEV_reads=0, native_W1_runs=0, continued_training=False,
        no_checkpoint_selected=True, no_checkpoint_written=True, scientific_status='response_match_not_established',
        interpretation='Fractional delta tests first-step overshoot; parameter-group deltas are not pure output interventions. Independent-bank descent is not final fit or conditional identification.'))
    write_json(out/'COMPLETE.json', dict(native_status='COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'), execution_counts=progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256: raise ValueError('033 external config SHA differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if not (p['status'] == 'ready_for_first_step_audit' and p['states'] == list(STATES) and len(p['risk_bank_seeds']) == 8
            and p['generated_bank_rows'] == 64 and p['real_bank_rows'] == 128
            and p['hypotheses'] == list(HYPOTHESES) and p['combined_AD_queries'] == p['diagnostic_G_updates'] == 1
            and set(p['inputs']) == {'initial032', 'noise005', 'dataset'}):
        raise ValueError('033 requires the ROOT-reviewed fixed audit, not the draft')
    if args.out.exists(): raise FileExistsError(args.out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory
        limit = p['resources']['GPU_GiB']*1024**3
        if physical < limit: raise ValueError('033 GPU is smaller than registered allocator bound')
        torch.cuda.set_per_process_memory_fraction(limit/physical, device)
    progress = dict(NN_calls=0, NN_rows=0, noise005_rows=0, extension_rows=0, generated_ADC_rows=0,
        checkpoint_loads=0, TRAIN_archives_decoded=0, AD_queries=0, diagnostic_G_updates=0, banks_cached=0,
        REAL_existing_observation_rows=0)
    try:
        run033(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error), execution_counts=progress,
                automatic_restart=False, scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
