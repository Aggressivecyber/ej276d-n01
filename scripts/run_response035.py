"""Zero-update mean/random parameter exchange on previously examined TRAIN banks."""
from __future__ import annotations

import argparse
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
from n01.model008 import canonical_charge008
from n01.model029 import N01Response029
from n01.observation008 import SAMPLE_INTERVAL_NS, observe_existing_carrier, observe_generated_carrier
from n01.response008 import HYPOTHESES, loss_tensor_statistics, make_extension, saturation_penalty, state_digest, statistics_digest
from run_response032 import check_limits
from run_response033 import NUMPY_STREAMS, TORCH_STREAMS, energy_parts, make_bank, summarize_tensor

ROOT = Path(__file__).resolve().parents[1]
CORNERS = ('M0R0', 'M0Rf', 'MfR0', 'MfRf')
STATES = CORNERS+('A0_only', 'Af_only')
DATA_CONTRASTS = {
    'M0Rf_minus_M0R0': {'M0Rf': 1, 'M0R0': -1},
    'MfR0_minus_M0R0': {'MfR0': 1, 'M0R0': -1},
    'MfRf_minus_M0R0': {'MfRf': 1, 'M0R0': -1},
    'A0_only_minus_M0R0': {'A0_only': 1, 'M0R0': -1},
    'Af_only_minus_M0R0': {'Af_only': 1, 'M0R0': -1},
    'Af_only_minus_A0_only': {'Af_only': 1, 'A0_only': -1},
    'M_effect_at_Rf': {'MfRf': 1, 'M0Rf': -1},
    'R_effect_at_Mf': {'MfRf': 1, 'MfR0': -1},
    'interaction': {'MfRf': 1, 'MfR0': -1, 'M0Rf': -1, 'M0R0': 1},
}


def geometry035(signal):
    x = signal.detach().double()
    global_peak = x.argmax(1)
    selected = x[:, 112:144].argmax(1)+112
    return dict(rows=len(x), scope='pure_pre_ADC_carrier384_no_electronics',
        canonical_Q_Vns=summarize_tensor(canonical_charge008(x)),
        negative_area_carrier_Vns=summarize_tensor(torch.relu(-x).sum(1)*SAMPLE_INTERVAL_NS),
        carrier_RMS_V=summarize_tensor(x.square().mean(1).sqrt()),
        absolute_peak_V=summarize_tensor(x.abs().amax(1)),
        positive_peak_V=summarize_tensor(x.amax(1)),
        global_peak_index=dict(minimum=int(global_peak.min()), maximum=int(global_peak.max()), mean=float(global_peak.double().mean())),
        global_peak_outside_search=dict(count=int(((global_peak < 112) | (global_peak >= 144)).sum()), denominator=len(x)),
        selected_peak_at_search_boundary=dict(count=int(((selected == 112) | (selected == 143)).sum()), denominator=len(x)),
        canonical_Q_nonpositive=dict(count=int((canonical_charge008(x) <= 0).sum()), denominator=len(x)),
        software_sample_interval_ns=SAMPLE_INTERVAL_NS)


def score_observations035(signals, bank, loss, carrier_scale, progress):
    observed = [{h: observe_generated_carrier(signal+item['noise'][h], ste=False,
        q_floor=float(loss.q_floor), return_carrier=True) for h in HYPOTHESES}
        for signal, item in zip(signals, bank['generated'])]
    progress['generated_ADC_rows'] += 128*len(HYPOTHESES)
    hypotheses = {}
    for h in HYPOTHESES:
        parts = energy_parts([observed[0][h], observed[1][h], *bank['targets']], loss, carrier_scale)
        wave = torch.cat([obs[h]['window_V'] for obs in observed])
        hypotheses[h] = dict(energy={key: float(value) for key, value in parts.items()},
            saturation=float(torch.stack([saturation_penalty(signal+item['noise'][h])
                for signal, item in zip(signals, bank['generated'])]).mean()),
            observed_Q_Vns=summarize_tensor(torch.cat([obs[h]['Q'] for obs in observed])),
            observed_negative_area_Vns=summarize_tensor(torch.relu(-wave).sum(1)*SAMPLE_INTERVAL_NS),
            flags={key: dict(count=int(torch.cat([obs[h]['flags'][key] for obs in observed]).sum()), denominator=128)
                   for key in observed[0][h]['flags']})
    data_metrics = {key: sum(hypotheses[h]['energy'][key] for h in HYPOTHESES)/3
                    for key in ('wave', 'integral', 'carrier', 'data')}
    data_metrics.update({h+'.'+key: value for h, record in hypotheses.items() for key, value in record['energy'].items()})
    return dict(data=data_metrics['data'], saturation=sum(record['saturation'] for record in hypotheses.values())/3,
        data_metrics=data_metrics, hypotheses=hypotheses, primary_comparison='data_energy_only',
        forward_ADC_is_hard=True, ADC_STE=False)


def forward_corner035(model, bank, loss, carrier_scale, progress):
    outputs, features = [], []
    handle = model.mean_model.unet.register_forward_hook(lambda module, args, result: features.append(dict(
        decoded_RMS=float(result[0].detach().double().square().mean().sqrt()),
        pooled_RMS=float(result[1].detach().double().square().mean().sqrt()))))
    try:
        for item in bank['generated']:
            outputs.append(model(item['source'], item['response']))
            progress['NN_calls'] += 1; progress['NN_rows'] += 64
    finally:
        handle.remove()
    signals = [result['signal_pre'] for result in outputs]
    means = [result['base_signal_pre_V'] for result in outputs]
    report = score_observations035(signals, bank, loss, carrier_scale, progress)
    regular = {name: float(torch.stack([result['regular_'+name] for result in outputs]).mean())
               for name in ('mean', 'basis', 'charge', 'local')}
    report.update(regular=float(torch.stack([result['regular'] for result in outputs]).mean()), regular_parts=regular,
        pre_ADC_mean=geometry035(torch.cat(means)), pre_ADC_signal=geometry035(torch.cat(signals)),
        mean_residual_saturation_fraction=float(torch.stack([result['mean_residual_saturation_fraction'] for result in outputs]).mean()),
        head_scales=dict(
            B_voltage_RMS=summarize_tensor(torch.cat([r['B'].square().mean((1, 2)).sqrt() for r in outputs])),
            B_RMS_over_A_RMS=summarize_tensor(torch.cat([r['B'].square().mean((1, 2)).sqrt()/r['base_signal_pre_V'].square().mean(1).sqrt() for r in outputs])),
            charge_v_norm=summarize_tensor(torch.cat([r['charge_v'].square().sum(1).sqrt() for r in outputs])),
            charge_log_variance=summarize_tensor(torch.cat([r['charge_log_variance'] for r in outputs])),
            alpha=summarize_tensor(torch.cat([r['alpha'] for r in outputs])),
            local_envelope_V=summarize_tensor(torch.cat([r['signal_envelope'] for r in outputs])),
            local_realization_V=summarize_tensor(torch.cat([r['signal_noise'] for r in outputs]))),
        same_forward_feature_RMS=features)
    report['total'] = report['data']+.01*(report['regular']+report['saturation'])
    for h, record in report['hypotheses'].items():
        record['total'] = record['energy']['data']+.01*(report['regular']+record['saturation'])
    report['full_objective_metrics'] = dict(regular=report['regular'], saturation=report['saturation'], total=report['total'],
        **{'regular.'+name: value for name, value in regular.items()},
        **{h+'.total': record['total'] for h, record in report['hypotheses'].items()})
    return report, means


def estimate035(values):
    x = np.asarray(values, np.float64)
    mean, mcse = float(x.mean()), float(x.std(ddof=1)/math.sqrt(8))
    return dict(mean=mean, MCSE=mcse, banks=8, bank_values=x.tolist(),
        descriptive_3MCSE_direction='positive' if mean > 3*mcse else 'negative' if mean < -3*mcse else 'unresolved',
        fresh_confirmation=False)


@torch.no_grad()
def run035(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True)
    write_json(out/'START.json', dict(config=p, UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), device=str(device)))
    paths = {}
    for name, record in p['inputs'].items():
        path = Path(record['path'])
        if sha256(path) != record['sha256']: raise ValueError('035 pinned input differs: '+name)
        paths[name] = path
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest: raise ValueError('035 pinned source differs: '+name)
    initial = torch.load(paths['initial034'], map_location='cpu', weights_only=False); progress['checkpoint_loads'] += 1
    final = torch.load(paths['final034'], map_location='cpu', weights_only=False); progress['checkpoint_loads'] += 1
    if not (initial['schema'] == final['schema'] == 'n01/response034-pilot-checkpoint/v1'
            and initial['G_updates'] == 0 and final['G_updates'] == 2000 and initial['config'] == final['config']
            and initial['config']['seed'] == 2026101732
            and statistics_digest(initial['statistics']) == statistics_digest(final['statistics'])
            and initial['integral_statistics'] == final['integral_statistics']
            and torch.equal(initial['carrier_scale'], final['carrier_scale'])
            and initial['statistics']['statistics_split'] == initial['statistics']['loss']['fit_split'] == 'TRAIN'
            and initial['statistics']['DEV_waveforms_decoded'] is False
            and initial['statistics']['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']
            and initial['frozen005_state_sha256'] == final['frozen005_state_sha256']):
        raise ValueError('035 endpoint identity/statistics/frozen005 binding differs')
    noise_cp = torch.load(paths['noise005'], map_location='cpu', weights_only=False); progress['checkpoint_loads'] += 1
    if not (noise_cp['schema_version'] == 5 and noise_cp['update'] == 5000
            and noise_cp['model_type'] == 'ObservedBaselineNoiseV3' and noise_cp['engineering'] is False):
        raise ValueError('035 original005 metadata differs')
    model = N01Response029(initial['statistics']['model'], seed=2026101732).eval().requires_grad_(False).to(device)
    extension = make_extension(noise_cp, device)
    frozen_digest = state_digest(extension.state_dict())
    if frozen_digest != initial['frozen005_state_sha256']: raise ValueError('035 electronics state binding differs')
    del noise_cp
    endpoint_states = {'0': initial['model'], 'f': final['model']}
    model.load_state_dict(endpoint_states['0'], strict=True)
    if state_digest(model.state_dict()) != p['expected_initial_model_state_sha256']:
        raise ValueError('035 initial model differs from saved native034/033 digest')
    ev = initial['config']['evaluation']
    loss = N01Loss014(loss_tensor_statistics(initial['statistics']['loss']), initial['integral_statistics'],
        seed=ev['projection_training_seed'], evaluation_seed=ev['projection_evaluation_seed'], integral_seed=ev['integral_seed']).to(device)
    carrier_scale = initial['carrier_scale'].to(device)
    write_json(out/'ENDPOINT_BINDING.json', dict(initial_model_state_sha256=state_digest(initial['model']),
        final_model_state_sha256=state_digest(final['model']), frozen005_extension_state_sha256=frozen_digest,
        M_block='all mean_model.* state entries', R_block='all other model entries: basis_head, charge_head, envelope_head, local_filter',
        stats_refitted=False, optimizer_loaded=False, source_mixture='uniform_count', physical_weights='unknown'))
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
        data['real_carrier384'] = archive['real_carrier384'].copy()
    progress['TRAIN_archives_decoded'] += 1
    validate_train(data)
    if not (data['real_carrier384'].shape == (21047, 384)
            and ('original_source_weights' not in data or np.array_equal(data['original_source_weights'], np.ones(6593)))):
        raise ValueError('035 original TRAIN mixture differs')
    real_cpu = observe_existing_carrier(data['real_carrier384'], already_digitized=True,
        q_floor=float(initial['statistics']['loss']['q_floor']), return_carrier=True)
    if not (np.array_equal(real_cpu['window_V'], data['real']) and np.array_equal(real_cpu['baseline100'], data['real_baseline100'])):
        raise ValueError('035 REAL CPU observation differs from original prepared target')
    real = {key: torch.as_tensor(data['real'] if key == 'window_V' else real_cpu[key], dtype=torch.float64, device=device)
            for key in ('window_V', 'carrier_adc', 'baseline_mean_V')}
    progress['REAL_existing_observation_rows'] += 21047
    del real_cpu, initial, final
    source = torch.as_tensor(data['source_carrier384'], dtype=torch.float64, device=device)
    records, bank_contrasts, bank_full_contrasts = [], [], []
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as stream:
        for seeds in p['risk_bank_seeds']:
            bank_index = seeds['bank_index']
            nr = {name: np.random.default_rng(seeds['numpy'][name]) for name in NUMPY_STREAMS}
            tr = {name: torch.Generator(device='cpu').manual_seed(seeds['torch'][name]) for name in TORCH_STREAMS}
            bank = make_bank(model, extension, source, real, data, nr, tr, device, progress)
            reports, means = {}, {}
            for corner in CORNERS:
                m, r = endpoint_states[corner[1]], endpoint_states[corner[3]]
                state = {key: m[key] if key.startswith('mean_model.') else r[key] for key in endpoint_states['0']}
                model.load_state_dict(state, strict=True)
                reports[corner], means[corner] = forward_corner035(model, bank, loss, carrier_scale, progress)
            for label, corner in (('A0_only', 'M0R0'), ('Af_only', 'MfRf')):
                signals = means[corner]
                reports[label] = score_observations035(signals, bank, loss, carrier_scale, progress)
                reports[label].update(pre_ADC_mean=geometry035(torch.cat(signals)),
                    regular_scope='not_reported_for_A_only; no inherited random regular or full-objective total',
                    interpretation='deterministic mean network plus electronics, not the conditional mean of nonlinear observed waveforms')
            for label in STATES:
                record = reports[label]
                record.update(state=label, bank_index=bank_index, identity=bank['identity'],
                    bank_role='previously_examined033_TRAIN_diagnostic_bank', fresh_confirmation=False)
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                check_limits(p, out, started)
            contrasts = {name: {metric: sum(coefficient*reports[state]['data_metrics'][metric] for state, coefficient in coefficients.items())
                for metric in reports['M0R0']['data_metrics']} for name, coefficients in DATA_CONTRASTS.items()}
            full_contrasts = {name: {metric: sum(coefficient*reports[state]['full_objective_metrics'][metric] for state, coefficient in coefficients.items())
                for metric in reports['M0R0']['full_objective_metrics']} for name, coefficients in DATA_CONTRASTS.items()
                if all(state in CORNERS for state in coefficients)}
            bank_contrasts.append(contrasts); bank_full_contrasts.append(full_contrasts)
            write_json(out/f'BANK_{bank_index:02d}_CONTRASTS.json', dict(data=contrasts, full_objective=full_contrasts,
                A_cross_R_absmax=dict(M0=float(max((a-b).abs().max() for a, b in zip(means['M0R0'], means['M0Rf']))),
                                     Mf=float(max((a-b).abs().max() for a, b in zip(means['MfR0'], means['MfRf'])))),
                paired_contrasts_computed_before_MCSE=True))
            print(json.dumps(dict(bank_index=bank_index, execution_counts=dict(progress))), flush=True)
    summary = {state: {metric: estimate035([r['data_metrics'][metric] for r in records if r['state'] == state])
        for metric in records[0]['data_metrics']} for state in STATES}
    paired = {name: {metric: estimate035([bank[name][metric] for bank in bank_contrasts])
        for metric in bank_contrasts[0][name]} for name in DATA_CONTRASTS}
    full_paired = {name: {metric: estimate035([bank[name][metric] for bank in bank_full_contrasts])
        for metric in bank_full_contrasts[0][name]} for name in bank_full_contrasts[0]}
    if state_digest(extension.state_dict()) != frozen_digest: raise RuntimeError('035 frozen electronics changed')
    expected = dict(NN_calls=64, NN_rows=4096, noise005_rows=1024, extension_rows=3072, generated_ADC_rows=18432,
        checkpoint_loads=3, TRAIN_archives_decoded=1, REAL_existing_observation_rows=21047, banks_cached=8,
        AD_queries=0, G_updates=0, DEV_reads=0, statistics_fits=0, checkpoint_writes=0, native_W1_runs=0)
    if progress != expected: raise RuntimeError('035 fixed execution budget differs')
    write_json(out/'RESULT.json', dict(schema='n01/response035-exchange-audit-result/v1', native_status='COMPLETE',
        execution_counts=progress, six_state_data_energy=summary, paired_data_contrasts=paired,
        four_corner_full_objective_contrasts=full_paired, primary='data_energy', A_only_full_objective_reported=False,
        previously_examined_TRAIN_banks=True, fresh_confirmation=False, conditional_identification=False,
        candidate_selected=False, training_history_causal_claim=False, scientific_status='response_match_not_established',
        interpretation='Parameter-block effects include changed features/amplitude and nonlinear observation. A-only marginal fit is not a conditional-mean truth test. No hard-FD/STE-gradient conclusion is established.'))
    write_json(out/'COMPLETE.json', dict(native_status='COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'), execution_counts=progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256: raise ValueError('035 external configuration SHA differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if not (p['status'] == 'ready_for_zero_update_exchange_audit' and p['seed'] == 2026101732
            and p['states'] == list(STATES) and len(p['risk_bank_seeds']) == 8
            and p['AD_queries'] == p['G_updates'] == 0
            and set(p['inputs']) == {'initial034', 'final034', 'noise005', 'dataset'}):
        raise ValueError('035 requires the ROOT-reviewed fixed zero-update design, not its draft')
    if args.out.exists(): raise FileExistsError(args.out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory
        limit = p['resources']['GPU_GiB']*1024**3
        if physical < limit: raise ValueError('035 GPU is smaller than the registered allocator bound')
        torch.cuda.set_per_process_memory_fraction(limit/physical, device)
    progress = dict(NN_calls=0, NN_rows=0, noise005_rows=0, extension_rows=0, generated_ADC_rows=0,
        checkpoint_loads=0, TRAIN_archives_decoded=0, REAL_existing_observation_rows=0, banks_cached=0,
        AD_queries=0, G_updates=0, DEV_reads=0, statistics_fits=0, checkpoint_writes=0, native_W1_runs=0)
    try:
        run035(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error), execution_counts=progress,
                automatic_restart=False, scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
