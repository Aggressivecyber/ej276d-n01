"""One fixed existing-DATASET008 energy pilot; draft, never auto-launches.

The caller freezes config/source and holds the existing whole-project flock.
Original A is a strict mean warmstart; 005/extension never train. Old W1 remains
the reported hard metric. No teacher, model grid or inferred REAL labels.
"""
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
from n01.electronics_v3 import ObservedBaselineNoiseV3
from n01.evaluate_window008 import _load_dev
from n01.losses008 import ADC_STEP, _wave_views
from n01.losses008_v2 import (WAVE_WEIGHTS, VIEWS, COORDINATE_VIEWS, SHORT_LAGS,
    empirical_wasserstein_1d, _pair_projection)
from n01.losses014 import N01Loss014, fit_integral_statistics014
from n01.model029 import N01Response029
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
from n01.response008 import (HYPOTHESES, loss_tensor_statistics, make_extension,
    saturation_penalty, state_digest, statistics_digest)
from n01.train import atomic_torch_save
from run_response031 import energy_data
from run_wp1_013 import REAL_METADATA, REAL_UNKNOWN_FLAGS
from wp1_evaluation import observable_features, raw_distance, output_q_conditions

ROOT = Path(__file__).resolve().parents[1]


def column_W1(fake, real, columns):
    """Original scalar-column CDF W1, with bounded temporary storage."""
    total = fake.new_zeros(())
    for start in range(0, fake.shape[1], columns):
        stop = min(start+columns, fake.shape[1])
        total = total+(stop-start)*empirical_wasserstein_1d(fake[:, start:stop], real[:, start:stop])
    return total/fake.shape[1]


class NativeMetricLoss034(N01Loss014):
    """Only evaluation memory layout differs from the original native loss."""
    column_chunk = 16

    @torch.no_grad()
    def wave_components(self, fake, real, weights, projection_context, mean_response=False):
        fv, rv = _wave_views(fake.double(), self.peak_floor), _wave_views(real.double(), self.peak_floor)
        f = {name: self._z(name, fv[name]) for name in VIEWS}
        r = {name: self._z(name, rv[name]) for name in VIEWS}
        swd = [column_W1(f[name]@projection_context.banks['wave:'+name].to(f[name]),
            r[name]@projection_context.banks['wave:'+name].to(r[name]), self.column_chunk) for name in VIEWS]
        coordinate, joint = [], []
        for name in COORDINATE_VIEWS:
            coordinate.append(column_W1(f[name], r[name], self.column_chunk))
            for lag in SHORT_LAGS:
                directions = projection_context.banks[f'lag:{name}:{lag}']
                joint.append(column_W1(_pair_projection(f[name], lag, directions),
                    _pair_projection(r[name], lag, directions), self.column_chunk))
        return dict(swd=torch.stack(swd).mean(), coordinate=torch.stack(coordinate).mean(),
                    short_lag=torch.stack(joint).mean())


def check_limits(p, out, started):
    if time.monotonic()-started > p['resources']['wall_seconds']:
        raise TimeoutError('034 registered wall limit exceeded')
    if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > p['resources']['output_MiB']*1024**2:
        raise OSError('034 registered output limit exceeded')


def response_noise(model, count, generator, device):
    return {key: value.to(device) for key, value in model.sample_noise(count, device='cpu', generator=generator).items()}


def electronic_noise(extension, count, noise_generator, extension_generator, device, progress):
    latents = {key: torch.randn((count,)+shape, generator=noise_generator, dtype=torch.float32).to(device)
               for key, shape in ObservedBaselineNoiseV3.noise_shapes.items()}
    epsilon = torch.randn(count, 1, 284, generator=extension_generator, dtype=torch.float32).to(device)
    result = extension.sample_all(latents, epsilon)['noise_pre']
    progress['noise005_rows'] += count
    progress['extension_rows'] += count*len(HYPOTHESES)
    return result


@torch.no_grad()
def hard_metrics(wave, baseline, real_wave, real_baseline, loss, context):
    device = loss.q_reference.device
    f, r, fb, rb = [torch.as_tensor(value, dtype=torch.float64, device=device)
                    for value in (wave, real_wave, baseline, real_baseline)]
    parts = loss.wave_components(f, r, None, context, mean_response=False)
    qpsd = loss._qpsd_with_context(f, r, None, context, mean_response=False)
    values = dict(wave=float(sum(WAVE_WEIGHTS[key]*value for key, value in parts.items())),
                  QPSD=float(qpsd), noise=float(loss.noise_loss(fb, rb, None, projection_context=context)))
    fake_features, fake_flags = observable_features(wave, baseline, float(loss.q_floor))
    real_features, real_flags = observable_features(real_wave, real_baseline, float(loss.q_floor))
    defined = {}
    for name in ('Q_Vns', 'front_Vns', 'tail_Vns', 'PSD_raw', 'PSD_guarded', 'peak_voltage_V',
                 'rise10_90_ns', 'FWHM_ns', 'baseline_mean_V', 'baseline_RMS_V'):
        defined[name] = raw_distance(fake_features[name], real_features[name])
        values[name] = defined[name]['value']
    for name in ('Q_nonpositive', 'PSD_undefined', 'PSD_guarded', 'rise_undefined', 'FWHM_undefined',
                 'temporal_nonpositive_peak', 'rise_prepeak_censored', 'FWHM_prepeak_censored', 'FWHM_postpeak_censored'):
        values[name+'_rate'] = float(abs(np.asarray(fake_flags[name]).mean()-np.asarray(real_flags[name]).mean()))
    conditions = output_q_conditions(fake_features, real_features, loss)
    del conditions['fake_bin_per_row'], conditions['real_bin_per_row']
    return dict(values=values, wave_components={key: float(value) for key, value in parts.items()},
        conditional_coverage=copy.deepcopy(loss.last_diagnostics), defined_populations=defined,
        output_Q_conditions=conditions,
        fake_rows=len(wave), real_rows=len(real_wave), all_rows_retained=True,
        fake_flags={key: dict(count=int(value.sum()), denominator=len(value)) for key, value in fake_flags.items()},
        real_flags={key: dict(count=int(value.sum()), denominator=len(value)) for key, value in real_flags.items()},
        REAL_pre_ADC_unknown_flags=list(REAL_UNKNOWN_FLAGS),
        projection_context=context.metadata(), standardization='original_frozen_TRAIN')


@torch.no_grad()
def evaluate034(model, A_reference, extension, source, real_wave, real_baseline, p, loss, context,
                scope, initial, out, progress, started):
    """One response pass per output; initial A is already in G0's returned dict."""
    model.eval()
    labels = ('A', 'G0') if initial else ('A', 'final') if A_reference is not None else ('final',)
    rows = np.repeat(np.arange(len(source)), p['evaluation']['draws_per_parent'])
    waves = {label: {h: np.empty((len(rows), 256), np.float64) for h in HYPOTHESES} for label in labels}
    baseline = np.empty((len(rows), 100), np.float64)
    flags = {label: {h: {} for h in HYPOTHESES} for label in labels}
    seed = p['evaluation'][scope+'_seed']
    generators = [torch.Generator(device='cpu').manual_seed(seed+i) for i in range(3)]
    device = loss.q_reference.device
    for start in range(0, len(rows), p['evaluation']['NN_chunk_rows']):
        stop = min(start+p['evaluation']['NN_chunk_rows'], len(rows))
        x = torch.as_tensor(source[rows[start:stop]], dtype=torch.float64, device=device)
        latent = response_noise(model, len(x), generators[0], device)
        result = model(x, latent)
        progress['NN_calls'] += 1; progress['NN_rows'] += len(x)
        signals = {'G0' if initial else 'final': result['signal_pre']}
        if initial:
            signals['A'] = result['base_signal_pre_V']
        elif A_reference is not None:
            zeros = {key: x.new_zeros((len(x),)+shape) for key, shape in A_reference.noise_shapes.items()}
            signals['A'] = A_reference.signal_forward(x, zeros)['signal_pre']
            progress['NN_calls'] += 1; progress['NN_rows'] += len(x)
        noise = electronic_noise(extension=extension, count=len(x),
            noise_generator=generators[1], extension_generator=generators[2], device=device, progress=progress)
        reference_prefix = None
        for label in labels:
            for h in HYPOTHESES:
                observed = observe_generated_carrier(signals[label]+noise[h], ste=False, q_floor=float(loss.q_floor))
                progress['generated_ADC_rows'] += len(x)
                waves[label][h][start:stop] = observed['window_V'].cpu().numpy()
                if reference_prefix is None:
                    reference_prefix = observed['baseline100']
                    baseline[start:stop] = reference_prefix.cpu().numpy()
                elif not torch.equal(reference_prefix, observed['baseline100']):
                    raise RuntimeError('034 paired prefix changed across endpoints/hypotheses')
                for key, value in observed['flags'].items():
                    flags[label][h].setdefault(key, []).append(value.cpu().numpy())
        check_limits(p, out, started)
    reports = {}
    for label in labels:
        reports[label] = {}
        for h in HYPOTHESES:
            report = hard_metrics(waves[label][h], baseline, real_wave, real_baseline, loss, context)
            report.update(endpoint=label, split=scope, previously_exposed_DEV=(scope == 'DEV'),
                source_parents=len(source), draws_per_parent=p['evaluation']['draws_per_parent'], MC_banks=1,
                original_observation_flags={key: dict(nonzero_rows=int(np.count_nonzero(np.concatenate(value))), denominator=len(rows))
                                            for key, value in flags[label][h].items()})
            reports[label][h] = report
            write_json(out/f'{scope}_{label}_{h}.json', report)
            selected = np.linspace(0, len(source)-1, 32, dtype=np.int64)*p['evaluation']['draws_per_parent']
            np.savez_compressed(out/f'{scope}_{label}_{h}_examples.npz', waveform256=waves[label][h][selected],
                baseline100=baseline[selected], source_parent_rows=rows[selected],
                population_mean_V=waves[label][h].mean(0), population_std_V=waves[label][h].std(0),
                example_row_indices=selected)
            check_limits(p, out, started)
    return reports


def run034(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True)
    write_json(out/'START.json', dict(config=p, UTC=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), device=str(device), automatic_restart=False, scientific_acceptance=False))
    paths = {}
    for name, record in p['inputs'].items():
        path = Path(record['path'])
        if not path.is_absolute(): path = ROOT/path
        if sha256(path) != record['sha256']:
            raise ValueError('034 pinned input differs: '+name)
        paths[name] = path
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('034 pinned source differs: '+name)
    statistics, domain, base = [json.loads(paths[key].read_text(encoding='utf8'))
                                for key in ('statistics', 'data_statistics', 'base_config')]
    if not (statistics['statistics_split'] == domain['split'] == statistics['loss']['fit_split'] == 'TRAIN'
            and statistics['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']
            and statistics['data_statistics_sha256'] == p['inputs']['data_statistics']['sha256']
            and statistics['DEV_waveforms_decoded'] is False and domain['DEV_waveforms_decoded'] is False):
        raise ValueError('034 original TRAIN statistics binding differs')
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
        data['real_carrier384'] = archive['real_carrier384'].copy()
        for key in REAL_METADATA: data[key] = archive[key].copy()
    progress['TRAIN_archives_decoded'] += 1
    validate_train(data)
    if data['real_carrier384'].shape != (21047, 384): raise ValueError('034 REAL carrier shape differs')
    if 'original_source_weights' in data and not np.array_equal(data['original_source_weights'], np.ones(6593)):
        raise ValueError('034 registered uniform count weights differ')
    np.savez_compressed(out/'TRAIN_LINEAGE.npz', source_ids=data['source_ids'], real_ids=data['real_ids'],
        real_groups=data['real_groups'], source_role=data['source_role'], real_role=data['real_role'],
        source_count_weights=np.ones(6593), calibrated_physical_weights=np.asarray('unknown'))
    a = torch.load(paths['deterministic'], map_location='cpu', weights_only=False)
    progress['checkpoint_loads'] += 1
    noise_cp = torch.load(paths['noise005'], map_location='cpu', weights_only=False)
    progress['checkpoint_loads'] += 1
    if not (a['schema'] == 'response008/v1' and a['arm'] == 'deterministic' and a['G_updates'] == 19000
            and a['stage'] == 'low_learning_rate' and a['config'] == base
            and a['D_updates'] == {h: 16000 for h in HYPOTHESES}
            and a['noise_updates'] == 0 and a['population_learning'] is False
            and statistics_digest(a['statistics']) == statistics_digest(statistics)
            and noise_cp['schema_version'] == 5 and noise_cp['update'] == 5000
            and noise_cp['model_type'] == 'ObservedBaselineNoiseV3' and noise_cp['engineering'] is False):
        raise ValueError('034 original A/005 metadata differs')
    model = N01Response029(statistics['model'], seed=p['seed'])
    model.mean_model.load_state_dict(a['model'], strict=True)
    model.mean_model.set_stage('mean_response')
    A_reference = copy.deepcopy(model.mean_model).eval().requires_grad_(False).to(device)
    model.to(device)
    extension = make_extension(noise_cp, device)
    if state_digest(extension.model005.state_dict()) != a['noise005_state_sha256']:
        raise ValueError('034 frozen005 does not match original A')
    old_A_digest = state_digest(A_reference.state_dict())
    noise_digest = state_digest(extension.state_dict())
    del a, noise_cp
    real_cpu = observe_existing_carrier(data['real_carrier384'],
        already_digitized=True, q_floor=float(statistics['loss']['q_floor']), return_carrier=True)
    if not (np.array_equal(real_cpu['window_V'], data['real'])
            and np.array_equal(real_cpu['baseline100'], data['real_baseline100'])):
        raise ValueError('034 existing REAL observation differs from prepared TRAIN')
    real_reference_flags = dict(TRAIN=dict(rows=21047,
        observation={key: dict(count=int(value.sum()), denominator=len(value)) for key, value in real_cpu['flags'].items()},
        saved_peak_tied=dict(count=int((data['real_peak_tied_samples'] > 1).sum()), denominator=21047),
        saved_peak_at_search_boundary=dict(count=int(data['real_peak_search_edge'].sum()), denominator=21047),
        saved_global_peak_outside_search=dict(count=int(data['real_global_peak_outside_search'].sum()), denominator=21047),
        pre_ADC_flags_unknown=list(REAL_UNKNOWN_FLAGS)),
        observation_replay=dict(device='cpu', prepared_window_bitwise_equal=True, prepared_baseline_bitwise_equal=True,
                                target_window='original_prepared_REAL256', second_ADC_applied=False))
    write_json(out/'REAL_REFERENCE_FLAGS.json', real_reference_flags)
    integral = fit_integral_statistics014(torch.as_tensor(data['real'], dtype=torch.float64))
    centered = torch.as_tensor(real_cpu['carrier_adc']-real_cpu['baseline_mean_V'][:, None], dtype=torch.float64)
    carrier_scale = centered.square().mean(0).sqrt().clamp_min(ADC_STEP).to(device)
    real = {key: torch.as_tensor(data['real'] if key == 'window_V' else real_cpu[key], dtype=torch.float64, device=device)
            for key in ('window_V', 'carrier_adc', 'baseline_mean_V')}
    del centered, real_cpu
    ev = p['evaluation']
    loss = NativeMetricLoss034(loss_tensor_statistics(statistics['loss']), integral,
        seed=ev['projection_training_seed'], evaluation_seed=ev['projection_evaluation_seed'], integral_seed=ev['integral_seed']).to(device)
    loss.column_chunk = ev['W1_column_chunk']
    context = loss.create_evaluation_context()
    if context.metadata()['fingerprint'] != ev['projection_fingerprint']:
        raise ValueError('034 frozen evaluation projection differs')
    write_json(out/'TRAIN_STATISTICS.json', dict(integrals=integral, carrier_scale=carrier_scale,
        original_native_statistics=p['inputs']['statistics'], fit_rows=21047, fit_split='TRAIN', DEV_used=False))
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([dict(params=groups['mean'], lr=p['training']['mean_lr']),
        dict(params=groups['shape'], lr=p['training']['shape_lr']),
        dict(params=groups['charge']+groups['local'], lr=p['training']['new_head_lr'])], weight_decay=0.)
    write_json(out/'WARMSTART_BINDING.json', dict(mean_state_sha256=old_A_digest,
        initial_model_state_sha256=state_digest(model.state_dict()),
        strict_whole_mean_loaded=True, mean_trainable=True, frozen005_extension_state_sha256=noise_digest,
        mean_parameters=sum(value.numel() for value in groups['mean']),
        new_parameters=sum(value.numel() for key in ('shape', 'charge', 'local') for value in groups[key]),
        source_mixture='uniform_count', calibrated_physical_weights='unknown', no_neural_identity_inputs=True))
    numpy_rng = {name: np.random.default_rng(p['seed']+100+i) for i, name in enumerate(('source_A', 'source_B', 'real_T', 'real_U'))}
    torch_rng = {name: torch.Generator(device='cpu').manual_seed(p['seed']+200+i)
                 for i, name in enumerate(('response_A', 'response_B', 'noise_A', 'noise_B', 'extension_A', 'extension_B'))}
    source_visits, real_visits = np.zeros(6593, np.int64), np.zeros(21047, np.int64)
    source = torch.as_tensor(data['source_carrier384'], dtype=torch.float64, device=device)
    def checkpoint(name):
        atomic_torch_save(dict(schema='n01/response034-pilot-checkpoint/v1', config=p,
            model={key: value.detach().cpu() for key, value in model.state_dict().items()}, optimizer=optimizer.state_dict(),
            G_updates=progress['G_updates'], numpy_streams={key: value.bit_generator.state for key, value in numpy_rng.items()},
            torch_streams={key: value.get_state() for key, value in torch_rng.items()},
            source_visits=source_visits, real_visits=real_visits, statistics=statistics, integral_statistics=integral,
            carrier_scale=carrier_scale.cpu(), original_A_state_sha256=old_A_digest, frozen005_state_sha256=noise_digest), out/name)
    checkpoint('initial.pt')
    initial = evaluate034(model, None, extension, data['source_carrier384'], data['real'], data['real_baseline100'],
        p, loss, context, 'TRAIN', True, out, progress, started)
    model.train()
    parameters = list(model.parameters())
    with (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as stream:
        for step in range(1, p['training']['updates']+1):
            optimizer.zero_grad(set_to_none=True)
            generated, response_outputs = [], []
            for bank in ('A', 'B'):
                indices = numpy_rng['source_'+bank].integers(0, len(source), p['training']['generated_bank_rows'])
                np.add.at(source_visits, indices, 1)
                latent = response_noise(model, len(indices), torch_rng['response_'+bank], device)
                result = model(source[torch.as_tensor(indices, device=device)], latent)
                progress['NN_calls'] += 1; progress['NN_rows'] += len(indices)
                noise = electronic_noise(extension, len(indices), torch_rng['noise_'+bank], torch_rng['extension_'+bank], device, progress)
                observations = {h: observe_generated_carrier(result['signal_pre']+noise[h], ste=True,
                    q_floor=float(loss.q_floor), return_carrier=True) for h in HYPOTHESES}
                progress['generated_ADC_rows'] += len(indices)*len(HYPOTHESES)
                generated.append(observations)
                response_outputs.append((result, noise))
            targets = []
            for bank in ('T', 'U'):
                indices = numpy_rng['real_'+bank].integers(0, 21047, p['training']['real_bank_rows'])
                np.add.at(real_visits, indices, 1)
                targets.append({key: real[key][torch.as_tensor(indices, device=device)]
                                for key in ('window_V', 'carrier_adc', 'baseline_mean_V')})
            data_terms = {h: energy_data([generated[0][h], generated[1][h], *targets], loss, carrier_scale) for h in HYPOTHESES}
            regular = torch.stack([result['regular'] for result, noise in response_outputs]).mean()
            saturation = torch.stack([saturation_penalty(result['signal_pre']+noise[h])
                for result, noise in response_outputs for h in HYPOTHESES]).mean()
            data_term = torch.stack(list(data_terms.values())).mean()
            objective = data_term+p['training']['regular_weight']*(regular+saturation)
            if not bool(torch.isfinite(objective)): raise FloatingPointError('034 nonfinite energy objective')
            # One backward of the combined three-hypothesis objective; no per-loss AD probes.
            objective.backward(); progress['training_AD_queries'] += 1
            if step == 1:
                gradients = {}
                named = {id(value): name for name, value in model.named_parameters()}
                for key, values in groups.items():
                    nonzero = [(named[id(value)], float(value.grad.double().norm())) for value in values
                               if value.grad is not None and bool(value.grad.ne(0).any())]
                    norm = math.sqrt(sum(float(value.grad.double().square().sum()) for value in values if value.grad is not None))
                    gradients[key] = dict(norm=norm, nonzero_parameter_tensors=len(nonzero), last_nonzero_parameter=None if not nonzero else nonzero[-1])
                write_json(out/'STARTUP_GROUP_GRADIENTS.json', gradients)
                if any(value['norm'] == 0 for value in gradients.values()): raise RuntimeError('034 trainable group has zero startup gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['clip_norm'], error_if_nonfinite=True)
            optimizer.step(); progress['G_updates'] += 1
            record = dict(step=step, data=float(data_term.detach()), regular=float(regular.detach()), saturation=float(saturation.detach()),
                objective=float(objective.detach()), preclip_gradient_norm=float(norm),
                hypothesis_energy={key: float(value.detach()) for key, value in data_terms.items()},
                mean_residual_saturation_fraction=float(torch.stack([result['mean_residual_saturation_fraction']
                    for result, noise in response_outputs]).mean()))
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            if step == 1 or step % p['training']['status_interval'] == 0:
                write_json(out/'STATUS.json', dict(**record, execution_counts=dict(progress)))
                print(json.dumps(record, allow_nan=False), flush=True)
                check_limits(p, out, started)
            if step % p['training']['checkpoint_interval'] == 0 and step < p['training']['updates']:
                checkpoint(f'periodic_{step:05d}.pt')
    model.eval()
    checkpoint('final.pt')
    write_json(out/'FINAL_ENDPOINT_FREEZE.json', dict(model_state_sha256=state_digest(model.state_dict()),
        G_updates=progress['G_updates'], evaluation=ev, training_complete=True, DEV_numeric_reads_so_far=0))
    final = evaluate034(model, None, extension, data['source_carrier384'], data['real'], data['real_baseline100'],
        p, loss, context, 'TRAIN', False, out, progress, started)
    dev = _load_dev(paths['original_container'], paths['identities'], base['window'], float(loss.q_floor))
    progress['DEV_archives_decoded'] += 1
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        if not (np.array_equal(archive['source_val_ids'], dev['source_ids'])
                and np.array_equal(archive['real_val_original_index'], dev['real_ids'])
                and np.array_equal(archive['real_val_group'], dev['real_groups'])
                and set(np.unique(archive['source_val_role'])) == {'DEV'}
                and set(np.unique(archive['real_val_role'])) == {'DEV'}
                and not np.intersect1d(data['source_ids'], dev['source_ids']).size
                and not np.intersect1d(data['real_ids'], dev['real_ids']).size):
            raise ValueError('034 DEV original identity/role isolation differs')
    progress['DEV_metadata_archives_decoded'] += 1
    np.savez_compressed(out/'DEV_LINEAGE.npz', source_ids=dev['source_ids'], real_ids=dev['real_ids'],
        real_groups=dev['real_groups'], source_role=np.full(512, 'DEV'), real_role=np.full(599, 'DEV'))
    real_reference_flags['DEV'] = dict(rows=599, previously_exposed=True,
        saved_peak_at_search_boundary=dict(count=int(dev['real_peak_search_edge'].sum()), denominator=599),
        saved_global_peak_outside_search=dict(count=int(dev['real_global_peak_outside_search'].sum()), denominator=599),
        pre_ADC_flags_unknown=list(REAL_UNKNOWN_FLAGS), original_identities_and_roles_verified=True)
    write_json(out/'REAL_REFERENCE_FLAGS.json', real_reference_flags)
    development = evaluate034(model, A_reference, extension, dev['source_carrier384'], dev['real'], dev['real_baseline100'],
        p, loss, context, 'DEV', False, out, progress, started)
    if state_digest(A_reference.state_dict()) != old_A_digest or state_digest(extension.state_dict()) != noise_digest:
        raise RuntimeError('034 original A reference or frozen005 state changed')
    tolerances = json.loads(paths['tolerances'].read_text(encoding='utf8'))['T_by_hypothesis']
    comparison = {}
    for split, reference, endpoint in (('TRAIN', initial['A'], final['final']), ('DEV', development['A'], development['final'])):
        comparison[split] = {}
        for h in HYPOTHESES:
            av, fv = reference[h]['values'], endpoint[h]['values']
            checks = {}
            for key, value in tolerances[h].items():
                if value is None or value <= 0:
                    checks[key] = dict(status='not_calibrated')
                elif fv[key] is None or av[key] is None:
                    checks[key] = dict(status='undefined', threshold=value)
                else:
                    checks[key] = dict(status='defined', threshold=value, final_over_T=fv[key]/value,
                        final_minus_A=fv[key]-av[key], noninferior=fv[key]-av[key] <= value, absolute_within_T=fv[key] <= value)
            improvement = (av['wave']-fv['wave'])/av['wave']
            comparison[split][h] = dict(relative_wave_improvement=improvement,
                useful_2percent=improvement >= ev['relative_wave_improvement_min'],
                A_wave=av['wave'], final_wave=fv['wave'], calibrated_components=checks,
                defined_positive_calibrated_components_noninferior=all(check['noninferior'] for check in checks.values() if check['status'] == 'defined'),
                undefined_positive_calibrated_components=[key for key, check in checks.items() if check['status'] == 'undefined'],
                RR_ratios={key: fv[key]/p['RR_reference'][key] for key in ('wave', 'QPSD', 'noise')}, RR_is_threshold=False)
            if split == 'TRAIN':
                comparison[split][h]['G0_values'] = initial['G0'][h]['values']
                comparison[split][h]['final_minus_G0'] = {key: None if value is None or initial['G0'][h]['values'][key] is None
                    else value-initial['G0'][h]['values'][key] for key, value in fv.items()}
    train_wave_progress = all(value['useful_2percent'] for value in comparison['TRAIN'].values())
    component_progress = all(value['defined_positive_calibrated_components_noninferior']
                             for split in comparison.values() for value in split.values())
    np.savez_compressed(out/'TRAIN_COVERAGE.npz', source_visits=source_visits, real_visits=real_visits,
        source_ids=data['source_ids'], real_ids=data['real_ids'])
    write_json(out/'RESULT.json', dict(schema='n01/response034-real-pilot-result/v1', native_status='COMPLETE',
        G_updates=progress['G_updates'], execution_counts=dict(progress), comparisons=comparison,
        all_TRAIN_hypotheses_wave_improve_2percent=train_wave_progress,
        all_TRAIN_DEV_defined_positive_calibrated_components_noninferior=component_progress,
        pilot_relative_progress=train_wave_progress and component_progress,
        actual_coverage=dict(source_unique=int((source_visits > 0).sum()), source_total=6593,
                            real_unique=int((real_visits > 0).sum()), real_total=21047),
        DEV_previously_exposed=True, DEV_used_for_training_or_selection=False, DEV_rows=599,
        source_physical_weights='unknown', source_mixture='uniform_count', conditional_identification=False,
        noise005_frozen=True, independent_acquisition_acceptance=False, scientific_status='response_match_not_established',
        original_A_state_sha256=old_A_digest, frozen005_extension_state_sha256=noise_digest,
        final_model_state_sha256=state_digest(model.state_dict()), one_training_seed=True, one_evaluation_MC_bank=True))
    write_json(out/'COMPLETE.json', dict(native_status='COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'),
        final_checkpoint_sha256=sha256(out/'final.pt'), execution_counts=dict(progress)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256: raise ValueError('034 configuration hash differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if not (p['status'] == 'ready_for_real_pilot' and p['objective'] == 'fixed_joint_energy_from031'
            and p['seed'] == 2026101732 and p['training']['updates'] == 2000
            and p['training']['generated_bank_rows'] == 64 and p['training']['real_bank_rows'] == 128
            and p['hypotheses'] == list(HYPOTHESES)):
        raise ValueError('034 requires the ROOT-reviewed fixed2000 energy pilot config, not the draft')
    if args.out.exists(): raise FileExistsError(args.out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory
        limit = p['resources']['GPU_GiB']*1024**3
        if physical < limit: raise ValueError('034 GPU has less memory than registered allocator limit')
        torch.cuda.set_per_process_memory_fraction(limit/physical, device)
    progress = dict(G_updates=0, NN_calls=0, NN_rows=0, noise005_rows=0, extension_rows=0,
        generated_ADC_rows=0, checkpoint_loads=0, TRAIN_archives_decoded=0, DEV_archives_decoded=0,
        DEV_metadata_archives_decoded=0, training_AD_queries=0)
    try:
        run034(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error), execution_counts=progress,
                automatic_restart=False, scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
