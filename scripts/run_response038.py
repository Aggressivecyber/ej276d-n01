"""One bounded M0-preserving mean-then-random response calibration.

Only the ROOT-frozen configuration may enter the data/CP path. FIT, refreshed
TRAIN_CHECK, report-only MONITOR and unopened FINAL banks have separate keys.
Historical loaders, models and scientific outputs are never rewritten.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from n01.data import sha256, write_json
from n01.diagnose_mean_random009 import read_train_members, validate_train
from n01.inference028 import load_reference028, verified_path028
from n01.losses008 import ADC_STEP
from n01.losses014 import fit_integral_statistics014
from n01.model import ADC_MIN, ADC_MAX, ADC_LEVELS
from n01.model008 import canonical_charge008
from n01.model016 import calibration_statistics016, inverse_softplus016, Q_REFERENCE016
from n01.model015 import _ExactZeroMeanAddition015
from n01.model014 import FrozenDeterministicVoltageShape014
from n01.model038 import N01Response038, MeanProposalOutsideDomain038
from n01.observation008 import FrozenObservedNoise008, observe_existing_carrier, observe_generated_carrier
from n01.response008 import HYPOTHESES, loss_tensor_statistics, state_digest
from run_response031 import standardized_banks
from run_response036 import electronic_noise, hard_metrics, NativeMetricLoss036
from run_response029 import (make_sources, projected_moments, teacher_projections, recovery_metrics,
                            cached_sample as cached_sample029)
from wp1_evaluation import observable_features, raw_distance

ROOT = Path(__file__).resolve().parents[1]


class TrackBudget038(Exception):
    """A registered resource boundary; its unfinished direction is rolled back."""


def check_deadline(ctx):
    if time.monotonic() >= ctx['deadline']:
        raise TrackBudget038('equal900s track training allocation exhausted')


def stream_seed(seed, purpose, *indices):
    key = '/'.join(map(str, ('n01/response038/v1', seed, purpose, *indices)))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & (2**63-1)


def weighted_distance(x, y, wx, wy):
    if x.shape[0] > 128 or y.shape[0] > 128:
        distance = torch.cat([torch.linalg.vector_norm(x[start:start+64, None]-y[None, :], dim=2)
                              for start in range(0, x.shape[0], 64)], dim=0)/math.sqrt(x.shape[1])
    else:
        distance = torch.linalg.vector_norm(x[:, None]-y[None, :], dim=2)/math.sqrt(x.shape[1])
    return (distance*wx[:, None]*wy[None, :]).sum()


def weighted_energy(banks, weights):
    a, b, t, u = banks
    wa, wb, wt, wu = weights
    return .5*(weighted_distance(a, t, wa, wt)+weighted_distance(a, u, wa, wu)
               +weighted_distance(b, t, wb, wt)+weighted_distance(b, u, wb, wu)) \
        -weighted_distance(a, b, wa, wb)-weighted_distance(t, u, wt, wu)


def energy_parts038(observations, weights, loss, carrier_scale):
    """Original joint features; all cross/self pairs carry design weights."""
    banks = standardized_banks(observations, loss, carrier_scale)
    wave = torch.stack([weighted_energy([bank['views'][name] for bank in banks], weights)
                        for name in banks[0]['views']]).mean()
    integral = weighted_energy([bank['integrals'] for bank in banks], weights)
    carrier = weighted_energy([bank['carrier'] for bank in banks], weights)
    return dict(wave=wave, integral=integral, carrier_report_only=carrier, data=wave+2.*integral)


def fit_strata(source_Q, source_PC1, real_Q, real_groups):
    """Stable rank splits use TRAIN only; strata never enter a neural module."""
    source = []
    for quarter in np.array_split(np.argsort(source_Q, kind='stable'), 4):
        source.extend(np.array_split(quarter[np.argsort(source_PC1[quarter], kind='stable')], 2))
    real = []
    groups, counts = np.unique(real_groups, return_counts=True)
    position = np.floor(4*(counts.cumsum()-.5*counts)/len(real_Q)).astype(np.int64)
    for quarter in range(4):
        rows = np.concatenate([np.flatnonzero(real_groups == group) for group in groups[position == quarter]])
        real.extend(np.array_split(rows[np.argsort(real_Q[rows], kind='stable')], 4))
    return source, real


def stratified_draw(strata, quota, rng):
    count = sum(map(len, strata))
    rows = np.concatenate([rng.choice(layer, quota, replace=True) for layer in strata])
    weights = np.concatenate([np.full(quota, len(layer)/(count*quota)) for layer in strata])
    return rows, weights


def fit_coverage(source_banks, source_population):
    rows = np.concatenate([bank[key] for bank in source_banks for key in ('source_A', 'source_B')])
    weights = np.concatenate([bank['weights'][lane].detach().cpu().numpy()
                              for bank in source_banks for lane in (0, 1)])/(2*len(source_banks))
    parent_weights = np.bincount(rows, weights=weights, minlength=source_population)
    return dict(source_draws=len(rows), source_unique=int(np.count_nonzero(parent_weights)),
                source_population=source_population, draw_Kish_ESS=float(1/np.square(weights).sum()),
                parent_aggregated_Kish_ESS=float(1/np.square(parent_weights).sum()),
                count_weighted_population='all legal TRAIN SOURCE rows',
                all_parents_have_nonzero_design_probability=True)


def frozen_electronics(count, noise_generator, extension_generator, extension, device, progress):
    progress['noise005_calls'] += 1
    progress['noise005_rows_attempted'] += count
    return electronic_noise(extension, count, noise_generator, extension_generator, device, progress)


@torch.no_grad()
def make_bank(seed, purpose, indices, source_strata, real_strata, extension, device,
              progress, *, stratified, teacher_factors=None, explicit_rows=None):
    """Allocate once; old/proposal/reference reuse these exact rows and latents."""
    base = stream_seed(seed, purpose, *indices)
    bank = dict(key=dict(seed=seed, purpose=purpose, indices=list(indices)), noise={})
    weights = []
    for lane, strata, quota in (('source_A', source_strata, 8), ('source_B', source_strata, 8),
                                ('real_T', real_strata, 8), ('real_U', real_strata, 8)):
        rng = np.random.default_rng(stream_seed(base, lane))
        if explicit_rows is not None:
            rows = explicit_rows[lane]
            weight = np.full(len(rows), 1/len(rows))
        elif stratified:
            rows, weight = stratified_draw(strata, quota, rng)
        else:
            count = 64 if lane.startswith('source') else 128
            rows = rng.integers(sum(map(len, strata)), size=count)
            weight = np.full(count, 1/count)
        bank[lane] = rows
        weights.append(torch.as_tensor(weight, dtype=torch.float64, device=device))
    bank['weights'] = tuple(weights)
    for lane in ('A', 'B'):
        count = len(bank['source_'+lane])
        shape_rng = torch.Generator(device='cpu').manual_seed(stream_seed(base, 'shape_'+lane))
        charge_rng = torch.Generator(device='cpu').manual_seed(stream_seed(base, 'charge_'+lane))
        bank['shape_'+lane] = torch.randn(count, 8, generator=shape_rng, dtype=torch.float32).to(device)
        bank['charge_'+lane] = torch.randn(count, generator=charge_rng, dtype=torch.float64).to(device)
        noise_rng = torch.Generator(device='cpu').manual_seed(stream_seed(base, 'noise005_'+lane))
        extension_rng = torch.Generator(device='cpu').manual_seed(stream_seed(base, 'extension_'+lane))
        bank['noise'][lane] = frozen_electronics(count, noise_rng, extension_rng, extension, device, progress)
    if teacher_factors is not None:
        bank['targets'] = {h: [] for h in HYPOTHESES}
        for lane in ('T', 'U'):
            rows = torch.as_tensor(bank['real_'+lane], device=device)
            shape_rng = torch.Generator().manual_seed(stream_seed(base, 'teacher_shape_'+lane))
            charge_rng = torch.Generator().manual_seed(stream_seed(base, 'teacher_charge_'+lane))
            shape = torch.randn(len(rows), 8, generator=shape_rng).to(device)
            charge = torch.randn(len(rows), generator=charge_rng, dtype=torch.float64).to(device)
            signal = cached_signal038(teacher_factors, rows, shape, charge)
            progress['teacher_cached_signal_rows'] += len(rows)
            noise_rng = torch.Generator().manual_seed(stream_seed(base, 'teacher_noise_'+lane))
            extension_rng = torch.Generator().manual_seed(stream_seed(base, 'teacher_extension_'+lane))
            noise = frozen_electronics(len(rows), noise_rng, extension_rng, extension, device, progress)
            for h in HYPOTHESES:
                bank['targets'][h].append(observe_generated_carrier(signal+noise[h], ste=False,
                    q_floor=teacher_factors['q_floor'], return_carrier=True, return_indices=True))
                progress['generated_observation_rows'] += len(rows)
    progress['allocated_banks'] += 1
    return bank


def score_bank(model, encoding, real, bank, loss, carrier_scale, *, ste, charge_enabled,
               progress, return_outputs=False):
    generated, outputs, regulars, saturations = [], [], [], []
    for lane, weight in zip(('A', 'B'), bank['weights'][:2]):
        indices = torch.as_tensor(bank['source_'+lane], device=encoding['source'].device)
        cache = model.select_encoding(encoding, indices)
        progress['response_calls'] += 1
        progress['response_rows'] += len(indices)
        output = model(cache['source'], bank['shape_'+lane], bank['charge_'+lane], cache=cache,
                       shape_enabled=True, mean_enabled=True, charge_enabled=charge_enabled)
        observations, saturation = {}, []
        for h in HYPOTHESES:
            pre = output['signal_pre']+bank['noise'][lane][h]
            observations[h] = observe_generated_carrier(pre, ste=ste, q_floor=float(loss.q_floor),
                                                       return_carrier=True, return_indices=True)
            excess = (F.relu(pre-ADC_MAX)+F.relu(ADC_MIN-pre))/(ADC_MAX-ADC_MIN)
            saturation.append((weight*excess.square().mean(1)).sum())
            progress['generated_observation_rows'] += len(indices)
        generated.append(observations)
        outputs.append(output)
        regulars.append((weight*output['regular_per_row']).sum())
        saturations.append(torch.stack(saturation).mean())
    targets = ({h: [{key: value[bank[lane]] for key, value in real.items()} for lane in ('real_T', 'real_U')]
                for h in HYPOTHESES} if 'targets' not in bank else bank['targets'])
    by_hypothesis = {h: energy_parts038([generated[0][h], generated[1][h], *targets[h]],
                                      bank['weights'], loss, carrier_scale) for h in HYPOTHESES}
    means = {key: torch.stack([value[key] for value in by_hypothesis.values()]).mean()
             for key in ('wave', 'integral', 'carrier_report_only', 'data')}
    means.update(regular=torch.stack(regulars).mean(), saturation=torch.stack(saturations).mean(),
                 by_hypothesis=by_hypothesis)
    if return_outputs:
        means.update(outputs=outputs, observations=generated)
    return means


def transaction_snapshot(model, optimizer, mutable, loss):
    """Frozen A/005 are protected references, never rewritten by rollback."""
    model.assert_frozen_unchanged()
    return dict(model={name: value.detach().clone() for name, value in model.state_dict().items()
                       if not name.startswith('baseline.')}, optimizer=copy.deepcopy(optimizer.state_dict()),
                modes={name: module.training for name, module in model.named_modules()
                       if not (name == 'baseline' or name.startswith('baseline.'))},
                requires_grad={name: value.requires_grad for name, value in model.named_parameters()
                               if not name.startswith('baseline.')},
                phase=model.phase, train_charge=model.train_charge, mutable=copy.deepcopy(mutable),
                loss_mutable={name: copy.deepcopy(getattr(loss, name)) for name in
                              ('last_diagnostics', 'last_wave_components', 'last_integral_diagnostics')},
                CPU_RNG=torch.get_rng_state().clone(),
                CUDA_RNG=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


@torch.no_grad()
def restore_transaction(model, optimizer, mutable, loss, snapshot):
    model.assert_frozen_unchanged()
    live = model.state_dict(keep_vars=True)
    for name, value in snapshot['model'].items():
        if not torch.equal(live[name], value):
            live[name].copy_(value)
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    for name, module in model.named_modules():
        if name in snapshot['modes']:
            module.training = snapshot['modes'][name]
    for name, parameter in model.named_parameters():
        if name in snapshot['requires_grad']:
            parameter.requires_grad_(snapshot['requires_grad'][name])
            parameter.grad = None
    model.phase, model.train_charge = snapshot['phase'], snapshot['train_charge']
    mutable.clear()
    mutable.update(copy.deepcopy(snapshot['mutable']))
    for name, value in snapshot['loss_mutable'].items():
        setattr(loss, name, copy.deepcopy(value))
    torch.set_rng_state(snapshot['CPU_RNG'])
    if snapshot['CUDA_RNG'] is not None:
        torch.cuda.set_rng_state_all(snapshot['CUDA_RNG'])


def adam_direction(model, optimizer, snapshot, progress):
    """Exactly one Adam step creates a direction and a tentative new state."""
    optimizer.step()
    progress['optimizer_steps'] += 1
    delta = {name: value.detach()-snapshot['model'][name]
             for name, value in model.named_parameters() if value.requires_grad}
    return delta, copy.deepcopy(optimizer.state_dict())


@torch.no_grad()
def apply_trial(model, snapshot, delta, scale, progress):
    live = dict(model.named_parameters())
    for name, difference in delta.items():
        live[name].copy_(snapshot['model'][name]+scale*difference)
    progress['parameter_amplitude_trials'] += 1


def fit_direction(model, optimizer, encoding, real, banks, loss, carrier_scale, p,
                  *, charge_enabled, progress, record_microgradients, deadline, source_visits, target_visits):
    """Accumulate independent microbanks at one theta; no optimizer here."""
    parameters = model.trainable_parameters()
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    gradients = []
    values = {name: [] for name in ('data', 'wave', 'integral', 'carrier_report_only', 'regular', 'saturation', 'total')}
    for bank in banks:
        if time.monotonic() >= deadline:
            raise TrackBudget038('track allocation exhausted at FIT microbank boundary')
        optimizer.zero_grad(set_to_none=True)
        result = score_bank(model, encoding, real, bank, loss, carrier_scale, ste=True,
                            charge_enabled=charge_enabled, progress=progress)
        objective = result['data']+p['training']['regular_weight']*(result['regular']+result['saturation'])
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError('038 FIT objective is nonfinite')
        objective.backward()
        progress['FIT_AD_queries'] += 1
        for lane in ('source_A', 'source_B'):
            np.add.at(source_visits, bank[lane], 1)
        for lane in ('real_T', 'real_U'):
            np.add.at(target_visits, bank[lane], 1)
        if record_microgradients:
            gradients.append(torch.cat([parameter.grad.detach().reshape(-1).double().cpu() for parameter in parameters]).numpy())
        for total, parameter in zip(accumulated, parameters):
            total.add_(parameter.grad, alpha=1/len(banks))
        for name in values:
            values[name].append(float(objective.detach() if name == 'total' else result[name].detach()))
    for parameter, gradient in zip(parameters, accumulated):
        parameter.grad = gradient
    if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters if parameter.grad is not None):
        raise FloatingPointError('038 FIT gradient is nonfinite')
    norm = torch.nn.utils.clip_grad_norm_(parameters, p['training']['clip_norm'], error_if_nonfinite=True)
    result = dict(means={name: float(np.mean(value)) for name, value in values.items()},
                per_microbank=values, gradient_norm_before_clip=float(norm),
                gradient_clip_applied=bool(norm > p['training']['clip_norm']),
                coverage=fit_coverage(banks, len(encoding['source'])),
                gradient_theta='same accepted pre-proposal state for every microbank')
    if record_microgradients:
        matrix = np.stack(gradients)
        result['microgradients'] = matrix
        result['block_gradient_variance'] = {str(size): dict(block_size=size, blocks=len(matrix)//size,
            empirical_covariance_trace=float(matrix.reshape(-1, size, matrix.shape[1]).mean(1).var(0, ddof=1).sum()))
            for size in (1, 2, 4, 8, 16)}
    return result


def estimate(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), MCSE=float(values.std(ddof=1)/math.sqrt(len(values))),
                banks=len(values), MCSE_scope='independent simulation banks; not experimental uncertainty')


@torch.no_grad()
def encode_all(model, source, chunk_rows, progress):
    """One frozen-A encoding per row, reused across directions and controls."""
    chunks = []
    for start in range(0, len(source), chunk_rows):
        progress['old_A_calls'] += 1
        progress['old_A_rows'] += len(source[start:start+chunk_rows])
        part = model.encode_source(source[start:start+chunk_rows])
        chunks.append(part)
    original = {}
    for name, first in chunks[0]['original'].items():
        values = [chunk['original'][name] for chunk in chunks]
        if isinstance(first, torch.Tensor) and first.ndim and first.shape[0] == len(chunks[0]['source']):
            original[name] = torch.cat(values)
        elif isinstance(first, torch.Tensor) and first.ndim == 0:
            original[name] = sum(value*len(chunk['source']) for value, chunk in zip(values, chunks))/len(source)
        else:
            original[name] = first
    return dict(source=source, original=original, decoded=torch.cat([chunk['decoded'] for chunk in chunks]),
                condition=torch.cat([chunk['condition'] for chunk in chunks]))


@torch.no_grad()
def factors038(model, encoding, charge_enabled, progress):
    rows, device = len(encoding['source']), encoding['source'].device
    progress['response_calls'] += 1
    progress['response_rows'] += rows
    output = model(encoding['source'], torch.zeros(rows, 8, device=device),
                   torch.zeros(rows, dtype=torch.float64, device=device), cache=encoding,
                   charge_enabled=charge_enabled)
    return dict(A=output['moment_mean_pre_V'], A0=output['A0_signal_pre_V'],
        delta=output['delta_mean_pre_V'], B=output['moment_basis_V'],
        cv2=output['moment_charge_cv2'], v=torch.cat((torch.log1p(output['moment_charge_cv2']).sqrt()[:, None],
            torch.zeros(rows, 8, device=device, dtype=torch.float64)), 1),
        envelope=torch.zeros_like(output['moment_mean_pre_V']), taps=output['signal_pre'].new_tensor([1.]+[0.]*16),
        gate=model.baseline.gate[0, 0], projection=model.baseline.projection[0, 0], q_floor=None)


def cached_signal038(factors, rows, shape, charge):
    M0 = factors['A0'][rows]+(factors['B'][rows]*shape.double()[..., None]).sum(1)
    mean_response = _ExactZeroMeanAddition015.apply(M0, factors['delta'][rows])
    logvar = torch.log1p(factors['cv2'][rows])
    return mean_response+factors['A'][rows]*torch.expm1(logvar.sqrt()*charge.double()-.5*logvar)[:, None]


def make_teacher_context(setup, p, device, progress):
    train, heldout, ignored_stats = make_sources(dict(source=p['teacher']['source']))
    teacher = N01Response038(setup['baseline'], setup['heads'][0], seed=p['teacher']['seed'],
        condition_state=setup['condition'], charge_initialization=setup['calibration'])
    train['source'] = train['source'].to(device)
    heldout['source'] = heldout['source'].to(device)
    encoding = encode_all(teacher, train['source'], p['evaluation']['NN_chunk_rows'], progress)
    held_encoding = encode_all(teacher, heldout['source'], p['evaluation']['NN_chunk_rows'], progress)
    calibration = calibration_statistics016(encoding['original']['signal_pre'], adc_step=.25*ADC_STEP)
    with torch.no_grad():
        teacher.mean_adapter[-1].weight.zero_()
        teacher.mean_adapter[-1].bias.copy_(teacher.mean_adapter[-1].bias.new_tensor(p['teacher']['mean_last_bias']))
        for parameter in teacher.basis_head.parameters():
            parameter.mul_(1.15)
        teacher.theta_a.copy_(inverse_softplus016(2*calibration['a_initial_Vns']/Q_REFERENCE016))
        teacher.theta_b.copy_(inverse_softplus016(2*calibration['b_initial']))
    teacher.set_phase('frozen').eval()
    truth = factors038(teacher, encoding, True, progress)
    held_truth = factors038(teacher, held_encoding, True, progress)
    truth['q_floor'] = held_truth['q_floor'] = float(setup['loss'].q_floor)
    Q = train['Q'].numpy()
    centered = train['source']-train['source'][:, :100].mean(1, keepdim=True)
    pc = (((centered/canonical_charge008(centered)[:, None]-teacher.condition_mean)
            *teacher.condition_axis).sum(1)/teacher.condition_scale).cpu().numpy()
    source_strata, target_strata = fit_strata(Q, pc, Q, np.arange(len(Q)))
    learners = [N01Response038(setup['baseline'], setup['heads'][index], seed=seed,
        condition_state=setup['condition'], charge_initialization=calibration) for index, seed in enumerate(p['seeds'])]
    return dict(train=train, heldout=heldout, encoding=encoding, held_encoding=held_encoding,
        teacher=teacher, factors=truth, held_factors=held_truth, models=learners, calibration=calibration,
        source_strata=source_strata, target_strata=target_strata,
        target_stratification='128 finite source parents treated as ordered singleton groups; nominal source Q; count weights',
        conditional_supervision_used=False)


def load_setup(p, device, progress):
    """Reuse028's six strict CP bindings, then construct wrappers on final device."""
    paths = {name: verified_path028(record) for name, record in p['inputs'].items()}
    reference = load_reference028(verified_path028(p['reference028']))
    progress['checkpoint_loads'] += reference.accounting['completed']['CP_loads']
    statistics, domain, base = [json.loads(paths[name].read_text(encoding='utf8'))
                                for name in ('statistics', 'data_statistics', 'base_config')]
    if not (statistics['statistics_split'] == domain['split'] == statistics['loss']['fit_split'] == 'TRAIN'
            and statistics['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']
            and statistics['data_statistics_sha256'] == p['inputs']['data_statistics']['sha256']
            and statistics['DEV_waveforms_decoded'] is False and domain['DEV_waveforms_decoded'] is False):
        raise ValueError('038 original TRAIN binding differs')
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
        data['real_carrier384'] = archive['real_carrier384'].copy()
    progress['TRAIN_archives_decoded'] += 1
    validate_train(data)
    if 'original_source_weights' in data and not np.array_equal(data['original_source_weights'], np.ones(6593)):
        raise ValueError('038 count-population SOURCE weights differ')
    PCA = p['PCA013']
    with np.load(verified_path028(PCA), allow_pickle=False) as archive:
        condition = dict(mean=archive[PCA['mean_key']].copy(),
                         axis=archive[PCA['components_key']][:, 0].copy(), scale=PCA['PC1_scale'])
    progress['PCA_archives_decoded'] += 1
    # The protected028 wrappers remain on CPU; do not move a layout-locked014.
    baseline = copy.deepcopy(reference.models[0].baseline).to(device)
    baseline.eval().requires_grad_(False)
    heads = [{key: value.detach().clone() for key, value in reference.models[index].basis_head.state_dict().items()}
             for index in (0, 1)]
    Q1 = [dict(theta_a_initial=float(reference.models[index].theta_a),
               theta_b_initial=float(reference.models[index].theta_b)) for index in (0, 1)]
    extension = FrozenObservedNoise008(copy.deepcopy(reference.extension.model005).to(device)).eval().requires_grad_(False)
    references = [N01Response038(baseline, heads[index], seed=reference.config['model_initialization_seeds'][index],
                            condition_state=condition, charge_initialization=Q1[index]) for index in (0, 1)]
    for model in references:
        model.set_phase('frozen').eval()
    models = [N01Response038(baseline, heads[index], seed=p['seeds'][index],
                            condition_state=condition, charge_initialization=Q1[index]) for index in (0, 1)]
    del reference
    source = torch.as_tensor(data['source_carrier384'], dtype=torch.float64, device=device)
    encoding = encode_all(references[0], source, p['evaluation']['NN_chunk_rows'], progress)
    calibration = calibration_statistics016(encoding['original']['signal_pre'], adc_step=.25*ADC_STEP)
    with torch.no_grad():
        for model in models:
            model.theta_a.fill_(calibration['theta_a_initial'])
            model.theta_b.fill_(calibration['theta_b_initial'])
    observed = observe_existing_carrier(data['real_carrier384'], already_digitized=True,
                                       q_floor=float(statistics['loss']['q_floor']), return_carrier=True)
    if not (np.array_equal(observed['window_V'], data['real'])
            and np.array_equal(observed['baseline100'], data['real_baseline100'])):
        raise ValueError('038 REAL replay differs from prepared TRAIN; no second ADC is allowed')
    integral = fit_integral_statistics014(torch.as_tensor(data['real'], dtype=torch.float64))
    centered = torch.as_tensor(observed['carrier_adc']-observed['baseline_mean_V'][:, None], dtype=torch.float64)
    carrier_scale = centered.square().mean(0).sqrt().clamp_min(ADC_STEP).to(device)
    real = {name: torch.as_tensor(observed[name], dtype=torch.float64, device=device)
            for name in ('window_V', 'baseline100', 'carrier_adc', 'baseline_mean_V')}
    ev = p['evaluation']
    loss = NativeMetricLoss036(loss_tensor_statistics(statistics['loss']), integral,
        seed=ev['projection_training_seed'], evaluation_seed=ev['projection_evaluation_seed'],
        integral_seed=ev['integral_seed']).to(device)
    loss.column_chunk = ev['W1_column_chunk']
    loss.last_diagnostics = {}
    context = loss.create_evaluation_context()
    if context.metadata()['fingerprint'] != ev['projection_fingerprint']:
        raise ValueError('038 original native evaluation directions differ')
    centered_source = source-source[:, :100].mean(1, keepdim=True)
    source_Q_tensor = canonical_charge008(centered_source)
    source_Q = source_Q_tensor.cpu().numpy()
    source_PC1 = (((centered_source/source_Q_tensor[:, None]
                   -torch.as_tensor(condition['mean'], device=device))
                  *torch.as_tensor(condition['axis'], device=device)).sum(1)/condition['scale']).cpu().numpy()
    real_Q = data['real'].sum(1)
    source_strata, real_strata = fit_strata(source_Q, source_PC1, real_Q, data['real_groups'])
    noise_floor = float(np.median(data['real_baseline100'].std(1)))
    return dict(paths=paths, statistics=statistics, base=base, data=data, source=source, encoding=encoding,
                models=models, references=references, heads=heads, Q1=Q1, condition=condition, calibration=calibration,
                baseline=baseline, extension=extension, noise_RMS_floor_V=noise_floor,
                real=real, loss=loss, context=context, integral=integral, carrier_scale=carrier_scale,
                real_reference_flags={name: dict(count=int(value.sum()), denominator=len(data['real']))
                                      for name, value in observed['flags'].items()},
                source_strata=source_strata, real_strata=real_strata,
                frozen_A_digest=state_digest(baseline.state_dict()),
                frozen005_digest=state_digest(extension.state_dict()))


@torch.no_grad()
def pooled_native(results, real, banks, loss, context):
    reports = {}
    for h in HYPOTHESES:
        targets = [target for bank in banks for target in (bank['targets'][h] if 'targets' in bank else [
            {key: value[bank[lane]] for key, value in real.items()} for lane in ('real_T', 'real_U')])]
        target_wave = torch.cat([target['window_V'] for target in targets]).cpu().numpy()
        target_baseline = torch.cat([target['baseline100'] for target in targets]).cpu().numpy()
        observations = [obs[h] for result in results for obs in result['observations']]
        wave = torch.cat([obs['window_V'] for obs in observations]).cpu().numpy()
        baseline = torch.cat([obs['baseline100'] for obs in observations]).cpu().numpy()
        reports[h] = hard_metrics(wave, baseline, target_wave, target_baseline, loss, context)
        reports[h]['values']['negative_lobe_area_Vns'] = raw_distance(np.maximum(-wave, 0).sum(1),
                                                                      np.maximum(-target_wave, 0).sum(1))['value']
        reports[h]['original_observation_flags'] = {
            name: dict(count=int(sum(obs['flags'][name].sum() for obs in observations)),
                       denominator=len(wave)) for name in observations[0]['flags']}
        reports[h]['pooling'] = dict(banks=len(banks), fake_rows=len(wave), target_rows=len(target_wave),
                                    native_not_mean_of_microbank_W1=True)
    return reports


def risk_record(result, regular_weight):
    values = {name: float(result[name]) for name in ('wave', 'integral', 'carrier_report_only', 'data', 'regular', 'saturation')}
    values['total'] = values['data']+regular_weight*(values['regular']+values['saturation'])
    values['by_hypothesis'] = {
        h: {name: float(value) for name, value in result['by_hypothesis'][h].items()} for h in HYPOTHESES}
    return values


def compact_native(reports):
    keys = ('values', 'wave_components', 'fake_rows', 'real_rows', 'fake_flags', 'real_flags',
            'original_observation_flags', 'pooling', 'conditional_coverage')
    return {h: {name: value[name] for name in keys if name in value} for h, value in reports.items()}


def paired_risk(candidate, reference):
    """Pairs are bank CRN contrasts; an undefined metric never becomes a pass."""
    result = {name: estimate([a[name]-b[name] for a, b in zip(candidate, reference)])
              for name in ('data', 'wave', 'integral', 'carrier_report_only', 'regular', 'saturation', 'total')}
    result['by_hypothesis'] = {}
    for h in HYPOTHESES:
        a, b = [row['by_hypothesis'][h] for row in candidate], [row['by_hypothesis'][h] for row in reference]
        parts = {name: estimate([x[name]-y[name] for x, y in zip(a, b)])
                 for name in ('data', 'wave', 'integral', 'carrier_report_only')}
        result['by_hypothesis'][h] = parts
    return result


def acceptance(candidate, current, M0, native_candidate, native_current, native_M0, tolerances,
               *, activation=False, cumulative=False):
    comparison = paired_risk(candidate, M0 if cumulative else current)
    energy, native, unresolved_domains = {}, {}, {}
    energy_ok, energy_resolved_bad, native_ok = True, False, True
    for h in HYPOTHESES:
        energy[h] = {}
        for name in ('data', 'wave'):
            value = comparison['by_hypothesis'][h][name]
            upper, lower = value['mean']+3*value['MCSE'], value['mean']-3*value['MCSE']
            allowance = .001*max(abs(np.mean([row['by_hypothesis'][h][name] for row in current])), 1e-4) if activation else 0.
            passed = upper <= allowance if activation or name == 'wave' else upper < 0.
            energy[h][name] = dict(**value, upper3MCSE=upper, lower3MCSE=lower, allowance=allowance, passed=passed)
            energy_ok = energy_ok and passed
            energy_resolved_bad = energy_resolved_bad or lower > allowance
        C, O, B = [reports[h] for reports in (native_candidate, native_current, native_M0)]
        cv, ov, bv = C['values'], O['values'], B['values']
        checks = dict(wave_current=(cv['wave']-ov['wave'] <= (.001 if activation else .002)*ov['wave']),
                      wave_starting_M0=(cv['wave']-bv['wave'] <= .005*bv['wave']))
        unresolved_domains[h] = []
        for name, threshold in tolerances[h].items():
            if name == 'wave':
                continue
            if threshold is None or threshold <= 0:
                unresolved_domains[h].append(name)
            elif any(value[name] is None for value in (cv, ov, bv)):
                checks[name+'_defined'] = False
            else:
                checks[name+'_current'] = cv[name]-ov[name] <= .01*threshold
                checks[name+'_starting_M0'] = cv[name]-bv[name] <= .05*threshold
        checks['negative_area_current'] = cv['negative_lobe_area_Vns']-ov['negative_lobe_area_Vns'] <= .1*ADC_STEP*16
        checks['negative_area_M0'] = cv['negative_lobe_area_Vns']-bv['negative_lobe_area_Vns'] <= ADC_STEP*16
        allowance = max(1/C['fake_rows'], .001)
        for family in ('fake_flags', 'original_observation_flags'):
            for name, item in C[family].items():
                denominator = item['denominator']
                count_key = 'count'
                rate = item[count_key]/denominator
                for label, reference in (('current', O), ('starting_M0', B)):
                    ref = reference[family][name]
                    checks[family+':'+name+':'+label] = rate-ref[count_key]/ref['denominator'] <= allowance
        native[h] = dict(checks=checks, passed=all(checks.values()), anomaly_extra_rate_allowance=allowance,
                         fake_rows=C['fake_rows'], target_rows=C['real_rows'])
        native_ok = native_ok and native[h]['passed']
    return dict(accepted=bool(energy_ok and native_ok),
                unresolved=bool(native_ok and not energy_ok and not energy_resolved_bad),
                energy=energy, native=native, paired=comparison, uncalibrated_domains=unresolved_domains,
                scope='internal TRAIN selection; repeated3MCSE is not simultaneous confidence')


def TR_pass(geometry, radius_LSB, *, random_phase):
    radius = radius_LSB*ADC_STEP
    checks = dict(RMS=geometry['signal_RMS_active_V'] <= radius,
        event99=geometry['event_RMS_p99_V'] <= 2*radius, eventmax=geometry['event_RMS_max_V'] <= 4*radius,
        coordinate=geometry['coordinate_absmax_V'] <= 16*radius,
        relative99=geometry['relative_event_RMS_p99'] <= .02, relativeMax=geometry['relative_event_RMS_max'] <= .05,
        tail=geometry['relative_tail_integral_absmax'] <= .01)
    if random_phase:
        checks.update(covariance99=geometry['preADC_covariance_relative_event_p99'] <= .05,
                      covarianceMax=geometry['preADC_covariance_relative_event_max'] <= .10)
    return dict(passed=all(checks.values()), checks=checks, radius_LSB=radius_LSB)


@torch.no_grad()
def score_library(model, ctx, banks, charge_enabled, progress):
    results = []
    for bank in banks:
        check_deadline(ctx)
        results.append(score_bank(model, ctx['encoding'], ctx['real'], bank, ctx['loss'], ctx['carrier_scale'],
            ste=False, charge_enabled=charge_enabled, progress=progress, return_outputs=True))
    return results


@torch.no_grad()
def pure_probe(model, ctx, banks, charge_enabled, progress):
    outputs = []
    for bank in banks[:4]:
        check_deadline(ctx)
        for lane in ('A', 'B'):
            rows = torch.as_tensor(bank['source_'+lane], device=ctx['encoding']['source'].device)
            cache = model.select_encoding(ctx['encoding'], rows)
            progress['response_calls'] += 1
            progress['response_rows'] += len(rows)
            outputs.append(model(cache['source'], bank['shape_'+lane], bank['charge_'+lane], cache=cache,
                                 charge_enabled=charge_enabled))
    return outputs


def joined_geometry(old_outputs, proposed_outputs, noise_floor):
    keys = ('signal_pre', 'A0_signal_pre_V', 'moment_mean_pre_V', 'moment_basis_V', 'moment_charge_cv2',
            'charge_Q_variance', 'normalization_gain', 'warp_Q_ratio')
    old, proposed = [{key: torch.cat([output[key] for output in outputs]) for key in keys}
                     for outputs in (old_outputs, proposed_outputs)]
    return trust_geometry(old, proposed, noise_RMS_floor_V=noise_floor,
                          tail_integral_floor_Vns=ADC_STEP*math.sqrt(228), tail_start=156)


def allocate_banks(ctx, seed, purpose, indices, start, count, progress, *, stratified=False):
    return [make_bank(seed, purpose, (*indices, index), ctx['source_strata'], ctx['target_strata'],
                     ctx['extension'], ctx['encoding']['source'].device, progress, stratified=stratified,
                     teacher_factors=ctx['teacher_factors']) for index in range(start, start+count)]


def library_decision(model, ctx, banks, old_results, M0_results, candidate_results, current_charge,
                     progress, p, *, activation=False, cumulative=False):
    check_deadline(ctx)
    native_current = pooled_native(old_results, ctx['real'], banks, ctx['loss'], ctx['context'])
    native_M0 = pooled_native(M0_results, ctx['real'], banks, ctx['loss'], ctx['context'])
    native_candidate = pooled_native(candidate_results, ctx['real'], banks, ctx['loss'], ctx['context'])
    records = [[risk_record(result, p['training']['regular_weight']) for result in results]
               for results in (candidate_results, old_results, M0_results)]
    decision = acceptance(*records, native_candidate, native_current, native_M0, ctx['tolerances'],
                          activation=activation, cumulative=cumulative)
    return decision, dict(candidate=records[0], current=records[1], M0=records[2],
                          pooled_native_candidate=native_candidate, pooled_native_current=native_current,
                          pooled_native_M0=native_M0)


def new_module_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith('baseline.')}


def save_state(path, model, optimizer, mutable, progress, p):
    from n01.train import atomic_torch_save
    atomic_torch_save(dict(schema='n01/response038-new-module-checkpoint/v1', config=p,
        new_modules=new_module_state(model), optimizer=optimizer.state_dict(), mutable=copy.deepcopy(mutable),
        CPU_RNG=torch.get_rng_state(), CUDA_RNG=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        immutable_A=p['inputs']['deterministic'], immutable_M0=p['inputs'][f'M0_head_seed{mutable["reference_index"]}'],
        immutable005=p['inputs']['noise005'], execution_counts=dict(progress), resume_allowed=False), path)


@torch.no_grad()
def copy_new_state(model, state):
    live = model.state_dict(keep_vars=True)
    for name, value in state.items():
        live[name].copy_(value.to(live[name]))


def phase_endpoint(model, ctx, seed, track, progress, p, out):
    banks = allocate_banks(ctx, seed, 'PHASE_ENDPOINT', (track,), 0, 8, progress)
    candidate = score_library(model, ctx, banks, False, progress)
    M0 = score_library(ctx['M0'], ctx, banks, False, progress)
    decision, details = library_decision(model, ctx, banks, candidate, M0, candidate, False,
                                        progress, p, cumulative=True)
    if decision['unresolved']:
        extra = allocate_banks(ctx, seed, 'PHASE_ENDPOINT', (track,), 8, 8, progress)
        candidate += score_library(model, ctx, extra, False, progress)
        M0 += score_library(ctx['M0'], ctx, extra, False, progress)
        banks += extra
        decision, details = library_decision(model, ctx, banks, candidate, M0, candidate, False,
                                            progress, p, cumulative=True)
    write_json(out/'MEAN_ENDPOINT_CHECK.json', dict(decision=decision, details=details,
        bank_keys=[bank['key'] for bank in banks], fresh_endpoint_check=True))
    return decision


def charge_activation(model, optimizer, ctx, state, seed, track, progress, p, out):
    snapshot = transaction_snapshot(model, optimizer, state, ctx['loss'])
    banks = allocate_banks(ctx, seed, 'CHARGE_ACTIVATION', (track,), 0, 8, progress)
    old = score_library(model, ctx, banks, False, progress)
    M0 = score_library(ctx['M0'], ctx, banks, False, progress)
    progress['charge_activation_trials'] += 1
    proposed = pure_probe(model, ctx, banks, True, progress)
    geometry = joined_geometry([output for result in old[:4] for output in result['outputs']],
                               proposed, ctx['noise_RMS_floor_V'])
    trust = TR_pass(geometry, state['radius_LSB'], random_phase=True)
    decision, details = None, None
    if trust['passed']:
        candidate = score_library(model, ctx, banks, True, progress)
        full_geometry = joined_geometry([output for row in old for output in row['outputs']],
            [output for row in candidate for output in row['outputs']], ctx['noise_RMS_floor_V'])
        trust = TR_pass(full_geometry, state['radius_LSB'], random_phase=True)
        geometry = full_geometry
        if not trust['passed']:
            restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
            model.set_phase('random', train_charge=False)
            write_json(out/'CHARGE_ACTIVATION.json', dict(enabled=False, geometry=full_geometry, trust=trust,
                decision='full_CHECK_TR_reject', optimizer_steps=0))
            return False
        decision, details = library_decision(model, ctx, banks, old, M0, candidate, False,
                                            progress, p, activation=True)
        if decision['unresolved']:
            extra = allocate_banks(ctx, seed, 'CHARGE_ACTIVATION', (track,), 8, 8, progress)
            old += score_library(model, ctx, extra, False, progress)
            M0 += score_library(ctx['M0'], ctx, extra, False, progress)
            candidate += score_library(model, ctx, extra, True, progress)
            banks += extra
            full_geometry = joined_geometry([output for row in old for output in row['outputs']],
                [output for row in candidate for output in row['outputs']], ctx['noise_RMS_floor_V'])
            trust = TR_pass(full_geometry, state['radius_LSB'], random_phase=True)
            geometry = full_geometry
            decision, details = library_decision(model, ctx, banks, old, M0, candidate, False,
                                                progress, p, activation=True)
            if not trust['passed']:
                decision['accepted'] = False
                decision['full_CHECK_TR_rejected'] = True
    enabled = bool(decision is not None and decision['accepted'])
    restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
    state['charge_enabled'] = enabled
    model.set_phase('random', train_charge=enabled)
    write_json(out/'CHARGE_ACTIVATION.json', dict(enabled=enabled, geometry=geometry, trust=trust,
        decision=decision, details=details, one_fixed_nonzero_initialization=True,
        bank_keys=[bank['key'] for bank in banks], optimizer_steps=0))
    return enabled


def make_monitor(ctx, seed, progress):
    nsource = len(ctx['encoding']['source'])
    rng = np.random.default_rng(stream_seed(seed, 'MONITOR_rows'))
    parents = np.arange(nsource) if ctx['teacher_factors'] is not None else rng.choice(nsource, 512, replace=False)
    repeats = 2 if ctx['teacher_factors'] is not None else 1
    target_rows = 256 if ctx['teacher_factors'] is not None else 1024
    explicit = dict(source_A=np.repeat(parents, repeats), source_B=np.repeat(parents, repeats),
        real_T=rng.integers(ctx['target_population'], size=target_rows),
        real_U=rng.integers(ctx['target_population'], size=target_rows))
    return make_bank(seed, 'MONITOR', (), ctx['source_strata'], ctx['target_strata'], ctx['extension'],
        ctx['encoding']['source'].device, progress, stratified=False, teacher_factors=ctx['teacher_factors'],
        explicit_rows=explicit)


@torch.no_grad()
def monitor(model, ctx, bank, state, progress, p, log):
    was_training = model.training
    model.eval()
    results = score_library(model, ctx, [bank], state['charge_enabled'], progress)
    native = pooled_native(results, ctx['real'], [bank], ctx['loss'], ctx['context'])
    record = dict(index=state['monitor_index'], phase=state['phase'], accepted_updates=state['accepted_updates'],
        risk=risk_record(results[0], p['training']['regular_weight']), native=compact_native(native),
        bank_key=bank['key'], selection=False, MC_banks=1, independent_parent_count_not_MC_rows=True)
    log.write(json.dumps(record, allow_nan=False)+'\n')
    log.flush()
    state['last_monitor'] = dict(index=state['monitor_index'], accepted_updates=state['accepted_updates'])
    state['monitor_index'] += 1
    model.train(was_training)
    return record


def run_track(model, ctx, seed, name, reference_index, p, out, progress, started):
    out.mkdir()
    track_started = time.monotonic()
    ctx = dict(ctx, deadline=min(started+p['resources']['internal_stop_save_seconds'], track_started+900))
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([dict(params=groups['mean'], lr=p['training']['mean_lr']),
        dict(params=groups['shape'], lr=p['training']['shape_lr']),
        dict(params=groups['charge'], lr=p['training']['charge_lr'])], weight_decay=0.)
    state = dict(track=name, reference_index=reference_index, phase='mean', charge_enabled=False,
                 radius_LSB=1., accepted_updates=0, consecutive_rejected=0, consecutive_accepted=0,
                 monitor_index=0, direction=0)
    model.set_phase('mean').train()
    monitor_bank = make_monitor(ctx, seed, progress)
    source_visits = np.zeros(len(ctx['encoding']['source']), np.int64)
    target_visits = np.zeros(ctx['target_population'], np.int64)
    result = dict(track=name, learner_seed=seed, matching_M0_reference_index=reference_index,
                  phases={}, charge_enabled=False, resource_stop=False)
    save_state(out/'initial.pt', model, optimizer, state, progress, p)
    snapshot = transaction_snapshot(model, optimizer, state, ctx['loss'])
    phase, phase_accepted = 'mean', 0
    try:
        with (out/'MONITOR.jsonl').open('x', encoding='utf8', buffering=1) as monitor_log, \
             (out/'metrics.jsonl').open('x', encoding='utf8', buffering=1) as log:
            monitor(model, ctx, monitor_bank, state, progress, p, monitor_log)
            for phase in ('mean', 'random'):
                state.update(phase=phase, radius_LSB=1., consecutive_rejected=0, consecutive_accepted=0)
                model.set_phase(phase, train_charge=state['charge_enabled']).train()
                phase_accepted, reason = 0, 'maximum_directions'
                if phase == 'random':
                    charge_activation(model, optimizer, ctx, state, seed, name, progress, p, out)
                for direction in range(1, 51):
                    if time.monotonic()-started >= p['resources']['internal_stop_save_seconds']:
                        reason, result['resource_stop'] = 'internal_wall_stop', True
                        break
                    state['direction'] = direction
                    snapshot = transaction_snapshot(model, optimizer, state, ctx['loss'])
                    check_limits(p, ctx['output_root'], started)
                    micros = 32 if direction == 1 else 16
                    banks_fit = allocate_banks(ctx, seed, 'FIT', (name, phase, direction), 0, micros, progress, stratified=True)
                    fit = fit_direction(model, optimizer, ctx['encoding'], ctx['real'], banks_fit, ctx['loss'],
                        ctx['carrier_scale'], p, charge_enabled=state['charge_enabled'], progress=progress,
                        record_microgradients=(direction == 1), deadline=ctx['deadline'],
                        source_visits=source_visits, target_visits=target_visits)
                    if direction == 1:
                        gradients = fit.pop('microgradients')
                        np.savez_compressed(out/f'{phase}_first_microgradients.npz', gradients=gradients,
                            parameter_names=np.asarray(model.trainable_parameter_names()),
                            parameter_numel=np.asarray([value.numel() for value in model.trainable_parameters()]))
                    del banks_fit
                    delta, proposed_Adam = adam_direction(model, optimizer, snapshot, progress)
                    restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
                    banks = allocate_banks(ctx, seed, 'TRAIN_CHECK', (name, phase, direction), 0, 8, progress)
                    old = score_library(model, ctx, banks, state['charge_enabled'], progress)
                    M0 = score_library(ctx['M0'], ctx, banks, False, progress)
                    old_geometry_outputs = [output for row in old[:4] for output in row['outputs']]
                    with torch.no_grad():
                        for key, parameter in model.named_parameters():
                            if key in delta:
                                parameter.copy_(snapshot['model'][key]+delta[key])
                    progress['raw_geometry_probes'] += 1
                    raw_outputs, raw_geometry, raw_failure = None, None, None
                    try:
                        raw_outputs = pure_probe(model, ctx, banks, state['charge_enabled'], progress)
                        raw_geometry = joined_geometry(old_geometry_outputs, raw_outputs, ctx['noise_RMS_floor_V'])
                    except MeanProposalOutsideDomain038 as error:
                        raw_failure = dict(reason=error.reason, rows=error.rows, values=error.values)
                    raw_RMS = None if raw_geometry is None else raw_geometry['signal_RMS_active_V']
                    base_scale = 1. if raw_RMS is None or raw_RMS == 0 else min(1., state['radius_LSB']*ADC_STEP/raw_RMS)
                    attempts, accepted = [], False
                    for factor in p['training']['backtrack_factors']:
                        check_deadline(ctx)
                        restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
                        scale = base_scale*factor
                        apply_trial(model, snapshot, delta, scale, progress)
                        attempt = dict(factor=factor, scale=scale)
                        try:
                            proposal_outputs = raw_outputs if scale == 1. and raw_outputs is not None else pure_probe(
                                model, ctx, banks, state['charge_enabled'], progress)
                            geometry = joined_geometry(old_geometry_outputs, proposal_outputs, ctx['noise_RMS_floor_V'])
                            attempt.update(geometry=geometry, trust=TR_pass(geometry, state['radius_LSB'], random_phase=(phase == 'random')))
                            if not attempt['trust']['passed']:
                                attempts.append(attempt)
                                continue
                            candidate = score_library(model, ctx, banks, state['charge_enabled'], progress)
                            full_geometry = joined_geometry([output for row in old for output in row['outputs']],
                                [output for row in candidate for output in row['outputs']], ctx['noise_RMS_floor_V'])
                            attempt['full_CHECK_geometry'] = full_geometry
                            attempt['full_CHECK_trust'] = TR_pass(full_geometry, state['radius_LSB'], random_phase=(phase == 'random'))
                            if not attempt['full_CHECK_trust']['passed']:
                                attempts.append(attempt)
                                continue
                            decision, details = library_decision(model, ctx, banks, old, M0, candidate,
                                                                state['charge_enabled'], progress, p)
                            if decision['unresolved'] and len(banks) == 8:
                                extra = allocate_banks(ctx, seed, 'TRAIN_CHECK', (name, phase, direction), 8, 8, progress)
                                old += score_library_at_snapshot(model, optimizer, state, ctx, extra, snapshot, delta, scale, progress)
                                M0 += score_library(ctx['M0'], ctx, extra, False, progress)
                                candidate += score_library(model, ctx, extra, state['charge_enabled'], progress)
                                banks += extra
                                full_geometry = joined_geometry([output for row in old for output in row['outputs']],
                                    [output for row in candidate for output in row['outputs']], ctx['noise_RMS_floor_V'])
                                attempt['full_CHECK_geometry'] = full_geometry
                                attempt['full_CHECK_trust'] = TR_pass(full_geometry, state['radius_LSB'], random_phase=(phase == 'random'))
                                decision, details = library_decision(model, ctx, banks, old, M0, candidate,
                                                                    state['charge_enabled'], progress, p)
                                if not attempt['full_CHECK_trust']['passed']:
                                    decision['accepted'] = False
                                    decision['full_CHECK_TR_rejected'] = True
                            attempt.update(decision=decision, bank_energy={key: details[key] for key in ('candidate', 'current', 'M0')},
                                pooled_native={key: compact_native(details[key]) for key in
                                    ('pooled_native_candidate', 'pooled_native_current', 'pooled_native_M0')})
                            accepted = decision['accepted']
                        except MeanProposalOutsideDomain038 as error:
                            attempt['outside_domain'] = dict(reason=error.reason, rows=error.rows, values=error.values)
                        attempts.append(attempt)
                        if accepted:
                            optimizer.load_state_dict(proposed_Adam)
                            state['accepted_updates'] += 1
                            progress['accepted_optimizer_updates'] += 1
                            phase_accepted += 1
                            state['consecutive_rejected'] = 0
                            state['consecutive_accepted'] += 1
                            if state['consecutive_accepted'] == 2:
                                state['radius_LSB'] = min(2., 1.25*state['radius_LSB'])
                                state['consecutive_accepted'] = 0
                            save_state(out/'best_accepted.pt', model, optimizer, state, progress, p)
                            snapshot = transaction_snapshot(model, optimizer, state, ctx['loss'])
                            monitor(model, ctx, monitor_bank, state, progress, p, monitor_log)
                            break
                    if not accepted:
                        restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
                        state['consecutive_rejected'] += 1
                        state['consecutive_accepted'] = 0
                        state['radius_LSB'] = max(.25, .5*state['radius_LSB'])
                    record = dict(track=name, phase=phase, direction=direction, FIT=fit, accepted=accepted,
                        accepted_updates=state['accepted_updates'], raw_geometry=raw_geometry, raw_failure=raw_failure,
                        base_scale=base_scale, attempts=attempts, CHECK_banks=len(banks),
                        bank_keys=[bank['key'] for bank in banks], radius_LSB_next=state['radius_LSB'],
                        last_monitor=state['last_monitor'], execution_counts=dict(progress),
                        elapsed_seconds=time.monotonic()-started)
                    log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                    progress['directions'] += 1
                    write_json(out/'STATUS.json', {key: record[key] for key in ('track', 'phase', 'direction', 'accepted',
                        'accepted_updates', 'radius_LSB_next', 'last_monitor', 'execution_counts', 'elapsed_seconds')})
                    snapshot = transaction_snapshot(model, optimizer, state, ctx['loss'])
                    if state['consecutive_rejected'] >= 5:
                        reason = 'five_consecutive_no_accept'
                        break
                result['phases'][phase] = dict(accepted=phase_accepted, stop_reason=reason, last_direction=state['direction'])
                monitor(model, ctx, monitor_bank, state, progress, p, monitor_log)
                if phase == 'mean':
                    save_state(out/'mean_endpoint.pt', model, optimizer, state, progress, p)
                    result['mean_state'] = new_module_state(model)
                    if result['resource_stop'] or phase_accepted == 0:
                        result['mean_to_random_supported'] = False
                        break
                    endpoint = phase_endpoint(model, ctx, seed, name, progress, p, out)
                    result['mean_to_random_supported'] = endpoint['accepted']
                    if not endpoint['accepted']:
                        break
    except TrackBudget038 as error:
        restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
        result['resource_stop'] = True
        result['phases'][phase] = dict(accepted=phase_accepted, stop_reason=str(error), last_direction=state['direction'])
        if 'mean_state' not in result:
            result['mean_state'] = new_module_state(model)
            result['mean_to_random_supported'] = False
        write_json(out/'TRACK_BUDGET_STOP.json', dict(reason=str(error), execution_counts=dict(progress), rollback=True))
    model.set_phase('frozen').eval()
    save_state(out/'final.pt', model, optimizer, state, progress, p)
    result.update(charge_enabled=state['charge_enabled'], accepted_updates=state['accepted_updates'],
        state=new_module_state(model), source_FIT_unique=int(np.count_nonzero(source_visits)),
        source_FIT_population=len(source_visits), target_FIT_unique=int(np.count_nonzero(target_visits)),
        target_FIT_population=len(target_visits), all_source_design_probabilities_positive=True,
        target_FIT_record_groups=int(len(np.unique(ctx['target_groups'][target_visits > 0]))),
        target_record_groups_total=int(len(np.unique(ctx['target_groups']))),
        elapsed_training_seconds=time.monotonic()-track_started, training_wall_limit_seconds=900)
    np.savez_compressed(out/'FIT_COVERAGE.npz', source_visits=source_visits, target_visits=target_visits)
    write_json(out/'TRACK_RESULT.json', {key: value for key, value in result.items() if key not in ('state', 'mean_state')})
    return result


def score_library_at_snapshot(model, optimizer, state, ctx, banks, snapshot, delta, scale, progress):
    restore_transaction(model, optimizer, state, ctx['loss'], snapshot)
    results = score_library(model, ctx, banks, state['charge_enabled'], progress)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in delta:
                parameter.copy_(snapshot['model'][name]+scale*delta[name])
    return results


@torch.no_grad()
def identity_gate(setup, p, out, progress):
    device = setup['source'].device
    generators = {name: torch.Generator().manual_seed(stream_seed(p['seeds'][0], 'G0_IDENTITY', name))
                  for name in ('shape', 'noise', 'extension')}
    reports = {str(index): dict(pure_bitwise=True, hard_bitwise={h: True for h in HYPOTHESES},
        pure_absmax_V=0., hard_absmax_V={h: 0. for h in HYPOTHESES}) for index in (0, 1)}
    for start in range(0, len(setup['source']), 128):
        rows = torch.arange(start, min(start+128, len(setup['source'])), device=device)
        cache = N01Response038.select_encoding(setup['encoding'], rows)
        shape = torch.randn(len(rows), 8, generator=generators['shape']).to(device)
        charge = torch.zeros(len(rows), device=device, dtype=torch.float64)
        noise = frozen_electronics(len(rows), generators['noise'], generators['extension'], setup['extension'], device, progress)
        for index, model in enumerate(setup['models']):
            model._active_encoding = cache
            progress['original014_calls'] += 1
            progress['original014_rows'] += len(rows)
            try:
                original = FrozenDeterministicVoltageShape014.random_forward(model, cache['source'], shape, enabled=True)
            finally:
                model._active_encoding = None
            progress['response_calls'] += 1
            progress['response_rows'] += len(rows)
            proposed = model(cache['source'], shape, charge, cache=cache, charge_enabled=False)
            report = reports[str(index)]
            report['pure_bitwise'] = report['pure_bitwise'] and torch.equal(
                original['signal_pre'].view(torch.uint8), proposed['signal_pre'].view(torch.uint8))
            report['pure_absmax_V'] = max(report['pure_absmax_V'], float((original['signal_pre']-proposed['signal_pre']).abs().max()))
            for h in HYPOTHESES:
                a, b = [observe_generated_carrier(value['signal_pre']+noise[h], ste=False,
                    q_floor=float(setup['loss'].q_floor)) for value in (original, proposed)]
                progress['generated_observation_rows'] += 2*len(rows)
                report['hard_bitwise'][h] = report['hard_bitwise'][h] and torch.equal(
                    a['window_V'].view(torch.uint8), b['window_V'].view(torch.uint8))
                report['hard_absmax_V'][h] = max(report['hard_absmax_V'][h], float((a['window_V']-b['window_V']).abs().max()))
    passed = all(item['pure_bitwise'] and all(item['hard_bitwise'].values()) for item in reports.values())
    write_json(out/'G0_ORIGINAL014_IDENTITY.json', dict(passed=passed, seeds=reports, parents=6593,
        original_entrypoint='FrozenDeterministicVoltageShape014.random_forward -> original010 assembly',
        same_device=True, cached_original_A0_and_decoded=True, CP_deserializations_extra=0,
        source_sha256={name: sha256(ROOT/name) for name in ('src/n01/model010.py', 'src/n01/model014.py', 'src/n01/model038.py')}))
    if not passed:
        raise RuntimeError('038 exact original014 G0 identity gate failed')


@torch.enable_grad()
def classifier256(fake, real, parent_ids, real_groups, stats, device, seed, out, progress, p, started):
    from n01.evaluate import _DomainClassifier, _split_groups, _weighted_auc, _auc_bootstrap
    ft, fe = _split_groups(parent_ids, stream_seed(p['seeds'][0], 'C2ST_SPLIT'))
    rt, re = _split_groups(real_groups, stream_seed(p['seeds'][0], 'C2ST_SPLIT_REAL'))
    train_x, test_x = np.concatenate((fake[ft], real[rt])), np.concatenate((fake[fe], real[re]))
    train_y, test_y = np.r_[np.zeros(len(ft)), np.ones(len(rt))], np.r_[np.zeros(len(fe)), np.ones(len(re))]
    center = torch.as_tensor(stats['views']['voltage']['center'], dtype=torch.float32, device=device)
    scale = torch.as_tensor(stats['views']['voltage']['scale'], dtype=torch.float32, device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        classifier = _DomainClassifier()
        classifier.head[1] = torch.nn.Linear(16*16, 32)
    classifier.to(device).train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    rng = np.random.default_rng(stream_seed(seed, 'MINIBATCH'))
    class_weight = np.where(train_y == 1, len(train_y)/(2*len(rt)), len(train_y)/(2*len(ft)))
    for step in range(200):
        check_limits(p, out.parent.parent, started)
        if time.monotonic()-started >= p['resources']['internal_stop_save_seconds']:
            raise TimeoutError('038 FINAL classifier resource reserve exhausted; partial counts retained')
        index = rng.choice(len(train_x), 128, replace=False)
        x, y, weight = [torch.as_tensor(value, dtype=torch.float32, device=device)
                        for value in (train_x[index], train_y[index], class_weight[index])]
        objective = (F.binary_cross_entropy_with_logits(classifier((x-center)/scale), y,
                                                        reduction='none')*weight).mean()
        progress['classifier_forward_calls'] += 1
        progress['classifier_forward_rows'] += len(x)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        progress['classifier_AD_queries'] += 1
        optimizer.step()
        progress['classifier_optimizer_steps'] += 1
    classifier.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_x), 128):
            x = torch.as_tensor(test_x[start:start+128], dtype=torch.float32, device=device)
            scores.append(classifier((x-center)/scale).cpu().numpy())
            progress['classifier_forward_calls'] += 1
            progress['classifier_forward_rows'] += len(x)
    scores = np.concatenate(scores)
    weights = np.ones(len(test_y))
    uncertainty = _auc_bootstrap(test_y, scores, parent_ids[fe], real_groups[re], weights, stream_seed(seed, 'BOOTSTRAP'))
    progress['classifier_bootstrap_resamples'] += 400
    out.mkdir()
    np.savez_compressed(out/'test_predictions.npz', labels=test_y, logits=scores,
        comparison_weights=weights, source_parent_ids=parent_ids[fe], REAL_groups=real_groups[re])
    torch.save(dict(model={name: value.detach().cpu() for name, value in classifier.state_dict().items()},
        updates=200, seed=seed, architecture='Conv1d8_16_MLP32_256', response_training=False), out/'classifier.pt')
    report = dict(raw_AUC=_weighted_auc(test_y, scores), count_weighted_AUC=_weighted_auc(test_y, scores, weights),
        uncertainty=uncertainty, updates=200, seed=seed, learning_rate=.001, batch_rows=128,
        source_train_rows=len(ft), source_test_rows=len(fe), REAL_train_rows=len(rt), REAL_test_rows=len(re),
        original_group_split=True, response_tuning=False, independent_acquisition_uncertainty=False)
    write_json(out/'RESULT.json', report)
    progress['classifiers_completed'] += 1
    return report


def composition_summary(wave, baseline, source_h, q_floor):
    from run_wp5_026 import finite_measure026
    features, flags = observable_features(wave, baseline, q_floor)
    features['negative_lobe_area_Vns'] = np.maximum(-wave, 0).sum(1)
    h = np.asarray(source_h, np.float64)
    raw = dict(nominal=np.ones(len(h)), low_h=1-.2*(2*h-1), high_h=1+.2*(2*h-1))
    report = {}
    for label, weight in raw.items():
        weight = weight/weight.sum()
        report[label] = dict(source_parent_ESS=float(1/np.square(weight).sum()),
            scalars={name: finite_measure026(value, weight, draws=4) for name, value in features.items()
                     if np.asarray(value).dtype.kind != 'b'},
            flags={name: finite_measure026(value.astype(np.float64), weight, draws=4) for name, value in flags.items()})
    return dict(weights=report, assumption='fixed026 raw1±.2*(2h-1), normalized over original SOURCE parents',
                h_is_not_REAL_label=True, separately_refitted_parameter_stability='not_established')


@torch.no_grad()
def teacher_recovery(model, teacher_ctx, index, charge_enabled, p, out, progress):
    truth = teacher_ctx['held_factors']
    predicted = factors038(model, teacher_ctx['held_encoding'], charge_enabled, progress)
    P, rawP, scales = teacher_projections(truth)
    tm, pm = projected_moments(truth, P), projected_moments(predicted, P)
    protocol = dict(source=p['teacher']['source'], evaluation=p['teacher']['evaluation'])
    result = recovery_metrics(predicted, truth, pm, tm, teacher_ctx['heldout'], protocol)
    rows = torch.as_tensor(p['teacher']['evaluation']['MC_parent_rows'], device=P.device)
    repeated = rows.repeat_interleave(256)
    generator = torch.Generator().manual_seed(stream_seed(p['seeds'][index], 'TEACHER_FINAL_MC'))
    joint = torch.randn(len(repeated), 9, generator=generator, dtype=torch.float64).to(P.device)
    arrays, reports = {}, {}
    for name, factor, moments in (('learner', predicted, pm), ('teacher', truth, tm)):
        wave = cached_signal038(factor, repeated, joint[:, 1:], joint[:, 0])
        score = wave.reshape(len(rows), 256, 384)@P[rows]
        q = canonical_charge008(wave).reshape(len(rows), 256)
        centered = score-score.mean(1, keepdim=True)
        covariance = centered.transpose(1, 2)@centered/255
        variance = q.var(1, correction=1)
        fourth = ((q-q.mean(1, keepdim=True))**4).mean(1)
        variance_SE = ((fourth-253/255*variance.square()).clamp_min(0)/256).sqrt()
        outer = centered[:, :, :, None]*centered[:, :, None, :]
        covariance_SE = outer.var(1, correction=1).div(256).sqrt()
        observed = observe_generated_carrier(wave, ste=False, q_floor=1e-8)
        progress['conditional_MC_rows'] += len(wave)
        progress['generated_observation_rows'] += len(wave)
        reports[name] = dict(Q_variance_relative_RMS=float((variance-moments['variance_Q'][rows]).norm()/tm['variance_Q'][rows].norm()),
            covariance_relative_Frobenius=float((covariance-moments['covariance'][rows]).norm()/tm['covariance'][rows].norm()),
            variance_SE_relative_RMS=float(variance_SE.norm()/tm['variance_Q'][rows].norm()),
            covariance_SE_relative_Frobenius=float(covariance_SE.norm()/tm['covariance'][rows].norm()),
            hard_without005_Q_mean=float(observed['Q'].mean()), hard_without005_Q_std=float(observed['Q'].std()),
            flags={name: int(value.sum()) for name, value in observed['flags'].items()})
        arrays[name+'_MC_Q'] = q.cpu().numpy()
        arrays[name+'_MC_projected'] = score.cpu().numpy()
    result.update(MC=reports, MC_is_gate=False, conditional_charge_enabled=charge_enabled,
        marginal_risk_report='track MONITOR/metrics and mean endpoint check; not substituted by conditional gate',
        mean_and_covariance_scope='exact preADC; MC hard report excludes005 to match original029 QA',
        independently_acquired_response_validation=False)
    write_json(out/f'TEACHER_seed{index}_RECOVERY.json', result)
    np.savez_compressed(out/f'TEACHER_seed{index}_SUMMARY.npz', teacher_mean=truth['A'].cpu().numpy(),
        learner_mean=predicted['A'].cpu().numpy(), teacher_covariance=tm['covariance'].cpu().numpy(),
        learner_covariance=pm['covariance'].cpu().numpy(), source_Q=teacher_ctx['heldout']['Q'].numpy(),
        template=teacher_ctx['heldout']['template'].numpy(), projections=rawP.cpu().numpy(),
        projection_scales=scales.cpu().numpy(), MC_parent_rows=rows.cpu().numpy(), **arrays)
    return result


@torch.no_grad()
def teacher_initial_recovery(models, teacher_ctx, p, out, progress):
    truth = teacher_ctx['held_factors']
    P, raw, scales = teacher_projections(truth)
    tm = projected_moments(truth, P)
    reports = {}
    for index, model in enumerate(models):
        initial = factors038(model, teacher_ctx['held_encoding'], False, progress)
        reports[str(index)] = recovery_metrics(initial, truth, projected_moments(initial, P), tm,
            teacher_ctx['heldout'], dict(source=p['teacher']['source'], evaluation=p['teacher']['evaluation']))
    write_json(out/'TEACHER_INITIAL_ANALYTIC.json', dict(seeds=reports, charge_off=True, MC_rows=0,
        no_heldout_parameter_selection=True, purpose='initial gates are not learned recovery'))
    return reports


def final_risk_confirmation(states, ctx, p, out, progress):
    banks = allocate_banks(ctx, p['seeds'][0], 'FINAL_RISK', (), 0, 16, progress)
    reports = {}
    for state in states:
        results = score_library(state['model'], ctx, banks, state['charge_enabled'], progress)
        reports[state['label']] = dict(energy=[risk_record(result, .01) for result in results],
            native=compact_native(pooled_native(results, ctx['real'], banks, ctx['loss'], ctx['context'])))
    contrasts = {}
    for index in (0, 1):
        full = reports[f'final_full_seed{index}']['energy']
        contrasts[str(index)] = {name: paired_risk(full, reports[label]['energy']) for name, label in
            (('full_minus_M0', f'M0_seed{index}'), ('full_minus_mean_endpoint', f'mean_endpoint_seed{index}'))}
    write_json(out/'FINAL_RISK_CONFIRMATION.json', dict(contrasts=contrasts, states=reports,
        bank_keys=[bank['key'] for bank in banks], fresh_after_endpoint_freeze=True, response_tuning=False))
    return contrasts


@torch.no_grad()
def source_pair_summary(states, setup, p, out, progress):
    information = json.loads(verified_path028(p['input_information']).read_text(encoding='utf8'))
    pairs = [pair for pair in information['same_Q_real_input_shape_feasibility']['pairs'] if pair['role'] == 'TRAIN']
    if len(pairs) != 121:
        raise ValueError('038 fixed013 TRAIN pair inventory differs')
    identity = {str(value): index for index, value in enumerate(setup['data']['source_ids'])}
    left = np.asarray([identity[pair['left_source_id']] for pair in pairs])
    right = np.asarray([identity[pair['right_source_id']] for pair in pairs])
    union, inverse = np.unique(np.r_[left, right], return_inverse=True)
    l, r = inverse[:121], inverse[121:]
    cache = N01Response038.select_encoding(setup['encoding'], torch.as_tensor(union, device=setup['source'].device))
    source = cache['source'].cpu().numpy()
    norm = source/np.max(np.abs(source), axis=1, keepdims=True)
    contrast = np.sqrt(np.mean((norm[l]-norm[r])**2, axis=1))
    pc = cache['condition'][:, 1].double().cpu().numpy()
    result = dict(pairs=pairs, prepared384_peak_normalized_RMS=contrast,
        fixed_PC1_difference=np.abs(pc[l]-pc[r]), original_pair_order_preserved=True,
        no_new_PCA=True, source_ids_only_for_retrieval=True, states={})
    for state in states:
        if state['label'] == 'A':
            continue
        factors = factors038(state['model'], cache, state['charge_enabled'], progress)
        mean = factors['A']
        B = factors['B']
        cov = torch.cat((B.transpose(1, 2), mean[:, :, None]*factors['cv2'].sqrt()[:, None, None]), 2)
        left_cov, right_cov = cov[l], cov[r]
        gl, gr, cross = left_cov.transpose(1, 2)@left_cov, right_cov.transpose(1, 2)@right_cov, left_cov.transpose(1, 2)@right_cov
        covariance_difference = (gl.square().sum((1, 2))+gr.square().sum((1, 2))-2*cross.square().sum((1, 2))).clamp_min(0).sqrt()
        generator = torch.Generator().manual_seed(stream_seed(p['seeds'][0], 'SOURCE_PAIRS'))
        z = torch.randn(121, 9, generator=generator, dtype=torch.float64).to(mean.device)
        a = cached_signal038(factors, torch.as_tensor(l, device=mean.device), z[:, 1:], z[:, 0])
        b = cached_signal038(factors, torch.as_tensor(r, device=mean.device), z[:, 1:], z[:, 0])
        progress['source_pair_cached_signal_rows'] += 242
        result['states'][state['label']] = dict(mean_difference_RMS_V=(mean[l]-mean[r]).square().mean(1).sqrt(),
            covariance_difference_Frobenius_V2=covariance_difference,
            one_common_latent_difference_RMS_V=(a-b).square().mean(1).sqrt(),
            latent_repeats=1, preADC_not_physical_identification=True)
    write_json(out/'SOURCE_PAIR_CONTENT_RESPONSE.json', result)
    return dict(pairs=121, prepared384_contrast_nonzero=int(np.count_nonzero(contrast)),
                fixed_PC1_contrast_nonzero=int(np.count_nonzero(pc[l]-pc[r])))


def compare_final(banks, tolerances, risk):
    report = {}
    for bank_name, results in banks.items():
        report[bank_name] = {}
        for index in (0, 1):
            rows = {}
            for h in HYPOTHESES:
                A, M0, mean, off, full = [results[label][h]['values'] for label in
                    ('A', f'M0_seed{index}', f'mean_endpoint_seed{index}', f'final_chargeoff_seed{index}', f'final_full_seed{index}')]
                calibrated, uncalibrated = {}, []
                for name, T in tolerances[h].items():
                    if T is None or T <= 0:
                        uncalibrated.append(name)
                    elif any(value[name] is None for value in (A, M0, full)):
                        calibrated[name] = dict(status='undefined_not_PASS', T=T)
                    else:
                        calibrated[name] = dict(status='defined', T=T, final_over_T=full[name]/T,
                            noninferior_to_A=full[name]-A[name] <= T,
                            noninferior_to_M0=full[name]-M0[name] <= T, absolute_within_T=full[name] <= T)
                morphology = {name: dict(full_minus_A=None if full[name] is None or A[name] is None else full[name]-A[name],
                    full_minus_M0=None if full[name] is None or M0[name] is None else full[name]-M0[name],
                    mean_endpoint_minus_M0=None if mean[name] is None or M0[name] is None else mean[name]-M0[name])
                    for name in ('peak_voltage_V', 'front_Vns', 'tail_Vns', 'rise10_90_ns', 'FWHM_ns')}
                rows[h] = dict(wave_improvement_vs_A=(A['wave']-full['wave'])/A['wave'],
                    wave_improvement_vs_M0=(M0['wave']-full['wave'])/M0['wave'],
                    wave_2percent_vs_both=(full['wave'] <= .98*A['wave'] and full['wave'] <= .98*M0['wave']),
                    calibrated=calibrated, uncalibrated_domains=uncalibrated, morphology=morphology,
                    full_minus_mean={name: None if value is None or mean[name] is None else value-mean[name] for name, value in full.items()},
                    chargeoff_minus_mean={name: None if value is None or mean[name] is None else value-mean[name] for name, value in off.items()},
                    full_original_flags=results[f'final_full_seed{index}'][h]['original_observation_flags'],
                    mean_original_flags=results[f'mean_endpoint_seed{index}'][h]['original_observation_flags'])
            report[bank_name][str(index)] = rows
    confirmed = {str(index): {h: all(risk[str(index)]['full_minus_M0']['by_hypothesis'][h][name]['mean']
        +3*risk[str(index)]['full_minus_M0']['by_hypothesis'][h][name]['MCSE'] < 0 for name in ('data', 'wave'))
        for h in HYPOTHESES} for index in (0, 1)}
    return dict(native=report, fresh_risk_improvement_resolved=confirmed,
        all_native_TRAIN_banks_two_seeds_wave_improve_2percent=all(
            cell['wave_2percent_vs_both'] for name in ('FINAL_TRAIN_0', 'FINAL_TRAIN_1')
            for seed in report[name].values() for cell in seed.values()),
        scientific_status='response_match_not_established',
        absolute_T_and_uncalibrated_domains_remain_explicit=True,
        scientific_acceptance_not_inferred_from_native_exit_or_loss=True)


def budget_closure038():
    rows = dict(FIT=835584, CHECK=4915200, geometry=1024000, phase_endpoint=16384,
        charge_activation=26624, MONITOR=316416, identity038=13186,
        initial_controls=105488, FINAL=547920, teacher_truth_factors=192,
        teacher_conditional_factors=256, source_pair_factors=2420, FINAL_risk=12288)
    noise = dict(FIT=1671168, CHECK=1638400, phase_endpoint=16384, charge_activation=16384,
        MONITOR=4096, identity=6593, initial_controls=26372, FINAL=54792, FINAL_risk=2048)
    algebra = dict(FIT_teacher_targets=835584, CHECK_teacher_targets=819200,
        phase_endpoint_teacher_targets=8192, activation_teacher_targets=8192,
        MONITOR_teacher_targets=1024, conditional_MC=16384, source_pairs=2420)
    adc = dict(FIT=5013504, CHECK=17203200, phase_endpoint=73728, charge_activation=98304,
        MONITOR=952320, identity=79116, initial_controls=395580, FINAL=1808136,
        conditional_MC=16384, FINAL_risk=36864)
    return dict(schema='n01/response038-operation-closure/v1', bounds_not_extra_queries=True,
        tracks=4, directions=400, FIT_AD_queries=6528, optimizer_steps=400,
        accepted_optimizer_updates=400, parameter_amplitude_trials=1600, charge_activation_trials=4,
        raw_geometry_probes=400, original_CP_loads=6,
        old_A_calls=58, old_A_rows=7297, original014_rows=13186,
        original014_calls=104, response_calls_max=112792, noise005_calls_max=39901,
        allocated_microbanks_max=13076, generated_observation_calls_max=331639,
        response_rows_by_path=rows, response_rows_max=sum(rows.values()),
        noise005_rows_by_path=noise, noise005_rows_max=sum(noise.values()),
        extension_hypothesis_rows_max=3*sum(noise.values()),
        cached_algebra_rows_by_path=algebra, generated_observation_rows_by_path=adc,
        generated_observation_rows_max=sum(adc.values()),
        classifier_AD_queries=6600, classifier_optimizer_steps=6600,
        classifier_training_rows=6600*128, classifier_test_rows_max=33*(6593*4+21047),
        classifier_forward_rows_max=6600*128+33*(6593*4+21047), classifier_bootstrap_resamples=33*400,
        classifier_forward_calls_max=6600+33*((6593*4+21047+127)//128), classifiers_completed_max=33,
        FINAL_private_crop_uint16_uncompressed_bytes=33*(2*6593*4+512*4)*256*2,
        FINAL_private_prefix_uint16_uncompressed_bytes=(2*6593*4+512*4)*100*2,
        log_estimate_MiB_max=300, total_output_cap_MiB=2048,
        track_training_seconds_max_each=900, hard_wall_seconds=7200, internal_stop_seconds=6900,
        row_definition='response_rows are actual038 head+map calls excluding immutable cached A; original014 separately',
        microbank_definition='64A+64B generated,128T+128U targets; CHECK128/256 total',
        native_checks='pool emitted8/16 banks; no new NN/ADC for pooled metrics or full-CHECK geometry',
        NN_counter_semantics='attempted call/row entry counted before execution, including explicit outside-domain proposals; noise005_rows completed and noise005_rows_attempted both retained',
        rejected_work_remains_in_cost=True, teacher_and_classifier_updates_not_REAL_response_acceptance=True)


@torch.no_grad()
def trust_geometry(old, proposed, *, noise_RMS_floor_V, tail_integral_floor_Vns, tail_start):
    """Pure signal and compact pre-ADC moments; no ADC, noise draw or384² C."""
    difference = proposed['signal_pre']-old['signal_pre']
    active = difference[:, 100:]
    event_RMS = active.square().mean(1).sqrt()
    fixed_A = old['A0_signal_pre_V']
    denominator = torch.maximum(fixed_A[:, 100:].square().mean(1).sqrt(),
                                fixed_A.new_tensor(noise_RMS_floor_V))
    relative = event_RMS/denominator
    Q0 = canonical_charge008(fixed_A)
    tail = difference[:, tail_start:].sum(1).abs()/torch.maximum(Q0, Q0.new_tensor(tail_integral_floor_Vns))
    mean_difference = proposed['moment_mean_pre_V']-old['moment_mean_pre_V']
    factors = [torch.cat((value['moment_basis_V'].transpose(1, 2),
                         value['moment_mean_pre_V'][:, :, None]*value['moment_charge_cv2'].sqrt()[:, None, None]), 2)
               for value in (old, proposed)]
    old_gram = factors[0].transpose(1, 2)@factors[0]
    new_gram = factors[1].transpose(1, 2)@factors[1]
    cross = factors[0].transpose(1, 2)@factors[1]
    old_norm2 = old_gram.square().sum((1, 2))
    new_norm2 = new_gram.square().sum((1, 2))
    covariance_difference2 = old_norm2+new_norm2-2*cross.square().sum((1, 2))
    # A Frobenius squared identity can round below zero at an identical state.
    covariance_difference2_nonnegative = covariance_difference2.clamp_min(0)
    covariance_relative_per_row = covariance_difference2_nonnegative.sqrt()/torch.maximum(
        old_norm2.sqrt(), old_norm2.new_tensor(noise_RMS_floor_V**2))
    trace_old = factors[0].square().sum((1, 2))
    trace_new = factors[1].square().sum((1, 2))
    return dict(rows=len(difference), active_samples=[100, 384],
        signal_RMS_active_V=float(active.square().mean().sqrt()),
        signal_RMS_full384_V=float(difference.square().mean().sqrt()),
        event_RMS_p99_V=float(event_RMS.quantile(.99)), event_RMS_max_V=float(event_RMS.max()),
        coordinate_absmax_V=float(difference.abs().max()), relative_event_RMS_p99=float(relative.quantile(.99)),
        relative_event_RMS_max=float(relative.max()), relative_tail_integral_absmax=float(tail.max()),
        mean_change_RMS_active_V=float(mean_difference[:, 100:].square().mean().sqrt()),
        preADC_covariance_relative_Frobenius=float((covariance_difference2_nonnegative.sum()/old_norm2.sum()).sqrt()),
        preADC_covariance_relative_event_p99=float(covariance_relative_per_row.quantile(.99)),
        preADC_covariance_relative_event_max=float(covariance_relative_per_row.max()),
        preADC_covariance_noise_floor_V2=noise_RMS_floor_V**2,
        preADC_covariance_change_Frobenius_RMS_V2=float(covariance_difference2_nonnegative.mean().sqrt()),
        preADC_covariance_trace_old_mean_V2=float(trace_old.mean()),
        preADC_covariance_trace_new_mean_V2=float(trace_new.mean()),
        preADC_covariance_difference_norm2_roundoff_negative_rows=int((covariance_difference2 < 0).sum()),
        preADC_charge_variance_old_mean_Vns2=float(old['charge_Q_variance'].mean()),
        preADC_charge_variance_new_mean_Vns2=float(proposed['charge_Q_variance'].mean()),
        normalization_gain_min=float(proposed['normalization_gain'].min()),
        normalization_gain_max=float(proposed['normalization_gain'].max()),
        warp_Q_ratio_min=float(proposed['warp_Q_ratio'].min()),
        scope='finite pure-signal CRN bank; software step control, not calibrated physical tolerance')


@torch.no_grad()
def evaluate_population(states, encoding, real_wave, real_baseline, extension, p, loss, context,
                        library_seed, scope, out, progress, started, *, save_codes=False,
                        source_h=None):
    """All controls/endpoints share rows, draws, electronics and hard readout."""
    ev, device = p['evaluation'], encoding['source'].device
    rows = np.repeat(np.arange(len(encoding['source'])), ev['draws_per_parent'])
    waves = {state['label']: {h: np.empty((len(rows), 256), np.float64) for h in HYPOTHESES} for state in states}
    flags = {state['label']: {h: {} for h in HYPOTHESES} for state in states}
    baseline = np.empty((len(rows), 100), np.float64)
    baseline_mean = np.empty(len(rows), np.float64)
    prefix_codes = np.empty((len(rows), 100), np.uint16) if save_codes else None
    codes = {state['label']: {h: np.empty((len(rows), 256), np.uint16) for h in HYPOTHESES}
             for state in states} if save_codes else None
    starts = {state['label']: {h: np.empty(len(rows), np.uint16) for h in HYPOTHESES}
              for state in states} if save_codes else None
    reconstruction = {state['label']: {h: dict(bitwise=True, classifier_float32_bitwise=True, max_abs_error_V=0.) for h in HYPOTHESES}
                      for state in states}
    generators = {name: torch.Generator(device='cpu').manual_seed(stream_seed(library_seed, name))
                  for name in ('shape', 'charge', 'noise005', 'extension')}
    modes = {id(state['model']): (state['model'], state['model'].training)
             for state in states if state['label'] != 'A'}
    for model, previous in modes.values():
        model.eval()
    for start in range(0, len(rows), ev['NN_chunk_rows']):
        stop = min(start+ev['NN_chunk_rows'], len(rows))
        indices = torch.as_tensor(rows[start:stop], device=device)
        cache = N01Response038.select_encoding(encoding, indices)
        count = stop-start
        shape = torch.randn(count, 8, generator=generators['shape'], dtype=torch.float32).to(device)
        charge = torch.randn(count, generator=generators['charge'], dtype=torch.float64).to(device)
        noise = frozen_electronics(count, generators['noise005'], generators['extension'], extension, device, progress)
        reference_prefix = None
        for state in states:
            if state['label'] == 'A':
                signal = cache['original']['signal_pre']
            else:
                progress['response_calls'] += 1
                progress['response_rows'] += count
                result = state['model'](cache['source'], shape, charge, cache=cache,
                    shape_enabled=state['shape_enabled'], mean_enabled=state['mean_enabled'],
                    charge_enabled=state['charge_enabled'])
                signal = result['signal_pre']
            for h in HYPOTHESES:
                observed = observe_generated_carrier(signal+noise[h], ste=False, q_floor=float(loss.q_floor),
                                                     return_carrier=save_codes, return_indices=save_codes)
                progress['generated_observation_rows'] += count
                label = state['label']
                waves[label][h][start:stop] = observed['window_V'].cpu().numpy()
                if reference_prefix is None:
                    reference_prefix = observed['baseline100']
                    baseline[start:stop] = reference_prefix.cpu().numpy()
                    baseline_mean[start:stop] = observed['baseline_mean_V'].cpu().numpy()
                    if save_codes:
                        prefix_codes[start:stop] = np.rint((baseline[start:stop]-ADC_MIN)*ADC_LEVELS).astype(np.uint16)
                elif not torch.equal(reference_prefix, observed['baseline100']):
                    raise RuntimeError('038 signal/control/hypothesis changed common prefix100')
                for name, value in observed['flags'].items():
                    flags[label][h].setdefault(name, []).append(value.cpu().numpy())
                if save_codes:
                    raw = observed['carrier_adc'].gather(1, observed['crop_indices']).cpu().numpy()
                    encoded = np.rint((raw-ADC_MIN)*ADC_LEVELS).astype(np.uint16)
                    restored = encoded.astype(np.float64)/ADC_LEVELS+ADC_MIN-baseline_mean[start:stop, None]
                    codes[label][h][start:stop] = encoded
                    starts[label][h][start:stop] = observed['crop_start'].cpu().numpy()
                    reconstruction[label][h]['bitwise'] = reconstruction[label][h]['bitwise'] and np.array_equal(
                        restored.view(np.uint64), waves[label][h][start:stop].view(np.uint64))
                    reconstruction[label][h]['classifier_float32_bitwise'] = reconstruction[label][h]['classifier_float32_bitwise'] and np.array_equal(
                        restored.astype(np.float32).view(np.uint32), waves[label][h][start:stop].astype(np.float32).view(np.uint32))
                    reconstruction[label][h]['max_abs_error_V'] = max(reconstruction[label][h]['max_abs_error_V'],
                        float(np.max(np.abs(restored-waves[label][h][start:stop]))))
        check_limits(p, out, started)
    reports = {}
    if save_codes:
        np.savez_compressed(out/f'{scope}_common_prefix_codes.npz', prefix_adc_codes=prefix_codes,
            observed_baseline_mean_V=baseline_mean, source_parent_rows=rows,
            draw=np.tile(np.arange(ev['draws_per_parent'], dtype=np.uint8), len(encoding['source'])))
    for state in states:
        label = state['label']
        reports[label] = {}
        for h in HYPOTHESES:
            report = hard_metrics(waves[label][h], baseline, real_wave, real_baseline, loss, context)
            report['values']['negative_lobe_area_Vns'] = raw_distance(np.maximum(-waves[label][h], 0).sum(1),
                                                                      np.maximum(-real_wave, 0).sum(1))['value']
            report.update(endpoint=label, scope=scope, source_parents=len(encoding['source']),
                draws_per_parent=ev['draws_per_parent'], MC_banks=1, draws_are_independent_parents=False,
                standardization='original frozen TRAIN; energy and nativeW1 are distinct',
                source_count_population=True, REAL_physical_weights='unknown',
                original_observation_flags={name: dict(nonzero_rows=int(np.count_nonzero(np.concatenate(value))),
                    denominator=len(rows)) for name, value in flags[label][h].items()})
            reports[label][h] = report
            if save_codes:
                private_path = out/f'{scope}_{label}_{h}_ADC_codes.npz'
                np.savez_compressed(private_path, raw_crop_adc_codes=codes[label][h],
                    crop_start=starts[label][h], **{name: np.packbits(np.concatenate(value))
                                                  for name, value in flags[label][h].items()})
                report['private_ADC_encoding'] = dict(path=private_path.name, raw_ADC_codes_lossless=True,
                    common_prefix_file=f'{scope}_common_prefix_codes.npz', reconstruction=reconstruction[label][h],
                    operator='float64(code)/16383-.5 - saved observed baseline_mean_V',
                    baseline_subtracted_window_not_rounded_to_ADC_codes=True)
            if source_h is not None:
                report['composition_sensitivity'] = composition_summary(waves[label][h], baseline, source_h, float(loss.q_floor))
            write_json(out/f'{scope}_{label}_{h}.json', report)
            selected = np.linspace(0, len(encoding['source'])-1, 32, dtype=np.int64)*ev['draws_per_parent']
            np.savez_compressed(out/f'{scope}_{label}_{h}_examples.npz', waveform256=waves[label][h][selected],
                baseline100=baseline[selected], source_parent_rows=rows[selected], example_row_indices=selected,
                population_mean_V=waves[label][h].mean(0), population_std_V=waves[label][h].std(0),
                example_selection=np.asarray('fixed evenly spaced parents; first draw; not output selected'),
                population_summary=np.asarray('generated count population; not REAL mean/std'))
            check_limits(p, out, started)
    for model, previous in modes.values():
        model.train(previous)
    return reports


def check_limits(p, out, started):
    if time.monotonic()-started >= p['resources']['internal_stop_save_seconds']:
        raise TimeoutError('038 internal6900s stop-save boundary reached; external hard7200s remains')
    if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > p['resources']['output_MiB']*1024**2:
        raise OSError('038 registered output limit exceeded')


def aggregate_REAL(data, out, scope):
    wave, baseline = data['real'], data['real_baseline100']
    noise_RMS = np.sqrt(np.mean((baseline-baseline.mean(1, keepdims=True))**2, axis=1))
    np.savez_compressed(out/f'{scope}_REAL_AGGREGATES.npz', waveform_mean_V=wave.mean(0),
        waveform_std_V=wave.std(0), prefix_population_mean_V=baseline.mean(0),
        prefix_population_std_V=baseline.std(0), baseline_mean_population_mean_V=baseline.mean(1).mean(),
        baseline_RMS_median_V=np.median(noise_RMS), rows=np.asarray(len(wave)))


def final_classifiers(reports, setup, p, out, device, progress, started):
    with np.load(out/'FINAL_TRAIN_0_common_prefix_codes.npz', allow_pickle=False) as archive:
        baseline_mean = archive['observed_baseline_mean_V'].copy()
    parent_ids = np.repeat(setup['data']['source_ids'], 4)
    results = {}
    (out/'C2ST').mkdir()
    for label, by_hypothesis in reports.items():
        results[label] = {}
        for h, report in by_hypothesis.items():
            encoding = report['private_ADC_encoding']
            if not encoding['reconstruction']['classifier_float32_bitwise']:
                raise RuntimeError('038 stored ADC reconstruction changed classifier float32 inputs')
            with np.load(out/encoding['path'], allow_pickle=False) as archive:
                fake = archive['raw_crop_adc_codes'].astype(np.float64)/ADC_LEVELS+ADC_MIN-baseline_mean[:, None]
            progress['saved_generated_code_archives_decoded'] += 1
            results[label][h] = classifier256(fake, setup['data']['real'], parent_ids, setup['data']['real_groups'],
                setup['statistics']['loss'], device, stream_seed(p['seeds'][0], 'C2ST', label, h),
                out/'C2ST'/f'{label}_{h}', progress, p, started)
            write_json(out/'CLASSIFIER_STATUS.json', dict(completed=progress['classifiers_completed'],
                label=label, hypothesis=h, execution_counts=dict(progress), primary_FINAL_already_saved=True))
    write_json(out/'C2ST_RESULTS.json', results)
    return results


def run038(p, out, device, progress, config_sha):
    from n01.evaluate_window008 import _load_dev
    started = time.monotonic()
    out.mkdir(parents=True)
    recovery = p.get('recovery')
    resume02 = recovery is not None and recovery['continuation_id'] == 'resume02_20261018'
    continuation = {}
    if recovery is not None:
        previous = json.loads(verified_path028(recovery['previous_config']).read_text(encoding='utf8'))
        for key, value in previous.items():
            if key not in ('root_design', 'previous_root_design', 'source_pins', 'resources',
                           'recovery', 'recovery_operation_budget') and p[key] != value:
                raise ValueError('038 continuation changed frozen setting: '+key)
        if any(p['resources'][key] != value for key, value in previous['resources'].items()
               if key not in ('GPU_headroom_basis', 'track_training_soft_seconds_total', 'reserve_for_unified_FINAL')):
            raise ValueError('038 continuation changed frozen resource limits')
        if p['resources']['track_training_soft_seconds_total'] != (900 if resume02 else 1800):
            raise ValueError('038 continuation training scope differs from remaining900s REAL tracks')
        if (p['previous_root_design'] != previous['root_design'] or
                set(p['source_pins']) != set(previous['source_pins']) or any(
                    digest != p['source_pins'][name] for name, digest in previous['source_pins'].items()
                    if name != 'scripts/run_response038.py')):
            raise ValueError('038 continuation changed frozen scientific source or previous design')
        prior_out = Path(recovery['previous_output_root'])
        for relative, record in recovery['preserved_files'].items():
            verified_path028(dict(record, path=str(prior_out/relative)))
        progress.update(recovery['prior_execution_counts_lower_bound'])
        continuation = dict(continuation_id=recovery['continuation_id'], teacher1_interrupted=True,
            all_four_tracks_complete=False, full_teacher_qualification_established=False,
            unified_scientific_acceptance_established=False,
            execution_counts_scope='attempt2 and explicit continuations verified lower bound; unpersisted restart tail excluded',
            old_unknown_tail=recovery['old_unknown_tail'], previous_runtime_seconds='unknown',
            prior_unpersisted_REAL_FINAL_DEV_execution='unknown', FINAL_first_historical_computation_claim=False,
            FINAL_stream_scope='original registered namespaces after current endpoint freeze; prior unpersisted calls unknown',
            new_runtime_clock_only=True, no_teacher_training_or_MONITOR_replay=True)
        if resume02:
            continuation.update(REAL_seed0_five_rejections_restored=True, REAL_seed0_training_replayed=False,
                REAL_seed0_post_rejection_RNG='not persisted; only initial checkpoint RNG is restored',
                remaining_response_training_track='REAL_seed1')
    write_json(out/'START.json', dict(config=p, config_sha256=config_sha, pid=os.getpid(),
        UTC=datetime.now(timezone.utc).isoformat(), device=str(device), scientific_dispatch_authority='ROOT frozen config',
        scientific_status='response_match_not_established', **continuation))
    write_json(out/'OPERATION_BUDGET.json', budget_closure038())
    if recovery is not None:
        write_json(out/'RECOVERY_OPERATION_BUDGET.json', p['recovery_operation_budget'])
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('038 pinned source differs: '+name)
    if sha256(ROOT/p['root_design']['path']) != p['root_design']['sha256']:
        raise ValueError('038 ROOT design differs')
    setup = load_setup(p, device, progress)
    write_json(out/'TRAIN_BINDINGS.json', dict(original_integral_statistics=setup['integral'],
        carrier_scale=setup['carrier_scale'], frozen_A_sha256=setup['frozen_A_digest'],
        frozen005_sha256=setup['frozen005_digest'], charge_calibration=setup['calibration'],
        noise_RMS_floor_V=setup['noise_RMS_floor_V'], REAL_known_flags=setup['real_reference_flags'],
        REAL_pre_ADC_status='unknown', source_physical_weights='unknown', source_count_weights='uniform',
        PCA013=p['PCA013'], no_PCA_refit=True, source_stratum_sizes=list(map(len, setup['source_strata'])),
        REAL_stratum_sizes=list(map(len, setup['real_strata'])), source_neural_identity_inputs=False))
    aggregate_REAL(setup['data'], out, 'TRAIN')
    if recovery is None:
        identity_gate(setup, p, out, progress)
    initial_states = [dict(label='A', model=setup['references'][0])]
    for prefix, charge in (('M0', False), ('Q1', True)):
        initial_states += [dict(label=f'{prefix}_seed{index}', model=model, shape_enabled=True,
            mean_enabled=False, charge_enabled=charge) for index, model in enumerate(setup['references'])]
    if recovery is None:
        evaluate_population(initial_states, setup['encoding'], setup['data']['real'], setup['data']['real_baseline100'],
            setup['extension'], p, setup['loss'], setup['context'], stream_seed(p['seeds'][0], 'INITIAL'),
            'INITIAL_TRAIN', out, progress, started)
        write_json(out/'G0_INITIAL_ALIASES.json', dict(aliases={f'G0_seed{index}': f'M0_seed{index}' for index in (0, 1)},
            exact_function_identity_proven_by='G0_ORIGINAL014_IDENTITY.json', new_evaluation_queries=0))
    else:
        for relative in recovery['reused_reports']:
            shutil.copy2(prior_out/relative, out/relative)
        write_json(out/'REUSED_INITIAL_PROVENANCE.json', dict(previous_output_root=str(prior_out),
            files={name: recovery['preserved_files'][name] for name in recovery['reused_reports']},
            new_identity_INITIAL_teacher_initial_queries=0, source_bytes_unchanged=True))
    teacher = make_teacher_context(setup, p, device, progress)
    if recovery is None:
        initial_teacher = teacher_initial_recovery(teacher['models'], teacher, p, out, progress)
    else:
        initial_teacher = json.loads((out/'TEACHER_INITIAL_ANALYTIC.json').read_text(encoding='utf8'))['seeds']
    tolerances = json.loads(setup['paths']['tolerances'].read_text(encoding='utf8'))['T_by_hypothesis']
    base_ctx = dict(loss=setup['loss'], carrier_scale=setup['carrier_scale'], extension=setup['extension'],
        context=setup['context'], noise_RMS_floor_V=setup['noise_RMS_floor_V'], tolerances=tolerances,
        output_root=out, deadline=started+p['resources']['internal_stop_save_seconds'])
    REAL_ctx = dict(base_ctx, encoding=setup['encoding'], real=setup['real'],
        source_strata=setup['source_strata'], target_strata=setup['real_strata'], teacher_factors=None,
        target_population=21047, target_groups=setup['data']['real_groups'])
    teacher_ctx = dict(base_ctx, encoding=teacher['encoding'], real={}, source_strata=teacher['source_strata'],
        target_strata=teacher['target_strata'], teacher_factors=teacher['factors'], target_population=128,
        target_groups=np.arange(128))
    tracks = {}
    populations = (('teacher', teacher_ctx, teacher['models']), ('REAL', REAL_ctx, setup['models']))
    if recovery is not None:
        restorations = (('teacher_seed0', 0, 'RESTORED_ENDPOINT.pt', teacher['models'][0]),
                        ('teacher_seed1', 1, 'RESTORED_ENDPOINT.pt', teacher['models'][1]),
                        ('REAL_seed0', 0, 'initial.pt', setup['models'][0])) if resume02 else (
                        ('teacher_seed0', 0, 'final.pt', teacher['models'][0]),
                        ('teacher_seed1', 1, 'best_accepted.pt', teacher['models'][1]))
        for name, index, filename, model in restorations:
            track_out = out/name
            shutil.copytree(prior_out/name, track_out)
            progress['checkpoint_loads'] += 1
            checkpoint = torch.load(prior_out/name/filename, map_location='cpu', weights_only=False)
            if (checkpoint['schema'] != 'n01/response038-new-module-checkpoint/v1' or
                    checkpoint['config'] != previous or checkpoint['resume_allowed'] is not False or
                    checkpoint['immutable_A'] != previous['inputs']['deterministic'] or
                    checkpoint['immutable_M0'] != previous['inputs'][f'M0_head_seed{index}'] or
                    checkpoint['immutable005'] != previous['inputs']['noise005']):
                raise ValueError('038 preserved endpoint checkpoint binding differs: '+name)
            copy_new_state(model, checkpoint['new_modules'])
            groups = model.parameter_groups()
            optimizer = torch.optim.AdamW([dict(params=groups['mean'], lr=p['training']['mean_lr']),
                dict(params=groups['shape'], lr=p['training']['shape_lr']),
                dict(params=groups['charge'], lr=p['training']['charge_lr'])], weight_decay=0.)
            optimizer.load_state_dict(checkpoint['optimizer'])
            mutable = copy.deepcopy(checkpoint['mutable'])
            model.set_phase(mutable['phase'], train_charge=mutable['charge_enabled']).train()
            torch.set_rng_state(checkpoint['CPU_RNG'])
            if checkpoint['CUDA_RNG'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['CUDA_RNG'])
            if name == 'REAL_seed0':
                records = [json.loads(line) for line in (track_out/'metrics.jsonl').read_text(encoding='utf8').splitlines()]
                last = records[-1]
                monitors = [json.loads(line) for line in (track_out/'MONITOR.jsonl').read_text(encoding='utf8').splitlines()]
                if (len(records) != 5 or [record['direction'] for record in records] != list(range(1, 6)) or
                        any(record['phase'] != 'mean' or record['accepted'] or record['accepted_updates'] != 0
                            for record in records) or len(monitors) != 1 or monitors[0]['index'] != 0 or
                        monitors[0]['accepted_updates'] != 0):
                    raise ValueError('038 REAL_seed0 is not the registered five rejected mean directions')
                mutable.update(phase='mean', direction=5, accepted_updates=0, consecutive_rejected=5,
                    consecutive_accepted=0, radius_LSB=last['radius_LSB_next'], last_monitor=last['last_monitor'],
                    monitor_index=last['last_monitor']['index']+1, charge_enabled=False)
                coverage = dict(scope='five complete rejected direction records; exact per-parent visits not saved',
                    source_unique_lower_bound=max(record['FIT']['coverage']['source_unique'] for record in records),
                    source_draws_lower_bound=sum(record['FIT']['coverage']['source_draws'] for record in records),
                    target_draws_lower_bound=256*sum(len(record['FIT']['per_microbank']['data']) for record in records),
                    completed_FIT_AD_queries=sum(len(record['FIT']['per_microbank']['data']) for record in records),
                    source_population=6593, target_population=21047, exact_per_parent_visits='unknown',
                    target_unique='unknown',
                    records=[dict(direction=record['direction'], coverage=record['FIT']['coverage']) for record in records])
                write_json(track_out/'FIT_COVERAGE_LOWER_BOUND.json', coverage)
                write_json(track_out/'MEAN_ENDPOINT_MONITOR_ALIAS.json', dict(source_file='MONITOR.jsonl',
                    source_index=0, bank_key=monitors[0]['bank_key'], new_NN_queries=0, new_AD_queries=0,
                    same_fixed_bank_and_initial_model=True, all_five_proposals_rejected=True,
                    original_report_reused=True, endpoint_report_evaluated_again=False,
                    reason='Original initial model/Adam restored after five rejected directions; fixed MONITOR unchanged.'))
                result = dict(track=name, learner_seed=p['seeds'][0], matching_M0_reference_index=0,
                    phases=dict(mean=dict(accepted=0, stop_reason='five_consecutive_no_accept', last_direction=5)),
                    charge_enabled=False, resource_stop=False, track_complete=True,
                    mean_to_random_supported=False, accepted_updates=0, elapsed_training_seconds=None,
                    training_wall_limit_seconds=900, source_FIT_population=6593, target_FIT_population=21047,
                    source_FIT_unique=None, target_FIT_unique=None, target_FIT_record_groups=None,
                    source_FIT_unique_lower_bound=coverage['source_unique_lower_bound'], coverage_scope=coverage['scope'],
                    original_complete_direction_records=5, continuation_training_directions=0,
                    native_report_failure_recovered='large MONITOR distance OOM; unchanged initial report is an explicit endpoint alias',
                    post_rejection_RNG='not saved; initial checkpoint RNG only, subsequent evaluation uses explicit keyed streams')
                write_json(track_out/'TRACK_RESULT.json', result)
            elif resume02 or index == 0:
                result = json.loads((track_out/'TRACK_RESULT.json').read_text(encoding='utf8'))
            else:
                records = [json.loads(line) for line in (track_out/'metrics.jsonl').read_text(encoding='utf8').splitlines()]
                last = records[-1]
                if (len(records) != 13 or last['phase'] != 'mean' or last['direction'] != 13 or
                        not last['accepted'] or last['accepted_updates'] != 5 or mutable['accepted_updates'] != 5):
                    raise ValueError('038 interrupted teacher endpoint is not the registered mean13/5accept')
                mutable['last_monitor'] = last['last_monitor']
                mutable['monitor_index'] = last['last_monitor']['index']+1
                coverage = dict(scope='complete thirteen direction records only; unknown in-flight tail excluded',
                    source_unique_lower_bound=max(record['FIT']['coverage']['source_unique'] for record in records),
                    source_draws_lower_bound=sum(record['FIT']['coverage']['source_draws'] for record in records),
                    target_draws_lower_bound=256*sum(len(record['FIT']['per_microbank']['data']) for record in records),
                    completed_FIT_AD_queries=sum(len(record['FIT']['per_microbank']['data']) for record in records),
                    source_population=128, target_population=128, exact_per_parent_visits='unknown',
                    target_unique='unknown', unknown_tail_visits='unknown',
                    records=[dict(direction=record['direction'], coverage=record['FIT']['coverage']) for record in records])
                write_json(track_out/'FIT_COVERAGE_LOWER_BOUND.json', coverage)
                result = dict(track=name, learner_seed=p['seeds'][index], matching_M0_reference_index=index,
                    phases=dict(mean=dict(accepted=5, stop_reason='SERVER_INTERRUPTED', last_direction=13)),
                    charge_enabled=False, resource_stop=False, server_interrupted=True, track_complete=False,
                    mean_to_random_supported=None, mean_to_random_status='not assessed after server interruption',
                    accepted_updates=5, elapsed_training_seconds=None,
                    training_wall_limit_seconds=900, source_FIT_population=128, target_FIT_population=128,
                    source_FIT_unique=None, target_FIT_unique=None, target_FIT_record_groups=None,
                    source_FIT_unique_lower_bound=coverage['source_unique_lower_bound'],
                    coverage_scope=coverage['scope'], exact_FIT_visits_and_old_runtime='unknown',
                    original_complete_direction_records=13, continuation_training_directions=0)
                write_json(track_out/'TRACK_RESULT.json', result)
            model.set_phase('frozen').eval()
            restored_path = track_out/('RESTORED_ENDPOINT_resume02.pt' if resume02 else 'RESTORED_ENDPOINT.pt')
            save_state(restored_path, model, optimizer, mutable, progress, p)
            if name == 'REAL_seed0' or (not resume02 and index == 1):
                shutil.copy2(restored_path, track_out/'final.pt')
                shutil.copy2(restored_path, track_out/'mean_endpoint.pt')
            result.update(state=new_module_state(model), mean_state=new_module_state(model))
            tracks[name] = result
            write_json(track_out/'ENDPOINT_RESTORATION.json', dict(source_checkpoint=filename,
                source_sha256=recovery['preserved_files'][f'{name}/{filename}']['sha256'],
                full_new_modules_Adam_mutable_CPU_CUDA_RNG_restored=True, teacher_training_queries=0,
                mutable_rebuilt_from_five_rejection_records=(name == 'REAL_seed0'),
                restored_RNG_scope='saved initial checkpoint, not post-five-rejection RNG' if name == 'REAL_seed0' else 'saved endpoint checkpoint',
                MONITOR_replayed=False, monitor_boundary=mutable['last_monitor'],
                monitor_index_reconciled_from_completed_log=(name == 'REAL_seed0' or (not resume02 and index == 1)),
                saved_phase=checkpoint['mutable']['phase'],
                saved_mode_and_requires_grad='not serialized; reconstructed from deterministic phase then frozen',
                lost_loss_diagnostic_cache='not serialized; report-only, not an input to subsequent computation',
                original_resume_allowed=False, explicit_one_time_authority=p['root_design'],
                old_checkpoint_bytes_unchanged=True, **continuation))
        populations = (('REAL', REAL_ctx, setup['models']),)
        write_json(out/'CONTINUATION_ENDPOINTS_PREPARED.json', dict(
            teacher_endpoints_restored=True, teacher_new_training_queries=0, REAL_tracks_started_in_this_invocation=False,
            previous_config_sha256=recovery['previous_config']['sha256'], execution_counts=dict(progress),
            **continuation))
    for population, context, models in populations:
        for index, seed in enumerate(p['seeds']):
            name = f'{population}_seed{index}'
            if resume02 and name == 'REAL_seed0':
                continue
            context = dict(context, M0=setup['references'][index])
            tracks[name] = run_track(models[index], context, seed, name, index, p, out/name, progress, started)
            write_json(out/'STATUS.json', dict(last_track=name,
                tracks={key: {field: value[field] for field in ('accepted_updates', 'resource_stop', 'charge_enabled')}
                        for key, value in tracks.items()}, execution_counts=dict(progress), elapsed=time.monotonic()-started,
                **continuation))
    endpoints = {name: dict(state_sha256=state_digest(value['state']), charge_enabled=value['charge_enabled'],
        accepted_updates=value['accepted_updates']) for name, value in tracks.items()}
    write_json(out/'ENDPOINTS_FROZEN.json', dict(endpoints=endpoints, config_sha256=config_sha,
        all_four_endpoints_frozen=True, FINAL_streams_opened=False, DEV_numeric_reads=0, no_settings_reopened=True,
        **(continuation if recovery is not None else dict(all_four_tracks_complete=True))))
    if recovery is not None:
        continuation['all_four_endpoints_frozen'] = True
    teacher_reports = {str(index): teacher_recovery(teacher['models'][index], teacher, index,
        tracks[f'teacher_seed{index}']['charge_enabled'], p, out, progress) for index in (0, 1)}
    if recovery is not None:
        for index in (0, 1):
            teacher_reports[str(index)].update(available_endpoint_only=True, track_interrupted=(index == 1),
                all_teacher_tracks_complete=False, full_teacher_qualification_established=False)
            write_json(out/f'TEACHER_seed{index}_RECOVERY.json', teacher_reports[str(index)])
    mean_models = []
    for index, seed in enumerate(p['seeds']):
        model = N01Response038(setup['baseline'], setup['heads'][index], seed=seed,
            condition_state=setup['condition'], charge_initialization=setup['calibration'])
        copy_new_state(model, tracks[f'REAL_seed{index}']['mean_state'])
        model.set_phase('frozen').eval()
        mean_models.append(model)
    states = list(initial_states)
    for prefix, models, charge in (('mean_endpoint', mean_models, False),
                                   ('final_chargeoff', setup['models'], False),
                                   ('final_full', setup['models'], None)):
        states += [dict(label=f'{prefix}_seed{index}', model=model, shape_enabled=True, mean_enabled=True,
            charge_enabled=tracks[f'REAL_seed{index}']['charge_enabled'] if charge is None else charge)
            for index, model in enumerate(models)]
    risk_states = [state for state in states if state['label'].startswith(('M0_', 'mean_endpoint_', 'final_full_'))]
    risk = final_risk_confirmation(risk_states, dict(REAL_ctx, M0=setup['references'][0]), p, out, progress)
    final = {}
    for bank in (0, 1):
        scope = f'FINAL_TRAIN_{bank}'
        final[scope] = evaluate_population(states, setup['encoding'], setup['data']['real'], setup['data']['real_baseline100'],
            setup['extension'], p, setup['loss'], setup['context'], stream_seed(p['seeds'][0], 'FINAL_TRAIN', bank),
            scope, out, progress, started, save_codes=True, source_h=setup['data']['h'])
    dev = _load_dev(setup['paths']['original_container'], setup['paths']['identities'], setup['base']['window'], float(setup['loss'].q_floor))
    progress['DEV_archives_decoded'] += 1
    with np.load(setup['paths']['dataset'], allow_pickle=False) as archive:
        if not (np.array_equal(archive['source_val_ids'], dev['source_ids'])
                and np.array_equal(archive['real_val_original_index'], dev['real_ids'])
                and np.array_equal(archive['real_val_group'], dev['real_groups'])
                and not np.intersect1d(setup['data']['source_ids'], dev['source_ids']).size
                and not np.intersect1d(setup['data']['real_ids'], dev['real_ids']).size):
            raise ValueError('038 original DEV identity isolation differs')
    progress['DEV_metadata_archives_decoded'] += 1
    aggregate_REAL(dev, out, 'DEV')
    dev_source = torch.as_tensor(dev['source_carrier384'], dtype=torch.float64, device=device)
    dev_encoding = encode_all(setup['references'][0], dev_source, 128, progress)
    final['FINAL_DEV'] = evaluate_population(states, dev_encoding, dev['real'], dev['real_baseline100'],
        setup['extension'], p, setup['loss'], setup['context'], stream_seed(p['seeds'][0], 'FINAL_DEV'),
        'FINAL_DEV', out, progress, started, save_codes=True, source_h=dev['h'])
    source_pairs = source_pair_summary(states, setup, p, out, progress)
    groups = np.unique(setup['data']['real_groups'])
    shuffled = np.random.default_rng(stream_seed(p['seeds'][0], 'FINAL_REAL_REAL')).permutation(groups)
    left = np.isin(setup['data']['real_groups'], shuffled[:len(groups)//2])
    RR = hard_metrics(setup['data']['real'][left], setup['data']['real_baseline100'][left],
        setup['data']['real'][~left], setup['data']['real_baseline100'][~left], setup['loss'], setup['context'])
    write_json(out/'FINAL_REAL_REAL.json', dict(report=RR, record_disjoint=True,
        left_groups=shuffled[:len(groups)//2], right_groups=shuffled[len(groups)//2:],
        prior_reference=p['prior_RR_reference'], thresholds_recalibrated=False))
    comparisons = compare_final(final, tolerances, risk)
    primary = dict(schema='n01/response038-primary-final/v1', comparisons=comparisons, teacher=teacher_reports,
        teacher_initial=initial_teacher, tracks={name: {key: value for key, value in result.items()
            if key not in ('state', 'mean_state')} for name, result in tracks.items()}, source_pairs=source_pairs,
        execution_counts=dict(progress), DEV_previously_exposed=True, DEV_used_for_tuning=False,
        independent_acquisition_acceptance=False, scientific_status='response_match_not_established', **continuation)
    write_json(out/'PRIMARY_FINAL.json', primary)
    classifiers = final_classifiers(final['FINAL_TRAIN_0'], setup, p, out, device, progress, started)
    if state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest'] or state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']:
        raise RuntimeError('038 immutable A0/005 changed')
    terminal = dict(primary)
    terminal.update(native_status='COMPLETE', C2ST=classifiers, execution_counts=dict(progress),
        elapsed_seconds=time.monotonic()-started, operation_budget=budget_closure038(), no_response_settings_reopened=True)
    write_json(out/'RESULT.json', terminal)
    write_json(out/'COMPLETE.json', dict(native_status='COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'),
        execution_counts=dict(progress), scientific_status='response_match_not_established', **continuation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256:
        raise ValueError('038 config SHA differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if p['status'] != 'frozen':
        raise ValueError('038 preparation draft cannot load CP/data or execute a scientific job')
    if p['schema'] != 'n01/response038-calibration-config/v1' or p['seeds'] != [2026101738, 2026101739]:
        raise ValueError('038 requires the one ROOT-reviewed fixed two-seed candidate')
    if p['operation_budget'] != budget_closure038():
        raise ValueError('038 registered operation closure differs')
    if p.get('recovery') is not None and (p['recovery']['mode'], p['recovery']['continuation_id']) not in (
            ('freeze_teacher_endpoints_then_REAL', 'resume01_20261018'),
            ('freeze_REAL_seed0_then_REAL_seed1', 'resume02_20261018')):
        raise ValueError('038 permits only the registered explicit continuation scope')
    if args.out.exists():
        raise FileExistsError(args.out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        physical = torch.cuda.get_device_properties(device).total_memory
        limit = p['resources']['GPU_GiB']*1024**3
        if physical < limit:
            raise ValueError('038 GPU has less memory than registered allocator cap')
        torch.cuda.set_per_process_memory_fraction(limit/physical, device)
    names = ('DEV_archives_decoded', 'DEV_metadata_archives_decoded', 'FIT_AD_queries', 'PCA_archives_decoded',
        'TRAIN_archives_decoded', 'accepted_optimizer_updates', 'allocated_banks', 'charge_activation_trials',
        'checkpoint_loads', 'classifier_AD_queries', 'classifier_bootstrap_resamples', 'classifier_forward_calls',
        'classifier_forward_rows', 'classifier_optimizer_steps', 'classifiers_completed', 'conditional_MC_rows',
        'directions', 'generated_observation_rows', 'noise005_calls', 'noise005_rows', 'noise005_rows_attempted', 'extension_rows',
        'old_A_calls', 'old_A_rows', 'optimizer_steps', 'original014_calls', 'original014_rows',
        'parameter_amplitude_trials', 'raw_geometry_probes', 'response_calls', 'response_rows',
        'saved_generated_code_archives_decoded', 'source_pair_cached_signal_rows', 'teacher_cached_signal_rows')
    progress = dict.fromkeys(names, 0)
    try:
        run038(p, args.out, device, progress, args.config_sha256)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            failure = dict(error_type=type(error).__name__, error=str(error), execution_counts=dict(progress),
                automatic_restart=False, scientific_status='response_match_not_established',
                completed_primary_FINAL=(args.out/'PRIMARY_FINAL.json').exists())
            if p.get('recovery') is not None:
                failure.update(continuation_id=p['recovery']['continuation_id'], teacher1_interrupted=True,
                    all_four_tracks_complete=False, all_four_endpoints_frozen=(args.out/'ENDPOINTS_FROZEN.json').exists(),
                    full_teacher_qualification_established=False, unified_scientific_acceptance_established=False,
                    execution_counts_scope='attempt2 and explicit continuations verified lower bound; unpersisted restart tail excluded',
                    prior_unpersisted_REAL_FINAL_DEV_execution='unknown', FINAL_first_historical_computation_claim=False,
                    old_unknown_tail=p['recovery']['old_unknown_tail'])
            write_json(args.out/'FAILED.json', failure)
            if failure['completed_primary_FINAL']:
                result = json.loads((args.out/'PRIMARY_FINAL.json').read_text(encoding='utf8'))
                result.update(native_status='INCOMPLETE_AFTER_PRIMARY_FINAL', terminal_failure=failure,
                              execution_counts=dict(progress))
                write_json(args.out/'RESULT.json', result)


if __name__ == '__main__':
    main()
