"""Bounded spline-mean restoration; frozen FIT memory and fixed final endpoint."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import n01.inference028 as inference028

inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from n01.data import write_json
from n01.global_memory01 import (GlobalDistributionMemory01, MeanHeadRepair01,
    GlobalRepairResponse01, adc_margin_risk, add_peak_selection_gradient)
from n01.mean_trend01 import ConditionedMeanRepair01, MeanTrendGrid01
from n01.charge_spline01 import ChargeSplineMean01
from n01.model008 import canonical_charge008, relative_time_warp008
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
from run_joint_response01 import load_joint_setup, random_bank, progress_file

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--config-sha256', required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
p = json.loads(args.config.read_bytes())
args.out.mkdir(parents=True, exist_ok=False)
counts, began = Counter(), time.monotonic()
torch.set_num_threads(p['resources']['CPU_threads'])
torch.use_deterministic_algorithms(True)
device = torch.device(args.device)
torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
    torch.cuda.get_device_properties(device).total_memory, device)


def budget():
    if time.monotonic() - began > p['resources']['internal_seconds']:
        raise TimeoutError('Registered spline-mean wall budget exhausted')
    for key in ('response_rows', 'conditioned_mean_rows', 'mean_adjoint_rows', 'optimizer_updates'):
        if counts[key] > p['operation_budget'][key]:
            raise RuntimeError('Registered budget exhausted: ' + key)


def banks(draws, seed):
    result = []
    for bank_index in range(2):
        chunks = []
        for j, rows in enumerate(parents.split(p['parent_chunk'])):
            b = random_bank(setup, rows, draws, seed + 100000 * bank_index + 1000 * j, counts)
            b['noise'] = b['noise'][p['hypothesis']]
            chunks.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in b.items()})
        result.append(chunks)
    return result


def observed(current, bank, selection, grad=False):
    rows, z = bank['rows'][selection].to(device), bank['z'][selection].to(device)
    counts['response_rows'] += len(rows)
    counts['response_calls'] += 1
    budget()
    signal = current(rows, z)
    obs = observe_generated_carrier(signal + current.noise_scale() * bank['noise'][selection].to(device),
        ste=grad, q_floor=q_floor, return_carrier=True)
    if grad:
        counts['STE_rows'] += len(rows)
        obs = add_peak_selection_gradient(obs)
    return obs


def pair(current, paired, collect=False):
    means, flags, rails, arrays = [], [], [], []
    with torch.no_grad():
        if isinstance(current.mean, ChargeSplineMean01):
            current.mean.refresh(parents, p['mean_chunk'])
        for bank in paired:
            total, flag, rail, records = torch.zeros_like(memory.target), Counter(), 0., []
            n = sum(len(b['rows']) for b in bank)
            for b in bank:
                for start in range(0, len(b['rows']), p['micro_rows']):
                    obs = observed(current, b, slice(start, start + p['micro_rows']))
                    total += memory.sums(obs)
                    flag.update({k: int(v.sum()) for k, v in obs['flags'].items()})
                    rail += float(adc_margin_risk(obs['pre_adc_carrier']).sum())
                    if collect:
                        records.append({k: obs[k].cpu().numpy() for k in ('Q', 'tail', 'PSD_raw', 'peak_index')})
            means.append(total / n)
            flags.append(dict(flag))
            rails.append(rail / n)
            arrays.append({k: np.concatenate([r[k] for r in records]) for k in records[0]} if collect else None)
    return means, flags, rails, arrays


def metric_tensors(means):
    avg = (means[0] + means[1]) / 2
    result = memory.losses(avg, avg)
    s = memory.sections['native_joint']
    raw = memory.joint.cholesky @ (avg - memory.target)[s]
    variance = memory.joint.cholesky.square().sum(1)
    for name, selection in memory.joint.block_slices.items():
        result['block_' + name] = (raw[selection].square() / variance[selection]).mean()
    result.update(zip(memory.wave_cell_names, memory.conditional_wave_risks(*means)))
    return result


def metrics(means):
    return {k: float(v) for k, v in metric_tensors(means).items()}


def memory_objective(means):
    distribution = sum(memory.losses(*means).values())
    positive = metric_tensors(means)
    global_penalty = sum((positive[k] / bound - 1).relu().square() for k, bound in global_limits.items())
    local_penalty = torch.stack([(positive[k] / cell_limits[k] - 1).relu().square() for k in supported]).mean()
    return distribution + p['constraint_weight'] * (global_penalty + local_penalty)


def gradient(paired):
    started = time.monotonic()
    means, flags, rails, _ = pair(model, paired)
    variables = [v.detach().requires_grad_(True) for v in means]
    loss = memory_objective(variables)
    coefficients = torch.autograd.grad(loss, variables)
    model.mean.cached.requires_grad_(True)
    coarse = [model.theta, model.mean.cached]
    accumulated = [torch.zeros_like(v) for v in coarse]
    for bank_index, bank in enumerate(paired):
        n = sum(len(b['rows']) for b in bank)
        for b in bank:
            for start in range(0, len(b['rows']), p['micro_rows']):
                obs = observed(model, b, slice(start, start + p['micro_rows']), grad=True)
                objective = (coefficients[bank_index] * memory.sums(obs)).sum() / n
                objective += p['adc_weight'] * adc_margin_risk(obs['pre_adc_carrier']).sum() / (2 * n)
                values = torch.autograd.grad(objective, coarse)
                for total, value in zip(accumulated, values):
                    total.add_(value)
                counts['adjoint_microblocks'] += 1
    mapped = model.mean.map_adjoints({'objective': accumulated[1]}, parents, p['mean_chunk'])['objective']
    counts['mean_adjoint_rows'] += len(parents)
    curve = grid.losses(model.mean) / trend_reference
    trend_grad = torch.autograd.grad(p['trend_weight'] * curve.mean(), trainable_mean)
    result = [accumulated[0], *[a + b for a, b in zip(mapped, trend_grad)]]
    assert all(bool(torch.isfinite(g).all()) for g in result)
    counts['conditioned_mean_rows'] = model.mean.forward_rows
    budget()
    return result, dict(positive_metrics=metrics(means), flags=flags, ADC_margin_risk=rails,
        trend_relative=curve.detach().tolist(), objective=float(loss.detach()) + p['trend_weight'] * float(curve.detach().mean())
        + p['adc_weight'] * sum(rails) / 2, gradient_seconds=time.monotonic() - started)


try:
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        with np.load(p['roles_file'], allow_pickle=False) as z:
            parent_array, real_rows = z['source_train_rows'].copy(), z['real_train_rows'].copy()
            assert not np.intersect1d(parent_array, z['source_heldout_rows']).size
            assert not np.intersect1d(real_rows, z['real_heldout_rows']).size
        assert len(parent_array) == 6465 and len(real_rows) == 18948
        parents = torch.as_tensor(parent_array, device=device)
        q_floor = float(setup['loss'].q_floor)
        real_cpu = observe_existing_carrier(torch.as_tensor(setup['data']['real_carrier384'][real_rows]),
                                            q_floor=q_floor, return_carrier=True)
        real = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in real_cpu.items()}
        real['flags'] = {k: v.to(device) for k, v in real_cpu['flags'].items()}
        memory = GlobalDistributionMemory01(real, q_floor=q_floor, **p['memory'])
        counts['REAL_memory_rows'] = len(real_rows)
        enc, baseline = setup['encoding'], setup['baseline']
        centered = enc['source'] - enc['source'][:, :100].mean(1, keepdim=True)
        unit = centered / centered.amax(1, keepdim=True)
        filtered = torch.cat([baseline.response_filter(unit[s:s + 128, None]) for s in range(0, len(unit), 128)])
        with np.load(p['candidate_arrays'][0], allow_pickle=False) as z:
            theta = torch.as_tensor(z['candidate'].copy(), device=device)
        inherited = GlobalRepairResponse01(setup['joint_models'][0], theta, MeanHeadRepair01(baseline, enc, filtered))
        saved = torch.load(p['replay_checkpoint'], map_location=device, weights_only=False)
        counts['repair_checkpoint_loads'] += 1
        assert saved['accepted_updates'] == p['replay_accepted_updates']
        inherited.load_state_dict(saved['model'], strict=True)
        inherited.requires_grad_(False)
        reference_mean = ConditionedMeanRepair01(baseline, enc, filtered, inherited.mean).requires_grad_(False)
        reference_mean.bind_reference(inherited.mean, parents, p['mean_chunk'])
        reference_mean.refresh(parents, p['mean_chunk'])
        grid = MeanTrendGrid01(reference_mean, parents, points=p['trend_points'])
        trend_reference = grid.losses(reference_mean).detach()
        assert bool((trend_reference > 0).all())
        excess_reference = grid.excess(reference_mean).detach()
        counts['reference_mean_rows'] = reference_mean.forward_rows
        del reference_mean
        mean = ChargeSplineMean01(baseline, enc, filtered, inherited.mean, parents)
        mean.prepare(parents, p['mean_chunk'])
        model = GlobalRepairResponse01(setup['joint_models'][0], torch.cat((inherited.theta, inherited.noise_coordinate)), mean)
        mean.refresh(parents, p['mean_chunk'])
        identity = 0.
        for rows in parents.split(p['mean_chunk']):
            residual = .20 * inherited.mean.mean_head(mean.decoded_anchor[rows]).tanh().double() * mean.gate
            scale = torch.exp(np.log(2.) * inherited.mean.mean_time(mean.scalars[rows]).squeeze(1).tanh().double())
            warped, _ = relative_time_warp008(filtered[rows], scale)
            raw = warped * mean.gate + residual
            control = (enc['original']['qbar'][rows, None, None] * raw / canonical_charge008(raw)[..., None])[:, 0]
            identity = max(identity, float((mean(rows) - control).abs().max()))
        assert identity <= p['anchor_identity_tolerance_V']
        initial_state = copy.deepcopy(model.state_dict())
        torch.save(dict(model=initial_state, inherited_updates=10, optimizer_updates=0,
            parent_checkpoint=p['replay_checkpoint'], architecture=p['architecture']), args.out / 'CANDIDATE_INITIAL.pt')
        original = GlobalRepairResponse01(setup['joint_models'][0], theta)
        paired = banks(p['draws_per_parent'], p['bank_seed'])
        values, flags, rails, _ = pair(original, paired)
        memory.normalize_from_initial(*values)
        original_metrics = metrics(values)
        old_memory = torch.load(p['replay_memory'], map_location=device, weights_only=False)
        prefix = len(old_memory['target'])
        assert torch.equal(memory.target[:prefix], old_memory['target'])
        assert torch.equal(memory.weights[:prefix], old_memory['weights'])
        counts['memory_checkpoint_loads'] += 1
        old_values, old_flags, old_rails, _ = pair(inherited, paired)
        inherited_metrics = metrics(old_values)
        supported = [k for k, ok in zip(memory.wave_cell_names, memory.wave_supported.tolist()) if ok]
        global_limits = {k: 1.02 * original_metrics[k] for k in ('Q_mass', 'block_wave', 'block_peak_codes')}
        global_limits['Q_wave'] = inherited_metrics['Q_wave']
        cell_limits = {k: 1.02 * inherited_metrics[k] for k in supported}
        values, flags, rails, _ = pair(model, paired)
        initial_curve = grid.losses(mean) / trend_reference
        initial_excess = grid.excess(mean)
        torch.save(memory.export(), args.out / 'TRAIN_DISTRIBUTION_MEMORY.pt')
        write_json(args.out / 'BINDINGS.json', dict(source_FIT=6465, REAL_FIT=18948, REPORT_training_rows=0,
            anchor_identity_max_V=identity, inherited_updates=10, spline_logQ_bounds=mean.logq_bounds.tolist(),
            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
            trainable_names=[k for k, v in model.named_parameters() if v.requires_grad],
            trend_reference=trend_reference.tolist(), source_groups=[len(v) for v in grid.groups],
            fixed_window_excess_reference=excess_reference.tolist(), fixed_memory_prefix_equal=True,
            original_metrics=original_metrics, inherited_metrics=inherited_metrics,
            anchored_initial_metrics=metrics(values), anchored_initial_flags=flags, anchored_initial_ADC_margin_risk=rails,
            anchored_initial_trend_relative=initial_curve.tolist(), anchored_initial_excess=initial_excess.tolist(),
            global_soft_limits=global_limits, cell_soft_limits=cell_limits,
            NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'CONDITIONAL_WAVE_MEMORY.json', memory.conditional_wave_summary())
        progress_file(args.out, counts, 'spline_anchor_and_FIT_memory_ready', elapsed_seconds=time.monotonic() - began)
    named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    assert [k for k, _ in named] == ['theta', 'mean.mean_head.weight', 'mean.mean_head.bias']
    trainable_mean = [v for v in mean.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam([dict(params=[v], lr=p['learning_rates']['response' if k == 'theta' else 'mean']) for k, v in named])
    for iteration in range(p['steps']):
        paired = banks(p['draws_per_parent'], p['training_seed'] + iteration * 1000000)
        grads, record = gradient(paired)
        for (_, value), grad in zip(named, grads):
            value.grad = grad
        optimizer.step()
        with torch.no_grad():
            model.theta.clamp_(-1., 1.)
        counts['optimizer_updates'] += 1
        record.update(iteration=iteration + 1, metrics_are_before_this_update=True,
            optimizer_updates=counts['optimizer_updates'], elapsed_seconds=time.monotonic() - began,
            update_is_candidate_only=True, scientific_acceptance=False)
        with (args.out / 'TRAINING_STEPS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            optimizer_updates=counts['optimizer_updates'], inherited_updates=10,
            parent_checkpoint=p['replay_checkpoint'], architecture=p['architecture']), args.out / 'B_CURRENT.pt')
        budget()
        progress_file(args.out, counts, 'spline_training_step_complete', **{k: record[k] for k in
            ('iteration', 'optimizer_updates', 'gradient_seconds', 'elapsed_seconds')})
    final_state = copy.deepcopy(model.state_dict())
    paired = banks(p['endpoint_draws'], p['endpoint_seed'])
    arrays = dict(source_parent_rows=parent_array, real_rows=real_rows,
                  **{'real_' + k: real[k].cpu().numpy() for k in ('Q', 'tail', 'PSD_raw')})
    endpoint = {}
    for name, current in [('initial', original), ('inherited', inherited), ('anchored', model), ('B', model)]:
        if name in ('anchored', 'B'):
            model.load_state_dict(initial_state if name == 'anchored' else final_state)
        values, flags, rails, records = pair(current, paired, collect=True)
        endpoint[name] = dict(positive_metrics=metrics(values), flags=flags, ADC_margin_risk=rails)
        for bank_index, record in enumerate(records):
            arrays.update({f'{name}_bank{bank_index}_{k}': v for k, v in record.items()})
            for key in ('Q', 'tail', 'PSD_raw'):
                sample = record[key].reshape(len(parents), p['endpoint_draws'])
                arrays[f'{name}_bank{bank_index}_parent_{key}_mean'] = sample.mean(1)
                arrays[f'{name}_bank{bank_index}_parent_{key}_std'] = sample.std(1, ddof=1)
        if name in ('anchored', 'B'):
            arrays[name + '_mean_Q'] = mean.cached[parents, 110:366].sum(1).detach().cpu().numpy()
            arrays[name + '_mean_PSD'] = (mean.cached[parents, 156:366].sum(1) /
                mean.cached[parents, 110:366].sum(1)).detach().cpu().numpy()
        progress_file(args.out, counts, 'spline_endpoint_complete', arm=name, elapsed_seconds=time.monotonic() - began)
    with torch.no_grad():
        curve, excess = grid.losses(mean) / trend_reference, grid.excess(mean)
    a, b, old = [endpoint[k]['positive_metrics'] for k in ('initial', 'B', 'inherited')]
    checks = dict(new_update=counts['optimizer_updates'] > 0, PSD_CDF_at_least_10pct_improvement=b['PSD_CDF'] <= .9 * a['PSD_CDF'],
        Q_mass_not_worse_2pct=b['Q_mass'] <= 1.02 * a['Q_mass'], wave_not_worse_2pct=b['block_wave'] <= 1.02 * a['block_wave'],
        peak_codes_not_worse_2pct=b['block_peak_codes'] <= 1.02 * a['block_peak_codes'],
        flags_not_increased=all(endpoint['B']['flags'][i][k] <= endpoint['initial']['flags'][i][k]
            for i in range(2) for k in endpoint['initial']['flags'][i]),
        Q_wave_improves=b['Q_wave'] < old['Q_wave'], supported_cells_not_worse_2pct=all(b[k] <= 1.02 * old[k] for k in supported),
        curvature_each_halved=bool((curve <= .5).all()), comb_excess_each_halved=bool((excess <= .5 * excess_reference).all()))
    endpoint['B'].update(pilot_checks=checks, passes_bounded_pilot=all(checks.values()))
    result = dict(status='COMPLETE', mode='charge_spline_mean_repair', endpoint=endpoint,
        trend_relative=curve.tolist(), fixed_window_excess=excess.tolist(),
        fixed_window_excess_reference=excess_reference.tolist(), optimizer_updates=counts['optimizer_updates'],
        fixed_final_step=p['steps'], endpoint_selection=False, inherited_updates=10, report_role_rows_used=0,
        scientific_acceptance=False, independent_final_random_banks=True, conditional_moments_use_actual_random_draws=True)
    write_json(args.out / 'TRAINING_RESULT.json', result)
    np.savez(args.out / 'ENDPOINT_SCALARS.npz', **arrays)
    setup['references'][0].assert_frozen_unchanged(deep=True)
    counts['conditioned_mean_rows'] = mean.forward_rows
    budget()
    write_json(args.out / 'RESULT.json', dict(status='COMPLETE', elapsed_seconds=time.monotonic() - began,
        execution_counts=dict(counts), optimizer_updates=counts['optimizer_updates'],
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), config_sha256=args.config_sha256,
        scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - began, automatic_restart=False))
