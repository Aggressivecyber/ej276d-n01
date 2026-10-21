"""Diagnose gradient conflict and Adam conditioning at the fixed152-step endpoint."""
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
chunk_count = (6465 + p['parent_chunk'] - 1) // p['parent_chunk']
training_seeds = {base + iteration * 1000000 + bank * 100000 + chunk * 1000 + offset
    for base, steps in [(2246102200, 24), (2406102200, 128)]
    for iteration in range(steps) for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
diagnostic_seeds = {p['diagnostic_seed'] + bank * 100000 + chunk * 1000 + offset
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
old_endpoint_and_normalization = {base + bank * 100000 + chunk * 1000 + offset
    for base in [p['bank_seed'], 2266102200, 2706102200]
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
assert not diagnostic_seeds.intersection(training_seeds | old_endpoint_and_normalization)

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
    for key in p['operation_budget']:
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


def component_gradients(paired):
    started = time.monotonic()
    means, flags, rails, _ = pair(model, paired)
    variables = [v.detach().requires_grad_(True) for v in means]
    losses = memory.losses(*variables)
    positive = metric_tensors(variables)
    objectives = dict(distribution=sum(losses.values()), PSD_CDF=losses['PSD_CDF'],
        global_penalty=p['constraint_weight'] * sum((positive[k] / bound - 1).relu().square() for k, bound in global_limits.items()),
        local_penalty=p['constraint_weight'] * torch.stack([(positive[k] / cell_limits[k] - 1).relu().square() for k in supported]).mean())
    coefficients = {k: torch.autograd.grad(v, variables, retain_graph=True) for k, v in objectives.items()}
    model.mean.cached.requires_grad_(True)
    coarse = [model.theta, model.mean.cached]
    keys = list(objectives) + ['ADC']
    accumulated = {k: [torch.zeros_like(v) for v in coarse] for k in keys}
    for bank_index, bank in enumerate(paired):
        n = sum(len(b['rows']) for b in bank)
        for b in bank:
            for begin in range(0, len(b['rows']), p['micro_rows']):
                obs = observed(model, b, slice(begin, begin + p['micro_rows']), grad=True)
                sums = memory.sums(obs)
                for i, key in enumerate(keys):
                    objective = p['adc_weight'] * adc_margin_risk(obs['pre_adc_carrier']).sum() / (2 * n) if key == 'ADC' else (coefficients[key][bank_index] * sums).sum() / n
                    values = torch.autograd.grad(objective, coarse, retain_graph=i + 1 < len(keys))
                    for total, value in zip(accumulated[key], values):
                        total.add_(value)
                    counts['adjoint_loss_rows'] += len(obs['Q'])
                counts['adjoint_microblocks'] += 1
    mapped = model.mean.map_adjoints({k: v[1] for k, v in accumulated.items()}, parents, p['mean_chunk'])
    counts['mean_adjoint_rows'] += len(parents) * len(keys)
    curve = grid.losses(model.mean) / trend_reference
    trend_grad = torch.autograd.grad(p['trend_weight'] * curve.mean(), trainable_mean)
    gradients = {k: [accumulated[k][0], *mapped[k]] for k in keys}
    gradients['trend'] = [torch.zeros_like(model.theta), *trend_grad]
    combined = [sum(gradients[k][i] for k in ['distribution', 'global_penalty', 'local_penalty', 'ADC', 'trend']) for i in range(len(named))]
    assert all(bool(torch.isfinite(g).all()) for values in gradients.values() for g in values)
    counts['conditioned_mean_rows'] = model.mean.forward_rows
    budget()
    return gradients, combined, dict(positive_metrics=metrics(means), flags=flags, ADC_margin_risk=rails,
        components={k: float(v.detach()) for k, v in objectives.items()} | dict(ADC=p['adc_weight'] * sum(rails) / 2, trend=p['trend_weight'] * float(curve.detach().mean())),
        objective=float(memory_objective(variables).detach()) + p['trend_weight'] * float(curve.detach().mean()) + p['adc_weight'] * sum(rails) / 2,
        gradient_seconds=time.monotonic() - started)


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
        resume = torch.load(p['initialization_checkpoint'], map_location=device, weights_only=False)
        counts['candidate_checkpoint_loads'] += 1
        assert resume['optimizer_updates'] == p['inherited_candidate_updates']
        model.load_state_dict(resume['model'], strict=True)
        assert all(torch.equal(v, resume['model'][k]) for k, v in model.state_dict().items())
        start_state = copy.deepcopy(model.state_dict())
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
            candidate_start_metrics=metrics(values), candidate_start_flags=flags, candidate_start_ADC_margin_risk=rails,
            candidate_start_trend_relative=initial_curve.tolist(), candidate_start_excess=initial_excess.tolist(),
            initialization_checkpoint=p['initialization_checkpoint'], inherited_candidate_updates=p['inherited_candidate_updates'],
            resumed_parameters_exactly_equal=True, diagnostic_seeds_disjoint_from_training_and_prior_endpoints=True,
            global_soft_limits=global_limits, cell_soft_limits=cell_limits,
            NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'CONDITIONAL_WAVE_MEMORY.json', memory.conditional_wave_summary())
        progress_file(args.out, counts, 'spline_resume_and_FIT_memory_ready', elapsed_seconds=time.monotonic() - began)
    named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    assert [k for k, _ in named] == ['theta', 'mean.mean_head.weight', 'mean.mean_head.bias']
    trainable_mean = [v for v in mean.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam([dict(params=[v], lr=p['learning_rates']['response' if k == 'theta' else 'mean']) for k, v in named])
    optimizer.load_state_dict(resume['optimizer'])
    assert all(group['lr'] == p['learning_rates']['response' if k == 'theta' else 'mean']
               for group, (k, _) in zip(optimizer.param_groups, named))
    paired = banks(p['draws_per_parent'], p['diagnostic_seed'])
    gradients, combined, baseline_record = component_gradients(paired)
    progress_file(args.out, counts, 'component_gradient_complete', elapsed_seconds=time.monotonic() - began,
                  gradient_seconds=baseline_record['gradient_seconds'], diagnostic_proposals_remaining=4)
    flat = {k: torch.cat([v.detach().double().flatten() for v in values]) for k, values in gradients.items()}
    flat['total'] = torch.cat([v.detach().double().flatten() for v in combined])
    blocks = dict(all=slice(0, 1019), deterministic=slice(0, 24), shared_L=slice(24, 45), M0_scale=slice(45, 53), spline_head=slice(53, 1019))
    geometry = {}
    for name, selection in blocks.items():
        g = {k: v[selection] for k, v in flat.items()}
        geometry[name] = dict(norm={k: float(v.norm()) for k, v in g.items()},
            cosine={a + '__' + b: float(torch.dot(g[a], g[b]) / (g[a].norm() * g[b].norm()))
                    if float(g[a].norm() * g[b].norm()) else None
                    for a, b in [('ADC', 'PSD_CDF'), ('ADC', 'global_penalty'), ('distribution', 'global_penalty'), ('total', 'PSD_CDF')]})
    proposals, delta_arrays = {}, {}
    for proposal in p['diagnostic_proposals']:
        model.load_state_dict(start_state)
        groups = [dict(params=[v], lr=group['lr'], betas=group['betas'], eps=group['eps'])
                  for (_, v), group in zip(named, resume['optimizer']['param_groups'])]
        trial_optimizer = torch.optim.Adam(groups)
        if proposal == 'saved_adam':
            trial_optimizer.load_state_dict(copy.deepcopy(resume['optimizer']))
        for (name, value), grad in zip(named, combined):
            enabled = not (proposal == 'reset_mean_only' and name == 'theta') and not (proposal == 'reset_response_only' and name != 'theta')
            value.grad = grad.detach().clone() if enabled else None
        trial_optimizer.step()
        with torch.no_grad():
            model.theta.clamp_(-1., 1.)
        counts['optimizer_updates'] += 1
        delta = torch.cat([(v.detach() - start_state[k]).double().flatten() for k, v in named])
        delta_arrays[proposal] = delta.cpu().numpy()
        values, flags, rails, _ = pair(model, paired)
        with torch.no_grad():
            curve = grid.losses(mean) / trend_reference
            loss = float(memory_objective(values)) + p['trend_weight'] * float(curve.mean()) + p['adc_weight'] * sum(rails) / 2
        proposals[proposal] = dict(objective=loss, positive_metrics=metrics(values), flags=flags, ADC_margin_risk=rails,
            trend_relative=curve.tolist(), objective_change=loss - baseline_record['objective'],
            delta_norms={name: float(delta[selection].norm()) for name, selection in blocks.items()},
            predicted_component_changes={k: float(torch.dot(v, delta)) for k, v in flat.items()},
            temporary_proposal_only=True, committed_training_updates=0)
        counts['conditioned_mean_rows'] = model.mean.forward_rows
        budget()
        progress_file(args.out, counts, 'diagnostic_proposal_complete', proposal=proposal, elapsed_seconds=time.monotonic() - began)
    model.load_state_dict(start_state)
    assert all(torch.equal(v, start_state[k]) for k, v in model.state_dict().items())
    setup['references'][0].assert_frozen_unchanged(deep=True)
    np.savez(args.out / 'COMPONENT_GRADIENTS_AND_PROPOSALS.npz', **{k: v.cpu().numpy() for k, v in flat.items()},
             **{'delta_' + k: v for k, v in delta_arrays.items()})
    write_json(args.out / 'DIAGNOSTIC_RESULT.json', dict(status='COMPLETE', baseline=baseline_record, geometry=geometry,
        proposals=proposals, committed_training_updates=0, temporary_optimizer_proposals=4, endpoint_selection=False,
        original_checkpoint_not_modified=True, parameter_state_restored=True, report_role_rows_used=0,
        scientific_acceptance=False, diagnostic_seed=p['diagnostic_seed']))
    write_json(args.out / 'RESULT.json', dict(status='COMPLETE', elapsed_seconds=time.monotonic() - began,
        execution_counts=dict(counts), committed_training_updates=0, temporary_optimizer_proposals=4,
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), config_sha256=args.config_sha256,
        scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - began, automatic_restart=False))
