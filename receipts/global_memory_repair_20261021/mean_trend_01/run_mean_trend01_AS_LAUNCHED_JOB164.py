"""Bounded conditioning-path repair; all acceptance uses native hard readout."""
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
    GlobalRepairResponse01, adc_margin_risk, add_peak_selection_gradient, joint_secant_weights)
from n01.mean_trend01 import ConditionedMeanRepair01, MeanTrendGrid01
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
        raise TimeoutError('Registered mean-trend wall budget exhausted')
    if counts['response_rows'] > p['operation_budget']['response_rows']:
        raise RuntimeError('Registered response-row budget exhausted')
    if counts['conditioned_mean_rows'] > p['operation_budget']['conditioned_mean_rows']:
        raise RuntimeError('Registered conditioned-mean budget exhausted')


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
        if isinstance(current.mean, ConditionedMeanRepair01):
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


def metrics(means):
    avg = (means[0] + means[1]) / 2
    result = {k: float(v) for k, v in memory.losses(avg, avg).items()}
    s = memory.sections['native_joint']
    raw = memory.joint.cholesky @ (avg - memory.target)[s]
    variance = memory.joint.cholesky.square().sum(1)
    for name, selection in memory.joint.block_slices.items():
        result['block_' + name] = float((raw[selection].square() / variance[selection]).mean())
    result.update(zip(memory.wave_cell_names, memory.conditional_wave_risks(*means).tolist()))
    return result


def trend_values():
    with torch.no_grad():
        values = grid.losses(model.mean) / trend_reference
    return values


def gradients(paired):
    started = time.monotonic()
    means, flags, rails, _ = pair(model, paired)
    model.mean.cached.requires_grad_(True)
    coarse = [model.theta, model.mean.cached]
    coefs = dict(objective=memory.coefficients(*means), wave=memory.positive_wave_coefficients(*means),
                 Q_wave=memory.conditional_wave_coefficients(*means))
    for name in p['active_cells']:
        index = memory.wave_cell_names.index(name)
        s = memory.sections['Q_wave']
        selected = slice(s.start + 56 * index, s.start + 56 * (index + 1))
        c = torch.zeros_like(memory.target)
        c[selected] = ((means[0] + means[1]) / 2 - memory.target)[selected] / (56 * memory.wave_denominator[index].square())
        coefs[name] = (c, c)
    accumulated = {k: [torch.zeros_like(v) for v in coarse] for k in [*coefs, 'ADC0', 'ADC1']}
    for bank_index, bank in enumerate(paired):
        n = sum(len(b['rows']) for b in bank)
        for b in bank:
            for start in range(0, len(b['rows']), p['micro_rows']):
                obs = observed(model, b, slice(start, start + p['micro_rows']), grad=True)
                f = memory.sums(obs) / n
                for key, coefficients in coefs.items():
                    g = torch.autograd.grad((coefficients[bank_index] * f).sum(), coarse, retain_graph=True)
                    for total, value in zip(accumulated[key], g):
                        total.add_(value)
                g = torch.autograd.grad(adc_margin_risk(obs['pre_adc_carrier']).sum() / n, coarse)
                for total, value in zip(accumulated['ADC' + str(bank_index)], g):
                    total.add_(value)
                counts['adjoint_microblocks'] += 1
    mapped = model.mean.map_adjoints({k: v[1] for k, v in accumulated.items()}, parents, p['mean_chunk'])
    result = {k: [v[0], *mapped[k]] for k, v in accumulated.items()}
    counts['mean_adjoint_rows'] += len(parents) * len(accumulated)
    curve = grid.losses(model.mean) / trend_reference
    tg = torch.autograd.grad(curve.mean(), [v for v in model.mean.parameters() if v.requires_grad])
    result['trend'] = [torch.zeros_like(model.theta), *tg]
    result['objective'] = [a + b for a, b in zip(result['objective'], result['trend'])]
    counts['conditioned_mean_rows'] = model.mean.forward_rows
    budget()
    return result, means, flags, rails, curve.detach(), time.monotonic() - started


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
        mean = ConditionedMeanRepair01(baseline, enc, filtered, inherited.mean)
        model = GlobalRepairResponse01(setup['joint_models'][0], torch.cat((inherited.theta, inherited.noise_coordinate)), mean)
        mean.refresh(parents, p['mean_chunk'])
        identity = max(float((mean(rows) - inherited.mean(rows)).abs().max()) for rows in parents.split(128))
        assert identity <= p['identity_tolerance_V']
        grid = MeanTrendGrid01(mean, parents, points=p['trend_points'])
        trend_reference = grid.losses(mean).detach()
        assert bool((trend_reference > 0).all())
        excess_reference = grid.excess(mean).detach()
        original = GlobalRepairResponse01(setup['joint_models'][0], theta)
        paired = banks(p['draws_per_parent'], p['bank_seed'])
        values, flags, _, _ = pair(original, paired)
        memory.normalize_from_initial(*values)
        old_memory = torch.load(p['replay_memory'], map_location=device, weights_only=False)
        prefix = len(old_memory['target'])
        assert torch.equal(memory.target[:prefix], old_memory['target'])
        assert torch.equal(memory.weights[:prefix], old_memory['weights'])
        counts['memory_checkpoint_loads'] += 1
        torch.save(memory.export(), args.out / 'TRAIN_DISTRIBUTION_MEMORY.pt')
        write_json(args.out / 'BINDINGS.json', dict(source_FIT=6465, REAL_FIT=18948, REPORT_training_rows=0,
            inherited_mean_identity_max_V=identity, inherited_updates=10,
            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
            trainable_names=[k for k, v in model.named_parameters() if v.requires_grad],
            trend_reference=trend_reference.tolist(), source_groups=[len(v) for v in grid.groups],
            fixed_window_excess_reference=excess_reference.tolist(),
            fixed_memory_prefix_equal=True, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'CONDITIONAL_WAVE_MEMORY.json', memory.conditional_wave_summary())
        progress_file(args.out, counts, 'mean_condition_path_and_TRAIN_memory_ready', elapsed_seconds=time.monotonic() - began)
    named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    sizes = [p['learning_rates']['response'] if k == 'theta' else
             p['learning_rates']['conditioning'] if k.startswith('mean.unet.') else p['learning_rates']['mean'] for k, _ in named]
    optimizer = torch.optim.Adam([dict(params=[v], lr=s) for (_, v), s in zip(named, sizes)])
    supported = [k for k, ok in zip(memory.wave_cell_names, memory.wave_supported.tolist()) if ok]
    guards = dict(Q_mass=1.005, block_wave=1., block_peak_codes=1.01, Q_wave=1., **{k: 1. for k in supported})
    accepted, rejected, steps = 0, 0, []
    for iteration in range(p['directions']):
        paired = banks(p['draws_per_parent'], p['direction_seed'] + iteration * 1000000)
        before, opt_before = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        grads, values, old_flags, old_rails, old_curve, gradient_seconds = gradients(paired)
        previous = metrics(values)
        old_objective = float(sum(memory.losses(*values).values())) + float(old_curve.mean())
        for (_, value), grad in zip(named, grads['objective']):
            value.grad = grad
        optimizer.step()
        counts['optimizer_directions'] += 1
        with torch.no_grad():
            model.theta.clamp_(-1., 1.)
        delta = [model.state_dict()[k] - before[k] for k, _ in named]
        norm = sum((d / s).double().square().sum() for d, s in zip(delta, sizes)).sqrt()
        directions = {'Adam_objective': delta}
        for label in ['wave', 'Q_wave', 'trend', *p['active_cells'], 'ADC0', 'ADC1']:
            g = grads[label]
            magnitude = sum((s * v).double().square().sum() for s, v in zip(sizes, g)).sqrt()
            if float(magnitude) > 0:
                directions[label] = [-norm / magnitude * s * s * v for s, v in zip(sizes, g)]
        model.load_state_dict(before)
        probes, actual = [], []
        for label, direction in directions.items():
            state = copy.deepcopy(before)
            for (key, _), d in zip(named, direction):
                state[key] = before[key] + d
            state['theta'].clamp_(-1., 1.)
            actual.append([state[k] - before[k] for k, _ in named])
            model.load_state_dict(state)
            values, flags, rails, _ = pair(model, paired)
            curve = trend_values()
            probes.append(dict(direction=label, positive_metrics=metrics(values), flags=flags,
                ADC_margin_risk=rails, trend=curve.tolist(),
                objective=float(sum(memory.losses(*values).values())) + float(curve.mean())))
            counts['native_secant_probes'] += 1
        model.load_state_dict(before)
        constraint_names = [*guards, 'PSD_CDF', 'trend_group0', 'trend_group1', 'ADC0', 'ADC1']
        changes = [[r['positive_metrics'][key] - previous[key] for r in probes] for key in [*guards, 'PSD_CDF']]
        allowances = [(factor - 1) * previous[key] for key, factor in guards.items()] + [0.]
        for group in range(2):
            changes.append([r['trend'][group] - float(old_curve[group]) for r in probes])
            allowances.append(0.)
        for bank in range(2):
            changes.append([r['ADC_margin_risk'][bank] - old_rails[bank] for r in probes])
            allowances.append(0.)
        objective_changes = np.array([r['objective'] - old_objective for r in probes])
        weights = joint_secant_weights(objective_changes, changes, allowances)
        predicted = float(objective_changes @ weights)
        torch.save(dict(before_trainable={k: before[k] for k, _ in named}, directions=actual,
            names=[k for k, _ in named], coefficients=weights, iteration=iteration + 1), args.out / f'DIRECTION_{iteration + 1:02d}.pt')
        passed, trials = False, []
        if predicted < 0:
            for scale in p['step_scales']:
                state = copy.deepcopy(before)
                for j, (key, _) in enumerate(named):
                    state[key] = before[key] + scale * sum(float(w) * d[j] for w, d in zip(weights, actual))
                state['theta'].clamp_(-1., 1.)
                model.load_state_dict(state)
                values, flags, rails, _ = pair(model, paired)
                now, curve = metrics(values), trend_values()
                objective = float(sum(memory.losses(*values).values())) + float(curve.mean())
                checks = {key: now[key] <= factor * previous[key] for key, factor in guards.items()}
                checks.update(PSD_improves=now['PSD_CDF'] < previous['PSD_CDF'],
                    Q_wave_improves=now['Q_wave'] < previous['Q_wave'], objective_improves=objective < old_objective,
                    trend_each_decreases=bool((curve < old_curve).all()),
                    ADC_margin_not_worse=all(rails[b] <= old_rails[b] for b in range(2)),
                    flags_not_increased=all(flags[b][k] <= old_flags[b][k] for b in range(2) for k in old_flags[b]))
                passed = all(checks.values())
                trials.append(dict(scale=scale, positive_metrics=now, trend=curve.tolist(),
                    ADC_margin_risk=rails, flags=flags, checks=checks, accepted=passed))
                counts['joint_hard_trials'] += 1
                if passed:
                    accepted += 1
                    counts['committed_optimizer_updates'] += 1
                    rejected = 0
                    break
        if not passed:
            model.load_state_dict(before)
            optimizer.load_state_dict(opt_before)
            rejected += 1
        record = dict(iteration=iteration + 1, gradient_seconds=gradient_seconds, previous_metrics=previous,
            previous_trend=old_curve.tolist(), previous_flags=old_flags, previous_ADC_margin_risk=old_rails,
            probes=probes, constraint_names=constraint_names, constraint_changes=changes, constraint_allowances=allowances,
            coefficients=weights.tolist(), predicted_objective_change=predicted, trials=trials,
            accepted=passed, new_committed_updates=accepted, elapsed_seconds=time.monotonic() - began)
        steps.append(record)
        with (args.out / 'TRAINING_STEPS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), new_committed_updates=accepted,
            inherited_updates=10, parent_checkpoint=p['replay_checkpoint'], direction_iteration=iteration + 1), args.out / 'B_CURRENT.pt')
        counts['conditioned_mean_rows'] = model.mean.forward_rows
        budget()
        progress_file(args.out, counts, 'mean_trend_direction_complete', **{k: record[k] for k in
            ('iteration', 'accepted', 'new_committed_updates', 'elapsed_seconds')})
        if rejected >= p['stop_consecutive_rejections']:
            break
    paired = banks(p['endpoint_draws'], p['endpoint_seed'])
    arrays = dict(source_parent_rows=parent_array, real_rows=real_rows,
                  **{'real_' + k: real[k].cpu().numpy() for k in ('Q', 'tail', 'PSD_raw')})
    endpoint = {}
    for name, current in [('initial', original), ('inherited', inherited), ('B', model)]:
        values, flags, rails, records = pair(current, paired, collect=True)
        endpoint[name] = dict(positive_metrics=metrics(values), flags=flags, ADC_margin_risk=rails)
        for b, record in enumerate(records):
            arrays.update({f'{name}_bank{b}_{k}': v for k, v in record.items()})
        progress_file(args.out, counts, 'mean_trend_endpoint_complete', arm=name, elapsed_seconds=time.monotonic() - began)
    with torch.no_grad():
        curve, excess = trend_values(), grid.excess(model.mean)
        arrays['B_mean_Q'] = model.mean.cached[parents, 110:366].sum(1).cpu().numpy()
        arrays['B_mean_PSD'] = (model.mean.cached[parents, 156:366].sum(1) / model.mean.cached[parents, 110:366].sum(1)).cpu().numpy()
    a, b, old = [endpoint[k]['positive_metrics'] for k in ('initial', 'B', 'inherited')]
    checks = dict(new_update=accepted > 0, PSD_CDF_at_least_10pct_improvement=b['PSD_CDF'] <= .9 * a['PSD_CDF'],
        Q_mass_not_worse_2pct=b['Q_mass'] <= 1.02 * a['Q_mass'], wave_not_worse_2pct=b['block_wave'] <= 1.02 * a['block_wave'],
        peak_codes_not_worse_2pct=b['block_peak_codes'] <= 1.02 * a['block_peak_codes'],
        flags_not_increased=all(endpoint['B']['flags'][i][k] <= endpoint['initial']['flags'][i][k]
            for i in range(2) for k in endpoint['initial']['flags'][i]),
        Q_wave_improves=b['Q_wave'] < old['Q_wave'], supported_cells_not_worse_2pct=all(b[k] <= 1.02 * old[k] for k in supported),
        curvature_each_halved=bool((curve <= .5).all()), comb_excess_each_halved=bool((excess <= .5 * excess_reference).all()))
    endpoint['B'].update(pilot_checks=checks, passes_bounded_pilot=all(checks.values()))
    result = dict(status='COMPLETE', endpoint=endpoint, trend_relative=curve.tolist(),
        fixed_window_excess=excess.tolist(), fixed_window_excess_reference=excess_reference.tolist(),
        new_committed_optimizer_updates=accepted, direction_records=len(steps), inherited_updates=10,
        report_role_rows_used=0, scientific_acceptance=False, independent_final_random_banks=True)
    write_json(args.out / 'TRAINING_RESULT.json', result)
    np.savez(args.out / 'ENDPOINT_SCALARS.npz', **arrays)
    setup['references'][0].assert_frozen_unchanged(deep=True)
    counts['conditioned_mean_rows'] = model.mean.forward_rows
    budget()
    write_json(args.out / 'RESULT.json', dict(status='COMPLETE', elapsed_seconds=time.monotonic() - began,
        execution_counts=dict(counts), committed_optimizer_updates=accepted,
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), config_sha256=args.config_sha256,
        scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - began, automatic_restart=False))
