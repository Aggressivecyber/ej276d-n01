"""Bounded N01 global-memory profile or paired terminal-head repair pilot."""
from __future__ import annotations

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

# User storage policy: actual input reads, no NAS path/size/hash verification.
inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from n01.data import write_json
from n01.global_memory01 import (GlobalDistributionMemory01, MeanHeadRepair01,
                                 GlobalRepairResponse01, add_peak_selection_gradient,
                                 adc_margin_risk, joint_secant_weights)
from n01.observation008 import observe_existing_carrier, observe_generated_carrier, crop_indices_torch
from run_joint_response01 import load_joint_setup, random_bank, progress_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    p = json.loads(args.config.read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(
        p['resources']['GPU_GiB'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    fixed_readout = False
    fixed_transform_peak = None

    def check_budget(rows=0):
        if time.monotonic() - started > p['resources']['internal_seconds']:
            raise TimeoutError('Registered wall budget exhausted')
        if counts['response_rows'] + rows > p['operation_budget']['response_rows']:
            raise RuntimeError('Registered response-row budget exhausted')

    def observed(model, bank, selection, *, grad=False):
        rows, z = bank['rows'][selection].to(device), bank['z'][selection].to(device)
        check_budget(len(rows))
        counts['response_calls'] += 1
        counts['response_rows'] += len(rows)
        signal = model(rows, z) if fixed_transform_peak is None else model(rows, z, fixed_peak=fixed_transform_peak)
        counts['hard_observation_rows'] += len(rows)
        if grad:
            counts['STE_rows'] += len(rows)
        obs = observe_generated_carrier(signal + model.noise_scale() * bank['noise'][selection].to(device),
                                        ste=grad, q_floor=q_floor, return_carrier=True)
        if grad and p.get('peak_selection_gradient', False):
            obs = add_peak_selection_gradient(obs)
        if fixed_readout:
            peaks = bank['fixed_readout_peak'][selection].to(device)
            window = (obs['carrier_adc'] - obs['baseline_mean_V'][:, None]).gather(1, crop_indices_torch(peaks))
            q, tail = window.sum(1), window[:, 46:].sum(1)
            safe_q = torch.where(q != 0, q, torch.ones_like(q))
            protected = torch.where(q < 0, -torch.full_like(q, q_floor), torch.full_like(q, q_floor))
            obs.update(window_V=window, waveform256=window, Q=q, tail=tail, peak_index=peaks,
                PSD_raw=torch.where(q != 0, tail / safe_q, torch.full_like(q, float('nan'))),
                PSD=tail / torch.where(q.abs() < q_floor, protected, q))
            obs['flags'].update(Q_nonpositive=q <= 0, PSD_undefined=q == 0, PSD_guarded=q.abs() < q_floor)
        return obs

    def banks(draws, seed, *, cpu=False):
        result = []
        for arm_bank in range(2):
            chunks = []
            for j, start in enumerate(range(0, len(parents), p['parent_chunk'])):
                b = random_bank(setup, parents[start:start + p['parent_chunk']], draws,
                                seed + 100000 * arm_bank + 1000 * j, counts)
                b['noise'] = b['noise'][p['hypothesis']]
                if cpu:
                    for key in ('rows', 'z', 'noise'):
                        b[key] = b[key].cpu()
                chunks.append(b)
            result.append(chunks)
        return result

    def bank_mean(model, bank, *, collect=False):
        total, flags, records = torch.zeros_like(memory.target), Counter(), []
        rail_risk = 0.
        count = sum(len(b['rows']) for b in bank)
        for b in bank:
            for start in range(0, len(b['rows']), p['micro_rows']):
                obs = observed(model, b, slice(start, start + p['micro_rows']))
                total += memory.sums(obs)
                flags.update({k: int(v.sum()) for k, v in obs['flags'].items()})
                if p['mode'] == 'joint_secant_pilot':
                    rail_risk += float(adc_margin_risk(obs['pre_adc_carrier']).sum())
                if collect:
                    records.append({k: obs[k].detach().cpu().numpy()
                                    for k in ('Q', 'tail', 'PSD_raw', 'peak_index')})
        arrays = {k: np.concatenate([r[k] for r in records]) for k in records[0]} if collect else None
        return total / count, dict(flags), arrays, rail_risk / count

    def pair_means(model, paired):
        with torch.no_grad():
            results = [bank_mean(model, b) for b in paired]
        model.adc_margin_risks = [r[3] for r in results]
        return [r[0] for r in results], [r[1] for r in results]

    def gradient(model, paired):
        torch.cuda.synchronize()
        stamp = time.monotonic()
        model.zero_grad(set_to_none=True)
        means, flags = pair_means(model, paired)
        coefficients = memory.coefficients(*means)
        project_wave = p['mode'] in ('projected_pilot', 'guard_diagnostic', 'joint_secant_pilot')
        diagnose_guard = p['mode'] in ('guard_diagnostic', 'joint_secant_pilot')
        parameters = [v for v in model.parameters() if v.requires_grad]
        if project_wave:
            wave_coefficients = memory.wave_coefficients(*means)
            accumulated = [torch.zeros_like(v) for v in parameters]
            wave_gradient = [torch.zeros_like(v) for v in parameters]
            if diagnose_guard:
                positive_coefficients = memory.positive_wave_coefficients(*means)
                positive_gradient = [torch.zeros_like(v) for v in parameters]
        for bank_index, (bank, coefficient) in enumerate(zip(paired, coefficients)):
            count = sum(len(b['rows']) for b in bank)
            for b in bank:
                for start in range(0, len(b['rows']), p['micro_rows']):
                    obs = observed(model, b, slice(start, start + p['micro_rows']), grad=True)
                    features = memory.sums(obs) / count
                    loss = (coefficient * features).sum()
                    if project_wave:
                        g = torch.autograd.grad(loss, parameters, retain_graph=True)
                        w = torch.autograd.grad((wave_coefficients[bank_index] * features).sum(),
                                                parameters, retain_graph=diagnose_guard)
                        for accumulator, wave_accumulator, gv, wv in zip(accumulated, wave_gradient, g, w):
                            accumulator.add_(gv)
                            wave_accumulator.add_(wv)
                        counts['wave_gradient_microblocks'] += 1
                        if diagnose_guard:
                            pg = torch.autograd.grad((positive_coefficients[bank_index] * features).sum(), parameters)
                            for accumulator, gv in zip(positive_gradient, pg):
                                accumulator.add_(gv)
                            counts['positive_wave_gradient_microblocks'] += 1
                    else:
                        loss.backward()
                    counts['backward_microblocks'] += 1
        if project_wave:
            for value, grad in zip(parameters, accumulated):
                value.grad = grad
            model.wave_gradient = wave_gradient
            if diagnose_guard:
                model.positive_wave_gradient = positive_gradient
        torch.cuda.synchronize()
        return means, flags, time.monotonic() - stamp

    def numbers(means):
        return {k: float(v) for k, v in memory.losses(*means).items()}

    def positive_metrics(means):
        avg = (means[0] + means[1]) / 2
        result = numbers([avg, avg])
        selection = memory.sections['native_joint']
        raw = memory.joint.cholesky @ (avg - memory.target)[selection]
        variance = memory.joint.cholesky.square().sum(1)
        for name, selected in memory.joint.block_slices.items():
            result['block_' + name] = float((raw[selected].square() / variance[selected]).mean())
        return result

    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            with np.load(p['roles_file'], allow_pickle=False) as archive:
                parent_array = archive['source_train_rows'].copy()
                real_rows = archive['real_train_rows'].copy()
                source_held = archive['source_heldout_rows'].copy()
                real_held = archive['real_heldout_rows'].copy()
            assert not np.intersect1d(parent_array, source_held).size
            assert not np.intersect1d(real_rows, real_held).size
            assert len(parent_array) == 6465 and len(real_rows) == 18948
            parents = torch.as_tensor(parent_array, device=device)
            q_floor = float(setup['loss'].q_floor)
            # Stored REAL canonical codes were defined by the CPU readout.
            # Replay there first, then transfer observables without re-digitizing.
            real_cpu = observe_existing_carrier(torch.as_tensor(
                setup['data']['real_carrier384'][real_rows]),
                q_floor=q_floor, return_carrier=True)
            real = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in real_cpu.items()}
            real['flags'] = {k: v.to(device) for k, v in real_cpu['flags'].items()}
            counts['REAL_memory_rows'] = len(real_rows)
            memory = GlobalDistributionMemory01(real, q_floor=q_floor, **p['memory'])
            encoding, baseline = setup['encoding'], setup['baseline']
            source = encoding['source']
            centered = source - source[:, :100].mean(1, keepdim=True)
            amplitude = centered.amax(1)
            filtered = torch.cat([baseline.response_filter(
                (centered[s:s + 128] / amplitude[s:s + 128, None])[:, None])
                for s in range(0, len(source), 128)])
            with np.load(p['candidate_arrays'][0], allow_pickle=False) as z:
                theta = torch.as_tensor(z['candidate'].copy(), device=device)
            mean = MeanHeadRepair01(baseline, encoding, filtered)
            models = dict(A=GlobalRepairResponse01(setup['joint_models'][0], theta),
                          B=GlobalRepairResponse01(setup['joint_models'][0], theta, mean))
            identity = max(float((mean(rows) - setup['joint_models'][0].cache['A'][rows]).abs().max())
                           for rows in parents.split(p['micro_rows']))
            if identity > p['identity_tolerance_V']:
                raise ValueError(f'Trainable copied mean does not reconstruct A0: {identity}')
            write_json(args.out / 'BINDINGS.json', dict(
                source_fit_rows=len(parents), real_fit_rows=len(real_rows),
                report_rows_used_for_memory=0, q_floor=q_floor,
                copied_mean_identity_max_V=identity, hypothesis=p['hypothesis'],
                seed_head=0, trainable_parameters={k: sum(x.numel() for x in m.parameters()
                    if x.requires_grad) for k, m in models.items()},
                NAS_file_IO_checks=False, NAS_file_hash_verification=False))
            progress_file(args.out, counts, 'TRAIN_memory_and_copied_heads_ready')
            paired = banks(p['draws_per_parent'], p['bank_seed'])
            initial, flags = pair_means(models['A'], paired)
            memory.normalize_from_initial(*initial)
            # Same initialization and data, identical frozen objective in both arms.
            torch.save(memory.export(), args.out / 'TRAIN_DISTRIBUTION_MEMORY.pt')
            np.savez(args.out / 'TRAIN_ROLE_ROWS.npz', source_fit=parent_array, real_fit=real_rows)
            initial_losses = numbers(initial)
            write_json(args.out / 'INITIAL.json', dict(losses=initial_losses, flags=flags,
                objective_normalization=memory.normalization))
            if p['mode'] in ('projected_pilot', 'endpoint_replay', 'guard_diagnostic', 'readout_diagnostic', 'joint_secant_pilot'):
                models = {'B': models['B']}

        if p['mode'] == 'profile':
            profile = dict(identity_max_V=identity, arms={}, initial_losses=initial_losses)
            for name, model in models.items():
                means, flags, elapsed = gradient(model, paired)
                grads = {k: dict(norm=float(v.grad.norm()), max_abs=float(v.grad.abs().max()))
                         for k, v in model.named_parameters() if v.requires_grad}
                assert all(np.isfinite(v['norm']) for v in grads.values())
                assert grads['theta']['norm'] > 0
                if name == 'B':
                    assert grads['mean.mean_head.weight']['norm'] > 0
                    assert grads['mean.mean_time.2.weight']['norm'] > 0
                # One registered Adam direction and three hard microbank probes,
                # always restored; these are not committed training updates.
                optimizer = torch.optim.Adam([
                    {'params': [model.theta], 'lr': p['learning_rates']['response']},
                    *([{'params': [v for k, v in model.named_parameters()
                                   if k != 'theta' and v.requires_grad],
                        'lr': p['learning_rates']['mean']}] if name == 'B' else [])])
                before = copy.deepcopy(model.state_dict())
                optimizer.step()
                with torch.no_grad():
                    model.theta.clamp_(-1., 1.)
                proposed = copy.deepcopy(model.state_dict())
                micro = [[bank[0]] for bank in paired]
                trials = []
                for scale in (0., 1., .5, .25):
                    model.load_state_dict({k: before[k] + scale * (proposed[k] - before[k])
                                           for k in before})
                    trial, trial_flags = pair_means(model, micro)
                    trials.append(dict(scale=scale, losses=numbers(trial), flags=trial_flags))
                model.load_state_dict(before)
                counts['temporary_Adam_directions'] += 1
                profile['arms'][name] = dict(full_population_two_pass_gradient_seconds=elapsed,
                                             gradients=grads, hard_microbank_trials=trials)
                progress_file(args.out, counts, 'profile_arm_complete', arm=name)
            with torch.no_grad():
                precision_banks = banks(p['precision_draws'], p['bank_seed'] + 10000000)
                precise, _ = pair_means(models['A'], precision_banks)
                ps = memory.sections['PSD_CDF']
                reference = ((precise[0] + precise[1]) / 2 - memory.target)[ps]
                difference = (((initial[0] + initial[1]) - (precise[0] + precise[1])) / 2)[ps]
                relative = float((memory.weights[ps] * difference.square()).sum().sqrt() /
                                 (memory.weights[ps] * reference.square()).sum().sqrt())
                profile['precision'] = dict(low_draws=p['draws_per_parent'],
                    high_draws=p['precision_draws'], relative_PSD_feature_mean_change=relative,
                    denominator='initial model-vs-REAL PSD memory discrepancy',
                    low_draws_supported_for_pilot=relative <= p['precision_relative_limit'])
            write_json(args.out / 'PROFILE.json', profile)
        elif p['mode'] in ('endpoint_replay', 'guard_diagnostic', 'readout_diagnostic', 'joint_secant_pilot'):
            # Load the fixed state. Endpoint replay constructs no optimizer.
            saved = torch.load(p['replay_checkpoint'], map_location=device, weights_only=False)
            counts['repair_checkpoint_loads'] += 1
            assert saved['accepted_updates'] == p['replay_accepted_updates']
            assert saved['direction_iteration'] == p['replay_direction_iteration']
            models['B'].load_state_dict(saved['model'], strict=True)
            saved_memory = torch.load(p['replay_memory'], map_location=device, weights_only=False)
            counts['memory_checkpoint_loads'] += 1
            assert torch.equal(memory.target, saved_memory['target'])
            assert torch.equal(memory.weights, saved_memory['weights'])
            accepted = Counter(B=saved['accepted_updates'])
            steps = [json.loads(line) for line in Path(p['replay_steps']).read_text().splitlines()]
            assert len(steps) == saved['direction_iteration']
            progress_file(args.out, counts, 'saved_endpoint_loaded_without_optimizer')
            if p['mode'] == 'guard_diagnostic':
                del paired
                paired = banks(p['draws_per_parent'], p['bank_seed'] + 2000000)
                model = models['B']
                means, old_flags, elapsed = gradient(model, paired)
                previous, guard_previous = numbers(means), positive_metrics(means)
                assert max(abs(guard_previous[k] - steps[1]['previous_metrics'][k])
                           for k in guard_previous) < 1e-8
                before = copy.deepcopy(model.state_dict())
                named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
                sizes = [p['learning_rates']['response'] if k == 'theta'
                         else p['learning_rates']['mean'] for k, _ in named]
                optimizer = torch.optim.Adam([
                    {'params': [model.theta], 'lr': p['learning_rates']['response']},
                    {'params': [v for k, v in named if k != 'theta'], 'lr': p['learning_rates']['mean']}])
                optimizer.load_state_dict(saved['optimizer'])
                optimizer.step()
                counts['temporary_Adam_directions'] += 1
                with torch.no_grad():
                    model.theta.clamp_(-1., 1.)
                proposal = copy.deepcopy(model.state_dict())
                displacement = [proposal[k] - before[k] for k, _ in named]
                directions = {}
                cross_label = 'cross_risk_tangent' if p.get('peak_selection_gradient', False) else 'original_cross_tangent'
                for label, grads in ((cross_label, model.wave_gradient),
                                     ('aligned_positive_tangent', model.positive_wave_gradient)):
                    derivative = sum((g * d).double().sum() for g, d in zip(grads, displacement))
                    denominator = sum((s * g).double().square().sum() for s, g in zip(sizes, grads))
                    correction = max(float(derivative), 0.) / float(denominator)
                    directions[label] = [d - correction * s * s * g
                                         for d, s, g in zip(displacement, sizes, grads)]
                norm = sum((d / s).double().square().sum() for d, s in zip(displacement, sizes)).sqrt()
                wave_norm = sum((s * g).double().square().sum()
                                for s, g in zip(sizes, model.positive_wave_gradient)).sqrt()
                directions['positive_wave_descent'] = [-norm / wave_norm * s * s * g
                                                       for s, g in zip(sizes, model.positive_wave_gradient)]
                torch.save(dict(before=before, directions=directions,
                    names=[k for k, _ in named], objective_gradient=[v.grad for _, v in named],
                    cross_wave_gradient=model.wave_gradient, positive_wave_gradient=model.positive_wave_gradient),
                    args.out / 'DIRECTION_DIAGNOSTIC.pt')
                model.load_state_dict(before)
                with torch.no_grad():
                    base_results = [bank_mean(model, b, collect=True) for b in paired]
                trials = []
                for label, delta in directions.items():
                    state = copy.deepcopy(before)
                    for (k, _), d in zip(named, delta):
                        state[k] = before[k] + d
                    state['theta'].clamp_(-1., 1.)
                    actual_delta = [state[k] - before[k] for k, _ in named]
                    model.load_state_dict(state)
                    with torch.no_grad():
                        results = [bank_mean(model, b, collect=True) for b in paired]
                    losses = numbers([r[0] for r in results])
                    metrics = positive_metrics([r[0] for r in results])
                    guards = {k: metrics[k] <= factor * guard_previous[k]
                              for k, factor in p['optimizer']['per_step_guards'].items()}
                    flags = [r[1] for r in results]
                    flags_pass = all(flags[b][k] <= old_flags[b][k] for b in range(2) for k in old_flags[b])
                    record = dict(direction=label, losses=losses, positive_metrics=metrics,
                        wave_derivative_cross=float(sum((g * d).double().sum()
                            for g, d in zip(model.wave_gradient, actual_delta))),
                        wave_derivative_positive=float(sum((g * d).double().sum()
                            for g, d in zip(model.positive_wave_gradient, actual_delta))),
                        hard_wave_change=metrics['block_wave'] - guard_previous['block_wave'],
                        guards=guards, flags=flags, flags_pass=flags_pass,
                        objective_decreased=sum(losses.values()) < sum(previous.values()),
                        peak_changes_by_bank=[int(np.count_nonzero(r[2]['peak_index'] != b[2]['peak_index']))
                                              for r, b in zip(results, base_results)])
                    trials.append(record)
                    progress_file(args.out, counts, 'guard_diagnostic_trial_complete', direction=label)
                if not p.get('peak_selection_gradient', False):
                    assert abs(trials[0]['positive_metrics']['block_wave'] -
                               steps[1]['trials'][0]['positive_metrics']['block_wave']) < 1e-8
                model.load_state_dict(before)
                write_json(args.out / 'GUARD_DIAGNOSTIC.json', dict(
                    replayed_original_rejection=not p.get('peak_selection_gradient', False),
                    peak_selection_gradient=p.get('peak_selection_gradient', False),
                    original_hard_baseline_reproduced=True, gradient_seconds=elapsed,
                    before_losses=previous, before_metrics=guard_previous, before_flags=old_flags, trials=trials,
                    committed_optimizer_updates=0, weights_restored=True, report_rows_used=0,
                    representation='Same247 parameters; frozen decoder/FiLM and M0 bases, original mean terminal heads.',
                    scientific_acceptance=False))
                setup['references'][0].assert_frozen_unchanged(deep=True)
            elif p['mode'] == 'readout_diagnostic':
                del paired
                paired = banks(p['draws_per_parent'], p['bank_seed'] + 2000000)
                model = models['B']
                direction = torch.load(p['direction_file'], map_location=device, weights_only=False)
                counts['diagnostic_direction_loads'] += 1
                before = copy.deepcopy(model.state_dict())
                assert all(torch.equal(before[k], direction['before'][k]) for k in before)
                with torch.no_grad():
                    base_results = [bank_mean(model, b, collect=True) for b in paired]
                    transform_peaks = torch.cat([model.mean(rows)[:, 112:144].argmax(1).double() + 112.
                        for rows in torch.arange(len(source), device=device).split(p['micro_rows'])])
                counts['diagnostic_mean_rows'] += len(source)
                for bank, base_result in zip(paired, base_results):
                    offset = 0
                    for b in bank:
                        b['fixed_readout_peak'] = torch.from_numpy(
                            base_result[2]['peak_index'][offset:offset + len(b['rows'])]).to(device)
                        offset += len(b['rows'])
                base_metrics = positive_metrics([r[0] for r in base_results])
                proposed = copy.deepcopy(before)
                for name, delta in zip(direction['names'], direction['directions']['positive_wave_descent']):
                    proposed[name] = before[name] + delta
                proposed['theta'].clamp_(-1., 1.)
                model.load_state_dict(proposed)
                trials = []
                for label in ('native', 'fixed_readout_peak', 'fixed_readout_and_transform_peaks'):
                    fixed_readout = label != 'native'
                    fixed_transform_peak = transform_peaks if label == 'fixed_readout_and_transform_peaks' else None
                    with torch.no_grad():
                        values = [bank_mean(model, b, collect=True) for b in paired]
                    metrics = positive_metrics([r[0] for r in values])
                    trials.append(dict(observation=label, positive_metrics=metrics,
                        wave_change=metrics['block_wave'] - base_metrics['block_wave'],
                        wave_change_percent=100 * (metrics['block_wave'] / base_metrics['block_wave'] - 1),
                        diagnostic_only=True, admissible_endpoint=label == 'native'))
                    progress_file(args.out, counts, 'fixed_peak_diagnostic_complete', observation=label)
                original = json.loads(Path(p['original_guard_diagnostic']).read_bytes())
                assert abs(trials[0]['positive_metrics']['block_wave'] -
                           original['trials'][2]['positive_metrics']['block_wave']) < 1e-8
                fixed_readout, fixed_transform_peak = False, None
                model.load_state_dict(before)
                write_json(args.out / 'READOUT_DIAGNOSTIC.json', dict(
                    replayed_native_trial=True, direction='saved_positive_wave_descent',
                    before_metrics=base_metrics, trials=trials, committed_optimizer_updates=0,
                    weights_restored=True, report_rows_used=0, scientific_acceptance=False,
                    interpretation='Fixed-peak variants are causal diagnostics only; original hard readout and acceptance remain unchanged.'))
                setup['references'][0].assert_frozen_unchanged(deep=True)
            elif p['mode'] == 'joint_secant_pilot':
                model = models['B']
                named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
                sizes = [p['learning_rates']['response'] if k == 'theta'
                         else p['learning_rates']['mean'] for k, _ in named]
                optimizer = torch.optim.Adam([
                    {'params': [model.theta], 'lr': p['learning_rates']['response']},
                    {'params': [v for k, v in named if k != 'theta'], 'lr': p['learning_rates']['mean']}])
                optimizer.load_state_dict(saved['optimizer'])
                seed_direction = torch.load(p['direction_file'], map_location=device, weights_only=False)
                counts['diagnostic_direction_loads'] += 1
                assert [k for k, _ in named] == seed_direction['names']
                assert all(torch.equal(model.state_dict()[k], v) for k, v in seed_direction['before'].items())
                steps, rejected_streak = [], 0
                radius = p.get('initial_secant_radius', 1.)
                for iteration in range(p['optimizer']['direction_attempts_per_arm']):
                    del paired
                    paired = banks(p['draws_per_parent'], p['bank_seed'] + (iteration + 2) * 1000000)
                    before = copy.deepcopy(model.state_dict())
                    opt_before = copy.deepcopy(optimizer.state_dict())
                    if iteration == 0:
                        means, old_flags = pair_means(model, paired)
                        elapsed = 0.
                        for (_, value), grad in zip(named, seed_direction['objective_gradient']):
                            value.grad = grad.clone()
                        directions = {k: seed_direction['directions'][k] for k in
                                      ('aligned_positive_tangent', 'positive_wave_descent')}
                    else:
                        means, old_flags, elapsed = gradient(model, paired)
                    previous, guard_previous = numbers(means), positive_metrics(means)
                    old_rail_risks = model.adc_margin_risks.copy()
                    optimizer.step()
                    counts['optimizer_directions'] += 1
                    with torch.no_grad():
                        model.theta.clamp_(-1., 1.)
                    if iteration > 0:
                        delta = [model.state_dict()[k] - before[k] for k, _ in named]
                        grads = model.positive_wave_gradient
                        derivative = sum((g * d).double().sum() for g, d in zip(grads, delta))
                        denominator = sum((s * g).double().square().sum() for s, g in zip(sizes, grads))
                        correction = max(float(derivative), 0.) / float(denominator)
                        norm = sum((d / s).double().square().sum() for d, s in zip(delta, sizes)).sqrt()
                        directions = dict(
                            aligned_positive_tangent=[d - correction * s * s * g
                                                      for d, s, g in zip(delta, sizes, grads)],
                            positive_wave_descent=[-norm / denominator.sqrt() * s * s * g
                                                   for s, g in zip(sizes, grads)])
                    model.load_state_dict(before)
                    if p.get('ADC_descent_basis', False):
                        # These gradients use continuous pre-ADC carriers, not a
                        # surrogate for rounded flags. Each bank gets its own direction.
                        reference_norm = sum((d / s).double().square().sum() for d, s in
                            zip(directions['positive_wave_descent'], sizes)).sqrt()
                        for bank_index, bank in enumerate(paired):
                            adc_gradient = [torch.zeros_like(v) for _, v in named]
                            count = sum(len(b['rows']) for b in bank)
                            for b in bank:
                                for start in range(0, len(b['rows']), p['micro_rows']):
                                    obs = observed(model, b, slice(start, start + p['micro_rows']))
                                    risk = adc_margin_risk(obs['pre_adc_carrier']).sum() / count
                                    grads = torch.autograd.grad(risk, [v for _, v in named])
                                    for g, value in zip(adc_gradient, grads):
                                        g.add_(value)
                                    counts['continuous_ADC_backward_microblocks'] += 1
                            grad_norm = sum((s * g).double().square().sum() for s, g in zip(sizes, adc_gradient)).sqrt()
                            # Interior banks have exactly zero hinge loss and gradient.
                            if float(grad_norm) > 0.:
                                directions[f'ADC_bank{bank_index}_descent'] = [
                                    -reference_norm / grad_norm * s * s * g for s, g in zip(sizes, adc_gradient)]
                    probes, actual_directions = [], []
                    for label, delta in directions.items():
                        state = copy.deepcopy(before)
                        for (k, _), d in zip(named, delta):
                            state[k] = before[k] + radius * d
                        state['theta'].clamp_(-1., 1.)
                        actual_directions.append([state[k] - before[k] for k, _ in named])
                        model.load_state_dict(state)
                        probe_means, probe_flags = pair_means(model, paired)
                        probes.append(dict(direction=label, losses=numbers(probe_means),
                            positive_metrics=positive_metrics(probe_means), flags=probe_flags,
                            ADC_margin_risk=model.adc_margin_risks.copy()))
                        counts['native_secant_probes'] += 1
                    model.load_state_dict(before)
                    guards = p['optimizer']['per_step_guards']
                    constraint_names = [*guards, 'PSD_CDF', 'ADC_margin_bank0', 'ADC_margin_bank1']
                    changes = [[v['positive_metrics'][k] - guard_previous[k] for v in probes]
                               for k in (*guards, 'PSD_CDF')]
                    allowances = [(factor - 1.) * guard_previous[k] for k, factor in guards.items()] + [0.]
                    for b in range(2):
                        changes.append([v['ADC_margin_risk'][b] - old_rail_risks[b] for v in probes])
                        allowances.append(0.)
                    objective_changes = np.array([sum(v['losses'].values()) - sum(previous.values()) for v in probes])
                    weights = joint_secant_weights(objective_changes, changes, allowances)
                    predicted_change = float(objective_changes @ weights)
                    torch.save(dict(before=before, directions=actual_directions, names=[k for k, _ in named],
                                    coefficients=weights, source='native_full_population_secants'),
                               args.out / f'DIRECTION_{iteration + 1:02d}.pt')
                    passed, trials = False, []
                    if predicted_change < 0.:
                        for scale in p['optimizer']['step_scales']:
                            state = copy.deepcopy(before)
                            for j, (k, _) in enumerate(named):
                                state[k] = before[k] + scale * sum(float(w) * d[j]
                                                                   for w, d in zip(weights, actual_directions))
                            state['theta'].clamp_(-1., 1.)
                            model.load_state_dict(state)
                            trial_means, flags = pair_means(model, paired)
                            losses, metrics = numbers(trial_means), positive_metrics(trial_means)
                            checks = {k: metrics[k] <= factor * guard_previous[k] for k, factor in guards.items()}
                            checks.update(PSD_CDF_decreased=metrics['PSD_CDF'] < guard_previous['PSD_CDF'],
                                global_objective_decreased=sum(losses.values()) < sum(previous.values()),
                                flags_not_increased=all(flags[b][k] <= old_flags[b][k]
                                                        for b in range(2) for k in old_flags[b]),
                                ADC_margin_not_worse=all(model.adc_margin_risks[b] <= old_rail_risks[b] for b in range(2)))
                            passed = all(checks.values())
                            trials.append(dict(scale=scale, losses=losses, positive_metrics=metrics,
                                flags=flags, ADC_margin_risk=model.adc_margin_risks.copy(), checks=checks, accepted=passed))
                            counts['joint_hard_trials'] += 1
                            if passed:
                                accepted['B'] += 1
                                counts['committed_optimizer_updates'] += 1
                                rejected_streak = 0
                                break
                    if not passed:
                        model.load_state_dict(before)
                        optimizer.load_state_dict(opt_before)
                        rejected_streak += 1
                    next_radius = radius
                    if p.get('adaptive_secant_radius', False):
                        if passed:
                            ratio = (sum(losses.values()) - sum(previous.values())) / (scale * predicted_change)
                            next_radius = min(1., radius * scale * (2. if ratio >= .75 else 1.))
                        else:
                            next_radius = radius * p['optimizer']['step_scales'][-1]
                    record = dict(iteration=iteration + 1, arm='B', gradient_seconds=elapsed,
                        reused_saved_gradient=iteration == 0, secant_radius=radius, next_secant_radius=next_radius,
                        previous_losses=previous,
                        previous_metrics=guard_previous, previous_flags=old_flags,
                        previous_ADC_margin_risk=old_rail_risks, probes=probes,
                        constraint_names=constraint_names, constraint_changes=changes,
                        constraint_allowances=allowances, coefficients=weights.tolist(),
                        predicted_objective_change=predicted_change, trials=trials,
                        accepted=passed, accepted_updates=accepted['B'],
                        new_committed_updates=counts['committed_optimizer_updates'])
                    radius = next_radius
                    steps.append(record)
                    with (args.out / 'TRAINING_STEPS.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(record) + '\n')
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        accepted_updates=accepted['B'], inherited_accepted_updates=saved['accepted_updates'],
                        new_committed_updates=counts['committed_optimizer_updates'], direction_iteration=iteration + 1,
                        parent_checkpoint=p['replay_checkpoint']), args.out / 'B_CURRENT.pt')
                    progress_file(args.out, counts, 'joint_secant_direction_complete', iteration=iteration + 1,
                        accepted_updates=dict(accepted), rejected_streak=rejected_streak,
                        elapsed_seconds=time.monotonic() - started)
                    if (iteration == 0 and not passed) or rejected_streak >= p['optimizer']['stop_consecutive_rejections']:
                        break
        else:
            assert p['mode'] in ('pilot', 'projected_pilot')
            accepted = Counter()
            rejected_streak = Counter()
            steps = []
            optimizers = {name: torch.optim.Adam([
                {'params': [model.theta], 'lr': p['learning_rates']['response']},
                *([{'params': [v for k, v in model.named_parameters()
                               if k != 'theta' and v.requires_grad],
                    'lr': p['learning_rates']['mean']}] if name == 'B' else [])
                ]) for name, model in models.items()}
            for iteration in range(p['optimizer']['direction_attempts_per_arm']):
                del paired
                paired = banks(p['draws_per_parent'], p['bank_seed'] + (iteration + 1) * 1000000)
                for name, model in models.items():
                    if rejected_streak[name] >= p['optimizer']['stop_consecutive_rejections']:
                        continue
                    means, old_flags, elapsed = gradient(model, paired)
                    previous = numbers(means)
                    guard_previous = positive_metrics(means)
                    before = copy.deepcopy(model.state_dict())
                    opt_before = copy.deepcopy(optimizers[name].state_dict())
                    gradient_norms = {k: float(v.grad.norm()) for k, v in model.named_parameters()
                                      if v.requires_grad}
                    if not all(np.isfinite(v) for v in gradient_norms.values()):
                        raise ValueError('Nonfinite pilot gradient')
                    optimizers[name].step()
                    counts['optimizer_directions'] += 1
                    with torch.no_grad():
                        model.theta.clamp_(-1., 1.)
                    proposed = copy.deepcopy(model.state_dict())
                    projection = None
                    if p['mode'] == 'projected_pilot':
                        named = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
                        stepsizes = [p['learning_rates']['response'] if k == 'theta'
                                     else p['learning_rates']['mean'] for k, _ in named]
                        derivative = sum((w * (proposed[k] - before[k])).double().sum()
                                         for (k, _), w in zip(named, model.wave_gradient))
                        denominator = sum((s * w).double().square().sum()
                                          for s, w in zip(stepsizes, model.wave_gradient))
                        assert denominator > 0
                        correction = max(float(derivative), 0.) / float(denominator)
                        for (k, _), w, s in zip(named, model.wave_gradient, stepsizes):
                            proposed[k] -= correction * s * s * w
                        # Domain clipping is part of the actual trial; recompute
                        # its directional derivative instead of claiming a proof.
                        proposed['theta'].clamp_(-1., 1.)
                        after_derivative = sum((w * (proposed[k] - before[k])).double().sum()
                                               for (k, _), w in zip(named, model.wave_gradient))
                        projection = dict(wave_directional_derivative_before=float(derivative),
                            wave_directional_derivative_after=float(after_derivative),
                            projection_active=correction > 0,
                            scope='First-order STE waveform halfspace only; native hard trial remains required.')
                    trials = []
                    passed = False
                    for scale in p['optimizer']['step_scales']:
                        model.load_state_dict({k: before[k] + scale * (proposed[k] - before[k])
                                               for k in before})
                        trial_means, new_flags = pair_means(model, paired)
                        trial_losses = numbers(trial_means)
                        guard_now = positive_metrics(trial_means)
                        objective_decreased = sum(trial_losses.values()) < sum(previous.values())
                        guards = {k: guard_now[k] <= factor * guard_previous[k]
                                  for k, factor in p['optimizer']['per_step_guards'].items()}
                        # Same independent banks before/after: no noise-driven
                        # difference in flags or batch-dependent reference counts.
                        flags_pass = all(new_flags[b][k] <= old_flags[b][k]
                                         for b in range(2) for k in old_flags[b])
                        passed = objective_decreased and all(guards.values()) and flags_pass
                        trials.append(dict(scale=scale, losses=trial_losses,
                                           positive_metrics=guard_now, guards=guards,
                                           flags=new_flags, flags_pass=flags_pass,
                                           objective_decreased=objective_decreased, accepted=passed))
                        if passed:
                            accepted[name] += 1
                            counts['committed_optimizer_updates'] += 1
                            rejected_streak[name] = 0
                            break
                    if not passed:
                        model.load_state_dict(before)
                        optimizers[name].load_state_dict(opt_before)
                        rejected_streak[name] += 1
                    record = dict(iteration=iteration + 1, arm=name,
                                  gradient_seconds=elapsed, gradient_norms=gradient_norms,
                                  previous_losses=previous, previous_metrics=guard_previous,
                                  previous_flags=old_flags, trials=trials, accepted=passed,
                                  accepted_updates=accepted[name], wave_projection=projection)
                    steps.append(record)
                    with (args.out / 'TRAINING_STEPS.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(record) + '\n')
                    torch.save(dict(model=model.state_dict(), optimizer=optimizers[name].state_dict(),
                                    accepted_updates=accepted[name], direction_iteration=iteration + 1),
                               args.out / f'{name}_CURRENT.pt')
                    progress_file(args.out, counts, 'paired_pilot_direction_complete',
                                  iteration=iteration + 1, arm=name, accepted_updates=dict(accepted),
                                  rejected_streak=dict(rejected_streak),
                                  elapsed_seconds=time.monotonic() - started)
                if all(rejected_streak[k] >= p['optimizer']['stop_consecutive_rejections'] for k in models):
                    break
        if p['mode'] in ('pilot', 'projected_pilot', 'endpoint_replay', 'joint_secant_pilot'):
            del paired
            # Fresh independent random banks are not used to choose settings or
            # the endpoint. 32 draws reproduce the requested full-Q figure scale.
            endpoint_banks = banks(p['endpoint_draws'], p.get('endpoint_seed', p['bank_seed'] + 50000000), cpu=True)
            reference = GlobalRepairResponse01(setup['joint_models'][0], theta)
            arrays = dict(source_parent_rows=parent_array, real_rows=real_rows)
            for key in ('Q', 'tail', 'PSD_raw'):
                arrays['real_' + key] = real[key].cpu().numpy()
            endpoint = {}
            with torch.no_grad():
                for name, model in dict(initial=reference, **models).items():
                    values, fcounts = [], []
                    for bank_index, bank in enumerate(endpoint_banks):
                        avg, flag, scalar, _ = bank_mean(model, bank, collect=True)
                        values.append(avg)
                        fcounts.append(flag)
                        arrays.update({f'{name}_bank{bank_index}_{k}': v for k, v in scalar.items()})
                    endpoint[name] = dict(independent_bank_loss=numbers(values),
                                          positive_metrics=positive_metrics(values), flags=fcounts)
                    progress_file(args.out, counts, 'pilot_endpoint_complete', arm=name)
            for name in models:
                metrics = endpoint[name]['positive_metrics']
                baseline_metrics = endpoint['initial']['positive_metrics']
                endpoint[name]['pilot_checks'] = dict(
                    PSD_CDF_at_least_10pct_improvement=metrics['PSD_CDF'] <= .90 * baseline_metrics['PSD_CDF'],
                    Q_mass_not_worse_2pct=metrics['Q_mass'] <= 1.02 * baseline_metrics['Q_mass'],
                    wave_not_worse_2pct=metrics['block_wave'] <= 1.02 * baseline_metrics['block_wave'],
                    peak_codes_not_worse_2pct=metrics['block_peak_codes'] <= 1.02 * baseline_metrics['block_peak_codes'],
                    flags_not_increased=all(endpoint[name]['flags'][b][k] <= endpoint['initial']['flags'][b][k]
                                           for b in range(2) for k in endpoint['initial']['flags'][b]))
                endpoint[name]['passes_bounded_pilot'] = all(endpoint[name]['pilot_checks'].values())
            archive_path = Path(p['endpoint_archive'])
            # Preserve small endpoint metrics even if the later array write fails.
            write_json(args.out / 'TRAINING_RESULT.json', dict(
                mode=p['mode'], accepted_updates={k: accepted[k] for k in models},
                new_committed_optimizer_updates=counts['committed_optimizer_updates'],
                direction_records=len(steps), endpoint=endpoint,
                endpoint_arrays=str(archive_path), seed_head=0, hypothesis=p['hypothesis'],
                independent_final_random_banks=True, report_role_rows_used=0,
                scientific_acceptance=False, historical_sources_modified=False))
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(archive_path, **arrays)
            # Original frozen baseline is still the same control object.
            setup['references'][0].assert_frozen_unchanged(deep=True)
        check_budget()
        write_json(args.out / 'RESULT.json', dict(status='COMPLETE', mode=p['mode'],
            elapsed_seconds=time.monotonic() - started, execution_counts=dict(counts),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
            config_sha256=args.config_sha256,
            committed_optimizer_updates=counts['committed_optimizer_updates'],
            scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                automatic_restart=False))


if __name__ == '__main__':
    main()
