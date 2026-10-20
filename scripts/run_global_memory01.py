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
from n01.global_memory01 import GlobalDistributionMemory01, MeanHeadRepair01, GlobalRepairResponse01
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
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
        signal = model(rows, z)
        counts['hard_observation_rows'] += len(rows)
        if grad:
            counts['STE_rows'] += len(rows)
        obs = observe_generated_carrier(signal + model.noise_scale() * bank['noise'][selection].to(device),
                                        ste=grad, q_floor=q_floor, return_carrier=True)
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
        count = sum(len(b['rows']) for b in bank)
        for b in bank:
            for start in range(0, len(b['rows']), p['micro_rows']):
                obs = observed(model, b, slice(start, start + p['micro_rows']))
                total += memory.sums(obs)
                flags.update({k: int(v.sum()) for k, v in obs['flags'].items()})
                if collect:
                    records.append({k: obs[k].detach().cpu().numpy()
                                    for k in ('Q', 'tail', 'PSD_raw', 'peak_index')})
        arrays = {k: np.concatenate([r[k] for r in records]) for k in records[0]} if collect else None
        return total / count, dict(flags), arrays

    def pair_means(model, paired):
        with torch.no_grad():
            results = [bank_mean(model, b) for b in paired]
        return [r[0] for r in results], [r[1] for r in results]

    def gradient(model, paired):
        torch.cuda.synchronize()
        stamp = time.monotonic()
        model.zero_grad(set_to_none=True)
        means, flags = pair_means(model, paired)
        coefficients = memory.coefficients(*means)
        project_wave = p['mode'] == 'projected_pilot'
        parameters = [v for v in model.parameters() if v.requires_grad]
        if project_wave:
            wave_coefficients = memory.wave_coefficients(*means)
            accumulated = [torch.zeros_like(v) for v in parameters]
            wave_gradient = [torch.zeros_like(v) for v in parameters]
        for bank_index, (bank, coefficient) in enumerate(zip(paired, coefficients)):
            count = sum(len(b['rows']) for b in bank)
            for b in bank:
                for start in range(0, len(b['rows']), p['micro_rows']):
                    obs = observed(model, b, slice(start, start + p['micro_rows']), grad=True)
                    features = memory.sums(obs) / count
                    loss = (coefficient * features).sum()
                    if project_wave:
                        g = torch.autograd.grad(loss, parameters, retain_graph=True)
                        w = torch.autograd.grad((wave_coefficients[bank_index] * features).sum(), parameters)
                        for accumulator, wave_accumulator, gv, wv in zip(accumulated, wave_gradient, g, w):
                            accumulator.add_(gv)
                            wave_accumulator.add_(wv)
                        counts['wave_gradient_microblocks'] += 1
                    else:
                        loss.backward()
                    counts['backward_microblocks'] += 1
        if project_wave:
            for value, grad in zip(parameters, accumulated):
                value.grad = grad
            model.wave_gradient = wave_gradient
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
            if p['mode'] == 'projected_pilot':
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
            del paired
            # Fresh independent random banks are not used to choose settings or
            # the endpoint. 32 draws reproduce the requested full-Q figure scale.
            endpoint_banks = banks(p['endpoint_draws'], p['bank_seed'] + 50000000, cpu=True)
            reference = GlobalRepairResponse01(setup['joint_models'][0], theta)
            arrays = dict(source_parent_rows=parent_array, real_rows=real_rows)
            for key in ('Q', 'tail', 'PSD_raw'):
                arrays['real_' + key] = real[key].cpu().numpy()
            endpoint = {}
            with torch.no_grad():
                for name, model in dict(initial=reference, **models).items():
                    values, fcounts = [], []
                    for bank_index, bank in enumerate(endpoint_banks):
                        avg, flag, scalar = bank_mean(model, bank, collect=True)
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
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(archive_path, **arrays)
            write_json(args.out / 'TRAINING_RESULT.json', dict(
                mode=p['mode'], accepted_updates={k: accepted[k] for k in models},
                direction_records=len(steps), endpoint=endpoint,
                endpoint_arrays=str(archive_path), seed_head=0, hypothesis=p['hypothesis'],
                independent_final_random_banks=True, report_role_rows_used=0,
                scientific_acceptance=False, historical_sources_modified=False))
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
