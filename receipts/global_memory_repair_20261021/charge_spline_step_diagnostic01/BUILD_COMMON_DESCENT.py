from pathlib import Path
import ast
import hashlib

root = Path('E:/GAN/research/N01')
base = (root / 'scripts/run_charge_spline_step_diagnostic01.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == '7e9b41656d3619003738b206740608e14f59cded5b5e1d40fbc7295b0e3e5d39'
text = base.decode()
text = text[:text.index('    optimizer = torch.optim.Adam(')]
text = text.replace('import torch\n', 'import torch\nfrom n01.common_descent01 import common_descent_direction\n', 1)
seed_start = text.index('training_seeds = ')
seed_end = text.index('\nargs.out.mkdir', seed_start)
text = text[:seed_start] + '''training_seeds = {base + iteration * 1000000 + bank * 100000 + chunk * 1000 + offset
    for base, steps in [(2246102200, 24), (2406102200, 128), (p['training_seed'], p['steps'])]
    for iteration in range(steps) for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
endpoint_seeds = {p['endpoint_seed'] + bank * 100000 + chunk * 1000 + offset
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
old_endpoint_and_normalization = {base + bank * 100000 + chunk * 1000 + offset
    for base in [p['bank_seed'], 2266102200, 2706102200, 2806102200]
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
assert not endpoint_seeds.intersection(training_seeds | old_endpoint_and_normalization)
assert not {p['training_seed'] + i * 1000000 + bank * 100000 + chunk * 1000 + offset
    for i in range(p['steps']) for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}.intersection(old_endpoint_and_normalization)
''' + text[seed_end:]
start = text.index('    objectives = dict(distribution=')
end = text.index('\n    coefficients = ', start)
text = text[:start] + '''    objectives = dict(distribution=sum(losses.values()),
        **{k: positive[k] for k in ['Q_mass', 'PSD_CDF', 'Q_wave', 'block_wave']},
        local_penalty=p['constraint_weight'] * torch.stack([(positive[k] / cell_limits[k] - 1).relu().square() for k in supported]).mean())
''' + text[end:]
text = text.replace("    combined = [sum(gradients[k][i] for k in ['distribution', 'global_penalty', 'local_penalty', 'ADC', 'trend']) for i in range(len(named))]\n", '')
text = text.replace('return gradients, combined, dict(', 'return gradients, dict(')
text = text.replace('diagnostic_seeds_disjoint_from_training_and_prior_endpoints=True', 'endpoint_seeds_disjoint_from_all_training_and_prior_diagnostics=True')
text += '''    rates = np.concatenate([np.full(v.numel(), p['learning_rates']['response' if k == 'theta' else 'mean']) for k, v in named])
    stop_reason = 'maximum_registered_directions_completed'
    torch.save(dict(model=model.state_dict(), optimizer_updates=p['inherited_candidate_updates'], stage_optimizer_updates=0,
        optimizer_method='common_descent_v1', parent_checkpoint=p['initialization_checkpoint'], architecture=p['architecture']), args.out / 'B_CURRENT.pt')
    for iteration in range(p['steps']):
        paired = banks(p['draws_per_parent'], p['training_seed'] + iteration * 1000000)
        before = copy.deepcopy(model.state_dict())
        gradients, baseline_record = component_gradients(paired)
        keys = [k for k in gradients if k != 'trend']
        matrix = np.stack([torch.cat([v.detach().double().flatten() for v in gradients[k]]).cpu().numpy() for k in keys])
        delta, solve = common_descent_direction(matrix, rates)
        counts['optimizer_directions'] += 1
        trials, accepted = [], False
        if solve['status'] == 'COMMON_DIRECTION':
            for alpha in p['step_factors']:
                model.load_state_dict(before)
                offset = 0
                with torch.no_grad():
                    for name, value in named:
                        change = torch.as_tensor(delta[offset:offset + value.numel()].reshape(value.shape), device=device, dtype=value.dtype)
                        value.add_(alpha * change)
                        offset += value.numel()
                    model.theta.clamp_(-1., 1.)
                values, flags, rails, _ = pair(model, paired)
                current = metrics(values)
                with torch.no_grad():
                    distribution = float(sum(memory.losses(*values).values()))
                    local = p['constraint_weight'] * np.mean([max(current[k] / cell_limits[k] - 1., 0.)**2 for k in supported])
                    curve = grid.losses(mean) / trend_reference
                    excess = grid.excess(mean)
                checks = dict(**{k + '_nonincrease': current[k] <= baseline_record['positive_metrics'][k] + 1e-10 * abs(baseline_record['positive_metrics'][k])
                                  for k in ['Q_mass', 'Q_wave', 'block_wave']},
                    PSD_strict_decrease=current['PSD_CDF'] < baseline_record['positive_metrics']['PSD_CDF'],
                    ADC_nonincrease=np.mean(rails) <= np.mean(baseline_record['ADC_margin_risk']) + 1e-10 * abs(np.mean(baseline_record['ADC_margin_risk'])),
                    distribution_nonincrease=distribution <= baseline_record['components']['distribution'] + 1e-10 * abs(baseline_record['components']['distribution']),
                    local_penalty_nonincrease=local <= baseline_record['components']['local_penalty'] + 1e-10 * abs(baseline_record['components']['local_penalty']),
                    peak_within_original_fixed_limit=current['block_peak_codes'] <= global_limits['block_peak_codes'],
                    flags_not_increased=all(flags[b][k] <= baseline_record['flags'][b][k] for b in range(2) for k in flags[b]),
                    curvature_each_halved=bool((curve <= .5).all()), comb_excess_each_halved=bool((excess <= .5 * excess_reference).all()))
                checks = {k: bool(v) for k, v in checks.items()}
                trials.append(dict(alpha=alpha, positive_metrics=current, flags=flags, ADC_margin_risk=rails,
                    distribution=distribution, local_penalty=float(local), trend_relative=curve.tolist(),
                    fixed_window_excess=excess.tolist(), checks=checks, accepted=all(checks.values())))
                counts['native_step_probes'] += 1
                counts['conditioned_mean_rows'] = mean.forward_rows
                budget()
                if all(checks.values()):
                    accepted = True
                    counts['optimizer_updates'] += 1
                    torch.save(dict(model=model.state_dict(), optimizer_updates=p['inherited_candidate_updates'] + counts['optimizer_updates'],
                        stage_optimizer_updates=counts['optimizer_updates'], optimizer_method='common_descent_v1',
                        parent_checkpoint=p['initialization_checkpoint'], architecture=p['architecture']), args.out / 'B_CURRENT.pt')
                    break
        record = dict(iteration=iteration + 1, baseline=baseline_record, component_names=keys, solve=solve,
            trials=trials, accepted=accepted, stage_optimizer_updates=counts['optimizer_updates'],
            cumulative_optimizer_updates=p['inherited_candidate_updates'] + counts['optimizer_updates'],
            elapsed_seconds=time.monotonic() - began, update_is_candidate_only=True, scientific_acceptance=False)
        with (args.out / 'TRAINING_STEPS.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(record) + '\\n')
        np.savez(args.out / 'LATEST_DIRECTION.npz', gradients=matrix, delta=delta, rates=rates)
        progress_file(args.out, counts, 'common_descent_direction_complete', iteration=iteration + 1,
            accepted=accepted, gradient_seconds=baseline_record['gradient_seconds'], elapsed_seconds=time.monotonic() - began)
        if not accepted:
            model.load_state_dict(before)
            stop_reason = 'no_verified_common_first_order_direction' if solve['status'] != 'COMMON_DIRECTION' else 'first_failed_finite_hard_step'
            break
'''
restoration = (root / 'scripts/run_charge_spline_restoration01.py').read_text()
tail = restoration[restoration.index('    final_state = copy.deepcopy(model.state_dict())'):]
tail = tail.replace("mode='charge_spline_restoration'", "mode='charge_spline_common_descent'")
tail = tail.replace("fixed_final_step=p['steps'], endpoint_selection=False", "fixed_final_step=counts['optimizer_updates'], maximum_directions=p['steps'], stop_reason=stop_reason, endpoint_selection=False")
text += tail
text = text.replace('Diagnose gradient conflict and Adam conditioning at the fixed152-step endpoint.',
                    'Bounded common-descent restoration with full-population hard step checks.')
ast.parse(text)
target = root / 'scripts/run_charge_spline_common_descent01.py'
target.write_bytes(text.encode())
print({'script': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
