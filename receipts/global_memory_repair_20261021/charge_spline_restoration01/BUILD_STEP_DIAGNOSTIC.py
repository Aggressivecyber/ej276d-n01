from pathlib import Path
import ast
import hashlib

root = Path('E:/GAN/research/N01')
source = (root / 'scripts/run_charge_spline_restoration01.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '8ae112a21311d2a4351b2914848939ca9e001bc3bbd8b8f60ca3d03935d97ba6'
text = source.decode()
text = text[:text.index('    for iteration in range(p[\'steps\']):')]
seed_start = text.index('training_seeds = ')
seed_end = text.index('\nargs.out.mkdir', seed_start)
text = text[:seed_start] + '''training_seeds = {base + iteration * 1000000 + bank * 100000 + chunk * 1000 + offset
    for base, steps in [(2246102200, 24), (2406102200, 128)]
    for iteration in range(steps) for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
diagnostic_seeds = {p['diagnostic_seed'] + bank * 100000 + chunk * 1000 + offset
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
old_endpoint_and_normalization = {base + bank * 100000 + chunk * 1000 + offset
    for base in [p['bank_seed'], 2266102200, 2706102200]
    for bank in range(2) for chunk in range(chunk_count) for offset in range(3)}
assert not diagnostic_seeds.intersection(training_seeds | old_endpoint_and_normalization)
''' + text[seed_end:]
start = text.index('def gradient(paired):')
end = text.index('\n\ntry:', start)
text = text[:start] + '''def component_gradients(paired):
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
''' + text[end:]
text = text.replace("        torch.save(dict(model=initial_state, inherited_updates=10, optimizer_updates=0,\n            parent_checkpoint=p['replay_checkpoint'], architecture=p['architecture']), args.out / 'CANDIDATE_INITIAL.pt')\n", '')
text = text.replace('endpoint_seeds_disjoint_from_both_training_stages=True', 'diagnostic_seeds_disjoint_from_training_and_prior_endpoints=True')
text += '''    paired = banks(p['draws_per_parent'], p['diagnostic_seed'])
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
'''
text = text.replace("for key in ('response_rows', 'conditioned_mean_rows', 'mean_adjoint_rows', 'optimizer_updates'):",
                    "for key in p['operation_budget']:")
text = text.replace('Resume the closed24-step spline candidate for one registered restoration stage.',
                    'Diagnose gradient conflict and Adam conditioning at the fixed152-step endpoint.')
ast.parse(text)
target = root / 'scripts/run_charge_spline_step_diagnostic01.py'
target.write_bytes(text.encode())
print({'script': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
