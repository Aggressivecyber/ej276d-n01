"""Read only saved thin-head checkpoints and logged minibatches; no models/data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def stats(value):
    a = value.detach().cpu().reshape(-1).double().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float64).reshape(-1)
    return dict(count=int(a.size), finite=bool(np.isfinite(a).all()), min=float(a.min()), max=float(a.max()),
                mean=float(a.mean()), std_population=float(a.std()), RMS=float(np.sqrt(np.mean(a*a))),
                L2=float(np.linalg.norm(a)), absmax=float(np.abs(a).max()),
                quantiles={str(q): float(np.quantile(a, q)) for q in (0., .25, .5, .75, 1.)})


def tensor_inventory(value, path=''):
    result = []
    if isinstance(value, torch.Tensor):
        result.append(dict(path=path, shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(), device=str(value.device)))
    elif isinstance(value, dict):
        for key, child in value.items():
            result.extend(tensor_inventory(child, path+'/'+str(key)))
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            result.extend(tensor_inventory(child, path+'/'+str(i)))
    return result


def summarize_sampler(state, total, seed):
    n, epochs, position = state['n'], state['completed_epochs'], state['position']
    order, visits = state['order'].numpy(), state['visits'].numpy()
    rng = np.random.default_rng(seed)
    expected_order = rng.permutation(n)
    for _ in range(total//n):
        expected_order = rng.permutation(n)
    expected_visits = np.full(n, total//n, dtype=np.int64)
    expected_visits[expected_order[:total % n]] += 1
    checks = dict(order_is_permutation=bool(np.array_equal(np.sort(order), np.arange(n))),
                  total_matches=bool(visits.sum() == total), epoch_position_matches=(epochs, position) == divmod(total, n),
                  visits_match_frozen_seed_and_total=bool(np.array_equal(visits, expected_visits)),
                  order_matches_frozen_seed=bool(np.array_equal(order, expected_order)), rng_state_matches_frozen_seed=state['rng'] == rng.bit_generator.state)
    assert all(checks.values()), checks
    values, counts = np.unique(visits, return_counts=True)
    return dict(n=n, completed_epochs=epochs, position=position, total_visits=int(visits.sum()),
                unique_visited=int((visits > 0).sum()), visits_min=int(visits.min()), visits_max=int(visits.max()),
                visits_histogram={str(int(v)): int(c) for v, c in zip(values, counts)}, checks=checks,
                order_sha256=tensor_digest(state['order']), visits_sha256=tensor_digest(state['visits']))


def summarize_checkpoint(cp, expected_update):
    assert cp['schema'] == 'n01/shape010-checkpoint/v1'
    assert cp['run_id'] == 'shape010_20261014' and cp['G_updates'] == expected_update
    assert cp['complete_step_boundary'] and cp['in_flight'] is None
    assert cp['DEV_numeric_read'] is False and cp['artificial_only'] is False
    assert cp['trainable_parameter_names'] == ['basis_head.weight', 'basis_head.bias']
    assert set(cp['new_head']) == {'weight', 'bias'}
    assert not ({'model', 'baseline', 'noise005', 'extension'} & set(cp))
    p = cp['policy']
    counters = cp['counters']
    assert counters['G_updates'] == counters['attempted_G_updates'] == expected_update
    assert all(counters[k] == 0 for k in ('D_updates', 'noise_updates', 'population_updates', 'C2ST_updates'))
    assert counters['train_response_rows'] == expected_update * 1024
    assert counters['train_noise005_rows'] == expected_update * 1024
    assert counters['train_hard_observation_rows'] == expected_update * 3072
    head = cp['new_head']
    flat = torch.cat([head[k].reshape(-1) for k in ('weight', 'bias')])
    assert flat.numel() == 1288 and torch.isfinite(flat).all()
    opt = cp['optimizer']
    assert len(opt['param_groups']) == 1 and opt['param_groups'][0]['params'] == [0, 1]
    assert opt['param_groups'][0]['lr'] == .0002 and opt['param_groups'][0]['weight_decay'] == .0001
    steps = {str(k): float(v['step']) for k, v in opt['state'].items()}
    assert not steps if expected_update == 0 else len(steps) == 2 and all(v == expected_update for v in steps.values())
    optimizer_state = {str(k): {name: stats(v) for name, v in s.items() if name != 'step'} for k, s in opt['state'].items()}
    result = dict(G_updates=expected_update, purpose=cp['purpose'], complete_step_boundary=True,
                  checkpoint_keys=list(cp), counters=counters, attempted_operations=cp['attempted'],
                  frozen_state_sha256=cp['frozen_state_sha256'], trainable_parameter_names=cp['trainable_parameter_names'],
                  head=dict(combined=stats(flat), parameters={k: stats(v) for k, v in head.items()},
                            weight_output_channel_L2=[float(v.double().norm()) for v in head['weight']],
                            tensor_sha256={k: tensor_digest(v) for k, v in head.items()}),
                  optimizer=dict(parameter_groups=opt['param_groups'], parameter_steps=steps, moments=optimizer_state),
                  source_sampler=summarize_sampler(cp['source_sampler'], expected_update*512, p['seeds']['source_sampler']),
                  real_sampler=summarize_sampler(cp['real_sampler'], expected_update*1024, p['seeds']['real_sampler']),
                  stream_states={k: dict(sha256=tensor_digest(v), shape=list(v.shape), dtype=str(v.dtype)) for k, v in cp['streams'].items()},
                  last_metrics=cp['last_metrics'], tensor_inventory=tensor_inventory(cp))
    return result, flat


def head_change(first, second):
    a, b = first.double().numpy(), second.double().numpy()
    delta = b-a
    return dict(delta=stats(delta), changed_parameters=int((a != b).sum()), L2_ratio=float(np.linalg.norm(b)/np.linalg.norm(a)),
                cosine=float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))))


def summarize_logs(logs):
    updates = [r['G_updates'] for r in logs]
    assert updates == [1]+list(range(100, 2001, 100))
    residuals = []
    conditional = []
    points = []
    fingerprints = []
    for r in logs:
        assert r['attempted_G_updates'] == r['G_updates'] and r['DEV_numeric_read'] is False
        assert r['changed_new_head_parameters'] == ['weight', 'bias']
        proxy = float(np.mean([h['wave']+2*h['QPSD'] for h in r['hypotheses'].values()]))
        residuals.append(r['objective']-proxy-.01*r['regular'])
        assert r['projection_context']['kind'] == 'training' and r['projection_context']['update'] == r['G_updates']
        fingerprints.append(r['projection_context']['fingerprint'])
        for h, m in r['hypotheses'].items():
            d = m['conditional_diagnostics']
            conditional.append(dict(update=r['G_updates'], hypothesis=h, complete=d['conditional_coverage_complete'],
                                    bins_used=d['conditional_bins_used'], bins_total=d['conditional_bins_total'],
                                    min_fake_count=min(b['fake_count'] for b in d['conditional_bins']),
                                    min_real_count=min(b['real_count'] for b in d['conditional_bins']),
                                    fake_nonpositive=d['fake_nonpositive_q_rows'], real_nonpositive=d['real_nonpositive_q_rows'],
                                    fake_guarded=d['fake_guarded_psd_rows'], real_guarded=d['real_guarded_psd_rows']))
        g = r['new_head_preclip_gradient_norm']
        points.append(dict(G_completed=r['G_updates'], objective_evaluated_at_G=r['G_updates']-1,
                           objective=r['objective'], batch_proxy=proxy, regular=r['regular'], weighted_regular=.01*r['regular'],
                           weighted_regular_fraction_of_objective=.01*r['regular']/r['objective'],
                           new_head_preclip_gradient_norm=g, clipping_active=g > 5., theoretical_clip_coefficient=min(1., 5./(g+1e-6)),
                           random_shape_RMS_absmax_V=r['random_shape_RMS_absmax_V'], coverage=r['coverage'],
                           signal_prefix_absmax_V=r['signal_prefix_absmax_V'], canonical_charge_identity_absmax=r['canonical_charge_identity_absmax'],
                           basis_canonical_charge_absmax=r['basis_canonical_charge_absmax'],
                           canonical_random_shape_charge_absmax=r['canonical_random_shape_charge_absmax'],
                           mean_wave=float(np.mean([v['wave'] for v in r['hypotheses'].values()])),
                           mean_QPSD=float(np.mean([v['QPSD'] for v in r['hypotheses'].values()])),
                           hypotheses={h: dict(wave=v['wave'], QPSD=v['QPSD']) for h, v in r['hypotheses'].items()}))
    assert max(abs(v) for v in residuals) < 1e-12
    assert len(set(fingerprints)) == len(fingerprints)
    def aggregate(group):
        return {k: stats([r[k] for r in group]) for k in ('objective', 'batch_proxy', 'regular', 'new_head_preclip_gradient_norm', 'random_shape_RMS_absmax_V')}
    return dict(saved_batch_records=len(logs), saved_completed_G_updates=updates, unsaved_G_updates=2000-len(logs),
                point_metrics_are_pre_step_single_batches=True, all_saved_training_projections_distinct=True,
                objective_reconstruction_residual_absmax=max(abs(v) for v in residuals), points=points,
                logged_clipping_active_count=sum(p['clipping_active'] for p in points),
                logged_clipping_denominator=len(points), actual_2000_step_clipping_frequency='unknown; only 21 batch records saved',
                aggregate_all_logged=aggregate(points), aggregate_last_five_logged=aggregate(points[-5:]),
                conditional=dict(saved_hypothesis_records=len(conditional), incomplete_count=sum(not r['complete'] for r in conditional),
                                 min_fake_bin_count=min(r['min_fake_count'] for r in conditional), min_real_bin_count=min(r['min_real_count'] for r in conditional),
                                 nonpositive_or_guarded_counts={k: sum(r[k] for r in conditional) for k in ('fake_nonpositive', 'real_nonpositive', 'fake_guarded', 'real_guarded')},
                                 all_2000_batch_conditional_coverage='unknown outside saved records', records=conditional),
                training_peak_change_counts='not recorded by this frozen training logger; not inferable as zero',
                random_RMS_statistic='maximum row RMS in each logged 1024-fake batch; not mean RMS or population uncertainty',
                regular_decomposition='regular contains realized RMS+second-difference terms and mean ADC saturation; separate components not logged')


def evaluation_summary(complete):
    ini, fin = complete['initial_evaluation'], complete['final_evaluation']
    views = dict(frozen_A=ini['frozen_A'], shape010_initial=ini['shape010_initial'], shape010_final=fin['shape010_final'])
    result = {}
    for view, hs in views.items():
        result[view] = {h: dict(**r['components'], proxy=r['components']['wave']+2*r['components']['QPSD'],
                               projection_context=r['projection_context']) for h, r in hs.items()}
        result[view]['mean_proxy'] = float(np.mean([r['components']['wave']+2*r['components']['QPSD'] for r in hs.values()]))
        result[view]['mean_wave'] = float(np.mean([r['components']['wave'] for r in hs.values()]))
        result[view]['mean_QPSD'] = float(np.mean([r['components']['QPSD'] for r in hs.values()]))
    a, initial, final = (result[k]['mean_proxy'] for k in ('frozen_A', 'shape010_initial', 'shape010_final'))
    result['changes'] = dict(final_relative_to_A=(final/a-1), initial_relative_to_A=(initial/a-1), final_relative_to_initial=(final/initial-1),
                             share_of_initial_to_final_proxy_reduction_returning_to_A=(initial-a)/(initial-final),
                             per_hypothesis={h: {k: result['shape010_final'][h][k]/result['frozen_A'][h][k]-1 for k in ('wave','QPSD','proxy')}
                                             for h in ('fast','slow','common_level')})
    control = ini['REAL_REAL']['components']
    result['REAL_REAL_control'] = dict(**control, proxy=control['wave']+2*control['QPSD'])
    result['interpretation'] = 'One fixed TRAIN bank and projection, 512 SOURCE parents x4 draws. Descriptive proxy improvement, not a scientific gate, uncertainty, convergence, or noGAN-vs-GAN causal result.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert args.output.resolve().drive.upper() in ('E:', 'D:')
    torch.set_num_threads(2)
    snapshot = json.loads(args.snapshot.read_text(encoding='utf8'))
    complete = snapshot['runs/primary/artifacts/COMPLETE.json']
    expected = complete['artifacts']
    filenames = ('training.jsonl', 'initial.pt', 'periodic_01000.pt', 'final.pt')
    hashes = {name: digest(args.artifacts/name) for name in filenames}
    assert all(hashes[name] == expected[name]['sha256'] for name in filenames), 'Native artifacts SHA mismatch before checkpoint decode'
    logs = [json.loads(line) for line in (args.artifacts/'training.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    raw, checkpoints, flats = {}, {}, {}
    for name, update in zip(filenames[1:], (0, 1000, 2000)):
        raw[name] = torch.load(args.artifacts/name, map_location='cpu', weights_only=False)
        checkpoints[name], flats[name] = summarize_checkpoint(raw[name], update)
    assert raw['periodic_01000.pt']['last_metrics'] == next(r for r in logs if r['G_updates'] == 1000)
    assert raw['final.pt']['last_metrics'] == logs[-1]
    assert raw['initial.pt']['last_metrics'] == {}
    assert all(cp['frozen_state_sha256'] == complete['frozen_state_sha256'] for cp in raw.values())
    assert all(raw[name]['policy'] == raw['initial.pt']['policy'] for name in filenames[1:])
    unchanged = {name: digest(args.artifacts/name) == hashes[name] for name in filenames}
    assert all(unchanged.values())
    result = dict(schema='n01/shape010-training-readback/v1', run_id='shape010_20261014', UTC=datetime.now(timezone.utc).isoformat(),
                  status='saved_training_tensor_and_log_analysis_only_no_model_execution',
                  source_artifacts=str(args.artifacts.resolve()), snapshot=dict(path=str(args.snapshot.resolve()), sha256=digest(args.snapshot)),
                  input_files={name: dict(sha256=hashes[name], bytes=(args.artifacts/name).stat().st_size, unchanged_after_read=unchanged[name]) for name in filenames},
                  checkpoints=checkpoints, head_changes={a+' -> '+b: head_change(flats[a], flats[b])
                  for a,b in (('initial.pt','periodic_01000.pt'),('periodic_01000.pt','final.pt'),('initial.pt','final.pt'))},
                  training=summarize_logs(logs), fixed_evaluation=evaluation_summary(complete),
                  native_exit_receipt=snapshot['runs/primary/supervisor/exit_receipt.json'], native_complete_coverage=complete['coverage'],
                  audit_scope=dict(loaded_thin_head_checkpoints=3, original_A_or_005_checkpoint_loads=0,
                                   scientific_model_forward=0, backward=0, optimizer_steps=0, raw_TRAIN_or_DEV_arrays_read=0,
                                   checkpoints_or_frozen_sources_modified=0),
                  source_line_evidence=dict(nonzero_head_initialization='src/n01/model010.py:89-98',
                                           tanh_control_and_random_voltage='src/n01/model010.py:190-207',
                                           optimizer_step_completed_counter='src/n01/train_shape010.py:296-303',
                                           saved_log_semantics='src/n01/train_shape010.py:305-318',
                                           thin_checkpoint_schema='src/n01/train_shape010.py:228-239',
                                           full_coverage_sampler='src/n01/data.py:120-150'),
                  parameter_tanh_interpretation='Small parameter values or their norms do not determine tanh saturation: controls=tanh(control_average(head(frozen decoded))). No decoded features or tanh activations were evaluated in this readback.',
                  loss_gradient_scope='Only total new-head preclip gradient norm is logged, no per-loss parameter-gradient attribution or cosines. Gradient mechanism requires a separately registered bounded analysis.',
                  recommendation=dict(immediate_GAN_registration=False,
                                      next_action='First use saved paired parent-cluster arrays to quantify whether the small final-vs-A gain is stable and separate pure-signal from ADC/peak/crop effects. If stable but mechanism unresolved, register fixed-bank per-loss new-head gradient attribution before a matched GAN control.',
                                      GAN_causal_control='Any future GAN comparison must use the same new head, initialization, data/noise banks, optimizer and G budget with adversarial on/off. Old full19k differs in architecture/randomness/stage/history and is not a causal control.',
                                      budget_is_convergence_claim=False, scientific_response_match_established=False,
                                      training_batch_metrics_vs_fixed_evaluation='Never splice these into one numerical learning curve; logged metrics use refreshed projections and current unpaired 1024-row batches and pre-step parameters. Fixed evaluation uses a different constant 2048-row parent bank and fixed projection.'))
    target = args.output/'TRAINING_READBACK.json'
    assert not target.exists()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    compact = dict(receipt=str(target), sha256=digest(target), input_files=hashes,
                   head_stats={name: r['head']['combined'] for name,r in checkpoints.items()},
                   head_changes=result['head_changes'], training_summary={k: v for k,v in result['training'].items() if k not in ('points','conditional')},
                   conditional={k:v for k,v in result['training']['conditional'].items() if k!='records'},
                   fixed_evaluation=result['fixed_evaluation'], final_sampler={k:checkpoints['final.pt'][k] for k in ('source_sampler','real_sampler')})
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
