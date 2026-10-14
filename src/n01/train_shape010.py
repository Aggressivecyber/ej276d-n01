"""One bounded TRAIN-only noGAN arm with an immutable deterministic response.

Only the new basis_head is optimized.  The independent010 binding is checked
before prepared TRAIN decoding or checkpoint loading.  There is no resume,
early-stop, discriminator, DEV decoder, or outcome-dependent configuration.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .data import FullCoverageSampler, sha256
from .diagnose_mean_random009 import (
    OutputStore, SOURCE_DEPENDENCIES, finite_moments, read_json,
    read_train_members, require, select_real, validate_train,
)
from .evaluate_window008 import _array, _clean, _memory_limit, array_sha, distance_receipt
from .electronics_v3 import ObservedBaselineNoiseV3
from .losses008_v2 import N01Loss008
from .model008 import N01Response008
from .observation008 import observe_generated_carrier
from .response008 import HYPOTHESES, fixed_noises, loss_tensor_statistics, make_extension, saturation_penalty, state_digest, statistics_digest
from .train import utc

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'shape010_20261014'
SEEDS = dict(initialization=2026101420, source_sampler=2026101421, real_sampler=2026101422,
    train_shape=2026101423, train_noise005=2026101424, train_extension=2026101425,
    train_projection=2026101426, evaluation_shape=2026101427, evaluation_noise005=2026101428,
    evaluation_extension=2026101429, evaluation_projection=2026101430,
    real_selection=2026101431, source_selection=2026101432)


def fixed_policy(path):
    p = read_json(path) if isinstance(path, (str, Path)) else copy.deepcopy(path)
    require(p.get('schema') == 'n01/shape010-policy/v1' and p.get('run_id') == RUN_ID
        and p.get('arm') == 'frozen_A_voltage_shape_noGAN' and p.get('seeds') == SEEDS,
        '010 identity or explicit seeds changed')
    require(p.get('TRAIN_counts') == dict(source=6593, real=21047) and p.get('G_updates') == 2000
        and all(p.get(k) == 0 for k in ('D_updates', 'noise_updates', 'population_updates', 'C2ST_updates'))
        and p.get('DEV_numeric_read') is False and p.get('automatic_restart') is False,
        '010 TRAIN/finite-update contract changed')
    t = p['training']
    expected = dict(source_parents_per_update=512, draws_per_parent=2, real_rows_per_update=1024,
        forward_chunk_rows=256, G_lr=.0002, G_weight_decay=.0001, clip_norm=5.,
        distribution_loss_stage='joint_stochastic', wave_weight=1., QPSD_weight=2., regular_weight=.01,
        adversarial_weight=0., discriminator_created=False, checkpoint_interval=1000, status_interval=100,
        early_stop_by_metric=False)
    require(all(t.get(k) == v for k, v in expected.items()), '010 frozen training recipe changed')
    require(p['shape']['basis_count'] == 8 and p['shape']['voltage_gain'] == .05
        and p['shape']['head_initialization_std'] == .01 and p['shape']['zero_canonical_Q'] is True
        and all(p['shape'][k] is False for k in ('charge_randomness', 'time_randomness', 'local_signal_noise')),
        '010 voltage-shape parameterization changed')
    e = p['evaluation']
    require((e['source_parents'], e['draws_per_parent'], e['matched_rows'], e['chunk_rows']) == (512, 4, 2048, 128)
        and e['split'] == 'TRAIN' and e['uses_DEV'] is False and e['adaptive_settings'] is False
        and e['before_training_fixed_baselines'] is True and e['REAL_REAL_disjoint'] is True
        and e['scientific_pass_threshold'] is None and e['final_evaluation_after_update'] == 2000,
        '010 fixed TRAIN evaluation changed')
    require(p['hypotheses'] == list(HYPOTHESES) and p['hypothesis_reduction'] == 'mean_of_separate_losses'
        and p['CPU_threads'] == 12 and p.get('overwrite_outputs') is False,
        '010 hypothesis/resource semantics changed')
    require(p['observation'] == dict(internal_samples=384, baseline_samples=100, effective_samples=256,
        peak_search=[112, 144], samples_before_peak=16, ADC_once=True, canonical_Q_slice=[110, 366]),
        '010 common observation changed')
    require(tuple(p['resource_limits'][k] for k in ('GPU_GiB', 'job_tree_RSS_GiB', 'wall_seconds', 'output_MiB'))
            == (10, 8, 7200, 1536), '010 resource bounds changed')
    return p


def bind_inputs(policy_path, bindings_path, output, *, root=ROOT):
    root = Path(root).resolve()
    pp, bp, out = [Path(v).resolve() for v in (policy_path, bindings_path, output)]
    p, b = fixed_policy(pp), read_json(bp)
    require(b['schema'] == 'n01/shape010-bindings/v1' and b.get('run_id') == p['run_id']
            and b['policy_sha256'] == sha256(pp), '010 binding identity changed')
    roots = [Path(v).resolve() for v in b['roots']]
    within = lambda x, allowed: any(x == r or r in x.parents for r in allowed)
    require(root in roots and within(pp, [root]) and within(bp, [root]) and within(out, [root])
        and out != root and not out.exists(), '010 fresh isolated-root paths required')
    if os.name == 'nt':
        require(out.drive.upper() in ('E:', 'D:'), '010 output must stay on E/D')
    else:
        require(root == Path(p['remote_root']).resolve()
            and set(roots) == {root, Path(p['immutable_endpoint_root']).resolve()}, '010 registered remote roots changed')
    record = b['source_manifest']
    mp = Path(record['path'])
    require(mp.is_absolute() and within(mp.resolve(), [root]) and sha256(mp) == record['sha256'], '010 source manifest changed')
    manifest = read_json(mp)
    require(manifest.get('schema') == 'n01/shape010-source/v1' and manifest.get('run_id') == RUN_ID
        and manifest.get('policy_sha256') == sha256(pp)
        and manifest.get('bound_inputs') == {name: record['sha256'] for name, record in b['files'].items()},
        '010 source manifest identity/policy/input binding differs')
    sources = {}
    for rel, digest in manifest['files'].items():
        path = (root/rel).resolve()
        require(not Path(rel).is_absolute() and within(path, [root]) and sha256(path) == digest, 'Unbound010 source: '+rel)
        sources[str(path)] = digest
    required_sources = {str(root/('src/n01/'+name+'.py')) for name in
                        set(SOURCE_DEPENDENCIES) | {'model010', 'train_shape010'}}
    require(required_sources <= set(sources), '010 numerical dependencies missing from new source manifest')
    required = {'dataset', 'statistics', 'data_statistics', 'noise005', 'deterministic', 'base_config', 'training_complete'}
    require(required <= set(b['files']) and 'full' not in b['files'], 'Wrong010 checkpoint/input inventory')
    files, paths = {}, {}
    for name, record in b['files'].items():
        path = Path(record['path'])
        require(path.is_absolute() and within(path.resolve(), roots) and sha256(path) == record['sha256'], '010 input changed before decode: '+name)
        paths[name] = path.resolve()
        files[str(path.resolve())] = record['sha256']
    files.update({str(pp): sha256(pp), str(bp): sha256(bp), str(mp.resolve()): sha256(mp)})
    return p, paths, dict(files_sha256=files, sources_sha256=sources, policy_sha256=sha256(pp),
        bindings_sha256=sha256(bp), before_TRAIN_decode=True, before_checkpoint_load=True, DEV_numeric_read=False)


def verify_files(freeze):
    for path, digest in {**freeze['files_sha256'], **freeze['sources_sha256']}.items():
        require(sha256(path) == digest, 'Frozen010 input/source changed: '+path)


class Streams010:
    def __init__(self, shape_seed, noise_seed, extension_seed):
        self.generators = {name: torch.Generator(device='cpu').manual_seed(seed) for name, seed in
            (('shape', shape_seed), ('noise005', noise_seed), ('extension', extension_seed))}

    def draw(self, rows, device):
        g = self.generators
        shape = torch.randn(rows, 8, generator=g['shape'], dtype=torch.float32).to(device)
        noise = {name: torch.randn((rows,)+dims, generator=g['noise005'], dtype=torch.float32).to(device)
                 for name, dims in ObservedBaselineNoiseV3.noise_shapes.items()}
        extension = torch.randn(rows, 1, 284, generator=g['extension'], dtype=torch.float32).to(device)
        return shape, noise, extension

    def state_dict(self):
        return {name: g.get_state().clone() for name, g in self.generators.items()}


class Progress010:
    def __init__(self, store, artificial=False):
        self.store, self.artificial = store, artificial
        self.counters = dict(G_updates=0, attempted_G_updates=0, D_updates=0, noise_updates=0,
            population_updates=0, C2ST_updates=0, train_response_rows=0, train_noise005_rows=0,
            train_hard_observation_rows=0, evaluation_response_rows=0, evaluation_noise005_rows=0,
            evaluation_hard_observation_rows=0)
        self.attempted = {k: 0 for k in self.counters}
        self.phase, self.in_flight, self.last_operation = 'initialized_before_numeric_work', None, None
        self.persist()

    def payload(self, status='running_shape010'):
        return dict(schema='n01/shape010-progress/v1', run_id=RUN_ID, UTC=utc(), pid=os.getpid(), status=status,
            phase=self.phase, counters=self.counters.copy(), attempted_operation_counters=self.attempted.copy(),
            in_flight=copy.deepcopy(self.in_flight), last_completed_operation=copy.deepcopy(self.last_operation),
            in_flight_completion='unknown until completed-operation receipt', artificial_only=self.artificial,
            DEV_numeric_read=False, automatic_restart=False)

    def persist(self, status='running_shape010'):
        self.store.status(self.payload(status))

    def call(self, operation, action, *, key=None, rows=0, **meta):
        if key:
            self.attempted[key] += rows
        self.phase = operation
        self.in_flight = dict(operation=operation, attempted_rows=rows, **meta)
        self.persist()
        result = action()
        if key:
            self.counters[key] += rows
        self.last_operation = dict(self.in_flight, completed=True)
        self.in_flight = None
        self.persist()
        return result


def frozen_hashes(model, extension):
    model.assert_frozen_unchanged(deep=True)
    extension.assert_frozen()
    return dict(baseline=state_digest(model.baseline.state_dict()), noise005=state_digest(extension.model005.state_dict()),
                extension=state_digest(extension.state_dict()))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


class Shape010Engine:
    def __init__(self, data, model, extension, statistics, policy, store, *, device='cpu', artificial=False, freeze=None):
        self.data, self.model, self.extension, self.statistics, self.policy = data, model, extension, statistics, policy
        self.store, self.device, self.freeze = store, torch.device(device), freeze
        self.progress = Progress010(store, artificial)
        validate_train(data, source_rows=policy['TRAIN_counts']['source'], real_rows=policy['TRAIN_counts']['real'])
        s, t = policy['seeds'], policy['training']
        self.source_sampler = FullCoverageSampler(len(data['source']), s['source_sampler'])
        self.real_sampler = FullCoverageSampler(len(data['real']), s['real_sampler'])
        self.streams = Streams010(s['train_shape'], s['train_noise005'], s['train_extension'])
        self.loss = N01Loss008(loss_tensor_statistics(statistics['loss']), seed=s['train_projection'],
                             evaluation_seed=s['evaluation_projection']).to(self.device)
        self.optimizer = torch.optim.AdamW(list(model.trainable_parameters()), lr=t['G_lr'], weight_decay=t['G_weight_decay'])
        model.assert_optimizer_membership(self.optimizer)
        self.whitelist = tuple(model.trainable_parameter_names())
        require(self.whitelist == ('basis_head.weight', 'basis_head.bias'), '010 trainable whitelist differs')
        self.frozen = frozen_hashes(model, extension)
        self.update, self.last_metrics = 0, {}

    def verify_frozen(self, files=False):
        self.model.assert_optimizer_membership(self.optimizer)
        require(frozen_hashes(self.model, self.extension) == self.frozen, '010 changed frozen A/005 parameters or buffers')
        if files and self.freeze:
            verify_files(self.freeze)

    def coverage(self):
        return dict(source_unique=int((self.source_sampler.visits > 0).sum()), source_total=self.source_sampler.n,
            real_unique=int((self.real_sampler.visits > 0).sum()), real_total=self.real_sampler.n,
            source_epochs=self.source_sampler.completed_epochs, real_epochs=self.real_sampler.completed_epochs,
            source_parent_visits=int(self.source_sampler.visits.sum()), real_row_visits=int(self.real_sampler.visits.sum()))

    def checkpoint(self, name, purpose):
        path = self.store.root/name
        require(not path.exists(), 'Preserve010 checkpoint: '+str(path))
        value = dict(schema='n01/shape010-checkpoint/v1', run_id=RUN_ID, purpose=purpose, G_updates=self.update,
            new_head=cpu_tree(self.model.basis_head.state_dict()), optimizer=cpu_tree(self.optimizer.state_dict()),
            source_sampler=self.source_sampler.state_dict(), real_sampler=self.real_sampler.state_dict(),
            streams=self.streams.state_dict(), counters=self.progress.counters.copy(),
            attempted=self.progress.attempted.copy(), in_flight=copy.deepcopy(self.progress.in_flight),
            frozen_state_sha256=self.frozen, trainable_parameter_names=list(self.whitelist),
            policy=self.policy, last_metrics=self.last_metrics, automatic_restart=False,
            artificial_only=self.progress.artificial,
            complete_step_boundary=purpose != 'failure', DEV_numeric_read=False)
        temporary = path.with_suffix('.tmp')
        with temporary.open('xb') as stream:
            torch.save(value, stream)
        temporary.replace(path)

    def step(self):
        p, t, progress = self.policy, self.policy['training'], self.progress
        self.store.guard(self.device)
        progress.counters['attempted_G_updates'] += 1
        progress.attempted['attempted_G_updates'] += 1
        attempted = self.update+1
        progress.phase = 'begin_G_attempt'
        progress.persist()
        parents = self.source_sampler.draw(t['source_parents_per_update'])
        si = np.repeat(parents, t['draws_per_parent'])
        ri = self.real_sampler.draw(t['real_rows_per_update'])
        shape, noise_latents, epsilon = self.streams.draw(len(si), self.device)
        noise = progress.call('training_frozen005_and_extensions', lambda: fixed_noises(self.extension, noise_latents, epsilon),
            key='train_noise005_rows', rows=len(si), attempted_G_update=attempted)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        signals, regulars = [], []
        prefix_max, q_error_max, random_rms_max, basis_q_max, random_q_max = 0., 0., 0., 0., 0.
        for start in range(0, len(si), t['forward_chunk_rows']):
            stop = min(start+t['forward_chunk_rows'], len(si))
            source = torch.as_tensor(self.data['source_carrier384'][si[start:stop]], dtype=torch.float64, device=self.device)
            output = progress.call('training_new_head_forward',
                lambda: self.model.random_forward(source, shape[start:stop], enabled=True),
                key='train_response_rows', rows=stop-start, start=start, stop=stop, attempted_G_update=attempted)
            signal_value = output['signal_pre']
            prefix_max = max(prefix_max, float(signal_value[:, :100].detach().abs().max()))
            q_error_max = max(q_error_max, float(output['canonical_charge_identity_residual'].detach().abs().max()))
            random_rms_max = max(random_rms_max, float(output['random_RMS_V'].detach().max()))
            basis_q_max = max(basis_q_max, float(output['basis_canonical_charge'].detach().abs().max()))
            random_q_max = max(random_q_max, float(output['canonical_random_shape_charge'].detach().abs().max()))
            require(torch.isfinite(signal_value).all() and prefix_max <= 1e-12 and q_error_max <= 1e-10,
                    '010 signal finite/prefix/canonical-Q engineering identity failed')
            signals.append(signal_value)
            regulars.append(output['regular']*(stop-start)/len(si))
        signal_value = torch.cat(signals)
        context = self.loss.create_projection_context(attempted)
        real = torch.as_tensor(self.data['real'][ri], dtype=torch.float64, device=self.device)
        rb = torch.as_tensor(self.data['real_baseline100'][ri], dtype=torch.float64, device=self.device)
        metrics, terms, saturation = {}, [], []
        for h in HYPOTHESES:
            obs = progress.call('training_ADC_peak_crop', lambda: observe_generated_carrier(signal_value+noise[h], ste=True),
                key='train_hard_observation_rows', rows=len(si), hypothesis=h, attempted_G_update=attempted)
            parts = self.loss.components(dict(waveform=obs['window_V'], baseline100=obs['baseline100']), real,
                stage='joint_stochastic', projection_context=context, real_baseline100=rb)
            terms.append(parts['wave']+2*parts['QPSD'])
            saturation.append(saturation_penalty(signal_value+noise[h]))
            metrics[h] = dict(wave=float(parts['wave'].detach()), QPSD=float(parts['QPSD'].detach()),
                              conditional_diagnostics=copy.deepcopy(self.loss.last_diagnostics))
        regular = sum(regulars)+torch.stack(saturation).mean()
        objective = torch.stack(terms).mean()+t['regular_weight']*regular
        require(torch.isfinite(objective), 'Nonfinite010 objective')
        progress.call('new_head_backward', lambda: objective.backward(), attempted_G_update=attempted)
        gradient = torch.nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), t['clip_norm'], error_if_nonfinite=True)
        before = {name: value.detach().clone() for name, value in self.model.basis_head.named_parameters()}
        # Count a G update immediately after AdamW returns; later QA failures do not erase it.
        progress.call('new_head_AdamW_step', lambda: self.optimizer.step(), key='G_updates', rows=1,
                      attempted_G_update=attempted)
        self.update = progress.counters['G_updates']
        changed = [name for name, value in self.model.basis_head.named_parameters() if not torch.equal(value.detach(), before[name])]
        require(all(torch.isfinite(value).all() for value in self.model.basis_head.parameters()), 'Nonfinite new-head parameter')
        self.last_metrics = dict(UTC=utc(), G_updates=self.update, attempted_G_updates=progress.counters['attempted_G_updates'],
            objective=float(objective.detach()), regular=float(regular.detach()), hypotheses=metrics,
            new_head_preclip_gradient_norm=float(gradient), changed_new_head_parameters=changed,
            signal_prefix_absmax_V=prefix_max, canonical_charge_identity_absmax=q_error_max,
            random_shape_RMS_absmax_V=random_rms_max, projection_context=context.metadata(), coverage=self.coverage(),
            basis_canonical_charge_absmax=basis_q_max, canonical_random_shape_charge_absmax=random_q_max,
            counters=progress.counters.copy(), D_updates=0, noise_updates=0, DEV_numeric_read=False)
        context.validate()
        if self.update == 1 or self.update % t['status_interval'] == 0:
            self.verify_frozen(files=self.update == 1)
            with (self.store.root/'training.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(_clean(self.last_metrics), allow_nan=False)+'\n')
            progress.phase = 'completed_G_update'
            progress.persist()
        if self.update == 1:
            require(float(gradient) > 0 and changed, 'First010 update has no gradient/parameter change')
            self.store.json('FIRST_UPDATE_PROOF.json', dict(schema='n01/shape010-first-update/v1', run_id=RUN_ID,
                pid=os.getpid(), actual_G_updates=1, frozen_state_sha256=self.frozen,
                frozen_A_005_unchanged=True, trainable_parameter_names=list(self.whitelist),
                TRAIN_counts=p['TRAIN_counts'], new_head_parameters_changed=True, new_head_gradient_nonzero=True,
                metrics=self.last_metrics, artificial_only=progress.artificial, DEV_numeric_read=False,
                no_discriminator_created=True, counters=progress.counters.copy()))
        return self.last_metrics


def evaluation_bank(data, policy):
    e, s = policy['evaluation'], policy['seeds']
    parents = np.random.default_rng(s['source_selection']).permutation(len(data['source']))[:e['source_parents']]
    a, b = select_real(len(data['real']), e['matched_rows'], s['real_selection'])
    streams = Streams010(s['evaluation_shape'], s['evaluation_noise005'], s['evaluation_extension'])
    shape, noise, epsilon = streams.draw(e['matched_rows'], 'cpu')
    return dict(parents=parents, real_a=a, real_b=b, shape=shape, noise=noise, extension=epsilon)


@torch.no_grad()
def evaluate(engine, bank, *, initial=False):
    model, data, p, progress = engine.model, engine.data, engine.policy, engine.progress
    e, device = p['evaluation'], engine.device
    parent_row = np.repeat(np.arange(e['source_parents']), e['draws_per_parent'])
    fake_views = ('frozen_A', 'shape010_initial') if initial else ('shape010_final',)
    pieces = {v: {h: {'waveform_V': [], 'pure_signal_V': [], 'peak_index': [], 'global_peak_index': [],
                            'crop_start': [], 'crop_stop': [], 'Q': [], 'Q_signal_crop_migration': [],
                            'Q_signal': [], 'Q_noise': [], 'Q_ADC_delta': []} for h in HYPOTHESES} for v in fake_views}
    baselines = []
    model.eval()
    for start in range(0, len(parent_row), e['chunk_rows']):
        stop = min(start+e['chunk_rows'], len(parent_row))
        source = torch.as_tensor(data['source_carrier384'][bank['parents'][parent_row[start:stop]]], dtype=torch.float64, device=device)
        noise = progress.call('evaluation_frozen005_and_extensions', lambda: fixed_noises(engine.extension,
            {k: value[start:stop].to(device) for k, value in bank['noise'].items()}, bank['extension'][start:stop].to(device)),
            key='evaluation_noise005_rows', rows=stop-start)
        shared = None
        for view in fake_views:
            out = progress.call('fixed_TRAIN_evaluation_forward',
                lambda: model.random_forward(source, bank['shape'][start:stop].to(device), enabled=view != 'frozen_A'),
                key='evaluation_response_rows', rows=stop-start, view=view, start=start, stop=stop)
            require(float(out['signal_pre'][:, :100].abs().max()) <= 1e-12
                and float(out['canonical_charge_identity_residual'].abs().max()) <= 1e-10, '010 evaluation signal engineering failed')
            for h in HYPOTHESES:
                signal_value = out['signal_pre']
                carrier = signal_value+noise[h]
                obs = progress.call('fixed_TRAIN_evaluation_hard_observation',
                    lambda: observe_generated_carrier(carrier, ste=False, return_indices=True, return_carrier=True),
                    key='evaluation_hard_observation_rows', rows=stop-start, view=view, hypothesis=h)
                if shared is None:
                    shared = obs['baseline100']
                    baselines.append(_array(shared))
                require(torch.equal(shared, obs['baseline100']), '010 evaluation noise is not paired')
                values = dict(waveform_V=obs['window_V'], pure_signal_V=signal_value,
                              **{k: obs[k] for k in ('peak_index', 'global_peak_index', 'crop_start', 'crop_stop', 'Q')})
                delta = obs['adc_carrier']-carrier
                for key, term in (('signal', signal_value), ('noise', noise[h]), ('ADC_delta', delta)):
                    crop = term.gather(1, obs['crop_indices'])-term[:, :100].mean(1, keepdim=True)
                    values['Q_'+key] = crop.sum(1)
                values['Q_signal_crop_migration'] = values['Q_signal']-out['qsig']
                require(float((values['Q']-sum(values['Q_'+k] for k in ('signal', 'noise', 'ADC_delta'))).abs().max()) <= 1e-10,
                        '010 same-observed-crop Q identity failed')
                for key, value in values.items():
                    pieces[view][h][key].append(_array(value))
        engine.store.guard(device)
    baseline = np.concatenate(baselines)
    context = engine.loss.create_evaluation_context()
    distances = {}
    for view in fake_views:
        distances[view] = {}
        for h in HYPOTHESES:
            values = {k: np.concatenate(v) for k, v in pieces[view][h].items()}
            engine.store.npz(view+'_'+h+'_TRAIN.npz', **values, source_parent_row=parent_row,
                             source_parent_ids=data['source_ids'][bank['parents'][parent_row]])
            engine.store.npz(view+'_'+h+'_PARENT_MOMENTS.npz', **finite_moments(values['waveform_V'].reshape(
                e['source_parents'], e['draws_per_parent'], 256)))
            distances[view][h] = distance_receipt(engine.loss, values['waveform_V'], data['real'][bank['real_a']],
                baseline, data['real_baseline100'][bank['real_a']], context)
    if initial:
        distances['REAL_REAL'] = distance_receipt(engine.loss, data['real'][bank['real_b']], data['real'][bank['real_a']],
            data['real_baseline100'][bank['real_b']], data['real_baseline100'][bank['real_a']], context)
    engine.verify_frozen(files=True)
    engine.store.json('INITIAL_EVALUATION.json' if initial else 'FINAL_EVALUATION.json', dict(
        distances=distances, projection_context=context.metadata(), source_parent_clusters=e['source_parents'],
        repeats=e['draws_per_parent'], matched_rows=e['matched_rows'], G_updates=engine.update,
        DEV_numeric_read=False, adaptive_settings=False, scientific_pass_threshold=None))
    return distances


def run_arrays(data, model, extension, statistics, policy, output, *, device='cpu', artificial=False, freeze=None, store=None):
    store = store or OutputStore(output, policy)
    engine = None
    try:
        engine = Shape010Engine(data, model, extension, statistics, policy, store, device=device, artificial=artificial, freeze=freeze)
        store.json('FROZEN_INPUTS.json', dict(run_id=RUN_ID, binding=freeze, frozen_state_sha256=engine.frozen,
            trainable_parameter_names=list(engine.whitelist), artificial_only=artificial, DEV_numeric_read=False))
        engine.checkpoint('initial.pt', 'initial')
        bank = evaluation_bank(data, policy)
        store.npz('EVALUATION_BANK.npz', selected_source_rows=bank['parents'], selected_source_ids=data['source_ids'][bank['parents']],
            real_reference_rows=bank['real_a'], real_control_rows=bank['real_b'], shape_z=_array(bank['shape']),
            extension=_array(bank['extension']), **{k: _array(v) for k, v in bank['noise'].items()})
        initial = evaluate(engine, bank, initial=True)
        while engine.update < policy['G_updates']:
            engine.step()
            if engine.update % policy['training']['checkpoint_interval'] == 0:
                engine.verify_frozen(files=True)
                engine.checkpoint('final.pt' if engine.update == policy['G_updates'] else 'periodic_%05d.pt'%engine.update,
                                  'final' if engine.update == policy['G_updates'] else 'periodic')
        if not (store.root/'final.pt').exists():
            engine.checkpoint('final.pt', 'final')
        final = evaluate(engine, bank)
        engine.verify_frozen(files=True)
        require(engine.coverage()['source_unique'] == len(data['source']) and engine.coverage()['real_unique'] == len(data['real']),
                'Bounded010 training failed full legal TRAIN coverage')
        engine.progress.phase = 'completed_fixed_budget_and_same_bank_TRAIN_evaluation'
        engine.progress.persist('shape010_training_and_TRAIN_evaluation_complete')
        complete = dict(schema='n01/shape010-complete/v1', run_id=RUN_ID, status='shape010_training_and_TRAIN_evaluation_complete',
            G_updates=engine.update, counters=engine.progress.counters.copy(), coverage=engine.coverage(),
            frozen_state_sha256=engine.frozen, frozen_A_005_unchanged=True, trainable_parameter_names=list(engine.whitelist),
            initial_evaluation=initial, final_evaluation=final, DEV_numeric_read=False, artificial_only=artificial,
            independent_acquisition_available=False, scientific_status='response_match_not_established',
            budget_is_convergence_claim=False, resources=store.guard(device),
            artifacts={v.name: dict(sha256=sha256(v), bytes=v.stat().st_size) for v in sorted(store.root.iterdir()) if v.is_file()})
        store.json('COMPLETE.json', complete)
        return _clean(complete)
    except BaseException as error:
        failure = dict(schema='n01/shape010-failure/v1', run_id=RUN_ID, UTC=utc(), error_type=type(error).__name__, error=str(error),
                       artificial_only=artificial, DEV_numeric_read=False, automatic_restart=False)
        if engine is not None:
            failure.update(progress=engine.progress.payload('failed_shape010'), completed_G_updates=engine.progress.counters['G_updates'],
                in_flight_actual_completion='unknown; never asserted complete', coverage=engine.coverage())
            engine.update = engine.progress.counters['G_updates']
        # Preserve original error and completed counts before potentially failing
        # serialization (OOM/disk interruption must not turn actual G into zero).
        store.json('FAILED.json', failure)
        store.status(failure)
        if engine is not None:
            try:
                engine.checkpoint('failure.pt', 'failure')
                store.json('FAILURE_CHECKPOINT_RECEIPT.json', dict(path='failure.pt',
                    sha256=sha256(store.root/'failure.pt'), completed_G_updates=engine.update,
                    purpose='failure', automatic_restart=False))
            except BaseException as checkpoint_error:
                store.json('FAILURE_CHECKPOINT_ERROR.json', dict(error_type=type(checkpoint_error).__name__,
                    error=str(checkpoint_error), original_error_type=type(error).__name__,
                    original_error=str(error), completed_G_updates=engine.update,
                    original_failure_receipt_preserved=True, checkpoint_completion='not confirmed'))
        raise


def run(policy_path, bindings_path, output, device='cuda:0'):
    policy, paths, freeze = bind_inputs(policy_path, bindings_path, output)
    store = OutputStore(output, policy)
    try:
        store.json('BINDING.json', freeze)
        stats = read_json(paths['statistics'])
        domain = read_json(paths['data_statistics'])
        cfg = read_json(paths['base_config'])
        require(stats.get('statistics_split') == domain.get('split') == stats['loss']['fit_split'] == 'TRAIN'
            and stats.get('DEV_waveforms_decoded') is False and domain.get('DEV_waveforms_decoded') is False
            and stats['prepared_dataset_sha256'] == sha256(paths['dataset'])
            and stats['data_statistics_sha256'] == sha256(paths['data_statistics']), '010 TRAIN preparation/statistics changed')
        torch.set_num_threads(policy['CPU_threads'])
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        freeze['CUDA_limit'] = _memory_limit(dict(gpu_memory_limit_GiB=policy['resource_limits']['GPU_GiB']), device)
        with np.load(paths['dataset'], allow_pickle=False) as archive:
            data = read_train_members(archive)
        validate_train(data)
        checkpoint = torch.load(paths['deterministic'], map_location='cpu', weights_only=False)
        require(checkpoint.get('schema') == 'response008/v1' and checkpoint.get('arm') == 'deterministic'
            and checkpoint.get('G_updates') == 19000 and checkpoint.get('stage') == 'low_learning_rate'
            and checkpoint.get('D_updates') == {h: 16000 for h in HYPOTHESES}
            and checkpoint.get('noise_updates') == 0 and checkpoint.get('population_learning') is False
            and checkpoint.get('config') == cfg and statistics_digest(checkpoint['statistics']) == statistics_digest(stats),
            '010 frozen A checkpoint metadata differs')
        baseline = N01Response008(stats['model'], 'deterministic')
        baseline.load_state_dict(checkpoint['model'], strict=True)
        baseline.set_stage(checkpoint['stage'])
        baseline.requires_grad_(False)
        baseline.eval()
        baseline.to(device)
        noise = torch.load(paths['noise005'], map_location='cpu', weights_only=False)
        require(noise.get('update') == 5000 and noise.get('schema_version') == 5
            and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
                '010 frozen005 endpoint differs')
        extension = make_extension(noise, device)
        require(state_digest(extension.model005.state_dict()) == checkpoint['noise005_state_sha256'], '010 A/005 reference mismatch')
        from .model010 import FrozenDeterministicVoltageShape010
        model = FrozenDeterministicVoltageShape010(baseline, seed=policy['seeds']['initialization'],
            amplitude_fraction=policy['shape']['voltage_gain'])
        verify_files(freeze)
        return run_arrays(data, model, extension, stats, policy, output, device=device, freeze=freeze, store=store)
    except BaseException as error:
        if not (store.root/'FAILED.json').exists():
            store.json('FAILED.json', dict(run_id=RUN_ID, error_type=type(error).__name__, error=str(error),
                completed_G_updates=0, phase='binding_or_prepared_TRAIN_or_frozen_checkpoint_loading', DEV_numeric_read=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned010 process received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = run(args.policy, args.bindings, args.output, args.device)
    print(json.dumps({k: result[k] for k in ('status', 'run_id', 'G_updates', 'counters', 'frozen_A_005_unchanged')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
