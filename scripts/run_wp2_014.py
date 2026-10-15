"""Bounded, serial WP2 C0/C1 pairs. Drafts cannot load private arrays or models.

Stdlib verifies the frozen source/input/contract inventory before numeric imports.
There is no resume, C2, adversary, population fit, DEV read, or implicit RNG.
The supervisor and deployment remain the existing, independently checked ops.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from wp1_ops import Store, clean, read_json, require, sha, unchanged, write_new


RUN_ID = 'wp2_014_20261015'
HYPOTHESES = ('fast', 'slow', 'common_level')
OWNED_FILES = ('scripts/run_wp2_014.py', 'src/n01/model014.py', 'src/n01/losses014.py')
TOLERANCE_SHA014 = 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'


def seed_set014(initialization):
    """Explicit stream addresses; no generator is shared between stream names."""
    return dict(initialization=initialization, source_sampler=initialization+1,
        real_sampler=initialization+2, train_shape=initialization+3,
        train_noise005=initialization+4, train_extension=initialization+5,
        train_projection=initialization+6, integral_projection=initialization+7,
        calibration_selection=initialization+8, calibration_shape=[initialization+9, initialization+10],
        probe_selection=initialization+11, probe_real=initialization+12,
        probe_banks=[dict(shape=initialization+13, noise005=initialization+14, extension=initialization+15),
                     dict(shape=initialization+16, noise005=initialization+17, extension=initialization+18)])


def settings014():
    """Non-executable settings proposal; root supplies bindings and final freeze."""
    return dict(schema='n01/wp2-014-config/v1', run_id=RUN_ID, status='draft',
        candidates=['C0', 'C1'], seed_sets=[seed_set014(2026101540), seed_set014(2026101640)],
        G_updates=2000, TRAIN_counts=dict(source=6593, real=21047),
        training=dict(source_parents_per_update=512, draws_per_parent=2, real_rows_per_update=1024,
            forward_chunk_rows=256, G_lr=.0002, G_weight_decay=.0001, clip_norm=5.,
            wave_weight=1., distribution_weight=2., regular_weight=.01, status_interval=100,
            checkpoint_steps=[1000, 2000]),
        calibration=dict(parents=128, draws=4, banks=2, RMS_upper_bound_ADC_LSB=1., bisections=16),
        probe=dict(steps=[0, 2000], parents=64, draws=2, banks=2,
            strata=['all', 'SOURCE_Q_quartiles_4', 'output_Q_quartiles_4', 'predeclared_single_parents_4'],
            single_parent_positions=[0, 16, 32, 48], scalar_AD_terms=3,
            real_reference_rows=128),
        evaluation=dict(parents=6593, draws=4, banks=2, chunk_rows=128,
            bank_seeds=[], projection_seed=None, whole_population=True,
            snapshot_arms=dict(initial=['A', 'initial'], final=['final'])),
        hypotheses=list(HYPOTHESES), CPU_threads=12,
        run_resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=7200, output_MiB=1536),
        resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=28800, output_MiB=6144,
            poll_seconds=10, TERM_grace_seconds=20),
        one_active_scientific_job=True, serial_runs=True, automatic_restart=False,
        overwrite_outputs=False, DEV_numeric_read=False,
        scientific_status='response_match_not_established')


def budget_caps014(p):
    """Worst-case actual operations, including extra scalar AD and calibration.

    S=seeds, R=2S runs, U=G/run, B=train parents*draws,
    E=all SOURCE parents*evaluation draws, K=eval banks,
    P=probe parents*draws, L=probe banks, J=probe snapshots, Z=13 strata.
    Old A/head/005 train=RUB, evaluation=2RKE, probe=R J L P.
    Initial hard observation has A+initial; final has final: 3 R K E H.
    Probe scalar AD <=R J L H Z*3; total vector is a sum, no fourth AD.
    Calibration A=S C; actual head=2 S C; maps<=S C D Kc*(16+2).
    Each step has one backward and one AdamW. Checkpoint loads=2 total.
    """
    seeds, runs, updates = len(p['seed_sets']), 2*len(p['seed_sets']), p['G_updates']
    t, c, v, e = (p[k] for k in ('training', 'calibration', 'probe', 'evaluation'))
    h = len(HYPOTHESES)
    train = runs*updates*t['source_parents_per_update']*t['draws_per_parent']
    evaluation = 2*runs*e['banks']*e['parents']*e['draws']
    probe = runs*len(v['steps'])*v['banks']*v['parents']*v['draws']
    probe_ad = runs*len(v['steps'])*v['banks']*h*13*3
    head = train+evaluation+probe+2*seeds*c['parents']
    calibration_maps = seeds*c['parents']*c['draws']*c['banks']*(c['bisections']+2)
    result = dict(old_A_rows=train+evaluation+probe+seeds*c['parents'],
        head_actual_parameter_forward_rows=head, head_total_kernel_rows=head+calibration_maps,
        calibration_map_rows=calibration_maps,
        noise005_rows=train+evaluation+probe,
        hard_ADC_rows=h*(train+probe)+3*runs*e['banks']*e['parents']*e['draws']*h,
        all_path_gather_rows=h*(train+probe)+3*runs*e['banks']*e['parents']*e['draws']*h,
        checkpoint_loads=2, optimizer_created=runs, optimizer_steps=runs*updates,
        G_updates=runs*updates, training_backward_queries=runs*updates,
        probe_AD_queries=probe_ad, new_AD_queries=runs*updates+probe_ad,
        integral_statistics_fits=1, D_updates=0, noise_updates=0,
        population_updates=0, C2ST_updates=0, DEV_model_rows=0)
    return result


def gate014(config_path, expected_sha, output):
    cp = Path(config_path).resolve()
    require(cp.is_file() and sha(cp) == expected_sha, '014 expected configuration SHA required')
    p = read_json(cp)
    require(p.get('schema') == 'n01/wp2-014-config/v1' and p.get('run_id') == RUN_ID
        and p.get('status') == 'frozen', '014 draft cannot execute scientific operations')
    reference = settings014()
    for key in ('candidates', 'seed_sets', 'G_updates', 'TRAIN_counts', 'training', 'calibration', 'probe',
                'hypotheses', 'CPU_threads', 'run_resource_limits', 'one_active_scientific_job',
                'serial_runs', 'automatic_restart', 'overwrite_outputs', 'DEV_numeric_read', 'resource_limits'):
        require(p.get(key) == reference[key], '014 predeclared setting changed: '+key)
    e = p['evaluation']
    for key in ('parents', 'draws', 'banks', 'chunk_rows', 'whole_population', 'snapshot_arms'):
        require(e.get(key) == reference['evaluation'][key], '014 full evaluation contract changed: '+key)
    require(len(e['bank_seeds']) == 2 and type(e['projection_seed']) is int,
        '014 explicit evaluation streams/projection required')
    require(p.get('budget_caps') == budget_caps014(p), '014 exact finite operation cap/formula mismatch')
    require(p.get('inputs') and set(('dataset', 'statistics', 'data_statistics', 'base_config', 'deterministic',
        'noise005', 'source_information')) <= set(p['inputs']), '014 bound input inventory incomplete')
    require(set(('observation', 'data_roles', 'acceptance', 'tolerance_table')) <= set(p.get('contracts', {})),
        '014 final frozen tolerance and contracts required before any candidate')
    require(p['contracts']['tolerance_table']['sha256'] == TOLERANCE_SHA014,
        '014 T must be the root-frozen pre-candidate exact WP1 table')
    require(set(OWNED_FILES) <= set(p.get('source_files', {})), '014 own source bytes not frozen')
    roots = [Path(r).resolve() for r in p['allowed_input_roots']]
    require(roots and cp.parent in roots, '014 launch root is not explicitly registered')
    checked, paths = {str(cp): expected_sha}, {}
    records = {**{'source:'+k:v for k, v in p['source_files'].items()},
        **{'input:'+k:v for k, v in p['inputs'].items()},
        **{'contract:'+k:v for k, v in p['contracts'].items()},
        **{'protected:'+k:v for k, v in p.get('protection_files', {}).items()}}
    for label, record in records.items():
        path = Path(record['path'])
        require(path.is_absolute() and path.resolve().is_file()
            and any(path.resolve().is_relative_to(root) for root in roots), '014 file escaped frozen roots: '+label)
        if os.name == 'nt':
            require(path.drive.upper() in ('E:', 'D:'), '014 scientific/source files must stay E/D')
        require(sha(path) == record['sha256'], '014 frozen bytes changed: '+label)
        if label.startswith('source:'):
            rel = label[7:]
            require(not Path(rel).is_absolute() and path.resolve() == (Path(__file__).resolve().parents[1]/rel).resolve(),
                '014 bound source does not name the actually imported launch-tree file: '+rel)
        key = str(path.resolve())
        require(key not in checked or checked[key] == record['sha256'], '014 conflicting SHA alias')
        checked[key] = record['sha256']
        if label.startswith('input:'):
            paths[label[6:]] = path.resolve()
    target = Path(output).resolve()
    require(not target.exists() and target != cp.parent and target.is_relative_to(cp.parent), '014 fresh output within launch root required')
    if os.name == 'nt':
        require(target.drive.upper() in ('E:', 'D:'), '014 output must stay on E/D')
    return p, paths, checked


class Store014(Store):
    """Reuse generic durable entered/completed accounting, with an honest 014 schema."""

    def base(self):
        row = super().base()
        row.update(schema='n01/wp2-014-state/v1', scientific_training_updates=self.counts.get('G_updates', 0),
                   scope='serial C0/C1 paired bounded optimization')
        return row

    def call(self, name, charges, function, **detail):
        charges = dict(charges)
        self.phase = name
        if 'head_actual_parameter_forward_rows' in charges or 'calibration_map_rows' in charges:
            charges['head_total_kernel_rows'] = charges.get('head_actual_parameter_forward_rows', 0)+charges.get('calibration_map_rows', 0)
        return super().call(name, charges, function, **detail)


def arr(value):
    return value.detach().cpu().numpy()


def current_rss014():
    """Existing Linux/tree reader, plus stdlib Windows process fallback.

    This worker starts no subprocess. The external owned-tree supervisor also
    enforces the 8 GiB process-tree cap. No dependency installation is needed.
    """
    from n01.diagnose_mean_random009 import current_rss
    rss = current_rss()
    if rss is not None or os.name != 'nt':
        return rss
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
            ('PrivateUsage', ctypes.c_size_t)]
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    query.restype = wintypes.BOOL
    row = Counters()
    row.cb = ctypes.sizeof(row)
    if not query(wintypes.HANDLE(-1), ctypes.byref(row), row.cb):
        raise OSError(ctypes.get_last_error(), '014 native process RSS query failed')
    return row.WorkingSetSize


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    require(not path.exists(), '014 preserve existing checkpoint: '+str(path))
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Runtime014:
    def __init__(self, p, data, baseline, extension, statistics, integral_statistics, store, device, source_info):
        import numpy as np
        import torch
        from n01.response008 import state_digest
        self.p, self.data, self.baseline, self.extension = p, data, baseline, extension
        self.statistics, self.integral_statistics, self.store = statistics, integral_statistics, store
        self.device, self.info = torch.device(device), source_info
        self.q_floor = float(torch.as_tensor(statistics['loss']['q_floor'], dtype=torch.float64))
        self.frozen = dict(baseline=state_digest(baseline.state_dict()),
            noise005=state_digest(extension.model005.state_dict()), extension=state_digest(extension.state_dict()))
        carrier = np.asarray(data['source_carrier384'], np.float64)
        centered = carrier-carrier[:, :100].mean(1, keepdims=True)
        self.source_Q = centered[:, 110:366].sum(1)
        self.source_Q_bounds = np.quantile(self.source_Q, [.25, .5, .75])
        self.output_Q_bounds = np.quantile(np.asarray(data['real']).sum(1), [.25, .5, .75])

    def call(self, name, charges, fn, **meta):
        import torch
        def synced():
            value = fn()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            return value
        return self.store.call(name, charges, synced, **meta)

    def verify(self, model=None, *, files=False):
        from n01.response008 import state_digest
        self.extension.assert_frozen()
        if model is not None:
            model.assert_frozen_unchanged(deep=True)
        current = dict(baseline=state_digest(self.baseline.state_dict()),
            noise005=state_digest(self.extension.model005.state_dict()), extension=state_digest(self.extension.state_dict()))
        require(current == self.frozen, '014 changed original A/005 parameters or buffers')
        if files:
            unchanged(self.store.checked)

    def forward(self, model, x, z):
        old = model._baseline_once
        def baseline_once(values):
            return self.call('original_frozen_A', dict(old_A_rows=len(values)), lambda: old(values))
        model._baseline_once = baseline_once
        try:
            out = self.call('actual_new_head_forward', dict(head_actual_parameter_forward_rows=len(x)), lambda: model(x, z))
        finally:
            del model._baseline_once
        require(float(out['canonical_charge_identity_residual'].detach().abs().max()) <= 1e-10
            and bool((out['signal_pre'][:, :100] == 0).all()), '014 signal prefix/canonical-Q identity failed')
        return out

    def noise(self, latents, epsilon):
        from n01.response008 import fixed_noises
        return self.call('frozen005_shared_extensions', dict(noise005_rows=len(epsilon)),
                         lambda: fixed_noises(self.extension, latents, epsilon))

    def observe(self, carrier, *, ste):
        from n01.observation008 import observe_generated_carrier
        return self.call('hard_ADC_first_argmax_crop_with_STE' if ste else 'hard_ADC_first_argmax_crop',
            dict(hard_ADC_rows=len(carrier), all_path_gather_rows=len(carrier)),
            lambda: observe_generated_carrier(carrier, ste=ste, q_floor=self.q_floor, return_carrier=True, return_indices=True))

    def guard(self, directory, started):
        import torch
        limits = self.p['run_resource_limits']
        require(time.monotonic()-started <= limits['wall_seconds'], '014 run wall cap exceeded')
        rss = current_rss014()
        require(rss is not None and rss <= limits['job_tree_RSS_GiB']*1024**3, '014 process RSS unavailable or cap exceeded')
        size = sum(q.stat().st_size for q in Path(directory).rglob('*') if q.is_file())
        require(size <= limits['output_MiB']*1024**2, '014 run output cap exceeded')
        if self.device.type == 'cuda':
            require(torch.cuda.memory_allocated(self.device) <= limits['GPU_GiB']*1024**3, '014 GPU cap exceeded')
        return dict(output_bytes=size, process_RSS_bytes=rss, elapsed_seconds=time.monotonic()-started)

    def calibrate(self, seed, directory):
        import numpy as np
        import torch
        from n01.losses008 import ADC_STEP
        from n01.model014 import FrozenDeterministicVoltageShape014, calibrate_initial_head014
        from n01.train_shape010 import cpu_tree
        c = self.p['calibration']
        model = FrozenDeterministicVoltageShape014(self.baseline, seed=seed['initialization'])
        parents = np.sort(np.random.default_rng(seed['calibration_selection']).choice(len(self.source_Q), c['parents'], replace=False))
        x = torch.as_tensor(self.data['source_carrier384'][parents], dtype=torch.float64, device=self.device)
        original, decoded = self.call('initial_RMS_frozen_A_cache', dict(old_A_rows=len(x)), lambda: model._baseline_once(x))
        banks = torch.stack([torch.randn(c['parents'], c['draws'], 8,
            generator=torch.Generator(device='cpu').manual_seed(s), dtype=torch.float32)
            for s in seed['calibration_shape']]).to(self.device)
        result = calibrate_initial_head014(model, decoded, original['signal_pre'], banks,
            target_RMS_V=c['RMS_upper_bound_ADC_LSB']*ADC_STEP, bisections=c['bisections'], operation=self.call)
        result.update(source_parent_rows=parents.tolist(), source_parent_ids=self.data['source_ids'][parents].tolist(),
            shape_bank_seeds=seed['calibration_shape'], RNG_global_preserved=model.initialization_global_CPU_RNG_preserved,
            paired_candidates_share_calibrated_head_bytes=True, actual_RMS_ADC_LSB=result['actual_RMS_V']/ADC_STEP)
        directory.mkdir()
        write_new(directory/'CALIBRATION.json', result)
        initial = cpu_tree(model.basis_head.state_dict())
        atomic_checkpoint(directory/'initial_head.pt', dict(schema='n01/wp2-calibrated-head/v1', new_head=initial, calibration=result))
        self.verify(model)
        return initial, result


class Engine014:
    def __init__(self, runtime, seed, candidate, initial_head, calibration, directory):
        import torch
        from n01.data import FullCoverageSampler
        from n01.model014 import FrozenDeterministicVoltageShape014
        from n01.losses014 import N01Loss014
        from n01.response008 import loss_tensor_statistics
        from n01.train_shape010 import Streams010
        self.rt, self.seed, self.candidate, self.out = runtime, seed, candidate, Path(directory)
        self.out.mkdir()
        self.started, self.update = time.monotonic(), 0
        self.model = FrozenDeterministicVoltageShape014(runtime.baseline, seed=seed['initialization'])
        self.model.basis_head.load_state_dict(initial_head, strict=True)
        self.loss = N01Loss014(loss_tensor_statistics(runtime.statistics['loss']), runtime.integral_statistics,
            seed=seed['train_projection'], evaluation_seed=runtime.p['evaluation']['projection_seed'],
            integral_seed=seed['integral_projection']).to(runtime.device)
        self.source_sampler = FullCoverageSampler(len(runtime.data['source']), seed['source_sampler'])
        self.real_sampler = FullCoverageSampler(len(runtime.data['real']), seed['real_sampler'])
        self.streams = Streams010(seed['train_shape'], seed['train_noise005'], seed['train_extension'])
        t = runtime.p['training']
        self.optimizer = runtime.call('new_head_AdamW_created', dict(optimizer_created=1),
            lambda: torch.optim.AdamW(list(self.model.trainable_parameters()), lr=t['G_lr'], weight_decay=t['G_weight_decay']))
        self.model.assert_optimizer_membership(self.optimizer)
        self.last_metrics, self.probe_initial_peaks = {}, {}
        write_new(self.out/'RUN_INPUTS.json', dict(schema='n01/wp2-run-inputs/v1', candidate=candidate,
            seed=seed, calibration=calibration, frozen_A_005_sha256=runtime.frozen,
            trainable_parameter_names=list(self.model.trainable_parameter_names()),
            only_candidate_difference='native QPSD versus fixed signed-asinh joint integrals',
            tolerance_table=runtime.p['contracts']['tolerance_table'], DEV_numeric_read=False,
            statistical_scope='TRAIN development evidence, not unseen confirmation'))

    def checkpoint(self, name, purpose):
        from n01.train_shape010 import cpu_tree
        atomic_checkpoint(self.out/name, dict(schema='n01/wp2-checkpoint/v1', candidate=self.candidate,
            initial_seed=self.seed['initialization'], G_updates=self.update, purpose=purpose,
            new_head=cpu_tree(self.model.basis_head.state_dict()), optimizer=cpu_tree(self.optimizer.state_dict()),
            source_sampler=self.source_sampler.state_dict(), real_sampler=self.real_sampler.state_dict(),
            streams=self.streams.state_dict(), last_metrics=self.last_metrics,
            frozen_state_sha256=self.rt.frozen, counters=self.rt.store.counts.copy(),
            inflight=copy.deepcopy(self.rt.store.inflight), complete_step_boundary=purpose != 'failure', automatic_restart=False))

    def coverage(self):
        return dict(source_unique=int((self.source_sampler.visits > 0).sum()), source_total=self.source_sampler.n,
            real_unique=int((self.real_sampler.visits > 0).sum()), real_total=self.real_sampler.n,
            source_parent_visits=int(self.source_sampler.visits.sum()), real_row_visits=int(self.real_sampler.visits.sum()),
            source_epochs=self.source_sampler.completed_epochs, real_epochs=self.real_sampler.completed_epochs)

    def step(self):
        import numpy as np
        import torch
        from n01.response008 import saturation_penalty
        rt, model, t = self.rt, self.model, self.rt.p['training']
        rt.guard(self.out, self.started)
        parents = self.source_sampler.draw(t['source_parents_per_update'])
        si = np.repeat(parents, t['draws_per_parent'])
        ri = self.real_sampler.draw(t['real_rows_per_update'])
        z, nz, epsilon = self.streams.draw(len(si), rt.device)
        noise = rt.noise(nz, epsilon)
        self.optimizer.zero_grad(set_to_none=True)
        model.train()
        outputs = []
        for start in range(0, len(si), t['forward_chunk_rows']):
            x = torch.as_tensor(rt.data['source_carrier384'][si[start:start+t['forward_chunk_rows']]], dtype=torch.float64, device=rt.device)
            outputs.append(rt.forward(model, x, z[start:start+len(x)]))
        signal_value = torch.cat([o['signal_pre'] for o in outputs])
        shape_regular = sum(o['regular']*len(o['signal_pre'])/len(si) for o in outputs)
        context = self.loss.create_projection_context(self.update+1)
        real, rb = [torch.as_tensor(rt.data[n][ri], dtype=torch.float64, device=rt.device) for n in ('real', 'real_baseline100')]
        waves, distributions, saturations, metrics = [], [], [], {}
        for h in HYPOTHESES:
            obs = rt.observe(signal_value+noise[h], ste=True)
            parts = self.loss.training_components(dict(waveform=obs['window_V'], baseline100=obs['baseline100']), real,
                candidate=self.candidate, projection_context=context, real_baseline100=rb)
            waves.append(parts['wave'])
            distributions.append(parts['distribution'])
            saturations.append(saturation_penalty(signal_value+noise[h]))
            metrics[h] = dict(wave=float(parts['wave'].detach()), training_distribution=float(parts['distribution'].detach()),
                QPSD_native=float(parts['QPSD_native'].detach()), diagnostics=parts['diagnostics'])
        wave = torch.stack(waves).mean()
        distribution = torch.stack(distributions).mean()
        regular = shape_regular+torch.stack(saturations).mean()
        objective = wave+2*distribution+t['regular_weight']*regular
        require(bool(torch.isfinite(objective)), '014 nonfinite objective')
        rt.call('one_training_backward', dict(training_backward_queries=1, new_AD_queries=1), lambda: objective.backward())
        params = list(model.trainable_parameters())
        gradient = torch.nn.utils.clip_grad_norm_(params, t['clip_norm'], error_if_nonfinite=True)
        before = torch.cat([v.detach().reshape(-1).double().clone() for v in params])
        model.assert_optimizer_membership(self.optimizer)
        rt.call('one_new_head_AdamW_step', dict(optimizer_steps=1, G_updates=1), lambda: self.optimizer.step())
        self.update += 1
        after = torch.cat([v.detach().reshape(-1).double() for v in params])
        delta = after-before
        require(bool(torch.isfinite(after).all()), '014 nonfinite updated head')
        self.last_metrics = dict(G_updates=self.update, candidate=self.candidate, initial_seed=self.seed['initialization'],
            objective=float(objective.detach()), wave=float(wave.detach()), weighted_distribution=float((2*distribution).detach()),
            weighted_regular=float((t['regular_weight']*regular).detach()), hypotheses=metrics,
            preclip_total_gradient_norm=float(gradient), actual_parameter_update_norm=float(delta.norm()),
            actual_parameter_update_absmax=float(delta.abs().max()), parameters_changed=bool((delta != 0).any()),
            source_parent_rows=parents.tolist(), real_row_count=len(ri), coverage=self.coverage(),
            projection_context=context.metadata(), all_legal_TRAIN_sampling=True, D_updates=0, noise_updates=0)
        if self.update == 1 or self.update % t['status_interval'] == 0:
            rt.verify(model, files=self.update == 1)
            with (self.out/'training.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(clean(self.last_metrics), allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            rt.store.status(dict(current_run=self.out.name, current_run_G_updates=self.update))
        if self.update == 1:
            require(float(gradient) > 0 and bool((delta != 0).any()), '014 first step has no actual gradient/update')
            write_new(self.out/'FIRST_UPDATE_PROOF.json', dict(schema='n01/wp2-first-real-update/v1',
                actual_G_updates=1, candidate=self.candidate, initial_seed=self.seed['initialization'],
                frozen_A_005_unchanged=True, frozen_state_sha256=rt.frozen,
                actual_optimizer='AdamW new-head-only', trainable_parameter_names=list(model.trainable_parameter_names()),
                head_parameters=1288, gradient_nonzero=True, actual_parameter_update_nonzero=True,
                artificial_only=rt.store.artificial, metrics=self.last_metrics))
            if not (rt.store.root/'FIRST_UPDATE_PROOF.json').exists():
                write_new(rt.store.root/'FIRST_UPDATE_PROOF.json', dict(rt.store.base(), schema='n01/wp2-first-real-update/v1',
                    actual_G_updates=1, batch_completed_G_updates=rt.store.counts['G_updates'],
                    trial=self.out.name, trial_proof_sha256=sha(self.out/'FIRST_UPDATE_PROOF.json'),
                    initial_seed=self.seed['initialization'], candidate=self.candidate,
                    frozen_A_005_unchanged=True, gradient_nonzero=True, actual_parameter_update_nonzero=True,
                    head_parameters=1288, artificial_only=rt.store.artificial,
                    trainable_parameter_names=list(model.trainable_parameter_names()), frozen_state_sha256=rt.frozen))
        context.validate()
        return self.last_metrics

    def probe(self):
        """Conditional subset diagnostics, never an additive event attribution."""
        import numpy as np
        import torch
        from n01.train_shape010 import Streams010
        from n01.response008 import saturation_penalty
        from wp1_evaluation import observable_features, output_q_bins
        from run_wp1_013 import observation_rows
        rt, p = self.rt, self.rt.p['probe']
        parents = np.sort(np.random.default_rng(self.seed['probe_selection']).choice(len(rt.source_Q), p['parents'], replace=False))
        source_rows = np.repeat(parents, p['draws'])
        real_rows = np.sort(np.random.default_rng(self.seed['probe_real']).choice(len(rt.data['real']), p['real_reference_rows'], replace=False))
        x = torch.as_tensor(rt.data['source_carrier384'][source_rows], dtype=torch.float64, device=rt.device)
        real, rb = [torch.as_tensor(rt.data[n][real_rows], dtype=torch.float64, device=rt.device) for n in ('real', 'real_baseline100')]
        all_records, raw_arrays, vectors = [], {}, {}
        self.model.train()
        parameters = list(self.model.trainable_parameters())
        accumulated_before = [None if v.grad is None else v.grad.detach().clone() for v in parameters]
        context = self.loss.create_projection_context(self.update)
        for bank, seeds in enumerate(self.seed['probe_banks']):
            streams = Streams010(seeds['shape'], seeds['noise005'], seeds['extension'])
            z, nz, epsilon = streams.draw(len(source_rows), rt.device)
            noise = rt.noise(nz, epsilon)
            out = rt.forward(self.model, x, z)
            for h in HYPOTHESES:
                carrier = out['signal_pre']+noise[h]
                obs = rt.observe(carrier, ste=True)
                raw = observation_rows(obs, carrier, self.loss)
                features, flags = observable_features(raw['window_V'], raw['baseline100'], float(self.loss.q_floor))
                key = f'bank{bank}_{h}'
                if self.update == 0:
                    self.probe_initial_peaks[key] = raw['peak_index'].copy()
                initial_peak = self.probe_initial_peaks.get(key)
                require(initial_peak is not None, '014 probe final requires its fixed initial peak reference')
                peak_switch = raw['peak_index'] != initial_peak
                source_bin = np.searchsorted(rt.source_Q_bounds, rt.source_Q[source_rows], side='left')
                output_bin = np.searchsorted(rt.output_Q_bounds, features['Q_Vns'], side='left')
                raw_arrays[key+'__source_parent_row'] = source_rows
                raw_arrays[key+'__source_parent_id'] = rt.data['source_ids'][source_rows]
                raw_arrays[key+'__draw_index'] = np.tile(np.arange(p['draws']), len(parents))
                raw_arrays[key+'__source_Q_quartile'] = source_bin
                raw_arrays[key+'__output_Q_quartile'] = output_bin
                raw_arrays[key+'__native_output_Q_bin'] = output_q_bins(features['Q_Vns'], self.loss)
                raw_arrays[key+'__peak_switch_vs_initial'] = peak_switch
                for n in ('Q_Vns', 'tail_Vns', 'front_Vns', 'PSD_raw', 'PSD_guarded'):
                    raw_arrays[key+'__'+n] = features[n]
                for n, value in raw.items():
                    if n.startswith(('flag__', 'diagnostic__')):
                        raw_arrays[key+'__'+n] = value
                strata = [('all', np.ones(len(source_rows), bool))]
                strata += [('SOURCE_Q_quartile_%d'%i, source_bin == i) for i in range(4)]
                strata += [('generated_output_Q_quartile_%d'%i, output_bin == i) for i in range(4)]
                strata += [('single_parent_position_%d'%i, source_rows == parents[pos])
                           for i, pos in enumerate(p['single_parent_positions'])]
                for name, mask in strata:
                    row = dict(bank=bank, hypothesis=h, stratum=name, G_updates=self.update,
                        fake_rows=int(mask.sum()), fake_parent_count=int(len(np.unique(source_rows[mask]))),
                        real_reference_rows=len(real_rows), real_reference_parent_unit='original REAL record',
                        whole_probe_rows=len(source_rows), whole_probe_parents=len(parents),
                        Q_nonpositive_rows=int((features['Q_Vns'][mask] <= 0).sum()),
                        PSD_undefined_rows=int((~np.isfinite(features['PSD_raw'][mask])).sum()),
                        PSD_guarded_rows=int((np.abs(features['Q_Vns'][mask]) < float(self.loss.q_floor)).sum()),
                        peak_switch_rows=int(peak_switch[mask].sum()),
                        interpretation='conditional subset loss gradients; not global event loss contributions or percentages',
                        subset_gradients_additive=False, fixed_real_reference_all_rows=True)
                    if not mask.any():
                        all_records.append(dict(row, status='empty_stratum', AD_queries=0, gradients=None))
                        continue
                    indices = torch.as_tensor(np.flatnonzero(mask), device=rt.device)
                    parts = self.loss.training_components(dict(waveform=obs['window_V'][indices], baseline100=obs['baseline100'][indices]),
                        real, candidate=self.candidate, projection_context=context, real_baseline100=rb)
                    random_shape = out['random_shape_pre_V'][indices]
                    regular = (random_shape.square().mean()+torch.diff(random_shape, n=2, dim=-1).square().mean())/self.model.baseline.a_ref.square()
                    regular = regular+saturation_penalty(carrier[indices])
                    terms = dict(wave=parts['wave'], weighted_distribution=2*parts['distribution'], weighted_regular=.01*regular)
                    gradients = {}
                    for term_name, value in terms.items():
                        def query(value=value):
                            gs = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=False)
                            return torch.cat([g.detach().double().reshape(-1) for g in gs])
                        vector = rt.call('predeclared_subset_scalar_AD', dict(probe_AD_queries=1, new_AD_queries=1), query,
                            G_updates=self.update, bank=bank, hypothesis=h, stratum=name, term=term_name)
                        require(bool(torch.isfinite(vector).all()), '014 nonfinite subset parameter gradient')
                        gradients[term_name] = vector
                    gradients['total'] = sum(gradients.values())
                    norms = {n: float(v.norm()) for n, v in gradients.items()}
                    def cosine(a, b):
                        return None if norms[a] == 0 or norms[b] == 0 else float(torch.dot(gradients[a], gradients[b])/(norms[a]*norms[b]))
                    cosines = {a+'__'+b: cosine(a, b) for ai, a in enumerate(gradients) for b in list(gradients)[ai+1:]}
                    for n, value in gradients.items():
                        vectors[key+'__'+name+'__'+n] = arr(value)
                    all_records.append(dict(row, status='computed', AD_queries=3,
                        scalar_values={n: float(v.detach()) for n, v in terms.items()}, gradient_norms=norms,
                        cosine=cosines, total_vector_rule='wave + weighted_distribution + weighted_regular; no fourth AD',
                        native_QPSD=float(parts['QPSD_native'].detach()), diagnostics=parts['diagnostics'],
                        actual_last_training_parameter_update_norm=self.last_metrics.get('actual_parameter_update_norm'),
                        probe_updates_parameters=False, source_parent_rows=np.unique(source_rows[mask]).tolist()))
        # autograd.grad must preserve even the preceding G step's accumulators.
        for value, before in zip(parameters, accumulated_before):
            require((before is None and value.grad is None) or
                (before is not None and value.grad is not None and torch.equal(before, value.grad)),
                '014 probe mutated a head gradient accumulator')
        rt.verify(self.model)
        directory = self.out/('probe_%04d'%self.update)
        directory.mkdir()
        with (directory/'RAW_PARENT_ROWS.npz').open('xb') as f:
            np.savez_compressed(f, **raw_arrays)
        with (directory/'GRADIENT_VECTORS.npz').open('xb') as f:
            np.savez_compressed(f, **vectors)
        write_new(directory/'SUMMARY.json', dict(schema='n01/wp2-stratified-gradient/v1', candidate=self.candidate,
            G_updates=self.update, seed=self.seed, source_Q_quartile_bounds=rt.source_Q_bounds,
            generated_output_Q_quartile_bounds_REAL_TRAIN=rt.output_Q_bounds,
            row_in_any_stratum_retained=True, records=all_records,
            extra_scalar_AD_queries=sum(r['AD_queries'] for r in all_records),
            no_per_event_loss_contribution_claim=True, no_optimizer_step=True,
            projection_context=context.metadata()))
        context.validate()
        rt.guard(self.out, self.started)
        return all_records

    def evaluate(self, *, initial):
        """All legal TRAIN parents, raw signed Q/PSD, both fixed MC banks."""
        import numpy as np
        import torch
        from n01.train_shape010 import Streams010
        from wp1_evaluation import evaluate_hard_population, component_distances, observable_features
        from run_wp1_013 import observation_rows, observation_booleans, lossless_window_codes, save_report, REAL_UNKNOWN_FLAGS
        rt, e = self.rt, self.rt.p['evaluation']
        directory = self.out/('initial_evaluation' if initial else 'final_evaluation')
        directory.mkdir()
        parent_rows = np.repeat(np.arange(len(rt.data['source'])), e['draws'])
        draw_index = np.tile(np.arange(e['draws']), len(rt.data['source']))
        arms = e['snapshot_arms']['initial' if initial else 'final']
        context = self.loss.create_evaluation_context()
        metrics = {}
        cached_real = observable_features(rt.data['real'], rt.data['real_baseline100'], float(self.loss.q_floor))
        self.model.eval()
        with torch.no_grad():
            for bank, seeds in enumerate(e['bank_seeds']):
                streams = Streams010(seeds['shape'], seeds['noise005'], seeds['extension'])
                pieces = {arm:{h:[] for h in HYPOTHESES} for arm in arms}
                codes = {arm:{h:[] for h in HYPOTHESES} for arm in arms}
                baseline_pieces = []
                for start in range(0, len(parent_rows), e['chunk_rows']):
                    selected = parent_rows[start:start+e['chunk_rows']]
                    x = torch.as_tensor(rt.data['source_carrier384'][selected], dtype=torch.float64, device=rt.device)
                    z, nz, epsilon = streams.draw(len(x), rt.device)
                    noise = rt.noise(nz, epsilon)
                    out = rt.forward(self.model, x, z)
                    shared_baseline = None
                    for arm in arms:
                        signal_value = out['base_signal_pre_V'] if arm == 'A' else out['signal_pre']
                        for h in HYPOTHESES:
                            carrier = signal_value+noise[h]
                            obs = rt.observe(carrier, ste=False)
                            rows = observation_rows(obs, carrier, self.loss)
                            if shared_baseline is None:
                                shared_baseline = rows['baseline100']
                                baseline_pieces.append(shared_baseline)
                            else:
                                require(np.array_equal(shared_baseline, rows['baseline100']), '014 unpaired electronics prefix')
                            codes[arm][h].append(lossless_window_codes(rows))
                            pieces[arm][h].append({n:v for n, v in rows.items() if n not in ('adc_carrier', 'baseline100')})
                    rt.guard(self.out, self.started)
                baseline = np.concatenate(baseline_pieces)
                # Each snapshot retains the exact used baseline, without silently
                # substituting its arithmetic from a previous run or WP1.
                with (directory/('bank%d_BASELINE100.npz'%bank)).open('xb') as f:
                    np.savez_compressed(f, baseline100=baseline, source_parent_row=parent_rows, draw_index=draw_index)
                for arm in arms:
                    for h in HYPOTHESES:
                        joined = {n:np.concatenate([r[n] for r in pieces[arm][h]]) for n in pieces[arm][h][0]}
                        storage = {n:np.concatenate([r[n] for r in codes[arm][h]]) for n in codes[arm][h][0]
                                   if n != 'IEEE754_reconstruction_XOR_uint64'}
                        # XOR can be absent for some chunks and present for others.
                        if any('IEEE754_reconstruction_XOR_uint64' in r for r in codes[arm][h]):
                            storage['IEEE754_reconstruction_XOR_uint64'] = np.concatenate([
                                r.get('IEEE754_reconstruction_XOR_uint64', np.zeros(r['ADC_crop_codes'].shape, np.uint64))
                                for r in codes[arm][h]])
                        stem = f'bank{bank}_{arm}_{h}'
                        storage.update(source_parent_row=parent_rows, source_parent_ids=rt.data['source_ids'][parent_rows],
                            draw_index=draw_index, bank_index=np.full(len(parent_rows), bank, np.int64))
                        storage.update({n:v for n, v in joined.items() if n != 'window_V'})
                        with (directory/(stem+'_HARD_ROWS.npz')).open('xb') as f:
                            np.savez_compressed(f, **storage)
                        report = evaluate_hard_population(joined['window_V'], baseline,
                            real_wave256=rt.data['real'], real_baseline100=rt.data['real_baseline100'],
                            parent_row=parent_rows, draw_index=draw_index, bank_index=np.full(len(parent_rows), bank, np.int64),
                            source_information=rt.info, loss=self.loss, context=context,
                            observation_flags=observation_booleans(joined), real_observation_flags=rt.data.get('real_observation_flags'),
                            real_unknown_flags=REAL_UNKNOWN_FLAGS, real_groups=rt.data['real_groups'], cached_real=cached_real)
                        save_report(directory, stem+'_REPORT', report)
                        metric = component_distances(joined['window_V'], baseline, rt.data['real'], rt.data['real_baseline100'],
                            self.loss, context, cached_reference=cached_real)
                        metrics[f'bank{bank}_{arm}_{h}'] = metric
                        rt.guard(self.out, self.started)
        write_new(directory/'METRICS.json', dict(schema='n01/wp2-full-hard-metrics/v1', metrics=metrics,
            source_parents=len(rt.data['source']), draws_per_bank=e['draws'], MC_banks=e['banks'], REAL_rows=len(rt.data['real']),
            statistical_unit='SOURCE_parent_and_original_REAL_group; MC draws are not new physical records',
            no_DEV=True, all_signed_rows_retained=True, tolerance_table=rt.p['contracts']['tolerance_table'],
            all_metric_classes=True, root_acceptance_pending=True, scientific_acceptance=False))
        rt.verify(self.model, files=True)
        return metrics


def compare_metrics014(initial, final, tolerance):
    result = []
    T = tolerance['T_by_hypothesis']
    for key, final_report in final.items():
        bank, _, hypothesis = key.split('_', 2)
        reference, start = initial[f'{bank}_A_{hypothesis}']['values'], initial[f'{bank}_initial_{hypothesis}']['values']
        values = final_report['values']
        checks = {}
        for metric, value in values.items():
            bound, a, i = T[hypothesis].get(metric), reference.get(metric), start.get(metric)
            valid = bound is not None and math.isfinite(bound) and bound > 0
            delta_A = None if value is None or a is None else value-a
            delta_init = None if value is None or i is None else value-i
            checks[metric] = dict(final=value, A=a, common_initial=i, T=bound,
                delta_final_minus_A=delta_A, delta_final_minus_initial=delta_init,
                absolute_within_T=None if not valid or value is None else value <= bound,
                noninferior_vs_A=None if not valid or delta_A is None else delta_A <= bound,
                noninferior_vs_initial=None if not valid or delta_init is None else delta_init <= bound,
                T_status='calibrated' if valid else 'not_calibrated_or_undefined_not_passed')
        improve_A = None if not reference.get('wave') else 1-values['wave']/reference['wave']
        improve_initial = None if not start.get('wave') else 1-values['wave']/start['wave']
        result.append(dict(bank=bank, hypothesis=hypothesis, checks=checks,
            relative_hard_wave_improvement_vs_A=improve_A,
            relative_hard_wave_improvement_vs_common_initial=improve_initial,
            wave_usefulness_2percent=None if improve_A is None else improve_A >= .02,
            native_QPSD_reported=True, legacy_J_is_auxiliary=True,
            root_scientific_acceptance_pending=True, scientific_acceptance=False))
    return result


def prepare_numeric014(p, paths, store, device):
    import numpy as np
    import torch
    from n01.model008 import N01Response008
    from n01.response008 import make_extension, statistics_digest, state_digest
    from n01.diagnose_mean_random009 import read_train_members, validate_train
    from n01.losses014 import fit_integral_statistics014
    from run_wp1_013 import information_join, REAL_METADATA
    stats, domain, cfg = [read_json(paths[k]) for k in ('statistics', 'data_statistics', 'base_config')]
    require(stats.get('statistics_split') == domain.get('split') == stats['loss']['fit_split'] == 'TRAIN'
        and stats.get('DEV_waveforms_decoded') is False and domain.get('DEV_waveforms_decoded') is False
        and stats['prepared_dataset_sha256'] == p['inputs']['dataset']['sha256']
        and stats['data_statistics_sha256'] == p['inputs']['data_statistics']['sha256'], '014 original TRAIN statistics provenance changed')
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
        for name in REAL_METADATA:
            data[name] = archive[name].copy()
    validate_train(data)
    data['real_observation_flags'] = dict(peak_tied=data['real_peak_tied_samples'] > 1,
        peak_at_search_boundary=data['real_peak_search_edge'], global_peak_outside_search=data['real_global_peak_outside_search'])
    def load(key):
        return store.call('checkpoint_'+key, dict(checkpoint_loads=1), lambda: torch.load(paths[key], map_location='cpu', weights_only=False))
    a = load('deterministic')
    require(a.get('schema') == 'response008/v1' and a.get('arm') == 'deterministic' and a.get('G_updates') == 19000
        and a.get('stage') == 'low_learning_rate' and a.get('D_updates') == {h:16000 for h in HYPOTHESES}
        and a.get('noise_updates') == 0 and a.get('population_learning') is False and a.get('config') == cfg
        and statistics_digest(a['statistics']) == statistics_digest(stats), '014 original A metadata mismatch')
    baseline = N01Response008(stats['model'], 'deterministic')
    baseline.load_state_dict(a['model'], strict=True)
    baseline.set_stage(a['stage'])
    baseline.eval()
    baseline.requires_grad_(False)
    baseline.to(device)
    noise = load('noise005')
    require(noise.get('schema_version') == 5 and noise.get('update') == 5000
        and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
        '014 original005 metadata mismatch')
    extension = make_extension(noise, device)
    require(state_digest(extension.model005.state_dict()) == a['noise005_state_sha256'], '014 frozen A/005 binding mismatch')
    integral = store.call('fixed_REAL_TRAIN_integral_scale', dict(integral_statistics_fits=1),
        lambda: fit_integral_statistics014(torch.as_tensor(data['real'], dtype=torch.float64)))
    store.json('INTEGRAL_TRAIN_STATISTICS.json', integral)
    info = information_join(data, paths['source_information'])
    return data, baseline, extension, stats, integral, info


def run014(config, config_sha256, output, device):
    p, paths, checked = gate014(config, config_sha256, output)
    # Numeric package import and every scientific action are below the full gate.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = str(p['CPU_threads'])
    import torch
    from n01.evaluate_window008 import _memory_limit
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']), device)
    store = Store014(output, p, checked)
    store.status()
    engine = None
    try:
        store.json('BINDING.json', dict(configuration_sha256=config_sha256, immutable_files=checked,
            tolerance_table=p['contracts']['tolerance_table'], budget_caps=p['budget_caps'], before_numeric_decode=True))
        data, baseline, extension, stats, integral, info = prepare_numeric014(p, paths, store, device)
        runtime = Runtime014(p, data, baseline, extension, stats, integral, store, device, info)
        tolerance = read_json(p['contracts']['tolerance_table']['path'])
        completed = []
        for index, seed in enumerate(p['seed_sets']):
            initial, calibration = runtime.calibrate(seed, store.root/('seed%d_CALIBRATION'%index))
            for candidate in p['candidates']:
                engine = Engine014(runtime, seed, candidate, initial, calibration, store.root/f'seed{index}_{candidate}')
                if not (store.root/'INIT_PROOF.json').exists():
                    store.json('INIT_PROOF.json', dict(store.base(), schema='n01/wp2-init-proof/v1', run_id=RUN_ID,
                        pid=os.getpid(), native_initialized=True, checkpoint_loads=store.counts['checkpoint_loads'],
                        frozen_state_sha256=runtime.frozen, head_parameters=1288,
                        trainable_parameter_names=list(engine.model.trainable_parameter_names()),
                        calibration=calibration, paired_initialization_bytes=True,
                        first_trial=engine.out.name, optimizer_created=store.counts['optimizer_created'],
                        G_updates=0, no_C2=True, artificial_only=store.artificial,
                        tolerance_table=p['contracts']['tolerance_table']))
                engine.checkpoint('initial.pt', 'initial')
                initial_metrics = engine.evaluate(initial=True)
                engine.probe()
                while engine.update < p['G_updates']:
                    engine.step()
                    if engine.update in p['training']['checkpoint_steps']:
                        runtime.verify(engine.model, files=True)
                        engine.checkpoint('final.pt' if engine.update == p['G_updates'] else 'step_%04d.pt'%engine.update,
                                          'final' if engine.update == p['G_updates'] else 'periodic')
                engine.probe()
                final_metrics = engine.evaluate(initial=False)
                comparison = compare_metrics014(initial_metrics, final_metrics, tolerance)
                write_new(engine.out/'HARD_COMPARISON.json', dict(schema='n01/wp2-hard-comparison/v1',
                    candidate=candidate, initial_seed=seed['initialization'], per_bank_hypothesis=comparison,
                    frozen_tolerance=p['contracts']['tolerance_table'], root_final_choice_pending=True))
                coverage = engine.coverage()
                require(coverage['source_unique'] == len(data['source']) and coverage['real_unique'] == len(data['real']),
                    '014 bounded training failed all legal TRAIN coverage')
                runtime.verify(engine.model, files=True)
                result = dict(schema='n01/wp2-trial-complete/v1', native_status='COMPLETE', candidate=candidate, seed=seed,
                    G_updates=engine.update, coverage=coverage, actual_calibration=calibration,
                    frozen_A_005_unchanged=True, frozen_state_sha256=runtime.frozen,
                    last_metrics=engine.last_metrics, per_bank_hypothesis=comparison,
                    resources=runtime.guard(engine.out, engine.started),
                    raw_artifacts_retained_on_execution_host=True, automatic_restart=False,
                    scientific_status='response_match_not_established', root_selection_pending=True)
                write_new(engine.out/'COMPLETE.json', result)
                completed.append(dict(directory=engine.out.name, candidate=candidate,
                    initial_seed=seed['initialization'], G_updates=engine.update, COMPLETE_sha256=sha(engine.out/'COMPLETE.json')))
                engine = None
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
        require(not store.inflight, '014 batch cannot complete with unknown in-flight scientific operations')
        unchanged(checked)
        inventory = {str(q.relative_to(store.root)).replace('\\','/'):dict(sha256=sha(q),bytes=q.stat().st_size)
            for q in sorted(store.root.rglob('*')) if q.is_file() and q.name != 'status.json'}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp2-artifact-inventory/v1', files=inventory,
            collection_policy='small immutable receipts first; raw private arrays remain on execution host until E/D space permits'))
        result = dict(store.base(), schema='n01/wp2-batch-complete/v1', native_status='COMPLETE',
            trials=completed, G_updates_per_trial=p['G_updates'], all_four_trials_completed=True,
            all_frozen_A_005_unchanged=True, no_C2=True, no_DEV=True, scientific_acceptance=False,
            scientific_status='response_match_not_established', root_choice_pending=True)
        store.json('COMPLETE.json', result)
        store.status(dict(native_status='COMPLETE'))
        return result
    except BaseException as error:
        failure = dict(store.base(), schema='n01/wp2-batch-failure/v1', native_status='FAILED',
            error_type=type(error).__name__, error=str(error), automatic_restart=False,
            completed_G_updates=store.counts.get('G_updates', 0),
            current_trial=None if engine is None else engine.out.name,
            current_trial_G_updates=None if engine is None else engine.update,
            inflight_completion='unknown; never asserted completed', preserve_all_completed_costs=True)
        store.json('FAILED.json', failure)
        store.status(failure)
        if engine is not None:
            try:
                engine.checkpoint('failure.pt', 'failure')
                write_new(engine.out/'FAILED.json', failure)
            except BaseException as saving_error:
                store.json('FAILURE_CHECKPOINT_ERROR.json', dict(error=repr(saving_error), original_error=repr(error),
                    completed_G_updates=store.counts.get('G_updates', 0)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned014 process received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = run014(args.config, args.config_sha256, args.output, args.device)
    print(json.dumps({k:result[k] for k in ('native_status', 'scientific_training_updates', 'all_four_trials_completed')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

