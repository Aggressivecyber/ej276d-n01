"""Recover completed019 seed0 evaluation, then train only original T1 unpaired seed1."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

from wp1_ops import clean, read_json, require, sha, unchanged, write_new
from run_wp4_019 import (settings019, budget_caps019, SCIENCE_SOURCES019,
    verify019 as verify020, teacher_data019 as teacher_data020, gate_trials019)

RUN_ID = 'wp4_020_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_020_20261016'
TRIALS = ('T1_unpaired_seed0', 'T1_unpaired_seed1')
HYPOTHESES = ('fast', 'slow', 'common_level')
SCIENCE_SOURCES020 = SCIENCE_SOURCES019 + ('scripts/run_wp4_020.py', 'scripts/mechanism012_losses.py')
SEED0_CP_SHA = 'ad285f938745d906df7295433ea9d1c3bc83d181e1c26db16375df7f2af48b79'
AUTHORIZATION_SHA = 'd1dcdd2002cf52c792aa307e14cc11fb3b1ad25d035ff0cd46a8bb84cc0040fa'


def settings020():
    p = settings019()
    p.update(schema='n01/wp4-020-config/v1', run_id=RUN_ID, remote_root=REMOTE,
        training_seed_sets=[1], seed0_new_G_AD=0, new_G_AD_cap=2000,
        restored_seed0_source_run_id='wp4_019_20261016', restored_seed0_source_G=2000,
        restored_seed0_checkpoint_sha256=SEED0_CP_SHA,
        restored_seed0_optimizer_created=False, restored_seed0_parent_samplers_created=0,
        autonomous_engineering_recovery_authorized=True,
        necessary_additional_computation_authorized=True, authorization_sha256=AUTHORIZATION_SHA,
        native_invocations_per_attempt=1, internal_retry_loop=False,
        repeated_failed_evaluation_prefix=dict(signal_rows=128, total_scalar_normal_requests=50432,
            all_three_extension_rows=384, hard_ADC_rows=128),
        original019_native1_preserved=True)
    return p


def budget_caps020(p):
    n, k, g, training_trials, evaluation_trials = 6593, 512, 2000, 1, 2
    evaluation = evaluation_trials*n*2*32
    teacher = training_trials*g*2*k
    student = teacher+evaluation
    total = student+teacher
    caps = {name: 0 for name in budget_caps019(settings019())}
    caps.update(trial_slots=1, G_updates=g, optimizer_steps=g,
        training_backward_queries=g, new_AD_queries=g, optimizers_created=1,
        checkpoint_loads=2, restored_seed0_checkpoint_loads=1,
        student_charge_module_rows=student, teacher_charge_module_rows=teacher,
        shape_cached_basis_assembly_rows=total, noise005_rows=total,
        all_three_extensions_rows=3*total, hard_ADC_rows=3*total, hard_first_argmax_gather_rows=3*total,
        training_student_rows=teacher, training_teacher_rows=teacher,
        evaluation_kernel_snapshots=evaluation_trials, evaluation_signal_rows=evaluation,
        normal_shape_scalar_requests=8*total, normal_charge_scalar_requests=total,
        normal_noise005_scalar_requests=101*total, normal_extension284_scalar_requests=284*total,
        total_scalar_normal_requests=394*total,
        parent_sampler_streams=4, parent_randint_calls=4*g, parent_randint_scalar_requests=4*g*k,
        unpaired_training_energy_pair_distances=g*(2*k)**2*2*3,
        unpaired_training_cross_retained_pairs=g*(2*k)**2*3,
        unpaired_training_self_ordered_retained_pairs=g*(2*k)*(2*k-1)*3,
        unpaired_training_self_diagonal_computed_then_masked=g*(2*k)*3,
        evaluation_energy_candidate_comparisons=evaluation_trials,
        evaluation_energy_pair_distances=evaluation_trials*4*n*3*32**2,
        checkpoint_saves_complete=2, noise005_checkpoint_loads=1,
        cached_array_mmap_files=3, reused_teacher_kernel_sets=1, reused_initial_score_sets=1,
        evaluation_observable_pair_distances=(1+2*evaluation_trials)*n*3*64**2)
    return caps


def validate_settings020(p, *, frozen=True, driver_source=None):
    reference = settings020()
    require(not frozen or p['status'] == 'frozen', '020 draft cannot execute')
    require(all(p.get(k) == value for k, value in reference.items() if k != 'status'),
            '020 changed the prospectively fixed settings')
    require(p['budget_caps'] == budget_caps020(p), '020 exact additive budget differs')



def gate020(config, expected_sha, output):
    cp = Path(config).resolve()
    require(sha(cp) == expected_sha, '020 frozen config SHA differs')
    p = read_json(cp)
    validate_settings020(p)
    require(set(SCIENCE_SOURCES020) <= set(p['source_files']), '020 mandatory science source closure is incomplete')
    require(all(Path(p['source_files'][name]['path']) == Path(p['remote_root'])/name for name in SCIENCE_SOURCES020),
            '020 mandatory imported source must be the bound owned copy')
    require(Path(output).absolute() == Path(p['remote_root'])/'runs/primary/artifacts'
            and not Path(output).exists(), '020 exact fresh owned output required')
    manifest_path = cp.parent/p['saved_input_manifest']
    require(sha(manifest_path) == p['saved_input_manifest_sha256'], '020 saved input manifest differs')
    manifest = read_json(manifest_path)
    require(manifest['file_count'] == 40 and p['inputs'] == manifest['entries'], '020 exact saved input whitelist differs')
    require(p['contracts']['prerequisite_018'] == manifest['entries']['prerequisite018'], '020 prerequisite binding differs')
    records = {**p['source_files'], **{'contract:'+k: v for k, v in p['contracts'].items()},
               **{'protection:'+k: v for k, v in p['protection_files'].items()},
               **{'input:'+k: v for k, v in manifest['entries'].items()}}
    checked = {str(cp): expected_sha, str(manifest_path): sha(manifest_path)}
    roots = [Path(x).resolve() for x in p['allowed_input_roots']]
    for label, record in records.items():
        path = Path(record['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(r) for r in roots),
                '020 input/source escaped frozen roots: '+label)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], '020 frozen bytes differ: '+label)
        checked[str(path)] = record['sha256']
    original018 = read_json(manifest['entries']['config018']['path'])
    original_gate = read_json(manifest['entries']['gate018']['path'])
    require(original018['scales'] == p['scales'] and original018['time_scale_software_ns'] == 1.
        and original018['new_observable_energy_tolerance'] == .10
        and original018['new_observable_MC_half_width_limit'] == .05
        and original_gate['status'] == 'pass' and original_gate['required_cell_count'] == 2544,
        '020 requires the same accepted018 numerical contract and complete gate')
    prerequisite = read_json(p['contracts']['prerequisite_018']['path'])
    require(prerequisite['schema'] == 'n01/wp4-018-final-result/v1' and
        prerequisite['run_id'] == 'wp4_018_20261016' and prerequisite['native_exit_code'] == 0 and
        prerequisite['qualified'] is True and prerequisite['required_cells'] == 2544 and
        prerequisite['independent_review_pass'] is True and prerequisite['old017_pending_preserved'] is True,
        '020 requires completed independently accepted018; producer pass alone is insufficient')
    require(manifest['entries']['seed0_final019']['sha256'] == SEED0_CP_SHA and
        manifest['entries']['seed0_final019']['bytes'] == 114968, '020 original completed seed0 CP identity')
    require(p['contracts']['result_019'] == manifest['entries']['result019'] and
        p['contracts']['user_authorization'] == manifest['entries']['authorization020'], '020 recovery evidence bindings')
    prior019 = read_json(p['contracts']['result_019']['path'])
    require(prior019['schema'] == 'n01/wp4-019-final-result/v1' and prior019['run_id'] == 'wp4_019_20261016'
        and prior019['native_exit_code'] == 1 and prior019['seed0_training_G_AD'] == 2000
        and prior019['seed1_started'] is False and prior019['gate_generated'] is False,
        '020 must preserve actual019 failure and completed seed0 scope')
    authorization = read_json(p['contracts']['user_authorization']['path'])
    require(p['contracts']['user_authorization']['sha256'] == AUTHORIZATION_SHA and
        authorization['resume_authorized'] is True and
        authorization['autonomous_engineering_restart_and_recovery_authorized'] is True and
        authorization['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True and
        authorization['immediate_registered_scope_to_implement']['new_run_id'] == RUN_ID,
        '020 latest direct human recovery authorization')
    return p, {k: Path(v['path']) for k, v in manifest['entries'].items()}, checked



def write_generated020(path, value, *, output, store):
    """Version only newly generated metadata; numerical algorithms/namespaces stay017."""
    require(Path(path).resolve().is_relative_to(output.resolve()), '020 metadata write escaped new output')
    schema = value['schema'] if 'schema' in value else None
    if schema is not None and 'wp4-017' in schema:
        value['implementation_schema_origin'] = schema
        value['schema'] = schema.replace('wp4-017', 'wp4-020')
    value['run_id'] = RUN_ID
    if Path(path).name == 'EVALUATION_AXES.json':
        value['source_index_root'] = store.config['reused_evaluation_root017']
    if Path(path).name == 'FIRST_UPDATE_PROOF.json':
        value.update(native_initialized=True, noise005_checkpoint_loads=store.counts['noise005_checkpoint_loads'],
            A_checkpoint_loads=0, M0_checkpoint_loads=0, A_original_forward_rows=0,
            fixed_M0_head_actual_kernel_rows=0, cache_reused_read_only=True,
            restored_seed0_checkpoint_loads=store.counts['restored_seed0_checkpoint_loads'],
            checkpoint_loads=store.counts['checkpoint_loads'], first_new_training_trial=TRIALS[1],
            A_M0_freeze_claim_scope='unchanged precomputed A/B/Q bytes and original cache binding; no fresh A/M0 forward')
    write_new(path, value)



def checkpoint020(engine, name, *, failure=False):
    from run_wp2_014 import atomic_checkpoint
    rt = engine.rt
    state = dict(schema='n01/wp4-020-checkpoint/v1', run_id=RUN_ID, trial=engine.out.name,
        teacher='T1', mode='unpaired', seed=engine.seed, G_updates=engine.update,
        parameters={k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()},
        optimizer=engine.optimizer.state_dict(), student_streams=engine.student_stream.state_dict(),
        teacher_streams=engine.teacher_stream.state_dict(),
        parent_sampler_states={k: dict(generator=v.generator.get_state().clone(), seen=v.seen.copy()) for k, v in engine.samplers.items()},
        initial_stream_states=engine.initial_stream_receipt, cache_manifest_sha256=sha(rt.cache['directory']/'CACHE_MANIFEST.json'),
        frozen_state_sha256=rt.frozen, complete_step_boundary=not failure, last_metrics=engine.last_metrics,
        counters=rt.store.counts.copy(), attempted_counters=rt.store.attempted.copy(), inflight=copy.deepcopy(rt.store.inflight),
        automatic_restart=False)
    if failure:
        require(not getattr(rt.store, 'failure_checkpoint_attempted', False), '020 whole-job failure CP already attempted')
        rt.store.failure_checkpoint_attempted = True
        return rt.call('whole_job_unique_failure_checkpoint', {}, lambda: atomic_checkpoint(engine.out/name, state))
    return rt.call('complete_2theta_checkpoint', dict(checkpoint_saves_complete=1), lambda: atomic_checkpoint(engine.out/name, state))



def restore_seed0_model020(state, p, cache_sha256, frozen, device):
    """Validate the external019 checkpoint; use its two theta and fixed qref only."""
    import torch
    from n01.model017 import CachedFrozenQ1Charge017
    require(state['schema'] == 'n01/wp4-019-checkpoint/v1' and state['run_id'] == 'wp4_019_20261016'
        and state['trial'] == TRIALS[0] and state['teacher'] == 'T1' and state['mode'] == 'unpaired'
        and state['seed'] == 0 and state['G_updates'] == 2000 and state['complete_step_boundary'] is True
        and state['inflight'] == {}, '020 source seed0 is not the complete registered019 endpoint')
    require(state['cache_manifest_sha256'] == cache_sha256 and state['frozen_state_sha256'] == frozen,
        '020 seed0 checkpoint cache or frozen005 identity differs')
    parameters = state['parameters']
    require(set(parameters) == {'theta_a', 'theta_b', 'q_reference'} and
        all(value.dtype == torch.float64 and value.ndim == 0 and bool(torch.isfinite(value))
            for value in parameters.values()) and float(parameters['q_reference']) == p['qref'],
        '020 seed0 checkpoint must contain exactly2 scalar theta and the fixed qref')
    metrics = state['last_metrics']
    require(metrics['G_updates'] == 2000 and metrics['teacher'] == 'T1' and metrics['mode'] == 'unpaired'
        and metrics['seed'] == 0 and metrics['theta_a'] == float(parameters['theta_a'])
        and metrics['theta_b'] == float(parameters['theta_b']), '020 seed0 final metrics/parameters differ')
    coverage = metrics['parent_coverage']
    require(set(coverage) == {'student0', 'student1', 'teacher0', 'teacher1'} and
        all(v['all_covered'] and v['unique'] == 5274 and v['fit_parents'] == 5274 for v in coverage.values()),
        '020 source seed0 did not complete original parent coverage')
    model = CachedFrozenQ1Charge017(device=device)
    model.load_state_dict(parameters, strict=True)
    model.requires_grad_(False)
    require(set(dict(model.named_parameters())) == {'theta_a', 'theta_b'} and
        all(torch.equal(model.state_dict()[k].cpu(), value.cpu()) for k, value in parameters.items()),
        '020 seed0 restored theta bytes differ')
    a, b = model.coefficients()
    restored_metrics = dict(metrics, a_Vns=float(a.detach()), b=float(b.detach()))
    return model, restored_metrics, coverage


def gate_trials020(trials):
    result = gate_trials019(trials)
    result['implementation_schema_origin'] = result['schema']
    result.update(schema='n01/wp4-020-unpaired-gate/v1', run_id=RUN_ID)
    return result



def run020(config, config_sha256, output, device):
    p, paths, checked = gate020(config, config_sha256, output)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(p['CPU_threads'])
    import functools
    import numpy as np
    # The018 import exposes its existing pure numerical helpers and namespace;
    # no private loading occurs merely by importing it.
    from run_wp4_018 import load_kernel018, replay_cells018, wave_energy018
    import run_wp4_017 as legacy
    from n01.stats018 import DISTANCE_COUNTS018, self_mean018, energy_discrepancy018, observable_cell018
    from n01.stats017 import groups_weights017, variance_curve_cell017, charge_identifiability017
    import torch
    from n01.model017 import deterministic_loading017
    from n01.response008 import make_extension, state_digest
    from n01.evaluate_window008 import _memory_limit
    torch.set_num_threads(p['CPU_threads']); torch.use_deterministic_algorithms(True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']), device)
    store = legacy.Store017(output, p, checked)
    legacy.write_new = functools.partial(write_generated020, output=store.root, store=store)
    store.status(); engine = None; active_trial = None
    try:
        store.json('BINDING.json', dict(store.base(), schema='n01/wp4-020-binding/v1',
            configuration_sha256=config_sha256, immutable_files=checked, budget_caps=p['budget_caps'], before_numeric_decode=True))
        cache_manifest = read_json(paths['cache_manifest'])
        prior = read_json(paths['prior_frozen'])['frozen_state_sha256']
        with np.load(paths['parent_split'], allow_pickle=False) as ar:
            cache = dict(fit=ar['fit_rows'], test=ar['test_rows'], ids=ar['source_ids'],
                source_features={k: ar[k] for k in ('q_source', 'source_peak_index', 'source_baseline_mean_V')},
                directory=paths['cache_manifest'].parent, manifest=cache_manifest)
        for name in ('A', 'B', 'Q'):
            cache[name] = store.call('read_only_cached_'+name, dict(cached_array_mmap_files=1),
                lambda name=name: np.load(paths['cache_'+name], mmap_mode='r', allow_pickle=False))
            require(not cache[name].flags.writeable, '020 requires read-only017 cache')
        require(len(cache['Q']) == 6593 and len(cache['fit']) == 5274 and len(cache['test']) == 1319,
                '020 original parent axes differ')
        with np.load(paths['physical_weights'], allow_pickle=False) as ar:
            np.testing.assert_array_equal(ar['source_ids'], cache['ids'])
            np.savez_compressed(store.root/'ORIGINAL_PHYSICAL_WEIGHTS.npz', weights=ar['weights'], source_ids=cache['ids'])
        with np.load(paths['parent_axes018'], allow_pickle=False) as ar:
            for key, value in (('source_ids', cache['ids']), ('fit_rows', cache['fit']), ('test_rows', cache['test']), ('fixed_A_Q_Vns', cache['Q'])):
                np.testing.assert_array_equal(ar[key], value)
        stats = read_json(paths['statistics'])
        noise = store.call('only_frozen005_checkpoint', dict(checkpoint_loads=1, noise005_checkpoint_loads=1),
            lambda: torch.load(paths['noise005'], map_location='cpu', weights_only=False))
        require(noise['schema_version'] == 5 and noise['update'] == 5000 and noise['model_type'] == 'ObservedBaselineNoiseV3'
            and noise['engineering'] is False, '020 original005 checkpoint metadata')
        with deterministic_loading017():
            extension = make_extension(noise, device)
        del noise
        require(state_digest(extension.model005.state_dict()) == prior['noise005'] and
                state_digest(extension.state_dict()) == prior['extension'], '020 original005 frozen state differs')
        rt = legacy.Runtime017(p, cache, extension, stats, prior, store, device)
        rt.verify = functools.partial(verify020, rt)
        store.call('reuse_T1_saved_evaluation', dict(reused_teacher_kernel_sets=1), lambda: None)
        store.call('reuse_shared_initial_scores', dict(reused_initial_score_sets=1), lambda: None)
        store.json('INIT_PROOF.json', dict(store.base(), schema='n01/wp4-020-init-proof/v1', native_initialized=True,
            first_trial=TRIALS[0], first_phase='restore_seed0_then_final_evaluation', new_G_AD_cap=2000,
            noise005_checkpoint_loads=1, restored_seed0_checkpoint_loads=0, A_checkpoint_loads=0, M0_checkpoint_loads=0,
            A_original_forward_rows=0, fixed_M0_head_actual_kernel_rows=0, cache_reused_read_only=True,
            cache_manifest_sha256=sha(paths['cache_manifest']), frozen_state_sha256=prior,
            trainable_parameters=2, trainable_parameter_names=['theta_a', 'theta_b'],
            initial=p['initial'], REAL_DEV_SOURCE_original_waveform_reads=0,
            teacher_initial_saved_evaluation_reused=True, old017_pending_preserved=True))
        restored = store.call('restore_completed019_seed0_checkpoint',
            dict(checkpoint_loads=1, restored_seed0_checkpoint_loads=1),
            lambda: torch.load(paths['seed0_final019'], map_location='cpu', weights_only=False))
        model, restored_metrics, coverage = restore_seed0_model020(restored, p,
            sha(paths['cache_manifest']), prior, rt.device)
        del restored
        seed0_out = store.root/TRIALS[0]
        seed0_out.mkdir()
        seed0_started = time.monotonic(); active_trial = TRIALS[0]
        restore_proof = dict(store.base(), schema='n01/wp4-020-restored-seed0/v1', run_id=RUN_ID,
            native_initialized=True, source_run_id='wp4_019_20261016', source_trial=TRIALS[0],
            source_G_updates=2000, source_checkpoint_sha256=sha(paths['seed0_final019']),
            source_checkpoint_bytes=paths['seed0_final019'].stat().st_size,
            restored_parameters=['theta_a', 'theta_b'], fixed_qref=p['qref'],
            restored_theta_values={name: float(value.detach()) for name, value in model.named_parameters()},
            restored_a_Vns=restored_metrics['a_Vns'], restored_b=restored_metrics['b'],
            seed0_new_G_AD=0, seed0_optimizer_created=False, seed0_parent_samplers_created=0,
            source_training_optimizer_and_RNG_states_used=False, gradients_enabled=False,
            checkpoint_loads=2, noise005_checkpoint_loads=1, restored_seed0_checkpoint_loads=1,
            cache_manifest_sha256=sha(paths['cache_manifest']), frozen_state_sha256=prior,
            evaluation_namespace=p['stream_addresses']['evaluation_student'],
            repeated_failed_prefix=p['repeated_failed_evaluation_prefix'],
            original019_native1_preserved=True)
        write_new(store.root/'RESTORED_SEED0_PROOF.json', restore_proof)
        store.status(dict(current_trial=TRIALS[0], current_trial_new_G_updates=0,
            phase='restored_seed0_final_evaluation', restored_seed0_source_G_updates=2000))
        legacy.evaluate_kernel017(rt, seed0_out/'evaluation', model=model)
        rt.verify(files=True)
        training_records = [dict(trial=TRIALS[0], source019_G_updates=2000, new_G_updates=0,
            recovered_completed_training=True, coverage=coverage, resources=rt.guard(seed0_out, seed0_started),
            last_metrics=restored_metrics, final_checkpoint_source=str(paths['seed0_final019']),
            final_checkpoint_sha256=sha(paths['seed0_final019']))]
        write_new(seed0_out/'RESTORED_EVALUATION_COMPLETE020.json', dict(schema='n01/wp4-020-restored-evaluation/v1',
            run_id=RUN_ID, **training_records[0], evaluation_gate_pending=True))
        del model
        seed, name = 1, TRIALS[1]
        active_trial = name
        engine = legacy.Engine017(rt, 'T1', 'unpaired', seed, store.root/name)
        while engine.update < p['G_updates']:
            engine.step()
            if engine.update in p['training']['checkpoints']:
                rt.verify(files=True)
                checkpoint020(engine, 'final.pt' if engine.update == 2000 else 'step_1000.pt')
        require(all(v['all_covered'] and v['unique'] == 5274 for v in engine.coverage().values()),
                '020 incomplete original fit-parent sampler coverage; no repair draws')
        engine.save_stream_states('end')
        legacy.evaluate_kernel017(rt, engine.out/'evaluation', model=engine.model)
        rt.verify(files=True)
        training_records.append(dict(trial=name, source019_G_updates=0, new_G_updates=2000,
            recovered_completed_training=False, coverage=engine.coverage(),
            resources=rt.guard(engine.out, engine.started), last_metrics=engine.last_metrics,
            final_checkpoint_sha256=sha(engine.out/'final.pt')))
        write_new(engine.out/'TRAINING_COMPLETE020.json', dict(schema='n01/wp4-020-training-complete/v1',
            run_id=RUN_ID, **training_records[-1], evaluation_gate_pending=True))
        engine = None; active_trial = None
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        # Both original final kernels finish before the common CPU scoring phase.
        DISTANCE_COUNTS018.clear()
        test, Q, ids = cache['test'], cache['Q'], cache['ids']
        groups, weights, cuts = groups_weights017(Q[cache['fit']], Q[test], ids[test])
        trials = []
        for record in training_records:
            a, b = record['last_metrics']['a_Vns'], record['last_metrics']['b']
            analytic = [dict(variance_curve_cell017(Q[test], a, b, g, weights[w]),
                metric='analytic_charge_variance', group=group, weights=w) for group, g in groups.items() for w in weights]
            trials.append(dict(schema='n01/wp4-020-comparison/v1', run_id=RUN_ID, trial=record['trial'], cells=analytic,
                selection_diagnostics=[], other_retained_flag_diagnostics=[],
                parameter_recovery_secondary=charge_identifiability017(Q[cache['fit']], Q[test], a, b),
                parent_ids=ids, test_rows=test, groups=groups, comparison_weights=weights, Q_group_cuts=cuts,
                original_physical_weights_file='../ORIGINAL_PHYSICAL_WEIGHTS.npz', no_REAL_hardware_acceptance=True))
        with np.load(paths['teacher_PIT'], allow_pickle=False) as ar:
            uniforms = ar['uniforms']
        for hi, h in enumerate(HYPOTHESES):
            T, null, reference, initial = teacher_data020(paths, h)
            zt = T['embedding'].reshape(6593, 64, 19)
            teacher_self = store.call('teacher018_O19_self', dict(evaluation_observable_pair_distances=6593*64**2), lambda: self_mean018(zt))
            for si, name in enumerate(TRIALS):
                score_started = time.monotonic()
                S = load_kernel018(Path(p['remote_root']), name+'/evaluation', h, ids, p, store.root, store.started, name)
                final = store.call('saved020_full257_energy', dict(evaluation_energy_pair_distances=4*6593*32**2),
                    lambda: wave_energy018(S['waves'], T['waves']))
                discrepancy = store.call('saved020_O19_energy', dict(evaluation_observable_pair_distances=2*6593*64**2),
                    lambda: energy_discrepancy018(S['embedding'].reshape(6593, 64, 19), zt, teacher_self=teacher_self))
                cells, extra, cov8, coverage = replay_cells018(S, T, uniforms[:, :, :, :, hi], test, groups, weights,
                    dict(final=final, initial=initial, reference=reference), h)
                selected = [c for c in cells if c['metric'] in ('conditional_cov9', 'time_rise10_90') or
                            (c['metric'] == 'PIT_coverage' and c['feature'] == 'rise10_90')]
                trials[si]['selection_diagnostics'].extend(selected)
                trials[si]['cells'].extend([c for c in cells if c['metric'] not in ('conditional_cov9', 'time_rise10_90') and
                    not (c['metric'] == 'PIT_coverage' and c['feature'] == 'rise10_90')])
                trials[si]['cells'].extend(cov8)
                trials[si]['other_retained_flag_diagnostics'].extend(extra)
                trials[si]['cells'].extend([dict(observable_cell018(discrepancy[test], null[test], g, weights[w]),
                    metric='observable_joint_energy', group=group, weights=w, hypothesis=h) for group, g in groups.items() for w in weights])
                np.savez_compressed(store.root/name/f'PARENT_COMPARISON020_{h}.npz', energy_final=final,
                    reused_energy_initial=initial, reused_energy_teacher_reference=reference, observable_discrepancy=discrepancy,
                    reused_teacher32_null=null, **{'coverage_'+k: v for k, v in coverage.items() if isinstance(v, np.ndarray)})
                training_records[si]['resources']['elapsed_seconds'] += time.monotonic()-score_started
                require(training_records[si]['resources']['elapsed_seconds'] <= 7200, '020 per-trial active wall cap')
                del S
            del T, zt
            store.status(dict(phase='scored_'+h))
        require(DISTANCE_COUNTS018 == {257: 162029568, 19: 405073920}, '020 exact018 evaluation distance budget')
        for trial in trials:
            store.call('one020_final_teacher_score_comparison', dict(evaluation_energy_candidate_comparisons=1), lambda: None)
            legacy.compact_report017(store.root/trial['trial']/'COMPARISON020.json', trial)
        verdict = gate_trials020(trials)
        legacy.compact_report017(store.root/'UNPAIRED_GATE020.json', verdict)
        for record, comparison in zip(training_records, trials):
            write_new(store.root/record['trial']/'COMPLETE.json', dict(schema='n01/wp4-020-trial-complete/v1', run_id=RUN_ID,
                native_status='COMPLETE', **record, all_required_cells_pass=all(c['status'] == 'pass' for c in comparison['cells']),
                required_cells=1272, scientific_acceptance=False))
            require(sum(q.stat().st_size for q in (store.root/record['trial']).rglob('*') if q.is_file()) <= 1536*1024**2,
                    '020 final per-trial output cap')
        rt.verify(files=True)
        require(not store.inflight and store.counts == store.attempted == p['budget_caps'], '020 exact terminal additive counters differ')
        files = {q.relative_to(store.root).as_posix(): dict(sha256=sha(q), bytes=q.stat().st_size)
                 for q in sorted(store.root.rglob('*')) if q.is_file() and q.name != 'status.json'}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp4-020-artifact-inventory/v1', run_id=RUN_ID,
            files=files, raw_arrays_retained_remote=True, selected_package_is_not_full_raw_evidence=True))
        result = dict(store.base(), schema='n01/wp4-020-complete/v1', native_status='COMPLETE', trials=training_records,
            terminal_scope='T1_unpaired_two_seeds_only', gate=verdict, old017_pending_preserved=True,
            original_A_M0_NN_forward_rows=0, noise005_rows=store.counts['noise005_rows'],
            new_SOURCE_REAL_DEV_reads=0, further_trials_started=False, scientific_acceptance=False,
            restored_seed0_source019_G_updates=2000, seed0_new_G_AD=0, seed1_new_G_AD=2000,
            original019_native1_preserved=True, autonomous_engineering_recovery_authorized=True,
            native_internal_retry_performed=False)
        store.json('COMPLETE.json', result); store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()) <= 5120*1024**2, '020 total output cap')
        shared_bytes = sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()
                           and q.relative_to(store.root).parts[0] not in TRIALS)
        require(shared_bytes <= 1024*1024**2, '020 shared scoring/log output cap')
        return result
    except BaseException as error:
        failure = dict(store.base(), schema='n01/wp4-020-failure/v1', native_status='FAILED',
            error_type=type(error).__name__, error=str(error), automatic_restart=False,
            current_trial=active_trial,
            current_trial_new_G_updates=0 if active_trial == TRIALS[0] else None if engine is None else engine.update,
            source019_seed0_G_updates=2000, original019_native1_preserved=True)
        store.json('FAILED.json', failure); store.status(failure)
        if engine is not None:
            try:
                checkpoint020(engine, 'failure.pt', failure=True)
            except BaseException as saving_error:
                store.json('FAILURE_CHECKPOINT_ERROR.json', dict(schema='n01/wp4-020-failure-checkpoint-error/v1',
                    run_id=RUN_ID, error=repr(saving_error), original_error=repr(error)))
        raise



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned020 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = run020(args.config, args.config_sha256, args.output, args.device)
    print(json.dumps(dict(native_status=result['native_status'], G_updates=result['counters']['G_updates'],
        unpaired_gate=result['gate']['status']), allow_nan=False), flush=True)



if __name__ == '__main__':
    main()
