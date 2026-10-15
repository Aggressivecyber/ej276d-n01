"""Two bounded T1 unpaired trials, with frozen017 algorithms and frozen018 gates."""
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
from run_wp4_017 import settings017, budget_caps017, seed017

RUN_ID = 'wp4_019_20261016'
REMOTE = '/home/rylan/projects/ej276d-n01/wp4_019_20261016'
TRIALS = ('T1_unpaired_seed0', 'T1_unpaired_seed1')
HYPOTHESES = ('fast', 'slow', 'common_level')
SCIENCE_SOURCES019 = tuple('scripts/'+name+'.py' for name in (
    'run_wp4_019', 'run_wp4_018', 'run_wp4_017', 'run_wp2_014', 'run_wp1_013', 'wp1_ops')) + tuple(
    'src/n01/'+name+'.py' for name in ('ablation009', 'data', 'data008', 'diagnose_mean_random009',
    'electronics_v3', 'evaluate', 'evaluate_window008', 'extension006', 'losses', 'losses008', 'losses008_v2',
    'model', 'model008', 'model008_fix01', 'model010', 'model014', 'model016', 'model017', 'observation008',
    'response008', 'saved_array_kernels017', 'stats017', 'stats018', 'train'))


def stream_addresses019():
    training, parents = {}, {}
    for seed in (0, 1):
        training[str(seed)] = {role: dict(namespace=dict(kind='training', replicate=seed, role=role),
            initial_seeds={stream: seed017(dict(kind='training', replicate=seed, role=role, stream=stream))
                           for stream in ('shape', 'charge', 'noise005', 'extension284', 'sign', 'PIT')})
            for role in ('student', 'teacher')}
        parents[str(seed)] = {role: dict(namespace=dict(kind='parent', mode='unpaired', replicate=seed, role=role),
            initial_seed=seed017(dict(kind='parent', mode='unpaired', replicate=seed, role=role)))
            for role in ('student0', 'student1', 'teacher0', 'teacher1')}
    return dict(training=training, parents=parents,
        evaluation_student={str(bank): {stream: seed017(dict(kind='evaluation', role='student', bank=bank, stream=stream))
            for stream in ('shape', 'charge', 'noise005', 'extension284', 'sign', 'PIT')} for bank in (0, 1)},
        Gaussian_namespace_unchanged_from017=True, parent_namespace_original_unpaired=True,
        teacher_evaluation_and_PIT_reused_without_new_draws=True)


def settings019():
    p = settings017()
    p.update(schema='n01/wp4-019-config/v1', run_id=RUN_ID, remote_root=REMOTE,
        status='draft', trial_order=list(TRIALS), candidates=['T1_unpaired'], seed_sets=[0, 1],
        teachers={'T1': dict(a_fraction_qref=.01, b=.0025)},
        shared_output_MiB=1024, selected_collection_MiB=128,
        resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=21600,
            output_MiB=5120, poll_seconds=10, TERM_grace_seconds=20),
        cache_read_only=True, new_A_or_M0_forward_rows=0,
        teacher_and_initial_evaluation_reused=True, final_evaluation_kernels=2,
        reused_evaluation_root017='/home/rylan/projects/ej276d-n01/wp4_017_20261015/runs/primary/artifacts',
        reused_evaluation_root018='/home/rylan/projects/ej276d-n01/wp4_018_20261016/runs/primary/artifacts',
        required_cells_per_trial=1272, required_cells_total=2544,
        new_observable_energy_tolerance=.10, new_observable_MC_half_width_limit=.05,
        time_scale_software_ns=1., representation_dimension=19,
        original017_pending_preserved=True, automatic_stage2_reopen=False,
        T2_enabled=False, REAL_numeric_read=False, DEV_numeric_read=False,
        stream_addresses=stream_addresses019(),
        scales=dict(fast=dict(Q_Vns=.14006821393078955, peak_voltage_V=.005499862543299458, FWHM_ns=.5129398878764326),
            slow=dict(Q_Vns=.1400669323527055, peak_voltage_V=.005499869994334829, FWHM_ns=.5149414509583063),
            common_level=dict(Q_Vns=.14006769980934886, peak_voltage_V=.00550009352539599, FWHM_ns=.5147888093676196)),
        resource_caps=dict(wall_seconds=21600, output_MiB=5120))
    for name in ('stage1_output_MiB', 'stage1_wall_seconds', 'proposal_sha256'):
        del p[name]
    return p


def budget_caps019(p):
    n, k, g, trials = 6593, 512, 2000, 2
    evaluation = trials*n*2*32
    teacher = trials*g*2*k
    student = teacher+evaluation
    total = student+teacher
    caps = {name: 0 for name in budget_caps017(settings017())}
    caps.update(trial_slots=trials, G_updates=trials*g, optimizer_steps=trials*g,
        training_backward_queries=trials*g, new_AD_queries=trials*g, optimizers_created=trials,
        checkpoint_loads=1, student_charge_module_rows=student, teacher_charge_module_rows=teacher,
        shape_cached_basis_assembly_rows=total, noise005_rows=total,
        all_three_extensions_rows=3*total, hard_ADC_rows=3*total, hard_first_argmax_gather_rows=3*total,
        training_student_rows=teacher, training_teacher_rows=teacher,
        evaluation_kernel_snapshots=trials, evaluation_signal_rows=evaluation,
        normal_shape_scalar_requests=8*total, normal_charge_scalar_requests=total,
        normal_noise005_scalar_requests=101*total, normal_extension284_scalar_requests=284*total,
        total_scalar_normal_requests=394*total,
        parent_sampler_streams=4*trials, parent_randint_calls=4*trials*g,
        parent_randint_scalar_requests=4*trials*g*k,
        unpaired_training_energy_pair_distances=trials*g*(2*k)**2*2*3,
        unpaired_training_cross_retained_pairs=trials*g*(2*k)**2*3,
        unpaired_training_self_ordered_retained_pairs=trials*g*(2*k)*(2*k-1)*3,
        unpaired_training_self_diagonal_computed_then_masked=trials*g*(2*k)*3,
        evaluation_energy_candidate_comparisons=trials,
        evaluation_energy_pair_distances=trials*4*n*3*32**2,
        checkpoint_saves_complete=4, noise005_checkpoint_loads=1,
        cached_array_mmap_files=3, reused_teacher_kernel_sets=1, reused_initial_score_sets=1,
        evaluation_observable_pair_distances=(1+2*trials)*n*3*64**2)
    return caps


def validate_settings019(p, *, frozen=True, driver_source=None):
    reference = settings019()
    require(not frozen or p['status'] == 'frozen', '019 draft cannot execute')
    require(all(p.get(k) == value for k, value in reference.items() if k != 'status'),
            '019 changed the prospectively fixed settings')
    require(p['budget_caps'] == budget_caps019(p), '019 exact additive budget differs')


def gate019(config, expected_sha, output):
    cp = Path(config).resolve()
    require(sha(cp) == expected_sha, '019 frozen config SHA differs')
    p = read_json(cp)
    validate_settings019(p)
    require(set(SCIENCE_SOURCES019) <= set(p['source_files']), '019 mandatory science source closure is incomplete')
    require(all(Path(p['source_files'][name]['path']) == Path(p['remote_root'])/name for name in SCIENCE_SOURCES019),
            '019 mandatory imported source must be the bound owned copy')
    require(Path(output).absolute() == Path(p['remote_root'])/'runs/primary/artifacts'
            and not Path(output).exists(), '019 exact fresh owned output required')
    manifest_path = cp.parent/p['saved_input_manifest']
    require(sha(manifest_path) == p['saved_input_manifest_sha256'], '019 saved input manifest differs')
    manifest = read_json(manifest_path)
    require(manifest['file_count'] == 37 and p['inputs'] == manifest['entries'], '019 exact saved input whitelist differs')
    require(p['contracts']['prerequisite_018'] == manifest['entries']['prerequisite018'], '019 prerequisite binding differs')
    records = {**p['source_files'], **{'contract:'+k: v for k, v in p['contracts'].items()},
               **{'protection:'+k: v for k, v in p['protection_files'].items()},
               **{'input:'+k: v for k, v in manifest['entries'].items()}}
    checked = {str(cp): expected_sha, str(manifest_path): sha(manifest_path)}
    roots = [Path(x).resolve() for x in p['allowed_input_roots']]
    for label, record in records.items():
        path = Path(record['path'])
        require(not path.is_symlink() and any(path.resolve().is_relative_to(r) for r in roots),
                '019 input/source escaped frozen roots: '+label)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], '019 frozen bytes differ: '+label)
        checked[str(path)] = record['sha256']
    original018 = read_json(manifest['entries']['config018']['path'])
    original_gate = read_json(manifest['entries']['gate018']['path'])
    require(original018['scales'] == p['scales'] and original018['time_scale_software_ns'] == 1.
        and original018['new_observable_energy_tolerance'] == .10
        and original018['new_observable_MC_half_width_limit'] == .05
        and original_gate['status'] == 'pass' and original_gate['required_cell_count'] == 2544,
        '019 requires the same accepted018 numerical contract and complete gate')
    prerequisite = read_json(p['contracts']['prerequisite_018']['path'])
    require(prerequisite['schema'] == 'n01/wp4-018-final-result/v1' and
        prerequisite['run_id'] == 'wp4_018_20261016' and prerequisite['native_exit_code'] == 0 and
        prerequisite['qualified'] is True and prerequisite['required_cells'] == 2544 and
        prerequisite['independent_review_pass'] is True and prerequisite['old017_pending_preserved'] is True,
        '019 requires completed independently accepted018; producer pass alone is insufficient')
    return p, {k: Path(v['path']) for k, v in manifest['entries'].items()}, checked


def write_generated019(path, value, *, output, store):
    """Version only newly generated metadata; numerical algorithms/namespaces stay017."""
    require(Path(path).resolve().is_relative_to(output.resolve()), '019 metadata write escaped new output')
    schema = value['schema'] if 'schema' in value else None
    if schema is not None and 'wp4-017' in schema:
        value['implementation_schema_origin'] = schema
        value['schema'] = schema.replace('wp4-017', 'wp4-019')
    value['run_id'] = RUN_ID
    if Path(path).name == 'EVALUATION_AXES.json':
        value['source_index_root'] = store.config['reused_evaluation_root017']
    if Path(path).name == 'FIRST_UPDATE_PROOF.json':
        value.update(native_initialized=True, noise005_checkpoint_loads=store.counts['noise005_checkpoint_loads'],
            A_checkpoint_loads=0, M0_checkpoint_loads=0, A_original_forward_rows=0,
            fixed_M0_head_actual_kernel_rows=0, cache_reused_read_only=True,
            A_M0_freeze_claim_scope='unchanged precomputed A/B/Q bytes and original cache binding; no fresh A/M0 forward')
    write_new(path, value)


def verify019(rt, *, files=False):
    from n01.response008 import state_digest
    rt.extension.assert_frozen()
    require(state_digest(rt.extension.model005.state_dict()) == rt.frozen['noise005'] and
            state_digest(rt.extension.state_dict()) == rt.frozen['extension'], '019 frozen005 or extension changed')
    if files:
        unchanged(rt.store.checked)


def checkpoint019(engine, name, *, failure=False):
    from run_wp2_014 import atomic_checkpoint
    rt = engine.rt
    state = dict(schema='n01/wp4-019-checkpoint/v1', run_id=RUN_ID, trial=engine.out.name,
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
        require(not getattr(rt.store, 'failure_checkpoint_attempted', False), '019 whole-job failure CP already attempted')
        rt.store.failure_checkpoint_attempted = True
        return rt.call('whole_job_unique_failure_checkpoint', {}, lambda: atomic_checkpoint(engine.out/name, state))
    return rt.call('complete_2theta_checkpoint', dict(checkpoint_saves_complete=1), lambda: atomic_checkpoint(engine.out/name, state))


def teacher_data019(paths, h):
    """Use the exact018 observable tuple; saved hardY is decoded only for fullwave ES."""
    import numpy as np
    from run_wp4_017 import decode_hard017
    from n01.stats017 import scalar_banks017, pooled_wave017
    waves, scalar, means, variances = [], [], [], []
    for bank in (0, 1):
        waves.append(decode_hard017(paths[f'teacher_{h}_hard{bank}'])[0].reshape(6593, 32, 256))
        with np.load(paths[f'teacher_{h}_report{bank}'], allow_pickle=False) as ar:
            scalar.append(ar['scalar9']); means.append(ar['wave_mean']); variances.append(ar['wave_variance'])
    scalar = np.stack(scalar, axis=1)
    moments = scalar_banks017(scalar)
    mean, variance = pooled_wave017(means[0], variances[0], means[1], variances[1])
    with np.load(paths[f'observable018_{h}'], allow_pickle=False) as ar:
        embedding, state = ar['embedding'], ar['state']
        flags = {k[len('original_full64_'):]: ar[k] for k in ar.files if k.startswith('original_full64_flag__')}
        np.testing.assert_array_equal(ar['original_full64_undefined_fraction'], moments['full_slot_undefined_fraction'])
        np.testing.assert_array_equal(state == 0, np.isfinite(scalar[:, :, :, 6]))
    with np.load(paths[f'control018_{h}'], allow_pickle=False) as ar:
        null, reference = ar['teacher_null'], ar['fullwave_teacher_reference']
    with np.load(paths[f'initial018_{h}'], allow_pickle=False) as ar:
        initial = ar['fullwave_energy']
    return dict(waves=np.stack(waves, axis=1), scalar=scalar, moments=moments,
        wave_mean=mean, wave_variance=variance, flags=flags, embedding=embedding), null, reference, initial


def gate_trials019(trials):
    """The exact018 grid, with honest019 unpaired trial identities."""
    from n01.stats017 import REQUIRED_MISSING017
    require([t['trial'] for t in trials] == list(TRIALS), '019 exact two unpaired trials')
    expected = set()
    for group in ('all_heldout', 'fit_Q_tertile0_heldout', 'fit_Q_tertile1_heldout', 'fit_Q_tertile2_heldout'):
        for weight in ('nominal', 'low', 'high'):
            expected.add(('analytic_charge_variance', group, weight, None, None, None))
            for h in HYPOTHESES:
                for metric in ('hard_wave_mean', 'hard_wave_variance', 'conditional_cov8', 'conditional_energy',
                               'time_peak_index', 'time_FWHM', 'observable_joint_energy'):
                    expected.add((metric, group, weight, h, None, None))
                for feature in ('Q', 'tail', 'peak_voltage', 'peak_index', 'FWHM'):
                    for level in (.5, .9):
                        expected.add(('PIT_coverage', group, weight, h, feature, level))
                for feature in REQUIRED_MISSING017:
                    expected.add(('missingness_or_censor', group, weight, h, feature, None))
    require(len(expected) == 1272, '019 fixed018 grid size')
    bad = []
    for trial in trials:
        signatures = [(c['metric'], c['group'], c['weights'], c.get('hypothesis'), c.get('feature'), c.get('level')) for c in trial['cells']]
        require(len(signatures) == 1272 and set(signatures) == expected, '019 missing/duplicate018 primary signature')
        require(all(c['status'] in ('pass', 'pending', 'fail') for c in trial['cells']), '019 primary status')
        bad += [dict(trial=trial['trial'], cell=i, metric=c['metric'], hypothesis=c.get('hypothesis'),
            group=c['group'], weights=c['weights'], status=c['status'], reasons=c['reasons'])
            for i, c in enumerate(trial['cells']) if c['status'] != 'pass']
    return dict(schema='n01/wp4-019-unpaired-gate/v1', run_id=RUN_ID, status='pass' if not bad else 'closed',
        T1_unpaired_qualified=not bad, required_cell_count=2544, nonpass=bad,
        pending_is_pass=False, further_trials_authorized=False, scientific_acceptance=False,
        scope='fixed A/B/005 conditional charge identification under the registered unpaired method and budget')


def run019(config, config_sha256, output, device):
    p, paths, checked = gate019(config, config_sha256, output)
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
    legacy.write_new = functools.partial(write_generated019, output=store.root, store=store)
    store.status(); engine = None
    try:
        store.json('BINDING.json', dict(store.base(), schema='n01/wp4-019-binding/v1',
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
            require(not cache[name].flags.writeable, '019 requires read-only017 cache')
        require(len(cache['Q']) == 6593 and len(cache['fit']) == 5274 and len(cache['test']) == 1319,
                '019 original parent axes differ')
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
            and noise['engineering'] is False, '019 original005 checkpoint metadata')
        with deterministic_loading017():
            extension = make_extension(noise, device)
        del noise
        require(state_digest(extension.model005.state_dict()) == prior['noise005'] and
                state_digest(extension.state_dict()) == prior['extension'], '019 original005 frozen state differs')
        rt = legacy.Runtime017(p, cache, extension, stats, prior, store, device)
        rt.verify = functools.partial(verify019, rt)
        store.call('reuse_T1_saved_evaluation', dict(reused_teacher_kernel_sets=1), lambda: None)
        store.call('reuse_shared_initial_scores', dict(reused_initial_score_sets=1), lambda: None)
        store.json('INIT_PROOF.json', dict(store.base(), schema='n01/wp4-019-init-proof/v1', native_initialized=True,
            first_trial=TRIALS[0], noise005_checkpoint_loads=1, A_checkpoint_loads=0, M0_checkpoint_loads=0,
            A_original_forward_rows=0, fixed_M0_head_actual_kernel_rows=0, cache_reused_read_only=True,
            cache_manifest_sha256=sha(paths['cache_manifest']), frozen_state_sha256=prior,
            trainable_parameters=2, trainable_parameter_names=['theta_a', 'theta_b'],
            initial=p['initial'], REAL_DEV_SOURCE_original_waveform_reads=0,
            teacher_initial_saved_evaluation_reused=True, old017_pending_preserved=True))
        training_records = []
        for seed, name in enumerate(TRIALS):
            engine = legacy.Engine017(rt, 'T1', 'unpaired', seed, store.root/name)
            while engine.update < p['G_updates']:
                engine.step()
                if engine.update in p['training']['checkpoints']:
                    rt.verify(files=True)
                    checkpoint019(engine, 'final.pt' if engine.update == 2000 else 'step_1000.pt')
            require(all(v['all_covered'] and v['unique'] == 5274 for v in engine.coverage().values()),
                    '019 incomplete original fit-parent sampler coverage; no repair draws')
            engine.save_stream_states('end')
            legacy.evaluate_kernel017(rt, engine.out/'evaluation', model=engine.model)
            rt.verify(files=True)
            training_records.append(dict(trial=name, G_updates=2000, coverage=engine.coverage(),
                resources=rt.guard(engine.out, engine.started), last_metrics=engine.last_metrics,
                final_checkpoint_sha256=sha(engine.out/'final.pt')))
            write_new(engine.out/'TRAINING_COMPLETE019.json', dict(schema='n01/wp4-019-training-complete/v1',
                run_id=RUN_ID, **training_records[-1], evaluation_gate_pending=True))
            engine = None
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        # Both fixed training slots finish before the common CPU scoring phase.
        DISTANCE_COUNTS018.clear()
        test, Q, ids = cache['test'], cache['Q'], cache['ids']
        groups, weights, cuts = groups_weights017(Q[cache['fit']], Q[test], ids[test])
        trials = []
        for record in training_records:
            a, b = record['last_metrics']['a_Vns'], record['last_metrics']['b']
            analytic = [dict(variance_curve_cell017(Q[test], a, b, g, weights[w]),
                metric='analytic_charge_variance', group=group, weights=w) for group, g in groups.items() for w in weights]
            trials.append(dict(schema='n01/wp4-019-comparison/v1', run_id=RUN_ID, trial=record['trial'], cells=analytic,
                selection_diagnostics=[], other_retained_flag_diagnostics=[],
                parameter_recovery_secondary=charge_identifiability017(Q[cache['fit']], Q[test], a, b),
                parent_ids=ids, test_rows=test, groups=groups, comparison_weights=weights, Q_group_cuts=cuts,
                original_physical_weights_file='../ORIGINAL_PHYSICAL_WEIGHTS.npz', no_REAL_hardware_acceptance=True))
        with np.load(paths['teacher_PIT'], allow_pickle=False) as ar:
            uniforms = ar['uniforms']
        for hi, h in enumerate(HYPOTHESES):
            T, null, reference, initial = teacher_data019(paths, h)
            zt = T['embedding'].reshape(6593, 64, 19)
            teacher_self = store.call('teacher018_O19_self', dict(evaluation_observable_pair_distances=6593*64**2), lambda: self_mean018(zt))
            for si, name in enumerate(TRIALS):
                score_started = time.monotonic()
                S = load_kernel018(Path(p['remote_root']), name+'/evaluation', h, ids, p, store.root, store.started, name)
                final = store.call('saved019_full257_energy', dict(evaluation_energy_pair_distances=4*6593*32**2),
                    lambda: wave_energy018(S['waves'], T['waves']))
                discrepancy = store.call('saved019_O19_energy', dict(evaluation_observable_pair_distances=2*6593*64**2),
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
                np.savez_compressed(store.root/name/f'PARENT_COMPARISON019_{h}.npz', energy_final=final,
                    reused_energy_initial=initial, reused_energy_teacher_reference=reference, observable_discrepancy=discrepancy,
                    reused_teacher32_null=null, **{'coverage_'+k: v for k, v in coverage.items() if isinstance(v, np.ndarray)})
                training_records[si]['resources']['elapsed_seconds'] += time.monotonic()-score_started
                require(training_records[si]['resources']['elapsed_seconds'] <= 7200, '019 per-trial active wall cap')
                del S
            del T, zt
            store.status(dict(phase='scored_'+h))
        require(DISTANCE_COUNTS018 == {257: 162029568, 19: 405073920}, '019 exact018 evaluation distance budget')
        for trial in trials:
            store.call('one019_final_teacher_score_comparison', dict(evaluation_energy_candidate_comparisons=1), lambda: None)
            legacy.compact_report017(store.root/trial['trial']/'COMPARISON019.json', trial)
        verdict = gate_trials019(trials)
        legacy.compact_report017(store.root/'UNPAIRED_GATE019.json', verdict)
        for record, comparison in zip(training_records, trials):
            write_new(store.root/record['trial']/'COMPLETE.json', dict(schema='n01/wp4-019-trial-complete/v1', run_id=RUN_ID,
                native_status='COMPLETE', **record, all_required_cells_pass=all(c['status'] == 'pass' for c in comparison['cells']),
                required_cells=1272, scientific_acceptance=False))
            require(sum(q.stat().st_size for q in (store.root/record['trial']).rglob('*') if q.is_file()) <= 1536*1024**2,
                    '019 final per-trial output cap')
        rt.verify(files=True)
        require(not store.inflight and store.counts == store.attempted == p['budget_caps'], '019 exact terminal additive counters differ')
        files = {q.relative_to(store.root).as_posix(): dict(sha256=sha(q), bytes=q.stat().st_size)
                 for q in sorted(store.root.rglob('*')) if q.is_file() and q.name != 'status.json'}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp4-019-artifact-inventory/v1', run_id=RUN_ID,
            files=files, raw_arrays_retained_remote=True, selected_package_is_not_full_raw_evidence=True))
        result = dict(store.base(), schema='n01/wp4-019-complete/v1', native_status='COMPLETE', trials=training_records,
            terminal_scope='T1_unpaired_two_seeds_only', gate=verdict, old017_pending_preserved=True,
            original_A_M0_NN_forward_rows=0, noise005_rows=store.counts['noise005_rows'],
            new_SOURCE_REAL_DEV_reads=0, further_trials_started=False, scientific_acceptance=False)
        store.json('COMPLETE.json', result); store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()) <= 5120*1024**2, '019 total output cap')
        shared_bytes = sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file()
                           and q.relative_to(store.root).parts[0] not in TRIALS)
        require(shared_bytes <= 1024*1024**2, '019 shared scoring/log output cap')
        return result
    except BaseException as error:
        failure = dict(store.base(), schema='n01/wp4-019-failure/v1', native_status='FAILED',
            error_type=type(error).__name__, error=str(error), automatic_restart=False,
            current_trial=None if engine is None else engine.out.name,
            current_trial_G_updates=None if engine is None else engine.update)
        store.json('FAILED.json', failure); store.status(failure)
        if engine is not None:
            try:
                checkpoint019(engine, 'failure.pt', failure=True)
            except BaseException as saving_error:
                store.json('FAILURE_CHECKPOINT_ERROR.json', dict(schema='n01/wp4-019-failure-checkpoint-error/v1',
                    run_id=RUN_ID, error=repr(saving_error), original_error=repr(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    def terminate(signum, frame):
        raise InterruptedError('Owned019 received signal '+str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = run019(args.config, args.config_sha256, args.output, args.device)
    print(json.dumps(dict(native_status=result['native_status'], G_updates=result['counters']['G_updates'],
        unpaired_gate=result['gate']['status']), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
