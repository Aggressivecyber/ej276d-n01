"""Fixed M0 SOURCE-composition stress on saved observables; no model or distances."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

from wp1_ops import Store, read_json, require, sha, unchanged

RUN_ID = 'wp5_026_20261016'
HYPOTHESES = ('fast', 'slow', 'common_level')
SCENARIOS = ('nominal', 'low_h', 'high_h')
SCALARS = ('Q_Vns', 'front_Vns', 'tail_Vns', 'PSD_raw', 'PSD_guarded',
           'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns')
FLAGS = ('Q_nonpositive', 'PSD_undefined', 'PSD_guarded', 'rise_undefined',
         'FWHM_undefined', 'temporal_nonpositive_peak', 'rise_prepeak_censored',
         'FWHM_prepeak_censored', 'FWHM_postpeak_censored')
PRIMARY = ('Q_Vns', 'front_Vns', 'tail_Vns', 'peak_voltage_V')
SCIENCE_SOURCES026 = ('scripts/run_wp5_026.py', 'scripts/wp1_ops.py')


def settings026():
    return dict(schema='n01/wp5-026-config/v1', run_id=RUN_ID, status='draft',
        parents=6593, source_information_rows=8192, draws=4, banks=[0, 1],
        hypotheses=list(HYPOTHESES), scenarios=list(SCENARIOS), scalar_names=list(SCALARS),
        flag_names=[name+'_rate' for name in FLAGS], decomposition_scalars=list(PRIMARY),
        h_tilt_amplitude=.20, raw_weight_bounds=[.8, 1.2],
        h_definition='0.15*edep_hadron/(edep_ee+0.15*edep_hadron)',
        within_empirical_denominator=4, identity_relative_tolerance=1e-12,
        physical_weight_calibration='unknown', comparison_weights_multiply_physical_weights=False,
        REAL_rows=21047, REAL_original_logical_groups=422, bank_pooling=False,
        fit_to_REAL=False, extra_scenarios=False, automatic_restart=False,
        training_enabled=False, scientific_training_updates=0,
        output_directory='E:/GAN/research/N01/runs/wp5_026_20261016/analysis_01',
        resource_limits=dict(execution='local Windows E/D only, reuse021 local bounded native pattern',
            CPU_affinity_count=1, BLAS_OMP_threads=1, RSS_MiB=1024, wall_seconds=300, output_MiB=64),
        scientific_status='response_match_not_established')


def budget_caps026(p):
    n, r = p['parents'], p['draws']
    libraries = len(p['banks'])*len(p['hypotheses'])
    cells = libraries*len(p['scenarios'])
    return dict(new_diagnostic_jobs=1, numeric_NPZ_files_decoded=9,
        saved_SOURCE_information_rows=8192, aligned_SOURCE_TRAIN_parents=n,
        saved_fake_observable_files=libraries, saved_fake_observable_rows=libraries*n*r,
        saved_REAL_observable_rows=21047, comparison_weight_scenarios=3,
        comparison_parent_weight_values=3*n, bank_hypothesis_weight_cells=cells,
        fake_scalar_moment_summaries=cells*8, fake_flag_rate_summaries=cells*9,
        REAL_scalar_moment_summaries=8, REAL_flag_rate_summaries=9,
        within_between_decompositions=cells*4, native4_parent_scalar_moment_rows=libraries*n*4,
        NN_forward_rows=0, checkpoint_loads=0, AD_queries=0, G_updates=0, optimizer_steps=0,
        new_RNG_requests=0, new_ADC_argmax_gather_rows=0, new_training_trials=0,
        new_parameter_points=0, parameter_fits=0, new_CDF_or_quantile_sorts=0,
        new_pair_distances=0, original_SOURCE_REAL_DEV_waveform_reads=0)


def validate_settings026(p, *, frozen=True, driver_source=None):
    require(not frozen or p['status'] == 'frozen', '026 draft cannot execute')
    require(all(p.get(k) == v for k, v in settings026().items() if k != 'status'),
            '026 fixed settings differ')
    require(p['budget_caps'] == budget_caps026(p), '026 exact additive budget differs')
    if driver_source is not None:
        require(hashlib.sha256(driver_source).hexdigest() ==
                p['source_files']['scripts/run_wp5_026.py']['sha256'], '026 driver SHA differs')


def gate026(config, expected_sha, output):
    cp = Path(config).resolve()
    require(os.name == 'nt' and cp.drive.upper() in ('E:', 'D:') and
            Path(output).resolve().drive.upper() in ('E:', 'D:'), '026 registered local Windows E/D execution only')
    require(sys.dont_write_bytecode and sha(cp) == expected_sha, '026 python -B and frozen config SHA required')
    p = read_json(cp)
    validate_settings026(p)
    require(Path(output).resolve() == Path(p['output_directory']).resolve() and not os.path.lexists(output),
            '026 exact fresh output required')
    root = Path(__file__).resolve().parents[1]
    require(set(SCIENCE_SOURCES026) <= set(p['source_files']) and
            all(Path(p['source_files'][name]['path']).resolve() == root/name for name in SCIENCE_SOURCES026),
            '026 actual scientific source closure missing')
    manifest_path = cp.parent/p['saved_input_manifest']
    require(sha(manifest_path) == p['saved_input_manifest_sha256'], '026 saved input manifest SHA differs')
    manifest = read_json(manifest_path)
    keys = {'SAVED024_AXES.npz', 'REAL_OBSERVABLES.npz', 'PARENT_INFORMATION.npz'}
    keys.update(f'zero_b{bank}_{hyp}_OBSERVABLES.npz' for bank in (0, 1) for hyp in HYPOTHESES)
    require(set(p['inputs']) == keys and manifest['entries'] == p['inputs'] and manifest['run_id'] == RUN_ID and
            manifest['file_count'] == 9 and manifest['total_frozen_input_bytes'] == 27790366,
            '026 exact nine saved inputs differ')
    records = {**p['source_files'], **{'input:'+k: v for k, v in p['inputs'].items()},
               **{'contract:'+k: v for k, v in p['contracts'].items()}}
    checked = {str(cp): expected_sha, str(manifest_path): p['saved_input_manifest_sha256']}
    for label, record in records.items():
        path = Path(record['path']).resolve()
        require(path.drive.upper() in ('E:', 'D:'), '026 bound payload outside E/D: '+label)
        if str(path) not in checked:
            require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                    '026 bound bytes differ: '+label)
            checked[str(path)] = record['sha256']
        else:
            require(checked[str(path)] == record['sha256'], '026 conflicting frozen aliases')
    scope = read_json(p['contracts']['prospective_scope']['path'])
    require(scope['schema'] == 'n01/wp5-026-prospective-scope/v1' and scope['run_id'] == RUN_ID and
            scope['inputs'] == p['inputs'] and scope['budget_caps'] == p['budget_caps'] and
            scope['resource_limits'] == p['resource_limits'], '026 registered scope differs')
    authority = read_json(p['contracts']['user_authorization']['path'])
    require(authority['necessary_additional_NN_AD_G_RNG_computation_authorized'] is True,
            '026 continuing registered-computation authorization absent')
    audit = read_json(p['contracts']['source_audit013']['path'])
    require(audit['counts']['TRAIN'] == 6593 and audit['metadata_formula_verified'] is True and
            audit['original_count_weights'] == 'uniform ones' and
            audit['calibrated_original_physical_weights'] == 'unknown', '026 SOURCE composition provenance differs')
    # ROOT binds the terminal025 native/independent closeout before freezing this execution.
    require('final025' in p['contracts'] and 'review025' in p['contracts'], '026 closed025 prerequisite bindings missing')
    return p, {name: Path(row['path']) for name, row in p['inputs'].items()}, checked


def finite_measure026(values, weights, *, draws, native4=None):
    """Weighted finite row measure; aggregate supported mass by original parent for ESS."""
    import numpy as np
    x, w = np.asarray(values, np.float64).reshape(len(weights), draws), np.asarray(weights, np.float64)
    finite = np.isfinite(x)
    mass = np.where(finite, w[:, None]/draws, 0.)
    unit_mass = mass.sum(1)
    supported_mass, original_mass = float(unit_mass.sum()), float(w.sum())
    result = dict(original_rows=x.size, defined_rows=int(finite.sum()), undefined_rows=int((~finite).sum()),
        original_weight_units=len(w), defined_weight_units=int((unit_mass > 0).sum()),
        original_raw_comparison_weight_mass=original_mass, defined_raw_comparison_weight_mass=supported_mass,
        defined_comparison_weight_mass_fraction=supported_mass/original_mass,
        effective_weight_units_ESS=None, mean=None, variance=None,
        finite_mask_normalization='independent_for_this_scalar', undefined_is_not_zero=True)
    if supported_mass == 0:
        return dict(result, status='undefined_no_finite_weight_mass')
    result['effective_weight_units_ESS'] = supported_mass**2/float(np.square(unit_mass).sum())
    mean = float((np.where(finite, x, 0.)*mass).sum()/supported_mass)
    variance = float((np.square(np.where(finite, x-mean, 0.))*mass).sum()/supported_mass)
    result.update(status='defined', mean=mean, variance=variance)
    if native4 is not None:
        means, within = native4
        alpha = w/original_mass
        between = float(np.dot(alpha, np.square(means-mean)))
        within_mean = float(np.dot(alpha, within))
        np.testing.assert_allclose(between+within_mean, variance, rtol=1e-12, atol=0.)
        result['finite_native4_decomposition'] = dict(between=between, within=within_mean,
            total=between+within_mean, direct_variance=variance, closure_residual=between+within_mean-variance,
            within_denominator=4, finite_empirical_measure_only=True, true_kernel_variance=False)
    return result


def run026(config, config_sha256, output):
    started = time.monotonic()
    p, paths, checked = gate026(config, config_sha256, output)
    store = Store(output, p, checked)
    cache = store.root/'runtime_cache'
    cache.mkdir()
    for name in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR'):
        os.environ[name] = str(cache)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    # Original021 native single-CPU boundary, without a new controller.
    process = wintypes.HANDLE(-1)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    affinity_query = kernel32.GetProcessAffinityMask
    affinity_query.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    affinity_query.restype = wintypes.BOOL
    affinity_set = kernel32.SetProcessAffinityMask
    affinity_set.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    affinity_set.restype = wintypes.BOOL
    allowed_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not affinity_query(process, ctypes.byref(allowed_mask), ctypes.byref(system_mask)):
        raise OSError(ctypes.get_last_error(), '026 owned process affinity query failed')
    initial_affinity_mask = allowed_mask.value
    selected_cpu_mask = initial_affinity_mask & -initial_affinity_mask
    if not affinity_set(process, selected_cpu_mask):
        raise OSError(ctypes.get_last_error(), '026 owned process single-CPU binding failed')
    store.status()
    store.json('STARTED.json', dict(store.base(), schema='n01/wp5-026-started/v1',
        configuration_sha256=config_sha256, initial_affinity_mask=initial_affinity_mask,
        selected_cpu_mask=selected_cpu_mask, physical_weight_calibration='unknown'))
    import numpy as np
    try:
        store.call('one_registered_composition_stress', dict(new_diagnostic_jobs=1), lambda: None)
        store.phase = 'SOURCE_identity_alignment'
        with np.load(paths['PARENT_INFORMATION.npz'], allow_pickle=False) as z:
            info = store.call('saved_SOURCE_metadata_once',
                dict(numeric_NPZ_files_decoded=1, saved_SOURCE_information_rows=8192),
                lambda: {name: z[name] for name in ('source_id', 'role', 'h_visible')})
        with np.load(paths['SAVED024_AXES.npz'], allow_pickle=False) as z:
            axes = store.call('saved_original_parent_axes_once', dict(numeric_NPZ_files_decoded=1),
                lambda: {name: z[name] for name in ('source_ids', 'original_physical_weights', 'source_parent_row', 'draw_index')})
        require(info['source_id'].shape == info['role'].shape == info['h_visible'].shape == (8192,) and
                info['source_id'].dtype.kind == info['role'].dtype.kind == 'U' and
                axes['source_ids'].shape == axes['original_physical_weights'].shape == (6593,) and
                np.array_equal(axes['source_parent_row'], np.repeat(np.arange(6593), 4)) and
                np.array_equal(axes['draw_index'], np.tile(np.arange(4), 6593)), '026 external SOURCE/axes schema differs')
        train = np.flatnonzero(info['role'] == 'TRAIN')
        ids = axes['source_ids'].astype(str)
        source_ids = info['source_id'][train].astype(str)
        require(len(train) == len(np.unique(source_ids)) == len(np.unique(ids)) == 6593 and
                set(source_ids) == set(ids), '026 full TRAIN ID sets differ; no positional join')
        lookup = {identifier: row for identifier, row in zip(source_ids, train)}
        aligned = np.asarray([lookup[identifier] for identifier in ids], np.int64)
        h = info['h_visible'][aligned]
        require(h.dtype == np.float64 and np.isfinite(h).all() and ((h >= 0) & (h <= 1)).all(),
                '026 bound h composition domain differs')
        store.call('complete_exact_SOURCE_TRAIN_join', dict(aligned_SOURCE_TRAIN_parents=6593), lambda: None)
        weights = store.call('three_fixed_positive_comparison_weights',
            dict(comparison_weight_scenarios=3, comparison_parent_weight_values=19779),
            lambda: np.stack((np.ones(6593), 1-.20*(2*h-1), 1+.20*(2*h-1))))
        shutil.copyfile(paths['SAVED024_AXES.npz'], store.root/'PRESERVED_SAVED024_AXES.npz')
        store.npz('SOURCE_ALIGNMENT_AND_COMPARISON_WEIGHTS.npz', source_ids=axes['source_ids'],
            original_SOURCE_information_row=aligned, source_role=info['role'][aligned], h_visible=h,
            source_parent_row=axes['source_parent_row'], draw_index=axes['draw_index'],
            original_physical_weights=axes['original_physical_weights'], scenario_names=np.asarray(SCENARIOS),
            raw_comparison_weights=weights, normalized_comparison_parent_weights=weights/weights.sum(1, keepdims=True))
        del info, lookup, source_ids, train, aligned, h
        store.phase = 'saved_REAL_target'
        names = tuple('observable__'+name for name in SCALARS)+tuple('derived_flag__'+name for name in FLAGS)
        with np.load(paths['REAL_OBSERVABLES.npz'], allow_pickle=False) as z:
            real = store.call('saved_REAL_observables_once',
                dict(numeric_NPZ_files_decoded=1, saved_REAL_observable_rows=21047),
                lambda: {name: z[name] for name in names+('real_ids', 'real_groups')})
        require(all(real[name].shape == (21047,) for name in real) and
                len(np.unique(real['real_ids'])) == 21047 and len(np.unique(real['real_groups'])) == 422 and
                all(real['derived_flag__'+name].dtype == np.bool_ for name in FLAGS) and
                all(np.isfinite(real['observable__'+name]).all() for name in PRIMARY), '026 saved REAL boundary differs')
        real_moments, real_flags = {}, {}
        for name in SCALARS:
            value = store.call('REAL_finite_scalar_moment', dict(REAL_scalar_moment_summaries=1),
                lambda name=name: finite_measure026(real['observable__'+name], np.ones(21047), draws=1), scalar=name)
            value.update(statistical_weight_unit='saved_REAL_row_not_independent_acquisition',
                         REAL_row_mass_ESS=value['effective_weight_units_ESS'])
            real_moments[name] = value
        for name in FLAGS:
            value = real['derived_flag__'+name]
            real_flags[name+'_rate'] = dict(count=int(value.sum()), original_rows=21047, rate=float(value.mean()),
                                           defined_rows=21047, original_row_weight_mass=1.)
            store.call('REAL_full_population_flag', dict(REAL_flag_rate_summaries=1), lambda: None, flag=name)
        store.npz('PRESERVED_REAL_IDENTITIES.npz', real_ids=real['real_ids'], real_groups=real['real_groups'])
        store.json('REAL_TARGET_MOMENTS.json', dict(schema='n01/wp5-026-REAL-target/v1', run_id=RUN_ID,
            scalar_moments=real_moments, flags=real_flags, role='original_exposed_TRAIN',
            original_logical_groups=422, independent_acquisitions=1, original_REAL_waveform_reads=0))
        del real
        cases = []
        for bank in (0, 1):
            for hyp in HYPOTHESES:
                label = f'b{bank}_{hyp}'
                store.phase = 'saved_M0_'+label
                with np.load(paths['zero_'+label+'_OBSERVABLES.npz'], allow_pickle=False) as z:
                    fake = store.call('saved_M0_scalar_and_flag_rows_once',
                        dict(numeric_NPZ_files_decoded=1, saved_fake_observable_files=1, saved_fake_observable_rows=26372),
                        lambda: {name: z[name] for name in names}, bank=bank, hypothesis=hyp)
                require(all(value.shape == (26372,) for value in fake.values()) and
                        all(fake['derived_flag__'+name].dtype == np.bool_ for name in FLAGS) and
                        all(np.isfinite(fake['observable__'+name]).all() for name in PRIMARY), '026 saved fake boundary differs')
                primary = np.stack([fake['observable__'+name] for name in PRIMARY], axis=1).reshape(6593, 4, 4)
                parent_means, parent_variances = store.call('native4_parent_scalar_moments_once',
                    dict(native4_parent_scalar_moment_rows=6593*4),
                    lambda: (primary.mean(1), primary.var(1, ddof=0)), bank=bank, hypothesis=hyp)
                store.npz(label+'_NATIVE4_PARENT_MOMENTS.npz', scalar_names=np.asarray(PRIMARY),
                    parent_means=parent_means, parent_variances_within_denominator4=parent_variances)
                support_arrays = {'defined__'+name: np.isfinite(fake['observable__'+name]) for name in SCALARS}
                store.npz(label+'_FINITE_SUPPORT.npz', **support_arrays)
                for scenario, w in zip(SCENARIOS, weights):
                    store.phase = f'statistics_{label}_{scenario}'
                    scalar_results, flag_results = {}, {}
                    for name in SCALARS:
                        native4 = ((parent_means[:, PRIMARY.index(name)], parent_variances[:, PRIMARY.index(name)])
                                   if name in PRIMARY else None)
                        charges = dict(fake_scalar_moment_summaries=1)
                        if native4 is not None:
                            charges['within_between_decompositions'] = 1
                        value = store.call('fixed_weight_finite_scalar_moment', charges,
                            lambda name=name, native4=native4: finite_measure026(fake['observable__'+name], w, draws=4, native4=native4),
                            bank=bank, hypothesis=hyp, scenario=scenario, scalar=name)
                        ref = real_moments[name]
                        value.update(parent_ESS=value['effective_weight_units_ESS'], statistical_weight_unit='SOURCE_parent',
                            signed_mean_minus_REAL=None if value['mean'] is None or ref['mean'] is None else value['mean']-ref['mean'],
                            signed_variance_minus_REAL=None if value['variance'] is None or ref['variance'] is None else value['variance']-ref['variance'])
                        scalar_results[name] = value
                    alpha = w/w.sum()
                    for name in FLAGS:
                        flags = fake['derived_flag__'+name].reshape(6593, 4)
                        rate = float(np.dot(alpha, flags.mean(1)))
                        flag_results[name+'_rate'] = dict(count=int(flags.sum()), original_rows=26372, original_parents=6593,
                            rate=rate, original_raw_comparison_parent_mass=float(w.sum()),
                            flag_raw_comparison_weight_mass=float(np.dot(w, flags.mean(1))),
                            parent_ESS=1/float(np.square(alpha).sum()), defined_rows=26372,
                            effective_weight_mass_fraction=1., signed_rate_minus_REAL=rate-real_flags[name+'_rate']['rate'])
                        store.call('fixed_weight_full_population_flag', dict(fake_flag_rate_summaries=1), lambda: None,
                                   bank=bank, hypothesis=hyp, scenario=scenario, flag=name)
                    cases.append(dict(bank=bank, hypothesis=hyp, scenario=scenario,
                        raw_comparison_weight_sum=float(w.sum()), parent_ESS=1/float(np.square(alpha).sum()),
                        scalar_moments=scalar_results, flags=flag_results, response_kernel_changed=False))
                    store.call('complete_fixed_bank_hypothesis_weight_cell', dict(bank_hypothesis_weight_cells=1),
                               lambda: None, bank=bank, hypothesis=hyp, scenario=scenario)
                del fake, primary, parent_means, parent_variances, support_arrays
        require(store.counts == p['budget_caps'] and store.counts == store.attempted and not store.inflight,
                '026 exact completed counts differ')
        unchanged(checked)
        require(sha(store.root/'PRESERVED_SAVED024_AXES.npz') == p['inputs']['SAVED024_AXES.npz']['sha256'],
                '026 original axes/weights preservation differs')
        if not affinity_query(process, ctypes.byref(allowed_mask), ctypes.byref(system_mask)) or allowed_mask.value != selected_cpu_mask:
            raise OSError(ctypes.get_last_error(), '026 single-CPU binding changed')
        memory_query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        memory_query.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
        memory_query.restype = wintypes.BOOL
        pointer_bytes = ctypes.sizeof(ctypes.c_size_t)
        memory = ctypes.create_string_buffer(8+9*pointer_bytes)
        ctypes.c_uint32.from_buffer(memory).value = ctypes.sizeof(memory)
        if not memory_query(process, ctypes.byref(memory), ctypes.sizeof(memory)):
            raise OSError(ctypes.get_last_error(), '026 native current/peak working-set query failed')
        resources = dict(elapsed_seconds=time.monotonic()-started,
            peak_working_set_bytes=ctypes.c_size_t.from_buffer(memory, 8).value,
            current_working_set_bytes=ctypes.c_size_t.from_buffer(memory, 8+pointer_bytes).value,
            initial_affinity_mask=initial_affinity_mask, selected_cpu_mask=selected_cpu_mask,
            actual_affinity_mask=allowed_mask.value, allowed_CPU_count=allowed_mask.value.bit_count(), BLAS_OMP_threads=1,
            snapshot_stage='after_analysis_before_final_result_serialization', whole_process_peak_RSS_measured=False,
            continuous_memory_kill_loop=False)
        require(resources['elapsed_seconds'] <= 300 and resources['peak_working_set_bytes'] <= 1024**3,
                '026 registered wall/working-set cap exceeded')
        store.json('ANALYSIS.json', dict(schema='n01/wp5-026-composition-stress/v1', run_id=RUN_ID,
            finite_saved_ensemble_only=True, M0_seed=0, cases=cases, REAL_scalar_moments=real_moments, REAL_flags=real_flags,
            exact_completed_counts=store.counts, resources=resources, within_empirical_denominator=4,
            physical_weight_calibration='unknown', original_weights_preserved=True,
            comparison_weights_multiply_physical_weights=False, true_conditional_kernel_variance=False,
            real_population_fit=False, Q_calibrated_neutron_energy=False, CDF_or_quantile_computed=False,
            Q1_re_evaluated=False, response_acceptance=False, full_WP5_WP6_complete=False))
        inventory = {str(path.relative_to(store.root)): dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(store.root.rglob('*')) if path.is_file() and path.name not in ('status.json', 'operations.jsonl')}
        store.json('ARTIFACT_INVENTORY.json', dict(schema='n01/wp5-026-artifacts/v1', run_id=RUN_ID, files=inventory))
        store.phase = 'complete_fixed_composition_stress'
        store.json('COMPLETE.json', dict(store.base(), schema='n01/wp5-026-complete/v1', native_analysis_complete=True,
            artifact_inventory_sha256=sha(store.root/'ARTIFACT_INVENTORY.json'), resources=resources,
            further_computation_started=False, scientific_acceptance=False))
        store.status()
        require(sum(path.stat().st_size for path in store.root.rglob('*') if path.is_file()) <= 64*1024**2 and
                time.monotonic()-started <= 300, '026 final output/wall cap exceeded')
        print(dict(run_id=RUN_ID, native_status='COMPLETE', cells=18, finite_decompositions=72), flush=True)
    except BaseException as error:
        store.error = repr(error)
        store.phase = 'failed'
        store.json('FAILED.json', dict(store.base(), schema='n01/wp5-026-failed/v1'))
        store.status()
        raise
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    return run026(args.config, args.config_sha256, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
