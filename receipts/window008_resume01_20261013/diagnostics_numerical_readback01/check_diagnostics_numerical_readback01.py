"""Independent, file-streamed CPU readback of completed Window008 diagnostics.

Only saved artifacts and frozen source byte hashes are opened. No torch import,
checkpoint load, original dataset/DEV decode, model call, fit, GPU or optimizer.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

ARMS = ('deterministic', 'full', 'simple')
HYPOTHESES = ('fast', 'slow', 'common_level')
RTOL, ATOL = 2e-12, 2e-14


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def peak_RSS():
    if os.name != 'nt':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[(k, ctypes.c_size_t) for k in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    values = Counters(); values.cb = ctypes.sizeof(values)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True); kernel.GetCurrentProcess.restype = wintypes.HANDLE
    memory = ctypes.WinDLL('psapi', use_last_error=True)
    memory.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not memory.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(values.PeakWorkingSetSize)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)


def summary(value):
    value = np.asarray(value, np.float64).ravel(); finite = np.isfinite(value); x = value[finite]
    return dict(rows=len(value), defined_rows=int(finite.sum()), undefined_rows=int((~finite).sum()),
                mean=float(x.mean()) if len(x) else None,
                minimum=float(x.min()) if len(x) else None, maximum=float(x.max()) if len(x) else None,
                p10_p50_p90=np.quantile(x, [.1, .5, .9]) if len(x) else [None]*3)


def raw_masks(archive, prefix, shape):
    q, raw = archive[prefix+'Q'], archive[prefix+'PSD_raw']
    assert q.shape == raw.shape == shape
    np.testing.assert_array_equal(np.isnan(raw), q == 0)
    np.testing.assert_array_equal(archive[prefix+'PSD_undefined'], q == 0)
    np.testing.assert_array_equal(archive[prefix+'Q_nonpositive'], q <= 0)
    if prefix+'rise10_90_ns' in archive.files:
        rise, width = archive[prefix+'rise10_90_ns'], archive[prefix+'FWHM_ns']
        np.testing.assert_array_equal(archive[prefix+'rise10_90_defined'], np.isfinite(rise))
        np.testing.assert_array_equal(archive[prefix+'FWHM_defined'], np.isfinite(width))
        close(rise, archive[prefix+'left_crossing90_ns']-archive[prefix+'left_crossing10_ns'])
        close(width, archive[prefix+'right_crossing50_ns']-archive[prefix+'left_crossing50_ns'])
        np.testing.assert_array_equal(archive[prefix+'selected_output_peak_index'], np.full(shape, 16))
        np.testing.assert_array_equal(archive[prefix+'temporal_nonpositive_peak'], archive[prefix+'peak_voltage_V'] <= 0)
    return dict(raw_PSD_undefined=int((q == 0).sum()), nonpositive_Q=int((q <= 0).sum()),
                guarded_PSD=int(archive[prefix+'PSD_guarded'].sum()),
                undefined_rise10_90=int((~archive[prefix+'rise10_90_defined']).sum()) if prefix+'rise10_90_defined' in archive.files else None,
                undefined_FWHM=int((~archive[prefix+'FWHM_defined']).sum()) if prefix+'FWHM_defined' in archive.files else None)


def bootstrap_counts(directory, filename, key, value):
    split = filename.split('_')[0]; metadata = read(directory/(split+'_PARENT_BOOTSTRAP.json'))
    if key in ('mean', 'std'):
        expected = metadata[key+'_defined_replicates']
    elif key.startswith(('deterministic_', 'simple_')):
        ref, metric = key.split('_', 1); expected = metadata['comparisons'][ref][metric]['defined_replicates']
    else:
        expected = metadata['hypothesis_sensitivity'][key]['defined_replicates']
    np.testing.assert_array_equal(np.isfinite(value).sum(0), np.asarray(expected))


def reopen_directory(directory):
    s = read(directory/'SUMMARY.json'); entries = s['artifact_sha256']; inventory = {}
    for name, expected in entries.items():
        assert Path(name).name == name
        actual = sha(directory/name)
        assert actual == expected, 'SHA256 mismatch '+str(directory/name)
    nan_members = {}; members = 0; files = 0
    for path in sorted(directory.glob('*.npz')):
        files += 1; per_file = {}
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]; members += 1
                assert value.dtype.kind != 'O', (path.name, key)
                descriptor = dict(shape=list(value.shape), dtype=str(value.dtype), nan_count=0)
                if value.dtype.kind in 'fc':
                    assert not np.isinf(value).any(), 'Infinity '+path.name+':'+key
                    count = int(np.isnan(value).sum()); descriptor['nan_count'] = count
                    if path.name.endswith('_parent_bootstrap.npz'):
                        bootstrap_counts(directory, path.name, key, value)
                    if count:
                        if path.name.endswith('_parent_bootstrap.npz'):
                            reason = 'bootstrap undefined ratios or unsupported membership, exact defined-replicate count matched JSON'
                        elif path.name.endswith('_curves.npz') and key.endswith(('_mean', '_std')):
                            rows_key = key.rsplit('_', 1)[0]+'_rows'
                            assert rows_key in archive.files
                            np.testing.assert_array_equal(np.isnan(value), np.repeat((archive[rows_key] == 0)[:, None], value.shape[-1], axis=1))
                            reason = 'empty conditional bin with exact saved row-count mask'
                        elif any(token in key for token in ('PSD_raw', 'crossing', 'rise10_90', 'FWHM', 'normalized_waveform', 'scalar_mean', 'scalar_sample_covariance')):
                            reason = 'registered raw ratio/time/normalized/counted scalar definition; detailed masks checked separately'
                        else:
                            raise AssertionError('Unregistered NaN '+path.name+':'+key)
                        nan_members[path.name+':'+key] = dict(count=count, reason=reason)
                per_file[key] = descriptor
                del value
        inventory[path.name] = per_file
    return dict(hash_bound_files=len(entries), npz_files=files, npz_members=members, all_allow_pickle_false=True,
                no_infinities=True, nan_members=nan_members, inventory=inventory, SUMMARY_sha256=sha(directory/'SUMMARY.json'))


def validate_selection(directory, primary):
    with np.load(directory/'SELECTIONS.npz', allow_pickle=False) as a:
        values = {k: a[k] for k in a.files}
    with np.load(primary/'DEV_observables.npz', allow_pickle=False) as a:
        ids = a['source_parent_ids']
    assert ids.shape == (512,)
    np.testing.assert_array_equal(values['all_SOURCE_ids'], ids)
    selected = np.random.default_rng(2026101399).permutation(512)[:128]
    np.testing.assert_array_equal(values['repeat_selected_rows'], selected)
    np.testing.assert_array_equal(values['repeat_selected_ids'], ids[selected])
    np.testing.assert_array_equal(values['selected_rows'], selected[:32])
    np.testing.assert_array_equal(values['selected_ids'], ids[selected[:32]])
    q = values['all_SOURCE_Q']; assert q.shape == (512,) and (q > 0).all()
    donors = []
    for parent in selected[:32]:
        gap = np.abs(np.log(q/q[parent])); gap[parent] = np.inf
        donors.append(np.lexsort((ids, gap))[0])
    np.testing.assert_array_equal(values['donor_rows'], donors)
    np.testing.assert_array_equal(values['donor_ids'], ids[donors])
    close(values['abs_log_charge_gap'], np.abs(np.log(q[donors]/q[selected[:32]])))
    x, donor, scaled = (values[k] for k in ('original_carrier384', 'donor_original_centered384', 'donor_same_Q_carrier384'))
    close(x[:, 110:366].sum(1), q[selected[:32]])
    close(scaled[:, 110:366].sum(1), q[selected[:32]])
    close(scaled, donor*(q[selected[:32]]/q[donors])[:, None])
    difference = donor/q[donors, None]-x/q[selected[:32], None]
    close(values['unit_shape_l2_difference'], np.linalg.norm(difference, axis=1))
    close(values['unit_shape_absmax_difference'], np.abs(difference).max(1))
    close(values['trend_carriers384'], x[None]*values['Q_node_scale'][:, :, None])
    close(values['trend_carriers384'][:, :, 110:366].sum(2), np.repeat(values['Q_nodes'][:, None], 32, axis=1))
    assert (scaled[:, 112:144].argmax(1)+112 == 126).all()
    assert (values['trend_carriers384'][:, :, 112:144].argmax(2)+112 == 126).all()
    return values, dict(selection_identity_exact=True, nearest_Q_donor_stable_ID_exact=True,
        parent_counts=dict(repeat=128, intervention=32), source_Q_preserved=True,
        abs_log_charge_gap=summary(values['abs_log_charge_gap']),
        input_unit_shape_L2_difference=summary(values['unit_shape_l2_difference']),
        input_unit_shape_absmax_difference=summary(values['unit_shape_absmax_difference']),
        zero_intervention_strength_parents=int((values['unit_shape_l2_difference'] == 0).sum()),
        Q_nodes=values['Q_nodes'], interpretation='Near-Q same-latent input interventions; measured strength is retained, not donor-selection success.')


def validate_latents(directory):
    metadata = read(directory/'LATENT_BANKS.json'); reports = {}
    for label, filename in (('repeat', 'repeat_latents.npz'), ('intervention', 'intervention_latents.npz')):
        binding = metadata[label]
        with np.load(directory/filename, allow_pickle=False) as a:
            for key, expected in binding['values_sha256'].items():
                x = a[key]; assert array_sha(x) == expected
                for chunk in binding['chunks']:
                    if key in chunk['latents_sha256']:
                        assert array_sha(x[chunk['start']:chunk['stop']]) == chunk['latents_sha256'][key]
                del x
        reports[label] = dict(parents=binding['parents'], draws_per_parent=binding['draws_per_parent'],
                              seeds=binding['seeds'], chunks=len(binding['chunks']), hashes_matched=True)
    assert metadata['B_C_share_exact_bank'] is True and metadata['RNG_not_advanced_per_variant_arm_or_hypothesis'] is True
    generation = read(directory/'GENERATION.json')
    for group in ('repeat', 'shape', 'Q_trend'):
        count = 8192 if group == 'repeat' else 256
        for chunk in generation['groups'][group]['chunks']:
            rows = np.arange(count, dtype=np.int64)[chunk['start']:chunk['stop']]
            assert array_sha(rows) == chunk['latent_row_sha256']
    return dict(banks=reports, same_bank_B_C=True, no_new_draw_or_forward=True)


def repeat_statistics(directory, selections):
    reports = {}; checked_parents = 0; global_errors = dict(waveform_covariance=0., waveform_mean=0., Var_Q=0., Var_PSD=0., Cov_Q_PSD=0.)
    generation = read(directory/'GENERATION.json')['groups']['repeat']['chunks']
    for arm in ARMS:
        reports[arm] = {}
        for h in HYPOTHESES:
            path = directory/('REPEAT_'+arm+'_'+h+'.npz')
            with np.load(path, allow_pickle=False) as a:
                y, cov, mean = a['waveform256'], a['waveform256_covariance'], a['waveform256_mean']
                q, psd = a['Q'], a['PSD']; vq, vp, cp = a['Var_Q_given_x'], a['Var_PSD_given_x'], a['Cov_Q_PSD_given_x']
                assert y.shape == (128, 64, 256) and cov.shape == (128, 256, 256) and mean.shape == (128, 256)
                assert q.shape == psd.shape == (128, 64) and int(a['sample_ddof']) == 1
                np.testing.assert_array_equal(a['source_parent_ids'], selections['repeat_selected_ids'])
                np.testing.assert_array_equal(a['draw'], np.arange(64))
                masks = raw_masks(a, '', (128, 64)); errors = {k: 0. for k in global_errors}
                for parent in range(128):
                    expected = np.cov(y[parent], rowvar=False, ddof=1)
                    close(cov[parent], expected); close(mean[parent], y[parent].mean(0))
                    expected_q = np.var(q[parent], ddof=1); expected_p = np.var(psd[parent], ddof=1)
                    expected_c = np.cov(q[parent], psd[parent], ddof=1)[0, 1]
                    close(vq[parent], expected_q); close(vp[parent], expected_p); close(cp[parent], expected_c)
                    errors['waveform_covariance'] = max(errors['waveform_covariance'], float(np.abs(cov[parent]-expected).max()))
                    errors['waveform_mean'] = max(errors['waveform_mean'], float(np.abs(mean[parent]-y[parent].mean(0)).max()))
                    for name, value in (('Var_Q', abs(vq[parent]-expected_q)), ('Var_PSD', abs(vp[parent]-expected_p)), ('Cov_Q_PSD', abs(cp[parent]-expected_c))):
                        errors[name] = max(errors[name], float(value))
                    checked_parents += 1
                scalar_fields = a['scalar_fields']; stack = np.stack([a[str(k)] for k in scalar_fields], axis=2)
                valid = np.isfinite(stack); counts = valid.sum(1)
                np.testing.assert_array_equal(a['scalar_defined_draws'], counts)
                np.testing.assert_array_equal(a['scalar_undefined_draws'], 64-counts)
                pair_count = valid.astype(np.int64).transpose(0, 2, 1)@valid.astype(np.int64)
                np.testing.assert_array_equal(a['scalar_covariance_defined_draws'], pair_count)
                np.testing.assert_array_equal(np.isnan(a['scalar_mean']), counts == 0)
                np.testing.assert_array_equal(np.isnan(a['scalar_sample_covariance']), pair_count < 2)
                normalized_count = np.repeat(a['normalized_waveform_defined'].sum(1)[:, None], 256, axis=1)
                np.testing.assert_array_equal(a['normalized_waveform_defined_draws'], normalized_count)
                np.testing.assert_array_equal(np.isnan(a['normalized_waveform_mean']), normalized_count == 0)
                np.testing.assert_array_equal(np.isnan(a['normalized_waveform_sample_std']), normalized_count < 2)
                np.testing.assert_array_equal(np.isnan(a['normalized_waveform_MonteCarlo_SE']), normalized_count < 2)
                close(a['waveform256_sample_std'], np.sqrt(cov.diagonal(axis1=1, axis2=2)))
                close(a['Q_identity_residual'], a['Q']-a['Q_signal']-a['Q_noise']-a['Q_ADC_delta'])
                close(a['tail_identity_residual'], a['tail']-a['tail_signal']-a['tail_noise']-a['tail_ADC_delta'])
                for chunk in generation:
                    n0, n1 = chunk['start'], chunk['stop']; record = chunk['outputs']['0_'+arm+'_'+h]
                    assert array_sha(y.reshape(-1, 256)[n0:n1]) == record['waveform256_sha256']
                qsig = a['qsig']; qbar = a['qbar']; times = a['time_scale']
                report = dict(parents=128, draws_per_parent=64, all_covariance_parents_recomputed=True,
                    recomputation_absmax=errors, undefined_and_flags=masks,
                    conditional_SD_Q_V_ns=summary(np.sqrt(vq)), conditional_SD_PSD=summary(np.sqrt(vp)),
                    conditional_Cov_Q_PSD=summary(cp), empirical_64draw_SD_qsig_V_ns=summary(qsig.std(1, ddof=1)),
                    empirical_64draw_SD_log_qsig=summary(np.log(qsig).std(1, ddof=1)),
                    empirical_64draw_CV_qsig=summary(qsig.std(1, ddof=1)/qsig.mean(1)),
                    conditional_SD_time_scale=summary(times.std(1, ddof=1)),
                    conditional_SD_carrier_peak_index=summary(a['peak_index'].std(1, ddof=1)),
                    same_crop_SD_Q_signal_V_ns=summary(a['Q_signal'].std(1, ddof=1)),
                    same_crop_SD_Q_noise_V_ns=summary(a['Q_noise'].std(1, ddof=1)),
                    same_crop_SD_Q_ADC_delta_V_ns=summary(a['Q_ADC_delta'].std(1, ddof=1)),
                    ADC_rail_rows=int(a['ADC_rail_hit'].sum()), search_boundary_rows=int(a['peak_at_search_boundary'].sum()),
                    global_peak_outside_search_rows=int(a['global_peak_outside_search'].sum()),
                    observation_decomposition_identity_absmax=dict(Q=float(np.abs(a['Q_identity_residual']).max()), tail=float(np.abs(a['tail_identity_residual']).max())),
                    log_sigma_interpretation='Empirical64-draw SD(log qsig), not the loaded model physical-head sigma or identified physical variance.')
                reports[arm][h] = report
                for key, value in errors.items():
                    global_errors[key] = max(global_errors[key], value)
            del y, cov, mean, q, psd, stack, valid, counts, pair_count, qsig, qbar, times
            print('REPEAT verified '+arm+'/'+h, flush=True)
    assert checked_parents == 128*9
    with np.load(directory/'REPEAT_COMMON.npz', allow_pickle=False) as a:
        common = dict(baseline_RMS_V=summary(a['baseline_RMS_V']), S=summary(a['S']),
                      baseline_shape=list(a['baseline100_ADC'].shape),
                      ACF_shape=list(a['baseline_ACF_lag0_99'].shape),
                      power_fraction_shape=list(a['baseline_power_fraction5'].shape))
        assert a['baseline100_ADC'].shape == (128, 64, 100) and a['S'].shape == (128, 64)
    return dict(checked_parent_covariances=checked_parents, waveform_covariance_entries_checked=checked_parents*256*256,
                frozen_tolerances=dict(rtol=RTOL, atol=ATOL, relaxed=False), maximum_absolute_recomputation_error=global_errors,
                arms=reports, shared_baseline_and_scale=common,
                physical_conditional_response_identified=False)


def interventions_statistics(directory, selections):
    reports = {}; trends = {}
    for arm in ARMS:
        reports[arm] = {}; trends[arm] = {}
        for h in HYPOTHESES:
            with np.load(directory/('SHAPE_'+arm+'_'+h+'.npz'), allow_pickle=False) as a:
                np.testing.assert_array_equal(a['source_parent_ids'], selections['selected_ids'])
                np.testing.assert_array_equal(a['draw'], np.arange(8))
                masks = {v: raw_masks(a, v+'_', (32, 8)) for v in ('original', 'donor_same_Q')}
                metrics = {}
                for metric in ('Q', 'tail', 'PSD', 'PSD_raw', 'qbar', 'qsig', 'time_scale', 'rise10_90_ns', 'FWHM_ns'):
                    delta = a['donor_same_Q_'+metric]-a['original_'+metric]
                    base = 'donor_same_Q_minus_original_delta_'+metric
                    close(a[base], delta)
                    count = np.isfinite(delta).sum(1); np.testing.assert_array_equal(a[base+'_defined_draws'], count)
                    close(a[base+'_mean'], np.nanmean(delta, axis=1))
                    close(a[base+'_std'], np.nanstd(delta, axis=1, ddof=1))
                    close(a[base+'_MonteCarlo_SE'], np.nanstd(delta, axis=1, ddof=1)/np.sqrt(count))
                    metrics[metric] = dict(paired_mean_delta=summary(a[base+'_mean']),
                        absolute_paired_mean_delta=summary(np.abs(a[base+'_mean'])),
                        paired_MonteCarlo_SE=summary(a[base+'_MonteCarlo_SE']),
                        positive_parent_means=int((a[base+'_mean'] > 0).sum()), negative_parent_means=int((a[base+'_mean'] < 0).sum()),
                        zero_parent_means=int((a[base+'_mean'] == 0).sum()))
                wave_delta = a['donor_same_Q_waveform256']-a['original_waveform256']
                close(a['donor_same_Q_minus_original_delta_waveform_mean'], wave_delta.mean(1))
                reports[arm][h] = dict(metrics=metrics, undefined_and_flags=masks,
                    mean_output_waveform_difference_RMS_V=summary(np.sqrt(np.square(wave_delta.mean(1)).mean(1))),
                    exact_zero_output_mean_difference_parents=int((np.abs(wave_delta.mean(1)).max(1) == 0).sum()))
            with np.load(directory/('Q_TREND_'+arm+'_'+h+'.npz'), allow_pickle=False) as a:
                np.testing.assert_array_equal(a['source_parent_ids'], selections['selected_ids'])
                np.testing.assert_array_equal(a['draw'], np.arange(8))
                node_masks = {v: raw_masks(a, v+'_', (32, 8)) for v in ('Q25', 'Q50', 'Q75')}
                intervals = {}
                for first, second in (('Q25', 'Q50'), ('Q50', 'Q75')):
                    tag = second+'_minus_'+first
                    qbar = a[second+'_qbar']-a[first+'_qbar']; observed = a[second+'_Q']-a[first+'_Q']
                    close(a[tag+'_delta_qbar'], qbar); close(a[tag+'_delta_Q'], observed)
                    np.testing.assert_array_equal(a[tag+'_qbar_nonnegative_per_draw'], qbar >= 0)
                    intervals[tag] = dict(draws=256, qbar_negative_draws=int((qbar < 0).sum()),
                        observed_Q_negative_draws=int((observed < 0).sum()),
                        observed_Q_negative_parent_means=int((observed.mean(1) < 0).sum()),
                        qbar_signed_delta=summary(qbar), observed_Q_mean_delta=summary(observed.mean(1)),
                        PSD_parent_mean_delta=summary(a[tag+'_delta_PSD_mean']))
                trends[arm][h] = dict(intervals=intervals, undefined_and_flags=node_masks,
                    node_observed_Q_means={v: summary(a[v+'_Q_mean']) for v in ('Q25', 'Q50', 'Q75')},
                    structural_qbar_check_separate_from_observed_Q=True)
    return dict(shape=reports, Q_trend=trends, Q_nodes=selections['Q_nodes'],
                no_energy_calibration_or_physical_nonlinearity_claim=True)


def replay_receipt(directory):
    j = read(directory/'REPLAY_IDENTITY_ORDER.json'); result = {}
    assert j['no_new_random_draws'] is True and j['signal_forward_rows'] == 1536 and j['observation_rows'] == 4608
    for name in ('rewritten_saved_ids', 'reversed_parent_storage_order'):
        r = j[name]; assert r['status'] == 'passed'
        bitwise = 0; maximum = 0.; exact_flags = 0; scalar_maximum = 0.
        assert len(r['results']) == 9
        for block in r['results'].values():
            bitwise += int(block['waveform256_bitwise_equal']); maximum = max(maximum, block['waveform256_max_abs_error'])
            for scalar in block['scalars'].values():
                if 'exact' in scalar:
                    assert scalar['exact'] is True; exact_flags += 1
                elif scalar['max_abs_error'] is not None:
                    scalar_maximum = max(scalar_maximum, scalar['max_abs_error'])
        result[name] = dict(status='passed', arm_hypothesis_blocks=9, bitwise_equal_wave_blocks=bitwise,
            waveform_absmax=maximum, scalar_absmax=scalar_maximum, exact_discrete_flag_entries=exact_flags,
            evidence='Hash-bound saved actual engineering replay receipt; no new forward in this readback.')
    return result


def population(directory, primary):
    reports = {}
    for split in ('TRAIN', 'DEV'):
        meta = read(directory/('POPULATION_'+split+'.json')); schemes = {}
        with np.load(primary/(split+'_observables.npz'), allow_pickle=False) as main, np.load(primary/(split+'_curves.npz'), allow_pickle=False) as curves, np.load(directory/('POPULATION_'+split+'.npz'), allow_pickle=False) as pop:
            bins = main['source_bins']; n = len(bins); rows = main['source_parent_row']
            weights = pop['parent_weights']; assert weights.shape == (3, n) and (weights > 0).all()
            np.testing.assert_array_equal(rows, np.repeat(np.arange(n), 4)); close(weights.sum(1), np.full(3, n))
            z = (2*bins-7)/7; raw = np.stack((np.ones(n), 1+.5*z, 1-.5*z)); expected = raw*n/raw.sum(1)[:, None]
            close(weights, expected)
            for wi, scheme in enumerate(('uniform', 'tilt_plus', 'tilt_minus')):
                r = meta['schemes'][scheme]; close(r['parent_ESS'], weights[wi].sum()**2/np.square(weights[wi]).sum())
                full = {}
                binw = np.ones(8) if wi == 0 else 1+(.5 if wi == 1 else -.5)*(2*np.arange(8)-7)/7
                binw *= n/binw[bins].sum()
                for arm in ARMS:
                    for h in HYPOTHESES:
                        label = arm+'_'+h; base = label+'_fixed_SOURCE_Q_in_SOURCE_TRAIN_bins'
                        means, stds, count = curves[base+'_mean'], curves[base+'_std'], curves[base+'_rows']
                        active = count > 0; mass = count[active]*binw[active]
                        mean = np.average(means[active], axis=0, weights=mass)
                        variance = np.average(stds[active]**2+(means[active]-mean)**2, axis=0, weights=mass)
                        close(pop[scheme+'_'+label+'_mean'], mean); close(pop[scheme+'_'+label+'_variance'], variance)
                        scalar_report = {}
                        for metric in ('Q', 'PSD'):
                            x = main[label+'_'+metric]; rw = weights[wi, rows]
                            expected_mean = np.average(x, weights=rw); expected_std = np.sqrt(np.average((x-expected_mean)**2, weights=rw))
                            got = r['arms'][arm][h]['population'][metric]
                            close(got['mean'], [expected_mean]); close(got['std'], [expected_std])
                            scalar_report[metric] = dict(mean=expected_mean, std=expected_std)
                        assert r['arms'][arm][h]['output_Q_conditional_weighted_waveform_moments'] is None
                        if arm == 'full':
                            full[h] = scalar_report
                schemes[scheme] = dict(ESS=r['parent_ESS'], weight_min=r['weight_min'], weight_max=r['weight_max'],
                    full_scalar_moments=full, hypothesis_sensitivity=r['hypothesis_sensitivity'],
                    exact_SOURCE_bin_weighted_waveform_verified=True,
                    output_Q_conditional_weighted_waveform_available=False)
        reports[split] = dict(parents=n, schemes=schemes, new_response_forward_rows=0,
                              no_REAL_weight_fit=True, h_physical_meaning='unknown')
    return reports


def run(args):
    start = time.perf_counter(); root = Path(args.project_root).resolve()
    destination = Path(args.report_dir).resolve(); destination.mkdir(parents=True, exist_ok=True)
    if (destination/'REPORT.json').exists():
        raise FileExistsError('Preserve existing readback report')
    directories = dict(diagnostics=Path(args.diagnostics).resolve(), primary=Path(args.primary).resolve(), replay=Path(args.replay).resolve())
    source_manifest = read(root/'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json')
    frozen = {name: sha(root/name) for name in source_manifest['files']}
    assert frozen == source_manifest['files'], 'Frozen source mismatch before readback'
    try:
        inventories = {}
        for name, path in directories.items():
            inventories[name] = reopen_directory(path)
            print(name+' all NPZ reopened members='+str(inventories[name]['npz_members']), flush=True)
        selections, selection_report = validate_selection(directories['diagnostics'], directories['primary'])
        latent_report = validate_latents(directories['diagnostics'])
        repeat_report = repeat_statistics(directories['diagnostics'], selections)
        intervention_report = interventions_statistics(directories['diagnostics'], selections)
        replay_report = replay_receipt(directories['diagnostics'])
        population_report = population(directories['diagnostics'], directories['primary'])
        actual = read(directories['diagnostics']/'ACTUAL_COUNTERS.json')
        assert actual['primary'] == dict(signal_forward_rows=28416, observation_rows=85248, noise005_forward_rows=8704)
        assert actual['total_signal_forward_rows'] == 29952 and actual['total_observation_rows'] == 89856
        assert {name: sha(root/name) for name in frozen} == frozen, 'Frozen source changed during readback'
        for label, directory in directories.items():
            assert sha(directory/'SUMMARY.json') == inventories[label]['SUMMARY_sha256']
        peak = peak_RSS(); assert peak < 1024**3, 'Readback peak exceeded1GiB'
        report = dict(schema='n01/window008-resume01-diagnostics-independent-readback/v1', status='passed',
            paths={k: str(v) for k, v in directories.items()}, tolerances=dict(rtol=RTOL, atol=ATOL, relaxed=False),
            npz_readback=inventories, selection=selection_report, latent_banks=latent_report,
            repeat=repeat_report, interventions=intervention_report, identity_order_replay=replay_report,
            population=population_report, recorded_scientific_execution_counts=actual,
            new_model_forwards=0, checkpoint_loads=0, original_SOURCE_REAL_DEV_decodes=0, GPU_calls=0,
            optimizer_updates=0, C2ST_updates=0, CPU_threads=2,
            frozen_source_files_verified_unchanged=len(frozen), frozen_source_sha256=frozen,
            peak_process_RSS_bytes=peak, wall_seconds=time.perf_counter()-start,
            scientific_status='response_match_not_established',
            gaps=['No repeated same-physical-event REAL response truth', 'No paired response truth or independent acquisition',
                  'Electronics extensions remain sensitivity hypotheses', 'Independent C2ST remains unexecuted',
                  'No output-Q-conditional weighted waveform moments in main artifacts', '64/8 draws are finite MonteCarlo samples'])
        save(destination/'REPORT.json', report)
        findings = {key: report[key] for key in ('selection', 'repeat', 'interventions', 'identity_order_replay', 'population', 'gaps', 'scientific_status')}
        save(destination/'FINDINGS.json', findings)
        save(destination/'PROCESS_RECEIPT.json', dict(status='checks_passed_pending_external_native_exit_receipt',
            script_sha256=sha(__file__), report_sha256=sha(destination/'REPORT.json'), findings_sha256=sha(destination/'FINDINGS.json'),
            CPU_threads=2, peak_process_RSS_bytes=peak, wall_seconds=report['wall_seconds'],
            new_model_forwards=0, checkpoint_loads=0, original_SOURCE_REAL_DEV_decodes=0, GPU_calls=0, optimizer_updates=0))
        print(json.dumps(dict(status='passed', covariance_parents=repeat_report['checked_parent_covariances'],
            covariance_absmax=repeat_report['maximum_absolute_recomputation_error'],
            npz_members={k:v['npz_members'] for k,v in inventories.items()}, peak_RSS_GiB=peak/1024**3,
            wall_seconds=report['wall_seconds'], new_model_forwards=0)), flush=True)
    except Exception as error:
        save(destination/'FAILED.json', dict(status='failed_preserved', error_type=type(error).__name__, error=str(error),
            tolerances=dict(rtol=RTOL, atol=ATOL, relaxed=False), new_model_forwards=0, checkpoint_loads=0,
            original_SOURCE_REAL_DEV_decodes=0, GPU_calls=0, optimizer_updates=0, peak_RSS_bytes=peak_RSS(),
            frozen_source_files_changed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True); parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--primary', required=True); parser.add_argument('--replay', required=True); parser.add_argument('--report-dir', required=True)
    run(parser.parse_args())
