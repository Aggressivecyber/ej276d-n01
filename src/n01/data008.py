"""TRAIN-only voltage-preserving 256-sample interface for N01 window008.

This module never reads source_val/target_val waveform members. It reuses the
bound N01 TRAIN loader (including canonical REAL ADC) before baseline removal.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .data import load_training, sha256


DEFAULT_SPEC = dict(schema='n01/window008-data/v1', input_length=480, carrier_length=384, length=256,
                    sample_interval_ns=1.0, baseline_start=0, baseline_stop=100,
                    peak_search_start=112, peak_search_stop=144, samples_before_peak=16,
                    tail_start=46, charge_floor_V_ns=1e-8,
                    global_peak_must_be_in_search=False)
INPUT_HASHES = dict(container='9e2f2c3c18470f4b2f3374a39590a9c27b18222bd1b22b5d7467e561409ad854',
                    identities='5145ea04d038510c9ca740d997e0cf9d060ed69210a4515e20110d884db7e845')


def window_spec(spec=None):
    result = copy.deepcopy(DEFAULT_SPEC)
    if spec is not None:
        unknown = set(spec) - set(result)
        if unknown:
            raise ValueError('Unknown window008 data keys: ' + ', '.join(sorted(unknown)))
        result.update(spec)
    if (result['schema'] != DEFAULT_SPEC['schema'] or result['input_length'] != 480
            or result['carrier_length'] != 384 or result['length'] != 256 or result['sample_interval_ns'] != 1.0
            or result['baseline_start'] != 0 or result['baseline_stop'] != 100
            or result['samples_before_peak'] != 16 or result['tail_start'] != 46
            or result['global_peak_must_be_in_search'] is not False):
        raise ValueError('Window008 fixed units, baseline, crop or readout changed')
    a, b = result['peak_search_start'], result['peak_search_stop']
    if (a, b) != (112, 144):
        raise ValueError('Root-frozen peak search is [112,144)')
    if not np.isfinite(result['charge_floor_V_ns']) or result['charge_floor_V_ns'] <= 0:
        raise ValueError('Invalid positive charge guard')
    return result


def readout256_np(waveform, spec=None):
    """New software readout: Q=sum(all256)*dt; tail starts at peak+30."""
    spec = window_spec(spec)
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != spec['length'] or not np.isfinite(x).all():
        raise ValueError('Expected finite [event,256] voltage waveforms')
    q = x.sum(axis=1, dtype=np.float64) * spec['sample_interval_ns']
    tail = x[:, spec['tail_start']:].sum(axis=1, dtype=np.float64) * spec['sample_interval_ns']
    if (q <= spec['charge_floor_V_ns']).any():
        raise ValueError('Nonpositive/guarded new Q256; retain all rows and diagnose')
    return dict(Q256=q, tail_charge256=tail, PSD256=tail / q)


def transform_domain(waveform, spec=None, *, domain='source'):
    spec = window_spec(spec)
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 480 or not np.isfinite(x).all():
        raise ValueError('Expected finite original [event,480] voltage waveforms')
    carrier = x[:, :spec['carrier_length']].copy()
    a, b = spec['peak_search_start'], spec['peak_search_stop']
    if domain == 'source':
        # SOURCE is a clean expected-voltage input, not an ADC measurement.
        from .observation008 import crop_existing_values
        observed = crop_existing_values(carrier)
        measured = carrier
        baseline = measured[:, :100].copy()
        offset = observed['baseline_mean_V']
        peak, cropped = observed['peak_index'], observed['window_V']
    elif domain == 'real':
        from .observation008 import observe_numpy
        observed = observe_numpy(carrier, already_digitized=True, return_carrier=True)
        measured = observed['carrier_adc']
        baseline = measured[:, :100].copy()
        offset = observed['baseline_mean_V']
        peak, cropped = observed['peak_index'], observed['window_V']
    else:
        raise ValueError('Unknown domain; choose source or real explicitly')
    global_peak = np.argmax(measured, axis=1)
    start = peak - spec['samples_before_peak']
    if (start < 0).any() or (start + 256 > spec['carrier_length']).any():
        raise ValueError('Crop would need padding; no truncation or padding is permitted')
    readout = readout256_np(cropped, spec)
    return dict(waveform=cropped, carrier384=carrier, baseline100=baseline,
                raw_baseline100=carrier[:, :100].copy(), baseline_mean_V=offset,
                peak_index480=peak.astype(np.int64), crop_start480=start.astype(np.int64),
                peak_amplitude_V=measured[np.arange(len(x)), peak] - offset,
                peak_tied_samples=(measured[:, a:b] == measured[np.arange(len(x)), peak, None]).sum(axis=1),
                peak_search_edge=(peak == a) | (peak == b - 1),
                global_peak_outside_search=(global_peak < a) | (global_peak >= b),
                **readout)


def load_training008(container, identity_path, spec=None, *, expected_hashes=None):
    """Retain every legal TRAIN parent and only DEV identity/group metadata."""
    spec = window_spec(spec)
    expected = INPUT_HASHES if expected_hashes is None else expected_hashes
    actual = dict(container=sha256(container), identities=sha256(identity_path))
    if actual != expected:
        raise ValueError('Original N01 input binding changed before numeric decode')
    old = load_training(container, identity_path)
    result = {key: value.copy() for key, value in old.items() if key not in ('source', 'real')}
    for domain in ('source', 'real'):
        values = transform_domain(old[domain], spec, domain=domain)
        result[domain] = values.pop('waveform')
        result.update({domain + '_' + key: value for key, value in values.items()})
        result[domain + '_role'] = np.full(len(old[domain]), 'TRAIN', dtype='<U5')
    # These named fields contain lineage, not held-out pulse values or hval.
    with np.load(container, allow_pickle=False) as archive:
        result['source_val_ids'] = archive['source_val_ids'].copy()
    with np.load(identity_path, allow_pickle=False) as archive:
        result['real_val_original_index'] = archive['real_val_original_index'].copy()
        result['real_val_group'] = archive['real_val_group'].copy()
    result['source_val_role'] = np.full(len(result['source_val_ids']), 'DEV', dtype='<U3')
    result['real_val_role'] = np.full(len(result['real_val_original_index']), 'DEV', dtype='<U3')
    if np.intersect1d(result['source_ids'], result['source_val_ids']).size:
        raise ValueError('SOURCE TRAIN/DEV parent overlap')
    if np.intersect1d(result['real_ids'], result['real_val_original_index']).size:
        raise ValueError('REAL TRAIN/DEV parent overlap')
    if result['source'].shape != (6593, 256) or result['real'].shape != (21047, 256):
        raise ValueError('Original legal TRAIN counts changed')
    return result


def fit_model_statistics008(data, spec=None):
    """New-window model scales, with names required by the original model API."""
    spec = window_spec(spec)
    q = readout256_np(data['source'], spec)['Q256']
    peak = np.asarray(data['source_peak_amplitude_V'], dtype=np.float64)
    baseline = np.asarray(data['real_baseline100'], dtype=np.float64)
    offset = baseline.mean(1)
    centered = baseline - offset[:, None]
    t = 2 * np.arange(100, dtype=np.float64) / (spec['carrier_length'] - 1) - 1
    tc = t - t.mean()
    slope = centered @ tc / (tc @ tc)
    midpoint_offset = offset - slope * t.mean()
    rms = (centered - slope[:, None] * tc).std(1, ddof=1)
    return dict(q_ref=float(np.median(q)), a_ref=float(np.median(peak)),
                log_q_min=float(np.log(q.min())), log_q_max=float(np.log(q.max())),
                electronics_rms=float(rms.mean()), electronics_offset_mean=float(midpoint_offset.mean()),
                electronics_offset_std=float(max(midpoint_offset.std(ddof=1), 1e-7)),
                electronics_log_rms_std=float(max(np.log(np.maximum(rms, 1e-8)).std(ddof=1), .01)),
                electronics_slope_std=float(max(slope.std(ddof=1), 1e-7)),
                amplitude_floor=1e-8, charge_floor=spec['charge_floor_V_ns'], alignment_t0=126.)


def fit_statistics008(data, spec=None):
    """Independent per-domain TRAIN summaries; no reused 480-point Q bins."""
    spec = window_spec(spec)
    levels = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
    result = dict(schema='n01/window008-statistics/v1', data_spec=spec,
                  split='TRAIN', units=dict(waveform='V', charge='V ns', PSD='dimensionless'),
                  quantile_levels=levels, domains={}, model=fit_model_statistics008(data, spec),
                  same_bin_number_across_domains_has_same_membership=False,
                  old_480_Q_statistics_reused=False, DEV_waveforms_decoded=False)
    for domain in ('source', 'real'):
        x = np.asarray(data[domain], dtype=np.float64)
        r = readout256_np(x, spec)
        q, psd = r['Q256'], r['PSD256']
        baseline = data[domain + '_baseline100']
        centered = baseline - data[domain + '_baseline_mean_V'][:, None]
        result['domains'][domain] = dict(
            rows=len(x), waveform_mean_V=x.mean(0).tolist(), waveform_std_V=x.std(0, ddof=1).tolist(),
            Q256_quantiles=np.quantile(q, levels).tolist(), Q256_mean=float(q.mean()),
            Q256_std=float(q.std(ddof=1)), Q256_median=float(np.median(q)),
            Q256_eight_quantile_bin_edges=np.quantile(q, np.linspace(0, 1, 9)).tolist(),
            PSD256_quantiles=np.quantile(psd, levels).tolist(), PSD256_mean=float(psd.mean()),
            PSD256_std=float(psd.std(ddof=1)), PSD256_negative_rows=int((psd < 0).sum()),
            PSD256_above_one_rows=int((psd > 1).sum()),
            peak_amplitude_V_quantiles=np.quantile(data[domain + '_peak_amplitude_V'], levels).tolist(),
            peak_index480_min=int(data[domain + '_peak_index480'].min()),
            peak_index480_max=int(data[domain + '_peak_index480'].max()),
            tied_peak_rows=int((data[domain + '_peak_tied_samples'] > 1).sum()),
            peak_search_edge_rows=int(data[domain + '_peak_search_edge'].sum()),
            global_peak_outside_search_rows=int(data[domain + '_global_peak_outside_search'].sum()),
            baseline_mean_V_quantiles=np.quantile(data[domain + '_baseline_mean_V'], levels).tolist(),
            baseline_centered_rms_V_quantiles=np.quantile(centered.std(1, ddof=1), levels).tolist(),
            nonpositive_or_guarded_Q_rows=0, rows_removed=0)
    return result


def self_test():
    """Artificial CPU array checks; no scientific data, fits or forwards."""
    t = np.arange(480)
    pulse = np.exp(-((t - 126) / 8) ** 2) * .03
    x = np.stack([pulse + .002, pulse * 2 - .001]).astype(np.float64)
    z = transform_domain(x)
    expected = x[:, 110:366] - x[:, :100].mean(1)[:, None]
    # NumPy and shared Torch float64 mean reductions can differ by roundoff.
    sample_tolerance = 8 * np.finfo(np.float64).eps * np.abs(x).max()
    np.testing.assert_allclose(z['waveform'], expected, rtol=0, atol=sample_tolerance)
    np.testing.assert_array_equal(z['peak_index480'], [126, 126])
    np.testing.assert_allclose(z['tail_charge256'], expected[:, 46:].sum(1), rtol=0, atol=256 * sample_tolerance)
    np.testing.assert_array_equal(z['carrier384'], x[:, :384])
    scaled = transform_domain(x * 3)
    np.testing.assert_allclose(scaled['Q256'], z['Q256'] * 3, rtol=1e-14, atol=1e-14)
    # A negative tail is data, and is neither clipped nor dropped.
    signed = np.zeros((1, 256)); signed[0, 16] = 1.; signed[0, 46:] = -.001
    assert readout256_np(signed)['PSD256'][0] < 0
    bad = x.copy(); bad[0, 300] = .2
    flagged = transform_domain(bad)
    assert flagged['global_peak_outside_search'][0] and len(flagged['waveform']) == len(bad)
    try:
        readout256_np(np.zeros((1, 256)))
    except ValueError:
        pass
    else:
        raise AssertionError('Zero-charge waveform was accepted')
    return dict(status='passed', artificial_CPU_checks=7, scientific_training_updates=0,
                scientific_model_forwards=0, DEV_waveforms_decoded=False)
