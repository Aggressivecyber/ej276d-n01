"""Saved-hard256 observables and prospectively fixed WP4-018 scores. NumPy only."""
from __future__ import annotations

import numpy as np

from .stats017 import (FLOORS017, FEATURES017, REQUIRED_MISSING017, require,
                       support_cell017, weighted017)

KEEP8 = np.asarray((0, 1, 2, 3, 4, 5, 7, 8))
STATES018 = ('measurable', 'prepeak_censored', 'other_undefined')
TIME_FIELDS018 = ('rise10_90_ns', 'left_crossing10_ns', 'left_crossing50_ns',
                  'left_crossing90_ns', 'right_crossing50_ns',
                  'rise_lower_bound_if_left10_exists_ns')
OBSERVABLE_FIELDS018 = tuple('state_' + s for s in STATES018) + tuple(
    'defined_' + s for s in TIME_FIELDS018) + tuple('masked_' + s for s in TIME_FIELDS018) + (
    'Q_over_T', 'peak_voltage_over_T', 'masked_FWHM_over_T', 'defined_FWHM')
OBSERVABLE_TOLERANCE018 = .10
OBSERVABLE_MC_HALF_WIDTH018 = .05
Z95 = 1.959963984540054
DISTANCE_COUNTS018 = {}


def morphology018(values):
    """Vectorized exact nearest-left/right crossing definition from 017, same window."""
    y = np.asarray(values, np.float64)
    require(y.ndim == 2 and y.shape[1] == 256 and np.isfinite(y).all(), '018 finite hard256')
    peak = y[:, 16]
    crossings = []
    rows = np.arange(len(y))
    for side in ('left', 'right'):
        side_values = []
        for fraction in (.1, .5, .9):
            target = fraction * peak
            if side == 'left':
                valid = (y[:, :16] <= target[:, None]) & (target[:, None] <= y[:, 1:17])
                lo = 15 - valid[:, ::-1].argmax(1)
            else:
                valid = (y[:, 17:] <= target[:, None]) & (target[:, None] <= y[:, 16:-1])
                lo = 16 + valid.argmax(1)
            a, b = y[rows, lo], y[rows, lo + 1]
            value = (lo + (1 if side == 'left' else 0)).astype(np.float64)
            np.divide(target-a, b-a, out=value, where=a != b)
            value[a != b] += lo[a != b]
            value[~valid.any(1) | (peak <= 0)] = np.nan
            side_values.append(value)
        crossings.append(np.column_stack(side_values))
    left, right = crossings
    rise, width = left[:, 2]-left[:, 0], right[:, 1]-left[:, 1]
    nonpositive = peak <= 0
    rise_missing, width_missing = ~np.isfinite(rise), ~np.isfinite(width)
    pre_rise = (~nonpositive & rise_missing &
        ((~np.isfinite(left[:, 0]) & (y[:, 0] > .1*peak)) |
         (~np.isfinite(left[:, 2]) & (y[:, 0] > .9*peak))))
    pre_width = ~nonpositive & width_missing & ~np.isfinite(left[:, 1]) & (y[:, 0] > .5*peak)
    post_width = ~nonpositive & width_missing & ~np.isfinite(right[:, 1]) & (y[:, -1] > .5*peak)
    return dict(peak_voltage_V=peak, rise10_90_ns=rise, FWHM_ns=width,
        left_crossing10_ns=left[:, 0], left_crossing50_ns=left[:, 1], left_crossing90_ns=left[:, 2],
        right_crossing50_ns=right[:, 1], rise_undefined=rise_missing, FWHM_undefined=width_missing,
        temporal_nonpositive_peak=nonpositive, rise_prepeak_censored=pre_rise,
        rise_other_undefined=rise_missing & ~nonpositive & ~pre_rise,
        FWHM_prepeak_censored=pre_width, FWHM_postpeak_censored=post_width,
        FWHM_other_undefined=width_missing & ~nonpositive & ~pre_width & ~post_width,
        selected_peak_below_global_crop_max=peak < y.max(1))


def scalar_features018(wave, baseline, peak_indices):
    """Recompute all original nine scalars from losslessly decoded Y and actual prefix."""
    y, b = np.asarray(wave, np.float64), np.asarray(baseline, np.float64)
    require(y.shape == (len(b), 256) and b.shape == (len(y), 100), '018 hard/baseline shapes')
    m = morphology018(y)
    q, tail = y.sum(1), y[:, 46:].sum(1)
    psd = np.full(len(y), np.nan)
    np.divide(tail, q, out=psd, where=q != 0)
    rms = np.sqrt(np.mean((b-b.mean(1, keepdims=True))**2, 1))
    return np.column_stack((q, tail, psd, b.mean(1), rms, m['peak_voltage_V'],
                            m['rise10_90_ns'], m['FWHM_ns'], peak_indices)), m


def observable018(wave, scalar9, morphology, scales):
    """Injective encoding of the stated observable tuple, never an imputed true rise.

    The bound channel is CONDITIONAL: if the unseen extension has a nearest-left
    10% crossing consistent with this definition, rise > t90_crop. Its existence
    is unobserved and is not asserted by this channel or by a nonzero mask.
    """
    y, s, m = np.asarray(wave, np.float64), np.asarray(scalar9, np.float64), morphology
    require(y.shape == (len(s), 256) and s.shape[1] == 9, '018 observable row schema')
    measurable, pre = np.isfinite(m['rise10_90_ns']), m['rise_prepeak_censored']
    state = np.where(measurable, 0, np.where(pre, 1, 2)).astype(np.uint8)
    timing = np.column_stack([m[k] for k in TIME_FIELDS018[:-1]] + [np.full(len(y), np.nan)])
    # Finite crossing positions are stored relative to the selected peak.
    timing[:, 1:5] -= 16.
    bound = (pre & (m['peak_voltage_V'] > 0) & ~np.isfinite(m['left_crossing10_ns']) &
             (y[:, 0] > .1*m['peak_voltage_V']) & np.isfinite(m['left_crossing90_ns']))
    timing[bound, 5] = m['left_crossing90_ns'][bound]
    defined = np.isfinite(timing)
    z = np.column_stack((np.eye(3)[state], defined.astype(np.float64),
        np.where(defined, timing, 0.), s[:, 0]/scales['Q_Vns'], s[:, 5]/scales['peak_voltage_V'],
        np.where(np.isfinite(s[:, 7]), s[:, 7]/scales['FWHM_ns'], 0.), np.isfinite(s[:, 7])))
    require(z.shape == (len(y), 19) and np.isfinite(z).all(), '018 nonfinite observable embedding')
    return dict(embedding=z, state=state, timing_values_ns=timing, timing_defined=defined,
                conditional_bound_defined=bound)


def covariance8_cell018(S, T, group, weights, *, moments):
    """Exactly the 017 covariance rule applied only to its other eight scalars."""
    sm, tm = moments
    support = (sm['feature_support'][:, KEEP8] & tm['feature_support'][:, KEEP8]).all(1)
    support &= (sm['pair_support'][:, KEEP8][:, :, KEEP8] &
                tm['pair_support'][:, KEEP8][:, :, KEEP8]).all((1, 2))
    cell = support_cell017(group, weights, support)
    if cell['status'] == 'pending':
        return cell
    ix = cell.pop('indices')
    a = weights[ix]/weights[ix].sum()
    scale = np.maximum(np.sqrt(tm['combined']['sample_variance'][ix][:, KEEP8]), FLOORS017[KEEP8])
    denominator = scale[:, :, None]*scale[:, None, :]
    sn = sm['combined']['sample_covariance'][ix][:, KEEP8][:, :, KEEP8]/denominator
    tn = tm['combined']['sample_covariance'][ix][:, KEEP8][:, :, KEEP8]/denominator
    cs, ct = np.einsum('p,pij->ij', a, sn), np.einsum('p,pij->ij', a, tn)
    c0, c1 = [np.einsum('p,pij->ij', a,
        item['sample_covariance'][ix][:, KEEP8][:, :, KEEP8]/denominator) for item in tm['native']]
    reference, split_reference = float(np.linalg.norm(ct)), float(np.linalg.norm(.5*(c0+c1)))
    if reference == 0 or split_reference == 0:
        return dict(cell, status='pending', reasons=['zero_covariance_reference'])
    error, resolution = float(np.linalg.norm(cs-ct)/reference), float(np.linalg.norm(c0-c1)/split_reference)
    status = 'pending' if resolution > .075 else 'pass' if error <= .15 else 'fail'
    return dict(cell, status=status, reasons=['MC_unresolved'] if status == 'pending' else [],
        group_relative_Frobenius=error, teacher32_split_resolution=resolution,
        student_normalized_conditional_covariance=cs, teacher_normalized_conditional_covariance=ct,
        supported_parent_indices=ix, teacher_scale=scale, no_between_parent_scatter=True,
        features=[FEATURES017[i] for i in KEEP8])


def pair_distances018(A, B, *, parent_block=8):
    """Direct float64 Euclidean distances; no dot-product cancellation, no RNG."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    require(A.ndim == B.ndim == 3 and len(A) == len(B) and A.shape[-1] == B.shape[-1]
            and np.isfinite(A).all() and np.isfinite(B).all(), '018 finite paired distance inputs')
    dimension = A.shape[-1]
    DISTANCE_COUNTS018[dimension] = DISTANCE_COUNTS018.get(dimension, 0) + len(A)*A.shape[1]*B.shape[1]
    result = np.empty((len(A), A.shape[1], B.shape[1]), np.float64)
    for start in range(0, len(A), parent_block):
        delta = A[start:start+parent_block, :, None, :] - B[start:start+parent_block, None, :, :]
        np.square(delta, out=delta)
        result[start:start+parent_block] = np.sqrt(delta.sum(-1))
    return result


def self_mean018(A):
    distances = pair_distances018(A, A)
    return distances.sum((1, 2))/(A.shape[1]*(A.shape[1]-1))


def energy_discrepancy018(S, T, *, student_self=None, teacher_self=None):
    """Unbiased half-energy distance. Negative finite estimates are retained."""
    require(S.shape == T.shape and S.ndim == 3 and S.shape[1] >= 2, '018 equal-draw paired distributions')
    ss = self_mean018(S) if student_self is None else student_self
    tt = self_mean018(T) if teacher_self is None else teacher_self
    return pair_distances018(S, T).mean((1, 2)) - .5*(ss+tt)


def observable_cell018(discrepancy, teacher_null, group, weights):
    require(np.isfinite(discrepancy).all() and np.isfinite(teacher_null).all(), '018 all-parent finite observable score')
    cell = support_cell017(group, weights, np.ones(len(discrepancy), bool))
    if cell['status'] == 'pending':
        return cell
    ix = cell.pop('indices')
    score, null = weighted017(discrepancy[ix], weights[ix]), weighted017(teacher_null[ix], weights[ix])
    half_width = Z95*score['SE']
    null_upper = abs(null['mean'])+Z95*null['SE']
    upper, lower = score['mean']+half_width, score['mean']-half_width
    if null_upper > OBSERVABLE_MC_HALF_WIDTH018:
        status, reasons = 'pending', ['teacher32_null_unresolved']
    elif lower > OBSERVABLE_TOLERANCE018:
        status, reasons = 'fail', []
    elif half_width > OBSERVABLE_MC_HALF_WIDTH018:
        status, reasons = 'pending', ['parent_cluster_MC_half_width_above005']
    elif upper <= OBSERVABLE_TOLERANCE018:
        status, reasons = 'pass', []
    else:
        status, reasons = 'pending', ['interval_intersects_tolerance']
    return dict(cell, status=status, reasons=reasons, score=score, teacher32_null=null,
        upper95=upper, lower95=lower, half_width95=half_width, teacher32_abs_upper95=null_upper,
        tolerance=OBSERVABLE_TOLERANCE018, MC_half_width_limit=OBSERVABLE_MC_HALF_WIDTH018,
        supported_parent_indices=ix, original_draw_slots_per_parent=64,
        thresholds_are_prospective_new_score_units=True)


def stage_gate018(trials):
    require([t['trial'] for t in trials] == ['T1_paired_seed0', 'T1_paired_seed1'], '018 exact two paired trials')
    expected = set()
    for group in ('all_heldout', 'fit_Q_tertile0_heldout', 'fit_Q_tertile1_heldout', 'fit_Q_tertile2_heldout'):
        for weight in ('nominal', 'low', 'high'):
            expected.add(('analytic_charge_variance', group, weight, None, None, None))
            for h in ('fast', 'slow', 'common_level'):
                for metric in ('hard_wave_mean', 'hard_wave_variance', 'conditional_cov8', 'conditional_energy',
                               'time_peak_index', 'time_FWHM', 'observable_joint_energy'):
                    expected.add((metric, group, weight, h, None, None))
                for feature in ('Q', 'tail', 'peak_voltage', 'peak_index', 'FWHM'):
                    for level in (.5, .9):
                        expected.add(('PIT_coverage', group, weight, h, feature, level))
                for feature in REQUIRED_MISSING017:
                    expected.add(('missingness_or_censor', group, weight, h, feature, None))
    require(len(expected) == 1272, '018 fixed primary signature count')
    nonpass = []
    for trial in trials:
        signature = [(c['metric'], c['group'], c['weights'], c.get('hypothesis'), c.get('feature'),
                      c.get('level')) for c in trial['cells']]
        require(len(signature) == len(expected) and set(signature) == expected, '018 missing/duplicate primary signature')
        require(all(c['status'] in ('pass', 'pending', 'fail') for c in trial['cells']), '018 primary status')
        nonpass += [dict(trial=trial['trial'], cell=i, metric=c['metric'], group=c['group'], weights=c['weights'],
                        hypothesis=c.get('hypothesis'), status=c['status'], reasons=c['reasons'])
                    for i, c in enumerate(trial['cells']) if c['status'] != 'pass']
    return dict(status='pass' if not nonpass else 'closed', T1_paired_018_qualified=not nonpass,
                required_cell_count=2544, nonpass=nonpass, pending_is_pass=False,
                stage2_training_authorized=False, scientific_acceptance=False,
                scope='saved fixed-A/B/005 conditional charge experiment; observable window only')
