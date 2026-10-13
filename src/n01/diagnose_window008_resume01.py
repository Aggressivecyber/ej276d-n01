"""Frozen resume01 endpoint diagnostics from USER_PROPOSAL section nine.

No optimizer, classifier, population fit or outcome-dependent selection lives in
this module. The public scientific entry point verifies independent source and
policy manifests and endpoint completion before reading DEV numerical values.
Small pure helpers and ``run_diagnostics_arrays`` support artificial CPU checks.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from .data import sha256
from .evaluate_window008 import (
    ROOT, _array, _clean, _json_sha, _load_dev, _read_json, _resolve,
    _validate_split, _verify_frozen, array_sha, output_bins, save_json,
    save_npz, scalar_summary, source_bins,
)
from .observation008 import observe_generated_carrier
from .response008 import ARMS, HYPOTHESES, PairedStreams
from .train import utc

POLICY_SCHEMA = 'n01/window008-resume01-diagnostics-policy/v1'
MANIFEST_SCHEMA = 'n01/window008-resume01-diagnostics-source-manifest/v1'
VARIANCE_FLOOR = (1. / 16383) ** 2 / 12.
BASELINE_BANDS = ((1, 4), (4, 9), (9, 17), (17, 33), (33, 51))
SCALAR_FIELDS = (
    'Q', 'tail', 'PSD', 'PSD_raw', 'qsig', 'qbar', 'time_scale',
    'Q_signal', 'Q_noise', 'Q_ADC_delta', 'tail_signal', 'tail_noise',
    'tail_ADC_delta', 'Q_signal_crop_migration', 'peak_voltage_V',
    'rise10_90_ns', 'FWHM_ns',
)
DECOMPOSITION_FIELDS = ('Q_signal', 'Q_noise', 'Q_ADC_delta',
                        'tail_signal', 'tail_noise', 'tail_ADC_delta')
SEEDS = dict(repeat_selection=2026101399, repeat_response=2026101400,
             repeat_noise005=2026101401, repeat_extension=2026101402,
             intervention_response=2026101403, intervention_noise005=2026101404,
             intervention_extension=2026101405)


def fixed_policy(value):
    """Reject changes to predeclared selection, budget, missingness and streams."""
    policy = _read_json(value)
    if (policy.get('schema') != POLICY_SCHEMA or policy.get('seeds') != SEEDS
            or policy.get('arms') != list(ARMS) or policy.get('hypotheses') != list(HYPOTHESES)
            or policy.get('endpoint_run_id') != 'window008_resume01_20261013'
            or policy.get('scientific_status') != 'response_match_not_established'
            or policy.get('scientific_pass_threshold') is not None
            or any(policy.get(k) != 0 for k in ('G_updates', 'D_updates', 'noise_updates', 'population_updates', 'C2ST_updates'))):
        raise ValueError('The preregistered diagnostic identity/seeds/zero-update contract changed')
    r, b, c, p = (policy[k] for k in ('repeat', 'shape_intervention', 'Q_trend', 'population'))
    if ((r['available_parents'], r['selected_parents'], r['draws_per_parent'], r['sample_ddof']) != (512, 128, 64, 1)
            or (b['parents'], b['draws_per_parent'], tuple(b['variants'])) != (32, 8, ('original', 'donor_same_Q'))
            or (c['parents'], c['draws_per_parent'], c['SOURCE_TRAIN_Q8_edge_indices']) != (32, 8, [2, 4, 6])
            or p['draws_per_parent'] != 4 or p['variance_ddof'] != 0 or p['response_forward_rows'] != 0
            or p['REAL_fit_or_reweight'] is not False or p['minimum_parents_per_bin'] != 32):
        raise ValueError('The preregistered diagnostic population/variance contract changed')
    resources = policy['resources']
    if any(resources[k] != v for k, v in dict(chunk_rows=128, cpu_threads=12, GPU_limit_GiB=10,
                                               process_tree_RSS_limit_GiB=8, output_limit_GiB=3).items()):
        raise ValueError('Diagnostic resource policy changed')
    if (policy['CPU_threads'] != 12 or policy['resource_limits'] != dict(GPU_GiB=10, job_tree_RSS_GiB=8,
            wall_seconds=21600, output_MiB=3072, poll_seconds=1, TERM_grace_seconds=20)
            or policy['preflight'] != dict(minimum_free_disk_GiB=8, minimum_available_RAM_GiB=10)):
        raise ValueError('Diagnostic supervisor/preflight contract changed')
    counts = dict(repeat_signal_rows_per_arm=8192, shape_signal_rows_per_arm=512,
                  Q_trend_signal_rows_per_arm=768, primary_signal_forward_rows=28416,
                  primary_observation_rows=85248, primary_noise005_forward_rows=8704,
                  identity_order_replay_signal_rows=1536, identity_order_replay_observation_rows=4608,
                  identity_order_replay_noise005_forward_rows=512,
                  total_signal_forward_rows_including_replay=29952,
                  total_observation_rows_including_replay=89856)
    if policy['counts'] != counts:
        raise ValueError('The preregistered actual-counter budget changed')
    if (b['canonical_charge_window'] != [110, 366] or b['baseline_center_samples'] != [0, 100]
            or b['SOURCE_peak_anchor'] != 126 or policy['time_morphology']['selected_output_peak_index'] != 16
            or policy['time_morphology']['threshold_fractions'] != [.1, .5, .9]
            or policy['baseline_statistics']['FFT_bins'] != [list(v) for v in BASELINE_BANDS]
            or policy['baseline_statistics']['ACF_lags'] != [0, 99]):
        raise ValueError('The preregistered observation/morphology contract changed')
    return policy


def select_repeat_parents(source_ids, policy):
    """Selection consumes identities only, in the already-bound original order."""
    ids = np.asarray(source_ids)
    n, selected = policy['repeat']['available_parents'], policy['repeat']['selected_parents']
    if ids.ndim != 1 or len(ids) != n or len(np.unique(ids)) != n or ids.dtype.kind == 'O':
        raise ValueError('Exactly512 unique non-object DEV identities are required')
    return np.random.default_rng(policy['seeds']['repeat_selection']).permutation(n)[:selected].astype(np.int64)


def shape_interventions(carriers384, source_ids, selected, edges9, policy):
    """Frozen donor selection and three Q nodes; never looks at generated/REAL data."""
    x = np.asarray(carriers384, np.float64)
    ids, selected, edges = np.asarray(source_ids), np.asarray(selected, np.int64), np.asarray(edges9, np.float64)
    if (x.shape != (len(ids), 384) or not np.isfinite(x).all() or len(np.unique(ids)) != len(ids)
            or selected.ndim != 1 or len(selected) < policy['shape_intervention']['parents']
            or (selected < 0).any() or (selected >= len(ids)).any() or len(np.unique(selected)) != len(selected)):
        raise ValueError('Invalid pristine SOURCE carrier/identity selection')
    if edges.shape != (9,) or not np.isfinite(edges).all() or (np.diff(edges) < 0).any():
        raise ValueError('Frozen SOURCE TRAIN edges must be finite nondecreasing9-vector')
    centered = x - x[:, :100].mean(1, keepdims=True)
    q = centered[:, 110:366].sum(1)
    peaks = centered[:, 112:144].argmax(1) + 112
    if not np.isfinite(q).all() or (q <= 1e-8).any() or (peaks != 126).any():
        raise ValueError('All SOURCE donors require positive canonical Q and anchor126; no row removal')
    chosen = selected[:policy['shape_intervention']['parents']]
    donors = np.empty(len(chosen), np.int64)
    for position, parent in enumerate(chosen):
        gap = np.abs(np.log(q / q[parent])); gap[parent] = np.inf
        # Original IDs are the stable tie breaker, independent of input storage.
        donors[position] = np.lexsort((ids, gap))[0]
    original = centered[chosen].copy()
    donor_same_q = centered[donors] * (q[chosen] / q[donors])[:, None]
    nodes = edges[policy['Q_trend']['SOURCE_TRAIN_Q8_edge_indices']]
    if (nodes <= 1e-8).any():
        raise ValueError('TRAIN Q nodes must be positive without clipping')
    trend = np.asarray([original * (node / q[chosen])[:, None] for node in nodes])
    atol = float(policy['engineering']['SOURCE_charge_preservation_atol_V_ns'])
    if (not np.allclose(donor_same_q[:, 110:366].sum(1), q[chosen], rtol=0, atol=atol)
            or not np.allclose(trend[:, :, 110:366].sum(2), nodes[:, None], rtol=0, atol=atol)
            or (donor_same_q[:, 112:144].argmax(1) + 112 != 126).any()
            or (trend[:, :, 112:144].argmax(2) + 112 != 126).any()):
        raise RuntimeError('SOURCE shape/Q intervention violated the frozen canonical contract')
    unit_original = original / q[chosen, None]
    unit_donor = centered[donors] / q[donors, None]
    return dict(selected_rows=chosen, selected_ids=ids[chosen], donor_rows=donors, donor_ids=ids[donors],
                all_SOURCE_Q=q, all_SOURCE_ids=ids, SOURCE_Q=q[chosen], donor_SOURCE_Q=q[donors],
                abs_log_charge_gap=np.abs(np.log(q[donors] / q[chosen])),
                scale_donor=q[chosen] / q[donors], original_carrier384=original,
                donor_original_centered384=centered[donors], donor_same_Q_carrier384=donor_same_q,
                unit_shape_l2_difference=np.sqrt(np.square(unit_donor-unit_original).sum(1)),
                unit_shape_absmax_difference=np.abs(unit_donor-unit_original).max(1),
                Q_nodes=nodes, Q_node_scale=nodes[:, None] / q[chosen][None],
                trend_carriers384=trend)


def _crossing(y, fraction, side):
    peak = float(y[16])
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    threshold = fraction * peak
    if side == 'left':
        pairs = ((j-1, j) for j in range(16, 0, -1))
    else:
        pairs = ((j, j+1) for j in range(16, len(y)-1))
    for lo, hi in pairs:
        a, b = float(y[lo]), float(y[hi])
        if (side == 'left' and a <= threshold <= b) or (side == 'right' and b <= threshold <= a):
            if a == b:
                return float(hi if side == 'left' else lo)
            return lo + (threshold-a) / (b-a)
    return np.nan


def temporal_readouts(waveform256, q_floor):
    """Descriptive nearest crossings, anchored to the selected output peak16."""
    y = np.asarray(waveform256, np.float64)
    if y.ndim != 2 or y.shape[1] != 256 or not np.isfinite(y).all():
        raise ValueError('Time morphology requires finite Bx256 without row removal')
    left = np.asarray([[_crossing(row, f, 'left') for f in (.1, .5, .9)] for row in y])
    right = np.asarray([[_crossing(row, f, 'right') for f in (.1, .5, .9)] for row in y])
    rise = left[:, 2] - left[:, 0]; width = right[:, 1] - left[:, 1]
    return dict(peak_voltage_V=y[:, 16], selected_output_peak_index=np.full(len(y), 16, np.int64),
                left_crossing10_ns=left[:, 0], left_crossing50_ns=left[:, 1], left_crossing90_ns=left[:, 2],
                right_crossing10_ns=right[:, 0], right_crossing50_ns=right[:, 1], right_crossing90_ns=right[:, 2],
                rise10_90_ns=rise, FWHM_ns=width, rise10_90_defined=np.isfinite(rise), FWHM_defined=np.isfinite(width),
                temporal_nonpositive_peak=y[:, 16] <= 0,
                normalized_waveform_defined=y.sum(1) > float(q_floor))


def baseline_readouts(baseline100):
    """Existing005 baseline convention extended explicitly to lag0..99."""
    x = np.asarray(baseline100, np.float64)
    if x.ndim != 2 or x.shape[1] != 100 or not np.isfinite(x).all():
        raise ValueError('Baseline diagnostics require original finite Bx100 ADC values')
    mean = x.mean(1); b = x - mean[:, None]
    raw_variance = np.square(b).mean(1); variance = np.maximum(raw_variance, VARIANCE_FLOOR)
    acf = np.asarray([(b[:, :100-lag]*b[:, lag:]).mean(1)/variance for lag in range(100)]).T
    power = np.abs(np.fft.rfft(b, axis=1)) ** 2
    fractions = np.asarray([power[:, lo:hi].sum(1) for lo, hi in BASELINE_BANDS]).T
    fractions /= np.maximum(power[:, 1:].sum(1, keepdims=True), 1e-30)
    return dict(baseline_mean_V=mean, baseline_RMS_V=np.sqrt(raw_variance),
                baseline_floor_RMS_V=np.sqrt(variance), baseline_ACF_lag0_99=acf,
                baseline_power_fraction5=fractions, baseline_variance_floor_used=raw_variance < VARIANCE_FLOOR)


def sample_statistics(value, ddof=1):
    """Parent x draw x feature moments; retain every row and explicit pair counts.

    Defined-subset means and pairwise jointly defined sample covariances are
    labelled by their counts. The pairwise covariance need not be PSD when
    missingness differs across features. Required Q/guarded-PSD are all finite.
    """
    x = np.asarray(value, np.float64)
    if x.ndim != 3 or x.shape[1] < 1 or ddof != 1 or np.isinf(x).any():
        raise ValueError('Sample covariance requires parent x draw x feature, ddof1, no infinities')
    valid = np.isfinite(x); v = valid.astype(np.float64); a = np.where(valid, x, 0.)
    counts = valid.sum(1)
    mean = np.full(counts.shape, np.nan)
    np.divide(a.sum(1), counts, out=mean, where=counts > 0)
    pair_count = (v.transpose(0, 2, 1) @ v).astype(np.int64)
    covariance = np.full(pair_count.shape, np.nan)
    # Center each jointly defined pair before multiplication. Subtracting E[XY]
    # and E[X]E[Y] would lose narrow response covariance at large mean charge.
    for i in range(x.shape[2]):
        for j in range(i, x.shape[2]):
            pair = valid[:, :, i] & valid[:, :, j]; count = pair_count[:, i, j]
            mi, mj = np.zeros(x.shape[0]), np.zeros(x.shape[0])
            np.divide(np.where(pair, x[:, :, i], 0.).sum(1), count, out=mi, where=count > 0)
            np.divide(np.where(pair, x[:, :, j], 0.).sum(1), count, out=mj, where=count > 0)
            product = np.where(pair, (x[:, :, i]-mi[:, None])*(x[:, :, j]-mj[:, None]), 0.)
            cov = np.full(x.shape[0], np.nan)
            np.divide(product.sum(1), count-ddof, out=cov, where=count > ddof)
            covariance[:, i, j] = covariance[:, j, i] = cov
    centered = np.where(valid, x-mean[:, None], 0.)
    variance = np.full(mean.shape, np.nan)
    np.divide(np.square(centered).sum(1), counts-ddof, out=variance, where=counts > ddof)
    for k in range(x.shape[2]):
        covariance[:, k, k] = variance[:, k]
    return dict(mean=mean, variance=variance, covariance=covariance,
                defined_draws=counts, covariance_defined_draws=pair_count,
                undefined_draws=x.shape[1]-counts, ddof=ddof)


def defined_moments(value):
    """Per-parent means/std/SE over explicitly counted finite draws."""
    x = np.asarray(value, np.float64)
    if x.ndim < 2 or np.isinf(x).any():
        raise ValueError('Defined moments require parent x draw x ... without infinity')
    valid = np.isfinite(x); count = valid.sum(1)
    mean = np.full(count.shape, np.nan)
    np.divide(np.where(valid, x, 0.).sum(1), count, out=mean, where=count > 0)
    variance = np.full(count.shape, np.nan)
    np.divide(np.where(valid, np.square(x-mean[:, None]), 0.).sum(1), count-1,
              out=variance, where=count > 1)
    std = np.sqrt(variance); se = np.full(count.shape, np.nan)
    np.divide(std, np.sqrt(count), out=se, where=count > 1)
    return dict(mean=mean, std=std, MonteCarlo_SE=se, defined_draws=count,
                undefined_draws=x.shape[1]-count)


def weighted_statistics(value, weights):
    """Positive-weight population moments (ddof0) with missing mass retained."""
    x, w = np.asarray(value, np.float64), np.asarray(weights, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if (x.ndim != 2 or w.shape != (len(x),) or not np.isfinite(w).all()
            or (w <= 0).any() or np.isinf(x).any()):
        raise ValueError('Weighted statistics require positive finite row weights and no infinities')
    valid = np.isfinite(x); mass = (valid*w[:, None]).sum(0)
    mean = np.full(x.shape[1], np.nan)
    np.divide((np.where(valid, x, 0.)*w[:, None]).sum(0), mass, out=mean, where=mass > 0)
    variance = np.full(x.shape[1], np.nan)
    np.divide((np.where(valid, np.square(x-mean[None]), 0.)*w[:, None]).sum(0), mass,
              out=variance, where=mass > 0)
    return dict(mean=mean, std=np.sqrt(variance), variance=variance, rows=len(x),
                defined_rows=valid.sum(0), undefined_rows=(~valid).sum(0),
                defined_weight=mass, undefined_weight=w.sum()-mass, total_weight=float(w.sum()),
                ddof=0, all_rows_retained=True)


def population_weights(source_parent_bins):
    bins = np.asarray(source_parent_bins)
    if bins.ndim != 1 or bins.dtype.kind not in 'iu' or not len(bins) or ((bins < 0) | (bins > 7)).any():
        raise ValueError('Fixed SOURCE bin must be an integer0..7 for every parent')
    z = (2.*bins-7.)/7.
    raw = np.asarray((np.ones(len(bins)), 1.+.5*z, 1.-.5*z))
    weights = raw*len(bins)/raw.sum(1, keepdims=True)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError('Preregistered parent weights are not positive')
    return weights


def exact_weighted_waveform(bin_mean, bin_std, bin_rows, bin_weights):
    """Exact mixture of within-bin ddof0 moments for weights constant in bin."""
    mean, std, rows, w = (np.asarray(v, np.float64) for v in (bin_mean, bin_std, bin_rows, bin_weights))
    if mean.shape != (8, 256) or std.shape != mean.shape or rows.shape != (8,) or w.shape != (8,):
        raise ValueError('Exact waveform mixture requires8x256 means/std and8 counts/weights')
    active = rows > 0
    if ((rows < 0).any() or (w <= 0).any() or not np.isfinite(rows).all() or not np.isfinite(w).all()
            or not active.any() or not np.isfinite(mean[active]).all()
            or not np.isfinite(std[active]).all() or (std[active] < 0).any()):
        raise ValueError('Nonfinite active SOURCE-bin waveform moments')
    mass = rows[active]*w[active]; total = mass.sum()
    result_mean = (mean[active]*mass[:, None]).sum(0)/total
    result_variance = ((np.square(std[active])+np.square(mean[active]-result_mean[None]))*mass[:, None]).sum(0)/total
    if not np.isfinite(result_variance).all() or (result_variance < 0).any():
        raise RuntimeError('Exact mixture produced invalid population variance')
    return dict(mean=result_mean, variance=result_variance, std=np.sqrt(result_variance),
                weighted_bin_rows=rows*w, total_row_weight=float(total), mathematical_exactness=True,
                source_bin_moments_ddof=0, result_ddof=0)


class ForwardCounters:
    """Count actual calls/rows after successful numerical generation."""
    def __init__(self):
        self.groups = {}

    def add(self, group, kind, rows, arm=None, hypothesis=None):
        entry = self.groups.setdefault(group, dict(signal_forward_rows=0, observation_rows=0,
                                                   noise005_forward_rows=0, signal_calls=0, observation_calls=0,
                                                   noise005_calls=0, per_arm={}, per_arm_hypothesis={}))
        entry[kind+'_rows'] += int(rows)
        entry[kind.replace('_forward', '')+'_calls'] += 1
        if kind == 'signal_forward':
            entry['per_arm'][arm] = entry['per_arm'].get(arm, 0)+int(rows)
        if kind == 'observation':
            key = arm+'_'+hypothesis
            entry['per_arm_hypothesis'][key] = entry['per_arm_hypothesis'].get(key, 0)+int(rows)

    def validate(self, policy):
        expected = dict(repeat=(24576, 73728, 8192), shape=(1536, 4608, 256),
                        Q_trend=(2304, 6912, 256), identity_order_replay=(1536, 4608, 512))
        for name, counts in expected.items():
            actual = self.groups.get(name, {})
            if tuple(actual.get(k) for k in ('signal_forward_rows', 'observation_rows', 'noise005_forward_rows')) != counts:
                raise RuntimeError('Actual forward/observation count differs: '+name)
            if actual.get('per_arm') != {a: counts[0]//3 for a in ARMS}:
                raise RuntimeError('Actual per-arm count differs: '+name)
            if actual.get('per_arm_hypothesis') != {a+'_'+h: counts[1]//9 for a in ARMS for h in HYPOTHESES}:
                raise RuntimeError('Actual per-arm/hypothesis count differs: '+name)
        primary = {k: sum(self.groups[n][k] for n in ('repeat', 'shape', 'Q_trend'))
                   for k in ('signal_forward_rows', 'observation_rows', 'noise005_forward_rows')}
        if (primary['signal_forward_rows'] != policy['counts']['primary_signal_forward_rows']
                or primary['observation_rows'] != policy['counts']['primary_observation_rows']
                or primary['noise005_forward_rows'] != policy['counts']['primary_noise005_forward_rows']):
            raise RuntimeError('Actual primary diagnostic total differs')
        return dict(primary=primary, groups=self.groups,
                    total_signal_forward_rows=sum(v['signal_forward_rows'] for v in self.groups.values()),
                    total_observation_rows=sum(v['observation_rows'] for v in self.groups.values()),
                    total_noise005_forward_rows=sum(v['noise005_forward_rows'] for v in self.groups.values()),
                    optimizer_updates=0, C2ST_updates=0, population_updates=0, noise_updates=0)


def _output_bound(destination, policy):
    size = sum(p.stat().st_size for p in Path(destination).rglob('*') if p.is_file())
    if size > policy['resources']['output_limit_GiB']*1024**3:
        raise RuntimeError('Preregistered diagnostic output limit exceeded')
    return size


def explicit_latent_bank(parents, draws, seeds, device='cpu', chunk_rows=128):
    """Generate once in fixed chunks; every scientific variant reuses these rows."""
    streams = PairedStreams(*seeds); initial = streams.state_dict()
    pieces, chunks = {}, []
    for start in range(0, parents*draws, chunk_rows):
        stop = min(start+chunk_rows, parents*draws)
        response, electronics, epsilon = streams.draw(stop-start, device)
        values = {**{'response_'+k: _array(v) for k, v in response.items()},
                  **{'noise005_'+k: _array(v) for k, v in electronics.items()}, 'extension_epsilon': _array(epsilon)}
        for name, value in values.items():
            pieces.setdefault(name, []).append(value)
        chunks.append(dict(start=start, stop=stop, latents_sha256={k: array_sha(v) for k, v in values.items()}))
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    result.update({'initial_'+k: _array(v) for k, v in initial.items()})
    result.update({'final_'+k: _array(v) for k, v in streams.state_dict().items()})
    return result, dict(parents=parents, draws_per_parent=draws, chunk_rows=chunk_rows, seeds=list(seeds),
                        chunks=chunks, values_sha256={k: array_sha(v) for k, v in result.items()})


def _latent_chunk(bank, rows, device):
    def tensor(name):
        return torch.as_tensor(np.ascontiguousarray(bank[name][rows]), device=device)
    response = {k[len('response_'):]: tensor(k) for k in bank if k.startswith('response_')}
    electronics = {k[len('noise005_'):]: tensor(k) for k in bank if k.startswith('noise005_')}
    return response, electronics, tensor('extension_epsilon')


def _electronics(extension, latents, epsilon, *, prefix_counter=None):
    extension.assert_frozen()
    shared = extension.sample_prefix(latents)
    if prefix_counter is not None:
        prefix_counter(len(shared['prefix_pre']))
    noises = {h: extension.extend_prefix(shared['prefix_pre'], shared['scale'], epsilon, h)['noise_pre'] for h in HYPOTHESES}
    if any(not torch.equal(noise[:, :100], shared['prefix_pre']) for noise in noises.values()):
        raise RuntimeError('Shared original005 prefix differs across noise hypotheses')
    return noises, shared['scale']


def _signal_fields(output):
    names = ('qsig', 'qbar', 'q_source', 'time_scale', 'canonical_charge', 'canonical_local_charge',
             'canonical_charge_identity_residual', 'query_outside_carrier', 'beyond_extension')
    return {k: _array(output[k]) for k in names}


@torch.no_grad()
def generate_family(group, carriers, ids, draws, bank, models, extension, statistics, policy, counters,
                    *, device='cpu', latent_parent_order=None):
    """Paired family generation; SOURCE IDs are saved metadata, never model input."""
    x = np.asarray(carriers, np.float64); ids = np.asarray(ids)
    if x.ndim != 3 or x.shape[2] != 384 or x.shape[1] != len(ids) or not np.isfinite(x).all():
        raise ValueError('Family input must be finite variant x parent x384')
    if tuple(models) != ARMS or not all(not p.requires_grad for m in models.values() for p in m.parameters()):
        raise ValueError('Exactly three frozen endpoint models are required')
    p, variants, n = len(ids), x.shape[0], len(ids)*draws
    parents = np.repeat(np.arange(p), draws); draw = np.tile(np.arange(draws), p)
    attached = np.arange(p) if latent_parent_order is None else np.asarray(latent_parent_order, np.int64)
    if attached.shape != (p,) or not np.array_equal(np.sort(attached), np.arange(p)):
        raise ValueError('Latents must remain attached to the original selected parent')
    latent_rows = attached[parents]*draws+draw
    pieces = {(v, a, h): [] for v in range(variants) for a in ARMS for h in HYPOTHESES}
    waves = {(v, a, h): np.empty((n, 256), np.float64) for v in range(variants) for a in ARMS for h in HYPOTHESES}
    baseline_all = np.empty((n, 100), np.float64); scale_all = np.empty(n, np.float64)
    noise_all = {h: np.empty((n, 384), np.float64) for h in HYPOTHESES}
    chunks = []
    engineering = dict(signal_prefix_absmax_V=0., canonical_charge_identity_absmax=0.,
                       Q_identity_absmax=0., tail_identity_absmax=0., baseline_shared=True)
    qfloor = float(statistics['loss']['q_floor']); tolerance = policy['engineering']
    for start in range(0, n, policy['resources']['chunk_rows']):
        stop = min(start+policy['resources']['chunk_rows'], n); pr = parents[start:stop]
        response, electronics, epsilon = _latent_chunk(bank, latent_rows[start:stop], device)
        noises, scale = _electronics(extension, electronics, epsilon,
            prefix_counter=lambda rows: counters.add(group, 'noise005_forward', rows))
        scale_all[start:stop] = _array(scale).reshape(-1)
        if not np.isfinite(scale_all[start:stop]).all() or (scale_all[start:stop] <= 0).any():
            raise RuntimeError('Nonfinite/nonpositive frozen005 S')
        for h in HYPOTHESES:
            noise_all[h][start:stop] = _array(noises[h])
        common_baseline = None
        record = dict(start=start, stop=stop, latent_row_sha256=array_sha(latent_rows[start:stop]), outputs={})
        for v in range(variants):
            source = torch.as_tensor(x[v, pr], dtype=torch.float64, device=device)
            for a, model in models.items():
                output = model.signal_forward(source, response); signal = output['signal_pre']
                counters.add(group, 'signal_forward', stop-start, arm=a)
                prefix_error = float(signal[:, :100].abs().max())
                canonical_error = float(output['canonical_charge_identity_residual'].abs().max())
                engineering['signal_prefix_absmax_V'] = max(engineering['signal_prefix_absmax_V'], prefix_error)
                engineering['canonical_charge_identity_absmax'] = max(engineering['canonical_charge_identity_absmax'], canonical_error)
                if (not bool(torch.isfinite(signal).all()) or prefix_error > tolerance['signal_prefix_max_V']
                        or canonical_error > tolerance['canonical_charge_identity_atol_V_ns']):
                    raise RuntimeError('Signal finite/prefix/canonical-charge engineering contract failed')
                common = _signal_fields(output)
                if any(not np.isfinite(value).all() for value in common.values()):
                    raise RuntimeError('Nonfinite required response scalar/counter')
                if (common['qsig'] <= 0).any() or (common['qbar'] <= 0).any() or (common['time_scale'] <= 0).any():
                    raise RuntimeError('Nonpositive required response charge/time-scale')
                for h in HYPOTHESES:
                    noise = noises[h]; pre = signal+noise
                    obs = observe_generated_carrier(pre, q_floor=qfloor, return_carrier=True)
                    counters.add(group, 'observation', stop-start, arm=a, hypothesis=h)
                    baseline = _array(obs['baseline100'])
                    if common_baseline is None:
                        common_baseline = baseline
                    if not np.array_equal(baseline, common_baseline):
                        engineering['baseline_shared'] = False
                        raise RuntimeError('Original100 ADC baseline changed across variants/arms/hypotheses')
                    y = _array(obs['waveform256'])
                    if not np.isfinite(y).all():
                        raise RuntimeError('Nonfinite observed waveform; no row deletion')
                    values = {k: _array(obs[k]) for k in ('Q', 'tail', 'PSD', 'PSD_raw', 'peak_index',
                                                         'global_peak_index', 'crop_start', 'crop_stop', 'baseline_mean_V')}
                    values.update({k: _array(val) for k, val in obs['flags'].items()}); values.update(common)
                    indices = obs['crop_indices']; delta = obs['adc_carrier']-pre
                    for name, term in (('signal', signal), ('noise', noise), ('ADC_delta', delta)):
                        crop = term.gather(1, indices)-term[:, :100].mean(1, keepdim=True)
                        values['Q_'+name] = _array(crop.sum(1)); values['tail_'+name] = _array(crop[:, 46:].sum(1))
                    values['Q_signal_crop_migration'] = values['Q_signal']-values['qsig']
                    values['Q_identity_residual'] = values['Q']-sum(values['Q_'+k] for k in ('signal', 'noise', 'ADC_delta'))
                    values['tail_identity_residual'] = values['tail']-sum(values['tail_'+k] for k in ('signal', 'noise', 'ADC_delta'))
                    for metric in ('Q', 'tail'):
                        error = float(np.abs(values[metric+'_identity_residual']).max())
                        engineering[metric+'_identity_absmax'] = max(engineering[metric+'_identity_absmax'], error)
                        if error > tolerance['observed_Q_tail_decomposition_atol_V_ns']:
                            raise RuntimeError('Same-observed-crop Q/tail identity failed')
                    if any(not np.isfinite(values[k]).all() for k in ('Q', 'tail', 'PSD')):
                        raise RuntimeError('Nonfinite Q/tail/guarded PSD')
                    if not np.array_equal(~np.isfinite(values['PSD_raw']), values['Q'] == 0):
                        raise RuntimeError('Raw PSD undefined mask must correspond exactly to Q==0')
                    values.update(temporal_readouts(y, qfloor))
                    waves[v, a, h][start:stop] = y; pieces[v, a, h].append(values)
                    record['outputs'][str(v)+'_'+a+'_'+h] = dict(waveform256_sha256=array_sha(y),
                        scalar_sha256={key: array_sha(value) for key, value in values.items()})
        baseline_all[start:stop] = common_baseline
        record['baseline100_ADC_sha256'] = array_sha(common_baseline); chunks.append(record)
    all_values = {key: {name: np.concatenate([piece[name] for piece in block]) for name in block[0]}
                  for key, block in pieces.items()}
    return dict(values=all_values, waveform256=waves, baseline100_ADC=baseline_all, S=scale_all,
                noise384=noise_all, parents=p, draws=draws, variants=variants,
                source_parent_ids=ids, source_parent_row=parents, draw=draw, chunks=chunks, engineering=engineering)


def _summary_fields(values):
    return {name: scalar_summary(value) for name, value in values.items()}


def _family_common(result, destination, label, policy):
    p, d = result['parents'], result['draws']
    baseline = baseline_readouts(result['baseline100_ADC'])
    values = dict(source_parent_ids=result['source_parent_ids'], source_parent_row=result['source_parent_row'],
                  draw=result['draw'], baseline100_ADC=result['baseline100_ADC'].reshape(p, d, 100),
                  S=result['S'].reshape(p, d))
    values.update({name: value.reshape((p, d)+value.shape[1:]) for name, value in baseline.items()})
    for h, noise in result['noise384'].items():
        n = noise.reshape(p, d, 384); centered = n-n[:, :, :100].mean(2, keepdims=True)
        values['noise384_'+h+'_conditional_mean'] = n.mean(1)
        values['noise384_'+h+'_conditional_std'] = n.std(1, ddof=1)
        values['noise384_'+h+'_RMS_V'] = np.sqrt(np.square(centered).mean(2))
        tail = noise[:, 100:]; tail = tail-tail.mean(1, keepdims=True)
        values['noise_tail284_'+h+'_centered_RMS_V'] = np.sqrt(np.square(tail).mean(1)).reshape(p, d)
    save_npz(destination/(label+'_COMMON.npz'), **values)
    _output_bound(destination, policy)
    return dict(original_baseline100=_summary_fields(baseline), S=scalar_summary(result['S']),
                original100_ACF_lags=np.arange(100), original100_ACF_mean_lag0_99=baseline['baseline_ACF_lag0_99'].mean(0),
                original100_ACF_sample_std_lag0_99=baseline['baseline_ACF_lag0_99'].std(0, ddof=1),
                original100_power_fraction5_mean=baseline['baseline_power_fraction5'].mean(0),
                electronics_hypotheses={h: dict(
                    RMS384_V=scalar_summary(values['noise384_'+h+'_RMS_V']),
                    centered_RMS_tail284_V=scalar_summary(values['noise_tail284_'+h+'_centered_RMS_V'])) for h in HYPOTHESES},
                baseline_shared_exactly=result['engineering']['baseline_shared'])


def save_repeat(result, destination, policy):
    """Save every64-draw row and full conditional waveform sample covariance."""
    p, d = result['parents'], result['draws']; summary = {}
    for a in ARMS:
        summary[a] = {}
        for h in HYPOTHESES:
            fields = result['values'][0, a, h]
            raw = {key: value.reshape((p, d)+value.shape[1:]) for key, value in fields.items()}
            y = result['waveform256'][0, a, h].reshape(p, d, 256)
            mean = y.mean(1); centered = y-mean[:, None]
            covariance = centered.transpose(0, 2, 1)@centered/(d-1)
            scalar = np.stack([raw[key] for key in SCALAR_FIELDS], axis=2)
            stats = sample_statistics(scalar)
            q = raw['Q']; normalized = np.full_like(y, np.nan)
            np.divide(y, q[:, :, None], out=normalized, where=q[:, :, None] > float(policy['_q_floor']))
            normalized_stats = defined_moments(normalized)
            component_indices = [SCALAR_FIELDS.index(k) for k in DECOMPOSITION_FIELDS]
            decomposition = stats['covariance'][:, component_indices][:, :, component_indices]
            values = dict(source_parent_ids=result['source_parent_ids'], draw=np.arange(d),
                          waveform256=y, waveform256_mean=mean, waveform256_covariance=covariance,
                          waveform256_sample_std=np.sqrt(covariance.diagonal(axis1=1, axis2=2)),
                          waveform256_covariance_mean_over_parents=covariance.mean(0),
                          waveform256_covariance_std_over_parents=covariance.std(0, ddof=1),
                          scalar_fields=np.asarray(SCALAR_FIELDS), scalar_mean=stats['mean'],
                          scalar_sample_covariance=stats['covariance'],
                          scalar_defined_draws=stats['defined_draws'],
                          scalar_covariance_defined_draws=stats['covariance_defined_draws'],
                          scalar_undefined_draws=stats['undefined_draws'],
                          Var_Q_given_x=stats['covariance'][:, 0, 0], Var_PSD_given_x=stats['covariance'][:, 2, 2],
                          Cov_Q_PSD_given_x=stats['covariance'][:, 0, 2],
                          decomposition_fields=np.asarray(DECOMPOSITION_FIELDS), decomposition_sample_covariance=decomposition,
                          decomposition_variance_Q_sum=decomposition[:, :3, :3].sum((1, 2)),
                          decomposition_variance_tail_sum=decomposition[:, 3:, 3:].sum((1, 2)),
                          decomposition_variance_Q_identity_residual=stats['covariance'][:, 0, 0]-decomposition[:, :3, :3].sum((1, 2)),
                          decomposition_variance_tail_identity_residual=stats['covariance'][:, 1, 1]-decomposition[:, 3:, 3:].sum((1, 2)),
                          normalized_waveform_defined_draws=normalized_stats['defined_draws'],
                          normalized_waveform_mean=normalized_stats['mean'], normalized_waveform_sample_std=normalized_stats['std'],
                          normalized_waveform_MonteCarlo_SE=normalized_stats['MonteCarlo_SE'], sample_ddof=np.asarray(1), **raw)
            save_npz(destination/('REPEAT_'+a+'_'+h+'.npz'), **values)
            summary[a][h] = dict(all_draws_saved=True, parents=p, draws_per_parent=d,
                population=_summary_fields(fields), conditional_variance_Q=scalar_summary(values['Var_Q_given_x']),
                conditional_variance_PSD=scalar_summary(values['Var_PSD_given_x']),
                conditional_covariance_Q_PSD=scalar_summary(values['Cov_Q_PSD_given_x']),
                per_parent_raw_PSD_defined=stats['defined_draws'][:, SCALAR_FIELDS.index('PSD_raw')],
                normalized_defined_draws_per_parent=raw['normalized_waveform_defined'].sum(1),
                scalar_covariance_rule='pairwise jointly defined draws with saved covariance counts; sample ddof1',
                waveform_covariance_shape=list(covariance.shape), waveform_covariance_ddof=1,
                time_undefined=dict(rise10_90=int((~raw['rise10_90_defined']).sum()), FWHM=int((~raw['FWHM_defined']).sum())),
                interpretation='Finite model conditional samples under three frozen electronics assumptions; physical conditional response not uniquely identified.')
            _output_bound(destination, policy)
    return summary


def _paired_block(first, second, y_first, y_second, parents, draws, q_floor):
    """Store per-draw paired deltas and explicit definition counts for each metric."""
    result, report = {}, {}
    for field in SCALAR_FIELDS:
        x = np.asarray(first[field]).reshape(parents, draws)
        y = np.asarray(second[field]).reshape(parents, draws)
        delta = y-x; stats = defined_moments(delta)
        result['delta_'+field] = delta
        result.update({'delta_'+field+'_'+key: value for key, value in stats.items()})
        report[field] = dict(mean_delta_summary=scalar_summary(stats['mean']),
                             MonteCarlo_SE_summary=scalar_summary(stats['MonteCarlo_SE']),
                             defined_draws_per_parent=stats['defined_draws'])
    first_wave, second_wave = y_first.reshape(parents, draws, 256), y_second.reshape(parents, draws, 256)
    wave_stats = defined_moments(second_wave-first_wave)
    result.update({'delta_waveform_'+key: value for key, value in wave_stats.items()})
    norm_first, norm_second = np.full_like(first_wave, np.nan), np.full_like(second_wave, np.nan)
    q_first, q_second = np.asarray(first['Q']).reshape(parents, draws), np.asarray(second['Q']).reshape(parents, draws)
    np.divide(first_wave, q_first[..., None], out=norm_first, where=q_first[..., None] > q_floor)
    np.divide(second_wave, q_second[..., None], out=norm_second, where=q_second[..., None] > q_floor)
    norm_stats = defined_moments(norm_second-norm_first)
    result.update({'delta_normalized_waveform_'+key: value for key, value in norm_stats.items()})
    report['normalized_waveform'] = dict(jointly_defined_draws_per_parent=norm_stats['defined_draws'][:, 0],
                                        missing_rule='paired normalized rows require both observed Q > frozen TRAIN q_floor')
    return result, report


def save_interventions(result, destination, policy, *, label, variant_names):
    p, d = result['parents'], result['draws']; summary = {}
    for a in ARMS:
        summary[a] = {}
        for h in HYPOTHESES:
            values = dict(source_parent_ids=result['source_parent_ids'], draw=np.arange(d), variant_names=np.asarray(variant_names))
            variant_reports, pairs = {}, {}
            for v, name in enumerate(variant_names):
                fields = result['values'][v, a, h]; y = result['waveform256'][v, a, h].reshape(p, d, 256)
                values.update({name+'_'+key: value.reshape((p, d)+value.shape[1:]) for key, value in fields.items()})
                values[name+'_waveform256'] = y
                values[name+'_waveform_mean'] = y.mean(1); values[name+'_waveform_sample_std'] = y.std(1, ddof=1)
                for key in SCALAR_FIELDS:
                    stats = defined_moments(fields[key].reshape(p, d))
                    values.update({name+'_'+key+'_'+kind: stat for kind, stat in stats.items()})
                variant_reports[name] = _summary_fields(fields)
            comparisons = [(0, 1)] if len(variant_names) == 2 else [(0, 1), (1, 2)]
            for first, second in comparisons:
                tag = variant_names[second]+'_minus_'+variant_names[first]
                paired, report = _paired_block(result['values'][first, a, h], result['values'][second, a, h],
                    result['waveform256'][first, a, h], result['waveform256'][second, a, h], p, d, policy['_q_floor'])
                values.update({tag+'_'+key: value for key, value in paired.items()}); pairs[tag] = report
                if label == 'Q_TREND':
                    qdelta = paired['delta_qbar']
                    values[tag+'_qbar_nonnegative_per_draw'] = qdelta >= 0
                    pairs[tag]['qbar_structure'] = dict(negative_draws=int((qdelta < 0).sum()),
                        nonnegative_draws=int((qdelta >= 0).sum()), minimum_signed_delta=float(qdelta.min()),
                        scientific_pass_threshold=None, separate_from_noisy_observed_Q=True)
            save_npz(destination/(label+'_'+a+'_'+h+'.npz'), **values)
            summary[a][h] = dict(variants=variant_reports, paired_comparisons=pairs,
                                 draws_per_parent=d, all_parent_draw_rows_saved=True,
                                 conditional_expectation_estimate='finite8-draw paired MonteCarlo estimate, not exact expectation')
            _output_bound(destination, policy)
    return summary


def compare_identity_order(reference, replay, policy, *, inverse_order=None):
    """Numeric replay with identical explicit latents and preserved undefined masks."""
    p, d = reference['parents'], reference['draws']
    order = np.arange(p) if inverse_order is None else np.asarray(inverse_order, np.int64)
    tolerances = policy['identity_order_replay']['tolerance']; results = {}
    for a in ARMS:
        for h in HYPOTHESES:
            expected = reference['waveform256'][0, a, h].reshape(p, d, 256)
            actual = replay['waveform256'][0, a, h].reshape(p, d, 256)[order]
            np.testing.assert_allclose(actual, expected, rtol=tolerances['rtol'], atol=tolerances['waveform_atol_V'])
            fields = reference['values'][0, a, h]; replay_fields = replay['values'][0, a, h]
            details = {}
            for key, before in fields.items():
                after = replay_fields[key].reshape((p, d)+replay_fields[key].shape[1:])[order].reshape(before.shape)
                if before.dtype.kind == 'b' or before.dtype.kind in 'iu':
                    if not np.array_equal(before, after):
                        raise RuntimeError('Identity/order replay changed discrete counter or flag: '+key)
                    details[key] = dict(exact=True)
                else:
                    if not np.array_equal(np.isfinite(before), np.isfinite(after)):
                        raise RuntimeError('Identity/order replay changed an undefined mask: '+key)
                    valid = np.isfinite(before)
                    np.testing.assert_allclose(after[valid], before[valid], rtol=tolerances['rtol'], atol=tolerances['scalar_atol'])
                    details[key] = dict(max_abs_error=float(np.abs(after[valid]-before[valid]).max()) if valid.any() else None,
                                        defined_rows=int(valid.sum()), undefined_rows=int((~valid).sum()))
            results[a+'_'+h] = dict(waveform256_max_abs_error=float(np.abs(actual-expected).max()),
                                    waveform256_bitwise_equal=np.array_equal(actual, expected), scalars=details)
    return dict(status='passed', numerical_response_only=True, tolerances=tolerances, results=results,
                metadata_ID_not_model_input=True, population_similarity_not_an_input_dependence_test=True)


def population_sensitivity(observables, curves, statistics, policy):
    """Four-draw population sensitivity from saved outputs, with zero forward.

    Weighted global wave moments are mathematically exact from fixed SOURCE-bin
    moments because all rows in a SOURCE bin share the same parent weight.
    Output-Q-conditional wave moments require unavailable joint-bin statistics;
    they are explicitly absent rather than inferred from marginal moments.
    """
    bins = np.asarray(observables['source_bins']); parent_ids = np.asarray(observables['source_parent_ids'])
    parent_rows = np.asarray(observables['source_parent_row']); draws = np.asarray(observables['draw'])
    n, d = len(parent_ids), policy['population']['draws_per_parent']
    if (bins.shape != (n,) or not np.array_equal(parent_rows, np.repeat(np.arange(n), d))
            or not np.array_equal(draws, np.tile(np.arange(d), n)) or len(np.unique(parent_ids)) != n
            or not np.array_equal(source_bins(observables['source_Q'], statistics), bins)):
        raise ValueError('Main output must retain exact parent-major four-draw population and frozen SOURCE bins')
    weights = population_weights(bins); row_weights = weights[:, parent_rows]
    bin_parent_counts = np.bincount(bins, minlength=8)
    reports, saved = {}, dict(source_parent_ids=parent_ids, source_parent_bins=bins, parent_weights=weights,
                             weight_scheme_names=np.asarray(policy['population']['weight_schemes']),
                             bin_parent_counts=bin_parent_counts, source_parent_row=parent_rows, draw=draws)
    moment_arrays = {}
    for wi, scheme in enumerate(policy['population']['weight_schemes']):
        wb = np.zeros(8, np.float64)
        z = (2.*np.arange(8)-7.)/7.
        unnormalized = np.ones(8) if wi == 0 else 1.+(.5 if wi == 1 else -.5)*z
        wb[:] = unnormalized*n/(unnormalized[bins].sum())
        if not np.allclose(wb[bins], weights[wi], rtol=0, atol=1e-14):
            raise RuntimeError('Parent and bin weights differ')
        bin_mass = np.bincount(bins, weights=weights[wi], minlength=8)
        report = dict(parents=n, retained_parents=n, weight_min=float(weights[wi].min()),
                      weight_max=float(weights[wi].max()), parent_weight_sum=float(weights[wi].sum()),
                      parent_ESS=float(weights[wi].sum()**2/np.square(weights[wi]).sum()),
                      bin_parent_count=bin_parent_counts, bin_parent_weight_sum=bin_mass,
                      bin_parent_weight_fraction=bin_mass/bin_mass.sum(), arms={})
        for a in ARMS:
            report['arms'][a] = {}
            for h in HYPOTHESES:
                label = a+'_'+h; metrics = {k: np.asarray(observables[label+'_'+k]) for k in ('Q', 'PSD', 'tail', 'PSD_raw')}
                if any(value.shape != (n*d,) for value in metrics.values()):
                    raise ValueError('Main scalar output row count changed')
                if any(not np.isfinite(metrics[k]).all() for k in ('Q', 'PSD', 'tail')):
                    raise ValueError('Nonfinite required main output')
                population = {k: weighted_statistics(value, row_weights[wi]) for k, value in metrics.items()}
                flags = {k: dict(rows=n*d, true_rows=int(np.asarray(observables[label+'_'+k]).sum()),
                    weighted_fraction=float(np.dot(row_weights[wi], observables[label+'_'+k])/row_weights[wi].sum()))
                    for k in ('Q_nonpositive', 'PSD_guarded', 'PSD_undefined')}
                memberships = dict(fixed_SOURCE_Q_in_SOURCE_TRAIN_bins=bins[parent_rows],
                                   output_Q_in_REAL_TRAIN_bins=output_bins(metrics['Q'], statistics))
                conditional = {}
                for membership_name, membership in memberships.items():
                    conditional[membership_name] = []
                    for b in range(8):
                        mask = membership == b; parents = np.unique(parent_rows[mask]); rows = int(mask.sum())
                        conditional[membership_name].append(dict(bin=b, rows=rows, parents=len(parents),
                            weight_sum=float(row_weights[wi, mask].sum()), supported=len(parents) >= policy['population']['minimum_parents_per_bin'],
                            metrics={k: weighted_statistics(value[mask], row_weights[wi, mask]) for k, value in metrics.items()}))
                base = label+'_fixed_SOURCE_Q_in_SOURCE_TRAIN_bins'
                mean, std, rows = (np.asarray(curves[base+'_'+k]) for k in ('mean', 'std', 'rows'))
                if not np.array_equal(rows, bin_parent_counts*d):
                    raise ValueError('Main SOURCE conditional waveform row counts differ from parent membership')
                mixture = exact_weighted_waveform(mean, std, rows, wb)
                if wi == 0:
                    for key in ('mean', 'std'):
                        np.testing.assert_allclose(mixture[key], curves[label+'_'+key], rtol=2e-12, atol=2e-13)
                saved.update({scheme+'_'+label+'_'+key: mixture[key] for key in ('mean', 'variance', 'std', 'weighted_bin_rows')})
                saved[scheme+'_'+label+'_SOURCE_conditional_mean'] = mean
                saved[scheme+'_'+label+'_SOURCE_conditional_std'] = std
                saved[scheme+'_'+label+'_SOURCE_conditional_supported'] = bin_parent_counts >= policy['population']['minimum_parents_per_bin']
                report['arms'][a][h] = dict(population=population, flags=flags, conditional=conditional,
                    waveform_global_moments='exact weighted SOURCE-bin mixture in NPZ',
                    SOURCE_conditional_waveform_moments='unchanged within bin; weight masses recorded',
                    output_Q_conditional_weighted_waveform_moments=None,
                    output_Q_conditional_weighted_waveform_reason='Main artifacts lack joint SOURCE-bin/output-Q-bin wave moments; no extra forward.')
                moment_arrays[wi, a, h] = {k: np.asarray([population[k][name][0] for name in ('mean', 'std')]) for k in ('Q', 'PSD')}
        report['hypothesis_sensitivity'] = {}
        report['full_vs_reference'] = {}
        for a in ARMS:
            report['hypothesis_sensitivity'][a] = {}
            for metric in ('Q', 'PSD'):
                values = np.asarray([moment_arrays[wi, a, h][metric] for h in HYPOTHESES])
                means, widths = values[:, 0], values[:, 1]
                report['hypothesis_sensitivity'][a][metric] = dict(mean_range=float(np.ptp(means)),
                    width_max_over_min=float(widths.max()/widths.min()) if widths.min() > 0 else None,
                    width_ratio_defined=bool(widths.min() > 0), all_three_hypotheses_retained=True)
        for reference in ('deterministic', 'simple'):
            report['full_vs_reference'][reference] = {}
            for h in HYPOTHESES:
                comparisons = {}
                for metric in ('Q', 'PSD'):
                    full = moment_arrays[wi, 'full', h][metric]; other = moment_arrays[wi, reference, h][metric]
                    comparisons[metric] = dict(mean_full_minus_reference=float(full[0]-other[0]),
                        std_full_minus_reference=float(full[1]-other[1]),
                        std_full_over_reference=float(full[1]/other[1]) if other[1] > 0 else None,
                        ratio_defined=bool(other[1] > 0))
                report['full_vs_reference'][reference][h] = comparisons
                for key in ('mean', 'std'):
                    saved[scheme+'_full_minus_'+reference+'_'+h+'_waveform_'+key] = (
                        saved[scheme+'_full_'+h+'_'+key]-saved[scheme+'_'+reference+'_'+h+'_'+key])
        reports[scheme] = report
    return dict(schemes=reports, response_forward_rows=0, noise005_forward_rows=0, optimizer_updates=0,
                C2ST_updates=0, scientific_status='response_match_not_established',
                waveform_mixture_exact_in_real_arithmetic=True, floating_roundoff_is_not_bitwise_identity=True,
                frozen_response_external_weight_invariance='Interface fact: SOURCE waveform and explicit latent inputs have no external population-weight argument.',
                no_new_same_x_forward_was_used_for_population_invariance=True,
                REAL_fit_or_reweight=False, h_physical_meaning='unknown',
                independent_acquisition_transfer_established=False), saved


def _save_population(primary_artifacts, destination, statistics, policy, source_ids=None):
    summaries = {}
    for split in ('TRAIN', 'DEV'):
        with np.load(Path(primary_artifacts)/(split+'_observables.npz'), allow_pickle=False) as archive:
            # REAL values are already-saved reference scalars; no original data decode.
            observables = {key: archive[key] for key in archive.files if not key.startswith('REAL_')}
        with np.load(Path(primary_artifacts)/(split+'_curves.npz'), allow_pickle=False) as archive:
            curves = {key: archive[key] for key in archive.files if not key.startswith('REAL_')}
        if source_ids is not None and split in source_ids and not np.array_equal(observables['source_parent_ids'], source_ids[split]):
            raise ValueError('Primary saved population IDs differ from the frozen preparation/DEV identity binding')
        if len(observables['source_parent_ids']) != (6593 if split == 'TRAIN' else 512):
            raise ValueError('Main four-draw population count changed: '+split)
        report, numerical = population_sensitivity(observables, curves, statistics, policy)
        save_json(destination/('POPULATION_'+split+'.json'), dict(split=split, **report))
        save_npz(destination/('POPULATION_'+split+'.npz'), **numerical)
        summaries[split] = dict(parents=len(observables['source_parent_ids']), rows=len(observables['draw']),
                                response_forward_rows=0, mathematical_waveform_mixture_exact=True,
                                output_Q_conditional_weighted_waveform_moments_available=False)
        _output_bound(destination, policy)
    return summaries


def run_diagnostics_arrays(data, models, extension, statistics, cfg, destination, policy, *, device='cpu',
                           artificial=False, freeze_receipt=None, counters=None,
                           primary_artifacts=None, population_source_ids=None):
    """Frozen-array driver. ``artificial=True`` is reserved for synthetic fixtures.

    Scientific callers must use ``diagnose_run`` so manifest/native/lineage
    gates precede array decoding. This driver never loads original science data.
    """
    policy = fixed_policy(policy); policy['_q_floor'] = float(statistics['loss']['q_floor'])
    if not np.isfinite(policy['_q_floor']) or policy['_q_floor'] <= 0:
        raise ValueError('Only the frozen positive TRAIN q_floor is allowed')
    if any(cfg['seeds'].get(k) != value for k, value in SEEDS.items()):
        raise ValueError('Addon streams differ from the original registered Window008 seeds')
    destination = Path(destination)
    if not destination.is_dir():
        raise ValueError('A fresh caller-owned artifact directory must already exist')
    if not artificial:
        if (freeze_receipt is None or not (destination/'FROZEN_INPUTS.json').is_file()
                or not (destination/'ADDON_FROZEN_INPUTS.json').is_file()
                or primary_artifacts is None):
            raise ValueError('Scientific array diagnostics require both freezes and primary artifacts')
        _verify_frozen(models, extension, freeze_receipt)
    counters = ForwardCounters() if counters is None else counters
    selected = select_repeat_parents(data['source_ids'], policy)
    edges = statistics['domains']['source']['Q256_eight_quantile_bin_edges']
    interventions = shape_interventions(data['source_carrier384'], data['source_ids'], selected, edges, policy)
    selection_values = {k: value for k, value in interventions.items()}
    selection_values.update(repeat_selected_rows=selected, repeat_selected_ids=np.asarray(data['source_ids'])[selected],
                            repeat_SOURCE_carrier384=np.asarray(data['source_carrier384'])[selected])
    save_npz(destination/'SELECTIONS.npz', **selection_values)
    rseeds = [policy['seeds']['repeat_'+name] for name in ('response', 'noise005', 'extension')]
    iseeds = [policy['seeds']['intervention_'+name] for name in ('response', 'noise005', 'extension')]
    repeat_bank, repeat_binding = explicit_latent_bank(128, 64, rseeds, chunk_rows=128)
    intervention_bank, intervention_binding = explicit_latent_bank(32, 8, iseeds, chunk_rows=128)
    save_npz(destination/'repeat_latents.npz', **repeat_bank)
    save_npz(destination/'intervention_latents.npz', **intervention_bank)
    save_json(destination/'LATENT_BANKS.json', dict(repeat=repeat_binding, intervention=intervention_binding,
        B_C_share_exact_bank=True, RNG_not_advanced_per_variant_arm_or_hypothesis=True))
    repeat = generate_family('repeat', np.asarray(data['source_carrier384'])[selected][None],
        np.asarray(data['source_ids'])[selected], 64, repeat_bank, models, extension, statistics, policy, counters, device=device)
    summaries = dict(repeat=save_repeat(repeat, destination, policy))
    common_reports = dict(repeat=_family_common(repeat, destination, 'REPEAT', policy))
    generation = dict(repeat=dict(chunks=repeat['chunks'], engineering=repeat['engineering']))
    del repeat, repeat_bank
    if not artificial:
        _verify_frozen(models, extension, freeze_receipt)
    shape_family = np.asarray((interventions['original_carrier384'], interventions['donor_same_Q_carrier384']))
    shape = generate_family('shape', shape_family, interventions['selected_ids'], 8, intervention_bank,
        models, extension, statistics, policy, counters, device=device)
    summaries['shape'] = save_interventions(shape, destination, policy, label='SHAPE',
                                           variant_names=policy['shape_intervention']['variants'])
    common_reports['shape'] = _family_common(shape, destination, 'SHAPE', policy)
    generation['shape'] = dict(chunks=shape['chunks'], engineering=shape['engineering'])
    if not artificial:
        _verify_frozen(models, extension, freeze_receipt)
    trend = generate_family('Q_trend', interventions['trend_carriers384'], interventions['selected_ids'], 8,
        intervention_bank, models, extension, statistics, policy, counters, device=device)
    summaries['Q_trend'] = save_interventions(trend, destination, policy, label='Q_TREND',
                                             variant_names=['Q25', 'Q50', 'Q75'])
    common_reports['Q_trend'] = _family_common(trend, destination, 'Q_TREND', policy)
    generation['Q_trend'] = dict(chunks=trend['chunks'], engineering=trend['engineering']); del trend
    # Replay both variants of metadata/order only on the original32x8 arm outputs.
    reference = dict(shape, variants=1,
        values={key: value for key, value in shape['values'].items() if key[0] == 0},
        waveform256={key: value for key, value in shape['waveform256'].items() if key[0] == 0})
    renamed = generate_family('identity_order_replay', interventions['original_carrier384'][None],
        np.asarray(['metadata_only_'+str(i) for i in range(32)]), 8, intervention_bank,
        models, extension, statistics, policy, counters, device=device)
    identity_report = compare_identity_order(reference, renamed, policy); del renamed
    order = np.arange(31, -1, -1)
    reordered = generate_family('identity_order_replay', interventions['original_carrier384'][order][None],
        interventions['selected_ids'][order], 8, intervention_bank, models, extension, statistics, policy, counters,
        device=device, latent_parent_order=order)
    order_report = compare_identity_order(reference, reordered, policy, inverse_order=np.argsort(order)); del reordered, shape, reference
    save_json(destination/'REPLAY_IDENTITY_ORDER.json', dict(rewritten_saved_ids=identity_report,
        reversed_parent_storage_order=order_report, latent_parent_order=order,
        no_new_random_draws=True, signal_forward_rows=1536, observation_rows=4608,
        engineering_replay_separately_registered=True))
    if not artificial:
        _verify_frozen(models, extension, freeze_receipt)
    if primary_artifacts is not None:
        summaries['population'] = _save_population(primary_artifacts, destination, statistics, policy, population_source_ids)
    else:
        summaries['population'] = dict(status='not_executed_in_array_fixture', response_forward_rows=0,
                                        independent_artificial_population_helper_check_required=True)
    count_receipt = counters.validate(policy)
    save_json(destination/'ACTUAL_COUNTERS.json', count_receipt)
    save_json(destination/'GENERATION.json', dict(groups=generation, common_baseline_noise_statistics=common_reports,
        row_removal_count=0, input_identity_never_model_feature=True, no_hypothesis_selection=True,
        all_selected_parent_draw_rows_saved=True, time_morphology_undefined_rows_retained=True))
    save_json(destination/'DIAGNOSTIC_STATISTICS.json', summaries)
    if not artificial:
        _verify_frozen(models, extension, freeze_receipt)
    size = _output_bound(destination, policy)
    artifacts = {path.name: sha256(path) for path in sorted(destination.iterdir()) if path.is_file()}
    summary = dict(schema='n01/window008-resume01-diagnostics/v1', UTC=utc(),
        status='artificial_shape_fixture_complete' if artificial else 'registered_frozen_endpoint_diagnostics_complete',
        diagnostic_id=policy['diagnostic_id'], scientific_status='response_match_not_established',
        scientific_pass_threshold=None, optimizer_updates=0, G_updates=0, D_updates=0,
        noise_updates=0, population_updates=0, C2ST_updates=0,
        DEV_numeric_decoded=not artificial, artificial_arrays_only=artificial,
        scientific_model_forward_rows=0 if artificial else count_receipt['total_signal_forward_rows'],
        actual_forward_counts=count_receipt, all_selected_parent_draw_rows_preserved=True,
        repeat_selection_parent_count=128, repeat_draws_per_parent=64,
        intervention_parent_count=32, intervention_draws_per_parent=8,
        population=summaries['population'], physical_response_match_established=False,
        paired_event_response_truth_available=False, independent_acquisition_available=False,
        frozen_states_rechecked=not artificial, output_bytes_before_SUMMARY=size,
        artifact_sha256=artifacts, interpretation=policy['claim_boundaries'])
    save_json(destination/'SUMMARY.json', summary)
    _output_bound(destination, policy)
    return _clean(summary)


def preregistration_gate(manifest_path, policy_path, main_manifest_path, *, root=None):
    """Read source/policy bytes only, before any model/data numerical operation."""
    root = ROOT if root is None else Path(root)
    manifest_path, policy_path, main_manifest_path = map(Path, (manifest_path, policy_path, main_manifest_path))
    before = sha256(manifest_path); manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('before_DEV_numeric_decode') is not True
            or manifest.get('before_scientific_model_forward') is not True
            or manifest.get('policy_sha256') != sha256(policy_path)):
        raise ValueError('Independent addon source/policy freeze is missing or changed')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or not files:
        raise ValueError('Independent addon source manifest needs explicit file hashes')
    required = {'src/n01/diagnose_window008_resume01.py', 'window008_resume01_diagnostics_policy.json',
                'scripts/check_window008_resume01_diagnostics.py', 'docs/WINDOW008_RESUME01_DIAGNOSTICS.md',
                'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'}
    if not required.issubset(files):
        raise ValueError('Independent addon manifest omits required source/policy/doc/check/main-freeze files')
    paths = {}
    for relative, expected in files.items():
        path = root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or sha256(path) != expected:
            raise ValueError('Independent addon source manifest file changed: '+relative)
        paths[str(path.resolve())] = expected
    if (files['window008_resume01_diagnostics_policy.json'] != sha256(policy_path)
            or files['WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'] != sha256(main_manifest_path)
            or sha256(manifest_path) != before):
        raise ValueError('Independent addon/main manifest binding changed')
    paths[str(manifest_path.resolve())] = before
    return dict(schema='n01/window008-resume01-addon-preregistration/v1', UTC=utc(),
                addon_source_manifest_sha256=before, addon_policy_sha256=sha256(policy_path),
                main_source_manifest_sha256=sha256(main_manifest_path), files_sha256=paths,
                before_DEV_numeric_decode=True, before_scientific_model_forward=True,
                optimizer_updates=0, C2ST_updates=0)


def _primary_binding(primary_artifacts, native_path, replay_native_path, freeze):
    """Hash-check completed main artifacts before reading any NPZ member."""
    directory = Path(primary_artifacts)
    paths = [directory/'SUMMARY.json', directory/'FROZEN_INPUTS.json', Path(native_path), Path(replay_native_path)]
    values = [json.loads(path.read_text(encoding='utf8')) for path in paths]
    summary, primary_freeze, native, replay_native = values
    if (summary.get('schema') != 'n01/window008-resume01-evaluation/v1'
            or summary.get('status') != 'fixed_resumed_endpoint_evaluation_complete'
            or native.get('exit_code') != 0 or native.get('reason') is not None
            or replay_native.get('exit_code') != 0 or replay_native.get('reason') is not None
            or summary.get('scientific_status') != 'response_match_not_established'
            or summary.get('C2ST_updates') != 0 or summary.get('evaluation_G_updates', summary.get('optimizer_updates')) != 0
            or summary.get('population_updates') != 0
            or primary_freeze.get('checkpoint_state_sha256') != freeze['checkpoint_state_sha256']
            or primary_freeze.get('noise_state_sha256') != freeze['noise_state_sha256']):
        raise ValueError('Population addon requires completed native0 primary/replay with identical frozen endpoint states')
    hashes = {str(path.resolve()): sha256(path) for path in paths}
    artifact_hashes = summary.get('artifact_sha256', {})
    for filename in ('TRAIN_observables.npz', 'TRAIN_curves.npz', 'DEV_observables.npz', 'DEV_curves.npz', 'FROZEN_INPUTS.json'):
        if filename not in artifact_hashes:
            raise ValueError('Main artifact hash list omits required population input: '+filename)
    for filename, expected in artifact_hashes.items():
        if Path(filename).name != filename or sha256(directory/filename) != expected:
            raise ValueError('Main artifact hash changed: '+filename)
        hashes[str((directory/filename).resolve())] = expected
    return dict(primary_artifacts=str(directory.resolve()), files_sha256=hashes,
                primary_native_exit_code=0, replay_native_exit_code=0,
                source='Already-generated main four-draw observables and exact fixed-SOURCE-bin moments, no additional forward.')


def _verify_file_binding(receipt):
    for name, expected in receipt['files_sha256'].items():
        if sha256(name) != expected:
            raise RuntimeError('Frozen addon or population input changed: '+name)


def diagnose_run(out, cfg, resume_cfg, dataset, statistics, raw_container, identities, *, source_manifest,
                 evaluation_policy, policy, addon_source_manifest, primary_artifacts,
                 primary_native_receipt, replay_native_receipt, destination, device='cuda:0', root=None):
    """Production entry point with all prelaunch, native, state and lineage gates."""
    from .evaluate_window008_resume01 import _freeze
    root = ROOT if root is None else Path(root)
    destination = _resolve(destination)
    if os.name == 'nt' and destination.drive.upper() not in ('E:', 'D:'):
        raise ValueError('All new diagnostic artifacts must stay on E or D')
    destination.mkdir(parents=True, exist_ok=False)
    counters = ForwardCounters(); decode_started = False; decoded = False
    try:
        policy_path = _resolve(policy); main_manifest = _resolve(source_manifest)
        preregistration = preregistration_gate(_resolve(addon_source_manifest), policy_path, main_manifest, root=root)
        policy_value = fixed_policy(policy_path)
        cfg_value, resume_value = _read_json(cfg), _read_json(resume_cfg)
        if (cfg_value['cpu_threads'] != policy_value['CPU_threads'] or cfg_value['gpu_memory_limit_GiB'] != policy_value['resource_limits']['GPU_GiB']):
            raise ValueError('Addon CPU/GPU limits differ from the frozen main configuration')
        save_json(destination/'ADDON_PREREGISTERED_INPUTS.json', preregistration)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(policy_value['CPU_threads']); torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        statistics_path = _resolve(statistics)
        stats_value = _read_json(statistics_path); dataset_path = _resolve(dataset)
        raw_path, identity_path = _resolve(raw_container), _resolve(identities)
        models, extension, freeze, _, data_statistics = _freeze(_resolve(out), cfg_value, resume_value,
            stats_value, dataset_path, statistics_path, raw_path, identity_path, device, destination,
            source_manifest=main_manifest, policy_path=_resolve(evaluation_policy), root=root)
        primary = _primary_binding(_resolve(primary_artifacts), _resolve(primary_native_receipt),
                                   _resolve(replay_native_receipt), freeze)
        addon_binding = dict(schema='n01/window008-resume01-addon-freeze/v1', UTC=utc(),
            before_DEV_numeric_decode=True, before_scientific_model_forward=True,
            policy_sha256=sha256(policy_path), resolved_policy=policy_value,
            checkpoint_state_sha256=freeze['checkpoint_state_sha256'], noise_state_sha256=freeze['noise_state_sha256'],
            files_sha256={**preregistration['files_sha256'], **primary['files_sha256']}, primary=primary,
            optimizer_updates=0, C2ST_updates=0, scientific_status='response_match_not_established')
        _verify_file_binding(addon_binding)
        save_json(destination/'ADDON_FROZEN_INPUTS.json', addon_binding)
        evaluation_statistics = copy.deepcopy(stats_value); evaluation_statistics['domains'] = data_statistics['domains']
        # No train/DEV member is opened until both independent freeze receipts exist.
        with np.load(dataset_path, allow_pickle=False) as archive:
            if {'source_val', 'target_val', 'hval'} & set(archive.files):
                raise ValueError('TRAIN preparation illegally contains DEV numerical values')
            train = {key: archive[key] for key in ('source_ids', 'real_ids', 'real_groups', 'source_val_ids', 'real_val_original_index')}
        decode_started = True
        dev = _load_dev(raw_path, identity_path, cfg_value['window'], stats_value['loss']['q_floor'])
        decoded = True; _validate_split(dev, 'DEV')
        if (not np.array_equal(train['source_val_ids'], dev['source_ids'])
                or not np.array_equal(train['real_val_original_index'], dev['real_ids'])
                or np.intersect1d(train['source_ids'], dev['source_ids']).size
                or np.intersect1d(train['real_ids'], dev['real_ids']).size
                or np.intersect1d(train['real_groups'], dev['real_groups']).size):
            raise ValueError('DEV lineage/role isolation differs from frozen preparation')
        result = run_diagnostics_arrays(dev, models, extension, evaluation_statistics, cfg_value, destination, policy_value,
            device=device, freeze_receipt=freeze, counters=counters, primary_artifacts=_resolve(primary_artifacts),
            population_source_ids={'TRAIN': train['source_ids'], 'DEV': dev['source_ids']})
        _verify_frozen(models, extension, freeze); _verify_file_binding(addon_binding)
        return result
    except Exception as error:
        save_json(destination/'FAILED.json', dict(UTC=utc(), status='diagnostics_failed_preserved',
            error_type=type(error).__name__, error=str(error), DEV_numeric_decode_started=decode_started,
            DEV_numeric_decoded=decoded, actual_partial_counters=counters.groups,
            optimizer_updates=0, C2ST_updates=0, scientific_status='response_match_not_established',
            partial_artifacts_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('out', 'config', 'resume-config', 'dataset', 'statistics', 'raw-container', 'identities',
                 'source-manifest', 'evaluation-policy', 'policy', 'addon-source-manifest', 'primary-artifacts',
                 'primary-native-receipt', 'replay-native-receipt', 'destination'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    result = diagnose_run(args.out, args.config, args.resume_config, args.dataset, args.statistics,
        args.raw_container, args.identities, source_manifest=args.source_manifest, evaluation_policy=args.evaluation_policy,
        policy=args.policy, addon_source_manifest=args.addon_source_manifest, primary_artifacts=args.primary_artifacts,
        primary_native_receipt=args.primary_native_receipt, replay_native_receipt=args.replay_native_receipt,
        destination=args.destination, device=args.device)
    print(json.dumps(dict(status=result['status'], optimizer_updates=0, C2ST_updates=0,
                         scientific_status=result['scientific_status'], counts=result['actual_forward_counts']), allow_nan=False))


if __name__ == '__main__':
    main()
