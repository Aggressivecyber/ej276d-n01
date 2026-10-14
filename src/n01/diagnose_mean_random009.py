"""Bound TRAIN-only A/B/C frozen-response diagnosis; D is derived, never run.

This entry point accepts its own isolated-root policy and source manifest.  It
does not call a legacy evaluator's inventory/completion gate, read DEV members,
create an optimizer, or alter historical endpoints.  Small pure helpers and the
array driver support artificial semantic checks without loading trained states.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .data import sha256
from .evaluate_window008 import (
    _array, _clean, _memory_limit, array_sha, distance_receipt, save_json,
    save_npz, scalar_summary,
)
from .losses008_v2 import N01Loss008
from .model008 import N01Response008
from .model008_fix01 import N01Response008Fix01
from .observation008 import observe_generated_carrier
from .response008 import (
    HYPOTHESES, PairedStreams, fixed_noises, loss_tensor_statistics,
    make_extension, state_digest, statistics_digest,
)
from .train import utc

ROOT = Path(__file__).resolve().parents[2]
POLICY_SCHEMA = 'n01/mean-random009-policy/v1'
BINDINGS_SCHEMA = 'n01/mean-random009-bindings/v1'
DIAGNOSTIC_ID = 'mean_random009_20261014'
VIEWS = ('A', 'B', 'C')
ALL_OFF = ('Q', 'time', 'basis', 'local')
SEEDS = dict(selection=2026101410, response=2026101411,
             noise005=2026101412, extension=2026101413,
             real_selection=2026101414, projections=2026101415,
             bootstrap=2026101416)
TRAIN_KEYS = (
    'source', 'source_carrier384', 'real', 'real_baseline100',
    'source_ids', 'real_ids', 'real_groups', 'source_role', 'real_role',
)
TRAIN_OPTIONAL_KEYS = ('original_source_weights', 'h')
SCALAR_FIELDS = (
    'Q', 'tail', 'PSD', 'qsig', 'qbar', 'time_scale', 'Q_signal',
    'Q_noise', 'Q_ADC_delta', 'tail_signal', 'tail_noise', 'tail_ADC_delta',
    'Q_signal_crop_migration', 'base_shape_Q', 'qsig_over_base_shape_Q',
    'stochastic_shape_voltage_RMS_V', 'local_RMS_V',
)
SOURCE_DEPENDENCIES = (
    '__init__', 'ablation009', 'data', 'data008', 'diagnose_mean_random009',
    'electronics_v3', 'evaluate', 'evaluate_window008', 'extension006',
    'losses', 'losses008', 'losses008_v2', 'model', 'model008',
    'model008_fix01', 'observation008', 'response008', 'train',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs)


def fixed_policy(value):
    p = read_json(value) if isinstance(value, (str, Path)) else copy.deepcopy(value)
    require(p.get('schema') == POLICY_SCHEMA and p.get('diagnostic_id') == DIAGNOSTIC_ID,
            '009 policy identity changed')
    require(p.get('arms') == list(VIEWS) and p.get('derived_mean') == 'D'
            and p.get('hypotheses') == list(HYPOTHESES) and p.get('seeds') == SEEDS,
            '009 views, hypotheses, or explicit streams changed')
    require((p.get('parents'), p.get('repeats'), p.get('chunk_rows'), p.get('CPU_threads'))
            == (512, 16, 128, 12) and p.get('TRAIN_counts') == dict(source=6593, real=21047),
            '009 fixed TRAIN counts or repeat budget changed')
    require(p['selection'].get('mode') == 'source_TRAIN_Q8_stratified_without_replacement'
            and p['selection'].get('bins') == 8 and p['selection'].get('parents_per_bin') == 64
            and p['selection'].get('population_weights_recorded_separately') is True,
            '009 source-selection rule changed')
    require(p.get('matched_real_rows') == 8192 and p.get('REAL_REAL_rows_each') == 8192
            and p.get('REAL_REAL_disjoint') is True and p.get('bootstrap_replicates') == 500,
            '009 matched REAL or parent-bootstrap rule changed')
    require(p['comparison'].get('paired_common_noise') is True
            and p['comparison'].get('D_means_of_individual_hard_observations') is True
            and p['comparison'].get('D_distribution_acceptance') is False
            and p['comparison'].get('sample_covariance_ddof') == 1
            and p['comparison'].get('waveform_covariance_storage') == 'centered_draw_factor',
            '009 pairing or covariance semantics changed')
    require(all(p.get(key) == 0 for key in ('optimizer_updates', 'G_updates', 'D_updates',
            'noise_updates', 'population_updates', 'C2ST_updates'))
            and p.get('DEV_numeric_read') is False and p.get('overwrite_outputs') is False
            and p.get('scientific_pass_threshold') is None,
            '009 zero-update/TRAIN-only/no-overwrite contract changed')
    limits = p['resource_limits']
    require((limits.get('GPU_GiB'), limits.get('job_tree_RSS_GiB'),
             limits.get('wall_seconds'), limits.get('output_MiB')) == (10, 8, 7200, 1536),
            '009 resource budget changed')
    require(p['observation'] == dict(internal_samples=384, baseline_samples=100,
            effective_samples=256, peak_search=[112, 144], samples_before_peak=16,
            ADC_once=True, canonical_Q_slice=[110, 366]), '009 observation changed')
    return p


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def bind_inputs(policy_path, bindings_path, output, *, root=ROOT):
    """Hash every registered input before any np.load or torch.load call."""
    root = Path(root).resolve()
    policy_path, bindings_path, output = [Path(v).resolve() for v in (policy_path, bindings_path, output)]
    require(policy_path.is_file() and bindings_path.is_file(), 'Missing explicit policy/bindings')
    p, b = fixed_policy(policy_path), read_json(bindings_path)
    require(b.get('schema') == BINDINGS_SCHEMA and b.get('policy_sha256') == sha256(policy_path),
            '009 binding schema/policy bytes differ')
    roots = [Path(v).resolve() for v in b['roots']]
    require(roots and root in roots and inside(policy_path, [root]) and inside(bindings_path, [root]),
            'Policy and bindings must belong to the isolated source root')
    require(inside(output, [root]) and output != root and not output.exists(),
            '009 requires a fresh output directory inside the isolated root')
    if os.name == 'nt':
        require(output.drive.upper() in ('E:', 'D:'), '009 output must be on E/D')
    else:
        require(root == Path(p['remote_root']).resolve()
                and set(roots) == {root, Path(p['immutable_endpoint_root']).resolve()},
                'Registered isolated/immutable remote roots differ')
    manifest_record = b['source_manifest']
    manifest_path = Path(manifest_record['path'])
    require(manifest_path.is_absolute(), 'Source manifest path must be absolute')
    manifest_path = manifest_path.resolve()
    require(inside(manifest_path, [root]) and sha256(manifest_path) == manifest_record['sha256'],
            '009 source manifest path/bytes differ')
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Missing explicit source file inventory')
    source_files = {}
    for relative, digest in manifest['files'].items():
        rel = Path(relative)
        path = (root / rel).resolve()
        require(not rel.is_absolute() and inside(path, [root]) and path.is_file(),
                'Unsafe/missing manifest source: ' + relative)
        require(sha256(path) == digest, 'Source bytes changed: ' + relative)
        source_files[str(path)] = digest
    required_sources = {str(root / ('src/n01/'+name+'.py')) for name in SOURCE_DEPENDENCIES}
    require(required_sources <= set(source_files),
            '009 numerical import dependencies are not all bound in their own manifest: '
            + str(sorted(required_sources-set(source_files))))
    required = {'dataset', 'statistics', 'data_statistics', 'noise005',
                'deterministic', 'full', 'base_config'}
    require(required <= set(b['files']), 'Missing 009 bound inputs: ' + str(required-set(b['files'])))
    files, paths = {}, {}
    for name, record in b['files'].items():
        path = Path(record['path'])
        require(path.is_absolute(), 'Input path must be absolute: ' + name)
        path = path.resolve()
        require(inside(path, roots) and path.is_file(), 'Unregistered input root/path: ' + name)
        require(sha256(path) == record['sha256'], 'Input bytes changed before decode: ' + name)
        if 'bytes' in record:
            require(path.stat().st_size == record['bytes'], 'Input size changed: ' + name)
        paths[name] = path
        files[str(path)] = record['sha256']
    files.update({str(policy_path): sha256(policy_path), str(bindings_path): sha256(bindings_path),
                  str(manifest_path): sha256(manifest_path)})
    return p, paths, dict(files_sha256=files, sources_sha256=source_files,
                         policy_sha256=sha256(policy_path), bindings_sha256=sha256(bindings_path),
                         manifest_sha256=sha256(manifest_path), isolated_root=str(root),
                         before_TRAIN_numeric_decode=True, before_checkpoint_load=True,
                         DEV_numeric_read=False, optimizer_updates=0)


def verify_files(freeze):
    for path, expected in {**freeze['files_sha256'], **freeze['sources_sha256']}.items():
        require(sha256(path) == expected, 'Frozen input/source changed: ' + path)


def read_train_members(archive):
    """Whitelist TRAIN fields; never iterate/decode the archive's other arrays."""
    require(set(TRAIN_KEYS) <= set(archive.files), 'Missing prepared TRAIN fields')
    names = list(TRAIN_KEYS) + [v for v in TRAIN_OPTIONAL_KEYS if v in archive.files]
    return {name: archive[name].copy() for name in names}


def validate_train(data, *, source_rows=6593, real_rows=21047):
    shapes = dict(source=(source_rows, 256), source_carrier384=(source_rows, 384),
                  real=(real_rows, 256), real_baseline100=(real_rows, 100),
                  source_ids=(source_rows,), real_ids=(real_rows,), real_groups=(real_rows,),
                  source_role=(source_rows,), real_role=(real_rows,))
    for name, shape in shapes.items():
        require(name in data and np.asarray(data[name]).shape == shape, 'Wrong TRAIN shape: ' + name)
        a = np.asarray(data[name])
        require(a.dtype.kind != 'O' and (a.dtype.kind not in 'fc' or np.isfinite(a).all()),
                'Object/nonfinite TRAIN field: ' + name)
    require(set(np.unique(data['source_role'])) == {'TRAIN'}
            and set(np.unique(data['real_role'])) == {'TRAIN'}, 'Non-TRAIN waveform role')
    require(len(np.unique(data['source_ids'])) == source_rows
            and len(np.unique(data['real_ids'])) == real_rows, 'Nonunique TRAIN identity')
    centered = np.asarray(data['source_carrier384'], np.float64)
    centered = centered-centered[:, :100].mean(1, keepdims=True)
    q = centered[:, 110:366].sum(1)
    require(np.isfinite(q).all() and (q > 1e-8).all()
            and ((centered[:, 112:144].argmax(1)+112) == 126).all(),
            'TRAIN SOURCE canonical charge/anchor changed; no row deletion')
    require(np.allclose(q, np.asarray(data['source'], np.float64).sum(1), rtol=0, atol=1e-10),
            'Prepared SOURCE carrier/window charge mismatch')
    for name in TRAIN_OPTIONAL_KEYS:
        if name in data:
            a = np.asarray(data[name])
            require(a.shape == (source_rows,) and np.isfinite(a).all(), 'Wrong TRAIN metadata: ' + name)
    return q


def select_parents(q, edges, *, parents_per_bin=64, seed=2026101410):
    edges = np.asarray(edges, np.float64)
    require(edges.shape == (9,) and np.isfinite(edges).all() and (np.diff(edges) >= 0).all(),
            'Frozen SOURCE TRAIN Q8 edges changed')
    bins = np.searchsorted(edges[1:-1], np.asarray(q), side='left')
    rng, selected = np.random.default_rng(seed), []
    counts = np.bincount(bins, minlength=8)
    for b in range(8):
        rows = np.flatnonzero(bins == b)
        require(len(rows) >= parents_per_bin, 'Insufficient original SOURCE Q stratum')
        selected.extend(rng.permutation(rows)[:parents_per_bin])
    selected = np.asarray(selected, np.int64)
    require(len(np.unique(selected)) == 8*parents_per_bin, 'Source selection repeats parents')
    # Statistical poststratification weights are not physical weights or inputs.
    post_weights = counts[bins[selected]] / float(parents_per_bin)
    post_weights *= len(selected) / post_weights.sum()
    return selected, bins[selected], counts, post_weights


def select_real(n, matched, seed):
    require(n >= 2*matched, 'Insufficient disjoint REAL-REAL TRAIN support')
    order = np.random.default_rng(seed).permutation(n)
    return order[:matched].astype(np.int64), order[matched:2*matched].astype(np.int64)


def explicit_bank(rows, chunk_rows, seeds):
    streams = PairedStreams(seeds['response'], seeds['noise005'], seeds['extension'])
    before = streams.state_dict()
    pieces = {'response': {}, 'noise005': {}, 'extension': []}
    chunks = []
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows-start)
        response, noise, epsilon = streams.draw(count, 'cpu')
        for family, values in (('response', response), ('noise005', noise)):
            for name, value in values.items():
                pieces[family].setdefault(name, []).append(value)
        pieces['extension'].append(epsilon)
        chunks.append(dict(start=start, stop=start+count,
            response={k: array_sha(v) for k, v in response.items()},
            noise005={k: array_sha(v) for k, v in noise.items()}, extension=array_sha(epsilon)))
    bank = {family: {k: torch.cat(v) for k, v in pieces[family].items()}
            for family in ('response', 'noise005')}
    bank['extension'] = torch.cat(pieces['extension'])
    states = {phase+'_'+name: _array(value) for phase, state in
              (('before', before), ('after', streams.state_dict())) for name, value in state.items()}
    values = {family+'_'+name: _array(v) for family in ('response', 'noise005') for name, v in bank[family].items()}
    values.update(extension=_array(bank['extension']), **states)
    return bank, values, dict(rows=rows, chunk_rows=chunk_rows, draw_order='parent_major',
                             seeds={k: seeds[k] for k in ('response', 'noise005', 'extension')}, chunks=chunks)


def finite_moments(values):
    """Parent x repeat x ...; NaN availability/morphology is explicitly counted."""
    x = np.asarray(values, np.float64)
    require(x.ndim >= 2 and not np.isinf(x).any(), 'Bad repeated moment array')
    valid = np.isfinite(x)
    count = valid.sum(1)
    mean = np.full(count.shape, np.nan)
    np.divide(np.where(valid, x, 0.).sum(1), count, out=mean, where=count > 0)
    centered = np.where(valid, x-mean[:, None], 0.)
    variance = np.full(count.shape, np.nan)
    np.divide(np.square(centered).sum(1), count-1, out=variance, where=count > 1)
    se = np.full(count.shape, np.nan)
    np.divide(np.sqrt(variance), np.sqrt(count), out=se, where=count > 1)
    return dict(mean=mean, sample_variance=variance, MonteCarlo_SE=se,
                defined_draws=count, undefined_draws=x.shape[1]-count)


def covariance_factor(values):
    x = np.asarray(values, np.float64)
    require(x.ndim == 3 and x.shape[1] >= 2 and np.isfinite(x).all(),
            'Wave covariance factor requires finite parent x repeat x voltage')
    mean = x.mean(1)
    factor = x-mean[:, None]
    variance = np.square(factor).sum(1)/(x.shape[1]-1)
    return dict(mean=mean, centered_draw_covariance_factor=factor,
                sample_variance=variance, MonteCarlo_SE=np.sqrt(variance/x.shape[1]),
                covariance_denominator=np.asarray(x.shape[1]-1, np.int64),
                covariance_rank_upper_bound=np.asarray(x.shape[1]-1, np.int64))


def scalar_covariance(values):
    x = np.asarray(values, np.float64)
    result = finite_moments(x)
    valid = np.isfinite(x)
    p, _, f = x.shape
    covariance = np.full((p, f, f), np.nan)
    counts = np.empty((p, f, f), np.int64)
    for i in range(f):
        for j in range(i, f):
            pair = valid[:, :, i] & valid[:, :, j]
            count = pair.sum(1)
            mi, mj = np.zeros(p), np.zeros(p)
            np.divide(np.where(pair, x[:, :, i], 0.).sum(1), count, out=mi, where=count > 0)
            np.divide(np.where(pair, x[:, :, j], 0.).sum(1), count, out=mj, where=count > 0)
            product = np.where(pair, (x[:, :, i]-mi[:, None])*(x[:, :, j]-mj[:, None]), 0.)
            cov = np.full(p, np.nan)
            np.divide(product.sum(1), count-1, out=cov, where=count > 1)
            covariance[:, i, j] = covariance[:, j, i] = cov
            counts[:, i, j] = counts[:, j, i] = count
    return dict(**result, sample_covariance=covariance, jointly_defined_draws=counts)


def _crossing(y, fraction, side):
    peak = float(y[16])
    if peak <= 0:
        return np.nan
    target = fraction*peak
    pairs = ((j-1, j) for j in range(16, 0, -1)) if side == 'left' else ((j, j+1) for j in range(16, 255))
    for lo, hi in pairs:
        a, b = float(y[lo]), float(y[hi])
        if (side == 'left' and a <= target <= b) or (side == 'right' and b <= target <= a):
            return float(hi if side == 'left' else lo) if a == b else lo+(target-a)/(b-a)
    return np.nan


def morphology(values):
    y = np.asarray(values, np.float64)
    require(y.ndim == 2 and y.shape[1] == 256 and np.isfinite(y).all(), 'Invalid hard observed wave')
    left = np.asarray([[_crossing(row, f, 'left') for f in (.1, .5, .9)] for row in y])
    right = np.asarray([[_crossing(row, f, 'right') for f in (.1, .5, .9)] for row in y])
    peak = y[:, 16]
    nonpositive = peak <= 0
    rise, width = left[:, 2]-left[:, 0], right[:, 1]-left[:, 1]
    rise_missing, width_missing = ~np.isfinite(rise), ~np.isfinite(width)
    pre_rise = (~nonpositive & rise_missing &
        ((~np.isfinite(left[:, 0]) & (y[:, 0] > .1*peak)) |
         (~np.isfinite(left[:, 2]) & (y[:, 0] > .9*peak))))
    pre_width = ~nonpositive & width_missing & ~np.isfinite(left[:, 1]) & (y[:, 0] > .5*peak)
    post_width = ~nonpositive & width_missing & ~np.isfinite(right[:, 1]) & (y[:, -1] > .5*peak)
    result = dict(peak_voltage_V=peak, rise10_90_ns=rise, FWHM_ns=width,
        left_crossing10_ns=left[:, 0], left_crossing50_ns=left[:, 1], left_crossing90_ns=left[:, 2],
        right_crossing50_ns=right[:, 1], rise_undefined=rise_missing, FWHM_undefined=width_missing,
        temporal_nonpositive_peak=nonpositive, rise_prepeak_censored=pre_rise,
        rise_other_undefined=rise_missing & ~nonpositive & ~pre_rise,
        FWHM_prepeak_censored=pre_width, FWHM_postpeak_censored=post_width,
        FWHM_other_undefined=width_missing & ~nonpositive & ~pre_width & ~post_width,
        selected_peak_below_global_crop_max=peak < y.max(1))
    require(np.array_equal(rise_missing, nonpositive | pre_rise | result['rise_other_undefined']),
            'Rise missingness causes do not cover all rows')
    require(np.array_equal(width_missing, nonpositive | pre_width | post_width | result['FWHM_other_undefined']),
            'FWHM missingness causes do not cover all rows')
    return result


def bootstrap_counts(parent_bins, replicates, seed):
    bins = np.asarray(parent_bins)
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(bins)), np.float64)
    for rep in range(replicates):
        for b in np.unique(bins):
            rows = np.flatnonzero(bins == b)
            chosen = rng.choice(rows, len(rows), replace=True)
            counts[rep] += np.bincount(chosen, minlength=len(bins))
    require((counts.sum(1) == len(bins)).all(), 'Parent cluster bootstrap lost a stratum')
    return counts/len(bins)


def paired_statistics(left, right, bootstrap_mass):
    difference = np.asarray(left, np.float64)-np.asarray(right, np.float64)
    require(np.isfinite(difference).all(), 'Paired voltage difference is nonfinite')
    moments = finite_moments(difference)
    parent_mean = moments['mean']
    sample = bootstrap_mass @ parent_mean
    return dict(**moments, equal_parent_mean=parent_mean.mean(0),
        conditional_global_MonteCarlo_SE=np.sqrt(moments['sample_variance'].sum(0)/
            (difference.shape[1]*difference.shape[0]**2)),
        parent_cluster_CI95=np.quantile(sample, [.025, .975], axis=0))


def current_rss():
    try:
        import psutil
        p = psutil.Process()
        return p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
    except ImportError:
        status = Path('/proc/self/status')
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])*1024
    return None


class OutputStore:
    def __init__(self, root, policy):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.policy = policy
        self.began = time.monotonic()
        self.peak_rss = 0

    def guard(self, device):
        limit = self.policy['resource_limits']
        require(time.monotonic()-self.began < limit['wall_seconds'], '009 wall budget exceeded')
        rss = current_rss()
        if rss is not None:
            self.peak_rss = max(self.peak_rss, rss)
            require(rss <= limit['job_tree_RSS_GiB']*1024**3, '009 RSS budget exceeded')
        gpu = dict(CUDA_allocated_bytes=0, CUDA_reserved_bytes=0)
        if torch.device(device).type == 'cuda':
            gpu = dict(CUDA_allocated_bytes=torch.cuda.memory_allocated(device),
                       CUDA_reserved_bytes=torch.cuda.memory_reserved(device),
                       CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            require(gpu['CUDA_reserved_bytes'] <= limit['GPU_GiB']*1024**3,
                    '009 CUDA allocator budget exceeded')
        size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        require(size <= limit['output_MiB']*1024**2, '009 output budget exceeded')
        return dict(output_bytes=size, RSS_bytes=rss, peak_RSS_bytes=self.peak_rss,
                    elapsed_seconds=time.monotonic()-self.began, CPU_threads=torch.get_num_threads(), **gpu)

    def json(self, name, value):
        save_json(self.root/name, value)

    def npz(self, name, **values):
        save_npz(self.root/name, **values)

    def status(self, value):
        """One explicitly mutable receipt; completed scientific artifacts stay immutable."""
        path = self.root/'status.json'
        temporary = self.root/'status.json.tmp'
        with temporary.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


class RunProgress:
    """Pre-call evidence survives TERM/KILL; attempted rows are never completed rows."""
    def __init__(self, store, *, artificial=False):
        self.store, self.artificial = store, artificial
        self.counters = dict(A_signal_forward_rows=0, B_signal_forward_rows=0,
            C_ablation_forward_rows=0, signal_forward_rows=0, hard_observation_rows=0,
            noise005_rows=0, D_model_forward_rows=0, optimizer_updates=0,
            G_updates=0, D_updates=0, noise_updates=0, population_updates=0, C2ST_updates=0)
        self.attempted = {k: 0 for k in self.counters}
        self.completed_family_rows = 0
        self.completed_families = 0
        self.in_flight = None
        self.last_completed_operation = None
        self.phase = 'bound_inputs_or_artificial_fixture_initialization'
        self.persist()

    def payload(self, status='running_frozen_diagnosis'):
        return dict(schema='n01/mean-random009-progress/v1', diagnostic_id=DIAGNOSTIC_ID,
            UTC=utc(), pid=os.getpid(), status=status, phase=self.phase,
            artificial_only=self.artificial, completed_operation_counters=self.counters.copy(),
            counters=self.counters.copy(), attempted_operation_counters=self.attempted.copy(),
            completed_family_rows=self.completed_family_rows, completed_families=self.completed_families,
            in_flight=copy.deepcopy(self.in_flight), last_completed_operation=copy.deepcopy(self.last_completed_operation),
            in_flight_completion='unknown until the corresponding completed-operation receipt is persisted',
            noise_operation_scope='frozen005 plus all three extension hypotheses; a failed call may have completed internal suboperations',
            DEV_numeric_read=False, optimizer_updates=0, automatic_restart=False)

    def persist(self, status='running_frozen_diagnosis'):
        self.store.status(self.payload(status))

    def execute(self, key, count, action, *, operation, **metadata):
        signal_call = key in ('A_signal_forward_rows', 'B_signal_forward_rows', 'C_ablation_forward_rows')
        self.attempted[key] += count
        if signal_call:
            self.attempted['signal_forward_rows'] += count
        self.phase = operation
        self.in_flight = dict(operation=operation, counter_key=key, attempted_rows=count,
            status='entered_call_not_asserted_completed', **metadata)
        self.persist()  # Before the actual call, including on default TERM/KILL.
        result = action()
        self.counters[key] += count
        if signal_call:
            self.counters['signal_forward_rows'] += count
        self.last_completed_operation = dict(self.in_flight, status='returned_complete')
        self.in_flight = None
        self.persist()
        return result

    def family_complete(self, record):
        self.completed_family_rows += record['stop']-record['start']
        self.completed_families += 1
        self.phase = 'completed_A_B_C_family_all_three_hypotheses'
        with (self.store.root/'CHUNK_RECEIPTS.jsonl').open('a', encoding='utf8', newline='\n') as stream:
            stream.write(json.dumps(_clean(dict(record=record, counters=self.counters.copy())), allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.persist()

    def failed(self, error):
        payload = self.payload('failed_frozen_diagnosis')
        payload.update(error_type=type(error).__name__, error=str(error),
            attempted_in_flight_rows=None if self.in_flight is None else self.in_flight['attempted_rows'],
            in_flight_actual_completion='unknown; not added to completed counters',
            scientific_completion=False, native_exit_confirmation='reported separately by the supervisor')
        self.store.json('FAILED.json', payload)
        self.store.status(payload)


def _augment_A(output):
    """The original A interface lacks its pre-normalization denominator."""
    output = dict(output)
    signal, local = output['signal_pre'], output['signal_noise']
    n = len(signal)
    missing = signal.new_full((n,), float('nan'))
    output.update(base_shape_Q=missing, qsig_over_base_shape_Q=missing,
        stochastic_shape_voltage_RMS_V=signal.new_zeros(n), local_RMS_V=local.square().mean(1).sqrt(),
        pure_signal_pre_V=signal, mean_component_pre_V=signal-local,
        stochastic_shape_pre_V=torch.zeros_like(signal))
    return output


def model_snapshot(models, extension):
    extension.assert_frozen()
    for model in models.values():
        require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                '009 requires frozen eval parameters without retained gradients')
    return dict(response={k: state_digest(v.state_dict()) for k, v in models.items()},
                extension=state_digest(extension.state_dict()), noise005=state_digest(extension.model005.state_dict()))


def _row_values(output, signal, noise, observation):
    indices = observation['crop_indices']
    delta = observation['adc_carrier']-(signal+noise)
    values = {name: _array(output[name]) for name in ('qsig', 'qbar', 'time_scale', 'base_shape_Q',
        'qsig_over_base_shape_Q', 'stochastic_shape_voltage_RMS_V', 'local_RMS_V')}
    values.update({name: _array(observation[name]) for name in ('Q', 'tail', 'PSD', 'PSD_raw',
        'peak_index', 'global_peak_index', 'crop_start', 'crop_stop', 'baseline_mean_V')})
    for name, term in (('signal', signal), ('noise', noise), ('ADC_delta', delta)):
        crop = term.gather(1, indices)-term[:, :100].mean(1, keepdim=True)
        values['Q_'+name] = _array(crop.sum(1))
        values['tail_'+name] = _array(crop[:, 46:].sum(1))
    values['Q_signal_crop_migration'] = values['Q_signal']-values['qsig']
    q_error = np.abs(values['Q']-sum(values['Q_'+k] for k in ('signal', 'noise', 'ADC_delta'))).max()
    tail_error = np.abs(values['tail']-sum(values['tail_'+k] for k in ('signal', 'noise', 'ADC_delta'))).max()
    require(q_error <= 1e-10 and tail_error <= 1e-10, 'Observed same-crop component identity failed')
    values.update({key: _array(value) for key, value in observation['flags'].items()})
    values.update(morphology(_array(observation['window_V'])))
    return values, float(q_error), float(tail_error)


@torch.no_grad()
def _run_arrays_impl(data, models, ablation, extension, statistics, data_statistics, policy,
                     destination, *, device='cpu', freeze=None, artificial=False, store, progress):
    """Full storage/counter path; injectable frozen fixtures use no checkpoints."""
    p, r, chunk = policy['parents'], policy['repeats'], policy['chunk_rows']
    require(chunk % r == 0 and p % 8 == 0, '009 chunks must keep complete parent clusters')
    if not artificial:
        fixed_policy(policy)
    q = validate_train(data, source_rows=policy['TRAIN_counts']['source'], real_rows=policy['TRAIN_counts']['real'])
    selected, bins, source_counts, post_weights = select_parents(q,
        data_statistics['domains']['source']['Q256_eight_quantile_bin_edges'],
        parents_per_bin=p//8, seed=policy['seeds']['selection'])
    require(len(selected) == p and p*r == policy['matched_real_rows'], '009 matched row count changed')
    real_a, real_b = select_real(len(data['real']), p*r, policy['seeds']['real_selection'])
    store.guard(device)
    snapshot = model_snapshot(models, extension)
    if freeze is not None:
        verify_files(freeze)
    store.json('FROZEN_INPUTS.json', dict(schema='n01/mean-random009-freeze/v1', UTC=utc(),
        artificial_only=artificial, model_states=snapshot, input_binding=freeze,
        DEV_numeric_read=False, optimizer_updates=0, checkpoint_classes=dict(A='N01Response008', B='N01Response008Fix01', C='same_B_ablation'),
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=str(device),
            CPU_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_TF32=torch.backends.cudnn.allow_tf32, CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))))
    selection_values = dict(source_parent_rows=selected, source_parent_ids=data['source_ids'][selected],
        source_Q=q[selected], source_TRAIN_Q_bins=bins, full_source_TRAIN_bin_counts=source_counts,
        diagnostic_equal_parent_weights=np.ones(p), statistical_poststratification_parent_weights=post_weights,
        original_SOURCE_count_weights_available=np.asarray('original_source_weights' in data),
        REAL_reference_rows=real_a, REAL_reference_ids=data['real_ids'][real_a],
        REAL_control_rows=real_b, REAL_control_ids=data['real_ids'][real_b])
    for name in TRAIN_OPTIONAL_KEYS:
        if name in data:
            selection_values[name] = data[name][selected]
    store.npz('SELECTION.npz', **selection_values)
    store.json('SELECTION.json', dict(TRAIN_counts=policy['TRAIN_counts'], selected_source_parents=p,
        repeats=r, rows=p*r, source_TRAIN_Q_bins=list(np.bincount(bins, minlength=8)),
        source_full_TRAIN_Q_bins=list(source_counts), statistical_weights_are_model_inputs=False,
        original_physical_weights='unknown', original_SOURCE_count_weights='preserved if available; separate from statistical poststratification',
        distribution_scope='fixed equal-quota SOURCE TRAIN diagnostic subset',
        REAL_REAL_disjoint=True, DEV_numeric_read=False))
    bank, bank_values, bank_binding = explicit_bank(p*r, chunk, policy['seeds'])
    store.npz('EXPLICIT_LATENTS.npz', **bank_values)
    store.json('LATENT_BANK.json', bank_binding)
    parent_rows = np.repeat(np.arange(p), r)
    draw_rows = np.tile(np.arange(r), p)
    pieces = {v: {'signal_pre_V': []} for v in VIEWS}
    observed = {v: {h: {} for h in HYPOTHESES} for v in VIEWS}
    common_baseline = []
    counters = progress.counters
    identities = {v: dict(Q_component_absmax=0., tail_component_absmax=0.,
                         signal_prefix_absmax_V=0., canonical_charge_identity_absmax=0.) for v in VIEWS}
    chunk_records = []
    qfloor = float(statistics['loss']['q_floor'])
    for start in range(0, p*r, chunk):
        stop = min(start+chunk, p*r)
        store.guard(device)
        source = torch.as_tensor(data['source_carrier384'][selected[parent_rows[start:stop]]], dtype=torch.float64, device=device)
        response = {k: value[start:stop].to(device) for k, value in bank['response'].items()}
        noise_latents = {k: value[start:stop].to(device) for k, value in bank['noise005'].items()}
        noise = progress.execute('noise005_rows', stop-start,
            lambda: fixed_noises(extension, noise_latents, bank['extension'][start:stop].to(device)),
            operation='frozen005_and_three_extensions', start=start, stop=stop)
        outputs = {}
        outputs['A'] = _augment_A(progress.execute('A_signal_forward_rows', stop-start,
            lambda: models['A'].signal_forward(source, response),
            operation='A_original_signal_forward', view='A', start=start, stop=stop))
        outputs['B'] = progress.execute('B_signal_forward_rows', stop-start,
            lambda: ablation.signal_forward(source, response, disable=()),
            operation='B_normal_signal_forward_and_decomposition', view='B', start=start, stop=stop)
        outputs['C'] = progress.execute('C_ablation_forward_rows', stop-start,
            lambda: ablation.signal_forward(source, response, disable=ALL_OFF),
            operation='C_all_signal_random_off', view='C', start=start, stop=stop)
        baseline = None
        record = dict(start=start, stop=stop, source_parent_rows=parent_rows[start:stop].tolist(), outputs={})
        for view, output in outputs.items():
            signal = output['signal_pre']
            prefix_error = float(signal[:, :100].abs().max())
            canonical_error = float(output['canonical_charge_identity_residual'].abs().max())
            require(torch.isfinite(signal).all() and prefix_error <= 1e-12 and canonical_error <= 1e-10,
                    'Frozen signal finite/prefix/canonical identity failed')
            identities[view]['signal_prefix_absmax_V'] = max(identities[view]['signal_prefix_absmax_V'], prefix_error)
            identities[view]['canonical_charge_identity_absmax'] = max(identities[view]['canonical_charge_identity_absmax'], canonical_error)
            require(torch.allclose(signal, output['mean_component_pre_V']+
                    output['stochastic_shape_pre_V']+output['signal_noise'], rtol=0, atol=1e-12),
                    'Mean/shape/local voltage decomposition does not sum to signal')
            pieces[view]['signal_pre_V'].append(_array(signal))
            record['outputs'][view] = dict(signal_pre_V_sha256=array_sha(signal), hypotheses={})
            for h in HYPOTHESES:
                carrier = signal+noise[h]
                obs = progress.execute('hard_observation_rows', stop-start,
                    lambda: observe_generated_carrier(carrier, ste=False, q_floor=qfloor,
                        return_indices=True, return_carrier=True),
                    operation='hard_ADC_peak_crop_observation', view=view, hypothesis=h, start=start, stop=stop)
                if baseline is None:
                    baseline = obs['baseline100']
                    common_baseline.append(_array(baseline))
                require(torch.equal(baseline, obs['baseline100']), 'A/B/C or hypothesis electronic baseline is not paired')
                values, q_error, tail_error = _row_values(output, signal, noise[h], obs)
                identities[view]['Q_component_absmax'] = max(identities[view]['Q_component_absmax'], q_error)
                identities[view]['tail_component_absmax'] = max(identities[view]['tail_component_absmax'], tail_error)
                values.update(preADC_carrier_V=_array(carrier), hard_observed_waveform_V=_array(obs['window_V']))
                for key, value in values.items():
                    observed[view][h].setdefault(key, []).append(value)
                record['outputs'][view]['hypotheses'][h] = dict(
                    preADC_carrier_V_sha256=array_sha(carrier), hard_waveform_V_sha256=array_sha(obs['window_V']))
        chunk_records.append(record)
        ablation.assert_unchanged(deep=start == 0)
        require(model_snapshot(models, extension) == snapshot, 'Frozen tensors changed during 009 family')
        progress.family_complete(record)
        if start == 0:
            if freeze is not None:
                verify_files(freeze)
            store.json('ARTIFICIAL_START_PROOF.json' if artificial else 'START_PROOF.json', dict(
                schema='n01/mean-random009-start-proof/v1', diagnostic_id=DIAGNOSTIC_ID, UTC=utc(), pid=os.getpid(),
                status='first_complete_A_B_C_family_with_all_three_hypotheses',
                actual_family_rows=stop-start, counters=counters.copy(), model_states=snapshot,
                tensors_unchanged=True, manifest_and_bound_files_unchanged=freeze is not None,
                artificial_only=artificial, input_binding=freeze, first_chunk=record,
                resources=store.guard(device), DEV_numeric_read=False, optimizer_updates=0))
        store.guard(device)
    progress.phase = 'derived_means_covariances_and_distribution_statistics'
    progress.persist()
    raw = {v: {h: {k: np.concatenate(value) for k, value in observed[v][h].items()}
               for h in HYPOTHESES} for v in VIEWS}
    signals = {v: {k: np.concatenate(value) for k, value in pieces[v].items()} for v in VIEWS}
    del pieces, observed, outputs
    baseline = np.concatenate(common_baseline)
    store.npz('SHARED_BASELINE100_ADC.npz', baseline100_ADC=baseline,
              source_parent_row=parent_rows, draw=draw_rows)
    bootstrap_mass = bootstrap_counts(bins, policy['bootstrap_replicates'], policy['seeds']['bootstrap'])
    store.npz('PARENT_BOOTSTRAP_MASS.npz', mass=bootstrap_mass, source_TRAIN_Q_bin=bins)
    descriptions = {}
    for view in VIEWS:
        store.guard(device)
        store.npz(view+'_SIGNAL_DRAWS.npz', **signals[view], source_parent_row=parent_rows, draw=draw_rows,
                  source_parent_ids=data['source_ids'][selected[parent_rows]])
        moment = covariance_factor(signals[view]['signal_pre_V'].reshape(p, r, 384))
        store.npz(view+'_SIGNAL_MOMENTS.npz', **moment)
        descriptions[view] = {}
        for h in HYPOTHESES:
            values = raw[view][h]
            store.npz(view+'_'+h+'_OBSERVED_DRAWS.npz', **values, source_parent_row=parent_rows, draw=draw_rows)
            for domain, field, dimension in (('PREADC', 'preADC_carrier_V', 384),
                                            ('HARD', 'hard_observed_waveform_V', 256)):
                moment = covariance_factor(values[field].reshape(p, r, dimension))
                store.npz(view+'_'+h+'_'+domain+'_MOMENTS.npz', **moment)
            matrix = np.stack([values[k] for k in SCALAR_FIELDS], axis=1).reshape(p, r, len(SCALAR_FIELDS))
            store.npz(view+'_'+h+'_SCALAR_MOMENTS.npz', scalar_names=np.asarray(SCALAR_FIELDS), **scalar_covariance(matrix))
            descriptions[view][h] = {k: scalar_summary(v) for k, v in values.items() if np.asarray(v).ndim == 1}
            descriptions[view][h]['base_shape_amplification_available'] = view != 'A'
            store.guard(device)
    # Every D value is a mean of stored B rows. No model or observation call.
    store.npz('D_SIGNAL_MC_MEAN.npz', pure_signal_pre_V=signals['B']['signal_pre_V'].reshape(p, r, 384).mean(1),
              source_parent_ids=data['source_ids'][selected], model_forward_rows=np.asarray(0))
    pair_summary = {}
    for h in HYPOTHESES:
        store.npz('D_'+h+'_MC_MEANS.npz',
            mean_preADC_carrier_V=raw['B'][h]['preADC_carrier_V'].reshape(p, r, 384).mean(1),
            mean_of_individual_hard_observations_V=raw['B'][h]['hard_observed_waveform_V'].reshape(p, r, 256).mean(1),
            source_parent_ids=data['source_ids'][selected], model_forward_rows=np.asarray(0))
    for label, left, right in (('D_minus_C', 'B', 'C'), ('D_minus_A', 'B', 'A'), ('C_minus_A', 'C', 'A')):
        pair_summary[label] = {}
        delta = paired_statistics(signals[left]['signal_pre_V'].reshape(p, r, 384),
                                  signals[right]['signal_pre_V'].reshape(p, r, 384), bootstrap_mass)
        store.npz(label+'_SIGNAL_PAIRED.npz', **delta)
        pair_summary[label]['signal_mean_difference_RMS_V'] = float(np.sqrt(np.square(delta['equal_parent_mean']).mean()))
        pair_summary[label]['signal_global_MCSE_RMS_V'] = float(np.sqrt(np.square(delta['conditional_global_MonteCarlo_SE']).mean()))
        for h in HYPOTHESES:
            pair_summary[label][h] = {}
            for domain, field, dim in (('PREADC', 'preADC_carrier_V', 384), ('HARD', 'hard_observed_waveform_V', 256)):
                delta = paired_statistics(raw[left][h][field].reshape(p, r, dim),
                                          raw[right][h][field].reshape(p, r, dim), bootstrap_mass)
                store.npz(label+'_'+h+'_'+domain+'_PAIRED.npz', **delta)
                pair_summary[label][h][domain] = dict(
                    mean_difference_RMS_V=float(np.sqrt(np.square(delta['equal_parent_mean']).mean())),
                    conditional_global_MCSE_RMS_V=float(np.sqrt(np.square(delta['conditional_global_MonteCarlo_SE']).mean())))
            store.guard(device)
    objective = N01Loss008(loss_tensor_statistics(statistics['loss']),
        seed=2026101387, evaluation_seed=policy['seeds']['projections']).to(device)
    context = objective.create_evaluation_context()
    real_wave = data['real'][real_a]
    real_baseline = data['real_baseline100'][real_a]
    distances = {v: {h: distance_receipt(objective, raw[v][h]['hard_observed_waveform_V'],
        real_wave, baseline, real_baseline, context) for h in HYPOTHESES} for v in VIEWS}
    distances['REAL_REAL'] = distance_receipt(objective, data['real'][real_b], real_wave,
        data['real_baseline100'][real_b], real_baseline, context)
    require('D' not in distances, 'D must not enter distribution acceptance')
    real_control = dict(reference={k: scalar_summary(v) for k, v in morphology(real_wave).items()},
                        control={k: scalar_summary(v) for k, v in morphology(data['real'][real_b]).items()})
    store.json('DISTRIBUTION_DISTANCES.json', dict(distances=distances, projection_context=context.metadata(),
        matched_rows=p*r, D_distribution_acceptance=False, real_reference='disjoint equal-N empirical control, not an irreducible-error bound'))
    store.json('MORPHOLOGY_AND_READOUT_SUMMARIES.json', dict(views=descriptions, REAL_REAL=real_control,
        missing_rows='retained; crossing categories descriptive, not physical validity gates'))
    store.json('CHUNK_RECEIPTS.json', dict(chunks=chunk_records, counters=counters))
    ablation.assert_unchanged(deep=True)
    require(model_snapshot(models, extension) == snapshot, '009 changed frozen endpoint/noise tensors')
    if freeze is not None:
        verify_files(freeze)
    expected = dict(signal_forward_rows=3*p*r, hard_observation_rows=9*p*r, noise005_rows=p*r,
                    D_model_forward_rows=0, optimizer_updates=0)
    require(all(counters[k] == value for k, value in expected.items()), '009 actual work counters differ')
    resources = store.guard(device)
    progress.phase = 'complete_zero_update_frozen_TRAIN_diagnosis'
    progress.persist('frozen_TRAIN_mean_random009_complete' if not artificial else 'artificial_driver_semantics_complete')
    artifacts = {v.name: dict(sha256=sha256(v), bytes=v.stat().st_size)
                 for v in sorted(store.root.iterdir()) if v.is_file()}
    summary = dict(schema='n01/mean-random009-summary/v1', diagnostic_id=DIAGNOSTIC_ID, UTC=utc(),
        status='artificial_driver_semantics_complete' if artificial else 'frozen_TRAIN_mean_random009_complete',
        artificial_only=artificial, TRAIN_counts=policy['TRAIN_counts'], parents=p, repeats=r,
        effective_independent_source_parent_clusters=p, counters=counters, engineering=identities,
        model_states_before=snapshot, model_states_after=model_snapshot(models, extension),
        frozen_tensors_unchanged=True, bound_files_unchanged=freeze is not None, distances=distances,
        paired_mean_summary=pair_summary, D_model_forward_rows=0, D_distribution_acceptance=False,
        D_observation_semantics='mean of individual hard observations; never observation of mean carrier',
        covariance_storage='raw centered repeat factor F; exact sample covariance F.T @ F / (repeats-1), rank <= repeats-1',
        uncertainty=dict(MonteCarlo='per-parent paired repeat SE and conditional global MCSE',
            parent_bootstrap='stratified TRAIN parent clusters; fixed repeat bank retained within each parent',
            bootstrap_replicates=policy['bootstrap_replicates'], bootstrap_seed=policy['seeds']['bootstrap'],
            independent_acquisition_uncertainty='unknown', final_acceptance=False),
        source_distribution_scope='fixed equal-quota SOURCE TRAIN diagnostic subset; full-source poststratification weights saved separately',
        units=dict(waveform='V', waveform_sample_covariance='V^2', charge='V ns',
            PSD='dimensionless guarded ratio', time_scale='dimensionless', sample_interval_ns=1,
            base_shape_Q='internal normalization shape sum returned by frozen008; not observed Q',
            qsig_over_base_shape_Q='internal voltage amplification returned by frozen008'),
        DEV_numeric_read=False, scientific_status='response_match_not_established', scientific_pass_threshold=None,
        resources=resources, artifacts=artifacts)
    store.json('SUMMARY.json', summary)
    store.guard(device)
    return _clean(summary)


@torch.no_grad()
def run_arrays(data, models, ablation, extension, statistics, data_statistics, policy,
               destination, *, device='cpu', freeze=None, artificial=False, _store=None, _progress=None):
    store = _store or OutputStore(destination, policy)
    progress = _progress or RunProgress(store, artificial=artificial)
    try:
        return _run_arrays_impl(data, models, ablation, extension, statistics, data_statistics,
            policy, destination, device=device, freeze=freeze, artificial=artificial,
            store=store, progress=progress)
    except BaseException as error:
        progress.failed(error)
        raise


def load_frozen_models(paths, statistics, base_config, device):
    """Only called after the explicit independent 009 binding has passed."""
    noise = torch.load(paths['noise005'], map_location='cpu', weights_only=False)
    require(noise.get('update') == 5000 and noise.get('schema_version') == 5
            and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
            'Wrong bound frozen005 endpoint')
    extension = make_extension(noise, device)
    noise_digest = state_digest(extension.model005.state_dict())
    models, common = {}, []
    for view, name, cls, schema in (('A', 'deterministic', N01Response008, 'response008/v1'),
                                    ('B', 'full', N01Response008Fix01, 'response008_resume01/v1')):
        checkpoint = torch.load(paths[name], map_location='cpu', weights_only=False)
        require(checkpoint.get('schema') == schema and checkpoint.get('arm') == name
                and checkpoint.get('G_updates') == 19000
                and checkpoint.get('D_updates') == {h: 16000 for h in HYPOTHESES}
                and checkpoint.get('stage') == 'low_learning_rate', 'Wrong fixed19k checkpoint: ' + name)
        require(checkpoint.get('config') == base_config
                and statistics_digest(checkpoint['statistics']) == statistics_digest(statistics)
                and checkpoint.get('noise005_state_sha256') == noise_digest
                and checkpoint.get('noise_updates') == 0 and checkpoint.get('population_learning') is False,
                'Checkpoint frozen config/statistics/noise mismatch: ' + name)
        if name == 'full':
            require(checkpoint.get('model_implementation') == 'n01.model008_fix01.N01Response008Fix01'
                    and checkpoint.get('checkpoint_purpose') == 'final'
                    and checkpoint.get('complete_step_boundary') is True
                    and checkpoint.get('engineering_fixture') is False, 'Wrong full endpoint implementation/completion')
        model = cls(statistics['model'], name)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_stage('low_learning_rate')
        model.requires_grad_(False)
        model.eval()
        model.to(device)
        models[view] = model
        common.append(checkpoint['common_initial_state_sha256'])
        del checkpoint
    require(len(set(common)) == 1, 'A/B historical common initialization differs')
    from .ablation009 import FrozenSignalAblation009
    return models, FrozenSignalAblation009(models['B']), extension


def run(policy_path, bindings_path, output, device='cuda:0'):
    policy, paths, freeze = bind_inputs(policy_path, bindings_path, output)
    store = OutputStore(output, policy)
    progress = RunProgress(store)
    try:
        return _run_bound(policy, paths, freeze, store, progress, output, device)
    except BaseException as error:
        if not (store.root/'FAILED.json').exists():
            progress.failed(error)
        raise


def _run_bound(policy, paths, freeze, store, progress, output, device):
    store.json('BINDING.json', dict(schema='n01/mean-random009-initial-binding/v1',
        diagnostic_id=DIAGNOSTIC_ID, UTC=utc(), **freeze))
    statistics, data_statistics, base_config = [read_json(paths[k]) for k in ('statistics', 'data_statistics', 'base_config')]
    require(statistics.get('statistics_split') == 'TRAIN' and statistics['loss'].get('fit_split') == 'TRAIN'
            and statistics.get('DEV_waveforms_decoded') is False
            and statistics.get('prepared_dataset_sha256') == freeze['files_sha256'][str(paths['dataset'])]
            and statistics.get('data_statistics_sha256') == freeze['files_sha256'][str(paths['data_statistics'])],
            'Frozen statistics TRAIN/preparation binding differs')
    require(data_statistics.get('split') == 'TRAIN' and data_statistics.get('DEV_waveforms_decoded') is False
            and {k: data_statistics['domains'][k]['rows'] for k in ('source', 'real')} == policy['TRAIN_counts'],
            'Original domain preparation counts/roles changed')
    torch.set_num_threads(policy['CPU_threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze['CUDA_allocator_limit'] = _memory_limit(dict(gpu_memory_limit_GiB=policy['resource_limits']['GPU_GiB']), device)
    # Explicitly whitelist prepared TRAIN arrays; no raw acquisition/DEV decoder.
    progress.phase = 'decoding_whitelisted_prepared_TRAIN_members'
    progress.persist()
    with np.load(paths['dataset'], allow_pickle=False) as archive:
        data = read_train_members(archive)
    validate_train(data)
    progress.phase = 'loading_only_bound_frozen005_and_A_B_endpoints'
    progress.persist()
    models, ablation, extension = load_frozen_models(paths, statistics, base_config, device)
    verify_files(freeze)
    return run_arrays(data, models, ablation, extension, statistics, data_statistics,
                      policy, output, device=device, freeze=freeze, _store=store, _progress=progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = run(args.policy, args.bindings, args.output, args.device)
    print(json.dumps({key: result[key] for key in ('status', 'diagnostic_id', 'counters', 'frozen_tensors_unchanged', 'DEV_numeric_read')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
