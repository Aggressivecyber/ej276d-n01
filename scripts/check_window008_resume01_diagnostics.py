"""Native artificial-only checks for the preregistered endpoint addons.

--full-shape exercises the entire declared128x64/32x8 storage and counter path
using minimal frozen-contract models. --architecture-128 separately benchmarks
fresh initializations of the actual mixed model008/fix01 classes on128 analytic
SOURCE rows per arm. Neither option opens trained checkpoints or science data.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
import torch
from torch import nn

from n01 import diagnose_window008_resume01 as diag
from n01.evaluate_window008 import _memory_limit, readouts256, save_json, save_npz, source_bins
from n01.model008 import N01Response008, canonical_charge008
from n01.model008_fix01 import N01Response008Fix01
from n01.noise_validation import baseline_statistics
from n01.response008 import ARMS, HYPOTHESES, state_digest


def analytic_carriers(n):
    t = np.arange(384, dtype=np.float64)[None]
    k = np.arange(n, dtype=np.float64)[:, None]
    rise = 3.+k % 5
    decay = 20.+k % 71
    shape = np.where(t < 126, np.exp(-.5*np.square((t-126)/rise)), np.exp(-(t-126)/decay))
    shape[:, :100] = 0.
    return (.008+.042*(k+1)/(n+1))*shape


class ArtificialFrozenResponse(nn.Module):
    """Minimal endpoint contract fixture, not a trained or real architecture."""
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.register_buffer('gain', torch.tensor(dict(deterministic=1., full=1.02, simple=.99)[arm], dtype=torch.float64))
        self.register_buffer('position', torch.arange(384, dtype=torch.float64))

    def signal_forward(self, source, latents):
        centered = source-source[:, :100].mean(1, keepdim=True)
        q = centered[:, 110:366].sum(1); qbar = self.gain*q
        stochastic = self.arm != 'deterministic'
        qsig = qbar*torch.exp(.02*latents['phys'][:, 0].double()-.5*.02**2) if stochastic else qbar
        scale = torch.exp(.02*latents['phys'][:, 1].double()) if stochastic else torch.ones_like(q)
        unit = centered/q[:, None]
        local = torch.zeros_like(centered)
        if self.arm == 'full':
            local = 2e-5*latents['local_white'][:, 0, :384].double()
            local[:, :100] = 0.
            projection = torch.zeros(384, dtype=torch.float64, device=source.device)
            projection[110:366] = 1./256
            local = local-local[:, 110:366].sum(1, keepdim=True)*projection[None]
        signal = qsig[:, None]*unit+local
        canonical = signal[:, 110:366].sum(1)
        zeros = torch.zeros_like(q, dtype=torch.int64)
        return dict(signal_pre=signal, qbar=qbar, qsig=qsig, q_source=q, time_scale=scale,
                    canonical_charge=canonical, canonical_local_charge=local[:, 110:366].sum(1),
                    canonical_charge_identity_residual=canonical-qsig,
                    query_outside_carrier=zeros, beyond_extension=zeros)


class ArtificialNoise005(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('log_scale_std', torch.tensor(.1, dtype=torch.float64))


class ArtificialFrozenExtension(nn.Module):
    """Explicitly artificial shared-prefix law with no optimizer or fit."""
    def __init__(self):
        super().__init__(); self.model005 = ArtificialNoise005()
        self.register_buffer('position', torch.arange(284, dtype=torch.float64))
        self._versions = {name: value._version for name, value in self.named_buffers()}

    def assert_frozen(self):
        assert {name: value._version for name, value in self.named_buffers()} == self._versions

    def sample_prefix(self, values):
        self.assert_frozen(); eta = self.model005.log_scale_std
        scale = torch.exp(eta*values['baseline_scale'].double()-eta.square())[:, 0]
        prefix = 2e-4*scale*values['baseline_gaussian'][:, 0].double()
        return dict(prefix_pre=prefix, scale=scale)

    def extend_prefix(self, prefix, scale, epsilon, hypothesis):
        factor = dict(fast=1., slow=1.3, common_level=.8)[hypothesis]
        tail = factor*2e-4*scale*epsilon[:, 0].double()
        if hypothesis != 'fast':
            tau = 32. if hypothesis == 'slow' else 300.
            tail = tail+prefix[:, -1, None]*torch.exp(-self.position[None]/tau)
        return dict(noise_pre=torch.cat((prefix, tail), 1))


class PeakSampler:
    def __init__(self):
        self.stop = threading.Event(); self.peak = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            total = current_RSS()
            self.peak = max(self.peak, total)
            self.stop.wait(.05)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join()


def current_RSS():
    if psutil is not None:
        process = psutil.Process(); total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        values = Counters(); values.cb = ctypes.sizeof(values)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True); kernel.GetCurrentProcess.restype = wintypes.HANDLE
        memory = ctypes.WinDLL('psapi', use_last_error=True)
        memory.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not memory.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(values.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def available_RAM():
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Status(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)]+[(name, ctypes.c_ulonglong) for name in (
                'ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
        values = Status(); values.dwLength = ctypes.sizeof(values)
        if not ctypes.WinDLL('kernel32', use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(values)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(values.ullAvailPhys)
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Cannot read available RAM for the artificial preflight')


def expect_error(callable_, kind=ValueError):
    try:
        callable_()
    except kind:
        return
    raise AssertionError('Expected '+kind.__name__)


def helper_checks(policy, out):
    ids = np.arange(512); chosen = diag.select_repeat_parents(ids, policy)
    expected = np.random.default_rng(2026101399).permutation(512)[:128]
    np.testing.assert_array_equal(chosen, expected)
    assert len(np.unique(chosen)) == 128
    malformed = copy.deepcopy(policy); malformed['seeds']['repeat_selection'] += 1
    expect_error(lambda: diag.fixed_policy(malformed))
    malformed = copy.deepcopy(policy); malformed['C2ST_updates'] = 1
    expect_error(lambda: diag.fixed_policy(malformed))
    x = analytic_carriers(512)
    edges = np.quantile(x[:, 110:366].sum(1), np.linspace(0, 1, 9))
    intervention = diag.shape_interventions(x, ids, chosen, edges, policy)
    np.testing.assert_allclose(intervention['donor_same_Q_carrier384'][:, 110:366].sum(1),
                               intervention['SOURCE_Q'], rtol=0, atol=1e-12)
    for k, q in enumerate(edges[[2, 4, 6]]):
        np.testing.assert_allclose(intervention['trend_carriers384'][k, :, 110:366].sum(1), q, rtol=0, atol=1e-12)
    tiny_policy = copy.deepcopy(policy); tiny_policy['shape_intervention']['parents'] = 1
    tied = np.repeat(x[:1], 3, axis=0)
    tie = diag.shape_interventions(tied, np.array([9, 3, 7]), [2], edges, tiny_policy)
    assert tie['donor_ids'].tolist() == [3]
    assert tie['unit_shape_absmax_difference'][0] == 0
    triangle = np.zeros((3, 256)); triangle[0, :17] = np.arange(17)/16
    triangle[0, 16:49] = 1.-np.arange(33)/32
    triangle[2] = 1.
    time_result = diag.temporal_readouts(triangle, .01)
    np.testing.assert_allclose(time_result['rise10_90_ns'][0], 12.8, atol=1e-12)
    np.testing.assert_allclose(time_result['FWHM_ns'][0], 24., atol=1e-12)
    assert not time_result['rise10_90_defined'][1] and not time_result['FWHM_defined'][2]
    nan_wave = triangle.copy(); nan_wave[0, 0] = np.nan
    expect_error(lambda: diag.temporal_readouts(nan_wave, .01))
    rng = np.random.default_rng(753)
    scalar = rng.normal(size=(3, 8, 4)); scalar[0, 0:2, 2] = np.nan; scalar[1, :, 3] = np.nan
    stat = diag.sample_statistics(scalar)
    for p in range(3):
        for i in range(4):
            for j in range(4):
                finite = np.isfinite(scalar[p, :, i]) & np.isfinite(scalar[p, :, j])
                assert stat['covariance_defined_draws'][p, i, j] == finite.sum()
                if finite.sum() > 1:
                    cov = np.cov(scalar[p, finite, i], scalar[p, finite, j], ddof=1)[0, 1]
                    np.testing.assert_allclose(stat['covariance'][p, i, j], cov, rtol=1e-13, atol=1e-14)
                else:
                    assert np.isnan(stat['covariance'][p, i, j])
    narrow = (1e6+1e-5*np.arange(8))[None, :, None]
    narrow = np.concatenate((narrow, 2*narrow), axis=2)
    narrow_stat = diag.sample_statistics(narrow)
    np.testing.assert_allclose(narrow_stat['covariance'][0], np.cov(narrow[0].T, ddof=1), rtol=1e-12, atol=1e-20)
    baseline = 1e-4*rng.normal(size=(12, 100))
    original = baseline_statistics(baseline); extended = diag.baseline_readouts(baseline)
    np.testing.assert_allclose(extended['baseline_ACF_lag0_99'][:, 1:21].mean(0), original['acf_mean'], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(extended['baseline_power_fraction5'].mean(0), original['psd_band_fraction_mean'], rtol=1e-13, atol=1e-14)
    assert extended['baseline_ACF_lag0_99'].shape == (12, 100)
    assert policy['baseline_statistics']['variance_floor_V2'] == diag.VARIANCE_FLOOR
    weights = diag.population_weights(np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]))
    np.testing.assert_allclose(weights.sum(1), 9., atol=1e-14)
    assert (weights > 0).all()
    values = np.array([1., 3., np.nan, 5.]); w = np.array([1., 2., 3., 4.])
    weighted = diag.weighted_statistics(values, w)
    np.testing.assert_allclose(weighted['mean'], 27/7)
    assert weighted['undefined_rows'].tolist() == [1] and weighted['undefined_weight'].tolist() == [3.]
    expect_error(lambda: diag.weighted_statistics(values, np.array([1., 2., 0., 4.])))
    # Missing/unsupported SOURCE bin moments remain explicitly undefined.
    bins = np.array([0, 0, 2, 2, 7, 7]); y = rng.normal(size=(6, 256)); m = np.full((8, 256), np.nan); s = m.copy()
    rows = np.bincount(bins, minlength=8)
    for b in (0, 2, 7):
        m[b] = y[bins == b].mean(0); s[b] = y[bins == b].std(0, ddof=0)
    bw = 1+.5*(2*np.arange(8)-7)/7; mix = diag.exact_weighted_waveform(m, s, rows, bw)
    direct_mean = np.average(y, axis=0, weights=bw[bins])
    direct_variance = np.average(np.square(y-direct_mean), axis=0, weights=bw[bins])
    np.testing.assert_allclose(mix['mean'], direct_mean, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(mix['variance'], direct_variance, rtol=2e-13, atol=2e-14)
    # All generated scientific entry paths must fail before numerical input/forward without freeze.
    destination = out/'missing_freeze'; destination.mkdir()
    cfg = dict(seeds=diag.SEEDS)
    counts = diag.ForwardCounters()
    expect_error(lambda: diag.run_diagnostics_arrays({}, {}, None, {'loss': {'q_floor': .01}}, cfg,
                 destination, policy, artificial=False, counters=counts))
    assert counts.groups == {}
    # Exercise the actual generation/observation/decomposition path at tiny shape.
    models = {a: ArtificialFrozenResponse(a).eval().requires_grad_(False) for a in ARMS}
    extension = ArtificialFrozenExtension().eval().requires_grad_(False)
    latent_bank, latent_binding = diag.explicit_latent_bank(2, 2, [2026101403, 2026101404, 2026101405])
    families = np.asarray((x[chosen[:2]], x[chosen[:2]]))
    tiny_counts = diag.ForwardCounters()
    generated = diag.generate_family('tiny_pairs', families, ids[chosen[:2]], 2, latent_bank, models,
                                     extension, {'loss': {'q_floor': .01}}, policy, tiny_counts)
    assert tiny_counts.groups['tiny_pairs']['signal_forward_rows'] == 24
    assert tiny_counts.groups['tiny_pairs']['observation_rows'] == 72
    assert generated['engineering']['baseline_shared']
    for a in ARMS:
        for h in HYPOTHESES:
            np.testing.assert_array_equal(generated['waveform256'][0, a, h], generated['waveform256'][1, a, h])
            np.testing.assert_allclose(generated['values'][0, a, h]['Q_identity_residual'], 0., atol=1e-12)
            np.testing.assert_allclose(generated['values'][0, a, h]['tail_identity_residual'], 0., atol=1e-12)
    permuted = diag.generate_family('tiny_order_replay', families[:1, ::-1], ids[chosen[:2]][::-1], 2,
        latent_bank, models, extension, {'loss': {'q_floor': .01}}, policy, tiny_counts, latent_parent_order=np.array([1, 0]))
    diag.compare_identity_order(generated, permuted, policy, inverse_order=np.array([1, 0]))
    assert tiny_counts.groups['tiny_order_replay']['signal_forward_rows'] == 12
    assert latent_binding['values_sha256'] == {key: diag.array_sha(value) for key, value in latent_bank.items()}
    return dict(selection_exact=True, donor_tie_original_ID_exact=True, zero_intervention_strength_retained=True,
                canonical_Q_and_same_shape_nodes_exact=True, temporal_interpolation_known_answer=True,
                undefined_time_rows_retained=True, sample_covariance_pair_counts_verified=True,
                narrow_covariance_stable=True, original005_ACF20_and_bands_reproduced=True,
                positive_weights_and_missing_mass_retained=True, exact_bin_wave_mixture_verified=True,
                missing_source_freeze_rejected_before_scientific_forward=True,
                paired_generation_common_baseline_and_decomposition=True,
                explicit_latent_order_replay_exact=True, artificial_family_counters=tiny_counts.groups)


def artificial_primary(out, dev_carrier, statistics):
    """Direct artificial arrays for population D; no response call is used."""
    out.mkdir()
    for split, n in (('TRAIN', 6593), ('DEV', 512)):
        carrier = analytic_carriers(n) if split == 'TRAIN' else dev_carrier
        qsource = carrier[:, 110:366].sum(1); bins = source_bins(qsource, statistics)
        parents = np.repeat(np.arange(n), 4); draw = np.tile(np.arange(4), n)
        flat = dict(source_parent_ids=np.arange(n)+(100000 if split == 'TRAIN' else 0), source_Q=qsource,
                    source_bins=bins, source_parent_row=parents, draw=draw)
        curves = {}; position = np.arange(256)[None]
        noise = 1e-5*np.sin(position/13.+draw[:, None])
        for ai, a in enumerate(ARMS):
            for hi, h in enumerate(HYPOTHESES):
                y = carrier[parents, 110:366]*(1.+.02*ai+.004*hi)+noise
                y[0] = 0.  # Retain one exactly undefined raw-PSD row.
                readout = readouts256(y, statistics['loss']['q_floor']); label = a+'_'+h
                flat.update({label+'_'+key: value for key, value in readout.items()})
                curves[label+'_mean'] = y.mean(0); curves[label+'_std'] = y.std(0, ddof=0)
                base = label+'_fixed_SOURCE_Q_in_SOURCE_TRAIN_bins'
                bm, bs = np.full((8, 256), np.nan), np.full((8, 256), np.nan)
                rows = np.bincount(bins[parents], minlength=8)
                for b in range(8):
                    if rows[b]:
                        bm[b] = y[bins[parents] == b].mean(0); bs[b] = y[bins[parents] == b].std(0, ddof=0)
                curves[base+'_mean'], curves[base+'_std'], curves[base+'_rows'] = bm, bs, rows
        save_npz(out/(split+'_observables.npz'), **flat)
        save_npz(out/(split+'_curves.npz'), **curves)


@torch.no_grad()
def architecture128(device, out):
    """Actual model classes; freshly initialized on analytic inputs only."""
    stats = dict(q_ref=3., a_ref=.1, log_q_min=math.log(.1), log_q_max=math.log(100.), detector_condition=0.)
    models = {}
    for arm in ARMS:
        torch.manual_seed(2026101380)
        cls = N01Response008 if arm == 'deterministic' else N01Response008Fix01
        model = cls(stats, arm)
        model.set_stage('low_learning_rate'); model.requires_grad_(False).eval().to(device); models[arm] = model
    carrier = torch.as_tensor(analytic_carriers(128), device=device)
    generator = torch.Generator(device='cpu').manual_seed(32109)
    latents = {key: torch.randn((128,)+shape, generator=generator, dtype=torch.float32).to(device)
               for key, shape in N01Response008.noise_shapes.items()}
    receipts = {}; start = time.perf_counter()
    for arm, model in models.items():
        before = state_digest(model.state_dict()); output = model.signal_forward(carrier, latents)
        assert torch.isfinite(output['signal_pre']).all() and (output['signal_pre'][:, :100] == 0).all()
        assert output['canonical_charge_identity_residual'].abs().max() < 1e-8
        assert state_digest(model.state_dict()) == before
        receipts[arm] = dict(rows=128, model_class=type(model).__name__, state_preserved=True,
                             signal_sha256=diag.array_sha(output['signal_pre']))
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    save_json(out/'ARCHITECTURE128.json', dict(arms=receipts, artificial_signal_forward_rows=384,
        scientific_signal_forward_rows=0, trained_checkpoint_reads=0, source_or_REAL_science_array_reads=0,
        wall_seconds=elapsed, classifier_or_optimizer_updates=0, DEV_numeric_decodes=0,
        interpretation='Freshly initialized actual architecture on128 artificial rows; no trained-endpoint response evidence.'))
    return dict(rows=384, seconds=elapsed, arms=receipts)


def reopen_fullshape(out):
    artifacts = out/'artifacts'; summary = json.loads((artifacts/'SUMMARY.json').read_text(encoding='utf8'))
    for name, expected in summary['artifact_sha256'].items():
        assert diag.sha256(artifacts/name) == expected, name
    max_cov_error = 0.; npz_members = 0; undefined_raw = 0
    for path in sorted(artifacts.glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]; npz_members += 1
                assert value.dtype.kind != 'O', (path.name, key)
                if value.dtype.kind in 'fc':
                    assert not np.isinf(value).any(), (path.name, key)
            if path.name.startswith('REPEAT_') and not path.name.endswith('_COMMON.npz'):
                y = archive['waveform256']; cov = archive['waveform256_covariance']
                assert y.shape == (128, 64, 256) and cov.shape == (128, 256, 256)
                expected = np.cov(y[0].T, ddof=1)
                error = float(np.abs(cov[0]-expected).max()); max_cov_error = max(max_cov_error, error)
                np.testing.assert_allclose(cov[0], expected, rtol=2e-12, atol=2e-14)
                np.testing.assert_allclose(archive['Var_Q_given_x'][0], np.var(archive['Q'][0], ddof=1), rtol=2e-12, atol=2e-14)
                np.testing.assert_allclose(archive['Cov_Q_PSD_given_x'][0], np.cov(archive['Q'][0], archive['PSD'][0], ddof=1)[0, 1], rtol=2e-12, atol=2e-14)
                assert (archive['scalar_defined_draws'][:, 0] == 64).all()
                assert (archive['scalar_covariance_defined_draws'][:, 0, 2] == 64).all()
            if path.name.startswith('POPULATION_'):
                split = 'TRAIN' if 'TRAIN' in path.name else 'DEV'
                report = json.loads((artifacts/('POPULATION_'+split+'.json')).read_text())
                for scheme in ('uniform', 'tilt_plus', 'tilt_minus'):
                    for arm in ARMS:
                        for h in HYPOTHESES:
                            undefined_raw += report['schemes'][scheme]['arms'][arm][h]['population']['PSD_raw']['undefined_rows'][0]
                            assert report['schemes'][scheme]['arms'][arm][h]['output_Q_conditional_weighted_waveform_moments'] is None
    assert summary['actual_forward_counts']['primary']['signal_forward_rows'] == 28416
    assert summary['actual_forward_counts']['primary']['observation_rows'] == 85248
    assert summary['actual_forward_counts']['total_signal_forward_rows'] == 29952
    assert summary['actual_forward_counts']['total_observation_rows'] == 89856
    assert undefined_raw == 54
    return dict(all_NPZ_allow_pickle_false_readable=True, NPZ_members=npz_members,
                full_conditional_wave_covariance_known_answer_absmax=max_cov_error,
                conditional_Q_PSD_covariance_known_answer=True, undefined_main_raw_PSD_retained=undefined_raw,
                full_counters_exact=True, all_SUMMARY_artifact_hashes_rechecked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--full-shape', action='store_true'); parser.add_argument('--architecture-128', action='store_true')
    args = parser.parse_args(); out = Path(args.out).resolve()
    if out.exists() or (os.name == 'nt' and out.drive.upper() not in ('E:', 'D:')):
        raise ValueError('An unused E/D or registered remote artifact path is required')
    out.mkdir(parents=True); start = time.perf_counter(); counts = diag.ForwardCounters()
    policy = diag.fixed_policy(ROOT/'window008_resume01_diagnostics_policy.json')
    source_hashes = {str(path.relative_to(ROOT)): diag.sha256(path) for path in
        [ROOT/'src/n01/diagnose_window008_resume01.py', ROOT/'scripts/check_window008_resume01_diagnostics.py',
         ROOT/'window008_resume01_diagnostics_policy.json']}
    free_disk, free_RAM = shutil.disk_usage(out).free, available_RAM()
    minimum_disk = (8 if args.full_shape or args.architecture_128 else .01)*1024**3
    minimum_RAM = (10 if args.full_shape or args.architecture_128 else .5)*1024**3
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(12); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        if free_disk < minimum_disk or free_RAM < minimum_RAM:
            raise RuntimeError('Artificial benchmark preflight resource gate failed: available disk/RAM='+str((free_disk, free_RAM)))
        allocator = _memory_limit({'gpu_memory_limit_GiB': 10}, args.device)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with PeakSampler() as peak:
            semantic = helper_checks(policy, out)
            architecture = architecture128(args.device, out) if args.architecture_128 else dict(status='not_requested')
            summary, readback = None, None
            if args.full_shape:
                x = analytic_carriers(512); ids = np.arange(512)
                train_q = analytic_carriers(6593)[:, 110:366].sum(1)
                edges = np.quantile(train_q, np.linspace(0, 1, 9))
                statistics = dict(domains={'source': {'Q256_eight_quantile_bin_edges': edges.tolist()}},
                    loss=dict(q_floor=.01, q_reference=1., charge_bin_boundaries=np.linspace(-2, 2, 7).tolist()))
                artificial_primary(out/'artificial_primary', x, statistics)
                models = {a: ArtificialFrozenResponse(a).eval().requires_grad_(False).to(args.device) for a in ARMS}
                extension = ArtificialFrozenExtension().eval().requires_grad_(False).to(args.device)
                before = {a: state_digest(m.state_dict()) for a, m in models.items()}; n_before = state_digest(extension.state_dict())
                artifacts = out/'artifacts'; artifacts.mkdir()
                summary = diag.run_diagnostics_arrays(dict(source_ids=ids, source_carrier384=x), models, extension,
                    statistics, {'seeds': diag.SEEDS}, artifacts, policy, device=args.device, artificial=True,
                    counters=counts, primary_artifacts=out/'artificial_primary')
                assert {a: state_digest(m.state_dict()) for a, m in models.items()} == before
                assert state_digest(extension.state_dict()) == n_before
                readback = reopen_fullshape(out)
            elapsed = time.perf_counter()-start
        gpu = dict(allocated_peak_bytes=None, reserved_peak_bytes=None, allocator_bound=allocator,
                   driver_context_not_in_torch_peak=True)
        if torch.device(args.device).type == 'cuda':
            gpu.update(allocated_peak_bytes=int(torch.cuda.max_memory_allocated()), reserved_peak_bytes=int(torch.cuda.max_memory_reserved()))
        size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        assert peak.peak < 8*1024**3 and size < 3*1024**3
        for name, expected in source_hashes.items():
            assert diag.sha256(ROOT/name) == expected
        receipt = dict(schema='n01/window008-resume01-diagnostics-artificial-check/v1', status='passed', native_exit_code=0,
            device=args.device, full_shape=args.full_shape, source_sha256=source_hashes, semantic_checks=semantic,
            architecture128=architecture, full_shape_readback=readback,
            actual_artificial_forward_counters=counts.groups, scientific_signal_forward_rows=0,
            actual_SOURCE_REAL_DEV_numeric_decodes=0, trained_checkpoint_reads=0,
            optimizer_updates=0, C2ST_updates=0, population_updates=0, noise_updates=0,
            scientific_status='response_match_not_established', wall_seconds=elapsed,
            preflight=dict(available_disk_bytes=free_disk, available_RAM_bytes=free_RAM,
                           minimum_disk_bytes=minimum_disk, minimum_RAM_bytes=minimum_RAM,
                           scope='full resource benchmark' if args.full_shape or args.architecture_128 else 'tiny semantic helpers only; no full resource proof'),
            sampled_process_tree_peak_RSS_bytes=peak.peak, RSS_sample_seconds=.05, GPU=gpu,
            RSS_scope='process plus children via psutil' if psutil is not None else 'single-process OS high water; fixture creates no subprocesses; independent supervisor enforces process tree',
            output_bytes_before_receipt=size, output_GiB=size/1024**3,
            full_shape_model='minimal artificial frozen contract' if args.full_shape else None,
            actual_architecture128_is_random_initialized=args.architecture_128,
            resource_scope='CPU covariance/storage and artificial generation only. Trained endpoint runtime and driver GPU peak require separately supervised execution.')
        save_json(out/'RECEIPT.json', receipt)
        print(json.dumps(dict(status='passed', full_shape=args.full_shape, device=args.device,
                             wall_seconds=elapsed, peak_RSS_GiB=peak.peak/1024**3, output_GiB=size/1024**3,
                             scientific_signal_forward_rows=0, actual_SOURCE_REAL_DEV_numeric_decodes=0)))
    except Exception as error:
        save_json(out/'FAILED.json', dict(status='artificial_fixture_failed_preserved', error_type=type(error).__name__,
            error=str(error), actual_artificial_forward_counters=counts.groups, scientific_signal_forward_rows=0,
            actual_SOURCE_REAL_DEV_numeric_decodes=0, optimizer_updates=0, C2ST_updates=0))
        raise


if __name__ == '__main__':
    main()
