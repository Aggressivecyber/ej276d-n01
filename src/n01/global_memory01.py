"""TRAIN-only multiscale joint-CDF memory and a separately trainable mean head.

Historical models and the hard observation operator are unchanged. Memory is
supervision only: inference consumes SOURCE-derived features and explicit z.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn

from .joint01_features import JointFeatureMap01, observable_blocks
from .joint01_model import JointResponse01
from .losses008 import ADC_STEP
from .model008 import canonical_charge008, relative_time_warp008


def add_peak_selection_gradient(obs):
    """Keep hard values exact; add a one-ADC-code soft peak-selection derivative.

    Candidate windows are detached, so the existing selected-value derivative
    is not counted twice. This approximates selection sensitivity only.
    """
    adc = obs['carrier_adc']
    probability = (adc[:, 112:144] / ADC_STEP).softmax(1)
    indices = torch.arange(112, 144, device=adc.device)[:, None] - 16 + torch.arange(256, device=adc.device)
    centered = (adc - obs['baseline_mean_V'][:, None]).detach()
    windows = centered[:, indices]
    expected = (probability[:, :, None] * windows).sum(1)
    correction = expected - expected.detach()
    obs['window_V'] = obs['waveform256'] = obs['window_V'] + correction
    obs['Q'] = obs['Q'] + correction.sum(1)
    obs['tail'] = obs['tail'] + correction[:, 46:].sum(1)
    q, tail = obs['Q'], obs['tail']
    safe = torch.where(q != 0, q, torch.ones_like(q))
    obs['PSD_raw'] = torch.where(q != 0, tail / safe, torch.full_like(q, float('nan')))
    floor = obs['q_floor']
    protected = torch.where(q < 0, -torch.full_like(q, floor), torch.full_like(q, floor))
    obs['PSD'] = tail / torch.where(q.abs() < floor, protected, q)
    return obs


class GlobalDistributionMemory01:
    """Joint local-Q masses/CDFs; all population rows retain their weight."""

    def __init__(self, real, *, q_floor, levels=(16, 32, 64), thresholds=128,
                 feature_seed=2026102107):
        self.q_floor = float(q_floor)
        self.levels = tuple(levels)
        self.threshold_count = thresholds
        self.q_ref = real['Q'].median().detach()
        self.tail_ref = real['tail'].abs().median().detach()
        assert self.q_ref > 0 and self.tail_ref > 0
        u = torch.asinh(real['Q'] / self.q_ref)
        valid = real['Q'].abs() >= self.q_floor
        self.centers = [torch.linspace(u.min(), u.max(), k, device=u.device,
                                      dtype=u.dtype) for k in levels]
        self.thresholds = []
        for y in (real['PSD_raw'][valid], torch.asinh(real['tail'] / self.tail_ref)):
            bounds = torch.quantile(y, y.new_tensor([.005, .995]))
            assert bounds[1] > bounds[0]
            self.thresholds.append(torch.linspace(*bounds, thresholds,
                                                 device=u.device, dtype=u.dtype))
        self.joint = JointFeatureMap01(observable_blocks(real), seed=feature_seed)
        self.sections, pieces, at = {}, [], 0
        for name, lengths, weights in (
            ('Q_mass', list(levels), [k / len(levels) for k in levels]),
            ('PSD_CDF', [k * thresholds for k in levels],
             [k / (len(levels) * thresholds) for k in levels]),
            ('Q_tail_CDF', [k * thresholds for k in levels],
             [k / (len(levels) * thresholds) for k in levels]),
            ('undefined_mass', [3], [1. / 3]),
            ('native_joint', [272], [1. / 272]),
        ):
            self.sections[name] = slice(at, at + sum(lengths))
            for length, weight in zip(lengths, weights):
                pieces.append(u.new_full((length,), weight))
            at += sum(lengths)
        self.weights = torch.cat(pieces)
        self.target = self.mean(real).detach()
        self.normalization = {name: 1. for name in self.sections}

    def sums(self, obs):
        q, tail = obs['Q'], obs['tail']
        u = torch.asinh(q / self.q_ref)
        # Q near zero is retained in Q/tail and explicit population masses.
        valid = q.abs() >= self.q_floor
        safe_q = torch.where(valid, q, torch.ones_like(q))
        psd = tail / safe_q
        local = [(-.5 * ((u[:, None] - c) / (c[1] - c[0])).square()).softmax(1)
                 for c in self.centers]
        parts = [a.sum(0) for a in local]
        for y, grid, mask in (
            (psd, self.thresholds[0], valid),
            (torch.asinh(tail / self.tail_ref), self.thresholds[1],
             torch.ones_like(valid)),
        ):
            cdf = ((grid[None] - y[:, None]) / (.5 * (grid[1] - grid[0]))).sigmoid()
            cdf = cdf * mask[:, None]
            parts.extend((a.T @ cdf).flatten() for a in local)
        parts.append(torch.stack(((~valid).sum(), (q == 0).sum(), (q <= 0).sum())).double())
        # Linear whitening permits exact streaming sums and cross-bank gradients.
        features = self.joint.features(observable_blocks(obs))
        parts.append(self.joint.whiten(features.sum(0)))
        return torch.cat(parts)

    def mean(self, obs):
        return self.sums(obs) / len(obs['Q'])

    def losses(self, a, b):
        product = self.weights * (a - self.target) * (b - self.target)
        return {name: product[selection].sum() for name, selection in self.sections.items()}

    def coefficients(self, a, b):
        return (self.weights * (b - self.target)).detach(), (self.weights * (a - self.target)).detach()

    def wave_coefficients(self, a, b):
        """Exact independent-bank waveform-block derivatives in memory space."""
        native = self.sections['native_joint']
        wave = self.joint.block_slices['wave']
        lower = self.joint.cholesky
        variance = lower.square().sum(1)[wave]
        result = []
        for other in (b, a):
            raw = lower @ (other - self.target)[native]
            coefficient = torch.zeros_like(self.target)
            coefficient[native] = lower[wave].T @ (raw[wave] / variance / len(variance))
            result.append(coefficient.detach())
        return tuple(result)

    def normalize_from_initial(self, a, b):
        """Freeze relative scales from initial TRAIN-only population residuals."""
        error = ((a + b) / 2 - self.target).square()
        for name, selection in self.sections.items():
            value = float((self.weights[selection] * error[selection]).sum())
            # These three initially zero event masses retain their absolute scale.
            denominator = 1. if name == 'undefined_mass' else max(value, 1e-8)
            self.weights[selection] /= denominator
            self.normalization[name] = denominator

    def positive_wave_coefficients(self, a, b):
        """Derivative of the actual guard: squared error of the two-bank mean."""
        native = self.sections['native_joint']
        wave = self.joint.block_slices['wave']
        lower = self.joint.cholesky
        variance = lower.square().sum(1)[wave]
        raw = lower @ ((a + b) / 2 - self.target)[native]
        coefficient = torch.zeros_like(self.target)
        coefficient[native] = lower[wave].T @ (raw[wave] / variance / len(variance))
        return coefficient.detach(), coefficient.detach()

    def export(self):
        return dict(q_floor=self.q_floor, q_ref=self.q_ref.cpu(), tail_ref=self.tail_ref.cpu(),
                    levels=self.levels, thresholds=[v.cpu() for v in self.thresholds],
                    centers=[v.cpu() for v in self.centers], target=self.target.cpu(),
                    weights=self.weights.cpu(), normalization=self.normalization,
                    joint_center={k: v.cpu() for k, v in self.joint.center.items()},
                    joint_scale={k: v.cpu() for k, v in self.joint.scale.items()},
                    joint_projections={k: v.cpu() for k, v in self.joint.projections.items()},
                    joint_cholesky=self.joint.cholesky.cpu())


class MeanHeadRepair01(nn.Module):
    """Original mean formula with copied trainable terminal modules only."""

    def __init__(self, baseline, encoding, filtered):
        super().__init__()
        self.mean_head = copy.deepcopy(baseline.mean_head).requires_grad_(True)
        self.mean_time = copy.deepcopy(baseline.mean_time).requires_grad_(False)
        self.mean_time[-1].requires_grad_(True)
        self.encoding, self.filtered = encoding, filtered
        self.gate = baseline.gate
        source = encoding['source']
        centered = source - source[:, :100].mean(1, keepdim=True)
        amplitude = centered.amax(1)
        q = encoding['original']['q_source']
        self.scalars = torch.stack(((q / baseline.q_ref).log(),
                                    (amplitude / baseline.a_ref).log(),
                                    baseline.detector_condition.expand(len(q))), 1).float()

    def forward(self, rows):
        residual = .20 * self.mean_head(self.encoding['decoded'][rows]).tanh().double() * self.gate
        scale = torch.exp(math.log(2.) * self.mean_time(self.scalars[rows]).squeeze(1).tanh().double())
        warped, _ = relative_time_warp008(self.filtered[rows], scale)
        raw = warped * self.gate + residual
        q = canonical_charge008(raw)
        if bool((q <= 1e-6).any()):
            raise ValueError('Trainable mean shape has nonpositive canonical charge')
        return (self.encoding['original']['qbar'][rows, None, None] * raw / q[..., None])[:, 0]


class GlobalRepairResponse01(nn.Module):
    """A: original response coordinates; B: same plus copied terminal mean heads."""

    def __init__(self, original, theta, mean_head=None):
        super().__init__()
        # Electronics remains frozen in both arms. The 53 response coordinates
        # retain the historical parameter domain; no tiny restoration radius.
        self.theta = nn.Parameter(theta[:53].clone())
        self.noise_coordinate = theta[-1:].detach().clone()
        self.original, self.mean = original, mean_head

    def forward(self, rows, z, *, fixed_peak=None):
        theta = torch.cat((self.theta, self.noise_coordinate))
        if self.mean is None:
            return self.original(theta, rows, z)
        mean = self.mean(rows)
        cache = dict(A=mean, B=self.original.cache['B'][rows],
                     condition=self.original.cache['condition'][rows],
                     peak=mean[:, 112:144].argmax(1).double() + 112.
                     if fixed_peak is None else fixed_peak[rows])
        current = JointResponse01(cache)
        return current(theta, torch.arange(len(rows), device=rows.device), z)

    def noise_scale(self):
        return torch.exp(math.log(1.03) * self.noise_coordinate[0])
