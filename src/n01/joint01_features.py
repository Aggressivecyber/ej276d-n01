"""Observable joint features, fixed random Fourier maps, and covariance scaling.

All inputs have already passed the unchanged hard 008 observation. Missing
crossings remain rows, with a zero measurement plus a separate undefined flag.
"""
from __future__ import annotations

import math
import torch
from .losses008 import ADC_STEP


BLOCKS = ('wave', 'charge_morphology', 'peak_codes', 'baseline')


def _crossing(wave, fraction, side):
    level = fraction * wave[:, 16]
    grid = torch.arange(256, device=wave.device)[None]
    if side == 'left':
        mask = (grid < 16) & (wave < level[:, None])
        k = torch.where(mask, grid, -torch.ones_like(grid)).amax(1)
        defined = k >= 0
        low = k.clamp_min(0)
        high = low + 1
    else:
        mask = (grid > 16) & (wave < level[:, None])
        k = torch.where(mask, grid, torch.full_like(grid, 256)).amin(1)
        defined = k < 256
        high = k.clamp_max(255)
        low = high - 1
    a, b = wave.gather(1, low[:, None])[:, 0], wave.gather(1, high[:, None])[:, 0]
    slope = b - a
    defined = defined & (slope != 0) & (wave[:, 16] > 0)
    position = low + (level - a) / torch.where(defined, slope, torch.ones_like(slope))
    return torch.where(defined, position, torch.zeros_like(position)), defined


def observable_blocks(obs):
    w, baseline = obs['window_V'], obs['baseline100']
    amplitude = w[:, 16].abs().clamp_min(ADC_STEP)
    wave = torch.cat((w, w / amplitude[:, None], w.diff(dim=1) / amplitude[:, None]), 1)
    left, left_ok = _crossing(w, .5, 'left')
    right, right_ok = _crossing(w, .5, 'right')
    t10, t10_ok = _crossing(w, .1, 'left')
    t90, t90_ok = _crossing(w, .9, 'left')
    width_ok, rise_ok = left_ok & right_ok, t10_ok & t90_ok
    width = torch.where(width_ok, right - left, torch.zeros_like(left))
    rise = torch.where(rise_ok, t90 - t10, torch.zeros_like(left))
    flags = obs['flags']
    morphology = torch.stack((obs['Q'], obs['tail'], w[:, 16], obs['PSD'], width, rise,
        (-w).clamp_min(0).sum(1), (~width_ok).double(), (~rise_ok).double(),
        flags['Q_nonpositive'].double(), flags['PSD_guarded'].double(),
        flags['peak_at_search_boundary'].double(), flags['global_peak_outside_search'].double(),
        flags['ADC_rail_hit'].double()), 1)
    adc = obs['carrier_adc']
    search_codes = torch.round(adc[:, 112:144] / ADC_STEP)
    top2 = search_codes.topk(2, dim=1).values
    ties = search_codes == top2[:, :1]
    index = torch.arange(32, device=adc.device)[None].expand_as(search_codes)
    reset = torch.where(ties, -torch.ones_like(index), index)
    lengths = index - reset.cummax(1).values
    plateau = lengths.amax(1)
    offsets = torch.arange(-4, 5, device=adc.device)[None]
    local = torch.round(adc.gather(1, obs['peak_index'][:, None] + offsets) / ADC_STEP)
    relative = local[:, 4:5] - local
    peak = torch.cat((relative[:, [0, 1, 2, 3, 5, 6, 7, 8]],
        (top2[:, 0] - top2[:, 1])[:, None], ties.sum(1)[:, None], plateau[:, None],
        (2 * local[:, 4] - local[:, 3] - local[:, 5])[:, None]), 1)
    centered = baseline - baseline.mean(1, keepdim=True)
    noise = torch.stack((baseline.mean(1), centered.square().mean(1).sqrt(),
        *[(centered[:, :-lag] * centered[:, lag:]).mean(1) for lag in (1, 2, 5, 10, 20)]), 1)
    return dict(zip(BLOCKS, (wave, morphology, peak, noise)))


class JointFeatureMap01:
    """272 distribution features; full shrunk covariance retains cross-block terms.

    Four blocks use 8 linear and 48 multi-scale Fourier features each. A further
    48 Fourier features mix all four blocks, so this is not four marginal fits.
    The feature covariance is a scale convention, not experimental precision.
    """

    def __init__(self, training_blocks, *, seed, shrinkage=.20):
        self.seed, self.shrinkage = seed, shrinkage
        self.center, self.scale, self.projections = {}, {}, {}
        generator = torch.Generator(device='cpu').manual_seed(seed)
        device = training_blocks['wave'].device
        total = 0
        for name in BLOCKS:
            values = training_blocks[name]
            self.center[name] = values.mean(0).float()
            scale = values.std(0, unbiased=False).float()
            self.scale[name] = torch.where(scale > 0, scale, torch.ones_like(scale))
            dimension = values.shape[1]
            self.projections[name] = torch.randn(dimension, 8, generator=generator).to(device) / math.sqrt(dimension)
            total += dimension
        self.projections['joint'] = torch.randn(total, 8, generator=generator).to(device)
        features = self.features(training_blocks)
        self.training_mean = features.mean(0)
        centered = features - self.training_mean
        covariance = centered.T @ centered / (len(features) - 1)
        diagonal = covariance.diag()
        omega = (1. - shrinkage) * covariance + shrinkage * torch.diag(diagonal)
        omega += torch.eye(len(omega), device=device, dtype=torch.float64) * (1e-5 * diagonal.mean())
        self.cholesky = torch.linalg.cholesky(omega)
        self.block_slices = {name: slice(i * 56, (i + 1) * 56) for i, name in enumerate(BLOCKS)}
        self.block_slices['joint'] = slice(224, 272)

    def features(self, blocks):
        features, joint = [], []
        for name in BLOCKS:
            standardized = (blocks[name].float() - self.center[name]) / self.scale[name]
            projected = standardized @ self.projections[name]
            fourier = [part for bandwidth in (.5, 1., 2.)
                       for part in ((projected / bandwidth).cos(), (projected / bandwidth).sin())]
            features.append(torch.cat((projected, *fourier), 1))
            joint.append(standardized / math.sqrt(4 * standardized.shape[1]))
        projected = torch.cat(joint, 1) @ self.projections['joint']
        features.append(torch.cat([part for bandwidth in (.5, 1., 2.)
            for part in ((projected / bandwidth).cos(), (projected / bandwidth).sin())], 1))
        return torch.cat(features, 1).double()

    def mean(self, blocks, weights=None):
        values = self.features(blocks)
        if weights is None:
            return values.mean(0)
        return (values * (weights / weights.sum())[:, None]).sum(0)

    def whiten(self, residual):
        return torch.linalg.solve_triangular(self.cholesky, residual[..., None], upper=False)[..., 0]

    def risks(self, residual):
        risk = dict(joint_whitened=float(self.whiten(residual).square().mean()))
        variance = self.cholesky.square().sum(1)
        for name, selection in self.block_slices.items():
            risk[name] = float((residual[selection].square() / variance[selection]).mean())
        return risk


def joint_step(residual, jacobian, *, ridge_fraction, radius):
    gram = jacobian.T @ jacobian
    ridge = ridge_fraction * gram.diag().mean()
    delta = torch.linalg.solve(gram + ridge * torch.eye(len(gram), device=gram.device, dtype=gram.dtype),
                               -(jacobian.T @ residual))
    norm = torch.linalg.vector_norm(delta)
    delta *= min(1., radius / float(norm))
    singular = torch.linalg.svdvals(jacobian)
    off_diagonal = gram - torch.diag(gram.diag())
    return delta, dict(singular_values=singular.cpu().tolist(),
        effective_rank_at_1pct=int((singular > .01 * singular[0]).sum()), ridge=float(ridge),
        off_diagonal_gram_fraction=float(off_diagonal.norm() / gram.norm()),
        proposed_step_norm=float(delta.norm()), full_cross_module_gram_used=True)
