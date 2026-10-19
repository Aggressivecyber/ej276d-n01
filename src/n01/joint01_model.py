"""Low-dimensional joint response. All stochastic coordinates share eight latents.

Time endpoints describe the finite software carrier, not a timing calibration.
The frozen full-waveform A0 and M0 bases are supplied by the caller. This module
does not load weights, draw noise, digitize, or fit population weights.
"""
from __future__ import annotations

import math
import torch


COORDINATES = ('gain', 'rise', 'fall', 'tail_fraction', 'tail_time', 'curvature')
COORDINATE_LIMITS = (.30, .50, .50, .45, .40, .30)


def condition_features(condition):
    bounded = torch.tanh(condition.double() / 2.)
    return torch.stack((torch.ones_like(bounded[:, 0]), bounded[:, 0], bounded[:, 1],
                        bounded[:, 0] * bounded[:, 1]), 1)


def response_cache(A, B, condition):
    return dict(A=A, B=B, condition=condition_features(condition),
                peak=A[:, 112:144].argmax(1).double() + 112.)


class JointResponse01:
    """53 response + 1 baseline-constrained noise-scale coordinate (neural).

    The physical comparator uses the same six transformations of clean SOURCE,
    with 24 mean and 21 stochastic coefficients, without neural M0 bases.
    """

    def __init__(self, cache, *, physical=False):
        self.cache, self.physical = cache, physical
        self.response_parameters = 45 if physical else 53
        self.parameter_count = self.response_parameters + 1
        self.lower = cache['A'].new_full((self.parameter_count,), -1.)
        self.upper = cache['A'].new_ones(self.parameter_count)
        self.lower[-1] = 0.
        self.tril = torch.tril_indices(6, 6, device=cache['A'].device)
        self.limits = cache['A'].new_tensor(COORDINATE_LIMITS)
        self.time = torch.arange(384, dtype=torch.float64, device=cache['A'].device)

    def initial(self):
        theta = self.lower.new_zeros(self.parameter_count)
        if self.physical:
            # A stated comparator initialization, not a measured fluctuation law.
            diagonal = self.tril[0] == self.tril[1]
            theta[24:45][diagonal] = .08
        return theta

    def coordinates(self, theta, rows, z):
        mean = self.cache['condition'][rows] @ theta[:24].reshape(6, 4).T
        L = theta.new_zeros(6, 6)
        L[self.tril[0], self.tril[1]] = theta[24:45]
        return self.limits * torch.tanh(mean + z[:, :6].double() @ L.T)

    def __call__(self, theta, rows, z, *, return_details=False):
        A = self.cache['A'][rows]
        if self.physical:
            base = A
        else:
            scales = torch.exp(.5 * math.log(2.) * theta[45:53])
            random = (self.cache['B'][rows] * (z.double() * scales)[..., None]).sum(1)
            base = torch.where(z.ne(0).any(1)[:, None], A + random, A)
        a = self.coordinates(theta, rows, z)
        p = self.cache['peak'][rows, None]
        mid = self.time[101:][None] - .5
        rise = torch.exp(-.5 * ((mid - p + 8.) / 8.).square())
        fall = torch.exp(-.5 * ((mid - p - 18.) / 22.).square())
        tail = torch.sigmoid((mid - p - 55.) / 15.)
        speed = torch.exp(a[:, 1:2] * rise + a[:, 2:3] * fall + a[:, 4:5] * tail)
        integral = speed.cumsum(1)
        queries = torch.cat((self.time[:101][None].expand(len(rows), -1),
                             100. + 283. * integral / integral[:, -1:]), 1)
        left = queries.floor().long().clamp_max(382)
        fraction = queries - left
        warped = base.gather(1, left) * (1. - fraction) + base.gather(1, left + 1) * fraction
        tail_amplitude = torch.sigmoid((self.time[None] - p - 30.) / 8.)
        peak_amplitude = torch.exp(-.5 * ((self.time[None] - p) / 4.).square())
        multiplier = torch.exp(a[:, 0:1] + a[:, 3:4] * tail_amplitude + a[:, 5:6] * peak_amplitude)
        transformed = warped * multiplier
        signal = torch.where(a.ne(0).any(1)[:, None], transformed, base)
        if return_details:
            return dict(signal=signal, coordinates=a, queries=queries,
                        noise_scale=torch.exp(math.log(1.03) * theta[-1]))
        return signal

    def noise_scale(self, theta):
        return torch.exp(math.log(1.03) * theta[-1])

    def parameter_names(self):
        names = [f'mean/{a}/{c}' for a in COORDINATES for c in ('intercept', 'logQ', 'PC1', 'interaction')]
        names += [f'shared_L/{COORDINATES[i]}/z{j}' for i, j in zip(*self.tril.cpu().tolist())]
        if not self.physical:
            names += [f'M0_log_scale/z{i}' for i in range(8)]
        return names + ['electronics/log_scale_1_to_1p03']
