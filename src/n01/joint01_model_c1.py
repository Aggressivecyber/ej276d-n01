"""Prospective C1 time interpolation; the original joint01 model is unchanged."""
from __future__ import annotations

import math
import torch

from .joint01_model import JointResponse01


def pchip_uniform(values, queries):
    """Shape-preserving cubic Hermite interpolation on the existing unit grid."""
    difference = values[:, 1:] - values[:, :-1]
    same_sign = difference[:, :-1] * difference[:, 1:] > 0
    denominator = torch.where(same_sign, difference[:, :-1] + difference[:, 1:], 1.)
    interior = torch.where(same_sign, 2. * difference[:, :-1] * difference[:, 1:] / denominator, 0.)
    ends = []
    for first, second in ((difference[:, 0], difference[:, 1]), (difference[:, -1], difference[:, -2])):
        slope = (3. * first - second) / 2.
        slope = torch.where(slope * first <= 0, 0., slope)
        slope = torch.where((first * second < 0) & (slope.abs() > 3. * first.abs()), 3. * first, slope)
        ends.append(slope[:, None])
    slopes = torch.cat((ends[0], interior, ends[1]), 1)
    left = queries.floor().long().clamp_max(values.shape[1] - 2)
    t = queries - left
    y0, y1 = values.gather(1, left), values.gather(1, left + 1)
    m0, m1 = slopes.gather(1, left), slopes.gather(1, left + 1)
    t2, t3 = t.square(), t.square() * t
    return (2. * t3 - 3. * t2 + 1.) * y0 + (t3 - 2. * t2 + t) * m0 + (-2. * t3 + 3. * t2) * y1 + (t3 - t2) * m1


class JointResponse01C1(JointResponse01):
    """Same coordinates, bounds, latent sharing and electronics as joint01."""

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
        warped = pchip_uniform(base, queries)
        tail_amplitude = torch.sigmoid((self.time[None] - p - 30.) / 8.)
        peak_amplitude = torch.exp(-.5 * ((self.time[None] - p) / 4.).square())
        multiplier = torch.exp(a[:, 0:1] + a[:, 3:4] * tail_amplitude + a[:, 5:6] * peak_amplitude)
        transformed = warped * multiplier
        signal = torch.where(a.ne(0).any(1)[:, None], transformed, base)
        if return_details:
            return dict(signal=signal, base=base, coordinates=a, queries=queries,
                        noise_scale=torch.exp(math.log(1.03) * theta[-1]))
        return signal
