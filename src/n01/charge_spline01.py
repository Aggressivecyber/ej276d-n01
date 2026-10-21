"""Replace oscillatory Q/amplitude FiLM conditioning by six spline heads."""
import copy
import math

import torch
from scipy.interpolate import BSpline

from .mean_trend01 import ConditionedMeanRepair01
from .model008 import canonical_charge008, relative_time_warp008


class ChargeSplineMean01(ConditionedMeanRepair01):
    def __init__(self, baseline, encoding, filtered, inherited, fit_parents):
        super().__init__(baseline, encoding, filtered, inherited)
        self.unet.requires_grad_(False)
        head = inherited.mean_head
        self.anchor_head = copy.deepcopy(head).requires_grad_(False)
        self.mean_head = torch.nn.Conv1d(head.in_channels, 6, head.kernel_size,
            padding=head.padding, device=head.weight.device, dtype=head.weight.dtype)
        with torch.no_grad():
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()
        logq = self.scalars[fit_parents, 0].double()
        self.register_buffer('logq_bounds', torch.stack((logq.min(), logq.max())))

    def spline_basis(self, logq):
        low, high = self.logq_bounds.detach().cpu().numpy()
        knots = [low] * 4 + [low + (high - low) / 3, low + 2 * (high - low) / 3] + [high] * 4
        matrix = BSpline.design_matrix(logq.detach().double().cpu().numpy(), knots, 3,
                                      extrapolate=True).toarray()
        return torch.as_tensor(matrix, device=logq.device, dtype=torch.float64)

    def anchored_decode(self, features, scalars):
        anchor = scalars.clone()
        anchor[:, :2] = 0
        with torch.no_grad():
            decoded, _ = self.unet(features, anchor)
        return decoded

    def prepare(self, parents, chunk):
        self.decoded_anchor = torch.zeros_like(self.encoding['decoded'])
        self.parent_basis = self.scalars.new_zeros((len(self.scalars), 6), dtype=torch.float64)
        with torch.no_grad():
            for rows in parents.split(chunk):
                self.decoded_anchor[rows] = self.anchored_decode(self.features[rows], self.scalars[rows])
                self.parent_basis[rows] = self.spline_basis(self.scalars[rows, 0])

    def shape_from_decoded(self, decoded, basis, filtered, scalars):
        self.forward_rows += len(decoded)
        delta = (self.mean_head(decoded).double() * basis[..., None]).sum(1, keepdim=True)
        logits = self.anchor_head(decoded) + delta.to(decoded.dtype)
        residual = .20 * logits.tanh().double() * self.gate
        scale = torch.exp(math.log(2.) * self.mean_time(scalars).squeeze(1).tanh().double())
        warped, _ = relative_time_warp008(filtered, scale)
        raw = warped * self.gate + residual
        charge = canonical_charge008(raw)
        if bool((charge <= 1e-6).any()):
            raise ValueError('Spline mean has nonpositive canonical charge')
        return (raw / charge[..., None])[:, 0]

    def unit_shape(self, features, filtered, scalars):
        return self.shape_from_decoded(self.anchored_decode(features, scalars),
            self.spline_basis(scalars[:, 0]), filtered, scalars)

    def uncached(self, rows):
        unit = self.shape_from_decoded(self.decoded_anchor[rows], self.parent_basis[rows],
                                       self.filtered[rows], self.scalars[rows])
        return self.encoding['original']['qbar'][rows, None] * unit
