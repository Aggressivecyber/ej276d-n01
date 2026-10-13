"""Window008 fix01: canonical-zero-charge stochastic shape bases.

The inherited constructor deliberately preserves every original parameter,
buffer, initialization draw and state ordering. Only the full-arm joint/low
stochastic basis definition changes. Electronics and observation stay external.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import InvalidWaveformError
from .model008 import (
    N01Response008, CARRIER_LENGTH, TIME_ANCHOR, canonical_charge008,
    control_average008, relative_time_warp008, _positive,
)


class N01Response008Fix01(N01Response008):
    """Same mean/control paths; full stochastic bases preserve canonical Q."""

    def signal_forward(self, source_carrier_V: Tensor, noise: Mapping[str, Tensor]) -> dict:
        x = source_carrier_V[:, None] if source_carrier_V.ndim == 2 else source_carrier_V
        if x.ndim != 3 or x.shape[1:] != (1, CARRIER_LENGTH) or len(x) == 0 or not x.is_floating_point():
            raise ValueError('Window008 SOURCE input must be nonempty floating Bx384/Bx1x384')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Nonfinite SOURCE; no row deletion')
        if set(noise) != set(self.noise_shapes):
            raise ValueError('Window008 requires explicit phys and local_white latents')
        for key, shape in self.noise_shapes.items():
            if noise[key].shape != (len(x),)+shape or noise[key].device != x.device or not bool(torch.isfinite(noise[key]).all()):
                raise ValueError('Invalid Window008 latent: '+key)
        x64 = x.double()
        baseline = x64[..., :100].mean(-1, keepdim=True)
        centered = x64-baseline
        peaks = centered[:, 0, 112:144].argmax(1)+112
        if bool((peaks != int(TIME_ANCHOR)).any()):
            raise InvalidWaveformError('SOURCE carrier peak differs from registered126', torch.nonzero(peaks != 126).flatten(), peaks[peaks != 126])
        amplitude = centered.amax(-1).squeeze(1)
        qsource = canonical_charge008(centered).squeeze(1)
        _positive(amplitude, 1e-8, 'Invalid Window008 SOURCE amplitude')
        _positive(qsource, 1e-8, 'Invalid Window008 SOURCE canonical charge')
        unit_peak = centered/amplitude[:, None, None]
        features = self._features(unit_peak)
        qbar = self.charge_head(qsource.log(), self.shape_encoder(features))
        _positive(qbar, 0., 'Invalid Window008 mean signal charge')
        scalars = torch.stack(((qsource/self.q_ref).log(), (amplitude/self.a_ref).log(), self.detector_condition.expand(len(x))), 1).to(features)
        zero = x64.new_zeros((len(x), 1, CARRIER_LENGTH))
        residual, basis, envelope = zero, zero.expand(-1, 8, -1), zero
        pooled = None
        if self.unet is not None and self.stage != 'baseline':
            decoded, pooled = self.unet(features, scalars)
            residual = .20*torch.tanh(self.mean_head(decoded)).double()*self.gate
        joint = self.stage in ('joint_stochastic', 'low_learning_rate')
        if joint and self.physical_head is not None:
            lower = self.physical_head(scalars, pooled if self.variant == 'full' else None,
                                       rank=10 if self.variant == 'full' else 2).double()
        else:
            lower = x64.new_zeros((len(x), 10, 10))
        eta = (lower@noise['phys'].double()[..., None]).squeeze(-1)
        sigma_q2 = lower[:, 0].square().sum(1)
        qsig = qbar*torch.exp(eta[:, 0]-.5*sigma_q2)
        scale = torch.exp(math.log(2.)*torch.tanh(self.mean_time(scalars).squeeze(1)).double()
                          + eta[:, 1]-.5*lower[:, 1].square().sum(1))
        _positive(qsig, 0., 'Invalid Window008 stochastic charge')
        _positive(scale, 0., 'Invalid Window008 time scale')
        if self.variant == 'full' and joint:
            controls = control_average008(self.basis_head(decoded)).double()
            raw_basis = torch.einsum('bkc,tc->bkt', controls, self.expansion)*self.gate
            # Apply the same smooth canonical projection as the local branch,
            # after expansion/gating and before norm or shape regularization.
            raw_basis = raw_basis-canonical_charge008(raw_basis)[..., None]*self.projection
            rms = raw_basis.square().mean(-1, keepdim=True).sqrt()
            if bool((~torch.isfinite(raw_basis)).any()) or bool(((~torch.isfinite(rms)) | (rms <= 1e-10)).any()):
                raise ValueError('Nonfinite or degenerate Window008 fix01 stochastic basis; no row deletion')
            basis = raw_basis/rms
        warped, diagnostics = relative_time_warp008(self.response_filter(unit_peak), scale)
        base_shape = warped*self.gate + residual
        base_shape_q = canonical_charge008(base_shape).squeeze(1)
        if self.variant == 'full' and joint:
            _positive(base_shape_q, 1e-6, 'Invalid Window008 fix01 base shape canonical charge')
        raw_shape = base_shape + (eta[:, 2:, None]*basis).sum(1, keepdim=True)
        shape_q = canonical_charge008(raw_shape).squeeze(1)
        _positive(shape_q, 1e-6, 'Invalid Window008 shape canonical charge')
        unit_shape = raw_shape/shape_q[:, None, None]
        local = zero
        if self.variant == 'full' and joint:
            controls = control_average008(self.envelope_head(decoded)).double()
            envelope = F.softplus(torch.einsum('bkc,tc->bkt', controls, self.expansion))*self.gate*self.a_ref*(qsig/self.q_ref)[:, None, None]
            raw_local = self.local_filter(F.pad(envelope, (16, 0))*noise['local_white'].double(), extended=True)*self.gate
            local = raw_local-canonical_charge008(raw_local)[..., None]*self.projection
        signal = qsig[:, None, None]*unit_shape + local
        if not bool(torch.isfinite(signal).all()) or bool((signal[..., :100] != 0).any()):
            raise ValueError('Window008 signal nonfinite or prefix100 not exactly zero')
        regular = residual.square().mean() + torch.diff(residual, dim=-1).square().mean()
        if self.variant == 'full' and joint:
            gram = basis@basis.transpose(1, 2)/CARRIER_LENGTH
            offdiag = gram-torch.diag_embed(gram.diagonal(dim1=1, dim2=2))
            regular = regular + .01*offdiag.square().mean() + torch.diff(basis, n=2, dim=-1).square().mean() + envelope.square().mean()/self.a_ref.square()
        return dict(signal_pre=signal[:, 0], qbar=qbar, qsig=qsig, q_source=qsource,
                    unit_shape=unit_shape[:, 0], signal_noise=local[:, 0], time_scale=scale,
                    basis=basis, mean_residual=residual[:, 0], signal_envelope=envelope[:, 0],
                    regular=regular, base_regular=regular, trend=qbar.sum()*0.,
                    physical_cholesky=lower, physical_eta=eta, sigma_q2=sigma_q2,
                    canonical_charge=canonical_charge008(signal).squeeze(1),
                    canonical_local_charge=canonical_charge008(local).squeeze(1),
                    canonical_charge_identity_residual=canonical_charge008(signal).squeeze(1)-qsig,
                    source_baseline_mean_V=baseline[:, 0, 0], source_peak_index=peaks,
                    base_shape_Q=base_shape_q, shape_Q=shape_q,
                    basis_canonical_charge=canonical_charge008(basis),
                    **diagnostics)


SignalResponse008Fix01 = N01Response008Fix01
