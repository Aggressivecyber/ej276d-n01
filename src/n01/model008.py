"""Window008 signal response: 384-point carrier, 256-point effective charge.

No electronics, ADC, baseline measurement or observed-peak recrop occurs here.
Only length-independent neural primitives are reused from immutable model.py.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import (CausalFilter, MonotoneCharge, PhysicalHead, ResponseUNet,
                    ShapeEncoder, InvalidWaveformError, cubic_expansion)

CARRIER_LENGTH = 384
EFFECTIVE_LENGTH = 256
TIME_ANCHOR = 126.0
CANONICAL_START, CANONICAL_STOP = 110, 366
CONTROLS = 48


def canonical_charge008(values: Tensor) -> Tensor:
    """Linear signal-only window sum, V ns on the bound 1 ns grid.

    Inputs must already be baseline centered. This is NOT observed Q after
    noisy peak recentering; that readout belongs to observation008.
    """
    if values.shape[-1] != CARRIER_LENGTH:
        raise ValueError('Window008 canonical charge requires 384 samples')
    return values.to(torch.float64)[..., CANONICAL_START:CANONICAL_STOP].sum(-1)


def _positive(values: Tensor, floor: float, reason: str) -> None:
    invalid = (~torch.isfinite(values)) | (values <= floor)
    if bool(invalid.any()):
        raise InvalidWaveformError(reason, torch.nonzero(invalid, as_tuple=False).flatten(), values[invalid])


def _onset(t: Tensor) -> Tensor:
    u = ((t - 100.) / 15.).clamp(0., 1.)
    return u**3 * (10. - 15. * u + 6. * u**2)


def control_average008(x: Tensor) -> Tensor:
    if x.ndim != 3 or x.shape[-1] != CARRIER_LENGTH:
        raise ValueError('Window008 controls require BxCx384')
    return x.reshape(x.shape[0], x.shape[1], CONTROLS, 8).mean(-1)


def relative_time_warp008(signal: Tensor, scale: Tensor) -> tuple[Tensor, dict]:
    """Peak-126 relative warp; C1 edge continuation is a numerical convention.

    The carrier endpoint is not zeroed. A 64-sample external Hermite segment
    matches its value/slope and ends at zero outside the carrier. Out-of-range
    queries are counted; this continuation is never called measured context.
    """
    if signal.ndim != 3 or signal.shape[1:] != (1, CARRIER_LENGTH):
        raise ValueError('Window008 warp requires Bx1x384')
    signal = signal.double()
    tau = torch.arange(1, 65, device=signal.device, dtype=signal.dtype) / 64.
    edge, slope = signal[..., -1:], signal[..., -1:] - signal[..., -2:-1]
    continuation = ((2*tau**3 - 3*tau**2 + 1)*edge
                    + (tau**3 - 2*tau**2 + tau)*64.*slope)
    extended = torch.cat((signal, continuation), -1)
    t = torch.arange(CARRIER_LENGTH, device=signal.device, dtype=signal.dtype)
    position = TIME_ANCHOR + (t[None] - TIME_ANCHOR) / scale[:, None].double()
    left = position.floor().long()
    fraction = position - left
    a = extended[:, 0].gather(1, left.clamp(0, extended.shape[-1]-1))
    b = extended[:, 0].gather(1, (left+1).clamp(0, extended.shape[-1]-1))
    valid = (position >= 0) & (position <= extended.shape[-1]-1)
    result = torch.where(valid, (1-fraction)*a + fraction*b, torch.zeros_like(a))[:, None]
    diagnostics = dict(query_outside_carrier=((position < 0) | (position > CARRIER_LENGTH-1)).sum(1),
                       beyond_extension=(~valid).sum(1),
                       warp_left_outside_carrier=(position < 0).sum(1),
                       warp_right_outside_carrier=(position > CARRIER_LENGTH-1).sum(1),
                       warp_outside_extension=(~valid).sum(1))
    return result, diagnostics


class N01Response008(nn.Module):
    """Three candidate signal models with explicit paired response latents.

    Input is original SOURCE voltage context Bx384/Bx1x384. The first100 mean
    is removed before source features/charge. SOURCE peak must be at126 under
    the registered [112,144) first-argmax. All observed outputs use a separate
    common observation operator and are finally 256 points.
    """
    noise_shapes = {'phys': (10,), 'local_white': (1, 400)}
    stages = ('baseline', 'mean_response', 'joint_stochastic', 'low_learning_rate')

    def __init__(self, stats: Mapping, variant: str = 'full'):
        super().__init__()
        if variant not in ('simple', 'deterministic', 'full'):
            raise ValueError('Unknown Window008 variant')
        required = ('q_ref', 'a_ref', 'log_q_min', 'log_q_max')
        if not isinstance(stats, Mapping) or any(k not in stats for k in required):
            raise ValueError('Window008 requires explicit new TRAIN model statistics')
        self.stats = {k: float(stats[k]) for k in required}
        self.stats['detector_condition'] = float(stats.get('detector_condition', 0.))
        if not all(math.isfinite(v) for v in self.stats.values()):
            raise ValueError('Nonfinite Window008 model statistics')
        if self.stats['q_ref'] <= 0 or self.stats['a_ref'] <= 0:
            raise ValueError('Window008 charge/amplitude references must be positive')
        self.variant = variant
        for name in ('q_ref', 'a_ref', 'detector_condition'):
            self.register_buffer(name, torch.tensor(self.stats[name], dtype=torch.float64))
        t = torch.arange(CARRIER_LENGTH, dtype=torch.float64)
        gate = _onset(t)
        self.register_buffer('t_norm', torch.linspace(-1., 1., CARRIER_LENGTH)[None, None])
        self.register_buffer('gate', gate[None, None])
        projection = gate * torch.exp(-.5*((t-180.)/65.)**2)
        self.register_buffer('projection', (projection/canonical_charge008(projection))[None, None])
        self.register_buffer('expansion', cubic_expansion(length=CARRIER_LENGTH, controls=CONTROLS))
        self.shape_encoder = ShapeEncoder()
        self.charge_head = MonotoneCharge(self.stats['log_q_min'], self.stats['log_q_max'])
        self.response_filter = CausalFilter(65, 'dc')
        self.mean_time = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mean_time[-1].weight)
        nn.init.zeros_(self.mean_time[-1].bias)
        self.unet = ResponseUNet() if variant != 'simple' else None
        self.mean_head = nn.Conv1d(32, 1, 5, padding=2) if self.unet is not None else None
        if self.mean_head is not None:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        self.physical_head = PhysicalHead(10 if variant == 'full' else 2) if variant != 'deterministic' else None
        self.basis_head = nn.Conv1d(32, 8, 5, padding=2) if variant == 'full' else None
        self.envelope_head = nn.Conv1d(32, 1, 5, padding=2) if variant == 'full' else None
        self.local_filter = CausalFilter(17, 'rms') if variant == 'full' else None
        if self.basis_head is not None:
            nn.init.normal_(self.basis_head.weight, std=.03)
            nn.init.zeros_(self.basis_head.bias)
            nn.init.normal_(self.envelope_head.weight, std=1e-3)
            nn.init.constant_(self.envelope_head.bias, -5.)
        self.set_stage('mean_response')

    def set_stage(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError('Unknown Window008 stage')
        self.stage = stage
        self.requires_grad_(stage != 'baseline')
        if stage in ('baseline', 'mean_response'):
            for module in (self.physical_head, self.basis_head, self.envelope_head, self.local_filter):
                if module is not None:
                    module.requires_grad_(False)

    def sample_noise(self, batch: int, device=None, generator=None) -> dict[str, Tensor]:
        if device is None:
            device = self.q_ref.device
        return {key: torch.randn((batch,)+shape, device=device, dtype=torch.float32, generator=generator)
                for key, shape in self.noise_shapes.items()}

    def _features(self, unit_peak: Tensor) -> Tensor:
        u = unit_peak.to(self.shape_encoder.net[0].weight.dtype)
        derivative = F.pad(u[..., 1:]-u[..., :-1], (1, 0))
        return torch.cat((u, derivative, self.t_norm.to(u).expand(len(u), -1, -1)), 1)

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
            rms = raw_basis.square().mean(-1, keepdim=True).sqrt()
            if bool((rms <= 1e-10).any()):
                raise ValueError('Degenerate Window008 stochastic basis; no row deletion')
            basis = raw_basis/rms
        warped, diagnostics = relative_time_warp008(self.response_filter(unit_peak), scale)
        raw_shape = warped*self.gate + residual + (eta[:, 2:, None]*basis).sum(1, keepdim=True)
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
                    **diagnostics)

    def forward(self, source_carrier_V: Tensor, noise: Mapping[str, Tensor]) -> dict:
        return self.signal_forward(source_carrier_V, noise)


# Explicit compatibility name for the signal-only API; neither name adds ADC.
SignalResponse008 = N01Response008
