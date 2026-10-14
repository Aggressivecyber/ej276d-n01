"""Frozen Window008 fix01 signal interventions for the paired TRAIN diagnosis.

This module does not load data or checkpoints, draw noise, change a model stage,
apply ADC/cropping, or train.  The normal path calls the original signal_forward.
Interventions mask OUTPUT rows of the same Cholesky matrix and recompute both
lognormal variance corrections.  They are interventions, not conditioning.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import InvalidWaveformError
from .model008 import (
    CARRIER_LENGTH, TIME_ANCHOR, canonical_charge008, control_average008,
    relative_time_warp008, _positive,
)
from .model008_fix01 import N01Response008Fix01


ALL_SIGNAL_RANDOM_OFF009 = ('Q', 'time', 'basis', 'local')


@dataclass(frozen=True)
class SignalMask009:
    """Output-coordinate switches; retained coordinates keep the original z."""

    charge: bool = True
    time: bool = True
    basis: tuple[bool, ...] = (True,) * 8
    local: bool = True

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.charge, self.time, self.local)):
            raise TypeError('009 charge/time/local masks must be bool')
        if not isinstance(self.basis, tuple) or len(self.basis) != 8 or any(type(v) is not bool for v in self.basis):
            raise TypeError('009 requires eight explicit bool basis-row masks')

    @classmethod
    def from_disable(cls, disable: Sequence[str] = ()) -> 'SignalMask009':
        if isinstance(disable, (str, bytes)):
            raise TypeError('009 disable must be a sequence of branch names')
        charge, time, local, basis = True, True, True, [True] * 8
        for name in disable:
            if name in ('Q', 'q', 'charge'):
                charge = False
            elif name == 'time':
                time = False
            elif name in ('basis', 'shape'):
                basis = [False] * 8
            elif name == 'local':
                local = False
            elif isinstance(name, str) and name.startswith('basis') and name[5:] in tuple(str(i) for i in range(8)):
                basis[int(name[5:])] = False
            else:
                raise ValueError('Unknown 009 disabled branch: ' + repr(name))
        return cls(charge=charge, time=time, basis=tuple(basis), local=local)

    @property
    def physical_rows(self) -> tuple[bool, ...]:
        return (self.charge, self.time) + self.basis

    @property
    def all_on(self) -> bool:
        return all(self.physical_rows) and self.local

    @property
    def all_off(self) -> bool:
        return not any(self.physical_rows) and not self.local


def tensor_state_digest009(model_or_state) -> str:
    """Deterministic SHA256 of state tensor names, layouts and exact bytes."""
    state = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    if not isinstance(state, Mapping):
        raise TypeError('009 state digest expects a model or state mapping')
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError('009 state digest requires named tensors')
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf8')
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        raw = value.detach().to(device='cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'little'))
        digest.update(raw)
    return digest.hexdigest()


def _frozen_layout009(model: N01Response008Fix01):
    if type(model) is not N01Response008Fix01 or model.variant != 'full':
        raise TypeError('009 interventions require the exact N01Response008Fix01 full class')
    if model.stage not in ('joint_stochastic', 'low_learning_rate'):
        raise ValueError('009 preserves the full endpoint joint/low stage; no stage switch is permitted')
    modes = tuple((name, module.training) for name, module in model.named_modules())
    if any(training for _, training in modes):
        raise ValueError('009 caller must put the complete original model in eval mode')
    parameters = tuple((name, id(value), value.requires_grad) for name, value in model.named_parameters())
    if any(trainable for _, _, trainable in parameters):
        raise ValueError('009 caller must freeze all original model parameters')
    layout = tuple((name, str(value.dtype), tuple(value.shape), value.data_ptr(), value._version)
                   for name, value in model.state_dict().items())
    stats = json.dumps(model.stats, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return model.stage, model.variant, modes, parameters, layout, stats


@torch.no_grad()
def _intervened_signal_forward009(model: N01Response008Fix01, source_carrier_V: Tensor,
                                  noise: Mapping[str, Tensor], mask: SignalMask009) -> dict:
    """Copy of fix01 assembly; only row masks and branch switches are added."""
    x = source_carrier_V[:, None] if source_carrier_V.ndim == 2 else source_carrier_V
    if x.ndim != 3 or x.shape[1:] != (1, CARRIER_LENGTH) or len(x) == 0 or not x.is_floating_point():
        raise ValueError('Window008 SOURCE input must be nonempty floating Bx384/Bx1x384')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite SOURCE; no row deletion')
    if set(noise) != set(model.noise_shapes):
        raise ValueError('Window008 requires explicit phys and local_white latents')
    for key, shape in model.noise_shapes.items():
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
    features = model._features(unit_peak)
    qbar = model.charge_head(qsource.log(), model.shape_encoder(features))
    _positive(qbar, 0., 'Invalid Window008 mean signal charge')
    scalars = torch.stack(((qsource/model.q_ref).log(), (amplitude/model.a_ref).log(), model.detector_condition.expand(len(x))), 1).to(features)
    zero = x64.new_zeros((len(x), 1, CARRIER_LENGTH))
    residual, basis, envelope = zero, zero.expand(-1, 8, -1), zero
    pooled = None
    if model.unet is not None and model.stage != 'baseline':
        decoded, pooled = model.unet(features, scalars)
        residual = .20*torch.tanh(model.mean_head(decoded)).double()*model.gate
    joint = model.stage in ('joint_stochastic', 'low_learning_rate')
    if joint and model.physical_head is not None and any(mask.physical_rows):
        lower = model.physical_head(scalars, pooled if model.variant == 'full' else None,
                                    rank=10 if model.variant == 'full' else 2).double()
        # Keep every z column and every retained output row exactly unchanged.
        if not all(mask.physical_rows):
            rows = lower.new_tensor(mask.physical_rows)
            lower = lower*rows[None, :, None]
    else:
        lower = x64.new_zeros((len(x), 10, 10))
    eta = (lower@noise['phys'].double()[..., None]).squeeze(-1)
    sigma_q2 = lower[:, 0].square().sum(1)
    qsig = qbar*torch.exp(eta[:, 0]-.5*sigma_q2)
    scale = torch.exp(math.log(2.)*torch.tanh(model.mean_time(scalars).squeeze(1)).double()
                      + eta[:, 1]-.5*lower[:, 1].square().sum(1))
    _positive(qsig, 0., 'Invalid Window008 stochastic charge')
    _positive(scale, 0., 'Invalid Window008 time scale')
    if model.variant == 'full' and joint and any(mask.basis):
        controls = control_average008(model.basis_head(decoded)).double()
        raw_basis = torch.einsum('bkc,tc->bkt', controls, model.expansion)*model.gate
        raw_basis = raw_basis-canonical_charge008(raw_basis)[..., None]*model.projection
        rms = raw_basis.square().mean(-1, keepdim=True).sqrt()
        if bool((~torch.isfinite(raw_basis)).any()) or bool(((~torch.isfinite(rms)) | (rms <= 1e-10)).any()):
            raise ValueError('Nonfinite or degenerate Window008 fix01 stochastic basis; no row deletion')
        basis = raw_basis/rms
    warped, diagnostics = relative_time_warp008(model.response_filter(unit_peak), scale)
    base_shape = warped*model.gate + residual
    base_shape_q = canonical_charge008(base_shape).squeeze(1)
    if model.variant == 'full' and joint:
        _positive(base_shape_q, 1e-6, 'Invalid Window008 fix01 base shape canonical charge')
    raw_shape = base_shape + (eta[:, 2:, None]*basis).sum(1, keepdim=True)
    shape_q = canonical_charge008(raw_shape).squeeze(1)
    _positive(shape_q, 1e-6, 'Invalid Window008 shape canonical charge')
    unit_shape = raw_shape/shape_q[:, None, None]
    local = zero
    if model.variant == 'full' and joint and mask.local:
        controls = control_average008(model.envelope_head(decoded)).double()
        envelope = F.softplus(torch.einsum('bkc,tc->bkt', controls, model.expansion))*model.gate*model.a_ref*(qsig/model.q_ref)[:, None, None]
        raw_local = model.local_filter(F.pad(envelope, (16, 0))*noise['local_white'].double(), extended=True)*model.gate
        local = raw_local-canonical_charge008(raw_local)[..., None]*model.projection
    signal = qsig[:, None, None]*unit_shape + local
    if not bool(torch.isfinite(signal).all()) or bool((signal[..., :100] != 0).any()):
        raise ValueError('Window008 signal nonfinite or prefix100 not exactly zero')
    regular = residual.square().mean() + torch.diff(residual, dim=-1).square().mean()
    if model.variant == 'full' and joint:
        gram = basis@basis.transpose(1, 2)/CARRIER_LENGTH
        offdiag = gram-torch.diag_embed(gram.diagonal(dim1=1, dim2=2))
        regular = regular + .01*offdiag.square().mean() + torch.diff(basis, n=2, dim=-1).square().mean() + envelope.square().mean()/model.a_ref.square()
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
                basis_canonical_charge=canonical_charge008(basis), **diagnostics)


def _engineering009(output: dict, mask: SignalMask009) -> dict:
    """Derive voltage components from existing outputs; no extra model forward."""
    result = dict(output)
    raw_shape_random = (output['physical_eta'][:, 2:, None]*output['basis']).sum(1)
    shape_voltage = output['qsig'][:, None]*(raw_shape_random/output['shape_Q'][:, None])
    local = output['signal_noise']
    mean_component = output['signal_pre']-local-shape_voltage
    remainder = output['signal_pre']-(mean_component+shape_voltage+local)
    result.update(
        pure_signal_pre_V=output['signal_pre'],
        qsig_over_base_shape_Q=output['qsig']/output['base_shape_Q'],
        stochastic_shape_pre_V=shape_voltage,
        stochastic_shape_voltage_RMS_V=shape_voltage.square().mean(1).sqrt(),
        local_RMS_V=local.square().mean(1).sqrt(),
        mean_component_pre_V=mean_component,
        decomposition_roundoff_absmax_V=remainder.abs().amax(1),
        # C is the full model's nominal path. B's nominal reference is the
        # independently requested C output, not an invisible extra forward.
        nominal_mean_signal_pre_V=output['signal_pre'] if mask.all_off else None,
        actual_physical_row_mask=torch.tensor(mask.physical_rows, device=output['signal_pre'].device, dtype=torch.bool),
        local_signal_noise_active=mask.local,
        intervention009=dict(schema='n01/signal-intervention009/v1',
                             physical_rows=list(mask.physical_rows), local=mask.local,
                             law='output-row intervention, not Gaussian conditioning',
                             mean_component='base contribution with the current Q/time draw; not the conditional mean',
                             extra_nominal_model_forwards=0),
    )
    return result


class FrozenSignalAblation009:
    """Inference-only adapter which refuses to mutate the original model."""

    def __init__(self, model: N01Response008Fix01):
        self.model = model
        self._layout = _frozen_layout009(model)
        self.initial_state_sha256 = tensor_state_digest009(model)

    def assert_unchanged(self, *, deep: bool = False) -> str:
        if _frozen_layout009(self.model) != self._layout:
            raise RuntimeError('009 original class/stage/modes/parameters/buffers/stats changed')
        if deep and tensor_state_digest009(self.model) != self.initial_state_sha256:
            raise RuntimeError('009 original state bytes changed')
        return self.initial_state_sha256

    @torch.no_grad()
    def signal_forward(self, source_carrier_V: Tensor, noise: Mapping[str, Tensor],
                       *, disable: Sequence[str] = (), mask: SignalMask009 | None = None) -> dict:
        self.assert_unchanged()
        if mask is not None and disable:
            raise ValueError('009 accepts either an explicit mask or disabled branch names')
        switches = SignalMask009.from_disable(disable) if mask is None else mask
        if not isinstance(switches, SignalMask009):
            raise TypeError('009 mask must be SignalMask009')
        try:
            output = (self.model.signal_forward(source_carrier_V, noise) if switches.all_on else
                      _intervened_signal_forward009(self.model, source_carrier_V, noise, switches))
            return _engineering009(output, switches)
        finally:
            self.assert_unchanged()


def ablated_signal_forward(model: N01Response008Fix01, source_carrier_V: Tensor,
                           noise: Mapping[str, Tensor], *, disable: Sequence[str] = ()) -> dict:
    """Convenience call; repeated batches should reuse FrozenSignalAblation009."""
    helper = FrozenSignalAblation009(model)
    result = helper.signal_forward(source_carrier_V, noise, disable=disable)
    helper.assert_unchanged(deep=True)
    return result
