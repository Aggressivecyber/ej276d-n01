"""Trainable mean plus correlated charge and zero-Q voltage fluctuations.

Electronics noise, ADC and observed-peak cropping belong to the caller. The
nine joint Gaussian coordinates are explicit: coordinate zero is available
for charge alone; coordinates one through eight also drive voltage shapes.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CausalFilter
from .model008 import N01Response008, canonical_charge008, control_average008


class N01Response029(nn.Module):
    """E[signal|x]=A(x), canonical Q=alpha*Q_A, with learned correlations.

    ``enabled=False`` returns A exactly. Explicit all-zero Gaussian draws
    instead give the mean-corrected lognormal's median charge, as intended.
    """

    noise_shapes = {'joint': (9,), 'local_white': (1, 400)}

    def __init__(self, stats: Mapping, *, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.mean_model = N01Response008(stats, 'deterministic')
            self.basis_head = nn.Conv1d(32, 8, 5, padding=2)
            self.charge_head = nn.Sequential(nn.Linear(131, 32), nn.SiLU(), nn.Linear(32, 9))
            self.envelope_head = nn.Conv1d(32, 1, 5, padding=2)
            self.local_filter = CausalFilter(17, 'rms')
            nn.init.normal_(self.basis_head.weight, std=1e-3)
            nn.init.zeros_(self.basis_head.bias)
            nn.init.zeros_(self.charge_head[-1].weight)
            nn.init.zeros_(self.charge_head[-1].bias)
            with torch.no_grad():
                self.charge_head[-1].bias[0] = .03
            nn.init.normal_(self.envelope_head.weight, std=1e-3)
            nn.init.constant_(self.envelope_head.bias, -7.)
        self.stats = dict(self.mean_model.stats)
        self.initialization_seed = seed

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return dict(mean=tuple(self.mean_model.parameters()),
                    shape=tuple(self.basis_head.parameters()),
                    charge=tuple(self.charge_head.parameters()),
                    local=tuple(self.envelope_head.parameters())+tuple(self.local_filter.parameters()))

    def sample_noise(self, batch: int, device=None, generator=None) -> dict[str, Tensor]:
        device = self.mean_model.q_ref.device if device is None else device
        return {key: torch.randn((batch,)+shape, device=device, dtype=torch.float32, generator=generator)
                for key, shape in self.noise_shapes.items()}

    def signal_forward(self, source_carrier_V: Tensor, noise: Mapping[str, Tensor], *, enabled: bool = True) -> dict:
        if type(enabled) is not bool or set(noise) != set(self.noise_shapes):
            raise ValueError('029 needs explicit joint/local_white draws and a boolean enable switch')
        for key, shape in self.noise_shapes.items():
            value = noise[key]
            if (value.shape != (len(source_carrier_V),)+shape or value.device != source_carrier_V.device
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid 029 latent: '+key)
        captured = []
        hook = self.mean_model.unet.register_forward_hook(
            lambda module, arguments, result: captured.append((result, arguments[1])))
        mean_noise = {key: source_carrier_V.new_zeros((len(source_carrier_V),)+shape)
                      for key, shape in self.mean_model.noise_shapes.items()}
        try:
            mean = self.mean_model.signal_forward(source_carrier_V, mean_noise)
        finally:
            hook.remove()
        (decoded, pooled), scalars = captured[0]
        A = mean['signal_pre']
        amplitude = A.abs().amax(1)
        controls = control_average008(self.basis_head(decoded)).double()
        smooth = torch.einsum('bkc,tc->bkt', controls, self.mean_model.expansion)*self.mean_model.gate
        projected = smooth-canonical_charge008(smooth)[..., None]*self.mean_model.projection
        B = amplitude[:, None, None]*projected
        charge_v = self.charge_head(torch.cat((scalars, pooled), 1)).double()
        log_variance = charge_v.square().sum(1)
        envelope_controls = control_average008(self.envelope_head(decoded)).double()
        envelope = F.softplus(torch.einsum('bkc,tc->bkt', envelope_controls, self.mean_model.expansion))
        envelope = envelope*self.mean_model.gate*amplitude[:, None, None]
        if enabled:
            z = noise['joint'].double()
            alpha = torch.exp((charge_v*z).sum(1)-.5*log_variance)
            shape = (B*z[:, 1:, None]).sum(1)
            raw_local = self.local_filter(F.pad(envelope, (16, 0))*noise['local_white'].double(), extended=True)
            raw_local = raw_local*self.mean_model.gate
            local = raw_local-canonical_charge008(raw_local)[..., None]*self.mean_model.projection
            signal = alpha[:, None]*A+shape+local[:, 0]
        else:
            alpha = torch.ones_like(log_variance)
            shape = torch.zeros_like(A)
            local = torch.zeros_like(A[:, None])
            signal = A
        if not bool(torch.isfinite(signal).all()):
            raise FloatingPointError('Nonfinite 029 signal; no rows clipped or dropped')
        normalizer = self.mean_model.a_ref.square()
        regular_basis = (B.square().mean()+torch.diff(B, n=2, dim=-1).square().mean())/normalizer
        regular_local = envelope.square().mean()/normalizer
        regular_charge = log_variance.mean()
        regular = mean['base_regular']+regular_basis+regular_local+regular_charge
        qsig = mean['qbar']*alpha
        canonical = canonical_charge008(signal)
        gate = self.mean_model.gate[0, 0]
        active_residual = mean['mean_residual'][:, gate > 0]/gate[gate > 0]
        residual_fraction = torch.zeros_like(mean['mean_residual'])
        residual_fraction[:, gate > 0] = active_residual.abs()/.20
        return dict(signal_pre=signal, base_signal_pre_V=A, pure_signal_pre_V=signal,
            B=B, voltage_basis_V=B, charge_v=charge_v, alpha=alpha,
            shape_coefficients=noise['joint'][:, 1:], random_shape_pre_V=shape,
            signal_noise=local[:, 0], signal_envelope=envelope[:, 0],
            qbar=mean['qbar'], qsig=qsig, q_source=mean['q_source'], time_scale=mean['time_scale'],
            mean_residual=mean['mean_residual'], mean_residual_bound_fraction=residual_fraction,
            mean_residual_saturation_fraction=(active_residual.abs() >= .198).double().mean(),
            sigma_q2=log_variance, charge_log_variance=log_variance,
            canonical_charge=canonical, canonical_charge_identity_residual=canonical-qsig,
            basis_canonical_charge=canonical_charge008(B), canonical_local_charge=canonical_charge008(local[:, 0]),
            regular=regular, base_regular=regular, trend=mean['trend'],
            regular_mean=mean['base_regular'], regular_basis=regular_basis,
            regular_local=regular_local, regular_charge=regular_charge,
            stochastic_enabled=enabled)

    forward = signal_forward
