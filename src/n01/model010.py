"""One voltage-domain random-shape head on the immutable deterministic 008 A.

The only learned parameters are basis_head.weight and basis_head.bias. A single
original signal_forward is evaluated under no_grad; a temporary U-Net hook
reuses its decoded features and is always removed. No old set_stage is called.
No data/checkpoints, random draws, ADC, observation, losses or optimizers are
created here. Mean and canonical-Q protection apply before nonlinear readout.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import Tensor, nn

from .model008 import N01Response008, canonical_charge008, control_average008


INITIALIZATION_SEED010 = 2026101420
HEAD_INITIALIZATION_STD010 = .01
VOLTAGE_GAIN010 = .05
SHAPE_DIMENSIONS010 = 8


def tensor_state_digest010(model_or_state) -> str:
    state = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    if not isinstance(state, Mapping):
        raise TypeError('010 state digest requires a model or state mapping')
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError('010 state digest requires named tensors')
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf8')
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'little'))
        digest.update(raw)
    return digest.hexdigest()


def _baseline_layout010(model: N01Response008):
    if type(model) is not N01Response008 or model.variant != 'deterministic':
        raise TypeError('010 requires the exact original N01Response008 deterministic class')
    if model.stage not in ('mean_response', 'joint_stochastic', 'low_learning_rate'):
        raise ValueError('010 requires an active original deterministic mean path')
    modes = tuple((name, module.training) for name, module in model.named_modules())
    if any(training for _, training in modes):
        raise ValueError('010 original A must already be entirely in eval mode')
    parameters = tuple((name, id(value), value.requires_grad, value.grad is None)
                       for name, value in model.named_parameters())
    if any(trainable or not grad_none for _, _, trainable, grad_none in parameters):
        raise ValueError('010 original A must be frozen with no accumulated gradients')
    state = tuple((name, str(value.dtype), tuple(value.shape), value.data_ptr(), value._version)
                  for name, value in model.state_dict().items())
    hooks = tuple((name, tuple((key, id(value)) for key, value in module._forward_hooks.items()),
                   tuple((key, id(value)) for key, value in module._forward_pre_hooks.items()),
                   tuple((key, id(value)) for key, value in module._backward_hooks.items()))
                  for name, module in model.named_modules())
    stats = json.dumps(model.stats, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return model.stage, model.variant, modes, parameters, state, hooks, stats


class FrozenDeterministicVoltageShape010(nn.Module):
    """Frozen A(x) + eight input-dependent canonical-zero-Q voltage bases."""

    noise_shapes = {'shape': (SHAPE_DIMENSIONS010,)}
    _TRAINABLE_NAMES = ('basis_head.weight', 'basis_head.bias')

    def __init__(self, baseline: N01Response008, *, seed: int = INITIALIZATION_SEED010,
                 amplitude_fraction: float = VOLTAGE_GAIN010):
        super().__init__()
        if type(seed) is not int or seed != INITIALIZATION_SEED010 or amplitude_fraction != VOLTAGE_GAIN010:
            raise ValueError('010 uses the registered initialization seed and voltage gain')
        layout = _baseline_layout010(baseline)
        if baseline.unet is None or baseline.mean_head is None:
            raise ValueError('010 requires the original deterministic U-Net and mean head')
        self.baseline = baseline
        self._baseline_layout = layout
        self.frozen_baseline_state_sha256 = tensor_state_digest010(baseline)
        self.initialization_seed = seed
        self.amplitude_fraction = float(amplitude_fraction)
        cpu_rng_before = torch.get_rng_state().clone()
        # Conv1d's constructor uses the global CPU generator. Fork/restore it,
        # then initialize the head from its own explicit CPU generator only.
        with torch.random.fork_rng(devices=[]):
            head = nn.Conv1d(32, SHAPE_DIMENSIONS010, 5, padding=2, device='cpu', dtype=torch.float32)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            nn.init.normal_(head.weight, mean=0., std=HEAD_INITIALIZATION_STD010, generator=generator)
            nn.init.zeros_(head.bias)
        if not torch.equal(cpu_rng_before, torch.get_rng_state()):
            raise RuntimeError('010 initialization changed the global CPU random stream')
        self.initialization_global_CPU_RNG_preserved = True
        reference = baseline.mean_head.weight
        self.basis_head = head.to(device=reference.device, dtype=reference.dtype)
        self._head_parameter_ids = tuple(id(value) for value in self.basis_head.parameters())
        self.initial_head_state_sha256 = tensor_state_digest010(self.basis_head)
        self.assert_frozen_unchanged(deep=True)

    @property
    def stage(self):
        """Read-only original stage; loss mode is controlled by the driver."""
        return self.baseline.stage

    def train(self, mode: bool = True):
        if type(mode) is not bool:
            raise TypeError('010 train mode must be bool')
        self.assert_frozen_unchanged()
        self.training = mode
        self.basis_head.train(mode)
        self.assert_frozen_unchanged()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """Only the new head can be enabled; A is never reopened recursively."""
        if type(requires_grad) is not bool:
            raise TypeError('010 requires_grad flag must be bool')
        self.assert_frozen_unchanged()
        self.basis_head.requires_grad_(requires_grad)
        self.assert_frozen_unchanged()
        return self

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._TRAINABLE_NAMES

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        values = tuple(self.basis_head.parameters())
        if tuple(id(value) for value in values) != self._head_parameter_ids:
            raise RuntimeError('010 registered new-head parameters were replaced')
        return values

    def assert_optimizer_membership(self, optimizer) -> None:
        self.assert_frozen_unchanged()
        expected = self.trainable_parameters()
        actual = tuple(value for group in optimizer.param_groups for value in group['params'])
        if len(actual) != len(expected) or {id(value) for value in actual} != {id(value) for value in expected}:
            raise ValueError('010 optimizer must contain exactly the new-head weight and bias')
        if not all(value.requires_grad for value in expected):
            raise ValueError('010 optimizer parameters must have the new head explicitly enabled')

    def assert_frozen_unchanged(self, *, deep: bool = False) -> str:
        if _baseline_layout010(self.baseline) != self._baseline_layout:
            raise RuntimeError('010 original A class/stage/eval/parameters/buffers/hooks/stats changed')
        if self.initialization_seed != INITIALIZATION_SEED010 or self.amplitude_fraction != VOLTAGE_GAIN010:
            raise RuntimeError('010 registered shape initialization or voltage gain changed')
        if deep and tensor_state_digest010(self.baseline) != self.frozen_baseline_state_sha256:
            raise RuntimeError('010 original A state bytes changed')
        return self.frozen_baseline_state_sha256

    def _baseline_once(self, source_carrier_V: Tensor) -> tuple[dict, Tensor]:
        captured = []

        def capture_decoded(module, arguments, result):
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError('010 original U-Net decoded interface changed')
            captured.append(result[0])

        hook = self.baseline.unet.register_forward_hook(capture_decoded)
        try:
            noise = {name: torch.zeros((len(source_carrier_V),)+shape, dtype=torch.float32,
                                       device=source_carrier_V.device)
                     for name, shape in self.baseline.noise_shapes.items()}
            with torch.no_grad():
                original = self.baseline.signal_forward(source_carrier_V, noise)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError('010 must reuse exactly one original U-Net decoded forward')
        decoded = captured[0].detach()
        if decoded.shape != (len(source_carrier_V), 32, 384):
            raise ValueError('010 decoded features must be Bx32x384')
        return original, decoded

    def random_forward(self, source_carrier_V: Tensor, z_shape: Tensor, *, enabled: bool = True) -> dict:
        self.assert_frozen_unchanged()
        if type(enabled) is not bool:
            raise TypeError('010 random-forward enable switch must be bool')
        if (not isinstance(source_carrier_V, Tensor) or source_carrier_V.ndim not in (2, 3)
                or len(source_carrier_V) == 0):
            raise ValueError('010 requires a nonempty original SOURCE carrier tensor')
        if (not isinstance(z_shape, Tensor) or z_shape.shape != (len(source_carrier_V), SHAPE_DIMENSIONS010)
                or not z_shape.is_floating_point() or z_shape.device != source_carrier_V.device
                or not bool(torch.isfinite(z_shape).all())):
            raise ValueError('010 requires explicit finite Bx8 shape coefficients on the source device')
        try:
            original, decoded = self._baseline_once(source_carrier_V)
            controls = torch.tanh(control_average008(self.basis_head(decoded)).double())
            smooth = torch.einsum('bkc,tc->bkt', controls, self.baseline.expansion)*self.baseline.gate
            projected = smooth-canonical_charge008(smooth)[..., None]*self.baseline.projection
            # The registered amplitude is max_abs(A), never a stochastic Qsig.
            amplitude = self.amplitude_fraction*original['signal_pre'].abs().amax(1)
            basis_voltage = amplitude[:, None, None]*projected
            if not bool(torch.isfinite(basis_voltage).all()) or bool((basis_voltage[..., :100] != 0).any()):
                raise ValueError('010 voltage bases must be finite with exactly zero prefix100')
            coefficients = z_shape.double() if enabled else torch.zeros_like(z_shape, dtype=torch.float64)
            random_shape = (basis_voltage*coefficients[..., None]).sum(1)
            # Explicit zero rows select the original bytes, including signed zero.
            active = coefficients.ne(0).any(1)
            signal = torch.where(active[:, None], original['signal_pre']+random_shape, original['signal_pre'])
            if not bool(torch.isfinite(signal).all()) or bool((signal[:, :100] != 0).any()):
                raise ValueError('010 signal must be finite with exactly zero prefix100')
            second_difference = torch.diff(random_shape, n=2, dim=-1)
            regular_rms = random_shape.square().mean()/self.baseline.a_ref.square()
            regular_curvature = second_difference.square().mean()/self.baseline.a_ref.square()
            regular = regular_rms+regular_curvature
            output = dict(original)
            output.update(
                signal_pre=signal, base_signal_pre_V=original['signal_pre'],
                pure_signal_pre_V=signal, voltage_basis_V=basis_voltage,
                basis_amplitude_V=amplitude, shape_coefficients=coefficients,
                random_shape_pre_V=random_shape,
                random_RMS_V=random_shape.square().mean(1).sqrt(),
                random_second_difference_RMS_V=second_difference.square().mean(1).sqrt(),
                regular_rms=regular_rms, regular_curvature=regular_curvature,
                regular=regular, base_regular=regular,
                frozen_original_regular=original['regular'],
                canonical_charge=canonical_charge008(signal),
                canonical_charge_identity_residual=canonical_charge008(signal)-original['qsig'],
                canonical_random_shape_charge=canonical_charge008(random_shape),
                basis_canonical_charge=canonical_charge008(basis_voltage),
                random_forward_enabled=enabled,
                baseline_signal_forward_calls=1, extra_baseline_signal_forward_calls=0,
                shape010=dict(schema='n01/frozen-A-voltage-shape010/v1',
                              basis_count=SHAPE_DIMENSIONS010, voltage_gain=self.amplitude_fraction,
                              amplitude='max_abs(original_A_signal_pre)',
                              initialization_seed=self.initialization_seed,
                              initialization_std=HEAD_INITIALIZATION_STD010,
                              frozen_A_stage=self.baseline.stage,
                              frozen_A_state_sha256=self.frozen_baseline_state_sha256,
                              mean_Q_protection='preADC only; not ADC/peak/crop observed Q or PSD'),
            )
            return output
        finally:
            self.assert_frozen_unchanged()

    def forward(self, source_carrier_V: Tensor, z_shape: Tensor, *, enabled: bool = True) -> dict:
        return self.random_forward(source_carrier_V, z_shape, enabled=enabled)
