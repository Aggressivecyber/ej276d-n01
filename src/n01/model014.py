"""WP2 uses the exact 010 head layout with an explicit, paired initial seed.

Only initialization differs from 010. The gain remains .05 throughout training;
calibration changes the initial head bytes, never the future representation.
No data, checkpoint, ADC, optimizer, or implicit random stream is opened here.
"""
from __future__ import annotations

import torch
from torch import nn

from .model008 import canonical_charge008, control_average008
from .model010 import (
    FrozenDeterministicVoltageShape010, _baseline_layout010,
    tensor_state_digest010, HEAD_INITIALIZATION_STD010, VOLTAGE_GAIN010,
    SHAPE_DIMENSIONS010,
)


class FrozenDeterministicVoltageShape014(FrozenDeterministicVoltageShape010):
    """Same A and 32x8x5+bias head, with a separately registered initial seed."""

    def __init__(self, baseline, *, seed: int):
        nn.Module.__init__(self)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('014 requires an explicit nonnegative integer head seed')
        layout = _baseline_layout010(baseline)
        if baseline.unet is None or baseline.mean_head is None:
            raise ValueError('014 requires the original deterministic U-Net and mean head')
        self.baseline, self._baseline_layout = baseline, layout
        self.frozen_baseline_state_sha256 = tensor_state_digest010(baseline)
        self.initialization_seed, self._registered_seed = seed, seed
        self.amplitude_fraction = VOLTAGE_GAIN010
        before = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            head = nn.Conv1d(32, SHAPE_DIMENSIONS010, 5, padding=2, device='cpu', dtype=torch.float32)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            nn.init.normal_(head.weight, mean=0., std=HEAD_INITIALIZATION_STD010, generator=generator)
            nn.init.zeros_(head.bias)
        if not torch.equal(before, torch.get_rng_state()):
            raise RuntimeError('014 initial head changed global CPU RNG')
        self.initialization_global_CPU_RNG_preserved = True
        reference = baseline.mean_head.weight
        self.basis_head = head.to(device=reference.device, dtype=reference.dtype)
        self._head_parameter_ids = tuple(id(value) for value in self.basis_head.parameters())
        self.initial_head_state_sha256 = tensor_state_digest010(self.basis_head)
        if sum(value.numel() for value in self.basis_head.parameters()) != 1288:
            raise RuntimeError('014 head layout changed')
        self.assert_frozen_unchanged(deep=True)

    def assert_frozen_unchanged(self, *, deep=False):
        if _baseline_layout010(self.baseline) != self._baseline_layout:
            raise RuntimeError('014 original A class/stage/eval/parameters/buffers/hooks/stats changed')
        if self.initialization_seed != self._registered_seed or self.amplitude_fraction != VOLTAGE_GAIN010:
            raise RuntimeError('014 initial seed or fixed .05 gain changed')
        if deep and tensor_state_digest010(self.baseline) != self.frozen_baseline_state_sha256:
            raise RuntimeError('014 original A state bytes changed')
        return self.frozen_baseline_state_sha256

    def random_forward(self, source_carrier_V, z_shape, *, enabled=True):
        # The actual 010 arithmetic and all its engineering guards are reused.
        out = super().random_forward(source_carrier_V, z_shape, enabled=enabled)
        metadata = dict(out.pop('shape010'))
        metadata.update(schema='n01/frozen-A-voltage-shape014/v1',
                        only_010_change='explicit registered initial seed and calibrated initial bytes',
                        persistent_calibration_multiplier=False)
        out['shape014'] = metadata
        return out

    forward = random_forward

    def cached_initial_logits(self, decoded):
        """One actual head kernel over cached, frozen decoded features."""
        return control_average008(self.basis_head(decoded)).double()

    def random_from_logits(self, logits, base_signal, z_shape, *, multiplier=1.):
        """The same voltage arithmetic as 010, used only for initial RMS fitting.

        This multiplier is a local argument, not model state. No output or loss
        calibration value can affect later forward calls.
        """
        if logits.shape != (len(base_signal), 8, 48) or base_signal.shape[1:] != (384,):
            raise ValueError('014 cached calibration layout changed')
        if z_shape.shape != (len(base_signal), 8):
            raise ValueError('014 requires explicit calibration Bx8 coefficients')
        controls = torch.tanh(logits * float(multiplier))
        smooth = torch.einsum('bkc,tc->bkt', controls, self.baseline.expansion)*self.baseline.gate
        projected = smooth-canonical_charge008(smooth)[..., None]*self.baseline.projection
        amplitude = self.amplitude_fraction*base_signal.abs().amax(1)
        basis = amplitude[:, None, None]*projected
        return (basis*z_shape.double()[..., None]).sum(1)


@torch.no_grad()
def calibrate_initial_head014(model, decoded, base_signal, z_banks, *, target_RMS_V,
                              bisections, operation=None):
    """Fixed finite RMS root solve on cached TRAIN A output, with fresh-head QA.

    decoded/base_signal contain one row per calibration parent. z_banks is
    KxPxDx8; no extra A forward is performed. Only initial head parameters are
    scaled. Every initial control/RMS evaluation is explicitly chargeable.
    """
    if target_RMS_V <= 0 or type(bisections) is not int or not 1 <= bisections <= 32:
        raise ValueError('014 calibration requires a positive target and finite 1..32 bisections')
    if z_banks.ndim != 4 or z_banks.shape[1] != len(decoded) or z_banks.shape[-1] != 8:
        raise ValueError('014 calibration banks must be KxPxDx8')
    if decoded.shape != (len(base_signal), 32, 384) or base_signal.shape != (len(decoded), 384):
        raise ValueError('014 frozen calibration cache shape changed')
    if not all(bool(torch.isfinite(x).all()) for x in (decoded, base_signal, z_banks)):
        raise ValueError('014 nonfinite calibration cache; no rows dropped')
    if operation is None:
        operation = lambda name, charges, fn, **meta: fn()
    parents, banks, draws = len(decoded), len(z_banks), z_banks.shape[2]
    initial = {name: value.detach().clone() for name, value in model.basis_head.named_parameters()}
    initial_sha = tensor_state_digest010(model.basis_head)
    logits = operation('calibration_actual_head_logits', dict(head_actual_parameter_forward_rows=parents),
                       lambda: model.cached_initial_logits(decoded))
    expanded_base = base_signal.repeat_interleave(draws, dim=0)
    expanded_logits = logits.repeat_interleave(draws, dim=0)

    def measure(alpha, cached_logits=expanded_logits):
        def evaluate():
            per_bank = [model.random_from_logits(cached_logits, expanded_base,
                        z_banks[bank].reshape(parents*draws, 8), multiplier=alpha)
                        for bank in range(banks)]
            return torch.stack(per_bank).square().mean().sqrt().item()
        return operation('calibration_voltage_RMS_map', dict(calibration_map_rows=banks*parents*draws),
                         evaluate, initial_weight_multiplier=float(alpha))

    history = [dict(multiplier=1., RMS_V=measure(1.))]
    already_below = history[0]['RMS_V'] <= target_RMS_V
    low, high = (1., 1.) if already_below else (0., 1.)
    if not already_below:
        for _ in range(bisections):
            middle = .5*(low+high)
            measured = measure(middle)
            history.append(dict(multiplier=middle, RMS_V=measured))
            if measured <= target_RMS_V:
                low = middle
            else:
                high = middle
    multiplier = low
    for name, value in model.basis_head.named_parameters():
        value.copy_(initial[name]*multiplier)
    # Verify actual scaled parameter arithmetic, including float32 rounding.
    actual_logits = operation('calibration_actual_scaled_head_logits', dict(head_actual_parameter_forward_rows=parents),
                             lambda: model.cached_initial_logits(decoded))
    actual = measure(1., actual_logits.repeat_interleave(draws, dim=0))
    if actual > target_RMS_V:
        raise ValueError('014 final actual-head RMS exceeds the registered upper bound')
    base_RMS = base_signal.square().mean().sqrt().item()
    model.assert_frozen_unchanged(deep=True)
    return dict(schema='n01/wp2-initial-RMS-calibration/v1', initial_seed=model.initialization_seed,
        uncalibrated_head_sha256=initial_sha, calibrated_head_sha256=tensor_state_digest010(model.basis_head),
        head_parameters=1288, parents=parents, banks=banks, draws_per_parent=draws,
        carrier_samples=384, RMS_scope='all bank,parent,draw,384 carrier samples including zero prefix100',
        target_RMS_V=float(target_RMS_V), target_is_upper_bound=True, actual_RMS_V=actual,
        base_A_RMS_V=base_RMS, actual_RMS_relative_A=actual/base_RMS if base_RMS else None,
        signed_residual_RMS_V=actual-target_RMS_V, relative_residual=(actual-target_RMS_V)/target_RMS_V,
        initial_weight_multiplier=multiplier, bisection_lower=low, bisection_upper=high,
        fixed_bisections=bisections, actual_bisections=0 if already_below else bisections,
        calibration_status='already_below' if already_below else 'lower_bracket_scaled', history=history,
        future_voltage_gain=.05, persistent_multiplier=False, loss_evaluations=0,
        actual_A_forward_rows_inside_solver=0, later_capacity_changed=False)
