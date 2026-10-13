"""N01-008 short carrier and common 256-sample observation.

This module does not load data, draw randomness, train, or infer hardware timing.
SOURCE values stay clean. REAL enters already canonical ADC; generated carriers
receive the original ADC once. Both observed domains then share baseline,
bounded first-argmax and crop code. The 005 prefix is nonstationary effective
observed noise; the three extensions remain explicit sensitivity hypotheses.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .electronics_v3 import ObservedBaselineNoiseV3
from .extension006 import HYPOTHESES006
from .model import ADC_MIN, ADC_MAX, ADC_BITS, ADC_LEVELS, digitize

CARRIER_LENGTH = 384
PREFIX_LENGTH = 100
EXTENSION_LENGTH = CARRIER_LENGTH - PREFIX_LENGTH
WINDOW_LENGTH = 256
PEAK_SEARCH_START = 112
PEAK_SEARCH_STOP = 144
PRE_PEAK = 16
POST_PEAK_EXCLUSIVE = 240
TAIL_START = 46
SAMPLE_INTERVAL_NS = 1.0  # bound software representation, not hardware proof
HYPOTHESES008 = tuple(spec["name"] for spec in HYPOTHESES006)
ObservedBaselineNoise005 = ObservedBaselineNoiseV3


def _carrier(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != CARRIER_LENGTH or value.shape[0] < 1:
        raise ValueError("008 requires a nonempty Bx384 carrier; no padding or row deletion")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("008 carrier must contain finite floating-point voltage values")
    return value.to(torch.float64)


def peak_indices_torch(carrier: Tensor) -> Tensor:
    """First argmax in the registered [112,144) software search interval."""
    value = _carrier(carrier)
    return value[:, PEAK_SEARCH_START:PEAK_SEARCH_STOP].argmax(1) + PEAK_SEARCH_START


def crop_indices_torch(peaks: Tensor) -> Tensor:
    if not isinstance(peaks, Tensor) or peaks.ndim != 1 or peaks.numel() < 1 or peaks.dtype != torch.long:
        raise ValueError("008 peak indices must be a nonempty int64 vector")
    if bool(((peaks < PEAK_SEARCH_START) | (peaks >= PEAK_SEARCH_STOP)).any()):
        raise ValueError("008 crop peak outside the frozen software search; do not silently clamp")
    indices = peaks[:, None] - PRE_PEAK + torch.arange(WINDOW_LENGTH, device=peaks.device)
    if bool(((indices < 0) | (indices >= CARRIER_LENGTH)).any()):
        raise ValueError("008 crop exceeds carrier support; no padding or extrapolation")
    return indices


def crop256_torch(carrier: Tensor, peaks: Tensor | None = None) -> Tensor:
    """Gather raw values; this helper performs neither ADC nor baseline removal.

    Argmax/crop indices have no gradient. Selected values retain their normal
    gather gradient. Use observation functions for the complete readout chain.
    """
    value = _carrier(carrier)
    peaks = peak_indices_torch(value) if peaks is None else peaks
    if peaks.device != value.device or peaks.shape != (len(value),):
        raise ValueError("008 peak vector must match carrier batch and device")
    return value.gather(1, crop_indices_torch(peaks))


def _observation(value: Tensor, *, mode: str, ste: bool, q_floor: float | None,
                 return_carrier: bool, return_indices: bool) -> dict:
    value = _carrier(value)
    if q_floor is not None and (not math.isfinite(float(q_floor)) or float(q_floor) <= 0):
        raise ValueError("q_floor must be explicitly finite and positive, or None")
    if mode == "new_ADC":
        measured = digitize(value, ste=ste)
    elif mode == "existing_ADC":
        canonical = digitize(value, ste=False)
        if not torch.equal(value, canonical):
            raise ValueError("already_digitized requires canonical float64 ADC values; canonicalize stored float32 REAL first")
        measured = value
    elif mode == "clean_values_no_ADC":
        measured = value
    else:
        raise ValueError("Unknown 008 observation mode")

    # Retain the actual pre-crop 100 values, never the first 100 of the 256 crop.
    prefix = measured[:, :PREFIX_LENGTH].clone()
    baseline = prefix.mean(1)
    centered = measured - baseline[:, None]
    peaks = peak_indices_torch(centered)
    indices = crop_indices_torch(peaks)
    waveform = centered.gather(1, indices)
    q = waveform.sum(1) * SAMPLE_INTERVAL_NS
    tail = waveform[:, TAIL_START:].sum(1) * SAMPLE_INTERVAL_NS
    nonzero = q != 0
    safe_q = torch.where(nonzero, q, torch.ones_like(q))
    raw_psd = torch.where(nonzero, tail / safe_q, torch.full_like(q, float("nan")))
    if q_floor is None:
        guarded = torch.zeros_like(q, dtype=torch.bool)
        psd = raw_psd
    else:
        floor = float(q_floor)
        guarded = q.abs() < floor
        protected = torch.where(q < 0, -torch.full_like(q, floor), torch.full_like(q, floor))
        psd = tail / torch.where(guarded, protected, q)
    global_peak = centered.argmax(1)
    flags = dict(
        peak_at_search_boundary=(peaks == PEAK_SEARCH_START) | (peaks == PEAK_SEARCH_STOP - 1),
        global_peak_outside_search=(global_peak < PEAK_SEARCH_START) | (global_peak >= PEAK_SEARCH_STOP),
        Q_nonpositive=q <= 0,
        PSD_undefined=~nonzero,
        PSD_guarded=guarded,
    )
    if mode != "clean_values_no_ADC":
        flags["ADC_rail_hit"] = ((measured == ADC_MIN) | (measured == ADC_MAX)).any(1)
    if mode == "new_ADC":
        flags["pre_ADC_out_of_range"] = ((value < ADC_MIN) | (value > ADC_MAX)).any(1)
    result = dict(
        waveform256=waveform, window_V=waveform,
        baseline100=prefix, baseline100_adc=prefix if mode != "clean_values_no_ADC" else None,
        baseline_mean_V=baseline, peak_index=peaks, global_peak_index=global_peak,
        crop_start=peaks - PRE_PEAK, crop_stop=peaks + POST_PEAK_EXCLUSIVE,
        Q=q, tail=tail, PSD_raw=raw_psd, PSD=psd, flags=flags,
        input_kind=mode, ADC_applied=(mode == "new_ADC"),
        q_floor=None if q_floor is None else float(q_floor),
    )
    if return_indices:
        result.update(crop_indices=indices, indices=indices)
    if return_carrier:
        adc = measured if mode != "clean_values_no_ADC" else None
        result.update(carrier_input=value, carrier_adc=adc, adc_carrier=adc,
                      pre_adc_carrier=value if mode == "new_ADC" else None)
    return result


def observe_torch(carrier: Tensor, *, ste: bool = False, already_digitized: bool = False,
                  q_floor: float | None = None, return_carrier: bool = False,
                  return_indices: bool = False) -> dict:
    """ADC (or verified ADC input) -> mean baseline -> peak -> 256 crop.

    Peak-edge and outside-global-peak flags are reported without filtering,
    stopping or changing the search interval. No second ADC follows subtraction.
    ``q_floor=None`` preserves a raw ratio, including NaN for Q==0. Callers own
    any predeclared physical/finite-output gates and must not silently drop rows.
    """
    return _observation(carrier, mode="existing_ADC" if already_digitized else "new_ADC", ste=ste,
                        q_floor=q_floor, return_carrier=return_carrier, return_indices=return_indices)


def _numpy_input(value) -> Tensor:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
        raise ValueError("008 NumPy carrier must be a floating-point Bx384 array")
    # No input mutation. All observations use the same CPU torch implementation.
    return torch.from_numpy(np.ascontiguousarray(array)).to(torch.float64)


def _numpy_result(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _numpy_result(item) for key, item in value.items()}
    return value


def observe_numpy(carrier, *, already_digitized: bool = False, q_floor: float | None = None,
                  return_carrier: bool = False, return_indices: bool = False) -> dict:
    """NumPy input/output adapter for the exact shared observation code."""
    with torch.no_grad():
        return _numpy_result(observe_torch(_numpy_input(carrier), already_digitized=already_digitized,
                             q_floor=q_floor, return_carrier=return_carrier, return_indices=return_indices))


def peak_indices_numpy(carrier):
    with torch.no_grad():
        return peak_indices_torch(_numpy_input(carrier)).cpu().numpy()


def crop256_numpy(carrier, peaks=None):
    with torch.no_grad():
        value = _numpy_input(carrier)
        indices = None if peaks is None else torch.as_tensor(peaks, dtype=torch.long)
        return crop256_torch(value, indices).cpu().numpy()


def crop_existing_values(carrier, *, q_floor: float | None = None, return_carrier: bool = False,
                         return_indices: bool = True) -> dict:
    """Clean SOURCE audit: baseline/crop only; never ADC or ADC-grid validation.

    This is not the generated/REAL ADC observation. It retains pristine analytic
    voltage values and uses the same baseline, peak and crop geometry.
    """
    if isinstance(carrier, Tensor):
        return _observation(carrier, mode="clean_values_no_ADC", ste=False, q_floor=q_floor,
                            return_carrier=return_carrier, return_indices=return_indices)
    with torch.no_grad():
        return _numpy_result(_observation(_numpy_input(carrier), mode="clean_values_no_ADC", ste=False,
                             q_floor=q_floor, return_carrier=return_carrier, return_indices=return_indices))


def observe_generated_carrier(pre_adc: Tensor, *, ste: bool = False, q_floor: float | None = None,
                              return_carrier: bool = False, return_indices: bool = True) -> dict:
    return observe_torch(pre_adc, ste=ste, q_floor=q_floor,
                         return_carrier=return_carrier, return_indices=return_indices)


def observe_existing_carrier(carrier, *, already_digitized: bool = True, q_floor: float | None = None,
                             return_carrier: bool = False, return_indices: bool = True) -> dict:
    options = dict(already_digitized=already_digitized, q_floor=q_floor,
                   return_carrier=return_carrier, return_indices=return_indices)
    return observe_torch(carrier, **options) if isinstance(carrier, Tensor) else observe_numpy(carrier, **options)


class FrozenObservedNoise008(nn.Module):
    """Frozen 005 prefix with the original three conditional laws on 384 points.

    Construction copies the supplied 005 model; no caller state is mutated.
    Explicit independent latents are supplied by the caller. No source, parent
    ID, observed peak or label is an input. Prefix, scale and extension epsilon
    are shared across hypotheses. The actual covariance uses C100, never K_BB.
    """
    length = CARRIER_LENGTH
    prefix_length = PREFIX_LENGTH
    extension_length = EXTENSION_LENGTH

    def __init__(self, model005: ObservedBaselineNoiseV3, hypotheses: Sequence | None = None,
                 ridge_relative: float = 1e-6):
        super().__init__()
        if not isinstance(model005, ObservedBaselineNoiseV3):
            raise ValueError("008 requires the original ObservedBaselineNoiseV3 model")
        if hypotheses is not None and list(hypotheses) not in (list(HYPOTHESES008), list(HYPOTHESES006)):
            raise ValueError("008 fixes exactly the unchanged fast/slow/common_level hypotheses")
        if ridge_relative != 1e-6:
            raise ValueError("008 retains the 006 full-kernel ridge 1e-6")
        if any(p.dtype != torch.float64 for p in model005.parameters()):
            raise ValueError("005 parameters must retain float64")
        self.model005 = copy.deepcopy(model005).eval().requires_grad_(False)
        self.hypotheses = HYPOTHESES008
        with torch.no_grad():
            mean = self.model005.mean_vector().detach().clone()
            lower = self.model005.cholesky_factor().detach().clone()
            eta = self.model005.log_scale_std.detach().clone()
            covariance = lower @ lower.T
            variance = covariance.diagonal().mean()
            if not bool(torch.isfinite(covariance).all() & torch.isfinite(mean).all() & torch.isfinite(eta)) or not bool(variance > 0):
                raise ValueError("Frozen 005 moments must be finite with positive variance")
            mean_reference = mean.mean()
            self.register_buffer("mean_prefix", mean)
            self.register_buffer("covariance_prefix", covariance)
            self.register_buffer("eta", eta)
            self.register_buffer("kernel_variance", variance)
            self.register_buffer("mean_reference", mean_reference)
            position = torch.arange(CARRIER_LENGTH, dtype=torch.float64, device=mean.device)
            distance = (position[:, None] - position[None, :]).abs()
            eye = torch.eye(CARRIER_LENGTH, dtype=torch.float64, device=mean.device)
            for spec in HYPOTHESES006:
                kernel = spec["short_weight"] * torch.exp(-distance / spec["short_length"])
                if spec["long_type"] == "exponential":
                    kernel = kernel + spec["long_weight"] * torch.exp(-distance / spec["long_length"])
                elif spec["long_type"] == "constant":
                    kernel = kernel + spec["long_weight"]
                kernel = variance * (kernel + ridge_relative * eye)
                kbb = kernel[:PREFIX_LENGTH, :PREFIX_LENGTH]
                kbt = kernel[:PREFIX_LENGTH, PREFIX_LENGTH:]
                a = torch.cholesky_solve(kbt, torch.linalg.cholesky(kbb)).T
                residual = kernel[PREFIX_LENGTH:, PREFIX_LENGTH:] - a @ kbt
                residual = (residual + residual.T) * 0.5
                factor = torch.linalg.cholesky(residual)  # no adaptive ridge/eigenvalue clipping
                tail_mean = mean_reference + a @ (mean - mean_reference)
                cross = covariance @ a.T
                joint_covariance = torch.cat((torch.cat((covariance, cross), 1),
                    torch.cat((cross.T, a @ covariance @ a.T + residual), 1)), 0)
                for key, value in dict(A=a, R=residual, cholesky_R=factor,
                                       mean=torch.cat((mean, tail_mean)), covariance=joint_covariance).items():
                    self.register_buffer(spec["name"] + "_" + key, value)
        self._frozen_versions = {name: value._version for name, value in self._named_state()}
        self.train(False)
        self.assert_frozen()

    def _named_state(self):
        yield from self.named_parameters()
        yield from self.named_buffers()

    def train(self, mode: bool = True):
        return super().train(False)

    def assert_frozen(self):
        for name, value in self._named_state():
            if name not in self._frozen_versions or value._version != self._frozen_versions[name]:
                raise ValueError("Frozen 008 noise state mutated: " + name)
            if value.is_floating_point() and value.dtype != torch.float64:
                raise ValueError("008 frozen noise requires float64 moments and parameters")
            if isinstance(value, nn.Parameter) and value.requires_grad:
                raise ValueError("008 noise may not receive training gradients")

    def _law(self, hypothesis: str, key: str) -> Tensor:
        if hypothesis not in self.hypotheses:
            raise ValueError("Unknown 008 extension hypothesis")
        return getattr(self, hypothesis + "_" + key)

    @torch.no_grad()
    def sample_prefix(self, noise005: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.assert_frozen()
        prefix = self.model005(noise005, length=PREFIX_LENGTH)[:, 0]
        scale = torch.exp(self.eta * noise005["baseline_scale"].to(torch.float64) - self.eta.square())[:, 0]
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("008 shared scale must be finite and strictly positive")
        return dict(prefix_pre=prefix, scale=scale)

    @torch.no_grad()
    def extend_prefix(self, prefix_pre: Tensor, scale: Tensor, extension_gaussian: Tensor,
                      hypothesis: str) -> dict[str, Tensor]:
        self.assert_frozen()
        prefix = prefix_pre[:, 0] if prefix_pre.ndim == 3 and prefix_pre.shape[1] == 1 else prefix_pre
        spread = scale[:, 0] if scale.ndim == 3 and scale.shape[1] == 1 else scale
        epsilon = extension_gaussian[:, 0] if extension_gaussian.ndim == 3 and extension_gaussian.shape[1] == 1 else extension_gaussian
        batch = len(prefix)
        for name, value, shape in (("prefix", prefix, (batch, PREFIX_LENGTH)), ("scale", spread, (batch, 1)),
                                   ("epsilon", epsilon, (batch, EXTENSION_LENGTH))):
            if not isinstance(value, Tensor) or value.shape != shape or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("008 malformed explicit " + name)
            if value.device != self.mean_prefix.device:
                raise ValueError("008 latents and frozen noise must share a device")
        if batch < 1 or prefix.dtype != torch.float64 or spread.dtype != torch.float64 or not bool((spread > 0).all()):
            raise ValueError("008 prefix/scale must be nonempty original float64 values")
        a, factor = self._law(hypothesis, "A"), self._law(hypothesis, "cholesky_R")
        tail = self._law(hypothesis, "mean")[PREFIX_LENGTH:][None] + (prefix - self.mean_prefix) @ a.T
        tail = tail + spread * (epsilon.to(torch.float64) @ factor.T)
        noise = torch.cat((prefix, tail), 1)
        if not bool(torch.isfinite(noise).all()):
            raise ValueError("008 nonfinite noise; no row removal")
        return dict(noise_pre=noise, prefix_pre=prefix, scale=spread)

    @torch.no_grad()
    def sample_all(self, noise005: Mapping[str, Tensor], extension_gaussian: Tensor) -> dict:
        common = self.sample_prefix(noise005)
        noises = {h: self.extend_prefix(common["prefix_pre"], common["scale"], extension_gaussian, h)["noise_pre"]
                  for h in self.hypotheses}
        return dict(noise_pre=noises, **common)

    @torch.no_grad()
    def extend(self, noise005: Mapping[str, Tensor], extension_gaussian: Tensor, hypothesis: str) -> dict:
        common = self.sample_prefix(noise005)
        return self.extend_prefix(common["prefix_pre"], common["scale"], extension_gaussian, hypothesis)

    def forward(self, noise005: Mapping[str, Tensor], extension_gaussian: Tensor) -> dict:
        return self.sample_all(noise005, extension_gaussian)

    @torch.no_grad()
    def theoretical(self, hypothesis: str) -> dict[str, Tensor]:
        """Unquantized carrier moments; not peak-selected or hardware moments."""
        self.assert_frozen()
        return {key: self._law(hypothesis, key).clone() for key in ("mean", "covariance", "A", "R", "cholesky_R")}


ConditionalExtension008 = FrozenObservedNoise008
