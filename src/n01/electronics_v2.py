"""Trainable stationary N01 electronics candidate; no source conditioning/drift.

This module is separate from the frozen N01 v1 implementation. Initialization
statistics must be supplied explicitly from TRAIN baselines, not legacy OLS
intercept/slope distributions. All pre-ADC construction is float64.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TAPS = 65
DEFAULT_LENGTH = 480
_REQUIRED_NOISE = ("electronics_white", "electronics_scale", "electronics_offset")
_INITIALIZATION_KEYS = {"rms", "log_scale_std", "offset_mean", "offset_std", "voltage_scale", "metadata"}
_DEFAULTS = {"rms": 1e-3, "log_scale_std": .02, "offset_mean": 0., "offset_std": 1e-4}


def _length(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("length must be a positive integer")
    return value


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class FreeStationaryFIR(nn.Module):
    """Free signed 65-tap causal FIR with exact L2 normalization.

    The unit impulse is only the initial parameter value. There is no tanh,
    residual-size restriction, DC normalization or zero-charge projection.
    """

    def __init__(self):
        super().__init__()
        initial = torch.zeros(TAPS, dtype=torch.float32)
        initial[0] = 1.
        self.raw = nn.Parameter(initial)

    def taps(self) -> Tensor:
        raw = self.raw.to(torch.float64)
        norm = torch.linalg.vector_norm(raw)
        if not bool(torch.isfinite(norm)) or bool(norm <= 1e-12):
            raise ValueError("FIR coefficients must have finite, nonzero L2 norm")
        return raw / norm

    def forward(self, extended_white: Tensor) -> Tensor:
        if extended_white.ndim != 3 or extended_white.shape[1] != 1 or extended_white.shape[2] < TAPS:
            raise ValueError("FIR input must be Bx1x(L+64) with L >= 1")
        # The provided history precedes the crop. No padding or circular wrap.
        white = extended_white.to(torch.float64)
        h = self.taps().to(device=white.device)
        return F.conv1d(white, h.flip(0)[None, None])


class ElectronicsNoiseV2(nn.Module):
    """Stationary correlated noise plus independent event baseline offset.

    y[b,t] = offset[b] + sigma[b] * sum_k h[k]*white[b,t+64-k].
    sigma is lognormal with its arithmetic mean corrected to rms; h has unit
    L2 energy. Offset and amplitude distributions are jointly trainable, with
    independent standard-normal latent draws. No source/particle/ID is accepted.

    ``forward(noise)`` returns Bx1x480 by default. ``length=100`` accepts either
    the original Bx1x544 white input or a dedicated Bx1x164 input, returning the
    same first 100 valid samples for identical leading white/scale/offset draws.
    Other dictionary keys (including the old random slope) are ignored.
    """

    def __init__(self, initialization: Mapping | None = None, *, length: int = DEFAULT_LENGTH):
        super().__init__()
        supplied = {} if initialization is None else dict(initialization)
        unknown = set(supplied) - _INITIALIZATION_KEYS
        if unknown:
            raise ValueError(f"unknown v2 initialization keys {sorted(unknown)}; legacy OLS statistics are not accepted")
        values = _DEFAULTS | supplied
        for key in ("rms", "log_scale_std", "offset_mean", "offset_std"):
            if not math.isfinite(float(values[key])):
                raise ValueError(f"nonfinite initialization {key}")
        for key in ("rms", "log_scale_std", "offset_std"):
            if float(values[key]) <= 0:
                raise ValueError(f"initialization {key} must be positive")
        scale = float(values.get("voltage_scale", values["rms"]))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("voltage_scale must be finite and positive")
        self.length = _length(length)
        self.initialization_metadata = dict(values.get("metadata", {}))
        self.filter = FreeStationaryFIR()
        self.register_buffer("voltage_scale", torch.tensor(scale, dtype=torch.float64))
        # The mean's optimized parameter is dimensionless; common Adam 1e-3
        # therefore changes physical mean at the fixed TRAIN voltage scale.
        self.offset_mean_units = nn.Parameter(torch.tensor(float(values["offset_mean"])/scale, dtype=torch.float32))
        self.log_offset_std = nn.Parameter(torch.tensor(math.log(float(values["offset_std"])), dtype=torch.float32))
        self.log_rms = nn.Parameter(torch.tensor(math.log(float(values["rms"])), dtype=torch.float32))
        self.raw_log_scale_std = nn.Parameter(torch.tensor(_inverse_softplus(float(values["log_scale_std"])), dtype=torch.float32))

    @property
    def offset_mean(self) -> Tensor:
        return self.voltage_scale * self.offset_mean_units.to(torch.float64)

    @property
    def offset_std(self) -> Tensor:
        return self.log_offset_std.to(torch.float64).exp()

    @property
    def log_scale_std(self) -> Tensor:
        return F.softplus(self.raw_log_scale_std.to(torch.float64))

    @property
    def rms_mean(self) -> Tensor:
        return self.log_rms.to(torch.float64).exp()

    @staticmethod
    def noise_shapes(length: int = DEFAULT_LENGTH) -> dict[str, tuple[int, ...]]:
        return {"electronics_white": (1, _length(length)+TAPS-1),
                "electronics_scale": (1, 1), "electronics_offset": (1, 1)}

    def sample_noise(self, batch: int, device=None, generator: torch.Generator | None = None,
                     *, length: int | None = None) -> dict[str, Tensor]:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("batch must be a positive integer")
        length = self.length if length is None else _length(length)
        device = self.voltage_scale.device if device is None else device
        return {key: torch.randn((batch,)+shape,device=device,dtype=torch.float32,generator=generator)
                for key,shape in self.noise_shapes(length).items()}

    def _validated_noise(self, noise: Mapping[str, Tensor], length: int) -> dict[str, Tensor]:
        missing = set(_REQUIRED_NOISE)-set(noise)
        if missing:
            raise ValueError(f"missing explicit electronics noise keys {sorted(missing)}")
        white = noise["electronics_white"]
        if not isinstance(white, Tensor) or white.ndim != 3 or white.shape[1] != 1 or white.shape[0] < 1:
            raise ValueError("electronics_white must be a nonempty Bx1x(L+64) tensor")
        if white.shape[-1] < length+TAPS-1:
            raise ValueError(f"electronics_white needs at least {length+TAPS-1} samples")
        batch = white.shape[0]
        result = {}
        for key in _REQUIRED_NOISE:
            value = noise[key]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise ValueError(f"{key} must be a floating point tensor")
            if key != "electronics_white" and value.shape != (batch,1,1):
                raise ValueError(f"{key} must be Bx1x1")
            if value.device != white.device:
                raise ValueError("explicit electronics noise tensors must be on the same device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite explicit noise {key}")
            result[key] = value.to(torch.float64)
        result["electronics_white"] = result["electronics_white"][..., :length+TAPS-1]
        return result

    def forward(self, noise: Mapping[str, Tensor], *, length: int | None = None) -> Tensor:
        length = self.length if length is None else _length(length)
        z = self._validated_noise(noise,length)
        spread = self.log_scale_std
        sigma = torch.exp(self.log_rms.to(torch.float64)+spread*z["electronics_scale"]-.5*spread.square())
        offset = self.offset_mean+self.offset_std*z["electronics_offset"]
        output = offset+sigma*self.filter(z["electronics_white"])
        if not bool(torch.isfinite(output).all()):
            raise ValueError("nonfinite stationary electronics output; no rows removed")
        return output

    def theoretical_stats(self) -> dict[str, Tensor]:
        """Differentiable ensemble moments before ADC; no empirical fit claim."""
        spread = self.log_scale_std
        sigma2 = torch.exp(2*self.log_rms.to(torch.float64)+spread.square())
        return {"offset_mean": self.offset_mean, "offset_std": self.offset_std,
                "rms_arithmetic_mean": self.rms_mean, "log_scale_std": spread,
                "sigma_squared_mean": sigma2,
                "sample_variance_stationary_component": sigma2,
                "sample_variance_total": sigma2+self.offset_std.square(),
                "voltage_scale": self.voltage_scale}

    def autocovariance(self, max_lag: int, *, include_offset: bool = True) -> Tensor:
        """Ensemble covariance at integer lags, including finite FIR support."""
        if isinstance(max_lag,bool) or not isinstance(max_lag,int) or max_lag < 0:
            raise ValueError("max_lag must be a nonnegative integer")
        h = self.filter.taps()
        sigma2 = self.theoretical_stats()["sigma_squared_mean"]
        result = torch.stack([torch.dot(h[:-lag],h[lag:]) if 0 < lag < TAPS
                              else h.square().sum() if lag==0 else h.sum()*0.
                              for lag in range(max_lag+1)])*sigma2
        return result+self.offset_std.square() if include_offset else result

    def linear_statistic_variance(self, weights: Tensor, *, include_offset: bool = True) -> Tensor:
        """Variance of sum_t weights[t]*y[t] for a stationary preADC process."""
        if weights.ndim != 1 or weights.numel() < 1 or not bool(torch.isfinite(weights).all()):
            raise ValueError("weights must be a finite nonempty one-dimensional tensor")
        w = weights.to(device=self.voltage_scale.device,dtype=torch.float64)
        length = w.numel()
        cov = self.autocovariance(min(length-1,TAPS-1),include_offset=False)
        variance = cov[0]*w.square().sum()
        for lag in range(1,len(cov)):
            variance = variance+2*cov[lag]*torch.dot(w[:-lag],w[lag:])
        if include_offset:
            variance = variance+self.offset_std.square()*w.sum().square()
        return variance


def fit_electronics_initialization(train_baselines, *, log_scale_std: float = .02,
                                   std_floor: float = 1e-7) -> dict:
    """Transparent, rough TRAIN-only initialization; no OLS coefficient fit.

    The caller supplies only baseline samples and owns the data-role check.
    Per-event first differences remove event offsets. mean(diff^2)/2 gives a
    rough white-initialization second-moment estimate, not a physically identified
    correlated-noise amplitude. Offset-mean variance is reduced by that estimated
    stationary sample-mean variance at the initial unit-impulse FIR. Negative
    residuals use a declared floor. logscale spread is the fixed near-zero .02
    engineering initial value; observed finite-window RMS spread is not copied.
    Joint baseline training must subsequently identify all free distributions.
    """
    if not math.isfinite(log_scale_std) or log_scale_std <= 0 or not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("initial spread and std_floor must be finite and positive")
    y = torch.as_tensor(train_baselines).detach().to(device="cpu",dtype=torch.float64)
    if y.ndim==3 and y.shape[1]==1:
        y=y[:,0]
    if y.ndim!=2 or min(y.shape)<2 or not bool(torch.isfinite(y).all()):
        raise ValueError("TRAIN baselines must be finite NxL with N,L >= 2; no rows removed")
    event_diff_sigma2 = (y[:,1:]-y[:,:-1]).square().mean(1)/2.
    rough_sigma2_mean = float(event_diff_sigma2.mean())
    if rough_sigma2_mean<=0:
        raise ValueError("TRAIN differences have zero variance; no positive stationary noise initialization")
    rms_mean = float(event_diff_sigma2.clamp_min(std_floor**2).sqrt().mean())
    mean = y.mean(1)
    observed_mean_var = float(mean.var(unbiased=True))
    # At the actual impulse initialization, E[sigma^2]=rms^2*exp(spread^2).
    impulse_sigma2_mean = rms_mean**2*math.exp(log_scale_std**2)
    initial_stationary_mean_var = impulse_sigma2_mean/y.shape[1]
    residual = observed_mean_var-initial_stationary_mean_var
    offset_std = math.sqrt(max(residual,std_floor**2))
    observed_log_rms_std = float(.5*event_diff_sigma2.clamp_min(std_floor**2).log().std(unbiased=True))
    return {"rms": rms_mean,"log_scale_std": log_scale_std,"offset_mean": float(mean.mean()),
            "offset_std": offset_std,"voltage_scale": rms_mean,
            "metadata": {"input_role": "caller_declared_TRAIN_baselines","events": y.shape[0],"samples": y.shape[1],
                         "mean_variance_ddof": 1,"observed_baseline_mean_variance": observed_mean_var,
                         "rough_difference_sigma_squared_mean": rough_sigma2_mean,
                         "initial_impulse_sigma_squared_mean": impulse_sigma2_mean,
                         "stationary_mean_variance_subtracted": initial_stationary_mean_var,
                         "offset_variance_residual_before_floor": residual,"offset_std_floor": std_floor,
                         "offset_floor_used": residual < std_floor**2,
                         "observed_finite_window_log_diff_rms_std_not_used_as_latent": observed_log_rms_std,
                         "log_scale_initialization": "fixed_near_zero_engineering_value",
                         "FIR_initialization": "unit_impulse_only",
                         "rough_assumptions": "difference amplitude uses white initialization; correlations and finite-window sampling are not latent OLS drift"}}


def make_n01_response_v2(stats: dict | None = None, variant: str = "full", *,
                          electronics_initialization: Mapping | None = None,
                          electronics: ElectronicsNoiseV2 | None = None,
                          stage: str = "joint_stochastic"):
    """Construct a new N01 response and replace only electronics, leaving v1 intact.

    Does not read data, copy legacy electronics weights or load any checkpoint.
    An explicitly supplied trained v2 module can be installed by the caller.
    The original full-six-key sample_noise API and all non-electronics behavior
    are retained. Calling set_stage after replacement keeps existing freeze rules.
    """
    from .model import N01Response
    if electronics is not None and electronics_initialization is not None:
        raise ValueError("supply either a v2 module or its initialization, not both")
    if electronics is not None and not isinstance(electronics,ElectronicsNoiseV2):
        raise TypeError("electronics must be ElectronicsNoiseV2")
    response=N01Response(stats,variant)
    response.electronics=electronics if electronics is not None else ElectronicsNoiseV2(electronics_initialization)
    if response.electronics.length!=DEFAULT_LENGTH:
        raise ValueError("N01 response electronics default length must be 480")
    response.set_stage(stage)
    return response
