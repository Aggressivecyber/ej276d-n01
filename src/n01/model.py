"""N01's proposed effective random response, with no inherited model weights.

Network features use float32; the Q operator and assembled pre-ADC waveform use
float64. IDs and source composition are deliberately absent from this API.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LENGTH = 480
TIME_ANCHOR = 120.0  # first source half-rise sample in the verified input code
ADC_MIN, ADC_MAX, ADC_BITS = -0.5, 0.5, 14
ADC_LEVELS = 2**ADC_BITS - 1
AMPLITUDE_FLOOR = 1e-8
SOURCE_Q_FLOOR = 1e-8
SHAPE_Q_FLOOR = 1e-6


class InvalidWaveformError(ValueError):
    """An invalid parent or generated shape; no row is silently removed."""

    def __init__(self, reason: str, rows: Tensor, values: Tensor | None = None):
        self.reason = reason
        self.rows = rows.detach().cpu().tolist()
        self.values = None if values is None else values.detach().cpu().tolist()
        super().__init__(f"{reason}; batch rows={self.rows}; values={self.values}")


def charge(x: Tensor) -> Tensor:
    """Window charge in V ns at 1 ns/sample, retaining leading dimensions."""
    if x.shape[-1] != LENGTH:
        raise ValueError(f"charge requires {LENGTH} samples, got {x.shape[-1]}")
    v = x.to(torch.float64)
    return v[..., 100:400].sum(-1) - 3.0 * v[..., :100].sum(-1)


def digitize(x: Tensor, ste: bool = True) -> Tensor:
    """Canonical 14-bit inclusive endpoint grid; nearest-even torch.round.

    STE is identity inside the ADC interval and has the clamp's zero gradient
    outside it. Its forward values are exactly the canonical rounded values.
    """
    clipped = x.clamp(ADC_MIN, ADC_MAX)
    canonical = torch.round((clipped - ADC_MIN) * ADC_LEVELS) / ADC_LEVELS + ADC_MIN
    if ste:
        return canonical.detach() + (clipped - clipped.detach())
    return canonical


def _invalid(values: Tensor, floor: float, reason: str) -> None:
    bad = (~torch.isfinite(values)) | (values <= floor)
    if bool(bad.any()):
        rows = torch.nonzero(bad, as_tuple=False).flatten()
        raise InvalidWaveformError(reason, rows, values[bad])


def causal_fir(x: Tensor, taps: Tensor, *, extended: bool = False) -> Tensor:
    """Linear causal convolution, never circular.

    For ``extended=True``, x begins with K-1 real history samples and only the
    valid output is returned. Otherwise the unknown history is zero padded.
    """
    if x.ndim != 3 or x.shape[1] != 1 or taps.ndim != 1:
        raise ValueError("causal_fir expects Bx1xT and K taps")
    if not extended:
        x = F.pad(x, (taps.numel() - 1, 0))
    return F.conv1d(x, taps.to(device=x.device, dtype=x.dtype).flip(0)[None, None])


class CausalFilter(nn.Module):
    """65/17 causal taps, identity initialized, safe nonzero DC or RMS norm."""

    def __init__(self, length: int, normalization: str):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(length))
        impulse = torch.zeros(length)
        impulse[0] = 1.0
        self.register_buffer("impulse", impulse)
        self.normalization = normalization

    def taps(self) -> Tensor:
        # Total absolute perturbation < .5, so DC and RMS denominators cannot
        # vanish. Signed tails are allowed; this is an effective correction FIR.
        h = self.impulse + 0.5 * torch.tanh(self.raw) / self.raw.numel()
        if self.normalization == "dc":
            return h / h.sum()
        if self.normalization == "rms":
            return h / torch.linalg.vector_norm(h)
        raise ValueError(self.normalization)

    def forward(self, x: Tensor, *, extended: bool = False) -> Tensor:
        return causal_fir(x, self.taps(), extended=extended)


def _smooth_step(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def support_gate(t: Tensor) -> Tensor:
    """C2 compact support: onset 100..115, tail taper 456..479.

    Added response terms have zero value AND two zero derivatives at their
    support boundary. This is not a discontinuous zeroing after generation.
    The original source branch remains available on its own natural support.
    """
    return _smooth_step((t - 100.0) / 15.0) * _smooth_step((479.0 - t) / 23.0)


def cubic_expansion(length: int = LENGTH, controls: int = 60) -> Tensor:
    """Fixed C2 uniform cubic B-spline matrix with replicated end controls."""
    position = torch.linspace(0.0, float(controls - 1), length, dtype=torch.float64)
    integer = position.floor().long()
    f = position - integer
    weights = torch.stack(((1 - f)**3 / 6,
                           (3 * f**3 - 6 * f**2 + 4) / 6,
                           (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6,
                           f**3 / 6), -1)
    indices = (integer[:, None] + torch.arange(-1, 3)[None]).clamp(0, controls - 1)
    matrix = torch.zeros(length, controls, dtype=torch.float64)
    matrix.scatter_add_(1, indices, weights)
    return matrix


def relative_time_warp(signal: Tensor, scale: Tensor, anchor: float = TIME_ANCHOR) -> Tensor:
    """Differentiable linear sampling about a fixed anchor, without time shift.

    The cropped right edge is continued by a 64-sample cubic Hermite segment
    matching its value and first difference, then going smoothly to zero. This
    avoids an artificial jump to zero when a contracted event queries beyond
    sample 479. This continuation is a numerical boundary binding, not data.
    """
    signal = signal.to(torch.float64)
    batch = signal.shape[0]
    tau = torch.arange(1, 65, device=signal.device, dtype=signal.dtype) / 64.0
    final, slope = signal[..., -1:], signal[..., -1:] - signal[..., -2:-1]
    continuation = ((2 * tau**3 - 3 * tau**2 + 1) * final
                    + (tau**3 - 2 * tau**2 + tau) * 64.0 * slope)
    extended = torch.cat((signal, continuation), -1)
    t = torch.arange(LENGTH, device=signal.device, dtype=signal.dtype)
    position = anchor + (t[None] - anchor) / scale[:, None].to(signal.dtype)
    left = position.floor().long()
    frac = position - left
    first = extended[:, 0].gather(1, left.clamp(0, extended.shape[-1] - 1))
    second = extended[:, 0].gather(1, (left + 1).clamp(0, extended.shape[-1] - 1))
    result = (1 - frac) * first + frac * second
    valid = (position >= 0) & (position <= extended.shape[-1] - 1)
    return torch.where(valid, result, result.new_zeros((batch, LENGTH)))[:, None]


class ResidualFiLM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.first = nn.Conv1d(in_channels, out_channels, 5, padding=2 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(128, 2 * out_channels)
        self.second = nn.Conv1d(out_channels, out_channels, 5, padding=2 * dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        # Identity FiLM start; parameters still receive gradients immediately.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        h = F.silu(self.norm1(self.first(x)))
        gamma, beta = self.film(condition).chunk(2, 1)
        h = h * (1.0 + gamma[..., None]) + beta[..., None]
        h = F.silu(self.norm2(self.second(h)))
        return h + self.skip(x)


class LowpassDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", torch.tensor([1., 4., 6., 4., 1.]) / 16)

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.to(x.dtype)[None, None].expand(x.shape[1], 1, 5)
        return F.conv1d(F.pad(x, (2, 2), mode="replicate"), kernel,
                        stride=2, groups=x.shape[1])


def deterministic_linear_upsample_2x(x: Tensor) -> Tensor:
    # Exactly the align_corners=False linear interpolation at 2x resolution.
    # Tensor slices and elementwise arithmetic avoid the unsupported CUDA
    # upsample_linear1d backward while keeping the prescribed interpolation.
    if x.ndim != 3 or x.shape[-1] < 1:
        raise ValueError("2x linear interpolation requires nonempty BxCxT")
    previous = torch.cat((x[..., :1], x[..., :-1]), dim=-1)
    following = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
    even = 0.75 * x + 0.25 * previous
    odd = 0.75 * x + 0.25 * following
    return torch.stack((even, odd), dim=-1).flatten(-2)


def deterministic_control_average(x: Tensor) -> Tensor:
    # The exact non-overlapping 480->60 adaptive average; mean backward is
    # deterministic and does not invoke adaptive_avg_pool2d CUDA backward.
    if x.ndim != 3 or x.shape[-1] != 480:
        raise ValueError("control averaging requires BxCx480")
    return x.reshape(x.shape[0], x.shape[1], 60, 8).mean(-1)


class ResponseUNet(nn.Module):
    widths = (32, 48, 64, 96, 128)

    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.encoder = nn.ModuleList([ResidualFiLM(3 if i == 0 else self.widths[i - 1], width)
                                      for i, width in enumerate(self.widths)])
        self.down = LowpassDown()
        self.bottleneck = nn.ModuleList([ResidualFiLM(128, 128, dilation=d) for d in (1, 2, 4)])
        self.up_convs = nn.ModuleList([nn.Conv1d(self.widths[i + 1], self.widths[i], 5, padding=2)
                                      for i in (3, 2, 1, 0)])
        self.decoder = nn.ModuleList([ResidualFiLM(2 * self.widths[i], self.widths[i])
                                      for i in (3, 2, 1, 0)])

    def forward(self, x: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        condition = self.condition(scalars)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.down(x), condition)
            skips.append(x)
        for block in self.bottleneck:
            x = block(x, condition)
        pooled = x.mean(-1)
        for conv, block, skip in zip(self.up_convs, self.decoder, reversed(skips[:-1])):
            if skip.shape[-1] != 2 * x.shape[-1]:
                raise ValueError("N01 decoder requires exact 2x resolution at every level")
            x = conv(deterministic_linear_upsample_2x(x))
            x = block(torch.cat((x, skip), 1), condition)
        return x, pooled


class ShapeEncoder(nn.Module):
    """Small separate amplitude-free encoder; no Q or A route exists here."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.GroupNorm(8, 16), nn.SiLU(),
                                 nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.AdaptiveAvgPool1d(1))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


class MonotoneCharge(nn.Module):
    """8 fixed-width monotone rational quadratic bins on log source charge.

    Positive heights and endpoint derivatives guarantee monotonicity while
    fixed shape features determine its 18 parameters. Outside TRAIN's knot
    range, endpoint-slope linear continuation remains positive and monotone.
    """

    def __init__(self, log_min: float, log_max: float):
        super().__init__()
        if not math.isfinite(log_min) or not math.isfinite(log_max) or log_max <= log_min:
            raise ValueError("log_q_min/log_q_max must be finite and strictly ordered")
        self.register_buffer("log_min", torch.tensor(log_min, dtype=torch.float64))
        self.register_buffer("log_max", torch.tensor(log_max, dtype=torch.float64))
        self.parameters_from_shape = nn.Linear(32, 18)
        nn.init.zeros_(self.parameters_from_shape.weight)
        nn.init.zeros_(self.parameters_from_shape.bias)

    def forward(self, log_q: Tensor, shape: Tensor) -> Tensor:
        raw = self.parameters_from_shape(shape).to(torch.float64)
        width = (self.log_max - self.log_min) / 8.0
        heights = width * torch.exp(2.0 * torch.tanh(raw[:, :8]))
        derivatives = torch.exp(2.0 * torch.tanh(raw[:, 8:17]))
        start = self.log_min + 2.0 * torch.tanh(raw[:, 17])
        knots = torch.cat((start[:, None], start[:, None] + heights.cumsum(1)), 1)
        index = ((log_q - self.log_min) / width).floor().long().clamp(0, 7)
        gather = lambda a, offset=0: a.gather(1, (index + offset)[:, None]).squeeze(1)
        h, y0 = gather(heights), gather(knots)
        d0, d1 = gather(derivatives), gather(derivatives, 1)
        raw_theta = (log_q - self.log_min - index * width) / width
        # A last-knot round-off can put theta just above 1. Keep the canonical
        # forward bin coordinate but its correct one-sided derivative. Outside
        # the knot interval the explicit linear continuation below is used.
        theta = raw_theta + (raw_theta.clamp(0, 1) - raw_theta).detach()
        slope = h / width
        product = theta * (1 - theta)
        numerator = h * (slope * theta**2 + d0 * product)
        denominator = slope + (d0 + d1 - 2 * slope) * product
        y = y0 + numerator / denominator
        y = torch.where(log_q < self.log_min, knots[:, 0] + derivatives[:, 0] * (log_q - self.log_min), y)
        y = torch.where(log_q > self.log_max, knots[:, -1] + derivatives[:, -1] * (log_q - self.log_max), y)
        return torch.exp(y)


class PhysicalHead(nn.Module):
    """55 Cholesky parameters (or the simple comparator's first 3)."""

    def __init__(self, dimensions: int):
        super().__init__()
        self.dimensions = dimensions
        count = dimensions * (dimensions + 1) // 2
        self.base = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, count))
        self.refinement = (nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, count))
                           if dimensions == 10 else None)
        rows, cols = torch.tril_indices(dimensions, dimensions)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        limits = torch.tensor([0.30, 0.20] + [0.15] * (dimensions - 2))
        initial_std = torch.tensor([0.04, 0.03] + [0.002] * (dimensions - 2))
        self.register_buffer("limits", limits)
        initial = torch.zeros(count)
        diagonal = rows == cols
        initial[diagonal] = torch.logit(initial_std / limits)
        nn.init.zeros_(self.base[-1].weight)
        with torch.no_grad():
            self.base[-1].bias.copy_(initial)
        if self.refinement is not None:
            nn.init.normal_(self.refinement[-1].weight, std=1e-3)
            nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, scalars: Tensor, pooled: Tensor | None = None, *, rank: int = 10) -> Tensor:
        raw = self.base(scalars)
        if pooled is not None and self.refinement is not None:
            raw = raw + self.refinement(pooled)
        diagonal = self.rows == self.cols
        row_limit = self.limits[self.rows]
        entries = torch.where(diagonal, row_limit * torch.sigmoid(raw),
                              0.5 * row_limit * torch.tanh(raw) / torch.sqrt(self.rows.clamp_min(1)))
        matrix = raw.new_zeros((raw.shape[0], 10, 10))
        matrix[:, self.rows, self.cols] = entries
        if rank < self.dimensions:
            active = (torch.arange(10, device=raw.device) < rank).to(raw.dtype)
            matrix = matrix * active[None, :, None] * active[None, None, :]
        return matrix


class ElectronicsNoise(nn.Module):
    """Input-independent electronics, including its nonzero window charge."""

    def __init__(self, stats: Mapping[str, float]):
        super().__init__()
        self.filter = CausalFilter(65, "rms")
        self.log_rms = nn.Parameter(torch.tensor(math.log(max(float(stats["electronics_rms"]), 1e-12))))
        spread = min(max(float(stats["electronics_log_rms_std"]), 1e-4), 0.7999)
        self.raw_spread = nn.Parameter(torch.tensor(math.log(spread / (0.8 - spread))))
        self.offset_mean = nn.Parameter(torch.tensor(float(stats["electronics_offset_mean"])))
        self.log_offset_std = nn.Parameter(torch.tensor(math.log(max(float(stats["electronics_offset_std"]), 1e-12))))
        self.slope_mean = nn.Parameter(torch.tensor(0.0))
        self.log_slope_std = nn.Parameter(torch.tensor(math.log(max(float(stats["electronics_slope_std"]), 1e-12))))
        self.register_buffer("t", torch.linspace(-1.0, 1.0, LENGTH, dtype=torch.float64)[None, None])

    def forward(self, noise: Mapping[str, Tensor]) -> Tensor:
        white = noise["electronics_white"].to(torch.float64)
        if white.ndim != 3 or white.shape[1:] != (1, 544):
            raise ValueError("electronics_white must be Bx1x544 (64 history + 480 valid samples)")
        spread = 0.8 * torch.sigmoid(self.raw_spread).to(torch.float64)
        sigma = torch.exp(self.log_rms.to(torch.float64)
                          + spread * noise["electronics_scale"].to(torch.float64) - 0.5 * spread**2)
        offset = self.offset_mean.to(torch.float64) + self.log_offset_std.exp().to(torch.float64) * noise["electronics_offset"].to(torch.float64)
        slope = self.slope_mean.to(torch.float64) + self.log_slope_std.exp().to(torch.float64) * noise["electronics_slope"].to(torch.float64)
        return offset + slope * self.t + sigma * self.filter(white, extended=True)


class N01Response(nn.Module):
    """Public proposed N01 response with explicit, reusable latent draws.

    ``forward`` accepts Bx480 or Bx1x480. Returned waveform/pre_adc and scalar
    per-event diagnostics use Bx480 and B, respectively. ``basis`` is Bx8x480.
    ``electronics(noise)`` can be fitted separately during the baseline stage.
    """

    noise_shapes = {"phys": (10,), "electronics_white": (1, 544),
                    "electronics_scale": (1, 1), "electronics_offset": (1, 1),
                    "electronics_slope": (1, 1), "local_white": (1, 496)}
    stages = ("baseline", "mean_response", "joint_stochastic", "low_learning_rate")

    def __init__(self, stats: dict | None = None, variant: str = "full"):
        super().__init__()
        if variant not in ("full", "deterministic", "simple"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        defaults = {"q_ref": 5.0, "a_ref": 0.1, "log_q_min": math.log(0.05),
                    "log_q_max": math.log(100.0), "detector_condition": 0.0,
                    "electronics_rms": 1e-3, "electronics_offset_mean": 0.0,
                    "electronics_offset_std": 2e-4, "electronics_slope_std": 1e-4,
                    "electronics_log_rms_std": 0.15}
        values = defaults | ({} if stats is None else stats)
        for key in defaults:
            if not math.isfinite(float(values[key])):
                raise ValueError(f"nonfinite statistic {key}")
        for key in ("q_ref", "a_ref", "electronics_rms"):
            if float(values[key]) <= 0:
                raise ValueError(f"statistic {key} must be positive")
        for key in ("electronics_offset_std", "electronics_slope_std", "electronics_log_rms_std"):
            if float(values[key]) < 0:
                raise ValueError(f"statistic {key} must be nonnegative")
        self.stats = {key: float(values[key]) for key in defaults}
        for key in ("q_ref", "a_ref", "detector_condition"):
            self.register_buffer(key, torch.tensor(self.stats[key], dtype=torch.float64))
        t = torch.arange(LENGTH, dtype=torch.float64)
        self.register_buffer("t_norm", torch.linspace(-1., 1., LENGTH)[None, None])
        self.register_buffer("gate", support_gate(t)[None, None])
        projection = support_gate(t) * torch.exp(-0.5 * ((t - 180.0) / 65.0)**2)
        self.register_buffer("projection", (projection / charge(projection))[None, None])
        self.register_buffer("expansion", cubic_expansion())
        self.shape_encoder = ShapeEncoder()
        self.charge_head = MonotoneCharge(self.stats["log_q_min"], self.stats["log_q_max"])
        self.response_filter = CausalFilter(65, "dc")
        self.mean_time = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.mean_time[-1].weight)
        nn.init.zeros_(self.mean_time[-1].bias)
        self.electronics = ElectronicsNoise(self.stats)
        self.unet = ResponseUNet() if variant != "simple" else None
        self.mean_head = nn.Conv1d(32, 1, 5, padding=2) if self.unet is not None else None
        if self.mean_head is not None:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        self.physical_head = PhysicalHead(10 if variant == "full" else 2) if variant != "deterministic" else None
        self.basis_head = nn.Conv1d(32, 8, 5, padding=2) if variant == "full" else None
        self.envelope_head = nn.Conv1d(32, 1, 5, padding=2) if variant == "full" else None
        self.local_filter = CausalFilter(17, "rms") if variant == "full" else None
        if self.basis_head is not None:
            nn.init.normal_(self.basis_head.weight, std=0.03)
            nn.init.zeros_(self.basis_head.bias)
            nn.init.normal_(self.envelope_head.weight, std=1e-3)
            nn.init.constant_(self.envelope_head.bias, -5.0)
        self.set_stage("joint_stochastic")

    def set_stage(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(f"unknown stage {stage!r}; expected {self.stages}")
        self.stage = stage
        joint = stage in ("joint_stochastic", "low_learning_rate")
        for p in self.parameters():
            p.requires_grad_(False)
        always = (self.shape_encoder, self.charge_head, self.response_filter, self.mean_time)
        active = list(always)
        if stage == "baseline":
            active.append(self.electronics)
        if stage != "baseline" and self.unet is not None:
            active.extend((self.unet, self.mean_head))
        if self.physical_head is not None and (stage == "baseline" or joint):
            active.append(self.physical_head.base)
        if self.variant == "full" and joint:
            active.extend((self.physical_head.refinement, self.basis_head,
                           self.envelope_head, self.local_filter))
        for module in active:
            for p in module.parameters():
                p.requires_grad_(True)

    def sample_noise(self, batch: int, device=None, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if batch < 1:
            raise ValueError("batch must be positive")
        if device is None:
            device = self.q_ref.device
        return {key: torch.randn((batch,) + shape, device=device, dtype=torch.float32, generator=generator)
                for key, shape in self.noise_shapes.items()}

    def load_shared_baseline(self, other: "N01Response | Mapping[str, Tensor]") -> None:
        """Copy the deliberately shared baseline into any of the three arms.

        Includes the trained low-dimensional Q/time covariance for stochastic
        arms. U-Net/mean/basis/local heads are not baseline-trained and are not
        copied here; initialize full/deterministic U-Nets with the same seed or
        separately copy their matching keys. The source state is not modified.
        """
        state = other.state_dict() if isinstance(other, N01Response) else other
        shared = ("shape_encoder", "charge_head", "response_filter", "mean_time", "electronics")
        for name in shared:
            prefix = name + "."
            subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not subset:
                raise ValueError(f"shared baseline is missing {name}")
            getattr(self, name).load_state_dict(subset, strict=True)
        for name in ("q_ref", "a_ref", "detector_condition"):
            if name not in state or not torch.equal(state[name].to(getattr(self, name)), getattr(self, name)):
                raise ValueError(f"shared baseline statistic mismatch: {name}")
        if self.physical_head is not None:
            prefix = "physical_head.base."
            subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not subset:
                raise ValueError("stochastic shared baseline is missing its Q/time covariance head")
            target = self.physical_head.base.state_dict()
            for name in ("0.weight", "0.bias"):
                if subset[name].shape != target[name].shape:
                    raise ValueError(f"shared covariance mismatch: {name}")
                target[name] = subset[name]
            # Row-major triangular order begins [L00,L10,L11] in both heads.
            for name in ("2.weight", "2.bias"):
                count = min(target[name].shape[0], subset[name].shape[0])
                copied = target[name].clone()
                copied[:count] = subset[name][:count]
                target[name] = copied
            self.physical_head.base.load_state_dict(target)

    def _noise(self, noise: Mapping[str, Tensor], batch: int, device) -> dict[str, Tensor]:
        if set(noise) != set(self.noise_shapes):
            raise ValueError(f"explicit noise keys must be {tuple(self.noise_shapes)}")
        result = {}
        for key, shape in self.noise_shapes.items():
            value = noise[key]
            if not isinstance(value, Tensor) or value.shape != (batch,) + shape:
                raise ValueError(f"noise[{key!r}] must have shape {(batch,) + shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"noise[{key!r}] contains nonfinite values")
            result[key] = value.to(device=device, dtype=torch.float32)
        return result

    def _features(self, unit_peak_shape: Tensor) -> Tensor:
        dtype = self.shape_encoder.net[0].weight.dtype
        u = unit_peak_shape.to(dtype)
        derivative = F.pad(u[..., 1:] - u[..., :-1], (1, 0))
        return torch.cat((u, derivative, self.t_norm.to(dtype).expand(u.shape[0], -1, -1)), 1)

    def mean_charge(self, log_q: Tensor, unit_peak_shape: Tensor) -> Tensor:
        """Fixed-shape monotonic response helper, also useful for diagnostics."""
        if unit_peak_shape.ndim == 2:
            unit_peak_shape = unit_peak_shape[:, None]
        return self.charge_head(log_q.to(torch.float64), self.shape_encoder(self._features(unit_peak_shape)))

    def forward(self, x: Tensor, noise: Mapping[str, Tensor] | None = None,
                generator: torch.Generator | None = None, adc_ste: bool = True) -> dict[str, Tensor]:
        if x.ndim == 2:
            x = x[:, None]
        if x.ndim != 3 or x.shape[1:] != (1, LENGTH) or x.shape[0] < 1:
            raise ValueError("x must have shape Bx480 or Bx1x480 with B > 0")
        if not x.is_floating_point():
            raise TypeError("x must be a floating point voltage waveform")
        nonfinite = ~torch.isfinite(x).flatten(1).all(1)
        if bool(nonfinite.any()):
            raise InvalidWaveformError("nonfinite source voltage", torch.nonzero(nonfinite).flatten())
        x64 = x.to(torch.float64)
        baseline = x64[..., :100].mean(-1, keepdim=True)
        centered = x64 - baseline
        amplitude = centered.amax(-1).squeeze(1)
        q_source = charge(x64).squeeze(1)
        _invalid(amplitude, AMPLITUDE_FLOOR, "invalid source amplitude")
        _invalid(q_source, SOURCE_Q_FLOOR, "invalid source window charge")
        unit_peak = centered / amplitude[:, None, None]
        features = self._features(unit_peak)
        log_q = q_source.log()
        qbar = self.charge_head(log_q, self.shape_encoder(features))
        _invalid(qbar, 0.0, "invalid mean signal charge")
        scalars = torch.stack(((q_source / self.q_ref).log(),
                               (amplitude / self.a_ref).log(),
                               self.detector_condition.expand(x.shape[0])), 1).to(features.dtype)
        noise = self._noise(self.sample_noise(x.shape[0], x.device, generator) if noise is None else noise,
                            x.shape[0], x.device)
        zero = x64.new_zeros((x.shape[0], 1, LENGTH))
        mean_residual, basis, envelope = zero, zero.expand(-1, 8, -1), zero
        decoded = pooled = None
        if self.unet is not None and self.stage != "baseline":
            decoded, pooled = self.unet(features, scalars)
            mean_residual = 0.20 * torch.tanh(self.mean_head(decoded)).to(torch.float64) * self.gate
        joint = self.stage in ("joint_stochastic", "low_learning_rate")
        physiology_on = self.variant != "deterministic" and self.stage != "mean_response"
        if physiology_on:
            full_rank = self.variant == "full" and joint
            lower = self.physical_head(scalars, pooled if full_rank else None, rank=10 if full_rank else 2).to(torch.float64)
        else:
            lower = x64.new_zeros((x.shape[0], 10, 10))
        eta = (lower @ noise["phys"].to(torch.float64)[..., None]).squeeze(-1)
        sigma_q2 = lower[:, 0].square().sum(1)
        qsig = qbar * torch.exp(eta[:, 0] - 0.5 * sigma_q2)
        _invalid(qsig, 0.0, "nonfinite or nonpositive stochastic signal charge")
        mean_log_time = math.log(2.0) * torch.tanh(self.mean_time(scalars).squeeze(1)).to(torch.float64)
        time_scale = torch.exp(mean_log_time + eta[:, 1] - 0.5 * lower[:, 1].square().sum(1))
        _invalid(time_scale, 0.0, "invalid relative time scale")
        if self.variant == "full" and joint:
            controls = deterministic_control_average(self.basis_head(decoded)).to(torch.float64)
            raw_basis = torch.einsum("bkc,tc->bkt", controls, self.expansion) * self.gate
            basis_rms = raw_basis.square().mean(-1, keepdim=True).sqrt()
            if bool((basis_rms <= 1e-10).any()):
                rows = torch.nonzero((basis_rms <= 1e-10).any(1).squeeze(-1)).flatten()
                raise InvalidWaveformError("degenerate stochastic shape basis", rows)
            basis = raw_basis / basis_rms
        raw_shape = relative_time_warp(self.response_filter(unit_peak), time_scale)
        raw_shape = raw_shape + mean_residual + (eta[:, 2:, None] * basis).sum(1, keepdim=True)
        q_shape = charge(raw_shape).squeeze(1)
        _invalid(q_shape, SHAPE_Q_FLOOR, "invalid generated shape window charge Q(v)")
        unit_shape = raw_shape / q_shape[:, None, None]
        signal_noise = zero
        if self.variant == "full" and joint:
            env_controls = deterministic_control_average(self.envelope_head(decoded)).to(torch.float64)
            envelope = F.softplus(torch.einsum("bkc,tc->bkt", env_controls, self.expansion))
            envelope = envelope * self.gate * self.a_ref * (qsig / self.q_ref)[:, None, None]
            extended_envelope = F.pad(envelope, (16, 0))
            raw_local = self.local_filter(extended_envelope * noise["local_white"].to(torch.float64), extended=True) * self.gate
            signal_noise = raw_local - charge(raw_local)[..., None] * self.projection
        electronics_noise = self.electronics(noise)
        pre_adc = qsig[:, None, None] * unit_shape + signal_noise + electronics_noise
        nonfinite_output = ~torch.isfinite(pre_adc).flatten(1).all(1)
        if bool(nonfinite_output.any()):
            raise InvalidWaveformError("nonfinite pre-ADC assembly", torch.nonzero(nonfinite_output).flatten())
        waveform = digitize(pre_adc, ste=adc_ste)
        mean_penalty = mean_residual.square().mean() + (mean_residual[..., 1:] - mean_residual[..., :-1]).square().mean()
        regular = mean_penalty
        if self.variant == "full" and joint:
            gram = (basis @ basis.transpose(1, 2)) / LENGTH
            off_diagonal = gram - torch.diag_embed(gram.diagonal(dim1=1, dim2=2))
            regular = regular + 0.01 * off_diagonal.square().mean()
            regular = regular + (basis[..., 2:] - 2 * basis[..., 1:-1] + basis[..., :-2]).square().mean()
            regular = regular + envelope.square().mean() / self.a_ref.square()
        adc_excess = (F.relu(pre_adc - ADC_MAX) + F.relu(ADC_MIN - pre_adc)) / (ADC_MAX - ADC_MIN)
        regular = regular + adc_excess.square().mean()
        # The architecture enforces the specified fixed-shape monotonic trend.
        # Returning a scalar zero keeps the trainer API without a redundant loss.
        trend = qbar.sum() * 0.0
        return {"waveform": waveform.squeeze(1), "pre_adc": pre_adc.squeeze(1),
                "qbar": qbar, "qsig": qsig, "q_source": q_source,
                "unit_shape": unit_shape.squeeze(1), "signal_noise": signal_noise.squeeze(1),
                "electronics_noise": electronics_noise.squeeze(1), "time_scale": time_scale,
                "basis": basis, "mean_residual": mean_residual.squeeze(1),
                "signal_envelope": envelope.squeeze(1), "regular": regular, "trend": trend,
                "saturation_fraction": ((pre_adc < ADC_MIN) | (pre_adc > ADC_MAX)).to(torch.float64).mean(),
                "saturation_fraction_per_event": ((pre_adc < ADC_MIN) | (pre_adc > ADC_MAX)).to(torch.float64).mean((1, 2)),
                "physical_cholesky": lower, "physical_eta": eta, "sigma_q2": sigma_q2,
                "shape_charge": q_shape, "pre_adc_charge": charge(pre_adc).squeeze(1),
                "electronics_charge": charge(electronics_noise).squeeze(1),
                "charge_identity_residual": charge(pre_adc).squeeze(1) - qsig - charge(electronics_noise).squeeze(1),
                "adc_charge_change": charge(waveform).squeeze(1) - charge(pre_adc).squeeze(1)}
