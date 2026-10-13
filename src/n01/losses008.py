"""N01 window008 objectives: baseline-subtracted 256-point pulses. Statistics are fitted to legal TRAIN only.

These losses compare empirical populations, never arbitrary SOURCE/REAL pairs.
The comparison-weight model is independent of the waveform response model.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


ADC_STEP = 1.0 / (2**14 - 1)
WAVEFORM_LENGTH = 256
MIN_CONDITIONAL_SUPPORT = 16


def _waveform(x: Tensor, length: int = WAVEFORM_LENGTH) -> Tensor:
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 2 or x.shape[1] != length or x.shape[0] == 0:
        raise ValueError(f"Expected nonempty (N,{length}) or (N,1,{length}), got {tuple(x.shape)}")
    return x


def window_charge(x: Tensor) -> Tensor:
    """Baseline-subtracted Q in V ns; 1 ns spacing, critical sum in float64."""
    x = _waveform(x).to(torch.float64)
    return x.sum(1)  # Already baseline-subtracted by the common carrier observation.


def _peak(x: Tensor) -> Tensor:
    x = _waveform(x)
    return x.amax(1)


def _extended_log_charge(q: Tensor, q_reference: Tensor, q_floor: Tensor) -> Tensor:
    """True log above the floor; monotone C1 asinh continuation below it."""
    above = (q.clamp_min(q_floor) / q_reference).log()
    below = (q_floor / q_reference).log() + torch.asinh((q - q_floor) / q_floor)
    return torch.where(q >= q_floor, above, below)


def _qpsd(x: Tensor, q_reference: Tensor, q_floor: Tensor) -> Tensor:
    x = _waveform(x).to(torch.float64)
    q = window_charge(x)
    tail = x[:, 46:256].sum(1)  # Software tail starts 30 ns after the located peak.
    # Retain every row. A guarded ratio is a numerical feature, not a claim that
    # PSD is physically meaningful for a zero/near-zero charge event.
    sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
    denominator = sign * q.abs().clamp_min(q_floor)
    return torch.stack((_extended_log_charge(q, q_reference, q_floor), tail / denominator), 1)


def _wave_views(x: Tensor, peak_floor: Tensor) -> dict[str, Tensor]:
    x = _waveform(x)
    peak = x.amax(1, keepdim=True).clamp_min(peak_floor)
    shape = x / peak
    views = {"voltage": x, "shape": shape,
             "difference": torch.diff(shape, dim=1),
             "curvature": torch.diff(shape, n=2, dim=1)}
    for size in (5, 17):
        smooth = F.avg_pool1d(F.pad(shape[:, None], (size // 2, size // 2), mode="replicate"), size, stride=1)[:, 0]
        views[f"difference_smooth_{size}"] = torch.diff(smooth, dim=1)
    return views


def _noise_features(x: Tensor) -> Tensor:
    """32 explicit raw ADC baseline features. Effective 256-point pulses are not baselines."""
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 100:
        raise ValueError("Noise features require the explicit uncentered ADC baseline (N,100)")
    b = x[:, :100].to(torch.float64)
    mean = b.mean(1, keepdim=True)
    centered = b - mean
    t = torch.linspace(-1, 1, 100, device=b.device, dtype=b.dtype)
    slope = (centered * t).mean(1, keepdim=True) / t.square().mean()
    variance_floor = ADC_STEP**2 / 12.0
    variance = centered.square().mean(1, keepdim=True).clamp_min(variance_floor)
    log_rms = 0.5 * variance.log()
    acf = torch.cat([(centered[:, :-lag] * centered[:, lag:]).mean(1, keepdim=True) / variance
                     for lag in range(1, 17)], 1)
    amplitudes = torch.quantile(centered, torch.tensor([0.01, 0.50, 0.99], device=b.device, dtype=b.dtype), dim=1).T
    power = torch.fft.rfft(centered, dim=1).abs().square()
    power_total = power[:, 1:].sum(1, keepdim=True).clamp_min(100.0 * variance_floor)
    bands = torch.cat([power[:, lo:hi].sum(1, keepdim=True) / power_total
                       for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 21), (21, 51))], 1)
    local = centered.reshape(-1, 5, 20)
    local_log_variance = (local - local.mean(2, keepdim=True)).square().mean(2).clamp_min(variance_floor).log()
    return torch.cat((mean, slope, log_rms, acf, amplitudes, bands, local_log_variance), 1)


def _robust_statistics(x: Tensor, floor: float | Tensor) -> dict[str, Tensor]:
    x = x.to(torch.float64)
    quantiles = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], dtype=x.dtype, device=x.device), dim=0)
    # IQR/1.349 estimates one Gaussian standard deviation, without requiring a
    # Gaussian population. Floors are numerical resolutions, not fitted wins.
    scale = ((quantiles[2] - quantiles[0]) / 1.349).clamp_min(torch.as_tensor(floor, device=x.device, dtype=x.dtype))
    return {"center": quantiles[1].float().cpu(), "scale": scale.float().cpu()}


@torch.no_grad()
def fit_loss_statistics(real_train: Tensor, source_train: Tensor, real_baseline100: Tensor) -> dict[str, Any]:
    """Serializable TRAIN statistics, with full row counts and validity metadata.

    Nonfinite rows and nonpositive TRAIN peaks fail explicitly. SOURCE must have
    positive usable Q. Small nonpositive REAL Q is retained with recorded guarded
    PSD/extended-log features. No masks remove low-energy or tail populations.
    """
    real = _waveform(real_train).detach().cpu().double()
    source = _waveform(source_train).detach().cpu().double()
    validity: dict[str, Any] = {}
    for name, x in (("real", real), ("source", source)):
        invalid = ~torch.isfinite(x).all(1)
        validity[f"{name}_nonfinite_rows"] = int(invalid.sum())
        if invalid.any():
            raise ValueError(f"{name} TRAIN contains {int(invalid.sum())} nonfinite rows; no rows were dropped")
    qr, qs = window_charge(real), window_charge(source)
    positive_real_q = qr[qr > 0]
    if positive_real_q.numel() == 0:
        raise ValueError("REAL TRAIN has no positive Q; logarithmic pulse population cannot be initialized")
    q_reference = positive_real_q.median()
    # The separate baseline100 is subtracted from all 256 pulse samples.
    # Quantization resolution uses 256 + 256**2/100; it is not a row cut.
    q_floor = torch.maximum(q_reference * 1e-6, torch.tensor(ADC_STEP * ((256 + 256**2 / 100) / 12)**0.5, dtype=torch.float64))
    peak_floor = torch.tensor(ADC_STEP, dtype=torch.float64)
    ar, ass = _peak(real).double(), _peak(source).double()
    validity.update({"real_rows": len(real), "source_rows": len(source),
                     "real_nonpositive_q_rows": int((qr <= 0).sum()),
                     "real_near_zero_q_rows": int((qr.abs() < q_floor).sum()),
                     "real_exact_zero_q_rows": int((qr == 0).sum()),
                     "source_q_below_resolution_rows": int((qs <= q_floor).sum()),
                     "real_nonpositive_peak_rows": int((ar <= 0).sum()),
                     "source_nonpositive_peak_rows": int((ass <= 0).sum()),
                     "real_q_min": float(qr.min()), "real_q_max": float(qr.max()),
                     "source_q_min": float(qs.min()), "source_q_max": float(qs.max()),
                     "row_selection": "none", "q_floor_is_selection_threshold": False})
    if validity["source_q_below_resolution_rows"] or validity["real_nonpositive_peak_rows"] or validity["source_nonpositive_peak_rows"]:
        raise ValueError(f"Unusable TRAIN pulse statistics, no rows dropped: {validity}")
    qpsd = _qpsd(real, q_reference, q_floor)
    boundaries = torch.quantile(qpsd[:, 0], torch.arange(1, 8, dtype=torch.float64) / 8)
    memberships = torch.bucketize(qpsd[:, 0].contiguous(), boundaries)
    validity["real_charge_bin_counts"] = torch.bincount(memberships, minlength=8).tolist()
    validity["charge_bin_coordinate"] = "log(Q/Qref) with C1 asinh continuation at quantization resolution"
    validity["charge_bin_outer_edges"] = ["-infinity", "+infinity"]
    noise = _noise_features(real_baseline100.detach().cpu().double())
    noise_floor = torch.tensor([ADC_STEP] * 2 + [0.02] + [0.02] * 16 + [ADC_STEP] * 3 + [0.01] * 5 + [0.04] * 5)
    views: dict[str, dict[str, Tensor]] = {}
    for name, view in _wave_views(real, peak_floor).items():
        floor = ADC_STEP if name == "voltage" else 1e-3
        views[name] = _robust_statistics(view, floor)
    return {"schema_version": 8, "window_schema": "n01/window008/v1", "fit_split": "TRAIN", "q_reference": q_reference.cpu(),
            "q_floor": q_floor.cpu(), "peak_floor": peak_floor.cpu(),
            "source_q_reference": qs.median().cpu(), "source_peak_reference": ass.median().cpu(),
            "real_q_reference": q_reference.cpu(), "charge_bin_boundaries": boundaries.cpu(),
            "minimum_conditional_support": MIN_CONDITIONAL_SUPPORT, "views": views,
            "qpsd": _robust_statistics(qpsd, torch.tensor([0.02, 0.01])),
            "noise": _robust_statistics(noise, noise_floor), "validity": validity}


def _probabilities(weights: Tensor | None, count: int, like: Tensor) -> Tensor:
    if weights is None:
        return torch.full((count,), 1.0 / count, device=like.device, dtype=like.dtype)
    weights = weights.reshape(-1).to(device=like.device, dtype=like.dtype)
    if len(weights) != count:
        raise ValueError(f"Weight count {len(weights)} does not match population {count}")
    if not torch.isfinite(weights).all() or (weights < 0).any() or not weights.sum() > 0:
        raise ValueError("Comparison weights must be finite, nonnegative, and have positive total mass")
    return weights / weights.sum()


def weighted_wasserstein_1d(x: Tensor, y: Tensor, weights: Tensor | None = None,
                            real_weights: Tensor | None = None) -> Tensor:
    """Exact empirical W1, averaged over coordinates, with unequal sample counts.

    Integrates |Fx-Fy| on the merged sorted support. Sample positions and weights
    have piecewise gradients; search/sort membership is discrete as expected for
    empirical optimal transport. No quantile grid approximation or SciPy is used.
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) == 0:
        raise ValueError("W1 requires nonempty (N,D)/(M,D) with identical D")
    y = y.to(device=x.device, dtype=x.dtype)
    px, py = _probabilities(weights, len(x), x), _probabilities(real_weights, len(y), x)
    xs, ix = x.sort(dim=0)
    ys, iy = y.sort(dim=0)
    cdfx = torch.cat((x.new_zeros(1, x.shape[1]), px[:, None].expand_as(x).gather(0, ix).cumsum(0)), 0)
    cdfy = torch.cat((x.new_zeros(1, x.shape[1]), py[:, None].expand_as(y).gather(0, iy).cumsum(0)), 0)
    points = torch.cat((xs, ys), 0).sort(dim=0).values
    left = points[:-1].T.contiguous()
    nx = torch.searchsorted(xs.T.contiguous(), left, right=True)
    ny = torch.searchsorted(ys.T.contiguous(), left, right=True)
    difference = cdfx.T.gather(1, nx) - cdfy.T.gather(1, ny)
    return (difference.abs().T * torch.diff(points, dim=0)).sum(0).mean()


def _directions(dimension: int, count: int, generator: torch.Generator) -> Tensor:
    directions = torch.randn(dimension, count, generator=generator)
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


class N01Loss(nn.Module):
    """Five non-adversarial components; adversarial loss is supplied separately."""

    def __init__(self, stats: dict[str, Any], seed: int = 20261012):
        super().__init__()
        if stats.get("fit_split") != "TRAIN":
            raise ValueError("N01 losses require explicitly TRAIN-fitted statistics")
        self.metadata = stats["validity"]
        self.seed = int(seed)
        self.minimum_support = int(stats.get("minimum_conditional_support", 16))
        self.last_diagnostics: dict[str, Any] = {}
        for key in ("q_reference", "q_floor", "peak_floor", "charge_bin_boundaries"):
            self.register_buffer(key, torch.as_tensor(stats[key]).clone())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.view_names = tuple(stats["views"])
        for name, value in stats["views"].items():
            self.register_buffer(f"{name}_center", value["center"].clone())
            self.register_buffer(f"{name}_scale", value["scale"].clone())
            self.register_buffer(f"{name}_directions", _directions(len(value["center"]), 32, generator))
        for name in ("qpsd", "noise"):
            self.register_buffer(f"{name}_center", stats[name]["center"].clone())
            self.register_buffer(f"{name}_scale", stats[name]["scale"].clone())
        self.register_buffer("qpsd_directions", _directions(2, 16, generator))
        self.register_buffer("noise_directions", _directions(len(stats["noise"]["center"]), 32, generator))

    def noise_features(self, x: Tensor) -> Tensor:
        return _noise_features(x)

    def _standardize(self, name: str, x: Tensor) -> Tensor:
        return ((x - getattr(self, f"{name}_center").to(x)) / getattr(self, f"{name}_scale").to(x)).float()

    def noise_loss(self, fake: Tensor, real: Tensor, weights: Tensor | None = None) -> Tensor:
        f = self._standardize("noise", self.noise_features(fake))
        r = self._standardize("noise", self.noise_features(real))
        marginal = weighted_wasserstein_1d(f, r, weights)
        projected = weighted_wasserstein_1d(f @ self.noise_directions.to(f), r @ self.noise_directions.to(r), weights)
        return 0.5 * (marginal + projected)

    def _qpsd_loss(self, fake: Tensor, real: Tensor, weights: Tensor | None,
                   mean_response: bool = False) -> Tensor:
        fraw = _qpsd(fake, self.q_reference, self.q_floor)
        rraw = _qpsd(real, self.q_reference, self.q_floor)
        f, r = self._standardize("qpsd", fraw), self._standardize("qpsd", rraw)
        probabilities = _probabilities(weights, len(f), f)
        if mean_response:
            global_term = ((f * probabilities[:, None]).sum(0) - r.mean(0)).abs().mean()
        else:
            marginal = weighted_wasserstein_1d(f, r, weights)
            joint = weighted_wasserstein_1d(f @ self.qpsd_directions.to(f), r @ self.qpsd_directions.to(r), weights)
            global_term = 0.5 * (marginal + joint)
        fb = torch.bucketize(fraw[:, 0].detach().contiguous(), self.charge_bin_boundaries)
        rb = torch.bucketize(rraw[:, 0].detach().contiguous(), self.charge_bin_boundaries)
        conditional, bin_diagnostics = [], []
        for index in range(8):
            fm, rm = fb == index, rb == index
            nf, nr = int(fm.sum()), int(rm.sum())
            supported = nf >= self.minimum_support and nr >= self.minimum_support
            bin_diagnostics.append({"bin": index, "fake_count": nf, "real_count": nr,
                                    "supported": supported,
                                    "status": "used" if supported else "insufficient_current_batch_support"})
            if supported:
                fw = None if weights is None else weights.reshape(-1)[fm]
                if mean_response:
                    fp = _probabilities(fw, nf, f)
                    term = ((f[fm, 1] * fp).sum() - r[rm, 1].mean()).abs()
                else:
                    term = weighted_wasserstein_1d(f[fm, 1], r[rm, 1], fw)
                conditional.append(term)
        fq, rq = window_charge(fake), window_charge(real)
        self.last_diagnostics = {
            "fake_global_rows": len(fake), "real_global_rows": len(real),
            "conditional_bins": bin_diagnostics, "minimum_conditional_support": self.minimum_support,
            "conditional_bins_used": len(conditional), "conditional_bins_total": 8,
            "conditional_coverage_complete": len(conditional) == 8,
            "conditional_rebatch_needed": len(conditional) != 8,
            "fake_nonpositive_q_rows": int((fq <= 0).sum()), "real_nonpositive_q_rows": int((rq <= 0).sum()),
            "fake_guarded_psd_rows": int((fq.abs() < self.q_floor).sum()),
            "real_guarded_psd_rows": int((rq.abs() < self.q_floor).sum()),
            "global_term_includes_all_rows": True,
            "conditional_term": "mean_PSD" if mean_response else "PSD_empirical_W1",
            "no_supported_bin_behavior": "global_term_only_with_incomplete_coverage_diagnostic",
        }
        # Missing conditional terms are not logged as zero distances. A trainer
        # may rebatch current-version forwards; no generation cache exists here.
        return 0.5 * (global_term + torch.stack(conditional).mean()) if conditional else global_term

    def population_objective(self, fake: Tensor, real: Tensor, weights: Tensor) -> Tensor:
        """Phi-only empirical QPSD comparison: response positions are detached."""
        return self._qpsd_loss(_waveform(fake).detach(), _waveform(real).detach(), weights)

    def components(self, outputs: dict[str, Tensor], real: Tensor,
                   comparison_weights: Tensor | None = None,
                   stage: str = "joint_stochastic", *,
                   real_baseline100: Tensor | None = None) -> dict[str, Tensor]:
        fake, real = _waveform(outputs["waveform"]), _waveform(real)
        if stage not in ("baseline", "mean_response", "joint_stochastic", "low_learning_rate"):
            raise ValueError(f"Unknown N01 stage: {stage}")
        # G may not backpropagate into the independent population model.
        weights = None if comparison_weights is None else comparison_weights.detach().reshape(-1).to(fake)
        mean_stage = stage == "mean_response"
        wave_terms = []
        fv, rv = _wave_views(fake, self.peak_floor), _wave_views(real, self.peak_floor)
        for name in self.view_names:
            f, r = self._standardize(name, fv[name]), self._standardize(name, rv[name])
            if mean_stage:
                p = _probabilities(weights, len(f), f)
                term = ((f * p[:, None]).sum(0) - r.mean(0)).abs().mean()
            else:
                direction = getattr(self, f"{name}_directions").to(f)
                term = weighted_wasserstein_1d(f @ direction, r @ direction, weights)
            wave_terms.append(term)
        qpsd = self._qpsd_loss(fake, real, weights, mean_response=mean_stage)
        regular = outputs.get("regular", fake.sum() * 0)
        trend = outputs.get("trend", fake.sum() * 0)
        if regular.numel() != 1 or trend.numel() != 1:
            raise ValueError("Generator regular and trend outputs must be scalar")
        return {"wave": torch.stack(wave_terms).mean(), "QPSD": qpsd,
                "noise": self.noise_loss(outputs["baseline100"], real_baseline100, weights),
                "regular": regular.reshape(()), "trend": trend.reshape(())}

    def forward(self, outputs: dict[str, Tensor], real: Tensor,
                comparison_weights: Tensor | None = None,
                stage: str = "joint_stochastic", *,
                real_baseline100: Tensor | None = None) -> dict[str, Tensor]:
        return self.components(outputs, real, comparison_weights, stage, real_baseline100=real_baseline100)


class WaveformDiscriminator(nn.Module):
    """Small spectral-normalized full-waveform CNN, with global and local heads."""

    def __init__(self, stats: dict[str, Any]):
        super().__init__()
        if stats.get("fit_split") != "TRAIN":
            raise ValueError("Discriminator scales must come from TRAIN")
        self.register_buffer("center", stats["views"]["voltage"]["center"].clone())
        self.register_buffer("scale", stats["views"]["voltage"]["scale"].clone())
        layers = []
        previous = 1
        for channels in (32, 64, 128):
            layers += [spectral_norm(nn.Conv1d(previous, channels, 7, stride=2, padding=3)), nn.LeakyReLU(0.2)]
            previous = channels
        self.body = nn.Sequential(*layers)
        self.global_head = spectral_norm(nn.Linear(128, 1))
        self.local_head = spectral_norm(nn.Conv1d(128, 1, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = _waveform(x)
        x = (x - self.center.to(x)) / self.scale.to(x)
        features = self.body(x[:, None])
        return self.global_head(features.mean(2))[:, 0], self.local_head(features)[:, 0]


def _head_mean(predictions: tuple[Tensor, Tensor], transform, weights: Tensor | None = None) -> Tensor:
    global_pred, local_pred = predictions
    if global_pred.ndim != 1 or local_pred.ndim != 2 or len(global_pred) != len(local_pred):
        raise ValueError("Discriminator predictions must be global (B,) and local (B,K)")
    p = _probabilities(weights, len(global_pred), global_pred)
    return 0.5 * ((transform(global_pred) * p).sum() + (transform(local_pred).mean(1) * p).sum())


def d_hinge(fakepred: tuple[Tensor, Tensor], realpred: tuple[Tensor, Tensor],
            weights: Tensor | None = None) -> Tensor:
    return _head_mean(fakepred, lambda x: F.relu(1 + x), weights) + _head_mean(realpred, lambda x: F.relu(1 - x))


def g_hinge(fakepred: tuple[Tensor, Tensor], weights: Tensor | None = None) -> Tensor:
    return _head_mean(fakepred, lambda x: -x, weights)


class PopulationWeights(nn.Module):
    """Three smooth source-population degrees of freedom; no ID features.

    Normalization uses the full SOURCE TRAIN population, even for a minibatch
    index request. Weights are separate comparison ratios, never physical weights.
    """

    def __init__(self, q_source: Tensor, h: Tensor):
        super().__init__()
        q = torch.as_tensor(q_source).detach().reshape(-1).double()
        h = torch.as_tensor(h, device=q.device).detach().reshape(-1).double()
        if len(q) != len(h) or len(q) == 0 or not torch.isfinite(q).all() or not torch.isfinite(h).all() or (q <= 0).any():
            raise ValueError("Population features require aligned finite source Q>0 and known composition h")
        logq = q.log()
        qstat = _robust_statistics(logq[:, None], 0.05)
        hstat = _robust_statistics(h[:, None], 0.05)
        linear = torch.tanh((logq - qstat["center"].to(q)[0]) / qstat["scale"].to(q)[0])
        quadratic = linear.square()
        composition = torch.tanh((h - hstat["center"].to(h)[0]) / hstat["scale"].to(h)[0])
        features = torch.stack((linear, quadratic - quadratic.mean(), composition - composition.mean()), 1)
        self.register_buffer("features", features.float())
        self.register_buffer("source_logq_center", qstat["center"].to(q.device))
        self.register_buffer("source_logq_scale", qstat["scale"].to(q.device))
        self.register_buffer("composition_center", hstat["center"].to(q.device))
        self.register_buffer("composition_scale", hstat["scale"].to(q.device))
        self.theta = nn.Parameter(torch.zeros(3, device=q.device))

    def forward(self, indices: Tensor | None = None) -> Tensor:
        logweight = 0.5 * torch.tanh(self.features @ self.theta)
        unnormalized = logweight.exp()
        weights = 0.5 + 0.5 * unnormalized / unnormalized.mean()
        return weights if indices is None else weights[indices.to(device=weights.device, dtype=torch.long)]

    def penalty(self) -> Tensor:
        """Unweighted theta L2 + KL-to-uniform; trainer uses coefficient 0.01."""
        weights = self()
        return self.theta.square().mean() + (weights * weights.log()).mean()
