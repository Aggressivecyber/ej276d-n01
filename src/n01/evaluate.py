"""Fixed-checkpoint descriptive N01 endpoint evaluation, with no model selection.

Scientific arrays are opened only inside evaluate_checkpoint. Every generation
batch is <=128. Evaluation seeds/splits/draw counts are fixed across arms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
import torch.nn.functional as F

from .data import charge_np, load_validation, sha256, write_json
from .losses import N01Loss
from .model import N01Response


EVALUATION_SEED = 20261012
GENERATION_BATCH = 128
MARGINAL_DRAWS = 4
REPEAT_PARENTS = 128
REPEAT_DRAWS = 64
CLASSIFIER_UPDATES = 200
CLASSIFIER_SEED = EVALUATION_SEED + 101
BOOTSTRAP_DRAWS = 400
SELECTED_TIME_SAMPLES = np.array([100, 120, 150, 200, 300, 399, 479], dtype=np.int64)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return _clean(value.detach().cpu().numpy())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _probability(weights: np.ndarray | None, count: int) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != count or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid evaluation comparison weights; no rows dropped")
    return weights / weights.sum()


def _summary(x: np.ndarray, weights: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p = _probability(weights, len(x))
    finite = np.isfinite(x)
    if not finite.any():
        return {"rows": len(x), "finite_rows": 0, "invalid_rows": len(x), "mean": None}
    # Used only for inherently undefined time diagnostics. Invalidity and the
    # denominator are always explicit; waveform/Q/global comparisons retain all.
    xx, pp = x[finite], p[finite]
    pp = pp / pp.sum()
    mean = float(pp @ xx)
    variance = float(pp @ (xx - mean)**2)
    order = np.argsort(xx, kind="stable")
    cdf = np.cumsum(pp[order])
    quantiles = xx[order][np.minimum(np.searchsorted(cdf, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]), len(xx) - 1)]
    return dict(rows=len(x), finite_rows=int(finite.sum()), invalid_rows=int((~finite).sum()),
                effective_rows=float(1 / (pp @ pp)), mean=mean, std=math.sqrt(max(variance, 0)),
                quantile_probabilities=[.01, .05, .25, .5, .75, .95, .99], quantiles=quantiles,
                minimum=float(xx.min()), maximum=float(xx.max()))


def _observables(waveform: np.ndarray, q_floor: float) -> dict[str, np.ndarray]:
    y = np.asarray(waveform, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 480 or not np.isfinite(y).all():
        raise ValueError("Nonfinite/malformed endpoint waveform; no event deletion")
    baseline = y[:, :100].mean(1)
    centered = y - baseline[:, None]
    q = charge_np(y)
    tail = y[:, 150:400].sum(1) - 2.5 * y[:, :100].sum(1)
    denominator = np.where(q < 0, -1., 1.) * np.maximum(np.abs(q), q_floor)
    peak_indices = centered[:, 100:400].argmax(1) + 100
    peak = centered[np.arange(len(y)), peak_indices]
    half_rise = np.full(len(y), np.nan)
    for i, (row, peak_index, amplitude) in enumerate(zip(centered, peak_indices, peak)):
        if amplitude <= 0:
            continue
        threshold = .5 * amplitude
        before = row[99:peak_index]
        after = row[100:peak_index + 1]
        crossing = np.flatnonzero((before < threshold) & (after >= threshold))
        if crossing.size:
            left = int(crossing[0]) + 99
            half_rise[i] = left + (threshold - row[left]) / (row[left + 1] - row[left])
    return {"Q": q, "PSD": tail / denominator, "peak_voltage": peak,
            "peak_time_ns": peak_indices.astype(np.float64), "half_rise_ns": half_rise,
            "relative_half_rise_ns": half_rise - 120., "relative_peak_ns": peak_indices - 120.,
            "baseline_rms_V": np.sqrt((centered[:, :100]**2).mean(1)),
            "Q_nonpositive": q <= 0, "PSD_guarded": np.abs(q) < q_floor}


def _validation_weights(population_state: dict, q: np.ndarray, h: np.ndarray,
                        train_h: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply the saved family to new rows without a validation normalizer."""
    if population_state is None:
        return np.ones(len(q)), {"mode": "uniform_no_population_state"}
    array = lambda key: torch.as_tensor(population_state[key]).detach().cpu().numpy().astype(np.float64)
    features_train, theta = array("features"), array("theta")
    qcenter, qscale = float(array("source_logq_center")[0]), float(array("source_logq_scale")[0])
    hcenter, hscale = float(array("composition_center")[0]), float(array("composition_scale")[0])
    if len(train_h) != len(features_train) or (q <= 0).any() or not np.isfinite(h).all():
        raise ValueError("Validation population feature provenance/validity mismatch")
    quadratic_center = float(np.mean(features_train[:, 0]**2))
    composition_center = float(np.mean(np.tanh((np.asarray(train_h) - hcenter) / hscale)))
    linear = np.tanh((np.log(q) - qcenter) / qscale)
    features_val = np.column_stack((linear, linear**2 - quadratic_center,
                                   np.tanh((h - hcenter) / hscale) - composition_center))
    train_unnormalized = np.exp(.5 * np.tanh(features_train @ theta))
    normalizer = float(train_unnormalized.mean())
    weights = .5 + .5 * np.exp(.5 * np.tanh(features_val @ theta)) / normalizer
    return weights, dict(mode="saved_TRAIN_population_family", train_normalizer=normalizer,
                         validation_mean_weight=float(weights.mean()), validation_refit=False,
                         quadratic_center_from="saved_TRAIN_features", composition_center_from="TRAIN_h_saved_TRAIN_scaling",
                         train_h_rows=len(train_h), theta=theta, quadratic_center=quadratic_center,
                         composition_tanh_mean=composition_center,
                         bounds=[.5 + .5 / math.e, .5 + .5 * math.e])


@torch.no_grad()
def _generate(model: N01Response, source: np.ndarray, parent_rows: np.ndarray,
              draws: int, seed: int, device: torch.device) -> dict[str, np.ndarray]:
    rows = np.repeat(np.asarray(parent_rows, dtype=np.int64), draws)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    collected: dict[str, list[np.ndarray]] = {}
    keys = ("waveform", "qbar", "qsig", "time_scale", "pre_adc_charge", "electronics_charge",
            "charge_identity_residual", "adc_charge_change", "saturation_fraction_per_event")
    for start in range(0, len(rows), GENERATION_BATCH):
        selected = rows[start:start + GENERATION_BATCH]
        noise = model.sample_noise(len(selected), device="cpu", generator=generator)
        output = model(torch.as_tensor(source[selected], dtype=torch.float32, device=device), noise=noise, adc_ste=False)
        for key in keys:
            collected.setdefault(key, []).append(output[key].detach().cpu().numpy())
    result = {key: np.concatenate(value) for key, value in collected.items()}
    result["parent_row"] = rows
    return result


def _fit_mixture(values: np.ndarray, weights: np.ndarray | None = None,
                 minimum: int = 16, resolution: float = 1e-4) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(x) < minimum:
        return dict(status="insufficient_current_validation_support", rows=len(x), minimum_rows=minimum)
    p = _probability(weights, len(x))
    n_eff = float(1 / np.sum(p**2))
    mean, variance = float(p @ x), float(p @ (x - p @ x)**2)
    sigma = math.sqrt(max(variance, resolution**2))
    results = []
    for probability_pair in ((.25, .75), (.1, .9), (.4, .6)):
        order = np.argsort(x)
        cdf = np.cumsum(p[order])
        mu = x[order][np.minimum(np.searchsorted(cdf, probability_pair), len(x) - 1)].copy()
        sd, alpha = np.full(2, sigma), np.full(2, .5)
        previous, converged = -np.inf, False
        for iteration in range(200):
            logpdf = -.5 * ((x[:, None] - mu) / sd)**2 - np.log(sd * math.sqrt(2 * math.pi))
            log_terms = logpdf + np.log(np.maximum(alpha, 1e-12))
            normalizer = logsumexp(log_terms, axis=1)
            responsibilities = np.exp(log_terms - normalizer[:, None]) * p[:, None]
            mass = responsibilities.sum(0).clip(1e-12)
            mu = (responsibilities * x[:, None]).sum(0) / mass
            sd = np.sqrt(np.maximum((responsibilities * (x[:, None] - mu)**2).sum(0) / mass, resolution**2))
            alpha = mass / mass.sum()
            log_likelihood = float(p @ normalizer * n_eff)
            if np.isfinite(previous) and abs(log_likelihood - previous) < 1e-7 * max(1., abs(previous)):
                converged = True
                break
            previous = log_likelihood
        final_terms = (-.5 * ((x[:, None] - mu) / sd)**2 - np.log(sd * math.sqrt(2 * math.pi))
                       + np.log(np.maximum(alpha, 1e-12)))
        log_likelihood = float(p @ logsumexp(final_terms, axis=1) * n_eff)
        order = np.argsort(mu)
        results.append(dict(positions=mu[order], original_PSD_std=sd[order], mixture_fraction=alpha[order],
                            effective_log_likelihood=log_likelihood, converged=converged,
                            iterations=iteration + 1, initial_quantile_pair=list(probability_pair)))
    best = max(results, key=lambda entry: entry["effective_log_likelihood"])
    ll_one = float(p @ (-.5 * ((x - mean) / sigma)**2 - math.log(sigma * math.sqrt(2 * math.pi))) * n_eff)
    bic_one = 2 * math.log(n_eff) - 2 * ll_one
    bic_two = 5 * math.log(n_eff) - 2 * best["effective_log_likelihood"]
    mu, sd = best["positions"], best["original_PSD_std"]
    peak_separation = float((mu[1] - mu[0]) / math.sqrt(.5 * np.sum(sd**2)))
    positions = np.stack([entry["positions"] for entry in results])
    return dict(status="descriptive_two_Gaussian_fit", rows=len(x), effective_rows=n_eff,
                component_names=["lower_PSD", "upper_PSD"], **best,
                BIC_one_Gaussian=bic_one, BIC_two_Gaussian=bic_two,
                BIC_one_minus_two=bic_one - bic_two, standardized_peak_separation=peak_separation,
                ambiguity={"one_component_BIC_preferred": bic_one <= bic_two,
                           "position_range_across_initializations": np.ptp(positions, axis=0),
                           "all_initializations_converged": all(r["converged"] for r in results),
                           "component_overlap_measure": peak_separation,
                           "interpretation": "mixture components are descriptive PSD peaks, not true particle labels"},
                sigma_numerical_floor=resolution, initializations=results)


def _conditional_peaks(fake: dict, real: dict, q_reference: float, q_floor: float,
                       boundaries: np.ndarray, weights: np.ndarray, parent_ids: np.ndarray) -> list[dict]:
    def coordinate(q):
        return np.where(q >= q_floor, np.log(np.maximum(q, q_floor) / q_reference),
                        np.log(q_floor / q_reference) + np.arcsinh((q - q_floor) / q_floor))
    fb, rb = np.searchsorted(boundaries, coordinate(fake["Q"]), side="left"), np.searchsorted(boundaries, coordinate(real["Q"]), side="left")
    results = []
    for index in range(8):
        fm, rm = fb == index, rb == index
        results.append(dict(bin=index, coordinate="generated_output_Q_and_REAL_Q_in_same_TRAIN_bins",
                            fake_rows=int(fm.sum()), real_rows=int(rm.sum()),
                            fake_distinct_parents=int(len(np.unique(parent_ids[fm]))), minimum_rows_per_fit=16,
                            raw=_fit_mixture(fake["PSD"][fm]),
                            comparison_weighted=_fit_mixture(fake["PSD"][fm], weights[fm]),
                            REAL=_fit_mixture(real["PSD"][rm]),
                            fake_guarded_PSD_rows=int(fake["PSD_guarded"][fm].sum()),
                            real_guarded_PSD_rows=int(real["PSD_guarded"][rm].sum())))
    return results


def _weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid AUC score/weight")
    order = np.argsort(scores, kind="stable")
    scores, labels, weights = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positive = np.add.reduceat(weights * (labels == 1), starts)
    negative = np.add.reduceat(weights * (labels == 0), starts)
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    before_negative = np.cumsum(negative) - negative
    return float(np.sum(positive * (before_negative + .5 * negative)) / (positive.sum() * negative.sum()))


def _auc_bootstrap(labels: np.ndarray, scores: np.ndarray, fake_groups: np.ndarray,
                   real_groups: np.ndarray, weights: np.ndarray, seed: int) -> dict:
    _, fi = np.unique(fake_groups, return_inverse=True)
    _, ri = np.unique(real_groups, return_inverse=True)
    nf, nr = int(fi.max()) + 1, int(ri.max()) + 1
    rng, raw, weighted = np.random.default_rng(seed), [], []
    for _ in range(BOOTSTRAP_DRAWS):
        fm = np.bincount(rng.integers(0, nf, nf), minlength=nf)[fi]
        rm = np.bincount(rng.integers(0, nr, nr), minlength=nr)[ri]
        multiplicity = np.concatenate((fm, rm))
        raw.append(_weighted_auc(labels, scores, multiplicity))
        weighted.append(_weighted_auc(labels, scores, multiplicity * weights))
    def receipt(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return dict(percentile_95_interval=np.quantile(finite, [.025, .975]) if len(finite) else None,
                    bootstrap_SE=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                    valid_resamples=len(finite))
    return dict(method="separate_SOURCE_parent_and_REAL_original_group_cluster_bootstrap",
                resamples=BOOTSTRAP_DRAWS, source_test_parent_groups=nf, real_test_original_groups=nr,
                raw=receipt(raw), comparison_weighted=receipt(weighted),
                limited_group_count=(nr < 10), independent_acquisition_uncertainty=False)


class _DomainClassifier(nn.Module):
    """Separate endpoint classifier: no training discriminator/state is loaded."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 8, 9, stride=4, padding=4), nn.SiLU(),
                                      nn.Conv1d(8, 16, 9, stride=4, padding=4), nn.SiLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 30, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.features(x[:, None]))[:, 0]


def _split_groups(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two original groups are required for classifier fit/test separation")
    shuffled = np.random.default_rng(seed).permutation(unique)
    split = min(max(int(.6 * len(unique)), 1), len(unique) - 1)
    train = np.isin(groups, shuffled[:split])
    return np.flatnonzero(train), np.flatnonzero(~train)


@torch.enable_grad()
def _classifier_evaluation(fake: np.ndarray, real: np.ndarray, parent_ids: np.ndarray,
                           real_groups: np.ndarray, comparison_weights: np.ndarray,
                           stats: dict, device: torch.device, output_dir: Path) -> dict:
    try:
        ft, fe = _split_groups(parent_ids, CLASSIFIER_SEED)
        rt, re = _split_groups(real_groups, CLASSIFIER_SEED + 1)
    except ValueError as error:
        return dict(status="not_estimable_with_original_group_split", reason=str(error))
    # Fit and test subsets are frozen before fitting and never used to select
    # architecture, update count, checkpoint, response arm, or classification seed.
    train_x, test_x = np.concatenate((fake[ft], real[rt])), np.concatenate((fake[fe], real[re]))
    train_y, test_y = np.r_[np.zeros(len(ft)), np.ones(len(rt))], np.r_[np.zeros(len(fe)), np.ones(len(re))]
    center = torch.as_tensor(stats["views"]["voltage"]["center"], device=device)
    scale = torch.as_tensor(stats["views"]["voltage"]["scale"], device=device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CLASSIFIER_SEED)
        classifier = _DomainClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    rng = np.random.default_rng(CLASSIFIER_SEED + 2)
    # Balanced domain contribution; original comparison weights are used only
    # in evaluation, keeping one independently fitted classifier for both views.
    class_weight = np.where(train_y == 1, len(train_y) / (2 * len(rt)), len(train_y) / (2 * len(ft)))
    classifier.train()
    for _ in range(CLASSIFIER_UPDATES):
        index = rng.choice(len(train_x), min(GENERATION_BATCH, len(train_x)), replace=False)
        x = torch.as_tensor(train_x[index], dtype=torch.float32, device=device)
        y = torch.as_tensor(train_y[index], dtype=torch.float32, device=device)
        weight = torch.as_tensor(class_weight[index], dtype=torch.float32, device=device)
        loss = (F.binary_cross_entropy_with_logits(classifier((x - center) / scale), y, reduction="none") * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    classifier.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_x), GENERATION_BATCH):
            x = torch.as_tensor(test_x[start:start + GENERATION_BATCH], dtype=torch.float32, device=device)
            scores.append(classifier((x - center) / scale).cpu().numpy())
    scores = np.concatenate(scores)
    weights = np.r_[comparison_weights[fe], np.ones(len(re))]
    uncertainty = _auc_bootstrap(test_y, scores, parent_ids[fe], real_groups[re], weights, CLASSIFIER_SEED + 3)
    classifier_path = output_dir / "independent_domain_classifier.pt"
    torch.save({"model": {key: value.cpu() for key, value in classifier.state_dict().items()},
                "seed": CLASSIFIER_SEED, "updates": CLASSIFIER_UPDATES,
                "source_train_parent_ids": np.unique(parent_ids[ft]).tolist(),
                "source_test_parent_ids": np.unique(parent_ids[fe]).tolist(),
                "real_train_groups": np.unique(real_groups[rt]).tolist(),
                "real_test_groups": np.unique(real_groups[re]).tolist()}, classifier_path)
    np.savez_compressed(output_dir / "classifier_test_predictions.npz", labels=test_y, logits=scores,
                        comparison_weights=weights, source_test_parent_ids=parent_ids[fe], real_test_groups=real_groups[re],
                        fake_train_rows=ft, fake_test_rows=fe, real_train_rows=rt, real_test_rows=re)
    return dict(status="independent_fixed_budget_classifier_evaluated", label_1="REAL", seed=CLASSIFIER_SEED,
                architecture="Conv1d8_16_flatten_MLP32", updates=CLASSIFIER_UPDATES, learning_rate=1e-3,
                source_train_rows=len(ft), real_train_rows=len(rt), source_test_rows=len(fe), real_test_rows=len(re),
                source_test_parents=len(np.unique(parent_ids[fe])), real_test_original_groups=len(np.unique(real_groups[re])),
                split="60percent_unique_groups_for_fit_remaining_groups_for_test",
                raw_ROC_AUC=_weighted_auc(test_y, scores), comparison_weighted_ROC_AUC=_weighted_auc(test_y, scores, weights),
                uncertainty=uncertainty, training_discriminator_reused=False, heldout_selection=False,
                fit_scope="endpoint_validation_classifier_fit_subset_only", checkpoint=str(classifier_path))


def _cluster_mean_uncertainty(observables: dict, groups: np.ndarray,
                              weights: np.ndarray | None, seed: int) -> dict:
    names = ["Q", "PSD", "peak_voltage", "relative_half_rise_ns", "relative_peak_ns", "baseline_rms_V"]
    values = np.column_stack([observables[name] for name in names])
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.)
    p = _probability(weights, len(values))
    _, membership = np.unique(groups, return_inverse=True)
    count = int(membership.max()) + 1
    rng, means = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAP_DRAWS):
        multiplicity = np.bincount(rng.integers(0, count, count), minlength=count)[membership]
        mass = p * multiplicity
        denominator = (mass[:, None] * finite).sum(0)
        means.append((mass[:, None] * safe).sum(0) / np.maximum(denominator, 1e-300))
    means = np.asarray(means)
    return dict(method="original_parent_or_original_REAL_group_cluster_bootstrap_of_means",
                groups=count, resamples=BOOTSTRAP_DRAWS, feature_names=names,
                mean_sampling_SE=means.std(0, ddof=1), mean_percentile_95_interval=np.quantile(means, [.025, .975], axis=0),
                undefined_feature_rows=(~finite).sum(0), independent_acquisition_uncertainty=False)


def _population_summary(observables: dict, waveform: np.ndarray, weights: np.ndarray | None,
                        objective: N01Loss, groups: np.ndarray, seed: int) -> dict:
    summaries = {name: _summary(values, weights) for name, values in observables.items()
                 if values.dtype != np.bool_}
    noise = objective.noise_features(torch.as_tensor(waveform)).cpu().numpy()
    p = _probability(weights, len(waveform))
    noise_mean = p @ noise
    noise_std = np.sqrt(p @ (noise - noise_mean)**2)
    joint = np.column_stack((observables["Q"], observables["PSD"]))
    joint_centered = joint - p @ joint
    joint_covariance = (joint_centered * p[:, None]).T @ joint_centered
    return dict(scalars=summaries, joint_QPSD_covariance=joint_covariance,
                nonpositive_Q_rows=int(observables["Q_nonpositive"].sum()),
                guarded_PSD_rows=int(observables["PSD_guarded"].sum()),
                PSD_scope="all_rows_with_signed_guarded_denominator_nonpositive_nearzero_not_physical_PSD",
                noise={"feature_order": ["mean_V", "slope_V_per_normalized_baseline_time", "log_RMS_V"]
                       + [f"ACF_lag_{i}" for i in range(1, 17)] + ["amplitude_q01_V", "amplitude_q50_V", "amplitude_q99_V"]
                       + [f"power_fraction_band_{i}" for i in range(5)] + [f"local_log_variance_{i}" for i in range(5)],
                       "mean": noise_mean, "std": noise_std,
                       "ACF_lags": list(range(1, 17)), "ACF_mean": noise_mean[3:19],
                       "spectrum_fraction_mean": noise_mean[22:27], "baseline_samples": 100},
                uncertainty=_cluster_mean_uncertainty(observables, groups, weights, seed))


def _waveform_curves(waveform: np.ndarray, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    p = _probability(weights, len(waveform))
    centered = waveform - waveform[:, :100].mean(1, keepdims=True)
    peak = centered.max(1).clip(1 / 16383)
    shape = centered / peak[:, None]
    result = {}
    for name, values in (("voltage", waveform), ("normalized_shape", shape)):
        mean = p @ values
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        cdf = np.cumsum(p[order], axis=0)
        quantiles = []
        for probability in (.05, .5, .95):
            index = np.minimum((cdf < probability).sum(0), len(values) - 1)
            quantiles.append(sorted_values[index, np.arange(values.shape[1])])
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = np.sqrt(p @ (values - mean)**2)
        result[f"{name}_quantile_05_50_95"] = np.stack(quantiles)
    baseline = waveform[:, :100] - waveform[:, :100].mean(1, keepdims=True)
    power = np.abs(np.fft.rfft(baseline, axis=1))**2
    result["baseline_fft_power_mean_V2"] = p @ power
    result["baseline_fft_frequency_GHz"] = np.fft.rfftfreq(100, d=1.)
    return result


@torch.no_grad()
def _distance_receipt(objective: N01Loss, fake: np.ndarray, real: np.ndarray,
                      weights: np.ndarray | None = None) -> dict:
    values = objective.components({"waveform": torch.as_tensor(fake)}, torch.as_tensor(real),
                                  None if weights is None else torch.as_tensor(weights), "joint_stochastic")
    return dict(components={key: float(value) for key, value in values.items()},
                conditional_support=objective.last_diagnostics,
                interpretation="descriptive_frozen_TRAIN_scale_distribution_distances_no_acceptance_cutoff")


def _repeat_receipt(generated: dict[str, np.ndarray], source: np.ndarray, parent_rows: np.ndarray,
                    parent_ids: np.ndarray, q_floor: float, output_dir: Path) -> dict:
    waveform = generated["waveform"].reshape(len(parent_rows), REPEAT_DRAWS, 480)
    observed_flat = _observables(generated["waveform"], q_floor)
    observed = {name: values.reshape(len(parent_rows), REPEAT_DRAWS) for name, values in observed_flat.items()}
    q, psd = observed["Q"], observed["PSD"]
    qmean, pmean = q.mean(1), psd.mean(1)
    qvar, pvar = q.var(1, ddof=1), psd.var(1, ddof=1)
    cov = ((q - qmean[:, None]) * (psd - pmean[:, None])).sum(1) / (REPEAT_DRAWS - 1)
    # Sampling SE for conditional variances/covariance uses within-parent draw
    # bootstrap; no Gaussian variance-SE approximation is imposed.
    rng, uncertainty = np.random.default_rng(EVALUATION_SEED + 21), []
    for start in range(0, len(parent_rows), 16):
        qchunk, pchunk = q[start:start + 16], psd[start:start + 16]
        indices = rng.integers(0, REPEAT_DRAWS, (len(qchunk), BOOTSTRAP_DRAWS, REPEAT_DRAWS))
        bq = np.take_along_axis(qchunk[:, None, :], indices, axis=2)
        bp = np.take_along_axis(pchunk[:, None, :], indices, axis=2)
        bqmean, bpmean = bq.mean(2), bp.mean(2)
        covariance = ((bq - bqmean[:, :, None]) * (bp - bpmean[:, :, None])).sum(2) / (REPEAT_DRAWS - 1)
        estimates = np.stack((bq.var(2, ddof=1), bp.var(2, ddof=1), covariance), 2)
        uncertainty.append(estimates.std(1, ddof=1))
    uncertainty = np.concatenate(uncertainty)
    selected = waveform[:, :, SELECTED_TIME_SAMPLES]
    centered = selected - selected.mean(1, keepdims=True)
    products = centered[:, :, :, None] * centered[:, :, None, :]
    selected_covariance = products.sum(1) / (REPEAT_DRAWS - 1)
    selected_covariance_SE = products.std(1, ddof=1) / math.sqrt(REPEAT_DRAWS) * REPEAT_DRAWS / (REPEAT_DRAWS - 1)
    source_observed = _observables(source[parent_rows], q_floor)
    metrics = np.stack((q, psd, observed["relative_half_rise_ns"], observed["relative_peak_ns"], observed["baseline_rms_V"]), 2)
    metric_centered = metrics - metrics.mean(1, keepdims=True)
    metric_covariance = np.einsum("nri,nrj->nij", metric_centered, metric_centered) / (REPEAT_DRAWS - 1)
    numeric = dict(waveform=waveform, source_parent_rows=parent_rows, source_parent_ids=parent_ids,
                   source_waveform=source[parent_rows], selected_time_samples=SELECTED_TIME_SAMPLES,
                   Q=q, PSD=psd, variance_Q=qvar, variance_PSD=pvar, covariance_Q_PSD=cov,
                   mean_Q=qmean, mean_PSD=pmean, mean_Q_sampling_SE=q.std(1, ddof=1) / math.sqrt(REPEAT_DRAWS),
                   mean_PSD_sampling_SE=psd.std(1, ddof=1) / math.sqrt(REPEAT_DRAWS),
                   variance_covariance_bootstrap_SE=uncertainty,
                   relative_half_rise_ns=observed["relative_half_rise_ns"], relative_peak_ns=observed["relative_peak_ns"],
                   half_rise_delta_from_source_ns=observed["half_rise_ns"] - source_observed["half_rise_ns"][:, None],
                   peak_delta_from_source_ns=observed["peak_time_ns"] - source_observed["peak_time_ns"][:, None],
                   baseline_rms_V=observed["baseline_rms_V"],
                   baseline_rms_sampling_SE=observed["baseline_rms_V"].std(1, ddof=1) / math.sqrt(REPEAT_DRAWS),
                   selected_waveform_covariance=selected_covariance, selected_waveform_covariance_sampling_SE=selected_covariance_SE,
                   metric_covariance=metric_covariance)
    for name in ("qbar", "qsig", "time_scale", "electronics_charge", "adc_charge_change", "saturation_fraction_per_event"):
        numeric[name] = generated[name].reshape(len(parent_rows), REPEAT_DRAWS)
    path = output_dir / "repeated_parent_responses.npz"
    np.savez_compressed(path, **numeric)
    return dict(parents=len(parent_rows), draws_per_parent=REPEAT_DRAWS, parent_ids=parent_ids,
                selected_waveform_time_samples=SELECTED_TIME_SAMPLES,
                conditional_variance_Q=_summary(qvar), conditional_variance_PSD=_summary(pvar),
                conditional_covariance_Q_PSD=_summary(cov),
                mean_Q_sampling_SE=_summary(numeric["mean_Q_sampling_SE"]),
                mean_PSD_sampling_SE=_summary(numeric["mean_PSD_sampling_SE"]),
                variance_covariance_bootstrap_SE_columns=["Var_Q_given_parent", "Var_PSD_given_parent", "Cov_Q_PSD_given_parent"],
                uncertainty_method="within_parent_noise_draw_bootstrap_400_resamples",
                relative_time_reference_ns=120.,
                relative_half_rise_ns=_summary(observed["relative_half_rise_ns"]),
                relative_peak_ns=_summary(observed["relative_peak_ns"]),
                undefined_half_rise_rows=int(np.isnan(observed["relative_half_rise_ns"]).sum()),
                noise_RMS_V=_summary(observed["baseline_rms_V"]),
                covariance_feature_order=["Q", "PSD", "half_rise_minus120_ns", "peak_minus120_ns", "baseline_RMS_V"],
                covariance_sampling_SE="selected_time_covariance_cross_product_SE_is_asymptotic;QPSD_covariance_SE_is_bootstrap",
                numeric_file=str(path),
                interpretation="conditional_model_draw_statistics_not_unique_identification_of_true_physical_noise")


def _input_dependency(model: N01Response, source: np.ndarray, ids: np.ndarray,
                      rows: np.ndarray, q_floor: float, device: torch.device, output_dir: Path) -> dict:
    selected = rows[:8]
    original = source[selected].astype(np.float64)
    baseline = original[:, :100].mean(1, keepdims=True)
    reference_row = 0
    reference = source[reference_row].astype(np.float64)
    reference -= reference[:100].mean()
    template = reference / charge_np(reference[None])[0]
    modified_shape = baseline + charge_np(original)[:, None] * template[None]
    raised_scale = baseline + 1.2 * (original - baseline)
    lowered_scale = baseline + .8 * (original - baseline)
    variants = {"original": original, "common_normalized_shape_same_Q": modified_shape,
                "same_shape_scale_1p2": raised_scale, "same_shape_scale_0p8": lowered_scale}
    noise_seed, draws = EVALUATION_SEED + 31, 32
    generated, observed = {}, {}
    for name, parents in variants.items():
        generated[name] = _generate(model, parents.astype(np.float32), np.arange(len(selected)), draws, noise_seed, device)
        observed[name] = _observables(generated[name]["waveform"], q_floor)
    numeric = {"source_parent_ids": ids[selected], "reference_shape_parent_id": ids[reference_row],
               "explicit_noise_seed": np.array(noise_seed), "draws": np.array(draws)}
    comparisons = {}
    for name, parents in variants.items():
        numeric[f"{name}_source_waveform"] = parents
        numeric[f"{name}_generated_waveform"] = generated[name]["waveform"].reshape(len(selected), draws, 480)
        if name == "original":
            continue
        metrics = {}
        for feature in ("Q", "PSD", "relative_half_rise_ns", "relative_peak_ns", "baseline_rms_V"):
            difference = (observed[name][feature] - observed["original"][feature]).reshape(len(selected), draws)
            numeric[f"{name}_{feature}_paired_difference"] = difference
            metrics[feature] = dict(parent_mean_paired_difference=np.nanmean(difference, axis=1),
                                    paired_sampling_SE=np.nanstd(difference, axis=1, ddof=1) / math.sqrt(draws),
                                    undefined_rows=int((~np.isfinite(difference)).sum()))
        mean_difference = (numeric[f"{name}_generated_waveform"] - numeric["original_generated_waveform"]).mean(1)
        metrics["mean_voltage_change_RMS_V"] = np.sqrt(np.mean(mean_difference**2, axis=1))
        metrics["input_voltage_change_RMS_V"] = np.sqrt(np.mean((parents - original)**2, axis=1))
        metrics["input_Q_ratio"] = charge_np(parents) / charge_np(original)
        metrics["electronics_charge_same_noise_max_abs_difference"] = float(np.max(np.abs(generated[name]["electronics_charge"] - generated["original"]["electronics_charge"])))
        comparisons[name] = metrics
    path = output_dir / "same_noise_input_interventions.npz"
    np.savez_compressed(path, **numeric)
    return dict(parents=len(selected), draws_per_parent=draws, parent_ids=ids[selected], explicit_noise_seed=noise_seed,
                reference_shape_parent_id=ids[reference_row], paired_latents_identical=True,
                changing_input_tests=comparisons, numeric_file=str(path), ID_as_network_input=False,
                criterion="descriptive_paired_changes_and_response_means;no_shuffle_marginal_invariance_criterion",
                interpretation="a_small_or_identical_input_shape_change_can_be_uninformative;input_change_magnitudes_are_reported")


def _population_stability(output_dir: Path, current_arm: str, container_hash: str, identity_hash: str, training_seed: int) -> dict:
    needed = {"full", "full_uniform"}
    if current_arm not in needed:
        return dict(status="pending_full_and_full_uniform_fixed_endpoints", required_arms=sorted(needed))
    counterpart = (needed - {current_arm}).pop()
    sibling = output_dir.parent.parent / counterpart / output_dir.name
    report_path, repeated_path = sibling / "endpoint_report.json", sibling / "repeated_parent_responses.npz"
    if not report_path.exists() or not repeated_path.exists():
        return dict(status="pending_counterpart_fixed_endpoint", current_arm=current_arm, required_counterpart=counterpart,
                    expected_counterpart_report=str(report_path), independent_retraining_required=True)
    other_report = json.loads(report_path.read_text(encoding="utf8"))
    if (other_report.get("experiment_id") != "N01" or other_report.get("arm") != counterpart or other_report.get("evaluation_seed") != EVALUATION_SEED or other_report.get("container_sha256") != container_hash or other_report.get("identity_sha256") != identity_hash):
        return dict(status="counterpart_provenance_mismatch", expected_counterpart=counterpart)
    with np.load(output_dir / "repeated_parent_responses.npz", allow_pickle=False) as a, np.load(repeated_path, allow_pickle=False) as b:
        if not np.array_equal(a["source_parent_ids"], b["source_parent_ids"]):
            return dict(status="counterpart_repeat_parent_identity_mismatch")
        changes = {}
        rng = np.random.default_rng(EVALUATION_SEED + 41)
        for feature in ("qbar", "Q", "PSD", "time_scale", "variance_Q", "variance_PSD", "covariance_Q_PSD"):
            av, bv = a[feature], b[feature]
            if av.ndim == 2:
                av, bv = av.mean(1), bv.mean(1)
            delta = av - bv
            bootstrap = delta[rng.integers(0, len(delta), (BOOTSTRAP_DRAWS, len(delta)))].mean(1)
            changes[feature] = dict(parent_difference_summary=_summary(delta),
                                    mean_parent_difference_95_interval=np.quantile(bootstrap, [.025, .975]),
                                    mean_parent_difference_sampling_SE=float(bootstrap.std(ddof=1)))
    return dict(status="descriptive_separately_retrained_endpoint_comparison_available",
                current_minus_counterpart=[current_arm, counterpart], counterpart_report=str(report_path),
                independent_retraining_required=True, matched_repeat_parents=REPEAT_PARENTS, changes=changes,
                matching_training_seed=other_report.get("training_seed") == training_seed,
                seed_difference_can_confound_population_sensitivity=other_report.get("training_seed") != training_seed,
                scientific_pass_threshold=None,
                interpretation="response_population_sensitivity_is_described;no_unique_physical_identification_claim")


def evaluate_checkpoint(checkpoint_path, container, identity_path, output_dir, device="cuda") -> dict:
    """Evaluate one caller-declared fixed endpoint; return the saved report dict.

    It generates endpoint validation responses, trains an independent diagnostic
    classifier on its designated fit subset, and saves all descriptive outputs.
    It never chooses a training checkpoint or changes the response parameters.
    """
    checkpoint_path, container, identity_path, output_dir = map(Path, (checkpoint_path, container, identity_path, output_dir))
    if os.name == "nt" and output_dir.drive.upper() not in ("E:", "D:"):
        raise ValueError("N01 evaluation outputs must be on E: or D:")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash, container_hash, identity_hash = sha256(checkpoint_path), sha256(container), sha256(identity_path)
    report_path = output_dir / "endpoint_report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf8"))
        complete_path = output_dir / "EVALUATION_COMPLETE.json"
        if previous.get("checkpoint_sha256") == checkpoint_hash and previous.get("container_sha256") == container_hash and previous.get("identity_sha256") == identity_hash:
            if complete_path.exists():
                complete = json.loads(complete_path.read_text(encoding="utf8"))
                if complete.get("report_sha256") == sha256(report_path):
                    for artifact in complete.get("artifacts", {}).values():
                        artifact_path = Path(artifact["path"])
                        if not artifact_path.exists() or sha256(artifact_path) != artifact["sha256"]:
                            raise ValueError(f"Completed evaluation artifact changed: {artifact_path}")
                    summarize_arm_comparison(output_dir.parent.parent)
                    return previous
            raise RuntimeError("Endpoint report has no verified completion receipt; preserve outputs and use a new directory")
        raise FileExistsError("Preserve previous evaluation: use a new E:/D: output directory for a different endpoint/input")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA endpoint evaluation but CUDA is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model", "population", "model_statistics", "loss_statistics", "loss_state", "seed", "variant", "arm"}
    if not required <= set(checkpoint):
        raise ValueError(f"Endpoint checkpoint missing {sorted(required - set(checkpoint))}")
    if checkpoint["arm"] == "baseline":
        raise ValueError("Baseline is not a completed response arm for endpoint evaluation")
    data = load_validation(container, identity_path)
    source, real, source_ids, real_ids, real_groups = (data[key] for key in ("source", "real", "source_ids", "real_ids", "real_groups"))
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(real_ids)) != len(real_ids):
        raise ValueError("Validation original parent identities are not unique")
    # Only TRAIN h is read for the saved population feature centering. No TRAIN
    # waveform or validation value is fitted into a new loss/model scale here.
    with np.load(container, allow_pickle=False) as archive:
        train_h = archive["h"].astype(np.float64)
    comparison_weights, weight_receipt = _validation_weights(checkpoint["population"], charge_np(source), data["h"], train_h)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EVALUATION_SEED)
        model = N01Response(checkpoint["model_statistics"], checkpoint["variant"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_stage("low_learning_rate")
    model.eval().to(target_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Fixed CPU statistics and projection state keep distance reductions bounded;
    # only response generation/classifier minibatches use the requested device.
    objective = N01Loss(checkpoint["loss_statistics"], seed=int(checkpoint["seed"]))
    objective.load_state_dict(checkpoint["loss_state"], strict=True)
    q_floor, q_reference = float(objective.q_floor), float(objective.q_reference)
    parent_rows = np.arange(len(source))
    generated = _generate(model, source, parent_rows, MARGINAL_DRAWS, EVALUATION_SEED + 1, target_device)
    fake = generated["waveform"]
    fake_ids = source_ids[generated["parent_row"]]
    fake_weights = comparison_weights[generated["parent_row"]]
    fake_observed, real_observed = _observables(fake, q_floor), _observables(real, q_floor)
    raw = _population_summary(fake_observed, fake, None, objective, fake_ids, EVALUATION_SEED + 2)
    weighted = _population_summary(fake_observed, fake, fake_weights, objective, fake_ids, EVALUATION_SEED + 2)
    real_summary = _population_summary(real_observed, real, None, objective, real_groups, EVALUATION_SEED + 3)
    raw["distribution_distances"] = _distance_receipt(objective, fake, real)
    weighted["distribution_distances"] = _distance_receipt(objective, fake, real, fake_weights)
    conditional = _conditional_peaks(fake_observed, real_observed, q_reference, q_floor,
                                     objective.charge_bin_boundaries.numpy(), fake_weights, fake_ids)
    numeric = dict(source_parent_ids=source_ids, source_parent_count_weights=np.ones(len(source)),
                   source_parent_comparison_weights=comparison_weights, source_h=data["h"],
                   real_original_ids=real_ids, real_original_groups=real_groups,
                   generated_source_parent_ids=fake_ids, generated_comparison_weights=fake_weights,
                   REAL_waveform=real, generated_waveform=fake)
    numeric.update({f"generated_{key}": value for key, value in generated.items() if key != "waveform"})
    numeric.update({f"generated_{key}": value for key, value in fake_observed.items()})
    numeric.update({f"REAL_{key}": value for key, value in real_observed.items()})
    for name, values in (("raw", _waveform_curves(fake)), ("comparison_weighted", _waveform_curves(fake, fake_weights)),
                         ("REAL", _waveform_curves(real))):
        numeric.update({f"{name}_{key}": value for key, value in values.items()})
    np.savez_compressed(output_dir / "endpoint_populations.npz", **numeric)
    try:
        first_real, second_real = _split_groups(real_groups, EVALUATION_SEED + 11)
        real_real = dict(status="original_group_disjoint_REAL_REAL_control",
                         first_original_ids=real_ids[first_real], second_original_ids=real_ids[second_real],
                         first_groups=np.unique(real_groups[first_real]), second_groups=np.unique(real_groups[second_real]),
                         distances=_distance_receipt(objective, real[first_real], real[second_real]))
    except ValueError as error:
        real_real = dict(status="not_estimable_original_groups_insufficient", reason=str(error))
    def adc_check(x):
        first = np.rint((np.clip(x, -.5, .5) + .5) * 16383) / 16383 - .5
        second = np.rint((first + .5) * 16383) / 16383 - .5
        return dict(input_rows=len(x), canonicalization_max_abs_change=float(np.max(np.abs(first - x))),
                    second_application_max_abs_change=float(np.max(np.abs(second - first))),
                    exact_second_application_idempotent=bool(np.array_equal(first, second)))
    controls = dict(REAL_REAL=real_real, common_ADC={"operator": "14bit_inclusive[-0.5,0.5]_nearest_even_1ns",
                    "REAL": adc_check(real), "generated": adc_check(fake),
                    "source_input_preclipping": False, "stochastic_forward_REAL_identity_requirement": False})
    classifier = _classifier_evaluation(fake, real, fake_ids, real_groups, fake_weights,
                                         checkpoint["loss_statistics"], target_device, output_dir)
    selected = np.sort(np.random.default_rng(EVALUATION_SEED + 20).choice(len(source), REPEAT_PARENTS, replace=False))
    repeated = _generate(model, source, selected, REPEAT_DRAWS, EVALUATION_SEED + 22, target_device)
    repeat_receipt = _repeat_receipt(repeated, source, selected, source_ids[selected], q_floor, output_dir)
    dependency = _input_dependency(model, source, source_ids, selected, q_floor, target_device, output_dir)
    stability = _population_stability(output_dir, str(checkpoint["arm"]), container_hash, identity_hash, int(checkpoint["seed"]))
    report = dict(schema_version=1, experiment_id="N01", arm=checkpoint["arm"], variant=checkpoint["variant"],
                  training_seed=int(checkpoint["seed"]), evaluation_seed=EVALUATION_SEED,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                  container=str(container), container_sha256=container_hash,
                  identity_path=str(identity_path), identity_sha256=identity_hash,
                  output_dir=str(output_dir), device=str(target_device), maximum_generation_batch=GENERATION_BATCH,
                  checkpoint_scope="caller_declared_fixed_endpoint_no_selection",
                  validation_loaded_only_in_endpoint_evaluator=True,
                  validation_source_parents=len(source), validation_REAL_original_parents=len(real),
                  marginal_draws_per_source_parent=MARGINAL_DRAWS, population_weights=weight_receipt,
                  raw_population=raw, comparison_weighted_population=weighted, REAL_population=real_summary,
                  conditional_PSD_peaks_and_original_widths=conditional, controls=controls,
                  independent_domain_classifier=classifier, repeated_parent_response=repeat_receipt,
                  same_noise_input_dependency=dependency, population_response_stability=stability,
                  charge_identity_max_abs_Vns=float(np.max(np.abs(generated["charge_identity_residual"]))),
                  generated_saturation_fraction_mean=float(generated["saturation_fraction_per_event"].mean()),
                  statistical_status="descriptive_endpoint_results_no_preexisting_hard_acceptance_thresholds",
                  physical_calibration="unknown", independent_acquisition_count=1,
                  uncertainty_scope="finite_original_groups_and_model_draws;not_independent_acquisition_uncertainty",
                  true_particle_labels_available=False, pseudolabel_accuracy_claimed=False,
                  original_parents_and_count_weights_preserved=True,
                  physical_weights_status="unknown_not_present_in_selected_interface",
                  artifacts={str(path.name): {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for path in sorted(output_dir.iterdir()) if path.is_file() and path.name not in ("endpoint_report.json", "EVALUATION_COMPLETE.json")})
    report = _clean(report)
    write_json(report_path, report)
    write_json(output_dir / "EVALUATION_COMPLETE.json", dict(
        status="complete", completed_UTC=datetime.now(timezone.utc).isoformat(),
        experiment_id="N01", arm=report["arm"], report_path=str(report_path),
        report_sha256=sha256(report_path), checkpoint_sha256=checkpoint_hash,
        container_sha256=container_hash, identity_sha256=identity_hash,
        evaluator_source_sha256=sha256(Path(__file__)),
        scientific_status="descriptive_no_scientific_pass_threshold",
        artifacts=report["artifacts"]))
    summarize_arm_comparison(output_dir.parent.parent)
    return report



def summarize_arm_comparison(run_root) -> dict:
    """Aggregate verified frozen arm evaluations; never rank/select endpoints."""
    run_root = Path(run_root)
    if os.name == "nt" and run_root.drive.upper() not in ("E:", "D:"):
        raise ValueError("N01 aggregate outputs must remain on E: or D:")
    reports = {}
    for marker in sorted(run_root.rglob("EVALUATION_COMPLETE.json")):
        complete = json.loads(marker.read_text(encoding="utf8"))
        if complete.get("experiment_id") != "N01" or complete.get("status") != "complete":
            continue
        report_path = Path(complete["report_path"])
        if not report_path.exists() or sha256(report_path) != complete["report_sha256"]:
            raise ValueError(f"Completed evaluation report hash mismatch: {report_path}")
        for artifact in complete.get("artifacts", {}).values():
            artifact_path = Path(artifact["path"])
            if not artifact_path.exists() or sha256(artifact_path) != artifact["sha256"]:
                raise ValueError(f"Completed evaluation artifact changed: {artifact_path}")
        report = json.loads(report_path.read_text(encoding="utf8"))
        arm = str(report["arm"])
        if arm in reports:
            raise ValueError(f"Multiple frozen endpoints for {arm}; narrow run_root rather than selecting a result")
        reports[arm] = report
    expected = ["full", "deterministic", "simple", "full_uniform"]
    rows = []
    for arm in expected:
        if arm not in reports:
            rows.append(dict(arm=arm, status="pending_fixed_endpoint_evaluation"))
            continue
        report = reports[arm]
        classifier = report["independent_domain_classifier"]
        row = dict(arm=arm, status="descriptive_completed_endpoint",
                   checkpoint=report["checkpoint"], checkpoint_sha256=report["checkpoint_sha256"],
                   raw_distances=report["raw_population"]["distribution_distances"]["components"],
                   comparison_weighted_distances=report["comparison_weighted_population"]["distribution_distances"]["components"],
                   raw_ROC_AUC=classifier.get("raw_ROC_AUC"),
                   comparison_weighted_ROC_AUC=classifier.get("comparison_weighted_ROC_AUC"),
                   classifier_uncertainty=classifier.get("uncertainty"), classifier_status=classifier["status"],
                   raw_mean_Q=report["raw_population"]["scalars"]["Q"]["mean"],
                   comparison_weighted_mean_Q=report["comparison_weighted_population"]["scalars"]["Q"]["mean"],
                   raw_mean_PSD=report["raw_population"]["scalars"]["PSD"]["mean"],
                   raw_mean_baseline_RMS_V=report["raw_population"]["scalars"]["baseline_rms_V"]["mean"],
                   saturation_fraction=report["generated_saturation_fraction_mean"],
                   conditional_bins_used=report["raw_population"]["distribution_distances"]["conditional_support"]["conditional_bins_used"],
                   repeat_variance_Q=report["repeated_parent_response"]["conditional_variance_Q"],
                   repeat_variance_PSD=report["repeated_parent_response"]["conditional_variance_PSD"])
        rows.append(row)
    stability = dict(status="pending_full_and_full_uniform_fixed_endpoints")
    if "full" in reports and "full_uniform" in reports:
        full, uniform = reports["full"], reports["full_uniform"]
        if full["container_sha256"] != uniform["container_sha256"] or full["identity_sha256"] != uniform["identity_sha256"]:
            raise ValueError("Population-sensitivity endpoints use different validation provenance")
        for report in (full, uniform):
            if sha256(report["checkpoint"]) != report["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint hash changed after evaluation")
        a = torch.load(full["checkpoint"], map_location="cpu", weights_only=False)
        b = torch.load(uniform["checkpoint"], map_location="cpu", weights_only=False)
        prefixes = ["shape_encoder", "charge_head", "response_filter", "mean_time", "electronics",
                    "unet", "mean_head", "physical_head", "basis_head", "envelope_head", "local_filter"]
        parameter_changes = {}
        for prefix in prefixes:
            keys = [key for key, value in a["model"].items()
                    if key.startswith(prefix + ".") and key in b["model"] and value.is_floating_point()
                    and key.rsplit(".", 1)[-1] not in ("impulse", "limits", "rows", "cols", "kernel")]
            if not keys:
                continue
            left = torch.cat([a["model"][key].double().reshape(-1) for key in keys])
            right = torch.cat([b["model"][key].double().reshape(-1) for key in keys])
            delta = left - right
            parameter_changes[prefix] = dict(entries=left.numel(), L2_difference=float(delta.norm()),
                                             relative_L2_difference=float(delta.norm() / left.norm().clamp_min(1e-12)),
                                             maximum_absolute_difference=float(delta.abs().max()))
        fpath = full["repeated_parent_response"]["numeric_file"]
        upath = uniform["repeated_parent_response"]["numeric_file"]
        common_changes = {}
        with np.load(fpath, allow_pickle=False) as f, np.load(upath, allow_pickle=False) as u:
            common_ids, fi, ui = np.intersect1d(f["source_parent_ids"], u["source_parent_ids"], return_indices=True)
            if not len(common_ids):
                raise ValueError("No common original repeat parents across full/full_uniform")
            rng = np.random.default_rng(EVALUATION_SEED + 51)
            for feature in ("qbar", "Q", "PSD", "time_scale", "variance_Q", "variance_PSD", "covariance_Q_PSD"):
                fv, uv = f[feature][fi], u[feature][ui]
                if fv.ndim == 2:
                    fv, uv = fv.mean(1), uv.mean(1)
                difference = fv - uv
                draws = difference[rng.integers(0, len(difference), (BOOTSTRAP_DRAWS, len(difference)))].mean(1)
                common_changes[feature] = dict(full_minus_full_uniform=_summary(difference),
                                               mean_difference_sampling_SE=float(draws.std(ddof=1)),
                                               mean_difference_percentile_95_interval=np.quantile(draws, [.025, .975]))
            mean_waveform_difference = f["waveform"][fi].mean(1) - u["waveform"][ui].mean(1)
            common_changes["mean_voltage_difference_RMS_V"] = _summary(np.sqrt(np.mean(mean_waveform_difference**2, axis=1)))
        stability = dict(status="descriptive_separately_retrained_endpoint_comparison_available",
                         shared_response_parameter_changes=parameter_changes,
                         full_population_theta=a["population"]["theta"],
                         full_uniform_population_theta=b["population"]["theta"],
                         common_original_parent_count=len(common_ids), common_original_parent_ids=common_ids,
                         common_parent_response_changes=common_changes,
                         matching_training_seed=full["training_seed"] == uniform["training_seed"],
                         seed_difference_can_confound_population_sensitivity=full["training_seed"] != uniform["training_seed"],
                         hard_pass_threshold=None,
                         interpretation="parameter_and_response_sensitivity_do_not_identify_a_unique_physical_response")
    summary = _clean(dict(schema_version=1, experiment_id="N01",
                          status="all_fixed_endpoints_described" if all(arm in reports for arm in expected) else "partial_pending_endpoints",
                          arm_metrics_side_by_side=rows, population_response_stability=stability,
                          endpoint_selection=False, hard_acceptance_thresholds=None,
                          independent_acquisition_count=1))
    write_json(run_root / "ARM_COMPARISON.json", summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.container, args.identity, args.output, args.device)
    print(json.dumps({"endpoint_report": str(Path(args.output) / "endpoint_report.json"),
                      "arm": report["arm"], "status": report["statistical_status"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
