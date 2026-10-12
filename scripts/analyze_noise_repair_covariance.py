"""Read-only TRAIN covariance identifiability audit of frozen N01 noise repair.

No model optimization, validation fitting, gate changes, or response training.
All statistical input is the existing canonical REAL TRAIN first100 window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_key] = "2"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from n01.data import canonical_real
from n01.electronics_v2 import ElectronicsNoiseV2

torch.set_num_threads(2)
SEED = 2026101251


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    return np.cov(x, rowvar=False, ddof=1)


def ols_operators(length=100):
    t = np.linspace(-1.0, 1.0, length)
    design = np.column_stack((np.ones(length), t))
    operator = np.vstack((np.ones(length)/length, t/(t @ t)))
    residual = np.eye(length) - design @ operator
    return t, design, operator, residual


def toeplitz(sequence, length):
    index = abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    return np.asarray(sequence, dtype=np.float64)[index]


def summarize_2(c):
    c = (np.asarray(c) + np.asarray(c).T)/2
    diagonal = np.diag(c)
    sd = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
    denominator = sd.prod()
    return dict(covariance_V2=c, marginal_SD_V=sd,
                mean_slope_correlation=c[0, 1]/denominator if denominator else None,
                eigenvalues_V2=np.linalg.eigvalsh(c))


def maximum_noise_scale(observed, noise_ols):
    """Largest alpha >=0 with observed-alpha*noise_ols positive semidefinite."""
    root = np.linalg.cholesky(observed)
    whitened = np.linalg.solve(root, noise_ols)
    whitened = np.linalg.solve(root, whitened.T).T
    largest = np.linalg.eigvalsh((whitened+whitened.T)/2)[-1]
    return float(1/largest) if largest > 0 else float("inf")


def empirical_stationary_estimate(y):
    """PSD lag covariance via zero-padded finite-window autocorrelation.

    This estimate is a descriptive pooled second moment, NOT identified K.
    It includes latent offset/drift and applies explicit finite-window taper.
    """
    centered = y - y.mean(0, keepdims=True)
    n, length = centered.shape
    transform = np.fft.rfft(centered, n=2*length, axis=1)
    sequence = np.fft.irfft(np.mean(abs(transform)**2, axis=0), n=2*length)[:length]
    return sequence/length * n/(n-1)


def fit_second_moment_family(c, maximum_lag=64):
    """Unconstrained linear covariance analysis; no neural/model training.

    C = X S X.T + Toeplitz(k[0..maximum_lag], zero thereafter).
    PSD of S, K and the infinite stationary spectral polynomial is reported,
    not imposed or repaired. Negative checks forbid interpreting it as a model.
    """
    length = len(c)
    _, x, _, _ = ols_operators(length)
    bases = [np.ones_like(c), np.outer(x[:, 1], x[:, 1]),
             x[:, 1, None]+x[None, :, 1]]
    distance = abs(np.arange(length)[:, None]-np.arange(length)[None, :])
    bases.extend((distance == lag).astype(np.float64) for lag in range(maximum_lag+1))
    design = np.stack([b.ravel() for b in bases], axis=1)
    values, _, rank, singular = np.linalg.lstsq(design, c.ravel(), rcond=None)
    latent = np.array([[values[0], values[2]], [values[2], values[1]]])
    sequence = np.r_[values[3:], np.zeros(length-maximum_lag-1)]
    k = toeplitz(sequence, length)
    estimate = x @ latent @ x.T+k
    frequencies = np.linspace(0, np.pi, 4097)
    spectral = sequence[0] + 2 * np.cos(frequencies[:, None]*np.arange(1, maximum_lag+1)) @ sequence[1:maximum_lag+1]
    info = dict(method="unconstrained_Frobenius_linear_second_moment_least_squares",
                neural_training_updates=0, fit_split="TRAIN", full_matrix_entries=length**2,
                design_rank=int(rank), parameters=len(values), condition_number=float(singular[0]/singular[-1]),
                latent=summarize_2(latent), K_minimum_finite100_eigenvalue_V2=float(np.linalg.eigvalsh(k)[0]),
                stationary_spectral_grid_points=len(frequencies),
                stationary_spectral_grid_minimum_V2=float(spectral.min()),
                negative_spectral_grid_is_a_definite_PSD_violation=True,
                positive_grid_alone_is_not_an_all_frequency_proof=True,
                relative_covariance_Frobenius_error=float(np.linalg.norm(c-estimate)/np.linalg.norm(c)),
                PSD_constraints_imposed=False, deployed=False,
                assumptions=["exact_linear_population_drift_on_first100",
                             "stationary_residual_and_drift_independence",
                             "finite_FIR65_covariance_support64",
                             "ADC_covariance_is_treated_as_preADC_approximation"])
    return info, latent, k, estimate


def group_bootstrap(ols, groups, frozen_noise_ols, resamples, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(unique), 2))
    squares = np.zeros((len(unique), 2, 2))
    np.add.at(sums, inverse, ols)
    np.add.at(squares, inverse, ols[:, :, None]*ols[:, None, :])
    rng = np.random.default_rng(seed)
    rho, eigen, alpha = [], [], []
    for _ in range(resamples):
        draw = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        n = draw @ counts
        mean_sum = draw @ sums
        second = np.einsum("g,gij->ij", draw, squares)
        c = (second-np.outer(mean_sum, mean_sum)/n)/(n-1)
        rho.append(c[0, 1]/np.sqrt(c[0, 0]*c[1, 1]))
        eigen.append(np.linalg.eigvalsh(c-frozen_noise_ols)[0])
        alpha.append(maximum_noise_scale(c, frozen_noise_ols))
    return dict(method="resample_original_TRAIN_groups_with_all_member_events",
                seed=seed, resamples=resamples, original_groups=len(unique),
                observed_correlation_percentile95=np.quantile(rho, [.025, .975]),
                frozen_K_latent_min_eigenvalue_percentile95_V2=np.quantile(eigen, [.025, .975]),
                frozen_K_latent_PSD_resample_fraction=float(np.mean(np.asarray(eigen) >= 0)),
                maximum_frozen_K_variance_scale_percentile95=np.quantile(alpha, [.025, .975]),
                scope="conditional_on_this_fitted_K_and_this_acquisition;no_retraining_or_independent_acquisition_uncertainty")


def self_checks(seed=SEED):
    rng = np.random.default_rng(seed)
    _, x, a, residual = ols_operators()
    h = rng.normal(size=65)
    h /= np.linalg.norm(h)
    sequence = np.r_[[h[:65-lag] @ h[lag:] for lag in range(65)], np.zeros(35)]*2e-7
    k = toeplitz(sequence, 100)
    noise_ols = a @ k @ a.T
    assert abs(noise_ols[0, 1]) < 1e-20
    assert np.linalg.norm(k-k[::-1, ::-1]) == 0
    known = np.array([[2e-8, -7e-9], [-7e-9, 1e-8]])
    observed = a @ (x @ known @ x.T+k) @ a.T
    np.testing.assert_allclose(observed-noise_ols, known, rtol=1e-12, atol=1e-20)
    assert np.linalg.eigvalsh(observed-noise_ols)[0] > 0
    assert np.linalg.norm(a @ residual) < 1e-14
    np.testing.assert_allclose(a @ (residual @ k @ residual.T) @ a.T, 0, atol=1e-20)
    change = np.array([[1, 380/99], [0, 479/99]])
    changed = change @ known @ change.T
    recovered = np.linalg.solve(change, changed)
    recovered = np.linalg.solve(change, recovered.T).T
    np.testing.assert_allclose(recovered, known, rtol=1e-12, atol=1e-20)
    assert np.linalg.eigvalsh(changed)[0] > 0
    # Copying observed OLS variance to latent adds K once more.
    copied = observed+noise_ols
    assert np.all(np.diag(copied) > np.diag(observed))
    # An artificially excessive stationary K makes latent subtraction illegal.
    excessive = np.diag(np.diag(observed)*1.1)
    assert np.linalg.eigvalsh(observed-excessive)[0] < 0
    return dict(seed=seed, checks=6, status="passed",
                meanings=["reflection_covariance_zero", "known_joint_latent_recovered",
                          "OLS_residual_cannot_identify_removed_modes",
                          "coordinate_congruence_preserves_PSD",
                          "copied_OLS_variance_double_counts_K", "nonPSD_subtraction_rejected"])


def analyze(checkpoint_path, container, identities, gate_path, output_dir,
            seed=SEED, bootstrap_resamples=512):
    out = Path(output_dir).resolve()
    if os.name == "nt" and out.drive.upper() not in ("E:", "D:"):
        raise ValueError("All audit artifacts must stay on E/D on Windows")
    if bootstrap_resamples < 1:
        raise ValueError("At least one declared bootstrap resample required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["engineering"] is not False or checkpoint["update"] != 5000:
        raise ValueError("Require frozen 5000-update scientific noise endpoint")
    if checkpoint["response_training_updates"] != 0:
        raise ValueError("Expected gate-stopped response training")
    if sha256(container) != checkpoint["config"]["train_container_sha256"]:
        raise ValueError("Registered TRAIN container hash mismatch")
    if sha256(identities) != checkpoint["config"]["identities_sha256"]:
        raise ValueError("Registered identities hash mismatch")
    with np.load(container, allow_pickle=False) as z:
        # Read the existing TRAIN member only. target_val/source_val are not read.
        y = canonical_real(z["target"][:, :100])
    with np.load(identities, allow_pickle=False) as z:
        groups = z["real_train_group"].copy()
        ids = z["real_train_original_index"].copy()
    if y.shape != (21047, 100) or len(ids) != len(y) or len(np.unique(ids)) != len(y):
        raise ValueError("Legal complete TRAIN row/parent identity mismatch")
    if len(groups) != len(y):
        raise ValueError("TRAIN group length mismatch")
    stats = checkpoint["loss_statistics"]
    if stats["fit_split"] != "TRAIN" or stats["fit_rows"] != len(y) or stats["baseline_samples"] != 100:
        raise ValueError("Frozen TRAIN fitted-statistics provenance mismatch")
    scale = stats["waveform"]["scale"].double().numpy()
    quantiles = np.quantile(y, [.25, .5, .75], axis=0)
    reproduced_scale = np.maximum((quantiles[2]-quantiles[0])/1.349, 1/16383)
    np.testing.assert_allclose(stats["waveform"]["center"].numpy(), quantiles[1], atol=1e-10, rtol=1e-6)
    np.testing.assert_allclose(scale, reproduced_scale, atol=1e-10, rtol=1e-6)
    t, x, a, residual = ols_operators()
    ols = y @ a.T
    c = covariance(y)
    observed = covariance(ols)
    np.testing.assert_allclose(observed, a @ c @ a.T, rtol=1e-12, atol=1e-20)
    model = ElectronicsNoiseV2(checkpoint["initialization"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        sequence = model.autocovariance(99, include_offset=False).numpy()
        k = toeplitz(sequence, 100)
        moments = {key: float(value) for key, value in model.theoretical_stats().items()}
    noise_ols = (a @ k @ a.T)
    noise_ols = (noise_ols+noise_ols.T)/2
    latent = observed-noise_ols
    latent_eigen = np.linalg.eigvalsh(latent)
    alpha = maximum_noise_scale(observed, noise_ols)
    boundary_latent = observed-alpha*noise_ols
    covariance_boundary = x @ boundary_latent @ x.T+alpha*k
    current_total = k+moments["offset_std"]**2*np.ones_like(k)
    detrended = residual @ c @ residual.T
    empirical_sequence = empirical_stationary_estimate(y)
    empirical_k = toeplitz(empirical_sequence, 100)
    empirical_ols = a @ empirical_k @ a.T
    ls_info, ls_latent, ls_k, ls_estimate = fit_second_moment_family(c)
    bootstrap = group_bootstrap(ols, groups, noise_ols, bootstrap_resamples, seed)
    unique, group_index = np.unique(groups, return_inverse=True)
    group_counts = np.bincount(group_index)
    group_sums = np.zeros((len(unique), 2))
    np.add.at(group_sums, group_index, ols)
    group_means = group_sums/group_counts[:, None]
    within = ols-group_means[group_index]
    within_covariance_contribution = within.T @ within/(len(ols)-1)
    group_centered = group_means-ols.mean(0)
    between_covariance_contribution = group_centered.T @ (group_counts[:, None]*group_centered)/(len(ols)-1)
    np.testing.assert_allclose(within_covariance_contribution+between_covariance_contribution,
                               observed, rtol=1e-12, atol=1e-20)
    expected_required_slope = observed[0, 1]**2/latent[0, 0]
    change = np.array([[1, 380/99], [0, 479/99]])
    # Reuse already-written gate summaries; no VAL member is opened or fitted.
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if gate["noise_checkpoint_sha256"] != sha256(checkpoint_path):
        raise ValueError("Saved gate/checkpoint hash mismatch")
    saved = {}
    for name in ("TRAIN", "REAL", "generated"):
        saved[name] = summarize_2(np.asarray(gate["statistics"][name]["ols_covariance_V2"]))
    np.testing.assert_allclose(observed, saved["TRAIN"]["covariance_V2"], rtol=1e-12, atol=1e-20)
    report = dict(
        schema_version=1, status="read_only_mechanism_analysis_not_a_new_candidate",
        new_noise_training_updates=0, new_response_training_updates=0,
        execution=dict(device="CPU", torch_num_threads=torch.get_num_threads(),
                       numerical_thread_environment={key:os.environ[key] for key in
                       ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}),
        gate_changed=False, model_changed=False, gate_passed=gate["passed"],
        frozen_endpoint_update=checkpoint["update"], seed=seed,
        files=dict(checkpoint_sha256=sha256(checkpoint_path), container_sha256=sha256(container),
                   identities_sha256=sha256(identities), script_sha256=sha256(__file__),
                   electronics_v2_source_sha256=sha256(Path(__file__).resolve().parents[1]/"src/n01/electronics_v2.py"),
                   data_source_sha256=sha256(Path(__file__).resolve().parents[1]/"src/n01/data.py")),
        provenance=dict(statistical_input="existing canonical REAL TRAIN target[:,:100] only",
                        TRAIN_rows=len(y), TRAIN_original_groups=len(np.unique(groups)),
                        no_row_deletion=True, new_VAL_fit=False, validation_members_read=False,
                        saved_gate_VAL_descriptive_covariance_reused=True, fitted_scale_split="TRAIN"),
        self_checks=self_checks(seed),
        OLS_coordinate=dict(center="first100_midpoint", slope_coordinate="linspace(-1,1,100)",
                            mean_operator="ones100/100", slope_operator="t/sum(t^2)",
                            global480_transform=change,
                            global480_intercept_mean_slope_distinction=True),
        observed_TRAIN=summarize_2(observed),
        within_between_TRAIN_groups=dict(
            denominator="both contributions use N-1;sum_equals_total_covariance",
            within_covariance_contribution=summarize_2(within_covariance_contribution),
            between_covariance_contribution=summarize_2(between_covariance_contribution),
            within_cross_covariance_fraction=float(within_covariance_contribution[0,1]/observed[0,1]),
            group_event_count_range=[int(group_counts.min()),int(group_counts.max())]),
        observable_point_SD_first_last20_V=[float(np.sqrt(np.diag(c))[:20].mean()),
                                          float(np.sqrt(np.diag(c))[80:].mean())],
        frozen_stationary_K=dict(component="FIR/logscale only;independent event offset excluded from proposed latent correction",
                                 **moments,
                                 covariance_reflection_relative_error=float(np.linalg.norm(k-k[::-1, ::-1])/np.linalg.norm(k)),
                                 OLS=summarize_2(noise_ols),
                                 full100_minimum_eigenvalue_V2=float(np.linalg.eigvalsh(k)[0])),
        current_model_including_independent_offset=summarize_2(a @ current_total @ a.T),
        saved_gate_summaries=saved,
        latent_subtraction=dict(
            formula="Sigma_latent = Cov_TRAIN(mean,slope) - A K_FIR A^T",
            independent_latent_noise_assumption=True, preADC_approximation_to_quantized_TRAIN=True,
            **summarize_2(latent), PSD=bool(latent_eigen[0] >= 0),
            required_slope_latent_variance_for_PSD_V2=float(expected_required_slope),
            actual_subtracted_slope_latent_variance_V2=float(latent[1, 1]),
            maximum_stationary_slope_contribution_at_fixed_mean_noise_V2=float(observed[1,1]-expected_required_slope),
            maximum_same_shape_frozen_K_variance_scale=alpha,
            corresponding_K_amplitude_scale=np.sqrt(alpha),
            coordinate_change_cannot_repair_nonPSD=True,
            clipping_negative_eigenvalues_is_not_an_exact_match_or_identification=True),
        boundary_scale_example=dict(
            purpose="algebraic_feasibility_bound_only;not_selected_or_deployed",
            variance_scale=alpha, latent=summarize_2(boundary_latent),
            full100_covariance_relative_error=float(np.linalg.norm(c-covariance_boundary)/np.linalg.norm(c)),
            TRAIN_scaled_covariance_relative_error=float(np.linalg.norm((c-covariance_boundary)/scale[:,None]/scale[None,:])/np.linalg.norm(c/scale[:,None]/scale[None,:])),
            observable_OLS_exact_by_construction=True),
        K_estimators=dict(
            OLS_detrended_empirical_K=dict(method="R Cov_TRAIN(y) R^T",
                minimum_eigenvalue_V2=float(np.linalg.eigvalsh(detrended)[0]),
                A_K_A_T=a @ detrended @ a.T,
                limitation="OLS projection erases exactly the noise mean/slope modes needed for subtraction;cannot identify their covariance"),
            pooled_stationary_empirical_K=dict(
                method="zero_padded_autocorrelation;divisor100;across_event_centered",
                minimum_eigenvalue_V2=float(np.linalg.eigvalsh(empirical_k)[0]),
                OLS=summarize_2(empirical_ols),
                limitation="includes latent offset/drift;finite_window_lag_taper;stationary projection forces mean-slope covariance zero;not a physically identified K")),
        TRAIN_covariance_reflection_relative_error=float(np.linalg.norm(c-c[::-1, ::-1])/np.linalg.norm(c)),
        unconstrained_second_moment_family=ls_info, group_bootstrap=bootstrap,
        identifiability=dict(
            identified_observables=["100point_joint_distribution", "100point_total_OLS_covariance"],
            not_identified=["stationary_low_frequency_K_vs_drift_without_structural_prior",
                            "independent_latent_drift_noise_decomposition",
                            "480point_drift_extrapolation", "hardware_or_independent_acquisition_equivalence"],
            conditional_formula_valid_if=["known_preADC_K_on_the_same_window_and_time_basis",
                                          "latent_linear_drift_independent_of_stationary_noise",
                                          "a_positive_semidefinite_residual_latent_covariance",
                                          "jointly_compatible_full100_residual_distribution"],
            missing_evidence=["noise-only_long_baseline_records_covering_480points_or_longer",
                              "recorded_trigger_and_alignment_selection_mechanism",
                              "independent_acquisition_and_group_effect_characterization",
                              "a_prior_or_measurement_for_drift_persistence_and_K_at_low_frequency"]),
        conclusion="Stationary scalar covariance gives zero centered OLS mean-slope covariance. Frozen K plus an independent latent linear drift cannot exactly recover TRAIN OLS covariance because the subtraction is nonPSD. More steps of this stationary-offset-only structure do not remove that structural restriction. Future coupled K/drift fitting requires new registered assumptions and validation, not copying OLS SDs."
    )
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"COVARIANCE_MATRICES.npz",
        time_first100=t, operator_A=a, design_X=x, residual_operator_R=residual,
        TRAIN_covariance_V2=c, frozen_FIR_K_V2=k, detrended_TRAIN_K_V2=detrended,
        pooled_stationary_K_V2=empirical_k, TRAIN_OLS_covariance_V2=observed,
        frozen_FIR_OLS_covariance_V2=noise_ols, latent_residual_covariance_V2=latent,
        boundary_scale_covariance_V2=covariance_boundary, TRAIN_OLS_features_V=ols,
        TRAIN_original_index=ids, TRAIN_original_group=groups,
        unconstrained_latent_covariance_V2=ls_latent, unconstrained_K_V2=ls_k,
        unconstrained_total_covariance_V2=ls_estimate)
    (out/"IDENTIFIABILITY_REPORT.json").write_text(
        json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    receipt = dict(status="read_only_covariance_audit_complete",
                   report_sha256=sha256(out/"IDENTIFIABILITY_REPORT.json"),
                   matrices_sha256=sha256(out/"COVARIANCE_MATRICES.npz"),
                   script_sha256=sha256(__file__), new_training_updates=0,
                   VAL_members_read=False, TRAIN_rows=len(y))
    (out/"AUDIT_COMPLETE.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--container")
    parser.add_argument("--identities")
    parser.add_argument("--gate")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(clean(self_checks(args.seed)), indent=2))
        return
    if not all((args.checkpoint, args.container, args.identities, args.gate, args.output_dir)):
        parser.error("all checkpoint/container/identities/gate/output-dir are required")
    report = analyze(args.checkpoint, args.container, args.identities, args.gate,
                     args.output_dir, args.seed, args.bootstrap_resamples)
    print(json.dumps(clean(dict(TRAIN=report["observed_TRAIN"],
                               latent=report["latent_subtraction"],
                               bootstrap=report["group_bootstrap"],
                               unconstrained_covariance_fit=report["unconstrained_second_moment_family"],
                               output_dir=str(Path(args.output_dir).resolve()))), indent=2))


if __name__ == "__main__":
    main()