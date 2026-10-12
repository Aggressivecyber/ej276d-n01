"""Reproduce N01 scientific figures from frozen endpoint artifacts only.

No model imports, fitting, training, or response generation. The remote payload
manifest is checked before every artifact is decoded. Original files are read
only. Derived arrays/statistics, plot caches and figures stay in analysis/figures.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLCONFIGDIR"] = str(ANALYSIS / ".mplconfig")
os.environ["TEMP"] = os.environ["TMP"] = "E:/GAN/runtime_tmp"
sys.dont_write_bytecode = True
# Import the existing NumPy before adding plotting-only dependencies.
import numpy as np
sys.path.insert(0, "E:/GAN/research/N01/runtime/plotting_deps")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, FormatStrFormatter
import hashlib
import json

ARMS = ("full", "deterministic", "simple", "full_uniform")
SERIES = ("REAL",) + ARMS
LABEL = {"REAL": "REAL", "full": "Full", "deterministic": "Deterministic U-Net",
         "simple": "Simple", "full_uniform": "Full / uniform population"}
COLOR = {"REAL": "#20242c", "full": "#0072B2", "deterministic": "#D55E00",
         "simple": "#CC79A7", "full_uniform": "#009E73"}
STYLE = {"REAL": "-", "full": "-", "deterministic": "--", "simple": ":", "full_uniform": "-."}
MARKER = {"REAL": "D", "full": "o", "deterministic": "s", "simple": "^", "full_uniform": "v"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
ANALYSIS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "train"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


manifest = read_json(ROOT / "REMOTE_MANIFEST.json")
expected = {entry["relative"]: entry["sha256"] for entry in manifest["files"]}
verified = []


def frozen(relative):
    path = TRAIN / relative
    digest = sha(path)
    if relative not in expected or digest != expected[relative]:
        raise ValueError(f"Frozen payload hash mismatch: {relative}")
    verified.append(dict(file="train/" + relative, sha256=digest, remote_manifest_match=True))
    return path


comparison = read_json(frozen("ARM_COMPARISON.json"))
reports, populations, repeats = {}, {}, {}
for arm in ARMS:
    prefix = arm + "/evaluation/"
    reports[arm] = read_json(frozen(prefix + "endpoint_report.json"))
    for name, destination in (("endpoint_populations.npz", populations),
                              ("repeated_parent_responses.npz", repeats)):
        path = frozen(prefix + name)
        if sha(path) != reports[arm]["artifacts"][name]["sha256"]:
            raise ValueError("Report artifact SHA does not match manifest")
        with np.load(path, allow_pickle=False) as z:
            destination[arm] = {key: z[key].copy() for key in z.files}
reference = populations["full"]
for arm in ARMS:
    for key in ("REAL_waveform", "REAL_Q", "REAL_PSD", "source_parent_ids", "real_original_ids"):
        if not np.array_equal(reference[key], populations[arm][key]):
            raise ValueError(f"Endpoint identity/REAL mismatch: {arm}/{key}")
if not np.array_equal(repeats["full"]["source_parent_ids"], repeats["full_uniform"]["source_parent_ids"]):
    raise ValueError("Repeated-parent order differs between full and full_uniform")


def vector_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    result = dict(rows=len(values), finite_rows=int(finite.sum()), invalid_rows=int((~finite).sum()))
    if finite.any():
        x = values[finite]
        result.update(mean=float(x.mean()), std_population=float(x.std(ddof=0)),
                      std_sample=float(x.std(ddof=1)) if len(x) > 1 else None,
                      minimum=float(x.min()), maximum=float(x.max()),
                      quantiles_05_50_95=np.quantile(x, [.05, .5, .95]).tolist())
    return result


def style(series):
    return dict(color=COLOR[series], linestyle=STYLE[series], linewidth=2.0 if series == "REAL" else 1.5)


def decorate(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=.17, linewidth=.7)


def legend_handles(series=SERIES):
    return [Line2D([0], [0], label=LABEL[k], **style(k)) for k in series]


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / (name + "." + suffix), dpi=220 if suffix == "png" else None)
    plt.close(fig)


def hist(ax, values, bins, series):
    density, edges = np.histogram(values, bins=bins, density=True)
    ax.stairs(density, edges, label=LABEL[series], **style(series))


metrics = dict(schema_version=1, experiment_id="N01", analysis_scope="frozen_fixed_endpoints_only",
               no_new_responses=True, no_new_fits=True, no_new_acceptance_thresholds=True,
               manifest_sha256=sha(ROOT / "REMOTE_MANIFEST.json"), inputs=verified,
               independent_acquisition_count=1, source_parents=512,
               marginal_draws_per_parent=4, marginal_generated_rows=2048,
               real_original_parents=599, repeated_parents=128, repeats_per_parent=64,
               uncertainty_scope="finite original groups and model draws, not independent acquisitions",
               arm_metrics={}, conditional_components={}, repeat_statistics={}, population_sensitivity={})
real_wave = reference["REAL_waveform"]
real_mean_baseline = real_wave[:, :100].mean(1)
real_sigma_mean = real_mean_baseline.std(ddof=0)
metrics["REAL"] = dict(scalars=reports["full"]["REAL_population"]["scalars"],
                       baseline_event_mean_V=vector_stats(real_mean_baseline))
for arm in ARMS:
    report, pop, rep = reports[arm], populations[arm], repeats[arm]
    baseline_mean = pop["generated_waveform"][:, :100].mean(1)
    classifier = report["independent_domain_classifier"]
    weights = pop["source_parent_comparison_weights"]
    saturation = rep["saturation_fraction_per_event"]
    metrics["arm_metrics"][arm] = dict(
        checkpoint_sha256=report["checkpoint_sha256"],
        raw_distances=report["raw_population"]["distribution_distances"]["components"],
        comparison_weighted_distances=report["comparison_weighted_population"]["distribution_distances"]["components"],
        raw_AUC=classifier["raw_ROC_AUC"], weighted_AUC=classifier["comparison_weighted_ROC_AUC"],
        AUC_uncertainty=classifier["uncertainty"], raw_scalars=report["raw_population"]["scalars"],
        comparison_weighted_scalars=report["comparison_weighted_population"]["scalars"],
        baseline_event_mean_V=vector_stats(baseline_mean),
        baseline_event_mean_std_ratio_to_REAL=float(baseline_mean.std(ddof=0) / real_sigma_mean),
        baseline_RMS_mean_difference_mV=float(1000 * (pop["generated_baseline_rms_V"].mean() - reference["REAL_baseline_rms_V"].mean())),
        peak_time_mean_difference_ns=float(pop["generated_peak_time_ns"].mean() - reference["REAL_peak_time_ns"].mean()),
        half_rise_mean_difference_ns=float(pop["generated_half_rise_ns"].mean() - reference["REAL_half_rise_ns"].mean()),
        validation_comparison_weight_range=[float(weights.min()), float(weights.max())],
        validation_comparison_weight_effective_parents=float(weights.sum()**2 / np.square(weights).sum()),
        marginal_saturation_fraction=report["generated_saturation_fraction_mean"],
        repeated_saturated_draws=int((saturation > 0).sum()),
        repeated_parents_with_saturation=int((saturation > 0).any(1).sum()),
        repeated_total_draws=int(saturation.size),
        repeated_saturation_max_fraction=float(saturation.max()),
        charge_identity_max_abs_Vns=report["charge_identity_max_abs_Vns"])
    metrics["repeat_statistics"][arm] = {
        key: vector_stats(rep[key]) for key in ("variance_Q", "variance_PSD", "covariance_Q_PSD", "mean_Q", "mean_PSD")}
    metrics["repeat_statistics"][arm]["conditional_SD_Q_Vns"] = vector_stats(np.sqrt(rep["variance_Q"]))
    metrics["repeat_statistics"][arm]["conditional_SD_PSD"] = vector_stats(np.sqrt(rep["variance_PSD"]))
    metrics["conditional_components"][arm] = []
    for row in report["conditional_PSD_peaks_and_original_widths"]:
        extracted = {key: row[key] for key in ("bin", "fake_rows", "real_rows", "fake_distinct_parents", "minimum_rows_per_fit")}
        for kind in ("raw", "comparison_weighted", "REAL"):
            fit = row[kind]
            extracted[kind] = {key: fit[key] for key in ("status", "positions", "original_PSD_std", "mixture_fraction", "converged", "BIC_one_minus_two", "ambiguity") if key in fit}
        metrics["conditional_components"][arm].append(extracted)
metrics["generated_baselines_identical_across_arms"] = all(
    np.array_equal(reference["generated_waveform"][:, :100], populations[a]["generated_waveform"][:, :100]) for a in ARMS)
metrics["evidence_boundaries"] = [
    "Projection distances are descriptive; low values do not establish successful response recovery.",
    "AUC uses a separately trained fixed-budget classifier; its intervals cover finite original groups, including only five REAL test groups.",
    "Two-Gaussian components are descriptive, not true particle labels; one-component BIC preference and non-convergence remain visible.",
    "Repeated model draws do not identify true event-level physical variance without repeated REAL truth.",
    "Same-seed full versus uniform arms were separately retrained; close population means do not prove response stability.",
    "REAL-REAL control uses 349/250 rows, versus generated/REAL 2048/599, so it is not a directly applicable pass threshold."]

# 1. Complete raw population and shared-electronics residuals.
fig, axes = plt.subplots(4, 2, figsize=(12.6, 13.1))
fig.subplots_adjust(left=.085, right=.975, top=.91, bottom=.095, hspace=.46, wspace=.28)
fig.suptitle("N01 fixed endpoints: complete raw populations", y=.995, fontsize=16, fontweight="bold")
fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.5, .966), ncol=5, frameon=False)
q_all = np.concatenate([reference["REAL_Q"]] + [populations[a]["generated_Q"] for a in ARMS])
psd_all = np.concatenate([reference["REAL_PSD"]] + [populations[a]["generated_PSD"] for a in ARMS])
q_bins = np.linspace(0, q_all.max() * 1.025, 46)
psd_bins = np.linspace(psd_all.min() - .015, psd_all.max() + .015, 58)
for series in SERIES:
    pop = reference if series == "REAL" else populations[series]
    prefix = "REAL" if series == "REAL" else "generated"
    mean_prefix = "REAL" if series == "REAL" else "raw"
    hist(axes[0, 0], pop[prefix + "_Q"], q_bins, series)
    hist(axes[0, 1], pop[prefix + "_PSD"], psd_bins, series)
    axes[1, 0].plot(np.arange(480), pop[mean_prefix + "_voltage_mean"] * 1000, **style(series))
    axes[1, 1].plot(np.arange(480), pop[mean_prefix + "_normalized_shape_mean"], **style(series))
    hist(axes[2, 0], pop[prefix + "_baseline_rms_V"] * 1000, np.linspace(.3, 1.8, 52), series)
    mean = pop[prefix + "_waveform"][:, :100].mean(1) * 1000
    # All observed values are retained in these common fixed display bins.
    means_all = np.concatenate([real_mean_baseline] + [populations[a]["generated_waveform"][:, :100].mean(1) for a in ARMS]) * 1000
    hist(axes[2, 1], mean, np.linspace(means_all.min() - .08, means_all.max() + .08, 58), series)
    noise = reports["full"]["REAL_population"]["noise"] if series == "REAL" else reports[series]["raw_population"]["noise"]
    axes[3, 0].plot(noise["ACF_lags"], noise["ACF_mean"], **style(series))
    power = pop[mean_prefix + "_baseline_fft_power_mean_V2"][1:]
    frequency = pop[mean_prefix + "_baseline_fft_frequency_GHz"][1:] * 1000
    axes[3, 1].plot(frequency, power / power.sum(), **style(series))
for ax, title, xlab, ylab in (
    (axes[0,0], "a  Window charge", "Q (V ns)", "Probability density"),
    (axes[0,1], "b  PSD, including observed tails", "PSD = Qtail / Q", "Probability density"),
    (axes[1,0], "c  Full mean voltage", "Aligned time (ns)", "Voltage (mV)"),
    (axes[1,1], "d  Full peak-normalized mean shape", "Aligned time (ns)", "Normalized voltage"),
    (axes[2,0], "e  Baseline RMS per response", "Baseline RMS (mV)", "Probability density (mV$^{-1}$)"),
    (axes[2,1], "f  Baseline mean per response", "Baseline mean (mV)", "Probability density (mV$^{-1}$)"),
    (axes[3,0], "g  Mean baseline autocorrelation", "Lag (ns)", "ACF"),
    (axes[3,1], "h  Normalized mean baseline FFT power", "Frequency (MHz)", "Power fraction per frequency bin")):
    decorate(ax, title, xlab, ylab)
for ax in axes[1]:
    ax.set_xlim(0, 479)
    ax.axvline(120, color="#777777", linestyle=":", alpha=.45)
axes[2,0].text(.98, .96, "Mean RMS: Full 0.6303 / REAL 0.6380 mV", transform=axes[2,0].transAxes,
               ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
ratio = metrics["arm_metrics"]["full"]["baseline_event_mean_std_ratio_to_REAL"]
axes[2,1].text(.98, .96, f"Across-event SD (ddof=0): {ratio:.2f}x REAL", transform=axes[2,1].transAxes,
               ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
fig.text(.085, .012, "Raw: 512 SOURCE parents x 4 draws = 2048 responses; 599 REAL parents. Shared generated baselines are identical across all four arms.\n"
         "Baseline-mean SD uses ddof=0; one historical REAL acquisition. Frozen waveform/shape summaries and noise features are used without refitting.", fontsize=9.5)
save(fig, "01_overall_endpoint")

# 2. Plot the original frozen two-component fits, preserving existing ambiguity.
fig, axes = plt.subplots(3, 2, figsize=(12.6, 11.7))
fig.subplots_adjust(left=.085, right=.975, top=.9, bottom=.135, hspace=.48, wspace=.28)
fig.suptitle("Conditional PSD: frozen component positions and original widths", y=.995, fontsize=16, fontweight="bold")
fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.5, .965), ncol=5, frameon=False)
flag_counts = {}
for series in SERIES:
    rows = reports["full" if series == "REAL" else series]["conditional_PSD_peaks_and_original_widths"]
    fits = [r["REAL" if series == "REAL" else "raw"] for r in rows]
    xs = np.arange(1, 9)
    flag_counts[series] = dict(one_Gaussian_BIC_preferred=0, not_all_initializations_converged=0, sparse_bins=0)
    for component in (0, 1):
        for row_index, field in ((0, "positions"), (1, "original_PSD_std")):
            ax = axes[row_index, component]
            ys = np.array([f.get(field, [np.nan, np.nan])[component] for f in fits])
            ax.plot(xs, ys, **style(series), alpha=.8)
            for x, y, fit in zip(xs, ys, fits):
                if not np.isfinite(y):
                    continue
                ambiguous = fit.get("ambiguity", {}).get("one_component_BIC_preferred", False) or not fit.get("ambiguity", {}).get("all_initializations_converged", fit.get("converged", False))
                ax.plot(x, y, marker=MARKER[series], linestyle="none", markersize=5,
                        color=COLOR[series], markerfacecolor="white" if ambiguous else COLOR[series], markeredgewidth=1.1)
                if field == "positions" and "initializations" in fit:
                    values = [f[field][component] for f in fit["initializations"]]
                    ax.vlines(x, min(values), max(values), color=COLOR[series], linewidth=2.8, alpha=.6)
    axes[2,0].plot(xs, [f.get("BIC_one_minus_two", np.nan) for f in fits], marker=MARKER[series], markersize=4, **style(series))
    counts = [r["real_rows"] if series == "REAL" else r["fake_rows"] for r in rows]
    axes[2,1].plot(xs, counts, marker=MARKER[series], markersize=4, **style(series))
    for fit in fits:
        flag_counts[series]["one_Gaussian_BIC_preferred"] += int(fit.get("ambiguity", {}).get("one_component_BIC_preferred", False))
        flag_counts[series]["not_all_initializations_converged"] += int(not fit.get("ambiguity", {}).get("all_initializations_converged", fit.get("converged", False)))
        flag_counts[series]["sparse_bins"] += int(fit.get("status") != "descriptive_two_Gaussian_fit")
for component, name in enumerate(("Lower PSD", "Upper PSD")):
    decorate(axes[0,component], f"{'ab'[component]}  {name} component position", "Frozen TRAIN Q bin (low to high)", "PSD position")
    decorate(axes[1,component], f"{'cd'[component]}  {name} component original width", "Frozen TRAIN Q bin (low to high)", "Original PSD SD")
decorate(axes[2,0], "e  Recorded model-selection ambiguity", "Frozen TRAIN Q bin (low to high)", "BIC(one) - BIC(two)")
axes[2,0].axhline(0, color="#555555", linestyle=":", linewidth=1)
axes[2,0].text(.02, .97, "Positive favors two; negative favors one", transform=axes[2,0].transAxes, va="top", fontsize=9)
decorate(axes[2,1], "f  Existing fit support", "Frozen TRAIN Q bin (low to high)", "Response rows in bin")
axes[2,1].axhline(16, color="#555555", linestyle=":", linewidth=1)
for ax in axes.flat:
    ax.set_xticks(np.arange(1,9))
    ax.set_xlim(.7, 8.3)
fig.text(.085, .023, "Open markers: recorded one-Gaussian BIC preference or incomplete convergence across initializations; components are not particle labels.\n"
         "Widths are frozen original PSD SDs. Counts are response rows (generated draws are clustered by parent); the original minimum fit support is 16.\n"
         "No peaks were refitted here and no new pass/fail threshold was introduced.", fontsize=9.5)
metrics["conditional_flag_counts"] = flag_counts
save(fig, "02_conditional_psd_components")

# 3. Same-seed, separately trained population sensitivity on the same 128 parents.
f, u = repeats["full"], repeats["full_uniform"]
delta_q = f["mean_Q"] - u["mean_Q"]
delta_psd = f["mean_PSD"] - u["mean_PSD"]
paired_q_se = (f["Q"] - u["Q"]).std(axis=1, ddof=1) / np.sqrt(64)
paired_psd_se = (f["PSD"] - u["PSD"]).std(axis=1, ddof=1) / np.sqrt(64)
mean_wave_rms = np.sqrt(np.square(f["waveform"].mean(1) - u["waveform"].mean(1)).mean(1))
metrics["population_sensitivity"] = dict(
    frozen_report=comparison["population_response_stability"],
    parent_mean_Q_difference_Vns=vector_stats(delta_q), parent_mean_PSD_difference=vector_stats(delta_psd),
    parent_mean_voltage_difference_RMS_V=vector_stats(mean_wave_rms),
    parent_mean_Q_difference_paired_sampling_SE=vector_stats(paired_q_se),
    parent_mean_PSD_difference_paired_sampling_SE=vector_stats(paired_psd_se),
    paired_by_saved_parent_id_and_draw_index=True,
    matching_evaluation_seed=reports["full"]["evaluation_seed"]==reports["full_uniform"]["evaluation_seed"],
    latent_noise_tensors_not_in_repeat_archive=True)
fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.7))
fig.subplots_adjust(left=.07, right=.978, top=.885, bottom=.145, hspace=.4, wspace=.33)
fig.suptitle("Full response sensitivity to the population-weight setting", y=.994, fontsize=16, fontweight="bold")
fig.text(.5, .947, "128 common SOURCE parents x 64 saved draws; same training seed, separately trained full and uniform arms", ha="center", fontsize=10.5)
for ax, field, title, unit in ((axes[0,0], "variance_Q", "a  Conditional charge SD", "V ns"),
                              (axes[0,1], "variance_PSD", "b  Conditional PSD SD", "PSD")):
    x, y = np.sqrt(u[field]), np.sqrt(f[field])
    ax.scatter(x, y, s=17, color=COLOR["full"], alpha=.6, edgecolors="none")
    lo, hi = min(x.min(), y.min())*.85, max(x.max(), y.max())*1.18
    ax.plot([lo,hi], [lo,hi], color=COLOR["full_uniform"], linestyle="--", linewidth=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ticks = [.2, .5, 1., 2.] if field == "variance_Q" else [.02, .04, .08, .12]
    ticks = [value for value in ticks if lo < value < hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    decorate(ax, title, "Uniform arm SD (" + unit + ")", "Full arm SD (" + unit + ")")
for ax, delta, se, title, ylabel in (
    (axes[0,2], delta_q, paired_q_se, "c  Paired parent-mean charge change", "Full - uniform mean Q (V ns)"),
    (axes[1,0], delta_psd, paired_psd_se, "d  Paired parent-mean PSD change", "Full - uniform mean PSD")):
    ax.errorbar(f["mean_Q"], delta, yerr=se, fmt="o", color=COLOR["full"], markersize=3,
                alpha=.65, linewidth=.8, capsize=0)
    ax.axhline(0, color=COLOR["full_uniform"], linestyle="--", linewidth=1.2)
    decorate(ax, title, "Full parent mean Q (V ns)", ylabel)
    ax.text(.03, .97, "Error bars: paired model-draw SE", transform=ax.transAxes, va="top", fontsize=8.5)
weights = reference["source_parent_comparison_weights"]
parent_qbar = np.zeros(512)
for i in range(512):
    parent_qbar[i] = reference["generated_qbar"][reference["generated_parent_row"] == i].mean()
axes[1,1].scatter(parent_qbar, weights, s=16, color=COLOR["full"], alpha=.55, edgecolors="none")
axes[1,1].axhline(1, color=COLOR["full_uniform"], linestyle="--", linewidth=1.5)
decorate(axes[1,1], "e  Saved validation comparison weights", "Full mean signal charge Qbar (V ns)", "Separate comparison weight")
axes[1,1].text(.03, .97, f"Full range {weights.min():.5f} to {weights.max():.5f}\nUniform = 1", transform=axes[1,1].transAxes, va="top", fontsize=9)
axes[1,2].scatter(f["mean_Q"], mean_wave_rms * 1000, s=17, color=COLOR["full"], alpha=.65, edgecolors="none")
decorate(axes[1,2], "f  Parent-mean full-waveform change", "Full parent mean Q (V ns)", "RMS(full - uniform mean V), mV")
axes[1,2].text(.03, .97, f"Mean {mean_wave_rms.mean()*1000:.4f} mV\nMax {mean_wave_rms.max()*1000:.4f} mV", transform=axes[1,2].transAxes, va="top", fontsize=9)
count = metrics["arm_metrics"]["full"]["repeated_saturated_draws"]
fig.text(.07, .03, "Conditional SDs and SEs describe model draws, not repeated physical measurements. Small overall Q/PSD mean differences do not prove response stability.\n"
         f"Full: {count}/8192 repeated draws have nonzero pre-ADC saturation (the 2048-draw marginal snapshot has none). All saved saturation values are retained.", fontsize=9.5)
save(fig, "03_population_sensitivity")

# 4. Descriptor distances and the independent classifier show distinct evidence.
fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
fig.subplots_adjust(left=.07, right=.97, top=.835, bottom=.23, wspace=.26)
fig.suptitle("Endpoint descriptor distances and independent domain separability", y=.985, fontsize=16, fontweight="bold")
x = np.arange(3); distance_keys = ("wave", "QPSD", "noise")
bar_width = .17
for index, arm in enumerate(ARMS):
    m = metrics["arm_metrics"][arm]
    location = x + (index - 1.5)*bar_width
    raw = [m["raw_distances"][key] for key in distance_keys]
    weighted = [m["comparison_weighted_distances"][key] for key in distance_keys]
    axes[0].bar(location, raw, width=bar_width*.91, color=COLOR[arm], alpha=.83, label=LABEL[arm])
    axes[0].plot(location, weighted, marker="o", markerfacecolor="white", markeredgecolor="#222222",
                 color="none", markersize=4, label=None)
control = reports["full"]["controls"]["REAL_REAL"]["distances"]["components"]
axes[0].plot(x, [control[key] for key in distance_keys], "D", color=COLOR["REAL"], markersize=7,
             linestyle="none", label="REAL-REAL control (349/250)")
axes[0].set_xticks(x, ["Waveform", "Joint Q-PSD", "Baseline noise"])
decorate(axes[0], "a  Frozen TRAIN-scale distribution distances", ylabel="Descriptor distance")
fig.legend(handles=[Patch(facecolor=COLOR[a], alpha=.83, label=LABEL[a]) for a in ARMS] +
           [Line2D([0],[0],marker="D",color=COLOR["REAL"],linestyle="none",label="REAL-REAL control (349/250)")],
           loc="upper center",bbox_to_anchor=(.5,.937),ncol=5,frameon=False,fontsize=9)
for i, arm in enumerate(ARMS):
    m = metrics["arm_metrics"][arm]
    auc = m["raw_AUC"]
    interval = m["AUC_uncertainty"]["raw"]["percentile_95_interval"]
    axes[1].errorbar(i, auc, yerr=[[auc-interval[0]], [interval[1]-auc]], fmt="o", color=COLOR[arm],
                    capsize=5, markersize=7, linewidth=1.8)
    axes[1].plot(i+.10, m["weighted_AUC"], "o", color=COLOR[arm], markerfacecolor="white", markersize=5)
    axes[1].text(i, auc-.055, f"{auc:.5f}", ha="center", fontsize=10)
axes[1].set_xticks(np.arange(4), ["Full", "Deterministic", "Simple", "Full / uniform"])
axes[1].set_ylim(.48, 1.025)
axes[1].axhline(.5, color="#777777", linestyle=":", linewidth=1.2)
axes[1].text(.03, .04, "0.5: chance discrimination", transform=axes[1].transAxes, fontsize=9)
decorate(axes[1], "b  Independent fixed-budget classifier", ylabel="ROC AUC (generated vs REAL)")
fig.text(.07, .037, "Solid bars/points: raw populations; open dots: saved comparison weights. AUC error bars: frozen 95% original-group/parent bootstrap intervals.\n"
         "Distance control sample sizes differ: REAL-REAL 349/250 versus generated/REAL 2048/599; its values are descriptive, not direct pass thresholds.\n"
         "AUC test: 205 SOURCE parents x 4 = 820 responses / 250 REAL rows (5 groups) from one acquisition. Low projected distances do not establish response recovery.", fontsize=9.5)
metrics["REAL_REAL_control"] = dict(rows=[349,250], distances=control, direct_pass_threshold=False)
save(fig, "04_distance_and_auc")

metrics["derived_comparisons"] = {
    "full_minus_deterministic_wave": metrics["arm_metrics"]["full"]["raw_distances"]["wave"] - metrics["arm_metrics"]["deterministic"]["raw_distances"]["wave"],
    "full_QPSD_relative_reduction_vs_deterministic": 1-metrics["arm_metrics"]["full"]["raw_distances"]["QPSD"]/metrics["arm_metrics"]["deterministic"]["raw_distances"]["QPSD"],
    "full_minus_deterministic_AUC": metrics["arm_metrics"]["full"]["raw_AUC"]-metrics["arm_metrics"]["deterministic"]["raw_AUC"],
    "full_mean_conditional_variance_Q_ratio_vs_deterministic": repeats["full"]["variance_Q"].mean()/repeats["deterministic"]["variance_Q"].mean(),
    "full_mean_conditional_variance_PSD_ratio_vs_deterministic": repeats["full"]["variance_PSD"].mean()/repeats["deterministic"]["variance_PSD"].mean()}
metrics["software"] = dict(numpy=np.__version__, matplotlib=matplotlib.__version__, python=sys.version,
                           plotting_script_sha256=sha(__file__))
metrics["figures"] = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha(p)}
                      for p in sorted(FIGURES.glob("0*.*"))]
(ANALYSIS / "METRICS.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
print(json.dumps(dict(verified_input_files=len(verified), generated_figures=len(metrics["figures"]),
                     baseline_mean_std_ratio=ratio, full_mean_wave_change_mV=float(mean_wave_rms.mean()*1000),
                     conditional_flags=flag_counts), ensure_ascii=False))