"""Artificial fixed-projection null-space audit; no training or GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from n01.losses import N01Loss, _wave_views, window_charge, weighted_wasserstein_1d

CONSTRUCTION_SEED = 91012
EVALUATION_SEED = 92012
REFRESH_SEEDS = (93012, 93013, 93014, 93015)
VIEWS = ("voltage", "shape", "difference", "curvature", "difference_smooth_5", "difference_smooth_17")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_waveforms(count=96, seed=CONSTRUCTION_SEED):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.arange(480, dtype=torch.float64)
    amplitude = torch.linspace(.08, .12, count, dtype=torch.float64)[:, None]
    decay = torch.linspace(28., 42., count, dtype=torch.float64)[:, None]
    rising = ((t - 100.) / 20.).clamp(0., 1.)
    shape = torch.where(t[None] <= 120, rising[None], torch.exp(-(t[None] - 120.) / decay))
    return amplitude * shape + .0003 * torch.randn(count, 480, generator=rng, dtype=torch.float64)


def linear_views(delta):
    result = dict(voltage=delta, shape=delta, difference=torch.diff(delta, dim=1),
                  curvature=torch.diff(delta, n=2, dim=1))
    for width in (5, 17):
        smooth = F.avg_pool1d(F.pad(delta[:, None], (width // 2, width // 2), mode="replicate"),
                              width, stride=1)[:, 0]
        result[f"difference_smooth_{width}"] = torch.diff(smooth, dim=1)
    return result


def frozen_bank(objective):
    return {name: getattr(objective, name + "_directions").detach().cpu().clone() for name in VIEWS}


def constraint_matrix(stats, directions, start=180):
    """Six linear view projection maps plus Q and tail preservation.

    Baseline is unchanged, and original peaks are before start. Therefore the
    shape map's event-dependent positive 1/peak factor does not change its kernel.
    """
    dimension = 480 - start
    basis = torch.zeros(dimension, 480, dtype=torch.float64)
    basis[:, start:] = torch.eye(dimension, dtype=torch.float64)
    views = linear_views(basis)
    blocks = []
    for name in VIEWS:
        scale = torch.as_tensor(stats["views"][name]["scale"]).double()
        blocks.append(((views[name] / scale) @ directions[name].double()).T)
    q = window_charge(basis)
    tail = basis[:, 150:400].sum(1) - 2.5 * basis[:, :100].sum(1)
    matrix = torch.cat(blocks + [q[None], tail[None]], dim=0)
    norms = matrix.norm(dim=1, keepdim=True)
    if (norms == 0).any():
        raise ValueError("Unexpected zero constraint row")
    return matrix / norms


def nullspace_perturbation(stats, bank, count=96, rms_V=.001, start=180, seed=CONSTRUCTION_SEED + 1):
    matrix = constraint_matrix(stats, bank, start)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * singular_values[0]
    rank = int((singular_values > tolerance).sum())
    null_basis = vh[rank:].T
    if null_basis.shape[1] == 0:
        raise RuntimeError("Joint fixed-projection null space is empty at the declared tolerance")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    coordinates = torch.randn(count, null_basis.shape[1], generator=rng, dtype=torch.float64)
    late = coordinates @ null_basis.T
    # RMS is over all 480 samples, not over only the changed 300 samples.
    late *= rms_V * math.sqrt(480) / late.norm(dim=1, keepdim=True)
    perturbation = torch.zeros(count, 480, dtype=torch.float64)
    perturbation[:, start:] = late
    return perturbation, dict(matrix=matrix, null_basis=null_basis, singular_values=singular_values,
                             rank=rank, nullity=null_basis.shape[1], tolerance=float(tolerance))


def standard_views(x, stats):
    views = _wave_views(x, torch.as_tensor(stats["peak_floor"]).to(x))
    return {name: (views[name] - torch.as_tensor(stats["views"][name]["center"]).to(x)) /
            torch.as_tensor(stats["views"][name]["scale"]).to(x) for name in VIEWS}


def wave_distance(x, y, stats, bank, dtype=torch.float64):
    x, y = x.to(dtype), y.to(dtype)
    xv, yv = standard_views(x, stats), standard_views(y, stats)
    terms, maximum_projection_errors = {}, {}
    for name in VIEWS:
        directions = bank[name].to(dtype)
        xp, yp = xv[name] @ directions, yv[name] @ directions
        terms[name] = float(weighted_wasserstein_1d(xp, yp))
        maximum_projection_errors[name] = float((xp - yp).abs().max())
    return dict(wave=float(np.mean(list(terms.values()))), per_view=terms,
                maximum_paired_projection_errors=maximum_projection_errors)


def coordinate_and_short_lag_detection(x, y, stats, seed=EVALUATION_SEED + 100):
    xv, yv = standard_views(x.double(), stats), standard_views(y.double(), stats)
    coordinate = {name: float(weighted_wasserstein_1d(xv[name], yv[name])) for name in VIEWS}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(2, 16, generator=rng, dtype=torch.float64)
    directions /= directions.norm(dim=0)
    short_lag = {}
    # Keep each time pair separate; do not pool different times into one cloud.
    a, b = xv["shape"], yv["shape"]
    for lag in (1, 2, 4, 8):
        ap = a[:, 180:480-lag, None]*directions[0] + a[:, 180+lag:, None]*directions[1]
        bp = b[:, 180:480-lag, None]*directions[0] + b[:, 180+lag:, None]*directions[1]
        ac = a - a.mean(0)
        bc = b - b.mean(0)
        covariance_a = (ac[:, 180:480-lag] * ac[:, 180+lag:]).mean(0)
        covariance_b = (bc[:, 180:480-lag] * bc[:, 180+lag:]).mean(0)
        short_lag[str(lag)] = dict(time_resolved_pair_projected_W1=float(weighted_wasserstein_1d(ap.flatten(1), bp.flatten(1))),
                                  absolute_covariance_change_mean=float((covariance_a - covariance_b).abs().mean()),
                                  pair_directions=16, pair_times=480-lag-180)
    return dict(coordinate_W1=coordinate, mean_coordinate_W1=float(np.mean(list(coordinate.values()))),
                short_lag_joint=short_lag, seed=seed)


def preservation(x, y):
    def psd(z):
        tail=z[:,150:400].sum(1)-2.5*z[:,:100].sum(1)
        return tail/window_charge(z)
    baseline_error=float((x[:,:100]-y[:,:100]).abs().max())
    xpeak=(x-x[:,:100].mean(1,keepdim=True)).amax(1)
    ypeak=(y-y[:,:100].mean(1,keepdim=True)).amax(1)
    return dict(baseline_max_abs_error_V=baseline_error,
                peak_max_abs_error_V=float((xpeak-ypeak).abs().max()),
                charge_max_abs_error_Vns=float((window_charge(x)-window_charge(y)).abs().max()),
                PSD_max_abs_error=float((psd(x)-psd(y)).abs().max()),
                peak_argmax_unchanged=bool(torch.equal(x.argmax(1),y.argmax(1))),
                all_original_peaks_before180=bool((x.argmax(1)<180).all()))


def fir_bounds(taps=65):
    """Bounds for stationary iid-white input filtered by the raw FIR alone."""
    a=.5/taps
    # "first order" here means truncating the tap-perturbation expansion.
    first_order=a/(1-a)
    conservative=((1+a)*a+(taps-2)*a*a)/(1-a)**2
    # The exact supremum is attained as h0 -> 1-a, all later taps -> +a.
    # Its negative counterpart uses alternating signs. Normalization cancels.
    exact=((1-a)*a+(taps-2)*a*a)/((1-a)**2+(taps-1)*a*a)
    return dict(taps=taps, per_tap_perturbation_bound=a,
                first_order_linearized_abs_rho1_bound=first_order,
                exact_stationary_white_FIR_abs_rho1_supremum=exact,
                conservative_triangle_abs_rho1_bound=conservative,
                first_order_bound_is_not_full_exact_bound=True,
                excludes=["baseline demeaning/detrending","random offset","random drift","ADC","finite-sample ACF estimator"],
                interpretation="Do not apply this stationary pure-FIR bound to a processed observed baseline.")


def kernel_rho1(kernel):
    return float(((kernel[:-1]*kernel[1:]).sum()/kernel.square().sum()).detach())


def run_audit(checkpoint_path, output_dir, count=96, rms_V=.001):
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=2:
        torch.set_num_interop_threads(2)
    checkpoint_path,output_dir=Path(checkpoint_path),Path(output_dir)
    if os.name=="nt" and output_dir.drive.upper() not in ("E:","D:"):
        raise ValueError("Audit outputs must remain on E:/D:")
    if count!=96 or rms_V!=.001:
        raise ValueError("This registered audit uses exactly96 rows and1mV full-window RMS")
    output_dir.mkdir(parents=True,exist_ok=True)
    source_paths=[ROOT/"src/n01"/name for name in ("losses.py","model.py","train.py")]
    before={str(path):sha256(path) for path in source_paths}
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    stats=checkpoint["loss_statistics"]
    training_seed=checkpoint.get("config",{}).get("projection_seed")
    if training_seed in (EVALUATION_SEED,)+REFRESH_SEEDS:
        raise ValueError("Independent evaluation/refresh seed coincides with recorded training seed")
    objective=N01Loss(stats,seed=0)
    objective.load_state_dict(checkpoint["loss_state"],strict=True)
    bank=frozen_bank(objective)
    original=synthetic_waveforms(count)
    delta,geometry=nullspace_perturbation(stats,bank,count,rms_V)
    altered=original+delta
    if not preservation(original,altered)["peak_argmax_unchanged"]:
        raise RuntimeError("Declared peak-preservation regime violated")
    directions={"frozen_training":bank}
    independent=frozen_bank(N01Loss(stats,seed=EVALUATION_SEED))
    directions["independent_evaluation"]=independent
    refreshed=[]
    for seed in REFRESH_SEEDS:
        fresh=frozen_bank(N01Loss(stats,seed=seed))
        directions[f"refresh_{seed}"]=fresh
        refreshed.append(dict(seed=seed,**wave_distance(original,altered,stats,fresh)))
    with torch.no_grad():
        actual64=objective.components(dict(waveform=altered),original)
        actual32=objective.components(dict(waveform=altered.float()),original.float())
    detection=coordinate_and_short_lag_detection(original,altered,stats)
    bounds=fir_bounds()
    a=.5/65
    limiting=torch.full((65,),a,dtype=torch.float64);limiting[0]=1-a
    rng=torch.Generator(device="cpu").manual_seed(CONSTRUCTION_SEED+2)
    raw=6*torch.randn(1000,65,generator=rng,dtype=torch.float64)
    kernels=.5*torch.tanh(raw)/65;kernels[:,0]+=1.
    random_rho=(kernels[:,:-1]*kernels[:,1:]).sum(1)/kernels.square().sum(1)
    bounds["limiting_kernel_numeric_rho1"]=kernel_rho1(limiting)
    bounds["random1000_raw_max_abs_rho1"]=float(random_rho.abs().max())
    key="electronics.filter.raw"
    if key in checkpoint["model"]:
        raw=checkpoint["model"][key].double()
        kernel=.5*torch.tanh(raw)/65;kernel[0]+=1.
        bounds["frozen_learned_kernel_stationary_FIR_rho1"]=kernel_rho1(kernel)
    np.savez_compressed(output_dir/"artificial_counterexample.npz",
        original=original.numpy(),altered=altered.numpy(),perturbation=delta.numpy(),
        constraint_matrix=geometry["matrix"].numpy(),null_basis=geometry["null_basis"].numpy(),
        singular_values=geometry["singular_values"].numpy())
    torch.save(dict(seed_protocol=dict(construction=CONSTRUCTION_SEED,training=training_seed,
        independent_evaluation=EVALUATION_SEED,refresh=list(REFRESH_SEEDS)),directions=directions),
        output_dir/"independent_direction_banks.pt")
    after={str(path):sha256(path) for path in source_paths}
    if before!=after:
        raise RuntimeError("Immutable scientific source hashes changed during the audit")
    report=dict(schema_version=1,audit="fixed_projection_joint_nullspace_artificial",
        checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),device="cpu",
        rows=count,length=480,change_support=[180,480],construction_seed=CONSTRUCTION_SEED,
        perturbation_seed=CONSTRUCTION_SEED+1,FIR_random_check_seed=CONSTRUCTION_SEED+2,
        script_sha256=sha256(__file__),
        training_projection_seed=training_seed,independent_evaluation_seed=EVALUATION_SEED,
        refreshed_direction_seeds=list(REFRESH_SEEDS),directions_per_view=32,views=list(VIEWS),
        target_full_window_RMS_V=rms_V,actual_full_window_RMS_min_V=float(delta.square().mean(1).sqrt().min()),
        actual_full_window_RMS_max_V=float(delta.square().mean(1).sqrt().max()),
        maximum_absolute_perturbation_V=float(delta.abs().max()),
        constraint_rows=geometry["matrix"].shape[0],free_samples=300,rank=geometry["rank"],
        nullity=geometry["nullity"],SVD_rank_tolerance=geometry["tolerance"],
        maximum_constraint_residual=float((geometry["matrix"]@delta[:,180:].T).abs().max()),
        float64_preservation=preservation(original,altered),float32_preservation=preservation(original.float(),altered.float()),
        frozen_training_float64_reference=wave_distance(original,altered,stats,bank),
        frozen_training_float32_reference=wave_distance(original,altered,stats,bank,torch.float32),
        current_implementation_with_float64_input={k:float(actual64[k]) for k in ("wave","QPSD","noise")},
        current_implementation_with_float32_input={k:float(actual32[k]) for k in ("wave","QPSD","noise")},
        current_standardization_forces_float32=True,
        independent_frozen_evaluation=wave_distance(original,altered,stats,independent),
        independent_frozen_evaluation_float32=wave_distance(original,altered,stats,independent,torch.float32),
        refreshed_bank_comparisons=refreshed,mean_refreshed_wave_distance=float(np.mean([entry["wave"] for entry in refreshed])),
        coordinate_and_short_lag_detection=detection,FIR_rho1_analysis=bounds,
        exclusions=["ADC","adversarial discriminator","generator regularization","generator admissibility constraints"],
        not_demonstrated=["current model exploiting the null space","physical plausibility of the artificial perturbations",
                         "failure of the entire combined objective","universal detection by one new frozen bank"],
        original_claim=dict(wave_loss=2.42e-7,maximum_perturbation_V=.00433,status="comparison_claim_not_forced_target"),
        immutable_source_sha256_before=before,immutable_source_sha256_after=after,
        authority="new_user_authorized_diagnostic;original_proposal_allows_projection_distribution_loss_without_specifying_refresh",
        proposal_difference="This isolates a fixed finite projection design;comparison banks and coordinate/lag diagnostics are new authorized audit choices.",
        artifacts={path.name:dict(bytes=path.stat().st_size,sha256=sha256(path))
                   for path in output_dir.iterdir() if path.is_file() and path.name!="PROJECTION_AUDIT.json"})
    (output_dir/"PROJECTION_AUDIT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps({key:report[key] for key in ("rank","nullity","maximum_absolute_perturbation_V",
        "current_implementation_with_float64_input","independent_frozen_evaluation","mean_refreshed_wave_distance")}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run_audit(args.checkpoint,args.output)


if __name__=="__main__":
    main()
