"""Frozen N01 random-electronics-slope mechanism diagnostic.

Only inspect mode is automatic. Paired mode requires an explicit --mode paired.
No optimizers, retraining, population refit, or extra source parents are used.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TEMP"] = os.environ["TMP"] = "E:/GAN/runtime_tmp"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import time
import traceback
from typing import Mapping
import numpy as np
import torch
from n01.model import N01Response

SEED = 20261034
STREAM_BATCH = 128
PARENTS = 128
DRAWS = 64
ADC_STEP = 1.0 / 16383
KEYS = ("phys", "electronics_white", "electronics_scale", "electronics_offset", "electronics_slope", "local_white")
DEFAULT_FROZEN = PROJECT / "runs/run_001_20261012/final_review_20261012_0900"
DEFAULT_OUTPUT = PROJECT / "runs/noise_audit_20261012/slope"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, torch.Tensor): return clean(value.detach().cpu().numpy())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def charge(wave):
    wave = np.asarray(wave, dtype=np.float64)
    return wave[..., 100:400].sum(-1) - 3 * wave[..., :100].sum(-1)


def tail_charge(wave):
    wave = np.asarray(wave, dtype=np.float64)
    return wave[..., 150:400].sum(-1) - 2.5 * wave[..., :100].sum(-1)


def raw_psd(wave):
    q, tail = charge(wave), tail_charge(wave)
    return np.divide(tail, q, out=np.full_like(q, np.nan), where=q != 0)


def slope_algebra():
    t = np.linspace(-1., 1., 480, dtype=np.float64)
    centered_t = t[:100] - t[:100].mean()
    cq = float(charge(t))
    ct = float(tail_charge(t))
    return dict(t=t, Q_coefficient=cq, Q_coefficient_exact_fraction="120000/479",
                Q_tail_coefficient=ct, Q_fast_coefficient=cq-ct,
                Q_tail_over_Q_coefficient=ct/cq, baseline_t_mean=float(t[:100].mean()),
                baseline_t_sum_squares=float(centered_t @ centered_t),
                baseline_t_sum_squares_exact_fraction="333300/229441")


def psd_slope_first_order(q, psd, delta_b1):
    a = slope_algebra()
    return (a["Q_tail_coefficient"] - psd * a["Q_coefficient"]) * delta_b1 / q


def fir_correlation(h, length):
    h = np.asarray(h, dtype=np.float64)
    autocorrelation = np.correlate(h, h, mode="full")[len(h)-1:]
    distance = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    return np.where(distance < len(h), autocorrelation[np.minimum(distance, len(h)-1)], 0.)


def noise_variance_theory(*, h, sigma2_mean, offset_std, slope_std):
    a = slope_algebra()
    t = a["t"][:100]
    mean_weights = np.full(100, .01)
    slope_weights = (t-t.mean()) / a["baseline_t_sum_squares"]
    covariance = fir_correlation(h, 100)
    white_mean_variance = sigma2_mean / 100
    colored_mean_variance = float(sigma2_mean * mean_weights @ covariance @ mean_weights)
    ols_iid_variance = sigma2_mean / a["baseline_t_sum_squares"]
    ols_colored_variance = float(sigma2_mean * slope_weights @ covariance @ slope_weights)
    q_weights = np.zeros(480)
    q_weights[:100], q_weights[100:400] = -3., 1.
    q_white_variance = float(sigma2_mean * q_weights @ fir_correlation(h, 480) @ q_weights)
    mean_drift_variance = a["baseline_t_mean"]**2 * slope_std**2
    return dict(OLS_iid_white_estimation_variance=ols_iid_variance,
                OLS_colored_estimation_variance=ols_colored_variance,
                OLS_model_total_variance=slope_std**2+ols_colored_variance,
                independent_offset_baseline_mean_variance=offset_std**2,
                independent_slope_baseline_mean_variance=mean_drift_variance,
                iid_white_baseline_mean_variance=white_mean_variance,
                filtered_white_baseline_mean_variance=colored_mean_variance,
                model_baseline_mean_variance=offset_std**2+mean_drift_variance+colored_mean_variance,
                model_baseline_mean_variance_slope_off=offset_std**2+colored_mean_variance,
                Q_filtered_white_variance=q_white_variance,
                Q_random_slope_variance=a["Q_coefficient"]**2*slope_std**2,
                Q_total_electronics_variance=q_white_variance+a["Q_coefficient"]**2*slope_std**2)


def ols_baseline(wave):
    b = np.asarray(wave, dtype=np.float64)[..., :100]
    t = np.linspace(-1., 1., 480)[:100]
    centered_t = t-t.mean()
    baseline_mean = b.mean(-1)
    slope = ((b-baseline_mean[..., None]) * centered_t).sum(-1) / (centered_t @ centered_t)
    midpoint_intercept = baseline_mean - slope*t.mean()
    return dict(mean=baseline_mean, slope=slope, midpoint_intercept=midpoint_intercept)


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    invalid = ~np.isfinite(values)
    # Undefined raw PSD is never removed to produce a deceptively finite metric.
    result = dict(rows=len(values), nonfinite_rows=int(invalid.sum()), includes_every_row=True)
    if invalid.any():
        result.update(mean=None, variance_ddof1=None, undefined_due_to_nonfinite_rows=True)
    else:
        result.update(mean=float(values.mean()), median=float(np.median(values)), variance_ddof1=float(values.var(ddof=1)) if len(values)>1 else None,
                      std_population=float(values.std(ddof=0)), minimum=float(values.min()), maximum=float(values.max()),
                      quantiles_05_50_95=np.quantile(values, [.05,.5,.95]))
    return result


def observables(wave, q_floor):
    wave = np.asarray(wave, dtype=np.float64)
    q, tail = charge(wave), tail_charge(wave)
    psd_raw = raw_psd(wave)
    denominator = np.where(q<0, -1., 1.) * np.maximum(np.abs(q), q_floor)
    b = wave[..., :100]
    return dict(Q=q, PSD_raw=psd_raw, PSD_guarded=tail/denominator,
                Q_nonpositive=q<=0, Q_exact_zero=q==0, PSD_guarded_rows=np.abs(q)<q_floor,
                PSD_raw_nonfinite=~np.isfinite(psd_raw), baseline_mean_V=b.mean(-1),
                baseline_RMS_V=np.sqrt(np.square(b-b.mean(-1,keepdims=True)).mean(-1)))


def baseline_features(wave):
    b = np.asarray(wave, dtype=np.float64)[..., :100].reshape(-1,100)
    c = b-b.mean(1,keepdims=True)
    variance = np.square(c).mean(1)
    denominator = np.maximum(variance, ADC_STEP**2/12)
    acf = np.stack([(c[:,:-lag]*c[:,lag:]).mean(1)/denominator for lag in range(1,17)], 1)
    power = np.abs(np.fft.rfft(c,axis=1))**2
    return dict(baseline_mean_V=summary(b.mean(1)), baseline_RMS_V=summary(np.sqrt(variance)),
                ACF_lags=list(range(1,17)), ACF_mean=acf.mean(0), ACF_event_std=acf.std(0,ddof=1),
                baseline_FFT_mean_power_V2=power.mean(0), FFT_frequency_MHz=np.fft.rfftfreq(100,d=1.)*1000,
                variance_guarded_rows=int((variance<ADC_STEP**2/12).sum()))


def zero_random_slope(noise: Mapping[str, torch.Tensor]):
    if tuple(noise) != KEYS:
        raise ValueError("Noise key/order differs from the frozen interface")
    result = dict(noise)
    result["electronics_slope"] = torch.zeros_like(noise["electronics_slope"])
    return result


def restored_noise(total, seed=SEED, stream_batch=STREAM_BATCH):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_state = generator.get_state().numpy().copy()
    blocks = {key: [] for key in KEYS}
    if tuple(N01Response.noise_shapes) != KEYS:
        raise ValueError("Frozen noise interface changed")
    for start in range(0,total,stream_batch):
        count = min(stream_batch,total-start)
        for key, shape in N01Response.noise_shapes.items():
            blocks[key].append(torch.randn((count,)+shape,generator=generator,dtype=torch.float32))
    return {key: torch.cat(value) for key,value in blocks.items()}, initial_state, generator.get_state().numpy().copy()


def state_hash(model):
    result = hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(key.encode()); result.update(str(array.dtype).encode()); result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


@torch.no_grad()
def paired_forward(model, source, noise, q_floor):
    on = model(source,noise=noise,adc_ste=False)
    off = model(source,noise=zero_random_slope(noise),adc_ste=False)
    for key in ("qbar","qsig","time_scale","unit_shape","signal_noise","basis","mean_residual"):
        if not torch.equal(on[key],off[key]):
            raise AssertionError("Slope intervention changed a non-electronics branch: " + key)
    delta_b1 = model.electronics.log_slope_std.exp().double().cpu().item()*noise["electronics_slope"].double().numpy().reshape(-1)
    expected_wave = delta_b1[:,None] * model.electronics.t.cpu().numpy().reshape(1,480)
    collected = {f"{label}_{key}":out[key].detach().cpu().numpy() for label,out in (("on",on),("off",off))
                 for key in ("waveform","pre_adc","qbar","qsig","time_scale","electronics_charge",
                             "pre_adc_charge","adc_charge_change","saturation_fraction_per_event")}
    on_pre, off_pre = collected["on_pre_adc"], collected["off_pre_adc"]
    delta_pre = on_pre-off_pre
    a = slope_algebra()
    pre_q_difference = charge(delta_pre)
    expected_q = a["Q_coefficient"]*delta_b1
    pre_tail_difference = tail_charge(delta_pre)
    qoff, qon = charge(off_pre), charge(on_pre)
    p_off, p_on = raw_psd(off_pre), raw_psd(on_pre)
    with np.errstate(divide="ignore",invalid="ignore"):
        first_order = psd_slope_first_order(qoff,p_off,delta_b1)
        exact = (a["Q_tail_coefficient"]-p_off*a["Q_coefficient"])*delta_b1/qon
        ratio = np.abs(expected_q/qoff)
    for label in ("on","off"):
        for key,value in observables(collected[label+"_waveform"],q_floor).items():
            collected[label+"_"+key] = value
    quantization_residual = collected["on_Q"]-collected["off_Q"]-pre_q_difference
    unsaturated = (collected["on_saturation_fraction_per_event"]==0)&(collected["off_saturation_fraction_per_event"]==0)
    if np.any(np.abs(quantization_residual[unsaturated]) > 600*ADC_STEP+1e-11):
        raise AssertionError("Unsaturated paired ADC Q residual exceeded the algebraic rounding bound")
    if np.max(np.abs(delta_pre-expected_wave)) > 2e-12:
        raise AssertionError("preADC pair differs by more than the random drift waveform")
    collected.update(delta_b1=delta_b1, expected_preADC_Q_difference=expected_q,
        preADC_wave_difference_residual_max=np.max(np.abs(delta_pre-expected_wave),axis=1),
        preADC_Q_difference_residual=pre_q_difference-expected_q,
        preADC_tail_difference_residual=pre_tail_difference-a["Q_tail_coefficient"]*delta_b1,
        preADC_baseline_mean_difference_residual=delta_pre[:,:100].mean(1)-a["baseline_t_mean"]*delta_b1,
        ADC_Q_difference_quantization_residual=quantization_residual,
        preADC_PSD_difference=p_on-p_off, preADC_PSD_first_order=first_order,
        preADC_PSD_exact_prediction=exact, preADC_PSD_exact_residual=p_on-p_off-exact,
        preADC_PSD_first_order_residual=p_on-p_off-first_order,
        preADC_relative_Q_slope_contribution=ratio,
        preADC_on_Q_nonpositive=qon<=0, preADC_off_Q_nonpositive=qoff<=0,
        non_electronics_branches_bitwise_identical=np.ones(len(source),dtype=bool))
    return collected


def parent_metrics(pair, parents, draws):
    result = {}
    for label in ("on","off"):
        q = pair[label+"_Q"].reshape(parents,draws)
        result[label+"_mean_Q"] = q.mean(1)
        result[label+"_variance_Q"] = q.var(1,ddof=1)
        for kind in ("raw","guarded"):
            psd = pair[label+"_PSD_"+kind].reshape(parents,draws)
            result[label+"_mean_PSD_"+kind] = psd.mean(1)
            result[label+"_variance_PSD_"+kind] = psd.var(1,ddof=1)
            result[label+"_covariance_Q_PSD_"+kind] = ((q-q.mean(1,keepdims=True))*(psd-psd.mean(1,keepdims=True))).sum(1)/(draws-1)
        for flag in ("Q_nonpositive","Q_exact_zero","PSD_guarded_rows","PSD_raw_nonfinite"):
            result[label+"_"+flag+"_count"] = pair[label+"_"+flag].reshape(parents,draws).sum(1)
        result[label+"_baseline_mean_V_mean"] = pair[label+"_baseline_mean_V"].reshape(parents,draws).mean(1)
        result[label+"_baseline_mean_V_variance"] = pair[label+"_baseline_mean_V"].reshape(parents,draws).var(1,ddof=1)
        result[label+"_saturated_draws"] = (pair[label+"_saturation_fraction_per_event"]>0).reshape(parents,draws).sum(1)
    if "old_baseline_Q" in pair:
        result["old_mean_Q"] = pair["old_baseline_Q"].reshape(parents,draws).mean(1)
        result["old_variance_Q"] = pair["old_baseline_Q"].reshape(parents,draws).var(1,ddof=1)
        result["old_mean_PSD_raw"] = pair["old_baseline_PSD_raw"].reshape(parents,draws).mean(1)
        result["old_variance_PSD_raw"] = pair["old_baseline_PSD_raw"].reshape(parents,draws).var(1,ddof=1)
    delta_q = (pair["on_Q"]-pair["off_Q"]).reshape(parents,draws)
    qoff = pair["off_Q"].reshape(parents,draws)
    interaction = 2*((qoff-qoff.mean(1,keepdims=True))*(delta_q-delta_q.mean(1,keepdims=True))).sum(1)/(draws-1)
    result["paired_Q_difference_mean"] = delta_q.mean(1)
    result["paired_Q_difference_variance"] = delta_q.var(1,ddof=1)
    result["paired_Q_difference_sampling_SE"] = delta_q.std(1,ddof=1)/np.sqrt(draws)
    result["Q_variance_interaction_2cov_off_delta"] = interaction
    result["Q_variance_on_minus_off"] = result["on_variance_Q"]-result["off_variance_Q"]
    result["Q_variance_decomposition_residual"] = result["Q_variance_on_minus_off"]-delta_q.var(1,ddof=1)-interaction
    for kind in ("raw","guarded"):
        delta = (pair["on_PSD_"+kind]-pair["off_PSD_"+kind]).reshape(parents,draws)
        result["paired_PSD_"+kind+"_difference_mean"] = delta.mean(1)
        result["paired_PSD_"+kind+"_difference_sampling_SE"] = delta.std(1,ddof=1)/np.sqrt(draws)
    return result


def source_hashes():
    return {str(p.relative_to(PROJECT)).replace("\\","/"):sha(p) for p in sorted((PROJECT/"src/n01").glob("*.py"))}


def load_frozen(frozen_root, arms):
    manifest = read_json(frozen_root/"REMOTE_MANIFEST.json")
    expected = {entry["relative"]:entry["sha256"] for entry in manifest["files"]}
    current = source_hashes()
    if current != manifest["source_hashes"]:
        raise ValueError("Current src/n01 hashes differ from the frozen endpoint source hashes")
    records, payload = [], {}
    def verified(relative):
        path = frozen_root/"train"/relative
        digest = sha(path)
        if expected.get(relative)!=digest: raise ValueError("Frozen artifact hash changed: "+relative)
        records.append(dict(file=relative,sha256=digest,bytes=path.stat().st_size))
        return path
    for arm in arms:
        report = read_json(verified(arm+"/evaluation/endpoint_report.json"))
        checkpoint_path = verified(arm+"/stage_20000.pt")
        checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        if sha(checkpoint_path)!=report["checkpoint_sha256"]: raise ValueError("Endpoint checkpoint/report mismatch")
        repeat_path = verified(arm+"/evaluation/repeated_parent_responses.npz")
        if sha(repeat_path)!=report["artifacts"]["repeated_parent_responses.npz"]["sha256"]: raise ValueError("Repeat artifact/report mismatch")
        with np.load(repeat_path,allow_pickle=False) as z:
            repeat = {key:z[key].copy() for key in ("source_parent_ids","source_parent_rows","source_waveform","waveform")}
        endpoint_path = verified(arm+"/evaluation/endpoint_populations.npz")
        with np.load(endpoint_path,allow_pickle=False) as z:
            real_waveform = z["REAL_waveform"].copy()
            validation_ids = z["source_parent_ids"].copy()
        if (repeat["source_waveform"].shape!=(PARENTS,480) or repeat["waveform"].shape!=(PARENTS,DRAWS,480)
                or len(np.unique(repeat["source_parent_ids"]))!=PARENTS
                or not np.array_equal(validation_ids[repeat["source_parent_rows"]],repeat["source_parent_ids"])):
            raise ValueError("Published repeat parents do not match the original legal validation identity order")
        if not np.isfinite(repeat["source_waveform"]).all() or np.any(charge(repeat["source_waveform"])<=0):
            raise ValueError("Invalid saved source input; no parent removal")
        payload[arm] = dict(report=report,checkpoint=checkpoint,repeat=repeat,real=real_waveform)
    first = payload[arms[0]]["repeat"]
    for arm in arms[1:]:
        if not np.array_equal(first["source_parent_ids"],payload[arm]["repeat"]["source_parent_ids"]) or not np.array_equal(first["source_waveform"],payload[arm]["repeat"]["source_waveform"]):
            raise ValueError("Arms do not share the exact published parent order/source values")
    return payload, records, current


def electronics_parameters(state):
    array = lambda name:state["electronics."+name].detach().cpu()
    h = array("filter.impulse")+.5*torch.tanh(array("filter.raw"))/array("filter.raw").numel()
    h = (h/torch.linalg.vector_norm(h)).double().numpy()
    spread = float((.8*torch.sigmoid(array("raw_spread"))).double())
    log_rms = float(array("log_rms").double())
    return dict(h=h,sigma2_mean=math.exp(2*log_rms+spread**2),
                sigma_mean=math.exp(log_rms),log_sigma_spread=spread,
                offset_std=float(array("log_offset_std").exp().double()),
                slope_std=float(array("log_slope_std").exp().double()),
                offset_mean=float(array("offset_mean").double()),slope_mean=float(array("slope_mean").double()))


def inspect_payload(payload):
    report = dict(algebra=slope_algebra(), intervention="electronics_slope standard-normal draw set to 0 only; slope_mean retained",
                  no_model_qualification=True, PSD_first_order="deltaPSD = CQ*(15/16-PSDoff)*delta_b1/Qoff",
                  PSD_exact="deltaPSD = (Ctail-PSDoff*CQ)*delta_b1/Qon", arms={})
    for arm,data in payload.items():
        parameters = electronics_parameters(data["checkpoint"]["model"])
        theory = noise_variance_theory(**{k:parameters[k] for k in ("h","sigma2_mean","offset_std","slope_std")})
        ols = ols_baseline(data["real"])
        report["arms"][arm] = dict(parameters=parameters,theory=theory,
            REAL_saved_validation_OLS={key:summary(value) for key,value in ols.items()},
            REAL_saved_validation_intercept_slope_covariance=float(np.cov(ols["midpoint_intercept"],ols["slope"],ddof=1)[0,1]),
            PSD_linearization_scope="first order about each saved off preADC draw; large relative slope/Q is reported, not filtered",
            independent_b0_b1_model=True)
    return report


def run_arm(arm,data,noise,parent_input,out,device,forward_batch):
    checkpoint = data["checkpoint"]
    model = N01Response(checkpoint["model_statistics"],variant=checkpoint["variant"])
    model.load_state_dict(checkpoint["model"],strict=True)
    model.set_stage(checkpoint["stage"])
    model.eval().requires_grad_(False)
    model.to(device)
    frozen_state_hash = state_hash(model)
    q_floor = float(torch.as_tensor(checkpoint["loss_statistics"]["q_floor"]))
    blocks = {}
    total = PARENTS*DRAWS
    expanded_parent = np.repeat(np.arange(PARENTS),DRAWS)
    start_time = time.monotonic()
    for start in range(0,total,forward_batch):
        stop = min(total,start+forward_batch)
        batch_noise = {key:value[start:stop] for key,value in noise.items()}
        source = torch.from_numpy(parent_input[expanded_parent[start:stop]]).to(device=device,dtype=torch.float32)
        try:
            values = paired_forward(model,source,batch_noise,q_floor)
        except BaseException:
            np.savez_compressed(out/"failure_batch.npz",source=source.detach().cpu().numpy(),
                flat_draw_indices=np.arange(start,stop),**{key:value.numpy() for key,value in batch_noise.items()})
            raise
        for key,value in values.items(): blocks.setdefault(key,[]).append(value)
        write_json(out/"progress.json",dict(arm=arm,status="frozen_paired_inference",finished_draws=stop,total_draws=total,
                  completed_actual_model_forwards=2*stop,elapsed_s=time.monotonic()-start_time))
        print(json.dumps(dict(arm=arm,draws_done=stop,total_draws=total,elapsed_s=time.monotonic()-start_time)),flush=True)
    pair = {key:np.concatenate(value) for key,value in blocks.items()}
    old = data["repeat"]["waveform"].reshape(total,480)
    lsb_difference = (pair["on_waveform"]-old)/ADC_STEP
    pair.update(source_parent_ids=data["repeat"]["source_parent_ids"],source_parent_rows=data["repeat"]["source_parent_rows"],
                flat_parent_index=expanded_parent,flat_draw_index=np.tile(np.arange(DRAWS),PARENTS),
                old_baseline_Q=charge(old),old_baseline_PSD_raw=raw_psd(old),
                old_replay_max_LSB_difference_per_draw=np.abs(lsb_difference).max(1),
                old_replay_changed_samples_per_draw=np.count_nonzero(pair["on_waveform"]!=old,axis=1))
    parent = parent_metrics(pair,PARENTS,DRAWS)
    parent.update(source_parent_ids=data["repeat"]["source_parent_ids"],source_charge=charge(parent_input))
    np.savez_compressed(out/"paired_outputs.npz",**pair)
    np.savez_compressed(out/"per_parent_metrics.npz",**parent)
    after = state_hash(model)
    if after!=frozen_state_hash: raise AssertionError("Frozen model state changed")
    report = dict(arm=arm,device=str(device),seed=SEED,stream_batch=STREAM_BATCH,forward_batch=forward_batch,
                  parents=PARENTS,draws_per_parent=DRAWS,actual_model_forward_rows=2*total,
                  conditional_variance_covariance_ddof=1,across_parent_summary_has_separate_mean_and_median=True,
                  intervention_key="electronics_slope",slope_mean_retained=float(model.electronics.slope_mean),
                  state_sha256_before=frozen_state_hash,state_sha256_after=after,state_unchanged=True,
                  checkpoint_sha256=data["report"]["checkpoint_sha256"],elapsed_s=time.monotonic()-start_time,
                  old_replay=dict(old_seed=SEED,old_CPU_stream_batch=STREAM_BATCH,
                    old_noise_dictionary_not_archived=True,old_forward_device=data["report"]["device"],
                    exact_waveform_bitwise_equal=bool(np.array_equal(pair["on_waveform"],old)),
                    changed_waveform_samples=int(np.count_nonzero(pair["on_waveform"]!=old)),
                    max_absolute_LSB_difference=float(np.abs(lsb_difference).max()),
                    new_baseline_is_comparison_reference_for_intervention=True),
                  diagnostics={},per_parent_summaries={key:summary(value) for key,value in parent.items() if key not in ("source_parent_ids",)},
                  populations={},no_model_qualification=True,
                  interpretation="frozen random-drift intervention; not a trained repaired model or a physical acceptance result")
    for key in ("preADC_wave_difference_residual_max","preADC_Q_difference_residual","preADC_tail_difference_residual",
                "preADC_baseline_mean_difference_residual","ADC_Q_difference_quantization_residual",
                "preADC_PSD_exact_residual","preADC_PSD_first_order_residual","preADC_relative_Q_slope_contribution"):
        report["diagnostics"][key] = summary(pair[key])
    for label in ("on","off"):
        report["populations"][label] = dict(Q=summary(pair[label+"_Q"]),PSD_raw=summary(pair[label+"_PSD_raw"]),
            PSD_guarded=summary(pair[label+"_PSD_guarded"]),baseline=baseline_features(pair[label+"_waveform"]),
            Q_nonpositive_rows=int(pair[label+"_Q_nonpositive"].sum()),Q_exact_zero_rows=int(pair[label+"_Q_exact_zero"].sum()),
            PSD_guarded_rows=int(pair[label+"_PSD_guarded_rows"].sum()),PSD_raw_nonfinite_rows=int(pair[label+"_PSD_raw_nonfinite"].sum()),
            preADC_Q_nonpositive_rows=int(pair["preADC_"+label+"_Q_nonpositive"].sum()),
            saturated_draws=int((pair[label+"_saturation_fraction_per_event"]>0).sum()),
            maximum_saturation_fraction=float(pair[label+"_saturation_fraction_per_event"].max()))
    report["populations"]["REAL_saved_validation"] = dict(baseline=baseline_features(data["real"]),
        Q=summary(charge(data["real"])),PSD_raw=summary(raw_psd(data["real"])))
    report["artifacts"] = {name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name)) for name in ("paired_outputs.npz","per_parent_metrics.npz")}
    write_json(out/"report.json",report)
    del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("inspect","paired"),default="inspect")
    parser.add_argument("--arms",nargs="+",choices=("full","deterministic"),default=["full","deterministic"])
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT/"inspection")
    parser.add_argument("--frozen-root",type=Path,default=DEFAULT_FROZEN)
    parser.add_argument("--forward-batch",type=int,choices=(16,32,64,128),default=128)
    parser.add_argument("--gpu-memory-limit-gib",type=float,default=2.)
    args = parser.parse_args(argv)
    if len(set(args.arms))!=len(args.arms): raise ValueError("Duplicate arm")
    if os.name=="nt" and args.output_dir.resolve().drive.upper() not in ("E:","D:"):
        raise ValueError("All diagnostic outputs must be on E or D")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE.json").exists(): raise FileExistsError("Completed diagnostic is immutable; choose a new output directory")
    initial_hashes=source_hashes()
    start_time=time.monotonic()
    try:
        payload,records,verified_src=load_frozen(args.frozen_root,args.arms)
        theory=inspect_payload(payload)
        write_json(out/"algebra_and_variance.json",theory)
        binding=dict(script_sha256=sha(__file__),mode=args.mode,arms=args.arms,device=args.device,seed=SEED,restored_declared_old_CPU_noise_stream=True,
            stream_generation_batch=STREAM_BATCH,forward_batch=args.forward_batch,parents=PARENTS,draws_per_parent=DRAWS,
            noise_keys_and_shapes=N01Response.noise_shapes,source_hashes=verified_src,frozen_inputs=records,
            torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,
            old_environment=read_json(args.frozen_root/"train/run_binding.json"),
            CPU_threads=2,intervention="random electronics_slope only, parameter slope_mean preserved",no_weight_update=True,
            old_noise_dictionary_not_archived=True,old_bitwise_replay_not_assumed=True,
            nonpositive_Q_and_PSD_policy="save all raw/guarded values and validity flags; no row deletion",
            no_model_qualification=True)
        write_json(out/"binding.json",binding)
        reports={}
        if args.mode=="paired":
            device=torch.device(args.device)
            if device.type=="cuda":
                if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
                total=torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit_gib*1024**3/total)
            noise,initial_rng,final_rng=restored_noise(PARENTS*DRAWS)
            first=payload[args.arms[0]]["repeat"]
            np.savez_compressed(out/"noise_inputs.npz",seed=np.array(SEED),stream_batch=np.array(STREAM_BATCH),
                CPU_rng_initial=initial_rng,CPU_rng_final=final_rng,
                source_parent_ids=first["source_parent_ids"],source_parent_rows=first["source_parent_rows"],
                source_waveform=first["source_waveform"],
                flat_parent_index=np.repeat(np.arange(PARENTS),DRAWS),flat_draw_index=np.tile(np.arange(DRAWS),PARENTS),
                **{key:value.numpy() for key,value in noise.items()})
            binding["noise_inputs_sha256"]=sha(out/"noise_inputs.npz")
            write_json(out/"binding.json",binding)
            for arm in args.arms:
                directory=out/arm; directory.mkdir(exist_ok=True)
                reports[arm]=run_arm(arm,payload[arm],noise,first["source_waveform"],directory,device,args.forward_batch)
        after_hashes=source_hashes()
        if initial_hashes!=after_hashes: raise AssertionError("src/n01 changed during the diagnostic")
        if args.mode=="paired":
            write_json(out/"report.json",dict(schema_version=1,status="paired_frozen_slope_diagnostic",
                arms=reports,algebra_and_variance=theory,binding=binding,variance_ddof=1,
                source_unchanged=True,scientific_training_updates=0,no_model_qualification=True))
        write_json(out/"COMPLETE.json",dict(status="inspection_complete" if args.mode=="inspect" else "paired_frozen_diagnostic_complete",
            completed_UTC=datetime.now(timezone.utc).isoformat(),mode=args.mode,arms=args.arms,elapsed_s=time.monotonic()-start_time,
            scientific_training_updates=0,paired_response_draws_per_arm=0 if args.mode=="inspect" else PARENTS*DRAWS,
            actual_forward_rows_total=0 if args.mode=="inspect" else 2*PARENTS*DRAWS*len(args.arms),
            source_hashes_before=initial_hashes,source_hashes_after=after_hashes,source_unchanged=True,
            no_model_qualification=True,reports={arm:str(out/arm/"report.json") for arm in reports}))
    except BaseException as error:
        write_json(out/"FAILED.json",dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            source_hashes_before=initial_hashes,source_hashes_after=source_hashes(),elapsed_s=time.monotonic()-start_time))
        raise


if __name__=="__main__": main()