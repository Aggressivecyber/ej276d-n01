"""Registered006 frozen-response conditional-noise sensitivity; zero training.

All raw file hashes bind before any scientific tensor/NPZ decode. The two old
response arms consume the same explicit CPU float32 latent draws per128 rows.
Only event readouts, online moments, and predetermined first128 waveforms are
saved. No optimizer, classifier, mixture fit, or noise projection is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch

from .data import charge_np, load_training, load_validation, sha256, write_json
from .electronics_v3 import ObservedBaselineNoiseV3
from .extension006 import ConditionalExtension006, HYPOTHESES006, signal_components
from .model import ADC_MIN, ADC_MAX, N01Response, charge, digitize

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "integration_006_20261012"
SCHEMA = "integration006/v1"
CONFIG_SHA = "38b7944a2927e5a496717a4d86aae2ee13e5353b9090e34fc40cbf3cabc72ec8"
PROTOCOL_SHA = "40ef6b403816b896ed1683e91b406f39ed43cfbdb41467b7ea653a94c9e02d02"
VIEWS = ("signal_only", "legacy", "fast", "slow", "common_level")
HYPOTHESES = VIEWS[2:]
ARMS = ("deterministic", "full")
SPLITS = ("TRAIN", "DEV")
UPDATES = dict(noise=0, response=0, classifier=0, GMM_fits=0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def clean(value):
    """Keep undefined diagnostics explicit as JSON null, never drop rows."""
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: cpu(item) for key, item in value.items()}
    return value


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(array(value)).tobytes()).hexdigest()


def tensor_dict_sha(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf8") + b"\0" + array_sha(values[key]).encode("ascii") + b"\n")
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "Output already exists; no replacement: " + str(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), "Partial output already exists: " + str(temporary))
    torch.save(value, temporary)
    temporary.replace(path)


def save_npz(path, values):
    path = Path(path)
    require(not path.exists(), "Output already exists: " + str(path))
    with path.with_suffix(path.suffix + ".tmp").open("xb") as handle:
        np.savez_compressed(handle, **values)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def fresh_output(path):
    path = Path(path).resolve()
    if os.name == "nt":
        require(path.drive.upper() in ("E:", "D:"), "006 outputs must be on E/D")
    require(not path.exists() or not any(path.iterdir()), "Output is not empty; no retry or deletion")
    path.mkdir(parents=True, exist_ok=True)
    return path


def bind(config_path, manifest_path, container, identities, *, root=ROOT,
         expected_config_sha=CONFIG_SHA, expected_protocol_sha=PROTOCOL_SHA):
    """Raw bytes and JSON only. Never follows005's historical internal paths."""
    root = Path(root).resolve()
    config_path, manifest_path = Path(config_path), Path(manifest_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    config_hash = sha256(config_path)
    require(config_hash == expected_config_sha == manifest["config_sha256"], "Frozen006 config mismatch")
    require(cfg["run_id"] == manifest["run_id"] == RUN_ID, "Run identity mismatch")
    require(manifest["before_new_scientific_array_decode"] is True and
            manifest["before_new_scientific_forward"] is True and
            manifest["scientific_update_budget"] == 0, "Source was not frozen before science")
    files = manifest["files"]
    for relative, digest in files.items():
        file = (root / relative).resolve()
        require(file.is_relative_to(root), "Manifest path escapes execution root")
        require(file.is_file() and sha256(file) == digest, "Frozen source/input mismatch: " + relative)
    required = {"src/n01/evaluate_integration006.py", "src/n01/extension006.py",
                "src/n01/model.py", "src/n01/electronics_v3.py", "src/n01/data.py"}
    require(required <= set(files), "Manifest omits required execution sources")
    protocol = manifest["protocol_path"]
    require(sha256(root / protocol) == expected_protocol_sha == manifest["protocol_sha256"],
            "Frozen006 protocol mismatch")
    require(files.get(protocol) == expected_protocol_sha, "Protocol missing from source binding")
    references = {}
    for name, item in cfg["references"].items():
        file = (root / item["path"]).resolve()
        require(file.is_relative_to(root) and file.is_file(), "Reference outside execution package")
        require(sha256(file) == item["sha256"] == files.get(item["path"]), "Reference mismatch: " + name)
        references[name] = item["sha256"]
    require(len(references) == 11, "006 requires all eleven frozen references")
    require(sha256(container) == cfg["train_container_sha256"], "TRAIN container mismatch before decode")
    require(sha256(identities) == cfg["identities_sha256"], "Parent identity mismatch before decode")
    require(cfg["hypotheses"] == list(HYPOTHESES006), "Hypotheses changed")
    ev = cfg["evaluation"]
    require(tuple(ev["arms"]) == ARMS and tuple(ev["views"]) == VIEWS and tuple(ev["splits"]) == SPLITS,
            "Evaluation order changed")
    require(ev["draws_per_parent"] == 4 and ev["chunk_rows"] == 128 and ev["bootstrap_draws"] == 1000,
            "Frozen generation/bootstrap sizes changed")
    require(ev["sample_waveform_rows"] == 128 and ev["noise_diagnostic_rows"] == 8192,
            "Frozen sample/diagnostic size changed")
    require(ev["original_count_weights"] is True and ev["old_population_weights_applied"] is False,
            "Population comparison weights are not permitted")
    require(all(cfg[key] == 0 for key in ("noise_updates", "response_updates", "classifier_updates", "GMM_fits")),
            "006 has zero update/fit budget")
    provenance = dict(config_sha256=config_hash, protocol_sha256=expected_protocol_sha,
                      source_manifest_sha256=sha256(manifest_path), references=references,
                      container_sha256=cfg["train_container_sha256"], identities_sha256=cfg["identities_sha256"],
                      source_files=files)
    return cfg, provenance


def frozen_models(cfg, *, root=ROOT, device="cpu"):
    """Called only after bind; checkpoint metadata/statistics must match refs."""
    root = Path(root)
    refs = cfg["references"]
    read = lambda name: json.loads((root / refs[name]["path"]).read_text(encoding="utf-8-sig"))
    cfg005, cfg001, stats = read("config005"), read("config001"), read("statistics001")
    state005 = torch.load(root / refs["noise005"]["path"], map_location="cpu", weights_only=False)
    require(state005["schema_version"] == 5 and state005["engineering"] is False and state005["update"] == 5000,
            "005 is not its fixed scientific endpoint")
    require(state005["config"] == cfg005 and state005["model_type"] == "ObservedBaselineNoiseV3",
            "005 checkpoint/config mismatch")
    require(state005["response_training_updates"] == 0 and state005["response_integration_allowed"] is False,
            "005 frozen scope changed")
    require(state005["config_sha256"] == refs["config005"]["sha256"] and
            state005["source_manifest_sha256"] == refs["source005"]["sha256"], "005 lineage mismatch")
    require(state005["initialization"]["length"] == 100 and state005["acf20_target_rows"] == 21047 and
            state005["initialization"]["metadata"]["events"] == 21047, "005 initialization/support changed")
    for old in (cfg005, cfg001):
        require(old["train_container_sha256"] == cfg["train_container_sha256"] and
                old["identities_sha256"] == cfg["identities_sha256"], "Old/new input lineage mismatch")
    torch.manual_seed(cfg["seeds"]["model_constructor"])
    noise = ObservedBaselineNoiseV3(state005["initialization"])
    noise.load_state_dict(state005["model"], strict=True)
    noise.to(device).eval().requires_grad_(False)
    snapshots = {"noise005": tensor_dict_sha(noise.state_dict())}
    models = {}
    for arm in ARMS:
        state = torch.load(root / refs[arm]["path"], map_location="cpu", weights_only=False)
        require(state["update"] == 20000 and state["stage"] == "low_learning_rate" and
                state["arm"] == state["variant"] == arm, "Old response endpoint/arm mismatch")
        require(state["config"] == cfg001 and state["model_statistics"] == stats["model"] and
                state["loss_statistics"] == stats["loss"], "Frozen001 model/loss statistics mismatch")
        model = N01Response(state["model_statistics"], variant=arm)
        model.load_state_dict(state["model"], strict=True)
        model.set_stage(state["stage"])
        model.to(device).eval().requires_grad_(False)
        models[arm] = model
        snapshots[arm] = tensor_dict_sha(model.state_dict())
        del state
    loss = stats["loss"]
    require(loss["fit_split"] == "TRAIN" and loss["q_reference"] > 0 and loss["q_floor"] > 0 and
            loss["peak_floor"] > 0 and len(loss["charge_bin_boundaries"]) == 7, "Frozen readout statistics invalid")
    extension = ConditionalExtension006(noise, cfg["hypotheses"], cfg["extension"]["kernel_ridge_relative"])
    return models, noise, extension, stats, snapshots


def readouts(waveform, q_floor, *, pre_adc=None):
    """Original deterministic N01 readouts plus explicit rawPSD/ADC accounting."""
    y = np.asarray(waveform, dtype=np.float64)
    require(y.ndim == 2 and y.shape[1] == 480 and np.isfinite(y).all(), "Malformed/nonfinite output; no deletion")
    pre = y if pre_adc is None else np.asarray(pre_adc, dtype=np.float64)
    require(pre.shape == y.shape and np.isfinite(pre).all(), "Malformed/nonfinite preADC output")
    baseline = y[:, :100].mean(1)
    centered = y - baseline[:, None]
    q = charge_np(y)
    tail = y[:, 150:400].sum(1) - 2.5 * y[:, :100].sum(1)
    denominator = np.where(q < 0, -1., 1.) * np.maximum(np.abs(q), q_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = tail / q
    # Undefined rawPSD uses NaN; all rows/guard flags remain in the NPZ.
    raw[~np.isfinite(raw)] = np.nan
    peak_indices = centered[:, 100:400].argmax(1) + 100
    peak = centered[np.arange(len(y)), peak_indices]
    half_rise = np.full(len(y), np.nan)
    for i, (row, peak_index, amplitude) in enumerate(zip(centered, peak_indices, peak)):
        if amplitude <= 0:
            continue
        threshold = .5 * amplitude
        crossing = np.flatnonzero((row[99:peak_index] < threshold) & (row[100:peak_index + 1] >= threshold))
        if crossing.size:
            left = int(crossing[0]) + 99
            half_rise[i] = left + (threshold - row[left]) / (row[left + 1] - row[left])
    saturation = (pre < ADC_MIN) | (pre > ADC_MAX)
    preq = charge_np(pre)
    return dict(Q=q, tail=tail, PSD=tail / denominator, PSD_raw=raw, peak_voltage=peak,
                peak_time_ns=peak_indices.astype(np.float64), half_rise_ns=half_rise,
                relative_half_rise_ns=half_rise - 120., relative_peak_ns=peak_indices - 120.,
                baseline_mean_V=baseline, baseline_rms_V=np.sqrt((centered[:, :100]**2).mean(1)),
                saturation_fraction_per_event=saturation.mean(1), saturation_samples=saturation.sum(1, dtype=np.int64),
                pre_adc_Q=preq, ADCcharge_change=q - preq, Q_nonpositive=q <= 0, PSD_guarded=np.abs(q) < q_floor)


def charge_bins(q, statistics):
    q = np.asarray(q, dtype=np.float64)
    require(np.isfinite(q).all(), "Nonfinite bin charge; no rows dropped")
    floor, reference = statistics["q_floor"], statistics["q_reference"]
    coordinate = np.where(q >= floor, np.log(np.maximum(q, floor) / reference),
                          np.log(floor / reference) + np.arcsinh((q - floor) / floor))
    # torch.bucketize(right=False): equality belongs to the bin on the left.
    return np.searchsorted(statistics["charge_bin_boundaries"], coordinate, side="left").astype(np.int64)


def scalar_summary(values):
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    result = dict(rows=len(x), finite_rows=int(finite.sum()), invalid_rows=int((~finite).sum()))
    if finite.all() and len(x):
        quartiles = np.quantile(x, [.25, .75], method="linear")
        result.update(mean=float(x.mean()), SD=float(x.std(ddof=0)), IQR=float(quartiles[1] - quartiles[0]),
                      minimum=float(x.min()), maximum=float(x.max()))
    else:
        result.update(mean=None, SD=None, IQR=None, minimum=None, maximum=None)
    return result


class Moments:
    """Online centered population moments, avoiding stored bulk480 outputs."""
    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape, np.float64)
        self.M2 = np.zeros(shape, np.float64)

    def add(self, values):
        x = np.asarray(values, dtype=np.float64)
        n = len(x)
        if not n:
            return
        mean = x.mean(0)
        centered = x - mean
        M2 = np.sum(centered * centered, axis=0)
        total = self.count + n
        delta = mean - self.mean
        self.M2 += M2 + delta * delta * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def values(self):
        if not self.count:
            return np.full_like(self.mean, np.nan), np.full_like(self.mean, np.nan)
        return self.mean.copy(), self.M2 / self.count


def fixed_bin_support(bins, parent_rows, minimum_parents, minimum_rows):
    bins, parent_rows = np.asarray(bins), np.asarray(parent_rows)
    rows = np.bincount(bins, minlength=8).astype(np.int64)
    parents = np.array([len(np.unique(parent_rows[bins == b])) for b in range(8)], np.int64)
    return rows, parents, (rows >= minimum_rows) & (parents >= minimum_parents)


def streams_for_split(split, cfg):
    seeds = {key: int(cfg["seeds"][split + "_" + key]) for key in ("response", "noise005", "extension")}
    generators = {key: torch.Generator(device="cpu").manual_seed(seed) for key, seed in seeds.items()}
    return seeds, generators


def draw_latents(batch, generators):
    response = {key: torch.randn((batch,) + shape, generator=generators["response"], dtype=torch.float32)
                for key, shape in N01Response.noise_shapes.items()}
    noise005 = {key: torch.randn((batch,) + shape, generator=generators["noise005"], dtype=torch.float32)
                for key, shape in ObservedBaselineNoiseV3.noise_shapes.items()}
    epsilon = torch.randn((batch, 1, 380), generator=generators["extension"], dtype=torch.float32)
    return response, noise005, epsilon


class Runtime:
    def __init__(self, cfg, device, *, started=None):
        self.cfg, self.start = cfg, time.monotonic() if started is None else started
        self.device = torch.device(device)
        threads = int(cfg["cpu_threads"])
        require(0 < threads <= 12, "CPU thread bound changed")
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device.type == "cuda":
            require(torch.cuda.is_available(), "Requested CUDA is unavailable; no fallback/retry")
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., cfg["gpu_memory_limit_GiB"] * 1024**3 / total), self.device)
        self.maximum_RSS_bytes = 0
        self.maximum_CUDA_bytes = 0

    def check(self, out=None):
        require(time.monotonic() - self.start <= self.cfg["wall_limit_s"], "Wall clock bound exceeded")
        # psutil is part of the preflight runtime. Include any children rather
        # than treating the Python parent RSS as the complete job tree.
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.maximum_RSS_bytes = max(self.maximum_RSS_bytes, rss)
        require(rss <= self.cfg["job_tree_RSS_limit_GiB"] * 1024**3, "Job tree RSS bound exceeded")
        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated(self.device)
            self.maximum_CUDA_bytes = max(self.maximum_CUDA_bytes, used)
            require(used <= self.cfg["gpu_memory_limit_GiB"] * 1024**3, "CUDA bound exceeded")
        if out is not None:
            used = sum(path.stat().st_size for path in Path(out).rglob("*") if path.is_file())
            require(used <= self.cfg["output_limit_MiB"] * 1024**2, "Scientific output bound exceeded")


def add_count(counts, category, role, batch):
    item = counts.setdefault(category, {}).setdefault(role, dict(batches=0, events=0))
    item["batches"] += 1
    item["events"] += int(batch)


def engineer_check(engineering, name, condition):
    # Algebra/prefix faults remain reportable interface failures. A malformed
    # waveform instead raises before any readout can silently remove a row.
    item = engineering.setdefault(name, dict(passed=True, failed_checks=0))
    if not bool(condition):
        item["passed"] = False
        item["failed_checks"] += 1


@torch.no_grad()
def paired_noise(batch, noise005, extension, noise_latents, epsilon, *, split, counts, engineering):
    add_count(counts, "noise005_forward_attempted", split, batch)
    prefix = noise005(noise_latents, length=100)
    add_count(counts, "noise005_forward", split, batch)
    eta = noise005.log_scale_std
    scale = torch.exp(eta * noise_latents["baseline_scale"].to(torch.float64) - eta.square())
    outputs = {}
    quantized_prefix = digitize(prefix, ste=False)
    for hypothesis in HYPOTHESES:
        add_count(counts, "extension_forward_attempted", split + "/" + hypothesis, batch)
        item = extension.extend_prefix(prefix, scale, epsilon, hypothesis)
        add_count(counts, "extension_forward", split + "/" + hypothesis, batch)
        outputs[hypothesis] = item["noise_pre"]
        engineer_check(engineering, "prefix_pre_bitwise", torch.equal(item["noise_pre"][..., :100], prefix))
        engineer_check(engineering, "prefix_post_bitwise", torch.equal(digitize(item["noise_pre"][..., :100], ste=False), quantized_prefix))
        engineer_check(engineering, "shared_scale", torch.equal(item["scale"], scale))
    return prefix, scale, outputs


@torch.no_grad()
def response_chunk(model, source, response_latents, new_noises, noise_prefix, *, split, arm, counts, engineering):
    """Direct components → one ADC per candidate; legacy is separate."""
    batch = len(source)
    add_count(counts, "model_forward_attempted", split + "/" + arm, batch)
    old = model(source, noise=response_latents, adc_ste=False)
    add_count(counts, "model_forward", split + "/" + arm, batch)
    signal = signal_components(old)
    direct_legacy = signal + old["electronics_noise"]
    engineer_check(engineering, "legacy_assembly_bitwise", torch.equal(direct_legacy, old["pre_adc"]))
    signalq = charge(signal)
    oldq = charge(old["electronics_noise"])
    original_charge_residual = charge(old["pre_adc"]) - old["qsig"] - oldq
    engineer_check(engineering, "legacy_Q_identity", torch.allclose(
        original_charge_residual, torch.zeros_like(original_charge_residual), atol=1e-12, rtol=0.))
    add_count(counts, "signal_reference_ADC_attempted", split + "/" + arm, batch)
    signal_waveform = digitize(signal, ste=False)
    add_count(counts, "signal_reference_ADC", split + "/" + arm, batch)
    waveforms = [signal_waveform, old["waveform"]]
    pres = [signal, old["pre_adc"]]
    noises = [torch.zeros_like(signal), old["electronics_noise"]]
    for hypothesis in HYPOTHESES:
        noise = new_noises[hypothesis][:, 0]
        pre = signal + noise
        require(torch.isfinite(pre).all(), "Nonfinite candidate preADC; no deletion")
        add_count(counts, "candidate_ADC_attempted", split + "/" + arm + "/" + hypothesis, batch)
        # The only candidate ADC call; digitized legacy and noise diagnostics
        # are not inputs to this expression.
        waveform = digitize(pre, ste=False)
        add_count(counts, "candidate_ADC", split + "/" + arm + "/" + hypothesis, batch)
        waveforms.append(waveform)
        pres.append(pre)
        noises.append(noise)
        residual = charge(pre) - signalq - charge(noise)
        engineer_check(engineering, "new_Q_identity", torch.allclose(
            residual, torch.zeros_like(residual), atol=1e-12, rtol=0.))
    wave = torch.stack(waveforms)
    pre = torch.stack(pres)
    noise = torch.stack(noises)
    engineer_check(engineering, "response_prefix_across_hypotheses_bitwise",
                   torch.equal(wave[2, :, :100], wave[3, :, :100]) and torch.equal(wave[2, :, :100], wave[4, :, :100]))
    baseline_change = (wave[2, :, :100] != digitize(noise_prefix, ste=False)[:, 0]).to(torch.float64).mean(1)
    return dict(waveform=wave, pre_adc=pre, noise_pre=noise, signal_pre=signal, signal_pre_Q=signalq,
                qsig=old["qsig"], signal_baseline_max_per_event=signal[:, :100].abs().amax(1),
                signal_prefix_ADC_change_fraction=baseline_change,
                legacy_assembly_max_abs_per_event=(direct_legacy - old["pre_adc"]).abs().amax(1),
                legacy_charge_identity_residual=original_charge_residual,
                old_electronics_noise=old["electronics_noise"])


def check_dataset(split, data, cfg):
    ev = cfg["evaluation"]
    key = "train" if split == "TRAIN" else "dev"
    n, r = ev[key + "_source_parents"], ev[key + "_real_rows"]
    require(data["source"].shape == (n, 480) and data["real"].shape == (r, 480), "Registered split size changed")
    require(len(data["source_ids"]) == n and len(np.unique(data["source_ids"])) == n, "SOURCE parent IDs invalid")
    require(len(data["real_ids"]) == r and len(np.unique(data["real_ids"])) == r and len(data["real_groups"]) == r,
            "REAL identity/group support invalid")
    h = data["h"]
    require(h.shape == (n,) and np.isfinite(h).all() and ((h >= 0) & (h <= 1)).all(), "Original h invalid")
    for kind in ("source", "real"):
        x = data[kind]
        require(np.isfinite(x).all() and (charge_np(x) > 1e-8).all(), "Invalid retained " + split + "/" + kind)
        require(((x - x[:, :100].mean(1, keepdims=True)).max(1) > 1e-8).all(), "Invalid retained amplitude")


def split_evaluation(split, data, cfg, models, noise005, extension, stats, runtime, out, counts, engineering):
    """Each batch begins/ends on a whole4-draw parent; all arms are paired."""
    n = len(data["source"])
    parents = np.repeat(np.arange(n, dtype=np.int64), 4)
    draws = np.tile(np.arange(4, dtype=np.int64), n)
    rows = len(parents)
    obs = {split + "_source_parent_ids": data["source_ids"], split + "_source_parent_row": parents,
           split + "_draw_index": draws, split + "_source_h": data["h"],
           split + "_source_count_weights": np.ones(n, np.float64), split + "_REAL_ids": data["real_ids"],
           split + "_REAL_groups": data["real_groups"], split + "_REAL_count_weights": np.ones(len(data["real"]), np.float64)}
    loss = stats["loss"]
    real_obs = readouts(data["real"], loss["q_floor"])
    for field, values in real_obs.items():
        obs[split + "_REAL_" + field] = values
    real_bins = charge_bins(real_obs["Q"], loss)
    obs[split + "_REAL_bins"] = real_bins
    curves = {split + "_REAL_mean": data["real"].mean(0), split + "_REAL_variance": data["real"].var(0, ddof=0)}
    real_bin_curves = [Moments((480,)) for _ in range(8)]
    for b in range(8):
        real_bin_curves[b].add(data["real"][real_bins == b])
    curves[split + "_REAL_bin_mean"] = np.stack([item.values()[0] for item in real_bin_curves])
    curves[split + "_REAL_bin_variance"] = np.stack([item.values()[1] for item in real_bin_curves])
    counts.setdefault("REAL_readout_rows", {})[split] = len(data["real"])
    seeds, generators = streams_for_split(split, cfg)
    latents = dict(schema_version=SCHEMA, split=split, seeds=seeds,
                   streams_initial={key: value.get_state().clone() for key, value in generators.items()}, chunks=[])
    accum = {arm: Moments((5, 480)) for arm in ARMS}
    bin_accum = {arm: [Moments((5, 480)) for _ in range(8)] for arm in ARMS}
    within = {arm: np.zeros((5, 480)) for arm in ARMS}
    replay = None
    samples = None
    for start in range(0, rows, 128):
        runtime.check(out)
        stop = min(start + 128, rows)
        batch = stop - start
        require(batch % 4 == 0, "Chunk split an original parent")
        response, baseline, epsilon = draw_latents(batch, generators)
        record = dict(row_start=start, row_stop=stop, response_sha256=tensor_dict_sha(response),
                      noise005_sha256=tensor_dict_sha(baseline), extension_sha256=array_sha(epsilon), arms={})
        x = torch.as_tensor(data["source"][parents[start:stop]], dtype=torch.float32, device=runtime.device)
        rd = {key: value.to(runtime.device) for key, value in response.items()}
        nd = {key: value.to(runtime.device) for key, value in baseline.items()}
        ep = epsilon.to(runtime.device)
        prefix, scale, extended = paired_noise(batch, noise005, extension, nd, ep, split=split,
                                               counts=counts, engineering=engineering)
        record.update(noise005_prefix_sha256=array_sha(prefix), scale_sha256=array_sha(scale),
                      extended_noise_sha256={h: array_sha(value) for h, value in extended.items()})
        if start == 0:
            first = dict(source=cpu(x), source_parent_row=torch.from_numpy(parents[:batch].copy()),
                         source_parent_ids=torch.as_tensor(data["source_ids"][parents[:batch]].copy()),
                         draw_index=torch.from_numpy(draws[:batch].copy()), response_latents=cpu(response),
                         noise005_latents=cpu(baseline), extension_gaussian=cpu(epsilon))
            latents["first128"] = first
            replay = dict(**first, noise005_pre=cpu(prefix), shared_scale=cpu(scale), arms={})
            samples = {key: cpu(first[key]) for key in ("source_parent_row", "source_parent_ids", "draw_index")}
            samples["arms"] = {}
        # rd/nd/ep are generated once above, outside this arm loop. They are
        # never redrawn, mutated, or seeded separately for a model/hypothesis.
        for arm in ARMS:
            item = response_chunk(models[arm], x, rd, extended, prefix, split=split, arm=arm,
                                  counts=counts, engineering=engineering)
            wave, pre, noise = (array(item[key]) for key in ("waveform", "pre_adc", "noise_pre"))
            signalq = array(item["signal_pre_Q"])
            read = [readouts(wave[i], loss["q_floor"], pre_adc=pre[i]) for i in range(5)]
            field_values = {field: np.stack([view[field] for view in read]) for field in read[0]}
            field_values["noise_pre_Q"] = charge_np(noise.reshape(5 * batch, 480)).reshape(5, batch)
            field_values["charge_identity_residual"] = field_values["pre_adc_Q"] - signalq[None, :] - field_values["noise_pre_Q"]
            tag = split + "_" + arm + "_"
            for field, values in field_values.items():
                if start == 0:
                    obs[tag + field] = np.empty((5, rows), dtype=values.dtype)
                obs[tag + field][:, start:stop] = values
            scalar_fields = ("signal_pre_Q", "qsig", "signal_baseline_max_per_event", "signal_prefix_ADC_change_fraction",
                             "legacy_assembly_max_abs_per_event", "legacy_charge_identity_residual")
            for field in scalar_fields:
                values = array(item[field])
                if start == 0:
                    obs[tag + field] = np.empty(rows, values.dtype)
                obs[tag + field][start:stop] = values
            fixed = charge_bins(signalq, loss)
            if start == 0:
                obs[tag + "fixed_bins"] = np.empty(rows, np.int64)
            obs[tag + "fixed_bins"][start:stop] = fixed
            accum[arm].add(wave.swapaxes(0, 1))
            for b in range(8):
                bin_accum[arm][b].add(wave[:, fixed == b].swapaxes(0, 1))
            within[arm] += wave.reshape(5, batch // 4, 4, 480).var(2, ddof=1).sum(1)
            record["arms"][arm] = dict(signal_sha256=array_sha(item["signal_pre"]), output_sha256=array_sha(wave),
                                         legacy_pre_sha256=array_sha(item["pre_adc"][1]))
            counts.setdefault("generated_view_rows", {}).setdefault(split, {}).setdefault(arm, 0)
            counts["generated_view_rows"][split][arm] += 5 * batch
            if start == 0:
                replay["arms"][arm] = dict(signal_pre=cpu(item["signal_pre"]), legacy_pre=cpu(item["pre_adc"][1]),
                    old_electronics_noise=cpu(item["old_electronics_noise"]), qsig=cpu(item["qsig"]),
                    expected_views=cpu(item["waveform"]), new_pre_adc=cpu(item["pre_adc"][2:]), new_noise_pre=cpu(item["noise_pre"][2:]))
                samples["arms"][arm] = dict(signal_pre=cpu(item["signal_pre"]), noise_pre=cpu(item["noise_pre"][2:]),
                    waveform=cpu(item["waveform"]), pre_adc=cpu(item["pre_adc"]))
        engineer_check(engineering, "stream_pairing", tensor_dict_sha(rd) == record["response_sha256"] and
                       tensor_dict_sha(nd) == record["noise005_sha256"] and array_sha(ep) == record["extension_sha256"])
        latents["chunks"].append(record)
    latents["streams_final"] = {key: value.get_state().clone() for key, value in generators.items()}
    for arm in ARMS:
        tag = split + "_" + arm + "_"
        fixed = obs[tag + "fixed_bins"]
        bins = charge_bins(obs[tag + "Q"], loss)
        obs[tag + "rebinned_bins"] = bins
        migration = np.zeros((5, 8, 8), np.int64)
        for i in range(5):
            np.add.at(migration[i], (fixed, bins[i]), 1)
        obs[tag + "migration"] = migration
        br, bp, supported = fixed_bin_support(fixed, parents, cfg["evaluation"]["minimum_parents_per_fixed_bin"],
                                               cfg["evaluation"]["minimum_rows_per_fixed_bin"])
        obs[tag + "fixed_bin_rows"], obs[tag + "fixed_bin_parents"] = br, bp
        curves[tag + "mean"], curves[tag + "variance"] = accum[arm].values()
        curves[tag + "fixed_bin_mean"] = np.stack([moment.values()[0] for moment in bin_accum[arm]])
        curves[tag + "fixed_bin_variance"] = np.stack([moment.values()[1] for moment in bin_accum[arm]])
        curves[tag + "mean_within_parent_variance"] = within[arm] / n
    atomic_save(latents, out / "latents" / (split + ".pt"))
    return obs, curves, replay, samples


class CovarianceMoments:
    def __init__(self, dimension):
        self.count = 0
        self.mean = np.zeros(dimension, np.float64)
        self.M2 = np.zeros((dimension, dimension), np.float64)

    def add(self, values):
        x = np.asarray(values, np.float64)
        n = len(x)
        if not n:
            return
        mean = x.mean(0)
        centered = x - mean
        total = self.count + n
        delta = mean - self.mean
        self.M2 += centered.T @ centered + np.outer(delta, delta) * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def values(self):
        require(self.count > 0, "No diagnostic observations")
        return self.mean.copy(), self.M2 / self.count


def noise_diagnostic(cfg, noise005, extension, runtime, out, counts, engineering):
    """8192 fixed diagnostics: analytic law and empirical moments, no refit."""
    theory = {hypothesis + "_" + key: array(value).copy()
              for hypothesis in HYPOTHESES for key, value in extension.theoretical(hypothesis).items()}
    n = cfg["evaluation"]["noise_diagnostic_rows"]
    seeds = dict(noise005=cfg["seeds"]["noise_diagnostic005"], extension=cfg["seeds"]["noise_diagnostic_extension"])
    generators = {key: torch.Generator(device="cpu").manual_seed(seed) for key, seed in seeds.items()}
    latents = dict(schema_version=SCHEMA, seeds=seeds,
                   streams_initial={key: value.get_state().clone() for key, value in generators.items()}, chunks=[])
    moments = {hypothesis: CovarianceMoments(480) for hypothesis in HYPOTHESES}
    fields = ("Q", "tail", "seam_increment", "scale", "ADC_Q", "ADC_tail")
    for h in HYPOTHESES:
        for field in fields:
            theory[h + "_diagnostic_" + field] = np.empty(n, np.float64)
    with torch.no_grad():
        for start in range(0, n, 128):
            runtime.check(out)
            stop = min(start + 128, n)
            batch = stop - start
            nd = {key: torch.randn((batch,) + shape, generator=generators["noise005"], dtype=torch.float32)
                  for key, shape in ObservedBaselineNoiseV3.noise_shapes.items()}
            epsilon = torch.randn((batch, 1, 380), generator=generators["extension"], dtype=torch.float32)
            record = dict(row_start=start, row_stop=stop, noise005_sha256=tensor_dict_sha(nd), extension_sha256=array_sha(epsilon))
            prefix, scale, extended = paired_noise(batch, noise005, extension,
                {key: value.to(runtime.device) for key, value in nd.items()}, epsilon.to(runtime.device),
                split="NOISE_DIAGNOSTIC", counts=counts, engineering=engineering)
            record.update(noise005_prefix_sha256=array_sha(prefix), scale_sha256=array_sha(scale),
                          extended_noise_sha256={h: array_sha(value) for h, value in extended.items()})
            for h, value in extended.items():
                pre = array(value[:, 0])
                post = array(digitize(value[:, 0], ste=False))
                moments[h].add(pre)
                q = charge_np(pre)
                tail = pre[:, 150:400].sum(1) - 2.5 * pre[:, :100].sum(1)
                adctail = post[:, 150:400].sum(1) - 2.5 * post[:, :100].sum(1)
                for field, values in dict(Q=q, tail=tail, seam_increment=pre[:, 100] - pre[:, 99],
                    scale=array(scale).reshape(batch), ADC_Q=charge_np(post), ADC_tail=adctail).items():
                    theory[h + "_diagnostic_" + field][start:stop] = values
            latents["chunks"].append(record)
    latents["streams_final"] = {key: value.get_state().clone() for key, value in generators.items()}
    summary = {}
    for h in HYPOTHESES:
        mean, covariance = moments[h].values()
        theory[h + "_diagnostic_mean"], theory[h + "_diagnostic_covariance"] = mean, covariance
        summary[h] = dict(rows=n, empirical_Q=scalar_summary(theory[h + "_diagnostic_Q"]),
            empirical_tail=scalar_summary(theory[h + "_diagnostic_tail"]),
            empirical_seam=scalar_summary(theory[h + "_diagnostic_seam_increment"]),
            analytical_mean_Q=float(theory[h + "_mean_Q"]), analytical_variance_Q=float(theory[h + "_variance_Q"]),
            analytical_tail400480_variance_mean=float(theory[h + "_tail400480_variance_mean"]),
            empirical_tail400480_variance_mean=float(covariance.diagonal()[400:480].mean()),
            seam_variance_over_prefix_increment_mean=float(theory[h + "_seam_variance_over_prefix_increment_mean"]),
            mean_absolute_error=float(np.max(np.abs(mean - theory[h + "_mean"]))),
            covariance_Frobenius_relative_error=float(np.linalg.norm(covariance - theory[h + "_covariance"]) /
                                                    np.linalg.norm(theory[h + "_covariance"])),
            diagnostic_not_acceptance_gate=True)
    variances = np.array([theory[h + "_tail400480_variance_mean"] for h in HYPOTHESES], np.float64)
    require(np.isfinite(variances).all() and (variances > 0).all(), "Analytic tail variance invalid")
    theory["noise_tail_variance_ratio"] = np.array(variances.max() / variances.min())
    counts["diagnostic_rows"] = n
    atomic_save(latents, out / "latents" / "NOISE_DIAGNOSTIC.pt")
    return theory, summary


METRIC_ORDER = ("Q_mean_span_over_REAL_sd", "PSD_mean_span_over_REAL_sd") + tuple(
    "fixed_bin_" + str(b) + "_PSD_sd_ratio" for b in range(8)) + ("fixed_bin_max_PSD_sd_ratio",)


def sd_ratio(sd):
    """A zero/zero width ratio is undefined; a positive/zero is infinity."""
    require(np.isfinite(sd).all() and (sd >= 0).all(), "Invalid SD values")
    smallest, largest = float(np.min(sd)), float(np.max(sd))
    if smallest > 0:
        return largest / smallest
    return np.nan if largest == 0 else np.inf


def extended_quantile(values, probabilities):
    """Linear quantiles preserving infinities, with no omission of NaN draws.

    NumPy's ordinary interpolation makes inf-inf a NaN. The mathematically
    same extended-real quantile returns inf if both endpoints are +inf.
    Undefined original ratios propagate NaN and cannot manufacture a pass.
    """
    a = np.asarray(values, np.float64)
    require(a.ndim == 2 and len(a) > 0, "Bootstrap values must be2D")
    ordered = np.sort(a, axis=0)
    result = np.empty((a.shape[1], len(probabilities)), np.float64)
    for j, probability in enumerate(probabilities):
        rank = probability * (len(a) - 1)
        lo, hi = int(math.floor(rank)), int(math.ceil(rank))
        weight = rank - lo
        left, right = ordered[lo], ordered[hi]
        with np.errstate(invalid="ignore"):
            q = left + weight * (right - left)
        q = np.where((left == right) | (weight == 0), left, q)
        q = np.where((right == np.inf) & (weight > 0), np.inf, q)
        q[np.isnan(a).any(0)] = np.nan
        result[:, j] = q
    return result


def bootstrap_records(obs, split, cfg):
    """Parent sufficient statistics; original support fixed before resampling."""
    n = len(obs[split + "_source_parent_ids"])
    parent = obs[split + "_source_parent_row"]
    ev = cfg["evaluation"]
    records, support = {}, {}
    for arm in ARMS:
        prefix = split + "_" + arm + "_"
        q, psd, bins = obs[prefix + "Q"][2:], obs[prefix + "PSD"][2:], obs[prefix + "fixed_bins"]
        c = np.zeros((8, n), np.float64)
        sums = np.zeros((8, 3, n), np.float64)
        squares = np.zeros_like(sums)
        rows, parents, sufficient = fixed_bin_support(bins, parent, ev["minimum_parents_per_fixed_bin"], ev["minimum_rows_per_fixed_bin"])
        for b in range(8):
            mask = bins == b
            c[b] = np.bincount(parent[mask], minlength=n)
            for h in range(3):
                # Fixed centering is only numerical arithmetic. It preserves
                # the ddof0 statistic and makes any nonzero constant's SD
                # exactly zero, preventing cancellation from fabricating a
                # positive width and an apparently stable0/0 ratio.
                center = psd[h, np.flatnonzero(mask)[0]] if mask.any() else 0.
                deviation = psd[h, mask] - center
                sums[b, h] = np.bincount(parent[mask], weights=deviation, minlength=n)
                squares[b, h] = np.bincount(parent[mask], weights=deviation**2, minlength=n)
        records[arm] = dict(Q=q.reshape(3, n, 4).mean(2), PSD=psd.reshape(3, n, 4).mean(2),
                            count=c, total=sums, square=squares)
        support[arm] = dict(rows=rows, parents=parents, supported=sufficient)
    return records, support


def bootstrap_statistic(record, weights, support, real_scales):
    den = weights.sum()
    qspan = np.ptp(record["Q"] @ weights / den) / real_scales[0]
    pspan = np.ptp(record["PSD"] @ weights / den) / real_scales[1]
    ratios = np.full(8, np.nan)
    supported = np.asarray(support["supported"], bool)
    for b in np.flatnonzero(supported):
        count = record["count"][b] @ weights
        if count == 0:
            continue
        mean = record["total"][b] @ weights / count
        # Roundoff can make an exact zero a tiny negative; max(variance,0)
        # is a variance calculation, not covariance eigenvalue clipping.
        variance = np.maximum(record["square"][b] @ weights / count - mean**2, 0.)
        ratios[b] = sd_ratio(np.sqrt(variance))
    maximum = np.max(ratios[supported]) if supported.any() else np.nan
    return np.r_[qspan, pspan, ratios, maximum]


def parent_bootstrap(obs, split, cfg, real_scales, *, runtime=None):
    require(np.isfinite(real_scales).all() and (np.asarray(real_scales) > 0).all(), "REAL TRAIN normalization is undefined")
    records, support = bootstrap_records(obs, split, cfg)
    n = len(obs[split + "_source_parent_ids"])
    count = cfg["evaluation"]["bootstrap_draws"]
    seed = cfg["seeds"]["bootstrap_" + split]
    rng = np.random.default_rng(seed)
    names = [arm + "/" + metric for arm in ARMS for metric in METRIC_ORDER]
    samples = np.empty((count, len(names)))
    points = np.concatenate([bootstrap_statistic(records[arm], np.ones(n), support[arm], real_scales) for arm in ARMS])
    hashes = []
    for draw in range(count):
        if runtime is not None and draw % 16 == 0:
            runtime.check()
        sampled = rng.integers(0, n, size=n)
        hashes.append(array_sha(sampled))
        weights = np.bincount(sampled, minlength=n).astype(np.float64)
        samples[draw] = np.concatenate([bootstrap_statistic(records[arm], weights, support[arm], real_scales) for arm in ARMS])
    intervals = extended_quantile(samples, cfg["decision"]["bootstrap_interval"])
    payload = {split + "_metric_names": np.asarray(names), split + "_metric_values": samples,
               split + "_point_metrics": points, split + "_intervals": intervals,
               split + "_seed": np.array(seed, np.int64), split + "_resample_parent_hashes": np.asarray(hashes)}
    summary = dict(metric_names=names, point_metrics=points, intervals=intervals, support=support,
                   seed=seed, draws=count, REAL_Q_sd=real_scales[0], REAL_PSD_sd=real_scales[1],
                   uncertainty="paired SOURCE-parent Monte Carlo, not independent acquisition generalization",
                   DEV_descriptive_only=split == "DEV", undefined_draws=np.isnan(samples).sum(0), infinite_draws=np.isinf(samples).sum(0))
    return payload, summary


def population_summary(obs, prefix, *, view=None):
    fields = ("Q", "tail", "PSD", "PSD_raw", "peak_voltage", "peak_time_ns", "half_rise_ns", "baseline_mean_V",
              "baseline_rms_V", "saturation_fraction_per_event", "ADCcharge_change")
    values = lambda field: obs[prefix + field] if view is None else obs[prefix + field][view]
    return dict(scalars={field: scalar_summary(values(field)) for field in fields},
                nonpositive_Q_rows=int(values("Q_nonpositive").sum()), guarded_PSD_rows=int(values("PSD_guarded").sum()),
                saturation_samples=int(values("saturation_samples").sum()), count_weights=1,
                invalid_rows_preserved=True, physical_PSD_not_claimed_for_nonpositive_or_guarded_rows=True)


def conditional_summary(obs, split, arm, cfg):
    prefix = split + "_" + arm + "_"
    fixed, rebinned = obs[prefix + "fixed_bins"], obs[prefix + "rebinned_bins"]
    parent = obs[split + "_source_parent_row"]
    br, bp, supported = fixed_bin_support(fixed, parent, cfg["evaluation"]["minimum_parents_per_fixed_bin"],
                                           cfg["evaluation"]["minimum_rows_per_fixed_bin"])
    result = dict(fixed_bins=[], rebinned_bins=[])
    for b in range(8):
        mask = fixed == b
        fixed_views, rebin_views = {}, {}
        for i, view in enumerate(VIEWS):
            p = obs[prefix + "PSD"][i]
            q = obs[prefix + "Q"][i]
            fixed_views[view] = dict(Q=scalar_summary(q[mask]), PSD=scalar_summary(p[mask]))
            rmask = rebinned[i] == b
            rebin_views[view] = dict(rows=int(rmask.sum()), parents=len(np.unique(parent[rmask])),
                                    Q=scalar_summary(q[rmask]), PSD=scalar_summary(p[rmask]))
        ratio = sd_ratio(np.array([obs[prefix + "PSD"][i, mask].std(ddof=0) for i in range(2, 5)])) if mask.any() else np.nan
        result["fixed_bins"].append(dict(bin=b, rows=int(br[b]), parents=int(bp[b]), supported=bool(supported[b]),
                                        PSD_SD_max_min_ratio=ratio, views=fixed_views))
        result["rebinned_bins"].append(dict(bin=b, views=rebin_views))
    return result


def invalid_rows(obs, split, arm):
    prefix = split + "_" + arm + "_"
    result = {}
    for i, view in enumerate(VIEWS):
        nonpositive = np.flatnonzero(obs[prefix + "Q_nonpositive"][i])
        guarded = np.flatnonzero(obs[prefix + "PSD_guarded"][i])
        result[view] = dict(Q_nonpositive_count=len(nonpositive), PSD_guarded_count=len(guarded),
                            Q_nonpositive_output_rows=nonpositive, PSD_guarded_output_rows=guarded,
                            identities="observables.npz parent_row, parent_ids and draw_index; no dropped rows")
    return result


def make_decision(cfg, obs, train_bootstrap, noise_ratio, engineering):
    """Only TRAIN thresholds decide; DEV leakage/readouts remain descriptive."""
    limits = cfg["decision"]
    failures = [name for name, item in engineering.items() if isinstance(item, dict) and item.get("passed") is False]
    for arm in ARMS:
        prefix = "TRAIN_" + arm + "_"
        if obs[prefix + "signal_baseline_max_per_event"].max() > limits["signal_baseline_absolute_max_V"]:
            failures.append("TRAIN/" + arm + "/signal_baseline_leakage")
        if np.any(obs[prefix + "Q_nonpositive"][2:]) or np.any(obs[prefix + "PSD_guarded"][2:]):
            failures.append("TRAIN/" + arm + "/invalid_new_readout")
    all_supported = all(np.asarray(item["supported"], bool).all() for item in train_bootstrap["support"].values())
    metrics, lower_above, upper_above, undefined = [], [], [], []
    for name, point, ci in zip(train_bootstrap["metric_names"], train_bootstrap["point_metrics"], train_bootstrap["intervals"]):
        metric = name.split("/", 1)[1]
        if metric.startswith("Q_mean"):
            threshold = limits["Q_mean_range_over_REAL_sd_max"]
        elif metric.startswith("PSD_mean"):
            threshold = limits["PSD_mean_range_over_REAL_sd_max"]
        else:
            threshold = limits["fixed_bin_PSD_sd_ratio_max"]
        if np.isnan(ci).any():
            undefined.append(name)
        else:
            if ci[0] > threshold:
                lower_above.append(name)
            if ci[1] > threshold:
                upper_above.append(name)
        metrics.append(dict(name=name, point=point, lower=ci[0], upper=ci[1], threshold=threshold,
                            undefined=bool(np.isnan(ci).any()), infinite_point=bool(np.isinf(point))))
    if failures:
        outcome = "interface_fail"
    elif lower_above or noise_ratio > limits["noise_tail_variance_ratio_max"]:
        outcome = "sensitivity_detected"
    elif all_supported and not upper_above and not undefined:
        outcome = "sensitivity_within_family"
    else:
        outcome = "MC_inconclusive"
    return dict(schema_version=SCHEMA, run_id=RUN_ID, outcome=outcome, engineering_pass=not failures,
                engineering_failures=failures, metrics=metrics, thresholds=limits, support=train_bootstrap["support"],
                all_fixed_bins_supported=bool(all_supported), lower_above=lower_above, upper_above=upper_above,
                undefined_intervals=undefined, noise_tail_variance_ratio=noise_ratio,
                TRAIN_only=True, DEV_descriptive_only=True, scientific_acceptance=False,
                next_response_protocol_allowed=outcome == "sensitivity_within_family", response_updates_authorized_here=0)


def payload_hashes(out):
    return {path.relative_to(out).as_posix(): sha256(path) for path in sorted(Path(out).rglob("*"))
            if path.is_file() and path.name not in ("COMPLETE.json", "FAILURE.json")}


def run(config_path, container, identities, out, *, source_manifest, device="cuda:0", root=ROOT):
    """One bounded run. Every exception writes an actual-count failure receipt."""
    out = fresh_output(out)
    started, start = utc(), time.monotonic()
    counts, engineering = {}, {}
    try:
        cfg, provenance = bind(config_path, source_manifest, container, identities, root=root)
        runtime = Runtime(cfg, device, started=start)
        runtime.check(out)
        models, noise005, extension, statistics, snapshots = frozen_models(cfg, root=root, device=runtime.device)
        data = {"TRAIN": load_training(container, identities), "DEV": load_validation(container, identities)}
        for split in SPLITS:
            check_dataset(split, data[split], cfg)
        require(not np.intersect1d(data["TRAIN"]["source_ids"], data["DEV"]["source_ids"]).size,
                "SOURCE TRAIN/DEV parent overlap")
        require(not np.intersect1d(data["TRAIN"]["real_ids"], data["DEV"]["real_ids"]).size and
                not np.intersect1d(data["TRAIN"]["real_groups"], data["DEV"]["real_groups"]).size,
                "REAL TRAIN/DEV identity/group overlap")
        global_rng = torch.get_rng_state().clone()
        obs, curves = {}, {}
        replay = dict(schema_version=SCHEMA, views=VIEWS, hypotheses=HYPOTHESES, splits={})
        samples = dict(schema_version=SCHEMA, views=VIEWS, hypotheses=HYPOTHESES, splits={})
        for split in SPLITS:
            o, c, r, s = split_evaluation(split, data[split], cfg, models, noise005, extension, statistics,
                                          runtime, out, counts, engineering)
            obs.update(o)
            curves.update(c)
            replay["splits"][split], samples["splits"][split] = r, s
        theory, diagnostic = noise_diagnostic(cfg, noise005, extension, runtime, out, counts, engineering)
        real_scales = (obs["TRAIN_REAL_Q"].std(ddof=0), obs["TRAIN_REAL_PSD"].std(ddof=0))
        bootstrap, bootstrap_summary = {}, {}
        for split in SPLITS:
            values, summary = parent_bootstrap(obs, split, cfg, real_scales, runtime=runtime)
            bootstrap.update(values)
            bootstrap_summary[split] = summary
        engineer_check(engineering, "global_RNG_untouched_by_generation", torch.equal(global_rng, torch.get_rng_state()))
        extension.assert_frozen()
        engineer_check(engineering, "frozen_states", tensor_dict_sha(noise005.state_dict()) == snapshots["noise005"] and
                       all(tensor_dict_sha(models[arm].state_dict()) == snapshots[arm] for arm in ARMS))
        engineer_check(engineering, "source_coverage", all(counts["model_forward"][split + "/" + arm]["events"] ==
                       len(data[split]["source"]) * 4 for split in SPLITS for arm in ARMS))
        engineer_check(engineering, "count_weights_only", True)
        engineer_check(engineering, "single_ADC", all(counts["candidate_ADC"][split + "/" + arm + "/" + h]["events"] ==
                       len(data[split]["source"]) * 4 for split in SPLITS for arm in ARMS for h in HYPOTHESES))
        leakage, algebra, invalid = {}, {}, {}
        for split in SPLITS:
            leakage[split], algebra[split], invalid[split] = {}, {}, {}
            for arm in ARMS:
                p = split + "_" + arm + "_"
                maximum = float(obs[p + "signal_baseline_max_per_event"].max())
                leakage[split][arm] = dict(absolute_max_V=maximum, absolute_max_ADC_steps=maximum * 16383,
                    ADC_grid_change_fraction_mean=float(obs[p + "signal_prefix_ADC_change_fraction"].mean()),
                    threshold_exceeded=maximum > cfg["decision"]["signal_baseline_absolute_max_V"],
                    participates_in_decision=split == "TRAIN")
                algebra[split][arm] = {field: float(np.max(np.abs(obs[p + field]))) for field in
                    ("legacy_assembly_max_abs_per_event", "legacy_charge_identity_residual", "charge_identity_residual")}
                invalid[split][arm] = invalid_rows(obs, split, arm)
        engineering.update(signal_baseline_limit=leakage, algebra_residuals=algebra, invalid_readouts=invalid,
                           no_silent_deletion=True, analytic_block_and_positive_definite=True)
        decision = make_decision(cfg, obs, bootstrap_summary["TRAIN"], float(theory["noise_tail_variance_ratio"]), engineering)
        coverage = {split: dict(source_parents=len(data[split]["source"]), source_rows=len(data[split]["source"]) * 4,
            draws_per_parent=4, REAL_rows=len(data[split]["real"]), REAL_original_groups=len(np.unique(data[split]["real_groups"])),
            parent_major=True, network_inputs="SOURCE voltage and explicit Gaussian latents only; no ID/h/row/group") for split in SPLITS}
        population, conditional = {}, {}
        for split in SPLITS:
            population[split] = {"REAL": population_summary(obs, split + "_REAL_")}
            conditional[split] = {}
            for arm in ARMS:
                population[split][arm] = {view: population_summary(obs, split + "_" + arm + "_", view=i)
                                         for i, view in enumerate(VIEWS)}
                conditional[split][arm] = conditional_summary(obs, split, arm, cfg)
        statistic_keys = ("q_reference", "q_floor", "peak_floor", "charge_bin_boundaries")
        summary = dict(schema_version=SCHEMA, run_id=RUN_ID, status="zero_training_completed", provenance=provenance,
            statistics={key: statistics["loss"][key] for key in statistic_keys}, coverage=coverage, engineering=engineering,
            population=population, conditional=conditional, noise_diagnostic=diagnostic, bootstrap=bootstrap_summary,
            actual_counts=counts, updates=UPDATES, scope=dict(scientific_acceptance=False, hardware_noise_identified=False,
                finite_hypothesis_family_not_confidence_set=True, frozen_response_counterfactual_not_repaired_performance=True,
                two_arms_separately_trained_not_architecture_causal_effect=True, original_count_weights_only=True,
                old_population_weights_applied=False, independent_acquisition_available=False, DEV_previously_exposed=True,
                DEV_descriptive_only=True, no_REAL_waveforms_saved=True, original_training_updates_remain=92000,
                software_grid_ns=1, native_hardware_sampling_verified=False,
                REAL_saturation_scope="Canonical recorded-ADC interval exceedance is zero; upstream clipping unknown",
                REAL_ADC_rail_samples={s: int(((data[s]["real"] == ADC_MIN) | (data[s]["real"] == ADC_MAX)).sum()) for s in SPLITS}))
        save_npz(out / "observables.npz", obs)
        save_npz(out / "curves.npz", curves)
        save_npz(out / "theory.npz", theory)
        save_npz(out / "bootstrap.npz", bootstrap)
        atomic_save(replay, out / "replay128.pt")
        atomic_save(samples, out / "sample_waveforms.pt")
        _, after = bind(config_path, source_manifest, container, identities, root=root)
        require(after == provenance, "Frozen input/source changed during run")
        write_json(out / "SUMMARY.json", clean(summary))
        write_json(out / "DECISION.json", clean(decision))
        runtime.check(out)
        size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        require(size + 65536 <= cfg["output_limit_MiB"] * 1024**2, "Output exceeds completion-receipt reserve")
        complete = dict(schema_version=SCHEMA, run_id=RUN_ID, status="zero_training_completed", started_UTC=started,
            completed_UTC=utc(), wall_seconds=time.monotonic() - start, provenance=provenance,
            payload_sha256=payload_hashes(out), actual_counts=counts, updates=UPDATES, output_bytes=size,
            decision_outcome=decision["outcome"], maximum_RSS_bytes=runtime.maximum_RSS_bytes,
            maximum_CUDA_bytes=runtime.maximum_CUDA_bytes)
        write_json(out / "COMPLETE.json", clean(complete))
        return complete
    except Exception as error:
        failure = dict(schema_version=SCHEMA, run_id=RUN_ID, status="zero_training_failed_no_retry", started_UTC=started,
            failed_UTC=utc(), error_type=type(error).__name__, error=str(error), actual_counts=counts, updates=UPDATES,
            partial_payload_sha256=payload_hashes(out))
        write_json(out / "FAILURE.json", clean(failure))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    complete = run(args.config, args.container, args.identities, args.out,
                   source_manifest=args.source_manifest, device=args.device)
    print(json.dumps(clean(complete), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
