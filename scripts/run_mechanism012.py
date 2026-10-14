"""One fixed-final-head mechanism diagnosis, without updates or fresh draws.

The stdlib-only binding gate runs before NumPy/Torch/project imports, archive
decoding or checkpoint restore. Registered forward/ADC/AD operations each have
an fsynced entry before execution and success record before status advancement.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import time

RUN_ID = "mechanism012_20261015"
HYPOTHESES = ("fast", "slow", "common_level")
HEAD_NAMES = ("basis_head.weight", "basis_head.bias")
TRAIN_KEYS = ("source", "source_carrier384", "real", "real_baseline100", "source_ids", "real_ids", "real_groups", "source_role", "real_role")
ZERO_KEYS = ("head_initial_or1000_rows", "optimizer_created", "optimizer_steps", "G_updates", "D_updates",
    "noise_updates", "population_updates", "C2ST_updates", "DEV_rows", "sampler_created", "new_latent_draws", "new_noise_draws")
MAIN_SLOTS = ("dataset", "statistics", "data_statistics", "noise005", "deterministic", "base_config", "training_complete",
    "head_initial", "head_1000", "head_final", "shape010_manifest", "shape010_bindings", "shape010_complete",
    "shape010_initial_eval", "shape010_final_eval", "shape010_eval_bank", "shape010_binding_receipt", "shape010_native_exit",
    "stability011_manifest", "stability011_bindings", "stability011_complete", "stability011_result", "stability011_native_exit", "stability011_frozen_inputs")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(flag, message):
    if not flag:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return value


def relative_path(root, name):
    part = PurePosixPath(name)
    require(not part.is_absolute() and ".." not in part.parts and "\\" not in name and str(part) == name
        and name not in ("", "."), "Unsafe012 inventory path")
    path = (root/name).resolve()
    require(path.is_relative_to(root.resolve()), "012 inventory path escaped root")
    return path


def fixed_policy(path):
    p = read_json(path)
    require(p.get("schema") == "n01/mechanism012-policy/v1" and p.get("run_id") == RUN_ID, "Wrong012 policy")
    expected = dict(source_file_count=69, source_parent_clusters=512, draws_per_parent=4, matched_rows=2048,
        head_update=2000, metadata_only_head_updates=[0, 1000], forward_chunk_rows=128,
        head_parameter_names=list(HEAD_NAMES), head_parameter_count=1288, initialization_seed=2026101420,
        voltage_gain=.05, hypotheses=list(HYPOTHESES), allowed_dataset_members=list(TRAIN_KEYS),
        TRAIN_counts=dict(source=6593, real=21047), DEV_numeric_read=False, automatic_restart=False, overwrite_outputs=False)
    require(all(p.get(k) == v for k, v in expected.items()), "012 fixed population/model/policy changed")
    b = p["budgets"]
    require(all(b.get(k) == 0 for k in ZERO_KEYS) and all(b.get(k) == 2048 for k in ("old_A_rows", "noise005_rows", "head_final_rows"))
        and b.get("scalar_AD_queries") == 9 and b.get("masked_endpoint_VJP_queries") == 2
        and b.get("ADC_hard_or_STE_rows") == 49152 and b.get("checkpoint_loads_max") == 5, "012 operation budgets changed")
    require(p["projection"] == dict(seed=2026101430, training_seed=2026101426, kind="evaluation", update=None,
        fingerprint="27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482", regenerate_once_then_freeze=True,
        new_random_projection_draws=0), "012 projection contract changed")
    require(p["objective"]["stage"] == "joint_stochastic" and p["objective"]["comparison_weights"] is None
        and tuple(p["objective"][k] for k in ("wave_weight", "QPSD_weight", "regular_weight")) == (1, 2, .01), "012 objective changed")
    require(p["probe"]["step_sizes_ADC_LSB"] == [.1, 1, 10] and p["probe"]["signs"] == [-1, 1]
        and p["probe"]["zero_direction_behavior"] == "record_six_undefined_skip_probe_ADC", "012 fixed probe changed")
    require(tuple(p["resource_limits"][k] for k in ("CPU_threads", "GPU_GiB", "job_tree_RSS_GiB", "wall_seconds", "output_MiB"))
        == (12, 10, 8, 900, 128), "012 resource bounds changed")
    require(p["source_Q8"]["preprocessing"] == "float64_source_minus_mean_of_first100_samples"
        and (p["source_Q8"]["canonical_start"], p["source_Q8"]["canonical_stop"]) == (110, 366)
        and p["source_Q8"]["boundaries"] == "frozen_TRAIN_loss.charge_bin_boundaries", "012 SOURCE strata changed")
    return p


def bind_inputs(policy_path, binding_path, p):
    """Hash complete transitive sources and all inputs using stdlib only."""
    b = read_json(binding_path)
    roots = [Path(p[k]).resolve() for k in ("remote_root", "shape010_remote_root", "stability011_remote_root", "reference_remote_root")]
    root, old010, old011, old008 = roots
    require(b.get("schema") == "n01/mechanism012-bindings/v1" and b.get("run_id") == RUN_ID
        and b.get("roots") == list(map(str, roots)) and b["policy_sha256"] == sha(policy_path), "012 binding identity/roots/policy mismatch")
    inventory = b["source_manifest"]
    mp = Path(inventory["path"]).resolve()
    require(mp.is_relative_to(root) and sha(mp) == inventory["sha256"], "012 source manifest mismatch")
    m = read_json(mp)
    require(m.get("schema") == "n01/mechanism012-source/v1" and m.get("run_id") == RUN_ID
        and m["policy_sha256"] == sha(policy_path) and len(m["files"]) == 69
        and m["bound_inputs"] == {k: v["sha256"] for k, v in b["files"].items()}, "012 full source/input contract mismatch")
    files, sources, paths = {}, {}, {}
    def add(path, digest, source=False):
        path = Path(path).resolve()
        require(any(path.is_relative_to(base) for base in roots) and path.is_file(), "Unregistered012 path: "+str(path))
        require(str(path) not in files or files[str(path)] == digest, "Conflicting012 input SHA")
        if str(path) in files:
            if source:
                sources[str(path)] = digest
            return
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) and sha(path) == digest,
            "012 frozen bytes changed: "+str(path))
        files[str(path)] = digest
        if source:
            sources[str(path)] = digest
    for name, digest in m["files"].items():
        add(relative_path(root, name), digest, source=True)
    required = set(MAIN_SLOTS) | {arm+"_"+h+"_TRAIN" for arm in ("frozen_A", "shape010_final") for h in HYPOTHESES}
    controls = {name for name in b["files"] if name.startswith("control_")}
    require(len(controls) == 6 and set(b["files"]) == required | controls, "012 fixed36 input slots changed")
    for key, row in b["files"].items():
        paths[key] = Path(row["path"]).resolve()
        add(paths[key], row["sha256"])
        if "bytes" in row:
            require(paths[key].stat().st_size == row["bytes"], "012 bound input size mismatch")
    add(policy_path, sha(policy_path)); add(binding_path, sha(binding_path)); add(mp, inventory["sha256"])
    old_m, old_b = read_json(paths["shape010_manifest"]), read_json(paths["shape010_bindings"])
    require(paths["shape010_manifest"] == old010/"SHAPE010_SOURCE_MANIFEST.json" and paths["shape010_bindings"] == old010/"SHAPE010_BINDINGS.json"
        and len(old_m["files"]) == 53 and len(old_m["protected_manifests"]) == 4
        and old_b["source_manifest"]["sha256"] == sha(paths["shape010_manifest"]), "010 inherited source chain mismatch")
    for name, digest in old_m["files"].items():
        require(m["files"].get(name) == digest, "012 numerical source copy differs from010: "+name)
        add(relative_path(old010, name), digest, source=True)
    for name, digest in old_m["protected_manifests"].items():
        require(m["files"].get(name) == digest and m["protected_manifests"].get(name) == digest, "012 old4 metadata copies differ")
        add(relative_path(root, name), digest)
    require(len(old_b["files"]) == 7 and old_m["bound_inputs"] == {k: v["sha256"] for k, v in old_b["files"].items()}, "010 old7 input chain mismatch")
    for key, row in old_b["files"].items():
        require(paths[key] == Path(row["path"]).resolve() and b["files"][key]["sha256"] == row["sha256"], "012 old input alias changed")
    require(paths["dataset"].is_relative_to(old008) and paths["statistics"].is_relative_to(old008), "012 prepared inputs outside original root")
    metadata = m["control_metadata_copies"]
    require(len(metadata) == 6 and {str(paths[k]) for k in controls} == {str(relative_path(root, n)) for n in metadata}, "012 control metadata layout differs")
    for name, digest in metadata.items():
        require(m["files"].get(name) == digest, "012 control copy absent source inventory")
        add(relative_path(root, name), digest)
    sm = read_json(paths["stability011_manifest"])
    require(paths["stability011_manifest"] == old011/"STABILITY011_SOURCE_MANIFEST.json" and len(sm["files"]) == 63,
        "012 old011 source inventory mismatch")
    for name, digest in sm["files"].items():
        add(relative_path(old011, name), digest, source=True)
    sb = read_json(paths["stability011_bindings"])
    require(sb["source_manifest"]["sha256"] == sha(paths["stability011_manifest"]), "011 source/binding chain mismatch")
    complete010 = read_json(paths["shape010_complete"])
    require(complete010["G_updates"] == 2000 and complete010["counters"]["attempted_G_updates"] == 2000,
        "012 requires actual completed θ2000")
    artifact010 = old010/"runs/primary/artifacts"
    require(paths["shape010_complete"] == artifact010/"COMPLETE.json" and not (artifact010/"FAILED.json").exists(), "010 artifact completion path/failure mismatch")
    for name, row in complete010["artifacts"].items():
        path = relative_path(artifact010, name)
        require(path.stat().st_size == row["bytes"], "010 immutable artifact size mismatch")
        add(path, row["sha256"])
    for slot, name in (("head_initial", "initial.pt"), ("head_1000", "periodic_01000.pt"), ("head_final", "final.pt"),
        ("shape010_eval_bank", "EVALUATION_BANK.npz"), ("shape010_initial_eval", "INITIAL_EVALUATION.json"),
        ("shape010_final_eval", "FINAL_EVALUATION.json"), ("shape010_binding_receipt", "BINDING.json")):
        require(paths[slot] == artifact010/name, "012 head/bank/anchor slot path changed")
    for arm in ("frozen_A", "shape010_final"):
        for h in HYPOTHESES:
            key = arm+"_"+h+"_TRAIN"
            require(paths[key] == artifact010/(key+".npz"), "012 saved-anchor slot path changed")
    for key in ("shape010_native_exit", "stability011_native_exit"):
        native = read_json(paths[key])
        require(native["child_native_exit_code"] == native["exit_code"] == 0 and native["reason"] is None and native["error"] is None,
            "Prior native completion not clean")
    sc, sr = read_json(paths["stability011_complete"]), read_json(paths["stability011_result"])
    require(sc["bootstrap_replicates_completed"] == sc["bootstrap_replicates_attempted"] == 500
        and sr["classification"] == p["prior_stability_required"]["classification"] and sr["all_replicate_conditional_coverage_complete"],
        "012 requires completed/stable prior011")
    for name, row in sc["artifacts"].items():
        path = relative_path(old011/"runs/primary/artifacts", name)
        require(path.stat().st_size == row["bytes"], "011 artifact size mismatch")
        add(path, row["sha256"])
    prior = b["prior_independent_readback"]
    require(prior.get("status") == "PASS" and prior.get("sha256") == p["prior_stability_required"]["independent_readback_sha256"]
        and prior.get("scope") == "root_verified_local_receipt", "012 prior independent readback registration mismatch")
    return dict(paths=paths, files_sha256=files, sources_sha256=sources,
        before_numeric_import=True, before_TRAIN_decode=True, before_checkpoint_load=True, old53_verified=True,
        old011_63_verified=True, new69_verified=True, DEV_numeric_read=False)


def verify_files(binding):
    for path, digest in {**binding["files_sha256"], **binding["sources_sha256"]}.items():
        require(sha(path) == digest, "012 protected file changed during diagnosis: "+path)


class OperationStore012:
    def __init__(self, output, policy, *, artificial=False, device="cpu"):
        self.root, self.policy, self.artificial, self.device = Path(output), policy, artificial, device
        require(not self.root.exists(), "Preserve existing012 output")
        self.root.mkdir(parents=True, exist_ok=False)
        self.started, self.phase = time.monotonic(), "binding"
        self.binding = None
        self.point_passed = False
        self.before_snapshot = None
        self.entered = self.completed = 0
        self.inflight = {}
        self.attempted = {k: 0 for k in policy["budgets"] if k != "checkpoint_loads_max"}
        self.counters = dict(self.attempted)
        self.attempted["checkpoint_loads"] = self.counters["checkpoint_loads"] = 0
        self.failed_operations = []
        (self.root/"operations.jsonl").open("x", encoding="utf8").close()

    def base(self):
        value = dict(schema="n01/mechanism012-result/v1", UTC=now(), run_id=RUN_ID, pid=os.getpid(),
            artificial_only=self.artificial, phase=self.phase, counters=self.counters.copy(), attempted_counters=self.attempted.copy(),
            operations_entered=self.entered, operations_completed=self.completed,
            inflight=copy.deepcopy(self.inflight), inflight_actual_completion="unknown_without_durable_completed_record",
            failed_operations=copy.deepcopy(self.failed_operations), point_checks_passed=self.point_passed,
            head_parameter_names=list(HEAD_NAMES), head_parameter_count=1288,
            scalar_AD_queries_completed=self.counters["scalar_AD_queries"],
            masked_VJP_queries_completed=self.counters["masked_endpoint_VJP_queries"],
            AD_queries_completed=self.counters["scalar_AD_queries"]+self.counters["masked_endpoint_VJP_queries"],
            optimizer_created=0, optimizer_steps=0, DEV_numeric_read=False,
            scientific_status="response_match_not_established", elapsed_seconds=time.monotonic()-self.started)
        if self.binding is not None:
            value.update(inputs_sha256=self.binding["files_sha256"], sources_sha256=self.binding["sources_sha256"])
        return value

    def json(self, name, value):
        with (self.root/name).open("x", encoding="utf8") as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())

    def status(self, extra=None):
        value = self.base()
        if extra:
            value.update(extra)
        temp = self.root/"status.json.tmp"
        require(not temp.exists(), "Preserve unexpected012 status temporary")
        with temp.open("x", encoding="utf8") as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(self.root/"status.json")

    def npz(self, name, **arrays):
        import numpy as np
        with (self.root/name).open("xb") as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())

    def journal(self, record):
        with (self.root/"operations.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(clean(record), allow_nan=False)+"\n")
            stream.flush(); os.fsync(stream.fileno())

    def guard(self):
        limits = self.policy["resource_limits"]
        require(time.monotonic()-self.started <= limits["wall_seconds"], "012 wall cap exceeded")
        require(sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()) <= limits["output_MiB"]*1024**2, "012 output cap exceeded")
        if sys.platform.startswith("linux"):
            import resource
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits["job_tree_RSS_GiB"]*1024**3, "012 RSS cap exceeded")

    def call(self, name, charges, function, **details):
        self.guard()
        for key, amount in charges.items():
            cap = self.policy["budgets"]["checkpoint_loads_max"] if key == "checkpoint_loads" else self.policy["budgets"][key]
            require(type(amount) is int and amount >= 0 and self.attempted[key]+amount <= cap, "012 operation exceeds cap: "+key)
        operation_id = self.entered
        entry = dict(operation_id=operation_id, event="entered", name=name, charges=charges, UTC=now(), details=details)
        self.journal(entry)
        self.entered += 1
        self.inflight[str(operation_id)] = entry
        for key, amount in charges.items():
            self.attempted[key] += amount
        self.status()
        try:
            value = function()
            if str(self.device).startswith("cuda"):
                import torch
                torch.cuda.synchronize(self.device)
        except BaseException as error:
            row = dict(operation_id=operation_id, event="failed", name=name, charges=charges, UTC=now(),
                error_type=type(error).__name__, error=str(error), actual_inflight_completion="unknown")
            self.journal(row); self.failed_operations.append(row)
            raise
        completed = self.counters.copy()
        for key, amount in charges.items():
            completed[key] += amount
        self.journal(dict(operation_id=operation_id, event="completed", name=name, charges=charges, UTC=now(), counters=completed))
        self.counters = completed
        self.completed += 1
        self.inflight.pop(str(operation_id))
        self.status()
        return value


def tensor_snapshot(model, extension):
    import torch
    from n01.response008 import state_digest
    model.assert_frozen_unchanged(deep=True); extension.assert_frozen()
    modules = dict(baseline=model.baseline, noise005=extension.model005, extension=extension, head=model.basis_head)
    result = dict(states={}, tensors={}, baseline_stage=model.baseline.stage)
    for name, module in modules.items():
        require(not module.training and all(p.grad is None for p in module.parameters()), "012 model mode/grad accumulator changed")
        if name != "head":
            require(all(not p.requires_grad for p in module.parameters()), "012 frozen parameter became trainable")
        else:
            require(all(p.requires_grad for p in module.parameters()), "012 new-head autograd disabled")
        result["states"][name] = state_digest(module.state_dict())
        result["tensors"][name] = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
            sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k, v in module.state_dict().items()}
    require(tuple(model.trainable_parameter_names()) == HEAD_NAMES and sum(p.numel() for p in model.trainable_parameters()) == 1288,
        "012 head whitelist or count changed")
    return result


def array(value):
    return value.detach().cpu().numpy()


def finite(value, label):
    import torch
    require(bool(torch.isfinite(value).all()), "012 nonfinite "+label)
    return value


def summarize_influences(rows, identities, parent_bins, repeats, actual_loss_differences):
    import numpy as np
    parent = np.asarray(identities["source_parent_row"])
    bins = np.repeat(parent_bins, repeats)
    result = {}
    union = np.zeros(len(parent_bins), bool)
    for h in HYPOTHESES:
        changed = rows[h+"_crop_changed"].astype(bool)
        union[np.unique(parent[changed])] = True
        categories = dict(all=np.ones(len(parent), bool), crop_changed=changed, crop_unchanged=~changed)
        categories.update({key: rows[h+"_actual_flag_"+key].astype(bool) for key in
            ("peak_at_search_boundary", "global_peak_outside_search", "Q_nonpositive", "PSD_undefined", "PSD_guarded",
             "ADC_rail_hit", "pre_ADC_out_of_range", "shape_peak_tied", "shape_peak_floor_active", "Q_exact_zero", "Q_negative", "ADC_halfgrid_tied")})
        categories["peak_tied"] = rows[h+"_peak_tied"].astype(bool)
        categories["peak_margin_zero"] = rows[h+"_first_runnerup_margin_V"] == 0
        def group(mask):
            count = int(mask.sum())
            return dict(draws=count, parents=int(len(np.unique(parent[mask]))), total_draws=len(parent), total_parents=len(parent_bins),
                wave_signed_influence_sum=float(rows[h+"_signed_influence_wave"][mask].sum()),
                QPSD_signed_influence_sum=float(rows[h+"_signed_influence_QPSD"][mask].sum()),
                wave_signed_influence_mean=None if count == 0 else float(rows[h+"_signed_influence_wave"][mask].mean()),
                QPSD_signed_influence_mean=None if count == 0 else float(rows[h+"_signed_influence_QPSD"][mask].mean()))
        result[h] = dict(categories={name: group(mask) for name, mask in categories.items()},
            source_Q8=[dict(bin=i, **group(bins == i), categories={name: group(mask & (bins == i)) for name, mask in categories.items()}) for i in range(8)],
            masks_are_reference_point_posthoc=True, entries_are_local_signed_influence_not_row_losses=True)
    remainders = {}
    for name in ("wave", "QPSD"):
        approximation = sum(float(rows[h+"_signed_influence_"+name].sum()) for h in HYPOTHESES)
        actual = actual_loss_differences[name]
        remainders[name] = dict(signed_influence_sum=approximation, true_fixed_membership_anchor_minus_actual_loss=actual,
            linearization_remainder=actual-approximation)
    return dict(hypotheses=result, union_crop_changed_parents=int(union.sum()), total_parents=len(parent_bins),
        total_draws=len(parent), linearization=remainders, conditional_Q_memberships_frozen=True,
        no_per_sample_loss_or_causal_label=True, norm_contributions_not_added_as_percentages=True)


def directional_probes(S, R, carriers, observations, real, loss, context, memberships, gS, identities, store):
    import numpy as np
    import torch
    import mechanism012_losses as m
    from n01.model import ADC_MIN, ADC_MAX, ADC_LEVELS
    from n01.observation008 import observe_generated_carrier
    n, chunk, p = len(S), store.policy["forward_chunk_rows"], store.policy
    paths = p["probe"]["paths"]
    parent = np.asarray(identities["source_parent_row"])
    maximum = float(R.detach().abs().max())
    signed, arrays = [], {}
    if maximum == 0:
        for step in p["probe"]["step_sizes_ADC_LSB"]:
            for sign in p["probe"]["signs"]:
                signed.append(dict(step_LSB=step, sign=sign, delta_V=sign*step/ADC_LEVELS,
                    paths={name: None for name in paths}, defined=False, reason="all_zero_random_shape"))
        return dict(direction_defined=False, reason="all_zero_random_shape", global_max_abs_random_shape_V=maximum,
            d_AD_loss_per_V=None, signed_probes=signed, central_differences=[], actual_probe_ADC_rows=0,
            all_steps_reported=True, best_step_selected=False), arrays
    v = R.detach()/maximum
    require(not bool(v[:, :100].ne(0).any()) and float(m.canonical_charge008(v).abs().max()) <= 1e-10,
        "012 probe direction prefix/canonical-Q invariant failed")
    arrays["probe_carrier_direction"] = array(v)
    dad = float((gS.detach()*v).sum())
    reference_sorting = {}
    with torch.no_grad():
        for h in HYPOTHESES:
            c0 = carriers[h].detach()
            proxy0 = observations[h]["adc_carrier"].detach()+(c0.clamp(ADC_MIN, ADC_MAX)-c0.clamp(ADC_MIN, ADC_MAX))
            ref = m.gather_observation(proxy0, observations[h]["peak_index"], float(loss.q_floor), input_carrier=c0)
            require(torch.equal(ref["window_V"], observations[h]["window_V"].detach()), "012 S-reference forward mismatch")
            reference_sorting[h] = m.sorting_signature(loss, ref["window_V"], context, memberships[h])
    with torch.no_grad():
        for step in p["probe"]["step_sizes_ADC_LSB"]:
            for sign in p["probe"]["signs"]:
                delta = sign*step/ADC_LEVELS
                values, reports = {name: [] for name in paths}, {name: {} for name in paths}
                prefix = "probe_"+str(step).replace(".", "p")+"_"+("minus" if sign < 0 else "plus")
                for h in HYPOTHESES:
                    input_value = carriers[h].detach()+delta*v
                    hard_pieces = []
                    for start in range(0, n, chunk):
                        stop = min(start+chunk, n)
                        hard_pieces.append(store.call("signed_probe_hard_ADC_shared_fixed_moving", dict(ADC_hard_or_STE_rows=stop-start),
                            lambda: observe_generated_carrier(input_value[start:stop], ste=False, q_floor=float(loss.q_floor),
                                return_carrier=True, return_indices=True), step_LSB=step, sign=sign, hypothesis=h, start=start, stop=stop))
                    moving = concatenate_observations(hard_pieces)
                    fixed = m.gather_observation(moving["adc_carrier"], observations[h]["peak_index"], float(loss.q_floor), input_carrier=input_value)
                    c0 = carriers[h].detach()
                    proxy_adc = observations[h]["adc_carrier"].detach()+(input_value.clamp(ADC_MIN, ADC_MAX)-c0.clamp(ADC_MIN, ADC_MAX))
                    surrogate = m.gather_observation(proxy_adc, observations[h]["peak_index"], float(loss.q_floor), input_carrier=input_value)
                    require(torch.equal(moving["baseline100"], observations[h]["baseline100"].detach())
                        and torch.equal(surrogate["baseline100"], observations[h]["baseline100"].detach()), "012 probe prefix pairing changed")
                    for path, obs in zip(paths, (surrogate, fixed, moving)):
                        w, parts = m.wave_value(loss, obs["window_V"], real, context)
                        q = m.frozen_qpsd(loss, obs["window_V"], real, context, memberships[h])
                        score = float(w+2*q); require(np.isfinite(score), "012 nonfinite directional probe")
                        values[path].append(score)
                        native_q = m.native_qpsd(loss, obs["window_V"], real, context)
                        native_diagnostics = copy.deepcopy(loss.last_diagnostics)
                        native_membership = m.FrozenMembership012.from_reference(loss, obs["window_V"], real)
                        changed = obs["peak_index"] != observations[h]["peak_index"]
                        flag = m.numeric_flags(obs, input_value, loss)
                        current_sort = m.sorting_signature(loss, obs["window_V"], context, memberships[h])
                        reports[path][h] = dict(wave=float(w), QPSD_fixed=float(q), QPSD_native=float(native_q), proxy_fixed=score,
                            wave_subcomponents={k: float(value) for k, value in parts.items()},
                            ADC_changed_cells=int((obs["adc_carrier"] != observations[h]["adc_carrier"].detach()).sum()),
                            peak_changed_draws=int(changed.sum()), peak_changed_parents=int(len(np.unique(parent[array(changed)]))),
                            source_draw_denominator=n, source_parent_denominator=len(identities["source_parent_ids"]),
                            native_Qbin_changed_draws=int((native_membership.fake_bins != memberships[h].fake_bins).sum()),
                            native_support_changed_bins=[i for i in range(8) if native_membership.supported[i] != memberships[h].supported[i]],
                            native_conditional_diagnostics=native_diagnostics,
                            sorting=m.sorting_changes(reference_sorting[h], current_sort),
                            sorting_diagnostic_scope="fake permutation and ties only; W1 absolute-pair/cross-domain support cusps unmeasured",
                            W1_absolute_pair_or_cross_domain_cusp="unknown_not_excluded",
                            flag_counts={k: int(value.sum()) for k, value in flag.items() if value.dtype == torch.bool})
                        stem = prefix+"_"+path+"_"+h
                        for key in ("peak_index", "global_peak_index", "Q", "tail", "PSD"):
                            arrays[stem+"_"+key] = array(obs[key])
                        for key, value in flag.items():
                            arrays[stem+"_flag_"+key] = array(value)
                        arrays[stem+"_native_Qbin"] = array(native_membership.fake_bins)
                        arrays[stem+"_peak_changed"] = array(changed)
                        del current_sort
                signed.append(dict(step_LSB=step, sign=sign, delta_V=delta, defined=True,
                    paths={name: dict(mean_h_proxy=float(np.mean(values[name])), hypotheses=reports[name]) for name in paths}))
    differences = []
    for step in p["probe"]["step_sizes_ADC_LSB"]:
        minus = next(row for row in signed if row["step_LSB"] == step and row["sign"] == -1)
        plus = next(row for row in signed if row["step_LSB"] == step and row["sign"] == 1)
        row = {}
        for name in paths:
            fd = (plus["paths"][name]["mean_h_proxy"]-minus["paths"][name]["mean_h_proxy"])/(2*step/ADC_LEVELS)
            row[name] = dict(central_difference_loss_per_V=fd, AD_directional_derivative_loss_per_V=dad, signed_difference_from_AD=fd-dad,
                relative_to_AD=None if dad == 0 else (fd-dad)/abs(dad), zero_AD_relative_reason="zero_AD_derivative" if dad == 0 else None,
                scale_normalized_difference=(fd-dad)/max(abs(fd), abs(dad), torch.finfo(torch.float64).tiny))
        differences.append(dict(step_LSB=step, paths=row,
            moving_minus_fixed_central_difference=row[paths[2]]["central_difference_loss_per_V"]-row[paths[1]]["central_difference_loss_per_V"]))
    return dict(direction_defined=True, reason=None, global_max_abs_random_shape_V=maximum,
        direction_prefix100_exact_zero=True, direction_canonical_Q_absmax=float(m.canonical_charge008(v).abs().max()),
        d_AD_loss_per_V=dad, S_reference_forward_equal=True, S_reference_gradient_check="existing_queries1_2_analytic_pullback_no_extra_AD",
        signed_probes=signed, central_differences=differences, actual_probe_ADC_rows=18*n,
        all_steps_reported=True, best_step_selected=False, parameter_finite_difference=False, physical_candidate=False,
        no_scientific_pass_threshold=True, hard_fixed_moving_share_identical_ADC=True), arrays


def prepare_numeric(paths, policy, store, device):
    """Restore frozen numeric classes directly; never construct an engine."""
    import numpy as np
    import torch
    from n01.model008 import N01Response008
    from n01.model010 import FrozenDeterministicVoltageShape010
    from n01.response008 import make_extension, state_digest, statistics_digest, loss_tensor_statistics
    from n01.losses008_v2 import N01Loss008
    stats, domain, cfg = [read_json(paths[k]) for k in ("statistics", "data_statistics", "base_config")]
    require(stats.get("statistics_split") == domain.get("split") == stats["loss"]["fit_split"] == "TRAIN"
        and stats.get("DEV_waveforms_decoded") is False and domain.get("DEV_waveforms_decoded") is False
        and stats["prepared_dataset_sha256"] == store.binding["files_sha256"][str(paths["dataset"])]
        and stats["data_statistics_sha256"] == store.binding["files_sha256"][str(paths["data_statistics"])], "012 TRAIN statistics lineage mismatch")
    with np.load(paths["dataset"], allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in TRAIN_KEYS}
    shapes = dict(source=(6593, 256), source_carrier384=(6593, 384), real=(21047, 256), real_baseline100=(21047, 100),
        source_ids=(6593,), real_ids=(21047,), real_groups=(21047,), source_role=(6593,), real_role=(21047,))
    for name, shape in shapes.items():
        a = data[name]
        require(a.shape == shape and a.dtype.kind != "O" and (a.dtype.kind not in "fc" or np.isfinite(a).all()), "012 legal TRAIN shape/nonfinite mismatch")
    require(set(np.unique(data["source_role"])) == set(np.unique(data["real_role"])) == {"TRAIN"}, "012 non-TRAIN role")
    require(len(np.unique(data["source_ids"])) == 6593 and len(np.unique(data["real_ids"])) == 21047, "012 nonunique TRAIN IDs")
    with np.load(paths["shape010_eval_bank"], allow_pickle=False) as archive:
        names = ("selected_source_rows", "selected_source_ids", "real_reference_rows", "real_control_rows", "shape_z", "extension", "baseline_gaussian", "baseline_scale")
        bank = {name: archive[name].copy() for name in names}
    sr, rr = bank["selected_source_rows"], bank["real_reference_rows"]
    require(sr.shape == bank["selected_source_ids"].shape == (512,) and rr.shape == (2048,)
        and len(np.unique(sr)) == 512 and len(np.unique(rr)) == 2048 and (sr >= 0).all() and (sr < 6593).all()
        and (rr >= 0).all() and (rr < 21047).all(), "012 fixed bank index/cluster mismatch")
    require(np.array_equal(data["source_ids"][sr], bank["selected_source_ids"]), "012 source parent identity mismatch")
    expected_bank_shapes = dict(shape_z=(2048, 8), extension=(2048, 1, 284), baseline_gaussian=(2048, 1, 100), baseline_scale=(2048, 1, 1))
    for name, shape in expected_bank_shapes.items():
        require(bank[name].shape == shape and bank[name].dtype == np.float32 and np.isfinite(bank[name]).all(), "012 latent bank dtype/shape mismatch")
    parents = np.repeat(np.arange(512), 4)
    selected = data["source_carrier384"][sr]
    source = torch.as_tensor(selected[parents], dtype=torch.float64, device=device)
    real = torch.as_tensor(data["real"][rr], dtype=torch.float64, device=device)
    identities = dict(source_parent_ids=bank["selected_source_ids"], source_parent_rows=sr,
        source_parent_row=parents, REAL_reference_rows=rr, REAL_reference_ids=data["real_ids"][rr], REAL_reference_groups=data["real_groups"][rr])
    latents = {name: torch.as_tensor(bank[name], device=device) for name in expected_bank_shapes}
    anchors = {}
    for arm in ("frozen_A", "shape010_final"):
        anchors[arm] = {}
        for h in HYPOTHESES:
            with np.load(paths[arm+"_"+h+"_TRAIN"], allow_pickle=False) as archive:
                row = {name: archive[name].copy() for name in ("waveform_V", "pure_signal_V", "peak_index", "crop_start", "crop_stop", "Q", "source_parent_row", "source_parent_ids")}
            require(np.array_equal(row["source_parent_row"], parents) and np.array_equal(row["source_parent_ids"], bank["selected_source_ids"][parents]), "012 saved anchor identities differ")
            require(row["waveform_V"].shape == (2048, 256) and row["pure_signal_V"].shape == (2048, 384), "012 anchor waveform geometry changed")
            anchors[arm][h] = {name: torch.as_tensor(value, device=device) for name, value in row.items() if name not in ("source_parent_row", "source_parent_ids")}
    del data, bank
    store.phase = "checkpoint_restore"
    a = store.call("load_frozen_A_G19000", dict(checkpoint_loads=1), lambda: torch.load(paths["deterministic"], map_location="cpu", weights_only=False))
    require(a.get("schema") == "response008/v1" and a.get("arm") == "deterministic" and a.get("G_updates") == 19000
        and a.get("stage") == "low_learning_rate" and a.get("D_updates") == {h: 16000 for h in HYPOTHESES}
        and a.get("noise_updates") == 0 and a.get("population_learning") is False and a.get("config") == cfg
        and statistics_digest(a["statistics"]) == statistics_digest(stats), "012 original A metadata mismatch")
    baseline = N01Response008(stats["model"], "deterministic")
    baseline.load_state_dict(a["model"], strict=True)
    baseline.set_stage(a["stage"]); baseline.requires_grad_(False); baseline.eval(); baseline.to(device)
    n = store.call("load_frozen005_update5000", dict(checkpoint_loads=1), lambda: torch.load(paths["noise005"], map_location="cpu", weights_only=False))
    require(n.get("schema_version") == 5 and n.get("update") == 5000 and n.get("model_type") == "ObservedBaselineNoiseV3"
        and n.get("engineering") is False, "012 original005 metadata mismatch")
    extension = make_extension(n, device)
    require(state_digest(extension.model005.state_dict()) == a["noise005_state_sha256"], "012 A/005 reference mismatch")
    model = FrozenDeterministicVoltageShape010(baseline, seed=policy["initialization_seed"], amplitude_fraction=policy["voltage_gain"])
    head_metadata, final_head = {}, None
    for slot, update in (("head_initial", 0), ("head_1000", 1000), ("head_final", 2000)):
        cp = store.call("load_thin_head_"+str(update), dict(checkpoint_loads=1), lambda slot=slot: torch.load(paths[slot], map_location="cpu", weights_only=False))
        require(cp.get("schema") == "n01/shape010-checkpoint/v1" and cp.get("run_id") == "shape010_20261014"
            and cp.get("G_updates") == update and cp.get("complete_step_boundary") is True
            and cp.get("trainable_parameter_names") == list(HEAD_NAMES) and set(cp["new_head"]) == {"weight", "bias"}, "012 thin head schema/update mismatch")
        require(cp["new_head"]["weight"].shape == (8, 32, 5) and cp["new_head"]["bias"].shape == (8,), "012 thin head shape mismatch")
        for value in cp["new_head"].values():
            require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), "012 final head dtype/nonfinite mismatch")
        steps = [int(value["step"]) for value in cp["optimizer"]["state"].values()]
        require(steps == [] if update == 0 else len(steps) == 2 and all(step == update for step in steps), "012 thin optimizer step metadata mismatch")
        head_metadata[str(update)] = dict(head_state_sha256=state_digest(cp["new_head"]), optimizer_state_steps=steps,
            metadata_only=update != 2000, new_forward_rows=0)
        if update == 2000:
            final_head = cp
    model.basis_head.load_state_dict(final_head["new_head"], strict=True)
    model.requires_grad_(True); model.eval()
    initial_snapshot = tensor_snapshot(model, extension)
    require({k: initial_snapshot["states"][k] for k in ("baseline", "noise005", "extension")} == final_head["frozen_state_sha256"], "012 restored frozen state mismatch")
    loss = N01Loss008(loss_tensor_statistics(stats["loss"]), seed=policy["projection"]["training_seed"], evaluation_seed=policy["projection"]["seed"]).to(device)
    context = loss.create_evaluation_context()
    require(context.metadata() == {k: policy["projection"][k] for k in ("kind", "seed", "update", "fingerprint")}, "012 native projection fingerprint mismatch")
    store.before_snapshot = initial_snapshot
    store.json("FROZEN_STATE.json", dict(before=initial_snapshot, thin_head_metadata=head_metadata,
        optimizer_instantiated=False, only_final_head_loaded_for_forward=True, wrapper_moved_after_construction=False,
        projection_context=context.metadata(), trainable_parameter_names=list(HEAD_NAMES), parameters1288=True))
    store.npz("EVALUATION_PROJECTIONS.npz", **{name: array(value) for name, value in context.banks.items()})
    return model, extension, source, latents, real, anchors, loss, context, identities


def concatenate_observations(pieces):
    import torch
    keys = ("window_V", "baseline100", "baseline_mean_V", "adc_carrier", "peak_index", "global_peak_index",
        "crop_indices", "crop_start", "crop_stop", "Q", "tail", "PSD_raw", "PSD")
    result = {key: torch.cat([row[key] for row in pieces]) for key in keys}
    result["flags"] = {key: torch.cat([row["flags"][key] for row in pieces]) for key in pieces[0]["flags"]}
    return result


def equal_hard_ste(hard, ste):
    import torch
    for key in ("window_V", "baseline100", "baseline_mean_V", "adc_carrier", "peak_index", "global_peak_index",
                "crop_indices", "crop_start", "crop_stop", "Q", "tail", "PSD"):
        require(torch.equal(hard[key], ste[key].detach()), "012 hard/STE forward mismatch: "+key)
    raw_h, raw_s = hard["PSD_raw"], ste["PSD_raw"].detach()
    require(torch.equal(torch.isnan(raw_h), torch.isnan(raw_s)) and torch.equal(raw_h[~torch.isnan(raw_h)], raw_s[~torch.isnan(raw_s)]),
        "012 raw PSD defined/undefined mismatch")
    require(all(torch.equal(hard["flags"][key], ste["flags"][key]) for key in hard["flags"]), "012 hard/STE flags mismatch")


def run_diagnostic(model, extension, source, latents, real, anchors, loss, context, identities, store,
                   *, snapshotter=tensor_snapshot, protection_guard=None, native_distances=None):
    """One graph and exactly nine scalar/two VJP queries; injectable tiny fixtures."""
    import numpy as np
    import torch
    import mechanism012_losses as m
    from n01.response008 import fixed_noises, saturation_penalty
    from n01.observation008 import observe_generated_carrier
    p, n, chunk = store.policy, len(source), store.policy["forward_chunk_rows"]
    require(native_distances is not None or store.artificial, "012 actual run requires original final distance point gate")
    require(n == p["matched_rows"] and len(real) == n and n % p["draws_per_parent"] == 0, "012 registered row count mismatch")
    params = tuple(model.trainable_parameters())
    require(len(params) == 2 and sum(x.numel() for x in params) == 1288 and all(x.requires_grad and x.grad is None for x in params), "012 two-leaf parameter whitelist mismatch")
    before = snapshotter(model, extension)
    store.before_snapshot = before
    signals, base, randoms, regular_rms, regular_curvature = [], [], [], [], []
    noise_pieces = {h: [] for h in HYPOTHESES}
    store.phase = "single_shared_family_forward"
    for start in range(0, n, chunk):
        stop = min(start+chunk, n)
        noise = store.call("noise005_and_shared_extensions", dict(noise005_rows=stop-start),
            lambda: fixed_noises(extension, {k: latents[k][start:stop] for k in ("baseline_gaussian", "baseline_scale")},
                latents["extension"][start:stop]), start=start, stop=stop)
        output = store.call("single_A_and_final_head_family", dict(old_A_rows=stop-start, head_final_rows=stop-start),
            lambda: model.random_forward(source[start:stop], latents["shape_z"][start:stop], enabled=True), start=start, stop=stop)
        require(output["baseline_signal_forward_calls"] == 1 and output["extra_baseline_signal_forward_calls"] == 0,
            "012 extra hidden baseline forward")
        finite(output["signal_pre"], "signal")
        require(not bool(output["signal_pre"][:, :100].ne(0).any()) and not bool(output["random_shape_pre_V"][:, :100].ne(0).any()), "012 nonzero signal prefix")
        require(float(output["canonical_charge_identity_residual"].detach().abs().max()) <= 1e-10
            and float(output["canonical_random_shape_charge"].detach().abs().max()) <= 1e-10, "012 canonical charge invariant failed")
        signals.append(output["signal_pre"]); base.append(output["base_signal_pre_V"]); randoms.append(output["random_shape_pre_V"])
        regular_rms.append(output["regular_rms"]*(stop-start)/n)
        regular_curvature.append(output["regular_curvature"]*(stop-start)/n)
        for h in HYPOTHESES:
            noise_pieces[h].append(noise[h])
        if start == 0:
            store.json("FIRST_FAMILY.json", dict(store.base(), first_family_start=0, first_family_rows=stop-start,
                family_complete=True, finite_signal=True, three_hypotheses_share_signal_and_noise_prefix=True,
                input_scope="artificial" if store.artificial else "fixed_saved_TRAIN_bank"))
    # S is the actual common ancestor: observations slice THIS concat, never
    # concatenate S as an unused side branch after observing original chunks.
    S, A, R = torch.cat(signals), torch.cat(base), torch.cat(randoms)
    noises = {h: torch.cat(noise_pieces[h]) for h in HYPOTHESES}
    del signals, base, randoms, noise_pieces, output
    carriers = {h: S+noises[h] for h in HYPOTHESES}
    observations, hard, anchor, memberships, native_q_diagnostics, anchor_native_q_diagnostics = {}, {}, {}, {}, {}, {}
    floor = float(loss.q_floor)
    rows = {}; actual_w, actual_q, anchor_w, anchor_q_fixed, anchor_q_native = [], [], [], [], []
    wave_parts, anchor_parts = {}, {}
    store.phase = "hard_STE_actual_and_same_carrier_anchor"
    for h in HYPOTHESES:
        hp, sp = [], []
        for start in range(0, n, chunk):
            stop = min(start+chunk, n)
            with torch.no_grad():
                hv = store.call("actual_hard_ADC", dict(ADC_hard_or_STE_rows=stop-start),
                    lambda: observe_generated_carrier(carriers[h][start:stop], ste=False, q_floor=floor, return_carrier=True, return_indices=True),
                    hypothesis=h, start=start, stop=stop)
            sv = store.call("actual_STE_ADC", dict(ADC_hard_or_STE_rows=stop-start),
                lambda: observe_generated_carrier(carriers[h][start:stop], ste=True, q_floor=floor, return_carrier=True, return_indices=True),
                hypothesis=h, start=start, stop=stop)
            equal_hard_ste(hv, sv); hp.append(hv); sp.append(sv)
        hard[h], observations[h] = concatenate_observations(hp), concatenate_observations(sp)
        current = observations[h]
        av = anchors["frozen_A"][h]
        fv = anchors["shape010_final"][h]
        tolerance = p["point_replay_abs_tolerance"]
        require(float((A-av["pure_signal_V"]).abs().max()) <= tolerance and float((S.detach()-fv["pure_signal_V"]).abs().max()) <= tolerance,
            "012 saved pure A/final signal replay mismatch")
        require(torch.equal(hard[h]["peak_index"], fv["peak_index"]) and torch.equal(hard[h]["crop_start"], fv["crop_start"])
            and torch.equal(hard[h]["crop_stop"], fv["crop_stop"]), "012 saved final peak/crop replay mismatch")
        require(float((hard[h]["window_V"]-fv["waveform_V"]).abs().max()) <= tolerance
            and float((hard[h]["Q"]-fv["Q"]).abs().max()) <= tolerance, "012 saved hard waveform/Q replay mismatch")
        anchor[h] = m.gather_observation(current["adc_carrier"], av["peak_index"], floor, input_carrier=carriers[h])
        memberships[h] = m.FrozenMembership012.from_reference(loss, hard[h]["window_V"], real)
        w, parts = m.wave_value(loss, current["window_V"], real, context)
        q = m.native_qpsd(loss, current["window_V"], real, context)
        native_q_diagnostics[h] = copy.deepcopy(loss.last_diagnostics)
        wa, pa = m.wave_value(loss, anchor[h]["window_V"], real, context)
        qa_fixed = m.frozen_qpsd(loss, anchor[h]["window_V"], real, context, memberships[h])
        with torch.no_grad():
            qa_native = m.native_qpsd(loss, anchor[h]["window_V"], real, context)
            anchor_native_q_diagnostics[h] = copy.deepcopy(loss.last_diagnostics)
        actual_w.append(w); actual_q.append(q); anchor_w.append(wa); anchor_q_fixed.append(qa_fixed); anchor_q_native.append(qa_native)
        wave_parts[h], anchor_parts[h] = parts, pa
        if native_distances is not None:
            expected = native_distances[h]
            require(abs(float(w.detach())-expected["components"]["wave"]) <= tolerance
                and abs(float(q.detach())-expected["components"]["QPSD"]) <= tolerance, "012 native final distance replay mismatch")
        changed = hard[h]["peak_index"] != av["peak_index"]
        decomposition = m.q_decomposition(S, noises[h], current)
        anchor_decomposition = m.q_decomposition(S, noises[h], anchor[h])
        for label, obs in (("actual", current), ("anchor", anchor[h])):
            for key in ("peak_index", "global_peak_index", "crop_start", "crop_stop", "Q", "tail", "PSD", "PSD_raw", "window_V"):
                rows[h+"_"+label+"_"+key] = array(obs[key])
            for key, value in m.numeric_flags(obs, carriers[h], loss).items():
                rows[h+"_"+label+"_flag_"+key] = array(value)
        for key, value in m.peak_diagnostics(current["adc_carrier"]).items():
            rows[h+"_"+key] = array(value)
        for label, dec in (("actual", decomposition), ("anchor", anchor_decomposition)):
            rows.update({h+"_"+label+"_"+k: array(v) for k, v in dec.items()})
        rows[h+"_crop_changed"] = array(changed)
        rows[h+"_fake_reference_Qbin"] = array(memberships[h].fake_bins)
        with torch.no_grad():
            anchor_membership = m.FrozenMembership012.from_reference(loss, anchor[h]["window_V"], real)
        rows[h+"_anchor_native_Qbin"] = array(anchor_membership.fake_bins)
        rows[h+"_native_Qbin_migrated"] = array(anchor_membership.fake_bins != memberships[h].fake_bins)
        rows[h+"_anchor_native_support"] = np.asarray(anchor_membership.supported)
        rows[h+"_reference_support"] = np.asarray(memberships[h].supported)
        rows[h+"_native_QPSD_actual_anchor"] = np.asarray([float(q.detach()), float(qa_native)])
        rows[h+"_anchor_native_conditional_counts"] = np.asarray(anchor_membership.counts)
    Lw, Lq = torch.stack(actual_w).mean(), 2*torch.stack(actual_q).mean()
    Lrms, Lcurv = .01*sum(regular_rms), .01*sum(regular_curvature)
    Lsat = .01*torch.stack([saturation_penalty(carriers[h]) for h in HYPOTHESES]).mean()
    terms = dict(wave=Lw, QPSD=Lq, RMS=Lrms, curvature=Lcurv, saturation=Lsat)
    total = sum(terms.values())
    for name, value in terms.items():
        finite(value, "weighted loss "+name)
    store.json("POINT_REPLAY.json", dict(hard_STE_all_defined_values_and_undefined_flags_equal=True,
        pure_A_and_final_signal_replayed=True, hard_final_peak_crop_Q_replayed=True, frozen_projection=context.metadata(),
        weighted_terms={k: float(v.detach()) for k, v in terms.items()}, actual_native_QPSD_diagnostics=native_q_diagnostics,
        anchor_native_QPSD_diagnostics=anchor_native_q_diagnostics,
        native_final_distance_gate_passed=native_distances is not None or store.artificial,
        comparisons={h: dict(actual_wave=float(actual_w[i].detach()), actual_QPSD=float(actual_q[i].detach()),
            anchor_wave=float(anchor_w[i].detach()), anchor_QPSD_fixed=float(anchor_q_fixed[i].detach()),
            anchor_QPSD_native=float(anchor_q_native[i])) for i, h in enumerate(HYPOTHESES)}))
    store.point_passed = True
    store.phase = "nine_formal_scalar_AD_queries"
    Y = tuple(observations[h]["window_V"] for h in HYPOTHESES)
    gradients, endpoint, shared_gradient = {}, {}, {}
    def grad_scalar(name, scalar, intermediates=False):
        targets = params+Y+(S,) if intermediates else params
        def query():
            values = torch.autograd.grad(scalar, targets, retain_graph=True, create_graph=False, allow_unused=False)
            for value in values:
                finite(value, "AD "+name)
            return values
        values = store.call(name, dict(scalar_AD_queries=1), query)
        gradients[name] = m.flatten_parameters(values[:2])
        if intermediates:
            endpoint[name] = tuple(v.detach() for v in values[2:5])
            shared_gradient[name] = values[5].detach()
        return values
    grad_scalar("wave", Lw, intermediates=True)
    require(store.counters["scalar_AD_queries"] == 1 and store.counters["old_A_rows"] == store.counters["head_final_rows"] == store.counters["noise005_rows"] == n,
        "012 first proof requires actual whole forward and first formal query")
    require(snapshotter(model, extension) == before, "012 state changed before first formal proof")
    if protection_guard is not None:
        protection_guard()
    store.json("FIRST_DIAGNOSTIC_PROOF.json", dict(store.base(), finite_formal_AD=True, first_family_completed=True,
        first_family_rows=min(chunk, n), inputs_unchanged=True, frozen_state_unchanged=True,
        first_global_query="wave", first_query_gradient_L2=float(gradients["wave"].norm()),
        frozen_state_before=before, projection_context=context.metadata(), extra_pilot_AD_queries=0))
    grad_scalar("QPSD", Lq, intermediates=True)
    grad_scalar("RMS", Lrms); grad_scalar("curvature", Lcurv); grad_scalar("saturation", Lsat)
    grad_scalar("total", total)
    fixed_actual = 2*torch.stack([m.frozen_qpsd(loss, observations[h]["window_V"], real, context, memberships[h]) for h in HYPOTHESES]).mean()
    grad_scalar("QPSD_fixed_reference", fixed_actual)
    require(abs(float((fixed_actual-Lq).detach())) <= 1e-10, "012 fixed-membership reference value mismatch")
    fixed_grad_error = float((gradients["QPSD_fixed_reference"]-gradients["QPSD"]).norm())
    native_q_norm = float(gradients["QPSD"].norm())
    fixed_grad_relative = None if native_q_norm == 0 else fixed_grad_error/native_q_norm
    require(fixed_grad_error <= 1e-12 or (fixed_grad_relative is not None and fixed_grad_relative <= 1e-5),
        "012 fixed-membership reference parameter gradient mismatch")
    grad_scalar("anchor_wave", torch.stack(anchor_w).mean())
    grad_scalar("anchor_QPSD_fixed", 2*torch.stack(anchor_q_fixed).mean())
    store.phase = "two_formal_masked_endpoint_VJP_queries"
    for name in ("wave", "QPSD"):
        seeds = tuple(endpoint[name][i]*torch.as_tensor(rows[h+"_crop_changed"], device=S.device)[:, None] for i, h in enumerate(HYPOTHESES))
        def query():
            values = torch.autograd.grad(Y, params, grad_outputs=seeds, retain_graph=True, create_graph=False, allow_unused=False)
            for value in values:
                finite(value, "masked VJP "+name)
            return values
        values = store.call(name+"_crop_changed", dict(masked_endpoint_VJP_queries=1), query)
        gradients[name+"_changed"] = m.flatten_parameters(values)
        gradients[name+"_unchanged"] = gradients[name]-gradients[name+"_changed"]
        require(float((gradients[name+"_changed"]+gradients[name+"_unchanged"]-gradients[name]).norm()) <= 1e-10,
            "012 changed/unchanged reconstruction failed")
    gradients["regular"] = gradients["RMS"]+gradients["curvature"]+gradients["saturation"]
    analytical_errors = {}
    for name in ("wave", "QPSD"):
        analytic = sum(m.analytic_window_pullback(endpoint[name][i], observations[h]["peak_index"], carriers[h]) for i, h in enumerate(HYPOTHESES))
        error = float((analytic-shared_gradient[name]).abs().max())
        require(error <= 1e-10, "012 original STE/S-surrogate analytic Jacobian mismatch")
        analytical_errors[name] = error
    scalar_values = {k: float(v.detach()) for k, v in terms.items()}
    for i, h in enumerate(HYPOTHESES):
        delta = anchor[h]["window_V"].detach()-Y[i].detach()
        for name in ("wave", "QPSD"):
            rows[h+"_signed_influence_"+name] = array((endpoint[name][i]*delta).sum(1))
    parent_index = torch.as_tensor(identities["source_parent_row"], device=S.device)
    qsource, qcoordinate, qbins = m.source_q_strata(source[::p["draws_per_parent"]], loss)
    rows.update(identities)
    rows.update(source_canonical_Q=array(qsource), source_native_logQ=array(qcoordinate), source_native_Q8=array(qbins),
        signal_pre_V=array(S), base_signal_pre_V=array(A), random_shape_pre_V=array(R))
    influence_summary = summarize_influences(rows, identities, array(qbins), p["draws_per_parent"],
        dict(wave=float(torch.stack(anchor_w).mean().detach()-Lw.detach()),
             QPSD=float((2*torch.stack(anchor_q_fixed).mean()-Lq).detach())))
    store.phase = "six_signed_carrier_probes_without_NN_or_AD"
    probe, probe_rows = directional_probes(S, R, carriers, observations, real, loss, context, memberships,
        shared_gradient["wave"]+shared_gradient["QPSD"], identities, store)
    rows.update(probe_rows)
    after = snapshotter(model, extension)
    require(after == before and all(param.grad is None for param in params), "012 tensor/state/grad accumulator changed")
    if protection_guard is not None:
        protection_guard()
    context.validate()
    summary = dict(weighted_loss_values=scalar_values, total_loss=float(total.detach()),
        gradient_attribution=m.gradient_summary(gradients), fixed_membership_reference=dict(value_abs_error=abs(float((fixed_actual-Lq).detach())),
            parameter_gradient_abs_error=fixed_grad_error, parameter_gradient_relative_error=fixed_grad_relative,
            zero_relative_reason="zero_native_QPSD_gradient_norm" if fixed_grad_relative is None else None),
        S_reference_analytic_pullback_absmax=analytical_errors, influence=influence_summary, probe=probe,
        frozen_state_before=before, frozen_state_after=after, frozen_state_unchanged=True,
        head_tensors_unchanged=True, optimizer_created=0, optimizer_steps=0,
        interpretation="fixed evaluation-bank local new-head gradient and observation-operator diagnosis; no causal/response-match acceptance",
        formal_response_acceptance=False, response_match_established=False)
    gradient_arrays = {name: array(value) for name, value in gradients.items()}
    for name in ("wave", "QPSD"):
        gradient_arrays[name+"_shared_S_partial_gradient"] = array(shared_gradient[name])
        gradient_arrays[name+"_endpoint_gradients"] = np.stack([array(value) for value in endpoint[name]])
    gradient_arrays["parameter_names"], gradient_arrays["parameter_shapes"] = np.asarray(HEAD_NAMES), np.asarray(["8x32x5", "8"])
    store.npz("GRADIENTS.npz", **gradient_arrays)
    store.npz("ROW_MECHANISMS.npz", **rows)
    store.json("DIRECTIONAL_PROBE.json", probe)
    store.json("SUMMARY.json", dict(store.base(), **summary))
    mandatory = dict(old_A_rows=n, noise005_rows=n, head_final_rows=n, scalar_AD_queries=9, masked_endpoint_VJP_queries=2)
    require(all(store.counters[k] == v for k, v in mandatory.items()) and all(store.counters[k] == 0 for k in ZERO_KEYS), "012 mandatory scientific accounting incomplete")
    require(store.counters["ADC_hard_or_STE_rows"] == n*(24 if probe["direction_defined"] else 6), "012 actual ADC row accounting mismatch")
    store.phase = "diagnostic_complete"
    store.status(dict(engineering_checks_passed=True, frozen_state_unchanged=True, direction_defined=probe["direction_defined"]))
    store.guard()
    artifacts = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(store.root.iterdir()) if path.is_file()}
    store.json("COMPLETE.json", dict(store.base(), artifacts=artifacts, inputs_unchanged=True,
        frozen_state_unchanged=True, finite_formal_AD=True, engineering_checks_passed=True,
        head_tensors_unchanged=True, direction_defined=probe["direction_defined"], formal_response_acceptance=False,
        diagnostic_only=True, protocol_budget_is_maximum_not_dummy_consumption=True))
    return summary


def write_failure(store, error):
    store.phase = "failed"
    value = dict(store.base(), error_type=type(error).__name__, error=str(error), automatic_restart=False,
        completed_counters_are_durable_lower_bounds=True, incomplete_internal_NN_AD_ADC_completion="unknown",
        partial_operation_journal_preserved=True, native_success_not_claimed=True)
    store.json("FAILED.json", value)
    try:
        store.status(dict(error_type=type(error).__name__, error=str(error)))
    except BaseException as status_error:
        value["failure_status_write_error"] = repr(status_error)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    p = fixed_policy(args.policy)
    root = Path(p["remote_root"]).resolve()
    output = Path(args.output).resolve()
    require(os.name == "posix" and str(root) == "/home/rylan/projects/ej276d-n01/mechanism012_20261015"
        and output == root/"runs/primary/artifacts" and args.device == "cuda:0", "012 native registered paths/device required")
    require(Path(args.policy).resolve().is_relative_to(root) and Path(args.bindings).resolve().is_relative_to(root), "012 control path outside root")
    store = OperationStore012(output, p, device=args.device)
    def interrupted(number, _frame):
        raise InterruptedError("012 received signal "+str(number))
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        binding = bind_inputs(args.policy, args.bindings, p)
        store.binding = binding
        store.json("BINDING.json", dict(store.base(), **{k: v for k, v in binding.items() if k != "paths"}))
        store.status()
        allowed = sorted(os.sched_getaffinity(0))
        require(len(allowed) >= 12, "012 insufficient CPU affinity")
        os.sched_setaffinity(0, allowed[:12])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "12" if key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") else "1"
        sys.path.insert(0, str(root/"src"))
        # Complete binding/hash validation above precedes these imports.
        import torch
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.is_available(), "012 registered CUDA device unavailable")
        torch.cuda.set_device(args.device)
        total = torch.cuda.get_device_properties(args.device).total_memory
        fraction = p["resource_limits"]["GPU_GiB"]*1024**3/total
        require(0 < fraction <= 1, "012 GPU cap exceeds physical device")
        torch.cuda.set_per_process_memory_fraction(fraction, args.device)
        store.json("RUNTIME.json", dict(Python=sys.version, Torch=torch.__version__, CUDA=torch.version.cuda,
            device=args.device, GPU_name=torch.cuda.get_device_name(args.device), CPU_threads=torch.get_num_threads(),
            CPU_affinity=sorted(os.sched_getaffinity(0)), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=False, matmul_TF32=False, cudnn_TF32=False, CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            parameter_dtype="float32_head_and_original_A", signal_noise_loss_dtype="original_float64",
            forward_chunk_rows=128, allocator_limit_GiB=10))
        model, extension, source, latents, real, anchors, loss, context, identities = prepare_numeric(binding["paths"], p, store, args.device)
        final_eval = read_json(binding["paths"]["shape010_final_eval"])
        require(final_eval["projection_context"] == context.metadata(), "012 saved evaluation projection mismatch")
        run_diagnostic(model, extension, source, latents, real, anchors, loss, context, identities, store,
            protection_guard=lambda: verify_files(binding), native_distances=final_eval["distances"]["shape010_final"])
        print(json.dumps(dict(run_id=RUN_ID, status="diagnostic_artifacts_complete_native_exit_pending",
            completed_counts=store.counters, scientific_status="response_match_not_established"), allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        failure = write_failure(store, error)
        print(json.dumps(clean(failure), allow_nan=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
