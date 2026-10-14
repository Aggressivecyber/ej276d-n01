"""One registered saved-TRAIN-bank stability analysis; no NN/CP/gradients.

All frozen paths and digests are checked with the standard library before any
numeric decode or imported project code. Only the original explicit CPU loss
projection factory is called; all distances thereafter use NumPy CDF caches.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import threading
import time

RUN_ID = "stability011_20261014"
ARMS = ("frozen_A", "shape010_final")
HYPOTHESES = ("fast", "slow", "common_level")
ZERO_COUNTERS = dict(checkpoint_loads=0, model_instances=0, model_forward_rows=0,
    gradient_evaluations=0, G_updates=0, D_updates=0, noise_updates=0, population_updates=0, C2ST_updates=0)


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
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def relative_path(root, relative):
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts and "\\" not in relative
        and str(part) == relative and relative not in ("", "."), "Unsafe relative inventory path")
    path = (root/relative).resolve()
    require(path.is_relative_to(root.resolve()), "Inventory path escapes its root")
    return path


def fixed_policy(path):
    policy = read_json(path)
    require(policy.get("schema") == "n01/stability011-policy/v1" and policy.get("run_id") == RUN_ID, "Wrong011 policy identity")
    for key, expected in dict(source_parent_clusters=512, draws_per_parent=4, source_rows=2048,
            REAL_reference_rows=2048, REAL_reference_original_groups=419, REAL_control_rows=2048,
            bootstrap_replicates=500, resample_source_clusters=512, resample_REAL_groups=419,
            minimum_conditional_support=16, arms=list(ARMS), hypotheses=list(HYPOTHESES),
            RNG="numpy.random.PCG64", within_cluster_resampling=False, Q_stratification=False,
            fresh_latent_or_noise=False, REAL_control_bootstrapped=False, automatic_restart=False,
            overwrite_outputs=False, SOURCE_waveform_decode=False, DEV_numeric_read=False,
            failed_or_missing_support_replicate_replacement=False,
            allowed_dataset_members=["real", "real_ids", "real_groups", "source_ids"]).items():
        require(policy.get(key) == expected, "Frozen011 policy field changed: "+key)
    require(policy["seeds"] == dict(source_parent_bootstrap=2026101451, REAL_group_bootstrap=2026101452,
            evaluation_projection=2026101430), "Frozen011 seeds changed")
    require(policy.get("multiplicity_generation") == "Generator.integers for full (500,512) source and (500,419) REAL-group draw matrices, then rowwise bincount with minlength",
            "Frozen011 multiplicity algorithm changed")
    require(all(policy.get(key) == 0 for key in ZERO_COUNTERS), "011 must have zero model/CP/gradient/update budgets")
    require(policy["resource_limits"]["CPU_threads"] == 12 and policy["resource_limits"]["GPU_computation"] is False,
            "011 CPU/GPU budget changed")
    require(tuple(policy["resource_limits"][k] for k in ("job_tree_RSS_GiB", "wall_seconds", "output_MiB")) == (8, 7200, 256),
            "011 resource caps changed")
    require(policy["interval"]["quantiles"] == [.025, .975] and policy["interval"]["method"] == "linear",
            "Frozen percentile interval changed")
    require(policy["interval"].get("formal_frequentist_coverage_established") is False
        and policy["interval"].get("formal_significance_test") is False, "011 interval interpretation changed")
    return policy


def verify_bindings(policy_path, bindings_path, policy):
    """Metadata parsing/hash only; no NumPy/Torch import or archive decode."""
    binding = read_json(bindings_path)
    root = Path(policy["remote_root"]).resolve()
    oldroot = Path(policy["shape010_remote_root"]).resolve()
    reference = Path(policy["reference_remote_root"]).resolve()
    require(binding.get("schema") == "n01/stability011-bindings/v1" and binding.get("run_id") == RUN_ID,
            "Wrong011 bindings identity")
    require(binding.get("roots") == [str(root), str(oldroot), str(reference)], "011 allowed roots changed")
    require(sha(policy_path) == binding["policy_sha256"], "011 policy SHA mismatch")
    metadata = binding["source_manifest"]
    source_manifest_path = Path(metadata["path"]).resolve()
    require(source_manifest_path.is_relative_to(root) and sha(source_manifest_path) == metadata["sha256"], "011 source manifest mismatch")
    manifest = read_json(source_manifest_path)
    require(manifest.get("schema") == "n01/stability011-source/v1" and manifest.get("run_id") == RUN_ID, "Wrong011 source inventory")
    protected, source_hashes = {}, {}
    def add(path, digest, source=False):
        path = Path(path).resolve()
        require(any(path.is_relative_to(base) for base in (root, oldroot, reference)), "Bound path outside frozen allowed roots")
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid SHA256")
        require(str(path) not in protected or protected[str(path)] == digest, "Conflicting frozen digest")
        require(path.is_file() and sha(path) == digest, "Changed frozen file: "+str(path))
        protected[str(path)] = digest
        if source:
            source_hashes[str(path)] = digest
    add(policy_path, binding["policy_sha256"])
    add(source_manifest_path, metadata["sha256"])
    protected[str(Path(bindings_path).resolve())] = sha(bindings_path)
    for relative, digest in manifest["files"].items():
        add(relative_path(root, relative), digest, source=True)
    required = {"dataset", "statistics", "shape010_manifest", "shape010_bindings", "shape010_complete",
        "shape010_initial_eval", "shape010_final_eval", "shape010_eval_bank", "shape010_native_exit", "shape010_binding_receipt"}
    required.update(arm+"_"+h+"_TRAIN" for arm in ARMS for h in HYPOTHESES)
    require(required == set(binding["files"]) and len(manifest["files"]) == 63, "Frozen011 63source/16input inventory changed")
    require(manifest["bound_inputs"] == {k: v["sha256"] for k, v in binding["files"].items()}
        and manifest["policy_sha256"] == binding["policy_sha256"], "011 source/binding/policy chain")
    for item in binding["files"].values():
        add(item["path"], item["sha256"])
    paths = {name: Path(item["path"]).resolve() for name, item in binding["files"].items()}
    expected_paths = dict(dataset=reference/"data/window008/DATASET008.npz", statistics=reference/"data/window008/FIT_STATISTICS008.json",
        shape010_manifest=oldroot/"SHAPE010_SOURCE_MANIFEST.json", shape010_bindings=oldroot/"SHAPE010_BINDINGS.json",
        shape010_complete=oldroot/"runs/primary/artifacts/COMPLETE.json", shape010_initial_eval=oldroot/"runs/primary/artifacts/INITIAL_EVALUATION.json",
        shape010_final_eval=oldroot/"runs/primary/artifacts/FINAL_EVALUATION.json", shape010_eval_bank=oldroot/"runs/primary/artifacts/EVALUATION_BANK.npz",
        shape010_native_exit=oldroot/"runs/primary/supervisor/exit_receipt.json",
        shape010_binding_receipt=oldroot/"runs/primary/artifacts/BINDING.json")
    expected_paths.update({arm+"_"+h+"_TRAIN": oldroot/"runs/primary/artifacts"/(arm+"_"+h+"_TRAIN.npz")
        for arm in ARMS for h in HYPOTHESES})
    require(all(paths[key] == value for key, value in expected_paths.items()), "Frozen input paths changed")
    old_manifest, old_binding = read_json(paths["shape010_manifest"]), read_json(paths["shape010_bindings"])
    require(len(old_manifest["files"]) == 53 and len(old_manifest["protected_manifests"]) == 4 and len(old_binding["files"]) == 7,
            "010 immutable inventory size changed")
    require(sha(paths["shape010_manifest"]) == old_binding["source_manifest"]["sha256"], "010 manifest/binding chain")
    require(manifest["shape010_source_manifest_sha256"] == sha(paths["shape010_manifest"]), "011 protected010 source manifest chain")
    require(old_manifest["bound_inputs"] == {k: v["sha256"] for k, v in old_binding["files"].items()}, "010 immutable inputs chain")
    for relative, digest in old_manifest["files"].items():
        require(manifest["files"].get(relative) == digest, "011 import copy differs from frozen010 source "+relative)
        add(relative_path(oldroot, relative), digest, source=True)
    for relative, digest in old_manifest["protected_manifests"].items():
        require(manifest["files"].get(relative) == digest and manifest["protected_manifests"].get(relative) == digest,
                "011 protected manifest metadata copy changed")
        add(relative_path(root, relative), digest)
    for item in old_binding["files"].values():
        add(item["path"], item["sha256"])
    add(oldroot/"shape010_policy.json", old_binding["policy_sha256"])
    old_receipt = read_json(paths["shape010_binding_receipt"])
    require(all(old_receipt["sources_sha256"].get(str(relative_path(oldroot, name))) == digest for name, digest in old_manifest["files"].items())
        and all(old_receipt["files_sha256"].get(str(Path(item["path"]).resolve())) == item["sha256"] for item in old_binding["files"].values()),
        "010 actual binding receipt chain changed")
    complete = read_json(paths["shape010_complete"])
    require(complete["G_updates"] == 2000 and complete["counters"]["attempted_G_updates"] == 2000,
            "010 complete update budget differs")
    require(all(complete["counters"][key] == 0 for key in ("D_updates", "noise_updates", "population_updates", "C2ST_updates")),
            "010 includes an unregistered optimizer")
    require(len(complete["artifacts"]) == 29 and not (paths["shape010_complete"].parent/"FAILED.json").exists(), "010 inventory/failure state")
    for relative, item in complete["artifacts"].items():
        path = relative_path(paths["shape010_complete"].parent, relative)
        require(path.stat().st_size == item["bytes"], "010 artifact size changed")
        add(path, item["sha256"])
    native = read_json(paths["shape010_native_exit"])
    require(native["child_native_exit_code"] == native["exit_code"] == 0 and native["reason"] is None and native["error"] is None,
            "010 native completion failed")
    return dict(paths=paths, inputs_sha256=protected, sources_sha256=source_hashes,
        old_source_count=53, old_protected_manifest_count=4, old_immutable_input_count=7,
        before_numeric_decode=True, before_old_projection_factory_import=True)


def assert_unchanged(protection):
    for path, digest in protection["inputs_sha256"].items():
        require(sha(path) == digest, "Protected input changed during011: "+path)


class Store:
    def __init__(self, root, policy, artificial=False):
        self.root, self.policy, self.artificial = Path(root), policy, artificial
        require(not self.root.exists(), "Preserve existing011 output directory")
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.phase = "binding"
        self.saved = self.attempted = self.finished = 0
        self.point_passed = False
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.entered_ids, self.computed_ids, self.worker_error_ids = set(), set(), set()
        self.protection = None
        for name in ("bootstrap.jsonl", "bootstrap_attempts.jsonl", "bootstrap_computed.jsonl", "bootstrap_worker_errors.jsonl"):
            (self.root/name).open("x", encoding="utf8").close()

    def base(self):
        with self.lock:
            attempted, finished = self.attempted, self.finished
            unfinished = sorted(self.entered_ids-self.computed_ids-self.worker_error_ids)
        value = dict(UTC=now(), schema="n01/stability011-result/v1", run_id=RUN_ID, pid=os.getpid(),
            artificial_only=self.artificial, phase=self.phase, point_checks_passed=self.point_passed,
            bootstrap_replicates_completed=self.saved, bootstrap_replicates_attempted=attempted,
            bootstrap_replicates_computed_successfully=finished, counters=ZERO_COUNTERS.copy(),
            recorded_unfinished_replicate_ids=unfinished,
            durability=dict(attempts_journal="bootstrap_attempts.jsonl", computed_journal="bootstrap_computed.jsonl",
                saved_journal="bootstrap.jsonl", worker_errors_journal="bootstrap_worker_errors.jsonl",
                journals_flushed_and_fsynced_before_counters=True, status_is_snapshot_only=True,
                interrupted_unjournaled_compute_state="unknown_do_not_infer_zero"),
            DEV_numeric_read=False, SOURCE_waveform_decode=False,
            scientific_status="response_match_not_established", elapsed_seconds=time.monotonic()-self.started)
        if self.protection is not None:
            value.update(inputs_sha256=self.protection["inputs_sha256"], sources_sha256=self.protection["sources_sha256"])
        return value

    def json(self, name, value):
        with (self.root/name).open("x", encoding="utf8") as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False)
            stream.write("\n")

    def status(self, extra=None):
        value = self.base()
        if extra:
            value.update(extra)
        temporary = self.root/"status.json.tmp"
        require(not temporary.exists(), "Preserve unexpected status temporary file")
        with temporary.open("x", encoding="utf8") as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(self.root/"status.json")

    def npz(self, name, **arrays):
        import numpy as np
        with (self.root/name).open("xb") as stream:
            np.savez(stream, **arrays)

    def append_durable(self, name, value):
        with self.lock:
            with (self.root/name).open("a", encoding="utf8") as stream:
                stream.write(json.dumps(value, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def guard(self):
        limits = self.policy["resource_limits"]
        require(time.monotonic()-self.started <= limits["wall_seconds"], "011 wall budget exceeded")
        size = sum(path.stat().st_size for path in self.root.iterdir() if path.is_file())
        require(size <= limits["output_MiB"]*1024**2, "011 output budget exceeded")
        if sys.platform.startswith("linux"):
            import resource
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits["job_tree_RSS_GiB"]*1024**3,
                    "011 RSS budget exceeded")


def multiplicities(policy):
    import numpy as np
    result = []
    for key, count in (("source_parent_bootstrap", policy["source_parent_clusters"]),
                       ("REAL_group_bootstrap", policy["REAL_reference_original_groups"])):
        generator = np.random.Generator(np.random.PCG64(policy["seeds"][key]))
        indices = generator.integers(0, count, size=(policy["bootstrap_replicates"], count))
        rows = np.stack([np.bincount(row, minlength=count) for row in indices]).astype(np.uint16)
        require((rows.sum(axis=1) == count).all(), "Multiplicity cluster mass changed")
        result.append(rows)
    return tuple(result)


def summarize(records, policy):
    import numpy as np
    quantiles = policy["interval"]["quantiles"]
    def item(values):
        x = np.asarray(values, np.float64)
        return dict(mean=float(x.mean()), standard_deviation=float(x.std(ddof=1)), minimum=float(x.min()), maximum=float(x.max()),
            percentile_resampling_interval95=np.quantile(x, quantiles, method="linear").tolist())
    primary = {key: item([record["paired"][key] for record in records]) for key in ("S_A", "S_final", "Delta", "relative_contrast_percent")}
    coverage = all(record["paired"]["conditional_coverage_complete"] for record in records)
    lower, upper = primary["Delta"]["percentile_resampling_interval95"]
    if not coverage:
        classification = "coverage_limited"
    elif upper < 0:
        classification = "saved_TRAIN_bank_gain_stable_under_paired_parent_group_reweighting"
    elif lower > 0:
        classification = "endpoint_worsening_under_this_reweighting"
    else:
        classification = "stability_unresolved"
    secondary = {}
    for hi, hypothesis in enumerate(HYPOTHESES):
        entry = {}
        for name in ("wave", "QPSD", "proxy", "swd", "coordinate", "short_lag"):
            def extract(value):
                return value["components"][name] if name in ("wave", "QPSD") else value["proxy"] if name == "proxy" else value["wave_subcomponents"][name]
            a = [extract(record["metrics"][0][hi]) for record in records]
            f = [extract(record["metrics"][1][hi]) for record in records]
            entry[name] = dict(A=item(a), final=item(f), final_minus_A=item(np.asarray(f)-a))
        secondary[hypothesis] = entry
    return dict(primary=primary, secondary_marginal_diagnostics=secondary, classification=classification,
        all_replicate_conditional_coverage_complete=coverage,
        coverage_limited_replicates=sum(not r["paired"]["conditional_coverage_complete"] for r in records),
        interval_scope="fixed trained/selected models, saved TRAIN bank, four saved draws per parent, artificial REAL blocks and fixed projections",
        formal_frequentist_coverage_established=False, formal_significance_test=False, response_match_established=False)


def run_bootstrap(caches, parent_index, real_group_index, policy, store, first_proof, protection_guard):
    """Bounded thread queue; successful saved records and attempts stay distinct."""
    import numpy as np
    from stability011_metrics import paired_record
    source_counts, real_counts = multiplicities(policy)
    store.npz("MULTIPLICITY.npz", source_parent_multiplicity=source_counts, REAL_group_multiplicity=real_counts,
        generation=np.asarray("independent PCG64 Generator.integers(0,C,size=(B,C)); row np.bincount minlength=C"))
    count_sha = sha(store.root/"MULTIPLICITY.npz")
    total = policy["bootstrap_replicates"]
    results = [None]*total
    store.phase = "bootstrap"
    store.status(dict(registered_bootstrap_replicates=total, maximum_queued_or_running=policy["resource_limits"]["CPU_threads"]))
    def task(index):
        with store.lock:
            if store.stop.is_set():
                raise InterruptedError("011 cancelled before replicate entry")
            store.append_durable("bootstrap_attempts.jsonl", dict(event="entered", replicate_index=index, UTC=now()))
            store.attempted += 1
            store.entered_ids.add(index)
        try:
            fw, rw = source_counts[index][parent_index], real_counts[index][real_group_index]
            metrics = []
            for arm in caches:
                values = []
                for cached in arm:
                    if store.stop.is_set():
                        raise InterruptedError("011 stopped during replicate")
                    values.append(cached.evaluate(fw, rw))
                metrics.append(values)
            value = dict(replicate_index=index, metrics=metrics, paired=paired_record(metrics))
        except BaseException as error:
            with store.lock:
                store.append_durable("bootstrap_worker_errors.jsonl", dict(replicate_index=index,
                    error_type=type(error).__name__, error=str(error), UTC=now()))
                store.worker_error_ids.add(index)
            raise
        with store.lock:
            store.append_durable("bootstrap_computed.jsonl", value)
            store.finished += 1
            store.computed_ids.add(index)
        return value
    workers = policy["resource_limits"]["CPU_threads"]
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="011-CDF")
    pending, all_futures, next_index = {}, {}, 0
    successful_loop = False
    def persist(record, verify_and_prove=True):
        index = record["replicate_index"]
        if results[index] is not None:
            return
        store.append_durable("bootstrap.jsonl", record)
        results[index] = record
        store.saved += 1
        if verify_and_prove:
            store.guard()
            if store.saved == 1:
                protection_guard()
                proof = dict(store.base(), **first_proof, first_replicate_index=index,
                    first_replicate=record, source_REAL_counts_sha256=count_sha, inputs_unchanged=True)
                store.json("FIRST_BOOTSTRAP_PROOF.json", proof)
            store.status()
    try:
        while next_index < min(workers, total):
            future = executor.submit(task, next_index)
            pending[future] = all_futures[future] = next_index
            next_index += 1
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                record = future.result()
                persist(record)
                if next_index < total:
                    future = executor.submit(task, next_index)
                    pending[future] = all_futures[future] = next_index
                    next_index += 1
        successful_loop = True
    finally:
        store.stop.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if not successful_loop:
            # Preserve every durable successful result even if another future
            # failed before the main thread consumed it; never fill the budget.
            for future, index in all_futures.items():
                if future.done() and not future.cancelled() and future.exception() is None and results[index] is None:
                    persist(future.result(), verify_and_prove=False)
    require(store.saved == total and store.attempted == total and store.finished == total and all(v is not None for v in results),
            "011 incomplete bootstrap accounting")
    return results


def save_results(store, records):
    import numpy as np
    metrics = [record["metrics"] for record in records]
    arrays = {key: np.asarray([r["paired"][key] for r in records], np.float64)
        for key in ("S_A", "S_final", "Delta", "relative_contrast_percent")}
    for name in ("wave", "QPSD"):
        arrays[name] = np.asarray([[[v["components"][name] for v in arm] for arm in row] for row in metrics])
    for name in ("swd", "coordinate", "short_lag"):
        arrays[name] = np.asarray([[[v["wave_subcomponents"][name] for v in arm] for arm in row] for row in metrics])
    for name in ("proxy", "QPSD_global", "conditional_bins_used", "conditional_coverage_complete", "fake_rows", "real_rows",
                 "fake_nonpositive_q_rows", "real_nonpositive_q_rows", "fake_guarded_psd_rows", "real_guarded_psd_rows"):
        arrays[name] = np.asarray([[[v[name] for v in arm] for arm in row] for row in metrics])
    for name in ("fake_count", "real_count", "supported"):
        arrays["Qbin_"+name] = np.asarray([[[[c[name] for c in v["conditional_counts"]] for v in arm] for arm in row] for row in metrics])
    arrays["arms"], arrays["hypotheses"] = np.asarray(ARMS), np.asarray(HYPOTHESES)
    store.npz("BOOTSTRAP_METRICS.npz", **arrays)


def write_failure(store, error):
    store.stop.set()
    store.phase = "failed"
    failure = dict(store.base(), error_type=type(error).__name__, error=str(error), automatic_restart=False,
        partial_bootstrap_jsonl_preserved=True, native_success_not_claimed=True)
    if store.protection is not None:
        failure.update(inputs_sha256=store.protection["inputs_sha256"], sources_sha256=store.protection["sources_sha256"])
    store.json("FAILED.json", failure)
    store.status(dict(error_type=type(error).__name__, error=str(error)))
    return failure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    policy = fixed_policy(args.policy)
    root = Path(policy["remote_root"]).resolve()
    output = Path(args.output).resolve()
    require(os.name == "posix" and output == root/"runs/primary/artifacts", "Only fresh registered native011 output allowed")
    require(Path(args.policy).resolve().is_relative_to(root) and Path(args.bindings).resolve().is_relative_to(root), "011 control paths outside run root")
    allowed = sorted(os.sched_getaffinity(0))
    require(len(allowed) >= policy["resource_limits"]["CPU_threads"], "Insufficient allowed CPUs")
    os.sched_setaffinity(0, allowed[:policy["resource_limits"]["CPU_threads"]])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    store = Store(output, policy)
    def received(number, _frame):
        store.stop.set()
        raise InterruptedError("011 received signal "+str(number))
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, received)
    try:
        protection = verify_bindings(args.policy, args.bindings, policy)
        store.protection = protection
        store.json("FROZEN_INPUTS.json", dict(store.base(), **{k: v for k, v in protection.items() if k != "paths"}))
        store.status()
        # Imports and numeric decode start only after the complete digest guard.
        import numpy as np
        from stability011_metrics import Representation, WholeProxyCache, paired_record, WAVE_COLUMNS
        paths = protection["paths"]
        with np.load(paths["shape010_eval_bank"], allow_pickle=False) as bank:
            selected = {name: bank[name].copy() for name in ("selected_source_rows", "selected_source_ids", "real_reference_rows", "real_control_rows")}
        with np.load(paths["dataset"], allow_pickle=False) as archive:
            train = {name: archive[name].copy() for name in policy["allowed_dataset_members"]}
        require(train["real"].shape == (21047, 256) and train["real_ids"].shape == train["real_groups"].shape == (21047,)
            and train["source_ids"].shape == (6593,), "Prepared legal TRAIN shapes changed")
        require(all(np.isfinite(v).all() for v in train.values() if v.dtype.kind in "bifuc"), "Nonfinite permitted TRAIN member")
        require(selected["selected_source_rows"].shape == selected["selected_source_ids"].shape == (512,), "Saved source parent bank shape changed")
        require(np.array_equal(train["source_ids"][selected["selected_source_rows"]], selected["selected_source_ids"]), "Saved source identity mismatch")
        source_rows = selected["selected_source_rows"]
        ra, rb = selected["real_reference_rows"], selected["real_control_rows"]
        require((source_rows >= 0).all() and (source_rows < 6593).all() and (ra >= 0).all() and (ra < 21047).all()
            and (rb >= 0).all() and (rb < 21047).all(), "Saved bank index outside legal TRAIN")
        require(len(np.unique(source_rows)) == 512 and ra.shape == rb.shape == (2048,) and len(np.unique(ra)) == len(np.unique(rb)) == 2048
            and not np.intersect1d(ra, rb).size, "Saved bank repeats/REAL control overlap")
        real, control = train["real"][ra].astype(np.float64), train["real"][rb].astype(np.float64)
        group_ids, group_index = np.unique(train["real_groups"][ra], return_inverse=True)
        require(len(group_ids) == 419, "Saved REAL reference group count differs")
        parent_index = np.repeat(np.arange(512), 4)
        statistics = read_json(paths["statistics"])
        stats = statistics["loss"]
        require(statistics["statistics_split"] == stats["fit_split"] == "TRAIN" and stats["minimum_conditional_support"] == 16
            and statistics["prepared_dataset_sha256"] == protection["inputs_sha256"][str(paths["dataset"])] , "Frozen loss statistics provenance")
        store.npz("SELECTED_IDENTITIES.npz", selected_source_rows=source_rows, selected_source_ids=selected["selected_source_ids"],
            source_parent_row=parent_index, real_reference_rows=ra, real_reference_ids=train["real_ids"][ra],
            real_reference_original_groups=train["real_groups"][ra], REAL_group_ids=group_ids, REAL_group_index=group_index,
            real_control_rows=rb, real_control_ids=train["real_ids"][rb])
        del train
        store.phase = "projection_factory"
        sys.path.insert(0, str(root/"src"))
        import torch
        torch.set_num_threads(policy["resource_limits"]["CPU_threads"])
        from n01.losses008_v2 import create_evaluation_context
        before_rng = torch.get_rng_state().clone()
        context = create_evaluation_context(policy["seeds"]["evaluation_projection"])
        torch.set_num_threads(1)
        require(torch.equal(before_rng, torch.get_rng_state()), "Explicit projection factory changed global CPU RNG")
        banks = {name: value.numpy().astype(np.float64) for name, value in context.banks.items()}
        store.npz("EVALUATION_PROJECTIONS.npz", **{name: value.numpy() for name, value in context.banks.items()})
        initial, final = read_json(paths["shape010_initial_eval"]), read_json(paths["shape010_final_eval"])
        require(initial["projection_context"] == final["projection_context"], "010 paired projection metadata changed")
        native_context, actual_context = initial["projection_context"], context.metadata()
        require(all(actual_context[k] == native_context[k] for k in ("kind", "seed", "update")), "Projection kind/seed/update changed")
        store.json("PROJECTION_CONTEXT.json", dict(native=native_context, actual=actual_context,
            fingerprint_match=actual_context["fingerprint"] == native_context["fingerprint"], CPU_global_RNG_unchanged=True,
            factory_called_once=True, Torch_version=torch.__version__, CPU_affinity=sorted(os.sched_getaffinity(0))))
        store.phase = "point_recompute_and_cache"
        store.status()
        real_repr = Representation.create(real, stats, banks)
        tolerance = policy["point_recompute"]["maximum_wave_QPSD_absolute_error"]
        require(tolerance == 1e-7, "Frozen original-point tolerance changed")
        comparisons, maximum_error = {}, 0.
        def point(name, value, expected):
            nonlocal maximum_error
            errors = {key: abs(value["components"][key]-expected["components"][key]) for key in ("wave", "QPSD")}
            suberrors = {key: abs(value["wave_subcomponents"][key]-expected["wave_subcomponents"][key]) for key in ("swd", "coordinate", "short_lag")}
            maximum_error = max(maximum_error, *errors.values(), *suberrors.values())
            require(max(errors.values()) <= tolerance and max(suberrors.values()) <= tolerance, "Original point mismatch: "+name)
            original_counts = [{k: v for k, v in item.items() if k in ("bin", "fake_count", "real_count", "supported")}
                for item in expected["conditional_diagnostics"]["conditional_bins"]]
            require(value["conditional_counts"] == original_counts, "Original output-Q support membership mismatch: "+name)
            comparisons[name] = dict(recomputed=value, native_components=expected["components"], absolute_errors=errors,
                wave_subcomponent_absolute_errors=suberrors, conditional_counts_exact=True)
        rr = WholeProxyCache.create(Representation.create(control, stats, banks), real_repr)
        point("REAL_REAL", rr.evaluate(np.ones(2048, int), np.ones(2048, int)), initial["distances"]["REAL_REAL"])
        del rr, control
        caches, point_metrics = [], []
        for arm in ARMS:
            arm_caches, arm_metrics = [], []
            for hypothesis in HYPOTHESES:
                with np.load(paths[arm+"_"+hypothesis+"_TRAIN"], allow_pickle=False) as archive:
                    wave = archive["waveform_V"].copy()
                    parent = archive["source_parent_row"].copy()
                    identities = archive["source_parent_ids"].copy()
                require(wave.shape == (2048, 256) and np.isfinite(wave).all() and np.array_equal(parent, parent_index)
                    and np.array_equal(identities, np.repeat(selected["selected_source_ids"], 4)), "Saved waveform/parent identity mismatch")
                cached = WholeProxyCache.create(Representation.create(wave, stats, banks), real_repr)
                value = cached.evaluate(np.ones(2048, int), np.ones(2048, int))
                expected = (initial if arm == "frozen_A" else final)["distances"][arm][hypothesis]
                point(arm+"_"+hypothesis, value, expected)
                arm_caches.append(cached)
                arm_metrics.append(value)
                del wave
                store.guard()
            caches.append(arm_caches)
            point_metrics.append(arm_metrics)
        del real_repr, real
        point_receipt = dict(comparisons=comparisons, maximum_absolute_error=maximum_error, tolerance=tolerance,
            passed=True, paired_point=paired_record(point_metrics), projection_context=actual_context,
            fingerprint_match=actual_context["fingerprint"] == native_context["fingerprint"],
            waveform_projection_columns=WAVE_COLUMNS, cache_bytes=sum(c.bytes for arm in caches for c in arm))
        store.json("POINT_COMPARISON.json", point_receipt)
        store.point_passed = True
        proof = dict(point_comparison_sha256=sha(store.root/"POINT_COMPARISON.json"), projection_context=actual_context,
            inputs_sha256=protection["inputs_sha256"], sources_sha256=protection["sources_sha256"])
        records = run_bootstrap(caches, parent_index, group_index, policy, store, proof, lambda: assert_unchanged(protection))
        save_results(store, records)
        store.phase = "final_readback_and_hash_verification"
        assert_unchanged(protection)
        summary = summarize(records, policy)
        store.json("STABILITY_RESULT.json", dict(store.base(), **summary, point=point_receipt["paired_point"],
            point_comparison_sha256=proof["point_comparison_sha256"], original_group_description=policy["REAL_original_group_description"]))
        store.phase = "complete"
        store.status(dict(classification=summary["classification"]))
        store.guard()
        artifacts = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(store.root.iterdir()) if path.is_file()}
        store.json("COMPLETE.json", dict(store.base(), inputs_sha256=protection["inputs_sha256"], sources_sha256=protection["sources_sha256"],
            artifacts=artifacts, classification=summary["classification"], inputs_unchanged=True,
            registered_bootstrap_replicates=500, automatic_restart=False, failed_or_replaced_replicates=0,
            projection_context=actual_context, projection_fingerprint_match=point_receipt["fingerprint_match"]))
        return 0
    except BaseException as error:
        failure = write_failure(store, error)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
