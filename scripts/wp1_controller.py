"""WP1 stdlib controller. Source/private payloads and scientific exit differ.

Construction checks use manufactured files only. Native actions require frozen
SHA pins; no action retries, installs packages, loads arrays, or creates models.
"""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import uuid

REMOTE = "/home/rylan/projects/ej276d-n01/wp1_013_runtime_fix01_20261015"
OLD008 = "/home/rylan/projects/ej276d-n01/window008_resume01_20261013"
OLD010 = "/home/rylan/projects/ej276d-n01/shape010_20261014"
OLD011 = "/home/rylan/projects/ej276d-n01/stability011_20261014"
OLD012 = "/home/rylan/projects/ej276d-n01/mechanism012_20261015"
HOST = "rylan@100.82.231.102"
PYTHON = "/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python"
RUN_ID, CFG, MANIFEST = "wp1_013_20261015", "wp1_013_config.json", "WP1_SOURCE_MANIFEST.json"
CONTROLLER = "scripts/wp1_controller.py"
SOURCE_ARCHIVE, PRIVATE_ARCHIVE = "wp1_source.tar.gz", "wp1_private_inputs.tar"
PRIVATE_KEYS = ("g4_cache", "training_container", "generator_source", "provenance_report")
SOURCE_LIMIT, PRIVATE_LIMIT, COLLECTION_LIMIT = 16*1024**2, 64*1024**2, 1040*1024**2
SELF_ROOT = Path(__file__).resolve().parents[1] if __file__ not in ("<stdin>", "<string>") else Path(REMOTE)
STATE_NAMES = ("START_RECEIPT.json", "REMOTE_INSTALL_RECEIPT.json", "DISPATCH_ATTEMPT.json",
    "runs/primary/supervisor/launch.json", "runs/primary/supervisor/resource_status.json",
    "runs/primary/supervisor/exit_receipt.json", "runs/primary/artifacts/status.json",
    "runs/primary/artifacts/FIRST_WP1_PROOF.json", "runs/primary/artifacts/FAILED.json",
    "runs/primary/artifacts/COMPLETE.json")
_CORE = _OPS = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, "missing stdlib operations module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def core(expected_sha=None):
    """Reuse frozen012 SHA, tar, exclusive path and owned-supervisor helpers."""
    global _CORE
    if _CORE is None:
        candidates = [SELF_ROOT/"scripts/mechanism012_ops.py", Path(OLD012)/"scripts/mechanism012_ops.py"]
        path = next((path for path in candidates if path.is_file()), None)
        require(path is not None, "frozen012 stdlib core unavailable")
        if expected_sha is None:
            source = SELF_ROOT/"MECHANISM012_SOURCE_MANIFEST.json"
            expected_sha = json.loads(source.read_text("utf8"))["files"]["scripts/mechanism012_ops.py"]
        require(valid_sha(expected_sha) and sha(path) == expected_sha, "frozen012 stdlib core changed")
        _CORE = load_module(path, "wp1_reused012_stdlib")
        _CORE._wp1_loaded_sha = expected_sha
    elif expected_sha is not None:
        require(_CORE._wp1_loaded_sha == expected_sha, "conflicting legacy core identity")
    return _CORE


def ops():
    global _OPS
    if _OPS is None:
        _OPS = load_module(SELF_ROOT/"scripts/wp1_ops.py", "wp1_reused_operations")
    return _OPS


def named_record(path):
    path = core().unredirected(path)
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def validate_config(value, frozen=True):
    require(value.get("schema") == "n01/wp1-013-config/v1" and value.get("run_id") == RUN_ID
        and value.get("remote_root") == REMOTE, "wrong common WP1 identity")
    require(value.get("scientific_training_updates") == 0 and value.get("automatic_restart") is False,
        "WP1 is zero training and never restarts")
    require(not frozen or value.get("status") == "frozen", "WP1 config is not frozen")
    require(value["reference"]["theta0"]["G_updates"] == 2000
        and value["calibration"]["reference_checkpoint"]["G_updates"] == 2000,
        "reference must be saved positive G2000")
    require(len(value["inputs"]) == 22, "WP1 input inventory changed")
    required_zero = ("optimizer_created", "optimizer_steps", "G_updates", "D_updates", "noise_updates",
        "population_updates", "C2ST_updates", "new_AD_queries", "DEV_model_rows")
    require(all(value["budget_caps"].get(name) == 0 for name in required_zero), "zero-update/query scope changed")
    require(value["resource_limits"] == dict(CPU_threads=12, GPU_GiB=10, job_tree_RSS_GiB=8,
        wall_seconds=7200, output_MiB=1024, poll_seconds=10, TERM_grace_seconds=20), "registered machine limits changed")
    private = {name: row for name, row in value["inputs"].items() if row["path"].startswith(REMOTE+"/private_inputs/")}
    require(set(private) == set(PRIVATE_KEYS), "private payload must contain exactly four records")
    require(not frozen or (value.get("source_files") and CFG not in value["source_files"]
        and MANIFEST not in value["source_files"]), "config/manifest source cycle")
    return private


def checked_local(record, checked):
    path = core().unredirected(record["local_path"])
    require(path.is_file() and valid_sha(record["sha256"]), "bad local immutable file")
    key = str(path)
    if key in checked:
        require(checked[key] == record["sha256"], "conflicting local aliases")
    else:
        require(sha(path) == record["sha256"], "local immutable bytes changed: "+key)
        checked[key] = record["sha256"]
    require(path.stat().st_size == record["bytes"], "local immutable size changed")
    return path


def protections(repo, checked):
    result = {}
    for name, root, count in (("SHAPE010_SOURCE_MANIFEST.json", OLD010, 53),
                            ("STABILITY011_SOURCE_MANIFEST.json", OLD011, 63),
                            ("MECHANISM012_SOURCE_MANIFEST.json", OLD012, 69)):
        manifest_path = core().unredirected(repo/name)
        manifest = core().read_json(manifest_path)
        require(len(manifest["files"]) == count, "old source manifest inventory changed")
        for relative, digest in manifest["files"].items():
            core().safe_name(relative)
            local = core().unredirected(repo/relative)
            row = dict(path=root+"/"+relative, local_path=str(local), sha256=digest, bytes=local.stat().st_size)
            checked_local(row, checked)
            result[root.rsplit("/", 1)[1]+":"+relative] = row
    return result


def add_regular(tar, path, name):
    core().safe_name(name)
    require(core().unredirected(path).is_file(), "package source is not a regular file")
    tar.add(path, arcname=name, recursive=False)


def index(path, maximum):
    # The old verifier is reused; add a strict sparse/type and Windows collision check.
    folded = set()
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(), "nonregular/sparse tar entry")
            name = core().safe_name(member.name)
            require(name.casefold() not in folded, "case-colliding tar members")
            folded.add(name.casefold())
    return core().tar_index(path, maximum)


def read_bundle(source_archive, private_archive, pins):
    require(sha(source_archive) == pins["source_package_sha256"] and sha(private_archive) == pins["private_package_sha256"], "package SHA differs")
    source_index, private_index = index(source_archive, SOURCE_LIMIT), index(private_archive, PRIVATE_LIMIT)
    with tarfile.open(source_archive, "r:*") as archive:
        raw_config = archive.extractfile(CFG).read()
        raw_manifest = archive.extractfile(MANIFEST).read()
    require(hashlib.sha256(raw_config).hexdigest() == pins["config_sha256"]
        and hashlib.sha256(raw_manifest).hexdigest() == pins["source_manifest_sha256"], "external control SHA differs")
    config, manifest = json.loads(raw_config), json.loads(raw_manifest)
    private = validate_config(config)
    require(manifest["schema"] == "n01/wp1-source/v1" and manifest["run_id"] == RUN_ID
        and manifest["config_sha256"] == pins["config_sha256"], "WP1 source manifest differs")
    require(set(source_index) == set(manifest["files"]) | {CFG, MANIFEST}, "source package inventory differs")
    require(set(config["source_files"]) == set(manifest["files"]), "common source inventory differs")
    for name, record in manifest["files"].items():
        require(not name.startswith("private_inputs/") and not name.endswith((".npz", ".pt", ".pth", ".bin")), "private numeric payload leaked into source")
        require(source_index[name] == record and config["source_files"][name]["sha256"] == record["sha256"]
            and config["source_files"][name]["bytes"] == record["bytes"]
            and config["source_files"][name]["path"] == REMOTE+"/"+name, "source bytes/path differ")
    require(manifest["files"][CONTROLLER]["sha256"] == pins["controller_sha256"], "controller SHA differs")
    expected_private = {PurePosixPath(row["path"]).relative_to(PurePosixPath(REMOTE)).as_posix():
        dict(bytes=row["bytes"], sha256=row["sha256"]) for row in private.values()}
    require(private_index == expected_private and all(name.startswith("private_inputs/") for name in private_index), "private inventory differs or crosses source payload")
    return config, manifest, source_index, private_index


def freeze(repo, output, expected_draft_sha, artificial=False):
    repo = core().unredirected(repo)
    require(sha(repo/CFG) == expected_draft_sha, "draft configuration changed before freeze")
    config = copy.deepcopy(core().read_json(repo/CFG))
    validate_config(config, frozen=False)
    checked = {str(repo/CFG): expected_draft_sha}
    require(sha(repo/config["authority"]["path"]) == config["authority"]["sha256"], "raw user authority changed")
    old = core().read_json(repo/"MECHANISM012_SOURCE_MANIFEST.json")
    require(len(old["files"]) == 69, "old69 inventory differs")
    dependency = ops().REUSED_CDF_FILE
    stability = repo/"STABILITY011_SOURCE_MANIFEST.json"
    require(sha(stability) == ops().STABILITY_MANIFEST_SHA, "original011 source manifest changed")
    dependency_sha = core().read_json(stability)["files"].get(dependency)
    require(dependency_sha == ops().REUSED_CDF_SHA, "original011 CDF source binding changed")
    names = set(old["files"]) | set(ops().NEW_FILES) | {CONTROLLER,
        "SHAPE010_SOURCE_MANIFEST.json", "STABILITY011_SOURCE_MANIFEST.json", "MECHANISM012_SOURCE_MANIFEST.json"}
    source_records = {}
    for name in sorted(names):
        core().safe_name(name)
        local = core().unredirected(repo/name)
        record = named_record(local)
        if name in old["files"]:
            require(record["sha256"] == old["files"][name], "frozen69 source bytes changed")
        if name == dependency:
            require(record["sha256"] == dependency_sha, "original011 CDF helper bytes changed")
        source_records[name] = record
        checked[str(local)] = record["sha256"]
    for collection in (config["inputs"], config["contracts"]):
        for row in collection.values():
            checked_local(row, checked)
    config["protection_files"] = protections(repo, checked)
    config["source_files"] = {name: dict(path=REMOTE+"/"+name, **record) for name, record in source_records.items()}
    config["status"] = "frozen"
    config["frozen_UTC"] = core().now()
    output = core().fresh_directory(output, local=True)
    core().write_new(output/CFG, config)
    config_sha = sha(output/CFG)
    manifest = dict(schema="n01/wp1-source/v1", run_id=RUN_ID, frozen_UTC=config["frozen_UTC"],
        config_sha256=config_sha, files=source_records, previous_mechanism012_manifest_sha256=sha(repo/"MECHANISM012_SOURCE_MANIFEST.json"),
        source_private_payloads_are_separate=True, private_input_count=4, before_scientific_decode=True)
    core().write_new(output/MANIFEST, manifest)
    source_archive, private_archive = output/SOURCE_ARCHIVE, output/PRIVATE_ARCHIVE
    with tarfile.open(source_archive, "x:gz") as archive:
        for name in sorted(source_records):
            add_regular(archive, repo/name, name)
        for name in (CFG, MANIFEST):
            add_regular(archive, output/name, name)
    with tarfile.open(private_archive, "x:") as archive:
        for key in PRIVATE_KEYS:
            row = config["inputs"][key]
            name = PurePosixPath(row["path"]).relative_to(PurePosixPath(REMOTE)).as_posix()
            add_regular(archive, row["local_path"], name)
    pins = dict(source_package_sha256=sha(source_archive), private_package_sha256=sha(private_archive),
        config_sha256=config_sha, source_manifest_sha256=sha(output/MANIFEST), controller_sha256=source_records[CONTROLLER]["sha256"],
        legacy_ops_sha256=old["files"]["scripts/mechanism012_ops.py"])
    _, _, source_index, private_index = read_bundle(source_archive, private_archive, pins)
    for path, digest in checked.items():
        require(sha(path) == digest, "file changed during source/private freeze")
    receipt = dict(schema="n01/wp1-freeze/v1", run_id=RUN_ID, UTC=core().now(), **pins,
        source_package=str(source_archive), private_package=str(private_archive),
        source_package_bytes=source_archive.stat().st_size, private_package_bytes=private_archive.stat().st_size,
        source_file_count=len(source_records), private_file_count=4, source_inventory=source_index,
        private_inventory=private_index, protected_source_records=len(config["protection_files"]),
        all_local_original_files_unchanged=True, artificial_only=artificial, scientific_started=False,
        numerical_array_decodes=0, checkpoint_loads=0, model_forwards=0, autograd_queries=0)
    core().write_new(output/"FREEZE_RECEIPT.json", receipt)
    return receipt


def load_freeze(directory):
    directory = core().unredirected(directory)
    receipt = core().read_json(directory/"FREEZE_RECEIPT.json")
    require(receipt["schema"] == "n01/wp1-freeze/v1" and receipt["run_id"] == RUN_ID, "wrong freeze receipt")
    bundle = read_bundle(directory/SOURCE_ARCHIVE, directory/PRIVATE_ARCHIVE, receipt)
    return receipt, bundle


def verify_runtime(pins, partial=False):
    root = core().unredirected(REMOTE)
    require(root.is_dir() and sha(root/CFG) == pins["config_sha256"]
        and sha(root/MANIFEST) == pins["source_manifest_sha256"]
        and sha(root/CONTROLLER) == pins["controller_sha256"], "installed WP1 controls changed")
    config, manifest = core().read_json(root/CFG), core().read_json(root/MANIFEST)
    validate_config(config)
    require(manifest["config_sha256"] == pins["config_sha256"]
        and set(manifest["files"]) == set(config["source_files"]), "installed source manifest differs")
    errors = []
    try:
        value, _, checked = ops().gate(root/CFG, pins["config_sha256"])
        require(value == config, "installed gate returned different config")
        for name, record in manifest["files"].items():
            require(named_record(root/name) == record, "installed source inventory changed")
    except Exception as error:
        if not partial:
            raise
        checked = {}; errors.append(repr(error))
    return config, manifest, checked, errors


def counters(config, attempted, completed):
    require(set(attempted) == set(completed) == set(config["budget_caps"]), "counter schema differs")
    for name, cap in config["budget_caps"].items():
        require(type(attempted[name]) is int and type(completed[name]) is int
            and 0 <= completed[name] <= attempted[name] <= cap, "counter cap or ordering differs: "+name)


def runtime_hashes(config, config_sha):
    expected = {REMOTE+"/"+CFG: config_sha}
    for records in (config["source_files"], config["inputs"], config["contracts"], config.get("protection_files", {})):
        for record in records.values():
            require(record["path"] not in expected or expected[record["path"]] == record["sha256"], "conflicting runtime aliases")
            expected[record["path"]] = record["sha256"]
    return expected


def proof_valid(state, config, require_current_boot=True):
    proof = state.get("runs/primary/artifacts/FIRST_WP1_PROOF.json")
    start, launch = state.get("START_RECEIPT.json"), state.get("runs/primary/supervisor/launch.json")
    require(isinstance(proof, dict) and isinstance(start, dict) and isinstance(launch, dict), "formal first family proof not yet available")
    require(proof.get("run_id") == RUN_ID and proof.get("artificial_only") is False
        and proof.get("actual_science") is True and proof.get("first_family_rows") == 128
        and proof.get("reference_head_G_updates") == 2000 and proof.get("frozen_state_unchanged") is True,
        "not an actual frozen G2000 first128 family")
    require(proof.get("first_family_completed", proof.get("A_theta_noise_family_completed")) is True
        and proof.get("all_hypotheses", proof.get("all_three_hypotheses")) in (True, ["fast", "slow", "common_level"])
        and proof.get("inputs_unchanged") is True and proof.get("extra_pilot_rows") == 0,
        "formal family/hash or extra-pilot contract differs")
    require(start["supervisor_pid"] == launch["supervisor_pid"] and launch["child_pid"] == proof["pid"], "first family PID chain differs")
    require(not require_current_boot or start["boot_id"] == state["current_boot_id"], "first proof belongs to another boot")
    counters(config, proof["attempted_counters"], proof["counters"])
    counts = proof["counters"]
    require(all(counts[name] >= 128 for name in ("old_A_rows", "head_actual_parameter_forward_rows", "noise005_rows"))
        and counts["hard_ADC_rows"] >= 768, "first family entered but not completed")
    require(not proof.get("inflight"), "first proof has unresolved operation")
    expected = state["_verified_freeze"]["immutable_files_sha256"]
    require(expected and all(proof["immutable_files_sha256"].get(path) == digest for path, digest in expected.items()), "first proof immutable identities differ")
    return dict(status="actual_first128_family_verified", supervisor_pid=start["supervisor_pid"], child_pid=proof["pid"],
        reference_head_G_updates=2000, completed_counters=counts, extra_pilot_rows=0, scientific_acceptance=False)


def complete_valid(state, config):
    native = state.get("runs/primary/supervisor/exit_receipt.json")
    complete = state.get("runs/primary/artifacts/COMPLETE.json")
    require(isinstance(native, dict) and isinstance(complete, dict), "native exit and producer COMPLETE are distinct and both required")
    child = native.get("child_native_returncode", native.get("child_native_exit_code"))
    supervisor = native.get("supervisor_native_returncode", native.get("exit_code"))
    require(type(child) is int and child == 0 and type(supervisor) is int and supervisor == 0
        and native.get("reason") is None and native.get("error") is None, "nonzero or resource-stopped native job")
    proof_valid(state, config, require_current_boot=False)
    require(complete.get("run_id") == RUN_ID and complete.get("artificial_only") is False
        and complete.get("engineering_checks_passed") is True and complete.get("inputs_unchanged") is True
        and complete.get("frozen_state_unchanged") is True and not state.get("runs/primary/artifacts/FAILED.json"), "producer failure or incomplete checks")
    counters(config, complete["attempted_counters"], complete["counters"])
    require(complete.get("all_SOURCE_TRAIN_parents") == 6593 and complete.get("all_REAL_TRAIN_rows") == 21047
        and complete.get("banks") == 2 and complete.get("reference_head_G_updates") == 2000
        and complete.get("probe_registered_states") == 30 and complete.get("calibration_cases") == 21,
        "full WP1 population/calibration/registered states missing")
    defined = complete.get("probe_defined_states")
    require(type(defined) is int and defined in (0, 6, 12, 18, 24, 30), "defined direction state count differs")
    required_counts = dict(old_A_rows=52744+8192*defined, head_actual_parameter_forward_rows=52744+8192*defined,
        noise005_rows=52744, hard_ADC_rows=832560+24576*defined, all_path_gather_rows=832560+73728*defined, checkpoint_loads=3)
    require(all(complete["counters"][name] == value for name, value in required_counts.items()), "completion work/counter formula differs")
    require(complete["counters"]["head_total_kernel_rows"] == complete["counters"]["head_actual_parameter_forward_rows"]
        +complete["counters"]["head_linear_direction_map_rows"], "actual and linear head kernel accounting differs")
    require(not complete.get("inflight") and not state["durable_child_progress"]["unknown_inflight_ids"]
        and not state["durable_child_progress"]["failed_ids"] and not state["durable_child_progress"]["truncated_tail"], "unresolved durable operation")
    for name, value in complete["counters"].items():
        require(state["durable_child_progress"]["completed_counts"].get(name, 0) == value, "producer/journal completed counts differ")
        require(state["durable_child_progress"]["entered_counts"].get(name, 0) == complete["attempted_counters"][name], "producer/journal attempted counts differ")
    expected = state["_verified_freeze"]["immutable_files_sha256"]
    require(all(complete["immutable_files_sha256"].get(path) == digest for path, digest in expected.items()), "completion immutable identities differ")
    status = state.get("runs/primary/artifacts/status.json")
    require(isinstance(status, dict) and status["counters"] == complete["counters"], "terminal status counts differ")
    require(native["supervisor_pid"] == state["START_RECEIPT.json"]["supervisor_pid"]
        and native["child_pid"] == complete["pid"] == state["runs/primary/supervisor/launch.json"]["child_pid"], "terminal PID chain differs")
    return dict(status="native_and_producer_complete", native_exit_code=0, counters=complete["counters"],
        method_choice=complete.get("method_choice"), scientific_status="response_match_not_established", scientific_acceptance=False)


def native_snapshot(pins, partial=False):
    config, _, checked, errors = verify_runtime(pins, partial)
    root = Path(REMOTE); state = {}
    for name in STATE_NAMES:
        path = core().unredirected(root/name)
        try:
            state[name] = core().read_json(path) if path.is_file() else None
        except Exception as error:
            if not partial:
                raise
            state[name] = None; errors.append(name+": "+repr(error))
    try:
        progress = ops().durable_progress(root/"runs/primary/artifacts")
    except Exception as error:
        if not partial:
            raise
        progress = dict(progress_error=repr(error), actual_completion="unknown"); errors.append(repr(error))
    try:
        resource = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True, timeout=10).strip()
    except Exception as error:
        if not partial:
            raise
        resource = None; errors.append("GPU metadata unavailable: "+repr(error))
    state.update(UTC=core().now(), current_boot_id=core().boot_id(), durable_child_progress=progress,
        protection_errors=errors, GPU_apps_current=resource, partial_evidence_only=partial,
        _verified_freeze=dict(config_sha256=pins["config_sha256"], source_manifest_sha256=pins["source_manifest_sha256"], immutable_files_sha256=checked))
    return config, state


def native_start(pins):
    require(os.name == "posix", "native Linux only")
    import fcntl
    root = core().unredirected(REMOTE)
    require(not any(os.path.lexists(root/name) for name in ("runs/primary", "START_RECEIPT.json", "DISPATCH_ATTEMPT.json",
        "supervisor_console.log", "tmp", "cache")), "preserve prior dispatch; no restart")
    config, _, _, _ = verify_runtime(pins)
    lock = core().unredirected(root.parent/".n01_scientific_dispatch.lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = core().runtime_preflight(root, config)
        active = [line for line in subprocess.check_output(["ps", "-eo", "pid,args"], text=True, timeout=10).splitlines()
            if "scripts/run_wp1_013.py" in line or "scripts/wp1_ops.py supervise" in line]
        require(not active, "another WP1 process is active")
        require(Path(PYTHON).is_file(), "registered native Python missing")
        core().write_new(root/"DISPATCH_ATTEMPT.json", dict(run_id=RUN_ID, UTC=core().now(), **pins,
            preflight=preflight, budget_caps=config["budget_caps"], automatic_restart=False))
        for name in ("tmp", "cache"):
            core().unredirected(root/name).mkdir(mode=0o700, exist_ok=False)
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONPATH=str(root/"src"),
            TMPDIR=str(root/"tmp"), XDG_CACHE_HOME=str(root/"cache"), CUDA_VISIBLE_DEVICES="0",
            OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12", NUMEXPR_NUM_THREADS="12")
        command = [PYTHON, "-B", str(root/"scripts/wp1_ops.py"), "supervise", "--config-sha256", pins["config_sha256"], "--lock-fd", str(fd)]
        child = core().spawn_supervisor(command, root, fd, env)
        result = dict(schema="n01/wp1-start/v1", run_id=RUN_ID, UTC=core().now(), **pins,
            supervisor_pid=child.pid, command=command, boot_id=preflight["boot_id"],
            exclusive_scientific_lock=str(lock), inherited_lock_fd=fd, lock_inherited_by_supervisor=True,
            first_family_verified=False, automatic_restart=False, scientific_acceptance=False, preflight=preflight)
        core().write_new(root/"START_RECEIPT.json", result)
        return result
    finally:
        os.close(fd)


def archive_inventory(state, config, partial=False):
    root = core().unredirected(REMOTE)
    if partial:
        start = state.get("START_RECEIPT.json")
        require(isinstance(start, dict), "no owned dispatch for partial collection")
        if start["boot_id"] == state["current_boot_id"]:
            launch = state.get("runs/primary/supervisor/launch.json") or {}
            for pid in (start.get("supervisor_pid"), launch.get("child_pid")):
                require(type(pid) is not int or not Path("/proc", str(pid)).exists(), "owned job still running; status only")
        verified = dict(status="partial_evidence_only", native=state.get("runs/primary/supervisor/exit_receipt.json"),
            durable_progress=state["durable_child_progress"], actual_inflight_completion="unknown",
            protection_errors=state["protection_errors"], scientific_acceptance=False)
    else:
        verified = complete_valid(state, config)
    names = []
    for directory, folders, files in os.walk(root/"runs/primary", followlinks=False):
        require(not any((Path(directory)/name).is_symlink() for name in folders+files), "redirected owned output")
        names.extend((Path(directory)/name).relative_to(root).as_posix() for name in files)
    for name in (CFG, MANIFEST, "START_RECEIPT.json", "REMOTE_INSTALL_RECEIPT.json", "DISPATCH_ATTEMPT.json", "supervisor_console.log"):
        if (root/name).is_file():
            names.append(name)
    inventory = {}
    for name in sorted(set(names)):
        core().safe_name(name)
        require(not name.endswith((".pt", ".pth", ".onnx")), "WP1 collection cannot export another model")
        inventory[name] = named_record(root/name)
    require(sum(record["bytes"] for record in inventory.values()) <= COLLECTION_LIMIT, "collection output cap exceeded")
    if not partial:
        for relative, record in state["runs/primary/artifacts/COMPLETE.json"]["artifacts"].items():
            core().safe_name(relative)
            require(inventory.get("runs/primary/"+relative) == record, "producer artifact inventory changed")
    return inventory, verified


def native_action(payload):
    require(os.name == "posix", "native action requires Linux")
    modes = ("prepare", "install", "start", "status", "collect", "collect-inspect", "collect-failed", "collect-failed-inspect")
    mode = payload.get("mode")
    require(mode in modes and all(valid_sha(payload.get(name)) for name in
        ("source_package_sha256", "private_package_sha256", "config_sha256", "source_manifest_sha256", "controller_sha256", "legacy_ops_sha256")), "invalid native action SHA pins")
    core(payload["legacy_ops_sha256"])
    root = core().unredirected(REMOTE)
    if mode in ("prepare", "install"):
        require(re.fullmatch("[0-9a-f]{32}", str(payload.get("token"))) is not None, "invalid incoming token")
        incoming = root.parent/(".wp1.incoming_"+payload["token"])
        require(not os.path.lexists(root), "WP1 native root already exists; do not overwrite")
        if mode == "prepare":
            incoming = core().fresh_directory(incoming)
            return dict(status="exclusive_incoming_prepared", incoming=str(incoming),
                source_archive=str(incoming/SOURCE_ARCHIVE), private_archive=str(incoming/PRIVATE_ARCHIVE), target_created=False)
        source, private = core().unredirected(incoming/SOURCE_ARCHIVE), core().unredirected(incoming/PRIVATE_ARCHIVE)
        config, _, source_index, private_index = read_bundle(source, private, payload)
        core().exclusive_extract(source, root, source_index, SOURCE_LIMIT)
        # Private members already include private_inputs/; only new files may be written.
        with tarfile.open(private, "r:*") as archive:
            for name in sorted(private_index):
                path = core().unredirected(root/name)
                require(path.is_relative_to(root/"private_inputs"), "private extraction escaped its root")
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    shutil.copyfileobj(archive.extractfile(name), stream, 1024**2)
                require(named_record(path) == private_index[name], "private file differs after extraction")
        _, _, checked, _ = verify_runtime(payload)
        from importlib.metadata import version
        result = dict(schema="n01/wp1-install/v1", run_id=RUN_ID, UTC=core().now(), **{key:payload[key] for key in
            ("source_package_sha256", "private_package_sha256", "config_sha256", "source_manifest_sha256", "controller_sha256")},
            source_file_count=len(source_index)-2, private_file_count=4, input_records=len(config["inputs"]),
            immutable_verified_paths=len(checked), native_python=sys.version, installed_torch_distribution=version("torch"),
            torch_imported=False, CUDA_Torch_runtime_not_queried=True, scientific_started=False,
            array_decodes=0, checkpoint_loads=0, model_forwards=0, preflight=core().runtime_preflight(root, config))
        core().write_new(root/"REMOTE_INSTALL_RECEIPT.json", result)
        return result
    require(root.is_dir() and sha(root/CONTROLLER) == payload["controller_sha256"], "installed controller differs")
    if mode == "start":
        return native_start(payload)
    partial = mode.startswith("collect-failed")
    config, state = native_snapshot(payload, partial)
    if mode == "status":
        return state
    inventory, verified = archive_inventory(state, config, partial)
    archive = root/("wp1_partial_01.tar" if partial else "wp1_complete_01.tar")
    if mode.endswith("inspect"):
        require(archive.is_file() and not archive.is_symlink(), "no prior archive; do not recreate")
    else:
        require(not os.path.lexists(archive), "preserve existing archive; inspect only")
        with tarfile.open(archive, "x:") as stream:
            for name in sorted(inventory):
                add_regular(stream, root/name, name)
    require(index(archive, COLLECTION_LIMIT) == inventory, "archived inventory differs")
    return dict(schema="n01/wp1-native-archive/v1", run_id=RUN_ID, UTC=core().now(), archive=str(archive),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, inventory=inventory,
        partial_evidence_only=partial, job=verified, config_sha256=payload["config_sha256"],
        source_manifest_sha256=payload["source_manifest_sha256"])


def ssh_action(payload, output):
    """Installed controller receives JSON stdin; prepare/install bootstrap is code stdin."""
    code = Path(__file__).read_bytes()
    require(hashlib.sha256(code).hexdigest() == payload["controller_sha256"], "controller changed before SSH")
    bootstrap = payload["mode"] in ("prepare", "install")
    if bootstrap:
        encoded = base64.urlsafe_b64encode(json.dumps(payload, allow_nan=False).encode("utf8")).decode("ascii")
        remote_command = shlex.join([PYTHON, "-B", "-", "--bootstrap", encoded])
        data = code
    else:
        remote_command = shlex.join([PYTHON, "-B", REMOTE+"/"+CONTROLLER, "--native-action"])
        data = json.dumps(payload, allow_nan=False).encode("utf8")
    command = ["ssh", "-x", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, remote_command]
    try:
        result = subprocess.run(command, input=data, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        core().write_new(output/(payload["mode"]+"_SSH_UNKNOWN.json"), dict(command=command, UTC=core().now(),
            unknown_remote_result=True, automatic_retry=False, stdout=(error.stdout or b"").decode("utf8", errors="replace"),
            stderr=(error.stderr or b"").decode("utf8", errors="replace")))
        raise
    stdout, stderr = result.stdout.decode("utf8", errors="replace"), result.stderr.decode("utf8", errors="replace")
    core().write_new(output/(payload["mode"]+"_NATIVE_RECEIPT.json"), dict(command=command, UTC=core().now(),
        native_ssh_exit_code=result.returncode, stdout=stdout, stderr=stderr, automatic_retry=False))
    require(result.returncode == 0, "native action failed; preserve state and do not retry")
    return json.loads(stdout)


def transfer(command, output, name, timeout=600):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        core().write_new(output/name, dict(command=command, unknown_transfer_result=True, automatic_retry=False))
        raise
    core().write_new(output/name, dict(command=command, native_scp_exit_code=result.returncode,
        stdout=result.stdout.decode("utf8", errors="replace"), stderr=result.stderr.decode("utf8", errors="replace"), automatic_retry=False))
    require(result.returncode == 0, "SCP failed; preserve partial transfer")


def controller_action(mode, freeze_directory, output, collected=None):
    receipt, bundle = load_freeze(freeze_directory)
    require(not receipt.get("artificial_only"), "artificial package cannot contact native host")
    config, _, _, _ = bundle
    output = core().fresh_directory(output, local=True)
    payload = {key:receipt[key] for key in ("source_package_sha256", "private_package_sha256", "config_sha256",
        "source_manifest_sha256", "controller_sha256", "legacy_ops_sha256")}
    payload["mode"] = mode
    if mode == "deploy":
        payload.update(mode="prepare", token=uuid.uuid4().hex)
        incoming = ssh_action(payload, output)
        expected_incoming = str(PurePosixPath(REMOTE).parent/(".wp1.incoming_"+payload["token"]))
        require(incoming.get("incoming") == expected_incoming
            and incoming.get("source_archive") == expected_incoming+"/"+SOURCE_ARCHIVE
            and incoming.get("private_archive") == expected_incoming+"/"+PRIVATE_ARCHIVE,
            "incoming transfer paths differ from the exclusive token")
        for filename, remote in ((SOURCE_ARCHIVE, incoming["source_archive"]), (PRIVATE_ARCHIVE, incoming["private_archive"])):
            command = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", str(Path(freeze_directory)/filename), HOST+":"+remote]
            transfer(command, output, filename+"_TRANSFER.json")
        payload["mode"] = "install"
        result = ssh_action(payload, output); core().write_new(output/"REMOTE_INSTALL_RECEIPT.json", result)
        return dict(status="native_install_verified", config_sha256=receipt["config_sha256"], scientific_started=False)
    if mode in ("status", "verify-start"):
        payload["mode"] = "status"
        state = ssh_action(payload, output); core().write_new(output/"REMOTE_STATE.json", state)
        if mode == "verify-start":
            failed = state.get("runs/primary/artifacts/FAILED.json")
            status = state.get("runs/primary/artifacts/status.json") or {}
            native = state.get("runs/primary/supervisor/exit_receipt.json") or {}
            child = native.get("child_native_returncode", native.get("child_native_exit_code"))
            supervisor = native.get("supervisor_native_returncode", native.get("exit_code"))
            require(failed is None and status.get("phase") != "failed"
                and child in (None, 0) and supervisor in (None, 0), "same snapshot already failed; historical first-family proof is insufficient")
            verified = proof_valid(state, config); core().write_new(output/"START_VALIDATION.json", verified); return verified
        return dict(status="one_snapshot_saved", receipt=str(output/"REMOTE_STATE.json"), receipt_sha256=sha(output/"REMOTE_STATE.json"),
            native_exit=state.get("runs/primary/supervisor/exit_receipt.json"), continuous_monitoring=False)
    if mode == "start":
        result = ssh_action(payload, output); core().write_new(output/"START_RECEIPT.json", result)
        return dict(status="native_supervisor_dispatched", supervisor_pid=result["supervisor_pid"], boot_id=result["boot_id"],
            first_family_verified=False, scientific_acceptance=False)
    require(mode in ("collect", "collect-inspect", "collect-failed", "collect-failed-inspect") and collected is not None, "fresh collection path required")
    collected = core().unredirected(collected)
    require(not os.path.lexists(collected), "preserve prior local collection")
    result = ssh_action(payload, output); core().write_new(output/"REMOTE_ARCHIVE_RECEIPT.json", result)
    archive = output/Path(result["archive"]).name
    part = archive.with_name(archive.name+".part")
    transfer(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST+":"+result["archive"], str(part)], output, "COLLECTION_TRANSFER.json")
    require(part.stat().st_size == result["archive_bytes"] and sha(part) == result["archive_sha256"], "downloaded archive SHA/size differs")
    require(index(part, COLLECTION_LIMIT) == result["inventory"], "downloaded member inventory differs")
    part.replace(archive)
    core().exclusive_extract(archive, collected, result["inventory"], COLLECTION_LIMIT)
    state, control_errors = {}, []
    for name in STATE_NAMES:
        try:
            state[name] = core().read_json(collected/name) if (collected/name).is_file() else None
        except Exception as error:
            if not result["partial_evidence_only"]:
                raise
            state[name] = None; control_errors.append(name+": "+repr(error))
    require(sha(collected/CFG) == receipt["config_sha256"] and sha(collected/MANIFEST) == receipt["source_manifest_sha256"],
        "collected frozen controls differ from the authorized freeze")
    state.update(current_boot_id=(state.get("START_RECEIPT.json") or {}).get("boot_id"),
        _verified_freeze=dict(immutable_files_sha256=runtime_hashes(config, receipt["config_sha256"])))
    try:
        state["durable_child_progress"] = ops().durable_progress(collected/"runs/primary/artifacts")
    except Exception as error:
        if not result["partial_evidence_only"]:
            raise
        state["durable_child_progress"] = dict(progress_error=repr(error), actual_completion="unknown")
        control_errors.append("operations journal: "+repr(error))
    if not result["partial_evidence_only"]:
        complete_valid(state, config)
    final = dict(schema="n01/wp1-collection/v1", status="partial_evidence_preserved" if result["partial_evidence_only"] else "native_output_inventory_verified",
        UTC=core().now(), archive=str(archive), archive_sha256=sha(archive), collected_root=str(collected), file_count=len(result["inventory"]),
        partial_evidence_only=result["partial_evidence_only"], job=result["job"], control_read_errors=control_errors,
        scientific_acceptance=False)
    core().write_new(output/"COLLECTION_RECEIPT.json", final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "deploy", "start", "status", "verify-start", "collect", "collect-inspect", "collect-failed", "collect-failed-inspect"))
    parser.add_argument("--repo", type=Path, default=SELF_ROOT)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--config-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--collected", type=Path)
    parser.add_argument("--native-action", action="store_true")
    parser.add_argument("--bootstrap")
    args = parser.parse_args(argv)
    if args.native_action or args.bootstrap:
        require(os.name == "posix", "remote action requires Linux")
        payload = json.loads(base64.urlsafe_b64decode(args.bootstrap)) if args.bootstrap else json.load(sys.stdin)
        result = native_action(payload)
    elif args.mode == "freeze":
        require(args.out and valid_sha(args.config_sha256), "freeze requires output and exact draft SHA")
        result = freeze(args.repo, args.out, args.config_sha256)
        result = {key:result[key] for key in ("schema", "config_sha256", "source_manifest_sha256", "source_package_sha256", "private_package_sha256", "source_file_count", "private_file_count")}
    else:
        require(args.mode and args.freeze and args.out, "controller mode, freeze directory and output are required")
        result = controller_action(args.mode, args.freeze, args.out, args.collected)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
