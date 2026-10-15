"""WP2 bounded serial controller, reusing frozen archive/journal/owned-tree tools.

Only explicit CLI actions contact the native host. A freeze packages public source
and contracts; existing scientific inputs stay read-only at their bound paths.
Collection verifies the remote whole inventory and transfers selected small files.
"""
from __future__ import annotations

import argparse
import ast
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
import signal
import subprocess
import sys
import tarfile
import time
import uuid


RUN_ID = "wp2_014_20261015"
REMOTE = "/home/rylan/projects/ej276d-n01/wp2_014_20261015"
OLD013 = "/home/rylan/projects/ej276d-n01/wp1_013_repeat_fix01_20261015"
OLD012 = "/home/rylan/projects/ej276d-n01/mechanism012_20261015"
HOST = "rylan@100.82.231.102"
PYTHON = "/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python"
CFG, MANIFEST = "wp2_014_config.json", "WP2_SOURCE_MANIFEST.json"
CONTROLLER, DRIVER = "scripts/wp2_controller.py", "scripts/run_wp2_014.py"
OWNED = (CONTROLLER, DRIVER, "src/n01/model014.py", "src/n01/losses014.py")
SOURCE_ARCHIVE = "wp2_source.tar.gz"
SOURCE_LIMIT, COLLECTION_LIMIT, HEAD_LIMIT = 16*1024**2, 512*1024**2, 64*1024**2
PRIOR_MANIFEST_SHA = "dccd1b6c5676fb481f04e1b94a346aff2c41ada72c41c32e6a9042ac30b1fc04"
CORE_SHA = "98a18eef018beebff82565da9ed5b31208cf2bff6db6bff0c492eff9bf2a7816"
JOURNAL_SHA = "f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5"
SUPERVISOR_SHA = "43b6fffd076a1e6d57db67e85e433e71d3f2cda91f3a3d5fe31f0fcb430486a4"
T_SHA = "f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b"
SELF_ROOT = Path(__file__).resolve().parents[1] if __file__ not in ("<stdin>", "<string>") else Path(REMOTE)
STATE_NAMES = ("START_RECEIPT.json", "REMOTE_INSTALL_RECEIPT.json", "DISPATCH_ATTEMPT.json",
    "runs/primary/supervisor/launch.json", "runs/primary/supervisor/resource_status.json",
    "runs/primary/supervisor/exit_receipt.json", "runs/primary/artifacts/status.json",
    "runs/primary/artifacts/INIT_PROOF.json", "runs/primary/artifacts/FIRST_UPDATE_PROOF.json",
    "runs/primary/artifacts/FAILED.json", "runs/primary/artifacts/COMPLETE.json")
TRIALS = ("seed0_C0", "seed0_C1", "seed1_C0", "seed1_C1")
_CORE = _JOURNAL = _SUPERVISOR = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def load_module(path, name, expected):
    require(sha(path) == expected, "reused stdlib source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, "missing reused stdlib module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def core():
    global _CORE
    if _CORE is None:
        path = SELF_ROOT/"scripts/mechanism012_ops.py"
        if not path.is_file():
            path = Path(OLD012)/"scripts/mechanism012_ops.py"
        _CORE = load_module(path, "wp2_reused012_stdlib", CORE_SHA)
    return _CORE


def journal_tools():
    global _JOURNAL
    if _JOURNAL is None:
        _JOURNAL = load_module(SELF_ROOT/"scripts/wp1_ops.py", "wp2_reused013_journal", JOURNAL_SHA)
    return _JOURNAL


def supervisor_tools():
    global _SUPERVISOR
    if _SUPERVISOR is None:
        path = SELF_ROOT/"scripts/supervise_mechanism012.py"
        # Its only top-level project import is the already SHA-verified stdlib core.
        require(sha(path) == SUPERVISOR_SHA, "reused owned-tree helpers changed")
        sys.modules.setdefault("mechanism012_ops", core())
        _SUPERVISOR = load_module(path, "wp2_reused012_owned_tree", SUPERVISOR_SHA)
    return _SUPERVISOR


def named_record(path):
    path = core().unredirected(path)
    require(path.is_file(), "expected regular file")
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def checked_local(row, checked):
    path = core().unredirected(row["local_path"])
    record = named_record(path)
    require(valid_sha(row["sha256"]) and record["sha256"] == row["sha256"]
        and ("bytes" not in row or record["bytes"] == row["bytes"]), "bound local file changed")
    key = str(path)
    require(key not in checked or checked[key] == row["sha256"], "conflicting local SHA aliases")
    checked[key] = row["sha256"]
    return path


def factory_contract(source):
    """Evaluate only the three pure stdlib factory functions from pinned bytes.

    No module top-level code, numeric imports, input loading or models execute.
    The same extractor works before the new archive has been installed.
    """
    tree = ast.parse(source)
    names = {"seed_set014", "settings014", "budget_caps014"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, "missing014 stdlib factories")
    allowed_calls = {"dict", "list", "len", "seed_set014"}
    for node in nodes:
        require(not any(isinstance(n, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)) for n in ast.walk(node)), "factory is not stdlib/pure")
        require(all(isinstance(n.func, ast.Name) and n.func.id in allowed_calls for n in ast.walk(node) if isinstance(n, ast.Call)), "unexpected factory operation")
    namespace = dict(RUN_ID=RUN_ID, HYPOTHESES=("fast", "slow", "common_level"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<verified014-pure-factories>", "exec"), namespace)
    return namespace["settings014"](), namespace["budget_caps014"]


def validate_config(p, *, frozen=True, driver_source=None):
    require(p.get("schema") == "n01/wp2-014-config/v1" and p.get("run_id") == RUN_ID
        and p.get("remote_root") == REMOTE, "wrong014 identity/root")
    require(not frozen or p.get("status") == "frozen", "draft014 cannot execute")
    if driver_source is None:
        driver_source = (SELF_ROOT/DRIVER).read_bytes()
    settings, budget_factory = factory_contract(driver_source)
    for key in ("candidates", "seed_sets", "G_updates", "TRAIN_counts", "training", "calibration", "probe",
                "hypotheses", "CPU_threads", "run_resource_limits", "one_active_scientific_job", "serial_runs",
                "automatic_restart", "overwrite_outputs", "DEV_numeric_read"):
        require(p.get(key) == settings[key], "registered014 setting differs: "+key)
    require(p["resource_limits"] == settings["resource_limits"], "whole-batch limits differ")
    for key in ("parents", "draws", "banks", "chunk_rows", "whole_population", "snapshot_arms"):
        require(p["evaluation"].get(key) == settings["evaluation"][key], "whole-population014 evaluation differs")
    require(len(p["evaluation"]["bank_seeds"]) == 2 and type(p["evaluation"]["projection_seed"]) is int, "fixed014 eval banks/projection required")
    require(p.get("budget_caps") == budget_factory(p), "exact014 operation caps differ")
    require(set(("dataset", "statistics", "data_statistics", "base_config", "deterministic", "noise005", "source_information")) <= set(p.get("inputs", {})), "014 input slots incomplete")
    require(set(("observation", "data_roles", "acceptance", "tolerance_table")) <= set(p.get("contracts", {})), "014 contracts incomplete")
    require(p["contracts"]["tolerance_table"]["sha256"] == T_SHA, "root-frozen pre-candidate T changed")
    require(p.get("allowed_input_roots") and REMOTE in p["allowed_input_roots"], "explicit014 input roots required")
    require(not frozen or (set(OWNED) <= set(p.get("source_files", {})) and CFG not in p["source_files"] and MANIFEST not in p["source_files"]), "source inventory missing or recursive")
    return settings


def project_import_closure(files, read_bytes):
    """Resolve lazy project import statements inside functions as well as imports.

    The inventory already includes all old84; this checks reachable project
    dependencies of the four new files rather than searching unrelated code.
    """
    names = set(files)
    modules = {}
    for name in names:
        if name.endswith(".py") and name.startswith("src/"):
            mod = name[4:-3].replace("/", ".")
            modules[mod.removesuffix(".__init__")] = name
        elif name.endswith(".py") and name.startswith("scripts/"):
            modules[PurePosixPath(name).stem] = name
    reached, queue = set(), list(OWNED)
    while queue:
        name = queue.pop()
        if name in reached or not name.endswith(".py"):
            continue
        require(name in names, "project import absent from source package: "+name)
        reached.add(name)
        tree = ast.parse(read_bytes(name), filename=name)
        current = name[4:-3].replace("/", ".") if name.startswith("src/") else PurePosixPath(name).stem
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Import):
                candidates = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = current.split(".")[:-node.level]
                    base = ".".join(parts+([base] if base else []))
                candidates = [base] + [base+"."+a.name for a in node.names if a.name != "*"]
            for module in candidates:
                if module in modules:
                    queue.append(modules[module])
                elif module.startswith("n01."):
                    # from n01.module import exported_symbol has an attribute
                    # suffix; only the actual from-module is mandatory.
                    base = node.module if isinstance(node, ast.ImportFrom) else module
                    if base and base.startswith("n01."):
                        require(base in modules, "lazy n01 dependency missing: "+base)
                elif "." not in module and (SELF_ROOT/"scripts"/(module+".py")).is_file():
                    require(module in modules, "lazy scripts dependency missing: "+module)
    require("scripts/stability011_metrics.py" in reached, "empirical CDF helper must be in actual import closure")
    return sorted(reached)


def add_regular(archive, path, name):
    core().safe_name(name)
    require(core().unredirected(path).is_file(), "nonregular payload")
    archive.add(path, arcname=name, recursive=False)


def index(path, maximum):
    seen = set()
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse(), "nonregular/sparse archive member")
            name = core().safe_name(member.name)
            require(name.casefold() not in seen, "case-colliding archive members")
            seen.add(name.casefold())
    return core().tar_index(path, maximum)


def read_bundle(archive_path, pins):
    require(sha(archive_path) == pins["source_package_sha256"], "source archive SHA differs")
    inventory = index(archive_path, SOURCE_LIMIT)
    with tarfile.open(archive_path, "r:*") as archive:
        config_bytes, manifest_bytes = [archive.extractfile(n).read() for n in (CFG, MANIFEST)]
        driver_bytes = archive.extractfile(DRIVER).read()
        require(hashlib.sha256(config_bytes).hexdigest() == pins["config_sha256"]
            and hashlib.sha256(manifest_bytes).hexdigest() == pins["source_manifest_sha256"], "frozen014 controls differ")
        p, manifest = json.loads(config_bytes), json.loads(manifest_bytes)
        validate_config(p, driver_source=driver_bytes)
        require(manifest.get("schema") == "n01/wp2-source/v1" and manifest.get("run_id") == RUN_ID
            and manifest.get("config_sha256") == pins["config_sha256"], "014 source manifest differs")
        require(set(inventory) == set(manifest["files"]) | {CFG, MANIFEST} and set(p["source_files"]) == set(manifest["files"]), "014 package/source inventory differs")
        for name, row in manifest["files"].items():
            require(inventory[name] == row and p["source_files"][name] == dict(path=REMOTE+"/"+name, **row), "014 packaged source bytes/path differ")
            require(not name.endswith((".npz", ".pt", ".pth", ".onnx", ".bin")), "numeric input leaked into public source payload")
        require(manifest["files"][CONTROLLER]["sha256"] == pins["controller_sha256"], "014 controller changed")
        closure = project_import_closure(manifest["files"], lambda name: archive.extractfile(name).read())
        require(closure == manifest["project_import_closure"], "packaged lazy dependency closure differs")
    return p, manifest, inventory


def freeze(repo, output, expected_draft_sha, *, artificial=False):
    repo = core().unredirected(repo)
    require(valid_sha(expected_draft_sha) and sha(repo/CFG) == expected_draft_sha, "draft014 changed")
    p = copy.deepcopy(core().read_json(repo/CFG))
    validate_config(p, frozen=False, driver_source=(repo/DRIVER).read_bytes())
    prior_path = repo/"WP1_SOURCE_MANIFEST.json"
    require(sha(prior_path) == PRIOR_MANIFEST_SHA, "frozen013 manifest changed")
    prior = core().read_json(prior_path)
    require(len(prior["files"]) == 84, "old84 source inventory changed")
    prior_config = core().read_json(repo/"wp1_013_config.json")
    require(sha(repo/"wp1_013_config.json") == prior["config_sha256"], "old013 configuration changed")
    checked = {str(repo/CFG):expected_draft_sha, str(prior_path):PRIOR_MANIFEST_SHA}
    paths = {name: repo/name for name in set(prior["files"]) | set(OWNED) | {"WP1_SOURCE_MANIFEST.json"}}
    protections = copy.deepcopy(prior_config.get("protection_files", {}))
    for key, row in protections.items():
        checked_local(row, checked)
    for name, record in prior["files"].items():
        require(named_record(repo/name) == record, "old84 original bytes changed: "+name)
        protections["wp1_013:"+name] = dict(path=OLD013+"/"+name, local_path=str(repo/name), **record)
    for name, digest in (("WP1_SOURCE_MANIFEST.json", PRIOR_MANIFEST_SHA), ("wp1_013_config.json", prior["config_sha256"])):
        protections["wp1_013:"+name] = dict(path=OLD013+"/"+name, local_path=str(repo/name), sha256=digest, bytes=(repo/name).stat().st_size)
    for collection in (p["inputs"], p["contracts"]):
        for row in collection.values():
            local = checked_local(row, checked)
            native = PurePosixPath(row["path"])
            if native.is_relative_to(PurePosixPath(REMOTE)):
                name = native.relative_to(PurePosixPath(REMOTE)).as_posix()
                core().safe_name(name)
                require(not name.endswith((".npz", ".pt", ".pth", ".bin")), "newroot numeric provisioning is not authorized")
                require(name not in paths or named_record(paths[name]) == named_record(local), "source/contract alias conflict")
                paths[name] = local
    source_records = {}
    for name, path in sorted(paths.items()):
        core().safe_name(name)
        source_records[name] = named_record(path)
        checked[str(path)] = source_records[name]["sha256"]
    closure = project_import_closure(source_records, lambda name: paths[name].read_bytes())
    p["source_files"] = {name:dict(path=REMOTE+"/"+name, **record) for name, record in source_records.items()}
    p["protection_files"] = protections
    p.update(status="frozen", frozen_UTC=core().now())
    validate_config(p, driver_source=(repo/DRIVER).read_bytes())
    # Every protection path must be inside an explicit registered read root.
    roots = [PurePosixPath(root) for root in p["allowed_input_roots"]]
    for collection in (p["source_files"],p["inputs"],p["contracts"],p["protection_files"]):
        require(all(any(PurePosixPath(row["path"]).is_relative_to(root) for root in roots) for row in collection.values()), "014 binding escaped registered input roots")
    output = core().fresh_directory(output, local=True)
    core().write_new(output/CFG,p)
    config_sha = sha(output/CFG)
    manifest = dict(schema="n01/wp2-source/v1",run_id=RUN_ID,config_sha256=config_sha,frozen_UTC=p["frozen_UTC"],files=source_records,
        prior_WP1_manifest_sha256=PRIOR_MANIFEST_SHA,prior_source_file_count=84,project_import_closure=closure,before_scientific_decode=True,
        public_source_only=True,scientific_inputs_copied=False)
    core().write_new(output/MANIFEST,manifest)
    package=output/SOURCE_ARCHIVE
    with tarfile.open(package,"x:gz") as archive:
        for name,path in sorted(paths.items()):add_regular(archive,path,name)
        for name in (CFG,MANIFEST):add_regular(archive,output/name,name)
    pins=dict(source_package_sha256=sha(package),config_sha256=config_sha,source_manifest_sha256=sha(output/MANIFEST),controller_sha256=source_records[CONTROLLER]["sha256"])
    _,_,inventory=read_bundle(package,pins)
    for path,digest in checked.items():require(sha(path)==digest,"file changed during014 freeze")
    result=dict(schema="n01/wp2-freeze/v1",run_id=RUN_ID,UTC=core().now(),**pins,source_package=str(package),source_package_bytes=package.stat().st_size,
        source_file_count=len(source_records),source_inventory=inventory,protected_source_records=len(protections),project_import_closure=closure,
        all_original_files_unchanged=True,artificial_only=artificial,scientific_started=False,array_decodes=0,checkpoint_loads=0,model_forwards=0)
    core().write_new(output/"FREEZE_RECEIPT.json",result)
    return result


def load_freeze(directory):
    directory=core().unredirected(directory)
    receipt=core().read_json(directory/"FREEZE_RECEIPT.json")
    require(receipt.get("schema")=="n01/wp2-freeze/v1" and receipt.get("run_id")==RUN_ID,"wrong014 freeze")
    return receipt,read_bundle(directory/SOURCE_ARCHIVE,receipt)


def runtime_gate(pins, *, partial=False):
    root=core().unredirected(REMOTE)
    require(sha(root/CFG)==pins["config_sha256"] and sha(root/MANIFEST)==pins["source_manifest_sha256"] and sha(root/CONTROLLER)==pins["controller_sha256"],"installed014 controls differ")
    p,manifest=core().read_json(root/CFG),core().read_json(root/MANIFEST)
    validate_config(p,driver_source=(root/DRIVER).read_bytes())
    require(manifest.get("config_sha256")==pins["config_sha256"] and set(manifest["files"])==set(p["source_files"]),"014 manifest/config source mismatch")
    checked,errors={str(root/CFG):pins["config_sha256"]},[]
    roots=[core().unredirected(Path(name)) for name in p["allowed_input_roots"]]
    for label,records in (("source",p["source_files"]),("input",p["inputs"]),("contract",p["contracts"]),("protected",p.get("protection_files",{}))):
        for name,row in records.items():
            try:
                path=core().unredirected(row["path"])
                require(path.is_absolute() and any(path.is_relative_to(r) for r in roots) and valid_sha(row["sha256"]),"014 input/source path escaped gate")
                if label=="source":
                    require(path==root/name and {k:row[k] for k in ("bytes","sha256")}==manifest["files"][name],"014 source actual path differs")
                if str(path) in checked:require(checked[str(path)]==row["sha256"],"conflicting014 SHA aliases")
                else:
                    require(sha(path)==row["sha256"],"014 immutable bytes changed: "+label+":"+name)
                    checked[str(path)]=row["sha256"]
                if "bytes" in row:require(path.stat().st_size==row["bytes"],"014 immutable size changed")
            except Exception as error:
                if not partial:raise
                errors.append(label+":"+name+": "+repr(error))
    return p,manifest,checked,errors


def expected_hashes(p,config_sha):
    result={REMOTE+"/"+CFG:config_sha}
    for records in (p["source_files"],p["inputs"],p["contracts"],p.get("protection_files",{})):
        for row in records.values():
            require(row["path"] not in result or result[row["path"]]==row["sha256"],"conflicting014 identity")
            result[row["path"]]=row["sha256"]
    return result


def validate_counters(p,attempted,completed):
    require(set(attempted)==set(completed)==set(p["budget_caps"]),"014 counter names differ")
    for name,cap in p["budget_caps"].items():
        require(type(attempted[name]) is int and type(completed[name]) is int and 0<=completed[name]<=attempted[name]<=cap,"014 counter cap/order differs: "+name)


def same_snapshot_not_failed(state):
    native=state.get("runs/primary/supervisor/exit_receipt.json") or {}
    status=state.get("runs/primary/artifacts/status.json") or {}
    require(state.get("runs/primary/artifacts/FAILED.json") is None and status.get("native_status")!="FAILED" and status.get("phase")!="failed"
        and native.get("child_native_exit_code") in (None,0) and native.get("exit_code") in (None,0),"same snapshot already failed; historical firstG1 proof is insufficient")


def first_valid(state,p,*,current_boot=True):
    same_snapshot_not_failed(state)
    start,launch=state.get("START_RECEIPT.json"),state.get("runs/primary/supervisor/launch.json")
    first,init=state.get("runs/primary/artifacts/FIRST_UPDATE_PROOF.json"),state.get("runs/primary/artifacts/INIT_PROOF.json")
    require(all(isinstance(x,dict) for x in (start,launch,first,init)),"014 source/init/actualG1 proof not ready")
    require(start["supervisor_pid"]==launch["supervisor_pid"] and launch["child_pid"]==first["pid"]==init["pid"],"014 firstG1 PID chain differs")
    require(not current_boot or start["boot_id"]==state["current_boot_id"],"014 proof belongs to prior boot")
    require(first.get("artificial_only") is False and init.get("artificial_only") is False and first.get("run_id")==RUN_ID,"artificial/wrong014 proof")
    require(init.get("native_initialized") is True and init.get("head_parameters")==1288 and init.get("no_C2") is True,"014 initialized state proof incomplete")
    require(init.get("checkpoint_loads")==2 and init.get("G_updates")==0 and init.get("optimizer_created")==1
        and init.get("first_trial")==TRIALS[0] and init.get("paired_initialization_bytes") is True
        and init.get("trainable_parameter_names")==["basis_head.weight","basis_head.bias"],"014 initialization/paired-head boundary differs")
    require(first.get("actual_G_updates")==1 and first.get("batch_completed_G_updates")==1 and first.get("trial")==TRIALS[0]
        and first.get("candidate")=="C0" and first.get("gradient_nonzero") is True and first.get("actual_parameter_update_nonzero") is True
        and first.get("frozen_A_005_unchanged") is True,"014 actual first optimizer update not proved")
    require(first.get("trainable_parameter_names")==["basis_head.weight","basis_head.bias"] and first.get("head_parameters")==1288,"014 head whitelist differs")
    require(first.get("frozen_state_sha256")==init.get("frozen_state_sha256") and bool(first.get("frozen_state_sha256")),"014 frozen tensor identities differ across initialization and G1")
    validate_counters(p,first["attempted_counters"],first["counters"])
    require(first["counters"]["G_updates"]==first["counters"]["optimizer_steps"]==1 and first["counters"]["checkpoint_loads"]==2 and not first.get("inflight"),"014 firstG1 counts incomplete")
    require(first["counters"]["training_backward_queries"]==1 and first["counters"]["new_AD_queries"]>=1
        and all(first["counters"][k]>=1024 for k in ("old_A_rows","head_actual_parameter_forward_rows","noise005_rows"))
        and first["counters"]["hard_ADC_rows"]>=3072,"014 firstG1 formal forward/backward family incomplete")
    expected=state["_verified_freeze"]["immutable_files_sha256"]
    require(expected and all(first["immutable_files_sha256"].get(k)==v and init["immutable_files_sha256"].get(k)==v for k,v in expected.items()),"014 proof immutable identities differ")
    return dict(status="actual_first_G1_source_and_initialization_verified",supervisor_pid=start["supervisor_pid"],child_pid=first["pid"],actual_completed_G_updates=1,
        current_batch_completed_G_updates=(state.get("runs/primary/artifacts/status.json") or {}).get("counters",{}).get("G_updates"),scientific_acceptance=False)


def complete_valid(state,p):
    same_snapshot_not_failed(state)
    native=state.get("runs/primary/supervisor/exit_receipt.json") or {}
    complete=state.get("runs/primary/artifacts/COMPLETE.json")
    require(native.get("child_native_exit_code")==native.get("exit_code")==0 and native.get("reason") is None and native.get("error") is None,"014 native failure/resource stop")
    require(isinstance(complete,dict) and complete.get("schema")=="n01/wp2-batch-complete/v1" and complete.get("native_status")=="COMPLETE"
        and complete.get("run_id")==RUN_ID and complete.get("artificial_only") is False and complete.get("all_four_trials_completed") is True
        and complete.get("all_frozen_A_005_unchanged") is True and complete.get("no_C2") is True and complete.get("no_DEV") is True,"014 producer not complete")
    first_valid(state,p,current_boot=False)
    validate_counters(p,complete["attempted_counters"],complete["counters"])
    require(complete["G_updates_per_trial"]==2000 and [r["directory"] for r in complete["trials"]]==list(TRIALS)
        and all(r["G_updates"]==2000 for r in complete["trials"]),"014 four serial trial boundary differs")
    counts=complete["counters"]
    require(counts["G_updates"]==counts["optimizer_steps"]==counts["training_backward_queries"]==8000 and counts["optimizer_created"]==4 and counts["checkpoint_loads"]==2,"014 actual optimizer budget incomplete")
    mandatory=("old_A_rows","head_actual_parameter_forward_rows","noise005_rows","hard_ADC_rows","all_path_gather_rows","integral_statistics_fits")
    require(all(counts[k]==p["budget_caps"][k] for k in mandatory),"014 mandatory full training/evaluation rows incomplete")
    require(counts["head_total_kernel_rows"]==counts["head_actual_parameter_forward_rows"]+counts["calibration_map_rows"],"014 actual/map/total kernel roles differ")
    require(counts["new_AD_queries"]==counts["training_backward_queries"]+counts["probe_AD_queries"],"014 AD role accounting differs")
    require(complete["attempted_counters"]==counts and not complete.get("inflight"),"014 incomplete operation remains")
    progress=state["durable_child_progress"]
    require(not progress["unknown_inflight_ids"] and not progress["failed_ids"] and not progress["truncated_tail"],"014 unresolved durable operation")
    for name,value in counts.items():
        require(progress["completed_counts"].get(name,0)==value and progress["entered_counts"].get(name,0)==value,"014 journal/counters differ")
    expected=state["_verified_freeze"]["immutable_files_sha256"]
    require(all(complete["immutable_files_sha256"].get(k)==v for k,v in expected.items()),"014 final immutable identities differ")
    terminal=state.get("runs/primary/artifacts/status.json") or {}
    require(terminal.get("counters")==counts and terminal.get("native_status")=="COMPLETE","014 terminal status differs")
    require(native["supervisor_pid"]==state["START_RECEIPT.json"]["supervisor_pid"] and native["child_pid"]==complete["pid"],"014 terminal PID chain differs")
    require(native.get("elapsed_seconds",float("inf"))<=p["resource_limits"]["wall_seconds"]
        and native.get("peak_output_bytes",float("inf"))<=p["resource_limits"]["output_MiB"]*1024**2,"014 terminal whole-batch resource cap exceeded")
    return dict(status="native_and_four_trial_producer_complete",counters=counts,trials=complete["trials"],scientific_acceptance=False,root_candidate_choice_pending=True)


def native_snapshot(pins,*,partial=False):
    p,_,checked,errors=runtime_gate(pins,partial=partial)
    root=Path(REMOTE);state={}
    for name in STATE_NAMES:
        try:
            path=core().unredirected(root/name)
            state[name]=core().read_json(path) if path.is_file() else None
        except Exception as error:
            if not partial:raise
            state[name]=None;errors.append(name+": "+repr(error))
    try:progress=journal_tools().durable_progress(root/"runs/primary/artifacts")
    except Exception as error:
        if not partial:raise
        progress=dict(progress_read_error=repr(error),actual_inflight_completion="unknown");errors.append(repr(error))
    state.update(UTC=core().now(),current_boot_id=core().boot_id(),durable_child_progress=progress,protection_errors=errors,partial_evidence_only=partial,
        _verified_freeze=dict(config_sha256=pins["config_sha256"],source_manifest_sha256=pins["source_manifest_sha256"],immutable_files_sha256=checked))
    return p,state


def supervise(config_sha,lock_fd):
    require(os.name=="posix","native014 Linux supervision only")
    import fcntl
    os.fstat(lock_fd);fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=core().unredirected(REMOTE)
    pins=dict(config_sha256=config_sha,source_manifest_sha256=sha(root/MANIFEST),controller_sha256=sha(root/CONTROLLER))
    p,_,_,_=runtime_gate(pins)
    job=core().fresh_directory(root/"runs/primary");sv=core().fresh_directory(job/"supervisor")
    helpers=supervisor_tools();clock=time.monotonic();interrupted={"signal":None}
    for number in (signal.SIGINT,signal.SIGTERM):signal.signal(number,lambda n,frame:interrupted.update(signal=n))
    child,reason,error,code=None,None,None,1;peak_rss=peak_gpu=peak_size=0
    command=[sys.executable,"-B",str(root/DRIVER),"--config",str(root/CFG),"--config-sha256",config_sha,"--output",str(job/"artifacts"),"--device","cuda:0"]
    limits=dict(p["resource_limits"],poll_seconds=10,TERM_grace_seconds=20)
    try:
        with (job/"console.log").open("xb") as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            core().write_new(sv/"launch.json",dict(schema="n01/wp2-launch/v1",run_id=RUN_ID,supervisor_pid=os.getpid(),child_pid=child.pid,command=command,UTC=core().now()))
            while child.poll() is None:
                ids,rss=helpers.process_tree(os.getpid());gpu=helpers.gpu_bytes(ids);size=helpers.output_bytes(job);elapsed=time.monotonic()-clock
                peak_rss,peak_gpu,peak_size=max(peak_rss,rss),max(peak_gpu,gpu),max(peak_size,size)
                reason=helpers.limit_reason(limits,rss,gpu,size,elapsed,interrupted["signal"])
                try:progress=journal_tools().durable_progress(job/"artifacts")
                except Exception as exc:progress=dict(progress_read_error=repr(exc),actual_inflight_completion="unknown")
                journal_tools().atomic_json(sv/"resource_status.json",dict(UTC=core().now(),owned_process_tree=sorted(ids),RSS_bytes=rss,GPU_bytes=gpu,
                    output_bytes=size,elapsed_seconds=elapsed,**progress))
                if reason:helpers.stop_owned(child,20);break
                time.sleep(10)
            code=child.wait();size=helpers.output_bytes(job);elapsed=time.monotonic()-clock;peak_size=max(peak_size,size)
            reason=reason or helpers.limit_reason(limits,0,0,size,elapsed,interrupted["signal"])
    except BaseException as exc:
        reason,error=reason or "supervisor_error",repr(exc)
        if child is not None:helpers.stop_owned(child,20);code=child.returncode
    try:progress=journal_tools().durable_progress(job/"artifacts")
    except Exception as exc:progress=dict(progress_read_error=repr(exc),actual_inflight_completion="unknown")
    receipt=dict(schema="n01/wp2-supervisor-exit/v1",run_id=RUN_ID,UTC=core().now(),supervisor_pid=os.getpid(),child_pid=None if child is None else child.pid,
        child_native_exit_code=code,exit_code=0 if code==0 and reason is None else 1,reason=reason,error=error,elapsed_seconds=time.monotonic()-clock,
        peak_RSS_bytes=peak_rss,peak_GPU_bytes=peak_gpu,peak_output_bytes=peak_size,automatic_restart=False,native_zero_is_scientific_acceptance=False,**progress)
    journal_tools().atomic_json(sv/"exit_receipt.json",receipt);os.close(lock_fd)
    return receipt["exit_code"]


def native_start(pins):
    require(os.name=="posix","native014 Linux only")
    import fcntl
    root=core().unredirected(REMOTE)
    require(not any(os.path.lexists(root/n) for n in ("runs/primary","START_RECEIPT.json","DISPATCH_ATTEMPT.json","supervisor_console.log","tmp","cache")),"preserve prior014 dispatch; no restart")
    p,_,_,_=runtime_gate(pins)
    lock=core().unredirected(root.parent/".n01_scientific_dispatch.lock")
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight=core().runtime_preflight(root,p)
        minimum=p.get("preflight",{}).get("minimum_free_disk_GiB",8)
        require(preflight["disk_free_bytes"]>=minimum*1024**3,"014 batch free-disk preflight failed")
        active=[line for line in subprocess.check_output(["ps","-eo","pid,args"],text=True,timeout=10).splitlines()
            if any(mark in line for mark in ("scripts/run_wp1_013.py","scripts/run_wp2_014.py","scripts/wp1_ops.py supervise","scripts/wp2_controller.py --mode supervise"))]
        require(not active,"another013/014 scientific job is active")
        core().write_new(root/"DISPATCH_ATTEMPT.json",dict(run_id=RUN_ID,UTC=core().now(),**pins,preflight=preflight,budget_caps=p["budget_caps"],automatic_restart=False))
        for name in ("tmp","cache"):core().unredirected(root/name).mkdir(mode=0o700,exist_ok=False)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONUTF8="1",PYTHONPATH=str(root/"src"),TMPDIR=str(root/"tmp"),XDG_CACHE_HOME=str(root/"cache"),
            CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="12",MKL_NUM_THREADS="12",OPENBLAS_NUM_THREADS="12",NUMEXPR_NUM_THREADS="12")
        command=[PYTHON,"-B",str(root/CONTROLLER),"--mode","supervise","--config-sha256",pins["config_sha256"],"--lock-fd",str(fd)]
        child=core().spawn_supervisor(command,root,fd,env)
        result=dict(schema="n01/wp2-start/v1",run_id=RUN_ID,UTC=core().now(),**pins,supervisor_pid=child.pid,command=command,boot_id=preflight["boot_id"],
            exclusive_scientific_lock=str(lock),lock_inherited_by_supervisor=True,actual_G1_verified=False,automatic_restart=False,scientific_acceptance=False,preflight=preflight)
        core().write_new(root/"START_RECEIPT.json",result)
        return result
    finally:os.close(fd)


def select_inventory(inventory,*,include_heads=False):
    selected,omitted={},{}
    for name,row in sorted(inventory.items()):
        core().safe_name(name)
        suffix=PurePosixPath(name).suffix.lower()
        small=suffix in (".json",".jsonl",".log",".txt",".md")
        head=suffix==".pt" and PurePosixPath(name).name in ("initial.pt","step_1000.pt","final.pt","failure.pt","initial_head.pt")
        if small or (include_heads and head and row["bytes"]<=HEAD_LIMIT):selected[name]=row
        else:omitted[name]=row
    require(sum(r["bytes"] for r in selected.values())<=COLLECTION_LIMIT,"selected014 collection exceeds512MiB; do not transfer raw")
    return selected,omitted


def archive_inventory(state,p,*,partial=False,include_heads=False):
    root=core().unredirected(REMOTE)
    if partial:
        start=state.get("START_RECEIPT.json")
        require(isinstance(start,dict),"no owned014 dispatch for partial collection")
        if start["boot_id"]==state["current_boot_id"]:
            launch=state.get("runs/primary/supervisor/launch.json") or {}
            require(all(type(pid) is not int or not Path("/proc",str(pid)).exists() for pid in (start.get("supervisor_pid"),launch.get("child_pid"))),"owned014 job is still active; status only")
        verified=dict(status="partial_evidence_only",native=state.get("runs/primary/supervisor/exit_receipt.json"),durable_progress=state["durable_child_progress"],actual_inflight_completion="unknown",scientific_acceptance=False)
    else:verified=complete_valid(state,p)
    names=[]
    for directory,folders,files in os.walk(root/"runs/primary",followlinks=False):
        require(not any((Path(directory)/n).is_symlink() for n in folders+files),"redirected014 owned output")
        names.extend((Path(directory)/n).relative_to(root).as_posix() for n in files)
    for name in (CFG,MANIFEST,"START_RECEIPT.json","REMOTE_INSTALL_RECEIPT.json","DISPATCH_ATTEMPT.json","supervisor_console.log"):
        if (root/name).is_file():names.append(name)
    inventory={name:named_record(root/name) for name in sorted(set(names))}
    require(sum(r["bytes"] for r in inventory.values())<=(p["resource_limits"]["output_MiB"]+16)*1024**2,"remote014 inventory exceeds job+control cap")
    if not partial:
        artifact_manifest=core().read_json(root/"runs/primary/artifacts/ARTIFACT_INVENTORY.json")
        for relative,row in artifact_manifest["files"].items():
            core().safe_name(relative)
            if relative=="status.json":continue  # Atomic terminal status follows producer inventory creation.
            require(inventory.get("runs/primary/artifacts/"+relative)==row,"producer014 immutable artifact inventory differs")
        for trial in state["runs/primary/artifacts/COMPLETE.json"]["trials"]:
            name="runs/primary/artifacts/"+trial["directory"]+"/COMPLETE.json"
            require(inventory[name]["sha256"]==trial["COMPLETE_sha256"],"014 trial COMPLETE changed")
            record=core().read_json(root/name)
            coverage=record.get("coverage",{});resources=record.get("resources",{})
            index=TRIALS.index(trial["directory"])
            require(record.get("schema")=="n01/wp2-trial-complete/v1" and record.get("G_updates")==2000
                and record.get("candidate")==p["candidates"][index%2] and record.get("seed")==p["seed_sets"][index//2]
                and record.get("frozen_A_005_unchanged") is True and coverage.get("source_unique")==6593 and coverage.get("real_unique")==21047,
                "014 trial identity/coverage/freeze differs")
            require(resources.get("output_bytes",float("inf"))<=p["run_resource_limits"]["output_MiB"]*1024**2
                and resources.get("elapsed_seconds",float("inf"))<=p["run_resource_limits"]["wall_seconds"]
                and resources.get("process_RSS_bytes",float("inf"))<=p["run_resource_limits"]["job_tree_RSS_GiB"]*1024**3,"014 trial resource cap exceeded")
    selected,omitted=select_inventory(inventory,include_heads=include_heads)
    return inventory,selected,omitted,verified


def native_action(payload):
    require(os.name=="posix","native014 action requires Linux")
    mode=payload.get("mode")
    require(mode in ("prepare","install","start","status","collect","collect-inspect","collect-failed","collect-failed-inspect")
        and all(valid_sha(payload.get(n)) for n in ("source_package_sha256","config_sha256","source_manifest_sha256","controller_sha256")),"bad014 native action pins")
    root=core().unredirected(REMOTE)
    if mode in ("prepare","install"):
        require(re.fullmatch("[0-9a-f]{32}",str(payload.get("token"))) is not None and not os.path.lexists(root),"014 exclusive new root/token required")
        incoming=root.parent/(".wp2.incoming_"+payload["token"])
        if mode=="prepare":
            core().fresh_directory(incoming)
            return dict(status="exclusive_incoming_prepared",incoming=str(incoming),source_archive=str(incoming/SOURCE_ARCHIVE),target_created=False)
        p,manifest,inventory=read_bundle(core().unredirected(incoming/SOURCE_ARCHIVE),payload)
        core().exclusive_extract(incoming/SOURCE_ARCHIVE,root,inventory,SOURCE_LIMIT)
        _,_,checked,_=runtime_gate(payload)
        result=dict(schema="n01/wp2-install/v1",run_id=RUN_ID,UTC=core().now(),source_file_count=len(manifest["files"]),config_sha256=payload["config_sha256"],
            source_manifest_sha256=payload["source_manifest_sha256"],immutable_verified_paths=len(checked),scientific_started=False,array_decodes=0,checkpoint_loads=0,model_forwards=0)
        core().write_new(root/"REMOTE_INSTALL_RECEIPT.json",result)
        return result
    require(root.is_dir() and sha(root/CONTROLLER)==payload["controller_sha256"],"installed014 controller changed")
    if mode=="start":return native_start(payload)
    partial=mode.startswith("collect-failed")
    p,state=native_snapshot(payload,partial=partial)
    if mode=="status":return state
    all_files,selected,omitted,verified=archive_inventory(state,p,partial=partial,include_heads=payload.get("include_head_checkpoints",False))
    archive=root/("wp2_partial_selected_01.tar" if partial else "wp2_complete_selected_01.tar")
    if mode.endswith("inspect"):require(archive.is_file() and not archive.is_symlink(),"no prior014 archive; do not recreate")
    else:
        require(not os.path.lexists(archive),"preserve existing014 archive; inspect only")
        with tarfile.open(archive,"x:") as stream:
            for name in sorted(selected):add_regular(stream,root/name,name)
    require(index(archive,COLLECTION_LIMIT)==selected,"selected014 archive bytes differ")
    return dict(schema="n01/wp2-native-selected-archive/v1",run_id=RUN_ID,UTC=core().now(),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        inventory=selected,whole_remote_inventory=all_files,omitted_remote_inventory=omitted,whole_remote_inventory_verified=True,raw_artifacts_retained_remote=True,
        all_raw_artifacts_collected=False,partial_evidence_only=partial,job=verified,config_sha256=payload["config_sha256"],source_manifest_sha256=payload["source_manifest_sha256"])


def ssh_action(payload,output):
    code=Path(__file__).read_bytes()
    require(hashlib.sha256(code).hexdigest()==payload["controller_sha256"],"014 controller changed before SSH")
    if payload["mode"] in ("prepare","install"):
        encoded=base64.urlsafe_b64encode(json.dumps(payload,allow_nan=False).encode()).decode("ascii")
        command=shlex.join([PYTHON,"-B","-","--bootstrap",encoded]);data=code
    else:
        command=shlex.join([PYTHON,"-B",REMOTE+"/"+CONTROLLER,"--native-action"]);data=json.dumps(payload,allow_nan=False).encode()
    argv=["ssh","-x","-o","BatchMode=yes","-o","ConnectTimeout=10",HOST,command]
    try:result=subprocess.run(argv,input=data,capture_output=True,timeout=900 if payload["mode"].startswith("collect") else 300)
    except subprocess.TimeoutExpired as error:
        core().write_new(output/(payload["mode"]+"_SSH_UNKNOWN.json"),dict(command=argv,unknown_remote_result=True,automatic_retry=False,stdout=(error.stdout or b"").decode("utf8",errors="replace"),stderr=(error.stderr or b"").decode("utf8",errors="replace")))
        raise
    stdout,stderr=result.stdout.decode("utf8",errors="replace"),result.stderr.decode("utf8",errors="replace")
    core().write_new(output/(payload["mode"]+"_NATIVE_RECEIPT.json"),dict(command=argv,native_ssh_exit_code=result.returncode,stdout=stdout,stderr=stderr,automatic_retry=False))
    require(result.returncode==0,"014 native action failed; preserve state, no retry")
    return json.loads(stdout)


def transfer(command,output,name):
    try:result=subprocess.run(command,capture_output=True,timeout=900)
    except subprocess.TimeoutExpired:
        core().write_new(output/name,dict(command=command,unknown_transfer_result=True,automatic_retry=False));raise
    core().write_new(output/name,dict(command=command,native_scp_exit_code=result.returncode,stdout=result.stdout.decode("utf8",errors="replace"),stderr=result.stderr.decode("utf8",errors="replace"),automatic_retry=False))
    require(result.returncode==0,"014 transfer failed; preserve partial transfer")


def controller_action(mode,freeze_directory,output,*,collected=None,include_heads=False):
    receipt,bundle=load_freeze(freeze_directory)
    require(receipt.get("artificial_only") is False,"artificial014 package cannot contact native host")
    p,_,_=bundle;output=core().fresh_directory(output,local=True)
    payload={k:receipt[k] for k in ("source_package_sha256","config_sha256","source_manifest_sha256","controller_sha256")};payload["mode"]=mode
    if mode=="deploy":
        payload.update(mode="prepare",token=uuid.uuid4().hex)
        incoming=ssh_action(payload,output);expected=str(PurePosixPath(REMOTE).parent/(".wp2.incoming_"+payload["token"]))
        require(incoming.get("incoming")==expected and incoming.get("source_archive")==expected+"/"+SOURCE_ARCHIVE,"014 incoming transfer target differs")
        transfer(["scp","-q","-o","BatchMode=yes","-o","ConnectTimeout=10",str(Path(freeze_directory)/SOURCE_ARCHIVE),HOST+":"+incoming["source_archive"]],output,"SOURCE_TRANSFER.json")
        payload["mode"]="install";result=ssh_action(payload,output);core().write_new(output/"REMOTE_INSTALL_RECEIPT.json",result)
        return dict(status="native014_install_verified",config_sha256=receipt["config_sha256"],scientific_started=False)
    if mode in ("status","verify-start"):
        payload["mode"]="status";state=ssh_action(payload,output);core().write_new(output/"REMOTE_STATE.json",state)
        if mode=="verify-start":
            result=first_valid(state,p);core().write_new(output/"START_VALIDATION.json",result);return result
        return dict(status="one_snapshot_saved",receipt=str(output/"REMOTE_STATE.json"),receipt_sha256=sha(output/"REMOTE_STATE.json"),continuous_monitoring=False)
    if mode=="start":
        result=ssh_action(payload,output);core().write_new(output/"START_RECEIPT.json",result)
        return dict(status="native014_supervisor_dispatched",supervisor_pid=result["supervisor_pid"],boot_id=result["boot_id"],actual_G1_verified=False,scientific_acceptance=False)
    require(mode in ("collect","collect-inspect","collect-failed","collect-failed-inspect") and collected is not None,"fresh014 selective collection target required")
    collected=core().unredirected(collected);require(not os.path.lexists(collected),"preserve previous014 local collection")
    payload["include_head_checkpoints"]=bool(include_heads)
    result=ssh_action(payload,output);core().write_new(output/"REMOTE_FULL_INVENTORY.json",result)
    archive=output/Path(result["archive"]).name;part=archive.with_name(archive.name+".part")
    require(shutil.disk_usage(output).free>=2*result["archive_bytes"]+128*1024**2,"insufficient E/D room for archive plus selective extraction")
    transfer(["scp","-q","-o","BatchMode=yes","-o","ConnectTimeout=10",HOST+":"+result["archive"],str(part)],output,"COLLECTION_TRANSFER.json")
    require(part.stat().st_size==result["archive_bytes"] and sha(part)==result["archive_sha256"] and index(part,COLLECTION_LIMIT)==result["inventory"],"014 transferred selective bytes differ")
    part.replace(archive);core().exclusive_extract(archive,collected,result["inventory"],COLLECTION_LIMIT)
    require(sha(collected/CFG)==receipt["config_sha256"] and sha(collected/MANIFEST)==receipt["source_manifest_sha256"],"collected014 frozen controls differ")
    state={};errors=[]
    for name in STATE_NAMES:
        try:state[name]=core().read_json(collected/name) if (collected/name).is_file() else None
        except Exception as error:
            if not result["partial_evidence_only"]:raise
            state[name]=None;errors.append(name+": "+repr(error))
    state.update(current_boot_id=(state.get("START_RECEIPT.json") or {}).get("boot_id"),_verified_freeze=dict(immutable_files_sha256=expected_hashes(p,receipt["config_sha256"])))
    try:state["durable_child_progress"]=journal_tools().durable_progress(collected/"runs/primary/artifacts")
    except Exception as error:
        if not result["partial_evidence_only"]:raise
        errors.append(repr(error));state["durable_child_progress"]=dict(progress_read_error=repr(error),actual_inflight_completion="unknown")
    if not result["partial_evidence_only"]:complete_valid(state,p)
    final=dict(schema="n01/wp2-selective-collection/v1",run_id=RUN_ID,status="partial_evidence_preserved" if result["partial_evidence_only"] else "native_inventory_and_selected_bytes_verified",
        collected_root=str(collected),archive=str(archive),archive_sha256=sha(archive),selected_file_count=len(result["inventory"]),
        whole_remote_file_count=len(result["whole_remote_inventory"]),omitted_file_count=len(result["omitted_remote_inventory"]),
        whole_remote_inventory_verified=True,all_raw_artifacts_collected=False,raw_artifacts_retained_remote=True,
        independent_numerical_readback_complete=False,partial_evidence_only=result["partial_evidence_only"],control_read_errors=errors,job=result["job"],scientific_acceptance=False)
    core().write_new(output/"COLLECTION_RECEIPT.json",final)
    return final


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("freeze","deploy","start","status","verify-start","collect","collect-inspect","collect-failed","collect-failed-inspect","supervise"))
    parser.add_argument("--repo",type=Path,default=SELF_ROOT);parser.add_argument("--freeze",type=Path);parser.add_argument("--out",type=Path)
    parser.add_argument("--config-sha256");parser.add_argument("--collected",type=Path);parser.add_argument("--include-head-checkpoints",action="store_true")
    parser.add_argument("--lock-fd",type=int);parser.add_argument("--native-action",action="store_true");parser.add_argument("--bootstrap")
    args=parser.parse_args(argv)
    if args.mode=="supervise":
        require(valid_sha(args.config_sha256) and args.lock_fd is not None,"014 supervision requires inherited lock/config SHA")
        return supervise(args.config_sha256,args.lock_fd)
    if args.native_action or args.bootstrap:
        payload=json.loads(base64.urlsafe_b64decode(args.bootstrap)) if args.bootstrap else json.load(sys.stdin)
        result=native_action(payload)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    if args.mode=="freeze":
        require(args.out and valid_sha(args.config_sha256),"014 freeze requires fresh output and exact draft SHA")
        result=freeze(args.repo,args.out,args.config_sha256)
    else:
        require(args.mode and args.freeze and args.out,"014 controller mode/freeze/output required")
        result=controller_action(args.mode,args.freeze,args.out,collected=args.collected,include_heads=args.include_head_checkpoints)
    print(json.dumps({k:result[k] for k in ("status","schema","source_file_count","config_sha256","source_manifest_sha256","source_package_sha256","supervisor_pid","child_pid","actual_completed_G_updates","partial_evidence_only","all_raw_artifacts_collected") if k in result},ensure_ascii=False,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
