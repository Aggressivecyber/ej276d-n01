"""Independent stdlib-only SHA bridge for the already frozen012 source package.

Does not decode arrays/checkpoints, import numeric libraries or contact a server.
"""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys
import tarfile


ROOT = Path("E:/GAN/research/N01")
DEST = ROOT / "receipts/mechanism012_20261015/independent_01/FROZEN_PACKAGE_REVIEW.json"
FREEZE = ROOT / "receipts/mechanism012_20261015/freeze_01/FREEZE_RECEIPT.json"
PRELAUNCH = ROOT / "receipts/mechanism012_20261015/independent_01/PRELAUNCH_REVIEW.json"
EXPECTED = {
    "package": "7946e2c0c33de78e30457cf7515cfcad13134744017283aab066a929872e2620",
    "manifest": "ba71785e8e6a0e34e71f2bd4a147f96f6ef8ad98a81ecb1e91e5d36d5d046eb8",
    "bindings": "7eba2f29cd720e4a4fe2da442482dce554aab5c52557d76f8b91636e3bf8c6cc",
    "policy": "336dde665ba1df09e88922642b4cff59ed21b5846b9ce01f6885cbd6edda16de",
    "protocol": "3b53fff9170ed86e61d8a09c739a1b571936f7976d3c27a359e85ba7f6dd2d0b",
    "prelaunch": "def26aef53f63b9b18635c909e795af20cf329a34845310d1159b2e95c4b78da",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def load(path):
    return json.loads(Path(path).read_bytes())


def require(value, message):
    if not value:
        raise AssertionError(message)


def check():
    require(not DEST.exists(), "Preserve an existing independent package review")
    freeze, pre = load(FREEZE), load(PRELAUNCH)
    require(sha(PRELAUNCH) == EXPECTED["prelaunch"] and pre["passed"] is True,
            "Independent PRELAUNCH bytes/status differ")
    package, backup = Path(freeze["package"]), Path(freeze["source_backup"])
    require(package.stat().st_size == backup.stat().st_size == freeze["package_bytes"] == 370420,
            "Package or D backup size differs")
    require(sha(package) == sha(backup) == freeze["package_sha256"] == EXPECTED["package"],
            "E/D package SHA differs")
    require(package.read_bytes() == backup.read_bytes(), "E/D package bytes differ")
    manifest_path, bindings_path = ROOT / "MECHANISM012_SOURCE_MANIFEST.json", ROOT / "MECHANISM012_BINDINGS.json"
    mf, bindings = load(manifest_path), load(bindings_path)
    require(sha(manifest_path) == freeze["source_manifest_sha256"] == EXPECTED["manifest"], "Manifest SHA differs")
    require(sha(bindings_path) == freeze["bindings_sha256"] == EXPECTED["bindings"], "Bindings SHA differs")
    for path in (manifest_path, bindings_path):
        require(path.read_bytes() == (FREEZE.parent / path.name).read_bytes(), "Root/freeze metadata bytes differ")
    require(mf["schema"] == "n01/mechanism012-source/v1" and bindings["schema"] == "n01/mechanism012-bindings/v1",
            "Frozen metadata schema differs")
    require(mf["run_id"] == bindings["run_id"] == "mechanism012_20261015", "Run ID differs")
    require(mf["before_scientific_numeric_decode"] is True and freeze["scientific_started"] is False,
            "Freeze is not before science")
    require(all(freeze[k] == 0 for k in ("numeric_decodes", "checkpoint_loads", "NN_rows", "AD_queries", "optimizer_steps")),
            "Freeze reports nonzero science")
    policy_path = ROOT / "mechanism012_policy.json"
    policy = load(policy_path)
    require(sha(policy_path) == mf["policy_sha256"] == bindings["policy_sha256"] == freeze["policy_sha256"] == EXPECTED["policy"],
            "Policy SHA differs")
    require(sha(ROOT / "docs/MECHANISM012_PROTOCOL.md") == EXPECTED["protocol"], "Protocol SHA differs")
    # Import only the already reviewed stdlib controller to reuse literal whitelist constants.
    spec = importlib.util.spec_from_file_location("frozen012_ops_review", ROOT / "scripts/mechanism012_ops.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    require(not any(x in sys.modules for x in ("torch", "numpy", "n01")), "Numeric library unexpectedly imported")
    require(mf["budgets"] == policy["budgets"] == ops.BUDGET, "Registered NN/AD/zero-update budget differs")
    require(policy["resource_limits"] == ops.RESOURCES, "Resource policy differs")
    require(bindings["roots"] == [ops.REMOTE, ops.OLD010, ops.OLD011, ops.OLD008], "Root whitelist differs")
    require(bindings["source_manifest"] == {"path": ops.REMOTE + "/" + ops.MANIFEST, "sha256": EXPECTED["manifest"]},
            "Manifest binding differs")
    require(len(mf["files"]) == freeze["source_file_count"] == 69 and len(bindings["files"]) == freeze["input_count"] == 36,
            "Inventory counts differ")
    require(set(bindings["files"]) == set(ops.INPUT_PATHS), "Input slot whitelist differs")
    require(mf["bound_inputs"] == {k: v["sha256"] for k, v in bindings["files"].items()}, "Bound digests disagree")
    for slot, record in bindings["files"].items():
        require(set(record) == {"path", "sha256"} and record["path"] == ops.INPUT_PATHS[slot]
                and ops.valid_sha(record["sha256"]), "Input record/path/SHA differs: " + slot)
    old53 = load(ROOT / "SHAPE010_SOURCE_MANIFEST.json")
    expected_sources = set(old53["files"]) | set(mf["protected_manifests"]) | set(ops.CONTROL_COPIES.values()) | set(ops.EXTRA)
    require(len(old53["files"]) == 53 and set(mf["files"]) == expected_sources,
            "Source69 is not exact old53+protected4+control6+new6")
    require(len(mf["protected_manifests"]) == 4 and len(mf["control_metadata_copies"]) == 6,
            "Protected/control metadata counts differ")
    require(mf["control_metadata_copies"] == {name: mf["files"][name] for name in ops.CONTROL_COPIES.values()},
            "Control6 source metadata map differs")
    for slot, name in ops.CONTROL_COPIES.items():
        require(bindings["files"][slot]["path"] == ops.REMOTE + "/" + name
                and bindings["files"][slot]["sha256"] == mf["files"][name], "Control copy binding differs")
    for name, d in mf["protected_manifests"].items():
        require(mf["files"][name] == sha(ROOT / name) == d, "Protected original manifest differs")
    for name, d in old53["files"].items():
        require(mf["files"][name] == d, "Frozen old53 source differs")
    for name, d in pre["sources_sha256"].items():
        require(mf["files"][name] == d == sha(ROOT / name), "Prelaunch source/protocol pin differs: " + name)
    expected_members = set(mf["files"]) | {ops.MANIFEST, ops.BINDINGS}
    checked_inventory = {}
    with tarfile.open(package, "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len(expected_members) == 71, "Tar member count differs")
        require(len({m.name for m in members}) == len(members), "Tar duplicate member")
        require({m.name for m in members} == expected_members, "Tar inventory differs")
        for member in members:
            ops.safe_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), "Tar nonregular member")
            stream = archive.extractfile(member)
            require(stream is not None, "Tar member unreadable")
            value = stream.read()
            require(len(value) == member.size and value == (ROOT / member.name).read_bytes(), "Tar/current source bytes differ")
            expected_d = (EXPECTED["manifest"] if member.name == ops.MANIFEST else
                          EXPECTED["bindings"] if member.name == ops.BINDINGS else mf["files"][member.name])
            require(digest(value) == expected_d, "Tar member SHA differs")
            checked_inventory[member.name] = {"bytes": len(value), "sha256": expected_d}
    require(freeze["old53_and_01163_verified"] is True and freeze["protected4_original_and_historical_files_verified"] is True,
            "Root historical guard is missing")
    prior = mf["prior_independent_readback"]
    require(prior == bindings["prior_independent_readback"] and prior["status"] == "PASS"
            and prior["scope"] == "root_verified_local_receipt", "Prior011 local prerequisite differs")
    receipt = {
        "schema": "n01/mechanism012-independent-frozen-package/v1",
        "run_id": "mechanism012_20261015", "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "PASS", "passed": True, "blocking_findings": [],
        "method": "stdlib source/JSON/tar bytes and whitelist comparison only; no extraction, execution or numeric decoding",
        "FREEZE_RECEIPT": {"path": str(FREEZE), "sha256": sha(FREEZE)},
        "PRELAUNCH_REVIEW": {"path": str(PRELAUNCH), "sha256": sha(PRELAUNCH), "status": "PASS"},
        "package": {"path": str(package), "bytes": 370420, "sha256": EXPECTED["package"]},
        "D_backup": {"path": str(backup), "bytes": 370420, "sha256": EXPECTED["package"], "byte_equal_to_E": True},
        "source_manifest_sha256": EXPECTED["manifest"], "bindings_sha256": EXPECTED["bindings"],
        "policy_sha256": EXPECTED["policy"], "protocol_sha256": EXPECTED["protocol"],
        "source_file_count": 69, "primary_input_slots": 30, "control_metadata_slots": 6, "input_slot_count": 36,
        "tar_regular_member_count": 71, "root_and_freeze_metadata_byte_equal": True,
        "source69_tar_current_original_sha_equal": True, "protected4_current_original_and_tar_sha_equal": True,
        "control6_current_original_and_tar_sha_equal": True, "source69_exact_composition_verified": True,
        "input36_absolute_path_and_digest_manifest_bridge_verified": True,
        "four_READY_scripts_policy_protocol_prelaunch_pins_equal": True,
        "budgets": mf["budgets"], "resource_limits": policy["resource_limits"],
        "root_historical_guard_receipt": {"old53_and_01163_verified": True, "protected4_original_and_historical_files_verified": True,
                                         "scope": "root FREEZE_RECEIPT confirms full old artifact/input guards; this bridge did not repeat historical raw-artifact or raw-input hashes"},
        "inventory": checked_inventory,
        "scientific_execution_in_this_bridge": {"numeric_library_imports": 0, "array_decodes": 0, "checkpoint_loads": 0,
             "NN_instances": 0, "NN_forward_rows": 0, "AD_queries": 0, "gradients": 0, "optimizer_created": 0,
             "updates": 0, "DEV_reads": 0, "SSH_SCP_calls": 0, "artificial_checks_rerun": 0},
        "unverified_here": ["Remote installed bytes and actual native PID/formal-FIRST chain require root native readback.",
                            "Actual model restore, saved-bank equality, real1288 graph and native peak resources are not exercised by this source-package bridge.",
                            "This engineering PASS does not establish causal or physical response match."],
        "reviewer_action_complete": "stop after this local SHA bridge; native deploy/start and existing one-time wake remain root operations",
        "script_sha256": sha(Path(__file__)), "original_sources_or_inputs_modified": False,
    }
    with DEST.open("x", encoding="utf8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    require(load(DEST)["passed"] is True, "Receipt reopen failed")
    print(json.dumps({"path": str(DEST), "sha256": sha(DEST), "bytes": DEST.stat().st_size,
                      "status": "PASS", "source_file_count": 69, "input_slot_count": 36,
                      "tar_regular_member_count": 71, "scientific_execution": receipt["scientific_execution_in_this_bridge"]}))


if __name__ == "__main__":
    check()
