"""Frozen real-checkpoint metadata only; no dataset decode/model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from n01 import evaluate_integration006 as driver


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def forbidden(*args, **kwargs):
    raise AssertionError("Dataset decode and model forward are forbidden in metadata-only verification")


def tensor_metadata(value, path=""):
    if isinstance(value, torch.Tensor):
        return [dict(path=path, shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))]
    if isinstance(value, dict):
        return [record for key, item in value.items() for record in tensor_metadata(item, path + "/" + key)]
    if isinstance(value, (list, tuple)):
        return [record for index, item in enumerate(value) for record in tensor_metadata(item, path + "/" + str(index))]
    return []


def main():
    torch.set_num_threads(2)
    config_path = ROOT / "integration006_config.json"
    manifest_path = ROOT / "INTEGRATION006_SOURCE_MANIFEST.json"
    cfg, manifest = read(config_path), read(manifest_path)
    driver.require(driver.sha256(config_path) == driver.CONFIG_SHA == manifest["config_sha256"], "Config bytes changed")
    driver.require(driver.sha256(ROOT / "docs/INTEGRATION_006_PROTOCOL.md") == driver.PROTOCOL_SHA == manifest["protocol_sha256"], "Protocol bytes changed")
    # Check every copied reference before the first checkpoint is opened.
    bound_refs = {}
    for role, item in cfg["references"].items():
        path = (ROOT / item["path"]).resolve()
        driver.require(path.is_relative_to(ROOT), "Reference path escaped ROOT")
        actual = driver.sha256(path)
        driver.require(actual == item["sha256"] == manifest["files"][item["path"]], "Frozen reference bytes changed: " + role)
        bound_refs[role] = actual
    driver.require(len(bound_refs) == 11, "Reference set changed")
    snapshot = ROOT / "provenance/integration006_failed_binding"
    for relative in ("src/n01/evaluate_integration006.py", "tests/test_evaluate_integration006.py"):
        driver.require(driver.sha256(snapshot / Path(relative).name) == manifest["files"][relative], "Original failed-run source snapshot mismatch")
    stats = read(ROOT / cfg["references"]["statistics001"]["path"])
    config001 = read(ROOT / cfg["references"]["config001"]["path"])
    findings = {}
    with patch.object(np, "load", side_effect=forbidden), \
         patch.object(driver, "load_training", side_effect=forbidden), \
         patch.object(driver, "load_validation", side_effect=forbidden), \
         patch.object(torch.nn.Module, "_call_impl", side_effect=forbidden):
        for arm in driver.ARMS:
            state = torch.load(ROOT / cfg["references"][arm]["path"], map_location="cpu", weights_only=False)
            driver.require(state["arm"] == state["variant"] == arm and state["update"] == 20000 and state["stage"] == "low_learning_rate", "Endpoint metadata changed")
            fields = {}
            for field, reference in (("config", config001), ("model_statistics", stats["model"]), ("loss_statistics", stats["loss"])):
                left = driver.metadata_json_bytes(state[field])
                right = driver.metadata_json_bytes(reference)
                driver.require(left == right, "Non-representation metadata difference: " + arm + "/" + field)
                fields[field] = dict(exact_canonical_bytes_equal=True, bytes=len(left),
                                    sha256=hashlib.sha256(left).hexdigest(), checkpoint_tensor_metadata=tensor_metadata(state[field]))
            findings[arm] = fields
            del state
    result = dict(run_id=cfg["run_id"], UTC=datetime.now(timezone.utc).isoformat(),
                  status="passed_exact_metadata_representation_only", CPU_threads=2, checkpoint_decodes=2,
                  scientific_dataset_decodes=0, scientific_model_forwards=0, updates=driver.UPDATES,
                  all_reference_hashes_checked_before_checkpoint_decode=True, reference_sha256=bound_refs,
                  source_manifest_sha256=driver.sha256(manifest_path), config_sha256=driver.CONFIG_SHA,
                  protocol_sha256=driver.PROTOCOL_SHA, driver_sha256=driver.sha256(ROOT / "src/n01/evaluate_integration006.py"),
                  checker_sha256=driver.sha256(Path(__file__)), comparisons=findings,
                  equality="Tensor.detach().cpu().tolist(), recursive finite JSON, sorted keys, compact separators, exact UTF8 bytes; no clean/tolerance/precision conversion")
    output = Path(__file__).with_name("METADATA_ONLY_RECEIPT.json")
    if output.exists():
        raise FileExistsError("Metadata receipt already exists; no replacement")
    driver.write_json(output, result)
    print(json.dumps(dict(status=result["status"], arms=list(findings), checkpoint_decodes=2,
                          dataset_decodes=0, forwards=0, receipt=str(output))))


if __name__ == "__main__":
    main()
