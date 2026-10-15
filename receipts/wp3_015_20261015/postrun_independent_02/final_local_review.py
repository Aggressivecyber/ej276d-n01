"""015 final closure from already collected JSON/journals; Python stdlib only.

This script never opens NPZ or checkpoint payloads and performs no model work.
All destinations are exclusive-created under this E-drive receipt directory.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("E:/GAN/research/N01")
OUT = ROOT / "receipts/wp3_015_20261015/postrun_independent_02"
RECEIPTS = ROOT / "receipts/wp3_015_20261015"
SELECTED = ROOT / "runs/wp3_015_20261015/selected_collected_01"
ART = SELECTED / "runs/primary/artifacts"
READBACK = ROOT / "runs/wp3_015_20261015/independent_readback_collected_01"
UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
REVIEWER = {
    "agent": "/root/wp3_015_solmax_review",
    "assigned_model": "GPT-6.1-Sol",
    "assigned_effort": "Max",
    "assignment_evidence": "Parent relayed the user's latest explicit all-future-work GPT-6.1-Sol-Max instruction.",
    "runtime_model_metadata_independently_exposed": False,
    "prior_authorship_and_receipts_preserved": True,
}
EVIDENCE: dict[str, dict] = {}
CHECKS: list[dict] = []
FORBIDDEN = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors"}


def require(value, name):
    if not value:
        raise AssertionError(name)


def rel(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def record(path):
    path = Path(path)
    require(path.suffix.lower() not in FORBIDDEN, "No raw-array/checkpoint byte reads")
    require(path.resolve().is_relative_to(Path("E:/GAN").resolve()), "Read confined to E project")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    item = {"path": rel(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    EVIDENCE[rel(path)] = item
    return item


def load(path):
    path = Path(path)
    require(path.suffix.lower() == ".json", "JSON-only numeric input")
    record(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bound(path, expected, name):
    actual = record(path)
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"], name)
    return actual


def check(name, details=None):
    CHECKS.append({"check": name, "passed": True, "details": details})


def finite(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def journal(path, expected_operations, counters):
    record(path)
    entered = completed = events = 0
    pending = {}
    charged_entered = collections.Counter()
    charged_completed = collections.Counter()
    by_name = collections.Counter()
    complete_seen = bytearray(expected_operations)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            require(line.endswith("\n"), "No truncated journal tail")
            j = json.loads(line)
            events += 1
            op, event = j["operation_id"], j["event"]
            require(isinstance(op, int) and 0 <= op < expected_operations, "Journal ID range")
            signature = (j["name"], j["charges"])
            if event == "entered":
                require(op == entered and op not in pending, "Unique sequential entered operation")
                pending[op] = signature
                charged_entered.update(j["charges"])
                entered += 1
            elif event == "completed":
                require(op in pending and pending.pop(op) == signature and not complete_seen[op], "Matching unique completion")
                complete_seen[op] = 1
                charged_completed.update(j["charges"])
                by_name[j["name"]] += 1
                completed += 1
            else:
                raise AssertionError("Unexpected failed/unknown journal event: " + event)
    require(not pending and entered == completed == expected_operations, "Every entered operation completed")
    require(events == expected_operations * 2 and all(complete_seen), "Exact complete journal coverage")
    for key, value in counters.items():
        require(charged_entered[key] == charged_completed[key] == value, "Journal charge: " + key)
    require(all(k in counters or v == 0 for k, v in charged_completed.items()), "No unregistered completed counter")
    return {"events": events, "entered": entered, "completed": completed, "operation_id_min": 0,
            "operation_id_max": expected_operations - 1, "failed": 0, "inflight": [],
            "truncated_tail": False, "completed_counts": dict(charged_completed), "operations_by_name": dict(by_name)}


def write_new(name, value):
    p = OUT / name
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return record(p)


def main():
    require(OUT.exists() and OUT.drive.upper() == "E:", "E-only destination")
    for name in ["FINAL_INDEPENDENT_REVIEW.json", "FINAL_DEVELOPMENT_SELECTION.json", "FINAL_CONCLUSION_ZH.md", "FINAL_OUTPUT_RECEIPT.json"]:
        require(not (OUT / name).exists(), "Preserve all existing outputs: " + name)
    record(Path(__file__))
    for p in [ROOT.parent.parent / "AGENTS.md", ROOT / "AGENTS.md", ROOT / "USER_MASTER_PLAN_20261015.txt",
              ROOT / "WP3_DATA_ROLES.json", ROOT / "docs/WP3_ACCEPTANCE_CONTRACT.md", ROOT / "docs/WP3_015_CONFIGURATION.md"]:
        record(p)

    wake_path = RECEIPTS / "readback_execution_01/WAKE_COMPLETION_SUMMARY.json"
    require(record(wake_path)["sha256"] == "1ad797ba820044e537c52ed8a3a8e83b5ffab95b382355b3a1df3ad383f03c82", "Assigned wake-summary hash")
    wake = load(wake_path)
    for item in wake["evidence"].values():
        bound(ROOT / item["path"], item, "Wake evidence binding: " + item["path"])
    check("wake_summary_and_all_10_bound_evidence_files_match_SHA_and_bytes")

    archive = load(RECEIPTS / "readback_collection_01/REMOTE_ARCHIVE.json")
    collection = load(RECEIPTS / "readback_collection_01/COLLECTION_RECEIPT.json")
    readback_files = [p for p in READBACK.rglob("*") if p.is_file()]
    require(len(readback_files) == len(archive["inventory"]) == collection["files"] == 19, "19 collected readback files")
    require(sum(p.stat().st_size for p in readback_files) == wake["collection"]["bytes"] == 513764, "Collected readback bytes")
    require({p.relative_to(READBACK).as_posix() for p in readback_files} == set(archive["inventory"]), "No absent/extra readback file")
    for path, item in archive["inventory"].items():
        bound(READBACK / path, item, "Collected readback inventory: " + path)
    bound(RECEIPTS / "readback_collection_01/readback_output_01.tar", {"bytes": archive["archive_bytes"], "sha256": archive["archive_sha256"]}, "Collected archive")
    canon = json.dumps(archive["inventory"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(hashlib.sha256(canon).hexdigest() == wake["collection"]["inventory_canonical_json_sha256"], "Canonical readback inventory")
    check("all19_collected_files_and_archive_match_registered_collection", {"files": 19, "bytes": 513764, "archive_bytes": archive["archive_bytes"]})

    plan = load(RECEIPTS / "postrun_independent_01/READBACK_PLAN.json")
    ready = load(RECEIPTS / "postrun_independent_01/READBACK_WORKER_READY.json")
    scope = load(RECEIPTS / "postrun_independent_01/READBACK_SCOPE_NOTE.json")
    prelaunch = load(RECEIPTS / "postrun_independent_01/ASTRA_PRELAUNCH_REVIEW.json")
    execution = load(RECEIPTS / "readback_registration_01/READBACK_EXECUTION.json")
    for document in [plan, ready]:
        for key in ["frozen_configuration", "frozen_manifest", "worker", "kernel", "inventory", "plan", "scope_note"]:
            if key in document:
                item = document[key]
                bound(Path(item["path"]), item, "Prelaunch binding: " + key)
    require(prelaunch["passed"] and not prelaunch["blockers"], "Recorded static prelaunch pass")
    require(execution["original_training_completed_G"] == execution["original_training_completed_AD"] == 8000, "Original execution budget")
    require(execution["cumulative_training_before_and_after_this_audit"] == 175667, "Cumulative training is not incremented by audit")
    cfg = load(ROOT / "wp3_015_config.json")
    for p, item in cfg["source_files"].items():
        bound(ROOT / p, item, "Frozen source105: " + p)
    require(len(cfg["source_files"]) == 105, "105 immutable source files")
    require(record(ROOT / "wp3_015_config.json")["sha256"] == execution["original_configuration_sha256"], "Frozen config")
    require(record(ROOT / "WP3_SOURCE_MANIFEST.json")["sha256"] == execution["original_source_manifest_sha256"], "Frozen source manifest")
    check("prelaunch_controls_and105_frozen_source_files_unchanged", {"source_files": 105})

    array = load(READBACK / "artifacts/ARRAY_READBACK.json")
    complete = load(READBACK / "artifacts/COMPLETE.json")
    status = load(READBACK / "artifacts/status.json")
    native = load(READBACK / "supervisor/exit_receipt.json")
    remote = load(RECEIPTS / "readback_wake_status_01/REMOTE_STATE.json")
    for p in ["supervisor/exit_receipt.json", "artifacts/COMPLETE.json", "artifacts/status.json", "artifacts/FIRST_PAIR_PROOF.json"]:
        require(remote[p] == load(READBACK / p), "Wake snapshot/local terminal match: " + p)
    require(remote["artifacts/FAILED.json"] is None
            and all(p.get("alive") is False for p in remote["live_processes"].values()), "Wake completed with no live registered process")
    require(native["child_native_exit_code"] == native["exit_code"] == 0 and native["reason"] is None and native["error"] is None, "Readback native0")
    require(array["passed"] and array["status"] == "PASS" and complete["native_status"] == "COMPLETE" and complete["passed"], "Array audit terminal pass")
    expected_readback = plan["expected_complete_counters"]
    for j in [array, complete, status]:
        require(j["counters"] == j["attempted_counters"] == expected_readback, "Exact terminal readback counters")
        require(j["operations_entered"] == j["operations_completed"] == 285 and not j["inflight"], "Readback operation closure")
    rb_journal = journal(READBACK / "artifacts/operations.jsonl", 285, expected_readback)
    require(not native["unknown_inflight_ids"] and not native["failed_ids"] and not native["truncated_tail"], "Native no incomplete work")
    require(array["reviewer_operations"] == wake["scientific_operations"], "Zero scientific work in completed array audit")
    require(all(v == 0 for v in array["reviewer_operations"].values()), "Array-audit scientific operation zeros")

    original_inventory = load(RECEIPTS / "collection_01/REMOTE_FULL_INVENTORY.json")
    omitted = original_inventory["omitted_remote_inventory"]
    require(len(original_inventory["whole_remote_inventory"]) == 300 and len(original_inventory["inventory"]) == 139 and len(omitted) == 161, "Original139/300 plus161raw")
    require(all(Path(p).suffix.lower() == ".npz" for p in omitted), "All omitted original files are NPZ")
    require(array["raw_files_SHA_checked"] == omitted, "Remote array audit matched all161 bound NPZ hashes")
    require(sum(x["bytes"] for x in omitted.values()) == 3064431039, "161 raw compressed bytes")
    require(not list(SELECTED.rglob("*.npz")) and not list(READBACK.rglob("*.npz")), "No original NPZ in local collections")
    selected_files = [p for p in SELECTED.rglob("*") if p.is_file()]
    require(len(selected_files) == 139, "Original selected collection139 files")
    metadata_verified = []
    for p, item in original_inventory["inventory"].items():
        target = SELECTED / p
        require(target.is_file() and target.stat().st_size == item["bytes"], "Selected original file presence and size")
        if target.suffix.lower() not in FORBIDDEN:
            bound(target, item, "Original selected metadata binding: " + p)
            metadata_verified.append(p)
    pairs = array["pairs"]
    expected_pairs = {(f"seed{s}_{c}", snap, arm, h)
                      for s in [0, 1] for c in ["M0", "M1"] for h in cfg["hypotheses"]
                      for snap, arms in [("initial_evaluation", ["A", "initial"]), ("final_evaluation", ["final"])] for arm in arms}
    require(len(pairs) == len(expected_pairs) == 36, "36 exact audit pairs")
    require({(p["trial"], p["snapshot"], p["arm"], p["hypothesis"]) for p in pairs} == expected_pairs, "Exact trial/snapshot/arm/hyp coverage")
    for p in pairs:
        require(p["parents"] == 6593 and p["rows_per_bank"] == 26372 and p["r4_native_moments_verified"]
                and p["r8_derived_from_saved4plus4"] and p["pairwise_finite_joint_counts_verified"], "Every pair passed declared numeric scope")
        require(p["covariance_denominator"] == p["waveform_rank_upper_bound"] == 7
                and p["full_scalar_covariance_shape"] == [6593, 9, 9]
                and not p["r8_native_report_exists"] and not p["dense256x256_covariance_materialized"], "Correct derived8 covariance interpretation")
        require(p["IEEE_correction_absmax_V"] <= 4e-15 and len(p["r4_by_bank"]) == 2, "Saved-window reconstruction bound")
    require(not array["native_complete_wave_QPSD_noise_distances_independently_replayed"]
            and not array["original_C1_median_fit_independently_replayed"]
            and not array["preADC384_and_fullcarrier_independent_readback"]
            and array["mean_evidence_REAL_summary_SHA_only_no_decode"], "Retain exact audit limitations")
    check("bounded_saved_array_result_scope_is_complete_and_correctly_limited", {"pairs": 36, "raw_SHA_files": 161, "decoded_evaluation_NPZ": 160, "decoded_hard_rows": 1898784})
    check("readback_journal_and_native_terminal_are_consistent", rb_journal)

    train = load(ART / "COMPLETE.json")
    train_native = load(SELECTED / "runs/primary/supervisor/exit_receipt.json")
    require(train_native["child_native_exit_code"] == train_native["exit_code"] == 0, "Original015 native0")
    require(train["native_status"] == "COMPLETE" and train["all_four_trials_completed"] and train["scientific_training_updates"] == 8000, "Original015 completed exactly8000G")
    tr_journal = journal(ART / "operations.jsonl", 154518, train["counters"])
    require(train["counters"] == train["attempted_counters"], "No excess/inflight original work")
    require(not train["inflight"] and train["all_frozen_A_005_unchanged"] and train["no_DEV"], "Original frozen/reference and DEV boundaries")
    require(train["counters"]["G_updates"] == train["counters"]["new_AD_queries"] == 8000, "Original G and AD are8000 each")
    for t in train["trials"]:
        trial = load(ART / t["directory"] / "COMPLETE.json")
        require(record(ART / t["directory"] / "COMPLETE.json")["sha256"] == t["COMPLETE_sha256"], "Trial COMPLETE hash")
        require(trial["native_status"] == "COMPLETE" and trial["G_updates"] == 2000 and trial["frozen_A_005_unchanged"], "Each original trial2000G")
        require(trial["coverage"]["source_unique"] == 6593 and trial["coverage"]["real_unique"] == 21047, "Each original trial all legal TRAIN covered")
        require((trial["candidate"] == "M0" and all(v == 0 for v in trial["mean_theta"])) or
                (trial["candidate"] == "M1" and trial["protected_C1_shape_unchanged"]), "Meanoff M0 / frozen-head M1")
    gate = load(ART / "MEAN_RESIDUAL_GATE.json")
    require(gate["all6_passed"] and len(gate["cells"]) == 6, "Original mean gate ALL6")
    for c in gate["cells"]:
        threshold = max(1e-7, 3 * math.sqrt(c["bank_MC_SE_RMS_V"] ** 2 + c["real_cluster_SE_RMS_V"] ** 2))
        require(abs(threshold - c["required_residual_RMS_V"]) <= 1e-15
                and c["residual_RMS_V"] > threshold and c["bank_residual_cosine"] > .5, "Saved mean-gate scalar formula")
    check("original015_training_journal_native_counts_coverage_and_gate_scalars_close", {"journal": tr_journal, "actual_counts": train["counters"], "cumulative_training_unchanged": 175667})

    tolerance = load(ROOT / "CALIBRATED_TOLERANCES_WP1_013.json")
    old = load(RECEIPTS / "postrun_independent_01/DEVELOPMENT_SELECTION.json")
    require(record(ROOT / "CALIBRATED_TOLERANCES_WP1_013.json")["sha256"] == old["tolerance_table_sha256"], "Frozen T table")
    tmap = tolerance["T_by_hypothesis"]
    cells, candidates, metrics_by_trial = [], {}, {}
    for s in [0, 1]:
        paired_initial = []
        for c in ["M0", "M1"]:
            name = f"seed{s}_{c}"
            initial = load(ART / name / "initial_evaluation/METRICS.json")
            final = load(ART / name / "final_evaluation/METRICS.json")
            metrics_by_trial[name] = {"initial": initial, "final": final}
            paired_initial.append(initial)
            for obj in [initial, final]:
                require(obj["source_parents"] == 6593 and obj["draws_per_bank"] == 4 and obj["MC_banks"] == 2
                        and obj["REAL_rows"] == 21047 and obj["no_DEV"] and obj["all_signed_rows_retained"], "Frozen evaluation population")
                for v in obj["metrics"].values():
                    require(v["all_rows_retained"] and v["reference_rows"] == 21047 and v["fake_rows"] == 26372
                            and v["coverage"]["conditional_coverage_complete"] and v["coverage"]["conditional_bins_used"] == 8,
                            "No missing evaluation rows/conditional bins")
            for h in cfg["hypotheses"]:
                components, required_pending, violations, uncalibrated = {}, [], [], []
                for feature, T in tmap[h].items():
                    refs = {}
                    for ref, obj, arm in [("A", initial, "A"), ("common_initial", initial, "initial"), ("final", final, "final")]:
                        vals = [obj["metrics"].get(f"bank{b}_{arm}_{h}", {}).get("values", {}).get(feature) for b in [0, 1]]
                        refs[ref] = sum(vals) / 2 if all(finite(v) for v in vals) else None
                    defined = all(finite(v) for v in refs.values())
                    calibrated = finite(T) and T > 0
                    da = refs["final"] - refs["A"] if defined else None
                    di = refs["final"] - refs["common_initial"] if defined else None
                    if not calibrated:
                        outcome = "pending_not_calibrated"
                        uncalibrated.append(feature)
                    elif not defined:
                        outcome = "pending_required_component_missing_or_undefined"
                        required_pending.append(feature)
                    elif da <= T and di <= T:
                        outcome = "noninferior_to_both_references"
                    else:
                        outcome = "failed_dual_reference_noninferiority"
                        violations.append(feature)
                    components[feature] = {**refs, "T": T, "delta_final_minus_A": da,
                                           "delta_final_minus_common_initial": di, "status": outcome,
                                           "absolute_distance_over_T": refs["final"] / T if calibrated and defined else None,
                                           "absolute_within_T": refs["final"] <= T if calibrated and defined else None}
                wave = components["wave"]
                improvement = (wave["A"] - wave["final"]) / wave["A"]
                cells.append({"candidate": c, "seed": s, "hypothesis": h, "eligible_cell": not required_pending and not violations,
                              "required_pending": required_pending, "noninferiority_failures": violations,
                              "wave_relative_improvement_vs_A": improvement,
                              "wave_relative_improvement_vs_common_initial": (wave["common_initial"] - wave["final"]) / wave["common_initial"],
                              "wave_usefulness_2percent": improvement >= .02, "components": components,
                              "pending_uncalibrated": uncalibrated,
                              "absolute_T_all_positive_defined": all(x["absolute_within_T"] is True for x in components.values() if finite(x["T"]) and x["T"] > 0)})
        require(paired_initial[0] == paired_initial[1], "M0 and M1 same-seed complete initial metrics identical")
    for c in ["M0", "M1"]:
        cc = [x for x in cells if x["candidate"] == c]
        values = [metrics_by_trial[f"seed{s}_{c}"]["final"]["metrics"][f"bank{b}_final_{h}"]["values"]["wave"] for s in [0, 1] for h in cfg["hypotheses"] for b in [0, 1]]
        initials = [metrics_by_trial[f"seed{s}_{c}"]["initial"]["metrics"][f"bank{b}_initial_{h}"]["values"]["wave"] for s in [0, 1] for h in cfg["hypotheses"] for b in [0, 1]]
        require(len(values) == len(initials) == 12, "Exactly12 wave distances per candidate")
        candidates[c] = {"eligible": all(x["eligible_cell"] for x in cc), "dual_reference_noninferior_cells": sum(x["eligible_cell"] for x in cc),
                         "final_wave_mean12": sum(values) / 12, "common_initial_wave_mean12": sum(initials) / 12,
                         "relative_wave_usefulness_cells": sum(x["wave_usefulness_2percent"] for x in cc),
                         "relative_wave_usefulness_ALL6": all(x["wave_usefulness_2percent"] for x in cc),
                         "absolute_T_ALL6": all(x["absolute_T_all_positive_defined"] for x in cc),
                         "calibrated_components_per_hypothesis": {h: sum(finite(v) and v > 0 for v in tmap[h].values()) for h in cfg["hypotheses"]}}
    eligible = [c for c in ["M0", "M1"] if candidates[c]["eligible"]]
    require(eligible, "At least one eligible development candidate")
    best = min(candidates[c]["final_wave_mean12"] for c in eligible)
    tied = [c for c in eligible if candidates[c]["final_wave_mean12"] - best <= 1e-12]
    winner = "M0" if "M0" in tied else tied[0]
    improved = candidates[winner]["common_initial_wave_mean12"] - candidates[winner]["final_wave_mean12"] > 1e-12
    selected = winner if improved else "common_C1_initial"
    for c in ["M0", "M1"]:
        require(candidates[c] == old["candidates"][c], "Independent aggregate agrees with provisional: " + c)
    old_cells = {(c["candidate"], c["seed"], c["hypothesis"]): c for c in old["cells"]}
    for c in cells:
        o = old_cells[(c["candidate"], c["seed"], c["hypothesis"])]
        require(c["components"] == o["components"] and c["eligible_cell"] == o["eligible_cell"]
                and c["wave_usefulness_2percent"] == o["wave_usefulness_2percent"], "Independent full cell/component algebra agrees")
    require(selected == "M0" and winner == old["winner_by_mean12"], "Final015 development reference M0")
    check("independent_saved_JSON_dualref_ALL6_and_mean12_selection_matches_frozen_rule", {"candidates": candidates, "selected": selected})

    residuals = {}
    for feature in tmap["fast"]:
        defined = [c["components"][feature] for c in cells if c["candidate"] == "M0"]
        ratios = [v["absolute_distance_over_T"] for v in defined if finite(v["absolute_distance_over_T"])]
        if ratios:
            residuals[feature] = {"distance_over_T_min": min(ratios), "distance_over_T_max": max(ratios),
                                  "absolute_pass_cells": sum(v["absolute_within_T"] is True for v in defined), "cells": 6}
    rep_path = ART / "seed0_M0/final_evaluation/bank0_final_fast_REPORT.json"
    report = load(rep_path)
    pointers = []
    def evidence_pointer(pointer, value, meaning):
        pointers.append({"file": rel(rep_path), "sha256": record(rep_path)["sha256"], "json_pointer": pointer, "value": value, "meaning_zh": meaning})
    for pointer, value, meaning in [
        ("/population/integrals/Q_Vns/fake/std_population", report["population"]["integrals"]["Q_Vns"]["fake"]["std_population"], "M0全人口Q标准差，包含SOURCE人口与硬观察效应"),
        ("/population/integrals/Q_Vns/real/std_population", report["population"]["integrals"]["Q_Vns"]["real"]["std_population"], "REAL全人口Q标准差；不是相同输入的条件方差"),
        ("/population/morphology/peak_voltage_V/fake/mean", report["population"]["morphology"]["peak_voltage_V"]["fake"]["mean"], "M0全人口峰值均值"),
        ("/population/morphology/peak_voltage_V/real/mean", report["population"]["morphology"]["peak_voltage_V"]["real"]["mean"], "REAL全人口峰值均值"),
        ("/population/morphology/rise10_90_ns/fake/mean", report["population"]["morphology"]["rise10_90_ns"]["fake"]["mean"], "M0已定义rise的全人口均值"),
        ("/population/morphology/rise10_90_ns/real/mean", report["population"]["morphology"]["rise10_90_ns"]["real"]["mean"], "REAL已定义rise的全人口均值"),
        ("/population/morphology/FWHM_ns/fake/std_population", report["population"]["morphology"]["FWHM_ns"]["fake"]["std_population"], "M0全人口FWHM标准差"),
        ("/population/morphology/FWHM_ns/real/std_population", report["population"]["morphology"]["FWHM_ns"]["real"]["std_population"], "REAL全人口FWHM标准差"),
        ("/output_Q_conditions/bins/0/fake/peak_voltage_V/mean", report["output_Q_conditions"]["bins"][0]["fake"]["peak_voltage_V"]["mean"], "低输出Q箱的M0峰值均值"),
        ("/output_Q_conditions/bins/0/real/peak_voltage_V/mean", report["output_Q_conditions"]["bins"][0]["real"]["peak_voltage_V"]["mean"], "同一输出Q箱的REAL峰值均值；与总体残差方向不同"),
        ("/output_Q_conditions/bins/0/fake/FWHM_ns/mean", report["output_Q_conditions"]["bins"][0]["fake"]["FWHM_ns"]["mean"], "低输出Q箱的M0 FWHM均值"),
        ("/output_Q_conditions/bins/0/real/FWHM_ns/mean", report["output_Q_conditions"]["bins"][0]["real"]["FWHM_ns"]["mean"], "低输出Q箱的REAL FWHM均值"),
    ]:
        evidence_pointer(pointer, value, meaning)
    next_step = {
        "recommendation": "support_one_bounded_structural_single_factor_q_sig_test_only",
        "recommendation_zh": "支持用WP3剩余两次短试检验全局a、b电荷分支；冻结M0/A/005并保持均值关闭。当前证据支持补足明确缺失的canonical电荷自由度，效果须由原硬观察评价决定。",
        "evidence": pointers,
        "meanoff_M0_absolute_residuals_ALL6": residuals,
        "parameterization": {"Q_mu_domain": "Q_mu>0; explicitly validate the frozen canonical mean on every legal parent before using a/Q_mu",
                             "variance": "Var(q_sig|x)=a*Q_mu+b*Q_mu^2, a>=0,b>=0",
                             "lognormal_s2": "s^2=log1p(a/Q_mu+b)", "sample": "q_sig=Q_mu*exp(s*Z-0.5*s^2), Z~N(0,1)",
                             "moments": "E(q_sig|x)=Q_mu and Var(q_sig|x)=a*Q_mu+b*Q_mu^2 analytically before hard ADC/cropping",
                             "assembly": "q_sig*u_mu + B_shape(x)*eta_shape + frozen005; B amplitude remains determined by frozen A/input",
                             "initial_independence": "Use an explicitly independent charge latent for the first two-parameter test; introduce no charge-shape correlation parameter",
                             "disabled_state": "Branch off returns q_sig exactly Q_mu with both random term and mean correction disabled",
                             "training_numerics_to_resolve_before_execution": "Handle the zero-variance sqrt boundary and nonnegative parameterization without singular gradients or a zero-gradient squared initialization; use one predeclared small positive initialization and preserve the exact disabled baseline"},
        "test_boundary": {"remaining_WP3_short_trials": 2, "trainable_parameters": 2, "new_shape_conditioning": False,
                          "new_time_or_local_noise_or_mean_parameters": False, "electronics_refit": False,
                          "population_adjustment": False, "DEV_numeric_reads": False, "implemented_or_launched_by_this_review": False},
        "decision_and_stop": "Freeze initialization, streams, per-seed2000G or an explicitly registered budget, and decision rule before launch. Compare each seed's frozen M0 q-off reference with the q-on candidate using mean2bank distances, ALL6 positive-defined-T noninferiority, full-wave usefulness, conditional PSD and failures, and absolute T separately. Retain q-off if the registered gain is absent; no near-threshold extension or extra candidate after the two WP3 slots.",
        "identifiability_limit_zh": "全人口Q标准差混合了SOURCE人口、条件均值、条件方差及ADC/寻峰/裁窗；3.0993与3.1528的差不能识别真实a、b，也不能把新分支参数解释成光电子效率。重复父事件给出模型条件方差，未配对REAL没有对应的相同输入真值。",
        "residual_interpretation_zh": "峰值、rise、FWHM和baseline DC各有独立残差：低Q峰值方向与全人口方向不同，低Q宽度存在显著形态差，不能统一归因于电荷分辨率。q分支对硬观察后的均值/形态可有非线性影响，需继续逐项报告；prefix100不变，DC残差保留为电子学限制。",
        "finite_draw_limit": "The mean correction is exact in expectation. A finite set of four/eight draws, even antithetic Gaussian Z pairs, need not have exactly Q_mu as its lognormal sample mean; report MCSE and never subtract the realized sample mean to manufacture the identity.",
        "scientific_status": "response_match_not_established",
    }

    selection = {"schema": "n01/wp3-final-independent-development-selection/v1", "UTC": UTC, "reviewer": REVIEWER,
                 "run_id": "wp3_015_20261015", "status": "independently_reviewed_development_research_reference", "passed": True,
                 "provisional": False, "readback_pending": False, "frozen_configuration_sha256": record(ROOT / "wp3_015_config.json")["sha256"],
                 "tolerance_table_sha256": record(ROOT / "CALIBRATED_TOLERANCES_WP1_013.json")["sha256"],
                 "inventory_sha256": record(RECEIPTS / "collection_01/REMOTE_FULL_INVENTORY.json")["sha256"],
                 "bank_aggregation": "mean2bank separate empirical distances, not pooled8", "candidates": candidates, "cells": cells,
                 "eligible_candidates": eligible, "winner_by_mean12": winner, "tie_epsilon": 1e-12, "tie_preference": "M0",
                 "winner_better_than_common_initial_by_more_than_epsilon": improved, "selected_research_reference": selected,
                 "selected_mean_adapter_enabled": False, "weights_promoted": False, "scientific_acceptance": False,
                 "absolute_T_ALL6_passed": False, "independent_raw_saved_array_readback_complete": True,
                 "full_empirical_loss_independently_replayed": False, "original_C1_median_fit_independently_replayed": False,
                 "preADC384_fullcarrier_independent_readback": False, "mean_evidence_REAL_summary_SHA_only_no_decode": True,
                 "raw_arrays_remain_remote": True, "original014_C1_historical_usefulness": "5/6 remains unchanged; this015 M0 continuation is a separate completed candidate",
                 "interpretation_zh": "015有限训练与保存数组审查已独立闭合。M0满足固定双参考ALL6非劣、用途6/6并以12格wave均值胜出，选为后续开发研究参考，均值分支关闭。M1用途0/6且不优于共同初始。绝对T仍失败，权重不作科学晋级。",
                 "scientific_status": "response_match_not_established", "evidence_scope": scope,
                 "completed_array_audit": {"path": rel(READBACK / "artifacts/ARRAY_READBACK.json"), "sha256": record(READBACK / "artifacts/ARRAY_READBACK.json")["sha256"], "pairs": 36},
                 "decision_recomputed_from": [rel(ART / f"seed{s}_{c}/{phase}/METRICS.json") for s in [0, 1] for c in ["M0", "M1"] for phase in ["initial_evaluation", "final_evaluation"]]}
    selection_record = write_new("FINAL_DEVELOPMENT_SELECTION.json", selection)
    review = {"schema": "n01/wp3-final-local-independent-review/v1", "UTC": UTC, "reviewer": REVIEWER,
              "run_id": "wp3_015_20261015", "status": "PASS_BOUNDED_LOCAL_CLOSURE", "passed": True,
              "conclusion_zh": selection["interpretation_zh"], "checks": CHECKS, "blockers": [],
              "selected_research_reference": selected, "selected_mean_adapter_enabled": False,
              "final_selection": selection_record, "original_training_counts": train["counters"],
              "cumulative_training_before_and_after_review": 175667,
              "array_readback_journal": rb_journal, "original_training_journal": tr_journal,
              "local_collection_scope": {"original_selected_files": 139, "original_remote_files": 300, "original_NPZ_retained_remote": 161,
                                         "original_raw_bytes_retained_remote": 3064431039, "original_metadata_files_SHA_verified_this_review": len(metadata_verified),
                                         "opaque_thin_checkpoint_payloads_not_opened_this_review": len(selected_files) - len(metadata_verified),
                                         "readback_files": 19, "readback_bytes": 513764, "local_original_NPZ": 0},
              "independently_completed_scope": scope["independently_recomputed"],
              "numeric_evidence_boundary": {"completed_registered_CPU_audit": True, "local_reviewer_redecoded_arrays": False,
                                           "full_empirical_CDF_or_loss_replay": False, "original_C1_median_refit": False,
                                           "preADC384_fullcarrier_reconstruction": False, "checkpoint_tensor_review": False,
                                           "REAL_mean_and_cluster_SE_array_replay": False, "r8_is_new_derived4plus4_not_native_report": True,
                                           "empirical_hard_sample_rank_le7_is_not_preADC_theoretical_rank": True,
                                           "pairwise_missing_scalar_covariance_automatic_PSD_claim": False},
              "remaining_absolute_residuals": residuals, "next_step_recommendation": next_step,
              "actual_local_review_operations": {"stdlib_only": True, "new_CP_loads": 0, "new_NPZ_or_NPY_decodes": 0,
                                                "new_NN_rows": 0, "new_AD_queries": 0, "new_training_updates": 0, "new_optimizer_or_sampler": 0,
                                                "new_random_draws": 0, "new_CDF_or_loss_replays": 0, "original_TRAIN_or_DEV_waveform_reads": 0,
                                                "GPU_work": 0, "SSH_calls": 0, "remote_status_or_collection_calls": 0,
                                                "numeric_fixture_reruns": 0, "root_metadata_edits": 0, "git_operations": 0,
                                                "new_data_source_cache_or_temp_paths": [OUT.as_posix()]},
              "weights_promoted": False, "scientific_acceptance": False, "scientific_status": "response_match_not_established",
              "evidence_manifest": list(EVIDENCE.values())}
    review_record = write_new("FINAL_INDEPENDENT_REVIEW.json", review)
    m0, m1 = candidates["M0"], candidates["M1"]
    gain = m0["common_initial_wave_mean12"] - m0["final_wave_mean12"]
    lines = [
        "015 已完成本地独立闭合，M0 成为下一阶段开发研究参考，均值分支关闭。",
        "",
        f"固定规则复算：M0/M1 均满足双参考 ALL6 正且有定义 T 非劣；12 格完整 wave 均值分别为 {m0['final_wave_mean12']:.17g} / {m1['final_wave_mean12']:.17g}，共同初始为 {m0['common_initial_wave_mean12']:.17g}。M0 相对共同初始改善 {gain:.17g}（{100*gain/m0['common_initial_wave_mean12']:.6f}%）；相对 A 的 2% 用途门为 6/6，M1 为 0/6。",
        "",
        "保存数组 CPU 审查已完成 36 个双库配对、161 个原文件 SHA、160 个评价 NPZ、1,898,784 条硬输出行；285 个操作全部闭合，worker/supervisor native exit 均为 0。本轮只读取回收 JSON/journal 并验证哈希，未解码原 NPZ、加载 checkpoint 或执行模型。",
        "",
        "绝对残差仍存在：M0 的 peak/rise/FWHM 分别约为 6.7/7.4/4.8 倍 T，baseline-mean 约为 33 倍 T；部分 QPSD/失效率也未达绝对 T。未标定分量继续 pending。科学状态保持 response_match_not_established，weights_promoted=false；014 原 C1 的 5/6 记录保留。",
        "",
        "支持下一步仅用剩余两次 WP3 短试检验全局两参数 q_sig：冻结 M0/A/005、mean off，使用正值且解析均值校正的 Var(q|x)=aQ+bQ²；B 的幅度仍由确定性 A/输入控制。建议 s²=log1p(a/Q+b)、q=Q exp(sZ−s²/2)，并在开跑前解决 Q>0 与零方差边界的稳定参数化。关闭分支应精确回到 q=Q。",
        "",
        "现有全人口 Q 标准差（代表格 M0 3.099316753724642、REAL 3.152829342399848）不能识别相同输入下的真实条件方差。峰值残差在低 Q 与全人口方向不同，rise/FWHM/DC 继续逐项保留；不把这些残差统一归因于电荷。新分支只作有界结构检验，未获原冻结用途/非劣门则保留 q-off，不补训扩大预算。",
        "",
        "审查边界：完成 savedADC 逆变换/roundtrip、行特征/标志、native 4-draw 矩/MCSE、9×9 pairwise-finite 协方差及新派生 4+4 draw 分母 7 合并。未独立重放完整 CDF/loss、C1 median 或完整 preADC384；REAL mean 摘要为 SHA-only。经验 hard sample rank≤7 与理论 preADC 秩分别解释，缺失配对协方差无自动 PSD 保证。",
    ]
    summary_path = OUT / "FINAL_CONCLUSION_ZH.md"
    with summary_path.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    summary_record = record(summary_path)
    output = {"schema": "n01/wp3-final-independent-output-receipt/v1", "UTC": UTC, "status": "PASS",
              "files": [selection_record, review_record, summary_record, record(Path(__file__))],
              "all_outputs_reopened_json_or_utf8": True, "scientific_acceptance": False}
    for p in [OUT / "FINAL_DEVELOPMENT_SELECTION.json", OUT / "FINAL_INDEPENDENT_REVIEW.json"]:
        require(json.loads(p.read_text(encoding="utf-8"))["passed"], "Final JSON reopened")
    require(summary_path.read_text(encoding="utf-8").startswith("015 已完成"), "Chinese summary reopened")
    receipt = write_new("FINAL_OUTPUT_RECEIPT.json", output)
    print(json.dumps({"status": "PASS", "outputs": output["files"], "output_receipt": receipt,
                      "selected": selected, "candidates": candidates, "checks": len(CHECKS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
