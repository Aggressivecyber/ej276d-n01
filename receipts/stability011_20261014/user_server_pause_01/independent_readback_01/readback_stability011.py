"""Local closure QA of saved 011 records; no source data, Torch, models or CDF calls.

Only the four small saved metric/multiplicity/identity/projection NPZ files are
decoded. The 500 original proxy distances are not recomputed. Outputs are new
files in this reviewer-owned E-drive directory; collected bytes stay read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import traceback

import numpy as np


ROOT = Path("E:/GAN/research/N01")
COLLECTED = ROOT / "runs/stability011_20261014/collected_01"
OUT = Path(__file__).resolve().parent
ART = COLLECTED / "runs/primary/artifacts"
COLLECTION = ROOT / "receipts/stability011_20261014/user_server_pause_01/collection_01/COLLECTION_RECEIPT.json"
FREEZE = ROOT / "receipts/stability011_20261014/freeze_01"
B = 500
ARMS = ["frozen_A", "shape010_final"]
HYPOTHESES = ["fast", "slow", "common_level"]
ZERO_KEYS = ["checkpoint_loads", "model_instances", "model_forward_rows", "gradient_evaluations",
             "G_updates", "D_updates", "noise_updates", "population_updates", "C2ST_updates"]
checks = []
numeric_errors = {}


def require(condition, name, detail=None):
    if not bool(condition):
        raise AssertionError(name + (": " + str(detail) if detail is not None else ""))
    checks.append({"name": name, "passed": True})


def same(actual, expected, name, atol=1e-13):
    a, e = np.asarray(actual), np.asarray(expected)
    require(a.shape == e.shape, name + "_shape", (a.shape, e.shape))
    error = float(np.max(np.abs(a.astype(float)-e.astype(float)))) if a.size else 0.
    numeric_errors[name] = error
    require(np.isfinite(a).all() and np.isfinite(e).all() and error <= atol, name, error)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def document(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def journal(path):
    data = Path(path).read_bytes()
    require(not data or data.endswith(b"\n"), Path(path).name + "_no_partial_last_line")
    return [json.loads(line) for line in data.splitlines()]


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "standard_deviation": float(x.std(ddof=1)),
            "minimum": float(x.min()), "maximum": float(x.max()),
            "percentile_resampling_interval95": np.quantile(x, [.025, .975], method="linear").tolist()}


def check_description(saved, actual, name):
    for key in actual:
        same(saved[key], actual[key], name + ":" + key)


def main():
    require(str(OUT).replace("\\", "/").startswith("E:/GAN/research/N01/receipts/stability011_20261014/user_server_pause_01/independent_readback_01"), "reviewer_E_output_scope")
    require(not (OUT/"READBACK.json").exists() and not (OUT/"CONCLUSION.md").exists(), "exclusive_new_deliverables")
    collection = document(COLLECTION)
    complete = document(ART/"COMPLETE.json")
    summary = document(ART/"STABILITY_RESULT.json")
    status = document(ART/"status.json")
    first = document(ART/"FIRST_BOOTSTRAP_PROOF.json")
    frozen_inputs = document(ART/"FROZEN_INPUTS.json")
    projection = document(ART/"PROJECTION_CONTEXT.json")
    point = document(ART/"POINT_COMPARISON.json")
    start = document(COLLECTED/"START_RECEIPT.json")
    launch = document(COLLECTED/"runs/primary/supervisor/launch.json")
    native = document(COLLECTED/"runs/primary/supervisor/exit_receipt.json")
    policy = document(COLLECTED/"stability011_policy.json")
    manifest = document(COLLECTED/"STABILITY011_SOURCE_MANIFEST.json")
    binding = document(COLLECTED/"STABILITY011_BINDINGS.json")

    require(Path(collection["collected_root"]).resolve() == COLLECTED.resolve(), "collection_exact_local_root")
    archive = Path(collection["archive"])
    require(digest_file(archive) == collection["archive_sha256"], "collection_tar_SHA256")
    full_inventory = {}
    with tarfile.open(archive, "r:") as tar:
        for member in tar.getmembers():
            name = member.name
            parsed = PurePosixPath(name)
            require(member.isfile() and not parsed.is_absolute() and str(parsed) == name
                    and ".." not in parsed.parts and "\\" not in name and ":" not in name
                    and name not in full_inventory, "regular_canonical_unique_tar_member:" + name)
            with tar.extractfile(member) as f:
                h = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: f.read(1024*1024), b""):
                    size += len(chunk)
                    h.update(chunk)
            target = COLLECTED.joinpath(*parsed.parts)
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == size == member.size
                    and digest_file(target) == h.hexdigest(), "collected_tar_SHA_and_size:" + name)
            full_inventory[name] = {"bytes": size, "sha256": h.hexdigest()}
    local_names = {p.relative_to(COLLECTED).as_posix() for p in COLLECTED.rglob("*") if p.is_file()}
    require(local_names == set(full_inventory) and len(full_inventory) == collection["file_count"] == 25,
            "exact_25_collected_regular_file_inventory")
    require(len(complete["artifacts"]) == collection["job"]["artifact_count"] == 14, "exact14_COMPLETE_artifacts")
    for name, expected in complete["artifacts"].items():
        require(full_inventory["runs/primary/artifacts/"+name] == expected, "COMPLETE_artifact_SHA_and_size:" + name)
    require(not (ART/"FAILED.json").exists() and not (ART/"SUMMARY.json").exists(), "no_FAILED_and_actual_summary_is_STABILITY_RESULT")

    require(policy["bootstrap_replicates"] == B and policy["source_parent_clusters"] == 512
            and policy["REAL_reference_original_groups"] == 419 and policy["draws_per_parent"] == 4,
            "registered500_joint512parent419group_scope")
    require(policy["interval"]["quantiles"] == [.025, .975] and policy["interval"]["method"] == "linear"
            and not policy["interval"]["formal_frequentist_coverage_established"]
            and not policy["interval"]["formal_significance_test"], "registered_descriptive_percentile_scope")
    require(len(manifest["files"]) == 63 and len(binding["files"]) == 16
            and manifest["bound_inputs"] == {k:v["sha256"] for k,v in binding["files"].items()}, "frozen63source16input_mapping")
    for name in ["STABILITY011_SOURCE_MANIFEST.json", "STABILITY011_BINDINGS.json", "stability011_policy.json"]:
        frozen_path = FREEZE/name if name != "stability011_policy.json" else ROOT/name
        require(digest_file(COLLECTED/name) == digest_file(frozen_path), "collected_exact_frozen_control:" + name)
    require(binding["source_manifest"]["sha256"] == digest_file(COLLECTED/"STABILITY011_SOURCE_MANIFEST.json")
            and manifest["policy_sha256"] == binding["policy_sha256"] == digest_file(COLLECTED/"stability011_policy.json"),
            "frozen_control_hash_chain")
    new_only = {"scripts/stability011_ops.py", "scripts/supervise_stability011.py", "scripts/run_stability011.py",
                "scripts/stability011_metrics.py", "stability011_policy.json", "docs/STABILITY011_PROTOCOL.md"}
    old53 = {k:v for k,v in manifest["files"].items() if k not in new_only and k not in manifest["protected_manifests"]}
    require(len(old53) == 53, "original53_manifest_subset_without_new_metadata_copies")
    expected_sources = {**{policy["remote_root"]+"/"+k:v for k,v in manifest["files"].items()},
                        **{policy["shape010_remote_root"]+"/"+k:v for k,v in old53.items()}}
    for doc in [frozen_inputs, first, status, complete, summary]:
        require(doc["sources_sha256"] == expected_sources and doc["inputs_sha256"] == complete["inputs_sha256"],
                "runtime_protection_map_identity:" + doc["phase"])
        require(doc["artificial_only"] is False and doc["run_id"] == "stability011_20261014"
                and set(doc["counters"]) == set(ZERO_KEYS) and all(doc["counters"][k] == 0 for k in ZERO_KEYS)
                and doc["DEV_numeric_read"] is False and doc["SOURCE_waveform_decode"] is False
                and doc["scientific_status"] == "response_match_not_established", "native_zero_new_model_gradient_update_DEV_scope:" + doc["phase"])
    for slot, entry in binding["files"].items():
        require(complete["inputs_sha256"].get(entry["path"]) == entry["sha256"], "runtime_exact_bound_input_hash:" + slot)
    require(start["source_manifest_sha256"] == collection["source_manifest_sha256"] == binding["source_manifest"]["sha256"]
            and start["bindings_sha256"] == collection["bindings_sha256"] == digest_file(COLLECTED/"STABILITY011_BINDINGS.json"),
            "dispatch_collection_frozen_identity")

    sup_pid, child_pid = launch["supervisor_pid"], launch["child_pid"]
    require(sup_pid == start["supervisor_pid"] == native["supervisor_pid"] == collection["job"]["supervisor_pid"]
            and child_pid == native["child_pid"] == collection["job"]["child_pid"]
            and all(v["pid"] == child_pid for v in [frozen_inputs, first, status, complete, summary]), "saved_native_supervisor_child_FIRST_PID_chain")
    require(native["child_native_exit_code"] == native["exit_code"] == 0 and native["reason"] is None
            and native["error"] is None and native["automatic_restart"] is False, "native_exit0_clean_no_restart")
    require(datetime.fromisoformat(start["UTC"]) <= datetime.fromisoformat(launch["started_UTC"])
            <= datetime.fromisoformat(first["UTC"]) <= datetime.fromisoformat(complete["UTC"])
            <= datetime.fromisoformat(native["finished_UTC"]), "saved_native_chronology")
    for doc in [complete, status, summary, native["last_persisted_status"]]:
        require([doc[k] for k in ["bootstrap_replicates_attempted", "bootstrap_replicates_computed_successfully", "bootstrap_replicates_completed"]] == [B, B, B]
                and doc["recorded_unfinished_replicate_ids"] == [], "terminal_exact_attempted_computed_saved500:" + doc["phase"])
    require(1 <= first["bootstrap_replicates_completed"] <= first["bootstrap_replicates_computed_successfully"]
            <= first["bootstrap_replicates_attempted"] <= B and first["point_checks_passed"]
            and first["inputs_unchanged"], "actual_first_persisted_bootstrap_proof")
    require(complete["failed_or_replaced_replicates"] == 0 and complete["inputs_unchanged"]
            and native["entered_not_computed_ids"] == native["computed_not_saved_ids"] == [], "no_replacement_or_terminal_gap")
    limits = policy["resource_limits"]
    require(native["peak_sampled_RSS_bytes"] <= limits["job_tree_RSS_GiB"]*1024**3
            and native["peak_sampled_GPU_bytes"] == 0 and native["peak_output_bytes"] <= limits["output_MiB"]*1024**2
            and native["elapsed_seconds"] <= limits["wall_seconds"], "recorded_sampled_resource_caps")

    attempts = journal(ART/"bootstrap_attempts.jsonl")
    computed = journal(ART/"bootstrap_computed.jsonl")
    saved = journal(ART/"bootstrap.jsonl")
    errors = journal(ART/"bootstrap_worker_errors.jsonl")
    journal_counts = {}
    for name, rows in [("entered", attempts), ("computed", computed), ("saved", saved)]:
        ids = [row["replicate_index"] for row in rows]
        require(len(ids) == len(set(ids)) == B and set(ids) == set(range(B)), name + "_500_unique_IDs0to499_no_gap")
        journal_counts[name] = {"records": len(ids), "unique_ids": len(set(ids)), "missing_ids": [], "duplicate_ids": []}
    require(not errors and all(v["event"] == "entered" for v in attempts), "zero_worker_errors_and_entered_events")
    computed_by_id = {v["replicate_index"]:v for v in computed}
    ordered = sorted(saved, key=lambda v:v["replicate_index"])
    require(all(v == computed_by_id[v["replicate_index"]] for v in saved), "every_saved_record_matches_computed_record")
    require(first["first_replicate"] == saved[0] and first["first_replicate_index"] == saved[0]["replicate_index"], "FIRST_matches_first_actual_saved_record")

    loaded = {}
    for name in ["MULTIPLICITY.npz", "SELECTED_IDENTITIES.npz", "BOOTSTRAP_METRICS.npz", "EVALUATION_PROJECTIONS.npz"]:
        with np.load(ART/name, allow_pickle=False) as saved_npz:
            loaded[name] = {k:saved_npz[k].copy() for k in saved_npz.files}
    multi, ids, metric, banks = [loaded[n] for n in ["MULTIPLICITY.npz", "SELECTED_IDENTITIES.npz", "BOOTSTRAP_METRICS.npz", "EVALUATION_PROJECTIONS.npz"]]
    require(metric["arms"].tolist() == ARMS and metric["hypotheses"].tolist() == HYPOTHESES, "canonical_saved_arm_hypothesis_order")
    for name, count, seed in [("source_parent_multiplicity",512,2026101451),("REAL_group_multiplicity",419,2026101452)]:
        generator = np.random.Generator(np.random.PCG64(seed))
        draw = generator.integers(0,count,size=(B,count))
        expected = np.stack([np.bincount(row,minlength=count) for row in draw]).astype(np.uint16)
        require(multi[name].shape == (B,count) and multi[name].dtype == np.uint16
                and np.array_equal(expected,multi[name]) and np.all(multi[name].sum(axis=1) == count), name + "_exact_registered_PCG64_plan")
    require(first["source_REAL_counts_sha256"] == digest_file(ART/"MULTIPLICITY.npz"), "FIRST_exact_saved_plan_digest")
    require(ids["selected_source_rows"].shape == ids["selected_source_ids"].shape == (512,)
            and len(np.unique(ids["selected_source_rows"])) == len(np.unique(ids["selected_source_ids"])) == 512
            and np.array_equal(ids["source_parent_row"],np.repeat(np.arange(512),4)), "512_unique_parents_with_whole4draw_packs")
    require(np.all((ids["selected_source_rows"] >= 0)&(ids["selected_source_rows"] < 6593))
            and all(ids[k].shape == (2048,) and len(np.unique(ids[k])) == 2048
                    and np.all((ids[k] >= 0)&(ids[k] < 21047)) for k in ["real_reference_rows","real_control_rows"])
            and np.intersect1d(ids["real_reference_rows"],ids["real_control_rows"]).size == 0, "saved_bank_TRAIN_legal_unique_disjoint_indices")
    group_ids, group_index = np.unique(ids["real_reference_original_groups"],return_inverse=True)
    require(len(group_ids) == 419 and np.array_equal(group_ids,ids["REAL_group_ids"])
            and np.array_equal(group_index,ids["REAL_group_index"]), "419_reference_group_order_and_all_rows_retained_mapping")
    real_n = multi["REAL_group_multiplicity"][:,group_index].sum(axis=1)
    require(np.array_equal(metric["real_rows"],np.broadcast_to(real_n[:,None,None],(B,2,3)))
            and np.all(metric["fake_rows"] == 2048), "variable_REAL_mass_and_fixed2048_fake_mass_across_all_arms_hypotheses")

    expected_shapes = {k:(B,2,3) for k in ["wave","QPSD","swd","coordinate","short_lag","proxy","QPSD_global",
        "conditional_bins_used","conditional_coverage_complete","fake_rows","real_rows","fake_nonpositive_q_rows",
        "real_nonpositive_q_rows","fake_guarded_psd_rows","real_guarded_psd_rows"]}
    expected_shapes.update({k:(B,2,3,8) for k in ["Qbin_fake_count","Qbin_real_count","Qbin_supported"]})
    expected_shapes.update({k:(B,) for k in ["S_A","S_final","Delta","relative_contrast_percent"]})
    require(all(metric[k].shape == v and np.isfinite(metric[k]).all() for k,v in expected_shapes.items()), "all_saved_numeric_metric_shapes_finite")
    wave = .5*metric["swd"]+.25*metric["coordinate"]+.25*metric["short_lag"]
    proxy = metric["wave"]+2*metric["QPSD"]
    sa, sf = proxy[:,0].mean(axis=1), proxy[:,1].mean(axis=1)
    delta, relative = sf-sa, 100*(sf/sa-1)
    for key,value in [("wave",wave),("proxy",proxy),("S_A",sa),("S_final",sf),("Delta",delta),("relative_contrast_percent",relative)]:
        same(metric[key],value,"saved_metric_algebra:"+key)
    require(np.all(sa > 0) and np.all(metric["wave"] >= 0) and np.all(metric["QPSD"] >= 0), "finite_positive_ratio_denominator_nonnegative_distances")
    support = (metric["Qbin_fake_count"] >= 16)&(metric["Qbin_real_count"] >= 16)
    same(metric["Qbin_supported"],support,"occurrence_count16_support_flags",atol=0)
    same(metric["conditional_bins_used"],support.sum(axis=-1),"supported_bin_counts",atol=0)
    same(metric["conditional_coverage_complete"],support.all(axis=-1),"all8_bin_coverage_flags",atol=0)
    same(metric["Qbin_fake_count"].sum(axis=-1),metric["fake_rows"],"all_fake_occurrences_assigned_once",atol=0)
    same(metric["Qbin_real_count"].sum(axis=-1),metric["real_rows"],"all_REAL_occurrences_assigned_once",atol=0)
    require(np.all(metric["Qbin_real_count"] == metric["Qbin_real_count"][:,0:1,0:1,:]), "same_REAL_conditional_counts_across_arms_and_hypotheses")
    fallback_cases = {"all8_supported_metric_calls":0,"partially_supported_metric_calls":0,"global_only_metric_calls":0}
    maximum_record_error = 0.
    for ri,row in enumerate(ordered):
        require(row["replicate_index"] == ri, "canonical_replicate_ID:"+str(ri))
        require(row["paired"]["conditional_coverage_complete"] == bool(support[ri].all()), "journal_paired_coverage:"+str(ri))
        for ai in range(2):
            for hi in range(3):
                item = row["metrics"][ai][hi]
                used = item["conditional_bins_used"]
                conditional = item["QPSD_conditional_mean"]
                q = .5*(item["QPSD_global"]+conditional) if used else item["QPSD_global"]
                require((conditional is not None) == (used > 0) and (conditional is None or np.isfinite(conditional)), "journal_conditional_mean_availability:%d:%d:%d"%(ri,ai,hi))
                errors_here = [abs(q-item["components"]["QPSD"])]
                for key in ["wave","QPSD"]:
                    errors_here.append(abs(item["components"][key]-metric[key][ri,ai,hi]))
                for key in ["swd","coordinate","short_lag"]:
                    errors_here.append(abs(item["wave_subcomponents"][key]-metric[key][ri,ai,hi]))
                for key in ["proxy","QPSD_global"]:
                    errors_here.append(abs(item[key]-metric[key][ri,ai,hi]))
                require(item["conditional_counts"] == [{"bin":bi,"fake_count":int(metric["Qbin_fake_count"][ri,ai,hi,bi]),
                    "real_count":int(metric["Qbin_real_count"][ri,ai,hi,bi]),"supported":bool(support[ri,ai,hi,bi])} for bi in range(8)], "journal_exact_Qbin_occurrences:%d:%d:%d"%(ri,ai,hi))
                maximum_record_error = max(maximum_record_error,*errors_here)
                fallback_cases["all8_supported_metric_calls" if used == 8 else "partially_supported_metric_calls" if used else "global_only_metric_calls"] += 1
        for key,value in [("S_A",sa),("S_final",sf),("Delta",delta),("relative_contrast_percent",relative)]:
            maximum_record_error = max(maximum_record_error,abs(row["paired"][key]-value[ri]))
    require(maximum_record_error <= 1e-13, "journal_metric_arrays_and_exact_QPSD_fallback_algebra", maximum_record_error)

    primary = {k:describe(v) for k,v in [("S_A",sa),("S_final",sf),("Delta",delta),("relative_contrast_percent",relative)]}
    for key,value in primary.items():
        check_description(summary["primary"][key],value,"summary_primary:"+key)
    secondary = {}
    for hi,hypothesis in enumerate(HYPOTHESES):
        secondary[hypothesis] = {}
        for key in ["wave","QPSD","proxy","swd","coordinate","short_lag"]:
            values = metric[key][:,:,hi]
            item = {"A":describe(values[:,0]),"final":describe(values[:,1]),"final_minus_A":describe(values[:,1]-values[:,0])}
            for part,value in item.items():
                check_description(summary["secondary_marginal_diagnostics"][hypothesis][key][part],value,"summary_secondary:%s:%s:%s"%(hypothesis,key,part))
            secondary[hypothesis][key] = item
    coverage = support.all(axis=(1,2,3))
    coverage_limited = int((~coverage).sum())
    lower, upper = primary["Delta"]["percentile_resampling_interval95"]
    classification = "coverage_limited" if coverage_limited else "saved_TRAIN_bank_gain_stable_under_paired_parent_group_reweighting" if upper < 0 else "endpoint_worsening_under_this_reweighting" if lower > 0 else "stability_unresolved"
    require(summary["all_replicate_conditional_coverage_complete"] == bool(coverage.all())
            and summary["coverage_limited_replicates"] == coverage_limited
            and summary["classification"] == status["classification"] == complete["classification"] == collection["job"]["classification"] == classification,
            "coverage_priority_registered_classification_and_all_terminal_documents")
    require(summary["formal_frequentist_coverage_established"] is False and summary["formal_significance_test"] is False
            and summary["response_match_established"] is False, "scientific_interpretation_does_not_expand_acceptance")

    h = hashlib.sha256()
    for key in sorted(banks):
        value = banks[key]
        require(value.dtype == np.float32 and np.isfinite(value).all(), "saved_projection_float32_finite:" + key)
        h.update(key.encode("utf8")); h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(b"torch.float32"); h.update(value.tobytes(order="C"))
    fingerprint = h.hexdigest()
    require(projection["native"] == projection["actual"] == first["projection_context"] == complete["projection_context"] == point["projection_context"]
            and projection["actual"] == {"kind":"evaluation","seed":2026101430,"update":None,"fingerprint":fingerprint}
            and projection["fingerprint_match"] and projection["CPU_global_RNG_unchanged"] and projection["factory_called_once"], "saved_projection_bytes_fingerprint_metadata_match_without_factory_call")
    require(point["passed"] and point["maximum_absolute_error"] <= point["tolerance"] == 1e-7
            and point["waveform_projection_columns"] == 8776 and first["point_comparison_sha256"] == summary["point_comparison_sha256"] == digest_file(ART/"POINT_COMPARISON.json"), "native_point_gate_receipt_and_bound_hash")
    same(summary["point"]["S_final"]-summary["point"]["S_A"],summary["point"]["Delta"],"original_point_Delta_algebra")
    same(100*(summary["point"]["S_final"]/summary["point"]["S_A"]-1),summary["point"]["relative_contrast_percent"],"original_point_relative_algebra")
    require("torch" not in sys.modules, "review_did_not_import_Torch_or_create_model")

    interpretation = {
        "classification":classification, "point":summary["point"], "reweighting_mean_is_not_original_point":True,
        "fixed_bank_relative_mean_percent":primary["relative_contrast_percent"]["mean"],
        "fixed_bank_relative_interval95_percent":primary["relative_contrast_percent"]["percentile_resampling_interval95"],
        "negative_Delta_replicates":int((delta < 0).sum()), "positive_Delta_replicates":int((delta > 0).sum()),
        "zero_Delta_replicates":int((delta == 0).sum()), "coverage_limited_replicates":coverage_limited,
        "interval_scope":summary["interval_scope"], "formal_significance_test":False,
        "formal_frequentist_coverage_established":False,"scientific_status":"response_match_not_established",
        "next_action":"Stop after local closure; await explicit user notice that the server and work may restart. No new diagnosis is registered or started by this readback."
    }
    conclusion = (
        "011 本地独立读回通过：回收 25 份文件和 COMPLETE 登记的 14 项产物 SHA256/大小均一致；原生退出码为 0，首份计算证明与监督进程链一致。entered/computed/saved 各为 500 个唯一编号（0–499），无失败、重复、缺口或替换；两个 PCG64 重采样计划逐元素匹配。\n\n"
        f"固定 TRAIN 库内，原始比较点相对改善为 {summary['point']['relative_contrast_percent']:.4f}%；重加权相对差值均值为 {relative.mean():.4f}%，95% 重采样分位区间 [{primary['relative_contrast_percent']['percentile_resampling_interval95'][0]:.4f}%, {primary['relative_contrast_percent']['percentile_resampling_interval95'][1]:.4f}%]。Delta 区间 [{lower:.10f}, {upper:.10f}] 小于 0，500 次均具有全部 8 个 Q 箱的条件支持，故符合预登记的“固定库收益方向稳定”规则；其中 {int((delta>0).sum())} 次 Delta 为正，不能称所有重复均改善。REAL 重复行数 {int(real_n.min())}–{int(real_n.max())}，与保存的 419 组整体重采样一致。\n\n"
        "本次只重建小型随机计划、读保存指标及投影元数据，核其代数和分位汇总；未重算 500 次完整 CDF 代理、未读原始数据或旧波形、未加载模型/CP、未做前向或梯度，未连接服务器。原生记录的全部新模型/梯度/更新计数为 0，DEV 和 SOURCE 波形读取标志为 false。此区间描述单个暴露 TRAIN 库、固定已选模型、每父 4 个已存抽样及人工 50-record 分组的重加权稳定性，不建立新采集独立性、正式检验或科学响应验收；科学状态仍为 response_match_not_established。\n\n"
        "未核项：暂停后的远端实时进程/锁和硬件状态（由根流程处理）；本轮未重新哈希或解码原始 16 项输入，未独立重算真实完整距离、原始 Q 箱成员分配或生成新的投影库。资源峰值为监督采样记录，非连续硬件测量。收尾完成后停止，等待用户通知重新启动；不开展下一诊断。\n"
    )
    with (OUT/"CONCLUSION.md").open("x",encoding="utf-8",newline="\n") as f:
        f.write(conclusion)
    readback = {
        "schema":"n01/stability011-independent-local-closure-readback/v1", "UTC":datetime.now(timezone.utc).isoformat(),
        "status":"PASS", "run_id":"stability011_20261014", "reviewer":"GPT-6.1-Sol-Max independent reviewer",
        "collected_root":str(COLLECTED), "collection_receipt":{"path":str(COLLECTION),"sha256":digest_file(COLLECTION)},
        "script":{"path":str(Path(__file__).resolve()),"sha256":digest_file(__file__)},
        "archive":{"path":str(archive),"sha256":digest_file(archive),"bytes":archive.stat().st_size,"file_count":25},
        "complete_SHA256":digest_file(ART/"COMPLETE.json"), "STABILITY_RESULT_SHA256":digest_file(ART/"STABILITY_RESULT.json"),
        "full25_inventory":full_inventory,"COMPLETE14_artifact_inventory":complete["artifacts"],
        "source_manifest_SHA256":digest_file(COLLECTED/"STABILITY011_SOURCE_MANIFEST.json"),
        "bindings_SHA256":digest_file(COLLECTED/"STABILITY011_BINDINGS.json"),
        "native":{"supervisor_pid":sup_pid,"child_pid":child_pid,"exit_code":native["child_native_exit_code"],
            "finished_UTC":native["finished_UTC"],"finished_plus08":datetime.fromisoformat(native["finished_UTC"]).astimezone(timezone(timedelta(hours=8))).isoformat(),
            "FIRST_replicate_index":first["first_replicate_index"],"FIRST_saved_attempted_computed":[first["bootstrap_replicates_completed"],first["bootstrap_replicates_attempted"],first["bootstrap_replicates_computed_successfully"]],
            "zero_counters":complete["counters"],"DEV_numeric_read":False,"SOURCE_waveform_decode":False,
            "elapsed_seconds":native["elapsed_seconds"],"peak_sampled_RSS_bytes":native["peak_sampled_RSS_bytes"],
            "peak_sampled_GPU_bytes":native["peak_sampled_GPU_bytes"],"peak_output_bytes":native["peak_output_bytes"],
            "resource_measurement_scope":"supervisor sampled peaks from saved native records, not continuous hardware verification"},
        "journal_accounting":journal_counts,"worker_errors":0,"failed_or_replaced_replicates":0,
        "multiplicity":{"source_shape":[500,512],"REAL_shape":[500,419],"exact_regenerated_PCG64_match":True,
            "seeds":{"source":2026101451,"REAL_group":2026101452},"fake_rows":2048,
            "REAL_variable_repeated_rows_range":[int(real_n.min()),int(real_n.max())],"REAL_mass_is_repeated_row_mass_not_equal_group_mean":True},
        "saved_NPZ_metadata":{name:{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in z.items()} for name,z in loaded.items()},
        "projection":{"fingerprint":fingerprint,"matches_native_saved_bytes_and_metadata":True,"new_factory_calls":0},
        "primary_readback":primary,"secondary_readback":secondary,"conditional_fallback_record_counts":fallback_cases,
        "Qbin_minimum_recorded_fake_occurrences":int(metric["Qbin_fake_count"].min()),
        "Qbin_minimum_recorded_REAL_occurrences":int(metric["Qbin_real_count"].min()),
        "maximum_saved_journal_metric_error":maximum_record_error,
        "maximum_summary_or_metric_algebra_error":max(numeric_errors.values()),"numeric_error_details":numeric_errors,
        "interpretation":interpretation,"checks":checks,"passed_checks":len(checks),
        "this_readback_activity":{"small_saved_NPZ_files_decoded":4,"reconstructed_multiplicity_plans":2,
            "new_full_proxy_CDF_evaluations":0,"new_scientific_bootstrap_metrics_computed":0,
            "original_dataset_reads":0,"old_waveform_bank_reads":0,"model_instances":0,"checkpoint_loads":0,
            "model_forwards":0,"gradient_calls":0,"optimizer_updates":0,"DEV_reads":0,"SOURCE_waveform_reads":0,
            "SSH_calls":0,"SCP_calls":0,"server_commands":0,"timers_created":0,"protected_edits":0,
            "TEMP_TMP":os.environ.get("TEMP"),"CPU_env_threads_requested":2,
            "CPU_native_thread_count_measured":False,"Torch_imported":False},
        "not_reverified":["Live remote processes, dispatch lock and server hardware after user pause; root owns stop proof.",
            "Original16 input bytes/data content and raw source/DEV arrays were not opened in this closure.",
            "True waveform CDF distances and original per-row Q-bin assignment were not recomputed.",
            "Projection bank was fingerprinted from saved bytes, not regenerated via Torch.",
            "Incomplete-support fallback was checked against saved records; none of these500 native replicates exercised missing support.",
            "New-acquisition independence, fresh latent draws, training/selection uncertainty and scientific response acceptance remain unestablished."],
        "conclusion":{"path":str(OUT/"CONCLUSION.md"),"sha256":digest_file(OUT/"CONCLUSION.md")},
        "follow_up":"None. Independent local closure is finished; stop and await user's explicit restart notice."
    }
    readback["local_checker_correction"] = {
        "preserved_script":str(OUT/"readback_stability011_attempt01.py"),
        "preserved_failure":str(OUT/"READBACK_FAILURE_attempt01.json"),
        "reason":"Attempt01 incorrectly expected runtime sources map to contain only new63; native map correctly also includes old01053, total116. All native documents agreed; no native output was changed.",
        "preserved_attempt02_script":str(OUT/"readback_stability011_attempt02.py"),
        "preserved_attempt02_failure":str(OUT/"READBACK_FAILURE_attempt02.json"),
        "attempt02_reason":"All numeric and native-record checks passed; a local archive-variable shadowing error prevented receipt serialization. Fixed only the reviewer script variable name; no native output or scientific calculation was changed."}
    with (OUT/"READBACK.json").open("x",encoding="utf-8",newline="\n") as f:
        json.dump(readback,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    assert document(OUT/"READBACK.json")["status"] == "PASS"
    print(json.dumps({"status":"PASS","READBACK":str(OUT/"READBACK.json"),"SHA256":digest_file(OUT/"READBACK.json"),
        "primary":primary,"classification":classification,"checks":len(checks)},ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {"schema":"n01/stability011-independent-local-closure-failure/v1","UTC":datetime.now(timezone.utc).isoformat(),
            "status":"FAILED_READBACK","error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc(),
            "checks_completed":checks,"scientific_new_CDF":0,"new_model_forwards":0,"new_gradients":0,"server_calls":0}
        path = OUT/"READBACK_FAILURE.json"
        if not path.exists():
            path.write_text(json.dumps(failure,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in failure.items() if k != "checks_completed"},ensure_ascii=False),file=sys.stderr)
        raise
