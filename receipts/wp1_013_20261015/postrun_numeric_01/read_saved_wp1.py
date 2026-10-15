"""Saved WP1/013 JSON and selected TRAIN-array readback; no model imports."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics


HYPOTHESES = ("fast", "slow", "common_level")
PRIMARY = ("negative_saved012_total", "prospective_AdamW2001")
KEYS = ("wave", "QPSD", "noise")
LSB = 1.0 / 16383
EPS = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(values):
    return sum(values) / len(values)


def finite(value):
    return value is not None and math.isfinite(value)


def numeric_range(values):
    values = [float(x) for x in values if finite(x)]
    return dict(n=len(values), minimum=min(values), mean=mean(values), maximum=max(values)) if values else dict(n=0)


class Reader:
    def __init__(self, root):
        self.root = root.resolve()
        self.job = self.root / "runs/primary"
        self.checked = {}
        self.decoded = {}
        complete_path = self.job / "artifacts/COMPLETE.json"
        self.complete = json.loads(complete_path.read_text(encoding="utf-8"))
        self.inventory = self.complete["artifacts"]
        self.add(complete_path)

    def add(self, path, inventory=False):
        path = path.resolve()
        require(path.is_relative_to(self.root), "Read outside collected root")
        require(path.is_file() and not path.is_symlink(), "Missing or linked collected file")
        if str(path) not in self.checked:
            digest = sha256(path)
            size = path.stat().st_size
            if inventory:
                relative = path.relative_to(self.job).as_posix()
                expected = self.inventory[relative]
                require(digest == expected["sha256"] and size == expected["bytes"], "Artifact inventory mismatch: " + relative)
            self.checked[str(path)] = dict(bytes=size, sha256=digest)
        return path

    def json(self, relative):
        path = self.add(self.job / relative, inventory=True)
        return json.loads(path.read_text(encoding="utf-8"))

    def json_external(self, relative):
        path = self.add(self.root / relative)
        return json.loads(path.read_text(encoding="utf-8"))

    def arrays(self, relative, members):
        import numpy as np
        path = self.add(self.job / relative, inventory=True)
        require(relative.startswith(("evaluation/REPEATS_8_", "artifacts/PARAMETER_")), "Unexpected numerical artifact")
        with np.load(path, allow_pickle=False) as bundle:
            require(set(members) <= set(bundle.files), "Missing saved-array member")
            result = {key: bundle[key] for key in members}
        self.decoded[relative] = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in result.items()}
        return result

    def unchanged(self):
        for filename, record in self.checked.items():
            path = Path(filename)
            require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"], "Readback input changed")


def full_population(reader, tolerances):
    original = reader.json("artifacts/FULL_POPULATION_COMPARISON.json")
    records = original["records"]
    require(len(records) == 12, "Expected twelve full-population records")
    expected = set(itertools.product((0, 1), ("A", "theta"), HYPOTHESES))
    require({(r["bank"], r["arm"], r["hypothesis"]) for r in records} == expected, "Full-population identity mismatch")
    require(all(r["metrics"]["fake_rows"] == 26372 and r["metrics"]["reference_rows"] == 21047 for r in records), "Population denominator changed")
    compact = [dict(bank=r["bank"], arm=r["arm"], hypothesis=r["hypothesis"], values=r["metrics"]["values"],
                    coverage=r["metrics"]["coverage"], wave_components=r["metrics"]["wave_components"],
                    auxiliary_legacy_objective=r["metrics"]["legacy_objective"]) for r in records]
    aggregates = {}
    for h in HYPOTHESES:
        by_arm = {}
        for arm in ("A", "theta"):
            rows = [r for r in records if r["hypothesis"] == h and r["arm"] == arm]
            by_arm[arm] = {key: mean([r["metrics"]["values"][key] for r in rows]) if all(finite(r["metrics"]["values"][key]) for r in rows) else None
                           for key in rows[0]["metrics"]["values"]}
        deltas = {key: by_arm["theta"][key] - by_arm["A"][key] if finite(by_arm["A"][key]) and finite(by_arm["theta"][key]) else None for key in by_arm["A"]}
        improvement = 1 - by_arm["theta"]["wave"] / by_arm["A"]["wave"]
        tests = {}
        for key, value in by_arm["theta"].items():
            threshold = tolerances[h][key]
            calibrated = finite(threshold) and threshold > 0
            tests[key] = dict(T_readback=threshold, absolute_distance_within_T=(value <= threshold) if calibrated and finite(value) else None,
                              theta_minus_A_within_T=(deltas[key] <= threshold) if calibrated and finite(deltas[key]) else None,
                              interpretation="defined application-assumption scale; uncalibrated is not PASS")
        aggregates[h] = dict(mean_of_two_bank_distances=by_arm, theta_minus_A=deltas, wave_relative_improvement=improvement,
                             wave_useful_threshold=0.02, wave_useful_threshold_met=improvement >= .02, component_scale_checks=tests)
    return dict(records=compact, per_hypothesis=aggregates,
                aggregation="arithmetic mean of two separately evaluated empirical distances, NOT pooled-eight-draw empirical distance",
                population_per_bank=dict(SOURCE_TRAIN_parents=6593, draws=4, fake_rows=26372, REAL_TRAIN_rows=21047),
                three_hypotheses_are_independent_model_fits=False)


def four_suites(reader):
    result = []
    for bank, arm, h in itertools.product((0, 1), ("A", "theta"), HYPOTHESES):
        name = f"evaluation/FULL_{bank}_{arm}_{h}.json"
        report = reader.json(name)
        population = report["population"]
        bins = report["output_Q_conditions"]
        trends = report["source_input_trends"]
        require(population["all_population_rows_retained"] and len(bins["bins"]) == 8, "Rows or output-Q bins lost")
        require(trends["parent_count"] == 6593, "Wrong SOURCE trend population")
        conditional = []
        for b in bins["bins"]:
            conditional.append(dict(bin=b["bin"], fake_rows=b["fake_rows"], real_rows=b["real_rows"], supported=b["supported"],
                                    defined_PSD_W1=b["defined_PSD_W1"],
                                    means={domain: {key: b[domain][key]["mean"] for key in ("Q_Vns", "tail_Vns", "PSD_raw", "peak_voltage_V", "rise10_90_ns", "FWHM_ns")} for domain in ("fake", "real")}))
        corr = trends["correlations"]
        strongest = {}
        for mode in ("response_mean", "response_variance"):
            strongest[mode] = {}
            for key in corr["canonical_Q_Vns"][mode]:
                candidates = [(n, c[mode][key]) for n, c in corr.items() if n.startswith("shape_PC_") and finite(c[mode][key]["value"])]
                n, value = max(candidates, key=lambda item: abs(item[1]["value"]))
                strongest[mode][key] = dict(descriptor=n, **value)
        repeat = report["repeated_parent"]
        result.append(dict(bank=bank, arm=arm, hypothesis=h, report=name,
                           global_population=dict(fake_rows=population["fake_rows"], real_rows=population["real_rows"],
                             fake_flags=population["fake_flags"], real_flags=population["real_flags"], real_unknown_flags=population["real_unknown_flags"],
                             REAL_group_count=population["REAL_group_count"], group_is_independent_acquisition=population["group_is_independent_acquisition"]),
                           output_Q_conditions=dict(bins=conditional, minimum_support=bins["minimum_support"],
                             all_rows_in_exactly_one_bin=bins["all_rows_in_exactly_one_bin"], Q_is_physical_energy_truth=False),
                           SOURCE_input_trends=dict(parents=trends["parent_count"], canonical_Q=corr["canonical_Q_Vns"], amplitude_384=corr["amplitude_384_V"],
                             strongest_absolute_shape_PC_associations=strongest, physical_source_weights=trends["physical_source_weights"],
                             conditional_REAL_truth=trends["unpaired_REAL_conditional_target"], interpretation=trends["inference"]),
                           same_parent_repeat_contract=dict(draws=repeat["draws_per_parent"], statistical_unit=repeat["statistical_unit"],
                             covariance_denominator=repeat["covariance_denominator"], covariance_rank_upper_bound=repeat["covariance_rank_upper_bound"])))
    return result


def calibration(reader):
    report = reader.json("artifacts/CALIBRATION.json")
    stored = reader.json("artifacts/CALIBRATED_TOLERANCES.json")
    records = report["records"]
    require(len(records) == 126, "Expected 21 cases x two banks x three hypotheses")
    require(len({(r["case_index"], r["bank"], r["hypothesis"]) for r in records}) == 126, "Duplicate calibration cell")
    recomputed, winners, max_error = {}, {}, 0.0
    for h in HYPOTHESES:
        relevant = [r for r in records if r["hypothesis"] == h and r["tolerance_case"]]
        indices = sorted({r["case_index"] for r in relevant})
        require(len(relevant) == 14 and len(indices) == 7, "Wrong tolerance-case population")
        recomputed[h], winners[h] = {}, {}
        for key in relevant[0]["self_distance"]["values"]:
            candidates, quality = [], True
            for index in indices:
                pair = sorted([r for r in relevant if r["case_index"] == index], key=lambda r: r["bank"])
                require([r["bank"] for r in pair] == [0, 1], "Calibration bank mismatch")
                values = [r["self_distance"]["values"][key] for r in pair]
                if not all(finite(v) for v in values):
                    quality = False
                    continue
                if key == "QPSD" and not all(r["self_distance"]["coverage"]["conditional_coverage_complete"] for r in pair):
                    quality = False
                candidates.append((mean(values), index, pair[0]["case"], values))
            best = max(candidates) if quality and len(candidates) == 7 else None
            value = best[0] if best and best[0] > 0 else None
            recomputed[h][key] = value
            expected = stored["T_by_hypothesis"][h][key]
            require((expected is None) == (value is None), "Tolerance definedness differs")
            if expected is not None:
                max_error = max(max_error, abs(expected - value))
                require(abs(expected - value) <= 1e-14 * max(1., abs(expected)), "Tolerance mapping differs")
                winners[h][key] = dict(case_index=best[1], case=best[2], two_bank_self_distances=best[3])
    compact = [dict(case_index=r["case_index"], case=r["case"], bank=r["bank"], hypothesis=r["hypothesis"], tolerance_case=r["tolerance_case"],
                    self_distance=r["self_distance"]["values"], distance_to_REAL=r["distance_to_REAL"]["values"], signed_REAL_changes=r["signed_REAL_changes"],
                    changed_crop_rows=r["changed_crop_rows"], guard_counts=r["guard_counts"],
                    all_conditional_bins_supported=r["self_distance"]["coverage"]["conditional_coverage_complete"]) for r in records]
    return dict(producer_T_readback_only=stored["T_by_hypothesis"], status=stored["status"], recomputed_T=recomputed, winning_cases=winners,
                mapping_max_abs_error=max_error, records=compact, formula=stored["rule"], application_assumptions=report["application_assumptions"],
                hardware_calibrated=False, RR_used_to_set_tolerance=False, ROOT_formal_T_written=False)


def parameter_validation(reader, tolerances):
    report = reader.json("artifacts/PARAMETER_VALIDATION.json")
    native = reader.json("artifacts/METHOD_SELECTION.json")
    rows = report["records"]
    directions = [d["name"] for d in report["directions"]]
    expected = set(itertools.product(directions, (.1, 1., 10.), (-1, 1), (0, 1), (0, 1), HYPOTHESES))
    observed = [(r["direction"], r["nominal_scale_LSB"], r["sign"], r["bank"], r["batch"], r["hypothesis"]) for r in rows]
    require(len(observed) == len(set(observed)) == 360 and set(observed) == expected, "Parameter cell cube mismatch")
    summaries, primary = [], {}
    delta_error = 0.
    for name, scale, sign in itertools.product(directions, (.1, 1., 10.), (-1, 1)):
        group = [r for r in rows if (r["direction"], r["nominal_scale_LSB"], r["sign"]) == (name, scale, sign)]
        cells = []
        for r in group:
            h, th = r["hypothesis"], tolerances[r["hypothesis"]]
            calibrated = all(finite(th[k]) and th[k] > 0 for k in KEYS)
            ratios = {k: r["S_H_self_distance"][k] / th[k] if finite(th[k]) and th[k] > 0 and finite(r["S_H_self_distance"][k]) else None for k in KEYS}
            material = any(finite(ratios[k]) and ratios[k] > 1 for k in ("wave", "QPSD"))
            adequate = calibrated and r["S_delta_objective"] < -EPS and r["H_delta_objective"] < -EPS and all(r["hard_component_delta"][k] <= th[k] for k in KEYS)
            repair = calibrated and r["S_delta_objective"] < -EPS and r["H_delta_objective"] > EPS and material
            for path, delta in (("S", r["S_delta_objective"]), ("H_moving", r["H_delta_objective"])):
                recompute = r["paths"][path]["legacy_objective"] - r["reference"]["legacy_objective"]
                delta_error = max(delta_error, abs(delta - recompute))
            for key in KEYS:
                delta_error = max(delta_error, abs(r["hard_component_delta"][key] - (r["paths"]["H_moving"]["values"][key] - r["reference"]["values"][key])))
            cell = dict(bank=r["bank"], batch=r["batch"], hypothesis=h, S_delta_objective=r["S_delta_objective"], H_moving_delta_objective=r["H_delta_objective"],
                        H_fixed_delta_objective=r["paths"]["H_fixed"]["legacy_objective"]-r["reference"]["legacy_objective"],
                        hard_component_delta=r["hard_component_delta"], S_H_self_distance=r["S_H_self_distance"], S_H_over_T=ratios,
                        changed_crop_rows=r["changed_crop_rows"], output_Q_bin_migration=r["output_Q_bin_migration"],
                        actual_carrier_delta_RMS_V=r["actual_carrier_delta_RMS_V"], actual_carrier_delta_absmax_V=r["actual_carrier_delta_absmax_V"],
                        critical_T_defined=calibrated, adequate_cell=adequate, repair_cell=repair,
                        guard_counts=r["guard_counts"] if name in PRIMARY and scale == 1. and sign == 1 else None)
            cells.append(cell)
        summary = dict(direction=name, nominal_scale_LSB=scale, sign=sign, cells=12,
                       S_descent_cells=sum(c["S_delta_objective"] < -EPS for c in cells), H_descent_cells=sum(c["H_moving_delta_objective"] < -EPS for c in cells),
                       proxy_down_hard_up_cells=sum(c["S_delta_objective"] < -EPS and c["H_moving_delta_objective"] > EPS for c in cells),
                       materially_proxy_down_hard_up_cells=sum(c["repair_cell"] for c in cells), adequate_cells=sum(c["adequate_cell"] for c in cells),
                       S_delta_objective=numeric_range([c["S_delta_objective"] for c in cells]), H_moving_delta_objective=numeric_range([c["H_moving_delta_objective"] for c in cells]),
                       S_H_over_T_max={key:max(c["S_H_over_T"][key] for c in cells) for key in KEYS},
                       changed_crop_rows_summed_over_12_distinct_hypothesis_cells=sum(c["changed_crop_rows"] for c in cells),
                       crop_denominator_12_cells=12*2048, cells_detail=cells)
        summaries.append(summary)
        if name in PRIMARY and scale == 1. and sign == 1:
            stored = native["direction_decisions"][name]
            require(stored["adequate_all12"] == (summary["adequate_cells"] == 12), "Adequate direction differs")
            require(stored["repair_all12"] == (summary["materially_proxy_down_hard_up_cells"] == 12), "Repair direction differs")
            primary[name] = summary
    good = [name for name in PRIMARY if primary[name]["adequate_cells"] == 12]
    unsafe = [name for name in PRIMARY if primary[name]["materially_proxy_down_hard_up_cells"] == 12]
    choice = "adequate" if good else "repair_needed" if unsafe else "inconclusive"
    require(choice == native["choice"] and good == native["adequate_directions"] and unsafe == native["unsafe_directions"], "Method machine-rule mismatch")
    require(delta_error <= 1e-14, "Saved objective/delta inconsistency")
    return dict(method_choice=choice, adequate_directions=good, unsafe_primary_directions=unsafe, primary_12_cell_results=primary,
                all_30_registered_state_summaries=summaries, directions=report["directions"], saved_delta_consistency_max_error=delta_error,
                new_AD_queries=0, executed_optimizer_steps=0, primary_scope="positive nominal1LSB; same direction must satisfy all2bank x2batch x3hyp cells",
                descriptive_states_are_not_posthoc_primary_choices=True, scientific_response_acceptance=False)


def selected_array_statistics(reader):
    import numpy as np
    repeated = []
    for arm, h in itertools.product(("A", "theta"), HYPOTHESES):
        relative = f"evaluation/REPEATS_8_{arm}_{h}"
        report = reader.json(relative + ".json")
        members = ["root/parent_rows", "root/bank_grid", "root/draw_grid", "root/scalars/sample_variance", "root/scalars/MonteCarlo_SE",
                   "root/scalars/defined_draws", "root/scalars/undefined_draws", "root/hard_wave_covariance_diagonal_V2"]
        if h == "fast":
            members.append("root/preADC_signal_covariance_diagonal_V2")
        arrays = reader.arrays(relative + ".npz", members)
        require(np.array_equal(arrays["root/parent_rows"], np.arange(6593)), "Repeat parent identity changed")
        require(np.array_equal(arrays["root/bank_grid"], np.tile(np.repeat([0, 1], 4), (6593, 1))), "Bank identity changed")
        require(np.array_equal(arrays["root/draw_grid"], np.tile(np.tile(np.arange(4), 2), (6593, 1))), "Draw identity changed")
        variance = arrays["root/hard_wave_covariance_diagonal_V2"]
        require(variance.shape == (6593, 256) and np.isfinite(variance).all() and (variance >= 0).all(), "Invalid saved hard-wave variance")
        hard_std = float(np.sqrt(variance.mean()))
        item = dict(arm=arm, hypothesis=h, parents=6593, repeats=8, sample_covariance_denominator=7, rank_upper_bound=7,
                    hard_wave_sqrt_mean_sample_variance_V=hard_std, hard_wave_RMS_conditional_mean_MCSE_V=hard_std/math.sqrt(8),
                    scalar_response_spread={}, statistical_unit="SOURCE parent", no_population_sampling_CI_computed=True)
        v, n = arrays["root/scalars/sample_variance"], arrays["root/scalars/defined_draws"]
        u, se = arrays["root/scalars/undefined_draws"], arrays["root/scalars/MonteCarlo_SE"]
        require(np.array_equal(n+u, np.full_like(n, 8)), "Defined/undefined draw accounting differs")
        se_error = 0.
        for column, name in enumerate(report["scalar_names"]):
            defined = n[:, column] >= 2
            require(np.isfinite(v[defined, column]).all() and (v[defined, column] >= 0).all(), "Invalid scalar sample variance")
            standard_deviations = np.sqrt(v[defined, column])
            expected = standard_deviations / np.sqrt(n[defined, column])
            se_error = max(se_error, float(np.max(np.abs(expected - se[defined, column]))))
            item["scalar_response_spread"][name] = dict(parents_with_2_or_more_defined_draws=int(defined.sum()), undefined_draws=int(u[:, column].sum()),
                                                       sqrt_mean_sample_variance=float(np.sqrt(np.mean(v[defined, column]))), median_parent_sample_STD=float(np.median(standard_deviations)),
                                                       p95_parent_sample_STD=float(np.quantile(standard_deviations, .95)), parents_with_any_undefined=int((u[:,column]>0).sum()))
        item["saved_MCSE_formula_max_abs_error"] = se_error
        require(se_error <= 1e-15, "Scalar MCSE formula differs")
        if h == "fast":
            pre = arrays["root/preADC_signal_covariance_diagonal_V2"]
            require(pre.shape == (6593,384) and np.isfinite(pre).all() and (pre >= 0).all(), "Invalid preADC variance")
            item["preADC_sqrt_mean_sample_variance_V"] = float(np.sqrt(pre.mean()))
            item["preADC_RMS_conditional_mean_MCSE_V"] = item["preADC_sqrt_mean_sample_variance_V"] / math.sqrt(8)
        repeated.append(item)
        del arrays, variance, v, n, u, se
    parameter = []
    for direction, bank in itertools.product(PRIMARY, (0, 1)):
        relative = f"artifacts/PARAMETER_{direction}_1_+1_bank{bank}.npz"
        arrays = reader.arrays(relative, ["carrier_delta_RMS_V", "carrier_delta_absmax_V", "baseline_replay_absmax_V", "source_parent_row", "draw_index"])
        rms, maximum = arrays["carrier_delta_RMS_V"], arrays["carrier_delta_absmax_V"]
        parents, draws = arrays["source_parent_row"], arrays["draw_index"]
        require(rms.shape == (4096,) and np.unique(parents).size == 1024, "Parameter subset shape differs")
        require(np.array_equal(draws, np.tile(np.arange(4),1024)) and np.all(parents.reshape(1024,4) == parents.reshape(1024,4)[:,:1]), "Parameter parent/draw identity differs")
        require(np.isfinite(rms).all() and (rms>=0).all() and np.isfinite(maximum).all(), "Invalid carrier residual")
        parameter.append(dict(direction=direction, bank=bank, rows=4096, parents=1024,
                              actual_global_RMS_V=float(np.sqrt(np.mean(rms*rms))), actual_global_RMS_LSB=float(np.sqrt(np.mean(rms*rms))/LSB),
                              actual_absmax_V=float(maximum.max()), actual_absmax_LSB=float(maximum.max()/LSB),
                              parent_draw_RMS_median_V=float(np.median(rms)), parent_draw_RMS_p95_V=float(np.quantile(rms,.95)),
                              baseline_replay_absmax_V=float(arrays["baseline_replay_absmax_V"].max()),
                              SOURCE_parent_rows_sha256=hashlib.sha256(parents.astype('<i8',copy=False).tobytes()).hexdigest()))
        del arrays, rms, maximum, parents, draws
    return dict(same_parent_eight_repeat_statistics=repeated, primary_actual_parameter_carrier_residuals=parameter,
                interpretation="conditional Monte Carlo spread from saved fixed banks, not detector conditional-truth validation or training-seed uncertainty")


def write_summary(path, report):
    full, method, tolerance = report["full_population"], report["parameter_validation"], report["calibration"]
    rr = report["REAL_REAL_control"]["metrics"]["values"]
    lines = ["# WP1/013 保存结果数值读回", "", "本回执只读取已收集 JSON 与指定 TRAIN 统计数组，不加载 checkpoint，不执行模型、AD、优化或新随机库。θ0 为原 010 的 G2000 头。", "",
             "原生子进程/监督均为 0；完整 6593 SOURCE 父事件、21047 REAL 行已执行。方法选择可用与响应匹配验收分别记录。", "",
             "| 假设 | A wave | θ0 wave | wave 改善 | A QPSD | θ0 QPSD | noise（A=θ0） |", "|---|---:|---:|---:|---:|---:|---:|"]
    for h in HYPOTHESES:
        d=full["per_hypothesis"][h]; a,t=d["mean_of_two_bank_distances"]["A"],d["mean_of_two_bank_distances"]["theta"]
        lines.append(f"| {h} | {a['wave']:.9f} | {t['wave']:.9f} | {100*d['wave_relative_improvement']:.3f}% | {a['QPSD']:.9f} | {t['QPSD']:.9f} | {t['noise']:.9f} |")
    lines += ["", "表中为两套随机库各自经验距离的算术均值，不是八次响应合并后的经验距离。三种假设均未达到相对 A 的 wave ≥2% 门槛。", "",
              f"RR 辅助对照 wave={rr['wave']:.9f}、QPSD={rr['QPSD']:.9f}、noise={rr['noise']:.9f}。两侧各 10475 行/210 组，来源仍为同一次采集；RR 不用作不可约下限或容忍度。", "",
              "| 假设 | T_wave | T_QPSD | T_noise | peak 距离/T_peak | rise 距离/T_rise | FWHM 距离/T_FWHM |", "|---|---:|---:|---:|---:|---:|---:|"]
    for h in HYPOTHESES:
        t=tolerance["producer_T_readback_only"][h]; v=full["per_hypothesis"][h]["mean_of_two_bank_distances"]["theta"]
        lines.append(f"| {h} | {t['wave']:.9f} | {t['QPSD']:.9f} | {t['noise']:.9f} | {v['peak_voltage_V']/t['peak_voltage_V']:.3f} | {v['rise10_90_ns']/t['rise10_90_ns']:.3f} | {v['FWHM_ns']/t['FWHM_ns']:.3f} |")
    lines += ["", "T 为已注册人工 gain±3%、tail±3%、noise±10%、额外 σ=1ns 七案例的两库 self-distance 均值最大值；独立重建与保存值一致。它是应用假设的标尺，尚未由真实硬件误差校准。本目录只读回数值，不写根目录正式 T 表。零灵敏度或未定义的分量仍记 not_calibrated。", "",
              f"机器选择独立重建为 **{method['method_choice']}**。", "", "| 正向 1LSB 主方向 | S 下降格 | 硬动态窗下降格 | adequate 格 | repair 格 |", "|---|---:|---:|---:|---:|"]
    for name,s in method["primary_12_cell_results"].items():
        lines.append(f"| {name} | {s['S_descent_cells']}/12 | {s['H_descent_cells']}/12 | {s['adequate_cells']}/12 | {s['materially_proxy_down_hard_up_cells']}/12 |")
    lines += ["", "negative_saved012_total 在全部 12 格满足有限步规则；prospective AdamW 是旧保存 moments 加旧 012 固定库梯度的纯数组方向，不是实际执行的新训练步。其余尺度、反方向与随机方向全部保留为描述性结果，不作事后主方向选择。", "",
              "四套输出均齐全：全人口保留全部行；8 个输出 Q 箱分报 PSD；SOURCE 趋势按原父事件身份记录；八次同父响应保存方差、协方差重建口径与 MCSE。输入趋势只说明生成器关联，无配对 REAL 条件真值。", "",
              "当前头的噪声分布与 A 相同，形态峰值/上升沿/半高宽距离仍超过人工 T；绝对响应匹配未建立。当前 finite1LSB 方法判据支持继续已登记的受控 WP2 损失表示比较，不要求新增裁窗代理；该结论不保证所有后续训练步、所有尺度或跨采集可复用。", "",
              "详尽数值、每格判据、每个分量的 T 与案例来源、保存数组成员及 SHA 见 NATIVE_NUMERIC_READBACK.json 和 READBACK_RECEIPT.json。", ""]
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--collected",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    require(args.out.resolve().drive.upper() in ("E:","D:"), "Output must remain on E/D")
    args.out.mkdir(parents=True,exist_ok=True)
    require(not (args.out/"NATIVE_NUMERIC_READBACK.json").exists(),"No readback overwrite")
    temporary=args.out/"tmp";temporary.mkdir(exist_ok=True)
    os.environ["TEMP"]=str(temporary.resolve());os.environ["TMP"]=str(temporary.resolve())
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[key]="4"
    reader=Reader(args.collected)
    complete=reader.complete
    require(complete["phase"]=="complete" and complete["engineering_checks_passed"] and complete["inputs_unchanged"] and complete["frozen_state_unchanged"] and not complete["inflight"] and not complete["error"],"Producer completion invalid")
    require(complete["counters"]==complete["attempted_counters"],"In-flight workload remains")
    require(all(complete["counters"][k]==0 for k in ("optimizer_created","optimizer_steps","G_updates","D_updates","noise_updates","population_updates","C2ST_updates","new_AD_queries","DEV_model_rows")),"Unexpected science/training role")
    native_path=reader.add(reader.job/"supervisor/exit_receipt.json")
    native=json.loads(native_path.read_text(encoding="utf-8"))
    require(native["child_native_exit_code"]==native["exit_code"]==0 and not native["reason"] and not native["unknown_inflight_ids"],"Nonzero or incomplete native run")
    require(not (reader.job/"artifacts/FAILED.json").exists(),"FAILED remains")
    configuration=reader.json_external("wp1_013_config.json")
    calib=calibration(reader)
    full=full_population(reader,calib["producer_T_readback_only"])
    suites=four_suites(reader)
    parameters=parameter_validation(reader,calib["producer_T_readback_only"])
    selected=selected_array_statistics(reader)
    real_control=reader.json("artifacts/REAL_REAL_CONTROL.json")
    repeat_residuals={name:reader.json("artifacts/"+name+".json") for name in ("A_NUMERIC_REPEATS_0","A_NUMERIC_REPEATS_1","A_NUMERIC_CROSS_BANK")}
    encoding=reader.json("artifacts/REFERENCE_ENCODING.json")
    report=dict(schema="n01/wp1-saved-numeric-readback/v1",UTC=dt.datetime.now(dt.timezone.utc).isoformat(),run_id=complete["run_id"],collected_root=str(reader.root),
                scope="saved JSON plus explicitly listed TRAIN statistics; no checkpoint/model/scientific new forward/AD/optimizer/DEV access",reference_head_G_updates=2000,
                native_execution=dict(child_exit=0,supervisor_exit=0,counters=complete["counters"],entered_equals_completed=True,peak_RSS_bytes=native["peak_RSS_bytes"],peak_GPU_bytes=native["peak_GPU_bytes"],peak_output_bytes=native["peak_output_bytes"]),
                full_population=full,four_suites=suites,REAL_REAL_control=real_control,calibration=calib,parameter_validation=parameters,selected_saved_array_statistics=selected,
                A_numerical_repeat_residuals=repeat_residuals,reference_storage=encoding,
                scientific_status="method adequate under registered finite-step scope; response_match_not_established",ROOT_formal_T_written=False,
                limitations=["All model/evaluation data derive from historically exposed single acquisition; no cross-acquisition validation",
                    "Two fixed random banks are not independent retrained model seeds","Same-parent variability has no paired REAL conditional target",
                    "Q is observed charge, not independently calibrated energy or particle label","Application T is derived from registered synthetic perturbations, not physical measurement calibration",
                    "No finite-step guarantee for all later iterates, boundaries or noise banks","No GAN causal control is created by comparing historical full19k"])
    reader.unchanged()
    output=args.out/"NATIVE_NUMERIC_READBACK.json"
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    write_summary(args.out/"NUMERIC_SUMMARY.md",report)
    receipt=dict(schema="n01/wp1-saved-readback-receipt/v1",UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status="saved_numeric_readback_complete",
                 source_script=dict(path=str(Path(__file__).resolve()),sha256=sha256(Path(__file__))),checked_files_sha256=reader.checked,decoded_members=reader.decoded,
                 threads_limit=4,new_science_counters=dict(checkpoint_loads=0,model_forward_rows=0,AD_queries=0,optimizer_created=0,optimizer_steps=0,training_updates=0,DEV_rows=0,new_random_banks=0,SSH_calls=0),
                 selected_input_hashes_unchanged=True,root_formal_T_written=False,
                 checks=dict(T_formula_max_abs_error=calib["mapping_max_abs_error"],method_machine_rule_matches=True,parameter_delta_max_abs_error=parameters["saved_delta_consistency_max_error"],FULL_records=12,calibration_records=126,parameter_cells=360,repeat_bundles=6),
                 outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in (output,args.out/"NUMERIC_SUMMARY.md")})
    with (args.out/"READBACK_RECEIPT.json").open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(status=receipt["status"],choice=parameters["method_choice"],T_max_abs_error=calib["mapping_max_abs_error"],
                          wave_improvement_percent={h:100*full["per_hypothesis"][h]["wave_relative_improvement"] for h in HYPOTHESES},
                          primary={n:{k:s[k] for k in ("S_descent_cells","H_descent_cells","adequate_cells","materially_proxy_down_hard_up_cells")} for n,s in parameters["primary_12_cell_results"].items()},
                          outputs=receipt["outputs"]),ensure_ascii=True))


if __name__=="__main__":
    main()
