"""Render the Chinese fixed-endpoint review from recovered immutable results.

No training, model forward pass, data fitting, or checkpoint selection occurs here.
The diagnosis text is the reviewed interpretation of the saved endpoint evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.review_root.resolve()
    recovery = read_json(root / "RECOVERY_VERIFICATION.json")
    audit = read_json(root / "CHECKPOINT_AND_SPLIT_AUDIT.json")
    diagnostic = read_json(root / "FULL_MODEL_ELECTRONICS_AND_INPUT_DIAGNOSTIC.json")
    comparison = read_json(root / "train/ARM_COMPARISON.json")
    arms = ["full", "deterministic", "simple", "full_uniform"]
    reports = {a: read_json(root / f"train/{a}/evaluation/endpoint_report.json") for a in arms}
    assert recovery["terminal_status"]["unique_updates"] == 77000
    assert recovery["native_exit"]["exit_code"] == 0
    assert all(audit["arms"][a]["endpoint"] == 20000 for a in arms)
    assert all(audit["arms"][a]["model_state_all_finite"] for a in arms)
    full = reports["full"]
    actual = full["REAL_population"]
    rows = comparison["arm_metrics_side_by_side"]
    by_arm = {row["arm"]: row for row in rows}
    improvement = 100 * (1 - by_arm["full"]["raw_distances"]["QPSD"] / by_arm["deterministic"]["raw_distances"]["QPSD"])
    baseline_sd = full["raw_population"]["noise"]["std"][0] * 1000
    real_sd = actual["noise"]["std"][0] * 1000
    ratio = baseline_sd / real_sd
    interval = full["independent_domain_classifier"]["uncertainty"]["raw"]["percentile_95_interval"]
    stability = comparison["population_response_stability"]
    changes = stability["common_parent_response_changes"]
    names = {"full": "完整随机 U-Net", "deterministic": "确定性 U-Net＋相同电子学噪声", "simple": "简单随机响应", "full_uniform": "完整随机 U-Net，固定均匀比较权重"}

    def metrics_table(weighted=False):
        cells = []
        key = "comparison_weighted_distances" if weighted else "raw_distances"
        auc_key = "comparison_weighted_ROC_AUC" if weighted else "raw_ROC_AUC"
        ci_key = "comparison_weighted" if weighted else "raw"
        for arm in arms:
            row = by_arm[arm]
            dist = row[key]
            ci = row["classifier_uncertainty"][ci_key]["percentile_95_interval"]
            cells.append(f'<tr><td><b>{arm}</b><br><span class="detail">{names[arm]}</span></td>'
                         f'<td>{dist["wave"]:.6f}</td><td>{dist["QPSD"]:.6f}</td>'
                         f'<td>{dist["noise"]:.6f}</td><td>{row[auc_key]:.6f}<br>'
                         f'<span class="detail">[{ci[0]:.6f}, {ci[1]:.6f}]</span></td></tr>')
        return '<div class="table-scroll"><table><thead><tr><th>固定终点</th><th>wave 距离</th><th>QPSD 距离</th><th>noise 距离</th><th>域分类器 AUC［95% 区间］</th></tr></thead><tbody>' + ''.join(cells) + '</tbody></table></div>'

    def figure(number, filename, caption):
        return f'<figure id="figure-{number}"><a href="figures/{filename}.png"><img src="figures/{filename}.png" alt="{html.escape(caption)}"></a><figcaption>图 {number}｜{caption} <a href="figures/{filename}.pdf">PDF 矢量版</a></figcaption></figure>'

    repeated_rows = []
    for arm in arms:
        rpt = reports[arm]["repeated_parent_response"]
        repeated_rows.append(f'<tr><td>{arm}</td><td>{rpt["conditional_variance_Q"]["mean"]:.6f}</td><td>{rpt["conditional_variance_PSD"]["mean"]:.6f}</td><td>{rpt["conditional_covariance_Q_PSD"]["mean"]:.6f}</td></tr>')
    conditional_rows = []
    for item in audit["arms"]["full"]["conditional_peaks"]:
        b = item["bin"]
        flag = "双组件拟合可描述；无真实粒子标签" if item["full_converged"] and item["full_BIC_one_minus_two"] > 0 and item["REAL_BIC_one_minus_two"] > 0 else "双组件证据不足；full EM 未收敛"
        conditional_rows.append(f'<tr><td>{b}</td><td>{item["fake_count"]} / {item["real_count"]}</td><td>{item["full_width"][0]:.5f} / {item["REAL_width"][0]:.5f}</td><td>{item["full_width"][1]:.5f} / {item["REAL_width"][1]:.5f}</td><td>{flag}</td></tr>')
    hashes = ''.join(f'<tr><td>{arm}</td><td class="hash">{reports[arm]["checkpoint_sha256"]}</td></tr>' for arm in arms)
    generated_time = datetime.now(timezone.utc).isoformat()
    summary = {
        "experiment_id": "N01", "run_id": "run_001_20261012", "review_UTC": generated_time,
        "execution_status": "completed_native_exit_0", "scientific_status": "response_match_not_established",
        "unique_training_updates": 77000, "review_added_training_updates": 0,
        "review_generated_new_responses": False, "checkpoint_selection": False,
        "baseline_standalone_evaluation_performed": False,
        "QPSD_distance_relative_improvement_full_vs_deterministic_percent": improvement,
        "full_AUC": full["independent_domain_classifier"]["raw_ROC_AUC"],
        "full_AUC_95_interval": interval,
        "baseline_mean_std_ratio_full_over_REAL_ddof0": ratio,
        "raw_and_weighted_arm_metrics": rows,
        "population_sensitivity_scope": "same_seed_retraining_under_learned_or_uniform_comparison_weights",
        "frozen_response_cross_population_transfer_validated": False,
        "independent_acquisition_count": 1, "physical_weights": "unknown", "absolute_calibration": "unknown",
        "next_priority": "joint_electronics_noise_decomposition_before_any_new_full_training",
        "next_training_authorized_or_started_by_review": False,
        "inputs": {name: digest(root / name) for name in ["RECOVERY_VERIFICATION.json", "CHECKPOINT_AND_SPLIT_AUDIT.json", "FULL_MODEL_ELECTRONICS_AND_INPUT_DIAGNOSTIC.json", "INDEPENDENT_REVIEW.md", "train/ARM_COMPARISON.json"]},
    }
    (root / "REVIEW_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 完整模型固定终点评估</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:#192d40;font:16px/1.75 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}main{{max-width:1160px;margin:auto;background:white;padding:36px 48px 60px}}h1{{font-size:31px;line-height:1.35;margin:0 0 16px}}h2{{font-size:23px;border-bottom:2px solid #d9e5f0;padding-bottom:8px;margin-top:36px}}h3{{font-size:18px;margin:24px 0 10px}}p{{margin:12px 0}}.eyebrow{{color:#35668c;letter-spacing:.08em;font-size:14px;font-weight:bold}}.lead{{font-size:19px}}.verdict{{background:#eef5fb;border-left:5px solid #2c6594;padding:16px 22px;margin:20px 0}}.residual{{background:#fff4e7;border-left:5px solid #b97828;padding:16px 22px;margin:20px 0}}.detail,figcaption{{font-size:14px;color:#40566b}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}.card{{border:1px solid #d6e1eb;border-radius:8px;padding:15px}}.card strong{{display:block;font-size:24px;color:#174a72}}a{{color:#195b8d;text-decoration:none}}a:hover{{text-decoration:underline}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin:14px 0}}th,td{{text-align:left;border-bottom:1px solid #dce4eb;padding:11px 12px;vertical-align:top}}th{{background:#edf3f8}}td.hash{{font:12px/1.6 Consolas,monospace;word-break:break-all}}.table-scroll{{overflow-x:auto}}figure{{margin:28px 0}}img{{display:block;width:100%;height:auto;border:1px solid #e0e6ec}}figcaption{{margin-top:8px}}code{{font-family:Consolas,monospace;background:#f2f5f7;padding:1px 4px}}li{{margin:5px 0}}.footer{{margin-top:30px;border-top:1px solid #d6e1eb;padding-top:16px}}@media(max-width:720px){{main{{padding:22px 16px}}h1{{font-size:27px}}.cards{{grid-template-columns:1fr}}th,td{{padding:8px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:0}}figure,table,.cards{{break-inside:avoid}}h2{{break-after:avoid}}}}
</style></head><body><main>
<div class="eyebrow">N01 · RUN 001 · 固定终点 · 2026-10-12</div>
<h1>完整模型训练完成；完整响应匹配仍有明确残差</h1>
<p class="lead">四组模型在同一固定计划下完成训练和自动终点评估。完整随机 U-Net 改善了部分 Q–PSD 统计，但波形可区分性、电子学噪声联合结构和部分条件峰宽仍不匹配。</p>
<div class="cards"><div class="card">实际独立优化更新<strong>77,000</strong>共享基线 1,000＋4×19,000</div><div class="card">完整模型域 AUC<strong>{full["independent_domain_classifier"]["raw_ROC_AUC"]:.4f}</strong>95% 区间 {interval[0]:.4f}–{interval[1]:.4f}</div><div class="card">基线均值的跨事件 SD<strong>{ratio:.2f} 倍</strong>生成 {baseline_sd:.3f} / REAL {real_sd:.3f} mV</div></div>
<div class="verdict"><b>可确认：</b>固定预算训练、模型落盘、自动终点评估和本次复核均已完成。<b>科学结论：</b>当前证据支持部分分布统计拟合与输入相关的随机响应表达；尚不能认定 sim2real 响应已验证、物理随机性已识别或可迁移到独立采集。</div>
<p>训练从 2026-10-12 00:51:53 至 07:00:28（Asia/Singapore），约 6 小时 9 分钟，监督进程退出码 0。所有臂达到名义 20,000 步；每臂均覆盖全部 6,593 个 SOURCE 和 21,047 个 REAL TRAIN 记录。本次按固定末点复核，未重新训练或挑选检查点。</p>

<h2>四组对照：随机分支有局部收益，尚无整体优势</h2>
<p>wave 包含完整电压、归一化波形及一、二阶差分的固定投影分布距离；QPSD 包含联合 Q–PSD 与按输出电荷分箱的 PSD 距离。三种距离均使用冻结的 TRAIN 尺度，数值越小表示该指标更接近。AUC 越接近 0.5 越难区分两域；这里没有追加历史通过线。</p>
<h3>原始计数总体</h3>{metrics_table()}
<h3>单独的比较权重</h3>{metrics_table(True)}
<p>full 相对 deterministic 的原始 QPSD 距离减少 <b>{improvement:.1f}%</b>，wave 距离略大，AUC 也更高。full_uniform 的 QPSD 距离更小。该比较不支持“增加随机分支后各项均改善”；simple 在本次结构、预算和初始化下整体较差，也不能外推为所有简单模型无效。</p>
<p>全 1 权重代表 SOURCE 记录的均匀计数参考，<b>真实物理事件权重仍未知</b>。学习到的比较权重只用于条件总体比较，不能当作绝对源谱校准。</p>
{figure(4, "04_distance_and_auc", "原始与比较加权距离、独立域分类器 AUC。REAL–REAL 对照样本量为 349 / 250，生成–REAL 为 2048 / 599，不能直接用前者设定通过阈值。")}

<h2>全波形和电子学噪声：RMS 接近掩盖了联合结构缺口</h2>
<p>full 的平均 Q 为 {full["raw_population"]["scalars"]["Q"]["mean"]:.4f} V·ns，REAL 为 {actual["scalars"]["Q"]["mean"]:.4f} V·ns；平均 PSD 为 {full["raw_population"]["scalars"]["PSD"]["mean"]:.5f} 与 {actual["scalars"]["PSD"]["mean"]:.5f}。峰时间均值偏移约 +1.19 ns，半高上升时间均值偏移约 +0.17 ns。总体均值接近仍不足以证明完整波形分布一致。</p>
{figure(1, "01_overall_endpoint", "完整模型与 REAL 的固定终点比较；均值曲线只能反映总体形状，不能替代逐事件相关结构。")}
<div class="residual"><b>最明确的残差：</b>平均基线 RMS 为 {1000*full["raw_population"]["scalars"]["baseline_rms_V"]["mean"]:.3f} 与 {1000*actual["scalars"]["baseline_rms_V"]["mean"]:.3f} mV，相差约 1.2%；基线均值的跨事件 SD 却是 {baseline_sd:.3f} 与 {real_sd:.3f} mV，约 {ratio:.2f} 倍。REAL 的奇偶延迟 ACF 结构也没有被近白噪声的生成基线恢复。</div>
<p>只读诊断给出了具体解释：前 100 点在全窗时间坐标中的平均位置为 −0.7933。REAL 短窗线性拟合的截距与斜率相关系数约 {diagnostic["REAL"]["OLS_intercept_slope_correlation"]:.3f}，其协方差会显著抵消基线均值的方差。当前电子学分支独立抽取截距和斜率，所学边际标准差约 0.381 / 0.581 mV；仅这一独立漂移项就预测约 {diagnostic["independent_latent_offset_slope_contribution_to_baseline_mean_std_mV"]:.3f} mV 的基线均值 SD，与观测生成结果接近。</p>
<p>短窗 OLS 系数的离散度还包含估计噪声；直接用这些边际离散度初始化潜在漂移，再叠加随机噪声，会有重复计入的风险。该结论指向<b>噪声分解及联合分布建模不足</b>。data.py 与 model.py 的时间坐标一致，未发现此处的坐标倍率错误。电子学分支在共享阶段后冻结，因此四臂共同继承该残差；它可能为域分类提供信息，尚未证明是高 AUC 的唯一原因。</p>

<h2>条件 PSD：部分峰宽残差与分量歧义必须保留</h2>
<p>8 个电荷箱均满足本次最小样本支持。分箱使用生成输出 Q 与 REAL Q；箱号从 0 开始。full 在箱 0、1、7 的最佳双 Gaussian EM 未收敛，且双方 BIC 均不支持两组件优于单组件，不能把强制双峰拟合的参数当作已恢复的真实双峰宽度。</p>
<div class="table-scroll"><table><thead><tr><th>箱</th><th>生成 / REAL 响应数</th><th>低 PSD 组件 SD<br>full / REAL</th><th>高 PSD 组件 SD<br>full / REAL</th><th>解释范围</th></tr></thead><tbody>{''.join(conditional_rows)}</tbody></table></div>
<p>例如箱 4 低 PSD 组件宽度为 0.03417 / 0.02227，约 1.53 倍；箱 2 高 PSD 组件约 1.43 倍。两组件没有真实 neutron / gamma 标签，重复响应不增加独立父事件数，当前峰位/峰宽没有按原始父组给出置信区间。</p>
{figure(2, "02_conditional_psd_components", "按输出电荷比较 PSD 组件位置与原始宽度；空心标记保留不收敛及 BIC 证据不足的箱。图中箱号 1–8 对应表内 0–7。")}

<h2>重复响应、输入依赖与总体权重敏感性</h2>
<p>每臂对固定 128 个 SOURCE 父事件各生成 64 次。下表为每父样本方差/协方差的平均值（方差分母 63）；这些是模型内部随机性，尚无同一真实输入的重复 REAL 真值证明其物理大小正确。</p>
<table><thead><tr><th>模型</th><th>平均 Var(Q|x)，(V·ns)²</th><th>平均 Var(PSD|x)</th><th>平均 Cov(Q,PSD|x)，V·ns</th></tr></thead><tbody>{''.join(repeated_rows)}</tbody></table>
<p>同随机变量的输入干预显示输出确实响应形状和幅度变化。不过，0.8 倍幅度测试有 3 / 8 个 SOURCE Q 低于冻结 TRAIN 下界，1.2 倍有 1 / 8 高于上界，属于外推。在一个低端外推父事件的 32 次响应中出现 1 次观测 Q=−0.0039065 V·ns，对应 PSD 离群值；该值保留在记录中，不能解释为合理物理依赖，也不应误报为 TRAIN 范围内的失败。</p>
<p>full 的比较权重范围为 0.98793–1.02705，属于很弱的总体扰动。与相同种子、固定均匀比较权重重训的 full_uniform 相比，128 个共同父事件的 Qbar 平均差为 {changes["qbar"]["full_minus_full_uniform"]["mean"]:.6f} V·ns，但逐父差异会相互抵消。每父平均波形差的 RMS 平均为 {1000*changes["mean_voltage_difference_RMS_V"]["mean"]:.3f} mV，最大 {1000*changes["mean_voltage_difference_RMS_V"]["maximum"]:.3f} mV；平均 Q–PSD 条件协方差变化为 {changes["covariance_Q_PSD"]["full_minus_full_uniform"]["mean"]:.6f}，bootstrap 区间约 [0.00120, 0.00336]。</p>
<p>该实验检验的是<b>改变比较总体假设后重新训练的敏感性</b>。它没有完成原方案所述“冻结同一响应，再改变 SOURCE 能谱 / 组成”的独立迁移验证。当前一个训练种子、一份真实采集及窄幅权重变化，均不足以证明响应核与事件总体可分别识别。</p>
{figure(3, "03_population_sensitivity", "128×64 重复响应与 full / full_uniform 比较；平均差小不等于每个父事件稳定。")}

<h2>数值有效性与证据边界</h2>
<ul><li>正式边际评估使用 512 个 SOURCE 父事件×4 响应及 599 个 REAL 原始事件。完整模型这 2048 条响应均有限、Q 为正，无近零分母保护行，未发生 ADC 饱和。pre-ADC 电荷恒等式最大残差 {full["charge_identity_max_abs_Vns"]:.3g} V·ns。</li>
<li>更大的 128×64 重复集合中 full 有 2 / 8192 个事件出现 pre-ADC 越界，45 / 8192 个 PSD 小于 0；这些没有静默删去。不能由边际样本零饱和推断所有生成响应均不饱和。</li>
<li>独立域分类器未复用训练判别器；SOURCE 父 ID 与 REAL 原始分组在 classifier fit/test 之间交叠为 0。测试仅含 205 个 SOURCE 父事件、250 个 REAL 记录 / 5 个原始块。AUC 区间反映当前冻结分类器下的抽样不确定性，没有包含重训种子或跨采集误差。</li>
<li>TRUE 粒子标签、绝对校准、实测物理事件权重均未知。原始记录组不等于独立采集。未追加 R79、旧 v5 或旧 A–F 硬阈值；低投影距离、近似均值和存在随机方差均不单独构成科学通过。</li>
<li>原始 endpoint_report 中 full 的 counterpart 状态是其生成时点的 pending；四臂完成后的总体比较以 <a href="train/ARM_COMPARISON.json">ARM_COMPARISON.json</a> 为准。旧报告原样保留，未改写历史回执。</li></ul>
<div class="verdict"><b>下一步建议：</b>优先审查电子学噪声的联合参数化、短窗估计误差与 FIR 约束，先用独立留出基线验证联合噪声特征，再决定完整模型的下一轮训练。当前材料支持这一具体方向，尚不支持单纯延长当前训练或扩大网络作为首选。本次未启动新训练。</div>

<h2>可复核交付</h2>
<p>回收包包含 68 个训练/终点评估文件，逐文件 SHA256 已重新核对；六个远端科学源文件与启动时 SHA 一致。四个末点的 stage、update、有限性和全部 77,000 行更新记录已重新打开检查。源文件、原始数据、启动回执和旧研究结果保持可追溯。</p>
<table><thead><tr><th>stage_20000.pt</th><th>SHA256</th></tr></thead><tbody>{hashes}</tbody></table>
<p><a href="RECOVERY_VERIFICATION.json">回收核验</a> · <a href="CHECKPOINT_AND_SPLIT_AUDIT.json">末点与划分审计</a> · <a href="FULL_MODEL_ELECTRONICS_AND_INPUT_DIAGNOSTIC.json">电子学与外推诊断</a> · <a href="INDEPENDENT_REVIEW.md">独立科学审核</a> · <a href="REVIEW_SUMMARY.json">机器可读总结</a></p>
<p>复现图表：<code>analysis/build_figures.py</code>。复现本报告：<code>scripts/build_completed_review.py --review-root &lt;本目录&gt;</code>。全部分析使用已保存固定终点数组，无新生成响应、无模型参数更新。</p>
<div class="footer detail">N01 独立遵循用户原方案。统计量依据已冻结的 run_001_20261012；图表与解释属于训练后复核，不用于重新选择终点。生成时间 UTC：{generated_time}。</div>
</main></body></html>'''
    destination = root / "N01_COMPLETE_MODEL_REVIEW.html"
    destination.write_text(output, encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(destination), "sha256": digest(destination), "summary": str(root / "REVIEW_SUMMARY.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
