# WP2/014 保存 JSON 开发选择

状态：`provisional_pending_independent_raw_readback`。开发路径选择 **C1**；不晋级权重，不宣告科学验收。

| 路径 | ALL6 非劣 | 12 个 final wave 均值 | ALL6 wave 相对 A ≥2% |
|---|---|---:|---|
| C0 | True | 0.15255321144257886 | False |
| C1 | True | 0.15042006897712382 | False |

| 路径 | seed | 假设 | A wave | final wave | 相对 A 改善 | 相对共同 initial 改善 | 绝对超 T 分量 |
|---|---:|---|---:|---:|---:|---:|---|
| C0 | 0 | fast | 0.153707296 | 0.152103660 | 1.043305172679% | 1.054541448300% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C0 | 0 | slow | 0.154674684 | 0.152934029 | 1.125365147174% | 1.112650007031% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C0 | 0 | common_level | 0.154605859 | 0.151987946 | 1.693281923767% | 1.661298014514% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C0 | 1 | fast | 0.153707296 | 0.152649597 | 0.688125888601% | 0.699597750634% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C0 | 1 | slow | 0.154674684 | 0.153329330 | 0.869796119859% | 0.879769026298% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C0 | 1 | common_level | 0.154605859 | 0.152314707 | 1.481930930574% | 1.472278723317% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 0 | fast | 0.153707296 | 0.150142574 | 2.319162882221% | 2.330254287513% | peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 0 | slow | 0.154674684 | 0.150895665 | 2.443204998363% | 2.430659330587% | QPSD, peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 0 | common_level | 0.154605859 | 0.149651004 | 3.204829589404% | 3.173337459394% | QPSD, peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 1 | fast | 0.153707296 | 0.150650398 | 1.988778858954% | 2.000100478011% | QPSD, peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 1 | slow | 0.154674684 | 0.151258463 | 2.208649063961% | 2.218487276284% | QPSD, peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |
| C1 | 1 | common_level | 0.154605859 | 0.149922309 | 3.029347994797% | 3.019847394147% | QPSD, peak_voltage_V, rise10_90_ns, FWHM_ns, baseline_mean_V, rise_undefined_rate, rise_prepeak_censored_rate |

C1 的 seed1-fast 改善精确为 0.019887788589535327，低于 0.02。不得因四舍五入接近 2% 判通过或追加补跑。

相对非劣使用同 seed/假设内两个 bank 的距离均值，再对全部 6 格取 ALL；不要求每个 bank 单独通过，也不把距离均值称为 8 draw 合并经验距离。零/未定义 T 分量继续 pending。

四 trial 均完成 2000 G；每 trial 实际 training backward=2000、probe AD=468、总 AD=2468。共享初始化/校准成本单列。总 G=8000、AD=9872、无在途；完整成本见 TRIAL_COST_READBACK.json。

梯度表为固定 64 父事件、2draw、2bank 条件子批经验损失的参数梯度，不能相加为全局事件贡献比例。日志只有 1、100…2000 共 21 个单批次点；与终点全人口评估分开。

选择性回收 142 文件、74,375,000 字节，包含薄头 checkpoint；0 个 NPZ。176 个原始文件约 3.077 GB 留在服务器。当前脚本未反序列化 checkpoint，未执行 NN/AD/优化、DEV 或 SSH。
