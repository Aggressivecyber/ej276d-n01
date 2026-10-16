# 025 独立结果复核

结论：可按已记录的负结果收口，无审核阻断。原生退出 0 与科学非劣通过分别判断；固定 plus2/index5 在六个 bank/电子学格均失败。90 项正 T 检查为 73 pass、17 fail，42 项未校准检查仍为 pending。

| 失败指标 | 失败格数 | 关键数值 |
|---|---:|---|
| rise10_90_ns | 6 | Δ=0.738064–0.739388 软件 ns，Δ/T=3.488121–3.503203 |
| wave | 5 | 失败格 Δ/T=1.003728–1.039734；bank1/fast=0.999812，按冻结规则通过 |
| rise undefined / prepeak censored | 各 3 | 均在 bank1；两标签在本数据上对应同一组截断行 |

六格的 SWD、coordinate、short-lag 三个聚合分量全部增大，Δ 范围分别为 0.263892–0.277819、0.119229–0.125607、0.106273–0.111184。36 个单独 SWD view 中 34 增、2 降；不能说所有单视图都变差。024 的有限人口均值 RMS 改善保留，但它未传递为本候选的完整 wave/rise 分布改善。Q、front、peak 距离减小；QPSD、tail、PSD、FWHM 距离增加但在各自非劣容差内。相对通过不代表绝对响应匹配。

独审核对 50 个选收文件及归档 SHA、34 条 producer inventory、13,722 行 journal；6,861 个操作 entered=completed，无 failed/inflight/truncated。原生 wall 433.470 s、CPU 489.142 s、峰 RSS 1,331,776 KiB、输出 44,110,147 B，均在冻结上限内。105,720 列 W1 和 6,816 次 CDF 构造全部完成。support=5,009,148,020，比 cap 少 5,464，精确来自 rise/FWHM 有限子人口缺失；96 个条件分箱全支持，最小 fake/REAL 支持为 3189/2630，不能把合法 cap 余量写成漏算。

本次成功数值读回为 **16 个 unique NPZ、16 次文件打开、588 次成员加载，仅 1 次 invocation，无重复数值 decode**。检查了 337,511 行保存观察量、坐标/分箱、flags 与分母、原父/draw/权重轴，重组 6,816 blocks→408 groups→12 scores，并复核 132 项判定及 66 个分 bank 描述均值；最大绝对差 4.44×10⁻¹⁶，分类差异 0。审核器的路径/字段错误及单视图方向断言错误保留在 JSON；它们没有改变原生结果，前置字段错误发生在 NPZ 加载前。

复核没有重排 CDF，也没有从原始 256 点波形重新提取特征；W1 分块值与原始特征提取仍依赖冻结 producer。本轮新增 NN/CP/AD/G/RNG/ADC、参数点、拟合、原波形读取和 SSH 均为 0。prefix 恒等只支持 noise/baseline 相对 Δ=0；REAL prefix 未读，绝对噪声距离 unknown。候选来自已暴露 TRAIN 的均值选择，本轮是开发检查；M0 zero 也不是 deterministic A 或旧 2% 门。

关闭固定 plus2 非劣问题，不晋级该候选；保留 024 三个有限均值见证及所有旧负结果，不推断全族不可达。源人口/冻结响应敏感性属于另行前瞻登记的问题，本审不启动新扫描、训练或 REAL 验收。完整 pins、17 项失败数值与六格分量见 [RESULT_REVIEW.json](RESULT_REVIEW.json)。
