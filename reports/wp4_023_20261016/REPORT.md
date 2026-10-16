# N01 023：固定 T2 硬观察检出前置检验结案

023 原生退出 0，完成 slow/common_level CPU 距离计算并复用 022 的 fast 父分数。独立读回确认全部 36 个分组、权重、electronics 签名和分类。三项 all_heldout / nominal 主 O19 对比均未在预登记 δ=0.10 下建立检出资格，MC、null 与 support 均已分辨。因此不启动 T2 训练。

| electronics | O19 mean | 95% CI | T1 null abs-upper | T2 null abs-upper |
|---|---:|---|---:|---:|
| fast | −0.00222308 | [−0.01029384, 0.00584768] | 0.01071521 | 0.01037345 |
| slow | −0.00319211 | [−0.01122099, 0.00483676] | 0.01191635 | 0.01071562 |
| common_level | −0.00356825 | [−0.01156471, 0.00442820] | 0.01268786 | 0.01064425 |

每项主格使用全部 1319 个原 heldout 父。T2b0–T1b1、T2b1–T1b0 先按父平均，再作原比较权重及 parent-cluster SE；两个方向不作为独立重复。36 格均为 not_established_at_registered_tolerance，最大 half-width 0.02031403、最大 T1/T2 null abs-upper 0.02511158，support 权重质量均为 1。负 U 分数保留原值。

完整 257 维分数只作描述，36 格中 4 格区间下界为正。其中 common_level / all_heldout / nominal 的 signed excess 为 0.02050289，95% CI [0.00408142, 0.03692437]。O19 未达到注册标准，不能据此断言所有 hard 观察等价、T2 机制不存在或任何尺度均不可检出。这是固定父、draw 库和尺度下的有限比较，未建立 power 曲线或同时检验校准。

023 新完成 257 维距离 54,009,856、O19 距离 94,517,248，共 148,527,104；加上只读复用的旧 fast 74,263,552，三项合计 222,790,656。新解码已保存 T1/T2 hard 行各 843,904；新 NN、CP、AD、G、RNG、ADC 和原 SOURCE／REAL／DEV 波形读取均为 0。项目累计 G 下界仍为 187,667。

全进程 native wall 42.9601 s，peak RSS 4,315,840 KiB，CPU user+system 43.6695 s，输出 89,758,327 B，无资源越界。durable journal 10 entered = 10 completed，无失败或在途操作。完整 inventory 37 文件核 SHA 后，选收 25 文件、4,087,036 B，tar 为 4,136,960 B；完整 console 和 operation 日志均已保留，未超 128 MiB。两份大 O19 数组继续保留远端。

独审读回 5 份不同 NPZ、108 数组，828 个闭式标量最大绝对差 4.54747e-13，12 项 U 向量代数 exact。独立复核未重算原始 hard/O19 距离。首次审计对跨平台 exp 权重要求逐 bit 相等，最大差 2.22e-16；该审计停止、诊断和有限重开已保留，未修改科学源或门。

022 的 ImportError、native1、已完成生成成本和撤回的源码就绪结论保持原样；023 成功不改写 022。017 原 72 pending 与 stage2 关闭、020 的 seed0 89 fail（seed1 全过）也保持。021 仍只是 WP5 canonical 入口分析，完整 WP5、WP6 未完成。

下一步另行登记旧六参数均值适配的有限诊断，再判断是否值得开展 time-warp 候选。015 的 42 条保存记录证明 M1 有移动、没有触及 10% RMS 域界，不能证明旧 basis 不可达。持续授权允许新增具名预算，原 WP3 六槽和 WP2 余两槽不作挪用。REAL 长训练停止边界及 response_match_not_established 保持。

证据：[最终回执](../../receipts/wp4_023_20261016/FINAL_RESULT_RECEIPT.json)、[独立数值复核](../../receipts/wp4_023_20261016/wake_20261016_1410_01/independent_result_01/RESULT_REVIEW.json)。
