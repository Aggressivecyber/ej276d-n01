# WP4 017：已知教师双参数恢复报告

更新：2026-10-16（新加坡时间）；运行 `wp4_017_20261015`。

017 第一阶段正常完成，原生退出码为 **0**，两个 T1 paired seed 各完成 2000 次更新，总 **G/AD = 4000**。2688 个必需评价单元中，**2616 pass、72 pending、0 fail**；第一阶段门为 `closed`，`stage2_allowed=false`，后六项试验未运行。72 个 pending 来自共同有效支持不足，不能改写成 `MC_unresolved` 或模型恢复失败。

两 seed 的 JSON 报告给出以下参数结果。相对教师误差取自 `parameter_recovery_secondary`；二者均满足已登记的 secondary 25% 条件，该结果保持 `secondary_only_not_primary_gate=true`。

| 试验 | a（V·ns） | b | a 相对误差 | b 相对误差 |
| --- | ---: | ---: | ---: | ---: |
| [T1 paired seed0](../../runs/wp4_017_20261015/selected_collected_01/runs/primary/artifacts/T1_paired_seed0/COMPARISON.json) | 0.0361930005881 | 0.00244156871351 | 2.455254% | 2.337251% |
| [T1 paired seed1](../../runs/wp4_017_20261015/selected_collected_01/runs/primary/artifacts/T1_paired_seed1/COMPARISON.json) | 0.0373999038316 | 0.00241770356107 | 0.797504% | 3.291858% |

两 seed 各有 1308 pass、36 pending。pending 全在低 Q 留出组 `fit_Q_tertile0_heldout`，该组原有 **435 父事件**；三个噪声假设的共同支持父数分别为 `fast: 402`、`slow: 403`、`common_level: 404`，计数保留率为 **92.4138%～92.8736%**。三种权重下的共同权重质量保留率如下，均低于固定 95% 门槛；两个 seed 的这些支持结果相同。

| 权重 | 共同权重质量 / 原组权重质量 |
| --- | ---: |
| nominal | 92.4138%～92.8736% |
| low | 93.0016%～93.5724% |
| high | 91.7805%～92.1444% |

72 个 pending 按指标分为 `conditional_cov9: 18`、`time_rise10_90: 18`、`PIT_coverage: 36`，每项均同时记录 `common_count_below95percent_original` 和 `common_weight_mass_below95percent_original`。门的判定保留 `pending_is_pass=false`。

唯一一轮独立数组复核已完成，原生退出码为 **0**。复核在既有远端路径读取 **18 个 NPZ + 1 个 Q.npy**，按原 inventory 验证字节数与 SHA256；72 个 pending 的分子、原 435 父分母、三种权重质量、ESS、reasons 和原 indices 全部一致。回执记录的远端 NumPy 计算耗时为 **0.206 s**、峰值 RSS **81452 KiB**；新增 NN、G、AD、随机抽样与 probe 均为 **0**。见[独立复核结论](../../receipts/wp4_017_20261015/wake_20261016_0223_01/saved_support_review_01/REVIEW.md)和[完整数值回执](../../receipts/wp4_017_20261015/wake_20261016_0223_01/saved_support_review_01/NUMERIC_READBACK.json)。

教师自身在低 Q 组的时间量支持已低于固定 95% 条件。两学生的支持计数相同，按假设分解如下：

| 假设 | 教师支持 / 435 | 教师支持率 | 每个学生支持 / 435 | 共同支持 / 435 |
| --- | ---: | ---: | ---: | ---: |
| fast | 407/435 | 93.5632% | 404/435 | 402/435 |
| slow | 407/435 | 93.5632% | 405/435 | 403/435 |
| common_level | 408/435 | 93.7931% | 405/435 | 404/435 |

低 Q 组的 cov9 支持掩码与 rise10–90 支持掩码完全一致，其余 8 个标量均为 435/435。保存 flag 中，教师的 rise undefined 槽数等于 prepeak-censored 槽数：三个假设依次为 **953、963、969 / 27840 原槽**；学生 seed0 为 955、957、950，seed1 为 957、952、950 / 27840 原槽。`temporal_nonpositive_peak` 与 `rise_other_undefined` 均为 0。因此，固定教师侧时间量的定义域和峰前截尾限制参与造成本轮 primary 支持 pending；参数次级通过不能据此打开 stage2。

**24 个解析方差单元复核一致**。使用全部 **6593 原父**，无删除、无无效方差；按 `v=aQ+bQ²` 复算，相对教师曲线的全部原父 RMS / 最大误差分别为 seed0 **2.411353% / 2.451065%**、seed1 **0.802128% / 1.266185%**。教师参数为 a=0.03710399804675581 V·ns、b=0.0025。


017 的最终 G、AD、optimizer 和 backward 计数均为 **4000**，286600 次操作 entered 与 completed 完全相同，无 inflight、failed ID 或截断日志。项目累计为 **179667 + 4000 = 183667**；本次苏醒相对先前已登记的 139 次，只新增记账 **3861** 次。冻结 A / 005 / M0 保持不变，保护错误为空；REAL 波形读取和原始 DEV 波形读取均为 **0**。

监督进程记录耗时 **1903.313 s**，峰值进程树 RSS **6169710592 B**、GPU **471859200 B（450 MiB）**、输出 **4042865100 B**，均在已冻结资源上限内。原生退出与计数见[终结快照](../../receipts/wp4_017_20261015/wake_20261016_0223_01/status_01/REMOTE_STATE.json)。

选择性回收已通过完整远端 inventory、归档 SHA256、成员及控制文件核验：远端共 147 文件，回收 45 文件（含 5 个小检查点），遗漏 102 文件，其中 **99 个 NPZ 全部保留远端**。归档为 **131471360 B**，低于 256 MiB；SHA256 为 `80132679a64170fdf26ef460a25859a3b551f66e289dcc6dd2117f9f92f83a8a`。见[回收回执](../../receipts/wp4_017_20261015/wake_20261016_0223_01/collection_01/COLLECTION_RECEIPT.json)和[完整 inventory](../../receipts/wp4_017_20261015/wake_20261016_0223_01/collection_01/REMOTE_FULL_INVENTORY.json)。

本报告汇总工程终结、producer 比较报告和这唯一一轮独立数组复核。独立复核覆盖保存的 derived 支持、coverage 有限计数/支持与 sentinel、原父/权重/分组及解析方差；没有从 native bank 原标量重新生成支持，也未重放 morphology、hardwave、energy、PIT/CDF、训练 loss 或 CP。**2616 pass 仍是 producer 计数，本轮未独立重算全部主门**。原始数组继续保留远端，科学状态保持 `response_match_not_established`，`scientific_acceptance=false`。

科学分支维持：**stage2 与真实长训继续关闭**；unpaired 和 T2 均未执行。最小下一步是在保留 **256 点硬观察口径**的前提下，先明确峰前截尾变量、可定义 rise 的可测范围及其联合评分表示，再单独登记并冻结未来合同。本轮 017 原规则、门槛与 pending 结果保留，不回填通过、不默认重训或进入 WP5 真实长训。WP5 / WP6 只能按主计划在本次结论范围内推进；T1 合成教师得到的 heads 保留为已知教师恢复产物，不能作为已经验证的实测部署权重。
