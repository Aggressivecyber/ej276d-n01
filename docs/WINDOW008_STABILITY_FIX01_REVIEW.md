# Window008 随机形状失稳：fix01 独立数学与恢复审查

日期：2026-10-13。状态：设计审查、独立模型核心、人工 CPU 语义检查及独立失败现场 CPU 审核完成；本作者未启动任何训练或 GPU 作业。恢复驱动与原生 CUDA 检查由其他明确分工完成，状态以各自回执为准。

本审查依据当前 `USER_PROPOSAL.md`、N01 `AGENTS.md`、`model008.py`、`response008.py`、`train_window008.py`、`window008_config.json`、`WINDOW008_SOURCE_MANIFEST.json` 和 Window008 冻结协议。协议优先级与 N01 独立性保持不变；Window008 的事实以本轮冻结文件和现场回执为准。后续实施授权已明确允许最小 fix01 施工，正式修订协议为 `docs/WINDOW008_STABILITY_FIX01_PROTOCOL.md`。

## 结论与唯一推荐路径

另立 `window008_fix01`：**只对 full 联合阶段的随机形状基加入 canonical 零积分投影，然后做原有 RMS 归一化，现有基函数正则作用于投影归一化后的基。** 不改变平均残差、时间头、正电荷头、滤波器、局部信号涨落、电子学、ADC、数据、损失系数或随机流。

保留原 det 的固定 19,000 G 终点；full 从原冻结 `stage_03000.pt` 导入完整状态，执行 12,000 joint + 4,000 low 更新；simple 从原共同初始化执行 19,000 G。这是一项**失败后登记的实现修订**，不能写成原 Window008 未改动协议已经完成。旧失败分支及所有实际优化计数保留。

该修复消除了随机形状系数直接改变规范化分母的一条结构性失稳通路，**不是对所有时间伸缩、平均残差或 Gaussian 极端取值的全局稳定性证明**。非正 base shape charge、非有限值、退化投影基仍必须保存现场并停止；不自动调整其他分支继续尝试。

## 1. 原实现的问题是随机分母，不是最终电荷被算了两遍

记线性 canonical 电荷为

\[
C(v)=\sum_{t=110}^{365}v_t,
\]

载体长 384，1 ns 网格，峰锚 126。它不同于经过电子学、ADC 和重新寻峰后的最终 256 点观测电荷。以下全部等式是信号载体的 canonical 等式。

令 `g` 是原 onset gate，`r_mu` 是原平均残差，

\[
m_a=g\,\mathcal T_a(h*u_x)+r_\mu,\qquad q_0(a)=C(m_a).
\]

原 `model008.py:205-216` 对展开并 gate 后的基只做 RMS 归一化。若 `c_k=C(B_k)` 非零，则

\[
C(v)=q_0(a)+\sum_{k=1}^8\eta_{k+1}c_k.
\]

最终 `v/C(v)` 在分母有效时本来就使 `C(signal)=Qsig`，所以不能把缺陷描述为最终总电荷方差必然被重复加了两遍。真正的问题是：本应描述形状的 Gaussian 系数同时给规范化分母加入有符号随机量，造成整体缩放、病态梯度及跨零风险。

联合头的元素限幅限制标准差，不限制 Gaussian 实现值。固定 SOURCE、时间和电荷坐标后，形状坐标仍有其下三角独立对角项；只要存在非零 canonical 基积分，通常仍存在使分母非正的形状实现。减小学习率或等到下一批样本不消除这个数学通路。当前 `_positive(shape_q, 1e-6, ...)` 的失败是应被保留的正确行为。

已核对保存的 `runs/window008_20261013/runs/science/FAILURE.json`：full 完成第 3,269 次 G 后，在下一次尝试中出现 `shape canonical charge = -0.6842861649644458`。独立现场诊断 `window008_failure_fix01/diagnosis/DIAGNOSIS.json` 及报告确认：CPU row22 的 warp/gate 积分 `31.6040391437034`、mean residual 积分 `10.104300498318358`，base 为 `41.708339642021755`；八条随机基合计贡献 `-42.39262093869865`，总和为 `-0.6842812966768888`。CPU 与原 CUDA 相差约 `4.87e-6`，不宣称跨设备逐位一致。512 行 base Q 最小为 `28.50189215393769`，该例时间尺度正常且无边界外查询；local 和电子学不在失败分母中。

本作者只读取上述现场 JSON/文字回执，未解码科学数组或调用现场模型；独立诊断代理的实际 SOURCE 前向单列于第 10 节，不能被归入零科学输入使用的人工检查。

## 2. 投影定义、放置位置及正则

复用原 local 分支已注册的固定平滑 `projection`：

\[
p_t=\frac{g_t\exp[-\tfrac12((t-180)/65)^2]}{C(g\exp[-\tfrac12((t-180)/65)^2])},\qquad C(p)=1.
\]

对每个 SOURCE 的每条基按下列唯一顺序处理：

1. 原 `basis_head -> 48 controls -> cubic expansion -> onset gate`，得到 `b_k`。
2. 在最终 384 点、峰锚 126 的载体坐标上投影：`b0_k = b_k - C(b_k) * p`。
3. 计算投影后的 `rms_k = sqrt(mean(b0_k**2))`；非有限或 `rms_k <= 1e-10` 时显式失败，不用 epsilon 或 clamp 代替检查。
4. `B_k = b0_k / rms_k`；后续随机组合、Gram 非对角惩罚和二阶差分惩罚全部使用这个最终 `B_k`。

投影算子 `P=I-p C` 满足 `C(Pb)=0`、`P(Pb)=Pb`；RMS 归一化保持零积分。该 `P` 沿固定平滑 `p` 去掉一个 canonical 电荷分量，通常不是欧氏正交投影。“最小修复”指修改范围和删除的电荷自由度最小，并不声称它是唯一物理解或最小 L2 改变量。

不能在 48 控制点上直接减均值，不能减全 384 点均值，也不能先投影后再乘 gate；这些操作使用不同线性泛函，会破坏 `C(B)=0`。当前随机基不经过时间 warp；若未来改变这一点，必须在 warp 和 gate 之后的最终坐标重新定义投影，而非假设零积分会自动保持。

现有局部涨落 `ns = raw_local - C(raw_local)*p` 已满足同一约束，不应重复归一化。电子学 `ne` 不投影；最终含噪波形也不投影。`p` 的载体末端不人为归零，保持原窗口及末端约定。

均值残差的幅度/一阶差分惩罚、基相关性系数、基二阶平滑惩罚、envelope 惩罚和所有全局 loss 权重保持原值。对投影前已被删除的 charge 方向额外加惩罚既不能代替结构约束，也会扩展本次修复的目标，故不加入。退化基必须失败；不能用任意替代基、SVD 旋转或新随机重初始化悄悄改变既定系数与随机流。

## 3. 修复后的分解及仍存在的风险

在精确算术中，

\[
v=m_a+\sum_{k=1}^8\eta_{k+1}B_k,\quad C(v)=q_0(a),\quad
\hat u=v/q_0(a),\quad s=Q_{sig}\hat u+n_s.
\]

因此在 `q0(a)>1e-6` 且全部有限时，`C(s)=Qsig>0`；加电子学后，`C(y_preADC)=Qsig+C(ne)`。共同观测重新寻峰仍可改变积分窗，不能据此宣称最终观测 Q 等于 `Qsig+C(ne)`。

继续使用原合成顺序及原最终 `shape_q=C(raw_shape)` 分母和检查。可在 full joint/low 的分解处显式计算并检查 `base_shape_q=C(m_a)`，同时记录 `C(warped*gate)`、`C(residual)`、每条基投影前/后电荷、`C(stochastic_shape)`、最终 `shape_q`。有限精度下 `C(stochastic_shape)` 只应有舍入残差；出现异常必须留存，而非修正最后一个采样点或更换分母。

“给定 Qsig 的形状涨落”表示所有该分支输出保持 canonical 信号电荷；它不意味着 `E[eta_shape | Qsig]=0`。原联合 `L z` 可继续表达电荷、时间和形状相关性，不应为满足零积分而拆开随机采样或对条件系数重新中心化。

尚未被消除的风险包括：

- 时间尺度的均值项被限在约 0.5–2，但其乘性 Gaussian 随机项仍使实现值具有正的无界支持。`scale>0` 不保证有限窗口积分为正。
- `r_mu=0.20*tanh(head)*gate` 允许有符号平均残差；gate 的 canonical 和约为 255.59，因此残差的 canonical 电荷可接近 ±51.12。它没有零积分约束。
- 原因果 FIR 总 DC 归一化且扰动有界，仍允许有符号尾部；其 DC 正性不等价于经过 warp、gate、截窗和 Hermite 延拓后任意组合的 canonical 正性。
- 即使时间均值下的 `C(m_a)` 为正，随机时间压缩可以减小正的源贡献，令原负残差占优。本轮人工反例在保持同一残差、identity FIR 时，scale=1 的 base Q 为 `9.73124656`，scale=0.4 时为 `-3.75697352`。
- 近零的正 base Q 仍可能使形状和梯度变大。原 `1e-6` hard guard、非有限检查和训练梯度检查必须保留；本次不新加根据失败事件调过的 barrier loss 或改分母的软保护。

因此本修复只隔离并修复随机基电荷方向。若随后出现 base Q 失败，应保留新的失败并重新审查平均形状/时间参数化；不能将这轮协议变成自动搜索稳定变体的循环。

## 4. det、simple 和 full mean 为什么可复用

新增操作只在 `variant=='full' and stage in ('joint_stochastic','low_learning_rate')` 内触发。其余操作、参数、buffer、模块构造顺序和随机 draw 顺序不变。

- deterministic 没有 physical/basis 分支，全部阶段的数值函数保持原实现。
- simple 没有 basis，全部阶段的数值函数保持原实现。
- full 的前 3,000 步 mean 阶段禁用并冻结 physical/basis/envelope/local，新增投影未参与 forward 或 regular，前缀训练路径保持原实现。

这给复用 det 19,000 及 full 3,000 提供了函数层面的理由。第 9 节人工差分和第 10 节独立真实 checkpoint 检查已确认对应输出/梯度及 state key/order 一致；独立诊断还确认原 full/det stage3000 的 178 个共有状态 tensor、167 项活跃 G optimizer state、D 与 D optimizer、sampler 和全部显式 RNG 逐位相同。实际迁移仍需逐项绑定这些状态，不能仅凭类名相同声称恢复轨迹正确。

## 5. 独立版本和明确迁移，不能伪装普通 resume

建议新增 `model008_fix01.py`、`response008_fix01.py`、`train_window008_fix01.py`、`window008_fix01_config.json`、`WINDOW008_FIX01_SOURCE_MANIFEST.json` 和单独协议。使用新的本地/远端 root，保留全部原冻结 68 文件及失败目录。新执行 schema 如 `response008_fix01/v1`，明确记录只修复 stochastic-basis canonical projection。

原 `Response008Engine.restore` 要求原 schema 和完整 config 相等，所以新版本应有**专用、一次性的可审计迁移入口**，不得放宽原 restore，也不得把旧 checkpoint 内的 config/schema 字段直接改写后冒充原生续跑。

迁移入口必须执行以下绑定与复制：

1. 按原 manifest/config 哈希验证原 full 的固定 `stage_03000.pt`、原 `common_initial_state.pt`、det `final.pt` 及其完成回执。full 入口要求 G=3000、三个 D 计数均为 0、stage=mean_response；检查噪声 005、统计和数据身份完全相同。
2. 使用原共同初始化的确切 state；比较所有参数/buffer 的名称、顺序、shape、dtype。投影使用已有 buffer，不新增可学习参数；simple 仍使用原先登记的兼容 tensor 与低维 physical-head 前缀映射。
3. 对 full 完整复制 G、三套 D、G optimizer、三套 D optimizer、SOURCE/REAL sampler 的顺序/游标/RNG/visits、三条显式 response/noise005/extension RNG、G/D 计数和原初始化身份。每项输出迁移前后 digest；optimizer 还需核对参数名映射与组顺序，不能只比较匿名序号。
4. 保持 full 的逻辑 `update=3000`，首个新更新为 3001，投影损失上下文仍由原 seed 和 update 3001 产生。阶段切换到 joint 的学习率行为与原程序相同。不能把本地追加计数 1 当作训练投影 update 1。
5. 保持每步两微批、每微批 256 父事件各 2 次响应、512 REAL，及全部 `phys/local_white/noise005/extension` draw shape 与 draw 顺序。任何恢复后校验用克隆 RNG/sampler 状态，不消耗正式流。
6. simple 用原共同初始 state、原种子和原数据流从 update=0 开始；det 不再执行 optimizer。新 root 用引用/带哈希副本声明 det 来自原 run，不能把其 checkpoint 的 config 和版本改成 fix01。

不从 `FAILURE_CHECKPOINT.pt` 继续：它在尝试失败时保存，sampler/RNG 可能已经消费部分批次，D 也可能在其他失败类型中已部分更新。即便本次最后 G=3269，直接恢复它还保留了错误基定义下的 269 次 joint 优化，并不能建立修复版本的固定轨迹。第 3,000 步是事先冻结、位于修复路径启用前的阶段边界；选择它不依赖训练质量、DEV 或最优 checkpoint。

## 6. 预算与报告口径

以下原运行计数已与保存的 FAILURE 回执及独立 checkpoint 诊断核对：det 19,000 G / 48,000 D，full 3,269 G / 807 D，原总计 22,269 G / 48,807 D。独立运行还须绑定原 native exit 回执。

| 项目 | 有效固定路径 G | 本次新增 G | 本次新增 D |
| --- | ---: | ---: | ---: |
| 原 det 终点 | 19,000 | 0 | 0 |
| 原 full mean 前缀 | 3,000 | 0 | 0 |
| fix01 full joint + low | 16,000 | 16,000 | 48,000 |
| simple fresh 全阶段 | 19,000 | 19,000 | 48,000 |
| 合计 | 57,000 | 35,000 | 96,000 |

有效三臂固定路径为 57,000 G / 144,000 D；包括失败计算在内的实际累计为 **57,269 G / 144,807 D**。旧 full joint 的 269 G / 807 D 单列为失败分支，仍进入实际计算账本，不能混入 fix01 full 的有效 19,000 步终点。新 run 的新增预算 35,000 G / 96,000 D 与比较 lineage 总量需使用不同字段，避免“跳过 det”后还按 57,000 新增步调度。

最终评价应有新版本 manifest，明确解析 det 原终点、full fix01 终点和 simple 新终点的不同 provenance；旧评价驱动若严格要求三个 checkpoint 与旧 run config 相同，则不能不经修订直接套用。共同观测、统计、三个电子学假设、固定采样、TRAIN/已暴露 DEV 分别报告和不选终点政策保持原定义。失败时没有完成三臂比较；修复后的结果也不能宣称原 008 原样完成。

这项修订已受到工程失败信息影响，应标记为失败后的明确 amendment；它不是新的独立采集验证或完全未接触结果的预注册试验。保持 det 是可证明未受修复影响的计算复用，不是挑选表现最好的对照。

## 7. 交付前必要工程验证

本作者已完成下表的人工代数、新旧 CPU 人工输入差分及模型语义检查，并读取独立现场模型审查回执。迁移、两遍引擎 VJP、原生 CUDA 及终点要求由相应执行者分别出具回执，不能据本文件提前认定完成。

| 验证 | 必须回答的问题 |
| --- | --- |
| 人工代数、0 模型调用 | 投影、RMS、prefix=0、电荷恒等式、退化拒绝以及剩余 mean/time 负积分反例是否成立；原 68 文件前后哈希是否匹配。 |
| 新旧 CPU 人工输入差分 | det/simple 全阶段及 full mean 的输出、regular、关键梯度、状态键和 RNG 消耗是否与原实现一致；仅 full joint/low 的目标基定义改变。不得读取 TRAIN/DEV。 |
| 新投影实际实现语义 | 最终基 C≈0、RMS≈1，退化显式失败，自动微分有限，基/联合头仍能获得非零形状梯度，local/electronics 分解不变，ADC 仍只执行一次。 |
| 现场诊断回执复核 | 原失败是否来自 basis charge；该事件 base Q 是否为正；不能用人工反例冒充失败事件的实证。任何现场模型调用应单列计数，不能记为零科学 forward。 |
| 迁移状态验证 | 原固定 checkpoint 身份、模型与 optimizer 全状态、下一批 sampler/latent、首个 loss context update=3001 是否精确绑定；错 hash、错 schema、错配置必须拒绝。 |
| 两遍 replay 与直接图比较 | 相同显式随机输入的观测波形 replay 一致；用人工数据比较两遍 VJP 与直接图梯度。该验证不要求 optimizer 更新。 |
| 新版本原生运行与限额 | 原 008 的 CUDA 回执不能替代新核心的原生工程证据；按 root 已授权的新版本工程范围另行验证、冻结，然后才可启动科学作业。 |
| 终点与退出回执 | 新工作严格按新增 35,000 G / 96,000 D 截止，三个臂 lineage 各 19,000 G，失败分支计数另列；native exit、最终哈希、冻结 005 及父事件覆盖齐全。 |

报告的科学范围仍是当前解析 SOURCE 到现有电压/读出域的有效随机响应候选。无配对分布匹配、工程可重复、训练完成或这项稳定性修复，都不能单独证明真实探测器响应恢复、独立采集泛化或不同物理过程的可辨识性。

## 8. 本次人工检查回执

脚本：`receipts/window008_20261013/stability_design_01/synthetic_projection_check.py`。
结果：同目录 `RESULTS.json`。Python 3.13 标准库、人工 384 点数组、`math.fsum`，未导入模型、torch 或 numpy；未解码科学数组或读取科学权重。

- 原冻结 68 文件运行前后全部哈希匹配；manifest SHA256 为 `97b020dc68e41fa1e893e66fdf1091b1bf9500b98281a5dd7e2545ee9ce00929`。
- 归一化后基的最大绝对 canonical 电荷 `3.52e-14`，RMS 误差 `1.11e-16`，投影幂等误差 `2.22e-16`。
- 人工原基组合 Q 为 `-37.7350`；投影组合 Q 为 `22.5107`，与其 base Q 相等至舍入精度。
- canonical 信号电荷恒等式误差 `4.44e-16`，加人工电子学后的恒等式误差 `1.11e-16`。
- 人工有限系数扫描仅检查代数舍入，最大 Q 差 `7.82e-14`；不代表随机尾概率或全局稳定性测试。
- 先投影后 gate 的反例残留 Q 为 `-0.04866`；投影后再 warp/gate 的反例残留 Q 为 `51.16884`，说明操作顺序不能互换。
- 与 `p` 共线的退化基被明确拒绝。模型 forward、optimizer update、科学 numeric decode、DEV decode、GPU 调用均为 0。

这份人工数学回执没有执行 PyTorch 新核心、失败事件 replay 或 GPU，所以不能替代上述独立工程及科学验证。

## 9. 已实现核心与本作者的人工 PyTorch CPU 检查

核心：`src/n01/model008_fix01.py`，导出 `N01Response008Fix01` 及别名 `SignalResponse008Fix01`。父类构造、`set_stage`、`noise_shapes` 和 `forward` 调度全部继承；只有 `signal_forward` 版本化复制。新增诊断字段为 `base_shape_Q`（B）、`shape_Q`（B）、`basis_canonical_charge`（B×8）；原 26 个返回键保留。

核心 SHA256：`3bc826a1867f2658ae507ab2be683a6b901b7f481b1efaf005f3cf4953429abc`。独立审核后未改动该文件。

人工检查脚本：`scripts/check_model008_fix01.py`；当前回执：`receipts/window008_fix01_20261013/model_cpu_02/ENGINEERING_RECEIPT.json`，SHA256 `ab7f27aab7c2c9c888a3b5607a86dd60e23b12207fa294dacfbfbb52a2811717`。先前 `model_cpu_01` 通过结果保留；第二次只修正 QA 输出中“baseline 无梯度检查”的空值标记、梯度范数标题及 tensor 标量转换，不改模型核心或测试判据。

每次运行各 27 个模型 forward、99 行人工 SOURCE，含 3 个预期守卫失败；两次合计 54 forward / 198 人工行。两次均没有科学数组/科学权重读取、actual SOURCE 前向、REAL 前向、optimizer 更新、DEV 解码或 GPU 调用。

- 四条解析脉冲，PyTorch `2.12.0+cu130` 的 CPU 路径，CUDA 隐藏，不以该版本字符串声称使用过 GPU。
- det/simple 各四阶段、full baseline/mean，共 10 个路径的原 26 输出逐位相同；其中 7 个可训练路径的参数梯度逐位相同。baseline 未做无意义的梯度比较。
- 所有构造后的 state tensor、参数/buffer 名称和顺序、构造后的 CPU RNG state 逐位相同。det/simple/full 分别有 178/32/195 个 state key、167/19/180 个参数项。forward 没有隐式 RNG draw。
- full joint 基积分最大绝对误差 `1.5332e-13`，RMS 误差 `2.2204e-16`；信号电荷恒等式最大误差 `7.1054e-15`；局部 canonical 电荷误差 `5.5511e-17`；prefix100 严格为零。
- regular 与最终投影归一化基计算一致；同显式随机输入重放逐位相同。charge/time/filter/mean/physical/basis/envelope/local 八个要求分支的梯度有限且非零。shape encoder 在恒等 charge 初始化下的梯度为零，回执如实保留，未声称所有网络参数初始梯度均非零。
- 只把保存于人工 fixture 的 phys 列 2–9 放大 100/10000 倍，不改电荷/时间潜变量、不重新采样，base Q 和 Qsig/time 保持一致，canonical 信号恒等式最大误差 `1.42e-14`。
- 人工负 base Q `-26.99725`、退化零基、非有限基分别正确抛错，无返回筛选批次或替代结果。
- 原 68 冻结文件前后哈希全部匹配。

## 10. 独立失败现场审核与真实输入计数

独立审核文件：`window008_failure_fix01/model_review/cpu_01/REVIEW.json`，SHA256 `da3fb8bbaa4bec82860126d222cfd643f1e746a13df1b88bd10449f07895d367`，状态 `passed_independent_CPU_engineering_review`，绑定同一模型 SHA256。

该审核读取已保存的 failure checkpoint/context 和原 prepared TRAIN 中 `source_carrier384/source_role/source_ids`。NPZ member 解码覆盖 6,593 个 SOURCE 父事件；实际选定失败批包含 256 个不同父事件的 512 条响应输入。没有 REAL 波形或 DEV 解码，没有新 latent 或随机值抽取，没有 optimizer/GPU。

审核总计 **12 次 forward / 1,569 行 actual SOURCE 前向**：原失败批 512，固定形状 latent ×100/×10000 各 512，4 组旧新函数比较各 2×4 行，共 32，另有 1 行人工均值参数干预的预期失败。最后一项虽然改的是人工参数，输入仍是保存的 actual SOURCE，不能把它计作人工 SOURCE。

- 固定失败批 512 行全部返回；原 row22 的 `base_shape_Q=41.70833964202177`、`shape_Q=41.70833964202175`；`Qsig` 与 canonical 信号 Q 均为 `7.752995144118847`。
- basis C 最大绝对误差 `4.5186e-14`，信号 Q 恒等式误差 `3.5527e-15`，prefix100 严格零。原 physical eta 与 mean residual 逐位不变。
- 两次确定性 latent 放大保持 Qsig/time/base/basis 逐位不变，×10000 时电荷恒等式最大误差 `4.8583e-12`。这只是数值压力检查，不是物理上合理的随机人口或科学分布评价。
- full stage3000 mean、det 完成终点、simple common mean/joint 的旧 26 输出及输入/参数梯度逐位一致；模型状态无改动。
- 同一 actual SOURCE 的人工负 base 干预 `-19.51365` 正确停止。原 checkpoint、context、68 个冻结文件和 global CPU RNG 都未改动。

独立原始故障诊断先前另发生 2×512=1,024 行旧模型前向，其中第一次只因格式化字符串 parent ID 时失败，第二次完成诊断；两次均保留计数。因此上述“故障诊断＋独立模型审核”共计 **2,593 行 actual SOURCE 工程前向**，不是零真实输入使用，也不是 2,593 次训练更新。本作者的 198 行人工前向另计，不重复执行上述现场批次。

目前这些证据支持所界定的最小投影修复及不受影响路径复用；不替代迁移驱动、原生 CUDA、固定终点或真实响应的后续验收。
