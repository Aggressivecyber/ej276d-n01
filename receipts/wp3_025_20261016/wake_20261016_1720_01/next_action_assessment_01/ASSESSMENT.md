# 下一步建议：WP5/026 冻结M0的组成敏感性准备

建议只登记一项`WP5_026_M0_COMPOSITION_STRESS`：保持025 zero对应的seed0 M0响应、原6593父和三电子学假设不动，用nominal及两种固定SOURCE组成倾斜，检查硬输出标量的人口混合量如何变化。本文件仅为实施建议；没有新增数组计算、SSH、源代码或科学执行。

## 为什么选这一步

025终态快照为native0、73 pass/17 fail/42 pending，固定plus2的90项正T有限非劣未通过。bank0/fast的wave从0.149554849升至0.338614005，rise W1从1.562763163升至2.301445214 ns。024均值witness成立与025分布退化同时保留，不追加theta、改门或从其余已暴露点继续挑选。[024 scope](../../../wp3_024_20261016/registration_01/PROSPECTIVE_SCOPE.json)；[025 scope](../../registration_01/PROSPECTIVE_SCOPE.json)；[025原生快照](../SNAPSHOT.json)。025独立终态读回由ROOT收口，不能把本建议写成该读回已经完成。

主计划WP5要求区分源人口、选择过程与响应；021已经说明canonical总体方差可掩盖核内差异。020无配对seed0有89失败、seed1通过，双seed资格未过；023固定T2的36格均未达到原O19 .10可检出标准，无support/MC unresolved，T2训练仍关闭。这些证据支持先做冻结系统的组成敏感性，不支持更多均值可达性尝试或扩展随机族。[020结案](../../../wp4_020_20261016/FINAL_RESULT_RECEIPT.json)；[021结案](../../../wp5_021_20261016/FINAL_RESULT_RECEIPT.json)；[023结案](../../../wp4_023_20261016/FINAL_RESULT_RECEIPT.json)。

## 唯一问题与固定情景

问题是：在一个固定M0核内，温和改变模拟SOURCE沉积组成的比较测度，会怎样改变已保存硬输出的均值、方差、missing/截尾支持及有限同父分解？它量化假设敏感性，不拟合真实源人口，不以分数改善晋级模型。

沿013审计已验证的`h_visible=.15*edep_hadron/(edep_ee+.15*edep_hadron)`，只用三个事先固定情景：

- nominal：`w_i=1`。
- h_high：`w_i=1+.20*(2*h_i-1)`。
- h_low：`w_i=1-.20*(2*h_i-1)`。

对合法`0≤h≤1`，raw比较权重始终在[.8,1.2]；这是权重倾斜幅度，不是把h、中子份额或中子能谱改变20%。h是SOURCE沉积组成代理；系数.15来自原生成器定义，不是新物理淬灭标定。输出Q和canonical Q均不能当已标定中子能量。输入关联和单场实测不识别唯一真实增益、人口或触发/QC选择函数。

原事件计数权重、原physical weights及其known/unknown状态逐项保存。013审计明确`calibrated_original_physical_weights=unknown`；未知继续未知。新比较权重独立存储，仅在各情景按总质量归一，不替代或乘上原physical weights，不从REAL或输出误差选择其幅度/符号。[013输入审计](../../../../INPUT_INFORMATION_AUDIT.json)。

## 已有输入与身份连接

数值输入已存在，共9个NPZ、27790366B：025的6份`zero_b{0,1}_{fast,slow,common_level}_OBSERVABLES.npz`、`SAVED024_AXES.npz`、`REAL_OBSERVABLES.npz`，以及013的`information/PARENT_INFORMATION.npz`。前8份的精确RAW SHA/bytes在025快照inventory；013文件7765055B、SHA `fdc020474cad666d940ce49d1b033cd5eb728f9c7355d9a4f5a9b8a3bc8dec8c`，已由024冻结输入绑定。本地该013文件存在，本评估未打开其数值成员。无需新原波形、hard解码、CP或NN；不使用plus2观测量。

013 schema提供8192条`source_id(<U75)/role/cache_row/h_visible/edep_ee_MeV/edep_hadron_MeV/original_count_weight/physical_weight_known`。执行时先按保存角色取TRAIN，再以完整canonical字符串ID与025 axes的6593父作唯一一一连接，检查集合一致并保存显式索引映射；不能假设两个文件当前行顺序相同。由`source_parent_row/draw_index`把父权重复制给原4draw。角色、原父ID和权重原件不改；INTERNAL_VAL/DEV不进入这次统计。

REAL只读025已保存21047行OBS及原IDs/groups，保持原已暴露TRAIN角色；不再读DATASET、baseline或原REAL波形，也不重分配粒子标签。REAL仅提供固定标量描述参照，不存在与SOURCE的逐父配对。422个逻辑组仍属于一次既有采集，不制造外部确认。

## 最小统计与预算建议

固定2bank×3hyp×3权重=18格。每格报告原8标量Q/front/tail/PSD_raw/PSD_guarded/peak/rise/FWHM的均值、方差，以及原9个flags的计数/加权质量/原分母。每个指标分别使用其有限mask并独立归一；保存有效行数、有效父数、有效比较权重质量和parent ESS。没有定义值的统计保留undefined，不能写0。原全人口nominal结果始终并列，不删除困难行或只展示较好情景。

Q/front/tail/peak另作18×4=72个**有限4draw经验测度**分解。令`m_i=mean_d z_id`、`v_i=mean_d (z_id-m_i)^2`，其中within分母为4；`alpha_i=w_i/sum_i w_i`。报告`mean=sum alpha_i*m_i`、`between=sum alpha_i*(m_i-mean)^2`、`within=sum alpha_i*v_i`与同一行测度`alpha_i/4`直接方差，核对`total=between+within`。若复用ddof1保存方差，必须先乘3/4；不能把分母3直接代入总方差恒等式。

该恒等式在保存经验测度上exact，不代表真实条件方差分离。4draw估计的父均值含MC误差，between不是真实核间方差；两bank分别报告，不能把draw算独立父。当前只覆盖一个M0 seed，不提供模型seed不确定性，也不新增bootstrap/置信区间。

建议登记上限：9个NPZ一次解码；6份fake OBS共158232行、REAL OBS21047行、SOURCE metadata8192行中连接TRAIN6593父；144个标量情景摘要、162个flag情景摘要、72个经验分解。新NN/CP/AD/G/RNG/ADC、CDF/energy距离、quantile、参数fit和theta点均为0。CPU1、RSS≤2GiB、wall≤120s、输出≤8MiB足够作为前瞻资源cap；准确执行资源由未来原生回执给出，不在本评估宣称已测。无需常驻监控。

只报告相对固定REAL均值的signed残差及三情景变化，不把原013的W1容差T直接套到均值或方差，不创设新pass阈值。全部结果保留，不能选最接近REAL的情景当“拟合人口”或回填025通过。

## 对主计划的推进与停止边界

这一步比重复可达性诊断更接近WP5：响应字节保持不变，人口测度单独改变，因而可直接展示一个受限系统对组成假设的依赖。即使指标变化大，也只能说明该stress方向有影响；若变化小，也只支持这两个温和倾斜方向，不能证明人口与响应已可辨识或跨源稳健。

既有015 M0与016 Q1均有两seed、两bank、三hyp的原硬评价及独立保存数组核验。MASTER当前研究参考仍是`016_Q1_mean_off_charge_on`，weights未晋级、绝对接受未通过。本次选M0是为直接复用025保存量并隔离核，不替换Q1、不合并不同随机库作配对检验；Q1既有证据继续作为受限参考档案。[015报告](../../../../reports/wp3_015_20261015/REPORT.md)；[016报告](../../../../reports/wp3_016_20261015/REPORT.md)；[MASTER](../../../../MASTER_EXECUTION_PLAN.json)。

本项只完成WP5的一项准备。受限physical-simple/deterministic/最终参考的统一确认、输入形状干预/条件关闭消融、模型seed/父/REAL组/新随机库不确定性、全流程分组CV、独立C2ST与REAL–REAL、适用域及软件交付仍未完成。不能用h stress代替这些要求，也不能据单场已暴露TRAIN默认新采集、高秩族、真实长训或完整WP5/6完成。ROOT决定是否登记026后，才进入实现和唯一审核流程。[用户主计划](../../../../USER_MASTER_PLAN_20261015.txt)。
