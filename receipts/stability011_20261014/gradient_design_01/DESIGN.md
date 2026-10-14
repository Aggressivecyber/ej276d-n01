# 010 新头梯度与裁窗机制：最小后续诊断设计

状态：**设计草案，未登记科学作业，未执行模型前向或反向**。根流程先完成 011 保存数组的父簇稳定性分析；本设计供之后单独登记。它不创建判别器、不延长训练，也不更新任何模型或统计量。010 的科学状态仍为 `response_match_not_established`。

最小主组合选 **θ2000 × 010 已保存的 512 个 SOURCE 父簇 ×4 draw**，比较同库的 2048 REAL TRAIN，三假设分别求损失后取均值。复用 `EVALUATION_BANK.npz` 中的 SOURCE/REAL 索引、shape z、005 latent 和 extension epsilon，不运行 FullCoverageSampler，不抽新库；使用原 `evaluation_projection=2026101430` 一次生成并冻结投影，核对 fingerprint `27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482`。θ0/θ1000 的薄 CP 只作参数/元数据参考，不进入主诊断的前向预算。若以后需要三个时期的梯度轨迹，应另登记两次批次，而不是自动扩展。

这里计算的是**固定 TRAIN 评价库上的局部梯度**，不是最后一次训练批次的梯度。训练 G2000 日志使用 θ1999、不同的 1024 行批次和刷新投影，不能直接与本诊断的 θ2000 范数比较。固定 512 父簇也不代表完整 TRAIN 种群，更不代表独立采集验收。

## 1. 一次前向和明确的参数恢复

新诊断需独立入口和绑定 schema，不能调用 `train_shape010.run`、`Shape010Engine` 或旧 `fixed_policy/bind_inputs`：前者会创建 optimizer/sampler，后者限定了 010 身份和路径。预检阶段只导入标准库，先核对新 policy、bindings、完整传递源码清单、全部输入 SHA256、路径归属及新输出目录，再导入数值模块；不得调用旧 evaluation binding 的历史 module inventory。

读取 prepared NPZ 时仅允许已登记 TRAIN 字段：`source/source_carrier384/real/real_baseline100/source_ids/real_ids/real_groups/source_role/real_role`，不读取 `val/DEV` 键。保留全合法 TRAIN 元数据计数 6593/21047，但只有固定库内的 SOURCE/REAL 进入科学计算。核对 `FIT_STATISTICS008` 和 `FIT_DATA_STATISTICS` 的 TRAIN 身份及 dataset/statistics 互相绑定；不重拟合标准化、分箱、阈值或噪声。

恢复顺序沿用冻结 010 的有效路径：

1. 原 A 用 `N01Response008(stats['model'], 'deterministic')`，严格恢复原 `response008/v1` 的模型状态，确认 G19000、stage=`low_learning_rate`、D 历史每假设16000、noise_updates=0、population_learning=false，以及 config/statistics 匹配。先执行 baseline 的 stage 恢复、`requires_grad_(False)`、`eval()`、`to(device)`。
2. 严格恢复原 005 schema5/update5000/model_type=`ObservedBaselineNoiseV3`，冻结并 eval，创建原三假设 extension。005/extension 的采样仅从固定库 latent 产生，禁止新的随机 draw。
3. 再构造 `FrozenDeterministicVoltageShape010`，使用原 seed2026101420/voltage_gain0.05。构造后**不要对整个 wrapper `.to(device)`**。只严格载入 `final.pt['new_head']`；不得载入、创建或推进 optimizer、sampler 或 streams。薄 CP 确认 G2000、完整步边界、两参数 optimizer step2000、白名单恰为 `basis_head.weight/bias`。
4. baseline/005/extension 全部参数 `requires_grad=False`，buffer/state hash 固定；仅现存 new head 两参数允许 autograd。head 用 eval，不使用 stage 去控制可训练参数。记录实际 θ2000 头 tensor digest，结束时值和 digest 必须不变。

SOURCE 按原 parent-major 顺序展开 2048 行，chunk128，与 010 保存评价相同。每 chunk 调用原 `model.random_forward(..., enabled=True)` **一次**；原 A 每行只前向一次，原 decoded 是 detached，旧 A 不参与反向。复用输出的 `base_signal_pre_V`，不增加额外 A 或 initial/1000 前向。005 每行生成一次，三个 extension 共用它的 prefix 和原 epsilon。所有假设共用一个新头信号 S、shape bank 和 prefix。

读入三份已保存 `frozen_A_*_TRAIN.npz` 的 parent IDs/peak/crop 作为 A-anchor，并核对银行/父簇顺序。可用其 pure signal 核验当前输出的 `base_signal_pre_V`，用保存 `shape010_final_*_TRAIN.npz` 核验当前 hard 波形、peak/crop/Q 和原距离。保持 chunk、数值环境、dtype、确定性和 TF32 设置；复现不一致应报告工程失败，不能换库、改精度或挑选结果。

## 2. 原目标的逐损失新头参数梯度

令 θ 为按 `weight.flatten(), bias.flatten()` 顺序排列的1288参数，h为三个噪声假设。所有 W/Q 均调用冻结 `N01Loss008`、`stage='joint_stochastic'`，comparison weights=`None`，同一固定评价投影和原 TRAIN statistics。

\[
L_w=\tfrac13\sum_h W_h,\quad
L_q=\tfrac23\sum_h QPSD_h,\quad
L_r=0.01(R_{V}+R_{d2}+\tfrac13\sum_h R_{sat,h}),\quad
L=L_w+L_q+L_r.
\]

`R_V=mean(random_shape_pre_V²)/a_ref²`、`R_d2=mean(diff2(random_shape_pre_V)²)/a_ref²`；正确按所有2048行加权 chunk 标量，不能把chunk均值再错误平均。`R_sat` 用原 preADC carrier 及原 saturation penalty。原 A regular、noise loss、trend、adv 不进入目标。

一次前向保留新头图，用 `torch.autograd.grad(..., retain_graph=True)` 返回梯度，不调用 `.backward()` 累积 `.grad`，不创建 optimizer。分别保存 **已加权** `g_w/g_q/g_rms/g_curvature/g_saturation`；`g_regular` 为后三项之和，独立再求 `g_total=grad(L,θ)`，与各项向量和核对。五分量加一次 total 共6个 AD 查询。记录1288完整向量、weight/bias各自 L2、8个输出 basis 的切片范数、全头 pairwise cosine、与total的投影以及抵消比例。零范数的 cosine 写 null 和原因，不伪造0。

总梯度求和一致性用 `||g_total−sum(g_i)|| / max(sum||g_i||,tiny)`，预登记工程容差1e-5，报告绝对误差。只记录相对010 clip5的假设裁剪系数，不实际裁剪。损失值大小不能替代梯度归因；这也只是**新头**归因，不能泛化为旧 charge/time/U-Net 分支冲突。

对 `L_w/L_q` 同时请求各假设 window endpoint 梯度以及共享 S 的梯度。它们用于后面的样本影响和观察算子 probe，避免再跑模型。

## 3. 硬观察、STE与A-anchor裁窗

同一 carrier `c_h=S+n_h` 分别调用 `observe_generated_carrier(ste=False/True, return_carrier=True, return_indices=True)`。硬/STE 的 ADC、baseline、peak、window、Q及loss前向数值应完全一致；保存 equality、绝对误差和所有 flags。STE 内区间传递 clamp 的导数，硬 `round` 的经典导数几乎处处为零；first-argmax/gather索引没有梯度。因而“forward相同”不证明 STE 捕获裁窗跳变，不能用硬 autograd 零梯度否定或验收 STE。

记录每行实际 `pθ`、原 A 的 `pA`、crop/global peak、搜索首峰与runner-up差及ties，所有 boundary/outside-search/ADC rail/nonpositive Q/PSD undefined或guard情况；保留全部2048行和512父簇，绝不删除困难事件。分别报告改变的draw数、改变的父簇数以及三假设联合父簇数。

在同一实际 ADC carrier 和同一prefix baseline上构造：

\[
Y_h^{act}=gather(ADC(c_h)-b_h,pθ),\qquad
Y_h^{A}=gather(ADC(c_h)-b_h,pA).
\]

这不增加模型或ADC前向，只改变 gather 索引。记录实际与anchor crop下的 Q、tail、PSD、wave/QPSD及Δ；在两个crop各自分解 `Q_signal+Q_noise+Q_ADC_delta` 并核验总Q identity。这里控制的是当前同一carrier，不能把 frozen-A 波形本身的差误叫裁窗效应。

**条件membership必须单独控制。** 原 QPSD 对 log-Q bucketize 的输入已 detach，但实际/anchor crop仍可能换 bin 或支持状态。以θ2000实际hard观察确定每h的 fake/REAL memberships和support列表，冻结它们；在新诊断模块显式实现 frozen-membership QPSD helper，复用原 `_qpsd/_z/empirical_wasserstein_1d/context` 和0.5权重/八bin均值，不修改旧loss或boundaries，不猴补 bucketize。先在实际参考点证明 helper 的值与1288参数梯度都与原 QPSD一致（1个额外AD查询）。

对 A-anchor crop 计算原 native QPSD 值和 frozen-membership QPSD 值，同时记录membership迁移和support切换。`native actual−native anchor` 含分箱变化；`fixed-membership actual−fixed-membership anchor` 才是在固定分箱条件下的裁窗对照。再保存anchor-crop的 `g_wave/g_QPSD_fixed`（2个AD查询），与实际对应梯度比较。regular不依赖crop，不重复求。

## 4. 裁窗改变对应哪些样本影响

分布损失没有天然逐行可加的loss，禁止按row臆造“每样本损失”。在实际参考点保留 `e_l,h=∂L_l/∂Y_h`，计算 `a_l,j,h=<e_l,j,h,Y_anchor−Y_actual>`。这只是局部一阶signed influence；同时报告其和、真实loss差以及线性化余项。按 fixed source-Q8 strata、crop改变/未变、peak margin/tie/flags分类汇总，Q8用原SOURCE canonical Q预先定义，不能按结果挑样本。

令 `m_j,h=1[pθ != pA]`，mask在参考点detach。用 `(e_l,h*m_h)` 对同一新头图做一次联合window VJP，得到 `g_l,changed`，另用 `g_l−g_l,changed` 得到 unchanged。wave/QPSD各一次，共2个VJP查询；核验 changed+unchanged重构原梯度。保存这两个向量、范数、余弦及实际row/parent denominators。两个范数不能相加当“百分比贡献”；mask本身是θ2000后验分组，这不是事件的因果标签。

## 5. 有限carrier probe：硬算子与代理的差异

无需新NN前向。固定方向 `v=random_shape_pre_V/max(abs(random_shape_pre_V))`；v在参考点detach，继承prefix0/canonical-zero-Q。若全为零，完整记录undefined，不换方向。令 δ∈{0.1,1,10}×原ADC LSB=1/16383 V，正负各一次，共6个signed扰动输入（3对），三假设使用共同 `S±δv`。probe只研究观察算子，**不是θ参数有限差分或新增物理候选**。

三个路径使用同一冻结membership/support和同一projection：

| 路径 | ADC值/代理 | crop | 解释 |
| --- | --- | --- | --- |
| S | `ADC(c0).detach()+clamp(c0±δv)−clamp(c0).detach()` | 固定当前pθ | 锚点值等于硬ADC、局部导数等于STE的代理 |
| H_fixed | 硬ADC一次 | 固定当前pθ | 量化台阶对方向响应的影响 |
| H_moving | 复用H_fixed的硬ADC | 重新first-argmax | 在同一量化结果上的crop迁移影响 |

S与H_fixed/H_moving都按各自prefix值重新mean-baseline；v前100为零可再次检查配对。分别计算 `mean_h(W+2QPSD_fixed)` 的中心差分和 `d_AD=<∂(L_w+L_q)/∂S,v>`。在参考点验证S值/梯度与原STE；输出所有步长结果，不挑“最好”的ε。S与AD若跨W1排序、peak-normalization ties、Q-floor等边界，记录相应变化，不把差分差异自动解释为错误。H_moving−H_fixed在分箱固定且共享硬ADC时定位crop引起的非连续响应；另记录native bin/support迁移，但不把它混入上述差分。

保存ADC变化cell数、crop/peak变化row/parent数、各类guard/floor/boundary，以及差分值/相对差。没有科学通过阈值，没有由probe决定重新训练或启用GAN的自动规则。

## 6. 待根单独登记的范围与预算

最小额外许可明确分开：

| 类别 | 后续登记范围 |
| --- | --- |
| 数据读取 | prepared DATASET008 **TRAIN白名单**和固定bank/保存A与final锚点数组；DEV=0 |
| 旧权重恢复 | 原A和005只读各一次，head薄CP0/1000/2000可读；仅final head进入科学图 |
| 旧模型前向 | 原A 2048行、005 2048行；三个extension共享prefix；θ0/1000 forward=0 |
| 新头前向 | θ2000 2048行，chunk128；probe追加head/旧A/005 forward均为0 |
| 参数/endpoint梯度 | 9个scalar AD查询+2个masked VJP=11；只对1288头参数及相应中间张量求导 |
| 观察与probe | actual hard/STE ADC各6144行；6个signed probe硬ADC共36864行；总硬/STE ADC budget49152行，moving/fixed共享ADC |
| 更新与其他模块 | optimizer创建/step=0，G/D/noise/pop/C2ST更新=0；无新D，无新的generator候选 |
| 候选资源 | CPU12、GPU10GiB、RSS8GiB、wall900秒、输出128MiB；根可先在人工fixture确认再固定 |

输出建议为 `BINDING.json/FROZEN_STATE.json/FIRST_DIAGNOSTIC_PROOF.json/status.json/FAILED.json/GRADIENTS.npz/ROW_MECHANISMS.npz/DIRECTIONAL_PROBE.json/SUMMARY.json/COMPLETE.json`，不输出raw REAL波形。对每个NN、AD和观察操作，先原子保存attempt/in-flight范围，再记录完成数；失败先保留原始错误和已完成数，无自动重启。首次证明需至少包含实际第一128行共同family及有限梯度完成，而非仅初始化。结束核对input/source/所有模型tensor hash，head参数值也保持不变。

根以后需要冻结新的policy/schema/ID/隔离remote root、传递源码manifest和bind slots，包括旧008输入7槽、010 thin CP3槽、010bank/COMPLETE及A/final anchor数组。旧53份source/原权重/010报告保持不变。允许复用它们的已冻结数值函数，但新的预检gate和frozen-membership helper必须是新增文件，人工fixture先验收其入口守卫、hard/STE forward一致、membership固定、changed-mask VJP重构、无optimizer及失败记账。

科学解释限定：这是有限局部机制诊断；即使gradient vectors可复现、crop效应明确，也不等于响应验收或noGAN优于GAN。未来GAN因果对照需相同新头、初始化、数据/随机库、optimizer/G预算下只切换adversarial，旧full19k不能替代这一对照。本设计未授权自动开始该对照。

源码依据：`train_shape010.py:228-239,253-318,330-336,350-405,478-515`；`model010.py:153-233`；`model.py:43-53`；`observation008.py:45-150`；`losses008_v2.py:199-292`；`losses008.py:40-67`；`response008.py:55-72,104-110,136-143`。
