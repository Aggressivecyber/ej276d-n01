# N01 模型实现绑定与人工数值验证

本文件记录 `src/n01/model.py` 的第一版实现。依据为 N01 `USER_PROPOSAL.md`、`README.md`、`config.json`；没有加载 R79 或其他历史权重、训练流、科学阈值或验收门。以下未由原文给定的数值是工程绑定，不是物理测量或已验证最优参数。训练统计由合法 TRAIN 单独拟合后传入，模型不读取数据文件或父身份。

## 接口与数组

```python
model = N01Response(stats, variant="full")  # full / deterministic / simple
model.set_stage("baseline")
noise = model.sample_noise(batch, device, generator)
baseline_noise = model.electronics(noise)   # B,1,480; 独立基线拟合接口
result = model(x, noise=noise, adc_ste=True)
```

- `x` 为 `B×480` 或 `B×1×480` 的浮点电压，源输入不裁 ADC；网络不存在 ID、文件名、序号或沉积标签输入。
- `waveform` 与 `pre_adc` 为 `B×480`；`qbar/qsig/q_source/time_scale` 为 `B`；`unit_shape/signal_noise/electronics_noise/mean_residual/signal_envelope` 为 `B×480`；`basis` 为 `B×8×480`。
- `regular` 与 `trend` 是可纳入训练目标的标量。另返回 `physical_cholesky/physical_eta/sigma_q2/shape_charge/pre_adc_charge/electronics_charge/charge_identity_residual/adc_charge_change` 及全局、逐事件饱和比例。
- `sample_noise` 的固定键顺序及形状：`phys: B×10`、`electronics_white: B×1×544`、`electronics_scale/offset/slope: B×1×1`、`local_white: B×1×496`。全部为标准正态。所有变体使用相同接口及抽样顺序；未启用分支仍保留其显式随机变量。
- 显式噪声下 forward 无随机抽样、无 BatchNorm、无状态更新。改变父事件存储顺序时，相应置换显式噪声可得到对应结果。未传噪声时才用传入 generator（或设备默认流）生成随机变量。

## 固定数值接口

- 480 点，1 ns/点；`Q(x)=sum(x[100:400])-3*sum(x[:100])`，单位 V ns。提取基线为前 100 点均值；幅度为减基线后完整窗口的最大正值。
- 源半升沿对齐代码绑定相对锚点 `t0=120`，不是积分开始点 100；只有相对伸缩，没有绝对时间平移头。
- ADC 区间 `[-0.5,0.5] V`、14 bit、16383 个端点间隔：`round((clamp(x)+0.5)*16383)/16383-0.5`，round 为 nearest-even。统一 float64 观测，不额外经过历史 float32 roundtrip。STE 在未饱和区为恒等梯度，超出区间使用 clamp 的零梯度。
- 网络参数/特征默认为 float32；Q、样条计算、时间重采样、噪声组装与 preADC 为 float64。建议只 `model.to(device)`，不要把整个模型的精确缓冲区转换成 float32。
- 幅度有效性门 `A>1e-8 V`，源 `Q>1e-8 V ns`，归一化形状分母 `Q(v)>1e-6`。非有限或不满足门时抛出 `InvalidWaveformError`，包含批次行与相关值；训练器可留账并中止诊断，模型不丢行、重抽、改变源电荷或隐藏裁剪。

## 结构与初始化

U-Net 为 3 输入通道 `[u,Δu,t]`，E0–E4 通道 32/48/64/96/128，长度 480/240/120/60/30，解码 96/64/48/32。每块 k5 卷积、GN8、SiLU、128 维 FiLM、k5 卷积、GN8、SiLU 与残差连接；FiLM 初始为恒等。固定 `[1,4,6,4,1]/16` 低通后步长 2 下采样；上采样为 linear 插值再卷积；瓶颈额外膨胀率 1/2/4。归一化不使用批次统计。

CUDA 工程检查发现 `upsample_linear1d_backward_out_cuda` 在确定性模式下不受支持，因此保留 deterministic_algorithms，并用固定 2 倍 align_corners=False 的等价公式展开：even=.75*x+.25*previous，odd=.75*x+.25*following，端点复制，再交错 even/odd。两处固定 480→60 控制点平均用 reshape(B,C,60,8).mean(-1)，等价于不重叠的 adaptive average，避免另一处潜在的非确定 CUDA backward。CPU float32/float64 前向与随机 VJP 等价测试通过，容差分别为浮点舍入量级（upsample 3e-7/2e-15，average 2e-7/1e-15）。通道、长度、卷积和控制点结构不变；未放宽确定性要求。

平均电荷用独立的 16/32 通道 GN8 shape encoder，唯一输入是幅度无关的上述形状通道。固定单形状时，其样条参数不随 Q/A 改变。8 个等 logQ 宽度的单调 rational quadratic 区间，输出 8 个正高度、9 个正导数和 1 个纵轴偏移，共 18 参数；纵轴高度和导数为 `exp(2*tanh(raw))`，偏移为 `2*tanh(raw)`。零初始参数严格给出 `Qbar=Qsource`。`log_q_min/max` 是合法 SOURCE TRAIN 的**绝对 log(Q)**，非 log(Q/Qref)；区间外沿端点正斜率线性外推，不把合法父电荷裁进区间。区间坐标的 forward 数值保护保留末端正确单侧导数；不会裁 Q/Qbar。

共享响应 FIR 长 65、初始单位脉冲。允许有符号有效修正尾：`h=delta+.5*tanh(raw)/K`，再归一 DC；总绝对扰动小于 .5，DC 分母不会为零。电子学 65 点和局部 17 点 FIR 使用同一参数族但各自独立，并归一 L2 范数，避免滤波增益与噪声 sigma 的尺度混淆。FIR 是线性因果卷积。

平均 `log(a)=log(2)*tanh(mean_time(scalars))`，初始化为 0，对应平均时间伸缩范围 0.5–2。标量条件仅 `log(Q/Qref),log(A/Aref),d`，单已知工况 `d=0`（或显式固定 stats 值）。随机 logtime 使用同一 Gaussian 头；`a=exp(mean_logtime+eta_a-Var(eta_a)/2)`，所以未截断期望伸缩为 `exp(mean_logtime)`。Gaussian 随机量无硬截断；极端非有限生成显式失败。

完整联合头预测 55 个下三角元素，10 个坐标是 Q、time、8 个形状系数。基线低维 head 为 `3→32→55`，仅其前 3 输出在基线有效；联合阶段再加入 `128→32→55` U-Net 瓶颈修正。对角为 `limit_i*sigmoid(raw_i)`，初始标准差 Q=.04、time=.03、shape=.002；工程上限分别 .30/.20/.15。非对角为 `.5*limit_row*tanh(raw)/sqrt(row)`，允许正负相关。`eta=Lz`，`Qsig=Qbar*exp(eta_Q-.5*sum(L_Q²))`，保持未截断对数正态均值和正确协方差；不对 Qsig 做后置裁剪。shape 原候选初始化 .02 在 2048 个 rise4/decay13 人工窄脉冲响应中产生 109 个无效 Q(v)，因此降为 .002；没有使用 REAL/SOURCE 或留出数据选择这一设置。更新后的初始化通过人工窄脉冲的 2048 次随机形状组装检查，不构成 Gaussian 任意极端尾的全局有效性保证。

全分辨率平均残差为 `.20*tanh(mean_head)*gate`，初始为零；8 条随机基从 decoder 预测 60 个控制点，使用复制端控制点的 uniform cubic B-spline 矩阵展开，随后 gate 并按 480 点 RMS 归一为 1。随机基不标注为确定物理过程。局部 sigma 包络同样用 60 控制点 cubic 展开并 softplus，初始 bias=-5、最后层权重 std=.001，再乘 gate、`Aref*Qsig/Qref`，显式依赖事件形状和 Qsig。

## 边界、两条噪声与电荷分解

新增平均残差、形状基和局部噪声使用 C2 的 quintic 支持门：100–115 点平滑开启，456–479 点平滑归零。0:100 的新增项严格为零，且支持端点值、一阶、二阶导数为零；原始 SOURCE 主分支保留自身脉冲支持，不把源形状切掉。60 控制点的 cubic B-spline 自身为 C2。这个 gate 是分支定义的一部分，不是生成后在第 100 点做不连续硬置零。

时间伸缩可能查询 479 点之后；源右端使用 64 点 Hermite 延拓，匹配最后点值和一阶差分，在延拓末端值/一阶导数归零，后面为零。延拓是冻结的数值边界绑定，没有补入测得尾数据。滤波与重采样均不循环回绕。

电子学为 `offset_mean+offset_std*z0+(slope_mean+slope_std*z1)*t+sigma_e*(h_e*white)`，其 t 是 `2*sample/479-1`。stats 的 offset/slope 必须以该坐标解释；从 `[0,1]` 回归转换时，slope 减半，offset 加原 slope 的一半。平均 sigma 初始为合法 REAL TRAIN 事件 RMS 的算术均值；事件 logRMS spread 的工程范围为 0–.8，初始化夹在 `[1e-4,.7999]` 内；sigma 的 lognormal 均值修正确保初始期望 sigma 等于该 RMS。电子学不接收脉冲或 Q/A。非负 offset/slope std 的数值初始化 floor=1e-12，方便零统计量的可微参数化；mean slope 初始 0，baseline 阶段可学习。

电子学使用 64 个独立历史白噪声 + 480 有效样本，因果 valid 卷积得到整个有效窗口，基线前端不缺噪声历史。局部白噪声使用 16 个历史样本 + 480 点；历史包络为零。局部 filtered noise 再乘平滑 gate 后，用整个窗口的平滑参考函数 `p=gate*Gaussian(center180,width65)/Q(p)` 做 `n_s=n_s0-Q(n_s0)*p`。参考函数不在 Q 积分边界 400 处骤停，因此不引入矩形投影的边界阶跃。

只对 `v` 做单位电荷规范，`preADC=Qsig*v/Q(v)+n_s+n_e`。`Q(n_s)=0`，电子学不投影，有 `Q(preADC)=Qsig+Q(n_e)`。最终 ADC 可以改变电荷，返回 `adc_charge_change` 和饱和比例。没有最终 Q 投影。

## 阶段与三组对照

|阶段|共同低维均值/滤波|电子学|U-Net均值|Q/time随机|8基/局部涨落|
|---|---|---|---|---|---|
|baseline|训练|训练|关闭/冻结|full/simple 前2坐标；det关闭|关闭/冻结|
|mean_response|训练|冻结|full/det训练；simple无U-Net|关闭/冻结|关闭/冻结|
|joint_stochastic|训练|冻结|full/det训练|full全部10维；simple前2；det关闭|仅full训练|
|low_learning_rate|同joint|冻结|同joint|同joint|同joint|

共同低维部分为幅度无关 charge encoder、单调 charge head、共享 FIR、平均 time MLP。simple 有自己的 2×2 Cholesky 3 输出 head，允许相关 Q/time 展宽，并具有同样 electronics；没有 U-Net 或局部信号分支。deterministic 有 U-Net 平均形状和平均 Q/time，仍保留随机电子学，不具生理随机头。full 在联合阶段开启全部提案分支。低学习率值由训练器绑定，模型只切换冻结状态。

`load_shared_baseline(other_model_or_state)` 有意复制 shape_encoder、charge_head、response_filter、mean_time、electronics；随机比较臂另复制低维 Cholesky 的前三个 triangular 输出。full→simple 因输出大小不同而显式取共同前三行；不假装 `strict=False` 能处理参数尺寸冲突。未做 baseline 优化的 U-Net/mean/basis/local 不由本方法复制，可用相同种子初始化 full/deterministic 的对应 U-Net，或另外复制匹配键。默认创建 stage=joint_stochastic，正式训练必须显式设置每一阶段。

`regular` 包含平均残差幅度/一阶平滑、完整模型基二阶平滑、内部权重 .01 的基间相关、局部包络幅度，以及 ADC 区间之外的 squared excess（按 1 V 范围无量纲化）。这使饱和样本即使 STE 被 mask，仍有修正梯度；外部 N01 regular 权重仍由训练器设置。`trend=0` 是因为固定形状电荷单调已由参数化保证，不额外伪造其他有物理依据的趋势约束。

## 已运行的验证与边界

2026-10-12，本地 CPU、线程 2，使用 E 盘现有 Python 环境；临时目录 E 盘，禁 bytecode。执行 `python -B -m unittest discover -s tests -p test_model.py -v`。14 项测试通过，含所有 3 变体×4 阶段、trainable 参数有限梯度、电子学保留电荷、preADC 恒等式、ADC 与 STE、单调样条/外推、55 参数 Gaussian 协方差/对数正态均值与方差、显式重放与批次置换、FIR 因果/valid history、平滑支持/基归一、锚点重采样、人工窄脉冲初始化及源与生成无效行显式失败。这些模型单元测试只使用人工波形和随机变量，没有解码 REAL/SOURCE 文件或执行正式科学训练。另以 16 对人工域样本完成 Engine 的 baseline/mean_response/joint_stochastic 各一步 CPU 接口检查：显式重放均一致、目标与梯度有限、联合阶段的人口更新完成；人工拟合与优化状态均丢弃，不是实际 TRAIN 结果或物理有效性证据。
