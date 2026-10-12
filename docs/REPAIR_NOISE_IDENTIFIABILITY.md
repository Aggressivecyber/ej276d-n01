# Repair_002 固定噪声终点：协方差机制与可辨识边界

本分析使用 5000-update 科学噪声 checkpoint、冻结 TRAIN 尺度和全部 21047 条合法 REAL TRAIN 的前 100 点；422 个原组，每组 48–50 行。没有新噪声/响应训练，没有修改模型、损失或 gate，没有读取容器中的 VAL 成员。VAL/generated 数字仅复用已保存 gate 的描述性协方差。response training updates 仍为 0，原 noise gate 的停止决定保留。

## 关键结论

1. 当前 ElectronicsNoiseV2 是平稳 FIR/logscale 噪声加独立事件常数 offset，其总体 centered OLS mean–slope covariance 严格为 0，与 TRAIN 的 −0.678247 相关不兼容。同结构增加优化 steps 不能改变此限制。
2. 保持冻结 FIR 和尺度，用 latent covariance = observed OLS covariance − A K Aᵀ 得到非半正定矩阵，最小特征值 −2.16932e−9 V²、形式上所需相关 −1.21546。不能直接采样该联合漂移；换坐标或复制 OLS SD 都不能解决。
3. 422 组的 512 次 bootstrap 中 residual covariance 0/512 满足 PSD。约 98.05% 的负 cross covariance 来自组内，不能仅解释为组间均值差异。
4. 降低同形 K 到 PSD 代数边界，可按构造恢复两个 OLS 统计，但完整 100×100 covariance 仍有约 34.9% 误差。这是可行性界，不是已选修复或成功结果。
5. 100 点基线不能证明漂移按同一随机直线延续到 480 点。强结构先验下可条件性识别二阶分解，但先验、完整分布和外推持续性尚未建立。

## 1. 平稳过程的解析限制

以前 100 点中点为原点：

~~~text
t_i = linspace(-1,1,100)
a_i = 1/100
b_i = t_i / sum(t_i²)
m = a y
s = b y
A = [a; b]
~~~

J 为时间反射矩阵。均值权重为偶、斜率权重为奇：aJ=a，bJ=−b。若 J K J=K，则：

~~~text
Cov(m,s) = a K bᵀ
         = a J K J bᵀ
         = −a K bᵀ = 0.
~~~

实标量平稳过程的 covariance 是对称 Toeplitz，满足此反射条件。当前 FIR 使用真实白噪声历史并裁取 valid 段，窗口起点没有零填充。独立 event logscale 只乘 E[sigma²]，独立常数 offset 只加 Var(offset)·11ᵀ，仍反射对称。该二阶结论不要求 FIR 系数本身反射对称。

这是总体约束，不要求有限样本 sample covariance 恰好为零。共同的逐坐标 ADC clipping/quantization 保持平稳性，不能让 offset-only 模型的总体 centered mean–slope covariance 变为目标负值。以下 K 是 pre-ADC ensemble covariance；不能直接当作已校准的 post-ADC covariance。

冻结 FIR 的理论 cross covariance 约 1.44e−24 V²，加入独立 offset 后约 4.72e−24。保存 gate 的 generated 8192 行相关为 −0.002233，TRAIN −0.678247，已有 VAL −0.656792。

## 2. 冻结 K 的协方差核算

合法 TRAIN 先使用 n01.data.canonical_real 的共同 ADC 格点，无行剔除。单位 V²：

~~~text
C_TRAIN(mean,slope) =
[[ 2.45006362e-8, -1.43751437e-8],
 [-1.43751437e-8,  1.83345880e-8]]

A K_FIR Aᵀ =
[[ 2.14872382e-9, 约0 ],
 [ 约0,           1.20767314e-8]]

S_latent = C_TRAIN − A K_FIR Aᵀ =
[[ 2.23519124e-8, -1.43751437e-8],
 [-1.43751437e-8,  6.25785659e-9]]
~~~

K_FIR 只含 stationary FIR/logscale，排除常数 offset；否则再将 offset 放入 latent b0 会重复计数。冻结 E[sigma²]=4.00410730e−7 V²、offset SD=0.120461 mV、logscale SD=0.0380093。

S_latent 特征值为 −2.16932005e−9 和 3.07790890e−8 V²。其对角为正，但 cross covariance 超过 PSD 的 Cauchy–Schwarz 上限，形式上相关 −1.215464，不能视为合法协方差。

固定 mean 噪声贡献时，所需 slope latent variance 至少 9.24505937e−9 V²，当前相减后只有 6.25785659e−9。当前 K 的 slope 贡献 1.20767314e−8 V²，允许上限约 9.08952866e−9。

若全部同形 K 乘 alpha，PSD 代数上限 alpha=0.76926004，振幅乘数 sqrt(alpha)=0.87707471。边界 latent covariance 为秩 1、相关 −1。该边界按构造恢复 OLS covariance，但完整 100×100 covariance 的相对 Frobenius 误差 0.348832；按冻结 TRAIN 每坐标尺度标准化后为 0.355812。这不是自动降低 RMS 的修复建议。

裁掉负特征值会改变目标 covariance，不能声称减法精确成立或物理漂移已经识别。改变 reference time 也不能修复非 PSD。

## 3. 组、不确定度和完整二阶族

重采样原组并保留组内全部事件，seed=2026101251，512 次：

| 量 | 结果 |
| --- | --- |
| TRAIN centered mean–slope correlation | −0.678247 |
| 组 bootstrap 95% 区间 | [−0.691516, −0.664684] |
| 冻结 K latent 最小特征值 95% 区间，V² | [−2.472919e−9, −1.871312e−9] |
| residual covariance 满足 PSD | 0/512 |
| 最大同形 K 方差乘数 95% 区间 | [0.737727, 0.799967] |
| cross covariance 的组内贡献比例 | 0.980515 |
| 组内 mean–slope correlation | −0.677802 |

区间条件于冻结 K，未重训，不包含 K 拟合不确定度或独立采集间不确定度。422 个原组不等于 422 次独立硬件采集；单次、曾查看的开发采集不能支持硬件等效性。

TRAIN covariance 与其反射的相对误差 0.510936。每坐标 population SD 的前 20 点平均 0.679673 mV、后 20 点平均 0.636363 mV。窗口统计不完全平稳，但仅凭此不能归因于硬件漂移、触发选择、预处理或某个器件。

另外只作 TRAIN 二阶线性分析：

~~~text
C ≈ X S Xᵀ + Toeplitz(k_0,...,k_64,0,...)
X = [1,t]
~~~

68 参数、10000 个 entry；设计矩阵秩 68、条件数 37.56。无约束最小二乘相对误差 0.179779，但 S 的对角为 −1.35848e−8、−3.38545e−8 V²，两特征值均负。因此不是有效 drift 模型。K 的有限 100 点矩阵和 4097 点谱网格未出现负值，但网格阳性不是全频率证明；没有谱因子化、部署或训练候选。

这并不证明所有 PSD 约束近似候选都不可能达到未来注册门槛。它说明不能把无约束负方差解作为物理机制。

## 4. K 估计、减法与可辨识条件

在 y=Xb+eta、A X=I、b 与 eta 独立、K=Cov(eta) 已知时：

~~~text
Cov(Ay) = Cov(b) + A K Aᵀ.
~~~

公式成立仍须检查 residual PSD 和完整 joint/residual 分布。drift–noise 相关项、非线性趋势、非平稳过程、ADC/触发选择都可能破坏其识别假设。仅凭 OLS 数字不能区分这些机制。

两种常见 K 估计并不自动识别所需 K：

- 每事件先 OLS 去趋势，得到 R C Rᵀ。A R=0，所以 A(R C Rᵀ)Aᵀ 恒为 0；它删除了恰好需要校正的 noise mean/slope 模式。相减后仍保留整个观测 OLS covariance，不能证明那就是 latent drift。
- 将 covariance 平稳化为 pooled autocorrelation，可构成 PSD Toeplitz 估计，但 offset/drift 混在其中，且有有限窗口 lag taper。脚本仅将其作为透明描述性比较，不视为识别 FIR K。

复制观测 OLS SD 到 latent 后再加 stationary noise，会把 A K Aᵀ 的采样方差重复计数。人工检查验证了这种错误。

在“严格 FIR65、最大 covariance lag64、独立精确线性 drift、同质总体、可忽略 ADC 偏差”等强先验下，本次 100 点二阶设计满秩，可以条件性求解；不能笼统声称所有结构假设下都不可辨识。但这些假设不能由 100 点数据自动证明，当前无约束最优分解也不在 latent PSD 域中。FIR 相位、event scale 分布与高阶结构同样不是仅凭 covariance 完全识别。

## 5. 坐标变换与 480 点外推

完整窗口 T=2*i/479−1，基线中点 u=2*i/99−1，u=(479/99)T+380/99：

~~~text
[intercept_at_T0]   [1  380/99] [mean_at_baseline_midpoint]
[slope_on_T]     = [0  479/99] [slope_on_u]
~~~

Covariance 为 M S Mᵀ。可逆坐标变换保持 PSD，不能消除上述负特征值。global intercept 不是 baseline mean；即使 centered mean–slope 不相关，外推 intercept 也可因代数变换与 slope 相关。原模型若把 global intercept/slope 独立建模，也可在 baseline mean 中产生位置相关。必须先写明时间原点再讨论“独立”。

前 100 点不能证明同一随机直线延伸至 480 点。任何在前 100 点恒为 0、后续变化的过程，都对当前观测不可见。exact-global-linear 假设下可以外推，但 slope 放大 479/99≈4.838，持续性须另有证据。

缺少覆盖至少 480 点的 noise-only/长 pretrigger 记录、触发与对齐选择机制、低频 K 与漂移持续时间的先验/测量、独立采集及组效应证据。当前支持“平稳 offset-only 候选有结构限制”，不支持“物理漂移已识别”或直接启动主波形训练。

## 6. 人工检查、运行和回执

脚本 scripts/analyze_noise_repair_covariance.py；CPU numerical threads=2，seed=2026101251，全部输出 E 盘，PYTHONDONTWRITEBYTECODE=1。self-test 六项通过：

1. Toeplitz reflection 使 centered OLS cross covariance 为零。
2. 已知 PSD joint latent 和 K，用减法精确恢复。
3. OLS residual projection 无法识别被删除的 K 模式。
4. centered/global congruence 变换保持 PSD 并可恢复。
5. 复制 OLS variance 重复计数 noise variance。
6. 检测 nonPSD subtraction。

完整 TRAIN 分析 native exit0（约 3.6 秒），含上述人工检查；没有优化器更新。输出位于 runs/repair_002_20261012/identifiability：

- IDENTIFIABILITY_REPORT.json：解析数值、假设和证据边界。
- COVARIANCE_MATRICES.npz：完整 covariance/K/OLS 数值和 TRAIN IDs/groups；没有复制完整原波形。
- AUDIT_COMPLETE.json：JSON/NPZ/script SHA 和 no-training/no-VAL 回执。

脚本核对容器、身份、checkpoint 与 gate SHA，以及保存的 TRAIN fit rows/scale。重跑：

~~~powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:TEMP='E:\GAN\runtime_tmp'
$env:TMP='E:\GAN\runtime_tmp'
$auditArgs=@(
  'E:\GAN\research\N01\scripts\analyze_noise_repair_covariance.py',
  '--checkpoint','E:\GAN\research\N01\runs\repair_002_20261012\runs\science\noise\final.pt',
  '--container','E:\GAN\research\sim2real_two_papers_20261008_0305\work\R23_ONLINE_STANDARDIZED_CRITIC_001\TRAIN_PRIVATE.npz',
  '--identities','E:\GAN\research\N01\data\PARENT_IDENTITIES.npz',
  '--gate','E:\GAN\research\N01\runs\repair_002_20261012\runs\science\noise_validation\NOISE_GATE.json',
  '--output-dir','E:\GAN\research\N01\runs\repair_002_20261012\identifiability',
  '--seed','2026101251','--bootstrap-resamples','512'
)
& 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe' @auditArgs
~~~

同脚本 --self-test 可单独执行人工检查，不读真实数据。