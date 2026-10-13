# Window008 resume01 冻结终点补充诊断

本模块实现 `USER_PROPOSAL.md` 第九节与 `docs/WINDOW008_REPORT_CONTENT.md` 第五节的四项补充诊断。实现与人工检查、实际终点评价、物理响应成立分别记录。科学状态始终为 `response_match_not_established`，科学通过阈值为 null；G、D、噪声、人口与 C2ST 更新均为 0。独立 C2ST 仍未执行。

源文件为 `src/n01/diagnose_window008_resume01.py`，预登记策略为 `window008_resume01_diagnostics_policy.json`。三臂采用主评价 adapter 已核验的混合终点：deterministic 为原 model008，full/simple 为恢复后的 model008_fix01；不转换或覆盖权重。实际训练 native0、端点 hash 与恢复链由 `evaluate_window008_resume01._freeze` 校验。

## 冻结与执行顺序

根流程在首次 DEV 数值解码前冻结主评价源码清单，其中包括本 addon 源码、策略、检查脚本和本说明。另建 `WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json`，schema 为 `n01/window008-resume01-diagnostics-source-manifest/v1`，记录 `before_DEV_numeric_decode=true`、`before_scientific_model_forward=true`、策略 SHA256 及相对路径到 SHA256 的 `files` 映射。该清单绑定主评价清单；主清单无需反向绑定 addon 清单，从而避免 hash 循环。

科学入口先核验 addon 源码/策略清单，再调用主 adapter 的 `_freeze` 保存原格式 `FROZEN_INPUTS.json`，核验主评价 primary 与 replay01 的 native exit0、主评价产物 SHA256 和相同模型状态，保存 `ADDON_PREREGISTERED_INPUTS.json` 与 `ADDON_FROZEN_INPUTS.json`。此后才读取 TRAIN 身份映射和 DEV 原始数值。DEV SOURCE/REAL 身份、REAL 原软件组须与 TRAIN 不交叠并与已准备身份映射完全一致。主产物只作为固定人口敏感性的已生成输入。

每组诊断后复核模型和噪声状态；结束时复核所有冻结源与人口输入文件。异常留下 `FAILED.json`、实际部分调用计数及已有产物。所有输出保存在新目录，既有文件不覆盖。Windows 输出只能位于 E/D；远程输出位于已登记项目。临时目录同样使用 E/D 或已登记远程目录。

## 四项统计

1. **同一 SOURCE 的 128×64 重复。** 用原 `repeat_selection` 种子在原准备顺序的 512 个 DEV 身份中无放回选 128 个父事件，固定 parent-major 顺序。64 次显式随机量在三臂间共享，三个电子学假设共享005前缀、S和 extension epsilon。保存全部行及其 Q、tail、guarded/raw PSD、qbar、qsig、time_scale、载体峰、crop、ADC/边界/guard flags 与同窗 Q/tail 分解。每个父事件保存完整 `256×256` 波形样本协方差、波形均值及 Var(Q|x)、Var(PSD|x)、Cov(Q,PSD|x)，均使用 ddof1。标量协方差对每一特征对单独保存共同定义次数；Q/guarded PSD 使用全部64次。raw PSD 的缺失不删除父事件。
2. **固定 latent、保持 Q 的形状替换。** 使用上述选择顺序的前32个父事件。在其余全部 DEV SOURCE 中按原 `abs(log(Qj/Qi))` 选 donor，并列按原身份升序破平。基线中心化 donor384 缩放到原 Qi，不裁切电压，保持峰锚点126和 `[110,366)` canonical Q。原形状与 donor 同 Q 形状各使用相同8次响应。保存实际 Q 间隔、原始归一形状差、每 draw 成对变化、均值差、样本标准差及 Monte Carlo SE，明确共同定义次数。实际干预很小时照常保留结果。
3. **同形状 Q 趋势。** 同一32个原形状缩放到已冻结 SOURCE TRAIN Q8 九边界数组的索引2/4/6，即25%/50%/75%节点。节点与 B 共用一份32×8显式 intervention bank。保存 qbar、qsig、最终 Q/tail/PSD、时间量及均值波形节点曲线和相邻成对差异。qbar 非负相邻差的结构检查单列；带电子学/ADC 的观测 Q 趋势不当作同一个结论。
4. **固定人口正权重敏感性。** 直接读取主评价完整 TRAIN/DEV 四 draw 输出。由原 SOURCE Q8 bin b 计算 `z=(2b−7)/7`，采用 `1`、`1+0.5z`、`1−0.5z` 三套正权重，分别使父权重和等于父数；四 draw 同权重。报告范围、ESS、各 bin 权重和、加权 Q/tail/PSD/raw PSD 人口与条件均值/方差、三假设敏感性及 full 对照差异。缺失 raw PSD 的行数和权重质量均保留；其均值属于定义子集。

人口波形合成使用主产物固定 SOURCE-bin 的均值 `μb`、人口标准差 `σb` 与行数 `nb`。权重在 bin 内恒定，令 `mb=wb nb`，则 `μ=Σmb μb/Σmb`，`Var=Σmb[σb²+(μb−μ)²]/Σmb`。该式在实数算术中精确，浮点合成允许舍入误差。uniform 合成须与主评价全人口曲线一致。SOURCE 条件波形矩在同 bin 内不变，报告变更后的权重质量和支持度。主产物只有 marginal 分箱曲线，缺少 SOURCE-bin × output-Q-bin 的联合波形矩，因而**不提供 output-Q 条件加权波形统计**；该限制写入 JSON，新增前向为 0。

## 时间与噪声口径

输出选中峰位于256波形索引16。以此峰的10%/50%/90%为阈值，从峰向左右寻找最近穿越并做线性插值；rise10–90 为左90减左10，FWHM 为右50减左50。峰非正或缺少穿越时返回 NaN，并保存定义 flags 和次数。载体峰与模型 time_scale 同时保存。归一波形仅在该行观测 Q 大于冻结 TRAIN `q_floor` 时定义，其余原始行仍保留；归一均值明确是定义子集。

噪声使用原始 ADC100 基线，沿用005 `noise_validation.baseline_statistics` 的事件去均值、方差 floor `(1/16383)^2/12` 和 FFT 区间 `[1,4)`、`[4,9)`、`[9,17)`、`[17,33)`、`[33,51)`。额外明确扩展 ACF 到 lag0–99；各事件的 ACF 单独计算后汇总。保存原 RMS、floor RMS、ACF、频带份额和 S；三假设的384点/后284点噪声统计单列。此口径与用于主分布距离的 losses008 噪声特征频带口径分别标明。

Q/tail 的 signal、electronics、ADC residual 分解始终使用最终观测峰所选的同一个 crop 与各分量原100均值。保存六分量的样本协方差及方差合成残差。它说明观测分解关系，不唯一识别物理噪声源。

## 实际计数、工程重放与资源

| 组 | 信号行前向 | 臂×假设观测行 | 005 前向行 |
|---|---:|---:|---:|
| 128×64 重复 | 24,576 | 73,728 | 8,192 |
| 32×8 两种形状 | 1,536 | 4,608 | 256 |
| 32×8 三个 Q 节点 | 2,304 | 6,912 | 256 |
| 三项主体合计 | 28,416 | 85,248 | 8,704 |
| 身份／顺序工程重放 | 1,536 | 4,608 | 512 |
| 含重放总计 | 29,952 | 89,856 | 9,216 |
| 人口敏感性 | 0 | 0 | 0 |

每次实际成功调用后计数，分别核验每组、每臂、每臂×假设与总数；预算不能冒充执行计数。B/C 共用一次显式 bank；在 C 中用同一 bank 重算005而不推进随机流，其实际005行数单独计入。

身份重放只改保存的32个身份文本，输入数值与 latent 不变。顺序重放逆转父存储顺序，latent 仍附着原父事件，结束后按原身份对齐数值。两次均重放 B 的原形状32×8，不重新抽随机数。连续值按已登记 rtol1e−6、波形 atol1e−8 V、标量 atol1e−6 比较，undefined masks、离散计数和 flags 必须完全一致；记录最大误差与 bitwise 状态。它是接口工程检查，输入内容依赖由 B 检查。

资源冻结为 GPU10GiB、进程树RSS8GiB、输出3072MiB、墙钟21600秒、CPU12，生成 chunk 最多128行。协方差按臂/假设流式写盘，每次只计算一个64MiB条件协方差块；不保存所有384中间模型张量。独立监督器负责驱动/上下文在内的GPU与进程树限额，PyTorch allocator 限额仅作配套约束。实际峰值与耗时以 native 人工测试或科学运行监督回执为准。

## 运行接口与产物

```powershell
python -B -m n01.diagnose_window008_resume01 `
  --out runs/science --config window008_config.json `
  --resume-config window008_resume01_config.json `
  --dataset <frozen_DATASET008.npz> --statistics <frozen_FIT_STATISTICS008.json> `
  --raw-container <bound_PRIVATE.npz> --identities <bound_identities.npz> `
  --source-manifest WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json `
  --evaluation-policy window008_resume01_evaluation_policy.json `
  --policy window008_resume01_diagnostics_policy.json `
  --addon-source-manifest WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json `
  --primary-artifacts runs/evaluation_jobs/primary/artifacts `
  --primary-native-receipt runs/evaluation_jobs/primary/supervisor/exit_receipt.json `
  --replay-native-receipt runs/evaluation_jobs/replay01/supervisor/exit_receipt.json `
  --destination runs/diagnostic_jobs/primary/artifacts --device cuda:0
```

| 产物 | 内容 |
|---|---|
| `SELECTIONS.npz` | 128父身份与原载体、32父 donor/Q差/形状差、三个冻结Q节点与缩放 |
| `repeat_latents.npz`、`intervention_latents.npz`、`LATENT_BANKS.json` | 全部显式随机量、stream起止状态、固定chunk hash |
| `REPEAT_{arm}_{hyp}.npz`（9份） | 所有64次原标量/flags/256波形、完整逐父样本协方差、分量协方差、定义次数和归一子集曲线 |
| `SHAPE_{arm}_{hyp}.npz`、`Q_TREND_{arm}_{hyp}.npz`（各9份） | 所有变体/节点每draw标量、256波形、成对差/均值/SE/定义次数 |
| `REPEAT_COMMON.npz`、`SHAPE_COMMON.npz`、`Q_TREND_COMMON.npz` | 原100 ADC基线、RMS/ACF/频带/S、三假设噪声条件均值和标准差 |
| `POPULATION_TRAIN/DEV.json/.npz` | 正权重、ESS、全人口/条件标量和精确波形合成及其限制 |
| `REPLAY_IDENTITY_ORDER.json` | 实际两次身份/顺序工程重放及各项误差 |
| `ACTUAL_COUNTERS.json`、`GENERATION.json`、`DIAGNOSTIC_STATISTICS.json` | 实际调用数、chunk输出 hash、工程/统计/undefined 汇总 |
| `SUMMARY.json` | 完成状态、科学边界、全文件 SHA256 |

人工检查入口为 `scripts/check_window008_resume01_diagnostics.py --out <fresh_path> --device cpu --full-shape`。可加 `--architecture-128`，对三个实际类的**新初始化**做每臂128行人工波形前向，不读取训练 checkpoint 或 SOURCE/REAL/DEV科学数值。full-shape 使用最小冻结接口模型检验完整调用计数、CPU协方差与存储/重读；实际类128行基准的人工前向384行单独登记。CUDA目标需在源码冻结后运行同一检查脚本，allocator早期设置为10GiB。人工性能不是实际训练终点的响应结果，也不代替科学运行监督回执。

本机 tiny-helper CPU native0 回执保存于 `receipts/window008_resume01_diagnostics_20261014/`。完整资源测试因当前本机可用RAM约2.4GiB未启动，保留10GiB预检失败；target CUDA人工测试由根流程在源码冻结后执行。该目录也保留可选psutil缺失的早期导入失败和人工接口fixture键名修正前失败；没有安装或改变共享环境。tiny-helper范围只证明算法语义与小形状的实际配对生成/同窗分解/顺序重放，不证明完整形状或GPU资源上限。

## 结论边界

这四项支持报告模型的逐输入有限抽样方差、同 latent 输入形状干预、同形状Q节点趋势和温和源Q人口倾斜敏感性。电子学扩展仍是三种假设。无同一物理事件重复REAL真值、无配对响应真值、无独立采集；人工软件组或低域差异不能补足这些证据。未核验 h 元数据物理含义时，不据此报告粒子组成重配或跨源条件复用。
