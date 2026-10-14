# N01 012：新头梯度与裁窗机制诊断

用户于 2026-10-15（Asia/Singapore）通知服务器启动并要求继续推进。本协议单独登记 `mechanism012_20261015`；以 `mechanism012_policy.json`、新 source/bindings manifests 和原生回执为执行依据。服务器重启预检已连通、N01 进程与 GPU 计算任务为空、作业锁可用；011 已完整完成 500 次配对重加权并独立读回。

011 的固定 TRAIN 库配对差值分位范围上界小于 0，满足已登记的方向稳定规则。这支持进入局部机制诊断；科学状态仍为 `response_match_not_established`。012 检查已训练新头的局部梯度和观察算子作用，不更新模型，不生成新训练候选。

## 固定输入、身份与旧状态

只使用 010 的 θ2000、原 512 个 SOURCE TRAIN 父事件、每父 4 个已存 draw、2048 个原 REAL TRAIN 参考行以及三假设 `fast/slow/common_level`。父事件顺序、shape latent、005 latent、extension epsilon、选择索引和投影保持原库不变。θ0/θ1000 薄 CP 只可作元数据和参数检查，不做它们的前向。全合法 TRAIN 元数据计数 6593/21047 保留，固定诊断库的统计范围另列。

prepared NPZ 仅解码 policy 列出的 9 个 TRAIN 字段；不读取 DEV/VAL 数值，不运行 sampler，不重拟合统计、分箱、阈值或噪声。哈希读取文件字节与解码波形是不同操作。新入口先用标准库核 policy、全部 source、bindings、外部输入和路径归属，再导入数值模块；不得调用旧 `Shape010Engine`、训练入口或旧身份固定 gate。

原 A 严格恢复 `N01Response008` deterministic 及 G19000/low_learning_rate 元数据；005 严格恢复 schema5、update5000 和 `ObservedBaselineNoiseV3`。先完成 baseline 的 stage、冻结、eval 和设备恢复，再构造原 `FrozenDeterministicVoltageShape010`，只装入 final 的新头状态，保留原初始化 seed2026101420/voltage_gain0.05。不能对整个 wrapper 再调用 `.to(device)`。仅 `basis_head.weight/bias` 共 1288 参数允许 autograd；旧 A、005、extension 的参数与 buffers 冻结。记录所有模型 tensor digest，结束时新头值也必须不变。

NN 使用原 chunk128：A、005 和 θ2000 新头各 2048 行，三假设共用同一次新头信号与 005 prefix。禁止额外 A、005、head 或 latent draw。先构建共享 S，观察窗口必须从这个 S 或其切片产生，使 S 是损失图的实际祖先。保持原数值环境、dtype、TF32 与确定性设置；原生回执记录实际版本和配置。

用保存 A 的 pure signal/parent/peak/crop 及保存 final 的 hard waveform/Q/peak/crop 核对当前恢复。原 point 重算和 helper 值绝对容差为 1e-7，peak/crop/身份必须一致；固定评价 seed2026101430、kind=evaluation、update=null 的 projection fingerprint 为 `27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482`。不因复现失败换库、改精度或挑假设。

## 新头的逐项梯度

每假设调用原 `N01Loss008`、stage=`joint_stochastic`、comparison weights=null、原 TRAIN statistics 和同一固定投影。定义

`L_w = mean_h(W_h)`；`L_q = 2 mean_h(QPSD_h)`；`L_r = .01 (RMS + curvature + mean_h(saturation_h))`；`L = L_w + L_q + L_r`。

RMS 为全部 2048 行随机形状电压平方均值除以 a_ref²；curvature 为其二阶差分平方均值除以 a_ref²。chunk 标量按实际行数汇总，不能误平均不等长 chunk。排除 frozen-A regular、noise、trend、adversarial 项。

以 weight.flatten 后接 bias.flatten 的顺序保存五个已加权分量 `g_wave/g_QPSD/g_RMS/g_curvature/g_saturation`，独立查询 total 并核对向量和。记录 weight/bias、8 个 basis 的范数、pairwise cosine、total 投影及抵消；零范数 cosine 写 null 和原因。求和误差 `norm(g_total-sum(g_i))/max(sum(norm(g_i)),tiny)` 的工程容差 1e-5，同时报告绝对误差。clip5 仅用于记录假设裁剪系数，不实际裁剪。

正式 AD 总预算为 **9 次 scalar 查询 + 2 次 masked endpoint VJP = 11**。wave/QPSD 两次查询同次返回参数、三假设窗口 endpoint 和共享 S 梯度。另一次查询核实际 frozen-membership QPSD 的值和参数梯度；anchor wave、anchor frozen-QPSD 各一次。wave/QPSD 的 changed-mask VJP 各一次。用 `autograd.grad`，不调用 backward，不累计参数 `.grad`，不创建 optimizer。

S-reference 梯度验证复用 wave/QPSD 两次正式查询：用固定 gather/scatter、prefix 均值和原 clamp 的解析 adjoint 与已得到的 shared-S 梯度核对；不增加第 12 次 AD。人工 fixture 先核其 Jacobian，包括 rail 的原生 inclusive subgradient 约定和 prefix 中间梯度可能非零的情况。

## 同一 carrier 上的实际与 A-anchor 裁窗

对同一 `S+n_h` 做 actual hard/STE，保存 ADC、baseline、peak/window/Q 和 loss 的前向一致性。round 的硬局部导数和 STE 的代理导数分别解释；first-argmax/gather index 不求梯度。相同前向值不保证代理捕获裁窗跳变。

在同一当前 ADC 与同一 prefix baseline 上，分别用 actual pθ 和保存 A 的 pA gather。记录 Q/tail/PSD、wave/native-QPSD/frozen-QPSD 差、peak margin/ties/搜索边界/rail/非正 Q/guard 等全部标签。两种 crop 各核 `Q_signal + Q_noise + Q_ADC_delta = Q_total`，包含 prefix baseline 对每个分量的贡献。

fake/REAL 条件 membership 和 support 在 θ2000 实际 hard 参考点冻结。新 helper 复用原 `_qpsd/_z/empirical_wasserstein_1d/context`、0.5 权重、八箱均值和 support16 fallback，保留全域项；不修改旧 loss/boundaries，不猴补 bucketize。实际点的值和 1288 参数梯度先与原 native QPSD 核对。anchor 同时报告 native 与 fixed-membership 值及分箱/support 迁移；固定分箱的差值才用于隔离裁窗对照。

用已保存 endpoint adjoint 计算 `inner(e, Y_anchor-Y_actual)`，这是局部一阶 signed influence，不是逐行可加 loss。报告其总和、真实损失差和线性化余项。按 actual crop changed 的 detached mask 作两次 VJP，unchanged 用 total-changed 得到；核两者向量重构。范数不能相加成百分比贡献，后验分组不解释为因果标签。

SOURCE 描述分层必须匹配原 A 的真实输入：先对 carrier384 的 float64 值减去前100点均值，再调用原 `canonical_charge008`，即 centered carrier 的固定 `[110:366]` 256 点和；随后用原 `_extended_log_charge(q,q_reference,q_floor)` 和冻结 TRAIN loss 的 `charge_bin_boundaries` bucketize。此 q_source 与动态裁窗的 observed Q 分别记录，不重拟合 source quantile，不用其分层重加权 loss。空箱和全部 2048 draw/512 parent 的困难事件均保留。

## 六个 signed carrier probe

固定 `v=detach(random_shape_pre_V)/global_max_abs(random_shape_pre_V)`，核 prefix100 与 canonical charge 为零；δ 为 ±0.1、±1、±10 个原 ADC LSB（1/16383 V）。扰动 carrier，不做 θ 参数有限差分，不生成物理候选，不增加 NN 前向。

三个路径共享固定 membership/projection：S 为参考硬 ADC 加 clamp 增量的代理，固定 actual crop；H_fixed 为扰动的硬 ADC、固定 actual crop；H_moving 复用同一次硬 ADC 并重新 first-argmax。各路径按自己的 prefix 求 baseline。报告所有步长的 `mean_h(W+2 QPSD_fixed)` 中心差分、`inner(dL/dS,v)`、ADC cell/crop/peak/parent 变化、guard/排序或归一化边界；不挑最好步长。固定/moving 共用硬 ADC 的差用于定位裁窗非连续作用，native 分箱迁移另列。

全零方向时保存 direction_defined=false 和原因，保留六个 probe 条目的 undefined/null 值，不换方向、不重算同一 carrier 占预算。actual hard+STE 仍为 12288 ADC 行；probe 额外 ADC 为0。非零方向的六次 probe 为36864 行，总最大 **49152** 行。ADC 是上限；完整 NN 与 11 个正式 AD 查询仍须完成。

## 操作记账、资源和结束规则

最大 CP 读取5（A、005、三个薄 head CP），NN各2048、θ0/1000前向0；optimizer/sampler创建和step、G/D/noise/pop/C2ST更新、DEV数值均0。输出不写 raw REAL 波形。每次 CP/NN/AD/ADC 先持久化 entered 范围，成功后 fsync completed 再更新 status；无法确认的 in-flight 完成数写 unknown，不推为0，不自动重启。首次实际证明使用首个正式 global AD 查询，且包含已完成第一128行共同 family；可以在全部2048前向后写，不能另加 pilot。

资源上限 CPU12、GPU10GiB、作业树 RSS8GiB、wall900秒、输出128MiB，监督采样10秒、TERM grace20秒；开始前 RAM≥10GiB、磁盘≥4GiB、GPU无计算任务，独占 N01 作业锁。监督仅管理本作业树。所有本地代码、缓存、日志、回执和备份存 E/D；远端沿用已核验项目环境，不修改共享环境或其他任务。

source package 为69份：旧53源码、4份原 protected metadata、6份006/008 provenance metadata原字节副本，以及4份012新脚本和policy/protocol。外部输入及011旧63源码、旧010/011产物在运行前后 hashguard。冻结后原协议、源、权重、统计、银行和旧负结果均不覆盖；E源包与D小源包备份 SHA 一致。

`COMPLETE`、原生退出0、数值读回和科学验收分别报告。完成必须包含完整 operation journals、11 AD、规定前向、全部 rows/parents、未变 source/input/model/head digest 和产物库存。任何复现、预算、有限性、求和或 hash 检查失败保留原始错误和已发生计数。cosine 或方向 undefined 有独立原因，不改成伪0。机制输出没有科学通过阈值，也没有自动重训或启用 GAN 的规则。

根流程核对实际正确启动后，更新同一个 n01 heartbeat 为预计结束后的单次苏醒，期间休息、不持续监控。苏醒后回收并独立读回结果，再据实际证据登记下一项。旧011完整重算不重跑。未来 adversarial 对照须另行登记同新头、初始化、随机库、optimizer与G预算的匹配实验，本协议不执行它。
