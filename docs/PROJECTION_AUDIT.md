# N01 固定投影联合盲区人工审查

本次按用户新授权执行独立 CPU 机制诊断。只新增本脚本、人工测试、说明和 projection 产物，不改模型、损失、训练代码，不运行 GPU 或响应训练。原论文方案建议“随机投影分布距离”，没有规定六个视图各 32 个方向必须永久冻结；当前冻结设计属于实现选择。本次独立方向、刷新方向、逐坐标和短 lag 比较是新授权的诊断，不自动变更旧训练协议或科学验收线。

## 固定规模与可运行入口

- 96 条人工 480 点波形；幅度 0.08–0.12 V、衰减参数 28–42 ns、固定 0.3 mV 白噪声。它们不是真实数据或旧模型输出。
- 只在样本 180..479 加扰动，每条按**全 480 点**计算 RMS=1 mV，保留首 100 点基线、峰、Q 和 PSD。
- 构造 seed=91012，扰动 seed=91013；冻结训练方向从检查点直接读，原 projection_seed=2026101202；独立评估 seed=92012；刷新 seeds=93012..93015；短 lag pair directions seed=92112。
- 冻结标准化来自正式 full/stage_20000.pt 中合法 TRAIN 的 loss_statistics；没有拟合验证集尺度，没有读取原始 TRAIN 波形。
- CPU/数值线程不超过 2；PYTHONDONTWRITEBYTECODE=1；TEMP/TMP 位于 E:/GAN/runtime_tmp。

PowerShell 入口（环境变量须与上面一致）：

    E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe research/N01/scripts/audit_projection_blindspot.py --checkpoint E:/GAN/research/N01/runs/run_001_20261012/final_review_20261012_0900/train/full/stage_20000.pt --output E:/GAN/research/N01/runs/noise_audit_20261012/projection

人工测试：

    E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe -m unittest discover -s research/N01/tests -p test_projection_audit.py -v

## 联合零空间的构造与复现结果

实际六个视图为完整 voltage、峰归一 shape、一阶差分、二阶差分、宽度 5/17 平滑后的一阶差分，每个有 32 个固定方向。基线和峰不变时 shape 对扰动是线性映射；每条波形的 1/peak 仅为一个正标量，不改变该视图的零空间。

构造各视图的线性映射，与冻结坐标尺度及方向组合，得到 192 个投影约束，再加 Q 与 tail 积分约束。t≥180 时两积分约束相同，所以 194 行矩阵实际秩为 193。修改区有 300 个自由样本，联合零空间为 **107 维**。SVD 的秩阈值为 1.28466e−13，行先做 L2 归一化；未通过调整秩阈值追求原 claim 数值。

96 条独立零空间扰动各自按全窗 RMS 归一，最大绝对扰动 **6.07734 mV**。全部峰仍位于 180 前且位置/幅度不变，最大归一约束残差 5.53e−18。

| 比较 | wave 距离 |
|---|---:|
| 冻结方向、全 float64 参考运算 | 3.13923e−15 |
| 当前损失实现，float64 输入 | 1.29025e−6 |
| 当前损失实现，float32 输入 | 1.51783e−6 |
| 独立冻结评估方向（float64参考） | 0.472031 |
| 四个刷新 bank 的平均距离 | 0.466457 |
| 六视图逐坐标 W1 平均 | 0.428403 |
| 短 lag=1/2/4/8，逐时间 pair 的联合投影 W1 | 约 0.476–0.478 |

当前 N01Loss._standardize() 强制输出 float32，因此“float64 输入”的当前实现不是全 float64 距离。审查单独保留全 float64 参考和 float32 参考，避免把舍入残差错当成不完全零空间。报告另给同一独立评估 bank 的 float32 比较以保证精度口径透明。

float64 中最大 Q/PSD 误差为 1.77636e−15 Vns / 1.66533e−16；float32 输入中为 8.74991e−9 Vns / 7.65051e−8。基线/峰误差均为 0。当前实现 float64 输入的 QPSD/noise 距离为 0；float32 输入的 QPSD 约 3.41e−9。

短 lag 比较保留每个时间 pair 的身份，不把不同时间混在一朵分布里；另记录其协方差绝对变化。独立评估与刷新种子在看本次扰动检出值前固定，没有选最优 seed。

原用户 2.42e−7 / 4.33 mV 为待复现 claim。本次没有复现相同数值，也没有强迫改变构造来贴合它；**定性复现了固定有限投影的联合盲区**。独立 bank 能检出本例不证明它不存在自己的零空间；逐坐标距离也不能单独覆盖所有联合相关结构。

## 必须保留的限制

本例**不含 ADC、adv、generator regular 或模型可实现性约束**。人工扰动可包含较强高频和边界结构，未证明现有平滑低秩生成器能产生它。不能由此说当前模型实际利用了盲区、全部联合目标失效、物理波形已被破坏而无人检出，或刷新投影已经解决所有分布问题。

此处“检出”只指同一已知扰动对应非零替代距离，不是实测统计检验的 p 值或新科学通过门槛。旧模型的实际误差归因仍应结合冻结终点、分窗口诊断及真实噪声结构。

如果后续另获实现授权，可考虑显式按逻辑更新步传入方向 bank/seed，同一组 microbatch、损失求导和重放使用同一 bank；训练方向与冻结独立评估库分离，checkpoint 保存 RNG/步号；再组合逐坐标与短 lag 联合结构。**本次没有修改该接口，也没有启动新训练。**

## 65-tap 电子学 FIR 的 rho1 边界

这里 rho1 指滞后 1 的理论相关，不是去趋势后首 100 点样本 ACF。对于独立同分布白噪声输入，只考虑 FIR 输出：

    h = impulse + a*tanh(raw), a=0.5/65=1/130
    rho1 = sum(h[i]*h[i+1]) / sum(h[i]^2)

RMS 归一化同时缩放分子分母，不改变该值。

- **线性化到扰动的一阶近似**上界为 a/(1−a)=1/129=0.00775194；不能把它当完整 rho1 的严格上界。
- 包含二阶 tap 乘积的保守三角上界为 194/16641=0.01165795。
- 本约束下完整纯 FIR 的精确绝对上确界为 **192/16705=0.01149356**。tanh 在有限 raw 下不取端点，故严格说是上确界。

证明要点：先取 tap 的绝对值只会增大正 numerator；上述保守界已小于 1/2。固定 h0 时，对 h1 的导数为正，使其饱和到 a；随后逐项有前邻 a，而 2*rho1*h_i<a，故各尾 tap 的最优值也为 a。最后对 h0 的导数为负，最大值在 h0→1−a。因此 numerator=(1−a)a+63a²，denominator=(1−a)²+64a²。交替尾 tap 给同幅负相关。边界核数值验证一致，1000 个固定 seed 随机 raw 均未超过该界；冻结学到的核纯 FIR rho1≈−0.00092859。

**不把以上界套到随机 offset/drift、基线去均值/去趋势、ADC 后的有限样本 ACF。** 去均值、漂移及估计误差均会改变该统计量，观测 baseline ACF 超过界不单独证明程序错误。

## 对根代理窗口诊断的只读审核

读取 scripts/diagnose_n01_windows.py、NOISE_AUDIT_PROTOCOL.md、NOISE_AUDIT_CONTROL_AMENDMENT.md，以及 windows / windows_stratified_control 的保存报告与 PREFIT splits；未修改这些文件，也未拟合任何分类器。

- 用保存 split 行索引重新连接冻结 endpoint 的原父 ID/REAL 组，四个 case 的 fit/test 父/组交叠均为 0；两种 REAL–REAL case 均无两域事件复用。matched_SOURCE_REAL 的 fit/test 每父只取一次预选响应，满足 128/96 每类配平。这里 SOURCE 指 full 生成响应，不是清洁原始 SOURCE。
- 当前代码只从 classifier fit 两域样本计算 center/scale，再应用于 test；逐事件去均值、去趋势、峰归一是固定的共同观察变换，不使用 test 总体估计。AUC label1=REAL（或任意 REAL domain1），不取 max(AUC,1−AUC)，方向口径正确。
- full=0:480、baseline=0:100、pulse=100:480；pulse_demean 使用首100点均值，pulse_shape 再按 pulse 最大峰归一。各窗口虽然用同一通道数/更新预算，flatten 后参数量随窗口长度变化，故 AUC 差异不能当严格等容量因果分解。
- 配平生成–REAL AUC：full .97830；baseline_raw .87945；demean .53483；detrend .51986；pulse_demean .95627；pulse_shape .98220。全支持 baseline_demean/detrend 为 .56120/.53956。去基线均值后的区分大降提示 offset 信息重要；归一脉冲仍高度区分，因此不能声称整体差异只来自 offset。pulse_shape 仍可能包含漂移、时序、随机形状或噪声影响，不单独证明“平均形状网络错误”。小预算 AUC 接近 .5 也不证明相关噪声已匹配。
- 原四向不同组 REAL–REAL AUC .397–.423，测试两域仅 3/2 组，应封存为组移位压力对照，不是可靠等分布阴性基准；不倒转 AUC 正确。
- 追加每组内随机等量标签 REAL–REAL 的六个 95% CI 均含 .5。其 fit/test 仍按 7/5 原组隔离，组内两标签无事件复用；共同组 bootstrap 同时重采样两域，符合共享组的依赖结构。
- 追加对照属于明确记录的观察后协议修订，不能描述成原先完整预注册阴性对照。PREFIT 保留原脚本 hash=b24b...，追加执行脚本/current hash=96b8...；版本差异有可追踪原因。追加 PREFIT 只记录原 protocol hash，未单列 CONTROL_AMENDMENT hash；这是回执追溯可完善项，实际 splits、脚本 hash 和明确补充文档已保留。
- 400 次组 bootstrap 仅 5 个测试块，且条件于一次固定分类器和随机标签分配；不包括重新分标签、重拟合或独立采集不确定性。CI 含 .5 是本诊断未发现标签区分，不是域等价证明。
- 已记录预算 3600+1200=4800 次分类器更新，response_model_updates=0。本地回收窗口文件为 JSON/PREFIT，未包含各 classifier 的 logits/归一化参数 NPZ/PT，故窗口 AUC 与实际保存 scaler 不能在本地再次数值复算；结论是代码和 split 审核，而非这些拟合产物的完整重开验证。

## 产物与核验

projection/PROJECTION_AUDIT.json 为完整结果与源 SHA 回执；artificial_counterexample.npz 保存原/扰动/修改波形、约束矩阵、零空间和奇异值；independent_direction_banks.pt 保存原训练、独立评估和四次刷新方向。保护的 losses/model/train 源文件 SHA 在人工运行前后相同。

三个人工测试覆盖联合约束与实际旧 wave 目标的逃逸、同扰动被独立/刷新/坐标/lag统计检出，以及完整 FIR 上确界和当前 CausalFilter 的一致性。均通过。它们是数值/接口验证，不是训练成功或物理验收。