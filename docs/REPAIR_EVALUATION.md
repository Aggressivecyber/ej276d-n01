# N01 repair_002 条件终点评价

实现状态：完整可调用代码与12项人工 CPU 检查已完成。当前正式噪声门未通过；两臂新响应训练和本评价均未科学执行。本实现任务没有读取实测 TRAIN/VAL 数组，没有产生新实测响应，也没有响应模型优化更新。人工检查只用合成过程、模拟身份/组编号及 E 盘临时小文件。旧 src/n01/evaluate.py 和 run_001 结果保持原样。

## 接口与强制绑定

根训练器在噪声门通过、确定性与完整随机两臂各达到固定19000次响应更新后调用：

```python
from n01.evaluate_repair import evaluate_repair_checkpoint
report = evaluate_repair_checkpoint(checkpoint, container, identities, out, device="cuda")
```

仅接受 schema_version>=2、loss_version=2、variant/arm 为相同的 deterministic 或 full、update=19000 且配置声明 updates_per_arm=19000 的新候选端点。缺失 noise_initialization、electronics_state_sha256、noise_gate_sha256 等新字段的旧检查点，以及中间检查点，均明确拒绝。人口比较权重固定为1；若记录表明启用了人口学习或保存 theta 非零，也拒绝评价。

在载入 VAL 前，实际容器/身份文件 SHA 必须分别等于 checkpoint.config.train_container_sha256 与 identities_sha256。读取 checkpoint.parent.parent/noise_validation/NOISE_GATE.json，核验实际文件 SHA 等于 checkpoint.noise_gate_sha256、passed 严格为 True、电子学状态 SHA 与检查点一致。通过 make_n01_response_v2 与显式 noise_initialization 构造，strict load 全部模型状态，再计算恢复后的电子学状态摘要，确保它是已验证的冻结候选。生成前后完整模型状态摘要不变。

完成目录不可混入其它端点或输入。复用完成结果也会重新检查实际数据绑定和当前噪声门；递归检验报告、分类器、数组和方向库等输出 SHA。没有完整回执的部分结果保留，调用者需换新 E/D 输出目录。两臂汇总再次核验噪声门和输出文件，不以旧缓存替代有效门条件。

## 独立冻结方向库

构造新的 BoundLossV2，只使用保存的 TRAIN loss_statistics 和 config.projection_seed。显式 create_evaluation_context(config.response.evaluation_projection_seed) 并绑定。评价 seed 必须不同于训练 projection seed。绝不加载 checkpoint.loss_state 或 last_projection_context 当作评价方向。

evaluation_projection_bank.pt 保存完整冻结评价方向及摘要。同一 bank 用于两臂、全支持生成–REAL、等 N256 生成–REAL及 REAL–REAL 距离。记录 SWD、逐坐标、短延迟联合三个波形子分量，QPSD 条件支持与噪声项。尺度及八个 Q 箱仍来自固定 TRAIN 统计，评价不重新拟合模型尺度、损失尺度、箱界、预算或随机种子。

## 固定响应样本与逐行诊断

生成 batch 固定128，CPU显式 torch.Generator 与原完整六键 sample_noise 顺序相同。两臂使用同一源父顺序、计数与随机流：

| 用途 | 数量 | 随机 seed |
| --- | --- | --- |
| VAL 总体 | 全512源父 ×4 | 20261013 |
| 条件重复 | 固定抽选128源父 ×64 | 20261034 |
| 同噪声输入干预 |8源父 ×32 ×4种输入 | 20261043 |
| 独立 C2ST 的 TRAIN 生成侧 |2048不同 TRAIN 源父 ×1 | 20261073 |

每臂总响应前向行数13312；两臂26624。这些是未来条件运行的预算，本轮实际科学执行数为0。逐行保存波形、Qbar/Qsig、时间伸缩、preADC电荷、电子学电荷、ADC电荷变化、电荷恒等式残差和饱和比例。输出仍使用相同14bit ADC、1ns采样及 Q=sum100:400−3sum0:100，SOURCE输入不预截断。生成器、数据与损失/评价帮助函数源码的 SHA 一并绑定，评价结束时检查源码未变化。

总体报告含电压/归一形状曲线、基线均值/RMS/ACF/频谱、Q–PSD协方差、八个固定箱内的描述性双高斯峰位置与原PSD标准差。稀疏箱记录支持不足；BIC、跨初值位置变化和重叠信息保留，不将每箱一律判为真正双峰，更不赋真实粒子标签。

重复数组保留每父64次 Q/PSD 样本均值、方差、协方差、选定时刻的波形协方差及抽样不确定性，方差统一 ddof=1。保留所有非正Q、精确零Q、守卫分母和非有限 raw PSD标记；原 all-row signed guarded PSD 用于损失和可比总体统计。raw PSD 直接按 tail/Q计算，零Q保存 NaN；一旦其总体含非有限行，不报告筛选有限子集后的均值/标准差。未定义上升时间也保留明确计数。

干预保存原输入、同Q公共归一形状、同形0.8/1.2缩放的配对输出与标量，明确标注输入 Q 超出冻结 TRAIN 范围的父事件、非正输出Q及饱和。保存实际完整噪声字典 same_noise_input_latents.npz。其它生成用途保存按实际 CPU/batch128/key顺序计算的随机张量流 SHA，汇总严格核对两臂相同。

## 等样本量距离

固定种子2026101251，从512源父均匀无放回选择256父，每父从已保存4次响应中均匀选择一次。REAL 按原始组等配额随机抽取512条，分成互不重叠、每组配额相同的256/256两域。生成–REAL与 REAL–REAL 均用N256和同一独立冻结 bank，保存具体行号、父ID和组配额。

全支持2048生成/599 REAL距离同时保留，但不能把不等样本的旧 REAL–REAL 分割距离当作直接通过阈值。新的等N对照也只是同次采集内的开发参照，不增加新的脉冲合格门，不把某个投影距离下降等同于完整响应校准。

## 主匹配 C2ST 与阴性对照

新主分类器使用每类 TRAIN2048 拟合、VAL256测试。源父/REAL组在 TRAIN–VAL 间交集必须为0。TRAIN源侧预定2048个不同合法父事件，各产生一次新冻结响应；VAL源侧预定256个不同父事件，每父取已保存4draw之一。合法 TRAIN 波形仅供独立 C2ST 构造/拟合，不用于响应更新、选择响应端点或重新拟合任何模型/损失统计。

REAL TRAIN/VAL分别按各自全部原始组的配额选择每域2048/256，并在每组内随机分两个不重叠标签。SOURCE–REAL真实侧直接采用 REAL–REAL随机标签控制第一侧的同一批事件，因此计数、支持与组配额完全相同。覆盖所有422 TRAIN组及12 VAL组；代码记录实际组数，支持不足时明确失败，不根据结果改计数或删组。

复用 diagnose_n01_windows.py 的六视图、WindowClassifier 和实际索引驱动 fit_classifier：full、baseline_raw、baseline_demean、baseline_detrend、pulse_demean、pulse_shape。每视图 SOURCE–REAL及组内随机标签 REAL–REAL各独立拟合200次固定Adam更新；归一化只在该分类器fit样本上拟合。每臂12个匹配分类器。另保留旧full域分类器200次更新用于连续描述，明确它采用 VAL内部原组分割，其样本数不等，不能代替新主结果。每臂分类器优化更新共2600；这些是独立域分类器更新，不是响应模型更新。

区间如实保留旧帮助函数的400次cluster bootstrap：SOURCE–REAL按源父/REAL组分开重采样，随机标签RR按同一REAL组跨标签共同重采样。所有新matched结果显式标 limited_original_group_count=True、limited_group_count=True，并保存旧帮助函数阈值标志为 legacy_helper_limited_group_count，避免把12组误读为充足的独立采集证据。没有翻转AUC朝向或按测试结果选择种子/视图/端点。

这些数据是已用于机制诊断的 development reusedVAL。即使训练/验证父与组隔离，也不构成未查看的独立采集终验；当前只有同次采集，绝对硬件校准、物理权重、真实粒子标签和跨采集验证仍未建立。

## 输出与两臂汇总

每臂输出 endpoint_report.json、endpoint_populations.npz、repeated_parent_responses.npz、same_noise_input_interventions.npz及噪声字典、equal_sample_selection.npz、evaluation_projection_bank.pt、evaluation_binding.json、分类器PT/预测NPZ/结果JSON。ARTIFACT_MANIFEST.json递归登记所有数组、方向库和分类器文件，EVALUATION_COMPLETE.json登记报告及文件SHA。

summarize_repair_comparison(run_root) 在两个固定路径 deterministic/evaluation 与 full/evaluation 汇总至新run根 ARM_COMPARISON.json。两臂齐备时严格核对容器/身份、冻结电子学、通过噪声门、评价 bank、所有生成随机流及等样本行选择相同。未完成的臂保持 pending，不挑选其它检查点替代。所有输出路径必须位于E/D。

## 人工验证回执

12项 CPU 测试通过（最终一次0.562s）：旧/中间检查点拒绝、strict v2恢复与电子学状态摘要、独立冻结评价 bank 与禁止读取训练 loss_state、人工正距离与同分布零距离、等N一父一draw和匹配组配额、TRAIN2048/VAL256源父/REAL组隔离及全组覆盖、非正/零Q逐行保留、两臂随机流一致、注册容器与实际通过门绑定、错误检查点在loadVAL前拒绝、递归artifact及完成后门修改检测、有限采集不确定性标记且不改变bootstrap区间。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:TEMP='E:\GAN\runtime_tmp'
$env:TMP='E:\GAN\runtime_tmp'
& 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe' -B -m unittest discover -s 'E:\GAN\research\N01\tests' -p test_repair_evaluation.py -v
```

人工测试验证实现接口与数值边界；没有执行完整实测评价或CUDA评价。噪声门失败触发正式停止条件，不能由已准备好的评价代码绕过，也不能称条件流程已产生两臂训练/评价结果。
