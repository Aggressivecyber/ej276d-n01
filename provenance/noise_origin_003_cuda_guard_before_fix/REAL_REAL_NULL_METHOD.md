# 固定 REAL–REAL 阴性对照校准方法

本程序属于 noise_origin_audit_003_20261012，仅校准评价分类器在已知随机标签零假设下的行为。禁止训练噪声或脉冲响应模型，也不重新解释或修改 repair_002 的失败噪声门。当前实现只完成合成语义测试；没有进行实测分类器拟合。

## 预先固定的数据角色

绑定 noise_origin_003_config.json 的 TRAIN 容器和身份文件 SHA256。读取科学数组时只访问 TRAIN_PRIVATE.npz 的 target 键，不访问 target_val、source 或 source_val。按 N01 canonical_real 将存储的 REAL 波形映射到 float64 inclusive14bit ADC格点，再取前100样本；这与当前N01二次digitization后的基线相同。无事件剪裁、删行或根据基线值选父事件。身份文件中的 TRAIN/VAL原ID和组编号可用于metadata-only交集检查，但不解码VAL波形。

全部21047条合法 TRAIN事件及422个原人工50-record组是候选池。按固定 seed2026101260，仅用组ID与合法行数，从 count>=44 的组选择12组为 pseudo-test，其余为 fit；所选组的所有合法TRAIN行整体分到同一role，不要求原50个槽全部通过早期QC。每次拟合/测试分别在其role池内无放回选两域，原组在fit/test间不重合，事件在两个随机域间不重合。

允许的metadata-only检查确认422组各有48–50行，预选组为 [12,591,1870,1970,1621,1714,155,1500,692,857,779,539]，合计596行；其余410组为fit。每域fit2048需每组4/5行，test256需每组21/22行，现有容量充分。每次同一role两域的组配额完全相同。程序复用原 balanced_real_domains，若任一组不能支持预定配额，明确失败；不减样本数、不改组、不采用基于结果的fallback。

这是固定采集内随机标签对照，不是独立采集试验。原组是人工连续记录分块，不能据此识别独立硬件采集或硬件噪声潜变量。

## 全重新拟合预算与随机流

正式预算128次全新分类器拟合，各200次Adam更新，共25600次评价分类器更新。每次2048fit/256test样本每域，主视图固定raw前100点。复用原 noise_validation.BaselineClassifier：Conv1d1→8→16，kernel9/stride4，SiLU，再flatten与MLP32；共有4897参数。Adam lr=1e-3，batch128，无放回抽当前fit样本；拟合4096行的逐坐标mean/std，只用fit行，std下限1/16383。测试分数不用于选择架构、预算、视图、初始状态或学习率。

每次重新创建分类器和optimizer；不warm-start、不重用先前拟合参数。保存全部float32模型参数、fit scaler、原始test IDs/组/分数/标签、fit行号与IDs、损失轨迹、组bootstrap与条件置换数组。固定seed规则（i=0..127）：

| 随机用途 | seed |
| --- | --- |
| pseudo role选择，仅一次 |2026101260 |
| fit域标签/抽样 |2026101261+i |
| test域标签/抽样 |2026101261+128+i |
| classifier初始化 |2026102261+i |
| optimizer minibatch NumPy流 |2026102261+i+1 |
| test条件标签置换 |2026103261+i |
| joint-group bootstrap |2026104261+i |

fit/test标签使用不重叠的角色seed范围；各次保存实际seed。classifier初始化使用Torch流，optimizer抽样使用NumPy流。相邻i的seed关系固定于协议，不按结果改种子。

## 旧joint-group bootstrap的严格同统计实现

保留旧方法1000次bootstrap。每次从12个test原组有放回抽12次，以某组的同一重复倍数同时加权该组两个随机标签；使用原tie-aware weighted AUC，取2.5%/97.5%分位区间。不将两个随机域的组独立重抽，也不把test事件当独立采集重抽。

为避免1000次重复排序，预先计算矩阵 C[a,b]：组a中label1分数胜过组b中label0分数的事件对数，ties计0.5。若组重复向量为m，则

    AUC(m) = (m^T C m) / ((m·n1)*(m·n0)).

包含全部交叉组项；n1/n0为各组两类计数。本优化只改变计算方式。NumPy rng.integers生成(1000,12)数组与旧循环1000次rng.integers(0,12,12)消耗同一随机流，分位区间方法不变。人工含大量ties的数据逐次检验全部1000个AUC，和旧 _weighted_auc 差异不超过1e-15。leave-one-group范围按同时删除该组两标签计算并保存每组AUC。

统计旧区间是否包含0.5；不包含则记一次旧方法假阳性。报告128次覆盖率、假阳性比例和95%Wilson区间。只按原方向AUC，不翻转到>=0.5。

## 每次固定分数的条件标签置换

在每次全拟合结束后，固定classifier、所选512个test事件与其分数。另做255次随机标签置换，每组保持原q/q标签配额，组间与事件集合不改变。AUC通过全test平均tie ranks计算，等价Mann–Whitney统计。双侧统计量为 abs(AUC−0.5)：

    p = (1 + sum(abs(AUC_perm−.5) >= abs(AUC_observed−.5))) / 256.

尾部使用>=，明确包含ties；+1规则保留Monte Carlo p值下界1/256。用固定alpha=.05统计条件检验假阳性比例和Wilson区间。score与test父事件集固定，置换时没有再次拟合classifier，也没有重新归一化。这种条件检验用于校准检验规则，不能替代前面的128次全重新拟合，不能把255置换误计为255独立模型。

## 推断和预算边界

全重拟合的重复对象是随机标签、事件抽样、初始化和minibatch流；原始TRAIN采集及12个pseudo-test组固定。Wilson区间只描述这一固定池/固定12组条件下的Monte Carlo假阳性比例，不代表跨采集不确定性。128次比率分辨率1/128，置换p值分辨率1/256；该有限次数本身也须保留。

报告null AUC均值、ddof1标准差、固定分位数以及两套FPR/Wilson区间。任何结果都不能追认旧VAL门、支持硬件漂移的唯一来源或构成噪声/脉冲模型合格证据。所有noise/response training updates始终0，只有evaluation classifier updates=25600。

资源由注册config与上层顺序runner绑定：wall1800s、RSS8GiB、GPU最多10GiB、输出最多1GiB，origin阶段结束才开始RR，保持单个科学job。脚本应用GPU allocator限额、固定更新计数和wall检查；上层runner监控整个进程树RSS/时限。128个float32小classifier checkpoint总量应低于5MiB，程序终点验证该上限。估计GPU128次200更新约1–3分钟，加I/O和CPU统计；这是工程估计，未实际测量。

## CLI与回执

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='E:\GAN\runtime_tmp'
$env:TMP='E:\GAN\runtime_tmp'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'E:\GAN\runtime_envs\local_cuda_retry_20261007073613_UTC\venv\Scripts\python.exe' -B `
  'E:\GAN\research\N01\scripts\calibrate_real_real_null.py' `
  --config 'E:\GAN\research\N01\noise_origin_003_config.json' `
  --out 'E:\GAN\research\N01\runs\noise_origin_audit_003_20261012\science\real_real' --device cuda
```

实际科学运行由根代理统一启动；上例仅接口说明。remote可传 --container/--identities覆写路径，实际SHA必须仍与config一致。默认相对路径以config目录解析，不改变文件绑定。Windows本地输出必须E/D。

TRAIN_POOL_BINDING.npz保存全部合法TRAIN IDs/组与完整role池行号。每次 repetition_000..127 保存 classifier.pt、audit.npz、result.json。ALL_REPETITIONS.npz保存AUC、旧区间覆盖、置换p与每个原事件fit/test访问次数。binding.json绑定config、数据/身份SHA、旧帮助函数和新脚本SHA、数据角色、资源与seed规则。

每轮写status.json并flush一条stdout进度，追加progress.jsonl。成功主报告名 RR_CALIBRATION_REPORT.json；完成后status=complete，COMPLETE.json绑定report与最终status SHA。status单独绑定，避免“报告包含旧status hash、status又包含report hash”的循环。失败保存FAILED.json：完成全块更新数、已知部分拟合成功更新数及失败时optimizer step是否仍在执行。不能把completedrepetitions*200误当作失败后的全部实际更新数。已完成或部分输出目录不可覆盖。

## 人工验证

9项CPU合成测试通过（2.230s）：ID/count-only全组role分配；固定seed双域相等组quota及fit/test隔离；tie平均rank AUC与旧AUC一致；1000次优化joint bootstrap逐次与旧weighted AUC及leave-one-group严格同统计；固定scores/事件集合/每组标签quota的255置换与+1尾规则；ties全部在tail以及人工强分离p小；Wilson边界数值；loader只读target且等价canonical first100；两次3-update人工分类器验证scaler仅fit、参数复现与部分更新计数。

最后一项共6次人工评价分类器更新；科学实测分类器拟合数0，科学noise/response更新0。除了获准的metadata-only身份容量检查，没有读取真实target波形。真实128次重拟合由根登记、核验实现和审阅后统一运行。
