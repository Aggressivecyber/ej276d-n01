# 025：固定plus2的硬输出分布开发检查

`wp3_025_20261016`只回答一个前瞻登记的问题：024已暴露的`plus2/index5`，相对`zero/index0`，是否在两bank、三electronics假设的每一格都满足原013正人工容差下的有限分布非劣。点固定为`[0,0,1,0,0,0]`，不再选点、扫描或拟合。

本次实际计算`D(plus2,REAL)-D(zero,REAL)`。两种状态各自完整26372行，与同一份21047行已暴露REAL TRAIN比较；不截齐样本数，不配对SOURCE与REAL父，不pool两个bank。它是已选均值候选的新增分布指标开发复核，不是独立heldout确认、原deterministic A的2%接受门或完整响应接受。

## 输入与入口

CLI固定为：

```text
python -B scripts/run_wp3_025.py --config /home/rylan/projects/ej276d-n01/wp3_025_20261016/wp3_025_config.json --config-sha256 <frozen SHA256> --output /home/rylan/projects/ej276d-n01/wp3_025_20261016/runs/primary/artifacts
```

入口只import标准库和标准库`wp1_ops`；`settings025()`、`budget_caps025(p)`、`validate_settings025(p, *, frozen=True, driver_source=None)`供ROOT冻结。输入/source/contract原件完整RAW SHA、bytes、白名单与新输出路径在数值import前一次核对。全部21输入198022817B的精确record与scope见[前瞻scope](../receipts/wp3_025_20261016/registration_01/PROSPECTIVE_SCOPE.json)。

输入为12个024保存hard文件、两个bank共享`BASELINE100`、`AXES_AND_TARGET`、`PROFILE`，以及原DATASET008、TRAIN统计、013容差与REAL identities/flags、011保存投影。`DATASET008.npz`只解压一次`real`成员；SOURCE/DEV及其他数值成员、REAL baseline不读。REAL的原TRAIN角色通过原始dataset RAW SHA、已验证角色的013 COMPLETE、DATA_ROLES和保存REAL IDs/groups链绑定，不另读`real_role`成员。原6593父ID、4draw顺序、physical weights完整保存；原013距离的测度仍为均匀原行计数，物理权重不参与重新拟合或改测度。

hard沿原lossless codec按uint16 ADC码、保存的实际baseline mean与可选IEEE XOR重建float64硬256输出；不再ADC、argmax、crop或神经前向。保留所有保存hard flags，另存原013定义的可观察量、未定义原因和原分母。

科学运行实际import闭包只有：

- `scripts/run_wp3_025.py`
- `scripts/wp1_ops.py`
- `scripts/stability011_metrics.py`
- `src/n01/saved_array_kernels017.py`

最后一个模块直接从其目录import，不触发`n01/__init__.py`或Torch。部署包另含ROOT绑定的ops/doc/manifest及小合同；不存在旧019/022科学driver的隐藏依赖。

## 固定计分

复用011的`wave_views/standard/q_features/CDFCache.create/distance`与017已验证的纯NumPy morphology。沿原TRAIN中心/尺度、011保存投影及fingerprint `27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482`。运行从保存float32 banks按原Torch fingerprint格式核身份，再转float64使用；不构造loss或projection factory，不抽新随机数。

每个view、coordinate或lag组独立流式处理最多16列CDF。stable sort、float64 support/gap/mass/cumsum、不等N的双域独立归一与ties语义保留。先求原组均值，再按`wave=.50*mean6SWD+.25*mean2coordinate+.25*mean8shortlag`聚合，不把8776列合池。不持有完整WholeProxyCache。

QPSD沿原2coordinate与16projection的global项、固定8个extended-log Q bins、`searchsorted(side='left')`、双域support≥16，使用原global/conditional各半的公式。原无bin时global-only数值仅保留为描述；zero、plus2、REAL任一缺bin时完整QPSD的非劣检查pending。

8个raw scalar为Q、front、tail、PSD_raw、PSD_guarded、peak、rise10–90、FWHM，分别在明确定义子群计算原W1，并保存定义数/原分母；不为某个scalar缺失删除全局行。9项原flags用相对REAL的绝对率差，再计算两状态距离之差。rise/FWHM截尾原因与未定义值保留。

024同一bank完整prefix100由相同base和prefix为零的原015均值adapter共享。025读取该保存prefix并核全部12份hard的baseline mean在bank内逐bit一致；完整prefix恒等的依据是冻结024算法/交付链，不伪称重新生成两份prefix。因此`noise/baseline_mean/baseline_RMS`的同目标距离差严格为0，标`deterministic_identity`。本次不计算它们对REAL的绝对距离或绝对接受。

每格原013全部22指标均报告`D_zero_REAL`、`D_plus2_REAL`、真实delta、T、可计算的D/T与delta/T及状态。正T沿原定义：可计算、QPSD完整且delta≤T才pass；超过则fail。原零/None/not_calibrated T永远`pending_not_calibrated`，不补epsilon。6格共90个正T检查、42个未校准pending，bank算术均值单独描述，不替代逐bank判据。原T来自登记人工response扰动，不能解释成硬件统计power。

## 预算、资源与证据

固定工作为12个保存hard/316464行、REAL21047行一次、特征337511行、12个到REAL评分、6个delta格。wave每格`6×32+2×256+2×4×(255+254+252+248)=8776`列，共105312；global QPSD216列。条件PSD与raw scalar各≤96列，总≤105720列。16列分块CDF create≤`12×(12+32+506+2+8+8)=6816`，sort support-column值≤`12×(8776+18+8+1)×(26372+21047)=5009153484`；最后的1是8个条件bin合计一次人口质量。支持不足可少于这些cap，不作为工程未完成。固定项须精确完成，Store逐操作持久保存entered/completed/inflight，不把未完成操作记作完成。

新增NN、CP、AD、G、optimizer、训练trial、参数fit、RNG、ADC/argmax/gather、energy/O19 pair distance均为0。CPU12，AS/RSS8GiB，wall7200s，CPU soft/hard21590/21600s，output1536MiB，选收128MiB。成熟CPU监督器持有单作业锁、原生退出码与owned process identity；无自动重试。

`INIT_PROOF.json`只证明输入/REAL读取/投影初始化；正确数值启动须见`FIRST_ACTUAL_CDF_PROOF.json`的`exact_float64_CDF_completed=true`、实际16列/758704 support值及Store真实completed计数，并由ROOT取同一快照owned PID/cwd/startticks/boot/CPU/RSS健康。若仍运行，ROOT设一次未来苏醒后休息，不持续轮询。

终态保存12格`*_DISTANCES.json`（含各组、每个CDF block的均值/列宽/支持数）、`*_OBSERVABLES.npz`、REAL observables/IDs/groups、原024 axes/weights、prefix摘要、`FULL_PREFIX_IDENTITY.json`、`COMPARISON.json`、资源、durable journal、COMPLETE/FAILED与inventory。大024 hard留远端。该读回材料可复核聚合/支持/flags/delta，不宣称独立重放全部原CDF排序。

唯一新增人工检查为[run_stream_fixture.py](../receipts/wp3_025_20261016/engineering_01/run_stream_fixture.py)：固定合成136/152不等行数，覆盖ties、六view/8lag分组、8Q bins及zero/negative Q，并与成熟完整Representation/WholeProxyCache和raw_distance比较。无私有数组或随机数。通过回执在[fixture_01/RESULT.json](../receipts/wp3_025_20261016/engineering_01/fixture_01/RESULT.json)；人工成本独立于025科学预算。

本次任何结果都不自动打开timewarp、额外theta、训练、T2或最终盲评。均值改善可能伴随分布退化，正T有限结果与42项未校准状态均原样保留。
