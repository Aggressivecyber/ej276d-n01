# WP4 022：原固定T2机制的硬观察可分辨性前置检查

022只生成一个完整T2 teacher hard kernel，判断原017固定非高斯零Q形状机制在现有hard256/O19口径和既定容忍下是否得到有限样本支持。021 canonical电荷诊断已独审收口；其方差分解不回答零Q形状机制经过电子学、ADC和寻峰裁窗之后的问题。018同族paired资格和020双seed非通过均保持原结论。

本任务G/AD、optimizer、训练trial、student/initial/已训练端点CP均为0。无论结果如何，不启动T2训练、不打开旧017 stage2、不改均值/形状头、人口、观察窗、尺度或科学门。一次有界native调用；持续用户授权支持必要后续工程修复，但没有内部盲重启循环。

## 固定机制与随机命名空间

直接复用Runtime017.signals、teacher_shape_coefficients017和evaluate_kernel017。T2仅将原B0对应系数变为eta0=.9×Rademacher+sqrt(1−.9²)×z0，另外7个shape系数仍为原标准高斯；原实现的浮点计算不改。B0为旧只读cache的第一列，canonical Q为零，不替换成更显著的方向。INIT记录实际B0按父/采样顺序的float64 SHA，以及整个B文件SHA和旧manifest作用域。

a=.01qref、b=.0025、qref=3.7103998046755806、A/B/Q、005与三电子学假设全固定。B0系数均值/方差和canonical电荷不因该替换发生目标改变；高阶形状变化能否在硬输出中分辨需要022实际检验，不能由preADC构造直接宣布。

6593原父、5274fit/1319heldout、两个bank、每bank32draw全部保留。Gaussian仍为原kind=evaluation,role=teacher,bank=0/1；T2与已保存T1同bank共享Gaussian前缀。T2另使用该bank的独立sign流，旧T1未使用sign。不能把同bank T1/T2当独立样本算cross，也不能pooled64。

原teacher评价入口附带生成kind=coverage,teacher=T2的7595136个PIT uniform并保存，022评分不用它们，但不得从预算中省去。原005实际生成421952行，不能把G/AD0写成NN0。

## 最小输入和新产物身份

WP4_022_SAVED_INPUTS.json绑定27份必要输入，共833,895,817字节：原A/B/Q、cache manifest、parent split/物理权重/冻结证明、005和statistics、T1评价metadata和六份hard bank、三份018 T1 O19和三份null/full257-reference control、parent axes、原018配置与最终资格、021最终独审回执、直接用户授权。前瞻scope另作合同绑定。未使用的T1原REPORT、旧PIT、initial scores和所有学生端点都不读取。

新root为/home/rylan/projects/ej276d-n01/wp4_022_20261016。旧源码/CFG/MF和结果不修改。科学闭包为旧019的30份实现，加run_wp4_022.py与漏项修复所用旧已冻mechanism012_losses.py，共32份，明确仅覆盖当前调用路径；不声称所有legacy lazy分支均经过测试。最小native ops由root另绑，仍复用旧监督原语。

只读取005 CP一次，再确定性构造原005/扩展；A/M0原网络、原CP和SOURCE/REAL/DEV原波形不读取或前向。新kernel/score文件均在022 owned输出。新JSON写诚实run_id022，旧017格式schema仅作implementation_schema_origin；Store/journal格式可沿用旧版，身份取新配置。

INIT_PROOF给CP1、冻结005/cache/B0、T2机制和bank/sign/PIT namespace。实际T2 sign请求、005/hard正进度由原durable entered/completed journal及status证明，不添加G1、额外probe或新监控helper。root确认真实进度和同快照资源后设一次未来苏醒并休息。

## 022数值组合

按同父分别算T2b0:T1b1、T2b1:T1b0。O19每方向为cross均值减半个两侧U-self；self分母为m(m−1)，cross为mn。保留有限负估计值。先将两个方向的结果在每父内部取平均，再应用原group/weight及parent-cluster SE；两方向不能当两个独立重复。

每hypothesis只计算四个within-bank self、两个跨family跨bank cross，另算一个T2b0:T2b1 cross复用T2self形成T2 null，共7×6593×32²距离。T1 teacher32 null直接复用018已验证向量。O19编码、状态mask、有限定义的截尾表示、1ns软件尺度和各hypothesis原固定T尺度不改，不恢复或补零窗外rise。

完整257维量使用原wave_energy018，将T2 bank轴反转而不反转父轴，再减已保存T1 reference；需2个cross+2个T2self，共4×6593×32²距离。它只报告有符号excess及逐父加权SE/区间，不借O19的.10阈值，也不套final−initial训练晋级门。

新T2 hard结果按018已有入口重建scalar/native32矩和O19并核对本次REPORT。原T1 hard/O19/control按SHA复用，不重复旧013/017/018全部工程套件。

## 36格报告与三格合取

完整报告4个原heldout组×3份原比较权重×3hypothesis=36格。主判据仅all_heldout/nominal三hypothesis，全三格均满足以下条件才写supported_at_registered_tolerance：

1. 原018支持规则成立：≥64父、原父计数与权重支持均≥95%、ESS≥原组父数一半。无删行、缺失rise填零或分母替换。
2. cross parent-cluster halfwidth95≤.05，T1及T2 teacher32 null的abs(mean)+1.96SE均≤.05。
3. 其后才检查cross lower95>.10。

support、null或MC不明确优先标MC_unresolved；控制已明确但区间跨.10标tolerance_straddling_unresolved；upper≤.10仅表示本容忍下未建立。三格部分支持且其余明确未建立时标电子学假设敏感，不从secondary格挑选结论。旧018的matching分类函数不直接反用，避免先lower>.10后再检查MC的优先级错误。

这是固定双bank的有界分辨基准，没有power曲线、同时检验校准或多次试验power估计。非通过不证明机制不存在、所有hard观测等价或所有训练方法不可恢复；通过不等于训练恢复、REAL或硬件验收。

## 精确预算和资源

| 022新增项 | 数值 |
|---|---:|
| 完整T2 teacher kernel | 1 |
| signal/teacher charge/shape cache/005行 | 各421952 |
| 三假设扩展/硬ADC/firstargmax裁窗行 | 各1265856 |
| shape/charge/005/extension normal scalar | 3375616 / 421952 / 42617152 / 119834368 |
| normal scalar合计 | 166249088 |
| T2 sign uniform / 原入口附带PIT uniform | 421952 / 7595136 |
| full257 / O19有序距离 | 81014784 / 141775872 |
| 总新距离 | 222790656 |
| CP加载 / 正常或失败CP保存 | 1份005 / 0 |
| A/M0前向、G/AD/optimizer、parent sampler、训练trial | 全0 |

settings022/budget_caps022/validate_settings022为纯stdlib接口，配置逐值固定。Store记录entered/completed/attempted/inflight；正常终态计数必须精确相等，失败不伪记完成。

CPU12、GPU10GiB、进程树RSS8GiB、wall21600s、总输出5120MiB。唯一kernel的7200s覆盖原生成评价以及随后CPU contrast评分，终态再次调用原per-kernel guard更新kernel_resources；1536MiB限制T2子目录；共享T2 O19/评分/日志为2048MiB，不能沿用旧仅复用T1任务的1024MiB。选收≤128MiB，其余完整数组留远端并给SHA清单。

唯一新增工程回归已用固定feature/metric-space人工数组执行：fixture624对，独立Python标量距离oracle1440对，共2064≤100000。核对crossbank/父轴、U-self负恒等值、逐父先平均后的SE，并区分错误same-bank、父轴反转及pooled语义；NN/CP/AD/G/RNG均0。它不生成实际硬波形或验证物理机制，不重测005/codec/旧套件。T2 sign与新增namespace由源码审核及实际启动journal核对。

入口同成熟driver：python scripts/run_wp4_022.py --config <frozen CFG> --config-sha256 <RAW SHA> --output <owned runs/primary/artifacts> --device cuda:0。root冻结、既有Sol独审后再部署，本施工不SSH或启动科学作业。
