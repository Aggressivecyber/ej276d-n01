# 适用范围、研究结果与欠账

Q1保留为当前开发研究参考：原A与015 M0形状头冻结，mean关闭，两个有效charge方差参数开启；M0作为charge关闭参考。两个seed终点分别保存，尚未选择部署seed。模型和数据身份见[REFERENCE_BINDINGS.json](REFERENCE_BINDINGS.json)，实际接口能力见[INTERFACE_SUPPORT.md](INTERFACE_SUPPORT.md)。

## 数据与物理口径

| 项 | 已绑定范围 |
| --- | --- |
| SOURCE | 原合法8192父；TRAIN6593、INTERNAL_VAL1599，固定DEV512已暴露。事件ID/角色属于审计与追溯，不能作为网络输入 |
| REAL | 原21047 TRAIN行，422个逻辑记录组属于一次既有采集；旧DEV和已暴露资料不重新命名blind |
| 输入/输出 | 已准备SOURCE carrier384，硬观察输出256；单位与观察链沿原冻结合同 |
| Q与h | canonical/硬Q是软件电荷量；h=.15·edep_hadron/(edep_ee+.15·edep_hadron)是SOURCE沉积组成代理。二者均不提供已标定中子能量或REAL粒子真值 |
| 权重 | 原计数、physical weights与比较权重分开保存；013物理校准权重状态为unknown。正比较tilt属于有限假设敏感性 |
| 电子学 | 005及fast/slow/common_level共享前缀分别报告；缺少独立长静默信息时继续保留长程假设依赖 |

来源：[数据角色](../../../DATA_ROLES.json)、[013输入审计](../../../INPUT_INFORMATION_AUDIT.json)、[观察合同](../../../OBSERVATION_CONTRACT.json)。真实条件响应、源人口与触发/QC选择过程在单采集未配对设计下继续按假设依赖解释。

## 已结案证据

| 工作 | 冻结结论与作用 | 证据 |
| --- | --- | --- |
| 015 M0/M1 | M0保留为研究参考，mean关闭；M0完整wave用途门6/6，M1 0/6；绝对T联合门未通过，weights未晋级 | [015结案](../../../receipts/wp3_015_20261015/FINAL_RESULT_RECEIPT.json) |
| 016 Q1/M0 | Q1 mean12 wave=0.15011008209012994，M0=0.15012505860962605；符合原开发替换规则。六个电子学/seed组合的绝对T联合门均未通过，保留peak/rise/FWHM/基线DC等残差 | [016结案](../../../receipts/wp3_016_20261015/FINAL_RESULT_RECEIPT.json)；[016报告](../../wp3_016_20261015/REPORT.md) |
| 017/018配对教师 | 原017的72个支持pending按原规则保存；018在新登记的censor-aware口径下完成同族配对资格。教师权重属于合成验证 | [018报告](../../wp4_018_20261016/REPORT.md) |
| 020无配对T1 | seed0为1183pass/89fail，seed1为1272pass；总2455pass/89fail/0pending，双seed资格未通过。有限结果不构成所有无配对方法的否定或数学不可识别证明 | [020结案](../../../receipts/wp4_020_20261016/FINAL_RESULT_RECEIPT.json) |
| 021解析入口 | canonical人口/核内方差及a/b代偿有解析证据；硬观察、真实总体/条件恢复是另一个问题 | [021结案](../../../receipts/wp5_021_20261016/FINAL_RESULT_RECEIPT.json) |
| 023固定T2前资格 | 原support成立，MC与两null均resolved；36格均未达到登记.10。原full257少数正下界只作描述；T2训练保持关闭 | [023结案](../../../receipts/wp4_023_20261016/FINAL_RESULT_RECEIPT.json) |
| 024固定均值profile | 原13点中plus2/minus3/plus4为三有限均值witness，支持原M0上的control-only诊断 | [024结案](../../../receipts/wp3_024_20261016/FINAL_RESULT_RECEIPT.json) |
| 025固定plus2 | 90正T非劣检查73pass/17fail，另42未校准pending；六格均非通过。rise六格及wave五格非劣失败，均值witness与完整分布退化同时保留，plus2不晋级 | [025结案](../../../receipts/wp3_025_20261016/FINAL_RESULT_RECEIPT.json) |
| 026固定M0组成stress | native0，18格与72项有限4draw分解完成；144个fake/8个REAL矩、162个fake/9个REAL flags独审核对。low_h/high_h的Q均值相对nominal约+7.48%/−6.41%，显示固定系统的人口统计依赖组成假设；权重未选优或拟合 | [026结案](../../../receipts/wp5_026_20261016/FINAL_RESULT_RECEIPT.json)；[结果独审](../../../receipts/wp5_026_20261016/postrun_independent_01/RESULT_REVIEW.json)；[报告](../../wp5_026_20261016/REPORT.md) |

017的支持pending、020的资格失败、023的固定容差未检出与025的分布退化是不同结论，按各自原规则保留。新比较权重、次级参数通过或工程native0都不回填这些门。

026结果独审的首轮浮点展开式使4475个h=1父的low_h成为0.7999999999999999并触发下界断言，原失败与修复成本已封存。依原前瞻公式补充读回后权重与producer逐值差0；累计24个唯一NPZ、48次打开/420成员加载，无第三次decode或科学重跑。这是026历史读回账，027没有新增数组读取。有限4draw分解与h倾斜继续按其经验测度和组成stress范围解释。

## 证据层级

本包在本机重新核验37个原016项目导入源码和所列JSON/文本的RAW身份，均可追到原冻结记录。六个CP引用来自既有config/inventory的bytes/SHA；027未读取或重hash CP二进制、未加载tensor，也未解码NPZ。历史原生训练/保存数组核验属于链接回执的范围；完整CDF/loss、随机流和CP重放等未完成项继续按原回执保留。

软件定位为带失败卡的研究参考。科学状态保持`response_match_not_established`，模型未部署、权重未晋级，独立硬件/跨采集接受尚无证据。

## 完整WP5/WP6的具体欠账

| 需要的结论 | 对应缺项 |
| --- | --- |
| 最终有效系统选择 | physical-simple、deterministic、最终模块化或受限fallback在同一确认合同下的三方比较；当前Q1开发参考不替代该选择 |
| 响应可用性 | 硬输出全电压/归一形状/局部结构、Q/tail/PSD联合与电荷条件统计、噪声和形态的预定用途/绝对容忍度及失效处理；未校准pending继续单列 |
| 输入依赖与模块解释 | 有效SOURCE形状/幅度干预、关闭输入条件的消融，以及与同父重复规律的对应；关联或人口重加权不能替代干预 |
| 稳健性与不确定性 | 原人口与可比人口分别报告，三电子学假设、模型seed/SOURCE父/REAL逻辑组/新随机库分层；有限MC不制造独立父或采集 |
| 独立统计确认 | 按父/采集组隔离的独立C2ST、同规模同组策略REAL–REAL；未拒绝差异不证明相等，REAL–REAL不是不可约下限 |
| 全pipeline CV | 每折重新拟合全部数据依赖统计、005、A及候选；原A19k使用全部原TRAIN，仅留出新头数据不能形成真正未见训练评价。若继续复用原A，结果应标冻结系统后验敏感性 |
| 可运行交付 | 统一推理loader/`response.sample`、显式流/父引用/flags及边界处理的原生验收、可复现实例、自足构造依赖与模型包 |
| 最终材料 | 适用卡、失败案例、三系统统一结果、无原始隐私波形的公开包、中文综合报告/英文图/论文初稿和针对性查新；最终确认后如部署，全TRAIN重拟合单独登记，成绩不作新未见测试 |

主计划来源：[用户主计划WP5/6](../../../USER_MASTER_PLAN_20261015.txt)。[MASTER](../../../MASTER_EXECUTION_PLAN.json)与[PROGRAM](../../../PROGRAM.json)仅供当前状态导航，不作为永久模型SHA绑定。

027固定保留现有结果与假设，不从三种h权重中挑最贴REAL者，不增加theta、timewarp、高秩族、T2训练、REAL长训或最终盲评。原WP3六slot已用完、WP2剩余slot不挪用，教师与最终确认仍按独立预算处理。
