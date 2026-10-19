# N01 当前阶段：原任务按预算终止，剩余比较已正确启动

原三拟合批循环 REAL 任务在 2026-10-20 02:47:16（UTC+8）触及内部 7200 秒上限，耗时 7201.39 秒、exit 1。659 份源文件、91 份输出及全部已保存状态已核验；完成 35 个方向、1806 个参数差分列和 19 次更新。第二头神经已完成 11 个方向，第 12 方向消耗 12 列、保存 9 列；物理对照尚未开始。见[终止审计](receipts/joint_response_01_20261020/wake_20261020_0254_01/TERMINAL_REVIEW.json)。第一头 24/96 个最终改善门失败的结论保留，科学验收仍为 false。

已前瞻登记并启动[固定方法补全](docs/JOINT_RESPONSE_01_MULTIBANK_REAL_COMPLETION_20261020.md)，仅完成第二头剩余 1 个神经方向、12 个物理方向及原最终评价和同父核比较。保持原算法、初值锚、数据与随机数、候选限制和每轨迹总计 12 个方向；重复加载、12 列差分及相应采样费用独立计入。新作业内部上限 3600 秒、外部 3900 秒加 20 秒，4CPU、GPU 2.75GiB、RSS 12GiB。

03:07:52（UTC+8）的[启动核验](receipts/joint_response_01_20261020/multibank_completion_01/STARTUP_ASSESSMENT.json)确认科学 PID79561 身份吻合、672 份部署文件和 30 项锁定输入通过；恢复基线及保存的 9 列差分共 36 个数组在原生环境逐字节一致。进度记录截至 03:07:41：第二头神经最后方向已完成 36 列，尚无新更新；此后的远端进度未观察。

按启动至 36 列的实测 64.51 秒估计，本次约需 18—25 分钟，预计 03:24—03:31 完成，已安排 **2026-10-20 03:35（UTC+8）单次接续**。见[耗时估计](receipts/joint_response_01_20261020/multibank_completion_01/WAIT_ESTIMATE.json)与[定时回执](receipts/joint_response_01_20261020/multibank_completion_01/AUTOMATION_CREATE_RECEIPT.json)。阶段 4 的 REAL 响应匹配尚未建立；阶段 5 及全流程验证未完成。

以下保留历史快照；当前状态以本节、三台账及对应回执为准。

---

# N01 当前阶段：第一头比较已审核，第二头仍在登记预算内运行

2026-10-20 02:19:27（UTC+8）快照确认原科学PID18472及启动身份一致：已完成1353/2400个参数差分列，总16次接受更新，正在seed1_joint第3方向。659份源码再次核验；该时刻RSS峰值约2.65GiB，预算未改变。见[运行与部分审核回执](receipts/joint_response_01_20261020/wake_20261020_0218_01/ASSESSMENT.json)。后续运行状态尚未观测。

第一头神经／物理各12方向、全部拟合角色覆盖、最终双库评价和同父核已完成。58份保存文件哈希通过；1200差分列、23次割线、47候选、188接受判定、参数链及原96门已复算。神经接受5次更新，最终joint剩余0.8852—0.8938、peak 0.9194—0.9312、native wave 0.9890—0.9922、QPSD 0.9926—1.0049，噪声比1；24个改善门仍失败，其他72门通过。物理接受10次更新，但最终native wave/QPSD仍约2.47—2.50／8.99—9.14倍M0。[第一头审核](receipts/joint_response_01_20261020/wake_20261020_0218_01/SEED0_REVIEW.json)、[保存证据诊断](receipts/joint_response_01_20261020/wake_20261020_0218_01/SEED0_DIAGNOSIS.json)。

神经同父实际MC均值改变约1.385%，wave方差为M0的1.0047—1.0056，协方差迹为0.9962—0.9970；没有REAL逐父随机核真值。保存向量与统计量的复算不替代原始候选波形或逐draw重新生成。

原70—85分钟估计未达到。实测第一头80.93分钟，整轮估计162—164分钟，预计超过原内部120分钟／外部125分钟上限；该上限保持。旧单次定时已消费并确认消失，新单次接续在**2026-10-20 02:54（UTC+8）**，位于原硬截止和kill grace之后。[估时依据](receipts/joint_response_01_20261020/wake_20261020_0218_01/WAIT_ESTIMATE.json)、[单次定时回执](receipts/joint_response_01_20261020/wake_20261020_0218_01/AUTOMATION_CREATE_RECEIPT.json)。届时核验终止证据；若比较未完成，先前瞻登记只补足原定剩余工作。完整比较、阶段5及全部合法TRAIN重拟合仍未完成，科学验收为false。

以下保留历史快照；当前状态以本节、三台账和对应回执为准。

---

# N01 当前阶段：M0起点三拟合批诊断已正确启动

2026-10-20恢复。655份部署文件、原生导入和派发前逐文件核验通过。00:20:20（UTC+08:00）的首次快照确认科学进程存活，两头各128行新随机数的M0零增量信号逐字节一致，共256行、2次生成调用，零提交更新。详见[正确启动回执](receipts/joint_response_01_20261020/multibank_m0_01/STARTUP_ASSESSMENT.json)。

按[已冻结配置](configs/joint_response_01_multibank_m0_20261020.json)执行[三拟合批联合设计](docs/JOINT_RESPONSE_01_MULTIBANK_M0_DIAGNOSTIC_20261019.md)：同一54参数方向保留全部交叉项，各批分别约束原生指标，每头最多3候选，首个通过后仅一次FRESH。拟合角色与原科学门不变，CHECK/FRESH不是全流程未见验证。

当前仍处于阶段4，REAL响应匹配尚未建立。已恢复10分钟定时接续，实际配置已回读核验。阶段5与最终全部合法TRAIN重拟合仍待完成。以下是历史快照，最新执行状态以三份台账和本次回执为准。

---

# N01 当前状态：按用户要求暂停，等待下次通知

2026-10-19。JOINT_RESPONSE_01停在阶段4，REAL响应匹配尚未建立。最近割线修复诊断已完成并通过保存数值的独立审核，但两种子局部方法门均未通过，零提交更新。已有科学作业全部终止，自动接续已删除；本次收尾没有新建服务器作业。

本机已保存[M0起点与三拟合批设计草案](docs/JOINT_RESPONSE_01_MULTIBANK_M0_DIAGNOSTIC_20261019.md)及脚本，通过语法、导入和4次CPU归一化检查。新诊断运行配置尚未冻结，未登记、未部署、未启动。完整结论与恢复步骤见[暂停与续接记录](docs/JOINT_RESPONSE_01_PAUSE_20261019.md)；[最新数值审核](receipts/joint_response_01_20261019/wake_20261019_1059_01/TERMINAL_REVIEW.json)与历史负结果保持不变。

GitHub发布代码、配置、方法和汇总验收证据；原始波形、逐行身份、权重与诊断数组保留在E盘。阶段5及最终全部合法TRAIN重拟合仍待完成。

以下保留历史阶段快照，其中运行、当前和自动接续描述只代表各自当时的状态。最新状态以本节和PROGRAM/MASTER_EXECUTION_PLAN/jobs三份台账为准。

---

# N01 当前阶段：JOINT_RESPONSE_01 原生指标约束的联合校准

2026-10-19，当前按[JOINT_RESPONSE_01_PLAN.md](JOINT_RESPONSE_01_PLAN.md)推进到阶段4 REAL联合校准，同步拟合低自由度物理基线。阶段1审计与阶段2工程通过；两种M0共13,186行及三种噪声硬输出精确回退已核验。

阶段3原64draw教师运行因第二种子协方差门失败，原判定保留。另行登记的冻结端点抽样诊断揭示MC敏感性；随后使用新随机数、每父2048draw、相同科学门限及新增MC精度要求的[独立确认](receipts/joint_response_01_20261019/kernel_precision_confirmation_01/TERMINAL_REVIEW.json)通过。均值误差2.33%–3.14%，协方差误差0.301–0.436；这只满足模拟教师先决条件，REAL响应匹配尚未建立。

[REAL完整回执](receipts/joint_response_01_20261019/wake_20261019_0426_01/TERMINAL_REVIEW.json)确认执行结束、科学门未通过：两神经轨迹各0/12更新，72次候选都被wave宏块拒绝，端点仍精确等于M0；24次接受全属于物理比较。物理模型改善自身起点，但原生wave/QPSD仍劣于M0。四条轨迹均覆盖全部6465个SOURCE拟合父和18,948条REAL；128父与42组2099条REAL未进拟合。

[约束方向诊断](receipts/joint_response_01_20261019/constrained_direction_diagnostic_01/TERMINAL_REVIEW.json)仍失败。[固定方向曲线](receipts/joint_response_01_20261019/linearization_diagnostic_01/TERMINAL_REVIEW.json)确认：相同FIT数据上已经存在明显的局部线性预测误差，改变特征投影还会扩大波形风险差异。随后[原生约束方向诊断](receipts/joint_response_01_20261019/native_direction_diagnostic_01/TERMINAL_REVIEW.json)通过两种子新库、新投影检查：联合风险下降6.72%/2.82%，native最大退化低于1%，零提交更新。这只是可用小步证据，未满足最终native改善要求。

[原生约束试验终止回执](receipts/joint_response_01_20261019/wake_20261019_0539_01/TERMINAL_REVIEW.json)确认触发2400秒内部时间上限、exit1；596份源码和19份输出哈希通过。已完成三轨迹全部拟合角色覆盖：两神经分别1、5次更新，seed0物理11次。seed0已失败24/96原门：joint风险仍为M0的0.9394–0.9416，native wave为1.0018–1.0077。实际同父核已重算，wave方差约1.053M0；Q方差比的近零分母另用绝对方差与逐父CV解释，不能当作物理增益。

[缺失比较补全回执](receipts/joint_response_01_20261019/wake_20261019_0602_01/TERMINAL_REVIEW.json)确认exit0、712.55秒、617源码与16输出哈希通过。四轨迹各12方向及全部拟合角色覆盖已完成；神经接受1/5次、物理11/9次。192个原final门逐条复算：两神经各失败24个改善门，noise/macro/flags均通过。seed1 joint剩余0.8723–0.8771，native wave0.9991–1.0049、QPSD0.9691–1.0034，仍未达到科学要求。物理虽然joint风险较低，原生wave/QPSD明显劣于M0。

两种子同父核均已独立复算：神经wave方差约1.053/1.020M0、实际均值改变约0.65%/1.39%，库间cov差约0.0574/0.0520；不据此声称REAL条件核已识别。[差分尺度诊断](receipts/joint_response_01_20261019/wake_20261019_0628_01/TERMINAL_REVIEW.json)已exit0结束，622源码与12输出核验通过，324FD/6求解/42试探/0更新。缩小尺度后Jacobian没有收敛；seed0三尺度均拒绝，seed1唯一局部FRESH通过方向的native wave仍劣于M0。当前已登记[读出算子归因](docs/JOINT_RESPONSE_01_READOUT_ATTRIBUTION_20261019.md)，复用固定FIT输入检查峰选择与ADC量化对导数和方向预测的贡献，零新求解、零更新，原科学评价不变。631份文件部署、原生导入及派发前逐文件核验通过。06:44 UTC的显存前检不足和后续等待回执保留；10:01:41 UTC显存释放后已按原预算首次派发。[正确启动回执](receipts/joint_response_01_20261019/wake_20261019_1000_01/STARTUP_ASSESSMENT.json)确认科学PID202850/start_ticks6588213，seed0尺度0.004第1差分列完成，原端点及首列精确复现，零参数更新。该运行的终止审核结果见下文。

每10分钟自动接续，阶段4/5及最终全部合法TRAIN重拟合仍待完成。原A0/M0/005/PCA与native尺度使用过全TRAIN，本轮是冻结系统后验研究；原native评价投影现已进入方向拟合，不能再作为独立验证证据。

S、070、039、M6和本次所有失败回执保持原样。工程、启动和科学验收分开记录，以PROGRAM/MASTER_EXECUTION_PLAN/jobs实时台账为准。全部工作由当前主助手完成。

以下保留以前的阶段快照。

---

# N01 当前状态：M6 负结果方法交付完成

2026-10-19。研究接口黄金验证通过，S 的科学验收未通过，`response_match_not_established`。没有合格最终模型，也没有活动科学作业；本阶段自动苏醒已关闭。

从[中文交付入口](reports/finalization_M6_20261019/README_CN.md)开始阅读：[中文方法报告](reports/finalization_M6_20261019/METHOD_REPORT_CN.md)、[API 使用说明](reports/finalization_M6_20261019/API_GUIDE_CN.md)、[英文论文初稿](reports/finalization_M6_20261019/MANUSCRIPT_EN.md)、[六张图](reports/finalization_M6_20261019/FIGURE_CAPTIONS_EN.md)，或下载[可分享复验包](reports/finalization_M6_20261019/packages/N01_M6_SHAREABLE.zip)。

S 在六条件下改善 wave 28.58–30.01%、joint 58.61–59.12%，但 QPSD/front/tail/PSD/负波瓣保护全部失败。523 项接口张量检查逐字节一致，24 份完整指标和 158,232 行原生 ADC/wave 复算精确。工程复验与科学验收分别记录，详见[S 决策](S_DECISION.md)及[权威台账](PROGRAM.json)。

公开材料包含代码、合成示例、人口级统计、图表和方法说明。私有波形、父标签/逐行索引、权重、拟合模板与私有快照保留在授权环境；完整神经推理依赖见 API 说明。S+R 未触发，新确认和 M5 全模型比较未运行。后续方法变更需另定有限研究设计。

以下原样保留历史阶段记录；其中“当前”、启动及等待描述均指当时快照。最新状态以上文及 PROGRAM/MASTER_EXECUTION_PLAN/jobs 三台账为准。

---

# N01 当前状态：038仅剩REAL1已正确启动，02:00单次苏醒

2026-10-18 01:28:43（Asia/Singapore）唯一启动快照：native20248/startticks420513、项目锁占用、CPU affinity0–3、RSS约2.82GiB/GPU4288MiB，无FAILED/exit。两教师与REAL0已严格恢复封存；REAL0五方向不重放，初始固定MONITOR明确作为无变化终点alias。REAL1.mean首方向32FIT微批完成，4096抽样/3049unique父事件，四幅度均拒绝、0接纳、radius1→0.5；原接纳门与完整Adam回退保持不变。

当前科学Adam构造确认下界49（教师43+REAL0的5+REAL1的1），FIT848，保留教师17、REAL接纳0；项目累计下界199718。CP23是旧14加本次9，离线5次metadata解码另记；快照以后的进度未知未计入。

前一resume01在REAL0五候选全部拒绝后，于阶段末report-only MONITOR申请3GiB临时距离张量而退出1。75文件12,688,039字节已收E并逐SHA核对。修复将大库全pair距离按64行分块，最大广播192MiB，全部权重和距离定义保留；唯一CPU合成检查差0逐位同，小FIT/CHECK路径与原run_track不变。初始大库MONITOR已GPU完成；阶段末与最终评价尚未核实。源代码/配置/设计独审PASS，51部署成员逐字匹配Git和远端，显存帽仍6GiB。

仅训练剩余REAL1原900秒，之后原统一FINAL/教师条件恢复/121同Q异形干预/026组成敏感性/RR及最后C2ST。教师1仍中断，四track完整资格始终不成立；科学状态仍为 `response_match_not_established`。已核验2026-10-18 02:00（Asia/Singapore）一次苏醒，期间休息、不持续查询。

[实际启动核验](receipts/response038_20261017/resume_20261018_01/wake_20261018_0110_01/repair_01/deployment_01/STARTUP_ASSESSMENT.json) · [单次苏醒回执](receipts/response038_20261017/resume_20261018_01/wake_20261018_0110_01/repair_01/deployment_01/FOLLOWUP_RECEIPT.json) · [修复独审](receipts/response038_20261017/resume_20261018_01/wake_20261018_0110_01/repair_01/independent_review/SOURCE_REVIEW.json) · [失败保存](receipts/response038_20261017/resume_20261018_01/wake_20261018_0110_01/repair_01/FAILED_ATTEMPT_PRESERVATION_RECEIPT.json) · [权威台账](PROGRAM.json)

以下为保留的历史阶段记录；其中“当前”、计划与运行快照均指当时状态。最新状态以上文和PROGRAM/MASTER_EXECUTION_PLAN/jobs三台账为准。

# N01 历史023记录：022失败保留，023评分恢复的当时快照

022完整T2教师生成和fast父分数已保存；首个分组评分因Z95错误导入退出1，完整科学结果未知。失败、完整日志、原源码和原审核撤回记录已封存。023新版本通过源码独审，仅修cell导入并续算CPU评分，复用所有完成部分，不重复教师/005/RNG或fast距离。

2026-10-16 13:54:48（Asia/Singapore）快照已确认：fast的12个分组分类完成，slow的两方保存hard读取完成，距离op4在途、完成unknown，未计入completed。当前CPU12和进程资源健康。已保存今日14:10单次苏醒，之后状态未持续查询。023新增NN/CP/AD/G/RNG/ADC为0，新距离预算148527104，项目累计G下界187667。

020为2455pass/89fail/0pending，89全部来自seed0、seed1全1272pass，双seed资格未过；021有限canonical解析及独立代数复核完成。完整WP5/WP6尚未完成，T2训练/REAL长训练未启动；scientific_status=response_match_not_established。后续必要工程恢复与具名有限新增计算沿用用户持续授权。

[023运行报告](reports/wp4_023_20261016/REPORT.md) · [023实际启动](receipts/wp4_023_20261016/EXECUTION_STARTED.json) · [022失败报告](reports/wp4_022_20261016/REPORT.md) · [021报告](reports/wp5_021_20261016/REPORT.md) · [020报告](reports/wp4_020_20261016/REPORT.md)

以下保留此前阶段说明，旧授权边界以最新直接人类授权为准。

# N01 历史019状态：训练部分完成，评价因漏依赖失败

019已native1退出：seed0完成2000G/AD，seed1未启动；首次评价缺少mechanism012_losses，完整评价与2544格主门未生成，科学判定未知。本次只补记新增1936G，项目累计下界185667。完整日志、三份检查点及64文件清单已核SHA、选回并备份；未重启或新增NN/AD/G/RNG。

源码打包和独审已更正此前“38成员闭包完整、足以执行”的工程结论，原件保留。最小修复草案仅补一份旧已冻依赖，尚未应用。后续可另行登记复用seed0检查点补评价、仅训练未启动seed1≤2000G/AD的有界恢复，需用户明确授权；原单次苏醒已履行，无活动科学作业或未来苏醒。

018新2544格合格及017旧72pending保持，T2/均值新候选/REAL长训练均未启动。scientific_status=response_match_not_established，六WP未全部完成。

[019失败结论与证据](reports/wp4_019_20261016/REPORT.md) · [最终回执](receipts/wp4_019_20261016/FINAL_RESULT_RECEIPT.json) · [终点独审](receipts/wp4_019_20261016/wake_20261016_0530_01/independent_result_01/REVIEW.md) · [未执行恢复方案](receipts/wp4_019_20261016/wake_20261016_0530_01/BOUNDED_RECOVERY_PROPOSAL.json) · [018报告](reports/wp4_018_20261016/REPORT.md)

以下保留此前阶段说明。

# N01 当前执行：WP3/015 训练完成，独立数组复核待执行

四次配对短试均2000G，原生子进程及监督进程均0，总计8000G/AD；所有154518操作完成，无在途或失败。原A/005保护通过，未读取DEV。本轮只补记7880次更新，累计175667。模型研究参考选择与完整数值接受，等待冻结合同规定的独立保存数组复核；科学状态仍为response_match_not_established。

[015原生终点](receipts/wp3_015_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [015合同](docs/WP3_015_CONFIGURATION.md)

以下保留此前阶段说明。

# N01 当前执行：WP3/015 已正确启动

015已通过首次真实AdamW更新、ALL6固定人口均值残差门和同一快照健康检查。四次配对短试串行、每次2000G，最多8000G/AD；M0继续C1形状头，M1冻结相同头而训练6参数均值适配器。A/005保持冻结，原硬观测和013数字T不变，005基线DC限制单列。

实际费用按唯一持久化快照记录，当前在途操作完成数未知；累计为原167667加本次已核G，只入账一次。完整结果、科学验收和跨采集物理复用仍待对应证据。已确认2026-10-15T13:20:00+08:00（Asia/Singapore）同一n01单次唤醒；当前休息，不持续监控。

[015配置](wp3_015_config.json) · [015合同](docs/WP3_015_CONFIGURATION.md) · [015启动回执](receipts/wp3_015_20261015/EXECUTION_STARTED.json) · [014中文报告](reports/wp2_014_20261015/N01_WP2_014_REPORT.md)

以下保留此前阶段说明。

# N01 当前执行：WP2/014 独立闭合，准备 WP3/015

014 四次配对短试全部完成，每次 2000G，总计 8000G / 9872AD，原生计算与监督均退出0。累计训练167667，搜索短试已完成4次；015正在实现，尚未开始新计算，WP3最多6次短试的预算保持不变。

独立复核后，选择 C1 作为后续开发代码路径。C0/C1均满足已标定分量ALL6非劣；12个final完整wave距离均值分别为0.15255321144257886 / 0.15042006897712382。C1的seed1-fast相对A改善1.9887788589535327%，全部6格至少2%的用途门未通过；权重未晋级，科学状态仍为response_match_not_established。

保存数组CPU独立读回已原生退出0并回收19个小文件：36对库、176个NPZ的SHA、1,898,784行硬输出、4/8次重复矩与标量协方差/MCSE及保存梯度向量核验通过。完整wave/QPSD/noise经验距离没有从raw重新计算，C1 median没有独立重拟合，176个原始NPZ仍保存在服务器。CPU复核新增CP/NN/AD/训练均0。

下一项015比较同预算继续C1形状头与冻结随机头后训练小型零初始化均值适配器；先检验固定人口假设下可辨均值残差，再按冻结门决定是否训练。均值残差不等于真实条件均值已识别，基线DC仍单列为电子学限制。014正式报告、三幅英文图和六份CSV已完成内容核验及代表性渲染检查。

[014中文报告](reports/wp2_014_20261015/N01_WP2_014_REPORT.md) · [014 HTML](reports/wp2_014_20261015/N01_WP2_014_REPORT.html) · [014最终回执](receipts/wp2_014_20261015/FINAL_RESULT_RECEIPT.json) · [开发选择](receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json) · [独立闭合](receipts/wp2_014_20261015/postrun_independent_02/CLOSEOUT_REVIEW.json) · [原生终点与计数](receipts/wp2_014_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [015准备设计](wp3_015_design.json)

以下保留此前阶段说明。

<!-- WP2_014_STARTED_20261015 -->
## 当前作业：WP2/014 C0/C1两种子配对短试

014已通过首次真实AdamW更新与同一快照健康检查。固定4trial串行、每次2000G，最多8000G；新AD最多9872，已与013零AD成本分开登记。保护A/005，C2关闭，硬ADC/first-argmax和013冻结数字T保持不变。运行期计数以最后持久化核验为下界，完整结果待回收。

已设置2026-10-15 10:30（Asia/Singapore）单次回收唤醒，期间休息。[014配置](wp2_014_config.json) · [014固定方案](docs/WP2_014_CONFIGURATION.md) · [启动回执](receipts/wp2_014_20261015/EXECUTION_STARTED.json) · [013报告](reports/wp1_013_20261015/N01_WP1_013_REPORT.md)

<!-- END_WP2_014_STARTED_20261015 -->

<!-- WP1_013_CLOSEOUT_20261015 -->
## 当前进展：WP1/013已独立闭合，推进WP2

013计算与监督均原生退出0，265文件库存及保存数组独立读回通过。固定总梯度方向通过全部12格有限步方法规则，选择adequate；数字容忍度已在014候选之前冻结。当前010头完整人口wave改善为0.9162%/0.9914%/1.6907%，均低于2%门槛，形态和基线均值仍有绝对偏差，科学状态保持response_match_not_established。

本轮训练与新AD均0，累计科学训练159667；前三次attempt全部消耗保留。下一步是C0/C1各2种子、每次≤2000G的配对短试，C2关闭。六工作包主计划继续执行。

[013中文报告](reports/wp1_013_20261015/N01_WP1_013_REPORT.md) · [数字T表](CALIBRATED_TOLERANCES_WP1_013.json) · [SOURCE审计](INPUT_INFORMATION_AUDIT.json) · [完成回执](receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json)

<!-- END_WP1_013_CLOSEOUT_20261015 -->

# N01：013 到项目交付主实施计划正在推进

013 的实际重复纯信号保存修复已在新服务器目录启动，首批与当时无失败均已核验。 前两次原生失败、25/38文件现场和各52744行参考计算成本保留；此次保存全部实际A输出并使用原数值回放标准。 本次训练更新和新 AD 查询均为 0；累计 N01 科学训练更新仍为 159667。

主线保留确定性平均响应与输入相关随机响应，固定 256 点硬观察输出。六个工作包已统一登记：013 收口与评价合同、目标与代理优化、受控响应模块、已知响应恢复、稳健确认、软件与论文材料交付。短候选最多 12 个，教师试验最多 8 个，最终确认预算单列。

当前 013 使用全部 6593 个 SOURCE TRAIN 父事件与 21047 条 REAL TRAIN 记录、两个独立随机库和三种电子学假设；仅验证 5 个新头参数方向及 3 个预定电压尺度，不因局部末位差异追加审计。原 DEV 保留已暴露历史；422 个原 REAL 组来自同一采集。

工程可用、在域有效、跨条件物理复用分别判定。科学状态仍为 **response_match_not_established**；跨采集复用尚未验证。

[用户主计划](USER_MASTER_PLAN_20261015.txt) · [工作包与预算](MASTER_EXECUTION_PLAN.json) · [观察合同](OBSERVATION_CONTRACT.json) · [数据角色](DATA_ROLES.json) · [实用误差合同](docs/ACCEPTANCE_CONTRACT.md) · [013 冻结配置](wp1_013_config.json) · [当前作业](jobs.json)

以下节点保留形成时的结果与作业状态。

<!-- N01 WP1 CURRENT STATUS END -->
## 历史完成节点：012 新头梯度与裁窗机制诊断

截至 2026-10-15，009 配对诊断、010 新头固定 2000 G 试验、011 的 500 次已存 TRAIN 库配对重加权、012 局部机制诊断均已完成并读回。当前科学作业为 0；012 新增训练更新为 0，累计科学训练更新仍为 159667。

012 完成 9 次 scalar AD 和 2 次 masked endpoint VJP，A、005 和 final 新头各前向 2048 行；2163 项独立保存输出核验和 32 项报告检查通过。固定 TRAIN 点的加权 QPSD 梯度约为波形项的 19 倍；单一 carrier 方向的 0.1 LSB moving-crop 中心差分与代理梯度反号，1/10 LSB 不反号。下一项先用保存输出诊断裁窗边界，训练预算为 0。

科学状态保持 **`response_match_not_established`**；这些结果描述固定 TRAIN 库和指定方向的局部机制。

[012 诊断报告](reports/mechanism012_20261015/N01_MECHANISM012_REPORT.md) · [012 最终结果回执](receipts/mechanism012_20261015/FINAL_RESULT_RECEIPT.json) · [独立数值读回](receipts/mechanism012_20261015/postrun_independent_01/NUMERIC_READBACK.json) · [报告检查](receipts/mechanism012_20261015/postrun_independent_01/REPORT_QA_FINAL.json) · [GitHub 核验回执](receipts/mechanism012_publication_20261015/PUBLICATION_RECEIPT.json)

[010 试验报告](reports/shape010_20261014/N01_SHAPE010_REPORT.md) · [011 配对重加权结果](receipts/stability011_20261014/user_server_pause_01/native_result_json/STABILITY_RESULT.json) · [009 结果](docs/MEAN_RANDOM009_RESULTS.md)

## 历史登记：009 平均响应与随机响应诊断

用户已授权设置计划并推进。本轮先做固定终点的配对 A/B/C/D 诊断：512 个 TRAIN 源父、每父 16 次，三种噪声假设，新增 G/D 更新均 0。[执行计划与协议](docs/MEAN_RANDOM009_PROTOCOL.md) · [登记回执](receipts/mean_random009_20261014/REGISTRATION_RECEIPT.json)。

009 开始时登记的阶段为实现、人工数值检查和源码冻结；启动后安排单次苏醒。其后 010 固定 2000 G 试验与 011/012 诊断已完成，最新状态见本文顶部。

## 已完成的 Window008 resume01

三组模型已达到各 19,000 G 的固定终点，合计 57,000 G / 144,000 D。256 点主评价回放与补充诊断回读通过，10 页报告已逐页核验。当前科学作业为 0，定时苏醒已暂停。

**物理响应匹配尚未建立**（`response_match_not_established`）：完整随机臂在 TRAIN、DEV 的主要整波形距离上仍劣于确定性对照。本轮训练完成、数值复核通过与科学目标验收分别记录。

[008 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-window008-resume01-20261014) · [10 页 PDF](https://github.com/Aggressivecyber/ej276d-n01/releases/download/n01-window008-resume01-20261014/N01_WINDOW008_RESUME01_REPORT.pdf) · [发布内容与使用方法](docs/WINDOW008_PUBLICATION.md) · [完整交付回执](receipts/window008_resume01_20261013/FINAL_DELIVERY_RECEIPT.json)

Release 提供报告与完整数值证据、冻结源码、三组推理权重及复用的 noise005 参数，附 SHA256 清单。2026-10-14 用户已授权本次 GitHub 发布；此前的本地交付回执保留其形成时的状态。

原008数值失败和fix01用户停机的现场与计数均已保留；Window008实际历史为57,667 G / 146,001 D，区别于最终有效终点计数。当前状态以 `PROGRAM.json`、`jobs.json` 和最终交付回执为准。

# N01｜物理分解式条件随机 1D U-Net

同源Gamma有效分类窗已核实为峰前16点、峰后239点，1 ns/点；原始采集记录为10000点。本轮已统一256点的观察、损失和Q/PSD定义，保留N01原电压处理口径。原480点续训被本轮替代。见 [008协议](docs/WINDOW008_PROTOCOL.md)、[冻结源码](WINDOW008_SOURCE_MANIFEST.json) 和 [窗口范围登记](receipts/planning_008_20261013/USER_SCOPE_WINDOW_ALIGNMENT.json)。

以下为历史节点；其中“当前科学作业数”只描述各节点关闭时的状态。

**007有限阶段已完成并核验：两臂各4,000 G，累计8,000 G（本执行新增7,999）及6,000训练D；原每臂19,000 G末点尚未完成，响应匹配未建立，当前科学作业0。** [三页报告](reports/response_007_20261013/deadline01/N01_RESPONSE007_STAGE_REPORT.html) · [完成回执](receipts/response_007_20261013/deadline01/FINAL_RECEIPT.json) · [比较口径与下一步](reports/response_007_20261013/deadline01/README.md)。本轮由主代理实施与自检。

**此前 006 已完成：接口检查通过，但低电荷 PSD 展宽对长程假设敏感；新增训练 0 次，当前科学作业 0。** [完整480点与PSD整合报告](reports/integration_006_20261012/N01_INTEGRATION006_REPORT.html) · [006 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-integration006-20261012) · [冻结协议](docs/INTEGRATION_006_PROTOCOL.md) · [完成回执](receipts/integration_006_20261012/FINAL_RECEIPT.json)。


**005 已完成：噪声更新 5000 次，响应更新 0；TRAIN ACF 匹配 True，改善 True；候选开发筛查 True，legacy 总门 False。控制测量状态仍不确定。当前科学作业为 0。** 见 [005 报告](reports/noise_acf005_20261012/README.md)、[分离评价回执](reports/noise_acf005_20261012/EVALUATION_RECEIPT.json)、[100/480 支持范围](reports/noise_acf005_20261012/SUPPORT_SCOPE.md) · 权重、日志和复核资料：[005 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-noise-acf005-20261012)。 和 [005 协议](docs/NOISE_ACF005_PROTOCOL.md)。

**前次已完成：noise_improvement_004 实际训练 5000 次，OLS 联合误差由 0.57436 降至 0.14193，通过原阈值；ACF 最大差 0.02185 仍超过 0.015，REAL–REAL 对照区间检查也未通过。噪声总门仍失败，响应训练为 0，当前科学作业为 0。** 见 [改进报告](reports/noise_improvement_004_20261012/README.md)、[004 协议](docs/NOISE_IMPROVEMENT_004_PROTOCOL.md) 和 [完成回执](receipts/noise_improvement_004_20261012/FINAL_RECEIPT.json)。本候选仅支持 100 点观测基线，不支持 480 点响应整合。 权重、日志和复核资料已上传 [004 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-noise-improvement-004-20261012)。

**前次诊断完成：21,047 条 TRAIN 原始→处理链逐位重建；128 次 REAL–REAL 全重新拟合完成，65,536 个分数重放逐位一致。本轮噪声/响应训练均为 0，评价分类器 25,600 更新。**

来源负相关早于已知处理链，固定取窗平移可复现增强；旧阴性对照判据的条件误报为 9/128，组内置换为 7/128，差异不足以证明后者更优。上一轮失败门不改判。见 [噪声来源与对照校准报告](reports/noise_origin_audit_003_20261012/README.md)、[预登记协议](docs/NOISE_ORIGIN_003_PROTOCOL.md) 和 [本轮 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-noise-origin-003-20261012)。

**上一轮修复状态：电子学与损失 V2 改造已完成；独立噪声训练 5000 次后的预注册门未通过，已按规则停止，新脉冲训练为 0。完整响应匹配尚未得到支持。**

上一轮中文报告见 [电子学修复与独立验证](reports/repair_002_20261012/README.md)，规则见 [repair_002 协议](docs/REPAIR_002_PROTOCOL.md) 与 `repair_run_config.json`。失败项为 ACF、基线均值—斜率联合统计和 REAL–REAL 阴性对照区间。确定性 / 完整随机两臂及独立终点评价代码已实现并测试，但由于噪声门未通过，本轮未运行这两臂。科学计数分别为：旧 run_001 共 77000 更新；本轮噪声 5000、响应 0；本轮 1200 次分类器评价更新另计。

原 run_001 四组固定终点中文报告见 [完整模型评估与四张科学图](reports/run_001_20261012/README.md)；原交付见 `receipts/run_001_20261012/FINAL_EVALUATION_RECEIPT.json`。它在 2026-10-12 07:00:28（Asia/Singapore）正常退出。当前科学作业为 0。

初始方案依据是用户提供的 [原文](USER_PROPOSAL.md)，后续由用户的噪声与损失审查修订。用户明确要求：**直接创建 N01，按粘贴方案，不继承 R79**。本文第 1 节以下及 `config.json` 保留初始设计；整体修复协议为 `docs/REPAIR_002_PROTOCOL.md`；最新独立噪声候选为 `docs/NOISE_ACF005_PROTOCOL.md` 与 `noise_acf005_config.json`，完成状态见 `PROGRAM.json`。原文的建议值不视为已验证最优值。

公开仓库：[https://github.com/Aggressivecyber/ej276d-n01](https://github.com/Aggressivecyber/ej276d-n01)。四组模型、检查点、日志及逐事件评估产物见 [固定终点 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-run-001-20261012)；上传范围与 SHA256 见 `PUBLICATION_MANIFEST.json`。

本轮噪声权重、逐步日志、分类器、复核结果和完整源码包见 [repair_002 Release](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-repair-002-20261012)，其中明确保留噪声门未通过状态。

## 2026-10-12 机制复核

已完成冻结模型分窗口分类器、同随机数斜率干预与固定投影人工盲区复现，见 [诊断报告与可运行入口](reports/noise_audit_20261012/README.md)。确定性模型条件 Q 方差中位数在关闭随机斜率后下降约 97.7%；归一脉冲域 AUC 仍为 0.982。核心响应模型与原结果未改，此诊断响应训练更新为 0；24 个诊断分类器共 4800 更新。随后进行的电子学 V2 运行已结束，其未通过结果见上方最新报告。

## repair_002 实现与复现

- [电子学 V2](docs/ELECTRONICS_V2_IMPLEMENTATION.md)：自由 FIR / L2 归一化，不把短窗 OLS 散布直接复制为全窗随机斜率。
- [分布目标 V2](docs/LOSS_V2_IMPLEMENTATION.md)：每更新刷新投影、双遍共享上下文、独立评价方向、逐坐标和短延迟联合分布。
- [条件两臂评价](docs/REPAIR_EVALUATION.md)：严格绑定实际通过的噪声门、权重与数据身份，当前没有本轮脉冲终点。
- [噪声协方差与可辨识性分析](docs/REPAIR_NOISE_IDENTIFIABILITY.md)：冻结 K 的潜在漂移相减非半正定，前 100 点不足以识别 480 点真实漂移。

运行入口为 `python -m n01.train_repair --help`，仅在用户授权并且新的独立噪声门确实通过后执行响应阶段。本轮固定终点、配置、全部失败指标和复核回执均保留，不能更改本次门槛后将同一结果重标为通过。

## 1. 目标与依据

在明确的仿真输入定义和探测器条件下，学习事件相关的随机响应模型，使同一仿真事件产生合理的幅度、相对时间、形状与噪声变化，同时区分探测器响应和两批数据的能谱/组成差异。

当前 SOURCE 是解析期望波形，含固定衰减模板与 `0.15 × had` 可见能量代理，已对齐裁窗。首先研究的是“相对于当前解析仿真输入的有效随机响应”；不能把学习参数直接称作独立测得的增益、淬灭或 PMT 传递函数。当前只有一份历史实测采集，人工分组不构成独立采集。

N01 无父协议。旧 R79/R80 设计、旧模型参数、v5/A–F 验收及旧固定运行数量均不自动适用。既有数据与采样/积分接口可以按来源引用，历史结果保持原样。

## 2. 输入与数据

- 输入 `x: B × 1 × 480`，提取基线 `b_x`、源窗口电荷 `Q_x` 和脉冲幅度 `A_x`。
- `u_x=(x-b_x)/A_x`；编码通道为 `[u_x, Δu_x, t_norm]`。
- 独立标量为 `[log(Q_x/Q_ref), log(A_x/A_ref), d]`；只有一种已知工况时 `d` 为固定配置，不编造工况变化。
- 不把父 ID、文件名、行号或批次位置送入网络。沉积组成可用于来源审计、趋势和单独的人口比较模型，不冒充两域共有真实粒子标签。
- 使用全部合法训练数据，保留原训练/验证划分，不人为挖空能区。已记录的 TRAIN 数量为 SOURCE 6593、REAL 21047；创建阶段仅核对元数据；正式运行使用全部合法 TRAIN，固定终点才加载既有验证划分。
- 小批次遍历覆盖全部合法训练父事件；“使用全部训练数据”不要求每一步全人口计算。

文件与字段引用见 `DATA_REFERENCES.json`。旧容器中保存的 `lambdas`、旧分箱、倾斜采样及旧预生成训练流不属于 N01 初始训练设置。幅度有效性阈值、数值保护和实际读出接口在实现时明确定义，异常事件留账。

## 3. 模型

### 3.1 主干

| 层级 | 尺寸（不含 batch） |
|---|---:|
| 输入 | 3 × 480 |
| E0 | 32 × 480 |
| E1 | 48 × 240 |
| E2 | 64 × 120 |
| E3 | 96 × 60 |
| E4 | 128 × 30 |
| D3/D2/D1/D0 | 96 × 60 / 64 × 120 / 48 × 240 / 32 × 480 |

残差块采用 `Conv1d(k=5) → GroupNorm(8) → SiLU → FiLM → Conv1d(k=5) → GroupNorm(8) → SiLU`。下采样前固定平滑低通；上采样采用插值后卷积；瓶颈膨胀率为 1、2、4。FiLM 条件向量 128 维。

### 3.2 平均电荷与联合随机响应

- 以 `log Q_x` 为主轴，归一化形状与已知条件为条件，采用 8 区间低自由度单调样条预测 `Qbar`，初始为恒等映射或经审查的简单增益基线。
- 对固定形状的单调性必须成立；用于生成样条参数的形状条件编码不能通过另一路无限制读入 `Q_x` 或幅度改写单调映射。
- `z_phys ~ N(0,I_10)`；下三角矩阵 `L` 预测 55 个参数，得到 `[eta_Q, eta_a, a_1,...,a_8] = L z_phys`。
- `Q_sig = Qbar exp(eta_Q - sigma_Q²/2)`。对数正态是第一版假设；均值关系只在相应未截断参数化成立时适用。`Q_sig` 是加入电子学噪声前的约定窗口信号电荷。

### 3.3 时间和形状

- 共享 65 点因果 FIR `h_d` 主要由探测器配置决定，初始为单位脉冲或简单基线滤波器。
- 学习正的时间伸缩 `a`，以固定对齐参考点施加 `T_a[s](t)=s(t0+(t-t0)/a)`；绝对时间平移头关闭。
- 解码器输出全分辨率平均残差 `r_mu` 和 8 条随机形状基。随机基先预测 60 个控制点，再平滑样条展开到 480 点；范数归一化并弱惩罚基间相关性。
- `v = T_a[h_d*u_x] + r_mu + Σ a_k B_k`。基函数不未经验证命名为具体物理过程。

### 3.4 两条噪声分支

- 电子学分支：`n_e = b0 + b1*t + sigma_e*(h_e*epsilon_e)`，65 点相关滤波，允许事件间幅度变化；主要从合法 REAL TRAIN 基线拟合，第一版不看完整脉冲或粒子代理标签。
- 局部信号分支：`n_s0 = h_s*[sigma_s(t|x,Q_sig)*epsilon_s]`，网络预测平滑幅度包络。
- 局部信号涨落投影为零窗口电荷，避免和 `Q_sig` 重复计算电荷方差。该限制不作用于电子学噪声。
- 延长随机序列再截取有效卷积段，防止 FFT 循环卷积使尾噪声绕回前端。

### 3.5 组装与读出

`u_hat = v/Q(v)`，`y_preADC = Q_sig*u_hat + n_s + n_e`，`y = A(y_preADC)`。

因此 `Q(n_s)=0`，一般 `Q(n_e)≠0`，并有 `Q(y_preADC)=Q_sig+Q(n_e)`。不得把最终含噪波形再次强制投影到 `Q_sig`。不用加减平均模板代替电荷与形状分解。`Q(v)` 非正或接近零时记录无效生成并追查，不以任意裁剪隐瞒。共同数字化、预处理、饱和和窗口损失均留账。

## 4. 无配对目标与事件人口

源电荷 `Q_x` 和实测电荷 `Q_y` 不是天然同条件，不能按相同数值直接配对或视为物理对应。

- 网络条件：完整源波形、源电荷、真实已知探测器配置。
- 输出条件比较：生成输出自己的电荷与实测电荷，使用相同定义。
- 来源审计：原父事件、原输入条件、沉积组成和原物理权重。
- 事件人口：单独低自由度比较权重，可随源电荷及已知沉积组成平滑变化，单独保存；不删事件或挑选输出。

原始人口和条件可比后的结果分别报告。权重形式、参数边界、拟合顺序和人口敏感性设置尚未由原文指定，将在 N01 实现时绑定，不读取旧倾斜表作为现成答案。没有独立标定或谱形信息，响应核与人口可能互相补偿；合理人口假设下响应不稳定时，应报告当前信息不足，不能只扩大神经网络。

随机前向响应不强制对 REAL 恒等。保留 REAL–REAL 和共同观测处理对照；它们不等于将 REAL 再送入前向响应模型后逐点不变。

## 5. 初始损失

各项使用合法 TRAIN 尺度无量纲化，记录梯度范数与夹角。

| 损失 | 内容 | 初始权重 |
|---|---|---:|
| wave | 完整电压、归一化形状、多尺度差分与联合投影分布距离 | 1 |
| QPSD | 联合 Q–PSD 和输出电荷条件 PSD 分布 | 2 |
| noise | 基线 RMS、ACF、频谱份额、幅度分布、局部相关 | 1 |
| adv | 全局/局部双头波形判别器 | 0.1 |
| regular | 平均残差幅度、平滑性、基函数退化 | 0.01 |
| trend | 有依据的条件趋势，例如固定形状下平均电荷单调 | 0.1 |

判别器为小型 1D CNN，通道 32、64、128，全局与局部头，谱归一化。评价域分类器独立于训练判别器。

不使用随机 SOURCE–REAL 配对的逐点 MSE、逐事件 PSD 严格排序、源群体禁止进入实测谷区或无上限鼓励多样性。条件样本不足时用同一参数版本的较大分布批次或跨微批联合计算，不混入旧参数生成缓存。

## 6. 训练与对照

| 阶段 | 建议更新数 | 内容 |
|---|---:|---|
| 基线 | 1000 | 电子学噪声及简单响应基线 |
| 平均响应 | 3000 | 平均电荷、共享滤波、平均形状 |
| 联合随机 | 12000 | 联合随机头、形状基、局部信号涨落与分布目标 |
| 低学习率 | 4000 | 相同结构，降低学习率，完成固定终点 |

原文建议合计 20000 更新；这是设计建议，实际完成量见第 9 节；更新数本身不构成收敛或科学有效性证明。对照各阶段具体开放项及更新统计在实现时明示。

- 每步 SOURCE 父事件 128、每父响应 2、REAL 波形 256。
- G 学习率 `2e-4`，D 学习率 `1e-4`，G weight decay `1e-4`，梯度裁剪范数 5，D:G 更新比 1:1。
- 网络 float32，电荷与关键归约 float64。实际显存和步时实测。
- 三组对照：简单随机响应基线；确定性 U-Net 加相同电子学噪声；完整随机 U-Net。三组使用相同合法训练数据，披露实际优化量与运行成本。简单基线也允许合理的随机展宽。
- 原文未指定初始化次数和种子列表；本次不自动加入 R79 的三个初始化，也不自动继承旧 25000/50000 步末点。

## 7. 验证

1. 联合检查完整电压、归一化波形、Q–PSD、条件峰位/原始峰宽、基线及相关噪声。原始人口和条件可比比较分开。
2. 固定 128 个仿真父事件，各生成 64 次，记录 `Var(Q|x)`、`Var(PSD|x)`、`Cov(Q,PSD|x)`、相对时间、波形协方差与噪声统计。重复响应不是独立父事件；无重复实测真值时不能声称逐输入条件方差被唯一识别。
3. 固定随机变量，在相近源电荷下替换归一化形状或改变输入内容，再检查尾部、上升沿和 PSD；用重复响应均值辅助确认。父 ID、存储顺序改变不应影响对应响应。仅做全局输入打乱后看总体分布不构成源依赖性证据。
4. 冻结响应后改变仿真能谱/组成，做内部人口敏感性验证；跨采集、跨源复用需要相应独立数据。

原文没有给出固定分类器通过线、4000 条重放门或已知响应恢复前置实验。本次不补入这些历史要求。必要数值自检随实现完成；定量评价口径在 N01 自身记录，不把旧阈值自动当成新方案验收。

## 8. 未指定的实现细节与下一步

原文尚未给出：随机种子/次数；精确分布距离与对抗损失；人口权重形式/边界/敏感性案例；单调样条与平滑样条细节；有效性阈值；局部噪声核；各阶段冻结策略与低学习率数值；条件区最小支持、统计不确定性和量化验收口径。

这些细节现已绑定在 `run_config.json`、`docs/TRAINING_IMPLEMENTATION.md`、`docs/MODEL_IMPLEMENTATION.md`、`docs/LOSS_IMPLEMENTATION.md` 和 `docs/EVALUATION_IMPLEMENTATION.md`。初始提案配置另存 `provenance/initial_config.json`；当前配置记录已实现的选择，不继承 R79 流程。

## 9. 存储与当前交付

N01 根目录为 `E:\GAN\research\N01`。模型权重、逐事件输出、数据、缓存与日志保存在 E/D，不进入版本库。当前已实现模型、训练器和终点评价。2026-10-12 00:51:53（Asia/Singapore）在 100.82.231.102 启动正式训练，PID 478257；实时位置与最后核实状态见 `jobs.json`，启动证据见 `receipts/run_001_20261012/START_RECEIPT.json`。正式运行结果保存在服务器独立目录，本地入口、代码、回执与备份在 E/D。固定终点评估现已完成：部分 QPSD 统计改善，但域 AUC 约 0.9874、基线均值跨事件 SD 约为 REAL 的 3.89 倍，条件峰宽和总体敏感性仍有残差；科学状态为 response_match_not_established。

资源仅继承工作环境的机器安全限制：服务器 CPU12、RSS48GiB、单 GPU allocated/reserved 各 10GiB；本机辅助共享 CPU4/RSS4GiB；一次一个科学作业。后续运行前核对实时资源。旧研究的科学计数、计算预算、评价门和审批链均不因数据复用而继承。

运行安排：共享基线 1000 更新，随后完整随机、确定性、简单随机三组各 19000 更新；另以同种子固定均匀比较权重重训完整模型，检查总体假设敏感性。每臂名义 20000，实际独立优化更新合计 77000。四个 128 父事件×2 响应微批联合形成同版本 1024 波形分布批次。未实测的物理事件权重保持 unknown；全 1 仅表示原始源记录的均匀计数参考。

读出归因已[正常结束并独立核验](receipts/joint_response_01_20261019/wake_20261019_1010_01/TERMINAL_REVIEW.json)：exit0、360.13秒、631源码和12输出哈希通过，216差分列与12探针、零更新；432个联合视图差分列、648个native视图差分列及全部24/36个探针向量已重算。冻结峰使native wave两尺度导数余弦约0.998–0.999，但联合与QPSD仍不稳定，连续固定峰仍保留较大步的预测误差。原硬读出和科学门不改。

据此登记[生成器与指标非线性归因](docs/JOINT_RESPONSE_01_SIGNAL_METRIC_ATTRIBUTION_20261019.md)：沿同两尺度保存方向比较实际生成信号与其有限尺度线性近似，通过原硬读出及原诊断视图复算；固定原FIT与随机库，216信号割线列、12实际探针、零更新。保存全部行的信号、噪声和完整误差向量；输入身份、原实际探针精确重放、数值分解和预算为执行验收依据。内部300秒/外部420秒，输出384MiB。具体启动与终止状态以PROGRAM/jobs及该任务回执为准。

[生成器/指标归因正确启动](receipts/joint_response_01_20261019/signal_metric_attribution_01/STARTUP_ASSESSMENT.json)已确认：640份源码部署及原生导入、派发前逐文件核验通过；10:24:01 UTC派发时GPU可用11605MiB，无其他N01进程。10:24:21 UTC科学PID207499/start_ticks6722131，已完成162信号割线列、8个实际探针精确重放和两个端点精确重放，零更新。该运行已完成，终止审核见下文。

生成器/指标归因已[完成并独立核验](receipts/joint_response_01_20261019/wake_20261019_1029_01/TERMINAL_REVIEW.json)：exit0、25.31秒，640源码/12输出哈希通过；全部296数组、10240角色行、216列前8行割线加权、12个全行信号误差、319488行独立NumPy硬读出及24/36个joint/native分解通过。四个大步的生成信号近似误差为实际变化的37.7%–54.6%，12个原硬wave比较中该项影响均大于剩余读出/指标项；小步仍有离散读出误差。零更新，原科学失败保留。

已前瞻登记[联合割线校正与局部信赖域试验](docs/JOINT_RESPONSE_01_SECANT_TRUST_DIAGNOSTIC_20261019.md)：固定两个神经端点、原0.004矩阵和已暴露FIT探针，以实际误差校正整个联合矩阵并在0.0125初始半径重解方向；失败只用FIT纠正，半径减半。每种子最多8个主支路候选，另1个只改半径首步对照，新CHECK/FRESH库及joint/native投影。保留原全部守门与M0锚；零提交更新，最多18求解/候选、52生成器calls，内部300秒/外部420秒。方法检查不替代科学成功。

联合割线试验首次执行于10:45:43 UTC派发，6.41秒后因第4次SLSQP求解的线搜索错误退出。645源码/9输出哈希及已保存首步对照的数值审核通过，3次候选实际检查、零提交更新；中途两次校正候选明细因仅在支路结束保存而缺失，计算次数及报错保留。[失败审核](receipts/joint_response_01_20261019/secant_trust_01/TERMINAL_REVIEW.json)。独立16次CPU求解检查中，原变量有1次同类失败，按半径归一化的8个等价问题全部可行；原第4次矩阵未保存，不能称其精确故障重放。拟单独登记变量尺度和逐试探持久化修复，原门不变。

已单独登记[FIX01工程修复](docs/JOINT_RESPONSE_01_SECANT_TRUST_FIX01_20261019.md)：只将求解变量按信赖半径归一化，并在每次求解前及实际检查后保存明细。原solver、模型、硬观察、科学接受门、候选上限及资源预算不变；另记18次方向尝试上限。重新加载与前缀重复计算费用计入新作业，原失败保留，零提交更新。

[FIX01正确启动](receipts/joint_response_01_20261019/secant_trust_fix01/STARTUP_ASSESSMENT.json)已确认：650份部署文件及原生导入、派发前逐文件核验通过；10:53:36 UTC派发时GPU可用11605MiB，无其他N01进程。10:53:43.620431 UTC科学PID213483/start_ticks6899668，首个端点精确复现，3次候选检查完成，正在第4次方向尝试，零更新。该时刻健康运行；现已结束，完整终止审核见下文。

FIX01已[正常完成并独立核验](receipts/joint_response_01_20261019/wake_20261019_1059_01/TERMINAL_REVIEW.json)：exit0、13.46秒，650源码/12输出哈希，163数组、30720角色行、24矩阵/24预测向量、10次割线校正、93风险视图及25接受判定通过。12次求解均成功，零更新；seed0八个主支路候选均拒绝，seed1第二候选通过FIT/CHECK但FRESH失败。seed0当前端点在FIT/CHECK的wave相对M0最高1.01567/1.01526，原方向只约束相对当前非增，未表达所需M0恢复量；seed1的FRESH joint比值0.93105，但QPSD相对当前1.01109/1.01284及slow wave相对M01.010235触发原拒绝门。新的批次与投影同时改变，不能仅据该结果断言是哪一种随机性造成失败。下一有界方法检查拟从原M0零增量起点建立三拟合批联合矩阵与多native投影约束，保留新CHECK/FRESH和原门，仍不提交训练。
