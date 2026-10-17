# N01 当前状态：038续行已独审通过，等待启动剩余REAL双种子

2026-10-18（Asia/Singapore）实时核对：服务器boot已改变，N01进程0、项目锁空闲。原作业没有保存完整退出、失败或终点回执，按中断处理，实际终止原因和未落盘成本未知。全6593父、两个种子的M0纯信号/三假设硬输出逐位一致及全部A/M0/Q1起点评价已保存，将直接复用。

教师种子0在900秒资源上限闭块，12次接纳；教师种子1保存至均值阶段第13方向，5次接纳，REAL两轨没有已保存进度，是否存在未落盘执行未知。已确认科学Adam方向构造43次（含拒绝方向），接纳17次均属教师；额外在途成本未知。项目累计计算更新下界199712，科学状态仍为 `response_match_not_established`。此前一次苏醒已过时，不把旧PID和台账活动标志当作仍运行。

现按用户“服务器重启了 继续推进”恢复同一038，保存模型、Adam、控制状态及RNG已核对，57个科学函数未改，续行通过独立源码审核；保留旧根、日志与已完成结果，仅继续剩余有界工作，不因中断调整模型、损失或接纳门。

[最新恢复检查](receipts/response038_20261017/resume_20261018_01/REMOTE_TERMINAL_AND_RESOURCES.json) · [用户继续授权](receipts/response038_20261017/resume_20261018_01/USER_RESUME_AUTHORIZATION.json) · [038续行配置](configs/response038_resume01_20261018.json) · [恢复设计](receipts/response038_20261017/resume_20261018_01/RESUMPTION_DESIGN.json) · [独立审核](receipts/response038_20261017/resume_20261018_01/independent_review/SOURCE_REVIEW.json) · [权威状态](PROGRAM.json)

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
