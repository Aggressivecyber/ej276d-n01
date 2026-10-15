# WP4 017：固定已知形状下的条件电荷核恢复

完整数值合同为 `receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json`，SHA256 `b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e`。同目录 `INDEPENDENT_CONTRACT_REVIEW.json` 已通过（SHA256 `31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7`）。本文给出执行入口；公式、门和有限支持以该固定合同为准。

固定原 A、005 和015 seed0 M0的1288参数形状头。一次6593父的正式A前向同时取得32×384 float32 decoded特征，逐128父经过固定头产生实际float64 A/B/Q；decoded逐块释放。A/B/Q只读缓存182336008原始数值字节（另有NPY头和索引），不以016保存A代替原始decoded。Q必须实际有限正值，无替代floor；T1与初值的预期电荷增量RMS差须≥3 ADC LSB，否则停在不可分辨状态。

学生只训练2个全局θ，`a=qref*softplus(theta_a)`、`b=softplus(theta_b)`，qref=3.7103998046755806；初值a=.001qref、b=.00025。T1固定a=.01qref、b=.0025；T2仅将第一个shape系数替换为`.9*Rademacher+sqrt(.19)*Normal`，其余与T1相同。均值适配关闭，B不乘电荷随机因子。这里恢复的是固定已知A/005/M0下的电荷条件核；1288形状参数、均值和时间机制未开放学习。

6593合法SOURCE TRAIN父按固定ID哈希隔离为5274拟合父、1319留出父；所有父评价，原物理权重独立保留或标unknown。A/005历史已使用原TRAIN，留出仅针对本次2θ拟合，不称全流程未见REAL交叉验证。REAL/DEV数值读取均0，ID/文件名/行号/教师latent不入部署输入。

两教师×paired/unpaired×2seed共最多8试验，各2000G。先T1 paired两个seed；所有预声明门通过后才进入其余6槽。任一fail或pending关闭stage2，但前两试验和评价正常完成可native0/COMPLETE，记录`stage1_closed_normally`；正常关门不改成执行失败。完整分支必须恰好8槽，失败保留partial/FAILED/原生退出与entered/completed/inflight，不自动重试。

paired每步512父、每侧2draw；unpaired每侧独立1024父。teacher与student所有shape/charge/005/extension流独立；初始和最终student可用固定CRN比较。优化完整256波形及固定积分坐标的257维proper energy U-score；student self排除对角，分母m(m−1)，cross独立，三hyp平均，一次backward。paired恰好4cross+1unique self距离；unpaired用64行`torch.cdist`块，保留exact U-statistic，不保存巨型差向量，不detach目标。有限负分数保留；firstG仅要求finite、两θ真实非零gradient和actual update。AdamW lr=.01、wd0、clip5；1000/2000保存小CP，初值共享1份，全作业失败CP最多1份。

硬观测保持原14bit、384carrier、实际prefix100、firstargmax[112:144]、crop[p−16:p+240]256、tail46、1软件ns。保存原013成熟code256+实际baseline均值+必要IEEE XOR，使原GPU Y可逐位重开；不存完整随机preADC carrier。原始baseline100、所有flags/低Q/undefined/censor原因、每父每draw身份保留。native32双bank与derived64矩明确区分。

评价每kernel为6593父×2bank×32draw×3hyp。T1teacher和共享initial各一次，8final各一次，T2teacher仅在stage1通过后一次，完整11kernel。每父保存波形均值/对角方差、scalar9有限pair covariance及实际valid计数。全矩阵留remote；≤256MiB选收汇总门、分母、权重/ESS、索引、SHA清单和小CP。总17GiB、shared5GiB、每trial1.5GiB、CPU12/GPU10GiB/RSS8GiB、shared和trial各7200s、全批64800s为硬cap；容量表仅预估，超cap保留失败，不静默删指标。

门覆盖两个T1 paired seed、留出all及3个fit-Q tertile组、nominal/low/high三权重、三hyp。每feature/每pair每bank至少24finite，合计至少48/64；共同父≥64且≥原组95%父数和95%权重、ESS≥原父数一半。无支持为pending，不补0。PIT用该父实际finite nS+1；teacher仅defined targets，先每父求覆盖，再父权重平均。全部原64槽的missing/censor差独立≤.01。

主variance curve采用`RMS=sqrt(sum(alpha*(vS-vT)^2)/sum(alpha*vT^2))≤.10`和最大相对误差≤.20；全波形均值归一RMS≤.30；组内conditional variance trace比误差≤.10。cov9先逐父用teacher64 SD/固定floor归一，再按组加权conditional covariance，相对Frobenius≤.15，不加入父间均值散布。teacher两32bank的同口径差≤.075才resolved。时间均值组差≤1ns、父RMSE≤2ns，teacher split须≤.5ns/1ns。该split是经验分辨率screen，不是64draw oracle误差保证。pairwise缺失cov不自动PSD。

coverage50/90分别容差.04/.03。energy以32draw两bank ES均值，teacher reference为两bank互评双方向均值；同父final−initial的加权均值+1.959963984540054×HC1 parent-cluster SE须<0，excess-reference使用同父差SE，门为max(.02×|reference均值|,3SE)。全组、三重权均需通过。独立a/b恢复精度25%只在scaled Gram特征值比≥1e−4时另报，不替代条件分布门。

运行源API：`settings017()`、`budget_caps017(p, stage1=False)`、`validate_settings017(p, frozen=True)`；控制器`--mode freeze/deploy/start/verify-start/status/collect[/failed]`复用原owned-tree、锁、atomic CP和durable journal。全批16000 G/AD，stage1为4000；原A/005/M0加载3次、原A/head仅6593行。所有精确逐项预算由factory生成并与完整合同逐字核对，未另加探针/REAL/DEV/原始科学重试。

当前交付范围为本地源码、人工工程检查、独审与本地冻结包。按用户最新要求不推送服务器、不查询服务器、不启动017科学任务；上述native接口仅作为源码交付保留。正式CP/SOURCE/REAL/DEV读取、正式NN/AD/G/RNG及GPU执行均未发生；人工检查不建立教师恢复或科学接受结论。
