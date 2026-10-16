# WP4 020：恢复019已完成seed0，仅训练未启动seed1

最新直接用户授权为“恢复 后面允许你自动重启以及新增”，原文及持续授权冻结在receipts/wp4_020_20261016/authorization_01/USER_AUTHORIZATION.json。020是原T1两seed无配对任务的工程恢复，不新增teacher、候选或科学门。原017的pending、018已通过资格与019 native1失败事实保留。工程修复和必要新增计算已授权；每次仍须前瞻登记、独立核查、单job和明确预算。当前native入口只执行一次，无内部盲重启循环；后续必要工程恢复由root按持续授权处理。

## 固定恢复路径

1. 核验018最终独审资格、019最终失败回执、最新授权、全部绑定源码/输入和原005/cache身份。
2. 005 CP加载一次，原A/B/Q只读mmap；不加载A/M0 CP、不构造或前向A/M0网络，不打开原SOURCE、REAL或DEV波形。
3. 加载019/T1_unpaired_seed0/final.pt一次：114968B、SHA256 ad285f938745d906df7295433ea9d1c3bc83d181e1c26db16375df7f2af48b79。严格核schema/run019/trial/teacherT1/modeunpaired/seed0/G2000/complete_step_boundary、无inflight、原cache/frozen绑定及qref。只将两个theta及固定qref载入CachedFrozenQ1Charge017并禁止梯度；不创建seed0 Engine、优化器、父采样器或训练随机流，不恢复optimizer/RNG状态，不重训seed0。
4. 先用该模型完成seed0原固定final评价，确认实际硬观测和诊断入口可达后，再创建seed1原Engine017。seed1保持原初值、unpaired算法、随机namespace、5274拟合父、2000G/AD、G1000/G2000正常CP两份；不改变T1真值或任何损失/尺度。
5. 完成seed1相同final评价，复用019成熟评分流程和018完整资格规则。所有新产物在020新root，旧文件不修改。

seed0在来源019已完成的2000G记作source019_G_updates；020新增G只来自seed1，终态必须恰为2000。恢复trial记录source019_G_updates=2000/new_G_updates=0，另一个trial记录0/2000；不能重复记入批次或项目累计。

seed0评价失败时记录020 FAILED与已有CP身份，不对无optimizer的恢复模型调用训练checkpoint保存。仅seed1 Engine存在时允许一次全job failure CP尝试，保存错误显式记录并重新抛出原异常，不吞错误或补采样。

## 字节与依赖边界

WP4_020_SAVED_INPUTS.json绑定40个精确输入，共1,141,935,072字节：保留019的37个原保存输入（018最终资格回执在020包内），加seed0 final.pt、019最终失败回执及最新用户授权。三份小回执只读本体，不沿其历史evidence引用扩大读取。005反序列化1次，seed0 CP反序列化1次；其他旧CP及原数据不在数值读取路径。

科学源码保留019的30份并增加run_wp4_020.py与遗漏的mechanism012_losses.py，共32份；run_wp4_019.py作为工厂、teacher_data和gate签名库复用，其文件不改。mechanism012_losses使用旧017已冻12854B原字节，SHA256 3fc84ccaf255b660d8eda4f96d43e2c236045a756e777c9b2d490f7137dc04ed。其直接5个及递归2个n01依赖已在原包；不为未调用WP1功能搬入其他模块。冻结时source_files必须包括这些实际owned路径，另绑定最小ops/文档/输入清单/证据闭包。

必要工程检查在E盘隔离源包中实际运行固定公开波形的evaluate_kernel017→Runtime017.observe→observation_rows→numeric_flags/peak_diagnostics→codec/native32报告；屏蔽仓库fallback，拦截CP/NN/AD/优化器/随机抽样。另以人工内存CP检查只恢复参数及错误来源元数据拒绝行为。这是新增恢复语义和漏依赖入口验证，不重跑旧工程套件，也不是科学资格评价。

## 随机流和硬观察不改

6593父、5274拟合/1319留出、每bank32draw、两个bank、三个噪声假设、256输出、384载体、prefix100、1软件ns、原ADC/firstargmax/crop/tail46均保持。所有原父、原权重与原分母保留。教师latent不成为部署输入，教师已知a/b不写入真实候选。

seed1训练Gaussian仍kind=training,replicate=1,role=student/teacher；父流仍kind=parent,mode=unpaired,replicate=1,role=student0/student1/teacher0/teacher1。原seed017哈希函数及完整初始seed表从019复用。学生final仍kind=evaluation,role=student,bank=0/1，从相同前缀重新生成；不把新run_id加到随机namespace。

019失败的首128行信号/all3noise已生成，但没有完整hard bank可复用。020完整重算该prefix，其中50432个normal请求、384行假设扩展及已完成的fast128行ADC将与019累计发生重复。下面020预算不扣掉重复工作，来源019的已执行量另记。

## 资格和产物

018的O19确定性映射、cov8、完整257波形energy、时间/覆盖/缺失/截尾、teacher32阴性和MC/支持规则全部不变。δ=.10、MC半宽=.05仍是前瞻经验筛选尺度；95%支持、24+24/48和原1%状态差门不降。完整rise只在定义成立处使用；条件下界不声称窗外交叉存在或恢复完整rise。

新final逐hard256重建scalar9/native32矩和O19；teacher/initial与原PIT按原SHA复用，不新增teacher/initial前向或RNG。full257距离重新计算，O19重算一次teacher64 self及两final cross/self；保留有符号U值，不夹零。

调用原019真实unpaired签名规则，每trial1272格、两trial2544格，输出UNPAIRED_GATE020.json。任一pending/fail均不合格，native0与科学通过分别记录；任何结果均不自动打开T2/旧stage2，不构成硬件、REAL或窗外响应验收。

INIT_PROOF.json、RESTORED_SEED0_PROOF.json与FIRST_UPDATE_PROOF.json均有真实run_id020。恢复证明给CP2、来源G2000、新G0、两theta、cache/005身份及未创建optimizer/sampler；FIRST_UPDATE_PROOF属于seed1，包含两个真实非零梯度与实际delta。旧Store/stream等格式若沿用017，只是implementation schema来源；新输出run_id必须020，来源019 CP不重新标记或覆盖。

## 精确预算和资源

| 020新增项 | 数值 |
|---|---:|
| 新训练trial / G / AD / optimizer steps | 1 / 2000 / 2000 / 2000 |
| CP加载 / 正常保存 / 失败保存尝试上限 | 2（005+seed0）/ 2（seed1）/ 1 |
| student训练行 / teacher训练行 | 各2,048,000 |
| 两final评价student行 | 843,904 |
| student电荷行 / teacher电荷行 | 2,891,904 / 2,048,000 |
| 005生成 / shape缓存组装行 | 各4,939,904 |
| 三假设扩展 / 硬ADC / 寻峰裁窗行 | 各14,819,712 |
| 标量normal请求 | 1,946,322,176 |
| parent流 / randint calls / scalars | 4 / 8000 / 4,096,000 |
| unpaired训练有序距离 | 12,582,912,000 |
| train cross / 去对角self / 计算后排除对角 | 6,291,456,000 / 6,285,312,000 / 6,144,000 |
| full257评价有序距离 / O19有序距离 | 162,029,568 / 405,073,920 |
| A/M0 CP及前向、原SOURCE/REAL/DEV、T2、PIT新抽样 | 全0 |

settings020()/budget_caps020()/validate_settings020()顶层纯stdlib，冻结CFG与机器工厂一致；Store记录entered/completed/attempted/inflight，终态全部计数恰等预算。来源seed0恢复另有restored_seed0_checkpoint_loads=1，不计新训练slot。

CPU12线程、GPU≤10GiB、进程树RSS≤8GiB；全job≤21600s/5120MiB，每trial活跃时间≤7200s、输出≤1536MiB，共享评分/日志≤1024MiB，选收≤128MiB。旧成熟owned-tree监督原语复用，不新增控制器。root核对真实恢复证明和实际硬评价已启动后设置一次未来苏醒并休息，无需等到seed1G1持续监视。
