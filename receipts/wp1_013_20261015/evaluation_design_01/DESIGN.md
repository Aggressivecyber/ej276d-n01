# WP1 统一硬观测与一次有界参数方向验证

这是供根流程登记、冻结的实现设计。依据 `USER_MASTER_PLAN_20261015.txt`（SHA256 `56190995374eb2a765a3c8589736514154dba7ff33aa1309b8b1de493e846ebb`）与 main `c9ad769697b982b607ccec320da9b727ad37b978`。本轮只读代码、JSON、NPZ头，没有波形数值解码、CP加载、NN/AD或科学派发。

## 已核事实和可执行方向

`src/n01/train_shape010.py:228–239` 在每个薄CP保存 `new_head`、完整 `optimizer.state_dict()`、samplers、显式streams、policy、计数与冻结state SHA。已有只读 `TRAINING_READBACK.json` 记录 final 的两组 moments 为1280+8参数，step均2000；lr=0.0002，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.0001，amsgrad/maximize=false。本轮没有再次decode CP。

012 `GRADIENTS.npz` 的头元数据包含1288维float64 wave、加权QPSD、total及两类endpoint/shared-S梯度。`ROW_MECHANISMS.npz` 有2048×384的signal/base signal和1555个保存成员；边界worker负责只读筛选和事件定位，不再前向012。012方向是carrier probe，不能当参数方向验证。

只定义5个参数方向：负012 total梯度、负012 wave梯度、负012加权QPSD梯度、由final AdamW moments及012 total梯度重构的prospective2001差向量、一个固定Rademacher方向。θ始终是010 final。prospective方向采用已登记clip5、bias correction、eps和decoupled weight decay，不实例化optimizer、不调用step，不写新CP；它不是实际新bank训练更新。缺/非法moments或零方向保留undefined slot，不偷换方向。科学阶段核CP的白名单、moment shape/finite、step/param_groups及原SHA后才使用。

## 一次运行和固定banks

完整评价用全部6593 SOURCE TRAIN父和全部21047 REAL TRAIN，SOURCE每bank4draw，两独立bank共52744响应行。两个bank固定父顺序，shape/005/extension用分别登记的独立显式流；每bank三hyp共用同一shape信号和电子学prefix。所有父有相同draw数，comparison weights与原计数/物理权重分开，物理权重未验证写unknown。

参数验证从SOURCE TRAIN无放回选1024父，分两个512父批次；每bank4draw，两个bank共8192行。选择是固定均匀选择，不按结果/Q删行。它直接复用完整评价对应的参考/noise/latent，不新增第三随机库。REAL参照始终是完整TRAIN21047。

每方向正负端点×三个预定名义输出RMS尺度0.1/1/10 ADC LSB，共30参数状态。用原 `FrozenDeterministicVoltageShape010.forward` 与 `torch.func.functional_call` 显式传入两白名单参数；不inplace更改θ0、基线、005或optimizer。不改变模型/观测源代码，也不使用carrier加性扰动冒充θ扰动。

尺度换算只用冻结decoded上的解析head方向映射：线性Conv(dW,db)、tanh导数、原control平均/expansion/gate/零canonical-Q projection和幅度。全8192参考行求共同RMS尺度，固定λ后只执行3尺度；无loss适配、无步长搜索、无新增autograd。保存名义尺度、λ、实际Δθ、有限更新的实际carrier ΔRMS/absmax、canonical-Q/prefix残差。tanh非线性导致名义与实际RMS不同必须报告，不能宣称恰好等尺度。

每状态比较三条路径：原参考硬ADC锚定的连续clamp增量、同次硬ADC固定参考crop、同次硬ADC动态first-argmax。fixed和moving共享硬量化。所有路径分别报告wave/QPSD/Q/T/PSD/guards；native output-Q membership与参考固定membership分开，避免把bin迁移写成crop误差。主判断依据两bank、两个父批次、三hyp下有限参数更新对真实硬目标的影响；不要求随机方向改善，不因微小反号自动改代理。

## 四套评价与五类指标

统一入口消费 observed256、真实uncentered prefix100、父/draw引用、SOURCE信息表、原REAL group、原冻结loss statistics/固定projection及全部观测flags。永久保留原100基线、384carrier、ADC一次、[112,144) first-argmax、pre16、256窗、tail46、1ns软件间隔。

1. 全人口：不等N的原W1精确归一CDF；分报原wave各view、多尺度lag、原生QPSD、Q/T联合描述和完整guard频数。legacy wave+2QPSD仅辅助。
2. 输出Q条件：原冻结TRAIN boundaries/support16，分报Q/tail/PSD分位数、宽度、样本数和fallback/缺箱；Q≤0/guarded/undefined不删行。输出Q不是入射能量或粒子真值。
3. SOURCE输入趋势：消费输入worker的canonical Q、幅度、形状PC及有效性mask，记录输出均值/方差/形态随输入的趋势。它是生成器输入依赖描述，未配对REAL不能提供同x的真实conditional target。metadata/ID不进网络。
4. 同父重复响应：分bank、合并8draw，记录preADC及hard域条件均值、方差/协方差摘要和Monte Carlo误差。父簇是统计单位；重复draw不是新增独立物理事件。

五类结果分别为wave、Q/tail/PSD、baseline noise、morphology、input dependence。noise使用真正prefix100，不用256窗前100点冒充；形态rise/width/recovery须保留negative/guarded/prepeak/postwindow-censored等原因mask与所有分母。完整 covariance 可由保留重复样本或因子重建，避免保存密集6593×256²矩阵。

REAL–REAL只作有限样本经验基准。全人口的6593/21047语义不能被匹配子集替代。RR若要求exact等N且原record groups不跨侧，应预冻group-isolated matched subsets并报告实际选择、N和group规模；未核metadata前不虚构两套完整21047无放回参照。RR的group定义可能是人工50-record组，不能称独立采集，也不能当不可约误差下限。

## TRAIN人工灵敏度标定

复用1024父×4draw×2bank当前参考的pure signal及相同noise；不解码另一套REALcarrier、不伪造REAL384载体。先由根声明应用容忍假设，再映射指标阈值，且在014新候选出现前冻结。

提案共21状态：signal gain±1/3/5%；smooth slow-tail gain±1/3/5%；electronics amplitude±5/10/20%；pure signal Gaussian展宽σ=0.5/1/2软件ns（positive only）。signal/电子学分别扰动、前100纯信号保持零、三hyp共享前缀；记录Q耦合及边缘影响。Gaussian核/尾gate定义必须随config冻结。这些是人工有效机制，不是新实测或硬件标定。无新NN；各状态经过原硬ADC/动态裁窗，记录对应五类指标灵敏度。实际容忍度不能由RR或“尚未通过”倒推。

## 提案预算、资源和运维

完整θ参考52744行；30参数状态×8192=245760行；原A及actual new-head各298504行。005仅52744行。解析方向Conv为5×8192=40960行，单列并纳入head kernel总339464，autograd为0。

硬ADC：完整A/θ参考316464行，参数状态737280行，21人工状态516096行，总1569840行；fixed/moving硬观测共享ADC，不重复计算。全路径gather上限3044400行。CP加载3（A、005、final）；initial/1000仅使用旧保存元数据。optimizer创建/step、G/D/noise/pop/C2ST训练、新AD与DEV模型行全部0。undefined方向/无法标定状态跳过，保存真实entered/completed和unknown in-flight，不耗dummy预算。

CPU12、GPU10GiB、job-tree RSS8GiB、wall7200s、output1024MiB、chunk128。每次只保留一个θ状态的完整评价工作数组；不保存30套完整波形。完整参考和逐父/逐draw机制量可保存；不导出raw REAL。根可选lossless ADC-code存储及原观测重建契约，不能悄悄float32改变评价口径。

`wp1_ops.py` 只组装统一config/bindings/运行入口。复用已验证012纯标准库SHA、safe regular tar、fresh paths与supervisor资源/only-owned stop函数，不再克隆69/36固定gate；common WP1 source/input manifest只冻结一次。逐操作journal复用entered-before/completed-fsync/failed-preserve语义；无自动restart。第一次实际128行A/θ/noise三hypfamily完成后写FIRST_WP1_PROOF；根验证正确开始后仅设置一次苏醒。

## 最小输入与接口

静态绑定：master plan；统一config/OBSERVATION_CONTRACT/DATA_ROLES/ACCEPTANCE_CONTRACT；dataset008；fit model/loss statistics；005/A/base config/training complete；010 final thinCP及完成/原生退出/source bindings chain；012 gradients、ROW_MECHANISMS、projections及完成/只读核验source chain；输入worker的G4原8192 cache和DATA_REFERENCES（按其whitelist）；原protected source/metadata链。全部stdlib SHA/path gate先于任何数值import/CP。

输入信息artifact可由同job的先行信息worker产生，登记其writer源码SHA、shape/role计数与实际artifact SHA，再由评价阶段单独hash/read；不凭未生成文件虚构绑定SHA。所有SOURCE join以完整source_id/原role核对；TRAIN-only forward，1599 internal_val及固定512DEV只在信息审计中标记。

API建议：`evaluate_hard_population(...)->{population,output_Q_conditions,source_input_trends,repeated_parent}`；`parameter_directions_from_saved_state(head_cp,gradients,config)`；`run_wp1_013 --config <frozen> --bindings <frozen> --output <fresh> --device cuda:0`。四新增脚本由本worker独占；输入/边界两模块由信息worker独占，根拥有数据角色、观察/验收正式合同与总运行登记。

新测试只覆盖新语义：原native forward与functional参数替换（恢复θ0/hash/无grad）、解析方向映射人工有限差分、AdamW纯数组公式人工参考、四套指标的角色/父簇/undefined与不同N/条件support。已有42运营检查引用，不再次为相同运维机制建研究协议。
