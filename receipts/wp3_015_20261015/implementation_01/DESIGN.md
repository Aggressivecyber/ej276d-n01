015 实现说明（代码与人工检查；未启动科学计算）

本实现比较 seed0_M0、seed0_M1、seed1_M0、seed1_M1，各最多2000次更新。M0继续优化014 C1的1288参数形状头；M1保护同一形状头，只优化六个零初始化均值参数。每个seed只从其014 C1 final薄checkpoint取new_head张量，两个arm各建fresh AdamW，不继承014优化器。SOURCE/REAL全覆盖sampler、shape/005/extension流和wave/integral投影均在同seed两arm间重置配对。

mean控制Q严格为去原prefix100均值后的SOURCE[110:366)，tail为[156:366)。q_reference/q_floor取原FIT_STATISTICS008.loss，保持原软件canonical窗口。三项均值基为原A、Gaussian sigma2ns/13点平滑差、causal exp tau16ns/64点归一滤波差。后两项经原gate与projection归零canonical Q，再以每事件RMS(A)/max(RMS(basis),1e-12V)规范。六参数产生三项0.1*tanh系数，叠加后逐事件缩放到RMS(deltaMean)<=0.10RMS(A)。B(x)z仍使用原A决定的0.05*max_abs(A)幅度；不乘适配后电荷。prefix100严格为零。

零adapter使用一阶导数为1的精确零加法，使初始IEEE字节保持不变，同时允许六参数从零学习；人工有限差分已验证此点。M1输出Q_mu=reference_A_qsig+canonical_Q(deltaMean)，当前qbar/qsig均为Q_mu。原qbar/qsig另存reference_A_*，原unit_shape/shape_Q明确为A参考字段。因零Q差基、gain系数绝对值<=0.1及trust缩放<=1，canonical均值电荷不低于0.9原A（数值残差单列）。未引入随机Q、时间或局部过程。

目标为三假设各自损失取均值：wave+2*C1 signed-asinh积分分布+0.01*(原随机形状regular+三假设saturation均值+mean((RMS(deltaMean)/max(RMS(A),1e-12V))^2))。M0 mean项恒0。只执行一次正常训练backward/update；无额外probe AD、RMS重校准或TRAIN统计重拟合。积分尺度直接绑定014保存的INTEGRAL_TRAIN_STATISTICS.json。

所有checkpoint/NN/优化之前先检查有限人口均值证据。只读prepared REAL TRAIN的real/real_groups/real_role/real_ids以及12个已保存C1 final REPORT中root/repeated_parent/{hard_wave_mean_V,parent_rows}。每seed/假设的r=RMS((mean_bank0+mean_bank1)/2-REALmean)，b=RMS(mean_bank0-mean_bank1)/sqrt(2)。REAL逐样点均值方差采用G/(G-1)*sum_g[sum_i_in_g(y_i-REALmean)]^2/N^2，G=422逻辑组，先逐样点计算再取SE的RMS。ALL6 r>max(1e-7V,3sqrt(b^2+realSE_RMS^2))且两bank残差cosine>0.5才允许短试。这是指定人口与逻辑分组下的有限证据，不识别真实条件均值，也不构成配对实测。合法阴性结果保存COMPLETE/training_skipped，checkpoint/NN/optimizer/AD/G全部0；verify-start和collect均支持该终态。硬256残差DC与电子学baseline100分开；完整硬评价另存ELECTRONICS_LIMITS，adapter不能修改冻结005。

精确最大计数：train=4*2000*512*2=8192000；模型evaluation=4*2个snapshot*2bank*6593*4=421952。A、head actual、head total、005各8613952。M1 adapter行数2*(2000*1024+2*2*6593*4)=4306976。ADC/gather=3*8192000+4*2bank*6593*4*3个arm-snapshot*3hyp=26474784。CP=4（A、005各1及每seed C1 head各1）；optimizer=4；G/backward/AD=8000。额外mean证据计数：REPORT12文件/79116父行、REAL21047行、6检验。maps、fits、probeAD、D/noise/population/C2ST更新、DEV模型行均0。阴性gate仅消耗mean证据计数。

硬评价直接复用014 Engine.evaluate：每trial A+共同C1 initial和final，全部6593父*4draw*2bank、21047 REAL，三假设、四套评价与全部readout flags不变。A/共同initial/final实际hard行均保留lossless编码、报告、父身份和重复统计，T表不变。每trial7200s/1536MiB、整批28800s/6144MiB、CPU12/GPU10GiB/owned RSS8GiB；原raw留server，选择回收上限512MiB/头64MiB。实际output/wall终点也检查。

控制器在新root使用SHA钉住的014纯stdlib运营AST，格式schema/archive/incoming标签映射到WP3，显式替换015 factory、freeze和proof/terminal合同。安全tar、新目录、fd继承全项目lock、仅停止owned child group、资源、native退出、partial证据和选择回收链不新增算法。原95源码及旧config/manifest继续原路径只读保护。12 REPORT无需下载：freeze只用已绑定318原库存的path/size/SHA小JSON证据，runtime仍核实物输入SHA后才允许decode。新包显式检查可达lazy项目依赖，包括stability011_metrics，避免遗漏。

CLI与接口：

- driver：python -B scripts/run_wp3_015.py --config PATH --config-sha256 SHA --output FRESH --device cuda:0。
- controller：python -B scripts/wp3_controller.py --mode freeze --repo REPO --config-sha256 DRAFT_SHA --out FRESH；后续deploy/start/status/verify-start/collect/collect-inspect/collect-failed沿014 --freeze PACKAGE --out FRESH接口，选择头用--include-head-checkpoints。
- 纯stdlib工厂settings015()/budget_caps015(p)。schema=n01/wp3-015-config/v1，run_id=wp3_015_20261015。
- 新OWNED：scripts/run_wp3_015.py、src/n01/model015.py、scripts/wp3_controller.py。不修改任何旧95/97文件。
- 必需inputs：dataset/statistics/data_statistics/base_config/deterministic/noise005/source_information/integral_statistics/C1_head_seed0/C1_head_seed1及mean_seed{0,1}_bank{0,1}_{fast,slow,common_level}。REMOTE-only REPORT可无local_path，必须与已绑定318库存一致。
- remote_inventory_evidence为{path:newroot/WP2_REMOTE_FULL_INVENTORY.json,local_path:E上的原collection库存JSON,sha256:4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5,bytes:129532}。
- contracts至少observation/data_roles/acceptance/tolerance_table；data_roles绑定新WP3_DATA_ROLES，configuration可绑定新WP3设计文档，其它原合同可作为额外保护。所有路径/size/SHA由根登记，不在本实现记录中构造科学输入。

人工证据位于engineering_01/artificial_cpu_02（9组）和controller_stdlib_03（5组），均native0。前者使用解析人工波形与未训练A，两个真实代码路径的人工step；后者纯stdlib、actual Store writer桥接与明确注入的计数，未计算任何真实科学行。真实CP读取、真实波形读取、真实NN/AD/训练及SSH在施工阶段均为0。根仍须完成正式输入/配置冻结、独立审查和启动；本文件不授权运行。
