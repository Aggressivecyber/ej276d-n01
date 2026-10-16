# 024：原015平均响应参数化的固定hard方向剖面

运行身份为`wp3_024_20261016`，owned root为`/home/rylan/projects/ej276d-n01/wp3_024_20261016`。[前瞻合同](../receipts/wp3_024_20261016/registration_01/PROSPECTIVE_SCOPE.json)固定13点、原观察口径及预算。本任务对冻结017 seed0 M0做control-only诊断；原015 M1保护的随机头是C1，因此本次不是015 M1重放，也不生成部署权重。

原6593个TRAIN父及原物理权重完整保留。输入为13个科学文件，加`contracts.init017`和`contracts.source_audit013`两个小元数据合同。执行先验证全部source/input/contract/protection原始SHA，再核017冻结CFG的015 seed0 M0 CP绑定（485683B、SHA`27ad34f2f7fa0c2c78f881b649ecd5ad664c91ff4bbe2bb08c109f3536f0915c`）、INIT缓存链及cache manifest中的M0 tensor摘要`0b1bdf84f58d9d9d7fb5fe29d572955611bfc870a972c98fa196f78ed7b13fdf`。不加载A/M0 CP，不做新A/M0前向。

SOURCE控制来自013的`canonical_Q_Vns`和`tail_clean_window256_Vns`，按原ID及TRAIN角色连接。013 audit和017 parent metadata必须证明peak126、crop110:366、prefix均值0，因此SOURCE tail正是原015的156:366积分。cq/cs沿原015公式恢复，不能用缓存A的Q替代。目标仅用015保存的21047个REAL TRAIN人口均值及422逻辑组得到的均值方差对角；无原始REAL/DEV/SOURCE波形读取、配对条件真值或跨时间协方差假设。

原015的Gaussian sigma2/13点、causal tau16/64点、zero-Q投影、RMS归一化、`.1*tanh`系数及相对A的10% RMS联合trust逐句保留。原model008 baseline gate/projection按其CPU缓冲构造后搬到设备；015 Gaussian/causal核按原015在当前设备构造。零点沿原`_ExactZeroMeanAddition015`保留base IEEE字节。只运行`zero,+e0,-e0,...,+e5,-e5`，h=1；不组合、不拟合、不扩大网格。

两个新随机库的seed为`seed017({'run_id':'wp3_024_20261016','purpose':'fixed_mean_axis_profile','bank':b,'stream':s})`。仅复用seed017的哈希算法，namespace独立于原017教师。

| bank | shape | noise005 | extension284 |
|---|---:|---:|---:|
| 0 | 8009511340382243432 | 8776897363944309080 | 4446727864989473062 |
| 1 | 8323415395242486091 | 3648971429808556269 | 4948005227162182137 |

`Streams010.draw`使用显式CPU float32生成器，父major/draw minor，signal chunk128。每bank每父4draw；005只加载一次，形状为原缓存`A+Bη`、charge off。先保存两bank三hyp完整float64 base384，全部13点共享其原始字节。三hyp共享成熟005 prefix及同一284维extension随机底稿。记录各流首末state，无charge/sign/PIT随机请求。

科学新增：52744个005 NN行及缓存M0组装行；158232个extension/base行；shape421952、0055327144、extension14979296，共20728392个normal标量请求；2057016次真实hard ADC/first-argmax/gather；85709个parent-theta adapter行；78个均值cell、72个与零点的配对比较。G/AD/optimizer/参数fit/train trial、A/M0 CP加载和原始波形读取均0。CPU12、GPU10GiB、RSS8GiB、wall7200s、output3072MiB、精选回收128MiB；不另扩shared预算。

每个cell使用原14bit ADC、112:144 first-argmax、256裁窗、真实prefix100基线；保存完整原013 lossless hard码及可选IEEE XOR。每父统计保留Q/front/tail均值与draw方差、共享底稿的配对变化方差、crop移动、rail及原hard flags，不删行。人口均值保存256向量，分别报告raw RMS、去DC形状RMS、signed meanQ/front/tail bias、trust使用；REAL对角不确定性原样保留。两个bank的一致性和固定库Monte Carlo量单列，不能把时间坐标或重复draw当独立物理父。

全部12个非零点均报告。同一点在六个bank/hypothesis cell中同时满足raw RMS ratio≤.90、去DC形状RMS严格改善且absolute meanQ bias不增加，才记有限profile witness。此新诊断标准不是013响应接受门。无witness只说明该13点profile未给出见证，不能证明整个六参数族不可达，也不自动授权timewarp、训练或更多点。

入口为`python -B scripts/run_wp3_024.py --config <owned CFG> --config-sha256 <SHA> --output <root>/runs/primary/artifacts --device cuda:0`。factory/validator顶层仅stdlib。原`Store.call`分别保存entered/completed/inflight，数值操作复用`Runtime017.call`的CUDA同步；失败保存FAILED并原样抛出，不重启。STARTED、INIT_PROOF、FIRST_ACTUAL_BASE_PROOF、FIRST_ACTUAL_HARD_PROOF及status支持ROOT确认真实005/base/hard进度与同快照owned进程/资源健康后设置一次未来苏醒。

大`*_BASE384.npy`及`*_HARD_ROWS.npz`留远端；精选JSON、AXES_AND_TARGET、RANDOM_STATES、POPULATION_MEANS、TRUST和PARENT_STATS足够核对有限结果，完整hard/base支持后续明确登记的读回。唯一新增[人工组合fixture](../receipts/wp3_024_20261016/engineering_01/COMPOSED_ADAPTER_HARD_FIXTURE.py)使用合成SOURCE，真实调用原015 adapter并与control-only入口逐值、最终hard和codec对齐；人工成本与科学预算分开，见[运行记录](../receipts/wp3_024_20261016/engineering_01/ARTIFICIAL_ATTEMPTS.json)及最终fixture收据。不复跑旧005/Streams/codec工程套件。
