# WP3 / 015：有限均值路径对照

本次回答固定人口假设下的小型均值修正是否有开发收益。014 的 C1 表示已独审；其权重用途门为5/6，尚未晋级。015引用各seed的C1 final头作研究起点，原A/005与失败记录保留。硬观测、数字T、原数据角色不变，不重训旧full19k。

先在未加载checkpoint、未运行NN前，读取全部21047条REAL TRAIN及12个既存C1 final REPORT的每父4draw硬波形均值。对2seed×3噪声假设，令r为两库均值相对REAL人口均值的RMS，b为两库均值差RMS/sqrt(2)，s为按原422逻辑组计算的REAL cluster-sandwich均值标准误RMS。仅当ALL6满足r>max(1e-7 V,3sqrt(b²+s²))且两库残差cosine>.5时训练。否则保存理由并以native COMPLETE / training_skipped结束，CP/NN/AD/G均0，不改门槛。该证据描述已记录人口的均值残差，无法识别真实条件均值。

固定顺序seed0_M0、seed0_M1、seed1_M0、seed1_M1，各2000G，最多8000G。M0继续原1288参数形状头，均值关闭；M1冻结同一形状头，只训练6个零初始化均值参数。每seed两arm加载相同C1头字节，均用新的AdamW（lr2e-4、weight_decay1e-4、clip5），不继承014 optimizer。全部训练/评价流配对，保存所有6593/21047 TRAIN的覆盖证据。目标仍为wave+2×C1积分分布+.01×(原regular+均值信赖域项)，无额外AD探针、初始化RMS重校准、005重拟合、人口调整或DEV波形读取。

均值增量使用三个基：A、σ2软件ns/13点Gaussian平滑差、τ16软件ns/64点因果指数滤波差。后两基以原gate/projection去除canonical Q，再按每事件RMS(A)归一，epsilon为1e-12 V。六个θ分别控制三基的截距和一个连续输入控制项，系数为.1tanh；控制Q来自SOURCE[110:366)、tail来自[156:366)，q_reference/q_floor来自冻结FIT_STATISTICS008.loss（3.7103998046755806 /0.0005319378913383611 V软件ns）。总增量RMS限制为.10RMS(A)，prefix100严格0。随机B(x)z继续以原A定幅，不与本次均值增量或随机电荷相乘。本次不增加电荷、时间、局部噪声分支。

每trial保留A、共同C1初始和final硬评价：全部6593父、每库4draw、2独立库、3假设，对全部21047 REAL TRAIN；全人口、输出Q条件、SOURCE输入趋势、同父重复分别保存。原正/零/undefined T和2%用途门保持原判定，低Q及失效行保留。基线DC单列为电子学限制。

开发选择预登记为：同一候选ALL6的每个正且已定义T组件，相对A与共同C1初始都非劣（mean2bank距离差≤T）；必需已标定分量缺失或undefined使候选pending、不eligible，未标定分量继续pending。合格候选按12个final完整wave距离均值排序，1e-12内tie选M0。胜者均值必须比共同初始低超过1e-12才替换研究参考；否则保留原C1参考、均值分支关闭。M1胜出仅支持保留该均值模块作下一阶段研究，不能把有限总体改善解释为真实条件均值恢复。逐格均值残差、原QPSD/噪声/形态/输入依赖、ALL6≥2%用途和绝对T分别报告；本阶段代码路径或研究参考选择不授予响应匹配成功。

精确最大计数：训练行4×2000×512×2=8192000；initial/final实际A/head/005各追加4×2×2×6593×4=421952。因此A、实际head、总head kernel、005均8613952行。硬ADC与gather各24576000+1898784=26474784行。仅M1调用的均值adapter共4306976行。CP4（A、005、每seed一个缓存C1头），optimizer4，G/step/training backward/AD各8000。初始化map、probe AD、统计fit、D/noise/pop/C2ST/DEV模型行均0。另登记前置只读REAL21047行、REPORT12文件/79116父行、6个测试。

单trial wall7200秒/输出1536MiB，整批wall28800秒/输出6144MiB，CPU12、GPU10GiB、job-tree RSS8GiB。新隔离目录，独占本项目科学锁，只管理本作业树；不自动重启，不覆盖现有输出。选择性回收最多512MiB，raw留服务器，不挤入C。启动后仅保存首次真实更新及同snapshot健康证据，安排同一n01一次未来唤醒后休息。

短候选已完成4/12；015预留4次，若完成为8/12。WP3最多6次，此块占4次，余2次供必要的下一单因素扩展。教师8次及最终全流程确认/部署预算另列，不能用近阈值补训扩大搜索。
