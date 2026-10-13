# 007实现约定与自检

主代理按用户要求独立实施；本文件和SELF_REVIEW回执不是第二位模型的独立审核。007仅新增文件，原model、losses、extension006、electronics_v3及所有历史协议/末点保持原字节。

`response007.py` 将原信号网络的electronics替换为无参数ZeroElectronics。网络输入仍只有SOURCE波形及明确的信号随机量；旧electronics随机键是API兼容的零张量，不消耗随机流。phys和local_white来自CPU float32响应流，005 z/u与延拓epsilon来自另外两个独立流。三假设噪声先作为常量形成，再在各微批的共同signal上各自执行一次实际候选ADC。

每假设的wave/QPSD/noise/adv目标乘1/3后对该waveform求梯度。第二遍前向保留真实的各自ADC饱和mask；通过同一signal图同时回传三个端点梯度。base regular先扣原信号单独ADC超界项，再加三候选超界项均值；regular/trend除微批数。不能先平均候选波形或先平均饱和mask。实际D更新在各自端点梯度前完成，其后D.eval并冻结谱归一化状态。

JSON保存的TRAIN统计恢复成float64张量以适配历史loss的clone接口；数值逐个原样恢复，不重新拟合、不转换成默认float32。V2本来就用float64做标准化，D则按其float32输入转换buffer。统计原JSON随检查点保留。

G更新从0计数，0..2999平均响应、3000..14999联合随机、15000..18999低学习率。两臂各自新初始化及同样显式随机流；所有共同state键从同一个新full初始化复制。每个D都重置到同一个初始state，之后优化器独立。96000个训练D步骤及4200个评价C2ST步骤与38000个响应G步骤分开计数。人工测试/容量测量的更新不进入科学计数。

固定末点评价只载入final.pt。全部人口统计用原count；另保存group平衡等样本量距离及C2ST，二者不能混称同一人口。C2ST结构为Conv1d(1,8,9,s2,p4)→SiLU→Conv1d(8,16,9,s2,p4)→SiLU→Conv1d(16,32,9,s2,p4)→SiLU→AdaptiveAvgPool1d(8)→Linear256,32→SiLU→Linear32,1。每个分类器200更新，固定初始化/采样种子，每批各类64，无D权重、方向反转或最佳点选择。归一化mean/std只取该分类器TRAIN，std最小1 ADC step。

生成波形/条件方差均用ddof0，IQR用linear quantile；条件均值/协方差不宣称无偏物理估计。父级bootstrap对固定信号Q箱使用父级count/sum/square充分统计，并预先中心化避免消减误差；未定义的0/0比例保留NaN→报告null，不删除父或补抽。两个臂共用同一次父级重采。

输入干预从DEV按SOURCE Q稳定排序后配不重叠邻对，再取logQ距离最小的32对。交换单位Q形状，保留父原Q；另记录转换到float32输入后的Q舍入差。排列等变性允许float32卷积/归约误差，明确界限为`64*eps32*max(abs(signal),abs(permuted_signal),ADC_step)`，记录实际最大误差及界限；不将数值等变性称为物理源保存验收。

人工测试覆盖：共同初始化与平均阶段输出一致；三个随机流配对/重放和005前缀；含单情景饱和的直接整图对照两遍梯度；三个训练阶段真实优化器更新与005冻结/随机头梯度；生成、分箱bootstrap、距离、重复父、输入干预、人口重权和C2ST人工端到端路径；候选整数ADC码零差和float64电压表示误差界限。

第一次扩展测试暴露了JSON统计的tensor/list接口差异，随后统一绑定；人工排列测试暴露了float32卷积误差，需要上述数值界限。所有问题在新科学数据解码和训练之前修复。工程测试历史保存在ENGINEERING_TEST_HISTORY.json，不把这些失败隐藏为科学训练成功。

RTX5070完整1024/假设批次人工容量测试：6个人工G步骤和12个D步骤；稳定平均阶段约0.35秒/步，联合/低学习率约0.52秒/步，峰值allocated约1.22GB、reserved约1.76GB、进程树RSS约2.82GB，原生退出0。它只支持资源估计，真实数据的训练时间和模型有效性仍未测试；8小时墙钟、8GiB RSS、10GiB GPU及2GiB输出限制保持原登记值。
