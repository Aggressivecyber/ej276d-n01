# WP4-023：只恢复022的CPU评分

023回答022原登记的同一个固定T2硬观测问题。022已保存完整T2 kernel和fast父分数，随后在新cell入口从`stats017`误导入`Z95`而以native1退出。023保留该失败及其源码；新`contrast_cell023`复制原cell，仅把`Z95`导入改到其实际定义处`stats018`。不改科学门、原随机命名空间、teacher或任何旧产物，也不运行旧`contrast_cell022`。

023的成功标准是：冻结输入和先决合同一致，原生退出0，36个完整签名及3个primary齐全，精确计数与资源通过，并保存父分数和完整结果。科学检测结果单独报告；native0不等于响应恢复、REAL或硬件验收。

## 固定输入与来源

`WP4_023_SAVED_INPUTS.json`逐项固定23份既有输入，共1,000,564,887字节：

- 022保存的fast父分数1份；018保存的PARENT_AXES1份。
- 022的slow/common_level：两bank的HARD_ROWS4份、REPORT4份，共用BASELINE100两份。
- T1的slow/common_level：原017保存的HARD_ROWS4份、018保存的O19两份及controls两份。
- 原018配置、原022配置、022的T2 EVALUATION三个JSON。

这些是已生成数组。禁止加载原SOURCE/REAL/DEV波形、A/B/Q缓存、005或端点checkpoint、PIT/随机状态数组。fast的所有父分数及其T1 null/reference按原值复制到新输出，不重新解码fast波形、生成fast O19或调用fast距离。

四份JSON先决合同单列于配置：018独立通过回执；022 native1且T2生成完整、fast向量已保存的终点回执；最新持久用户授权；023前瞻scope。它们固定字节和SHA，不把022改成成功。科学输入清单和合同由driver在任何数值解码前校核，终点再核不可变字节。

全部6593个父、原5274 fit/1319 heldout角色及source IDs保留。四个原组`all_heldout`、`fit_Q_tertile0_heldout`、`fit_Q_tertile1_heldout`、`fit_Q_tertile2_heldout`和`nominal/low/high`比较权重直接读018保存轴；原物理权重也完整保留。PARENT_AXES全部数组按原值重新保存，不声称NPZ字节相同。比较权重不替代原物理权重。

## 数值路径与判断

新driver只调用原`load_kernel018`、`teacher_data022`、`crossbank_scores022`及纯NumPy统计。先导入018所用的最小`n01` namespace，避免`n01/__init__`导入模型。顶层工厂只用标准库；实际CPU路径不导入Torch/CUDA，也不构造Runtime、Engine、模型、采样器或优化器。

slow/common_level各重建原两bank O19，并检查原REPORT。相同bank的T1/T2 Gaussian前缀相关，因此仍使用T2b0:T1b1及T2b1:T1b0；先在每父平均两个方向，再用原父权重和parent-cluster SE。方向不作为独立重复，不合并为64draw。四个within32 U-self分母仍为32×31；cross分母为32²，负U估计保留。原full257减T1 reference，只作描述。

36格为3hyp×4group×3weight；primary仅三个hyp各自的`all_heldout/nominal`。原支持规则保持：至少64父、原父count和weight mass均≥95%、ESS≥原组父数的一半。先检查支持、cross的95%半宽≤.05、T1/T2独立32bank null各自abs-upper95≤.05，再判断cross lower95>.10。阈值均沿022冻结值，Z95为1.959963984540054。

控制或MC不足是`MC_unresolved`；区间跨.10是`tolerance_straddling_unresolved`；resolved但upper95≤.10是`not_established_at_registered_tolerance`。三个primary全支持才报告`supported_at_registered_tolerance_all3`；其他结果保留具体噪声假设敏感性或未建立。没有power curve、同时检验校准、机制不存在、全硬观测等价或训练Hessian推论。023不启动T2或REAL训练。

## 精确新增预算与资源

- 新NN、005、CP、AD、G、optimizer、RNG/normal/sign/PIT、ADC/argmax、原波形读取、参数拟合和训练槽均为0。
- fast复用6593个父分数；slow/common新父分数共13186行。
- T2保存硬波形解码843904行，T1保存硬波形解码843904行，合计1687808行。这是保存结果解码，不是新增ADC或NN。T2新O19映射共843904行。
- 仅两hyp：257距离`4×6593×32²×2=54009856`，O19距离`7×6593×32²×2=94517248`，合计148527104。
- 022已经完成74263552距离；合计222790656，等于原022三hyp预算。fast不重复。20份唯一NPZ在原读取函数下共打开22次（两个共享baseline各按hyp重开）。

Linux CPU12，AS/RSS各8GiB，wall7200s，CPU soft/hard21590/21600s，输出1536MiB，选收128MiB；GPU数值工作0。driver设置AS/CPU rlimit，原018 CPU supervisor的023字面绑定负责单job锁、wall、准确native退出和进程资源回执。无新controller或盲重启循环。持久授权允许后续另行固定的必要工程恢复，并不把本入口变成自动重试。

原`Store.call`日志区分entered/attempted、completed和inflight。单hyp距离仅成功返回后计入completed；失败的inflight完成情况为unknown。`status.json`保存当前phase/inflight；`PHASE.json`保存最后完整hyp及其completed距离快照。失败回执中的kernel instrumentation计数明确标为completion unknown，不能替代completed账。fast分类进展可用于真实启动确认；仍运行时仅设置一次未来苏醒。

## 输出、源闭包与唯一工程检查

新输出固定在`wp4_023_20261016/runs/primary/artifacts`。保存STARTED、status/PHASE、durable operations、三个`PARENT_SCORES023_*`、两个新T2 O19、原轴副本、CONTRAST023 JSON/配套数组、COMPLETE或FAILED及artifact inventory。库存清单在COMPLETE和最终status之前写，排除可变status，最终全量库存由ROOT收口。终态记录elapsed、CPU、peakRSS和写终态回执前的output bytes，回执后再检查总输出上限。

实际科学调用闭包为9源：新023，原022/019/018/017/wp1_ops及stats017/stats018/saved_array_kernels017。配置固定8份旧源的bytes/SHA，ROOT冻结新023及其余包成员。不调用其余旧lazy分支，也不声称整个旧训练包已穷尽测试。

唯一新增人工检查在E盘隔离9源副本执行，实际调用`contrast_cell023`，覆盖正确Z95导入、正例、resolved负U例、cross MC优先级、两种null不足、支持不足、区间跨阈值及JSON/数组序列化；全部固定数组，距离/NN/CP/AD/G/RNG均0。禁止repo fallback，不重复crossbank、005、观察codec或旧工程套件。022先前fixture只验证crossbank数学，没有调用cell；其PASS不能证明完整022可执行。

执行CLI：

```text
python -B scripts/run_wp4_023.py --config wp4_023_config.json --config-sha256 <frozen RAW SHA256> --output /home/rylan/projects/ej276d-n01/wp4_023_20261016/runs/primary/artifacts
```

本文件和driver准备完成后由同一独审审核，ROOT仅改变draft的`status/source_files`进行冻结、注册单job并执行；本施工阶段不读取私有数组、不SSH、不运行科学评分。
