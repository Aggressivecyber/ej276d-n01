# WP5 021：canonical信号电荷的人口与同父响应方差分解

021仅实施已登记的一个前置问题：在已有Q与四个固定端点上，量化比较人口变化如何改变人口间方差及平均同父方差，并列出a/b两项和局部参数列的代偿结构。它不是新训练、完整WP5确认或新资格门。020独审结论保留：2455pass/89fail/0pending，89fail均为unpaired seed0；seed1的1272格全通过。

执行前必须绑定并核验020最终独审回执、直接用户持续授权、021前瞻登记，以及本脚本/本文/输入清单的真实字节。draft不可执行。最新用户授权允许必要新增计算和工程恢复，但不使旧科学非通过变为通过，也不取消前瞻预算和一次一个科学job要求。

## 六份科学输入与角色

`implementation_01/input_sources/INPUT_SOURCES.json`仅保存已有文件的路径、字节数、SHA和历史成员来源，不复制或重写原数据。六份科学输入共6,711,911字节：018的PARENT_AXES.npz、两份paired COMPARISON018.json、020的两份unpaired COMPARISON020.json、冻结020配置。最后一份只提供原T1与qref合同；四个a/b端点从已有secondary字段读取，不拟合或挑选更有利端点。020 final和授权是另外两份先决metadata，021登记是独立的范围合同。

PARENT_AXES保留6593个原身份、5274/1319原划分、原物理权重和既有group/weight。只分析1319个heldout Q，Q为`fixed_A_Q_Vns`，不是标定入射能量或REAL真值。四组严格为all_heldout及三个fit_Q_tertile*_heldout；三权重严格为nominal/low/high。低高权重只存在于heldout，不外推到fit/all。它们是固定比较人口，不是拟合得到或已经物理验证的源人口。

每组内按既有比较权重归一化alpha；报告原父数、比较权重和、ESS及无删行。原物理权重不被这些权重替换，也不静默相乘。原PARENT_AXES按原字节复制到输出PRESERVED_PARENT_AXES.npz，保持全部身份/角色/权重可追溯。

## 固定60项方差分解

4组×3权重×（已知T1+两个paired端点+两个unpaired端点）=60项。原T1为a=.01qref、b=.0025，qref=3.7103998046755806；端点值保持保存记录。对canonical信号电荷：

- E(q_sig|Q)=Q，Var(q_sig|Q)=aQ+bQ²。
- 人口间项B=E_alpha[(Q−E_alpha Q)²]。
- 同父项W=a E_alpha Q+b E_alpha Q²，两项分别保存。
- 总方差V=B+W，并报告人口/响应占比及同父a/b占比。
- 相对T1分别保存Delta a×E Q和Delta b×E Q²及其和，避免将相反作用合成后隐藏。

同一组、同一固定kernel的low/high相对nominal，只报告这些解析量的绝对和相对变化；nominal分母为零时，绝对差照报，相对变化标记undefined，不补零。该表不挑选人口或参数，不设置新性能门。所有量只属于005电子学和硬ADC之前的canonical信号电荷，不能外推到hard256、寻峰/截尾、REAL或硬件。

## 12项灵敏度与24次谱分解

对T1每个group/weight定义两列c=[a_T Q,b_T Q²]，表示方差对a/b分数变化的线性响应，单位Vns²。这里不除以v_T；与早期建议中的log(v)灵敏度不同，以正式021登记为准。

raw Gram为G=sum alpha c c.T，单位Vns⁴。unit-column Gram为G_ij/sqrt(G_ii G_jj)，无量纲；其off-diagonal是未中心化weighted column cosine，不能称为Pearson相关。每项分别调用一次eigvalsh，12项共24次，报告两特征值、lambda_min/lambda_max、条件数及机器精度数值秩。

数值秩截止为2×float64 epsilon×最大特征值，仅描述计算分辨率，不是科学可识别性门。小特征值保留原数值；rank<2时条件数为null并明确rank_deficient，不夹成零或当作pass。raw与unit-column两套结果同时保留，避免尺度被隐藏。此Gram不是实际无配对训练的梯度/Hessian，不证明完整分布可识别或不可识别，也不替代hard观察可分辨性。

## 成功、闭合和停止

正式运行直接执行新增代数闭合检查，不另造人工测试或重跑旧工程套件：W与aEQ+bEQ²一致，B+W与中心化/二阶矩写法一致；unit-column Gram的对角及1±cosine谱一致。float64数值容忍固定1e-12，二阶矩闭合按参与运算项的尺度归一；它只核对代数，不是物理容忍或新性能门。

成功是完整交付60项方差与12项灵敏度、全部原分母/ESS、24套谱和输入/输出SHA。存在或不存在明显代偿均为有效结果；不能以必须找到某种机制作为成功条件。完成即停止，不增加权重扫描、参数拟合、新seed、RNG、训练、T2或均值候选。旧020非通过、017 pending和018资格不改，完整WP5与WP6不据此宣称完成。

## 有界执行与产物

预算固定：response父贡献39570、Gram父贡献7914、NPZ数值打开1次；60方差项、12人口权重项、24个2×2 eigensystem。NN/CP/AD/G/RNG、原SOURCE/REAL/DEV波形、hard replay、pair distances、参数拟合和optimizer均为0。

本地Windows，CPU1、BLAS/OMP1、wall120秒、工作集/RSS上限2GiB、输出8MiB。依赖仅已有NumPy和Python标准库，不安装psutil。用ctypes直接将当前owned进程绑定到原允许mask的最低一核，并通过GetProcessMemoryInfo取得当前及迄今peak working set；该资源快照在分析完成、最终JSON序列化前取得。root外层负责整个native进程的timeout和完整实际资源回执。TEMP/TMP和请求的库缓存均位于E盘新输出runtime_cache，须使用python -B，不在C写数据。

输出为STARTED.json、原字节PRESERVED_PARENT_AXES.npz、完整ANALYSIS.json与COMPLETE.json；无新框架、helper类或controller。任一边界或代数错误直接抛出，由root保留原生失败；不覆盖旧输出或隐式重试。

入口：`python -B scripts/run_wp5_021.py --config wp5_021_config.json --config-sha256 <frozen RAW SHA256> --output E:/GAN/research/N01/runs/wp5_021_20261016/analysis_01`。只有root冻结配置与独审完成后才执行；施工期间不读取Q数值或运行真实分析。
