# 071：冻结070的原生选峰与Q/tail全人口分解

070已在两头三H完成独立复核，按CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP闭合。确定性前尾积分恢复到BASE054，五个tail保护和第二头两个QPSD保护仍失败。全人口保存标量中，相对BASE054选峰不同的981–1046/6593行占绝对tail差值约95%；78个fast载体例子支持精确分解，但不足以给出全人口路径贡献。071只补足这一证据，不预设新校准器，不打开确认库。

输入为原6593 SOURCE、21047 REAL、原身份/权重/NaN、A0/M0/005冻结状态、原BASE054构造、原062条件profile及070原始split_warp070。两FIT种子2026101839/2026101840和joint种子202610183900不变，fast/slow/common_level均保留。原M0/LEVEL069/SPLITWARP两头三H的18份完整指标与原case记录必须精确070，原72NPZ科学数组逐数组复核。选峰仍使用共同[112,144)首argmax，ADC、基线与256点裁窗均不改变。

唯一新增运算是在SPLITWARP每个H旁路观察BASE054信号，合计6次、39558行，没有新增hard_metrics调用。BASE054 signal按原顺序由bank M0加fixed_delta产生；它的全人口特征、flags和native峰必须离线对上已核验060/BASE054的对应保存行。新增代码不抽随机数，不改变bank或模型，不更改任何端点。新候选数、AD、优化、有限差分、LP、DEV、确认均为0。原candidate_head_evaluations计数更名为frozen_endpoint_head_replays=2，明确这些是冻结重放。

对同一父行，令m0为BASE054确定性均值，m1为070均值，r为M0−A0，dr为070随机校正，e为同一电子噪声；a0/a1分别为旧/新ADC载体，p0/p1为各自原生峰。浮点加法顺序项显式保留，不把它并入物理随机项。Q窗口为[p−16,p+240)，tail为[p+30,p+240)。对每个窗口定义：

`native差 = 新窗口内均值改变 + 新窗口内随机改变 + 新窗口内ADC误差改变 + 新窗口内浮点重结合项 + BASE054载体从旧窗口到新窗口的改变`。

各项逐行闭合误差不超过1e-12 V ns；同时与既有native Q/tail标量差一致。额外保存确定性参考峰窗口下的tail均值、随机及ADC改变。没有固定最终选峰或改写观察数据。

保存BASE054 signal、仅改均值的signal、耦合随机后的signal、BASE054/070 preADC、BASE054/070 ADC及参考均值峰。保存p1相对p0的峰竞争优势，分解为BASE054均值、均值改变、原随机、随机改变、电子噪声、070 ADC误差和浮点重结合项，合计须精确回到070 ADC优势（1e-12 V）。这是按既定计算路径的代数分解，不是唯一的物理因果识别。源/REAL仍未配对；Q分组不能当作能量真值。

新增6份DECOMPOSITION_ROWS.npz，共39558行，每份不超过52数值列和10布尔列；实际设计43数值列、9布尔列。另有6份小JSON汇总与BASE054观察flags。原70输出全部保留，因此预期78NPZ、9profiles73791行272组视图，18H、2bank、6checkpoint、1TRAIN/1PCA/0DEV保持。新增输出事前上界估计采用原070 run字节加52 float/int64列与10 bool列的全行原始字节，再加3MiB元数据/压缩封装余量，低于64MiB。运行中与RESULT后仍查实际64MiB界限；估计不作为放宽理由。

资源仍4CPU、GPU6GiB、RSS12GiB，内部780秒、外部900秒及20秒TERM宽限；一次一个作业，无自动重启。源码、配置与证据先冻结再部署。启动核验要求首头9H完整、全部原37NPZ科学数组与三case记录精确070、新3份分解表19779行、BASE054旧行精确、全行代数闭合及39个fast完整例子的独立NumPy复核。CPU/CUDA制造例子包含ADC同码首argmax、量化饱和、恒等零差和排列一致性，不读取训练数据或checkpoint。

成功仅定义为CLOSED_DIAGNOSTIC_COMPLETE：所有冻结重放、计数、全人口分解、hash、资源和最终深冻结通过；070科学负结果仍保留。之后依据全两头、全H结果比较同窗波形项和裁窗项、各阶段峰变化、Q分组/旧失败行，决定下一项有证据的最小工作。科学目标未完成时继续推进；不借诊断完成宣布响应接受。
