# WP2 / 014：C0与无分母积分表示的两种子短试验

013已原生退出0并完成265文件与保存数组独立读回。总梯度正向名义1LSB方向在2库×2父批×3假设全部12格通过，方法选择adequate；当前010头的全人口wave收益低于2%门槛，形态与基线均值仍有绝对偏差。本阶段按用户已登记的主实施计划推进有限C0/C1比较。

## 固定模型和唯一因素

保留010的1288参数形状头结构、8维高斯shape输入、保护canonical电荷和prefix100的电压域基构造。deterministic19k A、005及延拓的参数、缓冲和原源码冻结。优化器仅包含新头weight/bias；不创建D、GAN、人口重加权或C2代理。

C0使用原 `wave + 2*QPSD + .01*regular`。C1仅将分布表示改为Q、T和四段[0,16)、[16,46)、[46,96)、[96,256)积分，固定REAL TRAIN的median绝对积分尺度并以ADC-step×sqrt(段长)作为下限，作signed-asinh变换；距离为0.5坐标经验W1+0.5十六方向联合经验W1，外权仍为2。wave和regular路径、AdamW lr2e−4/weight_decay1e−4/clip5及三假设均值保持一致。

## 初始化与随机流

两个初始化种子2026101540、2026101640。每种子在128个固定SOURCE TRAIN父、每库4draw、2个shape库上计算384点实际纯随机电压RMS，以1 ADC LSB为上限。缓存一次冻结A的decoded，初始头权重在[0,1]作固定16次二分并选满足上限的lower bracket；原头已低于上限则保留，不放大。校准不用REAL loss，未来voltage gain仍为.05，不施加永久幅度乘子。两个候选复用同校准权重字节和回执。

各个sampler、shape、005、extension、wave projection、integral projection、校准及probe使用配置中的独立显式随机地址。每次训练reset本seed各流，C0/C1数据流逐步配对。全6593 SOURCE TRAIN和21047 REAL TRAIN以FullCoverageSampler循环；每步512父×2draw对1024 REAL，前向chunk256。评价用013原两库种子、固定评价投影与chunk128。

## 预算和资源

固定顺序seed0-C0、seed0-C1、seed1-C0、seed1-C1，各2000G，最多8000G。梯度probe见[014评价口径](WP2_ACCEPTANCE_CONTRACT.md)，额外AD最多1872，训练backward8000，总AD最多9872。真实NN/ADC/CP等完整上限由已冻结driver的 `budget_caps014` 公式写入 `wp2_014_config.json`，包含校准head map，entered与completed分开累计。

CPU12、GPU10 GiB、进程树RSS8 GiB；每trial墙钟7200秒/输出1536 MiB，整批28800秒/6144 MiB。一个科学作业树，监督仅管理本批。任一失败保留native exit、durable操作日志、已完成训练和在途未知；无自动重启。

## 启动与回收

固定源码、输入、角色与数字T的SHA后，在新目录 `/home/rylan/projects/ej276d-n01/wp2_014_20261015` 启动。读取首次真实G1证明及同snapshot健康状态后，设置一次预计回收时的唤醒并休息。回收按完整远端库存SHA验证；本地E空间有限时只取经SHA绑定的小件与新头checkpoint，完整raw保留远端并明确本地未回收/未独立读回范围。

保留旧009—013全部结果。四次短试计入最多12次搜索预算；教师8次、最终全流程折内重拟合及冻结后全TRAIN部署预算独立。最终硬评价、用途门槛和数字T保持固定；本阶段不把同TRAIN开发收益称独立采集泛化。
