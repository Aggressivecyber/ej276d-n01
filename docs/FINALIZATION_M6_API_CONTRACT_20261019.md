# M6 研究接口：冻结行为与黄金输出验证

2026-10-19。S 已因原生保护门失败关闭；本阶段交付可复验的研究接口，科学状态保持 `response_match_not_established`。所有 24 份原指标已从保存的完整原生载体独立重算并精确复现，见 [验收](../receipts/finalization_saved_replay_20261019/terminal_01/TERMINAL_REVIEW.json)。

## 接口与冻结边界

`n01.finalization.api.load_response(config_path, device)` 提供两个明确入口。原 `n01/wp6-028-config/v1` 继续调用原 `load_reference028` 和 `ResearchResponse028.sample`；新研究配置仅表示本次已关闭的 S。旧 028 文件及此前 543 个部署文件均保持原字节。

共同调用为 `sample(source_waveforms, parent_ids, n_draws, seed, model_seed, noise_hypothesis, return_contributions)`。旧 028 门面另保留原 `shape_enabled/charge_enabled` 开关。S 的标量电荷支路固定关闭，模型头固定为原 038/M0 的两个头，BASE054 和 S 算子、TRAIN profile、归一尺度、OSQP 1.1.3 设置均固定。

输入是已准备的 float64 `[P,384]` clean SOURCE 电压，单位 V，软件步长 1 ns；接口不按长度猜测 480 点准备方法。旧 028 仅 CPU；S 可在 CPU 或 CUDA 计算神经/观察部分，QP 使用同一 CPU 求解器。模型加载读取冻结权重/统计和 PCA，接口本身不打开 SOURCE/REAL 人口归档，也不拟合任何统计。

S 沿用 `seed039(seed,"FIT",0,stream)`，流为 shape/noise005/extension，块长 128。输入按父事件优先、draw 次序展开。每父解一次均值 QP，每 draw 应用同输入确定的线性随机投影；同一完整输入 batch 和 seed 精确重放。父 ID 仅作标签；调整行序或 batch 长度可能改变分配到该父的随机数。旧 028 继续使用其原完整 batch 随机流和 64 行神经块。

输出含硬 wave256、ADC384、基线、峰/裁窗索引、Q、front=Q−tail、tail、raw/guarded PSD、父/draw 映射及原异常标记。截尾/未定义形态标记沿用 018；另给出均值峰与终态峰不同、ADC 前两样本并列和 profile 端点钳位标记。诊断模式保存均值、原随机量、S 修正与电子学分量。结果带有 `model_accepted=false` 和当前 S 失败状态。

`finalization_run.py` 校验配置及输入 SHA，运行后保存完整 native NPZ，重新打开所有成员逐字节核验，并输出带身份、数组 schema 和调用计数的回执。原始 SOURCE 示例仍是私有输入，不进入公开包。

## 单次黄金输出作业

|检查|固定范围|通过标准|
|---|---|---|
|旧028兼容|原2父×2draw×2头×6调用；48输出行|原黄金文件的396个张量逐字节相等，原开关与状态不变|
|S集成|原SOURCE前128父×2头×3H，原FIT种子；768输出行|原保存ADC码、峰、原flags严格相等；float64电压/均值/随机修正绝对误差≤1e−12、rtol=0；同时记录是否逐位相等|
|重放与身份|S原128父重放一次，再仅改ID一次|完整原生输出和贡献逐字节不变，输出ID正确改变|
|多draw|2父×3draw，重复seed和另一seed；18输出行|父/draw映射准确，重复seed逐位重放，另一seed改变输出，诊断开关不改数值|
|输入边界|6个非法输入，无新推理|非float64、错误长度、NaN、零draw、负seed、未知H均在推理前拒绝|
|实际CLI|2父×2draw；4输出行|本机原生进程退出0，完整输出全部成员无损重新打开|

S 的浮点工程容差沿用既有 1e−12 V；这是跨 batch/实现布局的数值核对，原研究保护门保持不变。旧 028 黄金结果仍要求逐位相等。所有比较检查 NaN 位置，未定义项保留。

本次总推理输出 1094 行，18 次检查点加载，PCA 解码 2 次；原 TRAIN 归档只解码 SOURCE carrier/ID/role 三成员一次，REAL、DEV、AD、优化、人口级指标评价和新候选均为零。验证器 S 部分 11 调用、1030 个父均值 QP、1042 个输出；CLI 另 2 个父 QP、4 个输出。具体逐操作计数由配置固定并在终态精确核对。QP 零自由度若出现即与冻结黄金人口不符，不丢行。

资源为 4 CPU、GPU 分配器合计上限 6 GiB（主验证和 CLI 各 3 GiB）、RSS 12 GiB；内部 780 秒、外部 900 秒加 20 秒退出宽限，输出 32 MiB。原 028 阶段沿用 1 个 Torch CPU 线程。一次一个 N01 作业，不自动重试；错误保留当时调用阶段、已记计数和失败输出。配置/源码/黄金输入 hash、输出差异、计数或资源任一不符即失败。

验证通过只确认当前研究接口能重现冻结行为。当前没有合格的新端点；S+R、新确认库、四系统公平训练、三折全流程验证和新 C2ST 均未触发。后续交付据实际证据完成方法说明、失败卡、图表及论文材料。
