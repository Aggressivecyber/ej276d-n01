# 016 唯一替代执行已正确启动

唯一当前快照 2026-10-15T08:11:31.762570+00:00 通过 `current_owned_process_chain_and_resources_verified`；native exit、FAILED 和 COMPLETE 均未出现。分派 2026-10-15T08:08:15.165634+00:00，worker 启动 2026-10-15T08:08:15.327770+00:00，首真实 G1 2026-10-15T08:11:03.566864+00:00。首事件只等文件存在，用时 122.017 秒。

快照时 durable 已完成 G/optimizer/AD/backward 各 137；AD/backward 已进入 138，operation [8447] 的完成状态未知。A 200649、head/noise/charge 各 194056、ADC/gather 898632，calibration A6593、解析拟合1、CP3。无失败操作或截断尾。完整 entered/completed 与固定预算分开保存于 JSON。

首步两个 θ 均有非零实际更新：L2=0.014125471605424598，Linf=0.0099922099741469594。结合两个参数、fresh AdamW、wd0，可推两梯度均非零；未单独记录逐参数梯度值。全6593正Q与初始化完成证明有效，原1LSB解析合同不变，详细校准数值待后续回收。A/005/M0字节保护通过。

supervisor=191479 / tick5609686；worker=191486 / tick5609702；boot `75229583-9bb2-4225-9f0d-12e04e1c1726`。当前 RSS 4194267136 B、GPU 8854175744 B、输出 800094062 B、wall 196.467s，CPU12与本作业身份、命令和cwd均通过，全部低于原上限。

source commit `cc68f75d7e17f675a5ad0605167320cc3e9c93a3`；配置、manifest、冻包、controller 完整 pins 与所有原生调用回执 SHA 在 JSON。当前替代执行仅 deploy1/start1/presence-wait1/verify-start1，底层 SSH5/SCP1，无额外探针或轮询。原失败的非训练成本独立保留；新预算仍为同两个seed共4000G/AD、CP4。

所有 SSH/status/poll 已停止。快照之后的状态未知，根节点负责一次未来唤醒与元数据。权重未晋级，科学响应匹配尚未建立。
