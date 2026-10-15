# 016 修复版首次native启动

状态：START_NOT_CONFIRMED_OR_FAILED_AT_ONLY_SNAPSHOT。源码提交 `99464dd21544d4f6ef60f348015759b449fafaa2`；配置、manifest、冻包和controller完整SHA见同级JSON。

实际分派时间：2026-10-15T07:17:55.515144+00:00；首更新证明时间：None；唯一状态快照：2026-10-15T07:23:33.039797+00:00。supervisor=182393，child=182403，start ticks分别为5307721、5307736。

同快照健康：`native_failed_same_snapshot`。实际完成G=0、AD=0；producer计数与durable逐项一致，entered/complete各计数、unknown inflight、失效和journal尾部状态在JSON保存。当前owned-tree资源：RSS None字节、GPU None字节、输出None字节、wall None秒。若进程已原生退出，当前资源为空，原生退出回执的峰值另行保留。

本次动作：deploy1、start1、单次首事件存在等待1、verify-start快照1；底层SSH5、源码SCP1。首事件等待只检查文件存在，没有读取进度、追加数值探针或状态轮询。旧失败bootstrap记录单独保留，未自动科学重试。

冻结预算仍为两个seed各2000G/AD、共4000G/AD、CP4。实际已完成与预算分开保存；累计175667为本作业之前的既有累计，不由本施工代理改账。所有后续SSH/status/poll已停止，快照之后的远端变化未知。root负责后续处理与元数据。

保存的producer错误：`016 per-parent A canonical Q changed across draws`，失败时间2026-10-15T07:18:16.034425+00:00。有界等待超时只表示未见首事件；实际失败由唯一verify快照的FAILED和native exit1证明。

工程启动验证与科学响应匹配分开。权重未晋级，scientific_acceptance=false。
