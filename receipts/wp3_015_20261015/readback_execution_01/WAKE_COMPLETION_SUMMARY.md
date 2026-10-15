015 保存数组审计已完成并回收，worker / supervisor 均以 0 退出。完成时间为 2026-10-15T05:51:21.480786+00:00；本次唤醒只做一次快照、一次小型结果回收，远端访问已停止。

- 36/36 对，161 份 NPZ 一次 SHA，160 份评价 NPZ 一次解码，共 1,898,784 行；285 次进入与完成操作、570 条日志事件一致，无在途、失败或截断。
- 用时 60.446 秒；worker 生命周期峰值 RSS 为 466.668 MiB，supervisor 采样到的进程树峰值 RSS 为 383.621 MiB，均在登记上限内。
- 回收 19 个小文件、513,764 字节，逐文件 SHA/大小校验通过；原 161 份 NPZ 留在服务器。
- 新增 CP/NN/AD/G/RNG、原 TRAIN/DEV 波形读取均为 0；训练累计保持 175,667。完整 CDF/loss、C1 median 与 preADC384 未重放；本次通过不代表物理响应匹配通过。

回收目录：`E:\GAN\research\N01\runs\wp3_015_20261015\independent_readback_collected_01`。

结果 SHA：`d0ed2adfba7eb193647dba54eb99bcc029cf02f31f206c699b3a7854545df2c3`。

JSON 摘要 SHA：`1ad797ba820044e537c52ed8a3a8e83b5ffab95b382355b3a1df3ad383f03c82`。科学选择与下一工作包由根代理依据独审推进。
