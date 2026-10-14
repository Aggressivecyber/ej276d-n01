# 009 结果与 010 决策

009 已完成：原生退出 0，34.54 秒，冻结权重/缓冲和输入字节均不变。512 SOURCE TRAIN 父 × 16 次；A/B/C 共 24,576 行信号前向，73,728 行硬观测，优化器更新 0。独立复核通过全部 81 个结果文件 SHA256/大小、21 个波形协方差因子、9 个标量协方差块、配对均值/MCSE/父簇区间及全部硬观测链。

| 对象 | 噪声假设 | wave 距离 | QPSD 距离 |
| --- | --- | ---: | ---: |
| A | fast | 0.186019 | 0.091139 |
| A | slow | 0.187291 | 0.081210 |
| A | common_level | 0.186171 | 0.091266 |
| B | fast | 0.532406 | 0.057366 |
| B | slow | 0.531537 | 0.060472 |
| B | common_level | 0.530201 | 0.057215 |
| C | fast | 0.399108 | 0.311570 |
| C | slow | 0.401874 | 0.285240 |
| C | common_level | 0.400292 | 0.312327 |

wave/QPSD 使用同样的 8192 REAL 参考、TRAIN 拟合尺度与固定评价投影。A 为 deterministic 19k，B 为 full 19k，C 为 B 自身关闭全部信号随机项的路径。D 仅从 B 的重复数组派生，不进入分布验收。REAL–REAL 同数量、互不重叠参考的 wave=0.029848、QPSD=0.033959，属于本次经验控制，不是不可约误差下限。

逐父纯信号差异 RMS 的均值分别为 D−C 3.16579 mV、D−A 19.85873 mV、C−A 19.54524 mV。D−C 的逐父 MCSE RMS 均值为 3.90187 mV，去抽样方差后的均方区间跨零，16 次重复没有分辨 preADC 条件均值漂移。全局均值曲线会抵消，不能用全局 D−C=0.121 mV 推断所有父事件差异很小。

硬观测后 fast 的逐父 D−C RMS 均值为 8.38659 mV、MCSE 3.11661 mV；slow/common_level 相近。mean Obs(B) 与 Obs(mean preADC B) 的逐父差异 RMS 均值为 8.11714 mV。寻峰、ADC 和裁窗的非线性作用已在保存数组中检出。D 比 C 更接近 A 的父事件为 213/512，也说明 C−A 差异不能单独认定有害均值漂移。

本轮选择 010 单臂、2000 G 的受控候选：冻结 A 的全部参数/缓冲，以确定性输入条件化的 B(x) 与零均值显式 z 线性相乘，在电压域加入平滑零 canonical Q 随机形状；先使用原分布损失和 noGAN。目的在于检验固定均值是否能避免现有随机参数化的问题，尚未证明该候选有效。preADC 零均值/零 canonical Q 不扩张为 observed Q/PSD 不变。010 的设置在训练前独立登记，不根据已暴露 DEV 调参。

详见 [010 协议](SHAPE010_PROTOCOL.md)、[独立数值回执](../receipts/mean_random009_20261014/postrun_numeric_01/NUMERIC_READBACK.json) 与 [决策回执](../receipts/mean_random009_20261014/INTERPRETATION_RECEIPT.json)。科学状态仍为 response_match_not_established。
