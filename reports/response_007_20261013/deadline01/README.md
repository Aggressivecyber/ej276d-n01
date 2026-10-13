# N01-007 有限阶段结果

两臂各4,000 G，累计8,000 G（本执行新增7,999）及6,000训练判别器更新；原每臂19,000 G终点未完成。科学状态为 `response_match_not_established`。

完整随机臂相对确定性臂的三假设最坏波形距离降低18.2862%，Q/PSD距离降低28.1264%。计算式为 `1 - max_h(distance_full[h]) / max_h(distance_deterministic[h])`，两臂都是4,000 G末点，使用同一固定匹配选择（每侧2,048条）和投影上下文。不同指标、不同臂的最坏假设可以不同。这不是相对REAL–REAL、旧006或物理等价性的改善率。

完整随机臂的最坏wave/QPSD/noise距离仍为同采集REAL–REAL对照的1.9923/1.4716/3.0418倍。冻结基线未改善；输出Q条件宽度仍有偏差。低信号Q箱的三假设PSD宽度最大/最小为1.1006，父级95%区间为[1.0893, 1.1128]。各臂自身固定信号Q箱父人口不同，敏感性比值不能用作臂优劣排名。

在已追溯归档及合法TRAIN原记录审计中，没有得到480点静默窗口的证据。处理后的500点记录在120–140点出现半高上升沿；100点基线只是处理约定，不能证明纯电子学噪声。硬件时间标定、长程噪声和独立采集仍需补证。

[HTML报告](N01_RESPONSE007_STAGE_REPORT.html) · [PDF报告](N01_RESPONSE007_STAGE_REPORT.pdf) · [各假设完整摘要](SUMMARY.json) · [完成回执](../../../receipts/response_007_20261013/deadline01/FINAL_RECEIPT.json) · [本地原生重放回执](../../../runs/response_007_20261013/deadline01/runs/verification_deadline01/REPLAY_VERIFICATION.json)。服务器原生重放位置为 `/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01/runs/verification_deadline01/REPLAY_VERIFICATION.json`。

训练和完整重放均原生退出0。76个冻结源文件、22个完成产物、178个共同状态项与前3,000步配对指标核验通过；207个生成分块、213个数组成员、1,000次bootstrap和7个分布比较复现一致。复现不增加训练或独立随机样本，没有读取DEV。

三页PDF已全部渲染、视觉检查并重新打开。最终PDF SHA256：`deef3b4f0b3076e8389e0397895f3f74c09f542d09ff988318574ecba02c1ef8`。视觉检查绑定见 [VISUAL_QA](../../../receipts/response_007_20261013/deadline01/VISUAL_QA.json)。全部审核由主代理完成，不称第二方独立审核。

下一步：先核实后续Astra/Max/Fast运行设置；再登记长窗噪声约束与响应差异的最小可判别诊断。当前不追加训练，原007完整末点尚未执行。 先前的截止约束后来被取消，已完成的有限阶段及其冻结预算仍原样保留。当前没有新公开发布或上传共享。
