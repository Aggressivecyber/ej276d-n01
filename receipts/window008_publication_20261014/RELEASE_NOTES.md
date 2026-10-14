Window008 resume01 的 256 点固定训练、主评价和补充诊断已完成。deterministic、full、simple 各达到 19,000 G，有效终点合计 57,000 G / 144,000 D；包含原数值失败和用户停机在内的 Window008 实际历史为 57,667 G / 146,001 D。

**物理响应匹配尚未建立（response_match_not_established）**。完整随机臂在 TRAIN、DEV 的主要整波形距离上仍劣于确定性对照。主回放和补充诊断数值回读通过，报告 10 页均已渲染检查。

- `REPORT.pdf`：可直接下载的 10 页中文报告。
- `REPORT_EVIDENCE.zip`：PDF、HTML、10 张图、SUMMARY、完整 EVIDENCE 和 QA；解压后打开 HTML。
- `FROZEN_SOURCE.zip`：191 个冻结文本文件的原始字节，及 13 个未公开二进制输入的路径与 SHA256。
- `INFERENCE_WEIGHTS.zip`：三组响应终点和复用 noise005 的纯模型参数、构造统计、逐模型来源清单。所有导出均经 weights_only 回读、严格加载和张量摘要一致性检查；包含正确模型类与加载说明。
- `PUBLICATION_MANIFEST.json` 与 `PUBLIC_SHA256SUMS.txt`：附件大小、来源和 SHA256。

原始 REAL/SOURCE 波形、逐事件 REAL 观测量、优化器/采样器状态和完整私有归档继续保留在本地。发布包中的权重用于推理；完整训练复现需要已登记的私有输入。noise005 的长程延拓仍区分 fast/slow/common_level 三种假设。此次发布新增训练更新 0、模型前向 0。

需要逐字节复核冻结清单时，请使用本 Release 的 FROZEN_SOURCE 附件；GitHub 自动生成的 Source ZIP 可能采用 Git 换行规范。
