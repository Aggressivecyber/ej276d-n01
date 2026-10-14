# Window008 resume01 GitHub 交付

2026-10-14 用户要求上传 GitHub。本次公开交付对应已完成的 256 点 Window008 resume01；训练、评价、诊断和私有归档的历史回执保持原样。

[Release 下载入口](https://github.com/Aggressivecyber/ej276d-n01/releases/tag/n01-window008-resume01-20261014)

| 文件 | 内容 |
| --- | --- |
| `N01_WINDOW008_RESUME01_REPORT.pdf` | 已逐页检查的 10 页中文报告 |
| `N01_WINDOW008_RESUME01_REPORT_EVIDENCE.zip` | PDF、HTML、10 张图、SUMMARY、完整 EVIDENCE 和 QA；解压后打开 HTML，相对链接可直接使用 |
| `N01_WINDOW008_RESUME01_FROZEN_SOURCE.zip` | 三份训练/评价/诊断冻结清单对应的 191 个文本文件，保留原始字节；附 13 个未公开二进制输入的路径和 SHA256 |
| `N01_WINDOW008_RESUME01_INFERENCE_WEIGHTS.zip` | deterministic/full/simple 三组 19k 终点及复用 noise005 的纯模型 state_dict、构造统计、来源清单 |
| `PUBLICATION_MANIFEST.json` | 附件、原始检查点、模型张量和冻结清单的 SHA256 及验证记录 |
| `PUBLIC_SHA256SUMS.txt` | 上述附件的 SHA256 清单 |

三组有效终点合计 57,000 G / 144,000 D；主评价数值回放和补充诊断回读通过。完整随机臂在 TRAIN、DEV 的主要整波形距离上仍劣于确定性对照，科学状态为 `response_match_not_established`。数值回放通过只说明已保存计算可复核，物理响应匹配尚未建立。

推理权重均由已核验原始检查点导出，使用 `weights_only=True` 回读并严格载入对应模型，张量摘要与原始终点一致。包内保留各臂正确的模型类。noise005 的 100 点基线仍需按冻结代码使用 fast/slow/common_level 三种长程假设及既定 ADC、基线、寻峰和 256 点裁剪流程。发布过程新增优化更新 0、模型前向 0。

公开内容沿用既有 Release 的范围：模型参数、源码、生成结果与聚合证据。原始 REAL 和 SOURCE 波形、逐事件 REAL 观测量、采样器和优化器状态及完整私有归档保留在本地。推理权重不支持续训；完整训练复现仍需已登记的私有输入。

仓库自动生成的 Source ZIP 可能采用 Git 换行规范；需要逐字节检查冻结清单时，使用本 Release 的 `FROZEN_SOURCE.zip`。打包脚本为 `scripts/package_window008_publication.py`，输出目录必须是新的 E/D 盘目录。
