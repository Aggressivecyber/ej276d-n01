# N01 M6：方法、证据与公开复验材料

2026-10-19。研究接口验证和负结果材料交付已完成，科学状态仍为 `response_match_not_established`。S 在六条件下改善 wave 28.58–30.01%、joint 58.61–59.12%，但 QPSD/front/tail/PSD/负波瓣保护全部失败，没有合格最终模型。

|材料|入口|
|---|---|
|中文方法报告|[方法与结果](METHOD_REPORT_CN.md)、[HTML](METHOD_REPORT_CN.html)|
|接口与复验命令|[API_GUIDE_CN.md](API_GUIDE_CN.md)|
|英文论文和图表|[论文初稿](MANUSCRIPT_EN.md)、[六图说明](FIGURE_CAPTIONS_EN.md)、[相关工作](RELATED_WORK.md)|
|图文总览|[index.html](index.html)，下载后在浏览器打开|
|指标与分母|[数据说明](data/README.md)、[24 条完整条件指标](data/METRICS_ALL_CONDITIONS.csv)|
|数学复验|[合成演示](public_demo/README.md)、[实际运行结果](public_demo/verified_result/RESULT.json)|
|下载|[原始可分享包](packages/N01_M6_SHAREABLE.zip)|
|身份与范围|[原包成员清单](MANIFEST.json)、[本次发布范围](PUBLICATION_SCOPE.json)|

研究 API 的 523 项张量检查逐字节一致；完整保存结果复算精确重现 24 份指标和 158,232 行原生 ADC/wave。[API 验收](../../receipts/finalization_M6_20261019/terminal_01/TERMINAL_REVIEW.json)与[完整复算验收](../../receipts/finalization_saved_replay_20261019/terminal_01/TERMINAL_REVIEW.json)保留各自的实际范围。

公开目录包含已核验分享包的全部原始成员，并补充中文方法报告、接口说明和环境记录。`MANIFEST.json`只覆盖原包的39个内容成员；补充文件身份记录于[本次发布清单](../../receipts/github_M6_20261019/PUBLICATION_MANIFEST.json)。分享包 SHA256：`ddad2903bb576970d4bc03ab32f695952e9896c36797d885eb155ac40fc000ef`。

私有 SOURCE/REAL 波形、父标签/逐行索引、权重、拟合模板和私有快照保存在授权环境。公开包可复验数学算子和图表；完整神经推理需要接口说明所列的私有依赖与运行环境。历史设计文档中的私有文件路径保留作依赖说明。

M3 的 S+R 条件未触发，M4 新确认和 M5 公平比较、三折、同父随机核、新端点 C2ST 等没有运行。下一项科学工作需要为变更后的约束或目标另定有限设计。自动苏醒已关闭，当前无活动科学作业。
