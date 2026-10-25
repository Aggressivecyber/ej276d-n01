# 有限硬目标结果与PSD曲线：GitHub发布说明

本次发布覆盖245—252的有限硬目标实施记录、最小接口修正、冻结配置、关闭证据与PSD分布图。252进程退出0，768次搜索评价加一次唯一返回端点重放；完整wave改善1.766859949%，低于原2%门槛，QPSD改善0.200063850%，18风险/14异常及均值RMS保护通过。P2/P3未执行，端点未采用，受限平均响应和完整科学验收均为false。248/250/251的工程中断及本系列4次已启动拟合成本保留。

- [252关闭报告](../deliverables/finite_native_goal01_252_20261025/REPORT.md)
- [PSD分布曲线](../deliverables/finite_native_goal01_252_20261025/psd_distribution/PSD_DISTRIBUTION.png) / [SVG](../deliverables/finite_native_goal01_252_20261025/psd_distribution/PSD_DISTRIBUTION.svg) / [PDF](../deliverables/finite_native_goal01_252_20261025/psd_distribution/PSD_DISTRIBUTION.pdf)
- [PSD曲线数值](../deliverables/finite_native_goal01_252_20261025/psd_distribution/PSD_CURVES.csv) / [全范围CDF](../deliverables/finite_native_goal01_252_20261025/psd_distribution/PSD_FULL_RANGE_CDF.png)
- [全部768点](../deliverables/finite_native_goal01_252_20261025/ALL_768_SEARCH_TRIALS.csv) / [风险与异常](../deliverables/finite_native_goal01_252_20261025/RETURNED_POINT_18_RISKS_14_FLAGS.csv)
- [独立数值核收](../receipts/finite_native_goal01_recovery01_20261025/wake_20261025_0627/saved_readback/closure_readback_20261025/SAVED_ARRAY_VERIFICATION.json)
- [发布清单](../receipts/finite_native_goal01_recovery01_20261025/github_publication_20261025/PUBLICATION_MANIFEST.json)

本仓库提交源码、配置、汇总数据、图表和小型回执。原始波形、模型/端点NPZ数组、运行日志与缓存、完整数值ZIP结果包遵守既有仓库规则，留在E/D或原服务器活动盘；报告内指向这些内容的路径用于本地重算，GitHub中不提供对应载荷。现有冻结报告、哈希和结果包原件保留。NAS无访问或核验。

PSD图使用同一FIT人口：REAL18948条，M0与252各6465个SOURCE父×2库×8draw。采用原尾积分/总积分定义，统一高斯带宽；全部样本参与归一化，全范围ECDF保留负尾部。绘图无新拟合、端点选择、模型生成或随机draw。
