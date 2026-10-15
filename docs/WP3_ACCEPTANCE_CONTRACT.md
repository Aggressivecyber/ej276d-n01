# WP3 / 015：开发选择和验收口径

本次以[015固定配置](WP3_015_CONFIGURATION.md)和[WP3数据角色](../WP3_DATA_ROLES.json)为阶段合同，继承原[硬观察定义](../OBSERVATION_CONTRACT.json)及[完整评价合同](ACCEPTANCE_CONTRACT.md)。数字T沿用CALIBRATED_TOLERANCES_WP1_013.json，SHA256为f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b。它是本计划预登记的人工灵敏度标尺，不能称为硬件实测标定。

mean gate在任何checkpoint、NN、优化器和新随机数之前执行；阴性结果是合法的native COMPLETE / training_skipped，模型、更新和AD实际费用均0。阳性只允许这四次固定短试，不构成条件均值识别或科学成功。

每seed/hyp先平均两个库各自距离，不能改为pool8距离。候选M0/M1必须在同一ALL6的每个正且有定义T分量上同时满足mean2bank(final−A)≤T和mean2bank(final−共同C1初始)≤T。必需已标定分量缺失或undefined使候选pending、不eligible；零/undefined T的未标定分量继续pending，不自动PASS。合格候选以2seed×3hyp×2bank的12个final完整wave距离均值排序，1e-12内tie选M0；胜者分数未比共同初始低超过1e-12，则继续保留原C1参考、均值模块关闭。

这一选择只确定下一阶段的研究参考。ALL6相对A完整wave改善≥2%的用途门、绝对distance/T、原native QPSD/条件PSD/噪声/形态/输入依赖和科学确认分别报告。重复draw不能代替独立父事件，逻辑组不能代替独立采集。M1更低的总体误差不识别真实条件均值，代码路径和权重参考选择也不代表完整响应匹配。

工程完成、在域开发收益与跨条件物理复用沿用户主计划分别判断。当前科学状态为response_match_not_established；005基线DC不由signal prefix100之外的均值适配器修复，不通过删行或修改T使候选合格。
