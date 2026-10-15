015 已完成本地独立闭合，M0 成为下一阶段开发研究参考，均值分支关闭。

固定规则复算：M0/M1 均满足双参考 ALL6 正且有定义 T 非劣；12 格完整 wave 均值分别为 0.15012505860962605 / 0.15243191496746586，共同初始为 0.15042006897712382。M0 相对共同初始改善 0.00029501036749776666（0.196124%）；相对 A 的 2% 用途门为 6/6，M1 为 0/6。

保存数组 CPU 审查已完成 36 个双库配对、161 个原文件 SHA、160 个评价 NPZ、1,898,784 条硬输出行；285 个操作全部闭合，worker/supervisor native exit 均为 0。本轮只读取回收 JSON/journal 并验证哈希，未解码原 NPZ、加载 checkpoint 或执行模型。

绝对残差仍存在：M0 的 peak/rise/FWHM 分别约为 6.7/7.4/4.8 倍 T，baseline-mean 约为 33 倍 T；部分 QPSD/失效率也未达绝对 T。未标定分量继续 pending。科学状态保持 response_match_not_established，weights_promoted=false；014 原 C1 的 5/6 记录保留。

支持下一步仅用剩余两次 WP3 短试检验全局两参数 q_sig：冻结 M0/A/005、mean off，使用正值且解析均值校正的 Var(q|x)=aQ+bQ²；B 的幅度仍由确定性 A/输入控制。建议 s²=log1p(a/Q+b)、q=Q exp(sZ−s²/2)，并在开跑前解决 Q>0 与零方差边界的稳定参数化。关闭分支应精确回到 q=Q。

现有全人口 Q 标准差（代表格 M0 3.099316753724642、REAL 3.152829342399848）不能识别相同输入下的真实条件方差。峰值残差在低 Q 与全人口方向不同，rise/FWHM/DC 继续逐项保留；不把这些残差统一归因于电荷。新分支只作有界结构检验，未获原冻结用途/非劣门则保留 q-off，不补训扩大预算。

审查边界：完成 savedADC 逆变换/roundtrip、行特征/标志、native 4-draw 矩/MCSE、9×9 pairwise-finite 协方差及新派生 4+4 draw 分母 7 合并。未独立重放完整 CDF/loss、C1 median 或完整 preADC384；REAL mean 摘要为 SHA-only。经验 hard sample rank≤7 与理论 preADC 秩分别解释，缺失配对协方差无自动 PSD 保证。
