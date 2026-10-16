# 026 独立源码审核

结论：**no_blockers**。本回执绑定最终 CFG `0fef13df884415bf363d8926ff098d6c8169098c56e4eb033f3b350c14c1dfcf` 与 driver `8f6890a9db4027f289370dcabeca4a716b5e630e7739ba02b13ba11084005d2b`。9 个输入共 27,790,366 B、9 个 source/control、5 个合同的原字节一致；draft→frozen 仅 status/source_files 变化。025 已结案的 17 fail、42 pending 与独审回执均绑定，未回填通过。

实际 project import 路径只有 driver 与 wp1_ops 的 Store/读写/哈希辅助。旧 Linux supervision helper 的裸 import 不可达，未纳入026执行闭包。初次审核器把全模块 AST import 集合当作可达依赖而停止，现按实际调用图排除并记录；未重跑已通过的 SHA 检查。

SOURCE 按完整字符串 ID，从 8192 份保存 metadata 的 TRAIN 集合一一连接到原 6593 父顺序；h 沿原013已核公式与冻结输入。三套比较权重正且位于 [.8,1.2]，各自归一后每父质量等分四 draw；原 physical weights 与 axes 保留，不乘入新比较权重，也不赋予新的物理解释。两个 bank 分开，REAL ESS 明确属于保存行测度，422 逻辑组不作为422次独立采集。

每个标量独立有限 mask、质量归一、原分母与支持数均保留；ESS 先把有效 draw 质量聚到父上。全空 mask 返回 null。全行 flags 不借用标量 mask。Q/front/tail/peak 的 within 使用分母4，B+W 对直接加权总体方差的恒等式正确；72 次经验分解不等于真实条件核方差恢复。最终数学 AST 与唯一既有人工检查相同（fixture `58bcaa68…`），本审未重跑。

计数覆盖 18 格、144 fake/8 REAL 标量摘要、162 fake/9 REAL flags、72 分解，新增 NN/CP/AD/G/RNG/ADC/CDF/参数点全0。Windows 边界限定本进程最低允许单核、BLAS/OMP1、E盘 owned cache、阶段 peak working set≤1GiB、最终 output≤64MiB 与 wall≤300s；ROOT 按既定外层 subprocess timeout300 执行。内存快照止于最终结果序列化前，不能冒充整个 native 过程的峰 RSS 监测。

本审只做静态与 RAW SHA 核验，私有 NPZ 数值解码、科学执行、人工/旧套件重跑、SSH、共享状态/Git 修改均0。源审不替代原生终点、实际身份连接、完整签名与代数读回。026只提供固定 M0 组成 stress 开发证据，不识别真实源人口、不宣布 full-wave 稳健或响应接受，不自动打开额外权重、theta、T2 或 REAL 长训。
