# WP6-028 独立源码审核

结论：对本回执绑定的最终三文件，静态审核通过，无未解决阻断。配置冻结和源码审核仅覆盖登记的软件工程范围；实际原生执行及保存输出仍待各自回执。

审核人未参与实现。已按 N01 AGENTS、用户主计划、027 研究参考资料和原 016/015/008 接口核对；本轮仅阅读源码与 JSON/Markdown、重新计算文本身份及 AST，不 import 科学模块，不读 CP/NPZ 二进制，不运行 NN、RNG、ADC 或旧 suite。

| 最终文件 | bytes | SHA256 |
| --- | ---: | --- |
| src/n01/inference028.py | 14491 | 17e32e76ab86185c02e68e6a54dc9f9e7568cb1a9c2b2174581ade16514bc2f7 |
| scripts/run_wp6_028.py | 16622 | 4e2a132f9680fd13344a6fc6a436ca10596fe61f814dae84fa0431fdb928b3f5 |
| wp6_028_config.json | 15892 | 48fbc9639a9435dd3d02bfde4d3568f4feabcf9b6f18f2f4d0fc48b6acde4b0a |

11 个源码 pin 均重新读回一致；实际 n01 闭包为含 package init 的 10 文件，另加 driver。原 016 的完整 116 成员/37 项目导入清单继续作为独立 provenance 保留。六份 CP 的 bytes/SHA/远端路径逐项同 027 原记录；本轮没有重读 CP。

loader 只支持 ROOT 冻结的 028 config。A 与 005 先完成载入、冻结、eval 和设备设置，再构造 016/extension；Q1 每 seed 绑定原 M0、已有 charge initialization 与 final theta，不重新校准、创建 sampler 或 optimizer。当前冻结前后使用原 tensor_state_digest010，不把它冒充旧 response008.state_digest。固定六 CP 原 SHA 是本接口的明确适用边界。

原 016 的 chargeoff M0 对象/字节身份、shapeoff 仍保留 charge、原 005 prefix/scale 与选定假设的条件耦合均保留。全随机库先生成，再按固定 64 行分块；同 seed 是完整批次重放，不宣称任意拆批逐位等价。硬输出直接调用 observation008 的 ste=False，使用实际 ADC prefix100、原 first-argmax/裁窗、Q/tail/PSD 和七个现有 flags。没有额外 ADC、删行或新开关。

原生预算逐项同登记：12 次调用/48 行 A、head、005、选定 extension 和硬观察；6 次 CP load；采样 60 次 randn/18,912 normal scalars；构造 118 次 uniform/1,475,668 scalars 与 2 次 normal/2,560 scalars。原 SOURCE NPZ 只打开一次、按原三个成员各读取一次，不读 REAL/DEV。

本轮必要修复已复核：NPZ archive/member 的 attempted 前置、completed 后置；loader 明确 run_id/status=frozen；最终 eval/frozen/grad-none 检查包含 extension。driver 的 AS/CPU 限制早于 NumPy/torch import。监督器 21 项字面迁移独立重放完全一致；RSS 与 artifact 大小是终态验收，wall timeout 含原 20 秒 TERM grace 后仍按超 600 秒失败处理。

独立输出读取提案见 READBACK_PROPOSAL.json：一次 EXAMPLE.npz、37 成员、182,940 值、1,386,296 payload bytes。归约采用预定 2·gamma_(n+1)·sum(abs(v)) 误差界；重放、copies、latent 与开关身份保持 dtype/shape/raw bytes 精确。仅复核返回 ADC 上的代数，不重新 digitize。需先核实际 ARRAY_SCHEMA，再由 ROOT 登记该一次读回。

科学状态维持 response_match_not_established，未选部署 seed、未晋级权重，完整 WP5/WP6 未完成。
