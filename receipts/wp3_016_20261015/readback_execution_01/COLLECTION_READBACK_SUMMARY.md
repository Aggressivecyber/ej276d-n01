# 016 CPU读回回收闭合

一次CPU任务原生结束，child/supervisor均0；200项 entered=completed，400条journal逐项重聚合，16计数与冻结计划完全一致，inflight/failed为空。

19个小文件逐SHA核验，共470,489B；tar 512,000B，SHA `765db742ccd336b533975eb3057a21ff8cc0304acd976b5e6aadefddd5aefdd2`。ARRAY_READBACK为PASS，SHA `7337c5ab50929eb4381fc29ded4af138420b8257ba0c7b9945a7c82ca6be604a`。

112原始NPZ共2,169,655,337B只在登记CPU审计中各SHA/decode一次，原地保留，未传输或在本地重新解码。48HARD、48REPORT、8baseline、8charge；24双bankpair、1,265,856 hard行、210,976 actual A384/mu/q draw及52,744父moment记录完成。原SOURCE输入只核保存digest与lineage一致，不重读原SOURCE。

原生用时44.300s；worker getrusage峰RSS 493,142,016B，supervisor采样峰RSS 417,308,672B。二者不同测量，均在1536MiB内；CPU2、VM3072MiB、输出与时限按登记约束，退出后当前RSS/VM/亲和性无值。

prepare/start/presence/explicit snapshot/collect各一次，SSH6/SCP2；collect按成熟控制器内置终点验证，未追加独立snapshot命令。所有远端调用停止，无CPU唤醒。CP/NN/AD/G/RNG/原TRAINDEV读取/GPU预算均0，累计G保持179667；原失败native非训练成本单列保留。科学接受及权重晋级仍为false，最终研究参考交独审按冻结规则确认。
