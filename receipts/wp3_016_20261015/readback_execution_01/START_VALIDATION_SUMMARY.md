# 016 CPU审计在唯一启动快照已原生完成

首个正式pair 2026-10-15T09:23:04.709004+00:00；两bank charge检查在该证明前完成。唯一快照 2026-10-15T09:24:00.266454+00:00 已是完整native0：COMPLETE 2026-10-15T09:23:44.151883+00:00，supervisor退出 2026-10-15T09:23:45.748999+00:00，200项操作 entered=completed，全部16计数匹配，inflight/failed空。

112个原始NPZ共2,169,655,337B各SHA一次并decode一次：48HARD、48REPORT、8baseline、8charge；24双bankpair及210,976条实际A384记录完成。实际训练、模型、CP、AD、新随机、原TRAIN/DEV读取和GPU预算均0，累计G保持179667。

原生用时 44.300s，supervisor采样峰RSS 417308672B，输出 249037B，低于原cap。退出后无当前RSS/VM/CPU亲和性值；VM3072MiB和CPU2由登记子进程与worker约束，不把不存在的实时测量填0。

本次prepare/start/presence/snapshot各一次，SSH5/SCP1，全部远端调用已停止。ARRAY_READBACK声明SHA `7337c5ab50929eb4381fc29ded4af138420b8257ba0c7b9945a7c82ca6be604a`，尚待根授权小文件回收；本地未回收原始NPZ。无需运行中唤醒。科学接受与权重晋级仍待独审闭合。
