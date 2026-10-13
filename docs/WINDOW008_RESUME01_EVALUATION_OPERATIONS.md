# Window008 resume01 评价执行与回收

训练已在 2026-10-13 18:10:23 UTC 完成。新恢复任务 native exit0；原数值失败和用户停机两代 native1 分别保留。`receipts/window008_resume01_20261013/COMPLETE_COLLECTION.json` 记录31个文件的回收与逐字节校验。有效三臂各19000G，总57000G/144000D；历史实际57667G/146001D，N01全局157667G。

科学源码冻结前，独立审查评价适配器、严格回放、诊断统计与资源监督。主清单包含24个增量源文件，另建诊断清单绑定主清单而不形成哈希循环。主评价原数值核心不修改。冻结后新增文件不允许更改；如工程检查失败，保存原冻结和失败现场，明确修订版本后重新登记，不能按DEV结果调参。

服务器为 `rylan@100.82.231.102`，根目录 `/home/rylan/projects/ej276d-n01/window008_resume01_20261013`，Python为 `/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python`。本地根目录 `E:/GAN/research/N01`，Python为 `E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe`。本地统一 `-B`、`PYTHONUTF8=1`，TMP/TEMP指向 `E:/GAN/runtime_tmp`。

1. 本地源码提交后依次运行 `scripts/freeze_window008_resume01_evaluation.py` 与 `scripts/freeze_window008_resume01_diagnostics.py`。两份清单及冻结回执都独占新文件。
2. `scripts/stage_window008_resume01_evaluation.py --package <新E盘目录>` 生成增量包；`scripts/deploy_window008_resume01_evaluation.py --bundle <BUNDLE.json>` 转移并执行有哈希保护的安装。原177个训练源文件逐字节不变。
3. `scripts/check_window008_resume01_diagnostics_remote.py` 在服务器运行一次人工全形状及实际类128行CUDA基准，使用独立进程树RSS8GiB监督和600秒超时。人工模型、实际类随机初始化与实际科学前向分别计数；它不读取真实SOURCE/REAL/DEV数值或训练权重。成功native0后才进入科学评价。
4. `scripts/record_window008_resume01_evaluation_state.py --phase prepared` 登记三个零更新作业。
5. 服务器运行 `python -B scripts/start_window008_resume01_evaluation_remote.py --kind primary`。独占 `runs/evaluation_jobs/primary`；正确启动后保存一次 `scripts/snapshot_window008_resume01_evaluation.py --kind primary` 快照并用状态脚本登记。长计算更新同一个n01单次唤醒，期间休息。
6. primary独立native0、全部产物SHA和门通过后，以同样方式启动 `--kind replay01`。严格比较全部13份NPZ的所有实际成员、dtype/shape/value/member hash、chunk/parent/stream/hash、距离和1000父bootstrap数值。早期人工夹具66成员不是实际科学成员上限；只排除原声明的时间、ZIP容器和摘要产物哈希等volatile字段。
7. 用 `scripts/collect_window008_resume01_evaluation.py --kind primary` 和 `--kind replay01` 分别回收；每个已存在回执和目标文件不可覆盖。主评价和回放均native0才允许服务器运行 `python -B scripts/start_window008_resume01_diagnostics_remote.py --kind primary`。
8. 补充诊断用 `scripts/snapshot_window008_resume01_diagnostics.py --kind primary` 一次快照及 `scripts/collect_window008_resume01_diagnostics.py --kind primary` 回收。计划实际29952信号前向/89856臂×噪声假设观测行，其中独立新增主体28416信号行；身份和顺序工程重放1536行另列。全部G/D/noise/population/C2ST更新为0。
9. 按PDF技能先记录操作标记，再运行保存的报告生成器，渲染并查看全部页；报告实际训练完成、数值一致和物理响应匹配分别披露。源权重、日志、数值、原生回执在E盘私有完整归档，D盘仅作空间允许的小源备份并保留至少256MiB。只做本地提交。

每个阶段只接受真实当前进程身份、原生退出回执及不可变产物证据。SSH失败不等于科学失败；不对状态未知作业重启。不以评价选择终点、噪声假设或训练设置。有效波形256点，内部384点。TRAIN与已暴露DEV分别报告；单次采集、无独立标定或逐输入重复实测真值，科学状态保持 `response_match_not_established`。最终交付完成后暂停n01定时任务。
