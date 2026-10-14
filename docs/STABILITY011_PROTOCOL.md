# N01 011：010 完整代理指标的父簇与 REAL 原组稳定性

用户于 2026-10-14 要求“继续推进”。010 的固定 TRAIN 比较已完成 2000/2000 G，原生退出 0；三假设平均 wave+2QPSD 改善 5.45%。本轮先分析该完整指标的重加权稳定性，再决定如何推进新头逐损失梯度与裁窗机制诊断。

## 目标与范围

固定已经训练的 A 与 010 终点、010 评价库、统计标定和投影。主指标 S=mean_h(wave_h+2QPSD_h)，主对比 Δ=S_final−S_A；负值表示终点距离较小。分析直接重算全部混合分布的距离，不能平均各父独立计算的距离来替代完整指标。

仅读取原 shape010/runs/primary/artifacts 的 A/final × fast/slow/common_level 波形、EVALUATION_BANK、COMPLETE/INITIAL/FINAL/BINDING 等元数据，以及原 window008_resume01 的冻结 FIT_STATISTICS008 与 DATASET008 中 real、real_ids、real_groups、source_ids。SOURCE 波形、DEV/VAL、任何权重不解码。原 005 和 A/010 均不实例化或前向，不建立优化器。新 G/D/噪声/人口/C2ST 更新均为 0，新增模型前向及梯度均为 0。

分析在服务器独立目录 /home/rylan/projects/ej276d-n01/stability011_20261014 执行。原 008/009/010 的源文件、权重、输出、协议和报告只读保留；新增本地源、回执、归档、临时、缓存均写 E/D。

## 固定抽样设计

SOURCE 是 512 个原父，每父完整保留已保存的 4 次响应，共 2048 行。REAL reference 是原 bank 的 2048 行，分属 419 个旧人工组；这些组来自同一采集的 50-record 人工分块，不是新增的独立采集或已验证的物理独立单位。REAL control 的 2048 行只重算原点 REAL–REAL 参考，不参与主 bootstrap。

固定 B=500 次联合重采样。每次独立有放回抽 512 个源父和 419 个 REAL reference 原组。源父的 multiplicity 赋给它的全部 4 次响应；REAL 组的 multiplicity 赋给它在 reference bank 中的全部已选行，不再抽组内行。REAL 复制后的总行数可变。fake 总质量按实际复制行数 2048 归一，REAL 总质量按该次复制后的实际总行数归一。不能将不同 REAL 组改为等行权重，也不能补齐到 2048 行。

源父 seed=2026101451，REAL 原组 seed=2026101452，均为 NumPy PCG64。各用 Generator.integers 一次生成 (500,512) 与 (500,419) 的有放回索引矩阵，再逐行 bincount 得 multiplicity。固定源父顺序为 EVALUATION_BANK 保存的 selected_source_ids；REAL group 顺序为已选 reference 的 np.unique 原组 ID 顺序。保存全部 500 份父/组 multiplicity。两份抽样结果在 A/final、三种假设和各损失分解间配对复用。不抽新的 latent/noise，不在父内二次抽样，不分层强制维持 Q 箱组成，不增加投影或另选假设。

## 完整指标重算

使用原 TRAIN 标定、固定的六组 wave views、32 方向 SWD、voltage/shape coordinate、四个 short-lag ×4 方向 joint 以及 16 方向 QPSD 和原条件箱。wave=.5×SWD+.25×coordinate+.25×short_lag；QPSD 保留原 global/conditional 归约。代理指标不含 noise、regular 或 adversarial 项。

投影仍是原 evaluation seed=2026101430，kind=evaluation，update=None。用冻结的原 CPU 工厂生成一次投影，保存新 bank 与 fingerprint；先对 A/final 六组及 REAL–REAL 原点重算。kind/seed/update 必须相同，wave/QPSD 与原 native 点的最大绝对误差不得超过 1e-7。fingerprint 是否逐字节相同必须报告；数值通过且 fingerprint 不同只表示数值等价，不声称逐字节恢复。未通过原点核验即失败，不开始 bootstrap。

双域 CDF W1 可复用固定的排序和 gap，以 multiplicity 重加权；它必须与显式复制行、包含不同 N、ties、零 multiplicity 的参考实现一致。条件箱边界和每个已保存行的输出 Q membership 固定，A/final 各有自己的 membership；支持计数是复制后的 occurrence count，阈值仍为每域 >=16。每个箱内各域重新归一化质量。原缺箱 fallback 保持原定义，并保存全部缺箱、guard 和支持计数。

任何 bootstrap 重复均不得被丢弃、补抽或替换。缺箱重复仍保存准确的完整损失和 fallback；若任一主分析重复条件支持不完整，最终解释标注 coverage artifact，不能直接确认稳定收益。非有限数或实现/资源失败保留现场和已经完成的重复，不能将失败标作 500 次成功。

## 输出和解释规则

保存每次 A/final 的三个假设的 wave、QPSD 及 wave 分解、S、Δ、相对变化 100×(S_final/S_A−1)、每域实际行数和 Q 箱计数。主结果为 Δ 的 95% percentile 重采样区间，quantile method=linear；同时报告原点及相对百分比的区间。这里的 95% 是重采样分位覆盖，不宣称已证明频率学置信覆盖率或正式显著性检验。各假设、wave/QPSD 的区间由同一 500 次抽样得到，只作边际机制诊断，不据此选择最好假设。

若主区间上端 <0 且全部重复的条件支持完整，记录为 saved-TRAIN-bank gain stable under paired parent/group reweighting；若区间跨 0，记录为 stability unresolved；若下端 >0，记录为 endpoint worsening under this reweighting。若条件支持不完整，优先记录 coverage-limited。上述分类用于安排下一次有界诊断，不是响应匹配的科学验收门。

该区间条件于已经训练并选择的模型、这份 TRAIN 库、已保存的每父 4 次 draw 和固定投影；不包括训练随机性、模型选择偏差、新 latent 或新采集的不确定性。原点 REAL–REAL 是经验参考，原人工分组不是独立采集证明。科学状态仍为 response_match_not_established。

预算结束后独立读回 multiplicity、完整指标和 SHA256 链，交付中文结论。下一阶段只优先登记新头 wave/QPSD/regular 梯度及同 carrier 裁窗诊断，具体模型前向/梯度预算另行冻结；本轮不引入 D 或扩大训练。

## 启动、资源与停止

先冻结 policy、新增分析/监督/部署源和所有只读输入的 SHA256，再启动一次。启动前核对本项目没有其它科学进程和实际可用资源。CPU<=12，GPU计算为0，进程树 RSS<=8 GiB，墙钟<=7200秒，新增输出<=256 MiB。服务器需至少 10 GiB 可用 RAM 和 4 GiB 磁盘。只监督本作业进程树；不自动重启、不改现有环境。

第一个真实重复完成、原点核验通过、保护输入 hash 未变后即确认正确启动。若计算需持续运行，更新同一个 n01 为预计完成后的单次苏醒，然后休息，不持续监控。苏醒时取一次原生状态并安全回收，或按实际剩余工作再次设置单次苏醒；完成回收分析后暂停同一个任务。
