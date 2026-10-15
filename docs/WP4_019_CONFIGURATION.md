# WP4 019：T1 unpaired 两种子有界识别

`wp4_019_20261016`只推进已采纳路线的下一项：T1_unpaired_seed0/1各2000G。这消耗原八槽计划剩余六槽中的两个，不复跑paired，不运行T2，不自动打开旧017 stage2。017旧pending与全部017/018源码、配置和结果保持原样。

019必须在018最终独立复核通过后另行登记、冻结和启动。入口绑定root的`receipts/wp4_018_20261016/FINAL_RESULT_RECEIPT.json`真实路径/字节/SHA，检查018 native0、qualified=true、独审通过、2544格、旧017 pending保留；只有producer pass不足以放行。草案不能加载科学数组或005 CP。root负责登记、冻结、native ops与一次未来苏醒；本driver不含SSH、部署、重试或新监督框架。

## 固定科学问题和复用

比较相同已知A/B/005条件下，原无配对方法能否恢复T1的条件电荷核。学生仍只有两个全局theta：`a=qref*softplus(theta_a)`、`b=softplus(theta_b)`，qref=3.7103998046755806；初值a=.001qref、b=.00025；T1固定a=.01qref、b=.0025。原A、005、M0形状B与均值均不训练。教师值只作本次人工识别任务的真值，不写入REAL候选或部署。

6593个合法原SOURCE父的A/B/Q缓存按原字节和manifest只读mmap，5274拟合父、1319留出父完全保留。原物理权重、源ID、原采样角色及已保存源诊断坐标保留；ID/行号/文件名不作模型输入。不重新打开原SOURCE/REAL/DEV波形，不载入A/M0检查点、不构造A/M0网络、不做A/M0前向。005仍需一次原冻结CP数值加载，随后构造原三假设扩展；这1次明确计入CP预算。005有9,035,904行实际噪声生成，不能笼统称NN=0。

`WP4_019_SAVED_INPUTS.json`精确绑定37个输入，共1,141,808,275字节，包括：原017的A/B/Q、cache manifest、PARENT_SPLIT、原物理权重、原冻结状态证明、T1 hard/report及PIT uniforms；原005 CP及固定statistics JSON；018的T1 O19、teacher-null/fullwave-reference、initial energy、parent axes、配置、gate和COMPLETE；root最终018独审放行回执。配置inputs与该清单相等，运行前及终态核SHA。实际完整边界以37条绝对路径/字节/SHA为准，不用通配符扩大读取。原A/M0 CP既不在数值输入白名单，也不反序列化。根代理另绑定旧实现及ops的保护源字节。

原teacher与共享initial评价已由018全覆盖重放。019按SHA复用其保存波形和评分；不新增teacher或initial前向、硬观察、RNG或PIT抽样。新评价只生成两个final kernel，每个6593父×2bank×32draw×3hypothesis，保持原硬256、384载体、prefix100、ADC、firstargmax/crop、tail46及1软件ns。

## 训练和随机流原样

`Engine017.step`、`Runtime017`的信号/005/硬观察、`Streams017`及`ParentSampler017`直接复用，不另写训练算法。每G每侧两组独立512父，拼成1024行；四个parent角色为student0/student1/teacher0/teacher1，在各自流内按原IID有放回采样，不将两侧变成伪配对。

训练Gaussian namespace仍为`kind=training, replicate=seed, role=student/teacher`，不新增mode或019版本前缀；parent namespace仍为`kind=parent, mode=unpaired, replicate=seed, role=四角色`，哈希种子函数仍为原`N01-WP4-017-v1|...`。全部精确初始种子前瞻保存在config.stream_addresses，实际全generator初/末态与当时CP状态按旧成熟实现保存。教师与学生的所有shape/charge/005/extension流独立；教师latent不成为部署输入。

训练目标仍是完整256波形与固定积分坐标构成的原257维proper energy U-score，self去对角分母m(m−1)，保留有符号有限负分数；三hypothesis平均一次backward。unpaired保持原64行cdist块和完整U统计量，不改成近似距离、子采样对或新损失。AdamW lr=.01、wd0、clip5，两个theta真实梯度/实际更新的原G1门保留。

final评价仍使用原`kind=evaluation, role=student, bank=0/1`的相同前缀，故与共享initial/两个final形成原设计的CRN；teacher为另一个独立role，沿用原T1保存bank。seed的训练流彼此区分，MC draw仍不称独立物理父。

每seed在G1000和G2000保存CP，共4份正常CP；不额外保存共享initial CP、不从旧学生CP恢复。全job失败CP最多尝试1份；失败保存错误也显式记录，原异常继续抛出，不自动续训或补采样。正常结束须四个parent流覆盖全部5274拟合父，不补repair draws。

## 新产物身份与原实现

只新增019驱动、配置、输入清单和本文，旧代码文件不修改。新进程将旧模块的元数据writer绑定到限定019输出目录的writer：新JSON具有run_id019，原017格式版本另记implementation_schema_origin；这不改科学参数或随机namespace。CP由最小019保存函数写诚实的schema/run_id；不加载后再改旧CP。已有Store017的journal/status格式版本可复用，run_id始终取019配置。

Runtime017实例的verify只替换为原005/extension状态及只读缓存/绑定输入SHA核对，不要求重新构造A/M0。继承的“frozen A/M0”证明明确限定为原A/B/Q缓存及其来源绑定，不能被解读为又执行了一遍原网络。EVALUATION_AXES里原cache相对索引的source_index_root显式指回原017保存根。

INIT_PROOF和FIRST_UPDATE_PROOF包含run019、005 CP1、A/M0 CP0、A/M0前向0、缓存manifest身份、两theta白名单、005冻结hash。G1另保存两项真实非零梯度和actual delta，供root一次核对正确启动后设置苏醒，不持续人工轮询。

## 完整018评价门不改

019直接使用018的hard重建/O19、cov8、PIT、时间与波形评分内核。每个新final的scalar9与native32波形矩先由其hard256及实际baseline重新计算并逐项核原生REPORT，O19以同一确定性映射生成；完整父/64槽保留。teacher O19和阴性对照已冻且经过018核验，不重新标定、不重新取样。

完整波形ES重新从新final与原T1 hard256按原257维欧氏距离计算；仅initial ES与teacher reference复用018已核逐父向量。teacher64 O19 self项重算一次，两个final各重算cross和student self；保留负U值。原teacher32 null向量复用，MC门不随019分数改变。

O19尺度、δ=.10、MC半宽=.05、teacher32 null筛选、原1%缺失/截尾差门以及24+24/48、95%父/权重、ESS规则均不变。cov8相对Frobenius≤.15、teacher split≤.075；波形/电荷/时间/覆盖/energy门与018完全一致。旧rise数值门/cov9继续作为选择诊断，不再混入完整主门；有条件下界不证明窗外交叉存在或恢复完整rise。

新gate明确核真实T1_unpaired_seed0/1身份，不把标签改成paired调用旧身份检查。每trial1272格、共2544格完整签名：12个解析方差格，36个hypothesis×group×weight单元各35格，含wave2、cov8、fullwave energy、time2、coverage10、原因18、O19。任一fail/pending均不取得无配对资格。原018的2688格017重放、恒等、阴性和四反例证据按SHA引用，不复跑旧全工程套件。

两seed正常2000G且评价完成可native0，无配对gate仍可closed；native0不冒充识别成功。通过仅支持固定A/B/005下这两个电荷参数的指定条件核与可观察窗。未通过只限定当前无配对方法/信息和预算，不推导所有无配对算法无效；任何结果均不自动启动T2，不证明均值/形状学习或REAL/硬件响应验收。

## 精确预算和资源

机器可审预算由纯stdlib的settings019()/budget_caps019()/validate_settings019()生成，与配置逐字相等；原成熟Store记录entered/completed/attempted/inflight，终态全部计数须精确匹配，不用重试抹去失败。

| 预算项 | 019新增 |
|---|---:|
| G / AD / optimizer steps | 各4000 |
| 训练student / teacher行 | 各4,096,000 |
| 新final评价student行 | 843,904 |
| student电荷行 / teacher电荷行 | 4,939,904 / 4,096,000 |
| 005及shape cache组装行 | 各9,035,904 |
| 三假设扩展 / 硬ADC / 寻峰裁窗行 | 各27,107,712 |
| 标量normal请求 | 3,560,146,176 |
| parent randint calls / scalars | 16,000 / 8,192,000 |
| unpaired训练有序距离 | 25,165,824,000 |
| 训练cross / self去对角 / 计算后排除self对角 | 12,582,912,000 / 12,570,624,000 / 12,288,000 |
| 重新计算的257维评价有序距离 | 162,029,568 |
| O19评价有序距离 | 405,073,920 |
| CP加载 / 正常保存 | 1份005 / 4份新2theta CP |
| A/M0 CP数值加载、A/M0前向、SOURCE/REAL/DEV、T2、PIT新抽样 | 全0 |

相对另行生成同样T1/initial两kernel，复用直接避免843,904行信号/005及其3假设硬观察和对应随机请求；相对018再次全重放，不重算已接受的paired评分、teacher32 null/恒等/反例与initial energy。teacher64 self重算在405,073,920内明确计入，绝不把新final计算伪称复用。

资源沿用原每trial≤7200s、GPU≤10GiB、进程树RSS≤8GiB、输出≤1536MiB；两个trial训练与原生评价后共用CPU评分，按每trial实际活跃时间累计，不把另一trial耗时算作本trial计算。全job≤21600s、≤5120MiB，共享评分/日志预算1024MiB，12线程，选收≤128MiB。一次一个科学job，无自动重启。完整原生与O19数组留远端，root按最终完整SHA清单回收必要证据。
