# Window008 恢复执行与评价绑定设计 01

状态：只读设计，等待 root 确认模型类与配置契约；本文不是执行授权、源码冻结或训练完成回执。

## 1. 推荐路径与前提

保留原 deterministic 19,000 G 终点；full 仅从原 `stage_03000.pt` 恢复，重新执行 12,000 joint + 4,000 low G；simple 从原共同初始化新训 19,000 G。新作业独立 root / run_id，拟名 `window008_recovery01_20261013`，名称由 root 最终登记。

此路径成立的必要前提：修复只影响 full 的 joint/low 随机形状基；mean_response、deterministic、simple 的前向、损失、参数顺序、梯度和优化器语义均保持不变。最小 canonical 零积分 basis 修复若满足这些条件，不要求重跑已完成 det。修复不能悄悄改变平均形状、源电荷定义、观测窗、随机维数、paired draw 顺序、数据/统计或噪声005。

根因与数学修复由独立审核给出。本文不声称基投影已经解决科学响应匹配，也不放松非正/非有限停止条件。恢复仍为 TRAIN 固定预算；DEV 不选择修复、设置或终点。

## 2. 当前可核对的历史证据

- 原冻结清单：`WINDOW008_SOURCE_MANIFEST.json`，SHA256 `97b020dc68e41fa1e893e66fdf1091b1bf9500b98281a5dd7e2545ee9ce00929`，68 项。
- 原 config SHA256：`f5500c65f2d17a2fe6964bfc5908dedb68c27b9391cb30bf30c0ba161509d145`。
- 原 evaluation core：`src/n01/evaluate_window008.py`，SHA256 `75f8988415a181a31a7c2b5890418b1f7685e0d8fb141cdea84f4174af783cc0`。保留此文件字节，新增适配器。
- 收集证据：`receipts/window008_20261013/FAILURE_COLLECTION.json`；其已记录本地26个成员与远程SHA一致。原始 science/supervisor 文件在 `runs/window008_20261013/runs/`。
- 原退出链：supervisor 194237 / child 194239，2026-10-13T06:23:02.680315+00:00 原生 exit 1，reason=null；其失败回执保留，不改成 exit 0。
- 失败时全作业实际已完成 22,269 G / 48,807 D；det 19,000 G / 每 hypothesis 16,000 D；full 3,269 G / 每 hypothesis 269 D。
- det final SHA256：`2096523d53b62dbbc6b41d2788a8fba203af89c3cf753133bbbe81ab5c036849`。
- full stage_03000 SHA256：`48a20d256c3d87f2f087aa6a3cc3a2dd9279715c137ad1d5c073dcd22bfaab9c`。
- 原 common_initial_state.pt 文件SHA256：`ec5430fa4f5a0c0f2b330a7d38d8e210c5dfa93e12ffe911695de16e2a5d2f1a`；其记录的 model state digest 为 `b5e598c9251bf3fbf1408d9fc4382474a798817882f1d31276cecfc50a8b5f06`。文件SHA和state digest是不同概念。
- helpers 的实际人工工程回执名为 `evaluation_helpers_03/REVIEW.json` 与 `NATIVE_EXIT.json`，不是 ENGINEERING_RECEIPT.json；52/52 通过，未执行真实评价，尚待 root freeze。不可把其 artificial_* 下伪造的解析fixture当作科学原生回执。

## 3. 预算与计数必须分开

| 类别 | G | D（3 hypotheses合计） |
|---|---:|---:|
| 继承的有效det终点 | 19,000 | 48,000 |
| 继承的有效full平均阶段 | 3,000 | 0 |
| 新full joint + low | 16,000 | 48,000 |
| 新simple全部阶段 | 19,000 | 48,000 |
| 有效三臂固定终点 | 57,000 | 144,000 |
| 本恢复作业实际新增预算 | 35,000 | 96,000 |
| 原失败full废弃随机分支 | 269 | 807 |
| 原失败作业+新作业历史实际合计 | 57,269 | 144,807 |

失败尝试3270发生在生成前向，未完成额外G/D更新；其已消耗的sampler/RNG位置和失败上下文原样保留，不用于恢复。禁止从 FAILURE_CHECKPOINT、latest、失败时的 sampler/streams 或D269状态恢复此修复。

`G_updates` 在checkpoint内继续表示该臂累计有效更新，full从3000递增到19000；运行监控另记 `new_G_updates`、`new_D_updates`、`effective_endpoint_G_updates`、`effective_endpoint_D_updates` 与 `historical_attempted_completed_updates`。新输出不复制旧training.jsonl作为新增训练记录；继承前缀以路径+hash引用。

## 4. 新root、配置和物料

新增文件命名建议：

- `window008_recovery01_config.json`：独立run_id、原run/config/manifest的hash、明确修复版本、唯一获准的checkpoint引用、分臂执行计划、新预算与不变的有效终点预算。
- `docs/WINDOW008_RECOVERY01_PROTOCOL.md`：修复适用范围、继承条件、失败消耗口径、固定停止规则与评价绑定方式。
- `WINDOW008_RECOVERY01_SOURCE_MANIFEST.json`：新代码/协议/恢复config/工程回执的hash，原68文件的不可变引用，以及获准历史输入的hash；`before_recovery_scientific_forward=true`。
- `WINDOW008_RECOVERY01_INPUTS.json`：原native失败链、FAILURE.json、det完成回执/final、full stage3000、common initializer、数据/统计/noise005的hash、原路径及新root只读引用位置。
- 新模型、engine、driver文件：`model008_recovery01.py`、`response008_recovery01.py`、`train_window008_recovery01.py`；精确类名由root确认后施工。
- 新start/launch物料、严格新输出目录、独立supervisor回执。不得覆盖原run、旧失败文件或旧driver。

原 `window008_config.json`、原manifest和68项的字节不改。恢复config是执行计划；checkpoint中的 `config` 保持原科学recipe字典，另加 `recovery_run_id`、`recovery_config_sha256`、`recovery_source_manifest_sha256`、`model_implementation` 与 `lineage`。不能向原config偷偷加新字段再放宽旧restore的比较。

建议新root携带原冻结运行依赖的精确副本及 `provenance/window008_original/` 下的只读历史证据副本；所有副本验证来源hash。禁止软链接覆盖旧目录、重新序列化旧checkpoint、复制旧exit1并改0。若采用原root直接只读引用，必须把完整绝对允许路径及hash写入INPUTS，并在恢复和评价前后复验；不可接受调用者任意传入checkpoint路径。

训练与评价各自单独freeze。评价冻结放在恢复模块全部完成之后，避免原helpers的module inventory规则因后来新增模块而失败。原evaluation helpers保留；新恢复helpers绑定新root与恢复清单。

## 5. full stage3000严格导入契约

在任何科学前向前依序验证：

1. 核对原68文件、config、data/statistics/identities/noise005、历史native退出链、三个唯一获准的历史输入文件hash；校验失败立即退出，不自动选latest/failure/另一个stage。
2. 用CPU加载已验hash的full stage3000；要求schema=response008/v1、arm=full、config与原config精确相等、statistics内容digest和文件来源一致、G=3000、D={fast:0,slow:0,common_level:0}、stage=mean_response、noise_updates=0、population_learning=false。
3. common initializer从原文件读取并验state digest；不能只相信重设相同seed。核对原full/det初始checkpoint的共享参数与common逐tensor一致；simple按原make_signal的共有参数与physical_head前缀规则精确初始化。
4. 独立确认原full/det stage3000的共有mean参数、共有G optimizer slot、sampler、paired streams精确一致。比较优化器slot必须按参数名字映射，不能拿不同variant的裸整数parameter ID当成相同参数。
5. 新模型与原full的state_dict keys、每项shape/dtype、named_parameters顺序完全一致；strict load。最小投影修复不应新增trainable参数或改变已有参数顺序。默认禁止optimizer remap/缺项容忍。
6. G AdamW完整恢复state与param_groups，核对lr/weight_decay/betas/eps/amsgrad及各参数对应slot；mean训练过的参数保留step与moment，未训练随机分支应仍无optimizer slot。绝不能为了方便统一把moment清零。
7. 三个D恢复stage3000的完整state（含spectral_norm buffers）和optimizer；要求每个D的state digest等于其已登记initial_D_sha256，D optimizer state为空且param_groups一致。所谓“D归零”是恢复其真实零更新stage，绝非把269计数字段归零但留下权重/moment。
8. SOURCE sampler n=6593、REAL n=21047；order必须是完整不重复置换，position/epochs/visits为有效非负整数且匹配draw总量，numpy bit-generator类型与状态一致。3000步时SOURCE draw总量1,536,000=232*6593+6424，REAL 3,072,000=145*21047+20185；visits总和分别等于draw总量。恢复后state逐字段与保存值精确相等。
9. 三个显式CPU paired streams的键必须恰为response/noise005/extension，state tensor dtype/shape/值完整恢复；不要按seed重启流。保持phys10、local_white400、noise005各原形状及extension(1,284)的相同draw次序与microbatch分割。人工check可克隆流比较下一批，然后丢弃克隆，不能污染实际训练流。
10. projection context是seed与臂update的纯函数，无隐藏待恢复generator；第一恢复步必须使用update=3001，不能使用new_G_updates=1或全作业累计计数。
11. checkpoint保存的stage=mean_response是正确的：旧driver在完成第3000步后保存，下一步才切joint。恢复后由response_stage(3000)切joint，强制gradient audit与同参数/同latent replay；进入low时同样强制。不要把原stage3000文件重写成joint。
12. 先完成不依赖TRAIN数值的CPU预加载校验，再读取TRAIN数值、实例化engine并复验恢复后的state。在第一次科学前向前生成独占 `RESUME_VALIDATION.json`，绑定所有检查结果、输入hash、恢复前/后的state digest和预期计数。失败保留回执，无自动回退。

det保留应验证：原TRAINING_COMPLETE.json与FAILURE.completed_arms中的det记录一致，final文件hash正确；checkpoint为det/19000/三D16000/low，coverage6593/21047，common/statistics/noise引用正确。其来源是“原失败作业内已完成的臂”，不是“原作业原生成功”。

simple从common初始化开始，sampler/paired streams从原seed开始，D从同一原seed初始化；full从stage3000接续。不能让simple继承full已推进的流或优化器对象。

## 6. 训练complete与评价输入schema

建议新 `runs/science/TRAINING_COMPLETE.json` 使用 `schema=n01/window008-recovery-training-complete/v1` 与明确恢复status，不借用原单次57k作业成功status。以下结构为字段契约，hash由实际文件填写：

```json
{
  "schema": "n01/window008-recovery-training-complete/v1",
  "status": "fixed_recovered_endpoints_complete_pending_native_exit_and_evaluation",
  "run_id": "window008_recovery01_20261013",
  "base_run_id": "window008_20261013",
  "base_config_sha256": "BOUND_BASE_CONFIG_HASH",
  "base_source_manifest_sha256": "BOUND_BASE_MANIFEST_HASH",
  "recovery_config_sha256": "BOUND_RECOVERY_CONFIG_HASH",
  "recovery_source_manifest_sha256": "BOUND_RECOVERY_MANIFEST_HASH",
  "resume_validation_sha256": "BOUND_VALIDATION_HASH",
  "counts": {
    "inherited_effective": {"G": 22000, "D": 48000},
    "new_completed": {"G": 35000, "D": 96000},
    "effective_endpoints": {"G": 57000, "D": 144000},
    "discarded_old_full_branch": {"G": 269, "D": 807},
    "historical_completed_all_attempts": {"G": 57269, "D": 144807}
  },
  "arms": [
    {"arm": "deterministic", "origin": "retained_original_endpoint", "inherited_G": 19000, "new_G": 0, "G_updates": 19000, "D_updates": {"fast": 16000, "slow": 16000, "common_level": 16000}, "model_implementation": "n01.model008:N01Response008", "endpoint_path": "BOUND_DET_RELATIVE_PATH", "endpoint_sha256": "BOUND_DET_HASH", "legacy_arm_completion_sha256": "BOUND_OLD_ARM_RECEIPT"},
    {"arm": "full", "origin": "stage3000_recovery", "inherited_G": 3000, "new_G": 16000, "G_updates": 19000, "D_updates": {"fast": 16000, "slow": 16000, "common_level": 16000}, "model_implementation": "ROOT_CONFIRMED_RECOVERY_MODEL_CLASS", "endpoint_path": "runs/science/train/full/final.pt", "endpoint_sha256": "BOUND_FULL_HASH", "resume_source_sha256": "BOUND_STAGE3000_HASH"},
    {"arm": "simple", "origin": "fresh_common_initialization", "inherited_G": 0, "new_G": 19000, "G_updates": 19000, "D_updates": {"fast": 16000, "slow": 16000, "common_level": 16000}, "model_implementation": "n01.model008:N01Response008", "endpoint_path": "runs/science/train/simple/final.pt", "endpoint_sha256": "BOUND_SIMPLE_HASH"}
  ],
  "old_attempt": {"exit_code": 1, "failure_sha256": "BOUND_OLD_FAILURE_HASH", "launch_sha256": "BOUND_OLD_LAUNCH_HASH", "native_exit_sha256": "BOUND_OLD_EXIT1_HASH"},
  "awaiting_own_supervisor_native_exit": true,
  "DEV_waveforms_read": false,
  "C2ST_updates": 0,
  "noise_updates": 0,
  "population_updates": 0,
  "evaluation_complete": false,
  "scientific_status": "response_match_not_established"
}
```

实际各arm还必须包含coverage、stage=low_learning_rate、common/state/statistics/noise及有效代码hash，完整files列表绑定checkpoint及回执。`model_implementation`必须来自冻结的精确允许映射，不能以任意字符串动态import。

driver写complete时尚未原生退出，不能在该文件提前声称native0。评价门禁独立读取本恢复supervisor的launch与exit，检查实际新模块、命令、root、out、PID、时间链，要求本恢复job exit0/reason=null。同时按原hash读取旧exit1和旧FAILURE。两条链同时成立，绝不要求或制造旧exit0。

## 7. 评价适配策略

旧core的 `training_completion_gate`、`_freeze` 与主入口写死单run原生exit0、旧checkpoint schema、旧config路径、固定model008类；现有helpers也写死旧run/module/root。不能只放一份“看似57k”的汇总就直接调用旧入口。

新增 `evaluate_window008_recovery01.py` 及对应versioned helpers：

1. 新gate验证两代source/input manifest、旧失败链、新成功链、严格三臂endpoint map和三个预算口径；新训练未原生完成时在读取DEV/模型前即拒绝。
2. 新loader对det使用原model008，对full使用root批准修复类，对simple使用原simple类（或经bitwise证明的等价别名）；各自核对checkpoint schema及原config，新full/simple另核对recovery元数据。
3. 复用hash为75f898...的旧core数值工具函数，或者仅复制有明确差异审计的编排层；保留原损失/统计/生成/bootstrapping/观测函数，禁止monkey-patch旧gate返回成功。继承paired streams API、4 draws、chunk128及全部评价seed。
4. 新FROZEN_INPUTS绑定endpoint map及model implementation代码hash、两代manifests和native receipts，在DEV读取前冻结，结束复核。SUMMARY保留有效终点57k/144k，同时明确new_training35k/96k、discarded269/807及historical57269/144807。
5. 原生wrapper/module命令改为新adapter，primary/replay分别新目录；两次比较包括新lineage字段及其hash。现13 NPZ的全部dtype/shape/值/member哈希、chunk/流/投影/距离/bootstrap精确重放条件继续保留。
6. 旧9/9核心人工审查与helpers52/52仅是既有基线证据；新增adapter必须有独立人工fixture反例测试，不能把旧审查直接宣称为新接口通过。

## 8. 一次性实施与人工检查清单

- root确认修复类/函数语义和无新参数；独立工程证明mean/det/simple精确等价及mean共有state可继承。
- 实现versioned engine/driver和上面的严格一次性stage导入，先跑人工fixture（不读SCI/DEV、不用GPU）；至少拒绝错arm、3001/3269/failure、D非0、sampler/RNG错位、optimizer参数次序改变、config/statistics/common/noise/hash错误、缺失D谱归一化buffer、重复启动及path escape。
- 启动前完成恢复source/input freeze及完整物料hash验证；原68文件复验未变。只在新root运行，一个科学作业，保留原资源边界。
- 确保真正启动且第一恢复步的臂计数、projection3001、new计数和持久化回执正确后，按用户要求设置定时苏醒，不持续监控。
- 原生训练完成后验证三有效终点+双代退出链；再freeze/运行新adapter primary与同流replay。训练终点完成、原生成功、精确重放、科学有效性分开报告。

## 9. 什么时候才需重训

若修改影响共有mean/det计算、训练损失/观测算子/数据统计、全局初始化、共享参数次序或任何共有优化轨迹，不能继续把旧det当作同协议对照，需要登记新的完整三臂比较并说明不可避免原因。

如果仅full的mean路径改变，而det/simple仍精确不变，最小必要范围是full fresh19k+simple19k，仍保留det；不是自动三臂全重跑。若只有从未训练的随机参数结构变化，应先评估明确、可审查的name-based初始化/optimizer迁移能否保留full mean，而非默认丢弃。stage3000损坏或继承不满足时，必须报告具体失败证据后再选择最小重训范围，不把恢复困难当作浪费det19k的理由。

本设计阶段新增训练/科学前向/DEV数值读取/GPU使用均为0；未编写或启动恢复driver。

## 10. root确认后的实施交付

root随后明确选定 `window008_fix01_20261013` 与 `n01.model008_fix01.N01Response008Fix01`，并授权新增engine/driver。实施文件为 `src/n01/response008_fix01.py`、`src/n01/train_window008_fix01.py`；采用已登记 `window008_fix01_config.json` 的13项只读来源，不修改原recipe。

科学driver CLI为原参数加 `--recovery-config`：`--config`仍指原window008_config.json，`--source-manifest`指WINDOW008_FIX01_SOURCE_MANIFEST.json。engine科学模式只允许full/simple，full必须显式导入唯一stage3000；fix01 checkpoint schema为response008_fix01/v1，并附recovery身份、参数顺序、继承/新增计数与禁止自动恢复声明。det通过endpoint_map引用原文件，不重新序列化。

CPU人工QA脚本为 `scripts/check_window008_fix01_recovery.py`，最终回执 `receipts/window008_fix01_20261013/recovery_cpu_03/ENGINEERING_RECEIPT.json` 与 `NATIVE_EXIT.json`。63/63检查通过、原生exit0；仅人工数据7 G / 9 D工程更新，不读取实际SCI/DEV数组或科学signal checkpoint，不使用GPU。人工stage3000由一次人工mean更新后构造计数/AdamW step/sampler/RNG状态，仅用于解析及恢复反例，不冒充3000次实际优化。

额外覆盖：导入后验失败仍新增0计数、全部mean参数AdamW slot齐全、失败checkpoint/context与部分D更新不丢失、首次失败目录不可重启覆盖。Windows人工sampler置换dtype为int32而Linux通常int64；校验接受这两种合法整数索引但逐字段保持原dtype/shape/值，不转换保存状态。

旧原生链明确区分timeout wrapper child PID194239与trainer PID194240。旧回执无trainer parent PID，不能编造其相等或父子关系；新校验绑定原hash、native与launch的wrapper PID、完整注册命令参数及时间链，并在回执记录这个证据边界。

本轮没有创建新评价适配器、执行科学训练或冻结最终源码；由root完成独立审核、原生工程与物料冻结/启动。评价后续仍须实现第7节的新adapter，不可直接让旧gate接受拼合汇总。
