# Window008 resume01：从完整 full6000 周期检查点继续

本次运行身份为 `window008_resume01_20261013`，配置为 `window008_resume01_config.json`，schema 为 `n01/window008-resume01-config/v1`。用户已经直接通知服务器恢复并要求继续；此前暂停限制由该直接指令解除。所有新输出使用独立 Linux root `/home/rylan/projects/ej276d-n01/window008_resume01_20261013`。本地代码、回执、临时目录和缓存只放 E/D 盘。

科学模型仍为 `n01.model008_fix01.N01Response008Fix01`，损失、初始化、随机种子、数据、TRAIN/DEV 角色、噪声端点、观测算子和训练终点均按既有冻结配置执行。此次只实现完整周期检查点的状态接续。

## 唯一恢复来源

`provenance/window008_resume01_reference/fix01_full_periodic_06000.pt` 的 SHA256 必须为：

```
921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58
```

该文件来自用户停机归档中的 `runs/science/train/full/latest.pt`，为 `response008_fix01/v1` 的 full 周期检查点：6000 G、每个假设 3000 D、`joint_stochastic`。原件保留，复制件逐字节绑定到新配置。驱动只调用 `Response008Resume01Engine.import_fix01_full_periodic6000(path, expected_sha256=..., fix01_identity=...)`；通用 restore、原 stage3000 导入及任何 failure checkpoint 恢复均被禁止。

严格导入包含模型、三个判别器及谱归一化 buffers、G/D AdamW 全部已存在 slot 和超参数、SOURCE/REAL sampler 的置换/位置/访问次数/NumPy RNG，以及 response/noise005/extension 三条 CPU 随机流。G AdamW 的 mean 参数累计 6000 步、stochastic 参数累计 3000 步，必须保留混合步数；D 每个假设 3000 步。不得清空或改写 moment、重新播种、跳过父事件，或以 `new_G_updates=1` 代替 arm 的投影 update=6001。

导入完成后，`RESUME_VALIDATION.json` 在第一步科学前向之前落盘。第一步固定为 full6001，必须审计梯度并执行同版本、同 latent 的重放。随后执行冻结 joint 阶段到 full15000；下一步 full15001 进入既有低学习率阶段，再到 full19000。full 15000、19000 和周期检查点保留。simple 从原 common initializer 及原 fresh sampler/streams 开始，执行 3000、15000、19000 固定节点；原 deterministic19000 端点只读保留。

## 三种计数与历史弃用部分

下表 D 是三个假设的总更新数；各完成臂最终 D 均为每个假设 16000。

| 范围 | G | D |
|---|---:|---:|
| 保留原 deterministic 端点 | 19000 | 48000 |
| 继承完整 full6000 周期状态 | 6000 | 9000 |
| 本次新增 full6001–19000 | 13000 | 39000 |
| 本次 fresh simple1–19000 | 19000 | 48000 |
| 本次新增预算 | 32000 | 87000 |
| 有效比较的三个完成端点 | 57000 | 144000 |
| 原数值失败弃用部分 | 269 | 807 |
| 用户停机时 full6000 后弃用部分 | 398 | 1194 |
| 本次之前 Window008 历史实际累计 | 25667 | 59001 |
| 完成后的 Window008 历史实际累计 | 57667 | 146001 |

其他 N01 阶段此前累计 100000 G。因此本次科学训练之前的全局实际 G 为 125667，本次固定预算完成后为 157667。状态、失败回执和完成回执均分别记录 `new_execution_*`、`inherited_effective_*`、`effective_comparison_*` 与 `historical_actual_*`，同时披露两类弃用计数。部分 D 已更新而 G 尚未完成时，使用实际 D counters；不按 G 推断 D。绑定尚未通过的历史和继承计数记录为 null。

## 冻结与停机证据链

新驱动重新校验原 `WINDOW008_SOURCE_MANIFEST.json` 的全部 68 个文件以及 `WINDOW008_FIX01_SOURCE_MANIFEST.json` 的全部 114 个文件，保留两份 manifest 的原始字节。新 `WINDOW008_RESUME01_SOURCE_MANIFEST.json` 同时绑定两代 manifest SHA、原配置、新配置和全部 26 个历史引用。运行目录中所有 `src/n01/*.py` 必须属于三个清单的并集，所有实际模块都要有冻结哈希；缺项和多项均拒绝。

本地工作树还存在未执行草稿 `src/n01/train_response007_continuation01.py`，它不在两代旧冻结清单中。草稿保持原状，科学 stage 只取两代旧冻结模块与本次新增的 `response008_resume01.py`、`train_window008_resume01.py`，形成 36 模块的闭包。源清单与实际运行模块的精确一致检查发生在独立 staged/runtime root；本地工作树不是该科学运行目录。CPU QA 用仅含文件名的隔离人工目录验证 36 模块可接收、额外未冻结模块必须拒绝，不把该草稿加入科学运行。

历史链逐项核对原数值失败的 native1、launch、binding、det 完成凭据，以及 fix01 的 launch、trainer/timeout/supervisor PID、输入绑定、恢复凭据、SIGTERM 请求、停止核验、归档、暂停和 full6000 记录。原数值失败 native1 和用户要求 SIGTERM 的 native1 均保留为实际退出角色。暂停凭据的 `stop_receipt_sha256` 指向本地 JSON 原件（SHA `47aaf4...`）；远程 JSON 原件 SHA 为 `d2a3aa...`。两份 JSON 的内容必须精确一致，各自的原始字节分别冻结，且暂停凭据必须指向本地原件哈希。此次恢复既不加载用户停机时的 `FAILURE_CHECKPOINT.pt`，也不加载旧数值失败 checkpoint。

## 驱动与完成状态

入口为 `python -B -m n01.train_window008_resume01`，参数是 `--container`、`--identities`、`--config`（原 Window008 配置）、`--resume-config`（新 resume01 配置）、`--dataset`、`--statistics`、`--source-manifest`、`--out` 与 `--device cuda:0`。科学入口要求配置指定的新 Linux root，且输出必须为该 root 的 `runs/science`。输出目录必须全新；重复运行不会覆盖第一次尝试。

`BINDING.json` 在 prepared TRAIN 数值解码前记录新运行、两代历史、输入哈希和 PID。checkpoint schema 为 `response008_resume01/v1`。每次失败保留实际 G/D counters、两代旧 native1 链和本次失败状态；现场 checkpoint 标为可能处于更新中，禁止自动恢复。固定墙钟 21600 秒、CPU 12 线程、GPU 上限 10 GiB、作业树 RSS 8 GiB、输出 3072 MiB、每 1000 arm G 写周期检查点，均来自新冻结配置。

只有原 det19k、resume full19k 与 fresh simple19k 三个端点均满足固定步数、三个 D16000、全部父事件覆盖和代际 schema，驱动才写：

```
schema=n01/window008-resume01-training-complete/v1
status=fixed_resumed_endpoints_complete_pending_native_exit_and_evaluation
```

该回执仍为 `awaiting_own_supervisor_native_exit=true`、`native_training_success=false`、`evaluation_complete=false`，科学状态为 `response_match_not_established`。它不提前签署进程退出成功。新 root 的 supervisor 必须随后产生独立 native exit0；版本化 resume01 评价 adapter 仍待实现和冻结，才可独立读取 DEV 做固定评价。旧评价入口不得直接用于新 schema。

只允许一个科学作业。主流程核实一次正确启动和首步证据后，按用户偏好设置估计终点的一次苏醒，之后休息；后台自身资源监督器负责已登记边界。

## CPU 与人工元数据检查回执

`scripts/check_window008_resume01_driver.py` 全程封禁 waveform NPZ 解码、`nn.Module` 前向和 AdamW step。它对两代旧冻结文件和 26 个引用做原始字节校验，仅加载五个旧信号 CPU checkpoint 做已有 metadata 验证；full6000 的严格状态导入由 engine 的 CPU metadata QA 单独检查。

- `receipts/window008_resume01_20261013/driver_cpu_01/NATIVE_EXIT.json` 保留首次检查 native1；失败原因是人工模块清单只列新增两个模块，漏掉已有未列入旧冻结的 response007 continuation 模块。该次无科学前向/优化器更新。
- `receipts/window008_resume01_20261013/driver_cpu_02/ENGINEERING_RECEIPT.json` 和 `NATIVE_EXIT.json`：前版 CPU 检查 110/110，通过 native0；当时的本地模块清单检查包含工作树额外草稿。该版脚本保存在其 `source_snapshot/` 中。
- `receipts/window008_resume01_20261013/driver_cpu_03/ENGINEERING_RECEIPT.json` 和 `NATIVE_EXIT.json`：最终版 111/111，通过 native0。隔离人工 staged 目录验证了科学 36 模块闭包，并额外验证实际多出未冻结模块时拒绝；其余检查包括预算、恢复入口、两代停机/PID/raw hash 链、路径逃逸、重复 JSON keys、部分 D 失败计数、端点 provenance、完成状态与首次失败输出保留。

本轮驱动 QA 的科学模型前向、优化器更新、SCI/DEV 数值读取、failure checkpoint 加载、GPU 使用和实际科学作业启动均为 0。这些是 CPU 软件检查回执；它们不表示科学训练已经开始或端点已经完成。
