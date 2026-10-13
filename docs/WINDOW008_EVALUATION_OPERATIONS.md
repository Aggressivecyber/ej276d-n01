# Window008 固定终点评价与精确重放操作

训练冻结清单 `WINDOW008_SOURCE_MANIFEST.json` 的 SHA256 是
`97b020dc68e41fa1e893e66fdf1091b1bf9500b98281a5dd7e2545ee9ce00929`。
评价只新增另一份清单；不得修改原清单的 68 个文件或重新派发训练。
评价核心和数学口径见 `WINDOW008_EVALUATION.md`。

## 1. 人工小 fixture 自检与评价源码冻结

在 E 盘 N01 根目录，以项目 Python 执行：

```powershell
python -B scripts/check_window008_evaluation_helpers.py --output receipts/window008_20261013/evaluation_helpers_01
python -B scripts/freeze_window008_evaluation.py
python -B scripts/stage_window008_evaluation.py --package receipts/window008_20261013/evaluation_bundle_01
```

第一条只构造很小的人工数组、人工回执、隔离目录，检查反例，不调用科学模型，
不解码真实 SCI/DEV，也不使用 GPU。源码冻结和打包由 root 在审核通过后执行。
冻结清单包含全部评价辅助脚本、文档，以及 `src/n01` 中未被训练冻结的所有模块；
运行时模块清单必须完全相同。冻结后不得再向 `src/n01` 新增模块。

## 2. 传输和保留式安装

将 `window008_evaluation.tar.gz` 和 `BUNDLE.json` 传到注册远程目录的一个新 staging
目录，核对 archive SHA256。安装脚本和 `window008_evaluation_common.py` 可放在
独立临时工具目录作为安装入口；指定 `--root` 指向注册远程根目录：

```bash
python -B /path/to/staging-tools/stage_window008_evaluation.py \
  --install /path/to/new-stage/window008_evaluation.tar.gz \
  --root /home/rylan/projects/ej276d-n01/window008_20261013
```

安装只写评价清单内的新增路径。若未冻结的旧稿占用同名路径，先在新的
`receipts/window008_20261013/evaluation_stage_*/before/` 留存原字节。
训练冻结路径、符号链接归档、路径越界、清单以外文件均拒绝。
安装不会启动评价；已有评价目录时拒绝覆盖安装。

## 3. 训练原生退出后派发 primary

全部三组必须各完成 19,000 G / 每 hypothesis 16,000 D，合计 57,000 G / 144,000 D，
且存在匹配 PID/命令的训练 `runs/supervisor/exit_receipt.json`，`exit_code=0`、`reason=null`。
门禁还检查 completion、binding、全部 final checkpoint 和训练冻结文件的哈希。
失败不会启动评价。预检要求无其他 GPU/项目计算、GPU0 空闲至少 10 GiB、
RAM 可用至少 10 GiB、磁盘空闲至少 8 GiB。

在注册远程根目录，以原训练 Python 环境执行：

```bash
python -B scripts/start_window008_evaluation_remote.py --kind primary
```

正式产物为 `runs/evaluation_jobs/primary/artifacts/`。每个作业只允许一个永久
`START_ATTEMPT.json`；任何错误都保留目录，不自动重试、覆盖或恢复。
监管上限为 GPU 10 GiB、子进程树 RSS 8 GiB、28,800 秒、输出 3,072 MiB；
1 秒轮询，超限只向自身进程组发送 TERM，20 秒后仍未退出再 KILL。
GPU 分配器限额和实际进程显存检查同时生效。检查失败也会停止自身作业。

有效完成同时要求 `JOB_COMPLETE.json`、`EVALUATOR_NATIVE.json` 原生退出 0，及
本作业 `supervisor/exit_receipt.json` 退出 0 / 无停止原因；只看到 SUMMARY 不算完成。

## 4. 同流 replay01

primary 原生完成后执行：

```bash
python -B scripts/start_window008_evaluation_remote.py --kind replay01
```

重放产物保存在 `runs/evaluation_jobs/replay01/artifacts/`。三个固定模型、噪声005、
配置、seed、每 parent 4 draws、128 行 chunk、后端与 CPU 线程数均保持相同。
重放复现已经登记的随机样本，不加入新的独立样本，不合并扩大统计量。
评价及重放的 optimizer/noise/population/C2ST updates 均为 0。

重放包装器自动生成 `REPLAY_VERIFICATION.json`，检查全部 13 个 NPZ 的所有成员：
dtype、shape、逐值、包含 dtype/shape/连续字节的 SHA256 都必须一致。
NaN 的对应位置可作逐值相等，但仍要求原始字节哈希相同。也检查全部生成 chunk
记录、parent/draw 顺序、投影、初末随机流、匹配选择、距离、bootstrap 数组与 JSON。
只忽略明确列出的顶层 UTC 和 SUMMARY 内的容器文件哈希；每边文件哈希仍分别验真。
容器 ZIP 时间戳不作为跨运行比较对象，不使用误差容差。

若需要独立复核，两个作业原生退出完成后可写另一个新报告：

```bash
python -B scripts/verify_window008_evaluation.py --output /path/to/new/REPLAY_RECHECK.json
```

通过状态必须是 `exact_replay_verified` 且全部检查精确一致。输出完整、数值重放通过
和科学有效性分别记录；目前科学状态仍是 `response_match_not_established`。
单次 acquisition、此前暴露的 DEV 和无逐事件响应真值的限制不会因重放通过而解除。
