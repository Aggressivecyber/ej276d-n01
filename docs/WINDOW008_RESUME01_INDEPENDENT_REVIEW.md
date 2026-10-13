# Window008 resume01 独立审查

状态：实际周期 checkpoint、旧冻结文件、新 engine、driver 与运行助手的独立审查通过。已独立读取目标 Linux 的严格实际导入和 native QA 回执，全部 native state 摘要、参数名槽摘要和 PID/时间/命令/退出链闭合。科学训练尚未由本审查启动。

本审查仅写本文件及 `receipts/window008_resume01_20261013/independent_01/`。未构造科学模型，未执行模型前向、优化器更新、GPU 调用、DEV 数值解码或科学作业。

## 可恢复的完整周期点

- 路径：`runs/window008_fix01_20261013/runs/science/train/full/latest.pt`。
- SHA256：`921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58`；21404363 字节，读前后哈希一致。
- schema 为 `response008_fix01/v1`，arm 为 `full`，purpose 为 `periodic`，`failure_may_be_mid_step=false`，`engineering_fixture=false`。
- 科学配置逐项等于冻结的 `window008_config.json`；统计哈希一致，noise005 与 common initializer 的标识保留。
- `G_updates=6000`，三个假设的 `D_updates` 各为 3000，stage 为 `joint_stochastic`；last_metrics 的计数和阶段一致。
- 内部固定投影上下文为 `kind=training`、seed `2026101387`、update `6000`。下一次完整 G 更新与投影编号均须为 `6001`。

## 优化器和采样状态

G 的 180 个参数均有完整有限 AdamW 槽。167 个均值参数的 step 为 6000；13 个 `physical_head`、`basis_head`、`envelope_head`、`local_filter` 随机分支参数的 step 为 3000。随机分支在前 3000 步均值阶段未训练，恢复器不能把全部参数统一成 step 6000。

三个判别器各有 10 个完整有限 AdamW 槽，step 均为 3000。`spectral_norm` 的可训练权重是 `weight_orig`；`weight_u` 和 `weight_v` 属于 buffer，恢复须连同 buffer 严格保存。G/D 学习率、权重衰减与 AdamW 选项符合联合阶段的冻结配置。

| 状态 | 人口 | draw 总数 | 完整 epoch | 当前位置 | 覆盖数 |
|---|---:|---:|---:|---:|---:|
| SOURCE parent sampler | 6593 | 3072000 | 465 | 6255 | 6593 |
| REAL sampler | 21047 | 6144000 | 291 | 19323 | 21047 |

两个 sampler 的 order 均为完整排列，visits 逐元素等于完整 epoch 加当前排列前缀的访问次数，PCG64 状态可无损安装。三个 paired stream（`response`、`noise005`、`extension`）均为 CPU `uint8[5056]`，CPU generator 的 state 安装与读回逐字节一致。本轮没有重新生成科学波形或重放 6000 步随机流；恢复启动时仍须验证首步同版本重放。

## 用户停止点为何必须保留为历史

用户停止回执指向 full6398、native exit 1，错误为用户明确请求关闭前的 SIGTERM 15。`FAILURE_CHECKPOINT.pt` SHA256 为 `f4e6d6a30586159a7fcecc8ee34d3fa0a439f218151cb443d3007538c9beb6cd`。

该 checkpoint 的 G 计数为 6398、D 各为 3398，但 sampler 已消费尝试第 6399 步的输入：SOURCE draw 比 6398 个完整更新多 512，REAL draw 多 1024。其 purpose 为 `failure`，`failure_may_be_mid_step=true`。故其 optimizer、sampler 与 paired stream 不能当成完整周期起点；完整周期恢复必须从已绑定的 full6000 继续。停止后 398 个完整 G 更新和 1194 个 D 更新只计入历史实际计算，保留记录。

旧 Window008 native exit 1 是 full3269 后的形状规范电荷错误；deterministic19000 已有独立 arm 完成回执及 final checkpoint，旧整个作业的 exit 1 不取消该已完成 arm。旧失败、fix01 用户停止、新 resume01 的 native 退出链须分别保存。新比较完成仍要求新 native exit 0、两新执行 arm 的完整终点和最终训练完成回执。

## 预算与计数

| 项目 | G | D（全部三假设合计） |
|---|---:|---:|
| retained deterministic | 19000 | 48000 |
| imported full6000 | 6000 | 9000 |
| resume01 remaining full | 13000 | 39000 |
| resume01 fresh simple | 19000 | 48000 |
| resume01 新执行总额 | 32000 | 87000 |
| 有效三臂比较终点 | 57000 | 144000 |
| 旧 Window008 实际计算 | 22269 | 48807 |
| fix01 用户停止前实际新增 | 3398 | 10194 |
| 三轮历史实际终点 | 57667 | 146001 |

旧失效联合段的 269 G/807 D 和本次停止后不继承的 398 G/1194 D 均继续计入历史实际计算。导入的 6000 G/9000 D 不重复计入 resume01 新执行量；simple 仍从 common initializer 开始；deterministic 不再训练。

## 不可变文件与证据

`WINDOW008_SOURCE_MANIFEST.json` 的 68 个文件和 `WINDOW008_FIX01_SOURCE_MANIFEST.json` 的 114 个文件逐项哈希复验均无变化。当前恢复配置的 base、legacy manifest、fix01 config 和 fix01 manifest 的哈希连接均正确；26 个恢复引用按 raw bytes 核验。

恢复配置 SHA256：`b406f4b4a98e12c786248cb6857e84a956abbd0a1677def77a5b0381503ceb0b`。本地与远程 `USER_STOP_VERIFICATION.json` 的 JSON 内容相同，raw SHA256 分别为 `47aaf4b1837bb7013c52e2b91fb0441a1398e56c65ac8af2ffc6d24f2de2bdfc` 和 `d2a3aab816253aa31b2046ff1e80d58ed0bdf7c2132f12ec26b1f3e177221167`。pause 的 stop hash 指向本地原始序列化，collection 的原件 hash 指向远程原始序列化，两条 raw 链及语义等价均通过。

主要证据：

- `CHECKPOINT_METADATA_REVIEW.json`：实际周期点、optimizer 槽、sampler、paired stream、停止点反例和计数。
- `CHECKPOINT_METADATA_TREE.json`：CPU 元数据树与 tensor 的 dtype、shape、device、有限性和字节摘要。
- `IMMUTABILITY_REFERENCE_REVIEW.json`：原 68/fix01 114 的不可变复验及所有绑定引用的哈希。

## 新恢复实现必须满足

1. 仅允许一次显式 `import_fix01_full_periodic6000(path, expected_sha256, fix01_identity)`，以注册的 raw SHA256 及旧 fix01 身份绑定，不接受 FAILURE 或任意自动 restore。
2. 导入完整 model、D 参数及 buffer、G/D optimizer、SOURCE/REAL sampler、三个 paired stream、last_metrics 和固定参考状态，并安装后逐树核验一致。
3. 按分支验证 G 的 6000/3000 step；按假设验证 D 的 3000 step；下一次 G 更新为 6001，必须 audit 和同版本 waveform replay。
4. 不改变模型、loss、初始化和训练 seed、TRAIN 人口、数据身份、计数权重、noise005 或 ADC/baseline/peak/crop256 读出。
5. 仅运行 full 和 simple，新执行计数在导入前为零，导入完成后从继承量扣除，失败中途新增 D 按实际 optimizer 更新记录。
6. 科学状态始终保持 `response_match_not_established`。训练完成、数值重放和科学 response match 分别报告。

## 已完成的新实现审查

- engine：SHA256 `efcd7b473fdcbb3ecb69c736b5f2862d7e5769c60752f9c3912a235d5e8f32f1`，42 项独立元数据检查通过。使用仅返回已保存 tensor 的元数据 doubles，不构造科学模型或 optimizer。真实 full6000 是正例；错误 purpose、mid-step、身份、配置、混合 step、missing slot、moment dtype/有限性、负二阶矩、sampler、paired stream 和 last_metrics 均按预期拒绝。递归 CPU optimizer clone 保持槽和 tensor bytes，signed zero 的字节差异可检出。
- driver：SHA256 `bea28aa29eb6df8a750168815ae56cf1db4204c3d20ef8bd480e0334b1999cb9`，50 项独立元数据检查通过。覆盖真实 plan/native stop 链、错误预算和旧退出链、JSON 重复键、逃逸路径、full 导入 new=0、首次 6001、partial D、full 完成后的 simple、排除重复计数、三 endpoint 混合 schema 和完成回执等待 native exit/evaluation。
- 运行助手：真实 26 个引用和输入链、13 项人工 JSON 反例、11 个监控状态分支共 32 项通过。active 但尚无 G 更新时为等待首次科学更新；complete 无 native 回执、native0 无 completion、native1 均保留独立状态。新增 26 项 native/prelaunch JSON fixture 检查通过，涵盖 3 G/6 D/6144 人工 forward rows、scope、device、source、PID、时间、内存和 native0 门控。最终助手哈希以独立 JSON 的 `files_sha256` 为准。

本审查先发现新 checkpoint writer 把嵌套 optimizer state 交给旧浅层 `cpu_state` 的错误；engine 已在新文件内改用递归 `cpu_tree`，旧源码未改。另指出 native QA 不能对未导入的 fresh full 直接训练；root 已接受改为人工周期 parser 导入后 joint6001，加 fresh simple mean/joint，且科学 hardcoded SHA 和 full-import guard 保持。

## 平台验证边界

engine 作者的 Windows CPU QA 已执行 4 G/6 D 人工更新并通过 57 个人工检查，但真实 full6000 的严格 engine 导入在初始 D reference 检查处失败，该 native1/失败回执保留。作者定位到 Windows 重建的初始 spectral-normalization buffer 有末位差异；G/common/noise 一致。这里的独立 CPU 正例检查的是 checkpoint 的完整元数据及固定 raw bytes，并不构成初始化 D 重建或实际 engine 导入通过。

目标 Linux 已保留严格科学检查，完成实际 checkpoint 的零 forward/step/draw 导入及 native QA。24 项独立 native 回执检查通过：导入后的 model、D、G/D optimizer、sampler、paired stream 的七棵树摘要与实际 full6000 逐项一致；G 的 180 个 named slot 和 D 的 30 个 named slot 摘要逐项一致，moment 未重置。

native exit 为 0，监督器 PID `11207` → timeout wrapper `11211` → QA Python `11212`，父 PID、命令和开始/完成/退出时间一致。人工 full joint6001 和 simple mean/joint 共执行 3 G/6 D，6144 人工 forward rows，三步同版本重放均通过。最大 GPU 使用 `2319450112` 字节，作业树最大 RSS `2857353216` 字节。实际 SOURCE/REAL forward、数值解码、科学 G/D 更新和 DEV 解码全部为零。该检查的恢复 manifest 身份是明确的冻结前 metadata dummy；科学 driver 将使用正式新 manifest 的哈希。

工程标签下的人工 parser fixture 显式恢复其初始化 D snapshot；科学路径保持严格初始 D reference 检查。Windows 失败回执继续保留。正确科学启动时仍须核对实际首步 6001、同版本重放、计数、正式冻结来源及 PID/native 链。

最终检查发现新 `launch_window008_resume01.sh` 含 CRLF，root 已在科学 freeze 前规范为 LF。旧 shell 文件保持原始字节，已 stage 的前版新 launcher 由版本化 stage 保存。原生 Python QA 不使用该 shell；正式科学 launcher 的新 LF 字节单独纳入本审查和最终 manifest。
