# N01 研究接口使用与复验

2026-10-19。当前接口可复现旧 028 参考模型及冻结的 S 研究候选。S 的完整原生保护验收失败，结果携带 `model_accepted=false`。本次正式黄金验证覆盖 **CPU 上的旧 028** 和 **CUDA 上的 S/CLI**：523 项张量检查全部逐字节一致，1,094 个输出行，原生进程退出 0。S 的 CPU 路径存在于实现中，本次未做其完整黄金验证。

## 1. 输入、配置与随机语义

输入 `source_waveforms` 是非空 `torch.float64[P,384]` 张量，表示已准备好的 clean SOURCE 电压，单位 V，软件采样间隔 1 ns。480 点原始数据需要另外明确准备流程。探测器物理标定仍记录为 `unknown`。

|参数|含义|
|---|---|
|`parent_ids`|长度 P 的父标签；CLI 接受一维字符串或整数数组。标签不进入神经网络|
|`n_draws`|每父输出次数，正整数|
|`seed`|抽样种子，整数且 `0 <= seed < 2**63`|
|`model_seed`|冻结模型头索引 0 或 1，与抽样种子分开|
|`noise_hypothesis`|`fast`、`slow`、`common_level`，三者是噪声延展敏感性假设|
|`return_contributions`|是否返回分量诊断；不改变数值输出|

S 使用原 `039/FIT/0` 的 shape、noise005、extension 三条随机流，神经块长 128；父事件优先、draw 次序展开。完整输入 batch、顺序和种子相同可重放。改变 batch 大小或顺序可能重新分配随机数；ID 只作为标签。两父、每父三次的 `row_parent=[0,0,0,1,1,1]`，`draw=[0,1,2,0,1,2]`。

旧 028 门面保留原 CPU loader、64 行神经块、随机流和 `shape_enabled` / `charge_enabled` 开关，同时增加 `front=Q-tail`。旧配置每个已加载对象的预算仍为 12 次 sample、48 输出行等原上限；它是受限参考接口。S 的标量 Q1 电荷分支固定关闭。

## 2. 已验证环境中的调用

已验证目录位于服务器：

```text
/home/rylan/projects/ej276d-n01/finalization_M6_API_validation_20261019
```

运行时记录见 [VALIDATED_LINUX_ENVIRONMENT.json](runtime/VALIDATED_LINUX_ENVIRONMENT.json)：Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.12.0+cu130、OSQP 1.1.3。GPU 为 RTX 5070，驱动 580.159.03。完整已安装版本清单见 [requirements-linux-validated.txt](runtime/requirements-linux-validated.txt)，其中第三方包版本不替代平台、CUDA 和源码身份。

先在该目录设置独立 OSQP 及项目导入路径，缓存仍位于项目中：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/runtime/osqp_1_1_3:$PWD/src:$PWD/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="$PWD/cache"
export MPLCONFIGDIR="$PWD/cache/mpl"
PYTHON=/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python
```

库调用如下。这里的 `source_batch` 和 `parent_ids` 由调用方按上表准备，采样次数由该次任务预算确定：

```python
import torch
from n01.finalization.api import load_response

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
response = load_response(
    "configs/finalization_research_S_20261019.json", device="cuda:0"
)
result = response.sample(
    source_waveforms=source_batch,
    parent_ids=parent_ids,
    n_draws=2,
    seed=2026101906,
    model_seed=0,
    noise_hypothesis="fast",
    return_contributions=True,
)
```

命令行会核验输入和配置 SHA，保存完整原生数组，并逐成员重新打开核对字节。以下是已执行过的四行 CLI 示例的复验命令；新的输出目录必须不存在。`PRIVATE_CLI_INPUT.npz` 是私有原 SOURCE 派生输入，留在授权环境。

```bash
"$PYTHON" -B finalization_run.py \
  --config configs/finalization_research_S_20261019.json \
  --config-sha256 14c873e2119fcba932a7cce2b6548debf149593d7f4e3104931a3de5f632ad55 \
  --input run_01/PRIVATE_CLI_INPUT.npz \
  --input-sha256 1b233092666602ebc7962f5d64784e733ff624abe7e3214ea41e591013b0dc87 \
  --output user_replay_cli_01 --draws 2 --seed 2026101906 \
  --model-seed 0 --noise-hypothesis fast --device cuda:0 \
  --threads 4 --gpu-gib 3 --contributions
```

该调用的实际账本为：6 次 checkpoint 加载、1 次 PCA 解码、2 个父事件 QP、4 行完整输出，无原始 TRAIN/REAL 人口解码。`--gpu-gib` 限制 PyTorch 分配器；进程 RSS 与墙钟由外层作业监督器限制。日常调用按实际任务登记资源，不能把此示例的计数套给更大 batch。

## 3. 输出与读出定义

`OUTPUT.npz` 不使用 pickle，`RECEIPT.json` 保存每个数组的 dtype、shape、SHA256，以及配置、输入、随机流和实际调用计数。

|字段|解释|
|---|---|
|`window_V[N,256]`|硬观察波形，N=P×n_draws|
|`carrier_adc[N,384]`、`pre_adc_carrier`|ADC 后完整载体与 ADC 前载体；CLI 总是保存|
|`baseline100`、`baseline_mean_V`|前 100 点 ADC 基线及均值|
|`peak_index`、`crop_start/stop/indices`|原生 first-argmax 峰与裁窗索引|
|`Q`、`front`、`tail`|原生窗积分，V ns；`front=Q-tail`|
|`PSD_raw`、`PSD`|原始比值与采用冻结 q_floor 的保护比值|
|`parent_ids`、`row_parent`、`draw`|父标签列表及每输出行到父、draw 的映射|
|`flags__*`|原观测异常，包括边界峰、Q/PSD 异常及 ADC 范围标记|
|`readout_features__*`、`readout_flags__*`|S 的峰高、rise、FWHM 及未定义/截尾状态|
|`diagnostic_flags__*`|S 的 ADC 并列峰、终态峰偏离均值峰、profile 端点钳位|
|`contributions__*`|开启诊断时的均值、随机修正、电子学与各阶段分量|

索引采用左闭右开约定：搜索峰 `p ∈ [112,144)`；Q 窗 `[p-16,p+240)`，front 窗 `[p-16,p+30)`，tail 窗 `[p+30,p+240)`。在 256 点输出窗中 tail 从 46 开始。先 ADC、再基线扣除与选峰；均值约束的参考峰和每次抽样的原生峰分别保留。形态未定义值以原 NaN 和 flags 表达，不能填零。

S 直接库调用总是提供完整 carrier；旧 028 直接调用保留其原返回条件，取 carrier/分量时使用 `return_contributions=True`。CLI 内部开启完整返回，再依据诊断开关保留或移除分量。

## 4. 复验入口及其范围

`finalization_validate.py` 是接口黄金验证入口。固定配置 SHA 为 `3c55846cbc761e9f978a970f59792c3a6aa50be40ef23e458fde51b5769264c0`。正式执行资源为 4 CPU、主进程/CLI 各 3 GiB GPU 分配器、12 GiB RSS、780 秒内部限时、900 秒外部限时加 20 秒退出宽限、32 MiB 输出；总计 18 次 checkpoint、2 次 PCA、1,032 个 QP、1,094 输出行。实际墙钟 8.007278 秒，峰 RSS 4,762,599,424 B。

已完成的验证包括旧 028 的 396 项、S 六条件的 96 项、重放/ID/多 draw 的 31 项逐字节检查，另有输入拒绝、模型和全局 CPU/CUDA RNG 不变、种子敏感性、实际 CLI 和 NPZ 重开检查。已保存的三个示例在本地再次逐数组核验，共 143 数组。

科学指标复算由 `scripts/validate_finalization_saved_native.py` 完成：24 份完整 JSON 与原报告逐项一致，158,232 行 ADC/wave 原字节和原生峰精确。该路径读取已保存响应与 REAL 参照，运行响应 forward 为零。它验证既有科学结果的可重算性。

## 5. 源码、模型与可复现范围

S 配置绑定 `finalization_S_20261019_fix01.json`，SHA 为 `46e9495548d2d3a43fd1e5ee2d5e4346ee6ac342d94782b2d4f832ad3bfd7280`。旧 028 配置 SHA 为 `48fbc9639a9435dd3d02bfde4d3568f4feabcf9b6f18f2f4d0fc48b6acde4b0a`。配置、六权重、PCA、统计和外部绝对路径的逐项身份由私有绑定清单记录。

已经验证的 loader 会检查历史审计依赖，完整源码闭包有 550 文件、119,182,587 字节，其中包含 21 个派生 NPZ。私有快照保留该闭包；权重及统计保留原身份。移动安装位置需要显式重绑定与重新验证，当前包没有静默改写这些地址。

可分享演示只含两份原数学模块、一个合成事件和人口级图表统计，能复验样条 QP、线性投影及图表生成。完整神经响应的外部独立复验还需要授权权重、统计、历史依赖和相同运行环境。
