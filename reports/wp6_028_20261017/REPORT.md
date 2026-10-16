# 028：冻结研究响应接口与原生示例

`load_reference028` 与 `response.sample` 已实现，并通过服务器 CPU 原生示例和独立保存输出复核。接口使用原 A、005、两个 M0 及两个 Q1 终点，共六份冻结权重；本次没有训练或选择部署 seed。

这关闭了 WP6 的研究推理接口与固定示例任务。科学状态仍为 `response_match_not_established`；完整 WP5/WP6 尚未完成。原 020、023、025 的未通过结论保持有效，详见[既有适用范围与失败卡](../wp6_027_20261016/research_reference_01/APPLICABILITY_AND_FAILURES.md)。

## 已实现的输入和输出

输入为已准备的清洁 SOURCE `torch.float64[P,384]` CPU 张量，单位 V，软件采样间隔 1 ns。接口按父事件优先、draw 次之返回 `P*n_draws` 行 256 点硬观察波形、Q/tail、原始及保护后的 PSD、七项 flags、父事件与 draw 映射。`return_contributions=True` 额外返回各响应分量与随机输入。父事件 ID 仅用于追踪。

模型 seed 必须显式取 0 或 1；电子学假设必须显式取 `fast`、`slow` 或 `common_level`。`detector_config` 是冻结配置身份，未开放任意探测器参数修改。关闭 charge 保留原 M0，关闭 shape 后 M0 参考等于 A，charge 仍可开启。

以下展示调用方式，`source_batch`、`parent_ids` 与 `model_seed` 由调用方提供。在登记服务器的冻结目录中使用，原权重和统计文件路径由配置绑定；公开源码包不包含这些私有二进制文件。

```python
from n01.inference028 import load_reference028

response = load_reference028("wp6_028_config.json", device="cpu")
result = response.sample(
    source_waveforms=source_batch,
    n_draws=2,
    seed=2026101728,
    detector_config=response.detector_config,
    noise_hypothesis="fast",
    model_seed=model_seed,
    parent_ids=parent_ids,
    return_contributions=True,
)
```

相同完整 batch、顺序、draw 数、seed、开关、固定 64 行神经网络分块规则和软件环境可重放。每次同 seed 调用会从相同随机流起点开始，不能计为新的独立抽样；跨假设共享相同随机输入、005 前缀与 scale。005 与后续扩展保留原统计耦合。

## 原生执行与独立复核

固定使用 TRAIN 的前两个父事件，每父两次 draw；两个模型 seed 分别执行三种电子学假设、fast 重放、chargeoff 和 shapeoff，共 12 次调用、48 行输出。原生输入读取一次准备数据 NPZ 的三个成员，实际解码各成员全部 6593 行后选取两父；没有读取 REAL、DEV 或原 480 点成员。

| 检查 | 结果 |
| --- | --- |
| 六份 CP、源码及配置身份 | 全部符合冻结 SHA；执行前后模型状态一致 |
| 两 seed × 六调用 | 12/12 工程检查通过，全部 26 项运行计数与预算相等 |
| 完整 fast 重放与随机输入 | 12 组结果哈希、60 组随机输入哈希复核一致 |
| chargeoff / shapeoff | 保存输出逐字节身份成立；原生代码另核对 chargeoff 对象别名 |
| 硬观察链 | 保存 ADC 的基线、寻峰、裁窗、父/draw、flags 和 PSD 代数通过；独审未再次执行 ADC |
| Q / tail 复算最大绝对残差 | 3.1086×10⁻¹⁵ / 6.6613×10⁻¹⁶，均满足预登记的逐行浮点归约界 |
| 七项 flags | 本示例全部为 0；未覆盖异常边界压力测试 |

独审仅打开生成的一个 NPZ 一次，加载 37 成员、182940 值、1386296 B 数组载荷。NumPy 与 Torch 归约的比较使用读取前固定的 `2*gamma_(n+1)*sum(abs(values))` 界，未按结果调整；所有字节身份检查保持严格相等。完整计数和误差见[独立复核](../../receipts/wp6_028_20261017/postrun_independent_01/RESULT_REVIEW.json)。

服务器原生退出码 0，外层耗时 1.868452 秒，全程 child 峰值 RSS 843771904 B（约 0.786 GiB），产物 391057 B，无资源超限。实际环境为 Python 3.12.3、Torch 2.12.0+cu130、NumPy 1.26.4，使用单 CPU。独审耗时 1.554417 秒，最终序列化前记录的峰值工作集 33148928 B。结果已取回 E 盘并核对 RAW SHA。

新增计算为 48 行 A、48 行形状头、48 行 005/扩展/硬观察；采样请求 60 次、18912 个正态标量。构造阶段另有 120 次分布请求、1478228 个标量，随机初始化随后由冻结权重覆盖。新增 AD、G、优化、loss/CDF、拟合及训练均为 0；项目 G 累计下界保持 187667。

## 适用边界与后续

本次原生验收限定为 CPU、每次四行的固定示例，未验证跨分块的大 batch、任意拆分调用等价、CUDA、原始 480 点输入适配或异常事件压力测试。完整交付仍需同一确认合同下的必要基线比较、全流程分组内部验证与独立 C2ST/REAL–REAL、最终系统选择、自足模型包和综合论文材料；如进入部署，全 TRAIN 重拟合另行登记。当前接口是冻结研究参考，未建立硬件标定或跨采集物理接受。

028 已结束，当前没有活动科学作业。本次短作业在首次有界状态查询前已完成，无需为它设置后续唤醒。下一科学工作应直接围绕 WP5 最终确认缺项形成有限预算，不重开已关闭的候选或将本次软件通过计作分布匹配。

依据：[前瞻登记](../../receipts/wp6_028_20261017/registration_01/PROSPECTIVE_REGISTRATION.json)、[源码复核](../../receipts/wp6_028_20261017/independent_01/FINAL_SOURCE_REVIEW.json)、[原生终态](../../runs/wp6_028_20261017/selected_collected_01/runs/primary/supervisor/NATIVE_EXIT.json)、[回收回执](../../receipts/wp6_028_20261017/collection_01/COLLECTION_RECEIPT.json)。

源码提交：`c0432fba8867f9c79c0016e4c98b496cffa75ef6`。冻结配置 SHA256：`48fbc9639a9435dd3d02bfde4d3568f4feabcf9b6f18f2f4d0fc48b6acde4b0a`。完整文件身份与最终状态见[结果回执](../../receipts/wp6_028_20261017/FINAL_RESULT_RECEIPT.json)。
