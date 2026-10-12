# N01 分布损失 V2 实现与人工验证

状态：实现已冻结；本文件记录用户在噪声/投影盲区诊断后授权的新修复。旧 src/n01/losses.py、旧训练和旧终点评估结果保持原口径。V2 改变损失定义，不能把旧结果重新标成 V2 的科学证据。本次本代理未训练主模型、未读取 VAL、未下载依赖。

## 1. 实际 API

~~~python
from n01.losses_v2 import (
    ProjectionContext, N01LossV2, BaselineDistributionLoss,
    fit_baseline_statistics,
    create_projection_context, create_evaluation_context,
    baseline_ols_features, empirical_wasserstein_1d,
)

loss = N01LossV2(train_loss_statistics,
                 seed=2026101211, evaluation_seed=2026101212)
context = loss.create_projection_context(scientific_update)
parts = loss.components(outputs, real, comparison_weights=None,
                        stage="joint_stochastic",
                        projection_context=context)
# parts: wave, QPSD, noise, regular, trend; adv 仍由训练器处理。
# loss.last_wave_components: swd, coordinate, short_lag（用于记录，已 detach）。
# loss.last_diagnostics: Q/PSD 条件支撑及 context 的 seed/update/fingerprint。

# 独立电子学拟合仅用合法 REAL TRAIN 前 100 点。
baseline_stats = fit_baseline_statistics(real_train)
baseline_loss = BaselineDistributionLoss(
    baseline_stats, seed=2026101221, evaluation_seed=2026101222)
baseline_context = baseline_loss.create_projection_context(scientific_update)
baseline_parts = baseline_loss.components(
    fake_baseline, real_baseline, projection_context=baseline_context)
baseline_total = baseline_loss.total(baseline_parts)
~~~

两种类的 forward 返回 components 字典；BaselineDistributionLoss.total 才合成标量。两者必须显式传 projection_context；没有默认隐藏计数器。模块函数 create_projection_context(update, seed) 与实例方法均可用；create_evaluation_context(seed) 返回无训练 update 的冻结评价上下文。

N01LossV2.population_objective(fake, real, weights, *, projection_context=...) 只返回 QPSD 分布损失，波形位置 detach、比较权重保留梯度。常规 components 将比较权重 detach 后用于生成器梯度；它们不是物理校准权重。

## 2. 投影刷新与重放合同

一次科学优化更新开始时，只创建一次 context。该次更新的全部 microbatch、合并的同参数分布、波形端点梯度、显式噪声重放、VJP、参数梯度审计及比较权重更新必须使用同一对象或其完全相同的序列化恢复值。不要在 forward、每个 microbatch 或第二遍重放中创建新 context。科学 update 由训练器传入，而不是由损失调用次数推算。

每个 context 的局部 CPU torch.Generator 种子由 SHA256("N01-loss-v2|kind|seed|update") 确定；不推进全局 torch RNG。训练 update 不同会换方向；同 seed/update 和同版本实现可重建完全相同的 bank。context.metadata() 返回 kind、seed、update、SHA256 fingerprint；state_dict()/from_state_dict() 保存和核对实际 bank，而非仅依赖恢复时 RNG。

context 的 dataclass 和字典映射不可赋值，构造时 tensor 从输入独立 clone。validate() 检查 PyTorch 原位版本号。Tensor 存储本身不能强制只读；通过 .data 或共享可写 numpy 视图的修改可能绕过版本号，所以禁止这类操作。所谓不可变是调用合同，不是能拦截所有底层写入的安全边界。设备转换使用 bank.to(...) 的新值，不覆盖 CPU bank。

冻结评价用独立 seed 和 kind="evaluation"，update=None；构造器拒绝训练/评价 seed 相同，调用也核对这种冲突。同一端点及 REAL-REAL 控制共享该冻结评价 bank，不能每次测量刷新。训练 seed 默认 2026101211、评价 seed 默认 2026101212；独立 baseline 分别默认 2026101221/2026101222。若训练器注册了其他种子，必须明确记录。

N01LossV2 继承旧类的 TRAIN 统计和 state_dict 布局，其中保留的旧固定 direction buffers 不参与 V2 forward。保存/恢复必须同时标记 loss_version=2、实例种子和 context；不能把 V2 state_dict 交给旧 evaluate_checkpoint 的 N01Loss 来得到所谓 V2 评价。

## 3. 固定波形组合

标准化只用原合法 TRAIN 拟合的每坐标 median 与 max(IQR/1.349, resolution_floor)。中心/尺度沿用原保存张量；V2 的标准化、关键 Q/PSD 和 W1 运算用 float64。float32 输入已损失的精度不会因此恢复。

六个视图保持原定义：

| 视图 | 维数 | 定义 |
| --- | ---: | --- |
| voltage | 480 | 电压完整窗口 |
| shape | 480 | 减去前 100 点均值后除以事件正峰，使用既有 TRAIN 峰 floor |
| difference | 479 | shape 一阶差分 |
| curvature | 478 | shape 二阶差分 |
| difference_smooth_5 | 479 | shape 复制边界 5 点均值平滑后的一阶差分 |
| difference_smooth_17 | 479 | shape 复制边界 17 点均值平滑后的一阶差分 |

每个视图每更新生成 32 个单位方向，沿事件轴比较投影分布的经验 W1，再对方向/视图平均。

逐坐标项仅用于 voltage 和 shape：每个时间坐标分别沿事件轴比较经验 W1，最后平均。短 lag 联合项也仅用这两个视图，lag 固定为 1、2、4、8；对每一个合法时间 t 的 (z[t], z[t+lag]) 使用 4 个刷新二维方向。时间 × 方向保留为不同列，W1 只沿事件轴排序；不把全部时间点混合成一个分布。

总波形项固定为：

~~~text
wave = 0.50 * mean_six_view_SWD
     + 0.25 * mean_voltage_shape_coordinate_W1
     + 0.25 * mean_voltage_shape_lag1_2_4_8_joint_W1
~~~

该组合由诊断修复时预先绑定，未根据 VAL 或终点挑权重。mean_response 阶段使用对应均值差，而非强迫恢复随机宽度；其短 lag 投影均值不是联合分布约束，随机宽度应在 joint_stochastic/low_learning_rate 阶段检查。

这些投影、边缘和短 lag 联合项不是完整波形分布的充分统计，不能保证消除全部高阶相关盲区。它们可检出已构造的旧固定投影零空间例子，但不证明旧模型实际利用了该盲区，也不替代独立端点评价、ADC、adversarial 或结构 regular 检查。

## 4. 保留的 Q/PSD、权重与支撑语义

Q = sum(x[100:400]) - 3 * sum(x[:100])，单位 V ns、采样间隔 1 ns；PSD 分子为 sum(x[150:400]) - 2.5 * sum(x[:100])。关键求和用 float64。

Q 的正常区域使用 log(Q/Q_reference)，低 Q 使用既有单调延拓；PSD 分母按符号应用 Q floor。所有行都进入全局项，不丢弃负 Q、近零 Q 或范围外行。数值延拓/guarded ratio 不赋予非正 Q 物理 PSD 含义；last_diagnostics 明确记录 nonpositive 和 guarded 行数。

全局项为 Q/PSD 两个逐坐标 W1 与 16 个刷新联合投影 W1 的等权组合。条件箱沿用 REAL TRAIN 8 等概率 logQ 箱的 7 条边界，首尾开放保留 under/overflow；生成输出 Q 与 REAL Q 各自分箱，不把原始 SOURCE Q 箱当成输出等价箱。每箱 fake/real 都至少 16 行才计算条件 PSD W1。条件项只对当前支持的箱平均，与全局项等权；全部箱无支撑时保留全局项并明确报告不完整覆盖。每箱计数、支持状态、conditional_rebatch_needed 和全局纳入行数可审计。

主噪声项沿用前 100 点的 32 个 TRAIN 标准化特征：逐坐标 W1 和 32 个刷新联合方向 W1 各占一半。regular/trend 仍是生成器输出的标量，本文件未添加旧实验约束。

## 5. 独立 BaselineDistributionLoss

fit_baseline_statistics 接受 (N,100)/(N,480)/(N,1,T) 的合法 REAL TRAIN，实际只读取前 100 点。至少两行；非有限 baseline 行显式失败，无静默剔除。脉冲窗口不参与统计拟合。调用方仍须负责核对 TRAIN 角色和组/父事件隔离，函数名称不能自行证明输入 provenance。

固定总目标：

~~~text
baseline_total = 0.50   * full100_SWD_32_directions
               +0.25   * full100_coordinate_W1
               +0.125  * noise32_feature_marginal_W1
               +0.0625 * noise32_feature_joint_SWD_16_directions
               +0.0625 * OLS4_joint_SWD_16_directions
~~~

noise32 是前 100 点均值、斜率、去均值 logRMS、ACF lag1..16、去均值幅度分位点 1%/50%/99%、五个频带功率占比、五个局部去均值 logvariance。频带为 rFFT 的 [1:3]、[3:6]、[6:11]、[11:21]、[21:51]，DC 不进入总功率。

OLS4 是 baseline 均值、OLS slope、OLS intercept、OLS 去趋势 logRMS。OLS 时间坐标固定为 2*sample/479-1，在前 100 点上拟合；intercept 指完整窗口坐标 0 处的线性外推值，不是 baseline 中点。对四个观测量做联合投影；它们的定义并不假定潜在漂移系数独立，均值与截距的代数关联保留在 joint 中。

各特征尺度均只用上述 TRAIN median/IQR，采用 ADC/数值 floor；没有 SOURCE 输入、粒子标签、ID、全脉冲 Q 或 GAN 判别器。所有 baseline 输入先取前 100 点再转 float64。

根调度预先绑定的拟合预算建议为 4000 更新 Adam lr=1e-3，然后 1000 更新 lr=2e-4，固定末端，独立评价/门槛由根登记。该预算不是本代理执行的训练结果，也不以 VAL 最好点选择模型。

## 6. W1 快路与预算

新增 empirical_wasserstein_1d 的 equal-N uniform 快路是经验 W1 的数学等价实现：对每列排序后取 |sort(fake)-sort(real)| 的平均。只在两侧样本数相等、权重均为 None 或无梯度且严格正恒定张量时启用；不使用近似均匀容差。任何 requires_grad=True 的权重，即便数值恰好全 1，也进入旧合并 CDF W1，保留比较权重梯度。非恒定权重、unequal-N 同样回退。

短 lag 项共有 8 组比较，每组保留 (480-lag)*4 列。快路避免均匀同规模分布的合并 CDF 与双 searchsorted 开销；未把时间轴合并或改变数学目标。主项仍每视图 32 方向；baseline 只需 full100 的 32 方向及低维 joint。运行速度和设备内存必须由实际训练器工程测试确认，不能由 CPU 人工测试推断 GPU 每步上限。

## 7. 人工验证回执

本代理执行命令（工作目录 E:/GAN；全部输入为人工数据）：

~~~text
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
TEMP=E:/GAN/runtime_tmp
TMP=E:/GAN/runtime_tmp
E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe
    -m unittest discover -s research/N01/tests -p test_losses_v2.py -v
Ran 8 tests in 2.393s
OK
~~~

覆盖：

1. equal-N uniform 快路与旧 CDF 公式的 float64 数值及样本梯度等价（避开相同值非光滑点）。
2. 数值恒定但可学习的权重、近恒定权重、unequal-N 正确回退；权重梯度有限且非零。
3. 相同训练 seed/update 完全复现，更新方向变化、评价 seed 独立、全局 RNG 不变、序列化 fingerprint 一致及常规 torch 原位修改被拒绝。
4. 同一个 context 两次损失及波形梯度完全相同，梯度有限且非零；缺少 context 显式失败。
5. 在 96 条人工波形上构造仅尾部修改的旧六视图联合零空间。刻意保留旧 SWD bank 后，旧 SWD 仍近零，而 V2 coordinate 与 short_lag 均检出，因此不把检出简单归因于换 seed。
6. population_objective 只更新比较权重、不更新 detached 波形；全局保留所有行并报告稀疏箱支撑。
7. baseline-only 统计不读取后 380 点，相同分布所有 component 为零，人工线性漂移恢复 OLS 系数。
8. 人工单参数 baseline gain 的 12 步 Adam 拟合梯度有限，独立冻结评价目标下降；这是可微性工程检查，不是电子学科学拟合或主模型训练。

本地验证不使用 GPU、真实 TRAIN/VAL 波形或科学 checkpoint。测试对旧零空间的检出不包含 ADC、adv、regular，也不声称对所有未知分布偏差充分。根的远端训练器工程测试、固定末端电子学拟合和科学 gates 必须分别保存回执。