# 018 独立审核准备

范围：只读核对 017 合同与评分源；018 施工完成后审核其实际差异和必要人工例子。此文件尚未给出 018 实现结论，未读取或解码私有评价数组。

## 必需验收点

1. 保留 256 点硬输出、384 点载体、100 点基线及 ADC/first-argmax/crop；017 配置、源码、旧门与 pending 结论独立保留。
2. 原父事件、两 bank、每 bank 32 draw、三 noise hypothesis、原比较权重、原分组分母、异常/缺失原因和全部 flags 可追踪。明确 6593 全父诊断与 1319 留出父主门的角色。
3. 状态互斥且穷尽；每个数值带 mask；可测 rise、t10/t50/t90、右 t50 使用真实 crossing。masked 占位值只为编码，不能成为时间测量。
4. rise 下界仅用于正峰、左 10% crossing 缺失、左端确实高于 10% 峰、90% crossing 有定义的左截尾行。若完整 t10 存在，其应位于窗外，故 rise > t90_crop；其他行不生成下界。
5. cov8 仅去除不可全支持的 rise 维；保留其余八维条件协方差及原支持/95%/MC 门。rise 与状态、Q/峰高/FWHM 的同行联合关系由新观测表示评分补回；完整 rise 时间/PIT 保留为带选择范围的诊断。
6. 新 score、尺度、实际容忍度与 MC 精度前瞻冻结，标注新容忍度的选择依据；无学生结果后改门。保留有限负 U 差异，使用独立 cross 与排除对角的 self 分母。
7. 同 teacher 数组恒等检查分清经验分布恒等与 U 估计值；已有 native bank0/1 作阴性对照，原 independent 随机角色保留。预定义反例分别改变截尾率、只改截尾组 Q 关联、可测时间分布，必须实际被检出。
8. 全部主门范围精确列举并拒绝缺失/重复格；逐门说明是从 HARD_ROWS 重算、从 REPORT/moments 读取还是复用 producer 既有分数，执行状态与准备状态分别报告。

017 原 stage gate 的精确数目为每 trial 1344 格、两 trial 2688 格。若 018 用 cov8 替换 cov9，rise 时间及其两级 PIT 转诊断，并新增每组/权重/hypothesis 的观测联合能量，则预期每 trial 1272 主格、合计 2544 格；最终以冻结合同列举为准。

## 需要避免的科学死锁

- 64 draw 的条件 U 差异在高维、强尺度差异的表示下可能无法达到固定 MC 精度；阴性门 unresolved 应保留 pending，并用未来独立预算解决。
- 单一 Euclidean energy 在总体上区分嵌入分布，不保证现有有限样本能检出小的状态-电荷关联差异；三个反例必须分别检验。
- 沿用同一 teacher64 的标准差归一，会使 teacher 对角自归一，并使两 bank 的经验 MC 筛选与同一尺度相关。其结果是固定数组经验 screen；64 draw 的 oracle 精度与无偏性仍未建立。
- 017 旧 rise 支持不足不应继续把 cov8 或新全状态表示整体挂起。真正需要有限 rise 的量继续使用原 95% 支持规则。
- proper transformed energy 的可识别对象是当前窗内诱导的观测分布。窗外完整 rise 仍部分不可识别。

## 已读依据

- `AGENTS.md`、`research/N01/AGENTS.md`、`USER_MASTER_PLAN_20261015.txt`。
- `receipts/review_017_20261016/input_01/N01_017_review_and_next_actions.md`，19–38 行。
- `receipts/review_017_20261016/ACCEPTED_NEXT_ACTIONS.json`。
- `docs/WP4_017_CONFIGURATION.md`。
- `src/n01/saved_array_kernels017.py`，9–46 行。
- `src/n01/stats017.py`，39–144、239–299 行。
- `scripts/run_wp4_017.py`，644–692 行。

新 NN/AD/G/RNG、checkpoint 解码、原 SOURCE/REAL/DEV 波形读取和 SSH 均为 0。
