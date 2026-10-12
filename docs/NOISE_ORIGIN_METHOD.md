# TRAIN 原始档案→N01 处理链噪声来源方法

本脚本实现 noise_origin_audit_003_20261012 的逐事件分析，不训练，不改 gate，不读 VAL waveform member。当前只通过合成人工测试；正式科学执行由根任务登记、审核和固定 manifest 后启动。本文件不继承旧 R14 科学协议，只使用已复制并 SHA 绑定的精确数据接口。

raw_npz 是目前最早可追溯的波形档案，不等于已证明的仪器 native raw。全部结果条件于已经选定的合法 TRAIN IDs，selection_effect_identified=False；早窗、preCFD 窗口不自动视为 pure electronics。

## 输入与复用配置

使用根已登记 noise_origin_003_config.json，不修改其内容/hash：

- training_container：原 TRAIN_PRIVATE.npz 的 path/SHA，只读取 target。
- identities：PARENT_IDENTITIES.npz 的 path/SHA，只读取 real_train_original_index、real_train_group。
- raw_source：档案 path/bytes/SHA、ZIP member 的精确 shape/dtype/bytes/CRC 与新预算 io_caps。
- reference_interface：冻结 interface_core.py 的 path/SHA。动态 import 后仅使用 raw streaming、digitize、辅助 normalization、integer CFD、window helper；不运行旧 extract_train.py。
- origin_audit：21047 TRAIN rows、422 原 artificial groups、100 block/2000 raw samples、fixed starts [0,5,10,15,20]、bootstrap512/seed2026101259、保存 blockmeans。
- resource：使用新登记 wall limit；script numerical CPU threads 固定 2，在 root 上限内。source reader 保留精确 I/O reservation、CRC/EOF、同文件句柄 stat 和 deadline；memory bound 取 raw_source 与新 resource RSS 上限较小值。根 supervisor 另监整 phase wall/RSS/output。

CLI：

~~~text
python scripts/audit_noise_origin.py --config noise_origin_003_config.json --out <fresh-output-directory>
~~~

可选 --container、--identities、--raw-path、--reference-interface-path 只覆盖实际路径，不覆盖 config 中任何 SHA/schema。相对路径按 config 所在目录解析。Windows 输出仅 E/D；非空 output directory 一律拒绝，保留 partial/failure 或既有 complete 文件。

## 精确重放与流式读取

源 waveform member 为 100000×10000 float64，每采样间隔的软件 time_ns grid 为 0.05 ns。冻结 reader 整容器 hash 一次，inflate/time-check/CRC 一次；只有 TRAIN IDs 的 raw waveform rows 才 np.frombuffer 数值解码。其它行字节经过解压后跳过，不进行 waveform numeric decode。time_ns 的 10000 个时间值用于 grid 核对，其它 scalar/file_names 成员仅核对 ZIP 元数据，不做数值/字符串内容解析。

按 identity 顺序保存数据，raw stream 可按 sorted IDs 读取，再映射回原 target order。每事件：

1. raw10000 reshape(500,20).mean(axis=1)，得到 float64 blockmeans，1 ns 分辨率。
2. q64=reference digitize(block)，round→clip14bit[-0.5,0.5]→float64 voltage。
3. q32=q64.astype(float32)。
4. 辅助 float32 normalization 用前100点均值和正峰，仅供 CFD；原压不会替换成辅助归一波形。
5. CFD 0.5 是主峰前最后一个 below-threshold 样点之后的整数索引，没有亚样点插值；tp−120 开始窗口480点。reference window 输出 float32。
6. 该完整480 float32 window 与 target 对应行做 bytes 相等检查。任何 bit 不等立即停止，不继续后续行/阶段，不生成成功 marker。
7. N01 再 ADC 按 canonical_real 的合法电压检查和 rint 重新格点化，统计其前100点。

只有全21047 replay与source CRC/EOF均成功，才计算/保存完整结果。不保存任何逐事件10000点或2000点长 raw waveform。完整21047×500 blockmeans 的无压缩 numeric payload 为84,188,000字节，可重建后续接口；metadata与baseline派生features另存。

每1000个selected raw rows写status.json并print真实decoded/bitwise/current-ID计数；成功status=complete，失败保留FAILURE_RECEIPT/status。原波形不得因不合法值、正峰或重放错误而静默删除。

## 主链、固定窗和 paired 分支

primary_chain：

~~~text
raw_fixed_0 → block_fixed_0 → q64_fixed_0 → q32_fixed_0
            → postCFD_q32 → N01_readc
~~~

fixed start0/5/10/15/20：raw支持窗分别 raw[start*20:start*20+2000]，对应block[start:start+100]。每个raw窗都是相同100 ns采样支持长度，block均值代表20raw样点平均，而非单raw点抽样。

前100点辅助normalization和选定cohort已经可能含触发/峰相关选择，因此这些窗口不是被证明的独立电子学片段；本审计不从单条链变化宣称硬件原因。

两个位置定义：

- left_pre：block float64 自身先减前100均值、除正峰，再用同一 integerCFD 规则，未ADC/cast。
- left_post：实际q32和reference float32辅助normalization的CFD位置。

prospective paired contrasts：

| 比较 | 解释范围 |
| --- | --- |
| align(block,left_post) vs align(q64,left_post) | 固定实际位置，隔离ADC算子 |
| align(q64,left_post) vs align(q32,left_post) | 固定实际位置，隔离float32 cast |
| align(block,left_pre) vs align(block,left_post) | 同一个未ADC block，只改变CFD选的位置 |
| raw_aligned_post vs block_aligned_post | 相同选窗raw支持与block平均的比较 |

preADC/q64 paired数据直接float64 slice，不用会强制float32的旧window helper。exact target重放才调用reference window函数。非法full480 left(<0或>20)在paired分支表示NaN+valid mask，不padding、不作为有效早窗；报告n_total/n_valid/n_invalid。所有原事件仍保留在完整metadata及主target重放，全局不会因prebranch非法而缩小样本数。actual left、原重放padding、clip counts、tp_pre/post和shift差均逐事件保存并在报告汇总。

Cov(B−A)与Cov(B)−Cov(A)分别报告：covariance_of_paired_feature_difference_mean_physical_slope，以及 covariance_change_second_minus_first_same_valid_rows。二者都使用同valid-parent交集，不能互相替代或把Cov(B−A)误称处理造成的协方差增量。

## 统计、单位与群组

每baseline stage保留per-event三个features，列顺序：

~~~text
[mean_V, centered_slope_normalized_V, centered_slope_V_per_ns]
~~~

slope在该窗端点[-1,1]上做OLS；physical slope = 2*normalized slope/endpoint_span_ns。raw2000的端点span=99.95ns，block100端点span=99ns。raw/block数学event mean应一致到浮点精度；normalized slope端点不同，不能仅凭数值差异宣称减采样创造异常。人工线性斜坡检查证明physical slope一致，而normalized slope比为99.95/99。

raw阶段累加完整2000位置mean/SD vectors，不保存逐事件longrows，也不构造2000×2000 covariance。block/q64/q32/paired100点阶段保存完整100×100 covariance。ACF用每事件去均值序列，物理lag=1..20ns（raw对应20..400样点，block对应1..20），方差为0时ACF未定义而非填0，报告无效数。软件time grid记录start/end和dt mean/min/max；不称仪器时间基准已验收。

每stage报告mean–slope covariance/correlation、mean与SD、time profile、ACF，并分解within/between original groups；两个贡献都用Nvalid−1 denominator，和等于该有效子集total covariance。另报告与left/peak/Q的Pearson关联，只作描述，不宣称因果。

bootstrap固定512次、seed2026101259，以原422人工组为cluster，共享同一重采样plan across stages。各stage对自身明确valid subset给correlation和crosscov percentile95；这些是stage边缘区间，并非paired covariance-change区间。局部无效branch不改变主链counts，有限原组或无支持resample数明确报告。人工50record分组不等于独立采集；interval条件于选定TRAIN和本档案，不能覆盖真正独立acquisition不确定度。

## 输出和严格完成校验

- ORIGIN_REPORT.json：阶段定义/counts/profile/ACF/covariance/paired/关联/bootstrap/source-I/O与限制。
- BLOCK_MEANS.npz：完整TRAIN block_mean_V_f64、original_index、artificial_group。
- EVENT_FEATURES.npz：原IDs/groups、metadata_*、features_*、ACF_*，无longrawrows。
- STAGE_COVARIANCES.npz：100点各stage完整covariance。
- AUDIT_COMPLETE.json：实际行数/bitwise/source-I/O/implementation SHA及稳定输出SHA。
- COMPLETE.json：仅最后成功时写，status=complete、report_sha256、rows_completed、bitwise_target_equal=true、noise_training_updates=0、response_training_updates=0、validation_waveforms_decoded=false。
- FAILURE_RECEIPT.json：失败时保留目前selected decode和bitwise count、current ID、错误、source预算及traceback；没有COMPLETE。status.json只作progress，不代替验收。

根runner须检查exit0、COMPLETE、report SHA以及rows21047/全行bitwise后才开始后续RR校准。RR由另一个独立script实现，本脚本无classifier或训练更新。

## 人工回归回执

本代理未执行科学数据审计。所有测试输入人工生成，临时目录 E:/GAN/runtime_tmp，Python已指定E盘venv，PYTHONDONTWRITEBYTECODE=1，CPU threads2：

~~~text
python -m unittest discover -s research/N01/tests -p test_noise_origin.py -v
Ran 7 tests in 0.532s
OK
~~~

测试覆盖：reference convert_row精确q32重放与main-peak integerCFD；raw/block线性坡度坐标；float64 paired slices/非法分支NaN；量化改变CFD位置的人工例子；Cov(B−A)与ΔCov区别；selected stream跳过未选NaN raw且只访问TRAIN成员、保持target order、receipt SHA可重开；单bit target错误在首个selected事件立即停止、无成功marker、partial目录拒绝覆盖。人工成功不等于正式全21047事件重放已经验收，后者必须等待注册执行回执。