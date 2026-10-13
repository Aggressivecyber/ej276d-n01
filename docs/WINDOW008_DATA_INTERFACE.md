# WINDOW008：保留电压单位的 256 点 TRAIN 接口

本文件是待 root 冻结的数据接口实现说明，不启动训练、不改变原 007 配置或科学结论。用户已选择缩短有效窗口；本接口只复用绑定的合法 TRAIN480，不重新读取原始10000点归档。原20点blockmean链不重做，调用未修改的 `data.load_training`。另保存两域原前384点作为未扣基线的内部载体；最终有效输出和比较长度为256。

## 固定变换

两域各自在384点载体的 `[112,144)` 中取第一个最大值的位置p。SOURCE是解析清洁期望电压，保持pristine浮点，不做ADC；REAL在原loader恢复canonical ADC后经 `observation008.observe_numpy(...,already_digitized=True)`，共同ADC是REAL/GEN的观测约定，不适用于SOURCE输入。全384最大值在搜索外、或p命中搜索首尾时均保存flag，不丢行、不换峰、不改搜索范围。该范围保证p−16至p+239均有真实载体采样。首次只读TRAIN设计审计用较宽的几何合法搜索，SOURCE峰均126、REAL峰120–137；root据现有对齐定义将正式搜索固定为 `[112,144)`，未据DEV调范围。

取各域 `[0,100)` 100点均值b，输出 `v[p-16:p+240] - b`，共256点float64电压。峰位于输出index16。REAL的v已是原14bit inclusive[-0.5,0.5]V ADC；共算子只验证幂等并直接使用，不额外改值。SOURCE的v保持原期望电压。无SOURCE量化、峰归一化、附加噪声、2倍增益、±2限幅、补零、重采样或裁后ADC；原REAL ADC的饱和边界保持不变。基线扣除后不再量化。`baseline100`保存用于本域观测的100点，`raw_baseline100`保存载体原100点；均值、peak_index480与crop_start480单独保存，index名称保留相对原480的坐标。

## 新读出与统计

软件时间间隔1ns；`Q256=sum(wave256)*1ns`，`tail_charge256=sum(wave256[46:256])*1ns`，`PSD256=tail_charge256/Q256`。index46相当于峰后30ns。它们是新的软件读出约定，不宣称硬件已验证，不复用原 `[100,400)` Q或旧Q箱。Q≤1e-8 V ns时整个准备失败，不丢事件。带符号尾部可使PSD<0或>1，保留原值并统计，不clip或伪造保护值。

`fit_statistics008` 对全部SOURCE TRAIN和REAL TRAIN分别保存新Q/PSD分位数、Q八分位边界、256逐时均值/标准差、幅度、峰位及baseline统计。两域相同箱号不代表同一Q范围或同一人口成员；跨域条件损失的公共边界由root另行冻结。`model`字段来自 `fit_model_statistics008`：q_ref/a_ref/logq范围只用pristine新SOURCE256，保留原model API字段名；source峰126时Q精确取原 `[110,366)`；electronics统计只用REAL baseline100，时间坐标按384点载体计算；alignment_t0=126明确是内部载体的SOURCE峰参考。loss统计由root的独立008损失入口拟合，本接口不重用或生成旧loss统计。所有统计不读取DEV。

## 文件与接口

- `load_training008(container, identity_path, spec=None)` 保留原 `source/real/source_ids/real_ids/real_groups/h/original_source_weights` 键；source/real变为256点float64电压。每域另有 `_carrier384`、`_baseline100`、`_raw_baseline100`、`_baseline_mean_V`、`_peak_index480`、`_crop_start480`、`_peak_amplitude_V`、`_peak_tied_samples`、`_peak_search_edge`、`_global_peak_outside_search`、`_Q256`、`_tail_charge256`、`_PSD256`、`_role`。
- 保留原512个SOURCE DEV IDs及599个REAL DEV IDs/groups和DEV角色，不读取 `source_val`、`target_val` 或 `hval`。ID、角色、组号和h仅作谱系/审计，不作为神经输入。
- `readout256_np` 是统一新读出；`window_spec` 是纯数据spec解析器，默认值在 `DEFAULT_SPEC`，只接收该对象字段。原输入容器及身份文件SHA256在解码前核对。
- 准备入口创建 `DATASET008.npz`、`FIT_DATA_STATISTICS.json`、`PREPARATION_RECEIPT.json`，输出存在即拒绝覆盖。保存后每个数组重新读回并逐值精确核对，原文件不改。

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONDONTWRITEBYTECODE='1'
& 'E:/GAN/runtime_envs/local_cuda_retry_20261007073613_UTC/venv/Scripts/python.exe' -B scripts/prepare_window008.py --container 'E:/GAN/research/sim2real_two_papers_20261008_0305/work/R23_ONLINE_STANDARDIZED_CRITIC_001/TRAIN_PRIVATE.npz' --out 'E:/GAN/research/N01/runs/window008_20261013/prepared'
```

准备内置7组人工CPU检查：裁窗索引/峰位置、尾积分位置、未center载体原值保留、幅度信息保留、负尾PSD保留、搜索外峰标记且不删行、零电荷拒绝。人工检查和CPU数据准备均为0科学训练更新、0模型前向，不是响应匹配验收。

## 已知比较边界

各域以自身峰对齐会移除原相对峰时偏差，并使噪声条件化于寻峰规则。256点最前16点不是原100点baseline的替代；噪声模型应使用另存的baseline100。统计中的峰高及原Q不是硬件校准真值。新窗口仍保留原TRAIN/DEV角色及单采集边界，不能据此宣称独立泛化。
