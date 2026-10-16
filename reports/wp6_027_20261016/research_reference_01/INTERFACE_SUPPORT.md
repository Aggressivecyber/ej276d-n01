# 研究参考接口支持

当前Q1的构造链和硬观察原语可从冻结源码追溯。本包整理这些接口，统一`response.sample(...)`的封装与原生验收尚待完成。源码RAW身份见[REFERENCE_BINDINGS.json](REFERENCE_BINDINGS.json)；原016完整116成员manifest及37个项目导入项均保留引用。

## 已有构造与状态载入链

| 层 | 现有入口与字段 | 使用状态 |
| --- | --- | --- |
| 构造统计、A与005 | [prepare_numeric015:211](../../../scripts/run_wp3_015.py#L211)读取冻结TRAIN统计；236行构造`N01Response008`并载入A，238行载入005，241行构造extension并核A/005绑定 | 原训练运行已执行；该loader同时读取dataset，属于训练/评价入口 |
| 每seed M0头 | [run016:526](../../../scripts/run_wp3_016.py#L526)载入对应015 M0 CP；核`arm=M0`、G2000、原seed、完整step、mean_theta=0及冻结状态；533行取`new_head` | 016使用与seed对应的M0头，禁止混seed组装 |
| Q1模块 | [FrozenM0ChargeResponse016:60](../../../src/n01/model016.py#L60)沿A与M0形状头构造，持有`theta_a/theta_b`；[checkpoint:325](../../../scripts/run_wp3_016.py#L325)说明Q1 CP保存两theta、M0引用、初始化及训练状态 | Q1薄CP依赖外部A/M0/005与统计；独立推理loader待封装 |
| 完整依赖 | [原016 manifest](../../../WP3_016_SOURCE_MANIFEST.json)及其`project_import_closure` | 本包引用原闭包，未制作精简runtime或独立模型文件 |

`run_wp3_016.py`的历史main会创建数据sampler、训练流和优化器；它不是本包提供的推理命令。这里的链接用于查阅实现，027没有加载这些模块或checkpoint。

## 从目标API到实际原语

主计划目标签名如下，当前包没有导出该调用：

```text
response.sample(source_waveforms, n_draws, seed, detector_config, noise_hypothesis)
```

| 目标项 | 真实支持与源码定位 | 封装/验收欠项 |
| --- | --- | --- |
| `source_waveforms` | [transform_domain:59](../../../src/n01/data008.py#L59)处理有限原480点电压；SOURCE使用clean-values路径，REAL使用既有ADC路径。原016前向取准备好的`source_carrier384`；[model010:177](../../../src/n01/model010.py#L177)接收非空carrier tensor | 需明确外部输入是原480、carrier384还是已准备数据，并绑定转换规则；不能默许任意裁窗/单位 |
| 形状信号 | [model014.random_forward:60](../../../src/n01/model014.py#L60)、[model010.random_forward:177](../../../src/n01/model010.py#L177)沿原A与冻结M0头使用显式shape latent | source/batch/device、冻结状态和结果schema需统一验证 |
| 电荷随机性 | [model016.apply_charge:107](../../../src/n01/model016.py#L107)、[random_forward:141](../../../src/n01/model016.py#L141)：显式`z_charge`，正值均值校正law，条件为实际A的canonical Q；随机形状幅度沿原A | Q1 final theta应按原CP metadata载入；推理阶段冻结参数、no-grad与载入路径待原生验收 |
| `n_draws`与父引用 | 原016评价按父major/drawminor扩展保存行；[source_input_evidence016:179](../../../scripts/run_wp3_016.py#L179)核原输入/父/draw关系 | 模型forward本身不返回用户父ID；统一调用需返回完整父/draw映射，不能把重复draw当新父 |
| `seed` | [Streams010:127](../../../src/n01/train_shape010.py#L127)分离shape/005/extension的CPU float32流；[ChargeStreams016:118](../../../scripts/run_wp3_016.py#L118)另有独立charge流 | 单个外部seed到四流的namespace、连续调用/分块状态语义需冻结；没有新的统一seed接口 |
| `noise_hypothesis` | [fixed_noises:136](../../../src/n01/response008.py#L136)使用同一005 prefix与extension随机前缀分别形成fast/slow/common_level；[FrozenObservedNoise008.sample_prefix:308](../../../src/n01/observation008.py#L308)和[extend_prefix:317](../../../src/n01/observation008.py#L317)提供原语 | 统一接口应显式选择/返回假设身份；三假设不混池、不解释成全部真实噪声的覆盖 |
| 硬256输出 | [observe_generated_carrier:207](../../../src/n01/observation008.py#L207)经[_observation:75](../../../src/n01/observation008.py#L75)；`ste=False`为原硬ADC、100点基线、搜索范围first-argmax与256点gather | 封装须固定硬输出路径，禁止把可微训练代理冒充硬输出 |
| Q/tail/PSD与flags | 同一观察结果包含window、baseline100、peak/crop、Q/tail/PSD、低Q/分母保护、寻峰边界和ADC域flags；原语按开关返回carrier/indices | 与父ID/流标识/适用域组成统一结果对象，以及失败/越界返回行为，待封装和真实运行证明 |
| `detector_config` | 当前约束由冻结config、构造统计、ADC/窗规则、A/005/extension状态组成 | 目标参数对象与这些字节的绑定尚未实现；软件时间单位/名义采样间隔不等同独立硬件标定 |

原观察链为384点carrier、100点实际prefix基线、14bit ADC、[112:144)峰搜索first-argmax、[p−16:p+240)输出256点，tail从输出46点起。hard输出Q与preADC canonical Q/q_sig分别解释；基线不能用裁窗前100点代替。[观察合同](../../../OBSERVATION_CONTRACT.json)。

## 开关与可选诊断

| 项 | 已实现语义 |
| --- | --- |
| mean adapter | 当前Q1/M0研究参考均关闭；016没有对外的mean调参接口 |
| shape `enabled` | 控制原形状分支；charge是独立开关，不能把关闭shape解释为关闭全部随机性 |
| `charge_enabled=False` | [apply_charge:107](../../../src/n01/model016.py#L107)直接返回原M0信号tensor对象，含signed zero，不再加入随机量或均值补偿 |
| 005/长程extension | 原冻结状态与显式三hyp；保留实际prefix100和假设敏感性 |
| 时间warp、局部信号噪声、高秩/混合扩展 | 当前参考没有这些已验收接口；027未新增实现 |
| 可选贡献 | 现有signal/charge increment、005/extension与硬观察中间量可追溯；统一诊断返回、适用卡和原生示例仍待工程绑定 |

后续若制作可调用推理包，应另列CP载入、实际NN行、RNG请求、hard观察和示例的精确工程预算。027的静态身份核对不提供该原生执行证据。
