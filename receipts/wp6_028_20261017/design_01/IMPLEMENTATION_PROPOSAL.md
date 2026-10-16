# WP6-028 最小推理接口：静态实施提案

状态：仅完成静态设计，等待 ROOT 登记范围后实施和执行。本文与 `DESIGN_FACTS.json` 是本阶段仅有的两个新增文件。基线由 ROOT 报告为 `d2029ec44f241fe1e9c281993e5dbfbd4c2569af`；本文未调用 Git 或 SSH。

目标是在现有 Q1/M0 研究参考上提供实际可调用的 `response.sample`，并用一个固定小例验证接口连接、随机流、硬观察和身份。科学状态继续为 `response_match_not_established`；本工作不补做完整 WP5/WP6，不选择部署 seed。

## 1. 最小源码范围和 API

登记后建议只新增 `src/n01/inference028.py` 和 `scripts/example_response028.py`。前者是一份长期使用的 loader/API；后者直接执行下文固定示例和必要断言。旧原语保持原路径与字节；不导入训练 driver，不创建 optimizer/sampler/controller，不运行旧 suite。资源监督与执行登记由 ROOT 复用现有机制。

建议由 `load_reference028(root, device='cpu')` 一次构造共享 A、005/extension 与两个分别绑定的 Q1 对象，返回一个 `ResearchResponse028`。首个验收只覆盖 CPU 串行调用；不承诺 CUDA 或并发调用验收。模型 seed 是显式选择的 CP 身份，采样 seed 是本次随机库身份，两者分开：

```python
response = load_reference028(n01_root, device='cpu')
result = response.sample(
    source_waveforms=source_carrier384_V,  # float64 [P,384]，已准备的清洁 SOURCE
    n_draws=2,
    seed=2026101728,
    detector_config=response.detector_config,
    noise_hypothesis='fast',              # fast / slow / common_level，单选
    model_seed=0,                         # 0 或 1；没有默认部署 seed
    parent_ids=source_parent_ids,
    shape_enabled=True,
    charge_enabled=True,
    return_contributions=True,
)
```

`detector_config` 是 loader 绑定的配置身份，包含 CP/源码/统计/观察合同身份；调用边界只接受该绑定。它不接受未经拟合的任意 detector condition。`source_waveforms` 名称保留主计划目标，但此版明确只接受 CPU float64 tensor `[P,384]`，不接受原480、ADC code、256裁窗或任意时移波形。调用者从已保存 SOURCE carrier384 转成 tensor。原480到384的既有规则是 `data008.transform_domain` 取原电压前384点、走 clean-values 路径；此版不包装该转换，也不读原480。

输入必须保留原 V 单位、1 ns 软件网格及原 prefix100；模型内部继续去 prefix 均值，并要求搜索 `[112:144)` 的 first-argmax 为126、幅度和 canonical Q 合法。沿用内部原语的现有失败行为，不重复其检查，不删行、不补零、不夹改输入。API 只检查外部参数、shape/dtype/parent 长度和配置身份；错误直接传播。

输出保持原观察结果的平坦 `[P*D,256]` tensor 和字段名（`waveform256/window_V`、`Q/tail/PSD_raw/PSD`、`baseline100`、peak/crop 与 flags），再附父引用、`row_parent=[0,0,1,1]` / `draw=[0,1,0,1]`、模型/采样流身份和研究适用卡。父 major、draw minor；父 ID 不送入网络。可选贡献保留原 tensor 引用：base A、M0 reference、shape、charge、完整 electronics、preADC/ADC carrier 和 crop indices。charge 关闭时不要为不存在的 charge-on 字段造零值。PSD 的 undefined/guarded 行和 NaN 按原观察结果保留；公开 JSON 只写计数/身份，不序列化逐行波形或 NaN。

原观察 flags 实际为 `peak_at_search_boundary`、`global_peak_outside_search`、`Q_nonpositive`、`PSD_undefined`、`PSD_guarded`、`ADC_rail_hit`、`pre_ADC_out_of_range`。当前原语没有直接导出 peak-tie/halfgrid-tie flag；此步不增加新的形态或 censor 体系。

## 2. 真实依赖与 CP 构造顺序

静态追踪到的最小项目导入集合为 `n01/__init__.py`、`model.py`、`model008.py`、`model010.py`、`model014.py`、`model016.py`、`electronics_v3.py`、`extension006.py`、`observation008.py` 共9份。第三方直接依赖为 torch、numpy；版本和实际运行闭包要在原生例中记录。这里没有把原016的37项训练导入闭包冒充已经裁剪成功的 runtime。

旧链的源码依据：`run_wp3_015.prepare_numeric015:211-245`、`run_wp3_016.run016:526-537`、`Engine016.checkpoint:325-334`、`model016:60-70`。新 loader 取下列必要部分，不调用这些旧 main/loader：

1. 从027 bindings取得六个 CP 的原始路径/bytes/SHA和统计来源；实际打开前核对文件身份。读绑定的 `FIT_STATISTICS008.json`、`FIT_DATA_STATISTICS.json`、`window008_config.json` 与观察合同，核 A 的原统计/config/005绑定。`FIT_STATISTICS008.model` 用于 A 构造，`loss.q_floor=0.0005319378913383611` 用于硬观察；`q_reference=3.7103998046755806` 用于身份核对。014 integral statistics、训练数据、source information 和 loss 均非推理构造依赖，不加载或重新拟合。
2. `N01Response008(stats['model'], 'deterministic')` → 严格载入 A `model` → 按原 CP `stage` set_stage → eval/freeze → `.to(device)`。这些步骤全部在016对象捕获 A 的 layout/data_ptr/version 之前完成。
3. `ObservedBaselineNoiseV3(noise_cp['initialization'])` → 严格载入005 `model` → eval/freeze → `.to(device)` → `FrozenObservedNoise008(noise005)`。extension 是原 `ConditionalExtension008` 的实际类，不重写条件律。005初始化不消耗 RNG；extension 的三个100点 KBB和三个284点 residual Cholesky均为确定性构造。
4. 对 model_seed=0/1 各自读取015 M0 `new_head` 与016 Q1 `theta_a/theta_b`。核 `initial_seed`、M0 reference CP SHA、M0 head digest、冻结状态身份、final/complete 及 M0 `mean_theta==0`；禁止交叉拼装。用 Q1 CP 保存的 `charge_initialization` 传给 `FrozenM0ChargeResponse016`，然后原位拷入保存 theta；不调用 `calibration_statistics016` 或 `Runtime016.calibrate_charge`。
5. 在最终设备上完成所有构造/载入后冻结 theta、eval；此后不要整体 `.to(...)`，避免破坏已捕获的 data_ptr/version。用 `torch.no_grad()` 包住采样和构造期参数拷贝。不要用 `torch.inference_mode()` 构造这些依赖 `_version` 的对象。训练 CP 中的 optimizer/sampler/streams 会随 CP 对象反序列化，但不实例化、不恢复、不继续使用。

可复用旧 `tensor_state_digest010` 核新头和冻结前后 tensor；原 `response008.state_digest` 算法的历史绑定可核 metadata，实际二进制通过原 SHA+strict load 确认。不要复制该 hash 算法或为了取一个辅助函数导入整个 `response008` 训练依赖。源码与原始 CP pin 的完整事实见 `DESIGN_FACTS.json`。

模型0：M0 `27ad34f2…` + Q1 `557ec1bd…`，初始化 seed `2026101540`；模型1：M0 `8a14a66b…` + Q1 `03603cb5…`，初始化 seed `2026101640`。共享 A `2096523d…`、005 `1e45d336…`。六份 CP 总大小24,420,992 bytes；027的身份是历史 metadata，本次原生 loader 才会实际读/rehash/load。

## 3. 随机流和单假设硬输出

每次 `sample` 都是 stateless：以 `sha256('n01/wp6-028/v1/{seed}/{namespace}')` 的前8字节 little-endian、再取非负63位，分别建立 CPU generator；namespace 固定为 `shape`、`noise005`、`extension`、`charge`。模型 seed 不放入采样 namespace，因此两个模型的固定例使用共同随机数，但 CP 身份始终分别返回。不得读或继续旧训练 stream 状态。

先按完整 `R=P*n_draws` 生成整个 latent 库，再将 NN 前向按固定64行分块。请求顺序和 dtype 固定：shape `[R,8]` float32；005按原字典顺序 `baseline_gaussian [R,1,100]`、`baseline_scale [R,1,1]` float32，共用其 generator；extension `[R,1,284]` float32；charge `[R]` float64。即每次5个 randn 请求、每行394个 normal 标量。即使 shape/charge 关闭，也保留相同请求以稳定共同随机数。原 A 的 phys/local_white 在 `_baseline_once` 内仍是确定性 zeros，不新增抽样。

同完整 batch/顺序、draw数、seed、开关、执行环境与固定块计划的重复调用应逐字节一致。连续调用相同 seed 会重置并重放该随机库，不是新独立 draw；需要新库就明确换 seed。此版没有持续 stream 状态/断点续抽接口，不承诺拼接不同调用、改变 draw数或 NN块大小后逐位等价。先生成全库保证 NN分块不改变 latent 请求；浮点 NN 的块大小仍记入重放条件。

一次调用只处理一个 `noise_hypothesis`：复用 `sample_prefix` 一次，再调用原 `extend_prefix` 一次。它们正是 `fixed_noises` 的内核；不用 `fixed_noises` 展开未请求的另两假设。同 seed+同 R 的三次调用共享完全相同的005 prefix、scale、epsilon。005与extension存在原条件均值/scale耦合，“分别命名随机流”不表示统计独立。

信号走 `model016.random_forward(..., enabled=shape_enabled, charge_enabled=charge_enabled)`。charge-off 的原 `apply_charge:110-113` 返回原 M0 signal tensor 对象（保留 signed zero），不能改成 `+0`。shape-off 沿原014/010零系数路径；其实际形状头仍前向并计费。mean、timewarp新增项、local-signal扩展和噪声开关均不新增。

preADC=`signal_pre+noise_pre` 后只调用原 `observe_generated_carrier(..., ste=False, q_floor=bound_floor)` 一次：14bit hard ADC → 原 measured prefix100均值 → `[112:144)` first-argmax → `[p-16:p+240)` gather → 256点 Q、tail从46起、原PSD与flags。不把训练代理当硬输出。

## 4. 固定原生实例和精确预算

优先在 ROOT 已恢复连通的既有服务器 `rylan-GTi15 / 100.82.231.102`、原冻结 Python 环境运行 CPU；本地源与收集回执位于 E 盘。执行路径、source SHA、解释器/框架版本由 ROOT 注册，不静默切换 host/environment。所有新增本地缓存/TMP均在 E。执行前、torch/n01 import之前落实 CPU1/线程1、RLIMIT_AS=8GiB、RLIMIT_CPU soft590/hard600；外部wall600秒。RSS4GiB与artifacts64MiB是终态验收，collection32MiB；不称运行中RSS watchdog。ROOT复用既有 supervisor。

输入固定为 `DATASET008.npz` 的原行 `[0,1]`，要求角色为TRAIN；每父2draw。遇输入/模型失效记录并失败，不换父、不挑Q区。一次NPZ打开仅加载三个成员各一次：`source_carrier384`、`source_ids`、`source_role`。NPZ的成员读取会完整解压6593行，不把最后切片2行说成只读2行。历史 header inventory（不是本次NPZ decode）给出：

| 成员 | dtype/shape | 标量数 | payload bytes |
| --- | --- | ---: | ---: |
| source_carrier384 | f8 [6593,384] | 2,531,712 | 20,253,696 |
| source_ids | U75 [6593] | 6,593 | 1,977,900 |
| source_role | U5 [6593] | 6,593 | 131,860 |
| 合计 | 1 open / 3 member loads | 2,544,898 | 22,363,456 |

三个NPY ZIP成员未压缩合计22,363,840 bytes（含header），压缩payload合计7,598,009 bytes。完整NPZ SHA读取66,440,265个文件字节不进行数值解码。只将固定2行送NN；原480、REAL、DEV numeric 和其他 NPZ 成员均不加载。已有 header 身份和完整成员形状见 `DESIGN_FACTS.json`。

固定采样 seed `2026101728`，固定调用顺序为每模型seed：`Q1 fast`、`Q1 slow`、`Q1 common_level`、`fast exact replay`、`fast chargeoff`、`fast shapeoff`；先模型0后模型1。每调用4行，共12次。没有额外 reference NN/ADC 重放：

| 实际操作 | 精确预算 |
| --- | ---: |
| CP load / 唯一路径 | 6 / 6（A、005各1；两个M0、两个Q1各1） |
| response / A / 实际形状头 rows | 48 / 48 / 48 |
| 005 / 单hyp extension / hard观察 rows | 48 / 48 / 48 |
| 各模块batch调用数（R=4） | A/head/005/extension/观察各12 |
| charge-on module rows | 40（两个chargeoff调用共8行关闭） |
| shape-enabled rows / 仍实际执行head rows | 40 / 48 |
| hard ADC carrier samples | 18,432 = 48×384 |
| sampling randn 请求 / normal scalars | 60 / 18,912 |
| shape / 005 gaussian / 005 scale / extension f32 scalars | 384 / 4,800 / 48 / 13,632 |
| charge f64 scalars | 48 |
| sample generator创建 / latent payload bytes | 48 / 75,840 |
| AD、G/D/noise更新、fit、CDF、loss、训练trial、旧suite | 全0 |

构造期 RNG 另计，不能写成0：A 有57个 Conv/Linear，各 weight/bias随机初始化一次，共114个uniform请求、1,473,092个f32标量。两个临时shape Conv各2个uniform请求/1,288标量，再各1个显式normal请求/1,280标量。合计118个uniform请求/1,475,668标量 + 2个normal请求/2,560标量；构造期共120请求/1,478,228标量。005、extension、theta构造RNG均0。A分项与本机torch reset源码依据列在JSON；独审应确认执行环境仍使用同一标准reset布局。整个loader使用CPU `fork_rng(devices=[])` 保留外部全局状态；临时权重最终被CP覆盖。抽样60请求/18,912 normal与构造合计180请求/1,497,140请求标量；这是分布API请求量，不是底层随机引擎word数。

## 5. 验收成功标准与停止范围

一次小例应确认：六份CP实际身份及严格载入、9份冻结源码的实际依赖、双seed无交叉；全部参数frozen、grad为空、全局CPU RNG状态未变；父/draw映射和输入字节未变；全部48行及flags保留；硬观察返回的ADC prefix、baseline均值、first-argmax、crop indices/gather和Q/tail/PSD彼此一致。检查直接用已返回tensor的代数关系，不再调用ADC或模型做第二份参考。

fast replay与同seed同batch主调用逐字节一致（包括NaN的字节）；同seed跨hyp的四流及005 prefix/scale/epsilon一致。chargeoff的 `signal_pre is M0_reference_signal_pre_V`，且与主Q1输出中保存的M0 reference字节一致；shapeoff的M0 reference与同次base A字节一致，charge仍开启。无要求两个draw一定不同、经验矩接近理论、或某科学指标通过；2父×2draw仅为软件接口例。

失败则保留本次结果/成本并停止，由 ROOT 处理具名必要修复，不自动换CP、父、seed、假设、环境或重跑旧suite。输出的 CP/波形/逐父数组留在既有私有执行路径/E盘，不上传GitHub；公开代码/合同/汇总身份与失败卡即可。原生0和这些断言只建立工程可用性，单采集条件响应、人口/核可识别性、绝对容忍度、独立采集泛化、完整WP5/WP6与最终模型选择仍未完成。
