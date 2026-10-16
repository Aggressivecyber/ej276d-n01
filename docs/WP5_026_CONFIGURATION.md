# 026：冻结M0的SOURCE组成敏感性

026只实施[前瞻scope](../receipts/wp5_026_20261016/registration_01/PROSPECTIVE_SCOPE.json)中的18个固定情景。015 seed0 M0由024 zero、025保存observable表示；响应核、6593父、每库4draw及fast/slow/common_level保持不变。Q1仍是原研究参考，本次不重新评价或晋级M0/Q1。

## 本地入口与冻结

```text
python -B scripts/run_wp5_026.py --config E:/GAN/research/N01/wp5_026_config.json --config-sha256 <frozen SHA256> --output E:/GAN/research/N01/runs/wp5_026_20261016/analysis_01
```

`settings026()`、`budget_caps026(p)`、`validate_settings026(p, *, frozen=True, driver_source=None)`顶层只有标准库，供ROOT冻结。实际科学import闭包为`run_wp5_026.py`与标准库`wp1_ops.py`，数值库仅NumPy；不import021/025 driver或Torch。ROOT另绑定本文、输入manifest、025结案/独审、013审计、持续授权与026 scope原字节。draft不能执行，输入/source/contract完整RAW SHA和bytes在数值import之前核验；输出必须全新且仅E/D。

九个已存在NPZ共27790366B：六份025 zero observable、025 REAL observable、原SAVED024_AXES、013 PARENT_INFORMATION。入口只读取登记所需的标量/flags/IDs/groups/父轴，以及013的source_id、role、h_visible，不读原SOURCE/REAL/DEV波形、CP或hard码。013全部8192父中取6593 TRAIN，以完整字符串source_id与保存父轴一一连接，不假设位置一致。原axes按完整RAW字节复制，physical weights保存且校准状态继续unknown。

## 固定比较测度

三情景为`nominal=1`、`low_h=1-.20*(2h-1)`、`high_h=1+.20*(2h-1)`，h沿013已验证的`.15*edep_hadron/(edep_ee+.15*edep_hadron)`。比较权重为正、raw范围[.8,1.2]，按情景在原父上归一，每父质量等分给4draw；两bank不pool。原physical weights不与新比较权重相乘，也不被替代。

`.20`表示比较权重tilt幅度，不表示h或中子份额改变20%。h是SOURCE沉积组成代理，Q不是标定入射中子能量；本次不拟合真实源人口或真实响应，不从REAL选择权重情景。

每格报告8标量的均值/总体方差和9个原flags的全行加权率。每个标量各用其有限mask归一，保存原行/父分母、有效行/父数、有效原始比较质量及比例。ESS先把有效draw质量聚到原SOURCE父，再算`mass²/sum(parent_mass²)`；不会把4draw算成4个独立父。全未定义子群的数值与ESS为null并标undefined。

REAL只使用025保存的21047 TRAIN行；其有限行测度均匀归一，ESS明确标为saved-row measure，不能解释成独立SOURCE父或422次采集。REAL groups原样保存，无重采样或新增区间。signed均值/方差/flag差只是描述，不将013的W1容差T套到这些新摘要。

## 72个有限经验分解

Q/front/tail/peak全定义标量，每个库先预计算一次6593父的`m_i=mean4(z)`和`v_i=mean4((z-m_i)^2)`，within分母4。18格×4标量计算`B=sum alpha_i*(m_i-globalmean)^2`、`W=sum alpha_i*v_i`，与原行测度`alpha_i/4`的直接总体方差比较。复用021的`rtol=1e-12`只核数值恒等式，不是科学接受阈值；不使用分母3的样本方差替代within。

分解在有限保存4draw上exact；父均值自身带MC误差，不能把B/W宣称为真实核内/核间方差或可辨识性证明。不同bank、模型seed、父人口和采集组仍是不同证据层。

## 成本、资源与输出

精确科学计数沿scope：9个NPZ一次解码、8192来源metadata、6593父连接、6份fake158232行、REAL21047行、3套19779父权重、18格、144个fake/8个REAL标量摘要、162个fake/9个REAL flag摘要、72分解及158232个native4 parent-scalar moment行。新增NN/CP/AD/G/optimizer/RNG/ADC/trial/参数点/拟合/CDF/quantile/pair距离/原波形读取全为0。

复用021的原生Windows边界：最低可用单CPU affinity、BLAS/OMP1、输出内E盘runtime_cache，ROOT外层原生subprocess timeout300秒。RSS工作集cap1GiB、输出64MiB。GetProcessMemoryInfo的current/peak working set读数止于最后结果serialization前，保留测量阶段，不声称外层全程RSS测量；不新增持续memory-kill或监督循环。

`Store.call`逐操作记录entered/completed/inflight，完整计数须与scope相等。失败写FAILED并保留traceback向外退出，原失败不重写、不自动重试。输出包括原axes RAW副本、SOURCE连接/三套独立比较权重、REAL identities/target摘要、六库native4父矩与每指标finite mask、18格ANALYSIS、资源、durable journal、inventory和COMPLETE/FAILED。全部统计保留，无选优情景或新pass/fail响应门。

唯一人工语义检查见[run_finite_measure_fixture.py](../receipts/wp5_026_20261016/engineering_01/run_finite_measure_fixture.py)：两父×4draw、权重1:3，检查已知均值9、within5、between12、total17；部分mask核对定义质量、parent ESS与独立归一，空mask保留null。该检查调用实际026数学函数，无私有数据或随机数；人工24个scalar-row evaluations独立记账。

026只完成WP5组成stress准备。不能据此回填025通过、识别真实人口、宣称full-wave稳健、开启新theta/timewarp/T2/真实长训，或把WP5/6写成完成。
