# WP4 018：保存硬输出的截尾感知评价（零训练）

版本 `wp4_018_20261016`。本合同落实用户对017审核的“按照这个稳步推进”，只覆盖下一阶段A。017源码、配置、manifest、报告、门及pending结论不可改写。未完成的六项教师训练不在本次作业内。

## 输入与执行边界

只读017已经保存的4个kernel：T1、共享initial、T1_paired_seed0/1的final。每kernel保留6593个父、2个原生bank、每bank32draw、三个噪声假设；组合64draw不冒充第三个独立库。拟合/留出仍为5274/1319，原fit-Q三分组及nominal/low/high比较权重不变。原物理权重另存，不拿比较权重替代。

`WP4_018_SAVED_INPUTS.json`逐文件冻结75项（68个数值文件、7个JSON），压缩输入共3,364,811,496字节，来源是既有017完整远端清单；执行前须核当前字节数和SHA。远端原保存根为`/home/rylan/projects/ej276d-n01/wp4_017_20261015`。允许的文件模式严格限于：

- `runs/primary/artifacts/{T1,initial,T1_paired_seed0/evaluation,T1_paired_seed1/evaluation}/bank{0,1}_{fast,slow,common_level}_{HARD_ROWS,REPORT}.npz`及每kernel的两个`bank{0,1}_BASELINE100.npz`、`EVALUATION.json`。
- `T1/PIT_UNIFORMS.npz`；`cache/{Q.npy,PARENT_SPLIT.npz,ORIGINAL_PHYSICAL_WEIGHTS.npz,CACHE_MANIFEST.json}`。
- 两个paired trial的`COMPARISON.{json,npz}`及各自evaluation的三个`PARENT_COMPARISON_{hypothesis}.npz`。

模式仅供解释，真正白名单以75条精确路径、SHA和字节数为准。PARENT_SPLIT只解码fit_rows/test_rows/source_ids；原SOURCE波形及其诊断坐标不解码。PIT只复用保存uniforms，不读取/恢复随机状态。CP、A/B缓存、SOURCE/REAL/DEV原文件均不读；NN、G、AD、CP加载及新增RNG请求全部为0。保存数组的字节解码不称为新的模型前向。

输出只写新018目录。全部新代码、人工检查、合同与本地回执在E/D；远端输出在018新owned目录。root在源代码与合同独审后负责登记、冻结和启动，本driver没有SSH/部署/重启功能。

## 同一个256点观测映射

14bit ADC、384carrier、实际prefix100、first-argmax搜索[112:144]、`[peak−16:peak+240]`的256点输出、tail46和1软件ns完全沿用。使用原013 codec的code256、实际baseline均值和已有IEEE XOR逐位重建Y。不扩大输出窗，不重新做ADC/寻峰/裁窗。

018的vectorized crossings逐项保持017的定义：左侧从紧邻选中峰向左找最近的上穿越，右侧向右找最近的下穿越；阈值为选中峰电压的10/50/90%，端点线性插值，水平等值段规则不变。rise=t90_left−t10_left，FWHM=t50_right−t50_left。人工检查将与旧逐行实现核对，包括非单调、水平段和非正峰。

每个原始槽位恰有三态之一：`measurable`（rise有限）、`prepeak_censored`（原017的峰前截尾描述标记）、`other_undefined`（包括非正峰与其他不可定义）。原各细分flags仍保留并按原全64槽分母评分；三态不吞并其原因记录。截尾是当前窗的描述，不假定独立删失。

保存真实可测rise、可见left10/50/90及right50交叉（相对峰坐标），其他数值为NaN并有显式defined mask。只有正峰、left10不可见、首点严格高于10%峰且left90可见、原定义属于prepeak时，记录`rise_lower_bound_if_left10_exists_ns=t90_crop`。这个下界的含义严格为：**若**未见左扩展确有与同一nearest-left定义一致的10%交叉，则rise大于该数值；当前窗不能证明该交叉存在，也不能恢复窗外rise。其余情形不编造下界。

## O19联合可观察表示与前瞻新门

O19依次为三态onehot（3维）、下述6个时间值的defined mask（6维）、mask乘时间值（6维）、Q/T_Q、peak/T_peak、mask_FWHM×FWHM/T_FWHM、mask_FWHM（4维）。六个时间值是rise、left10、left50、left90、right50、上述有条件下界；时间值均除既有时间门的1软件ns。状态与mask尺度为1。三项标量T分别取013已冻结的对应hypothesis的Q_Vns、peak_voltage_V、FWHM_ns，数值保存在018配置，不做任何新拟合。缺值只在嵌入中编码0且mask=0；0不解释为真实rise。原始时间文件仍存NaN+mask。

这是一对预测/目标共同使用的确定性可观察变换。联合energy距离能观察到状态、可见时间、Q/peak/FWHM及其依赖；它不能区分映射到同一可观察分布的不同窗外机制。8个原完整标量的协方差另外评分，辅助energy不取代该协方差门。

逐父用完整64draw计算无偏half-energy discrepancy：

`D_U = mean_{i,j}||S_i−T_j|| − sum_{i≠j}||S_i−S_j||/[2m(m−1)] − sum_{i≠j}||T_i−T_j||/[2m(m−1)]`，m=64。

teacher两原生独立32bank按相同公式、m=32给阴性对照。所有有限负估计原样保留。相同数组S=T时，U统计量一般为负，必须等于`−self_U/m`；其V形式才为0，恒等测试不把U强制设0。

按原父权重求组均值和原017的HC1 parent-cluster SE（不将64draw当64个独立父）。新实际容忍δ=.10、新MC半宽限=.05均是**本次前瞻选择的新分数单位**，不是013对O19的扰动标定、不是通用1ns误差上界、不是硬件容忍度。确定性单坐标相差一个归一化单位时对应距离1，但联合总体距离与某项物理误差没有通用换算。

- teacher阴性对照仅当`abs(mean D_null)+1.959963984540054×SE_null <= .05`才resolved；否则该组新门pending。
- resolved后，候选95%下界>.10为fail；否则候选95%半宽>.05为pending；半宽达标且95%上界<=.10才pass；其余pending。
- 阈值固定，不按有限teacher差或学生分数放宽，不追加draw补救。teacher split与父cluster区间是固定库的经验精度筛选，不是条件64draw oracle误差保证、重训不确定性或总体同时覆盖保证。

全部原父/64槽参与O19，其表示不要求可测rise达到95%，因为截尾状态本身是观测值；不得靠删父获得支持。原95%父数、95%权重、最少64父、ESS≥原父数一半、每bank≥24且合计≥48的规则完全保留用于原有限标量门及cov8。

## 旧主门复核与018资格

从hard256及实际baseline重新计算全部9标量、原flags核对、原native32波形矩、derived64矩；以保存uniforms重算PIT。不拿REPORT中的旧矩或PARENT_COMPARISON中的energy作评分输入。旧257维fullwave energy严格沿用`[Y/(16Δ),sum(Y)/(256Δ)]`、Δ=1/16383，以NumPy直接欧氏距离重算原两bank32draw ES_U及teacher反向平均参照。旧PARENT_COMPARISON仅作全部逐父energy/coverage数值核对。

旧017全部1344格/trial、共2688格（以及其完整签名、支持分母、原指数、状态与数组数值）都重算对照producer。原共享stats017用于其既定门，距离内核改用直接NumPy差平方；这是保存数值的全覆盖重放，不能宣称独立另写了一套全部统计理论实现。独审检查关键公式和语义。浮点复核容差：energy rtol=1e−10/atol=1e−9；其他rtol=5e−12/atol=5e−14；整数、布尔、状态、支持及签名精确一致。超差/漏格为执行复核失败，不能自行放宽。

018保留原wave均值/方差、解析电荷方差、fullwave energy、peak_index/FWHM时间、除rise以外的PIT、全部18个missing/censor原因的1%门。cov9改为其余8维的同口径cov8：teacher64标准差/原floors逐父归一，组内条件协方差相对Frobenius≤.15，teacher32 split≤.075方resolved；不混入父间均值散布。这个归一与split是沿用的经验规则，不给出独立oracle协方差保证。旧rise时间/PIT及cov9完整保留为选择诊断并报告原支持范围，不能将旧pending改为pass。

018精确主门为1272格/trial、共2544格：每trial12个解析方差格；每36个hypothesis×group×weight单元含wave均值/方差2、cov8 1、fullwave energy1、peak_index/FWHM时间2、5特征×2覆盖率10、18原因1%门18、O19 energy1，共35格。代码须核完整签名，不能只数producer格数。两trial任一fail/pending均不取得018的T1 paired资格。即使全通过，这次作业也不执行其余六项训练，不等于完整响应学习、窗外rise恢复或REAL/硬件验收。

## 必要验证与反例

仅运行新表示所需人工语义检查，不复跑旧工程套件。四个固定、无需RNG的强反例均须在其预先指定效应下独立检出（O19 D_U>.10，且相对恒等U值增加>.10）；它们只证明这些效应的检出，不能代表任意小误差的功效。

1. 截尾率提高25个百分点。
2. 只在截尾组内固定反序置换Q，保持原整体和截尾组Q边际、状态边际以及完整组逐位不变，改变Q与截尾组可见时间/界的耦合。它保留state–Q依赖，不能据此声称检出了state–Q依赖。
3. 可测rise增加4ns，状态、Q、peak和FWHM不变。
4. 极小的跨状态Q重排：保持整体Q与状态边际，改变state–Q关联；与#2分开报告。

实际T1的固定截尾组Q重排另作敏感性诊断：按每父截尾槽的可见t90与槽号排序，将该组原Q排序后反序分配。完整组不变、Q与状态边际精确保留；没有足够不同槽位时原样保留并报告不可辨效应。记录被改父数、槽数、Q改变幅度、组内Q–t90协方差改变及O19分数。实际效应未预先保证超过δ，不把未检出改成通过，也不将其新增为必须越δ的教师资格门。

## 预算与交付

一次CPU-only保存重放：12线程；address-space/RSS预算6GiB；wall≤7200s；CPU软/硬≤21590/21600s；新输出≤1536MiB；预计42文件；选收汇总≤128MiB。75个白名单文件仅核SHA及解码所列成员；原文件不修改。主要距离预算为旧257维fullwave `6593×3×15×32²=303,805,440`个有序距离；新O19包括完整64、teacher32阴性对照、恒等和T1固定置换，共`6593×3×(10×64²+3×32²)=870,908,928`个有序距离，执行按该精确值核对。保存hard256解码共5,063,424行，新模型硬观察调用0。人工toy预算单独记录，不冒充私有重放。

产物为12份完整原父O19/时间/状态数组、原身份与物理权重绑定、旧2688格全覆盖数值复核、两trial018主门及rise选择诊断、teacher阴性/恒等检查、实际固定置换效应、输入/输出SHA清单、native退出与资源回执。完整大数组留远端；root只收必要汇总并安排一次未来苏醒，长计算正确启动后不持续轮询。

方法范围参考：[Pic et al. (2025), Aggregation and transformation of proper scoring rules](https://ascmo.copernicus.org/articles/11/23/2025/)。这里只引用对共同确定性变换后的可观察分布评分的思路；不引入独立删失假设，也不由proper一词推导窗外参数可识别性。
