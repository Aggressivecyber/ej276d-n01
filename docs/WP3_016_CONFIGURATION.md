# WP3 / 016：冻结 M0 后的信号电荷涨落短试

本次检验一个问题：给015各seed的M0 final增加两个全局电荷方差参数，能否改善相同硬观察口径下的开发误差。015的均值适配负结果保留，均值分支关闭。原A、005、M0形状头、C1积分尺度、数据角色、硬ADC/first-argmax/裁窗和013数字T保持冻结。两个seed各2000G，最多4000G/AD；占用WP3剩余两次短试，使WP3达到6/6、总搜索达到10/12。一次有限比较结束后推进主计划，不为近阈值追加短试。

015的零canonical-Q形状头在给定输入的preADC域没有新的信号电荷涨落自由度。无配对总体Q宽度无法识别真实条件方差；本次拟合参数只表示当前数据和人口假设下的有效响应。峰高、rise、FWHM、基线DC的残差分别报告。

令Q_mu为实际冻结A输出的canonical_charge008，要求全部合法SOURCE TRAIN输入均严格正。电荷分布为s²=log1p(a/Q_mu+b)，q_sig=Q_mu exp(s z_q−s²/2)，其中z_q为独立显式标准高斯，a=q_reference softplus(theta_a)、b=softplus(theta_b)，q_reference固定为3.7103998046755806 V软件ns。理论条件均值为Q_mu、方差为a Q_mu+b Q_mu²；a的单位为V软件ns，b无量纲。不给方差增加电子学常数项，不用分母floor改变这一规律。

组装采用原M0信号加A·expm1(s z_q−s²/2)。B(x)z_shape仍按原A的幅度生成，不乘本次q_sig。z_q与形状、005、长程扩展流独立。关闭电荷分支时直接返回原M0信号字节，同时关闭随机项和均值补偿；模型内部不创建随机数。prefix100保持严格0。均值保护针对preADC条件期望，硬观察后的期望由完整评价报告。

初始化只作一次覆盖全部6593 SOURCE TRAIN父事件的冻结A前向，不调用形状头、005、ADC、新随机或AD。令S0=mean(A²)、Sminus=mean(A²/Q_mu)，均值分母包括全部父事件和384点。以δ=1 ADC LSB设a0=δ²/(2 Sminus)、b0=δ²/(2 S0)，用稳定inverse-softplus写入两个初始θ。实际A输出决定的理论增量RMS为δ，正电荷范围、系数、RMS和初始评价库的实现RMS全部保存；不按有限MC偏差重新调初值。

仅theta_a/theta_b进入新的AdamW，lr=.01、weight_decay=0、clip5。每步512 SOURCE父事件×2draw和1024 REAL行，保留全部6593/21047 TRAIN覆盖证据。目标为wave+2×C1积分分布+.01×(原冻结形状regular+mean(a/Q_mu+b)+平均saturation penalty)。每步一次backward，无额外AD探针；1000/2000G保存checkpoint和明确随机流状态。原shape/noise/extension和投影前缀沿用015，新增charge流独立登记。时间、局部信号噪声、人口和GAN分支关闭。

每seed初始评价共享一次模型输出和同一噪声，分别保存A、M0电荷关闭、初始电荷开启；结束后保存final。每次全部6593父、每库4draw、两库、fast/slow/common_level三假设，对全部21047 REAL TRAIN。全人口、输出Q条件、SOURCE输入趋势、同父重复及全部flags/低Q/undefined PSD保留。原baseline100另存，不以脉冲首100点替代。增加小型preADC电荷语义和矩摘要，完整raw保留服务器。

研究参考替换规则：逐seed/h先平均两库各自距离，不能改成pool8距离。每个必需且正值有定义T组件在同一ALL6中须同时满足final−A≤T和final−M0≤T；必需组件缺失或undefined使候选pending、不eligible，零/undefined T继续未决。合格候选仅在12个final wave距离均值比对应M0低超过1e-12时替换研究参考；否则保留M0并关闭电荷分支。初始电荷开启的距离另报。ALL6相对A至少2%的用途门、绝对distance/T和科学确认分别报告，权重不据此自动晋级。

精确设计计数：训练NN行4096000；两seed的initial/final、两库各4draw共210976模型行。head、005、charge模块及charge标量draw各4306976行，A另加初始化6593行，共4313569。硬ADC/gather各13553856行。CP4为A、005、两seed的M0薄checkpoint；optimizer2，G/step/backward/AD各4000，charge初始化fit1。积分尺度fit、额外probe、mean adapter、D/noise/pop/C2ST/DEV模型行均0。最终冻结配置须与实现推导逐项一致。

单trial wall7200秒/输出1536MiB，整批wall14400秒/输出3072MiB，CPU12、GPU10GiB、job-tree RSS8GiB；选择性回收最多256MiB，raw留服务器。新隔离目录，只管理本项目拥有的作业树，禁止自动重启和覆盖既有结果。仅验证首次真实参数更新和同一快照健康，设置同一n01一次未来苏醒后休息。当前本文为实现设计，实际启动以冻结源码、配置、原生回执和jobs为准。

单场实测、旧DEV曝光和其它历史pool未知的边界继续保留。科学状态为response_match_not_established，完整六工作包尚待继续。
