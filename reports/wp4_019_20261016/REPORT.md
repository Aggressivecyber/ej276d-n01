# WP4/019：两项T1无配对试验已正确启动

019已按018资格单独前瞻登记并完成Sol Max源码/合同/ops独审。2026-10-16 04:38:28（新加坡）的同一实际快照确认T1_unpaired_seed0训练正在运行，首步两个theta的真实梯度与参数更新均非零。持久完成G/AD均64；019前累计G下界183667，本次只加新差额64，项目累计下界183731。在途operation3324当时unknown，未记作完成。未取得019终点或无配对资格。

监督PID38792、worker38806，boot `06177e53-90a1-4d8c-a942-03b487a0684d`，startticks1069418/1069461；当前父子关系、实际命令/cwd、12核worker亲和和原始SHA绑定均通过。快照RSS2.13175GiB、GPU0.66992GiB、输出2,077,014B、elapsed34.10s，资源健康。首步梯度为−0.02124247/−0.00542806，实际theta增量为0.0099999953/0.0099999816。

首步/INIT均证实005 CP数值加载1次，A/M0 CP及新前向0，原A/B/Q只读缓存及冻结005身份匹配。继承格式中的A/M0 frozen表述只指缓存来源与字节未变；本次未再执行A/M0。005后续确有新噪声生成，不能将019称为NN=0。

## 固定预算和下一次苏醒

源码提交`5585b11b79d975989221da287b031bb001cef930`；CFG SHA`e00ad2564dbf00bff0e210fc6b5425d8b7158662a07d7a5acca634b11c3e13d3`。38源/控制成员与Git原字节一致；37精确输入共1,141,808,275字节。原6593父/5274fit/1319heldout、物理权重、ID及角色保留。旧源码、随机namespace、训练257维U-score和018评分门保持。

只运行T1_unpaired_seed0/1，每个2000G/AD，总4000，消耗原WP4八teacher槽中的两个；017已完成两paired，019完成后最多共四槽。复用teacher/initial评价，只生成两个final，共843,904新评价信号行。全包005行9,035,904，正常CP4份，失败CP最多1份；新257维评价距离162,029,568、O19距离405,073,920。全批CPU12/GPU10GiB/RSS8GiB/wall21600s/output5120MiB；trial7200s/1536MiB/shared1024MiB，终点选收≤128MiB。无自动重启、补采样或追加试验。

已实际保存并核对n01于2026-10-16 05:30（新加坡）单次苏醒。该时间据首64步约34s、余训练及评价余量确定；启动快照后不再查询远端。下次先做一次真实状态核对；仍运行则按实际阶段另存一次合理未来苏醒并休息。退出则先核native/FAILED/COMPLETE、资源和持久计数，再按完整inventory/SHA选回必要证据。原数组、完整日志、CP和负结果保留，门和数值读回完成后才下019结论。

## 科学边界

018旧2688格保持2616pass+72pending，017 stage2仍关闭；前瞻018新2544格全部通过并取得本次登记资格。019同样要求每trial1272、共2544完整主门，failed/pending/MC_unresolved不算pass。实际teacher局部Q置换的未越门结果未建立关联缺陷检出，018有限库/自归一和窗外rise不可识别的限制继续保留。

本试验只检验固定正确A/B/005下两个电荷参数的指定无配对方法和预算。不自动进入T2、均值候选或REAL长训练；015 time-warp仍为草案，WP3六槽已耗尽，不能挪用WP2余两槽。科学状态保持`response_match_not_established`，六工作包尚未全部完成。

[实际启动回执](../../receipts/wp4_019_20261016/EXECUTION_STARTED.json) · [首更新与健康验证](../../receipts/wp4_019_20261016/start_verification_01/START_VALIDATION.json) · [单次苏醒](../../receipts/wp4_019_20261016/FOLLOWUP_RECEIPT.json) · [冻结合同](../../docs/WP4_019_CONFIGURATION.md) · [独审](../../receipts/wp4_019_20261016/independent_01/REVIEW.md) · [018完成报告](../wp4_018_20261016/REPORT.md)
