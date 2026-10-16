# 022：教师生成完成，分组评分因错误导入退出

022原生退出1，未生成完整36格或三主格判定，科学结论未知。故障在contrast_cell022从stats017导入Z95；常数实际定义于stats018。错误发生在首个fast父事件分数保存后、分组聚合开始时，不是训练或资源故障。

完整固定T2教师核已保存并核对22份输出数组的清单/SHA：6593父×2bank×32draw，实际005/教师电荷/形状各421952行；三hyp hardADC/gather各1265856行；normal166249088、sign421952、附带PIT uniform7595136、005 CP1全部计账。G/AD/optimizer为0，项目累计G下界187667不变。

fast的257维27004928及O19的47258624对距离已完成，父向量1024912B已保存且核SHA。其后两个hyp尚需257维54009856、O1994517248，共148527104对距离。首个分组入口即ImportError，没有完成任何cell判定，不能把已生成教师或fast距离称为可检出资格通过。

26393次entered=completed，journal没有inflight、failed IDs或截断；Python异常在收费操作外，空failed_ids不代表原生成功。elapsed203.7415秒；监督采样峰RSS6721544192B、GPU371195904B、输出1028809535B，均低于既定上限。完整inventory核SHA后85份中选回27份，12994560B归档低于128MiB；全部operations/console日志、失败与fast父向量保留，58份未选收中包括源码与大输出，原大数组留远端。

此前源码审核漏查了新cell入口的常数符号归属；交叉bank人工fixture未调用该入口。已撤回完整路径可执行判断，原源码和原审核回执保持字节不变，失败复核另存。恢复另行登记为023：只修cell导入并续算CPU评分，复用完整T2核和fast分数，不重复005、RNG或fast距离，不改科学门。

020的89项失败、018前瞻配对资格与017旧72pending继续保留。022不启动T2训练，REAL/硬件接受仍未建立，scientific_status=response_match_not_established。

[最终失败回执](../../receipts/wp4_022_20261016/FINAL_RESULT_RECEIPT.json) · [独立失败复核](../../receipts/wp4_022_20261016/wake_20261016_1325_01/independent_failure_01/FAILURE_REVIEW.json) · [完整清单与选收](../../receipts/wp4_022_20261016/wake_20261016_1325_01/collection_01/COLLECTION_RECEIPT.json)
