CPU 保存数组读回已完整完成并回收。child/supervisor=0/0，36/36 pair，308 entered/completed，无失败、截断或在途。

单次 raw SHA 遍历176 NPZ、3,077,281,496B；解码72 hard、72 report、16 baseline、8 probe raw、8 probe gradient，共1,898,784条硬窗行。CP、NN、AD、训练、新随机、原TRAIN/DEV波形、新CDF/损失重放、GPU均为0。

回收19个小文件535,842B，归档573,440B，逐文件大小/SHA与5source+config6核验通过；原176 NPZ留在服务器。运行60.4864s，监督采样owned RSS峰448,200,704B，worker自报峰499,855,360B，监督终点worker artifacts319,753B。

独立核验包括硬窗编码、身份、Q/tail/PSD及形态/flag、4draw矩与由两库合并的8draw矩、保存梯度向量/norm/cos。完整wave/QPSD/noise距离仍是冻结producer和JSON聚合证据，没有进行raw CDF/损失重放；没有独立重拟C1尺度、重建preADC全carrier，也没有实测原Y第二份副本。本次PASS是指定保存数组读回，科学响应匹配仍未建立。
