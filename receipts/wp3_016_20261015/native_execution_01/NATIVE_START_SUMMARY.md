# 016 native部署尝试

唯一一次deploy在bootstrap准备阶段失败，SSH退出1，未进入install或start；首事件等待和verify-start均未执行。

失败原因：stdin bootstrap使用尚未创建的016目录作为SELF_ROOT，新控制器operations仅查该目录，遗漏015已有的旧014固定SHA源码fallback。异常发生在native操作分派之前，本次bootstrap没有启动科学作业。

原失败回执：`deploy_native_01/prepare_NATIVE_RECEIPT.json`，SHA256 `223ffccae19fcb21905f8e3668cb61c4b7babe8b8e2a4ceb8792c7aafaf081a1`。

源码提交：`450ebb086a465e0dcbd1c1081e71edcf51c405b9`。配置、manifest、package和controller的原始pins保存在同级NATIVE_START_SUMMARY.json。

动作计数：deploy1、bootstrap SSH1、source SCP0、install0、start0、wait0、verify0；自动重试0。未追加任何SSH、进度读取、科学计算或状态快照。当前服务器资源及其它远端状态未经本次重新观察。

下一步由root处理最小fallback修复、重新冻包/独审/commit/D备份及新尝试授权。原pins和失败证据保留。
