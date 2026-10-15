# 019 首次评价依赖遗漏：静态诊断

本次019以native exit1结束。seed0已完成2000次G/AD更新并保存正常检查点；首次final评价生成128行信号、执行fast假设128行硬观测后，在提取诊断字段时缺少脚本模块。seed1未启动，完整final评价与2544格资格门均未生成。这里是运行包依赖遗漏，不能写成无配对方法的科学失败，也不能写成已通过。

## 已定位的实际路径

`run_wp4_019.py:337 → run_wp4_017.py:evaluate_kernel017:474 → run_wp1_013.py:observation_rows:67 → from mechanism012_losses import numeric_flags,peak_diagnostics`。

019冻结38成员没有`scripts/mechanism012_losses.py`。该文件在旧017已冻结：12854字节，RAW SHA256 `3fc84ccaf255b660d8eda4f96d43e2c236045a756e777c9b2d490f7137dc04ed`。本地原件、017配置记录、017基线提交、019部署源码提交和当前Git blob逐字节相同。Git blob为`2bffb20771425432e5220d2791faec8f856e4f4a`；最后修改提交`dc1b55a3a79308af0699a483f709d7104fdbd133`。详见SOURCE_GIT_MAPPING.json。

该文件直接导入5个项目模块：losses008、losses008_v2、model、model008、observation008；后两者递归带入electronics_v3、extension006。此8模块闭包中其余7份都已在019冻结包。numeric_flags/peak_diagnostics仅计算当前观察结果的标志/峰诊断，不要求新检查点、随机数或模型前向。

## 对原结论的更正

撤回本施工代理此前“30 science为完整运行时闭包、38成员足以完整执行019”的工程结论。原清单递归核对了n01相对/绝对导入，却没有递归遍历手工追加脚本的裸模块函数内导入；metadata、工厂和--help检查没有进入首次final评价。那些检查只能证明其实际覆盖的静态/入口行为，不能证明迟加载评价依赖存在。原源码、审查件及冻结回执保持不变，本诊断另存更正。

最小实际路径修复是保留原30 science并补上述1份旧源码，即31 science/39全包成员。MINIMAL_FIX_DRAFT.patch只是一份未应用草案；未来版本须有新身份、CFG/MF和冻结SHA。没有必要将run_wp1_013其他未调用功能的依赖一并搬入。完整模块AST的保守超集另列8个额外文件，其中仅mechanism012_losses属于当前缺陷；其余7个是其他lazy分支。该超集不是执行证明，也没有据此扩大修复。完整边、作用域及动态加载点见STATIC_DEPENDENCIES.json。

## 后续选择（均不在本次执行）

当前可直接按失败终点收口。如人明确授权工程恢复，应另行前瞻登记新版本：只加载已保存seed0 final.pt补其完整final评价，seed0不重训；仅对尚未启动的seed1执行原固定2000步unpaired。新G/AD至多2000，不开新teacher/候选、paired或T2，不改变018门。

seed0 final.pt在远端完整SHA清单中为114968字节，SHA256 `ad285f938745d906df7295433ea9d1c3bc83d181e1c26db16375df7f2af48b79`；此诊断只读取其清单记录，没有读取或反序列化检查点内容。新单进程恢复至少需要005与seed0各1次CP加载。两份完整final评价共843904信号行，seed1训练student/teacher各2048000行；总005/shape/charge组装4939904行。原失败的128行评价前缀必须从同一评价随机命名空间完整重算，新增normal请求50432，已执行fast硬观测128行也会重复；这些都必须计入新任务及累计账本。精确可计算NN、RNG、父采样、ADC和距离预算条目见RECOVERY_OPTIONS_NOT_EXECUTED.json；它不是可执行配置或运行授权。单独只评价seed0也不能取得双seed资格。

本次仅进行本地源文本、JSON元数据、Git对象和SHA的静态核查。没有SSH、CP加载、项目模块import、私有数组/原波形读取、NN、AD、G或RNG执行；没有修改已冻结019/旧源码、CFG/MF或共享登记，没有新driver/helper/controller，没有重跑旧工程套件。
