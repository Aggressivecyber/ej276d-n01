# N01 / WP6-027 研究参考资料

本包把当前Q1/M0研究参考、构造链、真实接口支持和已保留失败集中到一份静态资料中。模型定位仍为开发研究参考，科学状态为`response_match_not_established`；当前没有统一`response.sample`入口或独立可运行模型bundle。

**封存状态：026最终结案与唯一独立结果回执已绑定；独立文档审与发布结论由ROOT保存在包外027回执目录。** 027本身不执行新科学/native任务；审查与发布结果由ROOT在包外登记。

| 文件 | 阅读用途 |
| --- | --- |
| [REFERENCE_BINDINGS.json](REFERENCE_BINDINGS.json) | 本机RAW文本/源码核验、原016完整manifest引用、A/005/M0/Q1六份CP的历史metadata、构造链及026最终三文件绑定 |
| [INTERFACE_SUPPORT.md](INTERFACE_SUPPORT.md) | 现有carrier384/显式随机流/电荷/005/硬观察原语与目标API的逐项映射，附源码行定位 |
| [APPLICABILITY_AND_FAILURES.md](APPLICABILITY_AND_FAILURES.md) | 数据/物理适用口径、015至026的证据与失败、完整WP5/6欠账 |
| [MANIFEST.json](MANIFEST.json) | 四份资料的RAW bytes/SHA、027范围及026先决证据；manifest不递归包含自身 |

## 如何使用

先看适用与失败卡，再按接口表查原源码。CP与构造统计的原路径、bytes/SHA在bindings中；本包不带CP/NPZ或原波形，不提供执行历史训练driver的指令。原016的116成员冻结清单与37项目导入项完整保留引用，后续工程封装可据此登记真实依赖，避免凭静态资料承诺精简runtime可运行。

身份核验分两层：`local_raw_text_rehashed`表示027在本机读原文本RAW字节并重新计算SHA；`historical_manifest_only_no_binary_read`表示从已核验的旧config/inventory抄录CP的bytes/SHA，027没有重读CP二进制或验证tensor。后者不能当本次模型载入验收。MASTER/PROGRAM/jobs属于可继续更新的导航状态，未列为永久冻结模型绑定。

## 范围与后续

027新增NN、CP load、AD、G、RNG、ADC、CDF、参数点、fit、training trial、私有波形读取和NPZ数值解码全为0；只生成这四份文本资料及最终checksum manifest。包体上限16MiB，原件和源码保持原路径。

受限参考的软件资料可先交付；模型晋级、完整WP5确认和完整WP6交付分别依赖其原证据门。真正封装推理loader、生成多draw示例或部署需要另行登记CP/NN/RNG/观察预算，且不会因本包自动启动。一次文档审完成后由ROOT决定发布。

来源与登记：[027 scope](../../../receipts/wp6_027_20261016/registration_01/PROSPECTIVE_SCOPE.json)；[用户主计划](../../../USER_MASTER_PLAN_20261015.txt)。
