"""Prepare one human-readable wake after a verified real014 update."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import tomllib

r=Path('E:/GAN/research/N01');o=r/'receipts/wp2_014_20261015'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sv=read(o/'start_verification_01/START_VALIDATION.json')
assert sv['status']=='actual_first_G1_source_and_initialization_verified'
state=read(o/'start_verification_01/REMOTE_STATE.json')
assert not state.get('runs/primary/artifacts/FAILED.json') and not state.get('runs/primary/supervisor/exit_receipt.json')
source=read(o/'SOURCE_COMMIT_READBACK.json')
cfg=read(r/'wp2_014_config.json')
counts=state['durable_child_progress']['completed_counts']
assert counts['G_updates']==197
saved=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
old=tomllib.loads(saved.read_text(encoding='utf-8'))
assert old['kind']=='heartbeat' and old['target_thread_id']=='01a12bad-8839-7522-89a5-f028b107d7ff'
backup=o/'automation_before_01';backup.mkdir();shutil.copy2(saved,backup/'automation.toml')
wake=dt.datetime(2026,10,15,10,30,tzinfo=dt.timezone(dt.timedelta(hours=8)))
assert wake.astimezone(dt.timezone.utc)>dt.datetime.now(dt.timezone.utc)
prompt=f'''在本聊天于2026年10月15日10:30（Asia/Singapore，UTC+8）单次苏醒，回收已正确启动的N01 WP2/014，并继续用户《N01从013到项目交付：主实施计划》授权的六工作包路线。长计算正确启动后休息，单次唤醒再检查，不持续轮询。所有新增源码、数据、缓存、临时、日志、权重、归档与报告保存在E/D；科学服务器既有注册目录只读引用与本项目新目录写入。根负责规划登记，施工及独审按AGENTS使用Sol Max，优先复用已有代理/成熟工程，普通末位数值差异不重开整套宏观协议。

先读E:/GAN/research/N01/AGENTS.md、USER_MASTER_PLAN_20261015.txt、MASTER_EXECUTION_PLAN.json、PROGRAM.json、jobs.json、OBSERVATION_CONTRACT.json、DATA_ROLES.json、INPUT_INFORMATION_AUDIT.json、WP2_DATA_ROLES.json、docs/WP2_ACCEPTANCE_CONTRACT.md、docs/WP2_014_CONFIGURATION.md、wp2_014_config.json、WP2_SOURCE_MANIFEST.json、CALIBRATED_TOLERANCES_WP1_013.json和receipts/wp2_014_20261015的EXECUTION_STARTED.json、SOURCE_COMMIT_READBACK.json、FOLLOWUP_RECEIPT.json、runtime_package_01/FREEZE_RECEIPT.json、independent_01/PRELAUNCH_REVIEW.json及FROZEN_PACKAGE_REVIEW.json、start_native_01/START_RECEIPT.json、start_verification_01/REMOTE_STATE.json与START_VALIDATION.json。最新用户指令优先。

014服务器root /home/rylan/projects/ej276d-n01/wp2_014_20261015；SSH rylan@100.82.231.102，Python /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python。实际UTC2026-10-15T00:16:29启动，监督PID107073/计算107083只作历史身份。首次实际G1、非零梯度及参数更新、1288参数白名单、保护A/005、配置源码输入哈希与同snapshot无FAILED/native非0已验证。启动的唯一状态快照已完成197G、198次training backward、234probe AD、总AD432、CP2；在途操作真实完成数未知，不能补记为0或完成。累计最后核验训练为159667+197=159864，是运行期持久化下界。seed0初始化实际RMS6.100740429897056e-5V=0.9994843046300347 ADC LSB，为A_RMS的0.0014775124343396392；校准仅缩初始head权重，不改变未来gain .05或模型容量。

实际执行源码commit {source['source_commit']}，95source+2controls共97个原始Git blob相合，271旧源码/控制保护记录不变。cfg SHA {sha(r/'wp2_014_config.json')}，MF {sha(r/'WP2_SOURCE_MANIFEST.json')}，source tar1d25cf4a77e3722e8c86edfe7c5103f4905932ae37028eb3642b8690aacb056a。8input/7contracts，原硬14bit/384carrier/100baseline/firstargmax112:144/p−16:p240/256输出/1软件ns/tail46不变。CPU12、GPU10GiB、RSS8GiB，每trial7200s/1536MiB，整批28800s/6144MiB，独占本项目lock且只管理本作业树。当前10:30唤醒依据首197G/约131秒启动进度与评价开销给出保守工程余量，早于8小时资源硬上限；它不是保证已完成的时间。

只用scripts/wp2_controller.py --mode status --freeze receipts/wp2_014_20261015/runtime_package_01 --out 一个新E专属回执目录读一次远端状态。若健康未完，根据实际进度/余量更新同一个n01为下一次单次唤醒并休息，不反复轮询或重新start。SSH不可达则远端unknown。若失败，按collect-failed/collect-failed-inspect回收现场，保留native child/supervisor exit、FAILED、entered/completed/inflight/failed日志和各trial已耗G/AD/NN，不能自动整批重训、扩大预算或抹去已完成训练。

若四trial producer COMPLETE及native child/supervisor退出0，用controller --mode collect --freeze同目录 --out receipts/wp2_014_20261015/collection_01 --collected runs/wp2_014_20261015/selected_collected_01 --include-head-checkpoints安全选择性回收。任何目录已存在先inspect并使用新专属目录；远端归档已存在则collect-inspect恢复，禁止覆盖。先核完整远端库存SHA，再核所选文件传输SHA/大小、95source/8input/7contracts/271保护和native/完整日志。默认archive512MiB、单头CP≤64MiB；本地E空间约2.5GiB，禁止把全部raw强行拷回或转C。回执必须明确raw NPZ仍在服务器、本地未完整回收，不能把库存核验冒称原始数组独立数值读回。

014固定顺序seed0_C0、seed0_C1、seed1_C0、seed1_C1，各2000G，共8000G，2种子2026101540/2026101640。CP2、optimizer创建4/step8000，training backward8000，probe额外AD≤1872，总AD≤9872；oldA8616256、head实际8616512、校准map≤36864、总head=实际+map≤8653376、noise0058616000、ADC/gather各26480928、integral statistics fit1；D/noise/pop/C2ST/DEV模型行均0。maps与probeAD合法少用按实际保存，不能填满预算；总head与总AD按冻结恒等式检查。四次短试计入全计划最多12次搜索，原WP3预留≤6，教师8次与最终确认/部署重拟合预算独立。

独立审核先用保存JSON/日志/新头状态和选择性库存验证配对初始化字节、SOURCE/REAL/shape/noise/wave projection流、所有TRAIN覆盖、0/2000分层梯度与实际更新。训练source512父×2draw对1024REAL，FullCoverageSampler保留全部6593/21047TRAIN；ID/行号/Qbin/h审计元数据不进入NN。C0原wave+2QPSD+.01regular；C1仅换Q/T和4段积分[0,16)/[16,46)/[46,96)/[96,256)，REAL TRAIN固定尺度/signedasinh、0.5coordinate+0.5十六方向联合W1，外权2。C1独立投影流不消耗其它配对流。C2关闭，不依据本轮失败另开裁窗代理大工程。梯度是条件子批损失诊断，3分量vector相加得到total，不是全局逐事件loss贡献比例。

四套硬评价分别报告全人口、输出Q条件、SOURCE输入趋势、同父重复；wave、Q/tail/rawPSD/条件PSD、噪声、形态、输入依赖分开。使用原95source已冻结hard evaluator、全部6593父每库4draw×2库、21047REAL/422原逻辑组。Q<=0/guarded/undefined/ADC/peak/crop失效全部保留。SOURCE输入只有约一维规范形状信息及h关系；已有同Q不同形状配对是SOURCE现存输入，尚非真实响应干预。当前确认1次实测采集，旧DEV已曝光，其他历史pool曝光unknown；没有跨采集确认。

必须对真实保存hard rows做独立数值审核：raw在服务器时可登记一个仅仅读取已保存数组的小型远端readback，严格不CP/NN/AD/新随机库/原数据DEV波形，核整数codes+doublebaseline+可选IEEE XOR可逆、父/seed/bank/draw身份、flags、各4draw和合并8draw的均值/协方差因子与MCSE、保存标量指标和拟合统计SHA。也可在E剩余空间许可时只安全回收所需若干raw成员，必须完整库存与SHA绑定，不删原件。禁止为读回重新运行A/005/头或把缺raw写成已核；CPU分析若长也正确启动后单次wake休息。原native写入Y roundtrip与独立encoding复核的范围分别记录。

数字T沿用013正式表SHA f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b；7case max(mean2banks)公式、人工增益/尾3%/噪声10%/额外Gaussian1软件ns、2%wave用途门保持冻结。零或undefined T为not_calibrated不自动PASS，RR不用于定阈值。科学匹配、工程完成、在域开发收益和跨条件复用分别报告。

选择严格按config.development_selection和WP2_ACCEPTANCE：每seed/hyp先mean两bank各自经验距离（不换为pool8距离），final−A，对正且已标定T的所有分量检查ALL6同候选非劣；必需已标定分量缺失/undefined使候选pending、不eligible；未标定分量持续pending。合格候选的score为2seed×3hyp×2bank共12个final完整wave距离均值；1e-12以内tie选C0。合格集空保留原C0工程入口、不晋级权重。相对wave≥2%用途门按ALL6单独报告，共同initial比较、绝对distance/T、两种子差异与失败原因分别给出。代码路径选择不等于权重/响应匹配成功。

更新主计划/PROGRAM/jobs/E:/GAN/CURRENT_SIM2REAL_PROGRAM，真实8000G应在原159667上按已核actualG只加一次（197启动下界已入账，不能再重复加）；所有attempt费用、未完成原生状态与原009–013负结果保留。013已闭合：其源commit7be...，收尾本地commit5cfd71f92f0b782d2a1b215062a1d36dc17dac1e，报告reports/wp1_013_20261015/N01_WP1_013_REPORT.md SHAd8088e081e5fd324df903b14ddb809e32a85fdb8cae34430338a1fab891cdf00，wave收益0.9162/0.9914/1.6907%均未过2%，morph/DC绝对差仍大。013三attempt总A/head403992、map40960、headtotal444952、noise158232、ADC2202768、gather3677328、CP9，训练/AD全0；无需第4次013重跑。

在014正确完成和独立读回后，继续用户主计划WP3：基于实际硬误差先选一个优化表示，再登记必要的小模块/identity mean adapter或受控随机响应改动与单因素短试，遵守总搜索≤12及WP3预留≤6，不重训旧full19k、不启GAN或无界架构扫描。后续WP4两教师×paired/unpaired×两种子8次、WP5三折×两全流程种子×三系统的全部组件折内重拟合、冻结后全TRAIN部署与WP6中文报告/英文图/论文软件交付沿已登记主计划推进；教师恢复不是独立实测复用。必要配置、角色、source commit、精确预算和新语义工程检查在每个新计算前登记，复用成熟工程；不反复为末位差异重建协议或停在无尽诊断。

每个新长任务仅在首次真实计算及同snapshot健康证明保存后更新同一个n01为一次未来苏醒，然后结束本轮休息。没有待回收长任务时完成可交付工作后暂停同一个n01。当前014正确启动不是四trial终点或整个六工作包完成，科学状态保持response_match_not_established直到约定证据达成。'''
path=o/'FOLLOWUP_PROMPT.txt';assert not path.exists();path.write_text(prompt+'\n',encoding='utf-8')
plan=dict(schema='n01/wp2-one-wake-plan/v1',prepared_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    automation_id='n01',kind='heartbeat',name='N01 主实施计划014回收与推进',target_thread_id=old['target_thread_id'],
    status='ACTIVE',count=1,next_wake_local=wake.isoformat(),next_wake_UTC=wake.astimezone(dt.timezone.utc).isoformat(),
    prompt_path=str(path),prompt_sha256=sha(path),
    reason='Estimated completion from the saved197G/130.8s snapshot plus substantial evaluation/runtime allowance; if still running at wake, one new wake replaces polling.',
    expected_first_G1_verified=True,last_verified_G_updates=197,source_commit=source['source_commit'],
    no_remote_calls_after_correct_start=True,continuous_agent_monitoring=False)
(o/'FOLLOWUP_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(name=plan['name'],next_wake_local=plan['next_wake_local'],count=1,prompt_sha256=plan['prompt_sha256']),ensure_ascii=False))
