"""Prepare/confirm one native app heartbeat after the saved healthy015 start."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SETTINGS=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
CURRENT=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write(path,value,new=True):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)

ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['prepare','confirm'])
ap.add_argument('--minutes',type=int,default=65);args=ap.parse_args()
p=read(ROOT/'wp3_015_config.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
execution=read(OUT/'EXECUTION_STARTED.json');state=read(OUT/'start_verification_01/REMOTE_STATE.json')
validation=read(OUT/'start_verification_01/START_VALIDATION.json');start=read(OUT/'start_native_01/START_RECEIPT.json')
assert validation['status']=='actual_first_G1_and_mean_gate_verified'
assert state.get('runs/primary/artifacts/FAILED.json') is None
assert state.get('runs/primary/supervisor/exit_receipt.json') is None
progress=state['durable_child_progress'];counts=progress['completed_counts']
assert 1<=counts['G_updates']<8000 and not progress['failed_ids']
assert execution['current_training_updates']==counts['G_updates']
status_view=state['runs/primary/artifacts/status.json']
mean_adapter_rows=counts.get('mean_adapter_rows')
if mean_adapter_rows is None:
    # The durable journal omits zero charges. Bind this zero to the saved
    # explicit status counter rather than treating an arbitrary missing key as 0.
    assert status_view['counters']['mean_adapter_rows']==0
    assert status_view['counters']['G_updates']==counts['G_updates']
    mean_adapter_rows=0
if args.mode=='prepare':
    assert 15<=args.minutes<=180
    wake=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=args.minutes)
    wake=wake.replace(second=0,microsecond=0)+dt.timedelta(minutes=(-wake.minute)%5)
    local=wake.astimezone(dt.timezone(dt.timedelta(hours=8)))
    plan=dict(automation_id='n01',kind='heartbeat',name='N01 015回收与六工作包推进',status='ACTIVE',count=1,
        next_wake_local=local.isoformat(),next_wake_UTC=wake.isoformat(),
        target_thread_id='01a12bad-8839-7522-89a5-f028b107d7ff',source_commit=source['source_commit'],
        rrule=f'FREQ=DAILY;COUNT=1;BYHOUR={local.hour};BYMINUTE={local.minute};BYSECOND=0',
        basis='Conservative engineering margin after saved actual firstG1 and one healthy snapshot;014 full batch35.81min,015 adds bounded mean arithmetic. This is not a completion guarantee.',
        continuous_agent_monitoring=False)
    text=f'''在本聊天单次苏醒，回收已正确启动的N01 WP3/015，然后继续用户《N01从013到项目交付：主实施计划》的六工作包。所有新增源码、数据、缓存、临时、日志、权重、归档、图表保存在E/D；原服务器登记目录只读引用，新项目目录写入。根负责规划登记，施工及独审复用Sol Max和成熟工程。每个长任务保存首次真实计算及同snapshot健康证明后，只设同一个n01的一次未来wake并休息，不持续监控。最新用户指令优先。

先读E:/GAN/research/N01/AGENTS.md、USER_MASTER_PLAN_20261015.txt、MASTER_EXECUTION_PLAN.json、PROGRAM.json、jobs.json、OBSERVATION_CONTRACT.json、DATA_ROLES.json、INPUT_INFORMATION_AUDIT.json、WP3_DATA_ROLES.json、docs/WP3_015_CONFIGURATION.md、docs/WP3_ACCEPTANCE_CONTRACT.md、CALIBRATED_TOLERANCES_WP1_013.json、wp3_015_config.json、WP3_SOURCE_MANIFEST.json，以及receipts/wp3_015_20261015的EXECUTION_READY.json、EXECUTION_STARTED.json、SOURCE_COMMIT_READBACK.json、D_SOURCE_BACKUP_RECEIPT.json、FOLLOWUP_RECEIPT.json、runtime_package_01/FREEZE_RECEIPT.json、independent_01/PRELAUNCH_REVIEW.json与FROZEN_PACKAGE_REVIEW.json、start_native_01/START_RECEIPT.json、start_verification_01/REMOTE_STATE.json和START_VALIDATION.json。根本地closure和future状态可见CORRECT_START_AND_FOLLOWUP.json；不要把014/CPU旧启动记录当当前任务。

015源commit {source['source_commit']}；105source/107个原始Git blob相合，D113件源码备份已SHA核验。cfg {sha(ROOT/'wp3_015_config.json')}，MF {sha(ROOT/'WP3_SOURCE_MANIFEST.json')}，source tar9ea44db9ab3fd36cfeb8b68f877c30f89e9a441bcd98fcd31666c6b964bbf422。23inputs=原014全部8+积分统计1+C1头2+remote-only REPORT12；11contracts；原95/97及旧271保护不变，新增97为合计368。12个原REPORT未下载，freeze只核原318完整库存4cf27f6c771294561a474ef2de9574ab711d9322c588da488f0d9ce2e49792d5，native install/start/driver仍逐物理SHA核后才decode。安装已核497唯一immutable路径。

SSH rylan@100.82.231.102，Python /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python；当前root /home/rylan/projects/ej276d-n01/wp3_015_20261015；freeze E:/GAN/research/N01/receipts/wp3_015_20261015/runtime_package_01。start UTC {start.get('UTC',start.get('started_UTC','see START_RECEIPT'))}，supervisor {start['supervisor_pid']}、child {validation['child_pid']}、boot {start['boot_id']}仅历史身份。唯一启动snapshot保存真实G1、非零梯度/参数更新、M0白名单1288、原A/005保护、ALL6 mean gate阳性和同snapshot健康。snapshot durable已完成G{counts['G_updates']}、training backward{counts['training_backward_queries']}、AD{counts['new_AD_queries']}、CP{counts['checkpoint_loads']}、A/head/005各{counts['old_A_rows']}/{counts['head_actual_parameter_forward_rows']}/{counts['noise005_rows']}，mean adapter行{mean_adapter_rows}（来自同snapshot显式status零计数）；durable在途操作ID {progress.get('unknown_inflight_ids',[])}。较早status视图ADC/gather各686128、operation5434尚在途；较后durable视图各687152且在途为空，两视图均保存，不能据此推断G121或拼接其它采样点。后续若有在途操作，其完成数未知，不能补记完成或0。累计最后核验训练167667+{counts['G_updates']}={167667+counts['G_updates']}，该启动下界已入账。首证明不是四trial完成，当前原生终点未知。

只调用一次新015控制器状态：python -B scripts/wp3_controller.py --mode status --freeze receipts/wp3_015_20261015/runtime_package_01 --out 新专属receipts/wp3_015_20261015/wake_status_01。目录已存在先inspect换新名，不覆盖。健康未完就根据实际进度/资源余量更新同一个n01单次未来wake并休息，不连续查询或重新start。SSH不可达记unknown，不能把进程或成本记0。失败/资源停止/原生非0先collect-failed（归档已在则collect-failed-inspect）保留所有native exit、FAILED、entered/completed/inflight/failed和实际G/AD/NN/CP，不自动整批重训、扩大预算或抹去费用。

若producer COMPLETE且native child/supervisor均0，用同一控制器--mode collect --freeze同包 --out新receipts/wp3_015_20261015/collection_01 --collected runs/wp3_015_20261015/selected_collected_01 --include-head-checkpoints安全选收。已有远端归档用collect-inspect恢复，不覆盖。先核全远端库存，再核选择文件大小/SHA与源/输入/保护，默认archive512MiB/单头64MiB；E空间有限，raw留remote，不全搬或转C。完整数值审核要登记仅保存数组的有界CPU readback，复用014解码/行统计/4与8draw矩/MCSE/梯度等适用部分，禁止为读回重新CP/NN/AD/新随机或原DEV。只有实际完成的核验才可称独立；完整CDF/loss或尺度fit未重放就明确producer+独立聚合范围。CPU若长也正确启动后一次wake休息。

015四trial固定seed0_M0、seed0_M1、seed1_M0、seed1_M1，各2000G。M0从对应014C1 final头继续1288形状参数；M1冻结同一头，仅6个零θ均值参数；两者fresh AdamW2e-4/wd1e-4/clip5，samplers/shape/005/extension/wave/integral流同seed复位配对，不继承014 optimizer。C1 final只是研究参考，不是权重晋级。A/005原件冻结可回退，B(x)z仍按原A定幅，不乘随机Q或适配均值。mean基是A、σ2/13点Gaussian平滑差、τ16/64点因果差，后两canonical零Q，所有prefix100严格0，ΔRMS≤.1RMS(A)。SOURCE控制canonical[110:366)、tail[156:366)，原FIT_STATISTICS008.loss qref3.7103998046755806/floor0.0005319378913383611，当前Q_mu/qbar/qsig随均值同步，旧A值另存reference_A。没有引入charge/time/local-noise分支或005重拟合。

前置mean gate只读REAL TRAIN21047/422组及12个旧C1 final REPORT均值，未先读SOURCE、CP/NN/优化器/新随机；ALL6 r>max(1e-7V,3sqrt(bankdiff_RMS²/2+REAL_group_SE_RMS²))且两残差cos>.5才训练。它支持固定人口假设下均值试验，不是识别真实条件均值或配对真值；422旧组不是422采集。电子学baseline DC单列，信号mean不能修prefix100。若收到合法training_skipped终态必须保持CP/NN/AD/G0和原因、实际short trials0，不按失败重训或挪门。

完整四trial最大及正常终点精确费用：A/head实际/headtotal/005各8613952行，mean adapter4306976行，hardADC/gather各26474784；CP4（A+005+每seed C1头各1），optimizer4、G/steps/backward/AD各8000；calibration map/probe AD/integral fit/D/noise/pop/C2ST/DEV模型行均0。前置REPORT12文件/79116父行、REAL21047行、6检验单列。实际费用从durable/native日志取，inflight不填满预算。成功时累计167667+8000=175667；启动{counts['G_updates']}已计，只补8000-{counts['G_updates']}，不能重复加8000。原科学费用与所有attempt负结果永久保留。

原硬14bit/384carrier/100baseline/firstargmax[112:144)/p−16:p+240/256输出/1软件ns/tail46不变；6593父每库4draw×2库对全部21047 REAL、三噪声假设、全部flags和四评价轴（全人口/输出Q条件/SOURCE输入趋势/同父重复）分别报告。数字T SHA f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b不改，零/undefined T不自动PASS，RR不定阈值。逐seed/h先mean2bank各距离，不改pool8；候选ALL6每个positive-defined T相对A与共同C1 initial均非劣才eligible；required missing→pending。按mean12 final wave选择，1e-12 tieM0，若胜者未比共同initial低超过1e-12则保留C1参考mean关闭。研究参考选择、ALL6≥2%用途、绝对T与科学验收分开，按真实保存误差决定mean是否保留。

014已经完整闭合，不再调用旧014或CPU控制器status/start/collect。其FINAL_RESULT_RECEIPT SHA86072087796998e22d4c09740590e8e00ce53a10eb7bab3717fb17d20558bc7c，报告reports/wp2_014_20261015/N01_WP2_014_REPORT.md SHA b814487a015b87ed3e9bbd6373acc51f836ee540c5dc2580bdedaf03da5ab463，3英文图/6CSV/17项内容+根代表渲染/D47件备份；local closeout commitfd65a92ae2b623adc2bec6dccbd0cf4f0c38095e。014 G8000/AD9872，C1 codepath较低mean12但用途仅5/6（seed1fast1.9887788589535327%<2%），weights/science false。176raw留remote；36pair/176NPZ/1898784行独立复核已完成，完整lossCDF/median未重拟合。013也已闭合，无第4次013重跑。

完成015独审/中文报告/必要英文图与元数据后，继续WP3剩余受控模块，避免无尽诊断。全部成功则短试4+4=8/12、WP3已用4/6，最多余2次短试；失败/skip依实际计，不补满。按真实保存残差判断必要的正值均值校正q_sig(aQ+bQ²,a,b≥0)、零Q电压形状、最后必要的中心化时间或唯一受限分布扩展；不得重复计电子学常数方差，勿任意SOURCE/REAL配对MSE，不重训旧full19k或启GAN/无界架构搜索。每个新计算先登记精确角色/配置/源码commit/CPNNAD预算，复用成熟工程，普通末位差异不重建宏观协议。

WP4固定两教师×paired/unpaired×2seed共8次，每次≤2000G，用条件分布评分和父隔离，改变人口检验冻结条件核；同族paired失败则先修表达/优化，不继续真实长训练。教师成功不能称硬件或跨采集物理复用。WP5三组隔离折×2全流程种子×3系统，所有数据依赖组件（A/005/损失尺度/响应头/人口/C2ST）均折内重拟合，不能只训头沿用见过留出父的A；冻结设置后全TRAIN部署不作新未见成绩。WP6按主计划交付有效模型/统计/接口/适用卡/失败案例/中文报告/英文图/论文软件包，公开包不含原始私有波形；完整随机目标失败时交付独审最佳受限fallback并保留失败证据。当前仅确证一次实测采集，旧DEV已曝光，其它历史pool unknown，科学状态维持response_match_not_established直到约定证据达成。

每个新长任务只保存首个真实计算及同snapshot健康证明，更新同一个n01一次未来wake，立即结束本轮休息，不持续监控。没有待回收长任务时完成可交付工作后暂停n01。015正确启动不是四trial终点，更不是整个六工作包完成。
'''
    write(OUT/'FOLLOWUP_PLAN.json',plan)
    (OUT/'FOLLOWUP_PROMPT.txt').write_text(text,encoding='utf-8',newline='\n')
    before=OUT/'automation_before_01';before.mkdir(exist_ok=False)
    shutil.copy2(SETTINGS,before/'automation.toml')
    print(json.dumps(dict(status='one_followup_prepared_not_yet_applied',next_wake_local=plan['next_wake_local'],rrule=plan['rrule'])))
    raise SystemExit(0)

plan=read(OUT/'FOLLOWUP_PLAN.json');actual=tomllib.loads(SETTINGS.read_text(encoding='utf-8'))
expected=(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8').rstrip('\r\n')
assert actual['id']=='n01' and actual['kind']=='heartbeat' and actual['status']=='ACTIVE'
assert actual['rrule']==plan['rrule'] and actual['name']==plan['name']
assert actual['target_thread_id']==plan['target_thread_id'] and actual['prompt']==expected
follow=dict(schema='n01/wp3-single-followup-confirmation/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    automation_id='n01',kind='heartbeat',status='ACTIVE',count=1,next_wake_local=plan['next_wake_local'],
    next_wake_UTC=plan['next_wake_UTC'],target_thread_id=plan['target_thread_id'],
    actual_saved_settings_verified=True,settings_sha256=sha(SETTINGS),
    prepared_prompt_file_sha256=sha(OUT/'FOLLOWUP_PROMPT.txt'),actual_prompt_sha256=hashlib.sha256(expected.encode()).hexdigest(),
    continuous_agent_monitoring=False,no_remote_calls_after_correct_start_snapshot=True,source_commit=source['source_commit'])
write(OUT/'FOLLOWUP_RECEIPT.json',follow)
metadata=[ROOT/n for n in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md')]+[CURRENT]
snapshot=OUT/'metadata_before_followup';snapshot.mkdir(exist_ok=False)
for path in metadata:shutil.copy2(path,snapshot/path.name)
for path in [p for p in metadata if p.suffix=='.json']:
    value=read(path);assert value['active_scientific_jobs']==1
    value.update(followup=follow,followup_automation_id='n01',followup_automation_status='ACTIVE',
        followup_next_run_local=plan['next_wake_local'],followup_receipt='receipts/wp3_015_20261015/FOLLOWUP_RECEIPT.json',
        followup_receipt_sha256=sha(OUT/'FOLLOWUP_RECEIPT.json'))
    if 'wp3_execution' in value:value['wp3_execution']['followup']=follow
    for wp in value.get('work_packages',[]):
        if wp['id']=='WP3':wp['execution']['followup']=follow
    for job in value.get('jobs',[]):
        if job['job_id']=='N01_WP3_015_20261015':job.update(followup=follow);job['execution']['followup']=follow
    write(path,value,False)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig')
text=text.replace('正确启动后设置同一n01一次未来唤醒并休息。',
    '已确认'+plan['next_wake_local']+'（Asia/Singapore）同一n01单次唤醒；当前休息，不持续监控。',1)
readme.write_text(text,encoding='utf-8',newline='\n')
program=read(ROOT/'PROGRAM.json');program['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(ROOT/'PROGRAM.json',program,False)
current=read(CURRENT);current['plan_sha256']=sha(readme)
current['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(CURRENT,current,False)
write(OUT/'CORRECT_START_AND_FOLLOWUP.json',dict(schema='n01/wp3-correct-start-one-wake/v1',UTC=follow['UTC'],
    first_start=ref(OUT/'EXECUTION_STARTED.json'),followup=ref(OUT/'FOLLOWUP_RECEIPT.json'),
    source_commit=source['source_commit'],actual_snapshot_G=counts['G_updates'],
    cumulative_training_last_verified=167667+counts['G_updates'],unknown_inflight_ids=progress.get('unknown_inflight_ids',[]),
    response_match_established=False,whole_six_workpackage_project_complete=False,no_further_remote_calls_this_turn=True))
print(json.dumps(dict(status='one_future_wake_confirmed_rest',next_wake_local=plan['next_wake_local'],
    cumulative_training_last_verified=167667+counts['G_updates'],no_more_remote_calls=True)))
