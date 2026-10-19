from pathlib import Path
from datetime import datetime,timezone
import json

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
raw=json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
estimate=json.loads((here/'WAIT_ESTIMATE.json').read_bytes())
automation=json.loads((here/'AUTOMATION_CREATE_RECEIPT.json').read_bytes())
assert automation['status']=='PASS_SINGLE_WAKE_CREATED_AND_CONFIG_VERIFIED' and automation['count']==1
progress=raw['files']['run_01/PROGRESS.json'];counts=progress['execution_counts']
assert counts['finite_difference_columns']==54 and counts['teacher_accepted_joint_updates']==1
now=datetime.now(timezone.utc).isoformat()
prefix='receipts/joint_response_01_20261020/c1_precision_pilot_01/'
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    assert d['active_job_id']=='JOINT_RESPONSE_01_C1_PRECISION_PILOT_20261020' and d['active_scientific_jobs']==1
    d.update(status='JOINT_RESPONSE_01_C1_precision_pilot_running_single_wake',updated_UTC=now,
        execution_state_asof_UTC=raw['UTC'],current_stage='3_C1_precision_pilot',active_work_package='3_C1_precision_pilot',
        current_action='As of 07:30, first 32-draw joint iteration completed with one accepted update; remaining fixed iterations and dual-family reports pending.',
        next_action='At 07:48 consume one wake, verify terminal identity and audit actual high-precision trajectory and original gates; continue ready work.',
        followup_status='ACTIVE_SINGLE_WAKE',followup_automation_id='n01',followup_scheduled_local=automation['next_wake_local'],
        followup_next_run_local=automation['next_wake_local'],next_wake_local=automation['next_wake_local'],
        followup=dict(status='ACTIVE',kind='heartbeat',automation_id='n01',count=1,next_wake_local=automation['next_wake_local'],
            estimate_receipt=prefix+'WAIT_ESTIMATE.json',verification_receipt=prefix+'AUTOMATION_CREATE_RECEIPT.json'),scientific_acceptance=False)
    d['joint_response_01'].update(c1_precision_pilot_status='running_first_joint_update_verified',
        c1_precision_pilot_execution_counts_last_verified=counts,c1_precision_pilot_status_asof_UTC=raw['UTC'],
        c1_precision_pilot_estimated_finish_local=estimate['estimated_finish_local'],
        c1_precision_pilot_scientific_acceptance=False,c1_precision_pilot_automatic_REAL_promotion=False)
    if name=='jobs.json':
        d['jobs'][-1].update(status='running_first_joint_update_verified',status_asof_UTC=raw['UTC'],last_verified_execution_counts=counts,
            native_self_peak_RSS_bytes=raw['files']['run_01/seed0_joint_MEMORY_ITER00.json']['peak_RSS_bytes'],
            last_verified_supervisor_peak_RSS_bytes=raw['files']['control_01/resource_status.json']['peak_rss_bytes'],
            followup_local=automation['next_wake_local'],followup_automation_id='n01')
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')

current='''# N01 当前阶段：高精度联合方向已重复验证，正在做有界校准

[4/32draw方向对照及缺失第二头补全已独立审核](receipts/joint_response_01_20261020/c1_draw_completion_01/terminal_01/TERMINAL_REVIEW.json)。合并216个唯一差分列、4个联合方向、24个候选，均值和随机参数在同一完整硬读出目标中共同变化。32draw半步在CHECK/VERIFY的风险比：seed0为0.87008/0.83989，seed1为0.94718/0.90202，均通过原规则；4draw第二头没有两库共同通过的步长。此次方向诊断零参数更新，支持检验抽样精度对联合校准的影响。

原诊断因RSS峰15,096,520,704字节超过12GiB而停止，原资源失败与未保存0—9列的费用区间保留。[内存诊断](receipts/joint_response_01_20261020/c1_draw_precision_01/terminal_01/SAVED_MEMORY_DIAGNOSIS.json)定位到两个完整库的无效索引复制，共3,829,762,560字节。单独补全只重放第二头4列恢复检查并计算缺失内容，原生0退出、约150秒；775源文件、116锁定输入、19输出与17数值恢复检查通过，RSS峰12,385,734,656字节在原12GiB内。总实耗含恢复与旧未知为220—229差分列、5次求解，原失败没有删除。

已登记并正确启动[32draw有界联合校准](docs/JOINT_RESPONSE_01_C1_PRECISION_PILOT_20261020.md)：从原同家族对照两头冻结端点出发，每头固定4次新联合求解，共8方向／最多24候选。每轮使用全部6465拟合父、每父32draw；128报告父不参加拟合，原54坐标、差分、求解与接受规则保持。拟合和末端报告使用分离的新随机流；末端分别检查同家族C1和原教师的原硬观察及2048draw条件核门，不按中间Q方差调参或追加迭代。

[原生启动及进度核验](receipts/joint_response_01_20261020/c1_precision_pilot_01/STARTUP_ASSESSMENT.json)确认07:27:58派发、780源文件和120锁定输入、科学PID131951身份一致。截至07:30:09，第一头第1/4轮完成：54/432差分列、1联合方向、1候选并接受1次更新；9个原特征因子精确重现，之后进度未观测。顺序分配单个完整库并归还空闲内存后，第一轮进程RSS峰约6.08GiB。预算4CPU、GPU分配2.75GiB、RSS12GiB、内部1500秒／外部1740秒加20秒保持，无自动重启或延长。

07:08旧定时已消费并删除。按第一轮119.03秒、剩余7轮及已有同规格末端报告实测耗时，[预计07:45:11完成](receipts/joint_response_01_20261020/c1_precision_pilot_01/WAIT_ESTIMATE.json)，已核验 **2026-10-20 07:48（UTC+8）本聊天单次接续**，约168秒余量。[定时回执](receipts/joint_response_01_20261020/c1_precision_pilot_01/AUTOMATION_CREATE_RECEIPT.json)。醒来先处理实际结果，再推进当前可执行工作。

此前同家族两头正对照对C1和原教师均36/36硬门、16/18精度门通过，seed0 Q方差比约0.494仍失败；原跨家族完整父集约0.4895失败、原REAL两头各24/96改善门失败全部保留。同家族正对照、跨家族恢复、REAL接受和阶段5分别判定；科学响应匹配尚未建立。

'''
readme=(root/'README.md').read_text(encoding='utf-8')
boundary=readme.index('以下保留历史快照；当前状态以本节、三台账及对应回执为准。')
(root/'README.md').write_text(current+readme[boundary:],encoding='utf-8',newline='\r\n')
plan=(root/'JOINT_RESPONSE_01_PLAN.md').read_text(encoding='utf-8')
lines=plan.splitlines()
assert lines[2].startswith('> 当前执行状态')
lines[2]='> 当前执行状态（2026-10-20）：32draw联合方向在两头两独立库重复通过；资源停止后的缺失内容已补全审核。已启动每头固定4轮高精度联合校准，第一轮接受1次更新；07:48本聊天单次接续。原科学失败及阶段4/5未完成状态保持。'
plan='\n'.join(lines).rstrip()+'''\n\n[方向精度合并审核](receipts/joint_response_01_20261020/c1_draw_completion_01/terminal_01/TERMINAL_REVIEW.json)确认216个唯一FD、4方向与24候选可复算。32draw两头均在两个独立库有共同通过步长，4draw第二头没有；半步风险比为0.87008/0.83989和0.94718/0.90202。原作业RSS停止、0—9列未保存费用区间及补全重放计费保留，零模型更新。

据此前瞻登记[32draw有界联合校准](docs/JOINT_RESPONSE_01_C1_PRECISION_PILOT_20261020.md)，配置SHA256 `920df72317ba2cc31f39397a700011cc13a0680183ee82240adefa5a665a6df8`。两冻结端点各4轮、全部6465拟合父、54坐标完整联合求解；原模型／门限保持，新拟合与报告随机流分离，128报告父不拟合。两家族硬观察和2048draw条件核均在末端报告，同家族通过不替代原跨家族恢复。单个完整库顺序分配与分配器释放保持原dtype与汇总。

07:30:09快照确认780源／120pins和进程身份，第一轮54列、1方向／1候选／1接受更新完成，RSS峰约6.08GiB。按119.03秒实测首轮及余下7轮、同规格末端报告估计07:45:11完成，07:48单次接续已核验；内部1500秒、外部1740秒加20秒、RSS12GiB不延长。终止时使用已准备的收集与独立复算脚本；若失败先定位保存轨迹，不能直接用已暴露Q真值定向调参或自动开放REAL。
'''
(root/'JOINT_RESPONSE_01_PLAN.md').write_text(plan.rstrip()+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(dict(status='UPDATED_THREE_LEDGERS_AND_CURRENT_DOCS',UTC=now,science_asof_UTC=raw['UTC'],single_wake_local=automation['next_wake_local'],scientific_acceptance=False)))
