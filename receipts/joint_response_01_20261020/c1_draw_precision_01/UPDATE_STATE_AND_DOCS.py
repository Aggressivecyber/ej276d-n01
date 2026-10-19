from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,tomllib

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
now=datetime.now(timezone.utc).isoformat()
raw=Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_bytes()
automation=tomllib.loads(raw.decode('utf-8'))
request=json.loads((here/'AUTOMATION_CREATE_REQUEST.json').read_bytes())
estimate=json.loads((here/'WAIT_ESTIMATE.json').read_bytes())
assert automation['id']=='n01' and automation['kind']=='heartbeat' and automation['status']=='ACTIVE'
assert automation['rrule']==request['rrule'] and 'COUNT=1' in automation['rrule']
assert automation['prompt']==request['prompt'] and automation['target_thread_id']==request['targetThreadId']
(here/'AUTOMATION_CONFIGURATION.toml').write_bytes(raw)
receipt=dict(status='PASS_SINGLE_WAKE_CREATED_AND_CONFIG_VERIFIED',UTC=now,automation_id='n01',kind='heartbeat',
    target_thread_id=automation['target_thread_id'],destination=request['destination'],count=1,automation_status='ACTIVE',
    next_wake_local=estimate['wake_local'],next_wake_UTC=estimate['wake_UTC'],
    configuration_sha256=hashlib.sha256(raw).hexdigest(),prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
    wait_estimate='WAIT_ESTIMATE.json',old_wake_consumed_and_deleted=True,periodic_polling=False)
(here/'AUTOMATION_CREATE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
receipt_path=(here/'AUTOMATION_CREATE_RECEIPT.json').relative_to(root).as_posix()
receipt_sha=hashlib.sha256((here/'AUTOMATION_CREATE_RECEIPT.json').read_bytes()).hexdigest()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    assert d['active_scientific_jobs']==1 and d['active_job_id']=='JOINT_RESPONSE_01_C1_DRAW_PRECISION_20261020'
    d.update(updated_UTC=now,current_action='Matched control and frozen estimator audited; paired 4/32-draw direction diagnostic running.',
        next_action='At the measured single wake, collect the true terminal state and audit all saved directions and candidates; continue evidence-based work.',
        followup_status='SINGLE_WAKE_SCHEDULED',current_followup_status='SINGLE_WAKE_SCHEDULED',followup_automation_id='n01',
        followup_automation_status='ACTIVE',followup_scheduled_local=estimate['wake_local'],followup_next_run_local=estimate['wake_local'],
        next_wake_local=estimate['wake_local'],followup_receipt=receipt_path,followup_receipt_sha256=receipt_sha,
        followup_last_delivered_UTC='2026-10-19T22:43:10.010Z',followup_last_consumed_local='2026-10-20T06:43:37.802479+08:00')
    d['followup']=dict(automation_id='n01',status='ACTIVE',count=1,interval_minutes=None,next_wake_local=estimate['wake_local'],
        next_wake_UTC=estimate['wake_UTC'],receipt=receipt_path)
    d['joint_response_01'].update(c1_draw_precision_single_wake_local=estimate['wake_local'],
        c1_draw_precision_estimated_finish_local=estimate['estimated_finish_local'],
        c1_draw_precision_audit_prepared=True,c1_draw_precision_audit_executed=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')

section='''# N01 当前阶段：匹配模型仍未过关，正在检验联合方向的抽样精度

[同家族正对照已完成并审核](receipts/joint_response_01_20261020/c1_matched_control_01/terminal_01/TERMINAL_REVIEW.json)：24个联合方向、41个候选、20次接受更新，752源文件及66输出核验。对同家族C1和原教师分别评价，均为36/36硬观察门、16/18条件精度门通过；seed0两库Q方差比约0.494，低于原0.50门，seed1通过。目标家族失配并非唯一原因。此前完整父集约0.4895的失败和原REAL两头各24/96改善门失败保留，科学响应匹配仍未建立。

随后直接完成[匹配家族冻结估计器检验](receipts/joint_response_01_20261020/c1_matched_estimator_01/terminal_01/TERMINAL_REVIEW.json)，原生耗时16.97秒，零参数更新。760源文件、103锁定输入、12输出及144残差向量独立复算通过。两个头×三个投影的端点跨库风险U为5.01e-6—1.25e-5；32draw聚合MC项为4.33e-6—4.70e-6，单4draw对应噪声约为其8倍；教师自身U接近零。seed1最后一步在新FIT/CHECK库U增加约2.3e-6。八库jackknife仅作描述；证据支持先检查方向估计精度。

已登记并正确启动[4与32draw联合方向对照](docs/JOINT_RESPONSE_01_C1_DRAW_PRECISION_20261020.md)：从同一冻结端点，各算一次54参数联合方向；4draw取32draw每父前4个。全部6465拟合父保留，差分／求解／候选规则保持，两个方向的三个预设步长均在独立32draw CHECK和VERIFY库检查；不选择新训练端点，参数更新为零。64父分块与扩大后的种子间隔已检查，避免102块时跨库种子重叠。

[启动核验](receipts/joint_response_01_20261020/c1_draw_precision_01/STARTUP_ASSESSMENT.json)确认06:58:22（UTC+8）派发；766源文件、108锁定输入和科学PID126205身份一致。截至07:00:53，完成81/216差分列：seed0的4draw方向已完成，32draw为27/54列，第二头尚未开始；此后进度未核验。RSS峰10,786,500,608字节，资源限额保持4CPU、GPU分配2.75GiB、RSS12GiB、内部1200秒、外部1500秒加20秒，无自动重启或预算延长。

06:43旧定时已删除。按本次4draw与32draw实测1.558／2.122秒每列及剩余工作，[预计07:06:05完成](receipts/joint_response_01_20261020/c1_draw_precision_01/WAIT_ESTIMATE.json)；已核验 **2026-10-20 07:08（UTC+8）本聊天单次接续**，留约115秒余量。见[定时回执](receipts/joint_response_01_20261020/c1_draw_precision_01/AUTOMATION_CREATE_RECEIPT.json)。醒来先核对实际终止状态，再独立复算全部方向与候选；阶段3方法诊断、REAL接受和阶段5仍分别判定。

以下保留历史快照；当前状态以本节、三台账及对应回执为准。
'''
readme=(root/'README.md').read_text(encoding='utf-8')
position=readme.index('\n---\n')
(root/'README.md').write_text(section+'\n'+readme[position:],encoding='utf-8',newline='\r\n')
plan=(root/'JOINT_RESPONSE_01_PLAN.md').read_text(encoding='utf-8')
lines=plan.splitlines()
assert lines[2].startswith('> 当前执行状态')
lines[2]='> 当前执行状态（2026-10-20）：同家族正对照亦因seed0条件Q方差比约0.494失败；零更新独立库诊断确认4draw噪声高于端点残差，最后一步收益未稳定复现。已启动固定端点4/32draw联合方向对照，原门限保持。07:08本聊天单次接续；科学响应匹配与阶段5未完成。'
plan='\n'.join(lines)+'\n\n## 2026-10-20 06:43接续：同家族失败与估计精度\n\n'+'\n'.join(section.splitlines()[2:-1])+'\n'
(root/'JOINT_RESPONSE_01_PLAN.md').write_text(plan.rstrip()+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(dict(status='DOCS_LEDGERS_SINGLE_WAKE_VERIFIED',UTC=now,wake_local=estimate['wake_local'],active_science=1)))
