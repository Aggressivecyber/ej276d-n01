from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tomllib

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
request=json.loads((here/'AUTOMATION_CREATE_REQUEST.json').read_bytes())
result=json.loads((here/'AUTOMATION_CREATE_TOOL_RESULT.json').read_bytes())
estimate=json.loads((here/'WAIT_ESTIMATE.json').read_bytes())
raw=json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
registration=json.loads((here/'REGISTRATION.json').read_bytes())
meta=json.loads(result['content'][1]['text'])
assert not result['isError'] and meta==dict(automationId='n01',mode='create',status='ACTIVE')
config_raw=Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_bytes()
configuration=tomllib.loads(config_raw.decode('utf-8'))
for field in ('kind','name','prompt','status','rrule'):assert configuration[field]==request[field],field
assert configuration['id']=='n01' and configuration['target_thread_id']==request['targetThreadId']
assert 'COUNT=1' in configuration['rrule']
(here/'AUTOMATION_CONFIGURATION.toml').write_bytes(config_raw)
now=datetime.now(timezone.utc).isoformat()
receipt=dict(status='PASS_SINGLE_WAKE_CREATED_AND_READ_BACK',UTC=now,automation_id='n01',kind='heartbeat',count=1,
    next_wake_local=estimate['single_wake_local'],next_wake_UTC=estimate['single_wake_UTC'],
    target_thread_id=request['targetThreadId'],configuration_sha256=hashlib.sha256(config_raw).hexdigest(),
    prompt_matches_request=True,delete_or_confirm_consumed_on_wake=True,scientific_acceptance=False)
(here/'AUTOMATION_CREATE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
progress=raw['files']['run_01/PROGRESS.json'];counts=progress['execution_counts']
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    assert d['active_scientific_jobs']==1 and d['active_job_id']==registration['job_id']
    if name=='jobs.json':
        assert len(d['jobs'])==132
        d['jobs'][-1].update(status='running_confirmed_prefix_replay_pass',execution_counts_last_verified=counts,
            status_asof_UTC=raw['UTC'],progress_asof_UTC=progress['UTC'],estimated_finish_local=estimate['estimated_finish_local'])
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_fix01_running',updated_UTC=now,
        execution_state_asof_UTC=raw['UTC'],last_verified_UTC=raw['UTC'],latest_run_id=registration['job_id'],
        current_action='Same 32-draw centers and first nine FD columns replayed; exact native W1 repair runs the bounded diagnostic.',
        next_action='At the measured one-time wake, collect terminal outputs and audit full joint/native directions; no scientific updates.',
        followup=dict(kind='heartbeat',automation_id='n01',count=1,next_wake_local=estimate['single_wake_local'],
            next_wake_UTC=estimate['single_wake_UTC'],basis=(here/'WAIT_ESTIMATE.json').relative_to(root).as_posix(),
            delete_on_wake=True,status='ACTIVE'),
        followup_status='SINGLE_MEASURED_WAIT',followup_automation_id='n01',followup_scheduled_local=estimate['single_wake_local'],
        followup_next_run_local=estimate['single_wake_local'],next_wake_local=estimate['single_wake_local'])
    d['joint_response_01'].update(original_draw_precision_fix01_status='running_confirmed_prefix_replay_pass',
        original_draw_precision_fix01_execution_counts_last_verified=counts,original_draw_precision_fix01_status_asof_UTC=raw['UTC'],
        original_draw_precision_fix01_prefix_replay=estimate['prefix_replay'],
        original_draw_precision_fix01_estimated_finish_local=estimate['estimated_finish_local'])
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
status='> 当前执行状态（2026-10-20）：C1原教师恢复与目标排序诊断已闭合。原生成器32draw诊断FIX01运行中，原5中心及9差分列复现通过；同样9列从546.6秒降至44.4秒。预计09:02:16完成，09:04单次接续。当前1科学作业、零参数更新；REAL响应目标未通过。'
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md'):
    path=root/name;lines=path.read_text(encoding='utf-8').splitlines()
    for index,line in enumerate(lines):
        if line.startswith('> 当前'):lines[index]=status;break
    if name=='README.md':
        lines=[line.replace('接下来必须复现首9列后继续','首9列联合残差和参数数组已逐值复现，native最大差8.33e-17') for line in lines]
    else:
        lines.extend(['','FIX01在08:53:12进度处完成原9列复现：联合残差/参数逐值相同，native最大差8.3267e-17；首9列实耗44.4097秒。按剩余99列、至多30次高draw候选检查、第二头设置及输出余量，预计09:02:15.665完成；[估时回执](receipts/joint_response_01_20261020/original_draw_precision_fix01/WAIT_ESTIMATE.json)登记09:04单次苏醒，实际余量104.335秒。进程身份及815源文件在08:53:49核验，后续执行状态待接续；独立终止审计脚本已准备，尚未执行终止数据。'])
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(receipt))
