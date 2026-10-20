from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tomllib

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
request=json.loads((here/'AUTOMATION_CREATE_REQUEST.json').read_bytes())
result=json.loads((here/'AUTOMATION_CREATE_TOOL_RESULT.json').read_bytes())
estimate=json.loads((here/'WAIT_ESTIMATE.json').read_bytes())
raw=json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
registration=json.loads((here/'REGISTRATION.json').read_bytes())
meta=json.loads(result['content'][1]['text']); aid=meta['automationId']
assert not result['isError'] and meta['mode']=='create' and meta['status']=='ACTIVE'
configuration_path=Path('C:/Users/10629/.codex/automations')/aid/'automation.toml'
config_raw=configuration_path.read_bytes(); configuration=tomllib.loads(config_raw.decode('utf-8'))
for field in ('kind','name','prompt','status','rrule'):assert configuration[field]==request[field],field
assert configuration['id']==aid and configuration['target_thread_id']==request['targetThreadId']
assert 'COUNT=1' in configuration['rrule']
assert not Path('C:/Users/10629/.codex/automations/n01/automation.toml').exists()
(here/'AUTOMATION_CONFIGURATION.toml').write_bytes(config_raw)
now=datetime.now(timezone.utc).isoformat()
receipt=dict(status='PASS_SINGLE_WAKE_CREATED_AND_READ_BACK',UTC=now,automation_id=aid,kind='heartbeat',count=1,
    next_wake_local=estimate['single_wake_local'],next_wake_UTC=estimate['single_wake_UTC'],
    target_thread_id=request['targetThreadId'],configuration_sha256=hashlib.sha256(config_raw).hexdigest(),
    prompt_matches_request=True,previous_n01_automation_absent=True,delete_or_confirm_consumed_on_wake=True,scientific_acceptance=False)
(here/'AUTOMATION_CREATE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
progress=raw['files']['run_01/PROGRESS.json'];counts=progress['execution_counts']
assert counts['finite_difference_columns']==45 and counts.get('REAL_response_updates',0)==0
startup=json.loads((here/'STARTUP_ASSESSMENT.json').read_bytes())
assert raw['boot_id']==startup['boot_id']
assert {(x['pid'],x['start_ticks']) for x in raw['processes']}=={(x['pid'],x['start_ticks']) for x in startup['process_identity']}
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes());assert d['active_scientific_jobs']==1 and d['active_job_id']==registration['job_id']
    if name=='jobs.json':
        assert len(d['jobs'])==133
        d['jobs'][-1].update(status='running_confirmed_45_FD_columns',execution_counts_last_verified=counts,
            status_asof_UTC=raw['UTC'],progress_asof_UTC=progress['UTC'],estimated_finish_local=estimate['estimated_finish_local'])
    d.update(status='JOINT_RESPONSE_01_original_precision_REAL_pilot_running',updated_UTC=now,
        execution_state_asof_UTC=raw['UTC'],last_verified_UTC=raw['UTC'],latest_run_id=registration['job_id'],
        current_job_native_exit_code=None,current_job_ended_UTC=None,current_run_training_updates=0,
        current_action='32-draw matched REAL pilot confirmed at 45 of 600 finite-difference columns; 10:13 single continuation scheduled.',
        next_action='At measured one-time wake, collect and audit all four tracks, original 192 final gates and same-parent kernels before deciding the next step.',
        followup=dict(kind='heartbeat',automation_id=aid,count=1,next_wake_local=estimate['single_wake_local'],
            next_wake_UTC=estimate['single_wake_UTC'],basis=(here/'WAIT_ESTIMATE.json').relative_to(root).as_posix(),
            delete_on_wake=True,status='ACTIVE'),followup_status='SINGLE_MEASURED_WAIT',
        followup_automation_status='ACTIVE',followup_automation_id=aid,followup_scheduled_local=estimate['single_wake_local'],
        followup_next_run_local=estimate['single_wake_local'],next_wake_local=estimate['single_wake_local'],
        followup_receipt=(here/'AUTOMATION_CREATE_RECEIPT.json').relative_to(root).as_posix())
    d['joint_response_01'].update(original_precision_REAL_pilot_status='running_confirmed_45_FD_columns',
        original_precision_REAL_pilot_execution_counts_last_verified=counts,original_precision_REAL_pilot_status_asof_UTC=raw['UTC'],
        original_precision_REAL_pilot_estimated_finish_local=estimate['estimated_finish_local'])
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
status='> 当前执行状态（2026-10-20）：原生成器32draw局部诊断已完成并独立审核，两头FIT/CHECK/FRESH通过。三轮REAL校准与匹配物理比较已启动，09:15核验45/600差分列，预计10:05:24完成，10:13单次接续。当前1科学作业；REAL响应匹配尚未建立。'
for name in ('README.md','JOINT_RESPONSE_01_PLAN.md'):
    path=root/name;lines=path.read_text('utf-8').splitlines()
    for index,line in enumerate(lines):
        if line.startswith('> 当前'):lines[index]=status
    if name=='README.md':
        lines[0]='# N01 当前阶段：原生成器32draw三轮REAL校准运行中'
        for index,line in enumerate(lines):
            if line.startswith('当前唯一科学作业'):
                lines[index]='当前唯一科学作业是[32draw三轮REAL校准](docs/JOINT_RESPONSE_01_ORIGINAL_PRECISION_REAL_PILOT_20261020.md)，09:11派发。两神经从M0零增量、两物理从原物理初值开始，各3轮，完整覆盖6465拟合父和18948REAL行；匹配数据/随机流，原全部局部及最终门不变。819源文件、23输入及原进程身份核验通过。09:15快照为45/600差分列、尚无候选更新；后续状态待接续。两库硬报告与128×1024同父核无论是否通过均执行，原REAL失败与阶段5未完成状态保持。'
    else:
        lines.extend(['','已前瞻登记并启动[32draw三轮循环校准与匹配物理比较](docs/JOINT_RESPONSE_01_ORIGINAL_PRECISION_REAL_PILOT_20261020.md)，配置SHA256 `3272ef3806a6293d004a21859886d74d0de8f52137a4e0ae7c4907e64eed6445`。每条轨迹三轮9216父访问/36864REAL访问完整覆盖拟合池，不使用报告角色选择终点；全部原最终门和两库同父核保持。819源文件/23输入已冻结，09:15核验原进程、45差分列及RSS峰5,139,763,200字节。',
            '', '按本次229.877秒/278训练生成调用及余下最多3562调用，加同规格每头30.407秒末端报告与输出余量，[预计10:05:23.908完成](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/WAIT_ESTIMATE.json)；[10:13单次苏醒已读回核验](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/AUTOMATION_CREATE_RECEIPT.json)，留456秒余量。内部10:41:14/外部最晚10:46:34预算保持，不自动加轮或重启。终止收集及四轨迹完整复算脚本已准备；独立求解核心对54/46维保存方向核验差<2.26e-17，本次终止数据尚未审核。'])
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(receipt))
