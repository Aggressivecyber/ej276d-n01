from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
registration=json.loads((here/'REGISTRATION.json').read_bytes())
dispatch=json.loads((here/'DISPATCH_RECEIPT.json').read_bytes())
snapshot=json.loads((here/'STARTUP_RAW.json').read_bytes())
p=json.loads((root/registration['config']).read_bytes())
assert snapshot['config_sha256']==dispatch['config_sha256']==registration['config_sha256']
assert hashlib.sha256((root/registration['config']).read_bytes()).hexdigest()==registration['config_sha256']
assert snapshot['source_files_verified']==815 and snapshot['boot_id']==dispatch['boot_id']
files=snapshot['files'];assert all(name not in files for name in ('control_01/exit_receipt.json','run_01/FAILED.json','run_01/COMPLETE.json'))
by_pid={v['pid']:v for v in snapshot['processes']}
launch=files['control_01/launch.json'];supervisor=launch['supervisor_pid'];timeout=launch['child_pid']
science=[v for v in snapshot['processes'] if len(v['argv'])>2 and v['argv'][2].endswith('/scripts/diagnose_joint01_original_draw_precision_fix01.py')]
assert len(science)==1
science=science[0]
assert set(by_pid)=={dispatch['dispatcher_pid'],supervisor,timeout,science['pid']}
assert by_pid[dispatch['dispatcher_pid']]['start_ticks']==dispatch['dispatcher_start_ticks']
resource=files['control_01/resource_status.json']
assert resource['supervisor_pid']==supervisor and resource['child_pid']==timeout and resource['peak_rss_bytes']<12*1024**3
progress=files['run_01/PROGRESS.json'];counts=progress['execution_counts']
assert counts['M0_zero_signal_replays_verified']==2 and counts['low_draw_center_replays_verified']>=5
assert counts['nested_banks_verified']>=5 and counts.get('REAL_response_updates',0)==0
now=datetime.now(timezone.utc).isoformat()
assessment=dict(status='PASS_NATIVE_ORIGINAL_DRAW_PRECISION_STARTUP',UTC=now,status_asof_UTC=snapshot['UTC'],
    progress_asof_UTC=progress['UTC'],boot_id=snapshot['boot_id'],config_sha256=registration['config_sha256'],
    source_files_verified=815,source_pins=len(p['source_pins']),process_identity=[dict(pid=v['pid'],start_ticks=v['start_ticks']) for v in snapshot['processes']],
    science_pid=science['pid'],supervisor_pid=supervisor,execution_counts_last_verified=counts,
    resources=resource,original_M0_replay_pass=True,nested_draw_replay_pass=True,same_32draw_centers_verified=counts['same_32draw_centers_verified'],
    parameter_updates=0,scientific_acceptance=False)
(here/'STARTUP_ASSESSMENT.json').write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf-8',newline='\n')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        assert len(d['jobs'])==132 and d['jobs'][-1]['job_id']==p['job_id']
        d['jobs'][-1].update(status='running_confirmed',started_UTC=dispatch['UTC'],science_pid=science['pid'],supervisor_pid=supervisor,
            dispatcher_pid=dispatch['dispatcher_pid'],boot_id=dispatch['boot_id'],execution_counts_last_verified=counts,
            status_asof_UTC=snapshot['UTC'],progress_asof_UTC=progress['UTC'])
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_fix01_running',active_scientific_jobs=1,
        pending_job_id=None,pending_next_run_id=None,active_job_id=p['job_id'],active_run_id=p['job_id'],
        training_pid=science['pid'],supervisor_pid=supervisor,updated_UTC=now,last_verified_UTC=snapshot['UTC'],
        execution_state_asof_UTC=snapshot['UTC'],scientific_acceptance=False,
        current_action='Original-generator nested 32-draw direction diagnostic running; first five frozen centers and nested banks verified.',
        next_action='Audit terminal joint/native directions and paired four-versus-32-draw checks; no parameter updates.')
    d['joint_response_01'].update(original_draw_precision_fix01_started=True,original_draw_precision_fix01_status='running_confirmed',
        original_draw_precision_fix01_startup_assessment=(here/'STARTUP_ASSESSMENT.json').relative_to(root).as_posix(),
        original_draw_precision_fix01_execution_counts_last_verified=counts,original_draw_precision_fix01_status_asof_UTC=snapshot['UTC'])
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
status='> 当前执行状态（2026-10-20）：C1原教师恢复及目标排序诊断已闭合。原生成器32draw诊断因实测速度超出预算预测而保存9列停止；精确8:1距离计算FIX01已单独启动，首5组32draw中心复现通过。当前1科学作业、零参数更新；REAL响应目标未通过。'
path=root/'JOINT_RESPONSE_01_PLAN.md';lines=path.read_text(encoding='utf-8').splitlines();lines[2]=status
lines.extend(['','原32draw诊断[保存前缀审核](receipts/joint_response_01_20261020/original_draw_precision_01/terminal_01/TERMINAL_REVIEW.json)确认原生exit -15、9列及5组中心完整保留，零更新。首9列实耗546.638秒，108列预计6559.657秒，超过原5400秒预算；末次快照之后0—9列可能完成而未落盘的费用区间保留。这是运行可行性停止，未形成科学失败判定。',
'','[FIX01协议](docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020.md)使用精确均匀8:1次序统计量W1计算，原模型、事件、权重、分箱支持、距离定义和科学门保持。815文件/35锁定输入部署通过；原5组32draw中心复现，首9列还必须通过旧保存证据对照才继续。内部5400秒/外部5700秒预算另行登记，旧作业不重启、不延长。'])
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
path=root/'README.md';lines=path.read_text(encoding='utf-8').splitlines();lines[0]='# N01 当前阶段：原生成器32draw联合方向诊断FIX01运行中'
for index,line in enumerate(lines):
    if line.startswith('> 当前'):lines[index]=status
    if line.startswith('当前唯一科学作业为'):
        lines[index]='当前唯一科学作业是[原生成器32draw计算修复FIX01](docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_FIX01_20261020.md)，08:52派发。前项首9列用了546.638秒，预计108列需6559.657秒，超过原5400秒预算，已保存前缀并停止，零参数更新。FIX01在距离函数内部使用精确均匀8:1次序统计量公式，保留原事件数、分箱支持、模型、投影、联合求解及全部门；815部署文件通过校验，首5组32draw中心与原保存值一致，接下来必须复现首9列后继续。原REAL两头各24/96改善门失败及阶段5未完成状态保持。'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(assessment))
