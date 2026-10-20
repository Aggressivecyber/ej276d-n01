from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
registration=json.loads((here/'REGISTRATION.json').read_bytes())
dispatch=json.loads((here/'DISPATCH_RECEIPT.json').read_bytes())
snapshot=json.loads((here/'WAIT_ESTIMATE_RAW.json').read_bytes())
p=json.loads((root/registration['config']).read_bytes())
assert snapshot['config_sha256']==dispatch['config_sha256']==registration['config_sha256']
assert hashlib.sha256((root/registration['config']).read_bytes()).hexdigest()==registration['config_sha256']
assert snapshot['source_files_verified']==805 and snapshot['boot_id']==dispatch['boot_id']
files=snapshot['files'];assert all(name not in files for name in ('control_01/exit_receipt.json','run_01/FAILED.json','run_01/COMPLETE.json'))
by_pid={v['pid']:v for v in snapshot['processes']}
launch=files['control_01/launch.json'];supervisor=launch['supervisor_pid'];timeout=launch['child_pid']
science=[v for v in snapshot['processes'] if len(v['argv'])>2 and v['argv'][2].endswith('/scripts/diagnose_joint01_original_draw_precision.py')]
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
    source_files_verified=805,source_pins=len(p['source_pins']),process_identity=[dict(pid=v['pid'],start_ticks=v['start_ticks']) for v in snapshot['processes']],
    science_pid=science['pid'],supervisor_pid=supervisor,execution_counts_last_verified=counts,
    resources=resource,original_M0_replay_pass=True,nested_draw_replay_pass=True,
    parameter_updates=0,scientific_acceptance=False)
(here/'STARTUP_ASSESSMENT.json').write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf-8',newline='\n')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        assert len(d['jobs'])==131 and d['jobs'][-1]['job_id']==p['job_id']
        d['jobs'][-1].update(status='running_confirmed',started_UTC=dispatch['UTC'],science_pid=science['pid'],supervisor_pid=supervisor,
            dispatcher_pid=dispatch['dispatcher_pid'],boot_id=dispatch['boot_id'],execution_counts_last_verified=counts,
            status_asof_UTC=snapshot['UTC'],progress_asof_UTC=progress['UTC'])
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_running',active_scientific_jobs=1,
        pending_job_id=None,pending_next_run_id=None,active_job_id=p['job_id'],active_run_id=p['job_id'],
        training_pid=science['pid'],supervisor_pid=supervisor,updated_UTC=now,last_verified_UTC=snapshot['UTC'],
        execution_state_asof_UTC=snapshot['UTC'],scientific_acceptance=False,
        current_action='Original-generator nested 32-draw direction diagnostic running; first five frozen centers and nested banks verified.',
        next_action='Audit terminal joint/native directions and paired four-versus-32-draw checks; no parameter updates.')
    d['joint_response_01'].update(original_draw_precision_started=True,original_draw_precision_status='running_confirmed',
        original_draw_precision_startup_assessment=(here/'STARTUP_ASSESSMENT.json').relative_to(root).as_posix(),
        original_draw_precision_execution_counts_last_verified=counts,original_draw_precision_status_asof_UTC=snapshot['UTC'])
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
status='> 当前执行状态（2026-10-20）：原教师C1恢复36/36硬门、16/18精度门；固定端点目标排序诊断已闭合。现按原JointResponse01登记4→32draw嵌套REAL方向诊断，首5库原中心及样本子集复现通过；零参数更新。REAL响应目标仍未通过。'
path=root/'JOINT_RESPONSE_01_PLAN.md';lines=path.read_text(encoding='utf-8').splitlines()
lines[2]=status
lines.extend(['','原C1直接教师恢复的[终止审核](receipts/joint_response_01_20261020/c1_original_precision_01/terminal_01/TERMINAL_REVIEW.json)为36/36硬门、16/18精度门；seed1两库条件均值误差5.7628%/5.7221%超过原5%门。[冻结端点排序诊断](receipts/joint_response_01_20261020/c1_objective_alignment_01/terminal_01/TERMINAL_REVIEW.json)在24个配对视图中均显示原目标偏好条件响应较差端点，matched-C1目标排序全部反转。该证据支持停止沿同一跨家族目标继续校正C1；已有C1导数替代原生成器导数的负结果保留。',
    '', '已前瞻登记[原生成器4→32draw方向诊断](docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_20261020.md)。保留已通过教师恢复的原JointResponse01，固定旧M0三FIT诊断的SOURCE/REAL行、投影、完整联合/native求解与门；新增28draw并逐值保留旧4draw，两头各至多3候选，零提交更新。805部署文件及首5组旧中心/嵌套库检查通过，运行身份见[启动审核](receipts/joint_response_01_20261020/original_draw_precision_01/STARTUP_ASSESSMENT.json)。科学源代码冻结，按本次实际列速度估算单次苏醒；所有旧失败与原最终改善门保持。'])
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
path=root/'README.md';lines=path.read_text(encoding='utf-8').splitlines()
for index,line in enumerate(lines):
    if line.startswith('> 当前'):
        lines[index]=status;break
else:lines.insert(2,status)
path.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(assessment))
