from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
registration=json.loads((here/'REGISTRATION.json').read_bytes())
dispatch=json.loads((here/'DISPATCH_RECEIPT.json').read_bytes())
snapshot=json.loads((here/'STARTUP_RAW.json').read_bytes())
p=json.loads((root/registration['config']).read_bytes())
assert snapshot['config_sha256']==dispatch['config_sha256']==registration['config_sha256']
assert hashlib.sha256((root/registration['config']).read_bytes()).hexdigest()==registration['config_sha256']
assert snapshot['source_files_verified']==819 and snapshot['boot_id']==dispatch['boot_id']
files=snapshot['files']; assert all(name not in files for name in ('control_01/exit_receipt.json','run_01/FAILED.json','run_01/COMPLETE.json'))
by_pid={v['pid']:v for v in snapshot['processes']}
launch=files['control_01/launch.json']; supervisor=launch['supervisor_pid']; timeout=launch['child_pid']
science=[v for v in snapshot['processes'] if len(v['argv'])>2 and v['argv'][2].endswith('/scripts/run_joint_response01_original_precision_real.py')]
assert len(science)==1; science=science[0]
assert set(by_pid)=={dispatch['dispatcher_pid'],supervisor,timeout,science['pid']}
assert by_pid[dispatch['dispatcher_pid']]['start_ticks']==dispatch['dispatcher_start_ticks']
resource=files['control_01/resource_status.json']; assert resource['peak_rss_bytes']<12*1024**3
progress=files['run_01/PROGRESS.json']; counts=progress['execution_counts']
assert counts['checkpoint_loads']==6 and counts['M0_head_rows']==13186
now=datetime.now(timezone.utc).isoformat()
assessment=dict(status='PASS_NATIVE_SETUP_AND_PROCESS_IDENTITY',UTC=now,status_asof_UTC=snapshot['UTC'],
    progress_asof_UTC=progress['UTC'],boot_id=snapshot['boot_id'],config_sha256=registration['config_sha256'],
    source_files_verified=819,source_pins=len(p['source_pins']),
    process_identity=[dict(pid=v['pid'],start_ticks=v['start_ticks']) for v in snapshot['processes']],
    science_pid=science['pid'],supervisor_pid=supervisor,execution_counts_last_verified=counts,
    resources=resource,scientific_acceptance=False,scientific_FD_progress_verified=counts.get('finite_difference_columns',0)>0)
(here/'STARTUP_ASSESSMENT.json').write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf-8')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((root/name).read_bytes())
    if name=='jobs.json':
        assert len(d['jobs'])==133 and d['jobs'][-1]['job_id']==p['job_id']
        d['jobs'][-1].update(status='running_native_setup_confirmed',started_UTC=dispatch['UTC'],
            science_pid=science['pid'],supervisor_pid=supervisor,dispatcher_pid=dispatch['dispatcher_pid'],
            boot_id=dispatch['boot_id'],execution_counts_last_verified=counts,status_asof_UTC=snapshot['UTC'],progress_asof_UTC=progress['UTC'])
    d.update(status='JOINT_RESPONSE_01_original_precision_REAL_pilot_running',active_scientific_jobs=1,
        pending_job_id=None,pending_next_run_id=None,active_job_id=p['job_id'],active_run_id=p['job_id'],
        latest_run_id=p['job_id'],training_pid=science['pid'],supervisor_pid=supervisor,updated_UTC=now,
        last_verified_UTC=snapshot['UTC'],execution_state_asof_UTC=snapshot['UTC'],scientific_acceptance=False,
        current_action='Original 32-draw three-round REAL pilot with matched physical controls is running.',
        next_action='Prepare terminal audit and estimate one wake from native progress; retain final gates and all kernel controls.')
    d['joint_response_01'].update(original_precision_REAL_pilot_started=True,
        original_precision_REAL_pilot_status='running_native_setup_confirmed',
        original_precision_REAL_pilot_startup_assessment=(here/'STARTUP_ASSESSMENT.json').relative_to(root).as_posix(),
        original_precision_REAL_pilot_execution_counts_last_verified=counts,original_precision_REAL_pilot_status_asof_UTC=snapshot['UTC'])
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(assessment))
