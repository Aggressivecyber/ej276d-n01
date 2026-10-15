"""Record a saved healthy first-pair snapshot and the app-confirmed one-time wake."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BASE=OUT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

state=read(BASE/'readback_start_verification_01/REMOTE_STATE.json')
start=read(BASE/'readback_start_01/START_RECEIPT.json');proof=state['artifacts/FIRST_PAIR_PROOF.json']
cfg=read(OUT/'READBACK_EXECUTION.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
assert state['first_pair_verification']['first_formal_array_pair_verified'] is True
assert state['artifacts/FAILED.json'] is None
assert state['registration_sha256']==start['registration_sha256']==sha(OUT/'READBACK_EXECUTION.json')
launch=state['supervisor/launch.json'];supervisor=launch['supervisor_pid'];worker=launch['child_pid']
assert supervisor==start['supervisor_pid'] and worker==proof['pid']
assert state['current_boot_id']==start['boot_id'] and proof['scientific_CP_NN_AD_training']==0
assert proof['readback_script_sha256']==cfg['files']['readback_saved_arrays.py']['sha256']
assert proof['pure_kernel_sha256']==cfg['files']['saved_array_kernels.py']['sha256']
assert proof['inventory_sha256']==cfg['files']['BOUND_ORIGINAL_INVENTORY.json']['sha256']
assert proof['config_sha256']==cfg['original_configuration_sha256']
assert proof['source_manifest_sha256']==cfg['original_source_manifest_sha256']
assert proof['completed_pair_count']==1 and proof['actual_saved_array_pair_completed'] is True
durable=state['durable_progress'];counts=durable['completed_counts']
assert counts['bank_pairs_verified']>=1 and not durable['failed_ids'] and not durable['truncated_tail']
native=state['supervisor/exit_receipt.json']
if native is not None:
    assert native['child_native_exit_code']==0 and native['exit_code']==0
    assert state['artifacts/COMPLETE.json'] is not None
active=1 if native is None else 0
status='native_running_last_verified_first_pair_healthy' if active else 'completed_native0_pending_collection'
settings=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
a=tomllib.loads(settings.read_text(encoding='utf-8'));plan=read(OUT/'FOLLOWUP_PLAN.json')
assert a['id']=='n01' and a['kind']=='heartbeat' and a['status']=='ACTIVE'
assert a['rrule']==plan['rrule'] and a['name']==plan['name'] and a['target_thread_id']==plan['target_thread_id']
expected=(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8').rstrip('\r\n');assert a['prompt']==expected
now=dt.datetime.now(dt.timezone.utc)
assert dt.datetime.fromisoformat(plan['next_wake_UTC'])>now
follow=dict(schema='n01/wp3-readback-single-followup-confirmation/v1',UTC=now.isoformat(),automation_id='n01',kind='heartbeat',status='ACTIVE',count=1,
    next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],target_thread_id=a['target_thread_id'],
    actual_saved_settings_verified=True,settings_sha256=sha(settings),prepared_prompt_file_sha256=sha(OUT/'FOLLOWUP_PROMPT.txt'),
    actual_prompt_sha256=hashlib.sha256(a['prompt'].encode('utf-8')).hexdigest(),continuous_agent_monitoring=False,
    no_remote_calls_after_correct_start_snapshot=True,source_commit=source['source_commit'],requested_model='gpt-6-astra',requested_reasoning_effort='max')
write(OUT/'FOLLOWUP_RECEIPT.json',follow,True)
refs=dict(configuration=(OUT/'READBACK_EXECUTION.json').relative_to(ROOT).as_posix(),configuration_sha256=sha(OUT/'READBACK_EXECUTION.json'),
    source_commit=source['source_commit'],source_commit_pending=False,source_controls=6,run_id=cfg['run_id'],remote_root=cfg['remote_root'],
    status=status,supervisor_pid=supervisor,analysis_pid=worker,boot_id=start['boot_id'],first_formal_pair_verified=True,
    first_pair_verification=state['first_pair_verification'],last_verified_UTC=state['UTC'],
    actual_completed_counts_last_verified=counts,attempted_counts_last_verified=durable['entered_counts'],unknown_inflight_ids=durable['unknown_inflight_ids'],
    counts_are_durable_lower_bounds=True,actual_current_completion='unknown_after_single_saved_snapshot' if active else 'native0_at_saved_snapshot_collection_pending',
    native_child_exit_at_snapshot=None if active else native['child_native_exit_code'],native_supervisor_exit_at_snapshot=None if active else native['exit_code'],
    CP_NN_AD_training=0,resource_limits=cfg['resource_limits'],resource_snapshot=state['supervisor/resource_status.json'],
    start_receipt_sha256=sha(BASE/'readback_start_01/START_RECEIPT.json'),startup_snapshot_sha256=sha(BASE/'readback_start_verification_01/REMOTE_STATE.json'),
    startup_validation_sha256=sha(BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json'),followup=follow,
    independent_raw_readback_complete=False,original_raw_NPZ_transferred=False,scientific_status='response_match_not_established',scientific_acceptance=False)
write(OUT/'CORRECT_START_AND_FOLLOWUP.json',dict(schema='n01/wp3-readback-correct-start/v1',UTC=now.isoformat(),execution=refs,
    cumulative_training_updates=175667,original015_G_updates=8000,new_readback_training_updates=0,whole_project_complete=False),True)
paths=[ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup=OUT/'metadata_before_correct_start';backup.mkdir(exist_ok=False)
for p in paths:shutil.copy2(p,backup/p.name)
for p in [p for p in paths if p.suffix=='.json']:
    d=read(p);d.update(updated_UTC=now.isoformat(),active_scientific_jobs=active,followup=follow,
        active_scientific_jobs_status_basis='One saved first-pair healthy snapshot; subsequent state is not polled',
        scientific_status='response_match_not_established',latest_subagent_model_preference='GPT-6-Astra Max')
    if p.name=='jobs.json':
        job=next(x for x in d['jobs'] if x.get('run_id')==cfg['run_id']);job.update(status=status,execution=refs,
            worker_started=True,analysis_pid=worker,supervisor_pid=supervisor,actual_array_counts=counts,
            actual_array_count_basis='durable lower bounds at single startup snapshot; in-flight completion unknown',correct_start_verified=True,followup=follow)
    else:
        d.update(status='WP3_015_independent_readback_running_last_verified' if active else 'WP3_015_independent_readback_native0_pending_collection',
            wp3_readback_execution=refs,current_stage='WP3_015_independent_saved_array_readback',active_research=cfg['run_id'],
            current_work_package='WP3',latest_run_id=cfg['run_id'],current_run_training_updates=0,
            latest_training_run_id='wp3_015_20261015',latest_source_commit=source['source_commit'],new_training_updates=175667,cumulative_training_updates=175667,
            next_action='One scheduled wake to collect CPU015 readback, close015 and continue the six work packages under Astra Max.',
            followup_automation_id='n01',followup_automation_status='ACTIVE',followup_next_run_local=plan['next_wake_local'],
            followup_receipt=(OUT/'FOLLOWUP_RECEIPT.json').relative_to(ROOT).as_posix(),followup_receipt_sha256=sha(OUT/'FOLLOWUP_RECEIPT.json'))
        if p.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(x for x in d['work_packages'] if x['id']=='WP3');wp['saved_array_readback']=refs
            wp['status']='native0_selectively_collected_independent_saved_array_readback_running' if active else 'native0_independent_readback_native0_pending_collection'
            d['current_action']=d['next_action']
    write(p,d)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=text.replace('# N01 当前执行：WP3/015 保存数组独立复核准备','# N01 当前执行：WP3/015 保存数组独立复核已正确启动',1)
text=text.replace('当前仅已准备，启动以真实回执为准。',
    f"已保存首个正式完整pair与同snapshot健康证明；唯一快照完成{counts['bank_pairs_verified']}/36pair，在途ID {durable['unknown_inflight_ids']}。已确认{plan['next_wake_local']}（Asia/Singapore）单次苏醒，当前停止持续监控。施工、独审及后续调度均采用Astra Max。",1)
readme.write_text(text,encoding='utf-8',newline='\n')
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for p in [ROOT/'PROGRAM.json',paths[-1]]:
    d=read(p);d['master_plan_sha256']=master_sha
    if p==paths[-1]:d['plan_sha256']=sha(readme)
    write(p,d)
destination=Path('D:/GAN_backups/N01/wp3_015_20261015/readback_startup_01');assert not destination.exists();destination.mkdir(parents=True)
copy_paths=[*paths,OUT/'FOLLOWUP_PROMPT.txt',OUT/'FOLLOWUP_PLAN.json',OUT/'FOLLOWUP_RECEIPT.json',OUT/'CORRECT_START_AND_FOLLOWUP.json',
    OUT/'AGENT_MODEL_POLICY.json',OUT/'SOURCE_COMMIT_READBACK.json',OUT/'D_SOURCE_BACKUP_RECEIPT.json',
    BASE/'readback_start_01/START_RECEIPT.json',BASE/'readback_start_verification_01/REMOTE_STATE.json',BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json']
inventory={}
for i,p in enumerate(copy_paths):
    q=destination/f'{i:02d}_{p.name}';shutil.copy2(p,q);assert sha(q)==sha(p);inventory[q.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
manifest=destination/'SHA256SUMS.txt';manifest.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
write(OUT/'D_STARTUP_BACKUP_RECEIPT.json',dict(status='all_copied_files_SHA256_verified',destination=str(destination),files=len(inventory),
    bytes=sum(x['bytes'] for x in inventory.values()),manifest_sha256=sha(manifest)),True)
assert read(ROOT/'PROGRAM.json')['new_training_updates']==175667
print(json.dumps(dict(status='correct_start_one_wake_confirmed_rest',completed_pairs_at_only_snapshot=counts['bank_pairs_verified'],
    total_pairs=36,inflight_ids=durable['unknown_inflight_ids'],next_wake_local=plan['next_wake_local'],D_backup_files=len(inventory),whole_project_complete=False)))
