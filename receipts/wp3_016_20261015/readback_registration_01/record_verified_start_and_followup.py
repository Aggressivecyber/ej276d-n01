"""Record the one healthy CPU startup snapshot and app-confirmed future wake."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write(p,value,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

config_path=OUT/'READBACK_EXECUTION.json'
config=read(config_path)
source=read(OUT/'SOURCE_COMMIT_READBACK.json')
state_path=BASE/'readback_start_verification_01/REMOTE_STATE.json'
state=read(state_path)
start_path=BASE/'readback_start_01/START_RECEIPT.json'
start=read(start_path)
assert state['supervisor/exit_receipt.json'] is None, 'Terminal run must be collected instead.'
assert state['registration_sha256']==start['registration_sha256']==sha(config_path)
spec=importlib.util.spec_from_file_location('root016_readback_first_valid',config['files']['readback_control.py']['local_path'])
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
assert control.first_valid(state,config)==state['first_pair_verification']
assert state['first_pair_verification']['same_snapshot_health']=='current_owned_process_chain_and_resources_verified'
proof=state['artifacts/FIRST_PAIR_PROOF.json']
durable=state['durable_progress'];counts=durable['completed_counts']
assert counts['bank_pairs_verified']>=1 and counts['charge_npz_files_decoded']>=2
assert durable['failed_ids']==[] and durable['truncated_tail'] is False
assert proof['scientific_CP_NN_AD_training']==0
assert config['cumulative_training_before_and_after_this_audit']==179667
assert all(v==0 for v in config['scientific_budget'].values())
for name,row in source['files'].items():
    assert sha(ROOT/name)==row['sha256']
summary_path=BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json'
summary=read(summary_path)
assert summary_path.is_file()
plan=read(OUT/'FOLLOWUP_PLAN.json')
settings_path=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
settings=tomllib.loads(settings_path.read_text(encoding='utf-8'))
assert settings['id']=='n01' and settings['kind']=='heartbeat' and settings['status']=='ACTIVE'
assert settings['name']==plan['name'] and settings['rrule']==plan['rrule']
assert settings['target_thread_id']==plan['target_thread_id']
prompt=(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8').rstrip('\r\n')
assert settings['prompt']==prompt and hashlib.sha256(prompt.encode()).hexdigest()==plan['prompt_sha256']
now=dt.datetime.now(dt.timezone.utc)
assert dt.datetime.fromisoformat(plan['next_wake_UTC'])>now
follow=dict(schema='n01/wp3-016-CPU-single-followup-confirmation/v1',UTC=now.isoformat(),automation_id='n01',
    kind='heartbeat',status='ACTIVE',count=1,name=plan['name'],next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],
    target_thread_id=settings['target_thread_id'],actual_saved_settings_verified=True,settings_sha256=sha(settings_path),
    prepared_prompt_file_sha256=sha(OUT/'FOLLOWUP_PROMPT.txt'),actual_prompt_sha256=plan['prompt_sha256'],
    continuous_agent_monitoring=False,no_remote_calls_after_correct_start_snapshot=True,
    requested_model='gpt-6.1-sol',requested_reasoning_effort='max',source_commit=source['source_commit'])
write(OUT/'FOLLOWUP_RECEIPT.json',follow,True)
launch=state['supervisor/launch.json']
refs=dict(configuration=config_path.relative_to(ROOT).as_posix(),configuration_sha256=sha(config_path),
    source_commit=source['source_commit'],source_commit_pending=False,source_controls=6,run_id=config['run_id'],remote_root=config['remote_root'],
    status='native_running_last_verified_first_pair_healthy',supervisor_pid=launch['supervisor_pid'],analysis_pid=launch['child_pid'],
    boot_id=start['boot_id'],first_formal_pair_verified=True,first_pair_verification=state['first_pair_verification'],last_verified_UTC=state['UTC'],
    actual_completed_counts_last_verified=counts,attempted_counts_last_verified=durable['entered_counts'],unknown_inflight_ids=durable['unknown_inflight_ids'],
    counts_are_durable_lower_bounds=True,actual_current_completion='unknown_after_single_saved_snapshot',
    CP_NN_AD_training=0,resource_limits=config['resource_limits'],resource_snapshot=state['supervisor/resource_status.json'],
    start_receipt_sha256=sha(start_path),startup_snapshot_sha256=sha(state_path),startup_validation_sha256=sha(summary_path),
    source_commit_readback=dict(path=(OUT/'SOURCE_COMMIT_READBACK.json').relative_to(ROOT).as_posix(),sha256=sha(OUT/'SOURCE_COMMIT_READBACK.json')),
    D_source_backup=dict(path=(OUT/'D_SOURCE_BACKUP_RECEIPT.json').relative_to(ROOT).as_posix(),sha256=sha(OUT/'D_SOURCE_BACKUP_RECEIPT.json')),
    followup=follow,independent_raw_readback_complete=False,original_raw_NPZ_transferred=False,
    scientific_status='response_match_not_established',scientific_acceptance=False)
write(OUT/'CORRECT_START_AND_FOLLOWUP.json',dict(schema='n01/wp3-016-CPU-correct-start/v1',UTC=now.isoformat(),execution=refs,
    cumulative_training_updates=179667,original016_G_updates=4000,new_readback_training_updates=0,whole_project_complete=False),True)
paths=[ROOT/n for n in ['PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md']]+[CURRENT]
before=OUT/'metadata_before_correct_start';before.mkdir(exist_ok=False)
for path in paths:shutil.copy2(path,before/path.name)
for path in [p for p in paths if p.suffix=='.json']:
    value=read(path)
    value.update(updated_UTC=now.isoformat(),active_scientific_jobs=1,followup=follow,
        active_scientific_jobs_status_basis='One saved first-pair healthy snapshot; subsequent state not polled',
        scientific_status='response_match_not_established',latest_subagent_model_preference='GPT-6.1-Sol Max')
    if path.name=='jobs.json':
        job=next(j for j in value['jobs'] if j.get('run_id')==config['run_id'])
        job.update(status=refs['status'],execution=refs,worker_started=True,analysis_pid=launch['child_pid'],
            supervisor_pid=launch['supervisor_pid'],actual_array_counts=counts,
            actual_array_count_basis='Durable lower bounds at the only startup snapshot; in-flight completion unknown',
            correct_start_verified=True,followup=follow)
    else:
        assert value['new_training_updates']==179667
        value.update(status='WP3_016_independent_readback_running_last_verified',wp3_readback_execution=refs,
            wp3_016_readback_execution=refs,current_stage='WP3_016_independent_saved_array_readback',active_research=config['run_id'],
            active_audit=config['run_id'],current_work_package='WP3',latest_run_id=config['run_id'],current_run_training_updates=0,
            latest_training_run_id='wp3_016_20261015',latest_source_commit=source['source_commit'],new_training_updates=179667,cumulative_training_updates=179667,
            next_action='One scheduled wake to collect CPU016 readback, close016/report, then continue WP4-WP6 with Sol Max.',
            followup_automation_id='n01',followup_automation_status='ACTIVE',followup_next_run_local=plan['next_wake_local'],
            followup_receipt=(OUT/'FOLLOWUP_RECEIPT.json').relative_to(ROOT).as_posix(),followup_receipt_sha256=sha(OUT/'FOLLOWUP_RECEIPT.json'))
        if path.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            wp.update(saved_array_readback=refs,readback016=refs,status='native0_independent_saved_array_readback_running')
            value['current_action']=value['next_action']
    write(path,value)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=text.replace('# N01 当前执行：WP3/016 独立保存数组复核已登记','# N01 当前执行：WP3/016 独立保存数组复核已正确启动',1)
text=text.replace('当前尚未启动，启动以真实回执为准。',
    f"已保存首个正式完整pair及同snapshot当前健康；唯一快照完成{counts['bank_pairs_verified']}/24pair、{counts['charge_npz_files_decoded']}/8charge，在途ID {durable['unknown_inflight_ids']}。已确认{plan['next_wake_local']}（Asia/Singapore）单次苏醒，当前停止持续监控；规划、施工、独审和后续调度采用Sol Max。",1)
readme.write_text(text,encoding='utf-8',newline='\n')
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for path in [ROOT/'PROGRAM.json',CURRENT]:
    value=read(path);value['master_plan_sha256']=master_sha
    if path==CURRENT:value['plan_sha256']=sha(readme)
    write(path,value)
destination=Path('D:/GAN_backups/N01/wp3_016_20261015/readback_startup_01')
destination.mkdir(parents=True,exist_ok=False)
copies=[*paths,OUT/'FOLLOWUP_PROMPT.txt',OUT/'FOLLOWUP_PLAN.json',OUT/'FOLLOWUP_RECEIPT.json',OUT/'CORRECT_START_AND_FOLLOWUP.json',
    OUT/'SOURCE_COMMIT_READBACK.json',OUT/'D_SOURCE_BACKUP_RECEIPT.json',OUT/'SOURCE_BINDING_RECEIPT.json',
    start_path,state_path,summary_path,Path(__file__).resolve(),OUT/'prepare_followup.py',OUT/'FOLLOWUP_PROMPT_TEMPLATE.txt']
inventory={}
assert shutil.disk_usage(destination).free>sum(p.stat().st_size for p in copies)+16*1024**2
for index,path in enumerate(copies):
    target=destination/f'{index:02d}_{path.name}';shutil.copy2(path,target)
    assert sha(target)==sha(path)
    inventory[target.name]=dict(bytes=target.stat().st_size,sha256=sha(target))
write(destination/'MANIFEST.json',dict(UTC=now.isoformat(),files=inventory),True)
(destination/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+n+'\n' for n,r in sorted(inventory.items())),encoding='utf-8',newline='\n')
write(OUT/'D_STARTUP_BACKUP_RECEIPT.json',dict(status='all_copied_files_SHA256_verified',destination=str(destination),
    files=len(inventory),bytes=sum(r['bytes'] for r in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json')),True)
assert read(ROOT/'PROGRAM.json')['new_training_updates']==179667
print(json.dumps(dict(status='correct_start_one_wake_confirmed_rest',completed_pairs_at_only_snapshot=counts['bank_pairs_verified'],
    charge_banks_at_only_snapshot=counts['charge_npz_files_decoded'],total_pairs=24,inflight_ids=durable['unknown_inflight_ids'],
    next_wake_local=plan['next_wake_local'],D_backup_files=len(inventory),whole_project_complete=False)))
