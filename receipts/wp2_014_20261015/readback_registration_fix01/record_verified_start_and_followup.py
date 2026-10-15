"""Record saved first-pair health and the one app-confirmed wake; no SSH."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import tomllib

ROOT=Path('E:/GAN/research/N01');BASE=ROOT/'receipts/wp2_014_20261015';OUT=BASE/'readback_registration_fix01'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
state=read(BASE/'readback_start_verification_01/REMOTE_STATE.json')
start=read(BASE/'readback_start_01/START_RECEIPT.json');proof=state['artifacts/FIRST_PAIR_PROOF.json']
cfg=read(OUT/'READBACK_EXECUTION.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
assert state['first_pair_verification']['first_formal_array_pair_verified']
assert state['artifacts/FAILED.json'] is None and state['supervisor/exit_receipt.json'] is None
assert state['registration_sha256']==start['registration_sha256']==sha(OUT/'READBACK_EXECUTION.json')
assert start['supervisor_pid']==137650 and proof['pid']==137651 and proof['scientific_CP_NN_AD_training']==0
assert proof['readback_script_sha256']==cfg['files']['readback_saved_arrays.py']['sha256']
durable=state['durable_progress'];counts=durable['completed_counts']
assert counts['bank_pairs_verified']==27 and not durable['failed_ids'] and not durable['truncated_tail']
assert durable['unknown_inflight_ids']==[277,279]
settings=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
a=tomllib.loads(settings.read_text(encoding='utf-8'));plan=read(OUT/'FOLLOWUP_PLAN.json')
assert a['id']=='n01' and a['kind']=='heartbeat' and a['status']=='ACTIVE'
assert a['rrule']=='FREQ=DAILY;COUNT=1;BYHOUR=11;BYMINUTE=17;BYSECOND=0'
assert a['name']==plan['name'] and a['target_thread_id']==plan['target_thread_id']
expected=(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8').rstrip('\r\n')
assert a['prompt']==expected
now=dt.datetime.now(dt.timezone.utc).isoformat()
follow=dict(schema='n01/wp2-readback-single-followup-confirmation/v1',UTC=now,
    automation_id='n01',kind='heartbeat',status='ACTIVE',count=1,
    next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],
    target_thread_id=a['target_thread_id'],actual_saved_settings_verified=True,
    settings_sha256=sha(settings),prepared_prompt_file_sha256=sha(OUT/'FOLLOWUP_PROMPT.txt'),
    actual_prompt_sha256=hashlib.sha256(a['prompt'].encode('utf-8')).hexdigest(),
    prepared_trailing_linebreak_removed_before_tool=True,continuous_agent_monitoring=False,
    no_remote_calls_after_correct_start_snapshot=True,source_commit=source['source_commit'])
write(OUT/'FOLLOWUP_RECEIPT.json',follow,True)
refs=dict(configuration=str((OUT/'READBACK_EXECUTION.json').relative_to(ROOT)),configuration_sha256=sha(OUT/'READBACK_EXECUTION.json'),
    source_commit=source['source_commit'],source_commit_pending=False,source_controls=6,
    run_id=cfg['run_id'],remote_root=cfg['remote_root'],status='native_running_last_verified_first_pair_healthy',
    supervisor_pid=137650,analysis_pid=137651,boot_id=start['boot_id'],first_formal_pair_verified=True,
    last_verified_UTC=state['UTC'],actual_completed_counts_last_verified=counts,
    attempted_counts_last_verified=durable['entered_counts'],unknown_inflight_ids=[277,279],
    counts_are_durable_lower_bounds=True,actual_current_completion='unknown_after_single_saved_snapshot',
    native_child_exit_at_snapshot=None,native_supervisor_exit_at_snapshot=None,
    CP_NN_AD_training=0,resource_limits=cfg['resource_limits'],resource_snapshot=state['supervisor/resource_status.json'],
    start_receipt_sha256=sha(BASE/'readback_start_01/START_RECEIPT.json'),
    startup_snapshot_sha256=sha(BASE/'readback_start_verification_01/REMOTE_STATE.json'),
    startup_validation_sha256=sha(BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json'),
    followup=follow,independent_raw_readback_complete=False,original_raw_NPZ_transferred=False,
    scientific_status='response_match_not_established',scientific_acceptance=False)
write(OUT/'CORRECT_START_AND_FOLLOWUP.json',dict(schema='n01/wp2-readback-correct-start/v1',UTC=now,
    execution=refs,cumulative_training_updates=167667,original014_G_updates=8000,
    new_readback_training_updates=0,whole_project_complete=False),True)
paths=[ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
snapshot=OUT/'metadata_before_correct_start';snapshot.mkdir()
for p in paths:shutil.copy2(p,snapshot/p.name)
for p in [p for p in paths if p.suffix=='.json']:
    d=read(p);d.update(updated_UTC=now,active_scientific_jobs=1,followup=follow,
        active_scientific_jobs_status_basis='One saved first-pair healthy snapshot; later native completion unknown until scheduled wake',
        scientific_status='response_match_not_established')
    if p.name=='jobs.json':
        job=next(x for x in d['jobs'] if x.get('run_id')==cfg['run_id']);job.update(status=refs['status'],execution=refs,
            worker_started=True,analysis_pid=137651,supervisor_pid=137650,actual_array_counts=counts,
            actual_array_count_basis='durable lower bounds at single startup snapshot; in-flight completion unknown',
            correct_start_verified=True,followup=follow)
    else:
        d.update(status='WP2_014_independent_readback_running_last_verified',wp2_readback_execution=refs,
            current_stage='WP2_014_independent_saved_array_readback',active_research=cfg['run_id'],
            current_work_package='WP2',latest_run_id=cfg['run_id'],current_run_training_updates=0,
            latest_training_run_id='wp2_014_20261015',latest_source_commit=source['source_commit'],
            next_action='One scheduled wake to collect the CPU readback, close WP2 and continue bounded WP3.',
            followup_automation_id='n01',followup_automation_status='ACTIVE',followup_next_run_local=plan['next_wake_local'],
            followup_receipt=str((OUT/'FOLLOWUP_RECEIPT.json').relative_to(ROOT)),followup_receipt_sha256=sha(OUT/'FOLLOWUP_RECEIPT.json'))
        if p.name=='PROGRAM.json':d['new_training_updates']=167667
        if p.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(x for x in d['work_packages'] if x['id']=='WP2');wp['saved_array_readback']=refs
            wp['status']='native0_selectively_collected_independent_saved_array_readback_running'
            d['current_action']=d['next_action']
    write(p,d)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=text.replace('# N01 当前执行：WP2/014 保存数组独立复核准备','# N01 当前执行：WP2/014 保存数组独立复核已正确启动',1)
text=text.replace('当前仅已准备，真实启动以回执为准。','已保存首个正式完整pair与同snapshot健康证明。唯一快照完成27/36pair，仍有两项在途，原生终点未知；已确认2026-10-15 11:17（Asia/Singapore）单次苏醒，当前停止持续监控。',1)
readme.write_text(text,encoding='utf-8',newline='\n');selector=paths[-1];d=read(selector);d['plan_sha256']=sha(readme);write(selector,d)
destination=Path('D:/GAN_backups/N01/wp2_014_20261015/readback_startup_fix01');assert not destination.exists();destination.mkdir(parents=True)
copy_paths=[*paths,OUT/'FOLLOWUP_PROMPT.txt',OUT/'FOLLOWUP_PLAN.json',OUT/'FOLLOWUP_RECEIPT.json',OUT/'CORRECT_START_AND_FOLLOWUP.json',
    BASE/'readback_start_01/START_RECEIPT.json',BASE/'readback_start_verification_01/REMOTE_STATE.json',BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json']
inventory={}
for i,p in enumerate(copy_paths):
    q=destination/f'{i:02d}_{p.name}';shutil.copy2(p,q);assert sha(q)==sha(p);inventory[q.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
(destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
write(OUT/'D_STARTUP_BACKUP_RECEIPT.json',dict(status='all_copied_files_SHA256_verified',destination=str(destination),
    files=len(inventory),bytes=sum(x['bytes'] for x in inventory.values()),manifest_sha256=sha(destination/'SHA256SUMS.txt')),True)
assert read(ROOT/'PROGRAM.json')['new_training_updates']==167667
print(json.dumps(dict(status='correct_start_one_wake_confirmed_rest',completed_pairs_at_only_snapshot=27,
    total_pairs=36,inflight_ids=[277,279],next_wake_local=plan['next_wake_local'],D_backup_files=len(inventory),
    whole_project_complete=False)))
