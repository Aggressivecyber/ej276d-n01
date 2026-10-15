"""Verify the app-saved one-shot wake and register it without another SSH call."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import tomllib

r=Path('E:/GAN/research/N01');o=r/'receipts/wp2_014_20261015'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=read(o/'FOLLOWUP_PLAN.json')
saved=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
actual=tomllib.loads(saved.read_text(encoding='utf-8'))
assert actual['id']=='n01' and actual['kind']=='heartbeat' and actual['status']=='ACTIVE'
assert actual['name']==plan['name'] and actual['target_thread_id']==plan['target_thread_id']
prepared_prompt=(o/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8')
assert actual['prompt']==prepared_prompt.rstrip('\r\n'), 'Only the app trailing-linebreak normalization is permitted'
assert actual['rrule']=='FREQ=DAILY;COUNT=1;BYHOUR=10;BYMINUTE=30;BYSECOND=0'
assert dt.datetime.now().astimezone().utcoffset()==dt.timedelta(hours=8)
assert dt.datetime.fromisoformat(plan['next_wake_local'])>dt.datetime.now().astimezone()
source=read(o/'SOURCE_COMMIT_READBACK.json');started=read(o/'EXECUTION_STARTED.json')
assert started['current_training_updates']==197
record=dict(schema='n01/wp2-followup-confirmation/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    automation_id='n01',kind='heartbeat',status='ACTIVE',count=1,
    next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],
    host_UTC_offset_hours=8,actual_saved_settings_verified=True,settings_sha256=sha(saved),
    prompt_sha256=sha(o/'FOLLOWUP_PROMPT.txt'),saved_prompt_sha256=hashlib.sha256(actual['prompt'].encode('utf-8')).hexdigest(),
    app_removed_prepared_trailing_linebreak_only=True,target_thread_id=actual['target_thread_id'],
    source_commit=source['source_commit'],correct_first_G1_verified=True,
    start_receipt_sha256=sha(o/'start_native_01/START_RECEIPT.json'),
    startup_validation_sha256=sha(o/'start_verification_01/START_VALIDATION.json'),
    supervisor_pid=107073,analysis_pid=107083,last_verified_completed_G_updates=197,
    cumulative_training_last_verified=159864,new_AD_queries_last_verified=432,
    current_count_semantics='durable lower bound at the startup snapshot; running work can progress after this readback',
    continuous_agent_monitoring=False,no_new_remote_calls_after_correct_start=True,
    scientific_status='response_match_not_established')
assert not (o/'FOLLOWUP_RECEIPT.json').exists();write(o/'FOLLOWUP_RECEIPT.json',record)
ref={k:record[k] for k in ['automation_id','kind','status','count','next_wake_local','next_wake_UTC','source_commit','continuous_agent_monitoring']}
ref.update(receipt='receipts/wp2_014_20261015/FOLLOWUP_RECEIPT.json',receipt_sha256=sha(o/'FOLLOWUP_RECEIPT.json'),correct_start_verified=True)
paths=[r/'PROGRAM.json',r/'MASTER_EXECUTION_PLAN.json',r/'jobs.json',r/'README.md',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
snapshot=o/'metadata_before_followup';snapshot.mkdir()
for p in paths:shutil.copy2(p,snapshot/p.name)
for p in paths:
    if p.suffix!='.json':continue
    d=read(p);d['updated_UTC']=record['UTC'];d['followup']=ref
    if isinstance(d.get('wp2_execution'),dict):d['wp2_execution']['followup']=ref
    if p.name=='jobs.json':
        j=next(j for j in d['jobs'] if j['job_id']=='N01_WP2_014_20261015');j['followup']=ref;j['execution']['followup']=ref
    if p.name=='MASTER_EXECUTION_PLAN.json':
        next(w for w in d['work_packages'] if w['id']=='WP2')['execution']['followup']=ref
    write(p,d)
readme=r/'README.md';text=readme.read_text(encoding='utf-8')
old='正确启动后设置同一n01的一次唤醒并休息。'
assert old in text
text=text.replace(old,'已设置2026-10-15 10:30（Asia/Singapore）单次回收唤醒，期间休息。',1)
readme.write_text(text,encoding='utf-8',newline='\n')
p=r/'PROGRAM.json';d=read(p);d['master_plan_sha256']=sha(r/'MASTER_EXECUTION_PLAN.json');write(p,d)
p=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json');d=read(p);d['plan_sha256']=sha(readme);write(p,d)
print(json.dumps(dict(status='one_wake_saved_and_actual_settings_verified',next_wake_local=record['next_wake_local'],count=1,receipt_sha256=sha(o/'FOLLOWUP_RECEIPT.json'))))
