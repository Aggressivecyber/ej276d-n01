"""Prepare one future heartbeat from the saved, verified CPU startup snapshot."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BASE=OUT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state=read(BASE/'readback_start_verification_01/REMOTE_STATE.json')
assert state['first_pair_verification']['first_formal_array_pair_verified'] is True
assert state['artifacts/FAILED.json'] is None
cfg=read(OUT/'READBACK_EXECUTION.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
assert state['registration_sha256']==sha(OUT/'READBACK_EXECUTION.json')
for name,row in source['files'].items():assert sha(ROOT/name)==row['sha256']
template=(OUT/'FOLLOWUP_PROMPT_TEMPLATE.txt').read_text(encoding='utf-8')
prompt=template.replace('__SOURCE_COMMIT__',source['source_commit']).replace('__CONFIG_SHA__',sha(OUT/'READBACK_EXECUTION.json')).replace('__CONTROLLER_SHA__',cfg['files']['readback_control.py']['sha256']).rstrip('\r\n')
assert '__SOURCE_COMMIT__' not in prompt and '__CONFIG_SHA__' not in prompt and '__CONTROLLER_SHA__' not in prompt
with (OUT/'FOLLOWUP_PROMPT.txt').open('x',encoding='utf-8',newline='\n') as f:f.write(prompt+'\n')
now=dt.datetime.now(dt.timezone.utc);wake=(now+dt.timedelta(minutes=12)).replace(second=0,microsecond=0)
local=wake.astimezone(dt.timezone(dt.timedelta(hours=8)))
plan=dict(schema='n01/wp3-readback-single-followup-plan/v1',prepared_UTC=now.isoformat(),automation_id='n01',
    kind='heartbeat',name='N01 CPU015回收与六工作包推进',status='ACTIVE',
    target_thread_id='01a12bad-8839-7522-89a5-f028b107d7ff',
    next_wake_local=local.isoformat(),next_wake_UTC=wake.isoformat(),timezone='Asia/Singapore',
    rrule=f'FREQ=DAILY;COUNT=1;BYHOUR={local.hour};BYMINUTE={local.minute};BYSECOND=0',
    requested_model='gpt-6-astra',requested_reasoning_effort='max',
    prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),continuous_monitoring=False)
with (OUT/'FOLLOWUP_PLAN.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='one_followup_prepared',next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],rrule=plan['rrule'])))
