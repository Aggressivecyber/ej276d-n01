from pathlib import Path
from datetime import datetime,timezone
import json,tomllib,hashlib,shutil
r=Path('E:/GAN/research/N01');w=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
request=read(w/'AUTOMATION_UPDATE_REQUEST.json');d=tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text(encoding='utf-8'))
for a,b in [('id','id'),('kind','kind'),('name','name'),('status','status'),('rrule','rrule'),('target_thread_id','targetThreadId')]:assert d[a]==request[b],a
assert request['prompt']==d['prompt']
before=read(w/'AUTOMATION_BEFORE_UPDATE.json');assert d.get('notification_policy')==before.get('notification_policy')
assert d['created_at']==before['created_at'] and d['updated_at']>before['updated_at']
(w/'FOLLOWUP_PROMPT_EFFECTIVE.txt').write_text(d['prompt'],encoding='utf-8')
a=read(w/'STARTUP_ASSESSMENT.json');cfg=read(r/'configs/response070_20261018.json');assert sha(r/'configs/response070_20261018.json')==a['config_sha256']
for n,h in cfg['source_pins'].items():assert sha(r/n)==h,n
manifest=read(w/'SOURCE_MANIFEST.json')
for n,meta in manifest['files'].items():assert (r/n).stat().st_size==meta['bytes'] and sha(r/n)==meta['sha256'],n
now=datetime.now(timezone.utc).isoformat();next_wake=read(w/'NEXT_WAKE.json')['next_local_wake']
assert datetime.fromisoformat(now)<datetime.fromisoformat(next_wake)
snapshot=w/'before_followup';snapshot.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=r/name;j=read(path);assert j['status']=='response070_running_startup_verified' and j['active_scientific_jobs']==1
    assert j['response070']['active'] and j['response070']['stage']=='RUNNING_STARTUP_VERIFIED' and not j['response069']['active']
    (snapshot/name).write_bytes(path.read_bytes())
    j['response_goal_20261018'].update(next_wake_local=next_wake,followup_automation_id='n01',followup_receipt=(w/'FOLLOWUP_RECEIPT.json').relative_to(r).as_posix(),updated_UTC=now)
    j['updated_UTC']=now;path.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    assert read(path)['response_goal_20261018']['next_wake_local']==next_wake
assert read(r/'receipts/response070_20261018/preparation_01/before_state/jobs.json')['jobs']==read(r/'jobs.json')['jobs'][:-1]
f=dict(verified_UTC=now,automation_id=d['id'],status=d['status'],kind=d['kind'],target_thread_id=d['target_thread_id'],next_local_wake=next_wake,rrule=d['rrule'],prompt_exact_readback=True,prompt_characters=len(d['prompt']),
    prompt_sha256=sha(w/'FOLLOWUP_PROMPT_EFFECTIVE.txt'),original_continuous_root_only_prefix_preserved=d['prompt'].startswith(before['prompt'].split('当前具体交接：')[0]),
    updated_at=d['updated_at'],last_remote_startup_verified_UTC=a['verified_UTC'],no_remote_queries_after_correct_startup=True,
    prior88_job_records_unchanged=True,three_ledgers_coherent=True,three_ledger_goal_schedules_updated=True,source481pins_and490manifest_reverified=True,local_E_free_bytes=shutil.disk_usage(r).free)
assert f['original_continuous_root_only_prefix_preserved']
(w/'FOLLOWUP_RECEIPT.json').write_text(json.dumps(f,indent=2)+'\n')
(w/'ACTIVE_HANDOFF.json').write_text(json.dumps(dict(UTC=now,job_id=cfg['run_id'],status='RUNNING_STARTUP_VERIFIED',startup_assessment=(w/'STARTUP_ASSESSMENT.json').relative_to(r).as_posix(),
    followup_receipt=(w/'FOLLOWUP_RECEIPT.json').relative_to(r).as_posix(),last_remote_state_only=a['verified_UTC'],no_further_remote_polling=True,goal_achieved=False),indent=2)+'\n')
print(json.dumps(f))
