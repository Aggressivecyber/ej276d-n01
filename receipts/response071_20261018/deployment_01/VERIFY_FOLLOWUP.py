from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, tomllib

r = Path('E:/GAN/research/N01'); w = Path(__file__).parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
request = read(w/'AUTOMATION_UPDATE_REQUEST.json')
actual = tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text(encoding='utf-8'))
before = read(w/'AUTOMATION_BEFORE_UPDATE.json')
for a, b in [('id','id'), ('kind','kind'), ('name','name'), ('status','status'), ('rrule','rrule'), ('target_thread_id','targetThreadId')]:
    assert actual[a] == request[b], a
assert actual['prompt'] == request['prompt'].strip()
assert actual.get('notification_policy') == before.get('notification_policy')
assert actual['created_at'] == before['created_at'] and actual['updated_at'] > before['updated_at']
assert actual['prompt'].startswith(before['prompt'].split('当前具体交接：')[0])
(w/'FOLLOWUP_PROMPT_EFFECTIVE.txt').write_text(actual['prompt'], encoding='utf-8')
cfg = read(r/'configs/response071_20261018.json'); pkg = read(w/'PACKAGE_RECEIPT.json')
assert sha(r/'configs/response071_20261018.json') == pkg['config_sha256']
for name, digest in cfg['source_pins'].items(): assert sha(r/name) == digest, name
manifest = read(w/'SOURCE_MANIFEST.json')
for name, meta in manifest['files'].items():
    assert (r/name).stat().st_size == meta['bytes'] and sha(r/name) == meta['sha256'], name
now = datetime.now(timezone.utc).isoformat(); next_wake = '2026-10-18T21:18:00+08:00'
assert datetime.fromisoformat(now) < datetime.fromisoformat(next_wake)
snapshot = w/'before_followup'; snapshot.mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    path = r/name; d = read(path)
    assert d['status'] == 'response071_registered_awaiting_remote_host' and d['active_scientific_jobs'] == 0
    assert d['response071']['stage'] == 'REGISTERED_NOT_STARTED' and not d['response071']['active']
    (snapshot/name).write_bytes(path.read_bytes())
    d['response_goal_20261018'].update(next_wake_local=next_wake, followup_automation_id='n01', followup_receipt=(w/'FOLLOWUP_RECEIPT.json').relative_to(r).as_posix(), updated_UTC=now)
    d['updated_UTC'] = now
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
assert read(snapshot/'jobs.json')['jobs'] == read(r/'jobs.json')['jobs']
ledgers = [read(r/name) for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json')]
assert all(d['response_goal_20261018'] == ledgers[0]['response_goal_20261018'] and d['response071'] == ledgers[0]['response071'] for d in ledgers)
record = dict(verified_UTC=now, automation_id=actual['id'], status=actual['status'], kind=actual['kind'], target_thread_id=actual['target_thread_id'], next_local_wake=next_wake, rrule=actual['rrule'], prompt_match_after_tool_trailing_whitespace_trim=True, prompt_characters=len(actual['prompt']), prompt_sha256=sha(w/'FOLLOWUP_PROMPT_EFFECTIVE.txt'), original_continuous_root_only_prefix_preserved=True, updated_at=actual['updated_at'], current_job_status='REGISTERED_NOT_STARTED_AWAITING_REMOTE_HOST', native_started=False, no_remote_queries_after_offline_assessment=True, all90_job_records_unchanged=True, three_ledgers_coherent=True, three_ledger_goal_schedules_updated=True, source_pins_verified=len(cfg['source_pins']), source_manifest_verified=len(manifest['files']), local_E_free_bytes=shutil.disk_usage(r).free)
(w/'FOLLOWUP_RECEIPT.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
(w/'ACTIVE_HANDOFF.json').write_text(json.dumps(dict(UTC=now, job_id=cfg['run_id'], status=record['current_job_status'], native_started=False, active_scientific_jobs=0, deployment_wait_receipt=(w/'DEPLOYMENT_WAIT_RECEIPT.json').relative_to(r).as_posix(), followup_receipt=(w/'FOLLOWUP_RECEIPT.json').relative_to(r).as_posix(), next_local_wake=next_wake, no_further_remote_polling=True, goal_achieved=False), indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
