"""Record the second native failure without restarting or accepting science."""
import hashlib
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015/native_import_fix01'

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save(path, value, exclusive=False):
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

snapshot_path = base / 'start_verification_02/REMOTE_STATE.json'
snapshot = read(snapshot_path)
failed = snapshot['runs/primary/artifacts/FAILED.json']
native = snapshot['runs/primary/supervisor/exit_receipt.json']
assert failed['error'] == 'A repeat pure-signal bytes differ'
assert native['child_native_exit_code'] == native['exit_code'] == 1
assert native['completed_counts'] == native['entered_counts']
assert not native['unknown_inflight_ids'] and not native['failed_ids'] and not native['truncated_tail']
app = tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text('utf-8'))
assert app['status'] == 'PAUSED'
registration = read(base / 'RUNTIME_REPAIR_REGISTRATION.json')
cost2 = {key: native['completed_counts'].get(key, 0) for key in registration['prior_cost']}
total = {key: registration['prior_cost'][key] + cost2[key] for key in cost2}
stamp = datetime.now(timezone.utc).isoformat()
receipt = dict(schema='n01/wp1-second-native-failure/v1', UTC=stamp, attempt=2,
    failure='A repeat pure-signal bytes differ', native=native, snapshot=str(snapshot_path.relative_to(root)),
    snapshot_sha256=digest(snapshot_path), attempt2_cost=cost2, both_attempts_actual_cost=total,
    startup_guard_correctly_rejected_later_failed_snapshot=True, third_attempt_started=False,
    automation_status='PAUSED', original_source_and_failures_preserved=True,
    science_accepted=False, next_action='Quantify saved A signal variance and audit repeat lineage before selecting a minimal engineering correction.')
save(base / 'SECOND_NATIVE_FAILURE.json', receipt, True)
paths = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup = base / 'metadata_before_second_failure'
backup.mkdir(exist_ok=False)
for path in paths + [root / 'README.md']:
    shutil.copyfile(path, backup / path.name)
values = [read(path) for path in paths]
master, program, jobs, selector = values
state = 'WP1_013_native_failed_repeat_invariance_diagnosis'
refs = dict(receipt=str((base / 'SECOND_NATIVE_FAILURE.json').relative_to(root)),
    receipt_sha256=digest(base / 'SECOND_NATIVE_FAILURE.json'), actual_cost=total, third_attempt_started=False,
    source_commit='190553e063d5d4d2c11bdd9f68124aaef423fb59')
master.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, second_native_failure=refs,
    current_action=receipt['next_action'], followup=dict(automation_id='n01', status='PAUSED'))
master['work_packages'][0].update(status='native_failed', second_failure=refs)
for value in (program, selector):
    value.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, second_native_failure=refs,
        scientific_status='response_match_not_established', current_run_training_updates=0,
        next_action=receipt['next_action'], followup=dict(automation_id='n01', status='PAUSED'))
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
job.update(status='native_failed', second_native_failure=refs, attempt2_native_exit_code=1,
    completed_actual_cost_two_attempts=total, updated_UTC=stamp)
jobs.update(active_scientific_jobs=0, updated_UTC=stamp)
save(paths[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(paths[0])
for path, value in zip(paths[1:], (program, jobs, selector)):
    save(path, value)
readme = (root / 'README.md').read_text('utf-8')
old = '013 的一次包装修复已冻结，等待新服务器目录启动。'
new = '013 的包装修复通过了导入，但在重复纯信号逐字节检查处原生退出1；正在量化保存的方差并核对身份。'
assert old in readme
(root / 'README.md').write_bytes(readme.replace(old, new, 1).encode('utf-8'))
print(json.dumps(dict(status=state, active_scientific_jobs=0, actual_cost=total, third_attempt_started=False)))
