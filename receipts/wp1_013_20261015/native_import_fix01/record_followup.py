"""Prepare and verify the one-time app heartbeat for the repaired attempt."""
import argparse
import hashlib
import json
import shutil
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015/native_import_fix01'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=('prepare', 'verify'), required=True)
parser.add_argument('--start', type=Path, required=True)
parser.add_argument('--verification', type=Path, required=True)
args = parser.parse_args()

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save(path, value, exclusive=False):
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

start_path = args.start / 'START_RECEIPT.json'
proof_path = args.verification / 'START_VALIDATION.json'
start, proof, state = read(start_path), read(proof_path), read(args.verification / 'REMOTE_STATE.json')
assert proof['status'] == 'actual_first128_family_verified' and proof['extra_pilot_rows'] == 0
assert state.get('runs/primary/artifacts/FAILED.json') is None
assert (state.get('runs/primary/artifacts/status.json') or {}).get('phase') != 'failed'
native = state.get('runs/primary/supervisor/exit_receipt.json')
assert native is None or native['child_native_exit_code'] == native['exit_code'] == 0
code = read(base / 'SOURCE_COMMIT_READBACK.json')
assert code['all_passed'] and code['frozen_source_and_control_files'] == 86
sg = timezone(timedelta(hours=8))
planned = datetime.fromisoformat(start['UTC']).astimezone(sg) + timedelta(seconds=7200 + 600)
wake = planned.replace(second=0, microsecond=0)
if planned.second or planned.microsecond:
    wake += timedelta(minutes=1)
if wake.minute % 5:
    wake += timedelta(minutes=5 - wake.minute % 5)
rrule = 'FREQ=DAILY;COUNT=1;BYHOUR=%d;BYMINUTE=%d;BYSECOND=0' % (wake.hour, wake.minute)
assert datetime.now(timezone.utc) < wake
app_path = Path('C:/Users/10629/.codex/automations/n01/automation.toml')
planned_path = base / 'FOLLOWUP_PLAN.json'
prompt_path = base / 'FOLLOWUP_PROMPT.txt'
if args.stage == 'prepare':
    previous = tomllib.loads(app_path.read_text('utf-8'))
    assert previous['id'] == 'n01' and previous['status'] == 'PAUSED'
    before = base / 'automation_before_02'
    before.mkdir(exist_ok=False)
    shutil.copyfile(app_path, before / 'automation.toml')
    old_prompt = (root / 'receipts/wp1_013_20261015/FOLLOWUP_PROMPT.txt').read_text('utf-8')
    replacements = {
        '06:55': wake.strftime('%H:%M'),
        '/home/rylan/projects/ej276d-n01/wp1_013_20261015': '/home/rylan/projects/ej276d-n01/wp1_013_runtime_fix01_20261015',
        'UTC2026-10-14T20:43:34': 'UTC' + datetime.fromisoformat(start['UTC']).strftime('%Y-%m-%dT%H:%M:%S'),
        '监督PID49192、计算PID49199': '监督PID%d、计算PID%d' % (proof['supervisor_pid'], proof['child_pid']),
        'ffe88e43518cd79f01c116df421052550b1c874d': code['commit'],
        '85个源码及控制文件': '86个源码及控制文件',
        '5e871bfc871c92c9534ec9e214b377bf80fb9d020eaf5148db562dc29a23e214': start['config_sha256'],
        'ba10d2cb27713302e0709869b3d384882fb4b449766c27e33b1561fd946357fc': start['source_manifest_sha256'],
        '源码83文件': '源码84文件', '83源码/22输入': '84源码/22输入',
        'receipts/wp1_013_20261015/runtime_package_01': 'receipts/wp1_013_20261015/native_import_fix01/runtime_package_02',
        'runtime_package_01/FREEZE_RECEIPT.json': 'native_import_fix01/runtime_package_02/FREEZE_RECEIPT.json',
        'start_native_01/START_RECEIPT.json': 'native_import_fix01/start_native_02/START_RECEIPT.json',
        'start_verification_01/REMOTE_STATE.json': 'native_import_fix01/start_verification_02/REMOTE_STATE.json',
        'SOURCE_COMMIT_READBACK.json': 'native_import_fix01/SOURCE_COMMIT_READBACK.json',
        'FOLLOWUP_RECEIPT.json': 'native_import_fix01/FOLLOWUP_RECEIPT.json',
        'independent_01/PRELAUNCH_REVIEW.json': 'independent_01/native_import_fix01/PRELAUNCH_RUNTIME_FIX_REVIEW.json',
        'receipts/wp1_013_20261015/collection_01': 'receipts/wp1_013_20261015/native_import_fix01/collection_02',
        'runs/wp1_013_20261015/collected_01': 'runs/wp1_013_20261015/runtime_fix01_collected_02',
    }
    prompt = old_prompt
    for before_text, after_text in replacements.items():
        prompt = prompt.replace(before_text, after_text)
    prompt += ('\n本次是仅一次包装修复后的 attempt2。原attempt1首128通过后因遗漏原011 CDF脚本退出1，'
        '旧83源码包、25文件现场和3729完成操作均保存，不将它记为科学或工程完成。'
        '修复仅补入原011 stability011_metrics.py 原字节及原SHA；5个科学数值脚本和人口/方向/尺度/T公式不变。'
        '新verify-start已拒绝同一快照出现FAILED、phasefailed或非零native退出，历史FIRST128本身不足。'
        '本attempt的数值上限仍为原登记值；整个013累计预算必须加上已耗oldA/head/noise各52744、'
        'ADC/gather各316464、CP3。累计最大oldA/head各351248、linear40960、head_total392208、'
        'noise105488、ADC1886304、gather3360864、CP6；其余零预算保持。'
        '先读native_import_fix01/RUNTIME_REPAIR_REGISTRATION.json及其中cost和aggregate_caps；'
        '分别报告两attempt及总成本。若新attempt再次失败，先回收保存并报告具体工程缺项，'
        '不得自动开启第三次同范围重跑或隐去失败成本。\n')
    with prompt_path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(prompt)
    result = dict(schema='n01/wp1-repair-followup-plan/v1', name='N01 主实施计划013回收与推进',
        automation_id='n01', kind='heartbeat', target_thread_id=previous['target_thread_id'],
        next_wake_local=wake.isoformat(), next_wake_UTC=wake.astimezone(timezone.utc).isoformat(),
        rrule=rrule, count=1, prompt=str(prompt_path), prompt_sha256=digest(prompt_path),
        correct_start_verified=True, source_commit=code['commit'])
    save(planned_path, result, True)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0)

plan = read(planned_path)
assert plan['next_wake_local'] == wake.isoformat() and plan['rrule'] == rrule
app = tomllib.loads(app_path.read_text('utf-8'))
assert app['id'] == 'n01' and app['kind'] == 'heartbeat' and app['status'] == 'ACTIVE'
assert app['name'] == plan['name'] and app['target_thread_id'] == plan['target_thread_id']
assert app['rrule'] == rrule
assert app['prompt'].replace('\r\n', '\n').strip() == prompt_path.read_text('utf-8').replace('\r\n', '\n').strip()
stamp = datetime.now(timezone.utc).isoformat()
result = dict(UTC=stamp, attempt=2, **plan,
    status='ACTIVE', actual_saved_settings_verified=True, settings_sha256=digest(app_path),
    start_receipt_sha256=digest(start_path), startup_validation_sha256=digest(proof_path),
    supervisor_pid=proof['supervisor_pid'], analysis_pid=proof['child_pid'], first_completed_counters=proof['completed_counters'],
    continuous_agent_monitoring=False, no_new_remote_calls_after_this_startup=True,
    failed_attempt_cost_preserved=True, new_AD_queries=0, new_training_updates=0,
    scientific_status='response_match_not_established')
result['schema'] = 'n01/wp1-repair-one-time-followup/v1'
save(base / 'FOLLOWUP_RECEIPT.json', result, True)
paths = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
snapshot = base / 'metadata_before_followup'
snapshot.mkdir(exist_ok=False)
for path in paths:
    shutil.copyfile(path, snapshot / path.name)
values = [read(path) for path in paths]
master, program, jobs, selector = values
refs = dict(automation_id='n01', status='ACTIVE', count=1, next_wake_local=wake.isoformat(),
    receipt=str((base / 'FOLLOWUP_RECEIPT.json').relative_to(root)), receipt_sha256=digest(base / 'FOLLOWUP_RECEIPT.json'),
    source_commit=code['commit'], correct_start_verified=True, continuous_monitoring=False)
master.update(followup=refs, updated_UTC=stamp)
master['work_packages'][0]['execution'].update(followup=refs)
for value in (program, selector):
    assert value['active_scientific_jobs'] == 1 and value['scientific_status'] == 'response_match_not_established'
    value.update(followup=refs, updated_UTC=stamp)
    value['wp1_execution'].update(followup=refs)
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
job['execution'].update(followup=refs)
jobs.update(updated_UTC=stamp)
save(paths[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(paths[0])
for path, value in zip(paths[1:], (program, jobs, selector)):
    save(path, value)
print(json.dumps(dict(automation_id='n01', status='ACTIVE', count=1, next_wake_local=wake.isoformat(),
    actual_settings_verified=True, no_monitoring_after_startup=True)))
