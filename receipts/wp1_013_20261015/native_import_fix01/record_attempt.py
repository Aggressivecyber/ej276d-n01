"""Adopt, commit-readback, and register the single bounded packaging repair."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015/native_import_fix01'
parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=('adopt', 'commit', 'started'), required=True)
parser.add_argument('--freeze', required=True, type=Path)
parser.add_argument('--start', type=Path)
parser.add_argument('--verification', type=Path)
args = parser.parse_args()
freeze = args.freeze.resolve(strict=True)
assert freeze.is_relative_to(base)

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def save(path, value, exclusive=False):
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

frozen = read(freeze / 'FREEZE_RECEIPT.json')
assert frozen['source_file_count'] == 84 and frozen['private_file_count'] == 4
assert frozen['all_local_original_files_unchanged'] and not frozen['scientific_started']
pins = {'wp1_013_config.json': frozen['config_sha256'], 'WP1_SOURCE_MANIFEST.json': frozen['source_manifest_sha256']}
assert all(digest(freeze / name) == sha for name, sha in pins.items())
cfg = read(freeze / 'wp1_013_config.json')
assert cfg['status'] == 'frozen' and cfg['remote_root'] == '/home/rylan/projects/ej276d-n01/wp1_013_runtime_fix01_20261015'
assert len(cfg['source_files']) == 84 and cfg['engineering_repair']['attempt'] == 2
reg = read(base / 'RUNTIME_REPAIR_REGISTRATION.json')
assert cfg['budget_caps'] == reg['replacement_attempt_budget_caps']
assert cfg['engineering_repair']['aggregate_budget_caps'] == reg['aggregate_budget_caps']
for old_name in ('MECHANISM012_SOURCE_MANIFEST.json', 'STABILITY011_SOURCE_MANIFEST.json', 'SHAPE010_SOURCE_MANIFEST.json'):
    assert all(digest(root / rel) == sha for rel, sha in read(root / old_name)['files'].items())
assert cfg['source_files']['scripts/stability011_metrics.py']['sha256'] == read(root / 'STABILITY011_SOURCE_MANIFEST.json')['files']['scripts/stability011_metrics.py']

if args.stage == 'commit':
    assert all(digest(root / name) == sha for name, sha in pins.items())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    expected = {name: row['sha256'] for name, row in cfg['source_files'].items()}
    expected.update(pins)
    for name, want in expected.items():
        blob = subprocess.check_output(['git', 'show', head + ':' + name], cwd=root)
        assert hashlib.sha256(blob).hexdigest() == digest(root / name) == want, name
    result = dict(schema='n01/wp1-repair-source-commit-readback/v1', run_id=cfg['run_id'], attempt=2,
        commit=head, frozen_source_and_control_files=len(expected), exact_working_and_Git_blob_sha256=expected,
        all_passed=True, public_push_performed=False, original_source_commit='ffe88e43518cd79f01c116df421052550b1c874d')
    save(base / 'SOURCE_COMMIT_READBACK.json', result, True)
    print(json.dumps(dict(commit=head, source_controls=len(expected), all_passed=True)))
    raise SystemExit(0)

live = args.stage == 'started'
if not live:
    hist = base / 'before_control_adoption_02'
    hist.mkdir(exist_ok=False)
    for name in pins:
        shutil.copyfile(root / name, hist / name)
    assert read(root / 'wp1_013_config.json')['status'] == 'draft_registered'
    for name in pins:
        shutil.copyfile(freeze / name, root / name)
    assert all(digest(root / name) == sha for name, sha in pins.items())
else:
    assert args.start and args.verification
    start, proof, state = read(args.start / 'START_RECEIPT.json'), read(args.verification / 'START_VALIDATION.json'), read(args.verification / 'REMOTE_STATE.json')
    assert proof['status'] == 'actual_first128_family_verified'
    assert state.get('runs/primary/artifacts/FAILED.json') is None
    assert (state.get('runs/primary/artifacts/status.json') or {}).get('phase') != 'failed'
    native = state.get('runs/primary/supervisor/exit_receipt.json')
    assert native is None or native['child_native_exit_code'] == native['exit_code'] == 0
    assert start['supervisor_pid'] == proof['supervisor_pid']
    code = read(base / 'SOURCE_COMMIT_READBACK.json')
    assert code['all_passed'] and code['frozen_source_and_control_files'] == 86

metadata = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
history = base / ('metadata_before_' + args.stage)
history.mkdir(exist_ok=False)
for path in metadata + [root / 'README.md']:
    shutil.copyfile(path, history / path.name)
values = [read(path) for path in metadata]
master, program, jobs, selector = values
assert program['new_training_updates'] == 159667 and program['current_run_training_updates'] == 0
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
assert not any(row.get('scientific_job') and row.get('status') in ('running', 'native_running', 'started') for row in jobs['jobs'] if row is not job)
refs = dict(attempt=2, remote_root=cfg['remote_root'], configuration='wp1_013_config.json', configuration_sha256=pins['wp1_013_config.json'],
    source_manifest='WP1_SOURCE_MANIFEST.json', source_manifest_sha256=pins['WP1_SOURCE_MANIFEST.json'],
    source_file_count=84, source_controls=86, private_file_count=4,
    freeze_receipt=str((freeze / 'FREEZE_RECEIPT.json').relative_to(root)), freeze_receipt_sha256=digest(freeze / 'FREEZE_RECEIPT.json'),
    repair_registration=str((base / 'RUNTIME_REPAIR_REGISTRATION.json').relative_to(root)),
    repair_registration_sha256=digest(base / 'RUNTIME_REPAIR_REGISTRATION.json'),
    per_attempt_budget_caps=cfg['budget_caps'], prior_attempt_actual_cost=reg['prior_cost'], aggregate_budget_caps=reg['aggregate_budget_caps'],
    resources=cfg['resource_limits'], new_training_updates=0, new_AD_queries=0)
if live:
    refs.update(start_receipt=str((args.start / 'START_RECEIPT.json').relative_to(root)),
        startup_verification=str((args.verification / 'START_VALIDATION.json').relative_to(root)),
        start_receipt_sha256=digest(args.start / 'START_RECEIPT.json'), startup_verification_sha256=digest(args.verification / 'START_VALIDATION.json'),
        source_commit=code['commit'], supervisor_pid=proof['supervisor_pid'], analysis_pid=proof['child_pid'], correct_start_verified=True)
stamp = datetime.now(timezone.utc).isoformat()
state_name = 'WP1_013_runtime_fix01_native_running' if live else 'WP1_013_runtime_fix01_frozen_ready'
master.update(status=state_name, active_scientific_jobs=int(live), updated_UTC=stamp,
    current_action='One scheduled wake to collect the repaired WP1 and advance bounded WP2.' if live else 'Launch the one packaging repair and verify first computation plus absence of native failure.')
master['work_packages'][0].update(status='native_running' if live else 'frozen_ready', execution=refs)
master['runtime_repair'].update(status='native_running' if live else 'frozen_ready')
for value in (program, selector):
    value.update(status=state_name, active_scientific_jobs=int(live), updated_UTC=stamp, wp1_execution=refs,
        next_action=master['current_action'], current_run_training_updates=0, scientific_status='response_match_not_established')
    value['runtime_repair']['status'] = 'native_running' if live else 'frozen_ready'
job.update(status='native_running' if live else 'frozen_ready_for_launch', active_attempt=2, execution=refs,
    source_freeze_pending=False, exact_budget_pending=False, updated_UTC=stamp,
    analysis_pid=refs.get('analysis_pid'), supervisor_pid=refs.get('supervisor_pid'))
jobs.update(active_scientific_jobs=int(live), updated_UTC=stamp)
save(metadata[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(metadata[0])
for path, value in zip(metadata[1:], (program, jobs, selector)):
    save(path, value)
readme = (root / 'README.md').read_text('utf-8')
old_text = '013 首次计算已核验，但随后因公共包遗漏原011评价依赖而原生退出1；现场和已耗预算已保留，正在进行一次包装修复。'
new_text = ('013 的一次包装修复已在新服务器目录启动，首次计算及当时无失败均已核验。' if live else '013 的一次包装修复已冻结，等待新服务器目录启动。')
new_text += ' 原尝试退出1、25文件失败现场和52744行参考计算成本保留；补入原011评价依赖，没有改变科学设置。'
expected_text = old_text if not live else ('013 的一次包装修复已冻结，等待新服务器目录启动。'
    ' 原尝试退出1、25文件失败现场和52744行参考计算成本保留；补入原011评价依赖，没有改变科学设置。')
assert expected_text in readme
(root / 'README.md').write_bytes(readme.replace(expected_text, new_text, 1).encode('utf-8'))
result = dict(schema='n01/wp1-runtime-repair-execution/v1', UTC=stamp, stage=args.stage, attempt=2,
    status=state_name, active_scientific_jobs=int(live), execution=refs, scientific_status='response_match_not_established')
save(base / ('EXECUTION_' + args.stage.upper() + '.json'), result, True)
print(json.dumps(dict(status=state_name, active_scientific_jobs=int(live), source_count=84,
    analysis_pid=refs.get('analysis_pid'), aggregate_budget_caps=reg['aggregate_budget_caps'])))
