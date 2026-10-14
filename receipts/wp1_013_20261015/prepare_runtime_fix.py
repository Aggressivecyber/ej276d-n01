"""Preserve failed WP1 cost and register one packaging-only engineering repair."""
import hashlib
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015'
repair = base / 'native_import_fix01'
repair.mkdir(exist_ok=True)
snapshot = repair / 'metadata_before_repair_01'
snapshot.mkdir(exist_ok=False)
paths = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json',
    Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
for path in paths + [root / 'README.md', root / 'wp1_013_config.json', root / 'WP1_SOURCE_MANIFEST.json']:
    shutil.copyfile(path, snapshot / path.name)

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save(path, value, exclusive=False):
    mode = 'xb' if exclusive else 'wb'
    with Path(path).open(mode) as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

col_path = base / 'collection_failed_01/COLLECTION_RECEIPT.json'
col = json.loads(col_path.read_text('utf-8'))
assert col['status'] == 'partial_evidence_preserved' and col['file_count'] == 25
native, journal = col['job']['native'], col['job']['durable_progress']
assert native['child_native_exit_code'] == native['exit_code'] == 1
assert not journal['unknown_inflight_ids'] and not journal['failed_ids'] and not journal['truncated_tail']
assert journal['entered_counts'] == journal['completed_counts'] == native['completed_counts']
failed_path = root / 'runs/wp1_013_20261015/failed_attempt_01/runs/primary/artifacts/FAILED.json'
failed = json.loads(failed_path.read_text('utf-8'))
assert failed['error_type'] == 'ModuleNotFoundError' and failed['error'] == "No module named 'stability011_metrics'"
app_path = Path('C:/Users/10629/.codex/automations/n01/automation.toml')
app = tomllib.loads(app_path.read_text('utf-8'))
assert app['id'] == 'n01' and app['status'] == 'PAUSED'
old_cfg = json.loads((root / 'wp1_013_config.json').read_text('utf-8'))
assert old_cfg['status'] == 'frozen' and len(old_cfg['source_files']) == 83
assert digest(root / 'wp1_013_config.json') == '5e871bfc871c92c9534ec9e214b377bf80fb9d020eaf5148db562dc29a23e214'
prior = {key: journal['completed_counts'].get(key, 0) for key in old_cfg['budget_caps']}
total = {key: cap + prior[key] for key, cap in old_cfg['budget_caps'].items()}
assert total['old_A_rows'] == total['head_actual_parameter_forward_rows'] == 351248
assert total['noise005_rows'] == 105488 and total['checkpoint_loads'] == 6
assert total['hard_ADC_rows'] == 1886304 and total['all_path_gather_rows'] == 3360864
assert total['head_linear_direction_map_rows'] == 40960 and total['head_total_kernel_rows'] == 392208
assert all(total[key] == 0 for key in ('new_AD_queries', 'G_updates', 'D_updates', 'optimizer_steps', 'DEV_model_rows'))
stamp = datetime.now(timezone.utc).isoformat()
receipt = dict(schema='n01/wp1-runtime-packaging-repair/v1', UTC=stamp, run_id=old_cfg['run_id'],
    failed_attempt=1, native_exit_code=1, original_freeze='receipts/wp1_013_20261015/runtime_package_01/FREEZE_RECEIPT.json',
    collection=str(col_path.relative_to(root)), collection_sha256=digest(col_path),
    failed_artifact=str(failed_path.relative_to(root)), failed_artifact_sha256=digest(failed_path),
    failed_reason=failed['error'], failed_native=native, prior_cost=prior, aggregate_budget_caps=total,
    original_attempt_budget_caps=old_cfg['budget_caps'], replacement_attempt_budget_caps=old_cfg['budget_caps'],
    intended_repair='Bundle the unchanged, SHA-pinned original011 stability011_metrics dependency; use a fresh remote attempt root.',
    scientific_population_directions_scales_banks_hypotheses_and_tolerances_unchanged=True,
    automatic_restart=False, explicitly_registered_engineering_replacement_attempts=1,
    original_payloads_and_failed_artifacts_preserved=True, new_AD_queries=0, new_training_updates=0,
    science_accepted=False, premature_running_metadata_corrected=True,
    cancelled_followup=dict(automation_id='n01', actual_saved_status='PAUSED', settings_sha256=digest(app_path),
        former_followup_receipt='receipts/wp1_013_20261015/FOLLOWUP_RECEIPT.json',
        reason='The same saved startup snapshot contained a later native packaging failure; no running job awaits the old wake.'))
save(repair / 'RUNTIME_REPAIR_REGISTRATION.json', receipt, True)
old_remote = old_cfg['remote_root']
new_remote = '/home/rylan/projects/ej276d-n01/wp1_013_runtime_fix01_20261015'
def remap(value):
    if isinstance(value, dict):
        return {key: remap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item) for item in value]
    if isinstance(value, str) and value.startswith(old_remote):
        return new_remote + value[len(old_remote):]
    return value
cfg = remap(old_cfg)
cfg.update(status='draft_registered', source_files={}, protection_files={},
    engineering_repair=dict(attempt=2, previous_remote_root=old_remote, previous_cost=prior,
        aggregate_budget_caps=total, original_config_sha256=digest(snapshot / 'wp1_013_config.json'),
        registration='receipts/wp1_013_20261015/native_import_fix01/RUNTIME_REPAIR_REGISTRATION.json',
        registration_sha256=digest(repair / 'RUNTIME_REPAIR_REGISTRATION.json'),
        scope='Packaging dependency repair only; no changed scientific settings.'))
cfg.pop('frozen_UTC', None)
save(root / 'wp1_013_config.json', cfg)
values = [json.loads(path.read_text('utf-8')) for path in paths]
master, program, jobs, selector = values
state = 'WP1_013_runtime_repair_preparation'
refs = dict(failed_attempt=1, current_attempt=2, original_remote_root=old_remote, replacement_remote_root=new_remote,
    registration='receipts/wp1_013_20261015/native_import_fix01/RUNTIME_REPAIR_REGISTRATION.json',
    registration_sha256=digest(repair / 'RUNTIME_REPAIR_REGISTRATION.json'), prior_cost=prior, aggregate_budget_caps=total,
    frozen_source_commit_attempt1='ffe88e43518cd79f01c116df421052550b1c874d',
    status='preparation', automatic_restart=False, scientific_settings_unchanged=True)
master.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, runtime_repair=refs,
    current_action='Repair the single missing packaged dependency; validate an isolated package and launch one fresh bounded replacement attempt.')
master['budgets']['WP1'].update(original_attempt_budget_caps=old_cfg['budget_caps'], prior_actual_cost=prior,
    aggregate_budget_caps=total, replacement_attempt_budget_caps=old_cfg['budget_caps'])
master['work_packages'][0].update(status='runtime_repair_preparation', runtime_repair=refs)
for value in (program, selector):
    value.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, runtime_repair=refs,
        next_action=master['current_action'], scientific_status='response_match_not_established', current_run_training_updates=0,
        followup=dict(automation_id='n01', status='PAUSED', reason='native import failure; replacement not started'))
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
job.update(status='runtime_repair_preparation', active_attempt=2, runtime_repair=refs, source_freeze_pending=True,
    original_attempt_native_exit_code=1, original_attempt_cost=prior, aggregate_budget_caps=total, updated_UTC=stamp)
jobs.update(active_scientific_jobs=0, updated_UTC=stamp)
save(paths[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(paths[0])
for path, value in zip(paths[1:], (program, jobs, selector)):
    save(path, value)
readme = (root / 'README.md').read_text('utf-8')
readme = readme.replace('013 已在服务器启动并核验首次计算。',
    '013 首次计算已核验，但随后因公共包遗漏原011评价依赖而原生退出1；现场和已耗预算已保留，正在进行一次包装修复。', 1)
(root / 'README.md').write_bytes(readme.encode('utf-8'))
print(json.dumps(dict(status=state, active_scientific_jobs=0, failed_cost=prior, aggregate_caps=total,
    draft_config_sha256=digest(root / 'wp1_013_config.json'), automatic_restart=False)))
