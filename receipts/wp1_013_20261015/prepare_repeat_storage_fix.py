"""Register the quantified numerical storage correction with all prior costs."""
import hashlib
import json
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
oldbase = root / 'receipts/wp1_013_20261015/native_import_fix01'
base = root / 'receipts/wp1_013_20261015/native_repeat_fix01'
base.mkdir(exist_ok=True)

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def save(path, value, exclusive=False):
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

colpath = oldbase / 'collection_failed_02/COLLECTION_RECEIPT.json'
col = read(colpath)
assert col['file_count'] == 38 and col['job']['native']['child_native_exit_code'] == col['job']['native']['exit_code'] == 1
assert not col['job']['durable_progress']['unknown_inflight_ids'] and not col['job']['durable_progress']['truncated_tail']
prior = read(oldbase / 'SECOND_NATIVE_FAILURE.json')['both_attempts_actual_cost']
readback_path = root / 'receipts/wp1_013_20261015/native_repeat_readback01/saved_01/SAVED_REPEAT_READBACK.json'
readback = read(readback_path)
assert readback['A']['positive_variance_parents'] == 1
assert readback['A']['maximum_variance_location']['source_parent_row'] == 6592
assert readback['A']['maximum_pairwise_repeat_difference_upper_bound_V'] < 1e-7
assert readback['A']['upper_bounds_are_not_observed_exact_maxima']
cfgpath = root / 'wp1_013_config.json'
oldcfg = read(cfgpath)
assert oldcfg['status'] == 'frozen' and len(oldcfg['source_files']) == 84
assert digest(cfgpath) == 'eabcc194e5971f3e67d0831cc913b8558ba6bdb0174073e1d946759c8ecedad8'
app = tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text('utf-8'))
assert app['status'] == 'PAUSED'
total = {key: cap + prior[key] for key, cap in oldcfg['budget_caps'].items()}
assert total['old_A_rows'] == total['head_actual_parameter_forward_rows'] == 403992
assert total['head_total_kernel_rows'] == 444952 and total['head_linear_direction_map_rows'] == 40960
assert total['noise005_rows'] == 158232 and total['checkpoint_loads'] == 9
assert total['hard_ADC_rows'] == 2202768 and total['all_path_gather_rows'] == 3677328
stamp = datetime.now(timezone.utc).isoformat()
reg = dict(schema='n01/wp1-numerical-storage-repair/v1', UTC=stamp, run_id=oldcfg['run_id'], replacement_attempt=3,
    reason='Exact A equality was stronger than the already registered 1e-7 V baseline numerical replay gate; it prevented saving actual draws.',
    saved_readback=str(readback_path.relative_to(root)), saved_readback_sha256=digest(readback_path),
    known_upper_bound_V=readback['A']['maximum_pairwise_repeat_difference_upper_bound_V'],
    upper_bound_is_not_exact_observed_max=True, isolated_tail_parent=6592, original_replay_limit_V=1e-7,
    numerical_and_physical_randomness_reported_separately=True,
    required_correction='Retain all actual A draws in both reference banks, report numerical repeat/cross-bank residuals, validate against the existing numerical replay limit.',
    source_population_directions_scales_random_banks_observation_and_T_rules_unchanged=True,
    prior_actual_cost=prior, prior_cost=prior, aggregate_budget_caps=total,
    replacement_attempt_budget_caps=oldcfg['budget_caps'],
    necessity_of_regeneration='Saved moments and bank0 summaries cannot reconstruct the missing actual per-draw two-bank signals, waves, decoded logits or noise caches.',
    cache_recovery_as_zero_NN_claimed=False, automatic_restart=False, additional_corrected_regenerations_registered=1,
    no_automatic_fourth_attempt=True, original_two_attempts_preserved=True, new_training_updates=0, new_AD_queries=0,
    failure2_collection=str(colpath.relative_to(root)), failure2_collection_sha256=digest(colpath),
    fixed_scientific_problem=True, science_accepted=False)
save(base / 'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json', reg, True)
paths = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
history = base / 'metadata_before_repair_03'
history.mkdir(exist_ok=False)
for path in paths + [cfgpath, root / 'WP1_SOURCE_MANIFEST.json', root / 'README.md']:
    shutil.copyfile(path, history / path.name)
oldremote = oldcfg['remote_root']
newremote = '/home/rylan/projects/ej276d-n01/wp1_013_repeat_fix01_20261015'
def remap(value):
    if isinstance(value, dict):
        return {key: remap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item) for item in value]
    if isinstance(value, str) and value.startswith(oldremote):
        return newremote + value[len(oldremote):]
    return value
cfg = remap(oldcfg)
cfg.update(status='draft_registered', source_files={}, protection_files={},
    engineering_repair=dict(attempt=3, previous_remote_roots=['/home/rylan/projects/ej276d-n01/wp1_013_20261015', oldremote],
        previous_cost=prior, aggregate_budget_caps=total, original_config_sha256=digest(history / 'wp1_013_config.json'),
        registration='receipts/wp1_013_20261015/native_repeat_fix01/NUMERICAL_STORAGE_REPAIR_REGISTRATION.json',
        registration_sha256=digest(base / 'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json'),
        numerical_A_replay_max_abs_V=1e-7, A_storage='all_actual_float64_draws_per_bank_no_broadcast',
        scope='Retain actual numerical repeats; no changed scientific sampling or metric settings.'))
cfg.pop('frozen_UTC', None)
save(cfgpath, cfg)
values = [read(path) for path in paths]
master, program, jobs, selector = values
state = 'WP1_013_numerical_storage_repair_preparation'
refs = dict(attempt=3, status='preparation', registration=str((base / 'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json').relative_to(root)),
    registration_sha256=digest(base / 'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json'), prior_actual_cost=prior,
    aggregate_budget_caps=total, remote_root=newremote, automatic_restart=False, no_automatic_fourth_attempt=True)
master.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, numerical_storage_repair=refs,
    current_action='Validate actual-draw storage with the existing numerical replay criterion; launch only the registered corrected regeneration.')
master['work_packages'][0].update(status='numerical_storage_repair_preparation', numerical_storage_repair=refs)
master['budgets']['WP1'].update(prior_actual_cost=prior, aggregate_budget_caps=total,
    replacement_attempt_budget_caps=oldcfg['budget_caps'], regeneration_reason=reg['reason'])
for value in (program, selector):
    value.update(status=state, active_scientific_jobs=0, updated_UTC=stamp, numerical_storage_repair=refs,
        next_action=master['current_action'], current_run_training_updates=0, scientific_status='response_match_not_established')
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
job.update(status='numerical_storage_repair_preparation', active_attempt=3, numerical_storage_repair=refs,
    aggregate_budget_caps=total, source_freeze_pending=True, updated_UTC=stamp)
jobs.update(active_scientific_jobs=0, updated_UTC=stamp)
save(paths[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(paths[0])
for path, value in zip(paths[1:], (program, jobs, selector)):
    save(path, value)
print(json.dumps(dict(status=state, aggregate_budget_caps=total, draft_config_sha256=digest(cfgpath), automatic_restart=False)))
