"""Pre-register the disk-only cap needed to retain every actual diagnostic."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

root = Path('E:/GAN/research/N01')
base = root / 'receipts/wp1_013_20261015/native_repeat_fix01'
before = base / 'output_cap_amend01/config_before'
before.mkdir(parents=True, exist_ok=False)
cfgpath = root / 'wp1_013_config.json'
regpath = base / 'NUMERICAL_STORAGE_REPAIR_REGISTRATION.json'
meta = [root / 'MASTER_EXECUTION_PLAN.json', root / 'PROGRAM.json', root / 'jobs.json', Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
for path in [cfgpath, regpath] + meta:
    shutil.copyfile(path, before / path.name)

def read(path):
    return json.loads(Path(path).read_text('utf-8'))

def save(path, value, exclusive=False):
    with Path(path).open('xb' if exclusive else 'wb') as handle:
        handle.write((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))

def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

cfg = read(cfgpath)
assert cfg['status'] == 'draft_registered' and cfg['resource_limits']['output_MiB'] == 1024
assert digest(cfgpath) == '2dcaa7df34a769ada47bd354e8de2926afcf16ff0f70ac7389b17e4b3f751dde'
colpath = root / 'receipts/wp1_013_20261015/native_import_fix01/collection_failed_02/COLLECTION_RECEIPT.json'
collection = read(colpath)
peak = collection['job']['native']['peak_output_bytes']
assert peak == 322671367
partial = root / 'runs/wp1_013_20261015/failed_attempt_02/runs/primary'
observed = {str(path.relative_to(partial)): path.stat().st_size for path in partial.rglob('*') if path.is_file() and path.stat().st_size > 1000000}
stamp = datetime.now(timezone.utc).isoformat()
receipt = dict(schema='n01/wp1-output-cap-amendment/v1', UTC=stamp, attempt=3,
    prior_output_cap_MiB=1024, new_output_cap_MiB=1536, collection_archive_limit_MiB=1552, collection_control_allowance_MiB=16,
    actual_partial_peak_output_bytes=peak, actual_partial_large_files_bytes=observed,
    measured_scope='Two complete MC latent banks; bank0 six FULL reports and A/theta moments; before actual REFERENCE storage, bank1 evaluation, combined repeats, calibration and parameter reports.',
    expected_full_size='Approximately 1.1 to 1.4 GB inferred from existing compressed files and remaining record counts; not an observed final size.',
    reason='Retain actual per-draw A and hard output/flags plus all declared reports without silent precision loss or omitted records.',
    CPU_GPU_RSS_wall_limits_unchanged=True, NN_ADC_CP_and_all_zero_budgets_unchanged=True,
    scientific_population_directions_scales_observation_metrics_and_T_unchanged=True,
    disk_only_amendment_before_any_attempt3_dispatch=True, actual_new_model_calls=0,
    saved_collection_sha256=digest(colpath), original_draft_config_sha256=digest(before / cfgpath.name))
save(base / 'STORAGE_CAP_AMENDMENT.json', receipt, True)
reg = read(regpath)
reg.update(output_cap_amendment=dict(receipt=str((base / 'STORAGE_CAP_AMENDMENT.json').relative_to(root)),
    sha256=digest(base / 'STORAGE_CAP_AMENDMENT.json'), output_MiB=1536, collection_MiB=1552))
save(regpath, reg)
cfg['resource_limits']['output_MiB'] = 1536
cfg['engineering_repair'].update(registration_sha256=digest(regpath), output_cap_amendment_sha256=digest(base / 'STORAGE_CAP_AMENDMENT.json'))
save(cfgpath, cfg)
values = [read(path) for path in meta]
master, program, jobs, selector = values
master['budgets']['WP1']['resources']['output_MiB'] = 1536
master['numerical_storage_repair']['registration_sha256'] = digest(regpath)
master['numerical_storage_repair']['output_cap_amendment'] = receipt
master.update(updated_UTC=stamp)
for value in (program, selector):
    assert value['active_scientific_jobs'] == 0
    value['numerical_storage_repair']['registration_sha256'] = digest(regpath)
    value.update(updated_UTC=stamp)
job = next(row for row in jobs['jobs'] if row['job_id'] == 'N01_WP1_013_20261015')
job['resources']['output_MiB'] = 1536
job['numerical_storage_repair']['registration_sha256'] = digest(regpath)
job.update(updated_UTC=stamp)
save(meta[0], master)
for value in (program, selector):
    value['master_plan_sha256'] = digest(meta[0])
for path, value in zip(meta[1:], (program, jobs, selector)):
    save(path, value)
print(json.dumps(dict(output_cap_MiB=1536, collection_MiB=1552, actual_partial_peak_bytes=peak,
    new_draft_config_sha256=digest(cfgpath), scientific_settings_changed=False)))
