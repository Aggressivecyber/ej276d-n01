from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'recentered_derivative_diagnostic_01'
old_name = 'configs/joint_response_01_recentered_derivative_completion_20261020.json'
old_raw = (ROOT/old_name).read_bytes()
assert hashlib.sha256(old_raw).hexdigest() == '3f874facd9adad28674af7f1ebf3a8b983e16b19a6b5ff50f7d07a9d2c74f6bf'
assert not json.loads((HERE/'STOP_OWN_DEPLOYMENT_REPAIR_RECEIPT.json').read_bytes())['process_still_present']
config = json.loads(old_raw)
prior = json.loads((PREV/'REGISTRATION.json').read_bytes())['storage']['remote_root']
old_manifest_raw = (HERE/'SOURCE_MANIFEST.json').read_bytes()
(HERE/'SOURCE_MANIFEST_FULLCOPY_01.json').write_bytes(old_manifest_raw)
(HERE/'LOCAL_ADDITIONS_FULLCOPY_01.json').write_bytes((HERE/'LOCAL_ADDITIONS.json').read_bytes())
baseline = json.loads((PREV/'SOURCE_MANIFEST.json').read_bytes())['files']
records = {prior+'/'+name: entry for name, entry in baseline.items()}
additions = json.loads((HERE/'LOCAL_ADDITIONS.json').read_bytes())
additions.pop(old_name)
amendment = 'docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_THIN_20261020.md'
additions[amendment] = str(ROOT/amendment)
additions[old_name] = str(ROOT/old_name)
for name, path in additions.items():
    raw = Path(path).read_bytes()
    records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
assert len(records) == 191 and len(additions) == 6
config['method_contract'] = amendment
config['source_pins'] = {name: item['sha256'] for name, item in records.items()}
config['frozen_setup_config'] = prior+'/'+config['frozen_setup_config']
config['roles_file'] = prior+'/'+config['roles_file']
config['saved_arrays'] = [prior+'/'+name for name in config['saved_arrays']]
old_storage = copy.deepcopy(config['storage'])
config['storage'].update(
    remote_root='/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020',
    windows_root='Z:/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020',
    immutable_source_reuse_root=prior, local_text_files=7, external_immutable_sources=185)
now = datetime.now(timezone.utc).isoformat()
config['pre_execution_storage_amendment'] = dict(UTC=now, original_config=old_name,
    original_config_sha256=hashlib.sha256(old_raw).hexdigest(), original_storage=old_storage,
    scientific_calls_before_amendment=0, scientific_settings_and_budgets_unchanged=True)
name = 'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json'
(ROOT/name).write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8', newline='\n')
raw = (ROOT/name).read_bytes()
digest = hashlib.sha256(raw).hexdigest()
records[name] = dict(bytes=len(raw), sha256=digest)
additions[name] = str(ROOT/name)
manifest = dict(schema='n01/recentered-derivative-completion-thin-source/v1', config_sha256=digest,
    prior_NAS_root=prior, files=records, external_prefix_files=config['completed_prefix_inputs'],
    local_text_files=7, external_immutable_sources=185, bound_files=192)
assert len(records) == 192 and len(config['source_pins']) == 191
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'LOCAL_ADDITIONS.json').write_text(json.dumps(additions, indent=2)+'\n', encoding='utf-8', newline='\n')
for filename in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    target = ROOT/filename
    d = json.loads(target.read_bytes())
    assert d['active_scientific_jobs'] == 0 and d['pending_job_id'] == config['job_id']
    d.update(updated_UTC=now, current_action='Fixed completion uses seven new text files and 185 read-only pinned NAS sources after partial-copy deployment timeouts.')
    d['joint_response_01'].update(recentered_derivative_completion_config=name,
        recentered_derivative_completion_config_sha256=digest,
        recentered_derivative_completion_remote_root=config['storage']['remote_root'],
        recentered_derivative_completion_storage_amendment='receipts/joint_response_01_20261020/recentered_derivative_completion_01/STORAGE_AMENDMENT.json')
    if filename == 'jobs.json':
        job = d['jobs'][-1]
        assert job['status'] == 'registered_not_started'
        job.update(config=name, config_sha256=digest, remote_root=config['storage']['remote_root'],
            source_files=192, source_pins=191, local_text_files=7, external_immutable_sources=185,
            pre_execution_storage_amendment=config['pre_execution_storage_amendment'])
    target.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
receipt = dict(status='STORAGE_LAYOUT_AMENDED_BEFORE_SCIENCE', UTC=now, config=name,
    config_sha256=digest, source_files=192, source_pins=191, local_text_files=7,
    external_immutable_sources=185, external_prefix_files=12, new_scientific_calls=0,
    original_config_preserved=True, original_failed_job_preserved=True,
    scientific_settings_and_budgets_unchanged=True)
(HERE/'STORAGE_AMENDMENT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(receipt))
