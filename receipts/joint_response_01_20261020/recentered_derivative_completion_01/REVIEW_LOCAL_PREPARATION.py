from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'recentered_derivative_diagnostic_01'
manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
p = json.loads((ROOT/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json').read_bytes())
old_inputs = json.loads((PREV/'INPUT_MAP.json').read_bytes())
additions = json.loads((HERE/'LOCAL_ADDITIONS.json').read_bytes())
py_count = 0
for name, entry in manifest['files'].items():
    relative = name.removeprefix(manifest['prior_NAS_root']+'/')
    raw = Path(additions[name] if name in additions else old_inputs[relative]).read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], name
    if name.endswith('.py'):
        ast.parse(raw.decode('utf-8'))
        py_count += 1
assert len(manifest['files']) == 192 and len(p['source_pins']) == 191
assert manifest['external_prefix_files'] == p['completed_prefix_inputs']
assert len(p['completed_prefix_inputs']) == 12
assert p['completed_prefix_labels'] == {'0': ['FIT0', 'FIT1', 'FIT2', 'IID0', 'IID1', 'IID2'], '1': []}
assert p['response_coordinates'] == 53 and p['noise_correction'] == 0 and p['finite_difference_step'] == .004
b = p['operation_budget']
assert b['finite_difference_columns'] == 8*53 == 424
assert b['joint_response_calls'] == 8*(2+2*53) == 864
assert b['joint_response_rows'] == 864*32768
assert b['generated_observation_calls'] == 3*864 and b['generated_observation_rows'] == 3*864*32768
assert b['native_component_rows'] == 864*110592
assert b['noise005_rows'] == 8*32768 and b['extension_rows'] == 3*8*32768
assert b['shape_standard_normal_values'] == 8*32768*8
assert b['REAL_fit_feature_rows'] == 8*4096
assert b['saved_prefix_libraries_reused'] == 6 and b['saved_prefix_FD_columns_reused'] == 318
prefix = json.loads((PREV/'terminal_01/PREFIX_REVIEW.json').read_bytes())
old = prefix['driver_reported_failure_counts']
total = p['cumulative_accounting']
assert old['finite_difference_columns']+b['finite_difference_columns'] == total['executed_FD_columns'] == 751
assert 318+b['finite_difference_columns'] == total['unique_FD_columns_after_completion'] == 742
for key in ('joint_response_calls', 'joint_response_rows', 'generated_observation_calls', 'generated_observation_rows', 'native_component_rows', 'allocated_banks', 'REAL_fit_feature_rows', 'checkpoint_loads', 'TRAIN_archives_decoded', 'PCA_archives_decoded', 'old_A_calls', 'old_A_rows', 'M0_head_calls', 'M0_head_rows', 'REAL_observable_replay_calls', 'REAL_observable_replay_rows', 'feature_maps_constructed', 'new_native_projection_contexts'):
    assert old[key]+b[key] == total[key], key
assert total['repeated_generator_calls'] == old['joint_response_calls']-2*(318+6) == 20
assert total['repeated_executed_but_unsaved_FD_columns'] == 9
assert all(b[key] == 0 for key in ('joint_direction_solves', 'new_parameter_candidates', 'REAL_response_updates', 'DEV_waveforms_decoded', 'AD_queries', 'classifier_fits', 'REAL_CHECK_feature_rows'))
assert hashlib.sha256((ROOT/'src/n01/joint01_model.py').read_bytes()).hexdigest() == 'b9e11ea99abb738af7a396728ba17e40d3c548c2d8638eea1e37a881534497e6'
smoke = json.loads((HERE/'SUPERVISOR_FAULT_SMOKE.json').read_bytes())
assert smoke['owned_child_absent_after_failure'] and smoke['script_sha256'] == manifest['files']['scripts/supervise_joint01_nas.py']['sha256']
assert json.loads((HERE/'ATOMIC_NPZ_SMOKE.json').read_bytes())['status'] == 'PASS_NATIVE_NAS_NPZ_FLUSH_REPLACE_READBACK'
assert not (HERE/'DEPLOYMENT_VERIFICATION.json').exists()
auditors = {}
for name in ('AUDIT_COMPLETE.py', 'RUN_TERMINAL_AUDIT_REMOTE.py', 'REMOTE_DISPATCH_REQUEST.py', 'REMOTE_STATUS_REQUEST.py'):
    raw = (HERE/name).read_bytes()
    tree = ast.parse(raw.decode('utf-8'))
    if name == 'RUN_TERMINAL_AUDIT_REMOTE.py':
        remote = next(item.value for item in tree.body if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name) and item.targets[0].id == 'remote_script')
        ast.parse(ast.literal_eval(remote))
    auditors[name] = hashlib.sha256(raw).hexdigest()
assert "native_timing']['calls']==864" in (HERE/'AUDIT_COMPLETE.py').read_text(encoding='utf-8')
jobs = json.loads((ROOT/'jobs.json').read_bytes())
prior = json.loads(subprocess.run(['git', 'show', 'HEAD:jobs.json'], cwd=ROOT, capture_output=True, check=True).stdout)
assert len(jobs['jobs']) == 142 and jobs['jobs'][:140] == prior['jobs'][:140]
assert jobs['jobs'][-2]['status'] == 'failed_storage_prefix_audited'
assert jobs['active_scientific_jobs'] == 0 and jobs['followup']['count'] == 0 and jobs['pending_job_id'] == p['job_id']
review = dict(status='PASS_LOCAL_PREPARATION_ONLY_NAS_GATE_BLOCKED', UTC=datetime.now(timezone.utc).isoformat(),
    source_files=192, source_pins=191, local_source_files_rehashed=192, external_prefix_files=12,
    source_Python_AST=py_count, auditor_sha256=auditors, exact_new_and_cumulative_budgets_recomputed=True,
    original_model_loss_readout_preserved=True, original_failure_and_first_140_jobs_preserved=True,
    terminal_auditor_prepared=True, terminal_audit_executed=False, science_started=False,
    native_deployment_verified=False, dispatch_ready=False,
    supervisor_owned_child_cleanup_fault_test_passed=True, native_NAS_NPZ_flush_replace_test_passed=True)
(HERE/'LOCAL_PREPARATION_REVIEW.json').write_text(json.dumps(review, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(review))
