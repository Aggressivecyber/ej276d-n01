from datetime import datetime, timezone
from pathlib import Path
import ast
import copy
import hashlib
import json

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'recentered_derivative_diagnostic_01'
prefix = json.loads((PREV/'terminal_01/PREFIX_REVIEW.json').read_bytes())
failure = json.loads((PREV/'WAKE1402_FAILURE_ASSESSMENT.json').read_bytes())
probe = json.loads((PREV/'NAS_WRITE_PROBE_1402.json').read_bytes())
smoke = json.loads((HERE/'SUPERVISOR_FAULT_SMOKE.json').read_bytes())
assert prefix['status'] == 'SAVED_PREFIX_AUDITED_INCOMPLETE_JOB'
assert prefix['source_files_verified'] == 185 and not prefix['payload_errors']
assert prefix['audit_counts']['finite_difference_columns_recomputed'] == 318
assert prefix['unavailable_executed_columns'] == 9 and prefix['unexecuted_registered_columns'] == 415
assert probe['status'] == 'PASS_NAS_WRITE_FLUSH_RENAME_READBACK'
assert smoke['status'] == 'PASS_SUPERVISOR_IO_FAILURE_STOPS_OWNED_CHILD'
assert hashlib.sha256((ROOT/'scripts/supervise_joint01_nas.py').read_bytes()).hexdigest() == smoke['script_sha256']
old_manifest = json.loads((PREV/'SOURCE_MANIFEST.json').read_bytes())
records = copy.deepcopy(old_manifest['files'])
additions = {}
for name in ('scripts/complete_joint01_recentered_derivatives.py', 'scripts/supervise_joint01_nas.py',
             'docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020.md',
             'receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/terminal_01/PREFIX_REVIEW.json'):
    raw = (ROOT/name).read_bytes()
    if name.endswith('.py'):
        ast.parse(raw.decode('utf-8'))
    additions[name] = str(ROOT/name)
    records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
assert len(records) == 189
config = copy.deepcopy(json.loads((ROOT/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json').read_bytes()))
now = datetime.now(timezone.utc).isoformat()
complete_labels = ['FIT0', 'FIT1', 'FIT2', 'IID0', 'IID1', 'IID2']
old_remote = config['storage']['remote_root']
external = {old_remote+'/'+name: entry for name, entry in prefix['output_hashes'].items()
            if any(name == f'run_01/seed0_{label}_RECENTERED.{suffix}' for label in complete_labels for suffix in ('npz', 'json'))}
assert len(external) == 12
budget = copy.deepcopy(config['operation_budget'])
budget.update(REAL_fit_feature_rows=32768, allocated_banks=8, noise005_calls=8, noise005_rows=262144,
    extension_rows=786432, shape_standard_normal_values=2097152,
    joint_response_calls=864, joint_response_rows=28311552,
    generated_observation_calls=2592, generated_observation_rows=84934656,
    native_component_rows=95551488, finite_difference_columns=424,
    saved_previous_center_replays_verified=8, saved_new_center_replays_verified=8,
    saved_initial_anchors_reused=8, fixed_control_checks=8,
    saved_prefix_libraries_reused=6, saved_prefix_FD_columns_reused=318)
config.update(schema='n01/joint-response01-recentered-derivative-completion/v1',
    job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020', registered_UTC=now,
    method_contract='docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_COMPLETION_20261020.md',
    source_pins={name: item['sha256'] for name, item in records.items()},
    completed_prefix_labels={'0': complete_labels, '1': []}, completed_prefix_inputs=external,
    completed_prefix_review='receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01/terminal_01/PREFIX_REVIEW.json',
    operation_budget=budget,
    interrupted_job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020',
    cumulative_accounting=dict(executed_FD_columns=751, unique_FD_columns_after_completion=742,
        repeated_executed_but_unsaved_FD_columns=9, joint_response_calls=1532,
        joint_response_rows=50200576, generated_observation_calls=4596,
        generated_observation_rows=150601728, native_component_rows=169426944,
        repeated_generator_calls=20, allocated_banks=15, REAL_fit_feature_rows=61440,
        checkpoint_loads=12, TRAIN_archives_decoded=2, PCA_archives_decoded=2,
        old_A_calls=104, old_A_rows=13186, M0_head_calls=208, M0_head_rows=26372,
        REAL_observable_replay_calls=2, REAL_observable_replay_rows=42094,
        feature_maps_constructed=6, new_native_projection_contexts=6),
    storage=dict(remote_root='/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_20261020',
        windows_root='Z:/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_20261020',
        large_local_or_compute_disk_backups=False, existing_data_migration_performed=False))
config['resources'].update(internal_seconds=1800, hard_wall_seconds=2100)
name = 'configs/joint_response_01_recentered_derivative_completion_20261020.json'
(ROOT/name).write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8', newline='\n')
raw = (ROOT/name).read_bytes()
digest = hashlib.sha256(raw).hexdigest()
records[name] = dict(bytes=len(raw), sha256=digest)
additions[name] = str(ROOT/name)
assert len(records) == 190 and len(config['source_pins']) == 189
(HERE/'LOCAL_ADDITIONS.json').write_text(json.dumps(additions, indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(schema='n01/recentered-derivative-completion-source/v1',
    config_sha256=digest, prior_NAS_root=old_remote, files=records, external_prefix_files=external), indent=2)+'\n', encoding='utf-8', newline='\n')
rel = HERE.relative_to(ROOT).as_posix()
old_rel = PREV.relative_to(ROOT).as_posix()
for filename in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    target = ROOT/filename
    d = json.loads(target.read_bytes())
    assert d['active_scientific_jobs'] == 0 and d['followup']['count'] == 0
    if filename == 'jobs.json':
        assert len(d['jobs']) == 141
        before = copy.deepcopy(d['jobs'])
        d['jobs'][-1].update(status='failed_storage_prefix_audited',
            prefix_review=old_rel+'/terminal_01/PREFIX_REVIEW.json', saved_FD_columns=318,
            executed_but_unsaved_FD_columns=9, unexecuted_registered_FD_columns=415,
            complete=False, native_exit_code=None, full_terminal_audit_executed=False)
        d['jobs'].append(dict(job_id=config['job_id'], status='registered_not_started', registered_UTC=now,
            config=name, config_sha256=digest, remote_root=config['storage']['remote_root'],
            operation_budget=budget, resources=config['resources'], source_files=190, source_pins=189,
            external_prefix_files=12, reused_saved_FD_columns=318, automatic_restart=False,
            automatic_training=False, parameter_updates=0, scientific_acceptance=False))
        assert d['jobs'][:140] == before[:140]
    d.update(status='JOINT_RESPONSE_01_recentered_derivative_completion_registered', updated_UTC=now,
        execution_state_asof_UTC=prefix['UTC'], active_job_id=None, pending_job_id=config['job_id'],
        current_stage='4_recentered_derivative_completion', active_work_package='4_recentered_derivative_completion',
        current_action='Six saved libraries and 318 columns audited; fixed completion of eight missing libraries registered.',
        next_action='Verify NAS deployment and prepared completion auditor, then start one bounded fixed completion.',
        followup_receipt_sha256=hashlib.sha256((PREV/'WAKE1402_AUTOMATION_CONSUMED.json').read_bytes()).hexdigest(),
        followup_scheduled_local=None)
    d['joint_response_01'].update(recentered_derivative_status='failed_storage_prefix_audited',
        recentered_derivative_prefix_review=old_rel+'/terminal_01/PREFIX_REVIEW.json',
        recentered_derivative_saved_FD_columns=318, recentered_derivative_executed_unsaved_FD_columns=9,
        recentered_derivative_completion_status='registered_not_started',
        recentered_derivative_completion_config=name, recentered_derivative_completion_config_sha256=digest,
        recentered_derivative_completion_remote_root=config['storage']['remote_root'],
        recentered_derivative_combined_complete=False)
    target.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
receipt = dict(status='REGISTERED_FIXED_COMPLETION_BEFORE_EXECUTION', UTC=now, config=name,
    config_sha256=digest, source_files=190, source_pins=189, external_prefix_files=12,
    saved_prefix_FD_columns=318, new_FD_columns=424, repeated_unsaved_columns=9,
    first_140_jobs_unchanged=True, original_failure_preserved=True, scientific_job_started=False,
    no_new_local_or_compute_disk_payload_copies=True)
(HERE/'REGISTRATION.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(receipt))
