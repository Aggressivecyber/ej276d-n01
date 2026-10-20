from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
dispatch = json.loads((HERE/'DISPATCH_RECEIPT.json').read_bytes())
bound = json.loads((HERE/'STARTUP_STATUS_MEASURED.json').read_bytes())
event = json.loads((HERE/'STARTUP_PROGRESS_EVENT.json').read_bytes())
config_raw = (ROOT/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json').read_bytes()
p = json.loads(config_raw)
assert hashlib.sha256(config_raw).hexdigest() == dispatch['config_sha256'] == bound['config_sha256'] == 'e5727dcb8356a2212a6d1c05fe0e2ad471dab60661c4c97f69ffb8c355c10ce1'
assert bound['source_files_verified'] == 192 and len(p['source_pins']) == 191
assert dispatch['boot_id'] == bound['boot_id'] == event['boot_id'] == '83c9f761-c124-42e1-89b8-5f45f0b3acc4'
processes = {row['pid']: row for row in bound['processes']}
expected = {231686: '5524169', 231687: '5524177', 231735: '5524922', 231736: '5524922'}
assert set(processes) == set(expected)
for pid, ticks in expected.items():
    assert processes[pid]['start_ticks'] == ticks
assert processes[231736]['argv'][2].endswith('/scripts/complete_joint01_recentered_derivatives.py')
assert event['science_pid'] == 231736 and event['science_start_ticks'] == expected[231736]
for name in ('run_01/FAILED.json', 'run_01/COMPLETE.json', 'control_01/exit_receipt.json'):
    assert name not in bound['files'] and name not in event['files']
progress = event['files']['run_01/PROGRESS.json']
c = progress['execution_counts']
b = p['operation_budget']
assert progress['stage'] == 'recentered_response_Jacobian'
assert (progress['seed'], progress['role'], progress['column']) == (0, 'IID3', 27)
assert c['finite_difference_columns'] == 27 and c['joint_response_calls'] == 56 == 2+2*27
assert c['joint_response_rows'] == 56*32768 == 1835008
assert c['generated_observation_calls'] == 3*56 == 168
assert c['generated_observation_rows'] == 168*32768 == 5505024
assert c['native_component_rows'] == 56*110592 == 6193152
assert c['saved_prefix_libraries_reused'] == 6 and c['saved_prefix_FD_columns_reused'] == 318
assert c['allocated_banks'] == c['noise005_calls'] == 1
assert c['noise005_rows'] == 32768 and c['shape_standard_normal_values'] == 262144 and c['extension_rows'] == 98304
assert c['REAL_fit_feature_rows'] == 4096
assert all(c[key] == 1 for key in ('saved_previous_center_replays_verified', 'saved_initial_anchors_reused', 'saved_new_center_replays_verified', 'fixed_control_checks'))
for key in ('joint_direction_solves', 'new_parameter_candidates', 'REAL_response_updates', 'DEV_waveforms_decoded', 'AD_queries', 'classifier_fits', 'REAL_CHECK_feature_rows'):
    assert b[key] == 0 and key not in c
for key in ('checkpoint_loads', 'TRAIN_archives_decoded', 'PCA_archives_decoded', 'old_A_calls', 'old_A_rows', 'M0_head_calls', 'M0_head_rows', 'REAL_observable_replay_calls', 'REAL_observable_replay_rows', 'feature_maps_constructed', 'new_native_projection_contexts'):
    assert c[key] == b[key], key
resource = event['files']['control_01/resource_status.json']
assert resource['supervisor_pid'] == 231687 and resource['child_pid'] == 231735
assert resource['peak_rss_bytes'] < 12*1024**3
assert hashlib.sha256((ROOT/'src/n01/joint01_model.py').read_bytes()).hexdigest() == 'b9e11ea99abb738af7a396728ba17e40d3c548c2d8638eea1e37a881534497e6'
timing = progress['native_timing']
assert timing['calls'] == c['joint_response_calls'] == 56
remaining_calls = b['joint_response_calls']-c['joint_response_calls']
remaining_columns = b['finite_difference_columns']-c['finite_difference_columns']
assert remaining_calls == 808 == 2*26+7*108 and remaining_columns == 397 == 26+7*53
native_seconds_per_call = timing['seconds']/timing['calls']
progress_time = datetime.fromisoformat(progress['UTC'])
launch = bound['files']['control_01/launch.json']
launch_time = datetime.fromisoformat(launch['started_utc'])
native_projection = progress_time+timedelta(seconds=native_seconds_per_call*remaining_calls)
planning_projection = native_projection+timedelta(seconds=.15*native_seconds_per_call*remaining_calls+30)
TERM_time = launch_time+timedelta(seconds=p['resources']['hard_wall_seconds'])
KILL_time = TERM_time+timedelta(seconds=20)
wake_base = max(planning_projection, KILL_time+timedelta(seconds=30))
wake = datetime.fromtimestamp(math.ceil(wake_base.timestamp()/60)*60, timezone.utc)
local_zone = timezone(timedelta(hours=8))
wait = dict(status='ONE_WAKE_ESTIMATED_FROM_PROGRESS_AND_ORIGINAL_BUDGET', UTC=datetime.now(timezone.utc).isoformat(),
    snapshot_UTC=event['UTC'], progress_UTC=progress['UTC'], native_calls_measured=timing['calls'],
    native_seconds_measured=timing['seconds'], native_seconds_per_call=native_seconds_per_call,
    remaining_generator_calls=remaining_calls, remaining_FD_columns=remaining_columns, remaining_libraries_to_allocate=7,
    native_only_remaining_seconds=native_seconds_per_call*remaining_calls,
    native_only_projected_UTC=native_projection.isoformat(),
    native_timing_excludes_generation_features_NAS_IO_and_setup=True,
    provisional_full_projection_UTC=planning_projection.isoformat(),
    provisional_allowance='15 percent of native-only remaining time plus30 seconds; unmeasured generation/features/IO may differ.',
    full_duration_estimate_independently_measured=False,
    outer_timeout_TERM_UTC=TERM_time.isoformat(), outer_timeout_KILL_setting_UTC=KILL_time.isoformat(),
    actual_exit_verified=False, output_receipt_margin_after_KILL_setting_seconds=30,
    scheduled_wake_UTC=wake.isoformat(), scheduled_wake_local=wake.astimezone(local_zone).isoformat(),
    scheduling_rule='Ceil minute of later provisional projection or original TERM+20s KILL setting+30s receipt margin; one wake, no fixed polling.',
    scientific_internal_seconds=1800, scientific_external_seconds=2100, scientific_budget_extended=False,
    expected_native_calls_total=864, expected_FD_columns_total=424, new_solves=0, new_candidates=0, parameter_updates=0)
assessment = dict(status='PASS_CORRECT_NATIVE_COMPLETION_STARTUP', UTC=wait['UTC'],
    source_verification_snapshot_UTC=bound['UTC'], progress_snapshot_UTC=event['UTC'],
    boot_id=event['boot_id'], PID_start_ticks={str(pid): ticks for pid, ticks in expected.items()},
    config_sha256=dispatch['config_sha256'], source_files_verified=192, source_pins=191,
    original_prefix_libraries_reused=6, original_prefix_FD_columns_reused=318,
    progress=progress, resource=resource, registered_resources=p['resources'],
    new_solves=0, new_candidates=0, committed_parameter_updates=0, scientific_acceptance=False,
    terminal_audit_executed=False, private_NAS_session=True, no_new_large_local_or_compute_disk_payload_copies=True,
    source_proofs={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
        ('DISPATCH_RECEIPT.json', 'STARTUP_STATUS_MEASURED.json', 'STARTUP_PROGRESS_EVENT.json', 'PRE_DISPATCH_REVIEW.json')})
(HERE/'STARTUP_ASSESSMENT.json').write_text(json.dumps(assessment, indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'WAIT_ESTIMATE.json').write_text(json.dumps(wait, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status=assessment['status'], progress_UTC=progress['UTC'], new_FD_columns=27,
    generator_calls=56, native_seconds_per_call=native_seconds_per_call, remaining_calls=808,
    native_only_projected_UTC=native_projection.isoformat(), wake_local=wait['scheduled_wake_local'],
    TERM_UTC=TERM_time.isoformat(), KILL_setting_UTC=KILL_time.isoformat(), scientific_budget_extended=False)))
