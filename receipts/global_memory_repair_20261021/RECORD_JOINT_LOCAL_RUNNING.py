from pathlib import Path
from datetime import datetime, timezone
import json

root = Path('E:/GAN/research/N01')
out = Path(__file__).parent / 'joint_local_01'
startup = json.loads((out / 'NATIVE_FIRST_STEP_RECEIPT.json').read_bytes())
progress = startup['run_01/PROGRESS.json']
dispatch = json.loads((out / 'DISPATCH_RECEIPT.json').read_bytes())
steps = [json.loads(s) for s in startup['run_01/TRAINING_STEPS.jsonl'].splitlines()]
assert startup['control_01/exit_receipt.json'] is None and startup['run_01/FAILED.json'] is None
assert progress['execution_counts']['committed_optimizer_updates'] == 1
first = steps[0]
assert first['accepted'] and all(first['trials'][-1]['checks'].values())
stamp = datetime.now(timezone.utc).isoformat()
wake = dict(status='ACTIVE_SINGLE_WAKE_CONFIRMED', automation_id='n01-psd', UTC=stamp,
    scheduled_local='2026-10-21T04:29:00+08:00', measured_first_direction_with_setup_seconds=progress['elapsed_seconds'],
    completed_directions=1, remaining_maximum_directions=11,
    estimate_asof_UTC=progress['UTC'], estimated_remaining_seconds=360, margin_seconds=126,
    basis='First accepted step19.62s including setup. Allow30s per remaining direction with fresh global and ADC gradients, plus30s fresh endpoints, then126s margin.',
    fixed_interval_polling=False)
(out / 'WAKE_RECEIPT.json').write_bytes((json.dumps(wake, indent=2) + '\n').encode())
metrics = first['trials'][-1]['positive_metrics']
first_changes = {k: 100 * (v / first['previous_metrics'][k] - 1) for k, v in metrics.items()
                 if first['previous_metrics'][k] != 0}
(out / 'FIRST_ACCEPTED_UPDATE.json').write_bytes((json.dumps(dict(
    status='NATIVE_HARD_JOINT_STEP_ACCEPTED', UTC=progress['UTC'], new_committed_updates=1,
    inherited_updates=1, all_hard_checks_pass=True, training_metric_changes_percent=first_changes,
    denominator='Incoming checkpoint on the same two TRAIN random banks; not independent endpoint.',
    initial_radius=first['secant_radius'], next_radius=first['next_secant_radius'],
    scientific_acceptance=False), indent=2) + '\n').encode())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_joint_local_training_running', active_scientific_jobs=1,
        active_job_id='GLOBAL_MEMORY_JOINT_LOCAL_PILOT_20261021', active_run_id='global_memory_joint_local_20261021',
        latest_run_id='GLOBAL_MEMORY_JOINT_LOCAL_PILOT_20261021', pending_job_id=None, pending_registered_job_id=None,
        pending_next_run_id=None, current_job_native_exit_code=None, current_job_ended_UTC=None,
        training_pid=startup['control_01/resource_status.json']['child_tree'][-1],
        supervisor_pid=startup['control_01/launch.json']['supervisor_pid'], updated_UTC=stamp,
        execution_state_asof_UTC=progress['UTC'],
        current_action='Joint local native constrained training running; first new update passes Q/wave/peak/PSD/ADC guards.',
        next_action='At measured04:29 single wake collect native completion and fresh endpoints, render figures, apply unchanged gates.',
        followup_automation_id='n01-psd', followup_automation_status='ACTIVE', followup_status='ACTIVE_SINGLE_MEASURED_WAKE',
        followup_next_run_local=wake['scheduled_local'], followup_scheduled_local=wake['scheduled_local'], next_wake_local=wake['scheduled_local'])
    d['followup'] = dict(status='ACTIVE', automation_id='n01-psd', kind='heartbeat', count=1,
        next_wake_local=wake['scheduled_local'], basis=str((out / 'WAKE_RECEIPT.json').relative_to(root)))
    d['global_memory_repair'].update(status='joint_local_training_running', active_scientific_jobs=1,
        current_committed_training_updates=1, inherited_checkpoint_updates=1,
        total_accepted_updates_across_pilots=9, counts_asof_UTC=progress['UTC'],
        current_native_execution_counts=progress['execution_counts'], remote_root=dispatch['remote_root'],
        startup_receipt=str((out / 'NATIVE_FIRST_STEP_RECEIPT.json').relative_to(root)),
        dispatch_receipt=str((out / 'DISPATCH_RECEIPT.json').relative_to(root)),
        wake_receipt=str((out / 'WAKE_RECEIPT.json').relative_to(root)), automatic_training_continuation=True)
    if name == 'jobs.json':
        assert len(d['jobs']) == 158 and d['jobs'][-1]['job_id'] == d['active_job_id']
        d['jobs'][-1].update(status='running_first_joint_hard_update_accepted',
            dispatch_UTC=dispatch['UTC'], boot_id=dispatch['boot_id'], remote_root=dispatch['remote_root'],
            dispatcher_pid=dispatch['dispatcher_pid'], actual_execution_counts_last_verified=progress['execution_counts'],
            counts_asof_UTC=progress['UTC'], committed_optimizer_updates_last_verified=1)
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps(dict(wake=wake, first_update_changes=first_changes)))
