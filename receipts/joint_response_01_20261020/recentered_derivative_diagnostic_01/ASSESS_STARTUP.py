from datetime import datetime, timedelta, timezone
from pathlib import Path
import copy
import json
import math

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
rel = HERE.relative_to(ROOT).as_posix()
status = json.loads((HERE/'STARTUP_STATUS_RAW.json').read_bytes())
dispatch = json.loads((HERE/'DISPATCH_RECEIPT.json').read_bytes())
config = json.loads((ROOT/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json').read_bytes())
files = status['files']
progress = files['run_01/PROGRESS.json']
counts = progress['execution_counts']
launch = files['control_01/launch.json']
processes = {p['pid']: p for p in status['processes']}
assert status['boot_id'] == dispatch['boot_id']
assert processes[dispatch['dispatcher_pid']]['start_ticks'] == dispatch['dispatcher_start_ticks']
assert status['source_files_verified'] == 185
assert status['config_sha256'] == dispatch['config_sha256']
assert not any(name in files for name in ('run_01/FAILED.json', 'run_01/COMPLETE.json', 'control_01/exit_receipt.json'))
science = [p for p in status['processes'] if p['argv'][0].endswith('/python') and p['argv'][2].endswith('/diagnose_joint01_recentered_derivatives.py')]
assert len(science) == 1
assert launch['supervisor_pid'] in processes and launch['child_pid'] in processes
assert counts['finite_difference_columns'] == 71
assert counts['saved_previous_center_replays_verified'] == counts['saved_new_center_replays_verified'] == counts['fixed_control_checks'] == 2
assert counts['joint_response_calls'] == 146
assert files['control_01/resource_status.json']['peak_rss_bytes'] < 12*1024**3
for key, value in counts.items():
    assert value <= config['operation_budget'][key], key

assessment = dict(
    status='PASS_CORRECT_STARTUP', UTC=status['UTC'], progress_UTC=progress['UTC'],
    job_id=config['job_id'], boot_id=status['boot_id'], config_sha256=status['config_sha256'],
    source_files_verified=185, source_pins=184, process_identities=status['processes'],
    science_pid=science[0]['pid'], science_start_ticks=science[0]['start_ticks'],
    supervisor_pid=launch['supervisor_pid'], execution_counts=counts,
    peak_RSS_bytes=files['control_01/resource_status.json']['peak_rss_bytes'],
    original_and_recentered_saved_outputs_replayed=True,
    scientific_payloads_written_directly_to_NAS=True,
    full_terminal_audit_prepared=True, full_terminal_audit_executed=False,
    parameter_updates=0, scientific_acceptance=False,
    remote_progress_after_snapshot='not observed')
(HERE/'STARTUP_ASSESSMENT.json').write_text(json.dumps(assessment, indent=2)+'\n', encoding='utf-8')

start = datetime.fromisoformat(launch['started_utc'])
asof = datetime.fromisoformat(progress['UTC'])
elapsed = (asof-start).total_seconds()
rate = elapsed/counts['joint_response_calls']
remaining_calls = config['operation_budget']['joint_response_calls']-counts['joint_response_calls']
remaining = remaining_calls*rate+10
expected = asof+timedelta(seconds=remaining)
unrounded = expected+timedelta(seconds=.15*remaining)
wake = datetime.fromtimestamp(math.ceil(unrounded.timestamp()/60)*60, timezone.utc)
hard_end = start+timedelta(seconds=config['resources']['hard_wall_seconds']+20)
assert wake < hard_end
local = timezone(timedelta(hours=8))
estimate = dict(
    status='BOUNDED_MEASURED_WAIT_ESTIMATE', UTC=status['UTC'], job_id=config['job_id'],
    launch_UTC=start.isoformat(), progress_UTC=asof.isoformat(), elapsed_seconds=elapsed,
    completed_generator_calls=counts['joint_response_calls'],
    maximum_generator_calls=config['operation_budget']['joint_response_calls'],
    remaining_generator_calls=remaining_calls, measured_wall_seconds_per_generator_call=rate,
    includes_amortized_setup_and_completed_io=True,
    native_seconds_per_call=progress['native_timing']['seconds']/progress['native_timing']['calls'],
    summary_and_final_output_allowance_seconds=10, remaining_seconds_estimate=remaining,
    expected_completion_UTC=expected.isoformat(), expected_completion_local=expected.astimezone(local).isoformat(),
    margin_fraction_of_remaining=.15, margin_seconds=(wake-expected).total_seconds(),
    single_wake_UTC=wake.isoformat(), single_wake_local=wake.astimezone(local).isoformat(),
    earliest_internal_deadline_UTC=(start+timedelta(seconds=3600)).isoformat(),
    external_latest_UTC=hard_end.isoformat(), scientific_acceptance=False,
    uncertainty='Shared GPU load and NAS I/O can change. No automatic restart or budget extension; re-estimate once only if still waiting.')
(HERE/'WAIT_ESTIMATE.json').write_text(json.dumps(estimate, indent=2)+'\n', encoding='utf-8')

for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((ROOT/name).read_bytes())
    assert d['pending_job_id'] == config['job_id'] and d['active_scientific_jobs'] == 0
    old_jobs = copy.deepcopy(d['jobs']) if name == 'jobs.json' else None
    d.update(status='JOINT_RESPONSE_01_recentered_derivative_running',
        updated_UTC=status['UTC'], execution_state_asof_UTC=status['UTC'],
        active_scientific_jobs=1, active_job_id=config['job_id'], pending_job_id=None,
        training_pid=science[0]['pid'], supervisor_pid=launch['supervisor_pid'],
        active_scientific_jobs_status_basis='Verified remote process identities, saved-center replays, positive finite-difference progress and all deployed file hashes.',
        scientific_acceptance=False,
        current_action='Collecting fresh full response derivatives at the fixed measured control on NAS; 71/742 columns verified.',
        next_action='Create the measured single wake; finish local review and GitHub publication while the bounded collection runs.')
    d['joint_response_01'].update(
        recentered_derivative_status='running_correct_startup_verified',
        recentered_derivative_dispatch_receipt=rel+'/DISPATCH_RECEIPT.json',
        recentered_derivative_startup=rel+'/STARTUP_ASSESSMENT.json',
        recentered_derivative_counts_last_verified=counts,
        recentered_derivative_complete_terminal_audit_executed=False)
    if name == 'jobs.json':
        assert len(d['jobs']) == 141 and d['jobs'][-1]['job_id'] == config['job_id']
        d['jobs'][-1].update(status='running_correct_startup_verified',
            dispatched_UTC=dispatch['UTC'], verified_UTC=status['UTC'],
            boot_id=status['boot_id'], science_pid=science[0]['pid'],
            science_start_ticks=science[0]['start_ticks'], supervisor_pid=launch['supervisor_pid'],
            dispatch_receipt=rel+'/DISPATCH_RECEIPT.json', startup=rel+'/STARTUP_ASSESSMENT.json',
            execution_counts_last_verified=counts)
        assert d['jobs'][:-1] == old_jobs[:-1]
    (ROOT/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(startup=assessment['status'], science_pid=science[0]['pid'], finite_difference_columns=71, **estimate)))
