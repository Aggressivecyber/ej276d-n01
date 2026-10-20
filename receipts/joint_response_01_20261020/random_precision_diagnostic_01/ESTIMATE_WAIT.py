from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,math

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
s=json.loads((HERE/'WAIT_ESTIMATE_RAW.json').read_bytes());start=json.loads((HERE/'STARTUP_ASSESSMENT.json').read_bytes());p=json.loads((ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json').read_bytes())
assert s['boot_id']==start['boot_id'] and s['source_files_verified']==879 and 'run_01/FAILED.json' not in s['files'] and 'control_01/exit_receipt.json' not in s['files']
assert any(v['pid']==start['science_pid'] and v['start_ticks']==start['science_start_ticks'] for v in s['processes'])
progress=s['files']['run_01/PROGRESS.json'];launch=s['files']['control_01/launch.json'];begin=datetime.fromisoformat(launch['started_utc']);observed=datetime.fromisoformat(progress['UTC']);calls=progress['execution_counts']['joint_response_calls']
elapsed=(observed-begin).total_seconds();rate=elapsed/calls;remaining=p['operation_budget']['joint_response_calls']-calls
remaining_seconds=remaining*rate+10;expected=observed+timedelta(seconds=remaining_seconds)
wake_seconds=math.ceil((expected+timedelta(seconds=.15*remaining_seconds)).timestamp()/60)*60;wake=datetime.fromtimestamp(wake_seconds,timezone.utc)
internal=begin+timedelta(seconds=p['resources']['internal_seconds']);external=begin+timedelta(seconds=p['resources']['hard_wall_seconds']+20)
assert expected<internal and wake<external
record=dict(status='BOUNDED_MEASURED_WAIT_ESTIMATE',UTC=s['UTC'],job_id=p['job_id'],source_files_verified=879,source_pins=82,boot_id=s['boot_id'],science_pid=start['science_pid'],science_start_ticks=start['science_start_ticks'],launch_UTC=begin.isoformat(),progress_UTC=observed.isoformat(),execution_counts=progress['execution_counts'],progress_position={k:progress[k] for k in ('stage','seed','replicate','scheme','column')},elapsed_seconds=elapsed,completed_generator_calls=calls,maximum_generator_calls=1800,remaining_generator_calls=remaining,measured_wall_seconds_per_generator_call=rate,includes_amortized_setup_and_completed_io=True,native_seconds_per_call=progress['native_timing']['seconds']/progress['native_timing']['calls'],summary_and_final_output_allowance_seconds=10,remaining_seconds_estimate=remaining_seconds,expected_completion_UTC=expected.isoformat(),expected_completion_local=expected.astimezone(timezone(timedelta(hours=8))).isoformat(),margin_fraction_of_remaining=.15,margin_seconds=(wake-expected).total_seconds(),single_wake_UTC=wake.isoformat(),single_wake_local=wake.astimezone(timezone(timedelta(hours=8))).isoformat(),earliest_internal_deadline_UTC=internal.isoformat(),external_latest_UTC=external.isoformat(),scientific_acceptance=False,new_training_updates=0,uncertainty='Shared GPU load and scheme-dependent metric cost may change. No automatic budget extension or restart; if still active, recompute one bounded wake from actual progress.')
(HERE/'WAIT_ESTIMATE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=json.loads((ROOT/name).read_bytes());d.update(last_verified_UTC=s['UTC'],execution_state_asof_UTC=s['UTC']);d['joint_response_01'].update(random_precision_diagnostic_counts_last_verified=progress['execution_counts'],random_precision_diagnostic_wait_estimate=(HERE/'WAIT_ESTIMATE.json').relative_to(ROOT).as_posix())
 (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
print(json.dumps(record,indent=2))
