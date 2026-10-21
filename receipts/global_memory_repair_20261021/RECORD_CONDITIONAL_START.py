from pathlib import Path
from datetime import datetime, timezone
import json

root = Path('E:/GAN/research/N01')
out = Path(__file__).parent / 'conditional_wave_01'
dispatch = json.loads((out / 'DISPATCH_RECEIPT.json').read_bytes())
launch = json.loads((out / 'launch.json').read_bytes())
startup = json.loads((out / 'STARTUP_01.json').read_bytes())
binding = startup['run_01/BINDINGS.json']
assert startup['boot_id'] == dispatch['boot_id']
assert binding['source_fit_rows'] == 6465 and binding['real_fit_rows'] == 18948
assert binding['trainable_parameters']['B'] == 247 and binding['report_rows_used_for_memory'] == 0
assert startup['run_01/FAILED.json'] is None
record = dict(status='STARTED_WITH_REAL_DATA_AND_SAVED_MODEL', UTC=startup['UTC'],
    job_number=162, boot_id=dispatch['boot_id'], launch=launch, bindings=binding,
    progress=startup['run_01/PROGRESS.json'], inherited_updates=10,
    new_committed_updates=0, scientific_acceptance=False)
(out / 'STARTUP_SCIENTIFIC_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_conditional_wave_probe_running', active_scientific_jobs=1,
        active_job_id='GLOBAL_MEMORY_CONDITIONAL_WAVE_PROBE_20261021',
        active_run_id='GLOBAL_MEMORY_CONDITIONAL_WAVE_PROBE_20261021',
        pending_job_id=None, pending_registered_job_id=None, pending_next_run_id=None,
        training_pid=launch['child_pid'], supervisor_pid=launch['supervisor_pid'],
        current_job_native_exit_code=None, current_job_ended_UTC=None,
        updated_UTC=startup['UTC'], execution_state_asof_UTC=startup['UTC'],
        current_action='Job162 constructed TRAIN-only conditional wave memory and loaded fixed job158 B; bounded native direction check running.')
    d['global_memory_repair'].update(status='conditional_wave_probe_running', active_scientific_jobs=1,
        current_committed_training_updates=0, inherited_checkpoint_updates=10,
        current_native_execution_counts=startup['run_01/PROGRESS.json']['execution_counts'],
        counts_asof_UTC=startup['run_01/PROGRESS.json']['UTC'],
        current_startup_receipt=str((out / 'STARTUP_SCIENTIFIC_RECEIPT.json').relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 162
        d['jobs'][-1].update(status='running', started_UTC=launch['started_utc'],
            supervisor_pid=launch['supervisor_pid'], training_pid=launch['child_pid'],
            start_receipt=str((out / 'STARTUP_SCIENTIFIC_RECEIPT.json').relative_to(root)))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps(dict(status=record['status'], job_number=162)))
