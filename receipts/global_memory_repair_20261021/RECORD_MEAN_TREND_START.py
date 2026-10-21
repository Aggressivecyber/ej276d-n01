from pathlib import Path
from datetime import datetime, timezone
import json
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
out = here / 'mean_trend_fix02'
snapshot = json.loads((out / 'STARTUP_01.json').read_bytes())
assert snapshot['run_01/FAILED.json'] is None and snapshot['control_01/exit_receipt.json'] is None
binding, progress = snapshot['run_01/BINDINGS.json'], snapshot['run_01/PROGRESS.json']
assert binding['inherited_mean_identity_max_V'] <= 1e-7 and binding['fixed_memory_prefix_equal']
assert binding['trainable_parameters'] == 273174
stamp = datetime.now(timezone.utc).isoformat()
launch = snapshot['control_01/launch.json']
record = dict(status='NATIVE_DATA_AND_ZERO_START_IDENTITY_PASSED_FIRST_DIRECTION_RUNNING', UTC=stamp,
    job_number=166, boot_id=snapshot['boot_id'], launch=launch, bindings=binding, progress=progress,
    optimizer_updates_confirmed_at_progress=progress['execution_counts'].get('committed_optimizer_updates', 0),
    scientific_acceptance=False, engineering_failures_preserved=[164, 165])
(out / 'STARTUP_SCIENTIFIC_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    d = json.loads((root / name).read_bytes())
    d.update(status='N01_MEAN_TREND_repair_running', active_scientific_jobs=1,
        active_job_id='N01_MEAN_TREND_REPAIR_FIX02_20261021', active_run_id='mean_trend_repair_fix02_20261021',
        pending_job_id=None, pending_registered_job_id=None,
        updated_UTC=stamp, execution_state_asof_UTC=snapshot['UTC'],
        training_pid=launch['child_pid'], supervisor_pid=launch['supervisor_pid'],
        current_job_native_exit_code=None, current_job_ended_UTC=None,
        current_action='Native initialization and unchanged memory confirmed; conditioned-mean direction running. No repair acceptance yet.')
    d['global_memory_repair'].update(status='mean_trend_repair_running', active_scientific_jobs=1,
        current_native_execution_counts=progress['execution_counts'], counts_asof_UTC=progress['UTC'],
        current_committed_training_updates=progress['execution_counts'].get('committed_optimizer_updates', 0),
        current_startup_receipt=(out / 'STARTUP_SCIENTIFIC_RECEIPT.json').relative_to(root).as_posix(),
        current_remote_root='/home/rylan/projects/ej276d-n01/mean_trend_repair_fix02_20261021')
    if name == 'jobs.json':
        assert len(d['jobs']) == 166
        d['jobs'][-1].update(status='running', started_UTC=launch['started_utc'],
            startup_receipt=(out / 'STARTUP_SCIENTIFIC_RECEIPT.json').relative_to(root).as_posix(),
            last_verified_counts=progress['execution_counts'], counts_asof_UTC=progress['UTC'])
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
for folder, number in [('mean_trend_01', 164), ('mean_trend_fix01', 165)]:
    with tarfile.open(here / folder / 'runtime.tar') as archive:
        for source in ['src/n01/mean_trend01.py', 'scripts/run_mean_trend01.py']:
            destination = here / folder / (Path(source).stem + f'_AS_LAUNCHED_JOB{number}.py')
            destination.write_bytes(archive.extractfile(source).read())
print(json.dumps(dict(status=record['status'], identity=binding['inherited_mean_identity_max_V'],
    source_FIT=binding['source_FIT'], REAL_FIT=binding['REAL_FIT'], trainable_parameters=binding['trainable_parameters'])))
