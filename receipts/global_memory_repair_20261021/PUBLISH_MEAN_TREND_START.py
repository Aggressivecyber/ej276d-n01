from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import tarfile
import tomllib

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
run = here / 'mean_trend_fix02'
out = here / 'github_08'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
automation = tomllib.loads(Path('C:/Users/10629/.codex/automations/n01-psd/automation.toml').read_text('utf-8'))
assert automation['status'] == 'ACTIVE' and 'COUNT=1' in automation['rrule']
assert automation['target_thread_id'] == '01a12bad-8839-7522-89a5-f028b107d7ff'
observed = json.loads((run / 'FIRST_DIRECTION_01.json').read_bytes())
progress = observed['run_01/PROGRESS.json']
assert observed['run_01/FAILED.json'] is None and observed['control_01/exit_receipt.json'] is None
wake = dict(status='ACTIVE_SINGLE_WAKE_CONFIRMED', UTC=stamp, automation_id='n01-psd',
    scheduled_local='2026-10-21T16:24:30+08:00', measured_setup_seconds=22.992714635001903,
    measured_first_direction_with_setup_seconds=162.44779974200355,
    measured_direction_seconds=139.45508510700165, first_direction_accepted=False,
    estimate_basis='First direction rejected. Two further139.5s directions would reach the3-reject stop; allow60s endpoints plus about105s margin from the08:17:06Z completion. If any later direction commits, reestimate the remaining24-direction budget at this wake.',
    fixed_interval_polling=False, one_time_count=1)
(run / 'WAKE_RECEIPT.json').write_bytes((json.dumps(wake, indent=2) + '\n').encode())
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    d = json.loads((root / name).read_bytes())
    d.update(updated_UTC=stamp, execution_state_asof_UTC=observed['UTC'],
        followup_automation_id='n01-psd', followup_automation_status='ACTIVE',
        followup_status='SINGLE_ESTIMATED_STOP_OR_PROGRESS_WAKE',
        followup_scheduled_local=wake['scheduled_local'], next_wake_local=wake['scheduled_local'])
    d['global_memory_repair'].update(current_native_execution_counts=progress['execution_counts'],
        counts_asof_UTC=progress['UTC'], current_committed_training_updates=0,
        mean_trend_wake_receipt=(run / 'WAKE_RECEIPT.json').relative_to(root).as_posix(),
        mean_trend_first_direction_accepted=False)
    if name == 'jobs.json':
        d['jobs'][-1].update(last_verified_counts=progress['execution_counts'], counts_asof_UTC=progress['UTC'],
            first_direction_accepted=False, committed_optimizer_updates=0,
            wake_receipt=(run / 'WAKE_RECEIPT.json').relative_to(root).as_posix())
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with tarfile.open(run / 'runtime.tar') as archive:
    for name in ['src/n01/mean_trend01.py', 'scripts/run_mean_trend01.py',
                 'scripts/check_mean_trend01.py', 'configs/mean_trend_repair_fix02_20261021.json']:
        assert archive.extractfile(name).read() == (root / name).read_bytes(), name
old = json.loads(subprocess.check_output(['git', 'show', 'HEAD:jobs.json'], cwd=root))['jobs']
assert len(old) == 163 and json.loads((root / 'jobs.json').read_bytes())['jobs'][:163] == old
files = ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'JOINT_RESPONSE_01_PLAN.md',
    'src/n01/mean_trend01.py', 'scripts/run_mean_trend01.py', 'scripts/check_mean_trend01.py',
    'scripts/plot_global_memory01.py', 'configs/mean_trend_repair_20261021.json',
    'configs/mean_trend_repair_fix01_20261021.json', 'configs/mean_trend_repair_fix02_20261021.json']
files += [(here / name).relative_to(root).as_posix() for name in ['PREPARE_MEAN_TREND.py',
    'RECOVER_MEAN_TREND.py', 'RECOVER_MEAN_TREND_FIX02.py', 'RECORD_MEAN_TREND_START.py',
    'COLLECT_MEAN_TREND.py', 'COLLECT_MEAN_TREND_FIX01.py', 'COLLECT_MEAN_TREND_FIX02.py',
    'MEAN_TREND_REPAIR_DESIGN.md', 'PUBLISH_MEAN_TREND_START.py']]
for folder, number in [('mean_trend_01', 164), ('mean_trend_fix01', 165)]:
    names = ['REGISTRATION_RECEIPT.json', 'DISPATCH_RECEIPT.json', 'FAILED.json', 'exit_receipt.json', 'CLOSURE_RECEIPT.json',
        f'mean_trend01_AS_LAUNCHED_JOB{number}.py', f'run_mean_trend01_AS_LAUNCHED_JOB{number}.py']
    if number == 164:
        names += ['SYNTHETIC_CHECKS.json']
    else:
        names += ['IDENTITY_DIAGNOSTIC_REGISTRATION.json', 'IDENTITY_DIAGNOSTIC_STDOUT.json', 'DIAGNOSE_IDENTITY.py',
            'GPU_SYNTHETIC_CHECKS.json', 'GPU_DELTA_SYNTHETIC_CHECKS.json']
    files += [(here / folder / name).relative_to(root).as_posix() for name in names]
files += [(run / name).relative_to(root).as_posix() for name in ['REGISTRATION_RECEIPT.json',
    'DISPATCH_RECEIPT.json', 'GPU_SYNTHETIC_CHECKS.json', 'STARTUP_SCIENTIFIC_RECEIPT.json',
    'BINDINGS.json', 'TRAINING_STEPS.jsonl', 'WAKE_RECEIPT.json']]
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=root)
records = {}
for name in files:
    path = root / name
    assert path.suffix in ('.py', '.md', '.json', '.jsonl')
    path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    raw = path.read_bytes()
    records[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
(out / 'PUBLICATION_MANIFEST.json').write_bytes((json.dumps(records, indent=2) + '\n').encode())
subprocess.run(['git', 'add', '--', *files], cwd=root, check=True, capture_output=True)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=root, check=True, capture_output=True)
for name in files:
    assert subprocess.check_output(['git', 'show', ':' + name], cwd=root) == (root / name).read_bytes()
r = subprocess.run(['git', 'commit', '-m', 'Train conditioned mean with waveform trend and preserve native startup failures'], cwd=root, check=True, capture_output=True)
(out / 'COMMIT_STDOUT.txt').write_bytes(r.stdout)
r = subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=root, check=True, capture_output=True)
(out / 'PUSH_STDOUT.txt').write_bytes(r.stdout)
(out / 'PUSH_STDERR.txt').write_bytes(r.stderr)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip()
assert subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root).decode().split()[0] == head
assert not subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=root)
receipt = dict(status='COMMITTED_PUSHED_REMOTE_MAIN_VERIFIED', UTC=datetime.now(timezone.utc).isoformat(),
    commit=head, files=len(files), public_bytes=sum(v['bytes'] for v in records.values()),
    first163_jobs_unchanged=True, original_and_failed_runtime_sources_preserved=True,
    current_job=166, scientific_acceptance=False, new_accepted_updates_confirmed=0,
    single_wake_local=wake['scheduled_local'], private_arrays_and_checkpoints_not_published=True,
    NAS_file_IO_checks=False, NAS_file_hash_verification=False)
(out / 'DELIVERY_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
