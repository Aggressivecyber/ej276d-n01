"""One fixed 028 driver, one inherited N01 lock, exact native exit; no retry."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--config-sha256', required=True)
args = parser.parse_args()
root = Path('/home/rylan/projects/ej276d-n01/wp6_028_20261017')
if Path.cwd() != root or Path(args.config).resolve() != root/'wp6_028_config.json':
    raise ValueError('028 exact owned cwd/config required')
affinity = sorted(os.sched_getaffinity(0))
if len(affinity) < 1:
    raise ValueError('028 requires one available CPU')
os.sched_setaffinity(0, affinity[:1])
environment = os.environ.copy()
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    environment[name] = '1'
environment['PYTHONUNBUFFERED'] = '1'
environment['PYTHONDONTWRITEBYTECODE'] = '1'
(root/'tmp').mkdir(mode=0o700, exist_ok=False)
(root/'cache').mkdir(mode=0o700, exist_ok=False)
environment.update(TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'), PYTHONPATH=str(root/'src'))
lock = (root.parent/'.n01_scientific_dispatch.lock').open('a+b')
fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
directory = root/'runs/primary/supervisor'
directory.mkdir(parents=True, exist_ok=False)
output = root/'runs/primary/artifacts'
if output.exists():
    raise FileExistsError('028 output must be fresh')
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
supervisor_fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
supervisor = dict(pid=os.getpid(), parent_pid=os.getppid(), start_ticks=int(supervisor_fields[19]),
    command=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').split(b'\0')],
    cwd=str(root), boot_id=boot, CPU_affinity=affinity[:1])
command = [sys.executable, '-B', str(root/'scripts/run_wp6_028.py'),
    '--config', str(root/'wp6_028_config.json'), '--config-sha256', args.config_sha256,
    '--output', str(output)]
started = time.monotonic()
log_path = directory/'driver_console.log'
with log_path.open('xb') as log:
    child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
    try:
        fields = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()
        identity = dict(pid=child.pid, parent_pid=int(fields[1]), start_ticks=int(fields[19]),
            process_group=os.getpgid(child.pid),
            command=[v.decode() for v in Path('/proc', str(child.pid), 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')],
            cwd=os.readlink('/proc/'+str(child.pid)+'/cwd'), boot_id=boot,
            CPU_affinity=sorted(os.sched_getaffinity(child.pid)))
    except (FileNotFoundError, ProcessLookupError):
        identity = dict(pid=child.pid, identity_unavailable='driver_exited_before_proc_snapshot')
    launch = dict(schema='n01/wp6-028-native-launch/v1', run_id=root.name,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_started_monotonic=started,
        config_sha256=args.config_sha256, supervisor=supervisor, child=identity,
        log=str(log_path), lock=str(root.parent/'.n01_scientific_dispatch.lock'),
        lock_inherited_by_driver=True, automatic_restart=False,
        limits=dict(CPU_threads=1, driver_AS_bytes=8*1024**3, RSS_bytes=4*1024**3,
            wall_seconds=600, driver_CPU_soft_seconds=590, driver_CPU_hard_seconds=600,
            output_bytes=64*1024**2))
    with (directory/'LAUNCH.json').open('x', encoding='utf-8') as stream:
        json.dump(launch, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status='driver_spawned', supervisor_pid=os.getpid(), child_pid=child.pid)), flush=True)
    reason = None
    try:
        returncode = child.wait(timeout=max(0, 600-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        reason = 'wall_timeout_600_seconds'
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            reason += '; owned_group_exited_before_TERM'
        try:
            returncode = child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                reason += '; owned_group_exited_before_KILL'
            returncode = child.wait()
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
elapsed = time.monotonic()-started
output_bytes = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
violations = []
if usage.ru_maxrss*1024 > 4*1024**3:
    violations.append('RSS_cap_exceeded')
if output_bytes > 64*1024**2:
    violations.append('output_cap_exceeded')
if elapsed > 600:
    violations.append('wall_cap_exceeded')
receipt = dict(schema='n01/wp6-028-native-exit/v1', run_id=root.name,
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), config_sha256=args.config_sha256,
    supervisor=supervisor, child=identity, child_native_returncode=returncode,
    child_terminated_by_signal=(-returncode if returncode < 0 else None), reason=reason,
    elapsed_seconds=elapsed, peak_RSS_KiB=usage.ru_maxrss, output_bytes=output_bytes,
    child_user_CPU_seconds=usage.ru_utime, child_system_CPU_seconds=usage.ru_stime,
    resource_violations=violations, limits=launch['limits'], log=str(log_path),
    automatic_restart=False, scientific_acceptance=False)
with (directory/'NATIVE_EXIT.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
failed = returncode != 0 or reason is not None or bool(violations)
if failed:
    with (directory/'FAILED.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
lock.close()
print(json.dumps(dict(status='FAILED' if failed else 'native_zero', child_native_returncode=returncode,
    receipt=str(directory/'NATIVE_EXIT.json'))), flush=True)
raise SystemExit((returncode if returncode >= 0 else 128-returncode) if returncode else int(failed))
