"""One038 supplement command: existing RSS supervisor plus phase output bound."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path('/home/rylan/projects/ej276d-n01/response038_resume02_20261018')
python = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
config = root/'configs/response038_c2st_completion_20261018.json'
config_sha = 'f4a741361d54f852c62e30688cb80548e593c2f38ae86bc11902d245f750b3fa'
script = root/'scripts/complete_response038_c2st.py'
script_sha = 'e3a26d84f4d5a4c4bf7f1dbde7105ebf802832b2b999e83df090085a03f8750e'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=('probe', 'classify'), required=True)
parser.add_argument('--engineering-check-sha256')
args = parser.parse_args()
if (hashlib.sha256(config.read_bytes()).hexdigest() != config_sha or
        hashlib.sha256(script.read_bytes()).hexdigest() != script_sha):
    raise ValueError('Frozen038 supplement source/config differs')
p = json.loads(config.read_text())
if hashlib.sha256((root/'scripts/supervise.py').read_bytes()).hexdigest() != p['source_pins']['scripts/supervise.py']:
    raise ValueError('Original RSS supervisor differs')
if Path.cwd() != root or set(os.sched_getaffinity(0)) != {0, 1, 2, 3}:
    raise ValueError('038 deployment cwd/CPU affinity differs')
sys.path.insert(0, str(root/'scripts'))
from supervise import write

phase = 'probe' if args.phase == 'probe' else 'classifiers'
out = root/p['output_layout'][phase]
control = root/p['output_layout']['probe_control' if phase == 'probe' else 'classifier_control']
budget = p['engineering_probe']['budget_max'] if phase == 'probe' else p['classifier_completion']['resources']
wall = budget['wall_seconds'] if phase == 'probe' else budget['hard_wall_seconds']
if out.exists() or control.exists():
    raise FileExistsError('038 phase output/control must be fresh')
native = [python, '-B', str(script), '--phase', args.phase, '--config', str(config),
    '--config-sha256', config_sha, '--out', str(out), '--device', 'cuda:0']
if phase == 'classifiers':
    if args.engineering_check_sha256 is None:
        raise ValueError('Classifier command requires bound strict-PASS SHA')
    native += ['--engineering-check', str(root/p['output_layout']['probe']/'ENGINEERING_RECONSTRUCTION_CHECK.json'),
        '--engineering-check-sha256', args.engineering_check_sha256]
timeout_command = ['timeout', '--signal=TERM', '--kill-after=20s', str(wall)+'s']+native
command = [python, '-B', str(root/'scripts/supervise.py'), '--output', str(control),
    '--rss-limit-gib', str(budget['RSS_GiB']), '--']+timeout_command
control.mkdir(parents=True)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
identity = None
reason = None
peak_output = 0
started = time.monotonic()
with (control/'dispatcher.log').open('xb') as log:
    supervisor = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{supervisor.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    write(control/'OUTPUT_LIMIT_LAUNCH.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        boot_id=boot, output_guard_pid=os.getpid(), supervisor_pid=supervisor.pid,
        supervisor_start_ticks=ticks, cwd=str(root), command=command, output_limit_MiB=budget['output_MiB'],
        GPU_enforcement='Frozen scientific script torch allocator cap only; no NVML kill threshold',
        automatic_restart=False))
    while supervisor.poll() is None:
        launch = control/'launch.json'
        if identity is None and launch.exists():
            timeout_pid = json.loads(launch.read_text())['child_pid']
            proc = Path('/proc')/str(timeout_pid)
            if proc.exists():
                identity = dict(pid=timeout_pid, start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
                    boot_id=boot, cwd=os.readlink(proc/'cwd'), command=(proc/'cmdline').read_bytes().split(b'\0')[:-1])
                if identity['cwd'] != str(root) or identity['command'] != [v.encode() for v in timeout_command]:
                    raise ValueError('038 timeout process ownership differs')
                write(control/'TIMEOUT_IDENTITY.json', dict(identity,
                    command=[v.decode() for v in identity['command']]))
        size = 0
        for directory in (out, control):
            for path in directory.rglob('*'):
                if path.is_file():
                    try:
                        size += path.stat().st_size
                    except FileNotFoundError:  # Existing supervisor atomically renames status files.
                        pass
        peak_output = max(peak_output, size)
        write(control/'output_limit_status.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
            output_bytes=size, peak_output_bytes=peak_output, output_limit_MiB=budget['output_MiB'],
            timeout_pid=None if identity is None else identity['pid'], elapsed_seconds=time.monotonic()-started))
        if size > budget['output_MiB']*1024**2:
            reason = 'owned_phase_output_limit_exceeded'
            if identity is None:
                raise RuntimeError('Output limit exceeded before timeout ownership was recorded')
            for number in (signal.SIGTERM, signal.SIGKILL):
                proc = Path('/proc')/str(identity['pid'])
                if proc.exists():
                    current = dict(pid=identity['pid'], start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        cwd=os.readlink(proc/'cwd'), command=(proc/'cmdline').read_bytes().split(b'\0')[:-1])
                    if current != identity:
                        raise ValueError('038 timeout ownership changed; no signal sent')
                    os.killpg(identity['pid'], number)
                try:
                    supervisor.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    if number == signal.SIGKILL:
                        raise
            break
        time.sleep(1)
    code = supervisor.wait()
size = sum(path.stat().st_size for directory in (out, control) for path in directory.rglob('*') if path.is_file())
peak_output = max(peak_output, size)
if size > budget['output_MiB']*1024**2:
    reason = 'owned_phase_output_limit_exceeded'
write(control/'OUTPUT_LIMIT_EXIT.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
    supervisor_exit_code=code, output_guard_reason=reason, peak_output_bytes=peak_output,
    output_limit_MiB=budget['output_MiB'], elapsed_seconds=time.monotonic()-started,
    original_failed_run_preserved=True, automatic_restart=False))
raise SystemExit(code if reason is None else 1)
