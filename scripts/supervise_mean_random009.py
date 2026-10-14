"""Run one bounded frozen diagnostic and preserve its native process exit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temp.replace(path)


def tree(pid):
    rows = subprocess.check_output(['ps', '-eo', 'pid,ppid,rss'], text=True, timeout=10).splitlines()[1:]
    entries = [tuple(map(int, row.split())) for row in rows if len(row.split()) == 3]
    owned = {pid}
    while True:
        added = {p for p, parent, _ in entries if parent in owned}
        if added <= owned:
            break
        owned |= added
    return owned, sum(rss * 1024 for p, _, rss in entries if p in owned)


def gpu_bytes(pids):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    used = 0
    for row in rows.splitlines():
        pid, amount = row.split(',')
        if int(pid.strip()) in pids:
            used += int(amount.strip()) * 1024 ** 2
    return used


def stop(child, grace):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--lock-fd', required=True, type=int)
    args = parser.parse_args()
    os.fstat(args.lock_fd)
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy_path, bindings_path, job = (Path(v).resolve() for v in (args.policy, args.bindings, args.job_dir))
    policy = json.loads(policy_path.read_text(encoding='utf8'))
    limits = policy['resource_limits']
    root = Path(policy['remote_root'])
    if os.name != 'posix' or not job.is_relative_to(root) or job.name != 'primary' or job.exists():
        raise ValueError('Only a fresh owned Linux primary job is permitted')
    job.mkdir(parents=True)
    supervisor = job / 'supervisor'
    supervisor.mkdir()
    output = job / 'artifacts'
    command = [sys.executable, '-B', '-m', 'n01.diagnose_mean_random009', '--policy', str(policy_path),
               '--bindings', str(bindings_path), '--output', str(output), '--device', 'cuda:0']
    started = now()
    start_clock = time.monotonic()
    interrupted = {'signal': None}
    def received(number, _frame):
        interrupted['signal'] = number
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, received)
    child = None
    reason = None
    error = None
    peak_rss = peak_gpu = peak_output = 0
    try:
        with (job / 'console.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=root, start_new_session=True)
            write(supervisor / 'launch.json', dict(supervisor_pid=os.getpid(), child_pid=child.pid,
                  command=command, started_UTC=started, resource_limits=limits))
            while child.poll() is None:
                pids, rss = tree(child.pid)
                gpu = gpu_bytes(pids)
                size = sum(p.stat().st_size for p in job.rglob('*') if p.is_file() and not p.is_symlink())
                elapsed = time.monotonic() - start_clock
                peak_rss, peak_gpu, peak_output = max(peak_rss, rss), max(peak_gpu, gpu), max(peak_output, size)
                for failed, label in ((rss > limits['job_tree_RSS_GiB'] * 1024 ** 3, 'RSS_limit'),
                                     (gpu > limits['GPU_GiB'] * 1024 ** 3, 'GPU_limit'),
                                     (size > limits['output_MiB'] * 1024 ** 2, 'output_limit'),
                                     (elapsed > limits['wall_seconds'], 'wall_limit'),
                                     (interrupted['signal'] is not None, 'user_signal')):
                    if failed:
                        reason = label
                        break
                write(supervisor / 'resource_status.json', dict(UTC=now(), supervisor_pid=os.getpid(),
                      child_pid=child.pid, child_tree=sorted(pids), RSS_bytes=rss, GPU_bytes=gpu,
                      output_bytes=size, elapsed_seconds=elapsed, peak_RSS_bytes=peak_rss,
                      peak_GPU_bytes=peak_gpu, peak_output_bytes=peak_output, resource_limits=limits))
                if reason:
                    stop(child, limits['TERM_grace_seconds'])
                    break
                time.sleep(limits['poll_seconds'])
            code = child.wait()
    except BaseException as exc:
        reason = reason or 'supervisor_error'
        error = repr(exc)
        if child is not None:
            stop(child, limits['TERM_grace_seconds'])
        code = 1 if child is None or child.returncode is None else child.returncode
    receipt = dict(started_UTC=started, finished_UTC=now(), supervisor_pid=os.getpid(),
                   child_pid=None if child is None else child.pid, command=command,
                   child_native_exit_code=code, exit_code=0 if code == 0 and reason is None else 1,
                   reason=reason, error=error, elapsed_seconds=time.monotonic() - start_clock,
                   peak_RSS_bytes=peak_rss, peak_GPU_bytes=peak_gpu, peak_output_bytes=peak_output,
                   optimizer_updates=0)
    write(supervisor / 'exit_receipt.json', receipt)
    os.close(args.lock_fd)
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
