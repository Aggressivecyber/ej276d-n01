"""Bound one owned Linux evaluation process tree and preserve its native exit."""
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

from supervise import rss_tree
from window008_resume01_diagnostics_common import ROOT, evaluation_binding, paths_for, read, require, write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def update(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def output_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())


def owned_gpu_bytes(pids):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    values = []
    for row in output.splitlines():
        pid, memory = row.split(',')
        if int(pid.strip()) in pids:
            values.append(int(memory.strip()) * 1024 ** 2)
    return sum(values)


def stop_reason(rss, elapsed, size, gpu, limits):
    if rss > limits['job_tree_RSS_GiB'] * 1024 ** 3:
        return 'owned_child_RSS_limit_exceeded'
    if elapsed > limits['wall_seconds']:
        return 'owned_job_wall_limit_exceeded'
    if size > limits['output_MiB'] * 1024 ** 2:
        return 'owned_job_output_limit_exceeded'
    if gpu > limits['GPU_GiB'] * 1024 ** 3:
        return 'owned_child_GPU_memory_limit_exceeded'
    return None


def owned_group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_owned(child, grace):
    """The owned group may outlive its leader; stop remaining descendants too."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while owned_group_alive(child.pid):
        child.poll()  # Reap an exited leader while its group is being checked.
        if time.monotonic() >= deadline:
            break
        time.sleep(.1)
    if owned_group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def supervise(kind):
    cfg, training, policy, manifest = evaluation_binding(ROOT)
    require(os.name == 'posix' and str(ROOT) == read(ROOT / 'window008_resume01_config.json')['remote_root'], 'Supervisor must run on the registered Linux workspace')
    paths = paths_for(kind)
    require((paths['job'] / 'START_ATTEMPT.json').is_file(), 'Missing permanent dispatch claim')
    out = paths['supervisor']
    out.mkdir(exist_ok=False)
    import fcntl
    limits = policy['resource_limits']
    command = [sys.executable, '-B', str(ROOT / 'scripts/run_window008_resume01_diagnostics.py'), '--kind', kind]
    child = None
    started, clock_start = now(), time.monotonic()
    reason, monitor_error, exit_code = None, None, None
    peak_rss = peak_gpu = peak_output = 0
    interrupted = []
    def receive_signal(number, frame):
        interrupted.append(number)
    previous_handlers = {number: signal.signal(number, receive_signal) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (out / 'run.lock').open('x') as lock, (out / 'console.log').open('x', buffering=1) as log:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=ROOT, start_new_session=True)
            write_new(out / 'launch.json', dict(supervisor_pid=os.getpid(), child_pid=child.pid,
                command=command, started_utc=started, kind=kind, resource_limits=limits))
            while child.poll() is None:
                try:
                    rss, pids = rss_tree(child.pid)
                    size = output_bytes(paths['job'])
                    gpu = owned_gpu_bytes(set(pids))
                    elapsed = time.monotonic() - clock_start
                    peak_rss, peak_gpu, peak_output = max(peak_rss, rss), max(peak_gpu, gpu), max(peak_output, size)
                    update(out / 'resource_status.json', dict(supervisor_pid=os.getpid(), child_pid=child.pid,
                        rss_bytes=rss, peak_rss_bytes=peak_rss, GPU_memory_bytes=gpu, peak_GPU_memory_bytes=peak_gpu,
                        output_bytes=size, peak_output_bytes=peak_output, wall_seconds=elapsed,
                        child_tree=pids, utc=now(), resource_limits=limits))
                    reason = ('supervisor_received_signal_' + str(interrupted[0]) if interrupted else
                              stop_reason(rss, elapsed, size, gpu, limits))
                except Exception as error:
                    reason, monitor_error = 'resource_monitor_failed', repr(error)
                if reason is not None:
                    terminate_owned(child, limits['TERM_grace_seconds'])
                    break
                time.sleep(limits['poll_seconds'])
            exit_code = child.wait()
            if owned_group_alive(child.pid):
                reason = reason or 'owned_descendants_remained_after_wrapper_exit'
                terminate_owned(child, limits['TERM_grace_seconds'])
            # Catch a last write crossing the disk bound immediately before exit.
            peak_output = max(peak_output, output_bytes(paths['job']))
            if reason is None:
                reason = stop_reason(0, time.monotonic() - clock_start, peak_output, 0, limits)
    except BaseException as error:
        reason = reason or 'supervisor_failed'
        monitor_error = repr(error)
        if child is not None:
            terminate_owned(child, limits['TERM_grace_seconds'])
            exit_code = child.wait()
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        write_new(out / 'exit_receipt.json', dict(exit_code=exit_code, reason=reason,
            error=monitor_error, peak_rss_bytes=peak_rss, peak_GPU_memory_bytes=peak_gpu,
            peak_output_bytes=peak_output, started_utc=started, ended_utc=now(),
            wall_seconds=time.monotonic() - clock_start, supervisor_pid=os.getpid(),
            child_pid=None if child is None else child.pid, kind=kind, resource_limits=limits))
    return exit_code if reason is None and isinstance(exit_code, int) and 0 <= exit_code <= 255 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('primary',))
    sys.exit(supervise(parser.parse_args().kind))
