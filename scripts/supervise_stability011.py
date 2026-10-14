"""Bound one CPU-only saved-array bootstrap and preserve native exit/progress."""
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


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def validate_limits(policy):
    zeros = ('checkpoint_loads', 'model_instances', 'model_forward_rows', 'gradient_evaluations',
             'G_updates', 'D_updates', 'noise_updates', 'population_updates', 'C2ST_updates')
    if (policy.get('schema') != 'n01/stability011-policy/v1' or policy.get('run_id') != 'stability011_20261014'
            or policy.get('bootstrap_replicates') != 500 or any(policy.get(k) != 0 for k in zeros)
            or policy.get('DEV_numeric_read') is not False or policy.get('automatic_restart') is not False):
        raise ValueError('Wrong saved-array-only 011 budget/identity')
    limits = policy['resource_limits']
    expected = dict(CPU_threads=12, GPU_computation=False, job_tree_RSS_GiB=8, wall_seconds=7200,
                    output_MiB=256, poll_seconds=10, TERM_grace_seconds=20)
    if limits != expected:
        raise ValueError('011 explicit resource limits differ')
    return limits


def process_tree(pid):
    rows = subprocess.check_output(['ps', '-eo', 'pid,ppid,rss'], text=True, timeout=10).splitlines()[1:]
    entries = [tuple(map(int, line.split())) for line in rows if len(line.split()) == 3]
    owned = {pid}
    while True:
        children = {p for p, parent, _ in entries if parent in owned}
        if children <= owned:
            break
        owned |= children
    return owned, sum(rss*1024 for p, _, rss in entries if p in owned)


def gpu_bytes(pids):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    total = 0
    for row in rows.splitlines():
        pid, amount = row.split(',')
        if int(pid.strip()) in pids:
            total += int(amount.strip())*1024**2
    return total


def output_bytes(root):
    total = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders+files:
            p = Path(directory)/name
            if p.is_symlink():
                raise ValueError('Redirected path appeared in owned job output')
        for name in files:
            try:
                total += (Path(directory)/name).stat().st_size
            except FileNotFoundError:  # worker atomically renamed a temporary file
                pass
    return total


def limit_reason(limits, *, rss, gpu, size, elapsed, interrupted=None):
    for failed, name in ((rss > limits['job_tree_RSS_GiB']*1024**3, 'RSS_limit'),
                         (gpu != 0, 'unexpected_GPU_computation'),
                         (size > limits['output_MiB']*1024**2, 'output_limit'),
                         (elapsed > limits['wall_seconds'], 'wall_limit'),
                         (interrupted is not None, 'user_signal')):
        if failed:
            return name
    return None


def stop_owned(child, grace):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def last_progress(output):
    journals = {}
    for name in ('bootstrap_attempts.jsonl', 'bootstrap_computed.jsonl', 'bootstrap.jsonl', 'bootstrap_worker_errors.jsonl'):
        p = Path(output)/name
        ids, truncated = set(), False
        if p.exists():
            if p.is_symlink() or p.stat().st_size > 256*1024**2:
                raise ValueError('Invalid durable child journal path/size')
            with p.open('r', encoding='utf8') as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        truncated = True; break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        truncated = True; break
                    index = row.get('replicate_index')
                    if type(index) is not int or not 0 <= index < 500:
                        raise ValueError('Invalid durable replicate ID')
                    ids.add(index)
        journals[name] = dict(ids=sorted(ids), recorded_count_lower_bound=len(ids), truncated_tail=truncated)
    entered = set(journals['bootstrap_attempts.jsonl']['ids'])
    computed = set(journals['bootstrap_computed.jsonl']['ids'])
    saved = set(journals['bootstrap.jsonl']['ids'])
    durable = dict(durable_journals=journals, entered_replicates_lower_bound=len(entered),
                   computed_replicates_lower_bound=len(computed), saved_replicates_lower_bound=len(saved),
                   entered_not_computed_ids=sorted(entered-computed), computed_not_saved_ids=sorted(computed-saved),
                   unjournaled_or_in_flight_actual_completion='unknown; never inferred complete')
    p = Path(output)/'status.json'
    if not p.exists():
        return dict(status_available=False, completed_replicates=None, in_flight_actual_completion='unknown', **durable)
    if p.is_symlink() or p.stat().st_size > 16*1024**2:
        raise ValueError('Invalid child status path/size')
    d = json.loads(p.read_text(encoding='utf8'))
    return dict(status_available=True, last_persisted_status=d,
                completed_replicates=d.get('bootstrap_replicates_completed'),
                attempted_replicates_persisted_lower_bound=d.get('bootstrap_replicates_attempted'),
                actual_unpersisted_attempts='unknown outside durable child receipts',
                in_flight_actual_completion='only persisted completed count is asserted', **durable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--lock-fd', required=True, type=int)
    args = parser.parse_args()
    if os.name != 'posix':
        raise ValueError('Native011 supervisor is Linux-only')
    import fcntl
    os.fstat(args.lock_fd)
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pp, bp, job = [Path(v).resolve() for v in (args.policy, args.bindings, args.job_dir)]
    policy = json.loads(pp.read_text(encoding='utf8'))
    limits = validate_limits(policy)
    root = Path(policy['remote_root']).resolve()
    if (str(root) != '/home/rylan/projects/ej276d-n01/stability011_20261014'
            or pp != root/'stability011_policy.json' or bp != root/'STABILITY011_BINDINGS.json'
            or job != root/'runs/primary' or os.path.lexists(job)):
        raise ValueError('Only a fresh registered011 primary job is permitted')
    job.mkdir(parents=True, exist_ok=False)
    supervisor = job/'supervisor'; supervisor.mkdir()
    output = job/'artifacts'
    command = [sys.executable, '-B', str(root/'scripts/run_stability011.py'), '--policy', str(pp),
               '--bindings', str(bp), '--output', str(output)]
    started, clock = now(), time.monotonic()
    interrupted = dict(signal=None)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda n, frame: interrupted.update(signal=n))
    child, reason, error, code = None, None, None, 1
    peak_rss = peak_gpu = peak_size = 0
    try:
        with (job/'console.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(supervisor/'launch.json', dict(supervisor_pid=os.getpid(), child_pid=child.pid,
                command=command, started_UTC=started, resource_limits=limits, bootstrap_budget=500,
                GPU_computation_allowed=False, automatic_restart=False))
            while child.poll() is None:
                pids, rss = process_tree(os.getpid())
                gpu, size, elapsed = gpu_bytes(pids), output_bytes(job), time.monotonic()-clock
                peak_rss, peak_gpu, peak_size = max(peak_rss,rss), max(peak_gpu,gpu), max(peak_size,size)
                reason = limit_reason(limits, rss=rss, gpu=gpu, size=size, elapsed=elapsed,
                                      interrupted=interrupted['signal'])
                atomic_json(supervisor/'resource_status.json', dict(UTC=now(), supervisor_pid=os.getpid(),
                    child_pid=child.pid, owned_process_tree=sorted(pids), RSS_bytes=rss, GPU_bytes=gpu, output_bytes=size,
                    elapsed_seconds=elapsed, peak_sampled_RSS_bytes=peak_rss, peak_sampled_GPU_bytes=peak_gpu,
                    peak_output_bytes=peak_size, resource_limits=limits, **last_progress(output)))
                if reason:
                    stop_owned(child, limits['TERM_grace_seconds']); break
                time.sleep(limits['poll_seconds'])
            code = child.wait()
            size, elapsed = output_bytes(job), time.monotonic()-clock
            peak_size = max(peak_size, size)
            reason = reason or limit_reason(limits, rss=0, gpu=0, size=size, elapsed=elapsed,
                                             interrupted=interrupted['signal'])
    except BaseException as exc:
        reason, error = reason or 'supervisor_error', repr(exc)
        if child is not None:
            stop_owned(child, limits['TERM_grace_seconds'])
            code = child.returncode if child.returncode is not None else 1
    try:
        progress = last_progress(output)
    except BaseException as exc:
        progress = dict(status_available=False, status_read_error=repr(exc), completed_replicates=None)
    receipt = dict(schema='n01/stability011-supervisor-exit/v1', run_id=policy['run_id'], started_UTC=started,
        finished_UTC=now(), supervisor_pid=os.getpid(), child_pid=None if child is None else child.pid, command=command,
        child_native_exit_code=None if child is None else code, exit_code=0 if code == 0 and reason is None else 1, reason=reason, error=error,
        elapsed_seconds=time.monotonic()-clock, peak_sampled_RSS_bytes=peak_rss, peak_sampled_GPU_bytes=peak_gpu,
        peak_output_bytes=peak_size, registered_bootstrap_budget=500, automatic_restart=False, **progress)
    atomic_json(supervisor/'exit_receipt.json', receipt)
    os.close(args.lock_fd)
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
