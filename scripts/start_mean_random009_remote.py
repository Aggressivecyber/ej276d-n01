"""Verify a fresh isolated runtime and dispatch one owned diagnostic process."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import shutil
import fcntl


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    policy_path = ROOT / 'mean_random009_policy.json'
    bindings_path = ROOT / 'MEAN_RANDOM009_BINDINGS.json'
    policy = json.loads(policy_path.read_text(encoding='utf8'))
    bindings = json.loads(bindings_path.read_text(encoding='utf8'))
    if os.name != 'posix' or str(ROOT) != policy['remote_root']:
        raise ValueError('Dispatch only on the registered Linux workspace')
    job = ROOT / 'runs/primary'
    if job.exists() or (ROOT / 'START_RECEIPT.json').exists():
        raise ValueError('Preserve an earlier dispatch; no automatic restart')
    manifest_path = Path(bindings['source_manifest']['path'])
    if sha(manifest_path) != bindings['source_manifest']['sha256'] or sha(policy_path) != bindings['policy_sha256']:
        raise ValueError('Policy/source freeze changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    for rel, digest in manifest['files'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Frozen source changed: ' + rel)
    for key, row in bindings['files'].items():
        if sha(row['path']) != row['sha256']:
            raise ValueError('Frozen input changed: ' + key)
        if manifest['bound_inputs'][key] != row['sha256']:
            raise ValueError('Input is outside the source freeze: ' + key)
    lock_path = ROOT.parent / '.n01_scientific_dispatch.lock'
    if lock_path.is_symlink():
        raise ValueError('Refuse a redirected scientific lock')
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True).strip()
    if apps:
        raise ValueError('Another GPU compute process is active')
    markers = ('-m n01.', 'scripts/supervise_window', 'scripts/supervise_mean_random009.py',
               'scripts/run_window', 'scripts/supervise.py')
    active = [row for row in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
              if any(marker in row for marker in markers)]
    if active:
        raise ValueError('Another project scientific/CPU job is active: ' + repr(active))
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                                  '--format=csv,noheader'], text=True).strip()
    total, used = map(int, subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.used',
                                                   '--format=csv,noheader,nounits'], text=True).split(','))
    if (total - used) * 1024 ** 2 < policy['resource_limits']['GPU_GiB'] * 1024 ** 3:
        raise ValueError('Insufficient free GPU memory')
    free = shutil.disk_usage(ROOT).free
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))) * 1024
    if free < policy['preflight']['minimum_free_disk_GiB'] * 1024 ** 3 or available < policy['preflight']['minimum_available_RAM_GiB'] * 1024 ** 3:
        raise ValueError('Insufficient free disk/RAM')
    python = Path('/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python')
    if not python.is_file():
        raise ValueError('Registered existing Python runtime is missing')
    env = os.environ.copy()
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'), TMPDIR=str(ROOT / 'tmp'),
               XDG_CACHE_HOME=str(ROOT / 'cache'), OMP_NUM_THREADS='12', MKL_NUM_THREADS='12',
               OPENBLAS_NUM_THREADS='12', CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [str(python), '-B', str(ROOT / 'scripts/supervise_mean_random009.py'), '--policy', str(policy_path),
               '--bindings', str(bindings_path), '--job-dir', str(job), '--lock-fd', str(lock.fileno())]
    with (ROOT / 'supervisor_console.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, env=env,
                                   pass_fds=(lock.fileno(),))
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), status='supervisor_dispatched_awaiting_real_first_batch',
                   supervisor_pid=process.pid, command=command, remote_root=str(ROOT), GPU=gpu,
                   GPU_compute_processes_before=apps, active_project_jobs_before=active,
                   exclusive_scientific_lock=str(lock_path), lock_inherited_by_supervisor=True,
                   disk_free_bytes=free, RAM_available_bytes=available,
                   policy_sha256=sha(policy_path), bindings_sha256=sha(bindings_path),
                   source_manifest_sha256=sha(manifest_path), raw_array_decode=False, optimizer_updates=0)
    with (ROOT / 'START_RECEIPT.json').open('x', encoding='utf8') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
