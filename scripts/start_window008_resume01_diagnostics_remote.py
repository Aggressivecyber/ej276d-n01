"""Read-only training/resource gates, then one exclusive evaluation dispatch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from window008_resume01_diagnostics_common import (
    ROOT, EVAL_MANIFEST, TRAIN_MANIFEST, completed_job, evaluation_binding,
    paths_for, read, require, sha, training_completion, write_new,
)


def command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def preflight(kind, cfg, policy, manifest):
    paths = paths_for(kind)
    require(not paths['job'].exists(), 'Preserve prior evaluation attempt: ' + str(paths['job']))
    require((ROOT / 'tmp').is_dir() and (ROOT / 'cache').is_dir(), 'Registered remote temporary/cache directories are missing')
    completion = training_completion(ROOT, cfg)
    if kind == 'replay01':
        completed_job('primary', ROOT)
    require(sha(Path(cfg['remote_training_container'])) == cfg['train_container_sha256'], 'Original raw data changed')
    apps = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
    require(not apps, 'Another GPU compute job is active')
    markers = ('-m n01.', 'scripts/supervise.py', 'scripts/supervise_window008_resume01_diagnostics.py',
               'scripts/run_window008_resume01_diagnostics.py')
    active = [line for line in command(['ps', '-eo', 'pid,args']).splitlines()
              if any(marker in line for marker in markers)]
    require(not active, 'Another project/supervised job is active')
    gpu = command(['nvidia-smi', '--id=0', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'])
    total, used = map(int, command(['nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.used',
        '--format=csv,noheader,nounits']).split(','))
    require((total - used) * 1024 ** 2 >= policy['resource_limits']['GPU_GiB'] * 1024 ** 3,
            'Insufficient free GPU0 memory')
    disk = shutil.disk_usage(ROOT)
    require(disk.free >= policy['preflight']['minimum_free_disk_GiB'] * 1024 ** 3, 'Insufficient free output disk')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    require(available >= policy['preflight']['minimum_available_RAM_GiB'] * 1024 ** 3, 'Insufficient available RAM')
    return dict(UTC=datetime.now(timezone.utc).isoformat(), status='passed', kind=kind, GPU=gpu,
        GPU_compute_processes=apps, active_project_jobs=active, disk_free_bytes=disk.free,
        RAM_available_bytes=available, diagnostics_source_manifest_sha256=sha(ROOT / EVAL_MANIFEST),
        training_source_manifest_sha256=sha(ROOT / TRAIN_MANIFEST), config_sha256=sha(ROOT / 'window008_config.json'),
        evaluation_source_files_verified=len(manifest['files']), training_completion=completion,
        raw_array_decode=False, optimizer_updates=0, resource_limits=policy['resource_limits'])


def dispatch(kind):
    cfg, training, policy, manifest = evaluation_binding(ROOT)
    require(os.name == 'posix' and str(ROOT) == read(ROOT / 'window008_resume01_config.json')['remote_root'], 'Run only on the registered Linux workspace')
    import fcntl
    with (ROOT / 'runs/.evaluation_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks = preflight(kind, cfg, policy, manifest)
        paths = paths_for(kind)
        paths['job'].mkdir(parents=True, exist_ok=False)
        write_new(paths['job'] / 'START_ATTEMPT.json', dict(UTC=checks['UTC'], kind=kind,
            dispatcher_pid=os.getpid(), diagnostics_source_manifest_sha256=checks['diagnostics_source_manifest_sha256'],
            training_source_manifest_sha256=checks['training_source_manifest_sha256'], automatic_retry_allowed=False))
        process = None
        try:
            write_new(paths['job'] / 'PREFLIGHT.json', checks)
            with (paths['job'] / 'launch.log').open('xb') as log:
                process = subprocess.Popen(['bash', str(ROOT / 'scripts/launch_window008_resume01_diagnostics.sh'), kind],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(1.)
            code = process.poll()
            require(code is None or code == 0, 'Evaluation launcher exited with native code ' + str(code))
            result = dict(UTC=datetime.now(timezone.utc).isoformat(), kind=kind,
                status='dispatched_pending_native_confirmation', supervisor_pid=process.pid, remote_root=str(ROOT),
                diagnostics_source_manifest_sha256=checks['diagnostics_source_manifest_sha256'],
                training_source_manifest_sha256=checks['training_source_manifest_sha256'],
                preflight_sha256=sha(paths['job'] / 'PREFLIGHT.json'), optimizer_updates=0,
                additional_unique_draws=0 if kind == 'replay01' else None, native_exit_observed=False,
                automatic_retry_allowed=False)
            write_new(paths['job'] / 'REMOTE_START.json', result)
            return result
        except BaseException as error:
            write_new(paths['job'] / 'LAUNCH_FAILURE.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
                status='dispatch_failed_preserved', kind=kind, error=repr(error),
                supervisor_pid=None if process is None else process.pid,
                native_exit=None if process is None else process.poll(), automatic_retry_allowed=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('primary',))
    print(json.dumps(dispatch(parser.parse_args().kind)))
