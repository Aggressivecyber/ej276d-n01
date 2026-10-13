"""Owned wrapper for a single frozen endpoint evaluation or identical replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
import time

from window008_resume01_evaluation_common import (
    ROOT, EVAL_MANIFEST, TRAIN_MANIFEST, POLICY, artifact_integrity, completed_job,
    evaluation_binding, paths_for, read, require, sha, training_completion, write_new,
)


def run(kind):
    cfg, training, policy, manifest = evaluation_binding(ROOT)
    require(os.name == 'posix' and str(ROOT) == read(ROOT / 'window008_resume01_config.json')['remote_root'], 'Run only on the bound Linux workspace')
    paths = paths_for(kind)
    job = paths['job']
    for name in ('JOB_COMPLETE.json', 'JOB_FAILED.json', 'EVALUATOR_NATIVE.json', 'artifacts', 'WRAPPER_ATTEMPT.json'):
        require(not (job / name).exists(), 'Preserve existing evaluation attempt: ' + name)
    attempt = read(job / 'START_ATTEMPT.json')
    preflight = read(job / 'PREFLIGHT.json')
    current_hash = sha(ROOT / EVAL_MANIFEST)
    require(attempt['kind'] == kind and attempt['evaluation_source_manifest_sha256'] == current_hash
            and preflight['status'] == 'passed' and preflight['evaluation_source_manifest_sha256'] == current_hash,
            'Dispatch preflight/freeze differs')
    deadline = time.monotonic() + 5
    while not (paths['supervisor'] / 'launch.json').exists() and time.monotonic() < deadline:
        time.sleep(.05)
    launch = read(paths['supervisor'] / 'launch.json')
    require(launch['supervisor_pid'] == os.getppid() and launch['child_pid'] == os.getpid()
            and launch['kind'] == kind, 'Wrapper is not the owned supervisor child')
    write_new(job / 'WRAPPER_ATTEMPT.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        kind=kind, wrapper_pid=os.getpid(), supervisor_pid=os.getppid(),
        evaluation_source_manifest_sha256=current_hash, optimizer_updates=0, automatic_retry_allowed=False))
    try:
        before = training_completion(ROOT, cfg)
        reference = completed_job('primary', ROOT) if kind == 'replay01' else None
        command = [sys.executable, '-B', '-m', 'n01.evaluate_window008_resume01', '--out', str(paths['run']),
            '--config', str(ROOT / 'window008_config.json'), '--resume-config', str(ROOT / 'window008_resume01_config.json'), '--source-manifest', str(ROOT / EVAL_MANIFEST), '--policy', str(ROOT / POLICY), '--dataset', str(ROOT / cfg['dataset']),
            '--statistics', str(ROOT / cfg['statistics']), '--raw-container', cfg['remote_training_container'],
            '--identities', str(ROOT / 'data/PARENT_IDENTITIES.npz'), '--device', 'cuda:0',
            '--destination', str(paths['artifacts'])]
        started = datetime.now(timezone.utc).isoformat()
        native = subprocess.run(command, cwd=ROOT, check=False)
        write_new(job / 'EVALUATOR_NATIVE.json', dict(schema='n01/window008-resume01-evaluation-native/v1',
            module='n01.evaluate_window008_resume01', kind=kind, exit_code=native.returncode, command=command,
            started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
            wrapper_pid=os.getpid(), supervisor_pid=os.getppid(), evaluation_source_manifest_sha256=current_hash))
        require(native.returncode == 0, 'Core evaluator failed with native exit ' + str(native.returncode))
        artifact_integrity(paths['artifacts'])
        evaluation_binding(ROOT)
        require(training_completion(ROOT, cfg) == before, 'Training inputs or endpoint receipts changed during evaluation')
        verification_hash = None
        if reference is not None:
            # Recheck the completed primary immediately before comparison.
            reference = completed_job('primary', ROOT)
            from verify_window008_resume01_evaluation import compare_artifacts
            report = compare_artifacts(reference['paths']['artifacts'], paths['artifacts'])
            write_new(job / 'REPLAY_VERIFICATION.json', report)
            require(report['status'] == 'exact_replay_verified', 'Exact replay mismatch; outputs/report preserved')
            verification_hash = sha(job / 'REPLAY_VERIFICATION.json')
        write_new(job / 'JOB_COMPLETE.json', dict(schema='n01/window008-resume01-evaluation-job/v1',
            UTC=datetime.now(timezone.utc).isoformat(), status='evaluation_job_complete', kind=kind,
            evaluation_source_manifest_sha256=current_hash, training_source_manifest_sha256=sha(ROOT / TRAIN_MANIFEST),
            summary_sha256=sha(paths['artifacts'] / 'SUMMARY.json'), evaluator_native_sha256=sha(job / 'EVALUATOR_NATIVE.json'),
            replay_verification_sha256=verification_hash, optimizer_updates=0,
            additional_unique_draws=0 if kind == 'replay01' else None,
            awaiting_supervisor_native_exit=True, scientific_status='response_match_not_established'))
    except BaseException as error:
        write_new(job / 'JOB_FAILED.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
            status='evaluation_job_failed_preserved', kind=kind, error=repr(error),
            automatic_retry_allowed=False, optimizer_updates=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('primary', 'replay01'))
    run(parser.parse_args().kind)
