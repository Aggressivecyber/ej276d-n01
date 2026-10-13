"""Read-only bindings and exclusive receipts for the separate Window008 eval jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'window008_20261013'
TRAIN_MANIFEST = 'WINDOW008_SOURCE_MANIFEST.json'
EVAL_MANIFEST = 'WINDOW008_EVALUATION_SOURCE_MANIFEST.json'
POLICY = 'window008_evaluation_policy.json'
SOURCE_FILES = (
    'src/n01/evaluate_window008.py', POLICY,
    'scripts/window008_evaluation_common.py',
    'scripts/freeze_window008_evaluation.py',
    'scripts/stage_window008_evaluation.py',
    'scripts/start_window008_evaluation_remote.py',
    'scripts/launch_window008_evaluation.sh',
    'scripts/run_window008_evaluation.py',
    'scripts/supervise_window008_evaluation.py',
    'scripts/verify_window008_evaluation.py',
    'scripts/check_window008_evaluation_helpers.py',
    'docs/WINDOW008_EVALUATION.md',
    'docs/WINDOW008_EVALUATION_OPERATIONS.md',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def bound(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute(), 'Expected a relative bound path: ' + str(relative))
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes its bound root: ' + str(relative))
    return path


def allow_output(path):
    path = Path(path).resolve()
    if os.name == 'nt':
        require(path.drive.upper() in ('E:', 'D:'), 'Windows outputs must stay on E/D')
    return path


def verify_files(root, files):
    require(isinstance(files, dict) and bool(files), 'Empty source binding')
    for relative, digest in files.items():
        require(sha(bound(root, relative)) == digest, 'Bound bytes changed: ' + relative)


def training_binding(root=ROOT):
    root = Path(root).resolve()
    manifest = read(root / TRAIN_MANIFEST)
    cfg = read(root / 'window008_config.json')
    require(manifest['schema'] == 'n01/window008/source_manifest/v1'
            and manifest['run_id'] == cfg['run_id'] == RUN_ID, 'Wrong training version/run')
    require(manifest['before_scientific_model_forward'] is True, 'Training freeze provenance is missing')
    require(sha(root / 'window008_config.json') == manifest['config_sha256'], 'Training config changed')
    verify_files(root, manifest['files'])
    return cfg, manifest


def policy_binding(root=ROOT):
    root = Path(root).resolve()
    cfg, training = training_binding(root)
    policy = read(root / POLICY)
    require(policy['schema'] == 'n01/window008-evaluation-policy/v1' and policy['run_id'] == RUN_ID, 'Wrong evaluation policy')
    require(policy['training_source_manifest_sha256'] == sha(root / TRAIN_MANIFEST), 'Evaluation policy targets another training freeze')
    require(policy['training_run'] == 'runs/science' and policy['job_root'] == 'runs/evaluation_jobs'
            and policy['jobs'] == ['primary', 'replay01'], 'Evaluation layout differs')
    require(policy['evaluation_module'] == 'n01.evaluate_window008', 'Wrong evaluator module')
    require(policy['optimizer_updates'] == policy['C2ST_updates'] == 0, 'Evaluation may not optimize')
    limits = policy['resource_limits']
    for name, key in (('GPU_GiB', 'gpu_memory_limit_GiB'), ('job_tree_RSS_GiB', 'job_tree_RSS_limit_GiB'),
                      ('wall_seconds', 'wall_limit_s'), ('output_MiB', 'output_limit_MiB')):
        require(limits[name] == cfg[key], 'Evaluation resource bound differs: ' + name)
    require(limits['poll_seconds'] == 1 and limits['TERM_grace_seconds'] == 20, 'Supervision cadence differs')
    replay = policy['replay']
    require(replay['draws_per_parent'] == cfg['evaluation']['draws_per_parent'] == 4
            and replay['chunk_rows'] == cfg['evaluation']['chunk_rows'] == 128
            and replay['additional_unique_draws'] == 0 and replay['overwrite_allowed'] is False,
            'Replay must repeat the registered stream and chunking')
    require(all(replay[key] is True for key in ('same_training_endpoints', 'same_training_config_and_seeds',
            'compare_all_NPZ_members', 'compare_all_chunk_records', 'compare_distances_and_parent_bootstrap'))
            and replay['array_comparison'] == 'exact_dtype_shape_values_and_member_sha256', 'Replay checks may not be relaxed')
    return cfg, training, policy


def evaluation_binding(root=ROOT):
    root = Path(root).resolve()
    cfg, training, policy = policy_binding(root)
    manifest = read(root / EVAL_MANIFEST)
    require(manifest['schema'] == 'n01/window008-evaluation-source/v1' and manifest['run_id'] == RUN_ID, 'Wrong evaluation source manifest')
    require(manifest['training_source_manifest_sha256'] == sha(root / TRAIN_MANIFEST)
            and manifest['training_config_sha256'] == sha(root / 'window008_config.json')
            and manifest['policy_sha256'] == sha(root / POLICY), 'Evaluation version binding differs')
    require(manifest['before_evaluation_numeric_read'] is True and manifest['optimizer_updates'] == 0, 'Evaluation freeze scope differs')
    require(set(SOURCE_FILES) <= set(manifest['files']), 'Evaluation source manifest is incomplete')
    protected = set(training['files']) | {TRAIN_MANIFEST}
    require(not (set(manifest['files']) & protected), 'Evaluation payload would modify frozen training files')
    verify_files(root, manifest['files'])
    actual_modules = {path.relative_to(root).as_posix() for path in (root / 'src/n01').glob('*.py')}
    expected_modules = {name for name in set(training['files']) | set(manifest['files'])
                        if name.startswith('src/n01/') and name.endswith('.py')}
    require(actual_modules == expected_modules == set(manifest['module_inventory']), 'Evaluator module inventory changed')
    return cfg, training, policy, manifest


def paths_for(kind, root=ROOT):
    require(kind in ('primary', 'replay01'), 'Choose primary or replay01')
    root = Path(root).resolve()
    job = root / 'runs/evaluation_jobs' / kind
    return dict(root=root, run=root / 'runs/science', job=job, artifacts=job / 'artifacts',
                supervisor=job / 'supervisor', primary=root / 'runs/evaluation_jobs/primary')


def training_completion(root=ROOT, cfg=None):
    root = Path(root).resolve()
    if cfg is None:
        cfg, _ = training_binding(root)
    sys.path.insert(0, str(root / 'src'))
    from n01.evaluate_window008 import training_completion_gate
    return training_completion_gate(root / 'runs/science', cfg)


def artifact_integrity(destination):
    destination = Path(destination).resolve()
    require(not (destination / 'FAILED.json').exists(), 'Evaluation has a failure artifact')
    summary = read(destination / 'SUMMARY.json')
    require(summary['schema'] == 'n01/window008-evaluation/v1'
            and summary['status'] == 'fixed_endpoint_evaluation_complete', 'Evaluation is incomplete')
    require(summary['fixed_training_G_updates'] == 57000 and summary['fixed_training_D_updates'] == 144000, 'Wrong endpoints')
    for key in ('evaluation_G_updates', 'evaluation_noise_updates', 'population_updates', 'C2ST_updates'):
        require(summary[key] == 0, 'Evaluation optimized: ' + key)
    require(summary['DEV_numeric_decoded'] is True and summary['frozen_inputs_rechecked'] is True
            and summary['all_original_rows_preserved'] is True, 'Incomplete data/freeze receipt')
    expected = set(summary['artifact_sha256']) | {'SUMMARY.json'}
    actual = {path.name for path in destination.iterdir()}
    require(actual == expected, 'Evaluation artifact membership differs')
    require(all(path.is_file() and not path.is_symlink() for path in destination.iterdir()),
            'Evaluation artifacts must be regular files')
    verify_files(destination, summary['artifact_sha256'])
    return summary


def completed_job(kind, root=ROOT):
    """Requires both the core evaluator and its owned supervisor to exit cleanly."""
    cfg, training, policy, manifest = evaluation_binding(root)
    paths = paths_for(kind, root)
    job = paths['job']
    require(not (job / 'JOB_FAILED.json').exists() and not (job / 'LAUNCH_FAILURE.json').exists(),
            'Evaluation job has a preserved failure')
    native = read(paths['supervisor'] / 'exit_receipt.json')
    launch = read(paths['supervisor'] / 'launch.json')
    require(native['exit_code'] == 0 and native.get('reason') is None, 'Evaluation supervisor did not exit cleanly')
    require(native.get('supervisor_pid') == launch.get('supervisor_pid')
            and native.get('child_pid') == launch.get('child_pid')
            and isinstance(native.get('supervisor_pid'), int) and isinstance(native.get('child_pid'), int),
            'Evaluation native receipt/launch mismatch')
    command = launch.get('command', [])
    require(len(command) == 5 and command[1:] == ['-B', str(Path(root).resolve() / 'scripts/run_window008_evaluation.py'),
            '--kind', kind], 'Wrong evaluation native command')
    core = read(job / 'EVALUATOR_NATIVE.json')
    complete = read(job / 'JOB_COMPLETE.json')
    expected_core = [command[0], '-B', '-m', 'n01.evaluate_window008', '--out', str(paths['run']),
        '--config', str(Path(root).resolve() / 'window008_config.json'),
        '--dataset', str(Path(root).resolve() / cfg['dataset']),
        '--statistics', str(Path(root).resolve() / cfg['statistics']),
        '--raw-container', cfg['remote_training_container'], '--identities', str(Path(root).resolve() / 'data/PARENT_IDENTITIES.npz'),
        '--device', 'cuda:0', '--destination', str(paths['artifacts'])]
    require(core['exit_code'] == 0 and core['module'] == 'n01.evaluate_window008'
            and core['kind'] == kind and core['command'] == expected_core
            and core['wrapper_pid'] == native['child_pid'] and core['supervisor_pid'] == native['supervisor_pid']
            and core['evaluation_source_manifest_sha256'] == sha(Path(root) / EVAL_MANIFEST),
            'Core evaluator native command/freeze/PID differs')
    require(complete['status'] == 'evaluation_job_complete' and complete['kind'] == kind
            and complete['evaluation_source_manifest_sha256'] == sha(Path(root) / EVAL_MANIFEST)
            and complete['training_source_manifest_sha256'] == sha(Path(root) / TRAIN_MANIFEST)
            and complete['optimizer_updates'] == 0, 'Evaluation job receipt differs')
    require(complete['evaluator_native_sha256'] == sha(job / 'EVALUATOR_NATIVE.json'), 'Evaluator exit receipt changed')
    summary = artifact_integrity(paths['artifacts'])
    require(complete['summary_sha256'] == sha(paths['artifacts'] / 'SUMMARY.json'), 'Evaluation summary changed')
    if kind == 'replay01':
        verification = read(job / 'REPLAY_VERIFICATION.json')
        require(verification['status'] == 'exact_replay_verified'
                and complete['replay_verification_sha256'] == sha(job / 'REPLAY_VERIFICATION.json'), 'Replay verification missing/changed')
    return dict(paths=paths, summary=summary, complete=complete, native=native, core=core)
