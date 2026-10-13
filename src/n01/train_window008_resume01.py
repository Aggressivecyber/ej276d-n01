"""Resume only the bound, complete fix01 full6000 boundary in a fresh root.

Retain original det19k; continue full to19k; initialize simple from old common.
Failure checkpoints are forensic records and are never restoration inputs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import signal
import time
import traceback

import numpy as np
import torch

from .data import sha256, write_json
from .response008 import ARMS, HYPOTHESES, response_stage, state_digest
from .response008_fix01 import require
from .response008_resume01 import MODEL_IMPLEMENTATION, RUN_ID, Response008Resume01Engine
from .train import atomic_torch_save, utc
from .train_window008 import configure_runtime
from .train_window008_fix01 import (
    EXPECTED_REFERENCES as ORIGINAL_REFERENCES,
    argument,
    retained_endpoint,
    validate_legacy_receipts,
    validate_legacy_states,
    validate_plan as validate_fix01_plan,
    validate_training_data,
)

ROOT = Path(__file__).resolve().parents[2]
FIX01_RUN_ID = 'window008_fix01_20261013'
FULL6000_SHA256 = '921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58'
STOP_REFERENCES = {
    'fix01_full_periodic_06000', 'fix01_native_exit', 'fix01_launch',
    'fix01_binding', 'fix01_training_inputs', 'fix01_resume_validation',
    'fix01_failure', 'fix01_legacy_validation', 'fix01_user_stop_request',
    'fix01_user_stop_verification', 'fix01_user_stop_verification_local',
    'fix01_user_stop_collection', 'fix01_user_pause',
}
EXPECTED_REFERENCES = ORIGINAL_REFERENCES | STOP_REFERENCES
REQUIRED_SOURCES = {
    'src/n01/response008_resume01.py', 'src/n01/train_window008_resume01.py',
    'window008_resume01_config.json', 'docs/WINDOW008_RESUME01_EXECUTION.md',
    'scripts/check_window008_resume01_driver.py',
}
FIXED_COUNTS = dict(
    updates_per_valid_arm=19000,
    new_execution_G_updates=32000, new_execution_D_updates=87000,
    inherited_effective_G_updates=25000, inherited_effective_D_updates=57000,
    effective_comparison_G_updates=57000, effective_comparison_D_updates=144000,
    historical_actual_G_updates_before_resume=25667,
    historical_actual_D_updates_before_resume=59001,
    historical_actual_G_updates_after_success=57667,
    historical_actual_D_updates_after_success=146001,
    global_N01_G_updates_before_resume=125667,
    discarded_legacy_joint_G_updates=269, discarded_legacy_joint_D_updates=807,
    discarded_user_stop_G_updates=398, discarded_user_stop_D_updates=1194,
)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique)


def bound(root, relative):
    """Bind an immutable relative input, including cross-platform path checks."""
    root = Path(root).resolve()
    require(isinstance(relative, str) and bool(relative) and '\0' not in relative,
            'Expected a nonempty relative path')
    normalized = PurePosixPath(relative.replace('\\', '/'))
    candidate = Path(relative)
    require(not normalized.is_absolute() and not PureWindowsPath(relative).drive
            and not candidate.is_absolute() and '..' not in normalized.parts
            and bool(candidate.parts), 'Expected an in-root relative path: ' + relative)
    unresolved = root / candidate
    path = unresolved.resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes root: ' + relative)
    for component in (unresolved,) + tuple(unresolved.parents):
        if component == root:
            break
        require(not component.is_symlink(), 'Symlink is not an immutable bound input: ' + relative)
    return path


def verify_files(root, files):
    require(isinstance(files, dict) and bool(files), 'Missing bound file inventory')
    for relative, digest in files.items():
        require(isinstance(digest, str) and len(digest) == 64
                and all(char in '0123456789abcdef' for char in digest),
                'Invalid bound SHA256: ' + str(relative))
        path = bound(root, relative)
        require(path.is_file() and sha256(path) == digest, 'Bound file changed: ' + relative)


def fix01_identity(resume):
    return dict(run_id=FIX01_RUN_ID,
        config_sha256=resume['fix01_configuration_sha256'],
        source_manifest_sha256=resume['fix01_source_manifest_sha256'],
        base_configuration_sha256=resume['base_configuration_sha256'],
        base_source_manifest_sha256=resume['legacy_source_manifest_sha256'])


def resume_identity(resume, config_path, manifest_path):
    return dict(run_id=RUN_ID, config_sha256=sha256(config_path),
        source_manifest_sha256=sha256(manifest_path),
        base_configuration_sha256=resume['base_configuration_sha256'],
        base_source_manifest_sha256=resume['legacy_source_manifest_sha256'],
        fix01_configuration_sha256=resume['fix01_configuration_sha256'],
        fix01_source_manifest_sha256=resume['fix01_source_manifest_sha256'])


def validate_plan(base, fix, resume):
    validate_fix01_plan(base, fix)
    require(resume.get('schema') == 'n01/window008-resume01-config/v1'
            and resume.get('run_id') == RUN_ID, 'Resume01 identity differs')
    require(resume.get('model_implementation') == MODEL_IMPLEMENTATION
            and resume.get('model_loss_data_seed_readout_changes') is False,
            'Resume01 changed the frozen scientific implementation')
    require(resume.get('execution_arms') == ['full', 'simple']
            and tuple(resume.get('comparison_arms', [])) == ARMS,
            'Resume01 arm plan differs')
    require(all(type(resume.get(key)) is int and resume[key] == value
                for key, value in FIXED_COUNTS.items()), 'Resume01 update budgets differ')
    expected_resume = dict(arm='full', reference='fix01_full_periodic_06000',
        G_updates=6000, D_updates_per_hypothesis=3000, stage='joint_stochastic',
        next_G_update=6001, all_optimizer_sampler_stream_states_imported=True,
        first_update_audit=True, first_update_same_version_replay=True)
    require(resume.get('resume') == expected_resume, 'Resume01 full6000 policy differs')
    require(resume.get('fresh_simple') == fix['fresh_simple']
            and resume.get('retained_deterministic') == fix['retained_deterministic'],
            'Retained det/fresh simple policy differs')
    require(set(resume.get('reference_inputs', {})) == EXPECTED_REFERENCES,
            'Resume01 reference inventory differs')
    for name in ORIGINAL_REFERENCES:
        require(resume['reference_inputs'][name] == fix['reference_inputs'][name],
                'Original reference registration differs: ' + name)
    require(resume['reference_inputs']['fix01_full_periodic_06000']['sha256'] == FULL6000_SHA256,
            'Only the registered complete fix01 full6000 checkpoint is admissible')
    require(resume.get('effective_observation_samples') == 256
            and resume.get('internal_carrier_samples') == 384,
            'Resume01 observation dimensions differ')
    require(all(resume.get(key) == base.get(key) == fix.get(key) == 0
                for key in ('noise_updates', 'population_updates'))
            and resume.get('C2ST_updates') == 0
            and resume.get('DEV_waveforms_used_for_training') is False
            and resume.get('scientific_status') == 'response_match_not_established',
            'Resume01 role/update scope differs')
    for key in ('base_configuration', 'base_configuration_sha256',
                'legacy_source_manifest', 'legacy_source_manifest_sha256'):
        require(resume.get(key) == fix[key], 'Base/fix/resume lineage differs: ' + key)
    for key in ('cpu_threads', 'gpu_memory_limit_GiB', 'job_tree_RSS_limit_GiB',
                'output_limit_MiB', 'checkpoint_interval', 'status_interval', 'wall_limit_s'):
        require(resume.get(key) == fix[key], 'Resume01 runtime bound differs: ' + key)
    require(resume.get('one_active_scientific_job') is True,
            'Resume01 one-job boundary differs')


def validate_stop_chain(base, fix, resume, documents, original_chain):
    """JSON metadata only: preserve original failure and fix01 user-stop native1."""
    names = ('fix01_native_exit', 'fix01_launch', 'fix01_binding',
             'fix01_training_inputs', 'fix01_resume_validation', 'fix01_failure',
             'fix01_user_stop_request', 'fix01_user_stop_verification',
             'fix01_user_stop_collection', 'fix01_user_pause', 'fix01_legacy_validation')
    (native, launch, binding, inputs, restored, failure, request, verified,
     collection, pause, legacy) = (documents[name] for name in names)
    refs = resume['reference_inputs']
    previous_identity = fix01_identity(resume)
    require(native.get('exit_code') == 1 and native.get('reason') is None,
            'Fix01 user-stop native exit1 must be preserved')
    for key in ('supervisor_pid', 'child_pid'):
        require(type(native.get(key)) is int and native[key] > 0
                and native[key] == launch.get(key), 'Fix01 native PID chain differs: ' + key)
    command = launch.get('command', [])
    remote = fix['remote_root'].rstrip('/')
    require(isinstance(command, list) and len(command) >= 8
            and command[:4] == ['timeout', '--signal=TERM', '--kill-after=15s', '21600s']
            and PurePosixPath(command[4]).is_absolute()
            and command[5:8] == ['-B', '-m', 'n01.train_window008_fix01'],
            'Fix01 launch is not the registered timeout wrapper')
    flags = {'--container': base['remote_training_container'],
        '--identities': remote + '/data/PARENT_IDENTITIES.npz',
        '--config': remote + '/window008_config.json',
        '--recovery-config': remote + '/window008_fix01_config.json',
        '--dataset': remote + '/' + base['dataset'],
        '--statistics': remote + '/' + base['statistics'],
        '--source-manifest': remote + '/WINDOW008_FIX01_SOURCE_MANIFEST.json',
        '--out': remote + '/runs/science', '--device': 'cuda:0'}
    require(len(command) == 8 + 2 * len(flags)
            and all(argument(command, flag) == value for flag, value in flags.items()),
            'Fix01 native command flags differ')
    require(type(binding.get('pid')) is int and binding['pid'] > 0
            and binding.get('parent_pid') == native['child_pid']
            and binding.get('run_id') == FIX01_RUN_ID
            and binding.get('recovery') == previous_identity
            and binding.get('legacy_native_chain') == original_chain
            and binding.get('before_prepared_numeric_decode') is True,
            'Fix01 trainer binding/PID/recovery differs')
    for key, expected in (('dataset_sha256', base['dataset_sha256']),
                          ('statistics_sha256', base['statistics_sha256']),
                          ('raw_container_sha256', base['train_container_sha256']),
                          ('identities_sha256', base['identities_sha256'])):
        require(binding.get(key) == expected, 'Fix01 input binding differs: ' + key)
    require(binding.get('DEV_waveforms_read') is False
            and inputs.get('DEV_waveforms_read') is False
            and inputs.get('SOURCE_rows') == 6593 and inputs.get('REAL_rows') == 21047
            and inputs.get('effective_samples') == 256
            and inputs.get('internal_carrier_samples') == 384
            and inputs.get('source_ADC') is False and inputs.get('source_count_weights') == 'all_one'
            and inputs.get('noise_updates') == inputs.get('population_updates') == 0
            and inputs.get('recovery') == previous_identity,
            'Fix01 TRAIN role/dimensions/update policy differs')
    require(inputs.get('common_initial_state_file_sha256') == refs['common_initial_state']['sha256']
            and inputs.get('common_initial_state_sha256') == legacy.get('common_initial_state_sha256'),
            'Fix01 common initializer binding differs')
    require(legacy.get('status') == 'passed_CPU_checkpoint_metadata_only'
            and legacy.get('recovery') == previous_identity
            and legacy.get('native_chain') == original_chain
            and legacy.get('reference_files') == {name: refs[name]['sha256'] for name in ORIGINAL_REFERENCES}
            and legacy.get('CPU_scientific_model_forwards') == legacy.get('optimizer_updates') == 0
            and legacy.get('DEV_waveforms_decoded') is False,
            'Fix01 original CPU validation chain differs')
    require(restored.get('schema') == 'n01/window008-fix01-resume/v1'
            and restored.get('status') == 'passed' and restored.get('recovery') == previous_identity
            and restored.get('source_checkpoint_sha256') == refs['full_stage_03000']['sha256']
            and restored.get('G_updates') == 3000 and restored.get('D_updates') == {h: 0 for h in HYPOTHESES}
            and restored.get('next_G_update') == restored.get('projection_update') == 3001
            and restored.get('optimizer_sampler_streams_exact') is True
            and restored.get('first_update_requires_audit') is True
            and restored.get('first_update_requires_same_version_replay') is True,
            'Fix01 original mean-stage restore chain differs')
    expected_failure_counts = dict(new_execution_G_updates=3398, new_execution_D_updates=10194,
        inherited_effective_G_updates=22000, inherited_effective_D_updates=48000,
        effective_comparison_G_updates=25398, effective_comparison_D_updates=58194,
        historical_actual_G_updates=25667, historical_actual_D_updates=59001,
        discarded_legacy_joint_G_updates=269, discarded_legacy_joint_D_updates=807)
    require(failure.get('schema') == 'n01/window008-fix01-failure/v1'
            and failure.get('status') == 'fix01_failed_or_bounded_stop_preserved'
            and failure.get('run_id') == FIX01_RUN_ID
            and failure.get('error') == "TimeoutError('Owned fix01 process received termination signal 15')"
            and all(failure.get(key) == value for key, value in expected_failure_counts.items())
            and failure.get('current_arm') == 'full' and failure.get('current_G_updates') == 6398
            and failure.get('current_D_updates') == {h: 3398 for h in HYPOTHESES}
            and failure.get('legacy_references_validated') is True
            and failure.get('checkpoint_may_be_mid_step') is True
            and failure.get('automatic_resume_allowed') is False
            and failure.get('DEV_waveforms_read') is False and failure.get('evaluation_complete') is False
            and all(failure.get(key) == 0 for key in ('C2ST_updates', 'noise_updates', 'population_updates')),
            'Fix01 user-stop counts/mechanism differ')
    expected_det = retained_endpoint(fix, documents, legacy)
    require(failure.get('completed_arms') == [expected_det], 'Fix01 retained original det receipt differs')
    require(request.get('run_id') == verified.get('run_id') == FIX01_RUN_ID
            and request.get('trainer_pid') == verified.get('trainer_pid') == binding['pid']
            and request.get('binding') == verified.get('binding') == binding
            and request.get('UTC') == verified.get('UTC')
            and request.get('reason') == verified.get('reason')
            and request.get('signal_sent') is None
            and verified.get('signal_sent') == 'SIGTERM_to_verified_owned_trainer_only'
            and request.get('argv_before') == verified.get('argv_before') == command[4:]
            and request.get('proc_stat_before') == verified.get('proc_stat_before'),
            'Fix01 stop request/signal/trainer identity chain differs')
    proc_stat = verified.get('proc_stat_before', '')
    require(proc_stat.split(' ', 1)[0] == str(binding['pid']) and ') ' in proc_stat,
            'Fix01 pre-stop proc identity is missing')
    proc_fields = proc_stat.rsplit(') ', 1)[1].split()
    require(len(proc_fields) >= 2 and proc_fields[1] == str(binding['parent_pid']),
            'Fix01 pre-stop proc parent PID differs')
    require(verified.get('status') == 'user_requested_stop_verified'
            and verified.get('trainer_absent') is True and verified.get('owned_processes_after') == []
            and verified.get('files', {}).get('science/FAILURE.json') == failure
            and verified.get('files', {}).get('supervisor/exit_receipt.json') == native
            and verified.get('files', {}).get('supervisor/launch.json') == launch,
            'Fix01 stop verification does not match archived native/failure receipts')
    require(documents['fix01_user_stop_verification_local'] == verified,
            'Local/remote stop verification JSON differs')
    require(collection.get('status') == 'user_stopped_job_archived'
            and collection.get('native') == native and collection.get('failure') == failure
            and collection.get('frozen_manifest_sha256') == resume['fix01_source_manifest_sha256']
            and collection.get('all_transferred_member_hashes_verified') is True
            and collection.get('owned_training_processes') == []
            and collection.get('new_training_updates_during_collection') == collection.get('DEV_numeric_decodes') == 0
            and collection.get('resume_requires_coherent_boundary_checkpoint_validation') is True,
            'Fix01 user-stop collection chain differs')
    collected_names = dict(fix01_full_periodic_06000='runs/science/train/full/latest.pt',
        fix01_native_exit='runs/supervisor/exit_receipt.json', fix01_launch='runs/supervisor/launch.json',
        fix01_binding='runs/science/BINDING.json', fix01_training_inputs='runs/science/TRAINING_INPUTS.json',
        fix01_resume_validation='runs/science/RESUME_VALIDATION.json', fix01_failure='runs/science/FAILURE.json',
        fix01_legacy_validation='runs/science/LEGACY_VALIDATION.json',
        fix01_user_stop_request='runs/USER_STOP_REQUEST.json',
        fix01_user_stop_verification='runs/USER_STOP_VERIFICATION.json')
    for name, relative in collected_names.items():
        row = collection.get('files', {}).get(relative, {})
        require(row.get('sha256') == refs[name]['sha256'] and row.get('bytes') == refs[name]['bytes'],
                'Fix01 collected raw artifact differs: ' + name)
    periodic = pause.get('periodic_checkpoint', {})
    require(pause.get('status') == 'paused_by_user_before_server_shutdown'
            and pause.get('user_requested_pause') is True
            and pause.get('native_exit_code') == 1
            and pause.get('stop_receipt_sha256') == refs['fix01_user_stop_verification_local']['sha256']
            and pause.get('collection_receipt_sha256') == refs['fix01_user_stop_collection']['sha256']
            and pause.get('local_artifact_files') == len(collection['files'])
            and pause.get('all_artifact_hashes_verified') is True
            and pause.get('owned_trainer_and_supervisor_absent') is True
            and pause.get('actual_new_G_updates') == 3398 and pause.get('actual_new_D_updates') == 10194
            and pause.get('current_arm_G_updates') == 6398
            and periodic.get('sha256') == FULL6000_SHA256 and periodic.get('G_updates') == 6000
            and periodic.get('new_G_updates') == 3000 and periodic.get('D_updates') == {h: 3000 for h in HYPOTHESES}
            and periodic.get('metadata_role_verified') is True
            and pause.get('stop_checkpoint_may_be_mid_step') is True
            and pause.get('automatic_resume_allowed') is False
            and pause.get('model_forwards_for_pause_validation') == pause.get('optimizer_updates_for_pause_validation') == 0
            and pause.get('DEV_numeric_decodes') == 0,
            'Fix01 pause/full6000 raw receipt chain differs')
    require(native.get('started_utc') == launch.get('started_utc'), 'Fix01 start chronology differs')
    timestamps = [native['started_utc'], binding['UTC'], inputs['UTC'], request['UTC'],
                  failure['UTC'], native['ended_utc'], verified['verified_UTC'], collection['UTC'], pause['UTC']]
    times = [datetime.fromisoformat(value) for value in timestamps]
    require(all(first <= second for first, second in zip(times, times[1:])),
            'Fix01 stop receipt chronology differs')
    return dict(original_native_chain=copy.deepcopy(original_chain),
        fix01_native_exit_code=1, fix01_native_supervisor_pid=native['supervisor_pid'],
        fix01_native_child_pid=native['child_pid'], fix01_training_pid=binding['pid'],
        fix01_training_parent_pid=binding['parent_pid'], native_child_role='timeout_wrapper',
        original_native_role='preserved_numerical_failure', fix01_native_role='preserved_explicit_user_SIGTERM',
        old_jobs_native_success=False, retained_deterministic_arm_training_complete=True,
        fix01_actual_new_G_updates=3398, fix01_actual_new_D_updates=10194,
        full_boundary_G_updates=6000, full_boundary_D_updates={h: 3000 for h in HYPOTHESES},
        discarded_user_stop_G_updates=398, discarded_user_stop_D_updates=1194,
        previous_pause_superseded_by_direct_user_resume_authorization=True,
        mid_step_failure_checkpoint_loaded=False, DEV_waveforms_read=False)


def verify_module_inventory(root, *manifests):
    actual = {path.relative_to(root).as_posix() for path in (root / 'src/n01').glob('*.py')}
    expected = {name for manifest in manifests for name in manifest['files']
                if name.startswith('src/n01/') and name.endswith('.py')}
    require(actual == expected, 'Runtime Python module inventory is not fully frozen')


def bind(args, root=ROOT):
    root = Path(root).resolve()
    base, resume, manifest = (read_json(path) for path in (args.config, args.resume_config, args.source_manifest))
    fix_path = bound(root, resume['fix01_configuration'])
    fix = read_json(fix_path)
    validate_plan(base, fix, resume)
    require(Path(args.config).resolve() == bound(root, resume['base_configuration'])
            and Path(args.resume_config).resolve() == root / 'window008_resume01_config.json'
            and Path(args.source_manifest).resolve() == bound(root, resume['source_manifest']),
            'Resume01 configuration/manifest path differs')
    require(sha256(args.config) == resume['base_configuration_sha256']
            and sha256(fix_path) == resume['fix01_configuration_sha256'],
            'Original/fix01 configuration bytes changed')
    original_path = bound(root, resume['legacy_source_manifest'])
    fix_manifest_path = bound(root, resume['fix01_source_manifest'])
    require(sha256(original_path) == resume['legacy_source_manifest_sha256']
            and sha256(fix_manifest_path) == resume['fix01_source_manifest_sha256'],
            'Original/fix01 source manifest bytes changed')
    original_manifest, fix_manifest = read_json(original_path), read_json(fix_manifest_path)
    require(original_manifest.get('schema') == 'n01/window008/source_manifest/v1'
            and original_manifest.get('run_id') == base['run_id']
            and original_manifest.get('config_sha256') == resume['base_configuration_sha256']
            and original_manifest.get('before_scientific_model_forward') is True
            and len(original_manifest.get('files', {})) == 68,
            'Original 68-file freeze identity differs')
    require(fix_manifest.get('schema') == 'n01/window008-fix01-source/v1'
            and fix_manifest.get('run_id') == FIX01_RUN_ID
            and fix_manifest.get('config_sha256') == resume['fix01_configuration_sha256']
            and fix_manifest.get('base_manifest_sha256') == resume['legacy_source_manifest_sha256']
            and fix_manifest.get('before_scientific_model_forward') is True
            and len(fix_manifest.get('files', {})) == 114,
            'Fix01 114-file freeze identity differs')
    require(manifest.get('schema') == 'n01/window008-resume01-source/v1'
            and manifest.get('run_id') == RUN_ID
            and manifest.get('config_sha256') == sha256(args.resume_config)
            and manifest.get('base_manifest_sha256') == resume['legacy_source_manifest_sha256']
            and manifest.get('fix01_manifest_sha256') == resume['fix01_source_manifest_sha256']
            and manifest.get('before_scientific_model_forward') is True,
            'Resume01 source freeze differs')
    for item in (original_manifest, fix_manifest, manifest):
        verify_files(root, item['files'])
    mandatory = REQUIRED_SOURCES | {resume['protocol'], resume['base_configuration'],
        resume['fix01_configuration'], resume['legacy_source_manifest'], resume['fix01_source_manifest']}
    mandatory |= {row['path'] for row in resume['reference_inputs'].values()}
    require(mandatory <= set(manifest['files']), 'Resume01 freeze omits required code/config/reference inputs')
    verify_module_inventory(root, original_manifest, fix_manifest, manifest)
    checks = [(args.dataset, bound(root, base['dataset']), base['dataset_sha256']),
              (args.statistics, bound(root, base['statistics']), base['statistics_sha256']),
              (args.identities, bound(root, 'data/PARENT_IDENTITIES.npz'), base['identities_sha256'])]
    for path, expected_path, digest in checks:
        require(Path(path).resolve() == expected_path and sha256(path) == digest,
                'Prepared input path/hash differs: ' + str(path))
    require(str(Path(args.container).resolve()) in (base['remote_training_container'], base['local_training_container'])
            and sha256(args.container) == base['train_container_sha256'], 'Original TRAIN container differs')
    documents = {}
    for name, row in resume['reference_inputs'].items():
        path = bound(root, row['path'])
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'Historical reference bytes differ: ' + name)
        if path.suffix == '.json':
            documents[name] = read_json(path)
    original_chain = validate_legacy_receipts(base, fix, documents)
    native_chain = validate_stop_chain(base, fix, resume, documents, original_chain)
    return base, fix, resume, manifest, documents, native_chain


def update_accounting(completed, engine=None, *, legacy_validated=True):
    """Use actual completed optimizer counters, including a partial D-only step."""
    rows = [row for row in completed if row['arm'] in ('full', 'simple')
            and (engine is None or row['arm'] != engine.arm)]
    new_g = sum(row['new_G_updates'] for row in rows) + (engine.new_G_updates if engine is not None else 0)
    new_d = sum(sum(row['new_D_updates'].values()) for row in rows)
    if engine is not None:
        new_d += sum(engine.new_D_updates.values())
    require(type(new_g) is int and new_g >= 0 and type(new_d) is int and new_d >= 0,
            'New actual optimizer counters are invalid')
    return dict(new_execution_G_updates=new_g, new_execution_D_updates=new_d,
        inherited_effective_G_updates=25000 if legacy_validated else None,
        inherited_effective_D_updates=57000 if legacy_validated else None,
        effective_comparison_G_updates=25000 + new_g if legacy_validated else None,
        effective_comparison_D_updates=57000 + new_d if legacy_validated else None,
        historical_actual_G_updates_before_resume=25667 if legacy_validated else None,
        historical_actual_D_updates_before_resume=59001 if legacy_validated else None,
        historical_actual_G_updates=25667 + new_g if legacy_validated else None,
        historical_actual_D_updates=59001 + new_d if legacy_validated else None,
        global_N01_G_updates_before_resume=125667 if legacy_validated else None,
        global_N01_actual_G_updates=125667 + new_g if legacy_validated else None,
        discarded_legacy_joint_G_updates=269 if legacy_validated else None,
        discarded_legacy_joint_D_updates=807 if legacy_validated else None,
        discarded_user_stop_G_updates=398 if legacy_validated else None,
        discarded_user_stop_D_updates=1194 if legacy_validated else None,
        legacy_references_validated=legacy_validated)


def save_failure(out, error, completed, engine, *, began, legacy_validated, native_chain=None):
    """Preserve current actual counters and mid-step state without a resume claim."""
    out = Path(out)
    if engine is not None:
        try:
            engine.checkpoint(out / 'FAILURE_CHECKPOINT.pt', purpose='failure')
        except Exception as save_error:
            write_json(out / 'CHECKPOINT_SAVE_ERROR.json', dict(error=repr(save_error)))
        try:
            atomic_torch_save(engine.last_context, out / 'FAILURE_CONTEXT.pt')
        except Exception as save_error:
            write_json(out / 'CONTEXT_SAVE_ERROR.json', dict(error=repr(save_error)))
    result = dict(schema='n01/window008-resume01-failure/v1', UTC=utc(),
        status='resume01_failed_or_bounded_stop_preserved', run_id=RUN_ID,
        error=repr(error), traceback=traceback.format_exc(),
        **update_accounting(completed, engine, legacy_validated=legacy_validated),
        target_new_G_updates=32000, target_new_D_updates=87000,
        completed_arms=completed, current_arm=engine.arm if engine is not None else None,
        current_G_updates=engine.update if engine is not None else None,
        current_D_updates=dict(engine.d_updates) if engine is not None else None,
        legacy_native_chain=copy.deepcopy(native_chain),
        checkpoint_may_be_mid_step=engine is not None, automatic_resume_allowed=False,
        awaiting_own_supervisor_native_exit=True, native_training_success=False,
        DEV_waveforms_read=False, C2ST_updates=0, noise_updates=0, population_updates=0,
        elapsed_seconds=time.monotonic() - began,
        scientific_status='response_match_not_established', evaluation_complete=False)
    write_json(out / 'FAILURE.json', result)
    write_json(out / 'status.json', result)
    return result


def validate_endpoints(completed):
    require([row['arm'] for row in completed] == list(ARMS), 'Completed endpoint arm inventory differs')
    schemas = {'deterministic': 'response008/v1', 'full': 'response008_resume01/v1',
               'simple': 'response008_resume01/v1'}
    origins = {'deterministic': 'retained_original_endpoint', 'full': 'fix01_periodic6000_resume',
               'simple': 'fresh_common_initialization'}
    for row in completed:
        require(row['G_updates'] == 19000 and row['D_updates'] == {h: 16000 for h in HYPOTHESES}
                and row['stage'] == 'low_learning_rate'
                and row['coverage']['source_unique'] == 6593 and row['coverage']['real_unique'] == 21047,
                'Fixed endpoint counters/stage/coverage differ: ' + row['arm'])
        require(row['checkpoint_schema'] == schemas[row['arm']] and row['origin'] == origins[row['arm']]
                and row['model_implementation'] == ('n01.model008.N01Response008'
                    if row['arm'] == 'deterministic' else MODEL_IMPLEMENTATION),
                'Fixed endpoint schema/model/lineage differs: ' + row['arm'])
        for key in ('final_sha256', 'common_initial_state_sha256', 'noise005_state_sha256'):
            require(isinstance(row.get(key), str) and len(row[key]) == 64,
                    'Missing endpoint digest: ' + row['arm'] + '/' + key)
    require(len({row['common_initial_state_sha256'] for row in completed}) == 1
            and len({row['noise005_state_sha256'] for row in completed}) == 1,
            'Endpoint frozen common/noise references differ')
    expected = {'deterministic': (19000, 0, 0), 'full': (6000, 13000, 13000), 'simple': (0, 19000, 16000)}
    for row in completed:
        inherited, new_g, new_d = expected[row['arm']]
        require(row['inherited_G_updates'] == inherited and row['new_G_updates'] == new_g
                and row['new_D_updates'] == {h: new_d for h in HYPOTHESES},
                'Endpoint inherited/new ledger differs: ' + row['arm'])
    accounting = update_accounting(completed)
    require(accounting['new_execution_G_updates'] == 32000 and accounting['new_execution_D_updates'] == 87000
            and accounting['effective_comparison_G_updates'] == 57000
            and accounting['effective_comparison_D_updates'] == 144000
            and accounting['historical_actual_G_updates'] == 57667
            and accounting['historical_actual_D_updates'] == 146001,
            'Final resume01 accounting differs')
    return accounting


def completion_payload(base, resume, identity, completed, files, native_chain, validation_sha256, elapsed):
    accounting = validate_endpoints(completed)
    refs = resume['reference_inputs']
    det = completed[0]
    require(det['endpoint_path'] == refs['deterministic_final']['path']
            and det['final_sha256'] == refs['deterministic_final']['sha256']
            and det['legacy_arm_completion_sha256'] == refs['deterministic_training_complete']['sha256']
            and det.get('source_job_native_exit_code') == 1 and det.get('arm_training_complete') is True,
            'Retained original det endpoint provenance differs')
    for row in completed[1:]:
        require(row['recovery'] == identity
                and row['endpoint_path'] == 'runs/science/train/' + row['arm'] + '/final.pt',
                'New endpoint root/recovery differs: ' + row['arm'])
    require(completed[1]['lineage']['source_checkpoint_sha256'] == FULL6000_SHA256,
            'Completed full endpoint resume-source lineage differs')
    return dict(schema='n01/window008-resume01-training-complete/v1', UTC=utc(),
        status='fixed_resumed_endpoints_complete_pending_native_exit_and_evaluation', run_id=RUN_ID,
        base_run_id=base['run_id'], fix01_run_id=FIX01_RUN_ID, recovery=identity,
        **accounting, arms=completed, files=files,
        endpoint_map={row['arm']: dict(path=row['endpoint_path'], sha256=row['final_sha256'],
            schema=row['checkpoint_schema'], model_implementation=row['model_implementation'], origin=row['origin'])
            for row in completed},
        legacy_native_chain=native_chain,
        legacy_native_exit_sha256=refs['legacy_native_exit']['sha256'],
        legacy_failure_sha256=refs['legacy_failure']['sha256'],
        fix01_native_exit_sha256=refs['fix01_native_exit']['sha256'],
        fix01_failure_sha256=refs['fix01_failure']['sha256'],
        fix01_user_pause_sha256=refs['fix01_user_pause']['sha256'],
        resume_source_sha256=FULL6000_SHA256, resume_validation_sha256=validation_sha256,
        awaiting_own_supervisor_native_exit=True, native_training_success=False,
        evaluation_complete=False, evaluation_adapter_status='pending_versioned_resume01_adapter',
        DEV_waveforms_read=False, C2ST_updates=0, noise_updates=0, population_updates=0,
        elapsed_seconds=elapsed, scientific_status='response_match_not_established')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'resume-config', 'source-manifest', 'dataset', 'statistics', 'container', 'identities', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    require(not (os.name == 'nt' and out.drive.upper() not in ('E:', 'D:')), 'Windows output must stay on E/D')
    if out.exists():
        raise FileExistsError('Fresh resume01 output required; prior attempts are never resumed or overwritten')
    out.mkdir(parents=True, exist_ok=False)
    engine = None
    completed = []
    native_chain = None
    legacy_validated = False
    began = time.monotonic()
    def stop(signum, frame):
        raise TimeoutError('Owned resume01 process received termination signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    try:
        base, fix, resume, manifest, documents, native_chain = bind(args)
        require(os.name == 'posix' and str(ROOT.resolve()) == resume['remote_root']
                and out == ROOT / 'runs/science' and args.device == 'cuda:0',
                'Science must run in the registered fresh Linux root/device')
        statistics = read_json(args.statistics)
        require(statistics.get('prepared_dataset_sha256') == base['dataset_sha256']
                and statistics.get('statistics_split') == 'TRAIN', 'Statistics binding/role differs')
        common, cpu_validation = validate_legacy_states(ROOT, base, fix, statistics, documents)
        previous_validation = documents['fix01_legacy_validation']
        for key, value in cpu_validation.items():
            require(previous_validation.get(key) == value, 'Inherited CPU validation changed: ' + key)
        legacy_validated = True
        identity = resume_identity(resume, args.resume_config, args.source_manifest)
        write_json(out / 'BINDING.json', dict(schema='n01/window008-resume01-binding/v1', UTC=utc(),
            run_id=RUN_ID, pid=os.getpid(), parent_pid=os.getppid(), recovery=identity,
            dataset_sha256=sha256(args.dataset), statistics_sha256=sha256(args.statistics),
            raw_container_sha256=base['train_container_sha256'], identities_sha256=base['identities_sha256'],
            before_prepared_numeric_decode=True, DEV_waveforms_read=False,
            legacy_native_chain=native_chain, **update_accounting([])))
        write_json(out / 'LEGACY_VALIDATION.json', dict(UTC=utc(), **cpu_validation,
            native_chain=native_chain, reference_files={name: row['sha256'] for name, row in resume['reference_inputs'].items()},
            recovery=identity, mid_step_failure_checkpoint_loaded=False))
        completed.append(retained_endpoint(fix, documents, cpu_validation))
        device = torch.device(args.device)
        runtime = copy.deepcopy(base)
        for key in ('cpu_threads', 'gpu_memory_limit_GiB', 'job_tree_RSS_limit_GiB', 'wall_limit_s', 'output_limit_MiB'):
            runtime[key] = resume[key]
        configure_runtime(runtime, device)
        with np.load(args.dataset, allow_pickle=False) as archive:
            require(not ({'source_val', 'target_val', 'hval'} & set(archive.files)), 'Prepared input contains DEV values')
            data = {name: archive[name].copy() for name in archive.files}
        validate_training_data(data)
        noise_path = bound(ROOT, base['noise_reference']['path'])
        require(sha256(noise_path) == base['noise_reference']['sha256'], 'Frozen005 bytes changed')
        noise = torch.load(noise_path, map_location='cpu', weights_only=False)
        require(noise.get('update') == 5000 and noise.get('schema_version') == 5
                and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
                'Wrong frozen005 endpoint')
        write_json(out / 'TRAINING_INPUTS.json', dict(UTC=utc(), SOURCE_rows=6593, REAL_rows=21047,
            effective_samples=256, internal_carrier_samples=384, source_ADC=False,
            common_initial_state_sha256=state_digest(common),
            common_initial_state_file_sha256=resume['reference_inputs']['common_initial_state']['sha256'],
            source_count_weights='all_one', original_source_count_weights_preserved=True,
            original_physical_weights='unknown', DEV_waveforms_read=False,
            noise_updates=0, population_updates=0, recovery=identity))
        for arm in resume['execution_arms']:
            folder = out / 'train' / arm
            folder.mkdir(parents=True, exist_ok=False)
            engine = Response008Resume01Engine(data, statistics, base, noise, common,
                arm=arm, recovery_identity=identity, device=device)
            require(engine.noise_state_digest == cpu_validation['noise005_state_sha256']
                    and engine.initial_state_digest == cpu_validation['initial_model_state_sha256'][arm],
                    'Engine frozen reference construction differs')
            if arm == 'full':
                reference = resume['reference_inputs']['fix01_full_periodic_06000']
                validation = engine.import_fix01_full_periodic6000(bound(ROOT, reference['path']),
                    expected_sha256=reference['sha256'], fix01_identity=fix01_identity(resume))
                write_json(out / 'RESUME_VALIDATION.json', validation)
                engine.checkpoint(folder / 'restored_start.pt', purpose='restored_start')
            else:
                engine.set_stage('mean_response')
                engine.checkpoint(folder / 'initial.pt', purpose='initial')
            write_json(folder / 'INITIALIZATION.json', dict(UTC=utc(), arm=arm, recovery=identity,
                lineage=engine.lineage, model_replacement=engine.model_replacement_receipt,
                G_updates=engine.update, D_updates=dict(engine.d_updates), new_G_updates=0,
                new_D_updates={h: 0 for h in HYPOTHESES}))
            while engine.update < base['response']['updates_per_arm']:
                require(time.monotonic() - began < resume['wall_limit_s'] - 30,
                        'Registered resume01 wall bound reached before next update')
                stage = response_stage(engine.update, base)
                changed = stage != engine.stage
                if changed:
                    engine.set_stage(stage)
                update = engine.update + 1
                first_new = engine.new_G_updates == 0
                audit = first_new or update <= 2 or changed or update % base['engineering']['gradient_audit_interval'] == 0
                replay = first_new or update <= base['engineering']['waveform_replay_first_steps'] or changed
                metrics = engine.step(audit=audit, verify_replay=replay)
                with (folder / 'training.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + '\n')
                if audit or engine.update % resume['status_interval'] == 0:
                    status = dict(status='training_window008_resume01', run_id=RUN_ID, pid=os.getpid(),
                        elapsed_seconds=time.monotonic() - began, DEV_waveforms_read=False,
                        target_new_G_updates=32000, target_new_D_updates=87000,
                        **update_accounting(completed, engine), **metrics)
                    write_json(out / 'status.json', status)
                    print(json.dumps({key: status[key] for key in ('utc', 'status', 'arm', 'stage', 'G_updates',
                        'new_execution_G_updates', 'new_execution_D_updates', 'effective_comparison_G_updates',
                        'historical_actual_G_updates', 'weighted_total', 'update_seconds', 'coverage')}), flush=True)
                if engine.update % resume['checkpoint_interval'] == 0:
                    engine.enforce_frozen_references()
                    engine.checkpoint(folder / 'latest.pt', purpose='periodic', allow_overwrite=True)
                if engine.update in (3000, 15000, 19000):
                    path = folder / ('final.pt' if engine.update == 19000 else f'stage_{engine.update:05d}.pt')
                    engine.checkpoint(path, purpose='final' if engine.update == 19000 else 'stage_endpoint')
                if engine.update % 1000 == 0:
                    require(sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) <= resume['output_limit_MiB'] * 1024**2,
                            'Registered resume01 output bound exceeded')
            require(engine.update == 19000 and engine.d_updates == {h: 16000 for h in HYPOTHESES},
                    'Endpoint update accounting differs')
            engine.enforce_frozen_references()
            coverage = engine.last_metrics['coverage']
            require(coverage['source_unique'] == 6593 and coverage['real_unique'] == 21047,
                    'All-parent coverage incomplete')
            done = dict(UTC=utc(), arm=arm, origin=engine.lineage['origin'], G_updates=engine.update,
                D_updates=dict(engine.d_updates), inherited_G_updates=engine.imported_G_updates,
                new_G_updates=engine.new_G_updates, new_D_updates=engine.new_D_updates,
                coverage=coverage, checkpoint_schema='response008_resume01/v1', stage=engine.stage,
                model_implementation=MODEL_IMPLEMENTATION, endpoint_path=(folder / 'final.pt').relative_to(ROOT).as_posix(),
                final_sha256=sha256(folder / 'final.pt'), state_sha256=state_digest(engine.model.state_dict()),
                lineage=engine.lineage, common_initial_state_sha256=engine.common_state_digest,
                noise005_state_sha256=engine.noise_state_digest, recovery=identity)
            completed.append(done)
            write_json(folder / 'TRAINING_COMPLETE.json', done)
            del engine
            engine = None
            gc.collect()
            torch.cuda.empty_cache()
        validate_endpoints(completed)
        # Recheck all three generations and immutable inputs before signing endpoints.
        bind(args)
        files = {path.relative_to(out).as_posix(): sha256(path) for path in sorted(out.rglob('*'))
                 if path.is_file() and path not in (out / 'status.json', out / 'TRAINING_COMPLETE.json')}
        result = completion_payload(base, resume, identity, completed, files, native_chain,
            sha256(out / 'RESUME_VALIDATION.json'), time.monotonic() - began)
        write_json(out / 'TRAINING_COMPLETE.json', result)
        write_json(out / 'status.json', result)
    except BaseException as error:
        save_failure(out, error, completed, engine, began=began,
            legacy_validated=legacy_validated, native_chain=native_chain)
        raise


if __name__ == '__main__':
    main()
