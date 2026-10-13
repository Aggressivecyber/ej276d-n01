"""Frozen mixed-generation Window008 resume01 evaluation; no fitting or updates.

The original evaluator's numerical functions remain byte-for-byte frozen. This
module supplies a new completion gate, a strict mixed-model loader, and the
same orchestration for the recovered endpoints. Historical native1 receipts
retain their original failure/user-stop roles.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import platform

import numpy as np
import torch
from torch import nn

from . import evaluate_window008 as numeric
from .data import sha256
from .losses008 import WaveformDiscriminator
from .losses008_v2 import N01Loss008
from .model008 import N01Response008
from .model008_fix01 import N01Response008Fix01
from .response008 import (ARMS, HYPOTHESES, loss_tensor_statistics, make_extension,
                          make_signal, state_digest, statistics_digest)
from .response008_fix01 import (MODEL_IMPLEMENTATION, optimizer_named_state,
    parameter_signature, require, tree_digest, validate_optimizer,
    validate_sampler, validate_streams)
from .response008_resume01 import (assert_tree_bitwise_equal,
    validate_module_state, validate_phase_optimizer)
from .train import utc
from .train_window008_fix01 import (retained_endpoint, validate_legacy_receipts,
                                    validate_legacy_states)
from .train_window008_resume01 import (FIX01_RUN_ID, FULL6000_SHA256, RUN_ID,
    REQUIRED_SOURCES, bound, fix01_identity, resume_identity, update_accounting,
    validate_endpoints, validate_plan, validate_stop_chain, verify_files)


ROOT = Path(__file__).resolve().parents[2]
NUMERIC_CORE_SHA256 = '75f8988415a181a31a7c2b5890418b1f7685e0d8fb141cdea84f4174af783cc0'
EVALUATION_SCHEMA = 'n01/window008-resume01-evaluation/v1'
FREEZE_SCHEMA = 'n01/window008-resume01-evaluation-freeze/v1'
EVALUATION_STATUS = 'fixed_resumed_endpoint_evaluation_complete'
EVALUATION_MANIFEST = 'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'
EVALUATION_POLICY = 'window008_resume01_evaluation_policy.json'
NATIVE_TREE_KEYS = ('model', 'discriminators', 'g_optimizer', 'd_optimizers',
                    'source_sampler', 'real_sampler', 'streams')

# These are aliases to the frozen numerical implementation, never replacements.
SCHEMES, METRICS, COMPARATORS = numeric.SCHEMES, numeric.METRICS, numeric.COMPARATORS
save_json, save_npz, _array, _clean = numeric.save_json, numeric.save_npz, numeric._array, numeric._clean
generate_split, parent_bootstrap, distance_receipt = numeric.generate_split, numeric.parent_bootstrap, numeric.distance_receipt
balanced_real_domains, _validate_split, _load_dev = numeric.balanced_real_domains, numeric._validate_split, numeric._load_dev
_fixed_settings, _verify_frozen = numeric._fixed_settings, numeric._verify_frozen


def _json_text(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def _read_json(value):
    return _json_text(Path(value).read_text(encoding='utf8')) if isinstance(value, (str, Path)) else copy.deepcopy(value)


def _path(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else _bound(root, str(value))


def _bound(root, relative):
    require(isinstance(relative, str) and PurePosixPath(relative).as_posix() == relative
            and '\\' not in relative, 'Noncanonical bound path: ' + str(relative))
    return bound(root, relative)


def _hashes(root, files):
    # The driver supplies cross-platform traversal/symlink and SHA256 checks.
    verify_files(root, files)
    paths = [_bound(root, name) for name in files]
    require(len(paths) == len(set(paths)), 'Aliased file inventory')
    return {str(path): files[name] for name, path in zip(files, paths)}


def _time(value):
    require(isinstance(value, str), 'Missing UTC timestamp')
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'Naive UTC timestamp')
    return result


def _chronology(values, label):
    timestamps = [_time(value) for value in values]
    require(all(a <= b for a, b in zip(timestamps, timestamps[1:])), label + ' chronology differs')


def _load_checkpoint(path, expected):
    require(sha256(path) == expected, 'Checkpoint bytes changed before load: ' + str(path))
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(sha256(path) == expected, 'Checkpoint bytes changed during load: ' + str(path))
    require(isinstance(state, dict), 'Checkpoint must be a dictionary')
    return state


def _training_sources(cfg, resume_cfg, root):
    cfg, resume_cfg = _read_json(cfg), _read_json(resume_cfg)
    config_path = _bound(root, resume_cfg['base_configuration'])
    fix_path = _bound(root, resume_cfg['fix01_configuration'])
    resume_path = _bound(root, 'window008_resume01_config.json')
    require(_read_json(config_path) == cfg and _read_json(resume_path) == resume_cfg,
            'Registered base/resume configuration differs')
    fix = _read_json(fix_path)
    validate_plan(cfg, fix, resume_cfg)
    _fixed_settings(cfg)
    require(sha256(config_path) == resume_cfg['base_configuration_sha256']
            and sha256(fix_path) == resume_cfg['fix01_configuration_sha256'],
            'Registered base/fix01 configuration bytes differ')
    names = (resume_cfg['legacy_source_manifest'], resume_cfg['fix01_source_manifest'], resume_cfg['source_manifest'])
    paths = [_bound(root, name) for name in names]
    manifests = [_read_json(path) for path in paths]
    original, fixed, resumed = manifests
    require(sha256(paths[0]) == resume_cfg['legacy_source_manifest_sha256']
            and sha256(paths[1]) == resume_cfg['fix01_source_manifest_sha256'],
            'Historical source manifest bytes differ')
    require(original.get('schema') == 'n01/window008/source_manifest/v1'
            and original.get('run_id') == cfg['run_id']
            and original.get('config_sha256') == sha256(config_path)
            and original.get('before_scientific_model_forward') is True
            and len(original.get('files', {})) == 68, 'Original 68-file source freeze differs')
    require(fixed.get('schema') == 'n01/window008-fix01-source/v1'
            and fixed.get('run_id') == FIX01_RUN_ID
            and fixed.get('config_sha256') == sha256(fix_path)
            and fixed.get('base_manifest_sha256') == sha256(paths[0])
            and fixed.get('before_scientific_model_forward') is True
            and len(fixed.get('files', {})) == 114, 'Fix01 114-file source freeze differs')
    require(resumed.get('schema') == 'n01/window008-resume01-source/v1'
            and resumed.get('run_id') == RUN_ID
            and resumed.get('config_sha256') == sha256(resume_path)
            and resumed.get('base_manifest_sha256') == sha256(paths[0])
            and resumed.get('fix01_manifest_sha256') == sha256(paths[1])
            and resumed.get('before_scientific_model_forward') is True, 'Resume01 source freeze differs')
    mandatory = REQUIRED_SOURCES | {resume_cfg['protocol'], resume_cfg['base_configuration'],
        resume_cfg['fix01_configuration'], resume_cfg['legacy_source_manifest'], resume_cfg['fix01_source_manifest']}
    mandatory |= {row['path'] for row in resume_cfg['reference_inputs'].values()}
    require(mandatory <= set(resumed['files']), 'Resume01 freeze omits a required input/source')
    files = {str(path): sha256(path) for path in [config_path, fix_path, resume_path] + paths}
    for manifest in manifests:
        for path, digest in _hashes(root, manifest['files']).items():
            require(path not in files or files[path] == digest, 'Conflicting frozen source hashes')
            files[path] = digest
    require(sha256(_bound(root, 'src/n01/evaluate_window008.py')) == NUMERIC_CORE_SHA256,
            'Frozen original numeric evaluator changed')
    documents = {}
    for name, row in resume_cfg['reference_inputs'].items():
        path = _bound(root, row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'Historical reference bytes differ: ' + name)
        files[str(path)] = row['sha256']
        if path.suffix == '.json':
            documents[name] = _read_json(path)
    require(all(type(documents[name].get('exit_code')) is int and documents[name]['exit_code'] == 1
                for name in ('legacy_native_exit', 'fix01_native_exit')), 'Historical native1 must retain integer failure/user-stop roles')
    original_chain = validate_legacy_receipts(cfg, fix, documents)
    native_chain = validate_stop_chain(cfg, fix, resume_cfg, documents, original_chain)
    identity = resume_identity(resume_cfg, resume_path, paths[2])
    return cfg, fix, resume_cfg, manifests, documents, native_chain, identity, files


def _native_completion(run, cfg, resume_cfg, complete, binding, identity):
    native_path, launch_path = run.parent / 'supervisor/exit_receipt.json', run.parent / 'supervisor/launch.json'
    native, launch = _read_json(native_path), _read_json(launch_path)
    require(type(native.get('exit_code')) is int and native['exit_code'] == 0 and native.get('reason') is None,
            'Resume01 requires its own native exit0 without a termination reason')
    for key in ('supervisor_pid', 'child_pid'):
        require(type(native.get(key)) is int and native[key] > 0 and native[key] == launch.get(key),
                'Resume01 native PID binding differs: ' + key)
    require(type(binding.get('pid')) is int and binding['pid'] > 0
            and type(binding.get('parent_pid')) is int and binding['parent_pid'] == native['child_pid']
            and len({native['supervisor_pid'], native['child_pid'], binding['pid']}) == 3,
            'Resume01 trainer/timeout/supervisor roles differ')
    command = launch.get('command')
    remote = resume_cfg['remote_root'].rstrip('/')
    flags = {'--container': cfg['remote_training_container'], '--identities': remote + '/data/PARENT_IDENTITIES.npz',
        '--config': remote + '/' + resume_cfg['base_configuration'], '--resume-config': remote + '/window008_resume01_config.json',
        '--dataset': remote + '/' + cfg['dataset'], '--statistics': remote + '/' + cfg['statistics'],
        '--source-manifest': remote + '/' + resume_cfg['source_manifest'], '--out': remote + '/runs/science', '--device': 'cuda:0'}
    require(isinstance(command, list) and len(command) == 8 + 2 * len(flags)
            and command[:4] == ['timeout', '--signal=TERM', '--kill-after=15s', str(resume_cfg['wall_limit_s']) + 's']
            and isinstance(command[4], str) and PurePosixPath(command[4]).is_absolute()
            and command[5:8] == ['-B', '-m', 'n01.train_window008_resume01'],
            'Resume01 native command/module/timeout differs')
    pairs = command[8:]
    require(len(set(pairs[::2])) == len(flags) and dict(zip(pairs[::2], pairs[1::2])) == flags,
            'Resume01 native command argument inventory differs')
    require(native.get('started_utc') == launch.get('started_utc'), 'Resume01 launch/native start differs')
    _chronology([native['started_utc'], binding['UTC'], complete['UTC'], native['ended_utc']], 'Resume01 native')
    require(binding.get('schema') == 'n01/window008-resume01-binding/v1' and binding.get('run_id') == RUN_ID
            and binding.get('recovery') == identity and binding.get('before_prepared_numeric_decode') is True
            and binding.get('DEV_waveforms_read') is False
            and binding.get('dataset_sha256') == cfg['dataset_sha256']
            and binding.get('statistics_sha256') == cfg['statistics_sha256']
            and binding.get('raw_container_sha256') == cfg['train_container_sha256']
            and binding.get('identities_sha256') == cfg['identities_sha256'], 'Resume01 trainer input/identity binding differs')
    for key, value in update_accounting([]).items():
        require(binding.get(key) == value and type(binding.get(key)) is type(value), 'Initial resume01 accounting differs: ' + key)
    return dict(native_exit_code=0, native_receipt=str(native_path), launch_receipt=str(launch_path),
        supervisor_pid=native['supervisor_pid'], timeout_wrapper_pid=native['child_pid'], trainer_pid=binding['pid'],
        registered_remote_root=remote, registered_command=command, ended_UTC=native['ended_utc'])


def _expected_lineage(arm, cfg, resume_cfg):
    value = dict(origin='fresh_common_initialization', base_run_id=cfg['run_id'], parent_run_id=FIX01_RUN_ID,
        source_checkpoint_sha256=None, inherited_G_updates=0, inherited_D_updates={h: 0 for h in HYPOTHESES},
        model_implementation=MODEL_IMPLEMENTATION)
    if arm == 'full':
        value.update(origin='fix01_periodic6000_resume', source_checkpoint_sha256=FULL6000_SHA256,
            inherited_G_updates=6000, inherited_D_updates={h: 3000 for h in HYPOTHESES},
            parent_lineage=dict(origin='stage3000_recovery', base_run_id=cfg['run_id'],
                source_checkpoint_sha256=resume_cfg['reference_inputs']['full_stage_03000']['sha256'],
                inherited_G_updates=3000, inherited_D_updates={h: 0 for h in HYPOTHESES}))
    return value


def _optimizer_groups(model, original, lr):
    """Use the hash-bound original AdamW defaults without creating an optimizer."""
    require(isinstance(original, list) and len(original) == 1, 'Original AdamW group differs')
    groups = copy.deepcopy(original)
    groups[0]['params'] = list(range(len(list(model.parameters()))))
    groups[0]['lr'] = lr
    return groups


def _model(arm, statistics):
    with torch.random.fork_rng(devices=[]):
        return (N01Response008 if arm == 'deterministic' else N01Response008Fix01)(statistics['model'], arm)


def _discriminators(statistics):
    with torch.random.fork_rng(devices=[]):
        return nn.ModuleDict({h: WaveformDiscriminator(loss_tensor_statistics(statistics['loss'])) for h in HYPOTHESES})


def _validate_checkpoint(state, arm, cfg, statistics, identity, parent_identity, common_hash, noise_hash,
                         initial_hash, initial_d_hash, lineage, original_initial, *, update=19000,
                         stage='low_learning_rate', purpose='final'):
    model = _model(arm, statistics)
    d_updates = max(update - 3000, 0)
    require(state.get('schema') == ('response008/v1' if arm == 'deterministic' else 'response008_resume01/v1')
            and state.get('arm') == arm and state.get('config') == cfg
            and type(state.get('G_updates')) is int and state['G_updates'] == update
            and state.get('D_updates') == {h: d_updates for h in HYPOTHESES}
            and all(type(value) is int for value in state['D_updates'].values())
            and state.get('stage') == stage and state.get('noise_updates') == 0
            and state.get('population_learning') is False, 'Endpoint schema/config/counters/stage differ: ' + arm)
    digest = statistics_digest(statistics)
    require(state.get('statistics_sha256') == digest and statistics_digest(state['statistics']) == digest,
            'Endpoint TRAIN statistics differ: ' + arm)
    require(state.get('common_initial_state_sha256') == common_hash and state.get('noise005_state_sha256') == noise_hash
            and state.get('initial_state_sha256') == initial_hash and state.get('initial_D_sha256') == initial_d_hash,
            'Endpoint common/noise/initial-state binding differs: ' + arm)
    if arm != 'deterministic':
        inherited = 6000 if arm == 'full' else 0
        inherited_d = 3000 if arm == 'full' else 0
        require(state.get('model_implementation') == MODEL_IMPLEMENTATION
                and state.get('parameter_signature') == parameter_signature(model)
                and state.get('recovery') == identity and state.get('parent_fix01_recovery') == parent_identity
                and state.get('lineage') == lineage
                and type(state.get('new_G_updates')) is int and state['new_G_updates'] == update - inherited
                and state.get('new_D_updates') == {h: d_updates - inherited_d for h in HYPOTHESES}
                and state.get('engineering_fixture') is False and state.get('checkpoint_purpose') == purpose
                and state.get('automatic_resume_allowed') is False and state.get('failure_may_be_mid_step') is False
                and state.get('complete_step_boundary') is True, 'New endpoint lineage/recovery/purpose differs: ' + arm)
    validate_module_state(model, state['model'], arm + ' response')
    model.load_state_dict(state['model'], strict=True)
    ds = _discriminators(statistics)
    validate_module_state(ds, state['discriminators'], arm + ' discriminators')
    ds.load_state_dict(state['discriminators'], strict=True)
    low = stage == 'low_learning_rate'
    g_groups = _optimizer_groups(model, original_initial['g_optimizer']['param_groups'], cfg['loss']['G_low_lr' if low else 'G_lr'])
    if update:
        g_slots, g_steps = validate_phase_optimizer(model, state['g_optimizer'], g_groups,
            mean_steps=update, joint_steps=None if arm == 'deterministic' else d_updates)
    else:
        g_slots = validate_optimizer(model, state['g_optimizer'], g_groups, update=0)
        g_steps = {name: 0 for name in g_slots}
    require(set(state['d_optimizers']) == set(HYPOTHESES), 'Endpoint D optimizer hypotheses differ')
    for hypothesis in HYPOTHESES:
        groups = _optimizer_groups(ds[hypothesis], original_initial['d_optimizers'][hypothesis]['param_groups'],
                                   cfg['loss']['D_low_lr' if low else 'D_lr'])
        if d_updates:
            validate_phase_optimizer(ds[hypothesis], state['d_optimizers'][hypothesis], groups, mean_steps=d_updates)
        else:
            validate_optimizer(ds[hypothesis], state['d_optimizers'][hypothesis], groups, update=0)
    validate_sampler(state['source_sampler'], 6593, update * cfg['distribution_microbatches'] * cfg['source_parents_per_microbatch'])
    validate_sampler(state['real_sampler'], 21047, update * cfg['distribution_microbatches'] * cfg['real_rows_per_microbatch'])
    validate_streams(state['streams'])
    if update:
        metrics = state['last_metrics']
        require(metrics.get('arm') == arm and metrics.get('G_updates') == update
                and metrics.get('D_updates') == state['D_updates'] and metrics.get('stage') == stage
                and metrics.get('projection_context', {}).get('update') == update,
                'Endpoint last completed step/projection differs: ' + arm)
        if arm != 'deterministic' and purpose != 'restored_start':
            require(metrics.get('recovery_run_id') == RUN_ID and metrics.get('model_implementation') == MODEL_IMPLEMENTATION
                    and metrics.get('inherited_G_updates') == lineage['inherited_G_updates']
                    and metrics.get('inherited_D_updates') == lineage['inherited_D_updates']
                    and metrics.get('new_G_updates') == state['new_G_updates']
                    and metrics.get('new_D_updates') == state['new_D_updates'], 'Endpoint metrics lineage differs: ' + arm)
    return model, g_steps


def _verify_import(run, cfg, resume_cfg, statistics, identity, legacy_validation, original_initial, files):
    """Verify the native state import including every AdamW moment, without a step."""
    reference = resume_cfg['reference_inputs']['fix01_full_periodic_06000']
    source_path = Path(next(path for path in files if path.endswith('/' + reference['path'])
                            or path.replace('\\', '/').endswith('/' + reference['path'])))
    source = _load_checkpoint(source_path, reference['sha256'])
    parent_identity = fix01_identity(resume_cfg)
    require(source.get('schema') == 'response008_fix01/v1' and source.get('arm') == 'full'
            and source.get('G_updates') == 6000 and source.get('D_updates') == {h: 3000 for h in HYPOTHESES}
            and source.get('stage') == 'joint_stochastic' and source.get('recovery') == parent_identity
            and source.get('checkpoint_purpose') == 'periodic' and source.get('failure_may_be_mid_step') is False
            and source.get('engineering_fixture') is False and source.get('automatic_resume_allowed') is False,
            'Registered fix01 full6000 complete boundary differs')
    lineage = _expected_lineage('full', cfg, resume_cfg)
    require(source.get('lineage') == lineage['parent_lineage'] and source.get('new_G_updates') == 3000
            and source.get('new_D_updates') == {h: 3000 for h in HYPOTHESES}, 'Registered fix01 full6000 lineage differs')
    restored_path = _bound(run, 'train/full/restored_start.pt')
    restored = _load_checkpoint(restored_path, files[str(restored_path)])
    model, g_steps = _validate_checkpoint(restored, 'full', cfg, statistics, identity, parent_identity,
        legacy_validation['common_initial_state_sha256'], legacy_validation['noise005_state_sha256'],
        legacy_validation['initial_model_state_sha256']['full'], original_initial['initial_D_sha256'],
        lineage, original_initial, update=6000, stage='joint_stochastic', purpose='restored_start')
    for key in NATIVE_TREE_KEYS:
        assert_tree_bitwise_equal(source[key], restored[key], 'full6000 import/' + key)
    validation_path = run / 'RESUME_VALIDATION.json'
    validation = _read_json(validation_path)
    require(validation.get('schema') == 'n01/window008-resume01-import/v1' and validation.get('status') == 'passed'
            and validation.get('source_checkpoint_sha256') == FULL6000_SHA256
            and validation.get('source_checkpoint_purpose') == 'periodic'
            and validation.get('source_schema') == 'response008_fix01/v1' and validation.get('target_schema') == 'response008_resume01/v1'
            and validation.get('G_updates') == 6000 and validation.get('D_updates') == {h: 3000 for h in HYPOTHESES}
            and validation.get('new_G_updates') == 0 and validation.get('new_D_updates') == {h: 0 for h in HYPOTHESES}
            and validation.get('next_G_update') == validation.get('projection_update') == 6001
            and validation.get('optimizer_moments_reset') is False and validation.get('native_tree_bitwise_equal') is True
            and validation.get('optimizer_sampler_streams_exact') is True
            and validation.get('first_update_requires_audit') is True and validation.get('first_update_requires_same_version_replay') is True
            and validation.get('engineering_fixture') is False and validation.get('metadata_only_placeholder') is False
            and validation.get('recovery') == identity and validation.get('fix01_recovery') == parent_identity
            and validation.get('parameter_signature') == parameter_signature(model)
            and validation.get('G_optimizer_named_steps') == g_steps,
            'Resume01 first-step import validation differs')
    expected_digests = {key: tree_digest(source[key]) for key in NATIVE_TREE_KEYS}
    require(validation.get('state_digests') == expected_digests, 'Imported model/AdamW/sampler/stream digest differs')
    g_hashes = {name: tree_digest(slot) for name, slot in optimizer_named_state(model, source['g_optimizer']).items()}
    require(validation.get('G_optimizer_named_slot_sha256') == g_hashes, 'Imported named G AdamW moment digest differs')
    ds = _discriminators(statistics)
    d_hashes = {h: {name: tree_digest(slot) for name, slot in optimizer_named_state(ds[h], source['d_optimizers'][h]).items()}
                for h in HYPOTHESES}
    require(validation.get('D_optimizer_named_slot_sha256') == d_hashes, 'Imported named D AdamW moment digest differs')
    return validation, dict(source_checkpoint_sha256=FULL6000_SHA256, state_digests=expected_digests,
        G_optimizer_named_steps=g_steps, optimizer_moments_reset=False, native_tree_bitwise_equal=True,
        scientific_model_forwards=0, optimizer_updates=0)


def _log_boundaries(path, arm, cfg):
    """Read first/final/phase-change metadata, never waveform values."""
    records = {}
    expected_first = 6001 if arm == 'full' else 1
    wanted = {expected_first, 3000, 3001, 15000, 15001, 19000}
    count = 0
    with path.open(encoding='utf8') as stream:
        for line in stream:
            count += 1
            require(bool(line.strip()), 'Empty scientific training log row')
            if count in {1, 3000, 3001, 9000, 9001, 13000, 15000, 15001, 19000}:
                row = _json_text(line)
                if row.get('G_updates') in wanted:
                    records[row['G_updates']] = row
    require(count == (13000 if arm == 'full' else 19000), 'Completed training log length differs: ' + arm)
    required = {expected_first, 15000, 15001, 19000} | ({3000, 3001} if arm == 'simple' else set())
    require(set(records) == required, 'Completed training phase-boundary rows differ: ' + arm)
    for update, row in records.items():
        stage = 'mean_response' if update <= 3000 else 'joint_stochastic' if update <= 15000 else 'low_learning_rate'
        require(row.get('arm') == arm and row.get('G_updates') == update and row.get('stage') == stage
                and row.get('D_updates') == {h: max(update - 3000, 0) for h in HYPOTHESES}
                and row.get('projection_context', {}).get('update') == update
                and row.get('recovery_run_id') == RUN_ID and row.get('model_implementation') == MODEL_IMPLEMENTATION,
                'Completed step phase/counters/projection differs: ' + arm)
    require(records[expected_first].get('same_version_replay_verified') is True
            and records[expected_first].get('endpoint_gradients'), 'First new step audit/replay evidence missing: ' + arm)
    return {str(update): dict(UTC=row['utc'], stage=row['stage'], G_updates=row['G_updates'],
        D_updates=row['D_updates'], same_version_replay_verified=row['same_version_replay_verified'])
            for update, row in records.items()}


def training_completion_gate(run, cfg, resume_cfg, *, root=None):
    """CPU metadata gate for original native1, user-stop native1, and native0.

    ``root`` anchors immutable project sources/references. ``run`` anchors the
    collected resume01 science output, permitting a hash-verified local mirror.
    The gate never decodes an NPZ, calls a module forward, or creates an optimizer.
    """
    root, run = Path(root or ROOT).resolve(), Path(run).resolve()
    require(run.name == 'science' and run.parent.name == 'runs', 'Registered science output layout differs')
    cfg, fix, resume_cfg, manifests, documents, native_chain, identity, files = _training_sources(cfg, resume_cfg, root)
    complete_path = _bound(run, 'TRAINING_COMPLETE.json')
    complete = _read_json(complete_path)
    require(complete.get('schema') == 'n01/window008-resume01-training-complete/v1'
            and complete.get('status') == 'fixed_resumed_endpoints_complete_pending_native_exit_and_evaluation'
            and complete.get('run_id') == RUN_ID and complete.get('base_run_id') == cfg['run_id']
            and complete.get('fix01_run_id') == FIX01_RUN_ID and complete.get('recovery') == identity
            and complete.get('legacy_native_chain') == native_chain
            and complete.get('awaiting_own_supervisor_native_exit') is True
            and complete.get('native_training_success') is False and complete.get('evaluation_complete') is False
            and complete.get('DEV_waveforms_read') is False and complete.get('scientific_status') == 'response_match_not_established'
            and all(type(complete.get(key)) is int and complete[key] == 0 for key in ('C2ST_updates', 'noise_updates', 'population_updates')),
            'Resume01 pending-native completion contract differs')
    rows = complete.get('arms')
    accounting = validate_endpoints(rows)
    for key, value in accounting.items():
        require(complete.get(key) == value and type(complete.get(key)) is type(value), 'Final resume01 accounting differs: ' + key)
    inventory = complete.get('files')
    expected_files = {'BINDING.json', 'LEGACY_VALIDATION.json', 'RESUME_VALIDATION.json', 'TRAINING_INPUTS.json'}
    for arm in ('full', 'simple'):
        expected_files |= {'train/' + arm + '/' + name for name in ('INITIALIZATION.json', 'TRAINING_COMPLETE.json',
            'final.pt', 'latest.pt', 'stage_15000.pt', 'training.jsonl')}
    expected_files |= {'train/full/restored_start.pt', 'train/simple/initial.pt', 'train/simple/stage_03000.pt'}
    require(isinstance(inventory, dict) and set(inventory) == expected_files, 'Resume01 completion file inventory differs')
    actual = {path.relative_to(run).as_posix() for path in run.rglob('*') if path.is_file()
              and path not in (complete_path, run / 'status.json')}
    require(actual == expected_files, 'Resume01 output has missing or extra completion files')
    files.update(_hashes(run, inventory)); files[str(complete_path)] = sha256(complete_path)
    binding = _read_json(run / 'BINDING.json')
    require(binding.get('legacy_native_chain') == native_chain, 'Resume01 binding historical native chain differs')
    native = _native_completion(run, cfg, resume_cfg, complete, binding, identity)
    for name in ('native_receipt', 'launch_receipt'):
        files[native[name]] = sha256(native[name])
    provenance_fields = {'legacy_native_exit_sha256': 'legacy_native_exit', 'legacy_failure_sha256': 'legacy_failure',
        'fix01_native_exit_sha256': 'fix01_native_exit', 'fix01_failure_sha256': 'fix01_failure', 'fix01_user_pause_sha256': 'fix01_user_pause'}
    for key, name in provenance_fields.items():
        require(complete.get(key) == resume_cfg['reference_inputs'][name]['sha256'], 'Historical completion digest differs: ' + key)
    require(complete.get('resume_source_sha256') == FULL6000_SHA256
            and complete.get('resume_validation_sha256') == inventory['RESUME_VALIDATION.json'], 'Resume source/import digest differs')
    statistics_path = _bound(root, cfg['statistics'])
    require(sha256(statistics_path) == cfg['statistics_sha256'], 'Registered TRAIN statistics bytes changed')
    statistics = _read_json(statistics_path)
    require(statistics.get('statistics_split', statistics.get('split')) == 'TRAIN'
            and statistics['loss'].get('fit_split') == 'TRAIN'
            and statistics.get('prepared_dataset_sha256') == cfg['dataset_sha256'], 'Statistics roles/dataset binding differ')
    files[str(statistics_path)] = sha256(statistics_path)
    common, legacy_validation = validate_legacy_states(root, cfg, fix, statistics, documents)
    previous = documents['fix01_legacy_validation']
    current = _read_json(run / 'LEGACY_VALIDATION.json')
    for key, value in legacy_validation.items():
        require(previous.get(key) == value and current.get(key) == value, 'Inherited legacy metadata validation differs: ' + key)
    require(current.get('native_chain') == native_chain and current.get('recovery') == identity
            and current.get('reference_files') == {name: row['sha256'] for name, row in resume_cfg['reference_inputs'].items()}
            and current.get('mid_step_failure_checkpoint_loaded') is False, 'Current legacy-reference validation differs')
    initial_ref = resume_cfg['reference_inputs']['full_initial']
    original_initial = _load_checkpoint(_bound(root, initial_ref['path']), initial_ref['sha256'])
    noise_path = _bound(root, cfg['noise_reference']['path'])
    noise = _load_checkpoint(noise_path, cfg['noise_reference']['sha256'])
    require(noise.get('update') == 5000 and noise.get('schema_version') == 5
            and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
            'Registered frozen005 endpoint metadata differs')
    extension = make_extension(noise, 'cpu')
    noise_hash = state_digest(extension.model005.state_dict()); extension.assert_frozen()
    require(noise_hash == legacy_validation['noise005_state_sha256'], 'Frozen005 tensor digest differs')
    files[str(noise_path)] = cfg['noise_reference']['sha256']
    inputs = _read_json(run / 'TRAINING_INPUTS.json')
    require(inputs.get('SOURCE_rows') == 6593 and inputs.get('REAL_rows') == 21047
            and inputs.get('effective_samples') == 256 and inputs.get('internal_carrier_samples') == 384
            and inputs.get('source_ADC') is False and inputs.get('DEV_waveforms_read') is False
            and inputs.get('source_count_weights') == 'all_one' and inputs.get('original_source_count_weights_preserved') is True
            and inputs.get('original_physical_weights') == 'unknown'
            and inputs.get('common_initial_state_sha256') == state_digest(common)
            and inputs.get('common_initial_state_file_sha256') == resume_cfg['reference_inputs']['common_initial_state']['sha256']
            and inputs.get('noise_updates') == inputs.get('population_updates') == 0 and inputs.get('recovery') == identity,
            'Resume01 TRAIN input roles/common initializer differ')
    expected_det = retained_endpoint(fix, documents, legacy_validation)
    require(rows[0] == expected_det, 'Retained deterministic endpoint receipt differs')
    endpoint_map = complete.get('endpoint_map')
    require(isinstance(endpoint_map, dict) and list(endpoint_map) == list(ARMS), 'Mixed endpoint-map arm inventory/order differs')
    endpoint_paths, state_hashes, optimizer_steps = {}, {}, {}
    for row in rows:
        arm = row['arm']
        expected = dict(path=row['endpoint_path'], sha256=row['final_sha256'], schema=row['checkpoint_schema'],
                        model_implementation=row['model_implementation'], origin=row['origin'])
        require(endpoint_map[arm] == expected and set(endpoint_map[arm]) == set(expected), 'Mixed endpoint map differs: ' + arm)
        if arm == 'deterministic':
            require(row['endpoint_path'] == resume_cfg['reference_inputs']['deterministic_final']['path'], 'Deterministic endpoint path differs')
            path = _bound(root, row['endpoint_path'])
            initial_hash = documents['legacy_training_inputs']['common_initial_state_sha256']
            # Deterministic has only the common mean-branch subset, recorded in its original checkpoint.
            initial_model = make_signal(statistics['model'], arm, common, cfg['seeds']['initialization'], 'cpu')
            initial_hash = state_digest(initial_model.state_dict())
            lineage = None
        else:
            require(row['endpoint_path'] == 'runs/science/train/' + arm + '/final.pt'
                    and row.get('recovery') == identity, 'New endpoint path/recovery differs: ' + arm)
            path = _bound(run, 'train/' + arm + '/final.pt')
            require(inventory['train/' + arm + '/final.pt'] == row['final_sha256'], 'Indexed final digest differs: ' + arm)
            require(_read_json(run / 'train' / arm / 'TRAINING_COMPLETE.json') == row, 'Arm completion differs: ' + arm)
            lineage = _expected_lineage(arm, cfg, resume_cfg)
            require(row.get('lineage') == lineage, 'Arm endpoint lineage differs: ' + arm)
            initial_hash = legacy_validation['initial_model_state_sha256'][arm]
        require(sha256(path) == row['final_sha256'], 'Endpoint bytes differ: ' + arm)
        require(all(not os.path.samefile(path, previous) for previous in endpoint_paths.values()), 'Aliased mixed endpoints')
        state = _load_checkpoint(path, row['final_sha256'])
        model, steps = _validate_checkpoint(state, arm, cfg, statistics, identity, fix01_identity(resume_cfg),
            legacy_validation['common_initial_state_sha256'], noise_hash, initial_hash, original_initial['initial_D_sha256'],
            lineage, original_initial)
        digest = state_digest(model.state_dict())
        coverage = state['last_metrics']['coverage']
        require(digest == row['state_sha256'] and coverage == row['coverage']
                and coverage == dict(source_unique=6593, source_total=6593, real_unique=21047, real_total=21047,
                    source_epochs=state['source_sampler']['completed_epochs'], real_epochs=state['real_sampler']['completed_epochs'])
                and row['common_initial_state_sha256'] == legacy_validation['common_initial_state_sha256']
                and row['noise005_state_sha256'] == noise_hash, 'Endpoint state/coverage/reference digest differs: ' + arm)
        if arm != 'deterministic':
            _chronology([state['saved_UTC'], row['UTC'], complete['UTC']], 'Arm endpoint ' + arm)
        endpoint_paths[arm], state_hashes[arm], optimizer_steps[arm] = str(path), digest, steps
        files[str(path)] = row['final_sha256']
    validation, import_proof = _verify_import(run, cfg, resume_cfg, statistics, identity, legacy_validation, original_initial, files)
    simple_initial_path = _bound(run, 'train/simple/initial.pt')
    simple_initial = _load_checkpoint(simple_initial_path, inventory['train/simple/initial.pt'])
    _validate_checkpoint(simple_initial, 'simple', cfg, statistics, identity, fix01_identity(resume_cfg),
        legacy_validation['common_initial_state_sha256'], noise_hash, legacy_validation['initial_model_state_sha256']['simple'],
        original_initial['initial_D_sha256'], _expected_lineage('simple', cfg, resume_cfg), original_initial,
        update=0, stage='mean_response', purpose='initial')
    expected_simple = make_signal(statistics['model'], 'simple', common, cfg['seeds']['initialization'], 'cpu').state_dict()
    assert_tree_bitwise_equal(simple_initial['model'], expected_simple, 'fresh simple common initialization')
    for key in ('discriminators', 'd_optimizers'):
        assert_tree_bitwise_equal(simple_initial[key], original_initial[key], 'fresh simple/' + key)
    boundaries = {arm: _log_boundaries(_bound(run, 'train/' + arm + '/training.jsonl'), arm, cfg) for arm in ('full', 'simple')}
    _chronology([binding['UTC'], inputs['UTC'], validation['UTC'], boundaries['full']['6001']['UTC'], rows[1]['UTC'],
                 boundaries['simple']['1']['UTC'], rows[2]['UTC'], complete['UTC'], native['ended_UTC']], 'Resume01 execution')
    # Recheck the complete chain after CPU metadata loading; immutable failure
    # checkpoint bytes are hashed as evidence and are never torch.load inputs.
    for path, digest in files.items():
        require(sha256(path) == digest, 'Training input changed during completion gate: ' + path)
    return dict(schema='n01/window008-resume01-completion-gate/v1', status='mixed_endpoints_and_native_chain_verified',
        run_id=RUN_ID, recovery=identity, endpoint_map=endpoint_map, endpoint_paths=endpoint_paths,
        endpoint_files_sha256={arm: row['sha256'] for arm, row in endpoint_map.items()},
        checkpoint_state_sha256=state_hashes, common_initial_state_sha256=legacy_validation['common_initial_state_sha256'],
        noise_state_sha256=noise_hash, native_exit_code=0, native_completion=native, legacy_native_chain=native_chain,
        completion_G_updates=57000, completion_D_updates=144000, accounting=accounting,
        training_completion=str(complete_path), training_source_manifest=str(_bound(root, resume_cfg['source_manifest'])),
        files_sha256=files, optimizer_named_steps=optimizer_steps, resume_import=import_proof,
        training_boundary_metadata=boundaries, scientific_model_forwards=0, optimizer_updates=0,
        DEV_numeric_decoded=False, mid_step_failure_checkpoint_loaded=False, scientific_status='response_match_not_established')


def _evaluation_source_gate(root, cfg, resume_cfg, source_manifest, policy_path):
    """Require the additive, independently frozen evaluator policy/source set."""
    root = Path(root).resolve()
    manifest_path, policy_path = _path(source_manifest, root), _path(policy_path, root)
    require(manifest_path == _bound(root, EVALUATION_MANIFEST) and policy_path == _bound(root, EVALUATION_POLICY),
            'Versioned evaluation manifest/policy path differs')
    manifest, policy = _read_json(manifest_path), _read_json(policy_path)
    training_manifest = _bound(root, resume_cfg['source_manifest'])
    require(manifest.get('schema') == 'n01/window008-resume01-evaluation-source/v1'
            and manifest.get('run_id') == RUN_ID and manifest.get('before_evaluation_numeric_read') is True
            and type(manifest.get('optimizer_updates')) is int and manifest['optimizer_updates'] == 0
            and manifest.get('training_source_manifest_sha256') == sha256(training_manifest)
            and manifest.get('training_config_sha256') == sha256(_bound(root, 'window008_resume01_config.json'))
            and manifest.get('base_config_sha256') == sha256(_bound(root, resume_cfg['base_configuration']))
            and manifest.get('policy_sha256') == sha256(policy_path), 'Independent evaluation source/policy freeze differs')
    files = _hashes(root, manifest['files'])
    required = {'src/n01/evaluate_window008_resume01.py', EVALUATION_POLICY,
                'scripts/check_window008_resume01_evaluation.py', 'docs/WINDOW008_RESUME01_EVALUATION.md'}
    require(required <= set(manifest['files']), 'Evaluation freeze omits its adapter/policy/QA/documentation')
    historical = [_read_json(_bound(root, resume_cfg[name])) for name in ('legacy_source_manifest', 'fix01_source_manifest', 'source_manifest')]
    protected = {name for item in historical for name in item['files']}
    require(not protected.intersection(manifest['files']), 'Additive evaluation manifest aliases a frozen training input')
    modules = {name for item in historical + [manifest] for name in item['files'] if name.startswith('src/n01/') and name.endswith('.py')}
    actual = {path.relative_to(root).as_posix() for path in (root / 'src/n01').glob('*.py')}
    require(manifest.get('module_inventory') == sorted(modules) and modules == actual,
            'Runtime evaluator module inventory is not exactly frozen')
    require(policy.get('schema') == 'n01/window008-resume01-evaluation-policy/v1' and policy.get('run_id') == RUN_ID
            and policy.get('training_source_manifest_sha256') == sha256(training_manifest)
            and policy.get('evaluation_module') == 'n01.evaluate_window008_resume01'
            and policy.get('training_run') == 'runs/science' and policy.get('job_root') == 'runs/evaluation_jobs'
            and policy.get('jobs') == ['primary', 'replay01']
            and type(policy.get('optimizer_updates')) is int and policy['optimizer_updates'] == 0
            and type(policy.get('C2ST_updates')) is int and policy['C2ST_updates'] == 0
            and policy.get('scientific_status') == 'response_match_not_established', 'Evaluation policy identity/layout/update scope differs')
    limits = policy.get('resource_limits', {})
    require(limits.get('GPU_GiB') == resume_cfg['gpu_memory_limit_GiB']
            and limits.get('job_tree_RSS_GiB') == resume_cfg['job_tree_RSS_limit_GiB']
            and limits.get('output_MiB') == resume_cfg['output_limit_MiB']
            and type(limits.get('wall_seconds')) is int and limits['wall_seconds'] == resume_cfg['wall_limit_s']
            and type(resume_cfg.get('cpu_threads')) is int and resume_cfg['cpu_threads'] == 12
            and limits.get('poll_seconds') == 1 and limits.get('TERM_grace_seconds') == 20,
            'Independent evaluation resource policy differs')
    replay = policy.get('replay', {})
    require(all(replay.get(key) is True for key in ('same_training_endpoints', 'same_training_config_and_seeds',
            'compare_all_NPZ_members', 'compare_all_chunk_records', 'compare_distances_and_parent_bootstrap'))
            and replay.get('draws_per_parent') == 4 and replay.get('chunk_rows') == 128
            and replay.get('additional_unique_draws') == 0 and replay.get('overwrite_allowed') is False
            and replay.get('array_comparison') == 'exact_dtype_shape_values_and_member_sha256', 'Exact replay policy differs')
    require(sha256(_bound(root, 'src/n01/evaluate_window008.py')) == NUMERIC_CORE_SHA256, 'Frozen numerical core changed')
    files.update({str(manifest_path): sha256(manifest_path), str(policy_path): sha256(policy_path)})
    return policy, manifest, files


def load_endpoints(completion, cfg, statistics, device='cpu'):
    """Strict mixed-model loader after the CPU completion gate; no forward."""
    models = {}
    for arm in ARMS:
        path = Path(completion['endpoint_paths'][arm])
        state = _load_checkpoint(path, completion['endpoint_files_sha256'][arm])
        require(state['config'] == cfg and statistics_digest(state['statistics']) == statistics_digest(statistics)
                and state['schema'] == completion['endpoint_map'][arm]['schema'] and state['arm'] == arm
                and state['G_updates'] == 19000 and state['D_updates'] == {h: 16000 for h in HYPOTHESES}
                and state['stage'] == 'low_learning_rate'
                and state['common_initial_state_sha256'] == completion['common_initial_state_sha256']
                and state['noise005_state_sha256'] == completion['noise_state_sha256'], 'Mixed endpoint changed after gate: ' + arm)
        model = _model(arm, statistics)
        validate_module_state(model, state['model'], arm)
        model.load_state_dict(state['model'], strict=True)
        require(state_digest(model.state_dict()) == completion['checkpoint_state_sha256'][arm], 'Endpoint tensor state changed: ' + arm)
        model.set_stage('low_learning_rate'); model.requires_grad_(False); model.eval(); model.to(device)
        require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()), 'Endpoint is not frozen')
        models[arm] = model
    return models


def _freeze(run, cfg, resume_cfg, statistics, dataset_path, statistics_path, raw_container, identities, device,
            destination, *, source_manifest, policy_path, root=None):
    root = Path(root or ROOT).resolve()
    cfg, resume_cfg, statistics = _read_json(cfg), _read_json(resume_cfg), _read_json(statistics)
    policy, manifest, source_files = _evaluation_source_gate(root, cfg, resume_cfg, source_manifest, policy_path)
    settings = _fixed_settings(cfg)
    completion = training_completion_gate(run, cfg, resume_cfg, root=root)
    for value, relative, digest in ((dataset_path, cfg['dataset'], cfg['dataset_sha256']),
            (statistics_path, cfg['statistics'], cfg['statistics_sha256']),
            (identities, 'data/PARENT_IDENTITIES.npz', cfg['identities_sha256'])):
        require(value is not None and Path(value).resolve() == _bound(root, relative) and sha256(value) == digest,
                'Frozen evaluation input path/bytes differ: ' + relative)
    require(str(Path(raw_container).resolve()) in (cfg['local_training_container'], cfg['remote_training_container'])
            and sha256(raw_container) == cfg['train_container_sha256'], 'Original evaluation container path/bytes differ')
    require(statistics == _read_json(statistics_path)
            and statistics.get('statistics_split', statistics.get('split')) == 'TRAIN'
            and statistics['loss'].get('fit_split') == 'TRAIN' and statistics['prepared_dataset_sha256'] == cfg['dataset_sha256'],
            'Evaluation statistics must be the bound TRAIN-only bytes')
    data_statistics_path = Path(dataset_path).with_name('FIT_DATA_STATISTICS.json')
    require(sha256(data_statistics_path) == statistics['data_statistics_sha256'], 'Frozen domain preparation statistics changed')
    data_statistics = _read_json(data_statistics_path)
    require(data_statistics.get('split') == 'TRAIN' and data_statistics.get('DEV_waveforms_decoded') is False,
            'Domain preparation statistics are not TRAIN-only')
    numeric.source_bins(np.asarray([0.]), dict(loss=statistics['loss'], domains=data_statistics['domains']))
    memory_cfg = copy.deepcopy(cfg); memory_cfg['gpu_memory_limit_GiB'] = policy['resource_limits']['GPU_GiB']
    memory_limit = numeric._memory_limit(memory_cfg, device)
    models = load_endpoints(completion, cfg, statistics, device)
    noise_path = _bound(root, cfg['noise_reference']['path'])
    noise = _load_checkpoint(noise_path, cfg['noise_reference']['sha256'])
    extension = make_extension(noise, device); extension.assert_frozen()
    require(state_digest(extension.model005.state_dict()) == completion['noise_state_sha256'], 'Noise005 changed after completion gate')
    files = dict(completion['files_sha256'])
    for path in (dataset_path, statistics_path, data_statistics_path, raw_container, identities, noise_path):
        files[str(Path(path).resolve())] = sha256(path)
    sources = {str(path): sha256(path) for path in sorted((root / 'src/n01').glob('*.py'))}
    sources.update(source_files)
    receipt = dict(schema=FREEZE_SCHEMA, UTC=utc(), status='all_endpoints_frozen_before_DEV_numeric_read', run_id=RUN_ID,
        checkpoint_state_sha256=completion['checkpoint_state_sha256'], noise_state_sha256=completion['noise_state_sha256'],
        files_sha256=files, sources_sha256=sources, config_sha256=statistics_digest(cfg),
        statistics_sha256=statistics_digest(statistics), resolved_settings=settings, DEV_numeric_decoded=False,
        optimizer_updates=0, C2ST_updates=0, common_initial_state_sha256=completion['common_initial_state_sha256'],
        config=cfg, resume_config=resume_cfg, recovery=completion['recovery'], endpoint_map=completion['endpoint_map'],
        native_training_completion=completion, accounting=completion['accounting'], legacy_native_chain=completion['legacy_native_chain'],
        evaluation_source_manifest_sha256=sha256(_path(source_manifest, root)), evaluation_policy_sha256=sha256(_path(policy_path, root)),
        numeric_core_sha256=NUMERIC_CORE_SHA256, scientific_status='response_match_not_established',
        runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, device=str(device),
            CUDA_build=torch.version.cuda, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark, matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_TF32=torch.backends.cudnn.allow_tf32, CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            cpu_threads=torch.get_num_threads(), GPU_memory_limit=memory_limit))
    _verify_frozen(models, extension, receipt)
    save_json(Path(destination) / 'FROZEN_INPUTS.json', receipt)
    return models, extension, receipt, settings, data_statistics


def evaluate_run(out, cfg, resume_cfg, dataset, statistics, raw_container, identities, device='cuda:0', *,
                 destination, source_manifest, policy_path, root=None):
    """Evaluate the three recovered endpoints in a fresh primary/replay folder."""
    root, run, destination = Path(root or ROOT).resolve(), Path(out).resolve(), Path(destination).resolve()
    require(destination in {run.parent / 'evaluation_jobs' / kind / 'artifacts' for kind in ('primary', 'replay01')},
            'Evaluation requires the registered fresh primary/replay01 artifact directory')
    require(not (os.name == 'nt' and destination.drive.upper() not in ('E:', 'D:')), 'Evaluation outputs must stay on E/D')
    cfg, resume_cfg = _read_json(_path(cfg, root)) if isinstance(cfg, (str, Path)) else _read_json(cfg), \
                      _read_json(_path(resume_cfg, root)) if isinstance(resume_cfg, (str, Path)) else _read_json(resume_cfg)
    statistics_path, dataset_path = _path(statistics, root), _path(dataset, root)
    statistics = _read_json(statistics_path)
    raw_container, identities = _path(raw_container, root), _path(identities, root)
    destination.mkdir(parents=True, exist_ok=False)
    dev_decoded = False; dev_decode_started = False
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(int(resume_cfg['cpu_threads']))
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        models, extension, freeze, settings, data_statistics = _freeze(run, cfg, resume_cfg, statistics,
            dataset_path, statistics_path, raw_container, identities, device, destination,
            source_manifest=source_manifest, policy_path=policy_path, root=root)
        # Numerical orchestration below is the original frozen evaluation recipe.
        evaluation_statistics = copy.deepcopy(statistics)
        evaluation_statistics['domains'] = data_statistics['domains']
        with np.load(dataset_path, allow_pickle=False) as archive:
            if {'source_val', 'target_val', 'hval'} & set(archive.files):
                raise ValueError('Prepared dataset unexpectedly contains DEV numerical values')
            train = {key: archive[key].copy() for key in archive.files}
        _validate_split(train, 'TRAIN')
        dev_decode_started = True
        dev = _load_dev(raw_container, identities, cfg['window'], float(statistics['loss']['q_floor']))
        dev_decoded = True; _validate_split(dev, 'DEV')
        if (np.intersect1d(train['source_ids'], dev['source_ids']).size or np.intersect1d(train['real_ids'], dev['real_ids']).size
                or np.intersect1d(train['real_groups'], dev['real_groups']).size):
            raise ValueError('TRAIN/DEV parent or REAL-group overlap')
        if not np.array_equal(train['source_val_ids'], dev['source_ids']) or not np.array_equal(train['real_val_original_index'], dev['real_ids']):
            raise ValueError('DEV lineage differs from the prepared identity-only binding')
        objective = N01Loss008(loss_tensor_statistics(statistics['loss']), seed=cfg['seeds']['train_projection'],
                              evaluation_seed=cfg['seeds']['evaluation_projection']).to(device).eval().requires_grad_(False)
        context = objective.create_evaluation_context()
        save_npz(destination / 'evaluation_projection_bank.npz', **{key: _array(value) for key, value in context.banks.items()})
        selections, distances, comparisons, split_results = {}, {}, {}, {}
        for split, data in (('TRAIN', train), ('DEV', dev)):
            count = settings['matched_' + split]
            selected = np.random.default_rng(cfg['seeds']['selected_' + split]).permutation(len(data['source']))[:count]
            a, b = balanced_real_domains(data['real_groups'], count, cfg['seeds']['real_' + split])
            selections[split] = dict(SOURCE_rows=selected, SOURCE_parent_ids=data['source_ids'][selected],
                REAL_A_rows=a, REAL_B_rows=b, REAL_A_original_ids=data['real_ids'][a], REAL_B_original_ids=data['real_ids'][b],
                REAL_A_groups=data['real_groups'][a], REAL_B_groups=data['real_groups'][b])
            generated = generate_split(split, data, models, extension, evaluation_statistics, cfg, settings, destination, selected, device)
            bootstrap, replicates = parent_bootstrap(generated, evaluation_statistics, cfg, settings, split)
            save_json(destination / (split + '_PARENT_BOOTSTRAP.json'), bootstrap)
            save_npz(destination / (split + '_parent_bootstrap.npz'), **replicates)
            distances[split] = {arm: {h: distance_receipt(objective, generated['matched'][arm][h], data['real'][a],
                generated['matched_baseline'], data['real_baseline100'][a], context) for h in HYPOTHESES} for arm in ARMS}
            distances[split]['REAL_REAL'] = distance_receipt(objective, data['real'][b], data['real'][a],
                                                           data['real_baseline100'][b], data['real_baseline100'][a], context)
            comparisons[split] = {}
            for reference in COMPARATORS:
                comparisons[split][reference] = {h: {metric: dict(
                    full_minus_reference=distances[split]['full'][h]['components'][metric] - distances[split][reference][h]['components'][metric],
                    relative_distance_reduction=(1 - distances[split]['full'][h]['components'][metric] / distances[split][reference][h]['components'][metric])
                    if distances[split][reference][h]['components'][metric] > 0 else None)
                    for metric in ('wave', 'QPSD', 'noise')} for h in HYPOTHESES}
            matched = {arm + '_' + h: generated['matched'][arm][h] for arm in ARMS for h in HYPOTHESES}
            matched.update(generated_baseline100=generated['matched_baseline'], SOURCE_parent_ids=data['source_ids'][selected],
                REAL_A=data['real'][a], REAL_B=data['real'][b], REAL_A_baseline100=data['real_baseline100'][a],
                REAL_B_baseline100=data['real_baseline100'][b], REAL_A_original_ids=data['real_ids'][a], REAL_B_original_ids=data['real_ids'][b])
            save_npz(destination / (split + '_matched_waveforms.npz'), **matched)
            split_results[split] = dict(source_parents=len(data['source']), REAL_rows=len(data['real']),
                generated_rows_per_arm_hypothesis=len(generated['parents']), engineering=generated['engineering'],
                all_observation_rows_retained=True, bootstrap_draws=settings['bootstrap_draws'])
            context.validate()
            del generated, replicates
        save_json(destination / 'MATCHED_SELECTIONS.json', dict(selections=selections,
            REAL_rule='equal group quotas with randomized remainder; disjoint A/B within each group',
            SOURCE_rule='fixed seeded parents without replacement, draw0 only', original_population_summary_separate=True))
        save_json(destination / 'DISTRIBUTION_DISTANCES.json', dict(distances=distances, full_vs_reference=comparisons,
            projection_context=context.metadata(), hypothesis_selection=False, averaging_as_uncertainty=False))
        _verify_frozen(models, extension, freeze)
        artifacts = {path.name: sha256(path) for path in sorted(destination.iterdir()) if path.is_file()}
        summary = dict(schema=EVALUATION_SCHEMA, UTC=utc(), status=EVALUATION_STATUS, run_id=RUN_ID,
            fixed_training_G_updates=57000, fixed_training_D_updates=144000, evaluation_G_updates=0, evaluation_D_updates=0,
            evaluation_noise_updates=0, population_updates=0, C2ST_updates=0, DEV_numeric_decoded=dev_decoded,
            split_results=split_results, full_vs_reference=comparisons, distances=distances,
            scientific_status='response_match_not_established', scientific_pass_threshold=None,
            independent_acquisition_available=False, DEV_previously_exposed=cfg.get('validation_previously_exposed', True),
            paired_event_response_truth_available=False, hypotheses_are_sensitivity_assumptions_not_confidence_set=True,
            source_population_learning=False, all_original_rows_preserved=True, frozen_inputs_rechecked=True,
            endpoint_map=freeze['endpoint_map'], recovery=freeze['recovery'], accounting=freeze['accounting'],
            native_training_completion=freeze['native_training_completion']['native_completion'],
            legacy_native_chain=freeze['legacy_native_chain'], evaluation_source_manifest_sha256=freeze['evaluation_source_manifest_sha256'],
            evaluation_policy_sha256=freeze['evaluation_policy_sha256'], numeric_core_sha256=NUMERIC_CORE_SHA256,
            units=dict(waveform='V', charge='V ns', sample_interval_ns=1), artifact_sha256=artifacts,
            interpretation='Unpaired single-acquisition distribution/width comparisons do not establish physical detector-response identity.')
        save_json(destination / 'SUMMARY.json', summary)
        return _clean(summary)
    except BaseException as error:
        save_json(destination / 'FAILED.json', dict(UTC=utc(), status='resume01_evaluation_failed_preserved', run_id=RUN_ID,
            error_type=type(error).__name__, error=str(error), DEV_numeric_decoded=dev_decoded,
            DEV_numeric_decode_started=dev_decode_started, evaluation_optimizer_updates=0, C2ST_updates=0,
            partial_artifacts_preserved=True, automatic_restart_allowed=False, scientific_status='response_match_not_established'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'config', 'resume-config', 'dataset', 'statistics', 'raw-container', 'identities',
                 'destination', 'source-manifest', 'policy'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    evaluate_run(args.out, args.config, args.resume_config, args.dataset, args.statistics, args.raw_container,
        args.identities, args.device, destination=args.destination, source_manifest=args.source_manifest, policy_path=args.policy)


if __name__ == '__main__':
    main()
