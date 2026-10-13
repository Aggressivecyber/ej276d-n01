"""Fixed fix01 recovery: retain det19k, import full3000 + run16k, fresh simple19k."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import gc
import json
import os
from pathlib import Path, PurePosixPath
import signal
import time
import traceback

import numpy as np
import torch

from .data import sha256, write_json
from .model008 import N01Response008
from .response008 import ARMS, HYPOTHESES, make_signal, response_stage, state_digest, statistics_digest
from .response008_fix01 import (MODEL_IMPLEMENTATION, RUN_ID, Response008Fix01Engine,
    assert_tree_equal, optimizer_named_state, parameter_signature, require,
    validate_sampler, validate_streams)
from .train import atomic_torch_save, utc
from .train_window008 import configure_runtime

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REFERENCES = {'common_initial_state', 'full_stage_03000', 'full_initial',
    'deterministic_final', 'deterministic_stage_03000', 'deterministic_training_complete',
    'legacy_failure', 'legacy_native_exit', 'legacy_launch', 'legacy_binding',
    'legacy_training_inputs', 'legacy_failure_checkpoint', 'legacy_failure_context'}
REQUIRED_SOURCES = {'src/n01/model008_fix01.py', 'src/n01/response008_fix01.py',
    'src/n01/train_window008_fix01.py', 'window008_fix01_config.json'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def bound(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    require(not candidate.is_absolute() and bool(candidate.parts) and '..' not in candidate.parts,
            'Expected an in-root relative path: ' + str(relative))
    unresolved = root / candidate
    path = unresolved.resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes root: ' + str(relative))
    for component in (unresolved,) + tuple(unresolved.parents):
        if component == root:
            break
        require(not component.is_symlink(), 'Symlink is not an immutable bound input: ' + str(relative))
    return path


def verify_files(root, files):
    require(isinstance(files, dict) and bool(files), 'Missing bound file inventory')
    for relative, digest in files.items():
        path = bound(root, relative)
        require(path.is_file() and sha256(path) == digest, 'Bound file changed: ' + relative)


def argument(command, flag):
    require(command.count(flag) == 1 and command.index(flag) + 1 < len(command), 'Native command flag missing/duplicated: ' + flag)
    return command[command.index(flag) + 1]


def validate_legacy_receipts(base, fix, documents):
    native, launch, failure, binding, inputs, det = (documents[name] for name in
        ('legacy_native_exit', 'legacy_launch', 'legacy_failure', 'legacy_binding',
         'legacy_training_inputs', 'deterministic_training_complete'))
    require(native.get('exit_code') == 1 and native.get('reason') is None, 'Original native failure must remain exit1')
    for key in ('supervisor_pid', 'child_pid'):
        require(type(native.get(key)) is int and native[key] == launch.get(key), 'Legacy native PID chain differs: ' + key)
    command = launch.get('command', [])
    registered_root = base['remote_root'].rstrip('/')
    require(isinstance(command, list) and len(command) >= 8
            and command[:4] == ['timeout','--signal=TERM','--kill-after=15s','28800s']
            and PurePosixPath(command[4]).is_absolute() and command[5:8] == ['-B','-m','n01.train_window008'],
            'Legacy native launch is not the bound Window008 timeout wrapper')
    flags = {'--container':base['remote_training_container'],
        '--identities':registered_root+'/data/PARENT_IDENTITIES.npz',
        '--config':registered_root+'/window008_config.json',
        '--dataset':registered_root+'/'+base['dataset'], '--statistics':registered_root+'/'+base['statistics'],
        '--source-manifest':registered_root+'/WINDOW008_SOURCE_MANIFEST.json',
        '--out':registered_root+'/runs/science','--device':'cuda:0'}
    require(len(command) == 8 + 2*len(flags) and all(argument(command,flag)==value for flag,value in flags.items()),
            'Legacy native command input/output flags differ')
    require(type(binding.get('pid')) is int and binding['pid'] > 0, 'Legacy trainer PID is missing')
    require(native.get('started_utc') == launch.get('started_utc'), 'Legacy native start time differs')
    times = [datetime.fromisoformat(item) for item in
             (native['started_utc'], binding['UTC'], det['UTC'], failure['UTC'], native['ended_utc'])]
    require(all(a <= b for a,b in zip(times, times[1:])), 'Legacy receipt chronology differs')
    require(binding.get('run_id') == base['run_id']
            and binding.get('config_sha256') == fix['base_configuration_sha256']
            and binding.get('source_manifest_sha256') == fix['legacy_source_manifest_sha256']
            and binding.get('dataset_sha256') == base['dataset_sha256']
            and binding.get('statistics_sha256') == base['statistics_sha256']
            and binding.get('raw_container_sha256') == base['train_container_sha256']
            and binding.get('identities_sha256') == base['identities_sha256']
            and binding.get('DEV_waveforms_read') is False, 'Legacy training binding differs')
    require(failure.get('run_id') == base['run_id'] and failure.get('status') == 'failed_or_bounded_stop_preserved'
            and failure.get('G_updates') == 22269 and failure.get('D_updates') == 48807
            and failure.get('current_arm') == 'full' and failure.get('completed_arms') == [det]
            and failure.get('DEV_waveforms_read') is False and failure.get('evaluation_complete') is False,
            'Legacy failure/update accounting differs')
    require('Invalid Window008 shape canonical charge' in failure.get('error', ''), 'Legacy failure mechanism differs')
    require(all(failure.get(key) == 0 for key in ('C2ST_updates', 'noise_updates', 'population_updates')),
            'Legacy update categories differ')
    require(det.get('arm') == 'deterministic' and det.get('G_updates') == 19000
            and det.get('D_updates') == {h:16000 for h in HYPOTHESES}
            and det.get('final_sha256') == fix['reference_inputs']['deterministic_final']['sha256'], 'Retained det completion differs')
    require(det.get('coverage', {}).get('source_unique') == 6593 and det.get('coverage', {}).get('real_unique') == 21047,
            'Retained det parent coverage differs')
    require(inputs.get('SOURCE_rows') == 6593 and inputs.get('REAL_rows') == 21047
            and inputs.get('effective_samples') == 256 and inputs.get('internal_carrier_samples') == 384
            and inputs.get('DEV_waveforms_read') is False and inputs.get('source_count_weights') == 'all_one'
            and inputs.get('new_noise_updates') == inputs.get('new_population_updates') == 0,
            'Legacy input coverage/role differs')
    return dict(native_exit_code=1, native_supervisor_pid=native['supervisor_pid'], native_child_pid=native['child_pid'],
        legacy_training_pid=binding['pid'], original_completed_G_updates=22269, original_completed_D_updates=48807,
        native_child_role='timeout_wrapper',
        legacy_trainer_PID_evidence='original_hash_bound_binding_and_chronology; no trainer_parent_PID_in_old_receipt',
        retained_arm_training_complete=True, old_job_native_success=False,
        discarded_full_joint_G_updates=269, discarded_full_joint_D_updates=807)


def validate_plan(base, fix):
    require(fix.get('schema') == 'n01/window008-fix01-config/v1' and fix.get('run_id') == RUN_ID
            and base.get('run_id') == 'window008_20261013', 'Fix01/base run identity differs')
    require(fix.get('model_implementation') == MODEL_IMPLEMENTATION, 'Unregistered fix01 model')
    require(fix.get('execution_arms') == ['full', 'simple'] and tuple(fix.get('comparison_arms', [])) == ARMS,
            'Fix01 arm plan differs')
    require(tuple(base['arms']) == ARMS and base['response']['mean_response_updates'] == 3000
            and base['response']['joint_stochastic_updates'] == 12000 and base['response']['low_learning_rate_updates'] == 4000
            and base['response']['updates_per_arm'] == 19000 and base['response']['total_G_updates'] == 57000
            and base['response']['total_D_updates'] == 144000, 'Base stage budgets differ')
    counts = dict(updates_per_valid_arm=19000, new_execution_G_updates=35000, new_execution_D_updates=96000,
        effective_comparison_G_updates=57000, effective_comparison_D_updates=144000,
        historical_actual_G_updates_after_success=57269, historical_actual_D_updates_after_success=144807,
        discarded_legacy_joint_G_updates=269, discarded_legacy_joint_D_updates=807)
    require(all(fix.get(key) == expected for key,expected in counts.items()), 'Fix01 update budgets differ')
    require(fix.get('resume') == dict(arm='full', reference='full_stage_03000', G_updates=3000,
        D_updates_per_hypothesis=0, stage='mean_response', next_G_update=3001,
        all_optimizer_sampler_stream_states_imported=True, first_update_audit=True, first_update_same_version_replay=True),
        'Fix01 resume policy differs')
    require(fix.get('fresh_simple') == dict(common_reference='common_initial_state', G_updates=0, D_updates_per_hypothesis=0),
            'Simple initialization policy differs')
    require(fix.get('retained_deterministic') == dict(final_reference='deterministic_final', G_updates=19000,
        D_updates_per_hypothesis=16000, no_new_updates=True, source_execution_native_exit_code=1, arm_training_complete=True),
        'Retained det policy differs')
    require(all(base.get(key) == fix.get(key) == 0 for key in ('noise_updates', 'population_updates'))
            and fix.get('C2ST_updates') == 0 and fix.get('DEV_waveforms_used_for_training') is False
            and fix.get('loss_and_seeds_changed') is False, 'Fix01 scientific role/update scope differs')
    require(base.get('engineering_fixture') is not True, 'Artificial recipe cannot enter the scientific driver')
    require(set(fix.get('reference_inputs', {})) == EXPECTED_REFERENCES, 'Legacy reference inventory differs')
    require(fix.get('effective_observation_samples') == 256 and fix.get('internal_carrier_samples') == 384,
            'Fix01 observation dimensions differ')
    for key in ('cpu_threads', 'gpu_memory_limit_GiB', 'job_tree_RSS_limit_GiB', 'output_limit_MiB', 'checkpoint_interval', 'status_interval'):
        require(fix.get(key) == base[key], 'Fix01 runtime bound differs: ' + key)
    require(fix.get('wall_limit_s') == 21600 and fix.get('one_active_scientific_job') is True, 'Fix01 wall/job bound differs')


def bind(args, root=ROOT):
    root = Path(root).resolve()
    base, fix, manifest = (read_json(path) for path in (args.config, args.recovery_config, args.source_manifest))
    validate_plan(base, fix)
    require(Path(args.config).resolve() == bound(root, fix['base_configuration'])
            and Path(args.recovery_config).resolve() == root / 'window008_fix01_config.json'
            and Path(args.source_manifest).resolve() == bound(root, fix['source_manifest']), 'Configuration/manifest path differs')
    require(sha256(args.config) == fix['base_configuration_sha256'], 'Original config bytes changed')
    legacy_path = bound(root, fix['legacy_source_manifest'])
    require(sha256(legacy_path) == fix['legacy_source_manifest_sha256'], 'Original source manifest changed')
    legacy_manifest = read_json(legacy_path)
    require(legacy_manifest.get('schema') == 'n01/window008/source_manifest/v1'
            and legacy_manifest.get('run_id') == base['run_id']
            and legacy_manifest.get('config_sha256') == fix['base_configuration_sha256']
            and legacy_manifest.get('before_scientific_model_forward') is True
            and len(legacy_manifest.get('files', {})) == 68, 'Original freeze identity differs')
    require(manifest.get('schema') == 'n01/window008-fix01-source/v1' and manifest.get('run_id') == RUN_ID
            and manifest.get('config_sha256') == sha256(args.recovery_config)
            and manifest.get('base_manifest_sha256') == fix['legacy_source_manifest_sha256']
            and manifest.get('before_scientific_model_forward') is True, 'Fix01 source freeze differs')
    verify_files(root, legacy_manifest['files']); verify_files(root, manifest['files'])
    mandatory = REQUIRED_SOURCES | {fix['protocol'], fix['base_configuration'], fix['legacy_source_manifest']}
    mandatory |= {item['path'] for item in fix['reference_inputs'].values()}
    require(mandatory <= set(manifest['files']), 'Fix01 freeze omits required code/config/reference inputs')
    modules = {path.relative_to(root).as_posix() for path in (root/'src/n01').glob('*.py')}
    expected = {name for name in set(legacy_manifest['files']) | set(manifest['files'])
                if name.startswith('src/n01/') and name.endswith('.py')}
    require(modules == expected, 'Runtime Python module inventory is not fully frozen')
    checks = [(args.dataset, bound(root, base['dataset']), base['dataset_sha256']),
              (args.statistics, bound(root, base['statistics']), base['statistics_sha256']),
              (args.identities, bound(root, 'data/PARENT_IDENTITIES.npz'), base['identities_sha256'])]
    for path, expected_path, digest in checks:
        require(Path(path).resolve() == expected_path and sha256(path) == digest, 'Prepared input path/hash differs: ' + str(path))
    require(str(Path(args.container).resolve()) in (base['remote_training_container'], base['local_training_container'])
            and sha256(args.container) == base['train_container_sha256'], 'Original TRAIN container differs')
    documents = {}
    for name, row in fix['reference_inputs'].items():
        path = bound(root, row['path'])
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], 'Legacy reference bytes differ: ' + name)
        if path.suffix == '.json':
            documents[name] = read_json(path)
    legacy_receipt = validate_legacy_receipts(base, fix, documents)
    return base, fix, manifest, documents, legacy_receipt


def load_bound_checkpoint(root, reference):
    path = bound(root, reference['path'])
    require(sha256(path) == reference['sha256'], 'Checkpoint changed before CPU load')
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(sha256(path) == reference['sha256'], 'Checkpoint changed during CPU load')
    return state


def validate_legacy_states(root, base, fix, statistics, documents):
    """CPU checkpoint/metadata validation only; no waveform or model forward."""
    references = fix['reference_inputs']
    common_file = load_bound_checkpoint(root, references['common_initial_state'])
    require(common_file.get('seed') == base['seeds']['initialization']
            and common_file.get('statistics_sha256') == base['statistics_sha256'], 'Common initializer metadata differs')
    common = common_file['model']; common_hash = state_digest(common)
    require(common_hash == documents['legacy_training_inputs']['common_initial_state_sha256'], 'Common initializer digest differs')
    full = load_bound_checkpoint(root, references['full_stage_03000'])
    det_mean = load_bound_checkpoint(root, references['deterministic_stage_03000'])
    initial = load_bound_checkpoint(root, references['full_initial'])
    det = load_bound_checkpoint(root, references['deterministic_final'])
    for state, arm, step, d, stage in ((full,'full',3000,0,'mean_response'),
            (det_mean,'deterministic',3000,0,'mean_response'), (initial,'full',0,0,'mean_response'),
            (det,'deterministic',19000,16000,'low_learning_rate')):
        require(state.get('schema') == 'response008/v1' and state.get('arm') == arm and state.get('config') == base
                and state.get('G_updates') == step and state.get('D_updates') == {h:d for h in HYPOTHESES}
                and state.get('stage') == stage and state.get('noise_updates') == 0
                and state.get('population_learning') is False, 'Legacy checkpoint identity/counters differ: ' + arm + '/' + str(step))
        require(state.get('statistics_sha256') == statistics_digest(statistics)
                and statistics_digest(state['statistics']) == statistics_digest(statistics)
                and state.get('common_initial_state_sha256') == common_hash, 'Legacy statistics/common initializer differ')
    assert_tree_equal(initial['model'], common, 'original_full_initial')
    require(not initial['g_optimizer']['state'] and all(not state['state'] for state in initial['d_optimizers'].values()),
            'Original initialization already has optimizer history')
    for key in ('initial_state_sha256', 'initial_D_sha256', 'noise005_state_sha256'):
        require(full[key] == initial[key], 'Full initialization reference differs: ' + key)
    require(len({state['noise005_state_sha256'] for state in (full, det_mean, initial, det)}) == 1,
            'Legacy frozen005 references differ')
    shared = det_mean['model']
    require(set(shared) <= set(full['model']), 'Mean-stage shared model keys differ')
    for name, value in shared.items():
        assert_tree_equal(value, full['model'][name], 'mean_shared/' + name)
    for key in ('discriminators', 'd_optimizers', 'source_sampler', 'real_sampler', 'streams'):
        assert_tree_equal(full[key], det_mean[key], 'mean_pair/' + key)
    for key in ('discriminators', 'd_optimizers'):
        assert_tree_equal(full[key], initial[key], 'mean_stage_no_D_update/' + key)
    full_model = N01Response008(statistics['model'], 'full')
    det_model = N01Response008(statistics['model'], 'deterministic')
    full_model.load_state_dict(full['model'], strict=True); det_model.load_state_dict(det_mean['model'], strict=True)
    full_slots = optimizer_named_state(full_model, full['g_optimizer'])
    det_slots = optimizer_named_state(det_model, det_mean['g_optimizer'])
    for name, value in det_slots.items():
        assert_tree_equal(value, full_slots[name], 'mean_shared_AdamW/' + name)
    full_group = {k:v for k,v in full['g_optimizer']['param_groups'][0].items() if k != 'params'}
    det_group = {k:v for k,v in det_mean['g_optimizer']['param_groups'][0].items() if k != 'params'}
    assert_tree_equal(full_group, det_group, 'mean_shared_AdamW_hyperparameters')
    for state, steps in ((full,3000), (det,19000)):
        validate_sampler(state['source_sampler'],6593,steps*base['distribution_microbatches']*base['source_parents_per_microbatch'])
        validate_sampler(state['real_sampler'],21047,steps*base['distribution_microbatches']*base['real_rows_per_microbatch'])
        validate_streams(state['streams'])
    require(det['last_metrics'].get('G_updates') == 19000 and det['last_metrics'].get('D_updates') == {h:16000 for h in HYPOTHESES}
            and det['last_metrics'].get('coverage') == documents['deterministic_training_complete']['coverage'],
            'Retained det checkpoint/completion coverage differs')
    # Reapply the exact historical common-prefix initializer rules to both new arms.
    expected_initial = {}
    for arm in ('full', 'simple'):
        model = make_signal(statistics['model'], arm, common, base['seeds']['initialization'], 'cpu')
        expected_initial[arm] = state_digest(model.state_dict())
        if arm == 'full':
            require(expected_initial[arm] == initial['initial_state_sha256'], 'Original full initial state digest differs')
    receipt = dict(status='passed_CPU_checkpoint_metadata_only', common_initial_state_sha256=common_hash,
        mean_shared_model_tensors=len(shared), mean_shared_optimizer_parameters=sum(value is not None for value in det_slots.values()),
        mean_shared_D_samplers_streams_exact=True, noise005_state_sha256=full['noise005_state_sha256'],
        initial_model_state_sha256=expected_initial, det_final_model_state_sha256=state_digest(det['model']),
        CPU_scientific_model_forwards=0, optimizer_updates=0, DEV_waveforms_decoded=False)
    return common, receipt


def update_accounting(completed, engine=None, *, legacy_validated=True):
    rows = [row for row in completed if row['arm'] in ('full','simple') and (engine is None or row['arm'] != engine.arm)]
    new_g = sum(row['new_G_updates'] for row in rows) + (engine.new_G_updates if engine is not None else 0)
    new_d = sum(sum(row['new_D_updates'].values()) for row in rows) + (sum(engine.new_D_updates.values()) if engine is not None else 0)
    return dict(new_execution_G_updates=new_g, new_execution_D_updates=new_d,
        effective_comparison_G_updates=22000+new_g if legacy_validated else None,
        effective_comparison_D_updates=48000+new_d if legacy_validated else None,
        historical_actual_G_updates=22269+new_g if legacy_validated else None,
        historical_actual_D_updates=48807+new_d if legacy_validated else None,
        inherited_effective_G_updates=22000 if legacy_validated else None,
        inherited_effective_D_updates=48000 if legacy_validated else None,
        discarded_legacy_joint_G_updates=269 if legacy_validated else None,
        discarded_legacy_joint_D_updates=807 if legacy_validated else None,
        legacy_references_validated=legacy_validated)


def retained_endpoint(fix, documents, validation):
    previous = documents['deterministic_training_complete']
    return dict(arm='deterministic', origin='retained_original_endpoint', G_updates=19000,
        D_updates={h:16000 for h in HYPOTHESES}, inherited_G_updates=19000, new_G_updates=0,
        new_D_updates={h:0 for h in HYPOTHESES}, coverage=copy.deepcopy(previous['coverage']),
        checkpoint_schema='response008/v1', stage='low_learning_rate', model_implementation='n01.model008.N01Response008',
        endpoint_path=fix['reference_inputs']['deterministic_final']['path'],
        final_sha256=fix['reference_inputs']['deterministic_final']['sha256'],
        state_sha256=validation['det_final_model_state_sha256'],
        legacy_arm_completion_sha256=fix['reference_inputs']['deterministic_training_complete']['sha256'],
        source_job_native_exit_code=1, arm_training_complete=True,
        common_initial_state_sha256=validation['common_initial_state_sha256'],
        noise005_state_sha256=validation['noise005_state_sha256'])


def validate_training_data(data):
    require(data['source'].shape == (6593,256) and data['real'].shape == (21047,256), 'All-TRAIN effective dimensions changed')
    require(data['source_carrier384'].shape == (6593,384) and data['real_baseline100'].shape == (21047,100),
            'TRAIN carrier/baseline dimensions changed')
    require(set(np.unique(data['source_role'])) == {'TRAIN'} and set(np.unique(data['real_role'])) == {'TRAIN'},
            'Non-TRAIN waveform role')
    require(len(np.unique(data['source_ids'])) == 6593 and len(np.unique(data['real_ids'])) == 21047,
            'Parent identities missing/duplicated')
    require(not np.intersect1d(data['source_ids'], data['source_val_ids']).size
            and not np.intersect1d(data['real_ids'], data['real_val_original_index']).size, 'Parent role split overlap')


def save_failure(out, error, completed, engine, *, began, legacy_validated):
    """Preserve actual completed counters and any mid-step forensic state."""
    out = Path(out)
    if engine is not None:
        try:
            engine.checkpoint(out/'FAILURE_CHECKPOINT.pt',purpose='failure')
        except Exception as save_error:
            write_json(out/'CHECKPOINT_SAVE_ERROR.json',dict(error=repr(save_error)))
        try:
            atomic_torch_save(engine.last_context,out/'FAILURE_CONTEXT.pt')
        except Exception as save_error:
            write_json(out/'CONTEXT_SAVE_ERROR.json',dict(error=repr(save_error)))
    result = dict(schema='n01/window008-fix01-failure/v1',UTC=utc(),status='fix01_failed_or_bounded_stop_preserved',
        run_id=RUN_ID,error=repr(error),traceback=traceback.format_exc(),
        **update_accounting(completed,engine,legacy_validated=legacy_validated),
        completed_arms=completed,current_arm=engine.arm if engine is not None else None,
        current_G_updates=engine.update if engine is not None else None,
        current_D_updates=dict(engine.d_updates) if engine is not None else None,
        checkpoint_may_be_mid_step=engine is not None,automatic_resume_allowed=False,
        DEV_waveforms_read=False,C2ST_updates=0,noise_updates=0,population_updates=0,
        elapsed_seconds=time.monotonic()-began,scientific_status='response_match_not_established',evaluation_complete=False)
    write_json(out/'FAILURE.json',result); write_json(out/'status.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','recovery-config','source-manifest','dataset','statistics','container','identities','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    require(not (os.name == 'nt' and out.drive.upper() not in ('E:','D:')), 'Windows output must stay on E/D')
    if out.exists():
        raise FileExistsError('Fresh fix01 output required; failed or prior attempts are never resumed')
    out.mkdir(parents=True, exist_ok=False)
    engine = None; completed = []; legacy_validated = False; began = time.monotonic(); fix = None
    def stop(signum, frame):
        raise TimeoutError('Owned fix01 process received termination signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    try:
        base, fix, manifest, documents, legacy_chain = bind(args)
        require(os.name == 'posix' and str(ROOT.resolve()) == fix['remote_root']
                and out == ROOT/'runs/science', 'Science must run in the registered fresh Linux root')
        statistics = read_json(args.statistics)
        require(statistics.get('prepared_dataset_sha256') == base['dataset_sha256']
                and statistics.get('statistics_split') == 'TRAIN', 'Statistics binding/role differs')
        common, cpu_validation = validate_legacy_states(ROOT, base, fix, statistics, documents)
        legacy_validated = True
        identity = dict(run_id=RUN_ID, config_sha256=sha256(args.recovery_config), source_manifest_sha256=sha256(args.source_manifest),
            base_configuration_sha256=sha256(args.config), base_source_manifest_sha256=fix['legacy_source_manifest_sha256'])
        write_json(out/'BINDING.json', dict(UTC=utc(), run_id=RUN_ID, pid=os.getpid(), parent_pid=os.getppid(),
            recovery=identity, dataset_sha256=sha256(args.dataset), statistics_sha256=sha256(args.statistics),
            raw_container_sha256=base['train_container_sha256'], identities_sha256=base['identities_sha256'],
            before_prepared_numeric_decode=True, DEV_waveforms_read=False, legacy_native_chain=legacy_chain))
        write_json(out/'LEGACY_VALIDATION.json', dict(UTC=utc(), **cpu_validation, native_chain=legacy_chain,
            reference_files={name:row['sha256'] for name,row in fix['reference_inputs'].items()}, recovery=identity))
        completed.append(retained_endpoint(fix, documents, cpu_validation))
        device = torch.device(args.device)
        runtime = copy.deepcopy(base)
        for key in ('cpu_threads','gpu_memory_limit_GiB','job_tree_RSS_limit_GiB','wall_limit_s','output_limit_MiB'):
            runtime[key] = fix[key]
        configure_runtime(runtime, device)
        with np.load(args.dataset, allow_pickle=False) as archive:
            require(not ({'source_val','target_val','hval'} & set(archive.files)), 'Prepared input contains DEV values')
            data = {name:archive[name].copy() for name in archive.files}
        validate_training_data(data)
        noise_path = bound(ROOT, base['noise_reference']['path'])
        require(sha256(noise_path) == base['noise_reference']['sha256'], 'Frozen005 bytes changed')
        noise = torch.load(noise_path, map_location='cpu', weights_only=False)
        require(noise.get('update') == 5000 and noise.get('schema_version') == 5
                and noise.get('model_type') == 'ObservedBaselineNoiseV3' and noise.get('engineering') is False,
                'Wrong frozen005 endpoint')
        write_json(out/'TRAINING_INPUTS.json', dict(UTC=utc(), SOURCE_rows=6593, REAL_rows=21047,
            effective_samples=256, internal_carrier_samples=384, source_ADC=False,
            common_initial_state_sha256=state_digest(common), common_initial_state_file_sha256=fix['reference_inputs']['common_initial_state']['sha256'],
            source_count_weights='all_one', original_source_count_weights_preserved=True, original_physical_weights='unknown',
            DEV_waveforms_read=False, noise_updates=0, population_updates=0, recovery=identity))
        for arm in fix['execution_arms']:
            folder = out/'train'/arm
            folder.mkdir(parents=True, exist_ok=False)
            engine = Response008Fix01Engine(data, statistics, base, noise, common, arm=arm, recovery_identity=identity, device=device)
            require(engine.noise_state_digest == cpu_validation['noise005_state_sha256']
                    and engine.initial_state_digest == cpu_validation['initial_model_state_sha256'][arm], 'Engine reference construction differs')
            if arm == 'full':
                reference = fix['reference_inputs']['full_stage_03000']
                validation = engine.import_legacy_full_stage3000(bound(ROOT,reference['path']), expected_sha256=reference['sha256'])
                write_json(out/'RESUME_VALIDATION.json', validation)
                engine.checkpoint(folder/'restored_start.pt', purpose='restored_start')
            else:
                engine.set_stage('mean_response')
                engine.checkpoint(folder/'initial.pt', purpose='initial')
            write_json(folder/'INITIALIZATION.json', dict(UTC=utc(), arm=arm, recovery=identity,
                lineage=engine.lineage, model_replacement=engine.model_replacement_receipt,
                G_updates=engine.update, D_updates=dict(engine.d_updates), new_G_updates=0))
            while engine.update < base['response']['updates_per_arm']:
                require(time.monotonic()-began < fix['wall_limit_s']-30, 'Registered fix01 wall bound reached before next update')
                stage = response_stage(engine.update, base)
                changed = stage != engine.stage
                if changed:
                    engine.set_stage(stage)
                update = engine.update + 1
                first_new = engine.new_G_updates == 0
                audit = first_new or update <= 2 or changed or update % base['engineering']['gradient_audit_interval'] == 0
                replay = first_new or update <= base['engineering']['waveform_replay_first_steps'] or changed
                metrics = engine.step(audit=audit, verify_replay=replay)
                with (folder/'training.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps(metrics,allow_nan=False)+'\n')
                if audit or engine.update % fix['status_interval'] == 0:
                    status = dict(status='training_window008_fix01',run_id=RUN_ID,pid=os.getpid(),
                        elapsed_seconds=time.monotonic()-began,DEV_waveforms_read=False,
                        target_new_G_updates=35000,target_new_D_updates=96000,
                        **update_accounting(completed,engine), **metrics)
                    write_json(out/'status.json', status)
                    print(json.dumps({key:status[key] for key in ('utc','status','arm','stage','G_updates',
                        'new_execution_G_updates','new_execution_D_updates','effective_comparison_G_updates',
                        'historical_actual_G_updates','weighted_total','update_seconds','coverage')}),flush=True)
                if engine.update % fix['checkpoint_interval'] == 0:
                    engine.checkpoint(folder/'latest.pt', purpose='periodic', allow_overwrite=True)
                if engine.update in (3000,15000,19000):
                    path = folder/('final.pt' if engine.update == 19000 else f'stage_{engine.update:05d}.pt')
                    engine.checkpoint(path, purpose='final' if engine.update == 19000 else 'stage_endpoint')
                if engine.update % 1000 == 0:
                    require(sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) <= fix['output_limit_MiB']*1024**2,
                            'Registered fix01 output bound exceeded')
            require(engine.update == 19000 and engine.d_updates == {h:16000 for h in HYPOTHESES}, 'Endpoint update accounting differs')
            coverage = engine.last_metrics['coverage']
            require(coverage['source_unique'] == 6593 and coverage['real_unique'] == 21047, 'All-parent coverage incomplete')
            done = dict(UTC=utc(),arm=arm,origin=engine.lineage['origin'],G_updates=engine.update,D_updates=dict(engine.d_updates),
                inherited_G_updates=engine.imported_G_updates,new_G_updates=engine.new_G_updates,new_D_updates=engine.new_D_updates,
                coverage=coverage,checkpoint_schema='response008_fix01/v1',stage=engine.stage,
                model_implementation=MODEL_IMPLEMENTATION,endpoint_path=(folder/'final.pt').relative_to(ROOT).as_posix(),
                final_sha256=sha256(folder/'final.pt'),state_sha256=state_digest(engine.model.state_dict()),lineage=engine.lineage,
                common_initial_state_sha256=engine.common_state_digest,noise005_state_sha256=engine.noise_state_digest,recovery=identity)
            completed.append(done)
            write_json(folder/'TRAINING_COMPLETE.json',done)
            del engine; engine = None; gc.collect(); torch.cuda.empty_cache()
        accounting = update_accounting(completed)
        require(accounting['new_execution_G_updates'] == 35000 and accounting['new_execution_D_updates'] == 96000
                and accounting['effective_comparison_G_updates'] == 57000 and accounting['effective_comparison_D_updates'] == 144000
                and accounting['historical_actual_G_updates'] == 57269 and accounting['historical_actual_D_updates'] == 144807,
                'Final recovery accounting differs')
        # Recheck immutable sources and references before signing the endpoints.
        bind(args)
        files = {path.relative_to(out).as_posix():sha256(path) for path in sorted(out.rglob('*'))
                 if path.is_file() and path != out/'status.json' and path != out/'TRAINING_COMPLETE.json'}
        result = dict(schema='n01/window008-fix01-training-complete/v1', UTC=utc(),
            status='fixed_recovered_endpoints_complete_pending_native_exit_and_evaluation',run_id=RUN_ID,
            base_run_id=base['run_id'],recovery=identity,**accounting,arms=completed,files=files,
            endpoint_map={row['arm']:dict(path=row['endpoint_path'],sha256=row['final_sha256'],schema=row['checkpoint_schema'],
                model_implementation=row['model_implementation'],origin=row['origin']) for row in completed},
            legacy_native_chain=legacy_chain,legacy_native_exit_sha256=fix['reference_inputs']['legacy_native_exit']['sha256'],
            legacy_failure_sha256=fix['reference_inputs']['legacy_failure']['sha256'],
            resume_validation_sha256=sha256(out/'RESUME_VALIDATION.json'),
            awaiting_own_supervisor_native_exit=True,evaluation_complete=False,DEV_waveforms_read=False,
            C2ST_updates=0,noise_updates=0,population_updates=0,elapsed_seconds=time.monotonic()-began,
            scientific_status='response_match_not_established')
        write_json(out/'TRAINING_COMPLETE.json',result); write_json(out/'status.json',result)
    except BaseException as error:
        save_failure(out,error,completed,engine,began=began,legacy_validated=legacy_validated)
        raise


if __name__ == '__main__':
    main()
