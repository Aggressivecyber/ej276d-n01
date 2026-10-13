"""CPU metadata and artificial rejection QA; waveform decode/forward/steps blocked."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from n01 import evaluate_window008_resume01 as adapter
from n01 import evaluate_window008 as numeric
from n01.data import sha256, write_json
from n01.response008 import ARMS, HYPOTHESES, make_signal
from n01.response008_fix01 import parameter_signature
from n01.train import utc


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _mutate(document, keys, value):
    result = copy.deepcopy(document)
    parent = result
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value
    return result


def _source_fixture(out, cfg, resume):
    """Labelled metadata-only source root; no module from it is executed."""
    root = out / 'artificial_source_root'; root.mkdir()
    for relative, data in (
        ('src/n01/evaluate_window008.py', (ROOT / 'src/n01/evaluate_window008.py').read_bytes()),
        ('src/n01/evaluate_window008_resume01.py', b'# artificial inventory entry only\n'),
        ('scripts/check_window008_resume01_evaluation.py', b'# artificial inventory entry only\n'),
        ('docs/WINDOW008_RESUME01_EVALUATION.md', b'artificial metadata fixture\n'),
        ('window008_config.json', (ROOT / 'window008_config.json').read_bytes()),
        ('window008_resume01_config.json', (ROOT / 'window008_resume01_config.json').read_bytes()),
    ):
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for name in ('legacy_source_manifest', 'fix01_source_manifest', 'source_manifest'):
        _write(root / resume[name], dict(artificial_parser_fixture=True,
            files={'src/n01/evaluate_window008.py': adapter.NUMERIC_CORE_SHA256}))
    policy = dict(schema='n01/window008-resume01-evaluation-policy/v1', run_id=adapter.RUN_ID,
        artificial_parser_fixture=True, training_source_manifest_sha256=sha256(root / resume['source_manifest']),
        evaluation_module='n01.evaluate_window008_resume01', training_run='runs/science', job_root='runs/evaluation_jobs',
        jobs=['primary', 'replay01'], optimizer_updates=0, C2ST_updates=0, scientific_status='response_match_not_established',
        resource_limits=dict(GPU_GiB=10, job_tree_RSS_GiB=8, output_MiB=3072, wall_seconds=21600,
                             poll_seconds=1, TERM_grace_seconds=20),
        replay=dict(same_training_endpoints=True, same_training_config_and_seeds=True, draws_per_parent=4, chunk_rows=128,
            additional_unique_draws=0, compare_all_NPZ_members=True, compare_all_chunk_records=True,
            compare_distances_and_parent_bootstrap=True, array_comparison='exact_dtype_shape_values_and_member_sha256',
            overwrite_allowed=False))
    _write(root / adapter.EVALUATION_POLICY, policy)
    additions = ['src/n01/evaluate_window008_resume01.py', adapter.EVALUATION_POLICY,
                 'scripts/check_window008_resume01_evaluation.py', 'docs/WINDOW008_RESUME01_EVALUATION.md']
    manifest = dict(schema='n01/window008-resume01-evaluation-source/v1', run_id=adapter.RUN_ID,
        artificial_parser_fixture=True, before_evaluation_numeric_read=True, optimizer_updates=0,
        training_source_manifest_sha256=sha256(root / resume['source_manifest']),
        training_config_sha256=sha256(root / 'window008_resume01_config.json'),
        base_config_sha256=sha256(root / 'window008_config.json'), policy_sha256=sha256(root / adapter.EVALUATION_POLICY),
        module_inventory=['src/n01/evaluate_window008.py', 'src/n01/evaluate_window008_resume01.py'],
        files={name: sha256(root / name) for name in additions})
    _write(root / adapter.EVALUATION_MANIFEST, manifest)
    return root, policy, manifest


def run(output, science_run=None):
    output = Path(output).resolve()
    if os.name == 'nt' and output.drive.upper() not in ('E:', 'D:'):
        raise ValueError('CPU evaluation QA output must stay on E/D')
    if output.exists():
        raise FileExistsError('Fresh CPU evaluation QA output required')
    output.mkdir(parents=True)
    science = Path(science_run or ROOT / 'runs/window008_resume01_20261013/runs/science').resolve()
    checks, loaded = [], []
    began = time.monotonic()

    def passed(name):
        checks.append(dict(name=name, passed=True))

    def reject(name, action, expected=None):
        try:
            action()
        except (ValueError, FileExistsError, FileNotFoundError, KeyError, TypeError) as error:
            if expected and expected not in str(error):
                raise AssertionError(name + ': unexpected rejection ' + str(error)) from error
            checks.append(dict(name=name, passed=True, rejection=str(error)))
        else:
            raise AssertionError(name + ': accepted malformed artificial metadata')

    original_sha = sha256(ROOT / 'src/n01/evaluate_window008.py')
    require = adapter.require
    require(original_sha == adapter.NUMERIC_CORE_SHA256, 'Original evaluator is not frozen')
    cfg, resume = (adapter._read_json(ROOT / name) for name in ('window008_config.json', 'window008_resume01_config.json'))
    raw_load = torch.load
    def tracked_load(path, *args, **kwargs):
        require('failure_checkpoint' not in str(path).lower() and 'failure_context' not in str(path).lower(),
                'Forensic failure checkpoint load is forbidden')
        require(kwargs.get('map_location') == 'cpu', 'QA checkpoint loads must stay on CPU')
        loaded.append(str(Path(path).resolve()))
        return raw_load(path, *args, **kwargs)

    forbidden = AssertionError('CPU QA forbids waveform NPZ decode, module forward and optimizer construction/step')
    try:
        with mock.patch('numpy.load', side_effect=forbidden), \
             mock.patch.object(torch.nn.Module, '_call_impl', side_effect=forbidden), \
             mock.patch.object(torch.optim.AdamW, '__init__', side_effect=forbidden), \
             mock.patch.object(torch.optim.AdamW, 'step', side_effect=forbidden), \
             mock.patch('torch.load', side_effect=tracked_load):
            completion = adapter.training_completion_gate(science, cfg, resume)
            passed('actual_collected_three_generation_completion_gate')
            require(completion['legacy_native_chain']['original_native_chain']['native_exit_code'] == 1
                    and completion['legacy_native_chain']['fix01_native_exit_code'] == 1
                    and completion['native_exit_code'] == 0, 'Three native exit roles differ')
            passed('original_numeric_native1_fix01_user_stop_native1_resume01_native0_preserved')
            models = adapter.load_endpoints(completion, cfg, adapter._read_json(ROOT / cfg['statistics']), 'cpu')
            require(type(models['deterministic']).__name__ == 'N01Response008'
                    and type(models['full']).__name__ == type(models['simple']).__name__ == 'N01Response008Fix01'
                    and all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in models.values()),
                    'Mixed evaluation models are not strictly frozen')
            passed('actual_mixed_cpu_loader_original_det_and_fix01_models_strict_frozen')
            require(all(value in {16000, 19000} for arm in ARMS for value in completion['optimizer_named_steps'][arm].values())
                    and set(completion['optimizer_named_steps']['deterministic'].values()) == {19000}
                    and set(completion['optimizer_named_steps']['full'].values()) == {16000, 19000}, 'Final named AdamW step roles differ')
            passed('actual_complete_AdamW_slots_and_mixed_phase_steps_verified_without_optimizer')
            passed('actual_import_all_native_trees_moments_samplers_streams_bitwise_equal')
            require(completion['accounting']['new_execution_G_updates'] == 32000
                    and completion['accounting']['effective_comparison_G_updates'] == 57000
                    and completion['accounting']['historical_actual_G_updates'] == 57667,
                    'Actual update ledger roles differ')
            passed('new_effective_historical_update_ledgers_distinct')

            native_path = science.parent / 'supervisor/exit_receipt.json'
            launch_path = science.parent / 'supervisor/launch.json'
            native, launch = adapter._read_json(native_path), adapter._read_json(launch_path)
            binding, done = adapter._read_json(science / 'BINDING.json'), adapter._read_json(science / 'TRAINING_COMPLETE.json')
            raw_read = adapter._read_json
            def native_case(n=native, l=launch, b=binding, c=done):
                def read(path):
                    if Path(path) == native_path: return copy.deepcopy(n)
                    if Path(path) == launch_path: return copy.deepcopy(l)
                    return raw_read(path)
                with mock.patch.object(adapter, '_read_json', side_effect=read):
                    return adapter._native_completion(science, cfg, resume, c, b, completion['recovery'])
            for key, value in (('exit_code', 1), ('exit_code', False), ('reason', 'wall_limit'),
                               ('supervisor_pid', True), ('child_pid', native['child_pid'] + 1),
                               ('started_utc', '2026-10-13T20:00:00+00:00'), ('ended_utc', '2026-10-13T10:00:00+00:00')):
                reject('reject_native_' + key + '_' + str(value), lambda key=key, value=value: native_case(n=_mutate(native, [key], value)))
            for key, value in (('pid', True), ('parent_pid', binding['parent_pid'] + 1), ('before_prepared_numeric_decode', False),
                               ('DEV_waveforms_read', True), ('dataset_sha256', 'f' * 64), ('statistics_sha256', 'f' * 64),
                               ('identities_sha256', 'f' * 64), ('raw_container_sha256', 'f' * 64), ('new_execution_G_updates', 1)):
                reject('reject_binding_' + key, lambda key=key, value=value: native_case(b=_mutate(binding, [key], value)))
            for index, value in ((3, '28800s'), (7, 'n01.train_window008'), (7, 'n01.train_window008_fix01'),
                                 (launch['command'].index('--out') + 1, cfg['remote_root'] + '/runs/science'),
                                 (launch['command'].index('--device') + 1, 'cpu'),
                                 (launch['command'].index('--resume-config') + 1, '/tmp/other.json')):
                bad = copy.deepcopy(launch); bad['command'][index] = value
                reject('reject_launch_' + str(index) + '_' + value, lambda bad=bad: native_case(l=bad))
            bad = copy.deepcopy(launch); bad['command'] += ['--device', 'cuda:0']
            reject('reject_duplicate_command_flag', lambda: native_case(l=bad))

            sources = adapter._training_sources(cfg, resume, ROOT)
            def complete_case(bad):
                def read(path):
                    return copy.deepcopy(bad) if Path(path) == science / 'TRAINING_COMPLETE.json' else raw_read(path)
                with mock.patch.object(adapter, '_training_sources', return_value=copy.deepcopy(sources)), \
                     mock.patch.object(adapter, '_read_json', side_effect=read):
                    return adapter.training_completion_gate(science, cfg, resume)
            for key, value in (('schema', 'n01/window008-training-complete/v1'), ('native_training_success', True),
                               ('awaiting_own_supervisor_native_exit', False), ('run_id', cfg['run_id']), ('evaluation_complete', True),
                               ('DEV_waveforms_read', True), ('noise_updates', 1), ('C2ST_updates', 1), ('population_updates', 1),
                               ('new_execution_G_updates', 32001), ('effective_comparison_D_updates', 144001),
                               ('historical_actual_G_updates', 57668), ('discarded_user_stop_G_updates', 0)):
                reject('reject_completion_' + key, lambda key=key, value=value: complete_case(_mutate(done, [key], value)))
            for arm, key, value in (('full', 'G_updates', 18999), ('simple', 'checkpoint_schema', 'response008_fix01/v1'),
                                    ('full', 'model_implementation', 'n01.model008.N01Response008'),
                                    ('deterministic', 'new_G_updates', 1)):
                index = list(ARMS).index(arm)
                reject('reject_arm_' + arm + '_' + key,
                    lambda index=index, key=key, value=value: complete_case(_mutate(done, ['arms', index, key], value)))
            reject('reject_missing_complete_file', lambda: complete_case(_mutate(done, ['files'], {k:v for k,v in done['files'].items() if k != 'RESUME_VALIDATION.json'})))
            reject('reject_extra_complete_file', lambda: complete_case(_mutate(done, ['files'], dict(done['files'], unexpected='f' * 64))))

            statistics = raw_read(ROOT / cfg['statistics'])
            initial_ref = resume['reference_inputs']['full_initial']
            initial = raw_load(ROOT / initial_ref['path'], map_location='cpu', weights_only=False)
            state = raw_load(completion['endpoint_paths']['full'], map_location='cpu', weights_only=False)
            legacy = raw_read(science / 'LEGACY_VALIDATION.json')
            args = ('full', cfg, statistics, completion['recovery'], adapter.fix01_identity(resume),
                completion['common_initial_state_sha256'], completion['noise_state_sha256'],
                legacy['initial_model_state_sha256']['full'], initial['initial_D_sha256'], adapter._expected_lineage('full', cfg, resume), initial)
            for key, value in (('schema', 'response008_fix01/v1'), ('G_updates', 19001), ('stage', 'joint_stochastic'),
                               ('common_initial_state_sha256', 'f' * 64), ('noise005_state_sha256', 'f' * 64),
                               ('initial_state_sha256', 'f' * 64), ('checkpoint_purpose', 'failure'), ('engineering_fixture', True),
                               ('failure_may_be_mid_step', True), ('complete_step_boundary', False), ('automatic_resume_allowed', True),
                               ('model_implementation', 'n01.model008.N01Response008'), ('new_G_updates', 19000)):
                reject('reject_checkpoint_' + key,
                    lambda key=key, value=value: adapter._validate_checkpoint(dict(state, **{key:value}), *args))
            bad = dict(state); bad['model'] = dict(state['model']); bad['model'].pop(next(iter(bad['model'])))
            reject('reject_missing_model_tensor', lambda: adapter._validate_checkpoint(bad, *args))
            for field, value in (('step', torch.tensor(19001., dtype=torch.float32)),
                                 ('step', torch.tensor(19000., dtype=torch.float64)),
                                 ('exp_avg_sq', -torch.ones_like(state['g_optimizer']['state'][0]['exp_avg_sq'])),
                                 ('exp_avg', torch.full_like(state['g_optimizer']['state'][0]['exp_avg'], float('nan')))):
                bad = dict(state); bad['g_optimizer'] = copy.deepcopy(state['g_optimizer'])
                bad['g_optimizer']['state'][0][field] = value
                reject('reject_AdamW_' + field + '_' + str(value.dtype), lambda bad=bad: adapter._validate_checkpoint(bad, *args))
            bad = dict(state); bad['g_optimizer'] = copy.deepcopy(state['g_optimizer']); bad['g_optimizer']['state'].pop(0)
            reject('reject_missing_AdamW_named_history', lambda: adapter._validate_checkpoint(bad, *args))
            bad = dict(state); bad['g_optimizer'] = copy.deepcopy(state['g_optimizer']); bad['g_optimizer']['param_groups'][0]['lr'] = cfg['loss']['G_lr']
            reject('reject_wrong_final_AdamW_learning_rate', lambda: adapter._validate_checkpoint(bad, *args))
            bad = dict(state); bad['source_sampler'] = copy.deepcopy(state['source_sampler']); bad['source_sampler']['position'] += 1
            reject('reject_sampler_boundary_changed', lambda: adapter._validate_checkpoint(bad, *args))
            bad = dict(state); bad['streams'] = dict(state['streams']); bad['streams'].pop('extension')
            reject('reject_missing_training_random_stream', lambda: adapter._validate_checkpoint(bad, *args))

            fixture, policy, manifest = _source_fixture(output, cfg, resume)
            source_path, policy_path = fixture / adapter.EVALUATION_MANIFEST, fixture / adapter.EVALUATION_POLICY
            adapter._evaluation_source_gate(fixture, cfg, resume, source_path, policy_path)
            passed('artificial_independent_additive_source_policy_freeze_accepted')
            def source_case(m=manifest, p=policy):
                def read(path):
                    if Path(path) == source_path: return copy.deepcopy(m)
                    if Path(path) == policy_path: return copy.deepcopy(p)
                    return raw_read(path)
                with mock.patch.object(adapter, '_read_json', side_effect=read):
                    return adapter._evaluation_source_gate(fixture, cfg, resume, source_path, policy_path)
            for key, value in (('schema', 'n01/window008-evaluation-source/v1'), ('before_evaluation_numeric_read', False),
                               ('training_source_manifest_sha256', 'f' * 64), ('policy_sha256', 'f' * 64),
                               ('training_config_sha256', 'f' * 64), ('base_config_sha256', 'f' * 64),
                               ('module_inventory', ['src/n01/evaluate_window008.py']), ('optimizer_updates', 1)):
                reject('reject_eval_freeze_' + key, lambda key=key, value=value: source_case(m=_mutate(manifest, [key], value)))
            for keys, value in ((['schema'], 'n01/window008-evaluation-policy/v1'), (['evaluation_module'], 'n01.evaluate_window008'),
                                (['jobs'], ['primary']), (['optimizer_updates'], 1), (['C2ST_updates'], 1),
                                (['resource_limits', 'wall_seconds'], 0), (['resource_limits', 'wall_seconds'], 28800),
                                (['resource_limits', 'GPU_GiB'], 20),
                                (['replay', 'draws_per_parent'], 5), (['replay', 'chunk_rows'], 64),
                                (['replay', 'additional_unique_draws'], 1), (['replay', 'overwrite_allowed'], True),
                                (['replay', 'array_comparison'], 'allclose')):
                reject('reject_eval_policy_' + '_'.join(keys), lambda keys=keys, value=value: source_case(p=_mutate(policy, keys, value)))
            reject('reject_eval_manifest_outside_root', lambda: adapter._evaluation_source_gate(fixture, cfg, resume, output / adapter.EVALUATION_MANIFEST, policy_path))
            extra = fixture / 'src/n01/unfrozen.py'; extra.write_text('# artificial extra module\n', encoding='utf8')
            reject('reject_unfrozen_runtime_module', lambda: source_case())
            # The artificial extra module is retained as rejection evidence.

            reject('reject_duplicate_JSON_key', lambda: adapter._json_text('{"key":1,"key":2}'))
            reject('reject_nonfinite_JSON_constant', lambda: adapter._json_text('{"value":NaN}'))
            for value in ('../escape.pt', '/absolute.pt', 'C:/escape.pt', 'train//full/final.pt', 'train/./full/final.pt', 'train\\full\\final.pt'):
                reject('reject_unsafe_relative_path_' + value, lambda value=value: adapter._bound(output, value))
            for name in ('generate_split', 'parent_bootstrap', 'distance_receipt', '_fixed_settings', '_validate_split',
                         '_load_dev', '_verify_frozen', 'balanced_real_domains', 'save_json', 'save_npz'):
                require(getattr(adapter, name) is getattr(numeric, name), 'Frozen numeric function was replaced: ' + name)
                passed('same_frozen_numeric_function_' + name)
            # Artificial values exercise retention of nonpositive Q and matched
            # disjoint groups without a waveform-container read or neural call.
            artificial = np.zeros((3, 256), dtype=np.float64); artificial[1] = -1.; artificial[2] = 1.
            observed = numeric.readouts256(artificial, 1e-8)
            require(len(observed['Q']) == 3 and observed['Q_nonpositive'].tolist() == [True, True, False]
                    and observed['PSD_undefined'].tolist() == [True, False, False], 'Nonpositive-Q observation rows were removed')
            passed('artificial_nonpositive_Q_zero_Q_rows_retained')
            groups = np.repeat(np.arange(7), 20)
            a, b = adapter.balanced_real_domains(groups, 32, cfg['seeds']['real_DEV'])
            require(len(a) == len(b) == 32 and not np.intersect1d(a, b).size, 'Artificial REAL A/B is not disjoint')
            passed('artificial_balanced_REAL_A_B_disjoint_selection')
            require(sha256(ROOT / 'src/n01/evaluate_window008.py') == original_sha, 'Original numerical core was modified')
            passed('original_numeric_core_byte_hash_unchanged')
            _write(output / 'ACTUAL_COMPLETION_GATE.json', completion)
            receipt = dict(schema='n01/window008-resume01-evaluation-cpu-qa/v1', UTC=utc(), status='passed_cpu_metadata_artificial_checks',
                checks=len(checks), passed=sum(row['passed'] for row in checks), details=checks,
                actual_training_gate=str(output / 'ACTUAL_COMPLETION_GATE.json'), actual_training_gate_sha256=sha256(output / 'ACTUAL_COMPLETION_GATE.json'),
                numeric_core_sha256=original_sha, adapter_sha256=sha256(ROOT / 'src/n01/evaluate_window008_resume01.py'),
                loaded_CPU_checkpoint_paths=sorted(set(loaded)), scientific_model_forwards=0, waveform_NPZ_decodes=0,
                DEV_numeric_decodes=0, optimizer_constructions=0, optimizer_updates=0, GPU_work=0, scientific_job_dispatches=0,
                failure_checkpoint_loads=0, numerical_evaluation_executed=False, source_policy_fixture_is_artificial=True,
                scientific_status='response_match_not_established', elapsed_seconds=time.monotonic() - began)
            _write(output / 'ENGINEERING_RECEIPT.json', receipt)
            return receipt
    except BaseException as error:
        write_json(output / 'FAILED.json', dict(UTC=utc(), status='cpu_metadata_QA_failed_preserved', error_type=type(error).__name__,
            error=str(error), completed_checks=checks, loaded_CPU_checkpoint_paths=sorted(set(loaded)),
            scientific_model_forwards=0, DEV_numeric_decodes=0, optimizer_updates=0, numerical_evaluation_executed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--science-run')
    args = parser.parse_args()
    result = run(args.out, args.science_run)
    print(json.dumps({key: result[key] for key in ('status', 'checks', 'passed', 'scientific_model_forwards', 'DEV_numeric_decodes', 'optimizer_updates')}))


if __name__ == '__main__':
    main()
