"""Independent resume01 evaluation review: metadata and artificial arrays only.

Never decodes a scientific waveform NPZ, calls a trained response/noise model,
updates an optimizer, allocates CUDA, launches jobs, or changes global ledgers.
All output is preserved in a fresh E/D directory. Parser fixtures are explicitly
artificial; their native receipts must never be reported as scientific success.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch

from n01 import evaluate_window008 as core
from n01.model008 import N01Response008
from n01.model008_fix01 import N01Response008Fix01
from n01.losses008 import WaveformDiscriminator
from n01.response008 import ARMS, HYPOTHESES, loss_tensor_statistics, state_digest
from n01.response008_fix01 import STOCHASTIC_PREFIXES
from n01.response008_fix01 import tree_digest
import check_window008_evaluation_helpers as old_fixtures
import verify_window008_evaluation as old_compare

CORE_SHA256 = '75f8988415a181a31a7c2b5890418b1f7685e0d8fb141cdea84f4174af783cc0'
CORE_APIS = ('_fixed_settings', '_validate_split', '_load_dev', 'balanced_real_domains',
             'generate_split', 'parent_bootstrap', 'distance_receipt', '_verify_frozen',
             'save_json', 'save_npz', '_array', '_clean')
RUN_ID = 'window008_resume01_20261013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(core._clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, *, atol=2e-11, rtol=2e-11):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, label + ': shape')
    require(np.array_equal(np.isfinite(a), np.isfinite(b)), label + ': undefined mask')
    finite = np.isfinite(b)
    error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
    require(np.allclose(a[finite], b[finite], atol=atol, rtol=rtol), label + ': error ' + str(error))
    return error


def deny(*args, **kwargs):
    raise AssertionError('Forbidden scientific forward, optimizer step, or waveform decode during metadata review')


def source_contract(adapter):
    require(sha(ROOT / 'src/n01/evaluate_window008.py') == CORE_SHA256,
            'Original numerical evaluator changed')
    linked = {}
    for name in CORE_APIS:
        # The adapter can use a module-qualified core call rather than importing
        # a symbol; either way the numerical function itself must be unchanged.
        if hasattr(adapter, name):
            require(getattr(adapter, name) is getattr(core, name), 'Reimplemented core API: ' + name)
            linked[name] = 'identical_function_object'
        else:
            source = (ROOT / 'src/n01/evaluate_window008_resume01.py').read_text(encoding='utf8')
            require('core.' + name in source, 'Missing original numerical API reuse: ' + name)
            linked[name] = 'module_qualified_original'
    handoff = read(ROOT / 'receipts/window008_20261013/EVALUATION_HELPERS_HANDOFF.json')
    original = {}
    for relative, expected in handoff['helper_source_sha256'].items():
        actual = sha(ROOT / relative)
        require(actual == expected, 'Changed original helper/core source: ' + relative)
        original[relative] = actual
    require(core.ARMS == ('deterministic', 'full', 'simple')
            and core.HYPOTHESES == ('fast', 'slow', 'common_level'), 'Arm/hypothesis axis changed')
    return dict(core_sha256=CORE_SHA256, original_helper_files_sha256=original,
                numerical_API_reuse=linked, three_hypotheses_separate=True)


def optimizer_oracle(model, saved, *, mean_steps, joint_steps=None, inactive_stochastic=False, lr, weight_decay):
    """Independent named-slot oracle, without constructing/stepping AdamW."""
    params = list(model.named_parameters())
    require(set(saved) == {'state', 'param_groups'} and len(saved['param_groups']) == 1,
            'AdamW layout differs')
    group = saved['param_groups'][0]
    require(group['params'] == list(range(len(params))) and group['lr'] == lr
            and group['weight_decay'] == weight_decay and tuple(group['betas']) == (.9, .999)
            and group['eps'] == 1e-8 and group['amsgrad'] is False, 'AdamW IDs/hyperparameters differ')
    expected_ids = []
    named_steps = {}
    for index, (name, parameter) in enumerate(params):
        stochastic = name.startswith(STOCHASTIC_PREFIXES)
        if inactive_stochastic and stochastic:
            require(index not in saved['state'], 'Inactive deterministic stochastic slot present: ' + name)
            continue
        expected_ids.append(index)
        require(index in saved['state'], 'Missing trained AdamW slot: ' + name)
        slot = saved['state'][index]
        expected = joint_steps if stochastic and joint_steps is not None else mean_steps
        require(set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}, 'AdamW slot inventory: ' + name)
        step = slot['step']
        require(isinstance(step, torch.Tensor) and step.dtype == torch.float32 and step.shape == ()
                and bool(torch.isfinite(step)) and float(step) == expected, 'AdamW phase step: ' + name)
        for key in ('exp_avg', 'exp_avg_sq'):
            value = slot[key]
            require(isinstance(value, torch.Tensor) and value.shape == parameter.shape
                    and value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                    'AdamW moment shape/dtype/finite: ' + name + '/' + key)
        require(bool((slot['exp_avg_sq'] >= 0).all()), 'AdamW second moment sign: ' + name)
        named_steps[name] = expected
    require(set(saved['state']) == set(expected_ids), 'Unexpected/missing AdamW IDs')
    return dict(named_slots=len(named_steps), step_histogram={str(x): list(named_steps.values()).count(x)
                for x in sorted(set(named_steps.values()))})


def sampler_oracle(saved, *, population, draws):
    require(set(saved) == {'n', 'rng', 'order', 'position', 'completed_epochs', 'visits'}
            and type(saved['n']) is int and saved['n'] == population, 'Sampler population/layout')
    order, visits = saved['order'], saved['visits']
    require(order.dtype == torch.int64 and visits.dtype == torch.int64
            and tuple(order.shape) == tuple(visits.shape) == (population,), 'Sampler shapes/dtypes')
    order, visits = order.cpu().numpy(), visits.cpu().numpy()
    require(np.array_equal(np.sort(order), np.arange(population)), 'Sampler permutation')
    epochs, position = divmod(draws, population)
    expected = np.full(population, epochs, np.int64)
    expected[order[:position]] += 1
    require(saved['completed_epochs'] == epochs and saved['position'] == position
            and np.array_equal(visits, expected) and int(visits.sum()) == draws
            and int((visits > 0).sum()) == population, 'Sampler draws/coverage')
    require(saved['rng']['bit_generator'] == 'PCG64', 'Sampler RNG family')
    return dict(parents=population, draws=draws, epochs=epochs, position=position, covered=population)


def actual_metadata(run, cfg, resume_cfg, adapter):
    collection = read(ROOT / 'receipts/window008_resume01_20261013/COMPLETE_COLLECTION.json')
    local_run_root = run.parents[1]
    actual_files = {}
    for relative, binding in collection['files'].items():
        path = local_run_root / relative
        require(path.is_file() and path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'],
                'Collected file differs: ' + relative)
        actual_files[relative] = binding['sha256']
    complete = read(run / 'TRAINING_COMPLETE.json')
    native = read(run.parent / 'supervisor/exit_receipt.json')
    launch = read(run.parent / 'supervisor/launch.json')
    binding = read(run / 'BINDING.json')
    require(native['exit_code'] == 0 and native['reason'] is None and native == collection['native'],
            'Collected native completion differs')
    require(launch['supervisor_pid'] == native['supervisor_pid']
            and launch['child_pid'] == native['child_pid'] == binding['parent_pid']
            and binding['pid'] != binding['parent_pid'], 'Actual trainer/timeout/supervisor chain')
    require(complete['legacy_native_chain']['original_native_chain']['native_exit_code'] == 1
            and complete['legacy_native_chain']['fix01_native_exit_code'] == 1
            and complete['legacy_native_chain']['old_jobs_native_success'] is False,
            'Historical actual native1 roles lost')
    expected = {'deterministic': ('response008/v1', N01Response008, 'retained_original_endpoint'),
                'full': ('response008_resume01/v1', N01Response008Fix01, 'fix01_periodic6000_resume'),
                'simple': ('response008_resume01/v1', N01Response008Fix01, 'fresh_common_initialization')}
    rows = {row['arm']: row for row in complete['arms']}
    require(set(rows) == set(ARMS) and len(complete['arms']) == 3, 'Actual endpoint inventory')
    statistics = read(ROOT / cfg['statistics'])
    endpoints, common, noise = {}, set(), set()
    with mock.patch.object(np, 'load', side_effect=deny), \
            mock.patch.object(torch.nn.Module, '_call_impl', side_effect=deny), \
            mock.patch.object(torch.optim.AdamW, 'step', side_effect=deny):
        gate = adapter.training_completion_gate(run, cfg, resume_cfg, root=ROOT)
        for arm in ARMS:
            schema, model_class, origin = expected[arm]
            entry, row = complete['endpoint_map'][arm], rows[arm]
            path = ROOT / entry['path'] if arm == 'deterministic' else local_run_root / entry['path']
            require(entry['schema'] == schema and entry['origin'] == origin
                    and sha(path) == entry['sha256'] == row['final_sha256'], 'Endpoint file/schema/origin: ' + arm)
            before = sha(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            require(checkpoint['schema'] == schema and checkpoint['arm'] == arm
                    and checkpoint['stage'] == 'low_learning_rate' and checkpoint['G_updates'] == 19000
                    and checkpoint['D_updates'] == {h: 16000 for h in HYPOTHESES}
                    and checkpoint['config'] == cfg and core._json_sha(checkpoint['statistics']) == core._json_sha(statistics)
                    and checkpoint['noise_updates'] == 0 and checkpoint['population_learning'] is False,
                    'Endpoint actual checkpoint metadata: ' + arm)
            model = model_class(statistics['model'], arm)
            model.load_state_dict(checkpoint['model'], strict=True)
            model_digest = state_digest(model.state_dict())
            require(model_digest == row['state_sha256'], 'Actual model state digest: ' + arm)
            g = optimizer_oracle(model, checkpoint['g_optimizer'], mean_steps=19000, joint_steps=16000,
                                 inactive_stochastic=arm == 'deterministic',
                                 lr=cfg['loss']['G_low_lr'], weight_decay=cfg['loss']['G_weight_decay'])
            dmodel = WaveformDiscriminator(loss_tensor_statistics(statistics['loss']))
            d = {h: optimizer_oracle(dmodel, checkpoint['d_optimizers'][h], mean_steps=16000,
                                    lr=cfg['loss']['D_low_lr'], weight_decay=0.) for h in HYPOTHESES}
            source = sampler_oracle(checkpoint['source_sampler'], population=6593,
                                    draws=19000 * cfg['distribution_microbatches'] * cfg['source_parents_per_microbatch'])
            real = sampler_oracle(checkpoint['real_sampler'], population=21047,
                                  draws=19000 * cfg['distribution_microbatches'] * cfg['real_rows_per_microbatch'])
            common.add(checkpoint['common_initial_state_sha256']); noise.add(checkpoint['noise005_state_sha256'])
            if arm != 'deterministic':
                expected_new_g = 13000 if arm == 'full' else 19000
                expected_new_d = 13000 if arm == 'full' else 16000
                require(checkpoint['checkpoint_purpose'] == 'final'
                        and checkpoint['complete_step_boundary'] is True
                        and checkpoint['automatic_resume_allowed'] is False
                        and checkpoint['failure_may_be_mid_step'] is False
                        and checkpoint['engineering_fixture'] is False
                        and checkpoint['new_G_updates'] == expected_new_g
                        and checkpoint['new_D_updates'] == {h: expected_new_d for h in HYPOTHESES}
                        and checkpoint['lineage'] == row['lineage']
                        and checkpoint['recovery'] == row['recovery'] == complete['recovery'],
                        'Actual final checkpoint lineage/boundary/counter role: ' + arm)
                require(checkpoint['lineage']['origin'] == origin
                        and checkpoint['lineage']['source_checkpoint_sha256']
                        == (resume_cfg['reference_inputs']['fix01_full_periodic_06000']['sha256'] if arm == 'full' else None),
                        'Actual resume/fresh source identity: ' + arm)
            require(sha(path) == before, 'Metadata load changed endpoint bytes')
            endpoints[arm] = dict(file_sha256=before, schema=schema, origin=origin,
                                 model_state_sha256=model_digest, G_optimizer=g, D_optimizers=d,
                                 SOURCE_sampler=source, REAL_sampler=real)
            del model, dmodel, checkpoint
        prior = torch.load(ROOT / resume_cfg['reference_inputs']['fix01_full_periodic_06000']['path'],
                           map_location='cpu', weights_only=False)
        restored = torch.load(run / 'train/full/restored_start.pt', map_location='cpu', weights_only=False)
        validation = read(run / 'RESUME_VALIDATION.json')
        native_keys = ('model', 'discriminators', 'g_optimizer', 'd_optimizers',
                       'source_sampler', 'real_sampler', 'streams')
        native_trees = {}
        for name in native_keys:
            digest = tree_digest(prior[name])
            require(digest == tree_digest(restored[name]) == validation['state_digests'][name],
                    'Full6000 native tree import differs: ' + name)
            native_trees[name] = digest
        require(prior['G_updates'] == restored['G_updates'] == 6000
                and prior['D_updates'] == restored['D_updates'] == {h: 3000 for h in HYPOTHESES}
                and validation['next_G_update'] == validation['projection_update'] == 6001
                and validation['native_tree_bitwise_equal'] is True
                and validation['optimizer_moments_reset'] is False
                and validation['optimizer_sampler_streams_exact'] is True,
                'Full6000 import projection or actual counters differ')
        del prior, restored
    require(len(common) == len(noise) == 1, 'Actual common initializer/noise state differs')
    return dict(collected_files_sha256=actual_files, endpoints=endpoints, native=native,
                legacy_numeric_native_exit=1, fix01_user_stop_native_exit=1,
                common_state_sha256=next(iter(common)), noise005_state_sha256=next(iter(noise)),
                full6000_native_tree_sha256=native_trees,
                adapter_gate=gate, waveform_NPZ_decodes=0, trained_forwards=0, optimizer_steps=0)


def helper_replay_fixture(output):
    base = output / 'artificial_helper_reference'
    candidate = output / 'artificial_helper_replay'
    old_fixtures.build_artifacts(base)
    shutil.copytree(base, candidate)
    report = old_compare.compare_artifacts(base, candidate)
    require(report['status'] == 'exact_replay_verified' and report['NPZ_files_checked'] == 13
            and report['NPZ_members_checked'] == 66, 'Original helper 13-file/66-member exact replay')
    checks = []
    mutations = {
        'raw_array': lambda path: old_fixtures.mutate_array(path / 'DEV_observables.npz', 'numeric', lambda x: x + .01),
        'dtype': lambda path: old_fixtures.mutate_array(path / 'TRAIN_curves.npz', 'numeric', lambda x: x.astype(np.float32)),
        'stream_state': lambda path: old_fixtures.mutate_array(path / 'DEV_stream_states.npz', 'final_response', lambda x: x + 1),
        'bootstrap': lambda path: old_fixtures.mutate_array(path / 'DEV_parent_bootstrap.npz', 'bootstrap_std', lambda x: x + .01),
        'chunk_latent_hash': lambda path: old_fixtures.mutate_json(path / 'DEV_generation.json',
            lambda x: x['chunks'][0].update(extension_epsilon_sha256='f' * 64)),
        'distance': lambda path: old_fixtures.mutate_json(path / 'DISTRIBUTION_DISTANCES.json', lambda x: x.update(wave=.3)),
        'seed': lambda path: old_fixtures.mutate_json(path / 'FROZEN_INPUTS.json',
            lambda x: x['config']['seeds'].update(fixture=20)),
    }
    for name, mutate in mutations.items():
        path = output / ('artificial_replay_reject_' + name)
        shutil.copytree(base, path)
        mutate(path)
        old_fixtures.rebind_artifacts(path)
        result = old_compare.compare_artifacts(base, path)
        require(result['status'] == 'exact_replay_failed' and result['mismatches'], 'Accepted replay drift: ' + name)
        checks.append(dict(mutation=name, mismatches=result['mismatches']))
    write(output / 'ARTIFICIAL_13_NPZ_66_MEMBER_REPLAY.json', report)
    return dict(NPZ_files=13, members=66, artificial_helper_inventory_only=True,
                strict_all_actual_members_not_capped_at_66=True, rejected_mutations=checks)


def adapter_rejections(output, run, cfg, resume_cfg, adapter):
    from n01 import train_window008_resume01 as driver
    fix = read(ROOT / 'window008_fix01_config.json')
    documents = {name: read(ROOT / entry['path']) for name, entry in resume_cfg['reference_inputs'].items()
                 if entry['path'].endswith('.json')}
    original = driver.validate_legacy_receipts(cfg, fix, documents)
    driver.validate_stop_chain(cfg, fix, resume_cfg, documents, original)
    rejected = []
    for name in ('legacy_native_exit', 'fix01_native_exit'):
        bad = copy.deepcopy(documents); bad[name]['exit_code'] = 0
        try:
            legacy = driver.validate_legacy_receipts(cfg, fix, bad)
            driver.validate_stop_chain(cfg, fix, resume_cfg, bad, legacy)
        except (ValueError, KeyError):
            rejected.append(name + '_cannot_relabel_native0')
        else:
            raise AssertionError('Historical native1 was relabelled as success: ' + name)
    native_path, launch_path = run.parent / 'supervisor/exit_receipt.json', run.parent / 'supervisor/launch.json'
    native, launch = read(native_path), read(launch_path)
    binding, complete = read(run / 'BINDING.json'), read(run / 'TRAINING_COMPLETE.json')
    raw_read = adapter._read_json
    def invoke(n=native, l=launch, b=binding):
        def reader(path):
            if Path(path) == native_path: return copy.deepcopy(n)
            if Path(path) == launch_path: return copy.deepcopy(l)
            return raw_read(path)
        with mock.patch.object(adapter, '_read_json', side_effect=reader):
            return adapter._native_completion(run, cfg, resume_cfg, complete, b, complete['recovery'])
    invoke()
    variants = []
    for name, field, value in (('native1', 'exit_code', 1), ('bool_exit', 'exit_code', False),
                               ('resource_stop', 'reason', 'owned_job_wall_limit_exceeded'),
                               ('wrong_wrapper', 'child_pid', 999), ('time_before_completion', 'ended_utc', native['started_utc'])):
        value_dict = copy.deepcopy(native); value_dict[field] = value
        variants.append((name, lambda v=value_dict: invoke(n=v)))
    bad_binding = copy.deepcopy(binding); bad_binding['parent_pid'] = 999
    variants.append(('trainer_parent_pid', lambda: invoke(b=bad_binding)))
    for module in ('n01.train_window008', 'n01.train_window008_fix01'):
        value_dict = copy.deepcopy(launch); value_dict['command'][7] = module
        variants.append(('wrong_module_' + module, lambda v=value_dict: invoke(l=v)))
    bad_launch = copy.deepcopy(launch); bad_launch['command'] += ['--device', 'cuda:0']
    variants.append(('duplicate_command_flag', lambda: invoke(l=bad_launch)))
    for name, action in variants:
        try:
            action()
        except (ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Malformed native metadata accepted: ' + name)

    statistics = read(ROOT / cfg['statistics'])
    initial = torch.load(ROOT / resume_cfg['reference_inputs']['full_initial']['path'], map_location='cpu', weights_only=False)
    full = torch.load(run / 'train/full/final.pt', map_location='cpu', weights_only=False)
    validation = read(run / 'LEGACY_VALIDATION.json')
    expected_lineage = adapter._expected_lineage('full', cfg, resume_cfg)
    parent = driver.fix01_identity(resume_cfg)
    def checkpoint_case(candidate):
        with mock.patch.object(np, 'load', side_effect=deny), \
                mock.patch.object(torch.nn.Module, '_call_impl', side_effect=deny), \
                mock.patch.object(torch.optim.AdamW, 'step', side_effect=deny):
            return adapter._validate_checkpoint(candidate, 'full', cfg, statistics, complete['recovery'], parent,
                validation['common_initial_state_sha256'], validation['noise005_state_sha256'],
                validation['initial_model_state_sha256']['full'], initial['initial_D_sha256'],
                expected_lineage, initial)
    checkpoint_case(full)
    mutations = [('wrong_schema', lambda x: x.update(schema='response008_fix01/v1')),
                 ('wrong_stage', lambda x: x.update(stage='joint_stochastic')),
                 ('G18999', lambda x: x.update(G_updates=18999)),
                 ('D15999', lambda x: x['D_updates'].update(fast=15999)),
                 ('wrong_common', lambda x: x.update(common_initial_state_sha256='f' * 64)),
                 ('wrong_noise', lambda x: x.update(noise005_state_sha256='f' * 64)),
                 ('wrong_lineage', lambda x: x['lineage'].update(source_checkpoint_sha256='f' * 64)),
                 ('failure_as_final', lambda x: x.update(failure_may_be_mid_step=True)),
                 ('SOURCE_coverage_gap', lambda x: x['source_sampler']['visits'].__setitem__(0, 0))]
    parameter = next(index for index, row in enumerate(full['parameter_signature']) if row['name'].startswith(STOCHASTIC_PREFIXES))
    mutations.extend([
        ('stochastic_AdamW_step_wrongly19000', lambda x: x['g_optimizer']['state'][parameter]['step'].fill_(19000)),
        ('missing_AdamW_slot', lambda x: x['g_optimizer']['state'].pop(parameter)),
        ('negative_second_moment', lambda x: x['g_optimizer']['state'][parameter]['exp_avg_sq'].fill_(-1)),
    ])
    for name, mutate in mutations:
        candidate = copy.deepcopy(full); mutate(candidate)
        try:
            checkpoint_case(candidate)
        except (ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Malformed actual-state metadata accepted: ' + name)
    del initial, full
    # Fail the freeze before *any* numerical TRAIN or DEV decode. Failure flags
    # must represent the real execution boundary even for a parser fixture.
    fake_run = output / 'artificial_freeze_failure/runs/science'
    destination = fake_run.parent / 'evaluation_jobs/primary/artifacts'
    freeze_error = ValueError('Artificial failure before endpoint/input freeze')
    with mock.patch.object(adapter, '_freeze', side_effect=freeze_error), \
            mock.patch.object(adapter, '_load_dev', side_effect=deny) as dev_loader, \
            mock.patch.object(np, 'load', side_effect=deny) as decoder, \
            mock.patch.object(torch.nn.Module, '_call_impl', side_effect=deny), \
            mock.patch.object(torch.optim.AdamW, 'step', side_effect=deny):
        try:
            adapter.evaluate_run(fake_run, cfg, resume_cfg, ROOT / cfg['dataset'], ROOT / cfg['statistics'],
                                 cfg['local_training_container'], ROOT / 'data/PARENT_IDENTITIES.npz', 'cpu',
                                 destination=destination, source_manifest='artificial_unread_manifest.json',
                                 policy_path=ROOT / 'window008_resume01_evaluation_policy.json', root=ROOT)
        except ValueError as error:
            require(error is freeze_error, 'Unexpected artificial freeze failure')
        else:
            raise AssertionError('Freeze failure did not stop evaluation')
    failure = read(destination / 'FAILED.json')
    require(dev_loader.call_count == decoder.call_count == 0 and failure['DEV_numeric_decoded'] is False
            and failure['DEV_numeric_decode_started'] is False and failure['partial_artifacts_preserved'] is True,
            'Numeric decoding escaped the failed freeze')
    return dict(rejected_metadata_mutations=rejected, original_native1_preserved=True, user_stop_native1_preserved=True,
                DEV_loader_calls_before_failed_freeze=0, waveform_NPZ_decoder_calls=0,
                failed_attempt_preserved=True, response_optimizer_steps=0)


def operations_contract(output, cfg, resume_cfg):
    import window008_resume01_evaluation_common as ops
    import verify_window008_resume01_evaluation as replay
    import supervise_window008_resume01_evaluation as supervisor
    import stage_window008_resume01_evaluation as staging
    policy = read(ROOT / 'window008_resume01_evaluation_policy.json')
    require(policy['resource_limits'] == dict(GPU_GiB=10, job_tree_RSS_GiB=8, wall_seconds=21600,
                                              output_MiB=3072, poll_seconds=1, TERM_grace_seconds=20)
            and policy['CPU_threads'] == 12, 'Resume01 evaluation resource policy')
    # Compare the entire semantic AST of numerical comparison and all helpers;
    # only the module import that supplies integrity/native gates is versioned.
    def definitions(path):
        tree = ast.parse(Path(path).read_text(encoding='utf8'))
        return {item.name: ast.dump(item, include_attributes=False) for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name != 'main'}
    require(definitions(ROOT / 'scripts/verify_window008_evaluation.py')
            == definitions(ROOT / 'scripts/verify_window008_resume01_evaluation.py'), 'Numerical comparator logic changed')
    require(old_compare.NPZ_NAMES == replay.NPZ_NAMES and old_compare.JSON_IGNORED == replay.JSON_IGNORED,
            'Replay inventory or volatile exclusions changed')
    limits = policy['resource_limits']
    require(supervisor.stop_reason(8 * 1024**3, 21600, 3072 * 1024**2, 10 * 1024**3, limits) is None,
            'Budget inclusive boundary differs')
    reasons = [supervisor.stop_reason(rss, wall, size, gpu, limits) for rss, wall, size, gpu in (
        (8 * 1024**3 + 1, 0, 0, 0), (0, 21600.001, 0, 0),
        (0, 0, 3072 * 1024**2 + 1, 0), (0, 0, 0, 10 * 1024**3 + 1))]
    require(all(reasons) and len(set(reasons)) == 4, 'Individual resource stop rules')
    boundary_rejections = []
    for relative in ('../escape', '/absolute', 'sub/../../escape'):
        try:
            ops.bound(output, relative)
        except ValueError:
            boundary_rejections.append(relative)
        else:
            raise AssertionError('Accepted path escape: ' + relative)

    root = (output / 'artificial_native_ops').resolve()
    root.mkdir()
    write(root / ops.EVAL_MANIFEST, dict(fixture_only=True, schema='artificial_manifest'))
    write(root / ops.TRAIN_MANIFEST, dict(fixture_only=True, schema='artificial_training_manifest'))
    paths = ops.paths_for('primary', root)
    paths['job'].mkdir(parents=True); paths['supervisor'].mkdir()
    shutil.copytree(output / 'artificial_helper_reference', paths['artifacts'])
    summary = read(paths['artifacts'] / 'SUMMARY.json')
    summary.update(schema='n01/window008-resume01-evaluation/v1', status='fixed_resumed_endpoint_evaluation_complete')
    (paths['artifacts'] / 'SUMMARY.json').write_text(json.dumps(summary, allow_nan=False) + '\n', encoding='utf8')
    executable = '/artificial-fixture/python'
    launch = dict(fixture_only=True, supervisor_pid=201, child_pid=202,
                  command=[executable, '-B', str(root / 'scripts/run_window008_resume01_evaluation.py'), '--kind', 'primary'])
    native = dict(fixture_only=True, exit_code=0, reason=None, supervisor_pid=201, child_pid=202)
    command = [executable, '-B', '-m', 'n01.evaluate_window008_resume01', '--out', str(paths['run']),
               '--config', str(root / 'window008_config.json'), '--resume-config', str(root / 'window008_resume01_config.json'),
               '--source-manifest', str(root / ops.EVAL_MANIFEST), '--policy', str(root / ops.POLICY),
               '--dataset', str(root / cfg['dataset']), '--statistics', str(root / cfg['statistics']),
               '--raw-container', cfg['remote_training_container'], '--identities', str(root / 'data/PARENT_IDENTITIES.npz'),
               '--device', 'cuda:0', '--destination', str(paths['artifacts'])]
    evaluator = dict(fixture_only=True, exit_code=0, module='n01.evaluate_window008_resume01', kind='primary',
                     command=command, wrapper_pid=202, supervisor_pid=201,
                     evaluation_source_manifest_sha256=sha(root / ops.EVAL_MANIFEST))
    write(paths['supervisor'] / 'launch.json', launch)
    write(paths['supervisor'] / 'exit_receipt.json', native)
    write(paths['job'] / 'EVALUATOR_NATIVE.json', evaluator)
    job = dict(fixture_only=True, status='evaluation_job_complete', kind='primary',
               evaluation_source_manifest_sha256=sha(root / ops.EVAL_MANIFEST),
               training_source_manifest_sha256=sha(root / ops.TRAIN_MANIFEST), optimizer_updates=0,
               evaluator_native_sha256=sha(paths['job'] / 'EVALUATOR_NATIVE.json'),
               summary_sha256=sha(paths['artifacts'] / 'SUMMARY.json'))
    write(paths['job'] / 'JOB_COMPLETE.json', job)
    originals = {path: path.read_text(encoding='utf8') for path in
                 (paths['supervisor'] / 'launch.json', paths['supervisor'] / 'exit_receipt.json',
                  paths['job'] / 'EVALUATOR_NATIVE.json', paths['job'] / 'JOB_COMPLETE.json',
                  paths['artifacts'] / 'SUMMARY.json')}
    with mock.patch.object(ops, 'evaluation_binding', return_value=(cfg, {}, policy, {})):
        valid = ops.completed_job('primary', root)
        require(valid['native']['exit_code'] == 0 and valid['core']['exit_code'] == 0, 'Artificial native ops chain')
        mutations = [
            ('supervisor_exit1', paths['supervisor'] / 'exit_receipt.json', lambda x: x.update(exit_code=1)),
            ('resource_stop', paths['supervisor'] / 'exit_receipt.json', lambda x: x.update(reason='owned_job_wall_limit_exceeded')),
            ('wrapper_pid', paths['job'] / 'EVALUATOR_NATIVE.json', lambda x: x.update(wrapper_pid=999)),
            ('wrong_module', paths['job'] / 'EVALUATOR_NATIVE.json', lambda x: x.update(module='n01.evaluate_window008')),
            ('core_native1', paths['job'] / 'EVALUATOR_NATIVE.json', lambda x: x.update(exit_code=1)),
            ('old_summary_schema', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(schema='n01/window008-evaluation/v1')),
            ('evaluation_update', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(evaluation_G_updates=1)),
        ]
        rejected = []
        for name, path, mutate in mutations:
            value = json.loads(originals[path]); mutate(value)
            path.write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf8')
            # Rebind the outer job hashes so rejection is by the changed role,
            # not merely stale serialization. This is an artificial parser test.
            outer = copy.deepcopy(job)
            outer['evaluator_native_sha256'] = sha(paths['job'] / 'EVALUATOR_NATIVE.json')
            outer['summary_sha256'] = sha(paths['artifacts'] / 'SUMMARY.json')
            (paths['job'] / 'JOB_COMPLETE.json').write_text(json.dumps(outer) + '\n', encoding='utf8')
            try:
                ops.completed_job('primary', root)
            except (ValueError, AssertionError):
                rejected.append(name)
            else:
                raise AssertionError('Accepted artificial native-role mutation: ' + name)
            for target, text in originals.items():
                target.write_text(text, encoding='utf8')
    # The same exact comparator now uses resume01 artifact integrity.
    reference = output / 'artificial_resume_ops_reference'; candidate = output / 'artificial_resume_ops_replay'
    shutil.copytree(paths['artifacts'], reference); shutil.copytree(paths['artifacts'], candidate)
    exact = replay.compare_artifacts(reference, candidate)
    require(exact['status'] == 'exact_replay_verified' and exact['NPZ_members_checked'] == 66,
            'Resume ops exact artificial replay')
    write(output / 'ARTIFICIAL_RESUME_OPS_REPLAY.json', exact)
    return dict(comparator_semantic_AST_identical=True, ignored_fields_unchanged=True,
                resource_limits=limits, stop_reasons=reasons, path_escape_rejections=boundary_rejections,
                artificial_native_role_mutations_rejected=rejected, native_Linux_job_execution=False)


def diagnostic_operations(output, cfg):
    import window008_resume01_diagnostics_common as ops
    root = (output / 'artificial_diagnostic_native_ops').resolve(); root.mkdir()
    write(root / 'window008_config.json', cfg)
    write(root / ops.TRAIN_MANIFEST, dict(fixture_only=True))
    write(root / ops.EVAL_MANIFEST, dict(fixture_only=True))
    paths = ops.paths_for('primary', root)
    paths['artifacts'].mkdir(parents=True); paths['supervisor'].mkdir()
    families = dict(repeat=(24576, 73728, 8192), shape=(1536, 4608, 256),
                    Q_trend=(2304, 6912, 256), identity_order_replay=(1536, 4608, 512))
    counts = dict(primary=dict(signal_forward_rows=28416, observation_rows=85248, noise005_forward_rows=8704),
                  total_signal_forward_rows=29952, total_observation_rows=89856, total_noise005_forward_rows=9216,
                  groups={name: dict(signal_forward_rows=x[0], observation_rows=x[1], noise005_forward_rows=x[2],
                      per_arm={a: x[0] // 3 for a in ARMS},
                      per_arm_hypothesis={a + '_' + h: x[1] // 9 for a in ARMS for h in HYPOTHESES})
                          for name, x in families.items()})
    required = {'FROZEN_INPUTS.json', 'ADDON_PREREGISTERED_INPUTS.json', 'ADDON_FROZEN_INPUTS.json', 'SELECTIONS.npz',
                'repeat_latents.npz', 'intervention_latents.npz', 'LATENT_BANKS.json', 'REPLAY_IDENTITY_ORDER.json',
                'ACTUAL_COUNTERS.json', 'GENERATION.json', 'DIAGNOSTIC_STATISTICS.json'}
    required |= {family + '_' + a + '_' + h + '.npz' for family in ('REPEAT', 'SHAPE', 'Q_TREND')
                 for a in ARMS for h in HYPOTHESES}
    required |= {family + '_COMMON.npz' for family in ('REPEAT', 'SHAPE', 'Q_TREND')}
    required |= {'POPULATION_' + split + '.' + suffix for split in ('TRAIN', 'DEV') for suffix in ('json', 'npz')}
    require(len(required) == 45, 'Independent diagnostic payload inventory count')
    for name in sorted(required):
        if name == 'ACTUAL_COUNTERS.json':
            write(paths['artifacts'] / name, counts)
        elif name.endswith('.json'):
            write(paths['artifacts'] / name, dict(fixture_only=True))
        else:
            core.save_npz(paths['artifacts'] / name, artificial_fixture_only=np.array([True]))
    summary = dict(schema='n01/window008-resume01-diagnostics/v1',
                   status='registered_frozen_endpoint_diagnostics_complete', fixture_only=True,
                   scientific_status='response_match_not_established', optimizer_updates=0, G_updates=0, D_updates=0,
                   C2ST_updates=0, noise_updates=0, population_updates=0, DEV_numeric_decoded=True,
                   artificial_arrays_only=False, all_selected_parent_draw_rows_preserved=True, frozen_states_rechecked=True,
                   scientific_model_forward_rows=29952, repeat_selection_parent_count=128, repeat_draws_per_parent=64,
                   intervention_parent_count=32, intervention_draws_per_parent=8, actual_forward_counts=counts,
                   population={split: dict(parents=n, rows=n * 4, response_forward_rows=0,
                                           mathematical_waveform_mixture_exact=True) for split, n in (('TRAIN', 6593), ('DEV', 512))},
                   artifact_sha256={name: sha(paths['artifacts'] / name) for name in required})
    write(paths['artifacts'] / 'SUMMARY.json', summary)
    # This is a labelled parser fixture, not an actual job receipt. Production
    # fields are exercised explicitly so malformed role/count rows can be tested.
    executable = '/artificial-fixture/python'
    launch = dict(fixture_only=True, supervisor_pid=301, child_pid=302,
                  command=[executable, '-B', str(root / 'scripts/run_window008_resume01_diagnostics.py'), '--kind', 'primary'])
    native = dict(fixture_only=True, exit_code=0, reason=None, supervisor_pid=301, child_pid=302)
    evaluator = dict(fixture_only=True, exit_code=0, module='n01.diagnose_window008_resume01', kind='primary',
                     command=ops.core_command(root, executable), wrapper_pid=302, supervisor_pid=301,
                     diagnostics_source_manifest_sha256=sha(root / ops.EVAL_MANIFEST))
    write(paths['supervisor'] / 'launch.json', launch); write(paths['supervisor'] / 'exit_receipt.json', native)
    write(paths['job'] / 'EVALUATOR_NATIVE.json', evaluator)
    job = dict(fixture_only=True, status='diagnostics_job_complete', kind='primary', optimizer_updates=0,
               diagnostics_source_manifest_sha256=sha(root / ops.EVAL_MANIFEST),
               training_source_manifest_sha256=sha(root / ops.TRAIN_MANIFEST),
               evaluator_native_sha256=sha(paths['job'] / 'EVALUATOR_NATIVE.json'),
               summary_sha256=sha(paths['artifacts'] / 'SUMMARY.json'))
    write(paths['job'] / 'JOB_COMPLETE.json', job)
    originals = {path: path.read_text(encoding='utf8') for path in
                 (paths['supervisor'] / 'launch.json', paths['supervisor'] / 'exit_receipt.json',
                  paths['job'] / 'EVALUATOR_NATIVE.json', paths['job'] / 'JOB_COMPLETE.json',
                  paths['artifacts'] / 'SUMMARY.json', paths['artifacts'] / 'ACTUAL_COUNTERS.json')}
    mutations = [
        ('diag_native1', paths['supervisor'] / 'exit_receipt.json', lambda x: x.update(exit_code=1)),
        ('diag_resource_stop', paths['supervisor'] / 'exit_receipt.json', lambda x: x.update(reason='output_limit')),
        ('diag_wrapper_pid', paths['job'] / 'EVALUATOR_NATIVE.json', lambda x: x.update(wrapper_pid=999)),
        ('diag_old_module', paths['job'] / 'EVALUATOR_NATIVE.json', lambda x: x.update(module='n01.evaluate_window008')),
        ('diag_bad_schema', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(schema='wrong/v1')),
        ('diag_nonzero_noise_update', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(noise_updates=1)),
        ('diag_pre_freeze_decode', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(frozen_states_rechecked=False)),
        ('diag_repeat_reduced', paths['artifacts'] / 'SUMMARY.json', lambda x: x.update(repeat_selection_parent_count=127)),
        ('diag_budget_mixed_with_replay', paths['artifacts'] / 'ACTUAL_COUNTERS.json',
            lambda x: x['primary'].update(signal_forward_rows=29952)),
        ('diag_count_sum_truncated', paths['artifacts'] / 'ACTUAL_COUNTERS.json',
            lambda x: x.update(total_signal_forward_rows=29951)),
        ('diag_missing_arm_hypothesis', paths['artifacts'] / 'ACTUAL_COUNTERS.json',
            lambda x: x['groups']['shape']['per_arm_hypothesis'].pop('full_slow')),
        ('diag_population_additional_forward', paths['artifacts'] / 'SUMMARY.json',
            lambda x: x['population']['DEV'].update(response_forward_rows=1)),
    ]
    rejected = []
    with mock.patch.object(ops, 'evaluation_binding', return_value=(cfg, {}, {}, {})):
        ops.completed_job('primary', root)
        for name, path, mutate in mutations:
            value = json.loads(originals[path]); mutate(value)
            path.write_text(json.dumps(value, allow_nan=False) + '\n', encoding='utf8')
            # If counters change, keep the summary and file hashes internally
            # consistent so the family/total contracts themselves are tested.
            updated_summary = read(paths['artifacts'] / 'SUMMARY.json')
            if path.name == 'ACTUAL_COUNTERS.json':
                updated_summary['actual_forward_counts'] = value
                updated_summary['artifact_sha256']['ACTUAL_COUNTERS.json'] = sha(path)
                (paths['artifacts'] / 'SUMMARY.json').write_text(json.dumps(updated_summary) + '\n', encoding='utf8')
            updated_job = copy.deepcopy(job)
            updated_job['summary_sha256'] = sha(paths['artifacts'] / 'SUMMARY.json')
            updated_job['evaluator_native_sha256'] = sha(paths['job'] / 'EVALUATOR_NATIVE.json')
            (paths['job'] / 'JOB_COMPLETE.json').write_text(json.dumps(updated_job) + '\n', encoding='utf8')
            try:
                ops.completed_job('primary', root)
            except (ValueError, KeyError):
                rejected.append(name)
            else:
                raise AssertionError('Malformed diagnostic fixture accepted: ' + name)
            for target, text in originals.items():
                target.write_text(text, encoding='utf8')
    return dict(artificial_parser_fixture_only=True, payload_files=45, rejected_count_native_scope_mutations=rejected,
                primary_signal_rows=28416, primary_observation_rows=85248,
                separately_registered_engineering_replay_signal_rows=1536,
                separately_registered_engineering_replay_observation_rows=4608,
                actual_scientific_diagnostic_jobs_launched=0)


def diagnostic_math(output):
    from n01 import diagnose_window008_resume01 as diag
    policy = diag.fixed_policy(ROOT / 'window008_resume01_diagnostics_policy.json')
    ids = np.arange(512, dtype=np.int64)[::-1] + 70000
    selected = diag.select_repeat_parents(ids, policy)
    require(np.array_equal(selected, np.random.default_rng(2026101399).permutation(512)[:128])
            and len(np.unique(selected)) == 128, 'Independent identity-only 128-parent selection')
    t = np.arange(384, dtype=np.float64)[None]
    row = np.arange(512, dtype=np.float64)[:, None]
    shape = np.where(t < 126, np.exp(-((t-126)/(3+row%5))**2), np.exp(-np.maximum(t-126, 0)/(21+row%43)))
    shape[:, :100] = 0.
    q = .5 + np.arange(512) / 1000
    carrier = shape * (q / shape[:, 110:366].sum(1))[:, None] + .37
    centered = carrier - carrier[:, :100].mean(1, keepdims=True)
    actual_q = centered[:, 110:366].sum(1)
    edges = np.linspace(.2, 1.2, 9)
    result = diag.shape_interventions(carrier, ids, selected, edges, policy)
    oracle_donors = []
    for parent in selected[:32]:
        other = [j for j in range(512) if j != parent]
        oracle_donors.append(min(other, key=lambda j: (abs(math.log(actual_q[j]/actual_q[parent])), int(ids[j]))))
    require(np.array_equal(result['donor_rows'], oracle_donors)
            and np.array_equal(result['selected_rows'], selected[:32]), 'Independent nonself nearest-Q donor/tie selection')
    charge_error = close(result['donor_same_Q_carrier384'][:, 110:366].sum(1), actual_q[selected[:32]],
                         'Donor same canonical Q', atol=1e-12, rtol=0)
    require(np.array_equal(result['Q_nodes'], edges[[2, 4, 6]]), 'Q nodes changed from frozen TRAIN edge indices2/4/6')
    qnode_error = close(result['trend_carriers384'][:, :, 110:366].sum(2),
                        np.broadcast_to(edges[[2, 4, 6], None], (3, 32)), 'Fixed shape Q nodes', atol=1e-12, rtol=0)
    trend_unit = result['trend_carriers384'] / edges[[2, 4, 6], None, None]
    source_unit = centered[selected[:32]] / actual_q[selected[:32], None]
    close(trend_unit, np.broadcast_to(source_unit, trend_unit.shape), 'Q-trend normalized input shape', atol=1e-12, rtol=0)
    require((result['donor_same_Q_carrier384'][:, 112:144].argmax(1) + 112 == 126).all(), 'Donor peak anchor')
    tied_policy = copy.deepcopy(policy); tied_policy['shape_intervention']['parents'] = 1
    tie = diag.shape_interventions(np.repeat(carrier[:1], 3, axis=0), np.array([9, 3, 7]), [0], edges, tied_policy)
    require(tie['donor_ids'].tolist() == [3] and tie['unit_shape_absmax_difference'][0] == 0,
            'Stable original-ID tie or zero-strength intervention changed')
    # A later larger global peak must not change morphology's selected y[16].
    triangle = np.zeros((4, 256), np.float64)
    triangle[0, 8:17] = np.arange(9); triangle[0, 16:25] = np.arange(8, -1, -1)
    triangle[0, 50] = 30.
    triangle[2] = 1.; triangle[3] = -1.
    temporal = diag.temporal_readouts(triangle, .01)
    close(temporal['left_crossing10_ns'][0], 8.8, 'Selected-peak left10', atol=1e-13, rtol=0)
    close(temporal['left_crossing90_ns'][0], 15.2, 'Selected-peak left90', atol=1e-13, rtol=0)
    close(temporal['rise10_90_ns'][0], 6.4, 'Known rise', atol=1e-13, rtol=0)
    close(temporal['FWHM_ns'][0], 8., 'Known FWHM', atol=1e-13, rtol=0)
    require(np.isnan(temporal['FWHM_ns'][1:]).all() and not temporal['FWHM_defined'][1:].any()
            and temporal['peak_voltage_V'][0] == 8 and (temporal['selected_output_peak_index'] == 16).all(),
            'Undefined morphology rows or selected peak replaced by global maximum')
    rng = np.random.default_rng(27184)
    value = 1e6 + 1e-4 * rng.normal(size=(3, 8, 5))
    value[0, :, 2] = np.nan; value[1, 1:, 3] = np.nan
    value[2, :, 4] = 1e6; value[0, ::2, 1] = np.nan
    stats = diag.sample_statistics(value)
    reference_mean = np.full((3, 5), np.nan); reference_var = np.full((3, 5), np.nan)
    covariance = np.full((3, 5, 5), np.nan); pair_counts = np.zeros((3, 5, 5), np.int64)
    counts = np.isfinite(value).sum(1)
    for p in range(3):
        for i in range(5):
            valid = np.isfinite(value[p, :, i]); x = value[p, valid, i]
            if len(x): reference_mean[p, i] = x.mean()
            if len(x) > 1: reference_var[p, i] = x.var(ddof=1)
            for j in range(5):
                pair = np.isfinite(value[p, :, i]) & np.isfinite(value[p, :, j])
                pair_counts[p, i, j] = pair.sum()
                if pair.sum() > 1:
                    covariance[p, i, j] = np.cov(value[p, pair, i], value[p, pair, j], ddof=1)[0, 1]
    mean_error = close(stats['mean'], reference_mean, 'NaN-subset mean', atol=1e-9, rtol=0)
    variance_error = close(stats['variance'], reference_var, 'ddof1 variance', atol=1e-20, rtol=1e-10)
    covariance_error = close(stats['covariance'], covariance, 'Jointly defined pair covariance', atol=1e-20, rtol=1e-10)
    require(np.array_equal(stats['defined_draws'], counts)
            and np.array_equal(stats['undefined_draws'], 8-counts)
            and np.array_equal(stats['covariance_defined_draws'], pair_counts)
            and stats['variance'][2, 4] == 0 and stats['ddof'] == 1,
            'NaN pair counts, zero widths or sample variance role changed')
    # Compare the complete population implementation against its actual saved
    # per-row algebraic fixture, rather than another bin-mixture implementation.
    population_errors = {}
    main_stats = read(ROOT / read(ROOT / 'window008_config.json')['statistics'])
    main_stats['domains'] = read((ROOT / read(ROOT / 'window008_config.json')['dataset']).with_name('FIT_DATA_STATISTICS.json'))['domains']
    for split in ('TRAIN', 'DEV'):
        fixture = output / 'artificial_numeric_core'
        with np.load(fixture / (split + '_observables.npz'), allow_pickle=False) as archive:
            observed = {k: archive[k] for k in archive.files}
        with np.load(fixture / (split + '_curves.npz'), allow_pickle=False) as archive:
            curves = {k: archive[k] for k in archive.files}
        report, saved = diag.population_sensitivity(observed, curves, main_stats, policy)
        parents, bins = observed['source_parent_row'], observed['source_bins']
        n = len(bins); z = (2*bins-7)/7
        raw = np.array([np.ones(n), 1+.5*z, 1-.5*z]); expected_weights = raw*n/raw.sum(1, keepdims=True)
        close(saved['parent_weights'], expected_weights, 'Positive parent weights', atol=1e-14, rtol=0)
        require(np.isfinite(expected_weights).all() and (expected_weights > 0).all(), 'Population weights nonpositive')
        require(report['response_forward_rows'] == report['noise005_forward_rows'] == 0
                and report['REAL_fit_or_reweight'] is False, 'Population reweighting performed a forward or REAL fit')
        for wi, scheme in enumerate(policy['population']['weight_schemes']):
            record = report['schemes'][scheme]; weights = expected_weights[wi, parents]
            close(record['parent_ESS'], expected_weights[wi].sum()**2 / np.square(expected_weights[wi]).sum(),
                  'Parent ESS', atol=1e-12, rtol=0)
            for arm in ARMS:
                for h in HYPOTHESES:
                    label = arm + '_' + h
                    actual = record['arms'][arm][h]
                    require(actual['output_Q_conditional_weighted_waveform_moments'] is None, 'Invented joint-bin waveform moments')
                    for metric in ('Q', 'PSD', 'tail', 'PSD_raw'):
                        x = observed[label + '_' + metric]; finite = np.isfinite(x)
                        total = weights[finite].sum()
                        mean = np.dot(weights[finite], x[finite])/total if total > 0 else np.nan
                        variance = np.dot(weights[finite], (x[finite]-mean)**2)/total if total > 0 else np.nan
                        s = actual['population'][metric]
                        population_errors[split + '_' + scheme + '_' + label + '_' + metric] = max(
                            close(s['mean'][0], mean, 'Weighted mean', atol=1e-12, rtol=1e-12),
                            close(s['variance'][0], variance, 'Weighted population variance', atol=1e-12, rtol=1e-12))
                        require(s['defined_rows'][0] == finite.sum() and s['undefined_rows'][0] == (~finite).sum(),
                                'Population definition counts changed')
        write(output / ('ARTIFICIAL_POPULATION_' + split + '.json'), report)
        core.save_npz(output / ('ARTIFICIAL_POPULATION_' + split + '.npz'), **saved)
    # A nonuniform set of SOURCE bins is necessary to distinguish the two
    # population tilts. Use concrete row waveforms as an independent oracle.
    parent_bins = np.array([0, 0, 2, 2, 2, 6, 7, 7])
    row_bins = np.repeat(parent_bins, 4)
    waves = 1e3 + rng.normal(size=(32, 256))*.01 + row_bins[:, None]*.2
    means = np.full((8, 256), np.nan); stds = means.copy(); bin_rows = np.bincount(row_bins, minlength=8)
    for b in np.flatnonzero(bin_rows):
        means[b] = waves[row_bins == b].mean(0); stds[b] = waves[row_bins == b].std(0, ddof=0)
    maximum_mix_error = 0.
    for sign in (0., .5, -.5):
        bw = 1 + sign*(2*np.arange(8)-7)/7
        mixture = diag.exact_weighted_waveform(means, stds, bin_rows, bw)
        w = bw[row_bins]; mu = np.average(waves, axis=0, weights=w)
        var = np.average((waves-mu)**2, axis=0, weights=w)
        maximum_mix_error = max(maximum_mix_error, close(mixture['mean'], mu, 'Within+between-bin waveform mean', atol=1e-10, rtol=0),
                                close(mixture['variance'], var, 'Within+between-bin waveform variance', atol=1e-10, rtol=0))
    missing = np.array([1., np.nan, 3., np.nan]); w = np.array([.5, 2., 1.5, 4.])
    weighted = diag.weighted_statistics(missing, w)
    require(weighted['defined_rows'][0] == weighted['undefined_rows'][0] == 2
            and weighted['defined_weight'][0] == 2 and weighted['undefined_weight'][0] == 6,
            'Missing PSD mass silently discarded')
    close(weighted['mean'][0], 2.5, 'Known missing-subset weighted mean', atol=0, rtol=0)
    return dict(fixed128x64_selection_verified=True, fixed32x8_input_and_Q_nodes_verified=True,
                donor_canonical_charge_error=charge_error, Q_node_charge_error=qnode_error,
                selected_peak_morphology_known_answers=True, sample_mean_max_error=mean_error,
                sample_variance_max_error=variance_error, pair_covariance_max_error=covariance_error,
                missing_pair_counts_and_zero_widths_preserved=True,
                weighted_scalar_checks=len(population_errors), weighted_scalar_max_error=max(population_errors.values()),
                direct_waveform_weighted_mixture_max_error=maximum_mix_error,
                missing_scalar_weight_mass_preserved=True, population_response_noise_forwards=0,
                full128x64_covariance_storage_not_executed_by_this_reviewer=True)


def diagnostic_freeze_and_bank(output, cfg, resume_cfg):
    from n01 import diagnose_window008_resume01 as diag
    from n01.response008 import PairedStreams
    policy = diag.fixed_policy(ROOT / 'window008_resume01_diagnostics_policy.json')
    seeds = [policy['seeds']['intervention_' + k] for k in ('response', 'noise005', 'extension')]
    bank, binding = diag.explicit_latent_bank(32, 8, seeds, chunk_rows=128)
    stream = PairedStreams(*seeds)
    expected = {}
    for name, state in stream.state_dict().items():
        require(core.array_sha(bank['initial_' + name]) == core.array_sha(state), 'Addon initial stream state')
    for start in (0, 128):
        response, noise, epsilon = stream.draw(128, 'cpu')
        values = {**{'response_' + k: core._array(v) for k, v in response.items()},
                  **{'noise005_' + k: core._array(v) for k, v in noise.items()}, 'extension_epsilon': core._array(epsilon)}
        for key, value in values.items():
            require(core.array_sha(bank[key][start:start+128]) == core.array_sha(value), 'Addon explicit bank chunk: ' + key)
            expected.setdefault(key, []).append(value)
    for name, state in stream.state_dict().items():
        require(core.array_sha(bank['final_' + name]) == core.array_sha(state), 'Addon final stream state')
    require([(x['start'], x['stop']) for x in binding['chunks']] == [(0, 128), (128, 256)],
            'Fixed intervention latent bank chunking')
    class PrefixThenFailure:
        def assert_frozen(self): pass
        def sample_prefix(self, values):
            return dict(prefix_pre=torch.zeros((5, 100), dtype=torch.float64), scale=torch.ones((5, 1), dtype=torch.float64))
        def extend_prefix(self, *args):
            raise RuntimeError('Artificial extension failure after successful005prefix')
    counts = diag.ForwardCounters()
    try:
        diag._electronics(PrefixThenFailure(), {}, torch.zeros((5, 1, 284)),
                          prefix_counter=lambda rows: counts.add('artificial_failure', 'noise005_forward', rows))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected artificial extension failure')
    require(counts.groups['artificial_failure']['noise005_forward_rows'] == 5
            and counts.groups['artificial_failure']['signal_forward_rows'] == 0,
            'Successful prefix call lost from actual partial accounting')
    destination = output / 'artificial_addon_preregistration_failure'
    failure = ValueError('Artificial missing addon preregistration before any DEV decode')
    with mock.patch.object(diag, 'preregistration_gate', side_effect=failure), \
            mock.patch.object(diag, '_load_dev', side_effect=deny) as dev, \
            mock.patch.object(np, 'load', side_effect=deny) as arrays, \
            mock.patch.object(torch.nn.Module, '_call_impl', side_effect=deny), \
            mock.patch.object(torch.optim.AdamW, 'step', side_effect=deny):
        try:
            diag.diagnose_run(output / 'unexecuted/runs/science', cfg, resume_cfg, ROOT / cfg['dataset'], ROOT / cfg['statistics'],
                cfg['local_training_container'], ROOT / 'data/PARENT_IDENTITIES.npz', source_manifest='unread_main_manifest.json',
                evaluation_policy=ROOT / 'window008_resume01_evaluation_policy.json',
                policy=ROOT / 'window008_resume01_diagnostics_policy.json', addon_source_manifest='unread_addon_manifest.json',
                primary_artifacts=output / 'unread_primary', primary_native_receipt=output / 'unread_primary_native.json',
                replay_native_receipt=output / 'unread_replay_native.json', destination=destination, device='cpu', root=ROOT)
        except ValueError as error:
            require(error is failure, 'Unexpected preregistration failure')
        else:
            raise AssertionError('Missing preregistration allowed diagnostics')
    receipt = read(destination / 'FAILED.json')
    require(dev.call_count == arrays.call_count == 0 and receipt['DEV_numeric_decode_started'] is False
            and receipt['DEV_numeric_decoded'] is False and receipt['actual_partial_counters'] == {},
            'Addon data access escaped failed preregistration')
    return dict(intervention32x8_chunk128_random_bank_exact=True,
                initial_final_stream_states_exact=True, B_C_variants_use_the_same_saved_bank=True,
                successful_prefix_rows_preserved_after_extension_failure=5,
                no_TRAIN_DEV_decoder_called_before_addon_preregistration=True,
                failed_attempt_preserved=True, trained_response_noise_forwards=0)


class ArtificialSignal:
    """Algebraic fixture only, with no neural layers or trained parameters."""
    def __init__(self, arm):
        self.arm = arm
        self.calls = 0
        self.rows = 0

    def signal_forward(self, source, response):
        self.calls += 1; self.rows += len(source)
        index = ARMS.index(self.arm)
        center = source - source[:, :100].mean(1, keepdim=True)
        center[:, :100] = 0.
        qbar = center[:, 110:366].sum(1) * (1 + index * .03)
        latent = response['phys'].reshape(len(source), -1)[:, 0].double()
        factor = torch.ones_like(latent) if self.arm == 'deterministic' else 1 + .015 * latent
        signal = center * (1 + index * .03) * factor[:, None]
        q = signal[:, 110:366].sum(1)
        return dict(signal_pre=signal, qsig=q, qbar=qbar, time_scale=torch.ones_like(q), canonical_charge=q,
                    canonical_charge_identity_residual=torch.zeros_like(q), canonical_local_charge=torch.zeros_like(q),
                    query_outside_carrier=torch.zeros_like(signal, dtype=torch.bool),
                    beyond_extension=torch.zeros_like(signal, dtype=torch.bool))


def artificial_noises(extension, noise_latents, epsilon):
    n = len(epsilon)
    latent = next(iter(noise_latents.values())).reshape(n, -1).double()
    if latent.shape[1] < 100:
        latent = latent.repeat(1, 1 + 100 // latent.shape[1])
    prefix = latent[:, :100] * .0002
    suffix = epsilon.reshape(n, 284).double() * .00015
    return {h: torch.cat((prefix, suffix * (1 + .2 * hi)), 1) for hi, h in enumerate(HYPOTHESES)}


def artificial_data(n=34, nr=40, offset=0):
    t = np.arange(384, dtype=np.float64)
    pulse = np.maximum(0., 1 - np.abs(t - 126) / 10)
    pulse += np.where(t >= 126, .3 * np.exp(-np.maximum(t - 126, 0) / 32), 0.)
    pulse[:100] = 0.
    source = np.array([pulse * (.03 + .0005 * i) for i in range(n)])
    qfloor = 1e-8
    observed = core.crop_existing_values(source, q_floor=qfloor)
    rng = np.random.default_rng(777 + offset)
    real_carrier = np.array([pulse * (.032 + .0007 * (i % n)) for i in range(nr)])
    real_carrier += rng.normal(0., .0002, real_carrier.shape)
    real = core.observe_existing_carrier(real_carrier, already_digitized=False, q_floor=qfloor)
    return dict(source=core._array(observed['waveform256']), source_carrier384=source,
                source_ids=np.arange(offset + 10000, offset + 10000 + n),
                real=core._array(real['waveform256']), real_baseline100=core._array(real['baseline100']),
                real_ids=np.arange(offset + 20000, offset + 20000 + nr),
                real_groups=np.tile(np.arange(4) + offset, nr // 4))


def numerical_stream_regression(output, cfg, adapter):
    """Two independent calls on an algebraic fixture, saving every numeric member."""
    stats = read(ROOT / cfg['statistics'])
    data_stats = read((ROOT / cfg['dataset']).with_name('FIT_DATA_STATISTICS.json'))
    stats['domains'] = data_stats['domains']
    settings = core._fixed_settings(cfg)
    first, second = output / 'artificial_numeric_core', output / 'artificial_numeric_adapter'
    first.mkdir(); second.mkdir()
    objective = core.N01Loss008(loss_tensor_statistics(stats['loss']), seed=cfg['seeds']['train_projection'],
                              evaluation_seed=cfg['seeds']['evaluation_projection']).eval().requires_grad_(False)
    contexts = [objective.create_evaluation_context(), objective.create_evaluation_context()]
    fake_rows, observation_rows = 0, 0
    for destination, owner, context in ((first, core, contexts[0]), (second, adapter, contexts[1])):
        core.save_npz(destination / 'evaluation_projection_bank.npz',
                      **{key: core._array(value) for key, value in context.banks.items()})
        selections, distances = {}, {}
        for split, offset in (('TRAIN', 0), ('DEV', 50000)):
            data = artificial_data(offset=offset)
            selected = np.random.default_rng(cfg['seeds']['selected_' + split]).permutation(34)[:12]
            a, b = owner.balanced_real_domains(data['real_groups'], 12, cfg['seeds']['real_' + split])
            models = {arm: ArtificialSignal(arm) for arm in ARMS}
            with mock.patch.object(core, 'fixed_noises', side_effect=artificial_noises), \
                    mock.patch.object(torch.optim.AdamW, 'step', side_effect=deny):
                generated = owner.generate_split(split, data, models, None, stats, cfg, settings, destination, selected, 'cpu')
            require(all(model.rows == 136 and model.calls == 2 for model in models.values()),
                    'Artificial generation changed row/chunk accounting')
            fake_rows += sum(model.rows for model in models.values()); observation_rows += 136 * 9
            parent = np.repeat(np.arange(34), 4)
            require(np.array_equal(generated['parents'], parent), 'Numerical parent-major four-draw population')
            generation = read(destination / (split + '_generation.json'))
            require([(chunk['start'], chunk['stop']) for chunk in generation['chunks']] == [(0, 128), (128, 136)]
                    and generation['rows_removed'] == generation['extra_sample_forwards'] == 0,
                    'Numerical stream chunk or population change')
            report, payload = owner.parent_bootstrap(generated, stats, cfg, settings, split)
            core.save_json(destination / (split + '_PARENT_BOOTSTRAP.json'), report)
            core.save_npz(destination / (split + '_parent_bootstrap.npz'), **payload)
            distances[split] = {arm: {h: owner.distance_receipt(objective, generated['matched'][arm][h], data['real'][a],
                generated['matched_baseline'], data['real_baseline100'][a], context) for h in HYPOTHESES} for arm in ARMS}
            distances[split]['REAL_REAL'] = owner.distance_receipt(objective, data['real'][b], data['real'][a],
                                                                  data['real_baseline100'][b], data['real_baseline100'][a], context)
            matched = {arm + '_' + h: generated['matched'][arm][h] for arm in ARMS for h in HYPOTHESES}
            matched.update(generated_baseline100=generated['matched_baseline'], SOURCE_parent_ids=data['source_ids'][selected],
                           REAL_A=data['real'][a], REAL_B=data['real'][b], REAL_A_baseline100=data['real_baseline100'][a],
                           REAL_B_baseline100=data['real_baseline100'][b], REAL_A_original_ids=data['real_ids'][a],
                           REAL_B_original_ids=data['real_ids'][b])
            core.save_npz(destination / (split + '_matched_waveforms.npz'), **matched)
            selections[split] = dict(SOURCE_rows=selected, REAL_A_rows=a, REAL_B_rows=b)
        core.save_json(destination / 'MATCHED_SELECTIONS.json', dict(selections=selections, fixture_only=True))
        core.save_json(destination / 'DISTRIBUTION_DISTANCES.json', dict(distances=distances, fixture_only=True,
                                                                      projection_context=context.metadata()))
    require({path.name for path in first.glob('*.npz')} == old_compare.NPZ_NAMES, 'Actual numeric fixture 13 NPZ inventory')
    exact_members = {}
    for name in sorted(old_compare.NPZ_NAMES):
        with np.load(first / name, allow_pickle=False) as a, np.load(second / name, allow_pickle=False) as b:
            require(sorted(a.files) == sorted(b.files), 'Numeric fixture member inventory: ' + name)
            exact_members[name] = {}
            for member in a.files:
                x, y = a[member], b[member]
                require(x.dtype == y.dtype and x.shape == y.shape and core.array_sha(x) == core.array_sha(y)
                        and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x, y)),
                        'Numeric fixture exact replay differs: ' + name + '/' + member)
                exact_members[name][member] = core.array_sha(x)
    semantic_files = {}
    for name in sorted(path.name for path in first.glob('*.json')):
        a, b = read(first / name), read(second / name)
        if name.endswith('_generation.json'):
            a.pop('UTC'); b.pop('UTC')
        require(core._json_sha(a) == core._json_sha(b), 'Numeric JSON semantics: ' + name)
        semantic_files[name] = core._json_sha(a)
    evidence = dict(NPZ_files=13, NPZ_members=sum(len(x) for x in exact_members.values()),
                    arrays_sha256=exact_members, JSON_semantic_sha256=semantic_files,
                    fixed_draws_per_parent=4, chunk_rows=128, artificial_signal_rows=fake_rows,
                    artificial_observation_rows=observation_rows, trained_scientific_forwards=0,
                    all_chunk_random_hashes_exact=True, all_stream_states_exact=True,
                    projection_bank_distances_bootstrap_exact=True)
    write(output / 'ARTIFICIAL_NUMERIC_EQUIVALENCE.json', evidence)
    return {key: value for key, value in evidence.items() if key not in ('arrays_sha256', 'JSON_semantic_sha256')}


def bootstrap_oracle():
    n = 11
    parents = np.repeat(np.arange(n), 4); draw = np.tile(np.arange(4), n)
    qbase = 8 + parents * 3.2 + draw * .67
    pbase = .15 + parents * .03 + draw * .007
    stats = dict(loss=dict(q_reference=20., q_floor=1e-8,
                          charge_bin_boundaries=np.log(np.array([10, 15, 20, 25, 30, 35, 40]) / 20).tolist()))
    observations = {a: {} for a in ARMS}
    for ai, arm in enumerate(ARMS):
        for hi, hypothesis in enumerate(HYPOTHESES):
            q = (1 + hi * .2) * qbase + hi * .7
            psd = (1 + hi * .1) * pbase + hi * .013
            if arm == 'full':
                q, psd = 2 * q + 17, 3 * psd + .25
            elif arm == 'simple':
                q, psd = .5 * q - 13, .2 * psd + .1
            observations[arm][hypothesis] = dict(Q=q, PSD=psd)
    source_bins = np.repeat(np.array([0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 6]), 4)
    generated = dict(parents=parents, source_bins=source_bins, observables=observations,
                     real=dict(Q=np.arange(37.), PSD=np.arange(37.) / 37))
    cfg = {'seeds': {'bootstrap_TRAIN': 1123}}
    report, payload = core.parent_bootstrap(generated, stats, cfg,
                                         dict(bootstrap_draws=1000, minimum_parents_per_bin=32), 'TRAIN')
    means = np.full((1001, 2, 3, 3, 8, 2), np.nan); stds = np.full_like(means, np.nan)
    support = np.zeros((2, 3, 3, 8), np.int64); counts = np.zeros_like(support)
    rng = np.random.default_rng(1123)
    row_samples = [np.arange(n * 4)] + [(rng.integers(0, n, n)[:, None] * 4 + np.arange(4)).ravel()
                                      for _ in range(1000)]
    for ai, arm in enumerate(ARMS):
        for hi, hypothesis in enumerate(HYPOTHESES):
            values = observations[arm][hypothesis]
            q = values['Q']
            coordinate = np.where(q >= 1e-8, np.log(np.maximum(q, 1e-8) / 20),
                                  np.log(1e-8 / 20) + np.arcsinh((q - 1e-8) / 1e-8))
            output_bins = np.searchsorted(stats['loss']['charge_bin_boundaries'], coordinate, side='left')
            for si, membership in enumerate((output_bins, source_bins)):
                for bi in range(8):
                    support[si, ai, hi, bi] = len(np.unique(parents[membership == bi]))
                    counts[si, ai, hi, bi] = int((membership == bi).sum())
                    for ri, sampled in enumerate(row_samples):
                        selected = sampled[membership[sampled] == bi]
                        if not len(selected):
                            continue
                        for mi, metric in enumerate(core.METRICS):
                            x = values[metric][selected]
                            means[ri, si, ai, hi, bi, mi] = x.mean()
                            stds[ri, si, ai, hi, bi, mi] = 0. if x.min() == x.max() else x.std(ddof=0)
    errors = {name: close(actual, expected, name) for name, actual, expected in (
        ('point_mean', report['point_mean'], means[0]), ('point_std', report['point_std'], stds[0]),
        ('all_1000_mean', payload['mean'], means[1:]), ('all_1000_std', payload['std'], stds[1:]))}
    require(np.array_equal(report['parents_per_bin'], support)
            and np.array_equal(report['rows_per_bin'], counts)
            and np.array_equal(report['supported'], support >= 32), 'Bootstrap support/row counters')
    for comparator in core.COMPARATORS:
        index = ARMS.index(comparator)
        denominator = stds[1:, :, index]
        ratio = np.divide(stds[1:, :, 1], denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
        errors[comparator + '_ratio'] = close(payload[comparator + '_std_full_over_reference'], ratio, comparator + ' ratio')
        require(np.isnan(payload[comparator + '_std_full_over_reference'][denominator == 0]).all(),
                'Zero reference width ratio must remain undefined')
    return dict(parents=11, draws_per_parent=4, bootstrap_replicates=1000,
                scalar_mean_std_positions=576576, empty_positions=int(np.isnan(stds[1:]).sum()),
                exact_zero_width_positions=int((stds[1:] == 0).sum()), maximum_errors=errors,
                oracle='Explicitly repeat complete four-row parents, then direct NumPy population moments')


def run(output, metadata_run):
    output = Path(output).resolve(); metadata_run = Path(metadata_run).resolve()
    if os.name == 'nt':
        require(output.drive.upper() in ('E:', 'D:'), 'Review output must stay on E/D')
    require(not output.exists(), 'Fresh review directory required')
    output.mkdir(parents=True)
    began = time.monotonic()
    torch.set_num_threads(2)
    adapter = importlib.import_module('n01.evaluate_window008_resume01')
    cfg, resume_cfg = read(ROOT / 'window008_config.json'), read(ROOT / 'window008_resume01_config.json')
    import window008_resume01_evaluation_common as ops
    reviewed = sorted(set(ops.SOURCE_FILES) | {'src/n01/evaluate_window008.py'})
    initial_sources = {relative: sha(ROOT / relative) for relative in reviewed if (ROOT / relative).is_file()}
    report = dict(schema='n01/window008-resume01-independent-evaluation-review/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), status='review_running',
                  reviewer='independent_subagent', checks=[], artificial_only_scientific_arrays=True,
                  scientific_waveform_NPZ_decodes=0, trained_response_noise_forwards=0,
                  scientific_optimizer_updates=0, GPU_used=False, actual_job_launches=0,
                  actual_metadata_root=str(metadata_run), scientific_status='response_match_not_established')
    def test(name, action):
        try:
            evidence = action()
            report['checks'].append(dict(name=name, passed=True, evidence=evidence))
        except Exception as error:
            report['checks'].append(dict(name=name, passed=False, error_type=type(error).__name__, error=str(error)))
    test('all_declared_additive_evaluation_sources_present', lambda: require(set(initial_sources) == set(reviewed),
                                                                          'A declared source/policy/doc is still missing'))
    test('original_numerics_and_helper_bytes_unchanged', lambda: source_contract(adapter))
    test('actual_hash_bound_three_generation_endpoint_metadata', lambda: actual_metadata(metadata_run, cfg, resume_cfg, adapter))
    test('native_endpoint_metadata_mutations_and_no_DEV_before_freeze', lambda: adapter_rejections(output, metadata_run, cfg, resume_cfg, adapter))
    test('all_13_NPZ_66_artificial_members_strict_replay_and_drift_rejection', lambda: helper_replay_fixture(output))
    test('versioned_operations_native_roles_and_unrelaxed_replay', lambda: operations_contract(output, cfg, resume_cfg))
    test('diagnostic_native_roles_budget_inventory_and_population_no_forward', lambda: diagnostic_operations(output, cfg))
    test('every_numeric_artifact_member_chunk_stream_distance_bootstrap_exact', lambda: numerical_stream_regression(output, cfg, adapter))
    test('addon_fixed_interventions_morphology_covariance_and_population_oracles', lambda: diagnostic_math(output))
    test('addon_preregistration_random_bank_and_actual_partial_counter_boundaries',
         lambda: diagnostic_freeze_and_bank(output, cfg, resume_cfg))
    test('paired_parent_bootstrap_independent_direct_resampling', bootstrap_oracle)
    report['reviewed_source_sha256'] = {relative: sha(ROOT / relative) for relative in reviewed if (ROOT / relative).is_file()}
    test('reviewed_sources_stable_during_all_checks', lambda: require(report['reviewed_source_sha256'] == initial_sources,
                                                                    'A reviewed source changed during the checks'))
    report['runtime'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, device='cpu',
                             cpu_threads_final=torch.get_num_threads(), CUDA_initialized=torch.cuda.is_initialized(),
                             pid=os.getpid(), parent_pid=os.getppid())
    require(not torch.cuda.is_initialized(), 'Independent review unexpectedly initialized CUDA')
    report['passed'] = sum(check['passed'] for check in report['checks']); report['total'] = len(report['checks'])
    report['status'] = 'passed_independent_scoped_review' if report['passed'] == report['total'] else 'failed_independent_scoped_review'
    report['elapsed_seconds'] = time.monotonic() - began
    write(output / 'REVIEW.json', report)
    require(report['status'] == 'passed_independent_scoped_review', 'Review failed; see ' + str(output / 'REVIEW.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='Fresh E/D receipt and artificial fixture directory')
    parser.add_argument('--metadata-run', default=str(ROOT / 'runs' / RUN_ID / 'runs/science'))
    args = parser.parse_args()
    result = run(args.out, args.metadata_run)
    print(json.dumps(dict(status=result['status'], passed=result['passed'], total=result['total']), allow_nan=False))


if __name__ == '__main__':
    main()
