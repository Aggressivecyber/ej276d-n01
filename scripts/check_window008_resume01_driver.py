"""CPU metadata/artificial-JSON QA for the resume01 driver; zero forwards/steps.

Reads only frozen JSON/source bytes and five CPU legacy signal checkpoints.
Never opens a waveform NPZ, imports a failure checkpoint, or dispatches science.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from n01.data import sha256, write_json
from n01.train import utc
from n01 import train_window008_resume01 as driver


def run(output):
    output = Path(output).resolve()
    if os.name == 'nt' and output.drive.upper() not in ('E:', 'D:'):
        raise ValueError('CPU QA output must stay on E/D')
    if output.exists():
        raise FileExistsError('Fresh driver QA output required')
    output.mkdir(parents=True)
    checks = []
    loaded_checkpoints = []

    def passed(name):
        checks.append(dict(name=name, passed=True))

    def reject(name, action, expected=None):
        try:
            action()
        except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
            if expected is not None and expected not in str(error):
                raise AssertionError(name + ': unexpected error ' + str(error)) from error
            checks.append(dict(name=name, passed=True, rejection=str(error)))
        else:
            raise AssertionError(name + ': accepted an invalid artificial mutation')

    base = driver.read_json(ROOT / 'window008_config.json')
    fix = driver.read_json(ROOT / 'window008_fix01_config.json')
    resume = driver.read_json(ROOT / 'window008_resume01_config.json')
    documents = {name: driver.read_json(ROOT / row['path'])
                 for name, row in resume['reference_inputs'].items() if row['path'].endswith('.json')}
    driver.validate_plan(base, fix, resume)
    passed('registered_resume_plan_preserves_full6000_det19k_simplefresh')

    for key in driver.FIXED_COUNTS:
        bad = copy.deepcopy(resume)
        bad[key] += 1
        reject('reject_budget_' + key, lambda bad=bad: driver.validate_plan(base, fix, bad), 'update budgets')
    for key, value in (
        ('schema', 'n01/window008-fix01-config/v1'),
        ('run_id', 'window008_fix01_20261013'),
        ('execution_arms', ['deterministic', 'full', 'simple']),
        ('model_loss_data_seed_readout_changes', True),
        ('model_implementation', 'n01.model008.N01Response008'),
        ('DEV_waveforms_used_for_training', True),
        ('noise_updates', 1), ('population_updates', 1), ('C2ST_updates', 1),
        ('wall_limit_s', 43200), ('base_configuration_sha256', 'f' * 64),
    ):
        bad = copy.deepcopy(resume)
        bad[key] = value
        reject('reject_plan_' + key, lambda bad=bad: driver.validate_plan(base, fix, bad))
    for key, value in (('G_updates', 6398), ('D_updates_per_hypothesis', 3398),
                       ('stage', 'mean_response'), ('next_G_update', 1),
                       ('first_update_audit', False), ('first_update_same_version_replay', False),
                       ('reference', 'legacy_failure_checkpoint')):
        bad = copy.deepcopy(resume)
        bad['resume'][key] = value
        reject('reject_import_policy_' + key, lambda bad=bad: driver.validate_plan(base, fix, bad), 'full6000 policy')
    bad = copy.deepcopy(resume)
    bad['reference_inputs']['fix01_full_periodic_06000']['sha256'] = 'f' * 64
    reject('reject_unregistered_resume_checkpoint_SHA', lambda: driver.validate_plan(base, fix, bad), 'registered complete')
    bad = copy.deepcopy(resume)
    bad['reference_inputs']['common_initial_state']['sha256'] = 'f' * 64
    reject('reject_replaced_old_common_initializer', lambda: driver.validate_plan(base, fix, bad), 'Original reference')
    bad = copy.deepcopy(resume)
    del bad['reference_inputs']['fix01_user_stop_verification_local']
    reject('reject_missing_local_raw_pause_hash_reference', lambda: driver.validate_plan(base, fix, bad), 'reference inventory')

    original_chain = driver.validate_legacy_receipts(base, fix, documents)
    native_chain = driver.validate_stop_chain(base, fix, resume, documents, original_chain)
    if (original_chain['native_exit_code'] != 1 or native_chain['fix01_native_exit_code'] != 1
            or native_chain['old_jobs_native_success'] is not False):
        raise AssertionError('Both historical native1 roles must remain preserved')
    passed('original_numerical_failure_and_fix01_user_SIGTERM_native1_chains')
    if (documents['fix01_user_pause']['stop_receipt_sha256']
            == resume['reference_inputs']['fix01_user_stop_verification']['sha256']):
        raise AssertionError('Expected the observed remote/local serialization distinction')
    passed('remote_local_stop_JSON_equal_with_distinct_raw_hashes_and_full_pause_chain')

    mutations = [
        ('fix01_native_exit', 'exit_code', 0),
        ('fix01_native_exit', 'reason', 'claimed_success'),
        ('fix01_native_exit', 'child_pid', 1),
        ('fix01_binding', 'parent_pid', 1),
        ('fix01_binding', 'dataset_sha256', 'f' * 64),
        ('fix01_binding', 'DEV_waveforms_read', True),
        ('fix01_training_inputs', 'SOURCE_rows', 6592),
        ('fix01_training_inputs', 'source_count_weights', 'reweighted'),
        ('fix01_training_inputs', 'noise_updates', 1),
        ('fix01_resume_validation', 'G_updates', 6000),
        ('fix01_resume_validation', 'first_update_requires_same_version_replay', False),
        ('fix01_legacy_validation', 'optimizer_updates', 1),
        ('fix01_failure', 'new_execution_G_updates', 3000),
        ('fix01_failure', 'new_execution_D_updates', 9000),
        ('fix01_failure', 'historical_actual_G_updates', 25000),
        ('fix01_failure', 'current_G_updates', 6000),
        ('fix01_failure', 'automatic_resume_allowed', True),
        ('fix01_failure', 'error', 'numerical_failure'),
        ('fix01_user_stop_request', 'trainer_pid', 1),
        ('fix01_user_stop_verification', 'signal_sent', None),
        ('fix01_user_stop_verification', 'trainer_absent', False),
        ('fix01_user_stop_verification', 'owned_processes_after', [{'pid': 1}]),
        ('fix01_user_stop_verification_local', 'status', 'claimed_success'),
        ('fix01_user_stop_collection', 'frozen_manifest_sha256', 'f' * 64),
        ('fix01_user_stop_collection', 'all_transferred_member_hashes_verified', False),
        ('fix01_user_stop_collection', 'new_training_updates_during_collection', 1),
        ('fix01_user_pause', 'stop_receipt_sha256', 'f' * 64),
        ('fix01_user_pause', 'collection_receipt_sha256', 'f' * 64),
        ('fix01_user_pause', 'owned_trainer_and_supervisor_absent', False),
        ('fix01_user_pause', 'actual_new_G_updates', 3000),
        ('fix01_user_pause', 'native_exit_code', 0),
        ('fix01_user_pause', 'UTC', '2026-10-13T00:00:00+00:00'),
    ]
    for name, key, value in mutations:
        bad = copy.deepcopy(documents)
        bad[name][key] = value
        reject('reject_stop_chain_' + name + '_' + key,
               lambda bad=bad: driver.validate_stop_chain(base, fix, resume, bad, original_chain))
    bad = copy.deepcopy(documents)
    bad['fix01_launch']['command'][bad['fix01_launch']['command'].index('--out') + 1] = '/unbound/science'
    reject('reject_old_launch_output_root_rebinding',
           lambda: driver.validate_stop_chain(base, fix, resume, bad, original_chain), 'command flags')
    bad = copy.deepcopy(documents)
    bad['fix01_user_stop_collection']['files']['runs/science/train/full/latest.pt']['sha256'] = 'f' * 64
    reject('reject_collection_boundary_checkpoint_digest_mutation',
           lambda: driver.validate_stop_chain(base, fix, resume, bad, original_chain), 'collected raw')
    bad = copy.deepcopy(documents)
    bad['fix01_user_pause']['periodic_checkpoint']['D_updates']['fast'] = 3001
    reject('reject_pause_partial_D_as_complete_boundary',
           lambda: driver.validate_stop_chain(base, fix, resume, bad, original_chain), 'pause/full6000')

    fixture = output / 'artificial_paths'
    fixture.mkdir()
    small = fixture / 'fixed.json'
    small.write_text('{"fixed": true}\n', encoding='utf8')
    driver.verify_files(fixture, {'fixed.json': sha256(small)})
    passed('artificial_bound_file_hash_accepts_original_bytes')
    small.write_text('{"fixed": false}\n', encoding='utf8')
    reject('reject_bound_file_mutation', lambda: driver.verify_files(fixture, {'fixed.json': 'f' * 64}), 'Bound file changed')
    for path in ('../outside', '..\\outside', '/home/outside', 'C:\\outside', 'C:relative', '', '.'):
        reject('reject_path_' + repr(path), lambda path=path: driver.bound(fixture, path))
    duplicate = fixture / 'duplicate.json'
    duplicate.write_text('{"G_updates":6000,"G_updates":6398}', encoding='utf8')
    reject('reject_duplicate_JSON_metadata_keys', lambda: driver.read_json(duplicate), 'Duplicate JSON key')

    # Module closure is checked against both frozen generations and both new modules.
    original_manifest = driver.read_json(ROOT / resume['legacy_source_manifest'])
    fix_manifest = driver.read_json(ROOT / resume['fix01_source_manifest'])
    for label, manifest in (('original68', original_manifest), ('fix01_114', fix_manifest)):
        driver.verify_files(ROOT, manifest['files'])
        passed('immutable_source_' + label + '_all_hashes_unchanged')
    prior_modules = {name for manifest in (original_manifest, fix_manifest) for name in manifest['files']
                     if name.startswith('src/n01/') and name.endswith('.py')}
    new_inventory = dict(files={name: sha256(ROOT / name) for name in
        ('src/n01/response008_resume01.py', 'src/n01/train_window008_resume01.py')})
    staged_modules = prior_modules | set(new_inventory['files'])
    staged = output / 'artificial_staged_inventory'
    for name in sorted(staged_modules):
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Artificial module inventory only; no executable implementation.\n', encoding='utf8')
    driver.verify_module_inventory(staged, original_manifest, fix_manifest, new_inventory)
    passed('isolated_scientific_staged_36_module_inventory_closes_both_frozen_generations_and_new2')
    missing = copy.deepcopy(new_inventory)
    del missing['files']['src/n01/response008_resume01.py']
    reject('reject_new_engine_missing_from_module_inventory',
           lambda: driver.verify_module_inventory(staged, original_manifest, fix_manifest, missing), 'module inventory')
    unknown = copy.deepcopy(new_inventory)
    unknown['files']['src/n01/unbound_module.py'] = '0' * 64
    reject('reject_inventory_for_nonexistent_module',
           lambda: driver.verify_module_inventory(staged, original_manifest, fix_manifest, unknown), 'module inventory')
    (staged / 'src/n01/unbound_module.py').write_text('# Artificial extra module; must be rejected.\n', encoding='utf8')
    reject('reject_staged_root_with_real_extra_unfrozen_module',
           lambda: driver.verify_module_inventory(staged, original_manifest, fix_manifest, new_inventory), 'module inventory')
    local_modules = {path.relative_to(ROOT).as_posix() for path in (ROOT / 'src/n01').glob('*.py')}
    local_extras = sorted(local_modules - staged_modules)
    for name, row in resume['reference_inputs'].items():
        path = ROOT / row['path']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise AssertionError('Historical raw reference changed: ' + name)
    passed('all_26_historical_references_bytes_and_SHA256')

    real_torch_load = torch.load
    allowed = {Path(fix['reference_inputs'][name]['path']).name for name in
               ('common_initial_state', 'full_stage_03000', 'deterministic_stage_03000', 'full_initial', 'deterministic_final')}
    def checkpoint_metadata(path, *args, **kwargs):
        if Path(path).name not in allowed or kwargs.get('map_location') != 'cpu':
            raise AssertionError('Unregistered/non-CPU checkpoint read in metadata QA: ' + str(path))
        loaded_checkpoints.append(Path(path).relative_to(ROOT).as_posix())
        return real_torch_load(path, *args, **kwargs)
    with mock.patch.object(torch, 'load', side_effect=checkpoint_metadata):
        statistics = driver.read_json(ROOT / base['statistics'])
        common, validation = driver.validate_legacy_states(ROOT, base, fix, statistics, documents)
        for key, value in validation.items():
            if documents['fix01_legacy_validation'].get(key) != value:
                raise AssertionError('Original CPU validation changed: ' + key)
        del common
    if len(loaded_checkpoints) != 5:
        raise AssertionError('Unexpected legacy CPU checkpoint metadata inventory')
    passed('original_five_CPU_checkpoint_metadata_revalidated_without_forwards_or_failure_load')

    det = driver.retained_endpoint(fix, documents, validation)
    identity = dict(run_id=driver.RUN_ID, config_sha256=sha256(ROOT / 'window008_resume01_config.json'),
        source_manifest_sha256='0' * 64,
        base_configuration_sha256=resume['base_configuration_sha256'],
        base_source_manifest_sha256=resume['legacy_source_manifest_sha256'],
        fix01_configuration_sha256=resume['fix01_configuration_sha256'],
        fix01_source_manifest_sha256=resume['fix01_source_manifest_sha256'])
    rows = [det]
    for arm, inherited, new_g, new_d, origin in (
        ('full', 6000, 13000, 13000, 'fix01_periodic6000_resume'),
        ('simple', 0, 19000, 16000, 'fresh_common_initialization')):
        rows.append(dict(arm=arm, origin=origin, G_updates=19000,
            D_updates={h: 16000 for h in driver.HYPOTHESES}, inherited_G_updates=inherited,
            new_G_updates=new_g, new_D_updates={h: new_d for h in driver.HYPOTHESES},
            coverage=dict(source_unique=6593, real_unique=21047), stage='low_learning_rate',
            checkpoint_schema='response008_resume01/v1', model_implementation=driver.MODEL_IMPLEMENTATION,
            endpoint_path='runs/science/train/' + arm + '/final.pt', final_sha256='0' * 64,
            common_initial_state_sha256=validation['common_initial_state_sha256'],
            noise005_state_sha256=validation['noise005_state_sha256'], recovery=identity,
            lineage=dict(source_checkpoint_sha256=driver.FULL6000_SHA256 if arm == 'full' else None)))
    final_counts = driver.validate_endpoints(rows)
    if (final_counts['historical_actual_G_updates'], final_counts['historical_actual_D_updates'],
            final_counts['global_N01_actual_G_updates']) != (57667, 146001, 157667):
        raise AssertionError('History/global final ledger is incorrect')
    passed('final_32k87k_new_25k57k_inherited_57k144k_effective_57667G146001D_historical')
    start = driver.update_accounting([det])
    if (start['new_execution_G_updates'], start['inherited_effective_G_updates'],
        start['effective_comparison_G_updates'], start['historical_actual_G_updates'],
        start['global_N01_G_updates_before_resume']) != (0, 25000, 25000, 25667, 125667):
        raise AssertionError('Start accounting conflated new, inherited, effective and historical updates')
    passed('initial_ledger_separates_retained_valid_endpoints_and_discarded_actual_updates')
    unbound = driver.update_accounting([], legacy_validated=False)
    if unbound['new_execution_G_updates'] != 0 or any(unbound[key] is not None for key in
            ('inherited_effective_G_updates', 'historical_actual_G_updates', 'global_N01_actual_G_updates')):
        raise AssertionError('Unvalidated history must remain unknown')
    passed('pre_binding_failure_does_not_claim_validated_inherited_or_historical_updates')
    partial = SimpleNamespace(arm='full', update=6000,
        d_updates={'fast': 3001, 'slow': 3000, 'common_level': 3000},
        new_G_updates=0, new_D_updates={'fast': 1, 'slow': 0, 'common_level': 0})
    counts = driver.update_accounting([det], partial)
    if (counts['new_execution_G_updates'], counts['new_execution_D_updates'],
            counts['historical_actual_D_updates']) != (0, 1, 59002):
        raise AssertionError('Partial D step was omitted or inferred from G updates')
    passed('partial_D_without_completed_G_step_has_exact_actual_D_accounting')
    running = SimpleNamespace(arm='simple', new_G_updates=1,
        new_D_updates={h: 0 for h in driver.HYPOTHESES})
    counts = driver.update_accounting(rows[:2], running)
    if (counts['new_execution_G_updates'], counts['new_execution_D_updates']) != (13001, 39000):
        raise AssertionError('Completed full and active fresh simple accounting is incorrect')
    passed('simple_after_completed_full_adds_new_counters_without_double_counting_import')
    for index, key, value in ((0, 'G_updates', 18999), (1, 'new_G_updates', 16000),
                              (2, 'inherited_G_updates', 6000), (1, 'checkpoint_schema', 'response008_fix01/v1'),
                              (0, 'model_implementation', driver.MODEL_IMPLEMENTATION),
                              (2, 'noise005_state_sha256', 'f' * 64)):
        bad = copy.deepcopy(rows)
        bad[index][key] = value
        reject('reject_endpoint_' + str(index) + '_' + key, lambda bad=bad: driver.validate_endpoints(bad))
    reject('reject_missing_simple_endpoint', lambda: driver.validate_endpoints(rows[:2]), 'arm inventory')
    completion = driver.completion_payload(base, resume, identity, rows, {}, native_chain, '0' * 64, 0.)
    if (completion['awaiting_own_supervisor_native_exit'] is not True
            or completion['native_training_success'] is not False
            or completion['evaluation_complete'] is not False
            or completion['evaluation_adapter_status'] != 'pending_versioned_resume01_adapter'
            or completion['scientific_status'] != 'response_match_not_established'):
        raise AssertionError('Completion metadata must preserve pending native exit/evaluation/science boundaries')
    passed('artificial_completion_payload_cannot_claim_native_success_evaluation_or_response_match')
    bad = copy.deepcopy(rows)
    bad[1]['lineage']['source_checkpoint_sha256'] = 'f' * 64
    reject('reject_completed_full_from_unbound_source_checkpoint',
        lambda: driver.completion_payload(base, resume, identity, bad, {}, native_chain, '0' * 64, 0.), 'resume-source')

    failure_dir = output / 'artificial_partial_D_failure'
    failure_dir.mkdir()
    def forensic_checkpoint(path, *, purpose):
        if purpose != 'failure':
            raise AssertionError('Forensic fake checkpoint called with a non-failure purpose')
        Path(path).write_bytes(b'artificial metadata only; no signal checkpoint\n')
    partial.checkpoint = forensic_checkpoint
    partial.last_context = dict(artificial_metadata_only=True, completed_G_step=False)
    failure = driver.save_failure(failure_dir, RuntimeError('artificial partial D interruption'), [det], partial,
        began=time.monotonic(), legacy_validated=True, native_chain=native_chain)
    if (failure['new_execution_D_updates'] != 1 or failure['new_execution_G_updates'] != 0
            or failure['checkpoint_may_be_mid_step'] is not True
            or failure['automatic_resume_allowed'] is not False
            or failure['native_training_success'] is not False
            or failure['legacy_native_chain']['fix01_native_exit_code'] != 1):
        raise AssertionError('Failure metadata omitted partial actual counts or historical native1')
    passed('driver_failure_preserves_partial_D_actual_counters_forensics_and_two_native1_chains')

    argv = []
    for flag in ('config', 'resume-config', 'source-manifest', 'dataset', 'statistics', 'container', 'identities'):
        argv += ['--' + flag, str(output / ('nonexistent_' + flag))]
    attempt = output / 'artificial_binding_failure'
    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        reject('driver_rejects_unbound_start_before_numeric_decode',
               lambda: driver.main(argv + ['--out', str(attempt)]))
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    first_failure_sha = sha256(attempt / 'FAILURE.json')
    saved_failure = driver.read_json(attempt / 'FAILURE.json')
    if saved_failure['legacy_references_validated'] is not False or saved_failure['new_execution_G_updates'] != 0:
        raise AssertionError('Binding failure must not claim any scientific updates')
    reject('driver_rejects_existing_failed_output_and_preserves_first_attempt',
           lambda: driver.main(argv + ['--out', str(attempt)]), 'Fresh resume01 output')
    if sha256(attempt / 'FAILURE.json') != first_failure_sha:
        raise AssertionError('Existing attempt was overwritten')
    passed('first_failed_artificial_output_hash_unchanged_after_repeat_attempt')

    receipt = dict(schema='n01/window008-resume01-driver-CPU-QA/v1', UTC=utc(), status='passed',
        passed=len(checks), total=len(checks), checks=checks,
        metadata_config_sha256=sha256(ROOT / 'window008_resume01_config.json'),
        driver_sha256=sha256(ROOT / 'src/n01/train_window008_resume01.py'),
        QA_script_sha256=sha256(Path(__file__)), old_original_frozen_files_verified=68,
        old_fix01_frozen_files_verified=114, historical_reference_files_verified=26,
        new_scientific_module_inventory=new_inventory['files'],
        staged_scientific_runtime_module_count=len(staged_modules),
        artificial_staged_module_inventory_only=True,
        local_worktree_extra_modules_excluded_from_scientific_stage=local_extras,
        legacy_CPU_checkpoint_metadata_loaded=loaded_checkpoints,
        full6000_checkpoint_role='SHA_only_in_driver_QA; engine_QA_validates_CPU_state',
        artificial_JSON_failure_fixtures=True, artificial_signal_forwards=0,
        scientific_model_forwards=0, optimizer_updates=0,
        engineering_G_updates=0, engineering_D_updates=0,
        actual_SCI_DEV_arrays_decoded=0, DEV_numeric_decodes=0,
        failure_checkpoints_loaded=0, GPU_used=False, actual_native_training_dispatched=False,
        historical_native_exit_codes={'original': 1, 'fix01_user_stop': 1},
        scientific_status='response_match_not_established')
    write_json(output / 'ENGINEERING_RECEIPT.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'passed', 'total',
        'scientific_model_forwards', 'optimizer_updates', 'actual_SCI_DEV_arrays_decoded',
        'failure_checkpoints_loaded', 'GPU_used', 'actual_native_training_dispatched')}))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    # These guards apply to every metadata and artificial-driver check.
    with mock.patch.object(np, 'load', side_effect=AssertionError('Waveform NPZ decode forbidden in driver metadata QA')), \
         mock.patch.object(torch.nn.Module, '_call_impl', side_effect=AssertionError('Model forward forbidden in driver metadata QA')), \
         mock.patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Optimizer step forbidden in driver metadata QA')):
        run(args.out)


if __name__ == '__main__':
    main()
