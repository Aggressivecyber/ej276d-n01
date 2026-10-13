"""Small artificial CPU fixtures for the additive evaluation operations layer.

No n01 scientific module is imported, no original array is decoded, and no job
is dispatched. Fixture receipts are explicitly artificial and remain local.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from unittest import mock
import tarfile

import numpy as np

import freeze_window008_evaluation as freezer
import stage_window008_evaluation as staging
import supervise_window008_evaluation as supervisor
import window008_evaluation_common as common
from verify_window008_evaluation import NPZ_NAMES, JSON_IGNORED, array_sha, compare_artifacts


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def rebind_artifacts(path):
    summary = common.read(path / 'SUMMARY.json')
    summary['artifact_sha256'] = {item.name: common.sha(item) for item in path.iterdir()
                                 if item.name != 'SUMMARY.json' and item.is_file()}
    save(path / 'SUMMARY.json', summary)


def array_write(path, values, compressed=True):
    with path.open('wb') as stream:
        (np.savez_compressed if compressed else np.savez)(stream, **values)


def build_artifacts(path):
    path.mkdir()
    n, rows = 34, 136
    parents = np.repeat(np.arange(n, dtype=np.int64), 4)
    numeric = np.asarray([0., 1.25, np.nan, -3.5], dtype=np.float64)
    for name in sorted(NPZ_NAMES):
        values = dict(numeric=numeric, boolean=np.asarray([True, False]),
                      identities=np.asarray(['fixture-a', 'fixture-b']), scalar=np.asarray(17, dtype=np.int64))
        if name.endswith('_observables.npz'):
            values.update(source_parent_ids=np.arange(100, 100 + n, dtype=np.int64),
                source_parent_row=parents, draw=np.tile(np.arange(4, dtype=np.int64), n))
        if name.endswith('_stream_states.npz'):
            values.update(initial_response=np.arange(32, dtype=np.uint8), final_response=np.arange(32, 64, dtype=np.uint8))
        if name.endswith('_parent_bootstrap.npz'):
            values.update(bootstrap_mean=np.arange(240, dtype=np.float64).reshape(5, 2, 3, 8),
                          bootstrap_std=np.zeros((5, 2, 3, 8), dtype=np.float64))
        array_write(path / name, values)
    fake_hash = hashlib.sha256(b'artificial fixture only').hexdigest()
    for split in ('TRAIN', 'DEV'):
        chunks = []
        for start in range(0, rows, 128):
            stop = min(start + 128, rows)
            chunks.append(dict(start=start, stop=stop, source_parent_rows_sha256=array_sha(parents[start:stop]),
                response_sha256={'phys': fake_hash, 'local_white': fake_hash},
                noise_latents_sha256={'noise005': fake_hash}, extension_epsilon_sha256=fake_hash,
                baseline100_ADC_sha256=fake_hash,
                outputs={arm: {hypothesis: dict(waveform256_sha256=fake_hash, observables_sha256={'Q': fake_hash})
                    for hypothesis in ('fast', 'slow', 'common_level')} for arm in ('deterministic', 'full', 'simple')}))
        save(path / (split + '_generation.json'), dict(UTC='fixture-time-a', split=split, parents=n,
            draws_per_parent=4, rows=rows, response_forward_rows_per_arm=rows,
            noise005_forward_rows=rows, rows_removed=0, extra_sample_forwards=0, chunks=chunks,
            fixture_only=True, summary={'Q': 1.25, 'PSD': .125}))
        save(path / (split + '_PARENT_BOOTSTRAP.json'), dict(fixture_only=True,
            bootstrap_draws=5, mean_percentile95=[[1., 2.], [3., None]], point_std=[0., .1]))
    save(path / 'FROZEN_INPUTS.json', dict(UTC='fixture-time-a', fixture_only=True,
        optimizer_updates=0, DEV_numeric_decoded=False, checkpoint_state_sha256={'fixture': fake_hash},
        config={'seeds': {'fixture': 19}}, resolved_settings={'draws_per_parent': 4, 'chunk_rows': 128},
        runtime={'python': 'fixture-only', 'device': 'cpu', 'cpu_threads': 1}))
    save(path / 'MATCHED_SELECTIONS.json', dict(fixture_only=True, SOURCE_parent_ids=[100, 101], REAL_A=[1, 2], REAL_B=[3, 4]))
    save(path / 'DISTRIBUTION_DISTANCES.json', dict(fixture_only=True, wave=.2, QPSD=.3, noise=.4,
        relative={'full_minus_reference': -.05}))
    save(path / 'SUMMARY.json', dict(schema='n01/window008-evaluation/v1', UTC='fixture-time-a',
        status='fixed_endpoint_evaluation_complete', fixed_training_G_updates=57000,
        fixed_training_D_updates=144000, evaluation_G_updates=0, evaluation_noise_updates=0,
        population_updates=0, C2ST_updates=0, DEV_numeric_decoded=True, frozen_inputs_rechecked=True,
        all_original_rows_preserved=True, artifact_sha256={}, fixture_only=True))
    rebind_artifacts(path)


def mutate_array(path, member, operation):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values[member] = operation(values[member])
    array_write(path, values)


def mutate_json(path, operation):
    value = common.read(path)
    operation(value)
    save(path, value)


def fixture_root(path):
    path.mkdir()
    for relative in common.SOURCE_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(common.ROOT / relative, target)
    cfg = common.read(common.ROOT / 'window008_config.json')
    save(path / 'window008_config.json', cfg)
    protected = path / 'src/n01/artificial_training_fixture.py'
    protected.write_text('FIXTURE_ONLY = True\n', encoding='utf8')
    training = dict(schema='n01/window008/source_manifest/v1', run_id=common.RUN_ID,
        before_scientific_model_forward=True, config_sha256=common.sha(path / 'window008_config.json'),
        files={'window008_config.json': common.sha(path / 'window008_config.json'),
               'src/n01/artificial_training_fixture.py': common.sha(protected)}, fixture_only=True)
    save(path / common.TRAIN_MANIFEST, training)
    policy = common.read(path / common.POLICY)
    policy['training_source_manifest_sha256'] = common.sha(path / common.TRAIN_MANIFEST)
    save(path / common.POLICY, policy)
    return cfg, training


def rewrite_archive(target, payload, *, extra=None, duplicate=None, symlink=None):
    with tarfile.open(target, 'w:gz') as archive:
        for name, content in payload.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if extra is not None:
            info = tarfile.TarInfo(extra)
            info.size = 1
            archive.addfile(info, io.BytesIO(b'x'))
        if duplicate is not None:
            content = payload[duplicate]
            info = tarfile.TarInfo(duplicate)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if symlink is not None:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = '/tmp/outside-fixture'
            archive.addfile(info)


def bind_fake_native_job(root, artifact_source):
    """Fabricated receipts used only for parser and rejection fixtures."""
    cfg = common.read(root / 'window008_config.json')
    paths = common.paths_for('primary', root)
    paths['supervisor'].mkdir(parents=True)
    shutil.copytree(artifact_source, paths['artifacts'])
    executable = '/fixture-only/python'
    launch = dict(supervisor_pid=801, child_pid=802, kind='primary',
        command=[executable, '-B', str(root / 'scripts/run_window008_evaluation.py'), '--kind', 'primary'])
    save(paths['supervisor'] / 'launch.json', launch)
    save(paths['supervisor'] / 'exit_receipt.json', dict(exit_code=0, reason=None, supervisor_pid=801, child_pid=802))
    command = [executable, '-B', '-m', 'n01.evaluate_window008', '--out', str(paths['run']),
        '--config', str(root / 'window008_config.json'), '--dataset', str(root / cfg['dataset']),
        '--statistics', str(root / cfg['statistics']), '--raw-container', cfg['remote_training_container'],
        '--identities', str(root / 'data/PARENT_IDENTITIES.npz'), '--device', 'cuda:0', '--destination', str(paths['artifacts'])]
    save(paths['job'] / 'EVALUATOR_NATIVE.json', dict(exit_code=0, module='n01.evaluate_window008',
        kind='primary', command=command, wrapper_pid=802, supervisor_pid=801,
        evaluation_source_manifest_sha256=common.sha(root / common.EVAL_MANIFEST), fixture_only=True))
    save(paths['job'] / 'JOB_COMPLETE.json', dict(status='evaluation_job_complete', kind='primary',
        evaluation_source_manifest_sha256=common.sha(root / common.EVAL_MANIFEST),
        training_source_manifest_sha256=common.sha(root / common.TRAIN_MANIFEST), optimizer_updates=0,
        evaluator_native_sha256=common.sha(paths['job'] / 'EVALUATOR_NATIVE.json'),
        summary_sha256=common.sha(paths['artifacts'] / 'SUMMARY.json'), fixture_only=True))
    return paths


def run(output):
    output = common.allow_output(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg, real_training = common.training_binding()
    training_before = {name: common.sha(common.ROOT / name) for name in real_training['files']}
    results = []
    def check(name, function):
        try:
            details = function()
            results.append(dict(name=name, passed=True, details=details))
        except Exception as error:
            results.append(dict(name=name, passed=False, error=repr(error)))
    def rejected(function):
        try:
            function()
        except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
            return dict(rejected=True, exception=type(error).__name__)
        raise AssertionError('The invalid fixture was accepted')
    def equal(condition, message):
        if not condition:
            raise AssertionError(message)
    reference = output / 'artificial_reference'
    build_artifacts(reference)
    candidate = output / 'artificial_identical_replay'
    shutil.copytree(reference, candidate)
    # Deliberately change compression and UTC; file containers differ while members do not.
    for name in NPZ_NAMES:
        with np.load(candidate / name, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        array_write(candidate / name, values, compressed=False)
    for name, ignored in JSON_IGNORED.items():
        if 'UTC' in ignored:
            mutate_json(candidate / name, lambda value: value.update(UTC='fixture-time-b'))
    rebind_artifacts(candidate)
    def exact_positive():
        report = compare_artifacts(reference, candidate)
        common.write_new(output / 'ARTIFICIAL_EXACT_REPLAY.json', report)
        equal(report['status'] == 'exact_replay_verified', str(report['mismatches']))
        equal(report['NPZ_files_checked'] == 13 and report['NPZ_members_checked'] >= 60
              and report['JSON_files_checked'] == 8 and report['generation_chunk_records_checked'] == 4,
              'The verifier skipped an expected artifact class')
        equal(common.sha(reference / 'TRAIN_curves.npz') != common.sha(candidate / 'TRAIN_curves.npz'),
              'Fixture did not create distinct ZIP containers')
        return dict(NPZ_files=report['NPZ_files_checked'], members=report['NPZ_members_checked'],
                    json_files=report['JSON_files_checked'], chunk_records=report['generation_chunk_records_checked'])
    check('all_arrays_json_chunks_equal_despite_container_and_UTC_difference', exact_positive)
    mutations = {
        'array_numeric': lambda path: mutate_array(path / 'TRAIN_curves.npz', 'numeric', lambda a: np.where(np.isnan(a), a, a + 1)),
        'array_dtype': lambda path: mutate_array(path / 'DEV_curves.npz', 'numeric', lambda a: a.astype(np.float32)),
        'array_shape': lambda path: mutate_array(path / 'TRAIN_sample_waveforms.npz', 'numeric', lambda a: a.reshape(2, 2)),
        'signed_zero_bytes': lambda path: mutate_array(path / 'DEV_curves.npz', 'numeric', lambda a: np.asarray([-0., *a[1:]], dtype=a.dtype)),
        'stream_state': lambda path: mutate_array(path / 'DEV_stream_states.npz', 'final_response', lambda a: a + np.uint8(1)),
        'bootstrap_array': lambda path: mutate_array(path / 'DEV_parent_bootstrap.npz', 'bootstrap_std', lambda a: a + .01),
        'bootstrap_json': lambda path: mutate_json(path / 'TRAIN_PARENT_BOOTSTRAP.json', lambda a: a.update(point_std=[0., .2])),
        'distance': lambda path: mutate_json(path / 'DISTRIBUTION_DISTANCES.json', lambda a: a.update(QPSD=.4)),
        'chunk_hash': lambda path: mutate_json(path / 'TRAIN_generation.json', lambda a: a['chunks'][0].update(extension_epsilon_sha256='changed')),
        'chunk_partition': lambda path: mutate_json(path / 'DEV_generation.json', lambda a: a['chunks'][0].update(stop=127)),
        'seed': lambda path: mutate_json(path / 'FROZEN_INPUTS.json', lambda a: a['config']['seeds'].update(fixture=20)),
        'backend': lambda path: mutate_json(path / 'FROZEN_INPUTS.json', lambda a: a['runtime'].update(cpu_threads=2)),
        'parent_order': lambda path: mutate_array(path / 'TRAIN_observables.npz', 'source_parent_row', lambda a: a[::-1]),
        'draw_order': lambda path: mutate_array(path / 'DEV_observables.npz', 'draw', lambda a: a[::-1]),
        'extra_sample_forward': lambda path: mutate_json(path / 'TRAIN_generation.json', lambda a: a.update(extra_sample_forwards=1)),
        'missing_member': lambda path: array_write(path / 'evaluation_projection_bank.npz', {'numeric': np.ones(4)}),
        'extra_artifact': lambda path: (path / 'EXTRA.json').write_text('{}', encoding='utf8'),
        'failed_artifact': lambda path: (path / 'FAILED.json').write_text('{}', encoding='utf8'),
        'optimizer_update': lambda path: mutate_json(path / 'SUMMARY.json', lambda a: a.update(evaluation_G_updates=1)),
    }
    for name, mutation in mutations.items():
        def negative(name=name, mutation=mutation):
            path = output / ('artificial_mutation_' + name)
            shutil.copytree(reference, path)
            mutation(path)
            rebind_artifacts(path)  # Valid new file bindings must not conceal changed numbers.
            report = compare_artifacts(reference, path)
            common.write_new(output / ('REJECT_' + name + '.json'), report)
            equal(report['status'] == 'exact_replay_failed' and bool(report['mismatches']), 'Mutation escaped exact verification')
            return dict(mismatch_categories=sorted({item['kind'] for item in report['mismatches']}))
        check('replay_rejects_' + name, negative)
    check('same_directory_is_not_replay', lambda: rejected(lambda: compare_artifacts(reference, reference)))
    receipt_file = output / 'exclusive_receipt.json'
    common.write_new(receipt_file, {'first': True})
    check('exclusive_receipts_preserve_first_attempt', lambda: rejected(lambda: common.write_new(receipt_file, {'second': True})))
    check('path_escape_rejected', lambda: rejected(lambda: common.bound(output, '../outside-fixture')))
    source_root = output / 'artificial_source_root'
    fixture_cfg, fixture_training = fixture_root(source_root)
    def freeze_positive():
        with mock.patch.object(freezer.subprocess, 'check_output', return_value='fixture-commit-no-science'):
            result = freezer.freeze(source_root)
        equal(result['evaluation_executed'] is False and result['training_files_unchanged'] == 2, 'Fixture freeze scope differs')
        return result
    check('separate_freeze_keeps_artificial_training_files', freeze_positive)
    check('duplicate_freeze_rejected', lambda: rejected(lambda: freezer.freeze(source_root)))
    def module_changed():
        path = source_root / 'src/n01/unregistered_fixture.py'
        path.write_text('FIXTURE_ONLY = True\n', encoding='utf8')
        try:
            return rejected(lambda: common.evaluation_binding(source_root))
        finally:
            path.unlink()
    check('runtime_module_inventory_addition_rejected', module_changed)
    for name, relative in [('evaluation_source', 'scripts/run_window008_evaluation.py'),
                           ('training_source', 'src/n01/artificial_training_fixture.py')]:
        def changed(relative=relative):
            path = source_root / relative
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n# artificial mutation\n')
                return rejected(lambda: common.evaluation_binding(source_root))
            finally:
                path.write_bytes(original)
        check(name + '_byte_change_rejected', changed)
    bundle_dir = output / 'artificial_bundle'
    check('additive_package', lambda: staging.package(bundle_dir, source_root))
    archive_path = bundle_dir / 'window008_evaluation.tar.gz'
    manifest, payload = staging.archive_payload(archive_path)
    for name, kwargs in [('escape', {'extra': '../escaped.py'}), ('unlisted', {'extra': 'unlisted.py'}),
                          ('duplicate', {'duplicate': common.POLICY}), ('symlink', {'symlink': 'symbolic.py'})]:
        def malformed(name=name, kwargs=kwargs):
            target = output / ('bad_archive_' + name + '.tar.gz')
            rewrite_archive(target, payload, **kwargs)
            return rejected(lambda: staging.archive_payload(target))
        check('archive_rejects_' + name, malformed)
    def payload_changed():
        changed = dict(payload)
        changed[common.POLICY] += b'\n'
        target = output / 'bad_archive_changed_bytes.tar.gz'
        rewrite_archive(target, changed)
        return rejected(lambda: staging.archive_payload(target))
    check('archive_rejects_digest_mismatch', payload_changed)
    install_root = output / 'artificial_install_root'
    install_root.mkdir()
    for relative in set(fixture_training['files']) | {common.TRAIN_MANIFEST}:
        target = install_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    old_draft = install_root / 'src/n01/evaluate_window008.py'
    old_draft.write_bytes(b'# artificial old draft\n')
    def install_positive():
        result = staging.install_locked(archive_path, install_root, fixture_cfg, fixture_training)
        backups = list((install_root / 'receipts').glob('**/before/src/n01/evaluate_window008.py'))
        equal(len(backups) == 1 and backups[0].read_bytes() == b'# artificial old draft\n', 'Previous draft was not preserved')
        common.verify_files(install_root, fixture_training['files'])
        return dict(status=result['status'], preserved_previous_draft=True, actual_remote_execution=False)
    check('install_core_preserves_old_draft_and_training_bytes', install_positive)
    def protected_payload():
        changed = dict(payload)
        spec = copy.deepcopy(manifest)
        relative = 'src/n01/artificial_training_fixture.py'
        changed[relative] = b'# forbidden training replacement\n'
        spec['files'][relative] = hashlib.sha256(changed[relative]).hexdigest()
        changed[common.EVAL_MANIFEST] = json.dumps(spec).encode()
        target = output / 'bad_archive_protected_training.tar.gz'
        rewrite_archive(target, changed)
        return rejected(lambda: staging.install_locked(target, install_root, fixture_cfg, fixture_training))
    check('installation_rejects_any_protected_training_path', protected_payload)
    (install_root / 'runs/evaluation_jobs/primary').mkdir(parents=True)
    check('installation_rejects_already_attempted_job',
          lambda: rejected(lambda: staging.install_locked(archive_path, install_root, fixture_cfg, fixture_training)))
    # Pure supervisor boundary fixtures; Linux dispatch/GPU/native supervision remain unexecuted here.
    limits = common.read(common.ROOT / common.POLICY)['resource_limits']
    def resource_limits():
        equal(supervisor.stop_reason(8 * 1024 ** 3, 28800, 3072 * 1024 ** 2, 10 * 1024 ** 3, limits) is None,
              'Exact registered limits should not exceed the cap')
        expected = [((8 * 1024 ** 3 + 1, 1, 1, 1), 'owned_child_RSS_limit_exceeded'),
                    ((1, 28801, 1, 1), 'owned_job_wall_limit_exceeded'),
                    ((1, 1, 3072 * 1024 ** 2 + 1, 1), 'owned_job_output_limit_exceeded'),
                    ((1, 1, 1, 10 * 1024 ** 3 + 1), 'owned_child_GPU_memory_limit_exceeded')]
        for values, reason in expected:
            equal(supervisor.stop_reason(*values, limits) == reason, 'A resource cap was not enforced')
        return dict(checked_caps=4, values_at_boundary=True, Linux_native_supervision=False)
    check('resource_limit_boundaries', resource_limits)
    def owned_gpu_only():
        with mock.patch.object(supervisor.subprocess, 'check_output', return_value='101, 20\n202, 30\n303, 9999'):
            equal(supervisor.owned_gpu_bytes({101, 202}) == 50 * 1024 ** 2, 'Counted another job as owned')
        return dict(owned_PIDs=[101, 202], excluded_peer=303)
    check('GPU_accounting_uses_only_owned_tree', owned_gpu_only)
    def only_own_signals():
        child = mock.Mock(pid=707)
        child.poll.return_value = None
        with mock.patch.object(supervisor.os, 'killpg', create=True) as signal_group, \
                mock.patch.object(supervisor.signal, 'SIGKILL', 9, create=True), \
                mock.patch.object(supervisor.time, 'monotonic', side_effect=[0., 21.]):
            supervisor.terminate_owned(child, 20)
            sent = [item.args for item in signal_group.call_args_list if item.args[1] != 0]
            equal(sent == [(707, supervisor.signal.SIGTERM), (707, 9)], 'Termination targeted another group')
        return dict(TERM_then_KILL_target=707, actual_signals_sent=False)
    check('termination_targets_only_owned_process_group', only_own_signals)
    def leader_exited_descendants_remain():
        child = mock.Mock(pid=708)
        child.poll.return_value = 1
        with mock.patch.object(supervisor.os, 'killpg', create=True) as signal_group, \
                mock.patch.object(supervisor.signal, 'SIGKILL', 9, create=True), \
                mock.patch.object(supervisor.time, 'monotonic', side_effect=[0., 21.]):
            supervisor.terminate_owned(child, 20)
            sent = [item.args for item in signal_group.call_args_list if item.args[1] != 0]
            equal(sent == [(708, supervisor.signal.SIGTERM), (708, 9)], 'Exited wrapper left its owned descendants running')
        return dict(exited_leader_group_still_terminated=True, actual_signals_sent=False)
    check('exited_wrapper_does_not_leave_unbounded_owned_descendants', leader_exited_descendants_remain)
    native_paths = bind_fake_native_job(source_root, reference)
    check('native_receipt_chain_accepts_bound_artificial_fixture',
          lambda: dict(status=common.completed_job('primary', source_root)['complete']['status'], fixture_only=True))
    native_mutations = [
        ('supervisor_nonzero', native_paths['supervisor'] / 'exit_receipt.json', lambda a: a.update(exit_code=1)),
        ('supervisor_limit_reason', native_paths['supervisor'] / 'exit_receipt.json', lambda a: a.update(reason='cap')),
        ('supervisor_PID', native_paths['supervisor'] / 'exit_receipt.json', lambda a: a.update(child_pid=999)),
        ('launch_command', native_paths['supervisor'] / 'launch.json', lambda a: a.update(command=['wrong'])),
        ('core_nonzero', native_paths['job'] / 'EVALUATOR_NATIVE.json', lambda a: a.update(exit_code=1)),
        ('core_PID', native_paths['job'] / 'EVALUATOR_NATIVE.json', lambda a: a.update(wrapper_pid=999)),
        ('core_module', native_paths['job'] / 'EVALUATOR_NATIVE.json', lambda a: a.update(module='n01.train_window008')),
        ('completion_manifest', native_paths['job'] / 'JOB_COMPLETE.json', lambda a: a.update(evaluation_source_manifest_sha256='wrong')),
    ]
    for name, path, mutation in native_mutations:
        def changed_native(path=path, mutation=mutation):
            content = path.read_bytes()
            try:
                mutate_json(path, mutation)
                return rejected(lambda: common.completed_job('primary', source_root))
            finally:
                path.write_bytes(content)
        check('native_chain_rejects_' + name, changed_native)
    def static_sources():
        count = 0
        for relative in common.SOURCE_FILES:
            if relative.endswith('.py'):
                path = common.ROOT / relative
                ast.parse(path.read_text(encoding='utf8'), filename=str(path))
                count += 1
        return dict(Python_sources_parsed=count, Linux_launch_native_executed=False)
    check('all_helper_and_evaluator_Python_sources_parse', static_sources)
    training_after = {name: common.sha(common.ROOT / name) for name in real_training['files']}
    check('actual_training_frozen_bytes_unchanged', lambda: equal(training_before == training_after, 'Actual frozen training bytes changed'))
    result = dict(schema='n01/window008-evaluation-helper-qa/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='passed' if all(item['passed'] for item in results) else 'failed',
        passed=sum(item['passed'] for item in results), total=len(results), checks=results,
        artificial_fixtures_only=True, scientific_model_imports=0, scientific_forwards=0, optimizer_updates=0,
        actual_SCI_DEV_array_decodes=0, GPU_used=False, Linux_native_dispatch_supervision_executed=False,
        training_manifest_sha256=common.sha(common.ROOT / common.TRAIN_MANIFEST),
        actual_frozen_training_files=len(training_before), actual_training_files_unchanged=training_before == training_after,
        helper_source_sha256={name: common.sha(common.ROOT / name) for name in common.SOURCE_FILES},
        receipt_semantics='All native job receipts inside artificial_* are fabricated parser fixtures, not scientific execution receipts.')
    common.write_new(output / 'REVIEW.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'passed', 'total', 'actual_training_files_unchanged')}))
    common.require(result['status'] == 'passed', 'Helper QA failed; inspect ' + str(output / 'REVIEW.json'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New E/D directory preserving all artificial fixtures')
    run(parser.parse_args().output)
