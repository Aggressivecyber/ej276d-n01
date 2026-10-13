"""CPU metadata checks for runtime evidence and extracted monitor state logic.

The native engineering/prelaunch receipts below are labelled in-memory fixtures.
They do not certify any actual native run or authorize a dispatch.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from metadata_audit import ROOT, OUT, sha

sys.path.insert(0, str(ROOT / 'scripts'))
import window008_resume01_runtime as runtime


def main():
    runtime_path = ROOT / 'scripts/window008_resume01_runtime.py'
    monitor_path = ROOT / 'scripts/monitor_window008_resume01.py'
    checked_sha = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path)
                   for path in [runtime_path, monitor_path]}
    cfg, base, parent_manifest = runtime.validate_inputs(ROOT)
    refs = cfg['reference_inputs']
    stop_local = runtime.read(ROOT / refs['fix01_user_stop_verification_local']['path'])
    stop_remote = runtime.read(ROOT / refs['fix01_user_stop_verification']['path'])
    pause = runtime.read(ROOT / refs['fix01_user_pause']['path'])
    collection = runtime.read(ROOT / refs['fix01_user_stop_collection']['path'])
    assertions = dict(local_and_remote_stop_same_JSON=stop_local == stop_remote,
                      distinct_raw_serialization_hashes=refs['fix01_user_stop_verification_local']['sha256'] != refs['fix01_user_stop_verification']['sha256'],
                      pause_stop_exact_raw_hash=pause['stop_receipt_sha256'] == refs['fix01_user_stop_verification_local']['sha256'],
                      pause_collection_exact_raw_hash=pause['collection_receipt_sha256'] == refs['fix01_user_stop_collection']['sha256'],
                      collection_remote_stop_exact_raw_hash=collection['files']['runs/USER_STOP_VERIFICATION.json']['sha256'] == refs['fix01_user_stop_verification']['sha256'],
                      collection_native_matches=collection['native'] == runtime.read(ROOT / refs['fix01_native_exit']['path']),
                      collection_failure_matches=collection['failure'] == runtime.read(ROOT / refs['fix01_failure']['path']))
    results = [dict(case='actual26refs_transitive_runtime_input_validation', passed=True, fixture=False)]
    results.extend(dict(case=name, passed=bool(valid), fixture=False) for name, valid in assertions.items())

    # Metadata fixture overrides only JSON reads, preserving the real bound
    # SHA checks for all immutable files. No local project state is modified.
    originals = dict(read=runtime.read, validate_fix01=runtime.validate_fix01)
    documents = {ROOT / 'window008_resume01_config.json': copy.deepcopy(cfg)}
    for field in ['base_configuration', 'legacy_source_manifest', 'fix01_configuration', 'fix01_source_manifest']:
        documents[(ROOT / cfg[field]).resolve()] = originals['read'](ROOT / cfg[field])
    for reference in refs.values():
        path = (ROOT / reference['path']).resolve()
        if path.suffix == '.json':
            documents[path] = originals['read'](path)

    def activate(docs):
        runtime.read = lambda path: copy.deepcopy(docs.get(Path(path).resolve(), originals['read'](Path(path))))
        runtime.validate_fix01 = lambda root=ROOT: (cfg, base, parent_manifest)

    def reject(name, alter):
        fixture = copy.deepcopy(documents)
        alter(fixture)
        activate(fixture)
        try:
            runtime.validate_inputs(ROOT)
        except (ValueError, RuntimeError, TypeError, KeyError) as error:
            results.append(dict(case=name, passed=True, rejected=True, fixture=True, rejection=str(error)))
        else:
            results.append(dict(case=name, passed=False, rejected=False, fixture=True))

    config_path = (ROOT / 'window008_resume01_config.json').resolve()
    pause_path = (ROOT / refs['fix01_user_pause']['path']).resolve()
    native_path = (ROOT / refs['fix01_native_exit']['path']).resolve()
    failure_path = (ROOT / refs['fix01_failure']['path']).resolve()
    local_path = (ROOT / refs['fix01_user_stop_verification_local']['path']).resolve()
    remote_path = (ROOT / refs['fix01_user_stop_verification']['path']).resolve()
    reject('budget_wrong_new_G', lambda docs: docs[config_path].update(new_execution_G_updates=35000))
    reject('budget_wrong_new_D', lambda docs: docs[config_path].update(new_execution_D_updates=96000))
    reject('history_omits_discarded_user_stop', lambda docs: docs[config_path].update(historical_actual_G_updates_after_success=57269))
    reject('wrong_execution_arm_order', lambda docs: docs[config_path].update(execution_arms=['simple', 'full']))
    reject('pause_role_not_user_stop', lambda docs: docs[pause_path].update(status='failed'))
    reject('pause_points_to_wrong_raw_stop', lambda docs: docs[pause_path].update(stop_receipt_sha256=refs['fix01_user_stop_verification']['sha256']))
    reject('pause_collection_hash_wrong', lambda docs: docs[pause_path].update(collection_receipt_sha256='0' * 64))
    reject('local_remote_stop_JSON_differs', lambda docs: docs[local_path].update(native_exit_code=0))
    reject('owned_process_still_present', lambda docs: docs[remote_path].update(owned_processes_after=[{'pid': 1}]))
    reject('trainer_absence_not_verified', lambda docs: docs[remote_path].update(trainer_absent=False))
    reject('native_stop_exit_zero', lambda docs: docs[native_path].update(exit_code=0))
    reject('failure_count_chain_wrong', lambda docs: docs[failure_path].update(new_execution_G_updates=3400))
    reject('boundary_anchor_hash_wrong', lambda docs: docs[pause_path]['periodic_checkpoint'].update(sha256='0' * 64))

    runtime.read = originals['read']
    runtime.validate_fix01 = originals['validate_fix01']

    # Evaluate the actual AST branch that chooses state; none of the SSH,
    # filesystem writes, process or GPU commands in monitor.main executes.
    syntax = ast.parse(monitor_path.read_text('utf8'))
    function = next(item for item in syntax.body if isinstance(item, ast.FunctionDef) and item.name == 'main')
    state_assignment = next(item for item in function.body if isinstance(item, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == 'state' for target in item.targets))
    state_code = compile(ast.Expression(state_assignment.value), str(monitor_path), 'eval')
    scenarios = [
        ('not_started', {}, None, None, None, False, 0, 'window008_resume01_not_started'),
        ('dispatched_no_process', {}, None, None, {'x': 1}, False, 0, 'window008_resume01_dispatched_pending_scientific_process'),
        ('active_import_no_update', {}, None, None, {'x': 1}, True, 0, 'window008_resume01_dispatched_pending_first_observed_update'),
        ('first_scientific_update', {}, None, None, {'x': 1}, True, 1, 'window008_resume01_running'),
        ('missing_process_after_update', {}, None, None, {'x': 1}, False, 1, 'window008_resume01_process_missing_native_receipt_review_required'),
        ('complete_waiting_native', {}, {'complete': True}, None, {'x': 1}, False, 32000, 'window008_resume01_finished_waiting_native_receipt'),
        ('complete_with_native_zero', {}, {'complete': True}, {'exit_code': 0}, {'x': 1}, False, 32000, 'window008_resume01_complete_pending_endpoint_evaluation'),
        ('native_zero_missing_complete', {}, None, {'exit_code': 0}, {'x': 1}, False, 32000, 'window008_resume01_native_success_missing_completion'),
        ('native_failure_without_failure_doc', {}, None, {'exit_code': 1}, {'x': 1}, False, 1, 'window008_resume01_failed_or_bounded_stop_preserved'),
        ('failure_after_partial_D', {'failure': True}, None, {'exit_code': 1}, {'x': 1}, False, 1, 'window008_resume01_failed_or_bounded_stop_preserved'),
        ('complete_but_native_nonzero', {}, {'complete': True}, {'exit_code': 1}, {'x': 1}, False, 32000, 'window008_resume01_failed_or_bounded_stop_preserved'),
    ]
    for name, failure, complete, native, start, active, g, expected in scenarios:
        actual = eval(state_code, {}, dict(failure=failure, complete=complete, native=native, start=start, active=active, g=g))
        results.append(dict(case='monitor_' + name, passed=actual == expected, fixture=True,
                            actual_state=actual, expected_state=expected))

    for name, digest in checked_sha.items():
        assert sha(ROOT / name) == digest, 'Helper changed during metadata review: ' + name
    review = dict(schema='n01/window008-resume01-independent-helper-counterexamples/v1',
                  UTC=datetime.now(timezone.utc).isoformat(), files_sha256=checked_sha,
                  config_sha256=sha(ROOT / 'window008_resume01_config.json'),
                  status='passed' if all(row['passed'] for row in results) else 'failed', cases=results,
                  source_reference_count=len(refs), scientific_model_forwards=0, optimizer_updates=0,
                  DEV_numeric_decodes=0, GPU_calls=0, SSH_calls=0, project_state_writes=0,
                  scope='Actual input chain validation plus in-memory malformed JSON and monitor state AST fixtures.')
    (OUT / 'HELPER_METADATA_COUNTEREXAMPLES.json').write_text(json.dumps(review, indent=2) + '\n', 'utf8')
    print(json.dumps(dict(status=review['status'], cases=len(results), source_reference_count=len(refs),
                          files_sha256=checked_sha, failed=[row for row in results if not row['passed']]), indent=2))


if __name__ == '__main__':
    main()
