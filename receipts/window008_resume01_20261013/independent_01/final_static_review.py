"""Seal static/metadata evidence, explicitly leaving target native evidence open."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from datetime import datetime, timezone

from metadata_audit import ROOT, OUT, sha

sys.path.insert(0, str(ROOT / 'scripts'))
import window008_resume01_runtime as runtime


def main():
    cfg, _, parent = runtime.validate_inputs(ROOT)
    runtime_names = runtime.runtime_files(ROOT)
    expected_modules = sorted(name for name in runtime_names if name.startswith('src/n01/') and name.endswith('.py'))
    checks = dict(runtime_module_closure36=len(expected_modules) == 36,
                  unexecuted007_draft_excluded='src/n01/train_response007_continuation01.py' not in runtime_names,
                  actual_reference_count26=len(cfg['reference_inputs']) == 26,
                  no_changed_scientific_model=sha(ROOT / 'src/n01/model008_fix01.py') == parent['files']['src/n01/model008_fix01.py'],
                  no_changed_scientific_step=sha(ROOT / 'src/n01/response008.py') == parent['files']['src/n01/response008.py'],
                  no_changed_scientific_loss=sha(ROOT / 'src/n01/losses008_v2.py') == parent['files']['src/n01/losses008_v2.py'])
    names = list(runtime.NEW_FILES) + ['scripts/register_window008_resume01.py']
    syntax = {}
    for name in names:
        path = ROOT / name
        if path.suffix == '.py':
            ast.parse(path.read_text('utf8'), filename=str(path))
            syntax[name] = 'passed_AST_syntax'
    checks['all_new_python_AST_parse'] = len(syntax) == sum(name.endswith('.py') for name in names)
    shell = (ROOT / 'scripts/launch_window008_resume01.sh').read_bytes()
    checks['launch_unix_newlines'] = b'\r' not in shell
    checks['launch_bound_root_and_module'] = (cfg['remote_root'].encode() in shell
        and b'-m n01.train_window008_resume01' in shell and b'--resume-config' in shell)

    # Import the QA entry module to detect import-name errors without main,
    # scientific model construction, CUDA invocation or filesystem mutation.
    qa_path = ROOT / 'scripts/check_window008_resume01_cuda.py'
    spec = importlib.util.spec_from_file_location('independent_native_QA_import_check', qa_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked_native_sources = module.sources()
    checks['native_QA_entry_imports_without_main'] = True
    checks['native_QA_binds_executed_CPU_helper'] = 'scripts/check_window008_resume01_engine.py' in checked_native_sources
    native_text = qa_path.read_text('utf8')
    checks['actual_import_forward_step_draw_hard_guards'] = all(token in native_text for token in
        ["(torch.nn.Module,'_call_impl')", "(engine.model,'signal_forward')", "(torch.optim.AdamW,'step')",
         "(engine.source_sampler,'draw')", "(engine.real_sampler,'draw')", "(engine.streams,'draw')"])
    checks['full_artificial_uses_explicit_parser_import'] = "if arm=='full':engine.import_fix01_full_periodic6000" in native_text
    checks['native_artificial_budget3G6D6144'] = "report['engineering_G_updates']!=3" in native_text and "report['artificial_forward_rows']!=6144" in native_text

    evidence_names = ['CHECKPOINT_METADATA_REVIEW.json', 'IMMUTABILITY_REFERENCE_REVIEW.json',
                      'ENGINE_METADATA_COUNTEREXAMPLES.json', 'DRIVER_METADATA_COUNTEREXAMPLES.json',
                      'HELPER_METADATA_COUNTEREXAMPLES.json', 'ACTUAL_NATIVE_RECEIPT_REVIEW.json']
    evidence = {}
    for name in evidence_names:
        path = OUT / name
        record = json.loads(path.read_text('utf8'))
        evidence[name] = dict(sha256=sha(path), status=record['status'], cases=len(record.get('cases', [])))
        checks['evidence_passed_' + name] = record['status'] == 'passed'
    native_fixture_files = sorted(OUT.glob('NATIVE_HELPER_RECEIPT_FIXTURES_*.json'))
    matching_native = [path for path in native_fixture_files
                       if json.loads(path.read_text('utf8'))['source_sha256'] == sha(ROOT / 'scripts/window008_resume01_runtime.py')]
    checks['native_helper_current_source_fixture_passed'] = bool(matching_native) and all(
        json.loads(path.read_text('utf8'))['status'] == 'passed' for path in matching_native)
    for path in matching_native:
        record = json.loads(path.read_text('utf8'))
        evidence[path.name] = dict(sha256=sha(path), status=record['status'], cases=len(record['cases']))
    engine_review = json.loads((OUT / 'ENGINE_METADATA_COUNTEREXAMPLES.json').read_text('utf8'))
    driver_review = json.loads((OUT / 'DRIVER_METADATA_COUNTEREXAMPLES.json').read_text('utf8'))
    checks['engine_current_source_reviewed'] = engine_review['source_sha256'] == sha(ROOT / 'src/n01/response008_resume01.py')
    checks['driver_current_source_reviewed'] = driver_review['source_sha256'] == sha(ROOT / 'src/n01/train_window008_resume01.py')
    helper_review = json.loads((OUT / 'HELPER_METADATA_COUNTEREXAMPLES.json').read_text('utf8'))
    checks['helper_current_sources_reviewed'] = all(sha(ROOT / name) == expected for name, expected in helper_review['files_sha256'].items())
    native_review = json.loads((OUT / 'ACTUAL_NATIVE_RECEIPT_REVIEW.json').read_text('utf8'))
    checks['actual_target_native_receipts_unchanged'] = all(sha(ROOT / name) == expected
        for name, expected in native_review['files_sha256'].items())
    checks['actual_target_native_checked_sources_unchanged'] = all(sha(ROOT / name) == expected
        for name, expected in native_review['checked_QA_source_sha256'].items())
    native_passed = native_review['status'] == 'passed' and checks['actual_target_native_receipts_unchanged'] and checks['actual_target_native_checked_sources_unchanged']
    checks['actual_target_native_evidence_closed'] = native_passed

    failed_path = ROOT / 'receipts/window008_resume01_20261013/engine_cpu_01/ENGINEERING_RECEIPT.json'
    failed_cpu = json.loads(failed_path.read_text('utf8'))
    local_failure = dict(path=failed_path.relative_to(ROOT).as_posix(), sha256=sha(failed_path),
                         status=failed_cpu['status'], error=failed_cpu.get('error'),
                         passed_artificial_checks=len(failed_cpu['checks']),
                         engineering_G_updates=failed_cpu['engineering_G_updates'],
                         engineering_D_updates=failed_cpu['engineering_D_updates'],
                         actual_target_native_import_verified_by_this_review=False)
    doc = ROOT / 'docs/WINDOW008_RESUME01_INDEPENDENT_REVIEW.md'
    files = {name: sha(ROOT / name) for name in names}
    files[doc.relative_to(ROOT).as_posix()] = sha(doc)
    files.update({path.relative_to(ROOT).as_posix(): sha(path) for path in OUT.glob('*.py')})
    files.update({(OUT / name).relative_to(ROOT).as_posix(): record['sha256'] for name, record in evidence.items()})
    files.update(native_review['files_sha256'])
    review = dict(schema='n01/window008-resume01-independent-review/v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  status='passed_independent_prelaunch_review' if all(checks.values()) else 'failed_independent_static_review',
                  scope='Static sources, actual bound checkpoint metadata, malformed metadata, receipt fixtures and independently verified target native receipts.',
                  checks=checks, files_sha256=files, metadata_evidence=evidence,
                  scientific_runtime_module_inventory=expected_modules,
                  config_sha256=sha(ROOT / 'window008_resume01_config.json'),
                  parent_CPU_actual_import_failed_receipt=local_failure,
                  conditional_native_evidence=dict(required_before_scientific_dispatch=True,
                       native_evidence_condition_satisfied=native_passed,
                       actual_target_import_passed=native_passed, actual_target_native_engineering_exit0=native_passed,
                       require_strict_initial_D_frozen_reference=True,
                       require_actual_full6000_all_native_state_bitwise_import=True,
                       require_actual_scientific_forward_optimizer_sampler_stream_draws_zero=True,
                       require_native_artificial_G3_D6_forward_rows6144=True,
                       source_and_PID_time_receipt_chain_required=True,
                       no_waiver_from_CPU_metadata_positive_control=True),
                  scientific_model_construction=False, scientific_model_forwards=0, optimizer_updates=0,
                  DEV_numeric_decodes=0, GPU_calls=0, SSH_calls=0, new_scientific_jobs=0,
                  scientific_status='response_match_not_established')
    (OUT / 'INDEPENDENT_REVIEW.json').write_text(json.dumps(review, indent=2, ensure_ascii=False) + '\n', 'utf8')
    print(json.dumps(dict(status=review['status'], checked_files=len(files), source_modules=len(expected_modules),
                          failed_checks=[name for name, valid in checks.items() if not valid],
                          actual_target_native_evidence_required=True), indent=2))


if __name__ == '__main__':
    main()
