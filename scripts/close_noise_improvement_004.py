"""Close the fixed run from native, replay, and visual receipts; no fitting."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ID = 'noise_improvement_004_20261012'
RUN = ROOT / 'runs' / ID
REC = ROOT / 'receipts' / ID
REPORT = ROOT / 'reports' / ID


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    existing = read(REC / 'FINAL_RECEIPT.json') if (REC / 'FINAL_RECEIPT.json').exists() else None
    now = existing['closed_UTC'] if existing else dt.datetime.now(dt.timezone.utc).isoformat()
    science = RUN / 'runs/science'
    complete = read(science / 'COMPLETE.json')
    replay = read(RUN / 'runs/replay/REPLAY_VERIFICATION.json')
    native = read(RUN / 'runs/supervisor/exit_receipt.json')
    post = read(REC / 'POSTFLIGHT.json')
    visual = read(REC / 'VISUAL_QA.json')
    summary = read(REPORT / 'SUMMARY.json')
    tests = read(REC / 'TEST_SUMMARY.json')
    assert complete['status'] == 'complete' and native['exit_code'] == 0
    assert complete['noise_training_updates'] == 5000
    assert complete['classifier_training_updates'] == 1200
    assert complete['response_training_updates'] == 0
    assert replay['complete_sha256'] == sha(science / 'COMPLETE.json')
    assert replay['execution_source_manifest_sha256'] == sha(ROOT / 'EXECUTION_SOURCE_MANIFEST.json')
    for name, digest in complete['files'].items():
        assert sha(science / name) == digest, name
    for name, digest in complete['source_files'].items():
        assert sha(ROOT / name) == digest, name
    assert not post['active_N01_python'] and not post['GPU_compute_processes']
    assert visual['status'] == 'passed'
    assert sha(REPORT / 'N01_NOISE_IMPROVEMENT_REPORT.html') == visual['html_sha256']
    for name, digest in visual['screenshots'].items():
        assert sha(RUN / name) == digest, name
    for name, digest in visual['figures'].items():
        assert sha(REPORT / 'figures' / name) == digest, name
    assert replay['generated_evaluation_populations_bitwise_equal']
    assert all(x['bitwise_equal'] for x in replay['classifiers'])
    assert all(x['pre_ADC_bitwise'] and x['post_ADC_bitwise'] for x in replay['generator_chunks'])
    assert tests['distinct_tests'] == 21 and tests['status'] == 'passed'
    assert summary['noise_gate_passed'] is False

    review = dict(
        status='passed_for_bounded_partial_gain_delivery', UTC=now,
        reviewer='n01_repair_review', review_mode='independent_read_only_results_audit',
        blockers=[], scientific_gate_passed=False,
        checked=['28 scientific payload hashes', '28 execution source hashes',
                 'all 5000 updates and all 21047 TRAIN events',
                 'OLS joint covariance error independently recomputed',
                 'ACF residual maximum at lag17 independently recomputed',
                 'six classifiers and unchanged REAL-REAL control',
                 'native execution and replay receipts', 'report claim boundaries'],
        limitations=['100 observed samples only', 'no independent acquisition validation',
                     'ACF and REAL-REAL control gate remain failed',
                     'no hardware mechanism identification', 'no 480-sample integration or pulse training'],
        added_scientific_updates=0,
        pre_execution_review_unchanged=True)
    if (REC / 'RESULTS_INDEPENDENT_REVIEW.json').exists():
        assert read(REC / 'RESULTS_INDEPENDENT_REVIEW.json') == review
    else:
        write(REC / 'RESULTS_INDEPENDENT_REVIEW.json', review)
    final = dict(
        status='completed_partial_gain_stopped_at_failed_noise_gate', run_id=ID,
        closed_UTC=now, training_ended_UTC=native['ended_utc'], active_scientific_jobs=0,
        noise_training_updates=5000, classifier_evaluation_updates=1200, response_training_updates=0,
        cumulative_generator_updates=87000,
        accounting='77000 original + 5000 repair002 noise + 5000 improvement004 noise; evaluation classifier updates excluded',
        source_commit='0c3c3793d4ab033a18b8644756f447abe47d4ba2',
        native_training_exit_code=0, native_training_receipt=native,
        scientific_payloads_verified=len(complete['files']),
        execution_sources_verified=len(complete['source_files']),
        complete_sha256=sha(science / 'COMPLETE.json'),
        final_checkpoint_sha256=complete['final_checkpoint_sha256'],
        config_sha256=complete['config_sha256'],
        source_manifest_sha256=complete['source_manifest_sha256'],
        train_container_sha256=complete['train_container_sha256'],
        train_rows_used=complete['real_unique'], train_total_rows=complete['TRAIN_rows'],
        train_total_visits=replay['sampler_total_rows'], min_visits=replay['min_visits'], max_visits=replay['max_visits'],
        noise_gate_passed=False, noise_gate_status=complete['noise_gate_status'],
        failed_checks=summary['failed_checks'], old_gate_changed=False,
        old_metrics=summary['old_metrics'], new_metrics=summary['new_metrics'],
        model_scope='observed_baseline100_only', model_parameters=replay['parameter_count'],
        full480_integration_supported=False, full_detector_response_match_established=False,
        validation_previously_viewed=True, independent_acquisition_validations=0,
        replay=dict(path='runs/' + ID + '/runs/replay/REPLAY_VERIFICATION.json',
                    sha256=sha(RUN / 'runs/replay/REPLAY_VERIFICATION.json'),
                    explicit_generator_rows=1024, evaluation_generated_rows=10240,
                    classifiers=6, classifier_score_rows=3072,
                    all_bitwise_equal=True, added_training_updates=0),
        tests=tests, postflight=post, visual_QA=visual,
        results_review_receipt='receipts/' + ID + '/RESULTS_INDEPENDENT_REVIEW.json',
        results_review_sha256=sha(REC / 'RESULTS_INDEPENDENT_REVIEW.json'),
        report='reports/' + ID + '/N01_NOISE_IMPROVEMENT_REPORT.html',
        report_files={p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
                      for p in sorted(REPORT.rglob('*')) if p.is_file()},
        delivery_receipts='BACKUP_RECEIPT.json and DELIVERY_RECEIPT.json are post-packaging receipts',
        next_action='Preserve this fixed endpoint. A separately registered ACF objective study would be needed; no current scientific job.')
    if existing:
        assert existing == final, 'Preserve the closed scientific receipt'
    else:
        write(REC / 'FINAL_RECEIPT.json', final)

    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf8')
    old = '**当前执行：noise_improvement_004 已预登记，正在实现与检查 100 点观测基线模型。固定 5000 次噪声更新，旧评价门保持不变，本轮不整合 480 点响应。** 见 [004 协议](docs/NOISE_IMPROVEMENT_004_PROTOCOL.md) 和 `noise_improvement_004_config.json`。实际运行状态以 `PROGRAM.json`、`jobs.json` 和执行回执为准。'
    new = '**本轮已完成：noise_improvement_004 实际训练 5000 次，OLS 联合误差由 0.57436 降至 0.14193，通过原阈值；ACF 最大差 0.02185 仍超过 0.015，REAL–REAL 对照区间检查也未通过。噪声总门仍失败，响应训练为 0，当前科学作业为 0。** 见 [改进报告](reports/noise_improvement_004_20261012/README.md)、[004 协议](docs/NOISE_IMPROVEMENT_004_PROTOCOL.md) 和 [完成回执](receipts/noise_improvement_004_20261012/FINAL_RECEIPT.json)。本候选仅支持 100 点观测基线，不支持 480 点响应整合。'
    assert old in readme or new in readme
    readme = readme.replace(old, new, 1).replace('**最新诊断完成：', '**前次诊断完成：', 1)
    old_revision = '当前修订以 `docs/REPAIR_002_PROTOCOL.md`、`repair_run_config.json` 和 `PROGRAM.json` 为准。'
    new_revision = '整体修复协议为 `docs/REPAIR_002_PROTOCOL.md`；最新独立噪声候选为 `docs/NOISE_IMPROVEMENT_004_PROTOCOL.md` 与 `noise_improvement_004_config.json`，完成状态见 `PROGRAM.json`。'
    assert old_revision in readme or new_revision in readme
    readme = readme.replace(old_revision, new_revision, 1)
    readme_path.write_text(readme, encoding='utf8', newline='\n')

    program = read(ROOT / 'PROGRAM.json')
    program.update(status='noise_improvement_completed_partial_gain', active_repair=None,
        active_scientific_jobs=0, current_stage='bounded_noise100_evaluated_gate_failed',
        updated_UTC=now, last_verified_UTC=post['UTC'], new_training_updates=87000,
        current_run_training_updates=5000, latest_run_id=ID, latest_improvement=ID,
        improvement_noise_updates=5000, improvement_response_updates=0,
        improvement_classifier_evaluation_updates=1200, improvement_training_pid=None,
        improvement_supervisor_pid=None, improvement_historical_training_pid=628576,
        improvement_historical_supervisor_pid=628567, improvement_ended_UTC=native['ended_utc'],
        improvement_native_exit_code=0, improvement_noise_gate_passed=False,
        improvement_noise_gate_status=complete['noise_gate_status'],
        improvement_failed_checks=summary['failed_checks'],
        improvement_report=final['report'], improvement_completion_receipt='receipts/' + ID + '/FINAL_RECEIPT.json',
        improvement_full480_integration_supported=False,
        training_update_accounting=final['accounting'], next_action=final['next_action'])
    write(ROOT / 'PROGRAM.json', program)
    jobs = read(ROOT / 'jobs.json')
    jobs.update(updated_UTC=now, active_scientific_jobs=0)
    job = next(x for x in jobs['jobs'] if x['job_id'] == 'N01_NOISE_IMPROVEMENT_004_20261012')
    job.update(status='completed_partial_gain_stopped_at_failed_noise_gate', pid=None, supervisor_pid=None,
        historical_training_pid=628576, historical_supervisor_pid=628567, ended_UTC=native['ended_utc'],
        last_verified_UTC=post['UTC'], native_exit_code=0, observed_noise_updates=5000,
        noise_training_updates=5000, response_training_updates=0, classifier_evaluation_updates=1200,
        noise_gate_passed=False, noise_gate_status=complete['noise_gate_status'], failed_checks=summary['failed_checks'],
        report=final['report'], completion_receipt=program['improvement_completion_receipt'])
    write(ROOT / 'jobs.json', jobs)
    pointer_path = ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json'
    pointer = read(pointer_path)
    pointer.update(status=program['status'], active_repair=None, active_scientific_jobs=0,
        updated_UTC=now, plan_sha256=sha(readme_path), latest_run_id=ID,
        latest_improvement_report=str(ROOT / final['report']),
        improvement_configuration=str(ROOT / 'noise_improvement_004_config.json'),
        improvement_completion_receipt=str(ROOT / program['improvement_completion_receipt']),
        improvement_noise_training_updates=5000, improvement_response_training_updates=0,
        improvement_noise_gate_passed=False, improvement_full480_integration_supported=False)
    write(pointer_path, pointer)
    print(json.dumps(dict(status=final['status'], active_scientific_jobs=0,
        noise_updates=5000, response_updates=0, final_receipt_sha256=sha(REC / 'FINAL_RECEIPT.json'))))


if __name__ == '__main__':
    main()
