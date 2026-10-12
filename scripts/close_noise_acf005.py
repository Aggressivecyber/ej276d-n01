"""Finalize the one fixed005 run from native, independent, and visual evidence."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
ID='noise_acf005_20261012'
RUN=ROOT/'runs'/ID
REC=ROOT/'receipts'/ID
REPORT=ROOT/'reports'/ID

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')

def main():
    assert not (REC/'FINAL_RECEIPT.json').exists(),'Preserve closed005 receipt'
    science=RUN/'runs/science';now=dt.datetime.now(dt.timezone.utc).isoformat()
    complete=read(science/'COMPLETE.json');replay=read(RUN/'runs/replay/REPLAY_VERIFICATION.json')
    config=read(ROOT/'noise_acf005_config.json')
    native=read(RUN/'runs/supervisor/exit_receipt.json');post=read(REC/'POSTFLIGHT.json')
    visual=read(REC/'VISUAL_QA.json');summary=read(REPORT/'SUMMARY.json')
    review=read(REC/'RESULTS_INDEPENDENT_REVIEW.json');tests=read(REC/'TEST_SUMMARY.json')
    scope=read(REPORT/'SUPPORT_SCOPE.json');start=read(REC/'START_RECEIPT.json')
    assert complete['status']=='complete' and native['exit_code']==0
    assert complete['noise_training_updates']==5000 and complete['classifier_training_updates']==1200
    assert complete['response_training_updates']==0 and complete['scientifically_qualified'] is False
    assert replay['complete_sha256']==sha(science/'COMPLETE.json')
    assert replay['execution_source_manifest_sha256']==sha(ROOT/'NOISE_ACF005_SOURCE_MANIFEST.json')
    assert complete['config_sha256']==sha(ROOT/'noise_acf005_config.json')
    assert post['train_container_sha256']==config['train_container_sha256']
    for name,digest in complete['files'].items():assert sha(science/name)==digest,name
    for name,digest in complete['source_files'].items():assert sha(ROOT/name)==digest,name
    assert not post['active_N01_python'] and not post['GPU_compute_processes']
    assert visual['status']=='passed' and visual['html_sha256']==sha(REPORT/'N01_ACF005_REPORT.html')
    for name,digest in visual['screenshots'].items():assert sha(RUN/name)==digest,name
    for name,digest in visual['figures'].items():assert sha(REPORT/'figures'/name)==digest,name
    assert replay['generated_evaluation_populations_bitwise_equal']
    assert all(x['bitwise_equal'] for x in replay['classifiers'])
    assert all(x['pre_ADC_bitwise'] and x['post_ADC_bitwise'] for x in replay['generator_chunks'])
    assert replay['acf005']['all_24576_paired_rows_bitwise'] and replay['acf005']['all_2000_group_bootstrap_draws_reconstructed']
    assert replay['acf005']['conditional_permutation']['all_values_exact']
    assert review['status']=='passed_for_bounded_delivery' and not review['blockers']
    assert tests['status']=='passed' and tests['scientific_updates']==0
    assert scope['full480_integration_supported'] is False and scope['formal_response_training_updates']==0
    design=summary['read_only_480_design_check_allowed']
    status=('completed_100point_candidate_improvement_measurement_uncertain' if design else
            'completed_fixed_candidate_stopped_without_480_entry')
    next_action=('Read-only480 design check completed; any sensitivity simulation or response training requires a separately declared protocol.' if design else
                 'Preserve fixed005 endpoint and negative findings; no added updates, seed selection, model expansion, or480 integration.')
    final=dict(status=status,run_id=ID,closed_UTC=now,training_ended_UTC=native['ended_utc'],active_scientific_jobs=0,
        noise_training_updates=5000,classifier_evaluation_updates=1200,response_training_updates=0,
        cumulative_generator_updates=92000,
        accounting='77000 original +5000 repair002 +5000 improvement004 +5000 acf005; evaluation classifier updates excluded',
        source_commit=start['source_commit'],native_training_exit_code=0,native_training_receipt=native,
        scientific_payloads_verified=len(complete['files']),execution_sources_verified=len(complete['source_files']),
        complete_sha256=sha(science/'COMPLETE.json'),final_checkpoint_sha256=complete['final_checkpoint_sha256'],
        config_sha256=complete['config_sha256'],source_manifest_sha256=complete['source_manifest_sha256'],
        train_container_sha256=config['train_container_sha256'],train_rows_used=21047,train_total_rows=21047,
        train_total_visits=replay['sampler_total_rows'],min_visits=replay['min_visits'],max_visits=replay['max_visits'],
        TRAIN_match=summary['TRAIN_match'],TRAIN_improved=summary['TRAIN_improved'],
        candidate_checks_passed=summary['candidate_checks_passed'],candidate_failed_checks=summary['candidate_failed_checks'],
        legacy_gate_passed=summary['legacy_gate_passed'],legacy_failed_checks=summary['legacy_failed_checks'],
        control_status=summary['control_status'],read_only_480_design_check_allowed=design,
        scientifically_qualified=False,old_gate_changed=False,model_scope='observed_baseline100_only',model_parameters=5151,
        full480_integration_supported=False,full_detector_response_match_established=False,
        validation_previously_viewed=True,independent_acquisition_validations=0,
        support_scope='reports/'+ID+'/SUPPORT_SCOPE.json',support_scope_sha256=sha(REPORT/'SUPPORT_SCOPE.json'),
        replay=dict(path='runs/'+ID+'/runs/replay/REPLAY_VERIFICATION.json',sha256=sha(RUN/'runs/replay/REPLAY_VERIFICATION.json'),
            explicit_generator_rows=1024,paired_diagnostic_rows=24576,evaluation_generated_rows=10240,
            classifiers=6,classifier_score_rows=3072,ACF_bootstrap_draws=2000,conditional_permutations=9999,
            all_bitwise_equal=True,added_training_updates=0),
        tests=tests,postflight=post,visual_QA=visual,
        results_review_receipt='receipts/'+ID+'/RESULTS_INDEPENDENT_REVIEW.json',
        results_review_sha256=sha(REC/'RESULTS_INDEPENDENT_REVIEW.json'),
        report='reports/'+ID+'/N01_ACF005_REPORT.html',
        report_files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(REPORT.rglob('*')) if p.is_file()},
        next_action=next_action)
    write(REC/'FINAL_RECEIPT.json',final)
    p=ROOT/'README.md';text=p.read_text(encoding='utf8')
    old='**当前任务：005 ACF 目标覆盖修复已预登记。先做零训练初始化对照，再从相同初始化训练唯一 5000 次更新候选；旧门、候选检查与对照校准分别报告，正式脉冲训练预算为 0。** 见 [005 协议](docs/NOISE_ACF005_PROTOCOL.md) 与 `noise_acf005_config.json`。'
    new=f'**005 已完成：噪声更新 5000 次，响应更新 0；TRAIN ACF 匹配 {summary["TRAIN_match"]}，改善 {summary["TRAIN_improved"]}；候选开发筛查 {summary["candidate_checks_passed"]}，legacy 总门 {summary["legacy_gate_passed"]}。控制测量状态仍不确定。当前科学作业为 0。** 见 [005 报告](reports/{ID}/README.md)、[分离评价回执](reports/{ID}/EVALUATION_RECEIPT.json)、[100/480 支持范围](reports/{ID}/SUPPORT_SCOPE.md) 和 [005 协议](docs/NOISE_ACF005_PROTOCOL.md)。'
    assert old in text;text=text.replace(old,new,1).replace('**本轮已完成：noise_improvement_004','**前次已完成：noise_improvement_004',1)
    text=text.replace('最新独立噪声候选为 `docs/NOISE_IMPROVEMENT_004_PROTOCOL.md` 与 `noise_improvement_004_config.json`','最新独立噪声候选为 `docs/NOISE_ACF005_PROTOCOL.md` 与 `noise_acf005_config.json`',1)
    p.write_text(text,encoding='utf8',newline='\n')
    program=read(ROOT/'PROGRAM.json')
    program.update(status=status,active_repair=None,active_scientific_jobs=0,current_stage='fixed005_evaluated_and_stopped',
        updated_UTC=now,last_verified_UTC=post['UTC'],new_training_updates=92000,current_run_training_updates=5000,
        latest_run_id=ID,latest_acf005_run=ID,acf005_noise_updates=5000,acf005_response_updates=0,acf005_noise_training_updates=5000,acf005_response_training_updates=0,
        acf005_classifier_evaluation_updates=1200,acf005_training_pid=None,acf005_supervisor_pid=None,
        acf005_historical_training_pid=start['training_pid'],acf005_historical_supervisor_pid=start['supervisor_pid'],
        acf005_ended_UTC=native['ended_utc'],acf005_native_exit_code=0,acf005_candidate_checks_passed=summary['candidate_checks_passed'],
        acf005_legacy_gate_passed=summary['legacy_gate_passed'],acf005_control_status=summary['control_status'],
        acf005_read_only_480_design_check_allowed=design,acf005_report=final['report'],
        acf005_completion_receipt='receipts/'+ID+'/FINAL_RECEIPT.json',acf005_full480_integration_supported=False,
        training_update_accounting=final['accounting'],next_action=next_action)
    write(ROOT/'PROGRAM.json',program)
    jobs=read(ROOT/'jobs.json');jobs.update(updated_UTC=now,active_scientific_jobs=0)
    job=next(x for x in jobs['jobs'] if x['job_id']=='N01_NOISE_ACF005_20261012')
    job.update(status=status,pid=None,supervisor_pid=None,historical_training_pid=start['training_pid'],
        historical_supervisor_pid=start['supervisor_pid'],ended_UTC=native['ended_utc'],last_verified_UTC=post['UTC'],
        native_exit_code=0,observed_noise_updates=5000,noise_training_updates=5000,response_training_updates=0,
        classifier_evaluation_updates=1200,candidate_checks_passed=summary['candidate_checks_passed'],
        legacy_gate_passed=summary['legacy_gate_passed'],control_status=summary['control_status'],
        report=final['report'],completion_receipt=program['acf005_completion_receipt'])
    write(ROOT/'jobs.json',jobs)
    pointer_path=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';pointer=read(pointer_path)
    pointer.update(status=status,active_repair=None,active_scientific_jobs=0,updated_UTC=now,
        plan_sha256=sha(ROOT/'README.md'),latest_run_id=ID,latest_acf005_report=str(ROOT/final['report']),
        acf005_configuration=str(ROOT/'noise_acf005_config.json'),acf005_completion_receipt=str(ROOT/program['acf005_completion_receipt']),
        acf005_noise_training_updates=5000,acf005_response_training_updates=0,
        acf005_candidate_checks_passed=summary['candidate_checks_passed'],acf005_legacy_gate_passed=summary['legacy_gate_passed'],
        acf005_full480_integration_supported=False)
    write(pointer_path,pointer)
    print(json.dumps(dict(status=status,active_scientific_jobs=0,noise_updates=5000,response_updates=0,
        final_receipt_sha256=sha(REC/'FINAL_RECEIPT.json'))))

if __name__=='__main__':main()
