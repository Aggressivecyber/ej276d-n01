"""Close the finite006 run only from execution, replay, review and visual receipts."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
ID='integration_006_20261012';RUN=ROOT/'runs'/ID;REC=ROOT/'receipts'/ID;REPORT=ROOT/'reports'/ID
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')

def main():
    assert not (REC/'FINAL_RECEIPT.json').exists(),'Preserve closed006 results'
    science=RUN/'id_fix02/runs/science';complete=read(science/'COMPLETE.json');decision=read(science/'DECISION.json')
    replay=read(RUN/'id_fix02/runs/verification_adc_fix02/REPLAY_VERIFICATION.json');native=read(RUN/'id_fix02/runs/supervisor/exit_receipt.json')
    cfg=read(ROOT/'integration006_config.json');tests=read(REC/'id_fix02/TEST_SUMMARY.json')
    post=read(REC/'POSTFLIGHT.json');visual=read(REC/'VISUAL_QA.json');review=read(REC/'RESULTS_INDEPENDENT_REVIEW.json')
    start=read(REC/'id_fix02/START_RECEIPT.json')
    initial_failure=read(RUN/'runs/science/FAILURE.json');initial_native=read(RUN/'runs/supervisor/exit_receipt.json')
    assert initial_native['exit_code']==1 and initial_failure['actual_counts']=={}
    binding_failure=read(RUN/'binding_fix01/runs/science/FAILURE.json')
    assert read(RUN/'binding_fix01/runs/supervisor/exit_receipt.json')['exit_code']==1
    assert binding_failure['actual_counts']['noise005_forward']['TRAIN']['events']==128
    assert complete['status']=='zero_training_completed' and native['exit_code']==0
    assert complete['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0)
    assert replay['status']=='passed' and replay['passed'] and replay['complete_sha256']==sha(science/'COMPLETE.json')
    assert replay['source_manifest_sha256']==sha(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json')
    assert replay['verifier_sha256']==sha(ROOT/'scripts/verify_integration006_adc_fix02.py')
    assert read(RUN/'id_fix02/runs/verification_adc_fix02_supervisor/exit_receipt.json')['exit_code']==0
    for fix_name in ('replay_numeric_fix01','replay_adc_fix02'):
        for name,digest in read(REC/fix_name/'BINDING.json')['files'].items():assert sha(ROOT/name)==digest,name
    assert read(RUN/'id_fix02/runs/verification/REPLAY_VERIFICATION.json')['status']=='failed'
    assert read(RUN/'id_fix02/runs/verification_numeric_fix01/REPLAY_VERIFICATION.json')['status']=='failed'
    assert complete['provenance']['config_sha256']==sha(ROOT/'integration006_config.json')
    assert replay['independent_decision']['outcome']==decision['outcome']==complete['decision_outcome']
    assert not decision['scientific_acceptance'] and decision['response_updates_authorized_here']==0
    assert tests['status']=='passed' and tests['new_scientific_updates']==0
    assert review['status']=='passed_for_bounded_delivery' and not review['blockers']
    assert visual['status']=='passed'
    assert not post['active_N01_python'] and not post['GPU_compute_processes']
    assert post['train_container_sha256']==cfg['train_container_sha256']
    for name,digest in complete['payload_sha256'].items():assert sha(science/name)==digest,name
    for name,digest in complete['provenance']['source_files'].items():assert sha(ROOT/name)==digest,name
    for relative,digest in visual['artifact_sha256'].items():assert sha(ROOT/relative)==digest,relative
    outcome=decision['outcome'];status='completed_zero_training_integration006_'+outcome
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    next_action=('Register a separate hypothesis-explicit two-arm response protocol;006 is not physical noise qualification.'
        if decision['next_response_protocol_allowed'] else
        'Stop formal response training for this fixed006 candidate. Preserve all failures and assumption sensitivity; no scenario selection or added draws.')
    final=dict(run_id=ID,status=status,closed_UTC=now,active_scientific_jobs=0,outcome=outcome,
        updates=complete['updates'],cumulative_generator_updates=92000,
        training_update_accounting='77000 original +5000 repair002 +5000 improvement004 +5000 acf005 +0 integration006',
        source_commit=start['source_commit'],native_exit_code=0,actual_counts=complete['actual_counts'],
        source_manifest_sha256=sha(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json'),configuration_sha256=sha(ROOT/'integration006_config.json'),
        complete_sha256=sha(science/'COMPLETE.json'),decision=decision,
        replay_receipt='runs/'+ID+'/id_fix02/runs/verification_adc_fix02/REPLAY_VERIFICATION.json',
        replay_sha256=sha(RUN/'id_fix02/runs/verification_adc_fix02/REPLAY_VERIFICATION.json'),
        verification_attempts=[dict(attempt='initial',native_exit_code=1,response_noise_forward_rows=0,
            REAL_readout_rows_recomputed=21646,receipt_sha256=sha(RUN/'id_fix02/runs/verification/REPLAY_VERIFICATION.json')),
            dict(attempt='numeric_fix01',native_exit_code=1,
                reconstructed_counts=read(REC/'replay_adc_fix02/PREVIOUS_FAILURE.json')['reconstructed_counts'],
                count_evidence=read(REC/'replay_adc_fix02/PREVIOUS_FAILURE.json')['count_evidence'],
                first_batch_generated_waveform_hashes_matched=True,
                receipt_sha256=sha(RUN/'id_fix02/runs/verification_numeric_fix01/REPLAY_VERIFICATION.json')),
            dict(attempt='adc_fix02',native_exit_code=0,complete_frozen_input_replay=True,
                binding_sha256=sha(REC/'replay_adc_fix02/BINDING.json'))],
        engineering_attempts=[dict(attempt='initial',native_exit_code=1,scientific_forward_rows=0,
            failure_sha256=sha(RUN/'runs/science/FAILURE.json'),failure=initial_failure['error']),
            dict(attempt='binding_fix01',native_exit_code=1,actual_counts=binding_failure['actual_counts'],
                failure_sha256=sha(RUN/'binding_fix01/runs/science/FAILURE.json'),failure=binding_failure['error']),
            dict(attempt='id_fix02',native_exit_code=0,completed_scientific_evaluation=True)],
        original_scientific_protocol_and_configuration_unchanged=True,
        completed_attempt_frozen_inputs_and_generated_waveforms_verified=True,
        earlier_unsaved_partial_outputs_bitwise_verified=False,
        prior_repeated_engineering_computation=dict(noise005_forward_rows=128,extension_forward_rows=384,REAL_readout_rows=21047,response_forward_rows=0),
        all_attempts_actual_counts=dict(initial=initial_failure['actual_counts'],binding_fix01=binding_failure['actual_counts'],id_fix02=complete['actual_counts']),
        independent_acquisition_validations=0,physical_noise_identified=False,response_match_established=False,
        next_response_protocol_allowed=decision['next_response_protocol_allowed'],formal_response_updates=0,
        report='reports/'+ID+'/N01_INTEGRATION006_REPORT.html',
        report_files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(REPORT.rglob('*')) if p.is_file()},
        tests_receipt='receipts/'+ID+'/id_fix02/TEST_SUMMARY.json',tests_sha256=sha(REC/'id_fix02/TEST_SUMMARY.json'),
        independent_review_sha256=sha(REC/'RESULTS_INDEPENDENT_REVIEW.json'),visual_QA_sha256=sha(REC/'VISUAL_QA.json'),
        postflight=post,next_action=next_action)
    write(REC/'FINAL_RECEIPT.json',final)
    program=read(ROOT/'PROGRAM.json')
    program.update(status=status,active_repair=None,active_scientific_jobs=0,current_stage='integration006_evaluated_and_closed',
        updated_UTC=now,last_verified_UTC=post['UTC'],new_training_updates=92000,current_run_training_updates=0,
        latest_run_id=ID,latest_integration006_run=ID,integration006_outcome=outcome,
        integration006_noise_updates=0,integration006_response_updates=0,integration006_classifier_updates=0,
        integration006_PID=None,integration006_supervisor_pid=None,
        integration006_report=final['report'],integration006_completion_receipt='receipts/'+ID+'/FINAL_RECEIPT.json',
        integration006_next_response_protocol_allowed=decision['next_response_protocol_allowed'],
        training_update_accounting=final['training_update_accounting'],next_action=next_action)
    write(ROOT/'PROGRAM.json',program)
    jobs=read(ROOT/'jobs.json');jobs.update(updated_UTC=now,active_scientific_jobs=0)
    job=next(item for item in jobs['jobs'] if item['job_id']=='N01_INTEGRATION006_ID_FIX02_20261012')
    job.update(status=status,pid=None,supervisor_pid=None,historical_evaluation_pid=start['evaluation_pid'],
        historical_supervisor_pid=start['supervisor_pid'],ended_UTC=native['ended_utc'],last_verified_UTC=post['UTC'],
        native_exit_code=0,observed_noise_updates=0,response_training_updates=0,classifier_evaluation_updates=0,
        verification_status='passed',verification_supervisor_pid=None,
        actual_counts=complete['actual_counts'],outcome=outcome,report=final['report'],
        completion_receipt=program['integration006_completion_receipt'])
    write(ROOT/'jobs.json',jobs)
    p=ROOT/'README.md';text=p.read_text(encoding='utf8')
    outcome_text=dict(interface_fail='接口检查失败',sensitivity_detected='接口检查通过，但低电荷 PSD 展宽对长程假设敏感',
        MC_inconclusive='固定预算下稳定性尚不确定',sensitivity_within_family='在三个有限假设内满足稳定性容差')[outcome]
    header=f'**当前主线 006 已完成：{outcome_text}；新增训练 0 次，当前科学作业 0。** [完整480点与PSD整合报告](reports/{ID}/N01_INTEGRATION006_REPORT.html) · [冻结协议](docs/INTEGRATION_006_PROTOCOL.md) · [完成回执](receipts/{ID}/FINAL_RECEIPT.json)。\n\n'
    assert '当前主线 006 已完成' not in text
    lines=text.splitlines(keepends=True);lines.insert(1,'\n'+header);p.write_text(''.join(lines),encoding='utf8',newline='\n')
    pointer_path=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';pointer=read(pointer_path)
    pointer.update(status=status,active_repair=None,active_scientific_jobs=0,updated_UTC=now,plan_sha256=sha(p),
        latest_run_id=ID,integration006_outcome=outcome,integration006_report=str(ROOT/final['report']),
        integration006_completion_receipt=str(REC/'FINAL_RECEIPT.json'),integration006_noise_updates=0,
        integration006_response_updates=0,integration006_next_response_protocol_allowed=decision['next_response_protocol_allowed'])
    write(pointer_path,pointer)
    print(json.dumps(dict(status=status,active_scientific_jobs=0,cumulative_generator_updates=92000,final_receipt_sha256=sha(REC/'FINAL_RECEIPT.json'))))

if __name__=='__main__':main()
