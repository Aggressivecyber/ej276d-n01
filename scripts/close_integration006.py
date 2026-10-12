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
    science=RUN/'runs/science';complete=read(science/'COMPLETE.json');decision=read(science/'DECISION.json')
    replay=read(RUN/'runs/verification/REPLAY_VERIFICATION.json');native=read(RUN/'runs/supervisor/exit_receipt.json')
    cfg=read(ROOT/'integration006_config.json');tests=read(REC/'TEST_SUMMARY.json')
    post=read(REC/'POSTFLIGHT.json');visual=read(REC/'VISUAL_QA.json');review=read(REC/'RESULTS_INDEPENDENT_REVIEW.json')
    start=read(REC/'START_RECEIPT.json')
    assert complete['status']=='zero_training_completed' and native['exit_code']==0
    assert complete['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0)
    assert replay['status']=='passed' and replay['passed'] and replay['complete_sha256']==sha(science/'COMPLETE.json')
    assert replay['source_manifest_sha256']==sha(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json')
    assert complete['provenance']['config_sha256']==sha(ROOT/'integration006_config.json')
    assert replay['independent_decision']['outcome']==decision['outcome']==complete['decision_outcome']
    assert not decision['scientific_acceptance'] and decision['response_updates_authorized_here']==0
    assert tests['status']=='passed' and tests['scientific_updates']==0
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
        source_manifest_sha256=sha(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json'),configuration_sha256=sha(ROOT/'integration006_config.json'),
        complete_sha256=sha(science/'COMPLETE.json'),decision=decision,
        replay_receipt='runs/'+ID+'/runs/verification/REPLAY_VERIFICATION.json',
        replay_sha256=sha(RUN/'runs/verification/REPLAY_VERIFICATION.json'),
        all_frozen_inputs_and_generated_waveforms_verified=True,
        independent_acquisition_validations=0,physical_noise_identified=False,response_match_established=False,
        next_response_protocol_allowed=decision['next_response_protocol_allowed'],formal_response_updates=0,
        report='reports/'+ID+'/N01_INTEGRATION006_REPORT.html',
        report_files={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(REPORT.rglob('*')) if p.is_file()},
        tests_receipt='receipts/'+ID+'/TEST_SUMMARY.json',tests_sha256=sha(REC/'TEST_SUMMARY.json'),
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
    job=next(item for item in jobs['jobs'] if item['job_id']=='N01_INTEGRATION006_20261012')
    job.update(status=status,pid=None,supervisor_pid=None,historical_evaluation_pid=start['evaluation_pid'],
        historical_supervisor_pid=start['supervisor_pid'],ended_UTC=native['ended_utc'],last_verified_UTC=post['UTC'],
        native_exit_code=0,observed_noise_updates=0,response_training_updates=0,classifier_evaluation_updates=0,
        actual_counts=complete['actual_counts'],outcome=outcome,report=final['report'],
        completion_receipt=program['integration006_completion_receipt'])
    write(ROOT/'jobs.json',jobs)
    p=ROOT/'README.md';text=p.read_text(encoding='utf8')
    header=f'**当前主线 006 已完成：{outcome}；新增训练 0 次，当前科学作业 0。** [完整480点与PSD整合报告](reports/{ID}/N01_INTEGRATION006_REPORT.html) · [冻结协议](docs/INTEGRATION_006_PROTOCOL.md) · [完成回执](receipts/{ID}/FINAL_RECEIPT.json)。\n\n'
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
