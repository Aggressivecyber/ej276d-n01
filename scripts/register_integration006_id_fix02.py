"""Register an identity metadata repair while retaining actual partial counts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1];ID='integration_006_20261012'
RUN=ROOT/'runs'/ID;REC=ROOT/'receipts'/ID;FIX=REC/'id_fix02'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    FIX.mkdir(exist_ok=True);dest=FIX/'PREREGISTRATION.json';assert not dest.exists()
    prior=RUN/'binding_fix01/runs';failure=read(prior/'science/FAILURE.json');native=read(prior/'supervisor/exit_receipt.json')
    post=read(REC/'binding_fix01/POSTFLIGHT.json');old=read(ROOT/'INTEGRATION006_BINDING_FIX01_SOURCE_MANIFEST.json')
    assert native['exit_code']==1 and failure['error_type']=='TypeError' and 'numpy.str_' in failure['error']
    counts=failure['actual_counts'];assert counts['REAL_readout_rows']=={'TRAIN':21047}
    assert counts['noise005_forward']=={'TRAIN':{'batches':1,'events':128}} and 'model_forward' not in counts
    assert counts['extension_forward']=={'TRAIN/'+h:{'batches':1,'events':128} for h in ['fast','slow','common_level']}
    assert failure['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0) and not failure['partial_payload_sha256']
    assert not post['active_N01_python'] and not post['GPU_compute_processes']
    assert sha(ROOT/'integration006_config.json')==old['config_sha256']
    assert sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md')==old['protocol_sha256']
    for rel in ['science/FAILURE.json','supervisor/launch.json','supervisor/exit_receipt.json','supervisor/resource_status.json','supervisor/console.log']:
        target=REC/'failed_binding_fix01'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();shutil.copy2(prior/rel,target);assert sha(target)==sha(prior/rel)
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    reg=dict(run_id=ID,attempt_id='id_fix02',registered_UTC=now,
        amendment='docs/INTEGRATION_006_ID_FIX02.md',amendment_sha256=sha(ROOT/'docs/INTEGRATION_006_ID_FIX02.md'),
        unchanged_configuration_sha256=old['config_sha256'],unchanged_protocol_sha256=old['protocol_sha256'],
        previous_source_manifest_sha256=sha(ROOT/'INTEGRATION006_BINDING_FIX01_SOURCE_MANIFEST.json'),
        previous_source_commit='e1b8ab0cc1609bd21f18a390fd85bd397dc0d889',previous_native_exit_code=1,
        previous_failure_sha256=sha(prior/'science/FAILURE.json'),previous_actual_counts=counts,
        SOURCE_REAL_datasets_previously_decoded=True,previous_response_forward_rows=0,
        previous_noise005_forward_rows=128,previous_extension_forward_rows=384,
        fixed_seeds_replayed_without_added_unique_draws=True,scientific_update_budget=0,new_attempt_budget=1,
        both_failed_attempts_preserved=True,no_automatic_retry_loop=True,response_training_allowed=False)
    write(dest,reg)
    p=read(ROOT/'PROGRAM.json');p.update(status='integration006_id_fix02_registered',active_scientific_jobs=0,
        current_stage='original_string_identity_serialization_repair',updated_UTC=now,last_verified_UTC=post['UTC'],
        integration006_PID=None,integration006_supervisor_pid=None,
        integration006_binding_fix01_native_exit_code=1,integration006_partial_noise_forward_rows=128,
        integration006_partial_extension_forward_rows=384,integration006_partial_response_forward_rows=0)
    write(ROOT/'PROGRAM.json',p)
    jobs=read(ROOT/'jobs.json');jobs.update(updated_UTC=now,active_scientific_jobs=0)
    oldjob=next(j for j in jobs['jobs'] if j['job_id']=='N01_INTEGRATION006_BINDING_FIX01_20261012')
    oldjob.update(status='failed_string_ID_serialization_before_response_forward',native_exit_code=1,pid=None,supervisor_pid=None,
        ended_UTC=native['ended_utc'],historical_evaluation_pid=12568,historical_supervisor_pid=native['supervisor_pid'],
        actual_counts=counts,noise_updates=0,response_updates=0,
        failure_receipt='receipts/'+ID+'/failed_binding_fix01/science/FAILURE.json')
    jobs['jobs'].append(dict(job_id='N01_INTEGRATION006_ID_FIX02_20261012',status='registered',registered_UTC=now,
        host='100.82.231.102',pid=None,supervisor_pid=None,noise_updates_planned=0,response_updates_planned=0,
        classifier_updates_planned=0,GMM_fits_planned=0,configuration='integration006_config.json',
        protocol='docs/INTEGRATION_006_PROTOCOL.md',amendment=reg['amendment'],
        preregistration='receipts/'+ID+'/id_fix02/PREREGISTRATION.json',
        remote_root='/home/rylan/projects/ej276d-n01/integration_006_20261012_id_fix02'))
    write(ROOT/'jobs.json',jobs)
    ptr=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';value=read(ptr)
    value.update(status=p['status'],active_scientific_jobs=0,updated_UTC=now,
        integration006_binding_fix01_native_exit_code=1,integration006_partial_noise_forward_rows=128)
    write(ptr,value);print(json.dumps(reg))
if __name__=='__main__':main()
