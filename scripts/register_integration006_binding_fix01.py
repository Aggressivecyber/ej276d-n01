"""Record the startup failure and a narrow metadata representation continuation."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
ID='integration_006_20261012';RUN=ROOT/'runs'/ID;REC=ROOT/'receipts'/ID
FIX=REC/'binding_fix01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    FIX.mkdir(exist_ok=True)
    dest=FIX/'PREREGISTRATION.json';assert not dest.exists()
    failure=read(RUN/'runs/science/FAILURE.json');native=read(RUN/'runs/supervisor/exit_receipt.json')
    post=read(REC/'FAILED_ATTEMPT_POSTFLIGHT.json');manifest=read(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json')
    assert failure['error']=='Frozen001 model/loss statistics mismatch' and not failure['actual_counts']
    assert failure['updates']==dict(noise=0,response=0,classifier=0,GMM_fits=0) and native['exit_code']==1
    assert not (RUN/'runs/science/COMPLETE.json').exists()
    assert not post['active_N01_python'] and not post['GPU_compute_processes']
    assert sha(ROOT/'integration006_config.json')==manifest['config_sha256']
    assert sha(ROOT/'docs/INTEGRATION_006_PROTOCOL.md')==manifest['protocol_sha256']
    package=read(REC/'EXECUTION_PACKAGE.json');assert sha(RUN/'EXECUTION_SOURCE.tar.gz')==package['sha256']
    for rel in ['runs/science/FAILURE.json','runs/supervisor/launch.json','runs/supervisor/exit_receipt.json','runs/supervisor/resource_status.json','runs/supervisor/console.log']:
        target=REC/'failed_attempt'/Path(rel).relative_to('runs')
        target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        shutil.copy2(RUN/rel,target);assert sha(target)==sha(RUN/rel)
    snapshot=ROOT/'provenance/integration006_before_binding_continuation';snapshot.mkdir()
    for name in ['PROGRAM.json','jobs.json']:
        shutil.copy2(ROOT/name,snapshot/name)
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    reg=dict(run_id=ID,attempt_id='binding_fix01',registered_UTC=now,
        amendment='docs/INTEGRATION_006_BINDING_FIX01.md',amendment_sha256=sha(ROOT/'docs/INTEGRATION_006_BINDING_FIX01.md'),
        unchanged_configuration_sha256=manifest['config_sha256'],unchanged_protocol_sha256=manifest['protocol_sha256'],
        initial_source_commit='f697698023bf228155c14dbee66d2a1a8ac76a8e',initial_source_manifest_sha256=sha(ROOT/'INTEGRATION006_SOURCE_MANIFEST.json'),
        initial_execution_package_sha256=package['sha256'],initial_native_exit_code=1,
        initial_failure_sha256=sha(RUN/'runs/science/FAILURE.json'),
        initial_scientific_dataset_decodes=0,initial_scientific_forward_rows=0,initial_checkpoint_metadata_decoded=True,
        stage_scientific_updates=0,scientific_updates_budget=0,new_attempt_budget=1,
        original_failed_output_preserved=True,no_automatic_retry_loop=True,
        response_training_allowed=False,source_freeze_and_independent_review_required=True)
    write(dest,reg)
    program=read(ROOT/'PROGRAM.json');program.update(status='integration006_binding_fix01_registered',active_scientific_jobs=0,
        current_stage='exact_checkpoint_metadata_representation_repair',updated_UTC=now,last_verified_UTC=post['UTC'],
        integration006_initial_attempt_native_exit_code=1,integration006_initial_attempt_forward_rows=0,
        integration006_initial_failure_receipt='receipts/'+ID+'/failed_attempt/science/FAILURE.json')
    write(ROOT/'PROGRAM.json',program)
    jobs=read(ROOT/'jobs.json');jobs.update(updated_UTC=now,active_scientific_jobs=0)
    old=next(j for j in jobs['jobs'] if j['job_id']=='N01_INTEGRATION006_20261012')
    old.update(status='failed_before_dataset_decode_metadata_representation',native_exit_code=1,
        started_UTC=native['started_utc'],ended_UTC=native['ended_utc'],historical_supervisor_pid=native['supervisor_pid'],
        historical_timeout_pid=native['child_pid'],scientific_forward_rows=0,noise_updates=0,response_updates=0,
        source_commit=reg['initial_source_commit'],failure_receipt=program['integration006_initial_failure_receipt'])
    jobs['jobs'].append(dict(job_id='N01_INTEGRATION006_BINDING_FIX01_20261012',status='registered',registered_UTC=now,
        host='100.82.231.102',pid=None,supervisor_pid=None,noise_updates_planned=0,response_updates_planned=0,
        classifier_updates_planned=0,GMM_fits_planned=0,configuration='integration006_config.json',
        protocol='docs/INTEGRATION_006_PROTOCOL.md',amendment=reg['amendment'],
        preregistration='receipts/'+ID+'/binding_fix01/PREREGISTRATION.json',
        remote_root='/home/rylan/projects/ej276d-n01/integration_006_20261012_binding_fix01'))
    write(ROOT/'jobs.json',jobs)
    pointer_path=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';pointer=read(pointer_path)
    pointer.update(status=program['status'],active_scientific_jobs=0,updated_UTC=now,
        integration006_initial_attempt_native_exit_code=1,integration006_initial_attempt_forward_rows=0)
    write(pointer_path,pointer)
    print(json.dumps(reg))
if __name__=='__main__':main()
