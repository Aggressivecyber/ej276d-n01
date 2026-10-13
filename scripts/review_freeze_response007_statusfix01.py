"""Self-review, log first-update failure, and freeze its bounded continuation."""
from pathlib import Path
import ast
import datetime as dt
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];ID='response_007_20261013';REC=ROOT/'receipts'/ID/'status_fix01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    cfg=read(ROOT/'response007_config.json');reg=read(REC/'PREREGISTRATION.json');old=read(ROOT/'RESPONSE007_SOURCE_MANIFEST.json')
    for n,d in old['files'].items():assert sha(ROOT/n)==d,n
    assert sha(ROOT/'response007_config.json')==reg['config_sha256'] and sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md')==reg['protocol_sha256']
    newfiles=['src/n01/train_response007_statusfix01.py','scripts/prepare_response007_statusfix01.py',
        'scripts/launch_response007_statusfix01.sh','scripts/start_response007_statusfix01_remote.py',
        'scripts/review_freeze_response007_statusfix01.py','tests/test_response007_resume.py','docs/RESPONSE_007_STATUS_FIX01.md']
    for name in newfiles:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf8'))
    oldtext=(ROOT/'src/n01/train_response007.py').read_text(encoding='utf8')
    newtext=(ROOT/'src/n01/train_response007_statusfix01.py').read_text(encoding='utf8')
    (REC/'ENTRYPOINT_DIFF.txt').write_text(''.join(difflib.unified_diff(oldtext.splitlines(True),newtext.splitlines(True),fromfile='original',tofile='statusfix01')),encoding='utf8')
    environment=os.environ.copy();environment.update(PYTHONPATH=str(ROOT/'src'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
        TEMP='E:/GAN/runtime_tmp',TMP='E:/GAN/runtime_tmp',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_response007_resume.py','-v'],cwd=ROOT,env=environment,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8')
    (REC/'SELF_TEST_OUTPUT.txt').write_text(result.stdout,encoding='utf8')
    if result.returncode:raise RuntimeError(result.stdout)
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    write(REC/'SELF_REVIEW.json',dict(UTC=now,status='passed_primary_agent_self_review_for_exact_continuation',
        original_files_unchanged=len(old['files']),new_source_hashes={n:sha(ROOT/n) for n in newfiles},
        native_exit_code=result.returncode,artificial_tests=2,core_model_loss_evaluation_unchanged=True,
        artificial_continuation_parameters_optimizer_rng_and_sampler_exact=True,
        previous_scientific_G_updates=1,new_execution_G_budget=37999,cumulative_G_budget=38000,
        independent_reviewer_used=False,scientific_response_match_established=False,
        diff_sha256=sha(REC/'ENTRYPOINT_DIFF.txt'),test_output_sha256=sha(REC/'SELF_TEST_OUTPUT.txt')))
    paths={ROOT/n for n in old['files']};paths.update(ROOT/n for n in newfiles)
    paths.update((ROOT/'provenance/response007_statusfix01').iterdir())
    paths.update(REC/n for n in ['PREREGISTRATION.json','SELF_REVIEW.json','SELF_TEST_OUTPUT.txt','ENTRYPOINT_DIFF.txt'])
    paths.add(ROOT/'RESPONSE007_SOURCE_MANIFEST.json')
    manifest=dict(old);manifest.update(UTC=now,attempt_id='status_fix01',previous_G_updates=1,previous_D_updates=0,
        previous_scientific_data_decoded=True,new_execution_G_budget=37999,
        amendment_sha256=sha(ROOT/'docs/RESPONSE_007_STATUS_FIX01.md'),
        original_source_manifest_sha256=sha(ROOT/'RESPONSE007_SOURCE_MANIFEST.json'),
        freeze_scope='Before this continuation; original completed update retained and resumed once',
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    target=ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json';assert not target.exists();write(target,manifest);paths.add(target)
    run=ROOT/'runs'/ID/'status_fix01';run.mkdir(exist_ok=False)
    package=run/'EXECUTION_SOURCE.tar.gz'
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(paths):archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(package,'r:gz') as archive:
        for member in archive.getmembers():assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==sha(ROOT/member.name)
    write(REC/'EXECUTION_PACKAGE.json',dict(UTC=now,path=package.relative_to(ROOT).as_posix(),sha256=sha(package),bytes=package.stat().st_size,
        source_manifest_sha256=sha(target),members=len(paths),all_members_reopened_verified=True))
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json');pointer=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';ptr=read(pointer)
    program.update(status='response007_statusfix01_registered',current_stage='resume_after_status_recording_failure',active_scientific_jobs=0,
        new_training_updates=92001,current_run_training_updates=1,response007_G_updates=1,response007_D_updates=0,response007_C2ST_updates=0,
        response007_initial_native_exit_code=1,response007_statusfix01_manifest=target.name,
        next_action='Resume the preserved first update in a new directory; no change to 38000 cumulative G budget.',updated_UTC=now)
    for job in jobs['jobs']:
        if job['job_id']=='N01_RESPONSE007_20261013':
            job.update(status='failed_status_recording_after_first_G_update',native_exit_code=1,G_updates=1,D_updates=0,C2ST_updates=0,
                started_UTC='2026-10-13T01:34:23.013561+00:00',ended_UTC='2026-10-13T01:34:27.016058+00:00',
                historical_supervisor_pid=157422,historical_training_pid=157434,failure_receipt='provenance/response007_statusfix01/FAILURE.json')
    jobs['jobs'].append(dict(job_id='N01_RESPONSE007_STATUSFIX01_20261013',status='registered_exact_continuation',registered_UTC=now,
        host='100.82.231.102',pid=None,supervisor_pid=None,previous_G_updates=1,new_execution_G_updates_planned=37999,
        cumulative_G_updates_planned=38000,D_updates_planned=96000,C2ST_updates_planned=4200,noise_updates_planned=0,
        remote_root=reg['remote_root'],configuration='response007_config.json',protocol='docs/RESPONSE_007_PROTOCOL.md',
        amendment='docs/RESPONSE_007_STATUS_FIX01.md',source_manifest=target.name))
    jobs['updated_UTC']=now;write(ROOT/'PROGRAM.json',program);write(ROOT/'jobs.json',jobs)
    ptr.update(status=program['status'],active_scientific_jobs=0,response007_G_updates=1,response007_D_updates=0,
        response007_initial_native_exit_code=1,response007_statusfix01_manifest=str(target),updated_UTC=now)
    write(pointer,ptr)
    print(json.dumps(dict(status='frozen_continuation',package_sha256=sha(package),package_bytes=package.stat().st_size,
        source_manifest_sha256=sha(target),previous_G_updates=1,new_G_budget=37999)))
if __name__=='__main__':main()
