"""Freeze a time-limited partial comparison, preserving uncompleted endpoints."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];ID='response_007_20261013';REC=ROOT/'receipts'/ID/'deadline01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    REC.mkdir(parents=True,exist_ok=False)
    old=read(ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json')
    for name,digest in old['files'].items():assert sha(ROOT/name)==digest,name
    gpuroot=ROOT/'runs'/ID/'status_fix01'
    for suffix,name in [('resume_engineering_supervisor','GPU_OPTIMIZER_DEVICE_TEST_FAILED'),('resume_values_engineering_supervisor','GPU_RESUME_VALUES_PASSED')]:
        for original,ending in [('exit_receipt.json','.json'),('console.log','.txt')]:
            shutil.copy2(gpuroot/suffix/original,REC/(name+ending))
    assert read(REC/'GPU_RESUME_VALUES_PASSED.json')['exit_code']==0
    reg=read(ROOT/'receipts'/ID/'status_fix01/PREREGISTRATION.json');now=dt.datetime.now(dt.timezone.utc).isoformat()
    bound=dict(run_id=ID,attempt_id='deadline01',registered_UTC=now,updates_per_arm=4000,previous_G_updates=1,new_G_budget=7999,
        cumulative_G_budget=8000,D_budget=6000,C2ST_budget=0,noise_updates=0,
        stages=dict(mean_response=3000,joint_stochastic=1000,low_learning_rate=0),
        stop_updates_UTC='2026-10-13T03:30:00+00:00',user_closeout_deadline_UTC='2026-10-13T04:00:00+00:00',wall_limit_s=5400,
        remote_root='/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01',
        resume_checkpoint_sha256=reg['resume_checkpoint_sha256'],
        diagnostic_seeds=dict(TRAIN_response=2026101341,TRAIN_noise005=2026101342,TRAIN_extension=2026101343,
            selected_TRAIN=2026101344,real_TRAIN=2026101345,bootstrap_TRAIN=2026101346,evaluation_projection=2026101347),
        diagnostic_SOURCE_parents=6593,draws_per_parent=4,REAL_TRAIN_rows=21047,matched_per_class=2048,bootstrap_draws=1000,
        DEV_waveforms_read=False,original_19000_endpoint_remains_uncompleted=True,source_hypotheses_or_loss_changed=False,
        new_public_release_authorized=False,coordination_source_thread='01a132e8-765f-7689-aea6-75d85ca86080')
    write(ROOT/'response007_deadline01_config.json',bound)
    launcher=(ROOT/'scripts/launch_response007_statusfix01.sh').read_text(encoding='utf8')
    launcher=launcher.replace('response_007_20261013_statusfix01','response_007_20261013_deadline01').replace('28800s','5400s')
    launcher=launcher.replace('n01.train_response007_statusfix01','n01.train_response007_deadline01')
    launcher=launcher.replace('RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json','RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json')
    launcher=launcher.replace('--config "$root/response007_config.json"','--config "$root/response007_config.json" --bound-config "$root/response007_deadline01_config.json"')
    (ROOT/'scripts/launch_response007_deadline01.sh').write_text(launcher,encoding='utf8',newline='\n')
    starter=(ROOT/'scripts/start_response007_statusfix01_remote.py').read_text(encoding='utf8')
    starter=starter.replace('response_007_20261013_statusfix01','response_007_20261013_deadline01')
    starter=starter.replace('RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json','RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json')
    starter=starter.replace('scripts/launch_response007_statusfix01.sh','scripts/launch_response007_deadline01.sh')
    (ROOT/'scripts/start_response007_deadline01_remote.py').write_text(starter,encoding='utf8',newline='\n')
    new=['src/n01/train_response007_deadline01.py','response007_deadline01_config.json','docs/RESPONSE_007_DEADLINE01.md',
        'scripts/register_freeze_response007_deadline01.py','scripts/launch_response007_deadline01.sh','scripts/start_response007_deadline01_remote.py',
        'tests/test_response007_resume_values.py']
    for name in new:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf8'))
    code=(ROOT/'src/n01/train_response007_deadline01.py').read_text(encoding='utf8')
    assert 'load_validation' not in code and 'fit_classifier' not in code and 'evaluate_run(' not in code
    assert "engine.update<bound['updates_per_arm']" in code and "engine.update%250==0" in code
    write(REC/'PREREGISTRATION.json',dict(UTC=now,run_id=ID,attempt_id='deadline01',
        parent_config_sha256=sha(ROOT/'response007_config.json'),bound_config_sha256=sha(ROOT/'response007_deadline01_config.json'),
        protocol_sha256=sha(ROOT/'docs/RESPONSE_007_DEADLINE01.md'),previous_G_updates=1,new_G_budget=7999,cumulative_G_budget=8000,
        D_budget=6000,C2ST_budget=0,noise_updates=0,before_this_execution_scientific_forward=True,
        prior_scientific_data_decoded=True,prior_failed_attempt_preserved=True,
        source_hashes={name:sha(ROOT/name) for name in new},self_review_mode='primary_agent_no_subagents',
        review=dict(core_response_loss_unchanged=True,successful_GPU_resume_value_check=True,
            no_DEV_or_C2ST_code_path=True,status_builder_reuses_tested_fix=True,absolute_deadline_and_wall_cap=True,
            final_original_endpoint_not_relabelled=True,blockers=[])))
    paths={ROOT/name for name in old['files']};paths.update(ROOT/name for name in new)
    paths.add(ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json');paths.update(p for p in REC.iterdir() if p.is_file())
    manifest=dict(old);manifest.update(UTC=now,attempt_id='deadline01',new_execution_G_budget=7999,cumulative_G_budget=8000,
        actual_D_budget=6000,actual_C2ST_budget=0,bound_config_sha256=sha(ROOT/'response007_deadline01_config.json'),
        bound_protocol_sha256=sha(ROOT/'docs/RESPONSE_007_DEADLINE01.md'),
        stop_updates_UTC=bound['stop_updates_UTC'],files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)})
    target=ROOT/'RESPONSE007_DEADLINE01_SOURCE_MANIFEST.json';assert not target.exists();write(target,manifest);paths.add(target)
    folder=ROOT/'runs'/ID/'deadline01';folder.mkdir(exist_ok=False)
    package=folder/'EXECUTION_SOURCE.tar.gz'
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(paths):archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(package,'r:gz') as archive:
        for member in archive.getmembers():assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==sha(ROOT/member.name)
    write(REC/'EXECUTION_PACKAGE.json',dict(UTC=now,path=package.relative_to(ROOT).as_posix(),sha256=sha(package),bytes=package.stat().st_size,
        source_manifest_sha256=sha(target),members=len(paths),all_reopened_hashes_verified=True))
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json');pointer=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';ptr=read(pointer)
    program.update(status='response007_deadline01_registered',current_stage='time_bounded_partial_comparison',active_scientific_jobs=0,
        new_training_updates=92001,current_run_training_updates=1,response007_G_updates=1,response007_D_updates=0,response007_C2ST_updates=0,
        response007_deadline01_manifest=target.name,response007_deadline01_configuration='response007_deadline01_config.json',
        next_action='Run a single bounded GPU comparison to 4000 per arm, TRAIN-only diagnostics, then preserve by noon.',updated_UTC=now)
    for job in jobs['jobs']:
        if job['job_id']=='N01_RESPONSE007_STATUSFIX01_20261013':job.update(status='not_started_superseded_by_noon_bound',new_execution_G_updates=0,D_updates=0,C2ST_updates=0)
    jobs['jobs'].append(dict(job_id='N01_RESPONSE007_DEADLINE01_20261013',status='registered',registered_UTC=now,host='100.82.231.102',pid=None,supervisor_pid=None,
        previous_G_updates=1,new_execution_G_updates_planned=7999,cumulative_G_updates_planned=8000,D_updates_planned=6000,C2ST_updates_planned=0,
        noise_updates_planned=0,wall_limit_s=5400,stop_updates_UTC=bound['stop_updates_UTC'],remote_root=bound['remote_root'],
        configuration='response007_deadline01_config.json',protocol='docs/RESPONSE_007_DEADLINE01.md',source_manifest=target.name))
    jobs['updated_UTC']=now;write(ROOT/'PROGRAM.json',program);write(ROOT/'jobs.json',jobs)
    ptr.update(status=program['status'],active_scientific_jobs=0,response007_deadline01_manifest=str(target),response007_G_updates=1,
        response007_deadline01_configuration=str(ROOT/'response007_deadline01_config.json'),updated_UTC=now);write(pointer,ptr)
    print(json.dumps(dict(status='bounded_frozen',sha256=sha(package),bytes=package.stat().st_size,source_manifest_sha256=sha(target),new_G_budget=7999)))
if __name__=='__main__':main()
