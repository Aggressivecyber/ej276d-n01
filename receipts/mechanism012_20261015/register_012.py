"""Register the separately bounded012 diagnostic after a human server-start notice."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT=Path('E:/GAN/research/N01')
RECEIPTS=ROOT/'receipts/mechanism012_20261015'
OUT=RECEIPTS/'registration_01'
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value,new=False):
    path=Path(path)
    if new and path.exists():
        raise RuntimeError(f'Preserve existing file: {path}')
    temporary=path.with_name(path.name+'.register012_tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def main():
    OUT.mkdir(exist_ok=False)
    policy=read(ROOT/'mechanism012_policy.json')
    preflight=read(RECEIPTS/'restart_preflight_01/RESTART_PREFLIGHT.json')
    native=preflight['native']
    assert preflight['SSH_native_exit_code']==0 and not native['active_project_python_processes']
    assert native['exclusive_project_lock_free'] and not native['GPU_compute_apps'] and not native['new_runtime_exists']
    assert native['RAM_available_bytes'] >= 10*1024**3 and native['disk_free_bytes'] >= 4*1024**3
    prior_root=ROOT/'receipts/stability011_20261014/user_server_pause_01'
    prior=read(prior_root/'FINAL_SERVER_PAUSE_RECEIPT.json')
    independent_path=prior_root/'independent_readback_01/READBACK.json'
    assert read(independent_path)['status']=='PASS'
    assert sha(independent_path)==policy['prior_stability_required']['independent_readback_sha256']
    assert prior['bootstrap']==dict(attempted=500,computed=500,saved=500)
    assert prior['classification']==policy['prior_stability_required']['classification']
    old_complete=ROOT/'runs/stability011_20261014/collected_01/runs/primary/artifacts/COMPLETE.json'
    remote_complete=policy['stability011_remote_root']+'/runs/primary/artifacts/COMPLETE.json'
    assert native['prior_artifacts'][remote_complete]['sha256']==sha(old_complete)
    design=read(ROOT/'receipts/stability011_20261014/gradient_design_01/DESIGN_RECEIPT.json')
    for name in ('final.pt','EVALUATION_BANK.npz'):
        row=design['bound_input_references']['shape010_artifacts'][name]
        assert native['prior_artifacts'][row['path']]['sha256']==row['sha256']
    local_manifest=read(ROOT/'SHAPE010_SOURCE_MANIFEST.json')
    old_source_verified={name:sha(ROOT/name) for name in local_manifest['files']}
    assert old_source_verified==local_manifest['files'] and len(old_source_verified)==53
    timestamp=datetime.now(timezone.utc)
    utc=timestamp.isoformat()
    for path in (ROOT/'PROGRAM.json',ROOT/'jobs.json',CURRENT):
        shutil.copyfile(path,OUT/path.name)
    program=read(ROOT/'PROGRAM.json')
    jobs=read(ROOT/'jobs.json')
    assert program['new_training_updates']==159667 and jobs['active_scientific_jobs']==0
    assert not any(job.get('run_id')=='mechanism012_20261015' for job in jobs['jobs'])
    assert program['stability011_closeout_complete'] is True
    assert program['followup_automation_status']=='PAUSED'
    next_action='Complete012 engineering and independent contract checks; freeze new69-source/36-input schema, deploy once and launch the bounded11-query zero-update diagnosis. After correct native first proof schedule one wake in the same n01 heartbeat and rest.'
    common=dict(status='mechanism012_registered_not_started',updated_UTC=utc,active_scientific_jobs=0,
        active_scientific_jobs_status_basis='fresh restart preflight empty;012 not started',
        latest_run_id='mechanism012_20261015',latest_training_run_id='shape010_20261014',
        active_research='mechanism012_20261015',current_stage='mechanism012_engineering_and_source_freeze',
        current_run_training_updates=0,user_requested_pause=False,resume_requires_user_server_started_notice=False,
        resumed_by_user_notice='2026-10-15: 继续推进 服务器启动了',resume_verified_UTC=native['UTC'],
        server_boot_id_at_resume=native['boot_id'],next_action=next_action,
        mechanism012_configuration='mechanism012_policy.json',mechanism012_protocol='docs/MECHANISM012_PROTOCOL.md',
        mechanism012_source_file_count=69,mechanism012_source_freeze_pending=True,
        mechanism012_planned_old_A_rows=2048,mechanism012_planned_noise005_rows=2048,mechanism012_planned_head_final_rows=2048,
        mechanism012_planned_scalar_AD_queries=9,mechanism012_planned_masked_VJP_queries=2,
        mechanism012_max_ADC_rows=49152,mechanism012_model_forward_rows=0,mechanism012_gradient_evaluations=0,
        mechanism012_optimizer_created=0,mechanism012_G_updates=0,mechanism012_D_updates=0,
        mechanism012_analysis_pid=None,mechanism012_supervisor_pid=None,
        mechanism012_registration_receipt='receipts/mechanism012_20261015/registration_01/REGISTRATION_RECEIPT.json',
        mechanism012_restart_preflight='receipts/mechanism012_20261015/restart_preflight_01/RESTART_PREFLIGHT.json',
        followup_automation_status='PAUSED',followup_scheduled_local=None,followup_next_run_local=None,
        scientific_status='response_match_not_established')
    program.update(common)
    write(ROOT/'PROGRAM.json',program)
    current=read(CURRENT)
    current.update(common)
    for key in ('mechanism012_configuration','mechanism012_protocol','mechanism012_registration_receipt','mechanism012_restart_preflight'):
        current[key]=str(ROOT/common[key])
    write(CURRENT,current)
    job=dict(job_id='N01_MECHANISM012_20261015',run_id='mechanism012_20261015',
        status='registered_not_started',registered_UTC=utc,user_authorization=policy['user_authorization'],
        scientific_job=True,diagnostic_only=True,host='100.82.231.102',remote_root=policy['remote_root'],
        policy='mechanism012_policy.json',protocol='docs/MECHANISM012_PROTOCOL.md',
        source_manifest='MECHANISM012_SOURCE_MANIFEST.json',bindings='MECHANISM012_BINDINGS.json',
        planned_budgets=policy['budgets'],old_A_rows=0,noise005_rows=0,head_final_rows=0,
        scalar_AD_queries=0,masked_endpoint_VJP_queries=0,ADC_hard_or_STE_rows=0,
        optimizer_created=0,optimizer_steps=0,G_updates=0,D_updates=0,noise_updates=0,
        population_updates=0,C2ST_updates=0,DEV_rows=0,DEV_numeric_read=False,
        source_freeze_pending=True,supervisor_pid=None,analysis_pid=None,
        source_parent_clusters=512,draws_per_parent=4,matched_REAL_rows=2048,
        head_update=2000,head_parameter_count=1288,resource_limits=policy['resource_limits'],
        scientific_status='response_match_not_established',automatic_restart=False)
    jobs['jobs'].append(job);jobs['updated_UTC']=utc;jobs['active_scientific_jobs']=0
    jobs['user_requested_pause']=False;jobs['resume_requires_user_server_started_notice']=False
    write(ROOT/'jobs.json',jobs)
    receipt=dict(schema='n01/mechanism012-registration/v1',UTC=utc,
        local_Asia_Singapore=timestamp.astimezone(timezone(timedelta(hours=8))).isoformat(),
        run_id=policy['run_id'],status='bounded_diagnostic_registered_not_started',
        policy_sha256=sha(ROOT/'mechanism012_policy.json'),protocol_sha256=sha(ROOT/'docs/MECHANISM012_PROTOCOL.md'),
        user_notice_revokes_prior_temporary_server_pause=True,restart_preflight_sha256=sha(RECEIPTS/'restart_preflight_01/RESTART_PREFLIGHT.json'),
        server_boot_id=native['boot_id'],prior011_complete_verified=True,prior011_readback_sha256=sha(independent_path),
        old53_sources_verified=old_source_verified,planned_budgets=policy['budgets'],
        source_freeze_pending=True,active_scientific_jobs=0,actual_NN_rows=0,actual_AD_queries=0,
        actual_checkpoint_loads=0,actual_optimizer_updates=0,DEV_numeric_read=False,
        followup_automation_id='n01',followup_automation_status='PAUSED_until_correct_native_start',
        scientific_status='response_match_not_established',new_training_updates_global=159667)
    write(OUT/'REGISTRATION_RECEIPT.json',receipt,new=True)
    assert read(ROOT/'PROGRAM.json')['mechanism012_gradient_evaluations']==0
    assert read(ROOT/'jobs.json')['jobs'][-1]['run_id']==policy['run_id']
    print(json.dumps(dict(status=receipt['status'],registered_UTC=utc,active_jobs=0,
        NN_rows=0,AD_queries=0,source_freeze_pending=True,policy_sha256=receipt['policy_sha256']),ensure_ascii=False))

if __name__=='__main__':
    main()
