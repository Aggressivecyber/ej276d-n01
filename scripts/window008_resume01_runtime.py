"""Immutable source/reference bindings for the explicit post-shutdown continuation."""
from pathlib import Path
import copy
from window008_fix01_runtime import ROOT,read,sha,bound,require,write_new,validate_evidence as validate_fix01

RUN_ID='window008_resume01_20261013'
NEW_FILES=('window008_resume01_config.json','src/n01/response008_resume01.py','src/n01/train_window008_resume01.py',
    'docs/WINDOW008_RESUME01_EXECUTION.md','scripts/check_window008_resume01_engine.py','scripts/check_window008_resume01_driver.py',
    'scripts/prepare_window008_resume01.py','scripts/window008_resume01_runtime.py','scripts/stage_window008_resume01.py',
    'scripts/freeze_window008_resume01.py','scripts/start_window008_resume01_remote.py','scripts/launch_window008_resume01.sh',
    'scripts/check_window008_resume01_cuda.py','scripts/execute_window008_resume01_engineering.py','scripts/monitor_window008_resume01.py')
ENGINEERING_FILES=('receipts/window008_resume01_20261013/engine_cpu_01/ENGINEERING_FIXTURE_INPUTS_FORMAL_BATCH.pt',
    'receipts/window008_resume01_20261013/engine_cpu_01/fabricated_fix01_periodic6000.formal_batch.engineering.pt')

def fix01_identity(cfg):
    fix=read(ROOT/cfg['fix01_configuration'])
    return dict(run_id=fix['run_id'],config_sha256=cfg['fix01_configuration_sha256'],
        source_manifest_sha256=cfg['fix01_source_manifest_sha256'],base_configuration_sha256=cfg['base_configuration_sha256'],
        base_source_manifest_sha256=cfg['legacy_source_manifest_sha256'])

def identity(cfg,manifest_sha256):
    return dict(run_id=cfg['run_id'],config_sha256=sha(ROOT/'window008_resume01_config.json'),source_manifest_sha256=manifest_sha256,
        base_configuration_sha256=cfg['base_configuration_sha256'],base_source_manifest_sha256=cfg['legacy_source_manifest_sha256'],
        fix01_configuration_sha256=cfg['fix01_configuration_sha256'],fix01_source_manifest_sha256=cfg['fix01_source_manifest_sha256'])

def validate_inputs(root=ROOT):
    validate_fix01(root)
    cfg=read(root/'window008_resume01_config.json');base=read(bound(root,cfg['base_configuration']))
    require(cfg['schema']=='n01/window008-resume01-config/v1' and cfg['run_id']==RUN_ID,'Resume identity differs')
    for name in ('base_configuration','legacy_source_manifest','fix01_configuration','fix01_source_manifest'):
        require(sha(bound(root,cfg[name]))==cfg[name+'_sha256'],'Frozen reference differs: '+name)
    fixmanifest=read(bound(root,cfg['fix01_source_manifest']))
    for name,digest in fixmanifest['files'].items():require(sha(bound(root,name))==digest,'Fix01 frozen file differs: '+name)
    for name,item in cfg['reference_inputs'].items():require(sha(bound(root,item['path']))==item['sha256'],'Resume reference differs: '+name)
    require(cfg['reference_inputs']['fix01_full_periodic_06000']['sha256']=='921ec4c58387a5ad53fdf55ade10169930536106141e9070f5e4a8df60ab6c58','Unregistered boundary')
    expected=dict(new_execution_G_updates=32000,new_execution_D_updates=87000,inherited_effective_G_updates=25000,
        inherited_effective_D_updates=57000,effective_comparison_G_updates=57000,effective_comparison_D_updates=144000,
        historical_actual_G_updates_before_resume=25667,historical_actual_D_updates_before_resume=59001,
        historical_actual_G_updates_after_success=57667,historical_actual_D_updates_after_success=146001,
        discarded_user_stop_G_updates=398,discarded_user_stop_D_updates=1194,global_N01_G_updates_before_resume=125667)
    for name,value in expected.items():require(cfg[name]==value,'Resume count differs: '+name)
    require(cfg['execution_arms']==['full','simple'] and cfg['comparison_arms']==['deterministic','full','simple'],'Resume arms differ')
    refs=cfg['reference_inputs'];pause=read(bound(root,refs['fix01_user_pause']['path']));stop=read(bound(root,refs['fix01_user_stop_verification']['path']))
    native=read(bound(root,refs['fix01_native_exit']['path']));failure=read(bound(root,refs['fix01_failure']['path']))
    require(pause['status']=='paused_by_user_before_server_shutdown' and stop['status']=='user_requested_stop_verified'
        and stop['trainer_absent'] and not stop['owned_processes_after'],'User-stop provenance differs')
    require(native==stop['files']['supervisor/exit_receipt.json'] and native['exit_code']==1
        and failure==stop['files']['science/FAILURE.json'] and failure['new_execution_G_updates']==3398
        and failure['new_execution_D_updates']==10194,'Parent stop/native/accounting differs')
    require(pause['periodic_checkpoint']['sha256']==refs['fix01_full_periodic_06000']['sha256'],'Pause/boundary differs')
    local_stop=read(bound(root,refs['fix01_user_stop_verification_local']['path']))
    require(local_stop==stop and pause['stop_receipt_sha256']==refs['fix01_user_stop_verification_local']['sha256']
        and pause['collection_receipt_sha256']==refs['fix01_user_stop_collection']['sha256'],'Local/remote stop serialization or pause receipt chain differs')
    return cfg,base,fixmanifest

def runtime_files(root=ROOT):
    cfg,_,fix=validate_inputs(root)
    return sorted(set(fix['files'])|{cfg['fix01_source_manifest']}|set(NEW_FILES)|set(ENGINEERING_FILES)|{x['path'] for x in cfg['reference_inputs'].values()})

def validate_engineering(root=ROOT):
    cfg,base,fix=validate_inputs(root)
    rec=root/'receipts/window008_resume01_20261013'
    qa=read(rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json');native=read(rec/'engineering_supervisor_01/exit_receipt.json')
    launch=read(rec/'engineering_supervisor_01/launch.json')
    require(qa['status']=='passed_native_resume01_engineering_and_import','Native continuation QA missing')
    require(qa['scientific_G_updates']==qa['scientific_D_updates']==qa['actual_SOURCE_forward_rows']==qa['actual_REAL_forward_rows']==qa['DEV_numeric_decodes']==0,'Native engineering scope differs')
    require(qa['engineering_G_updates']==3 and qa['engineering_D_updates']==6 and qa['periodic_import']['status']=='passed','Native artificial/import QA differs')
    require(qa['device']=='cuda:0' and qa['actual_SOURCE_numeric_decodes']==qa['actual_REAL_numeric_decodes']==0
        and qa['artificial_forward_rows']==6144 and len(qa['steps'])==3,'Native engineering device/data scope differs')
    require(qa['config_sha256']==sha(root/'window008_resume01_config.json'),'Native QA recipe differs')
    for name,digest in qa['source_sha256'].items():require(sha(bound(root,name))==digest,'Native checked source changed: '+name)
    require(native['exit_code']==0 and native.get('reason') is None and native['peak_rss_bytes']<=8*1024**3,'Native engineering exit/resource differs')
    require(all(launch[k]==native[k] for k in ('supervisor_pid','child_pid','started_utc')),'Native engineering parent chain differs')
    require(qa['parent_pid']==launch['child_pid'] and 'scripts/check_window008_resume01_cuda.py' in launch['command']
        and launch['command'][launch['command'].index('--out')+1]==cfg['remote_root']+'/runs/engineering_cuda_01','Native QA command/output differs')
    import datetime as dt
    require(qa['pid']!=qa['parent_pid'] and dt.datetime.fromisoformat(native['started_utc'])<=dt.datetime.fromisoformat(qa['started_UTC'])
        <=dt.datetime.fromisoformat(qa['finished_UTC'])<=dt.datetime.fromisoformat(native['ended_utc']),'Native engineering PID/time chain differs')
    require(all(x['peak_GPU_bytes']<=10*1024**3 for x in qa['steps']),'Engineering GPU cap differs')
    return cfg,base,fix

def validate_evidence(root=ROOT):
    cfg,base,fix=validate_engineering(root)
    review=read(root/'receipts/window008_resume01_20261013/PRELAUNCH_REVIEW.json')
    require(review['status']=='passed_independent_prelaunch_review','Independent continuation review missing')
    for name,digest in review['files_sha256'].items():require(sha(bound(root,name))==digest,'Reviewed continuation artifact changed: '+name)
    require({'src/n01/response008_resume01.py','src/n01/train_window008_resume01.py'}<=set(review['files_sha256']),'Continuation core review incomplete')
    return cfg,base,fix
