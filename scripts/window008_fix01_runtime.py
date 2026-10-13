"""Shared immutable bindings for the single Window008 fix01 dispatch."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]
RUN_ID='window008_fix01_20261013'
NEW_FILES=('window008_fix01_config.json','src/n01/model008_fix01.py','src/n01/response008_fix01.py',
    'src/n01/train_window008_fix01.py','src/n01/evaluate_window008.py','docs/WINDOW008_STABILITY_FIX01_PROTOCOL.md',
    'scripts/window008_fix01_runtime.py','scripts/prepare_window008_fix01.py',
    'scripts/stage_window008_fix01.py','scripts/freeze_window008_fix01.py',
    'scripts/start_window008_fix01_remote.py','scripts/launch_window008_fix01.sh',
    'scripts/check_window008_fix01_cuda.py','scripts/monitor_window008_fix01.py')


def require(condition,message):
    if not condition:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()
def bound(root,name):
    require(not Path(name).is_absolute(),'Relative path required')
    path=(root/name).resolve();require(path.is_relative_to(root.resolve()),'Path escapes root: '+str(name))
    return path
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def validate_inputs(root=ROOT):
    cfg=read(root/'window008_fix01_config.json');base=read(bound(root,cfg['base_configuration']))
    require(cfg['schema']=='n01/window008-fix01-config/v1' and cfg['run_id']==RUN_ID,'Wrong recovery identity')
    require(sha(bound(root,cfg['base_configuration']))==cfg['base_configuration_sha256'],'Base recipe changed')
    require(sha(bound(root,cfg['legacy_source_manifest']))==cfg['legacy_source_manifest_sha256'],'Legacy manifest changed')
    legacy=read(bound(root,cfg['legacy_source_manifest']))
    for name,digest in legacy['files'].items():require(sha(bound(root,name))==digest,'Original frozen file changed: '+name)
    for name,record in cfg['reference_inputs'].items():require(sha(bound(root,record['path']))==record['sha256'],'Legacy reference changed: '+name)
    require(cfg['execution_arms']==['full','simple'] and cfg['comparison_arms']==['deterministic','full','simple'],'Arm order differs')
    for name,value in dict(new_execution_G_updates=35000,new_execution_D_updates=96000,effective_comparison_G_updates=57000,
        effective_comparison_D_updates=144000,historical_actual_G_updates_after_success=57269,
        historical_actual_D_updates_after_success=144807,discarded_legacy_joint_G_updates=269,discarded_legacy_joint_D_updates=807).items():
        require(cfg[name]==value,'Recovery budget differs: '+name)
    refs=cfg['reference_inputs'];failure=read(bound(root,refs['legacy_failure']['path']))
    native=read(bound(root,refs['legacy_native_exit']['path']))
    require(native['exit_code']==1 and native.get('reason') is None and failure['G_updates']==22269 and failure['D_updates']==48807,'Not the registered failed job')
    require(failure['current_arm']=='full' and failure['DEV_waveforms_read'] is False,'Failed role differs')
    done=read(bound(root,refs['deterministic_training_complete']['path']))
    require(done['G_updates']==19000 and done['D_updates']==dict(fast=16000,slow=16000,common_level=16000)
            and done['final_sha256']==refs['deterministic_final']['sha256'],'Retained deterministic endpoint differs')
    require(sha(bound(root,base['dataset']))==base['dataset_sha256'] and sha(bound(root,base['statistics']))==base['statistics_sha256'],'Prepared TRAIN differs')
    return cfg,base,legacy


def runtime_files(root=ROOT):
    cfg,base,legacy=validate_inputs(root)
    names=set(legacy['files'])|{cfg['legacy_source_manifest']}|set(NEW_FILES)
    names|={record['path'] for record in cfg['reference_inputs'].values()}
    return sorted(names)


def validate_evidence(root=ROOT):
    cfg,base,legacy=validate_inputs(root)
    rec=root/'receipts/window008_fix01_20261013'
    qa=read(rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json')
    native=read(rec/'engineering_supervisor_01/exit_receipt.json')
    launch=read(rec/'engineering_supervisor_01/launch.json')
    review=read(rec/'PRELAUNCH_REVIEW.json')
    require(qa['status']=='passed_native_fix01_engineering_and_historical_failure_inference','Native engineering not passed')
    require(qa['device']=='cuda:0' and qa['scientific_G_updates']==qa['scientific_D_updates']==qa['DEV_numeric_decodes']==0,'Engineering scope differs')
    require(qa['engineering_G_updates']==4 and qa['engineering_D_updates']==6,'Synthetic optimizer accounting differs')
    require(qa['actual_SOURCE_inference_rows']==1024 and qa['actual_REAL_numeric_rows_read']==0,'Fixed historical inference scope differs')
    require(qa['recovery_config_sha256']==sha(root/'window008_fix01_config.json')
            and qa['base_config_sha256']==cfg['base_configuration_sha256'],'QA config differs')
    for name,digest in qa['source_sha256'].items():require(sha(bound(root,name))==digest,'Native checked source changed: '+name)
    require(native['exit_code']==0 and native.get('reason') is None and native['peak_rss_bytes']<=cfg['job_tree_RSS_limit_GiB']*1024**3,'Native engineering resource/exit failure')
    require(all(launch[key]==native[key] for key in ('supervisor_pid','child_pid','started_utc')),'Engineering native command/PID chain differs')
    command=launch['command']
    require(command.count('scripts/check_window008_fix01_cuda.py')==1 and command.count('--out')==1,'Engineering native command differs')
    require(command[command.index('--out')+1]==cfg['remote_root']+'/runs/engineering_cuda_01','Engineering native output differs')
    require(qa['parent_pid']==launch['child_pid'] and qa['pid']!=qa['parent_pid'],'Engineering native process chain differs')
    require(dt.datetime.fromisoformat(native['started_utc'])<=dt.datetime.fromisoformat(qa['started_UTC'])
            <=dt.datetime.fromisoformat(qa['finished_UTC'])<=dt.datetime.fromisoformat(native['ended_utc']),'Engineering native time chain differs')
    require(all(row['peak_GPU_bytes']<=cfg['gpu_memory_limit_GiB']*1024**3 for row in qa['steps']),'Native GPU cap exceeded')
    require(review['status']=='passed_independent_prelaunch_review','Independent prelaunch review missing')
    for name,digest in review['files_sha256'].items():require(sha(bound(root,name))==digest,'Reviewed artifact changed: '+name)
    require({'src/n01/model008_fix01.py','src/n01/response008_fix01.py','src/n01/train_window008_fix01.py'}<=set(review['files_sha256']),'Reviewed core coverage incomplete')
    return cfg,base,legacy
