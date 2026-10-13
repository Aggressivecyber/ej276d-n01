"""Record explicit user pause, scheduler state and the retained boundary metadata."""
import datetime as dt
import json
import os
from pathlib import Path
import tomllib
import torch
from window008_fix01_runtime import ROOT,read,sha,write_new,require

def replace_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(temp,path)

def main():
    rec=ROOT/'receipts/window008_fix01_20261013';stop=read(rec/'USER_STOP_VERIFICATION.json');collection=read(rec/'USER_STOP_COLLECTION.json')
    require(stop['status']=='user_requested_stop_verified' and stop['trainer_absent'] and not stop['owned_processes_after'],'Stop not verified')
    require(collection['all_transferred_member_hashes_verified'] is True,'Local recovery not verified')
    automation_path=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
    automation=tomllib.loads(automation_path.read_text(encoding='utf8'))
    require(automation['id']=='n01' and automation['status']=='PAUSED','Scheduled wakeup not paused')
    checkpoint_path=ROOT/'runs/window008_fix01_20261013/runs/science/train/full/latest.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    require(checkpoint['schema']=='response008_fix01/v1' and checkpoint['checkpoint_purpose']=='periodic'
            and checkpoint['failure_may_be_mid_step'] is False,'Last periodic checkpoint role differs')
    require(checkpoint['G_updates']==6000 and checkpoint['new_G_updates']==3000
            and checkpoint['D_updates']==dict(fast=3000,slow=3000,common_level=3000),'Boundary checkpoint counters differ')
    require(checkpoint['last_metrics']['G_updates']==checkpoint['G_updates']
            and checkpoint['last_metrics']['D_updates']==checkpoint['D_updates'],'Boundary checkpoint/metrics differ')
    require(checkpoint['recovery']['source_manifest_sha256']==sha(ROOT/'WINDOW008_FIX01_SOURCE_MANIFEST.json'),'Boundary source identity differs')
    failure=collection['failure'];g=failure['new_execution_G_updates'];d=failure['new_execution_D_updates'];stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    record=dict(UTC=stamp,status='paused_by_user_before_server_shutdown',user_requested_pause=True,
        stop_receipt_sha256=sha(rec/'USER_STOP_VERIFICATION.json'),collection_receipt_sha256=sha(rec/'USER_STOP_COLLECTION.json'),
        local_artifact_files=len(collection['files']),all_artifact_hashes_verified=True,owned_trainer_and_supervisor_absent=True,
        native_exit_code=collection['native']['exit_code'],native_exit_role='Intentional SIGTERM at explicit user shutdown request; training not complete.',
        actual_new_G_updates=g,actual_new_D_updates=d,current_arm_G_updates=failure['current_G_updates'],
        periodic_checkpoint=dict(path=checkpoint_path.relative_to(ROOT).as_posix(),sha256=sha(checkpoint_path),
            G_updates=6000,D_updates=checkpoint['D_updates'],new_G_updates=3000,metadata_role_verified=True),
        stop_checkpoint_may_be_mid_step=True,automatic_resume_allowed=False,requires_user_server_online_notice=True,
        resume_next_action='After user confirms server online, validate and implement a versioned boundary-checkpoint resume; never directly resume the potentially mid-step FAILURE_CHECKPOINT.',
        automation_id='n01',automation_status='PAUSED',automation_config_sha256=sha(automation_path),
        model_forwards_for_pause_validation=0,optimizer_updates_for_pause_validation=0,DEV_numeric_decodes=0,
        scientific_status='response_match_not_established')
    write_new(rec/'USER_PAUSE_RECEIPT.json',record)
    backup=rec/'user_stop_state_before';backup.mkdir(exist_ok=False)
    state_paths=[ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']
    for path in state_paths:(backup/path.name).write_bytes(path.read_bytes())
    jobs=read(ROOT/'jobs.json');job=next(x for x in jobs['jobs'] if x['job_id']=='N01_WINDOW008_FIX01_20261013')
    job.update(status=record['status'],pid=None,supervisor_pid=None,native_exit_code=collection['native']['exit_code'],
        observed_new_G_updates=g,observed_new_D_updates=d,last_verified_UTC=stop['verified_UTC'],
        ended_UTC=collection['native']['ended_utc'],user_requested_pause=True,
        stop_receipt='receipts/window008_fix01_20261013/USER_PAUSE_RECEIPT.json',local_artifacts_root=collection['local_collection_root'])
    replace_json(ROOT/'jobs.json',jobs)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(path);value.update(status=record['status'],active_scientific_jobs=0,current_arm=failure['current_arm'],current_stage='user_requested_pause',
            window008_fix01_training_pid=None,window008_fix01_supervisor_pid=None,window008_fix01_new_G_updates=g,window008_fix01_new_D_updates=d,
            window008_fix01_native_exit_code=collection['native']['exit_code'],window008_fix01_user_requested_pause=True,
            last_verified_UTC=stop['verified_UTC'],updated_UTC=stamp,followup_automation_status='PAUSED',followup_scheduled_local=None,
            next_action=record['resume_next_action'],scientific_status='response_match_not_established')
        if path.name=='PROGRAM.json':value.update(new_training_updates=122269+g,current_run_training_updates=g)
        replace_json(path,value)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');first,rest=text.split('\n',1)
    readme.write_text(first+'\n\n当前暂停：用户将关闭服务器，已按明确要求停止Window008 fix01；训练和监督进程均已退出，GPU已释放，n01定时苏醒已暂停。24个作业文件已回收到E盘并逐文件哈希验证；最近完整周期检查点为full 6000 G，停止现场保留到6398 G。等待用户晚上开机通知，随后核对状态和完整检查点再恢复。暂停回执：`receipts/window008_fix01_20261013/USER_PAUSE_RECEIPT.json`。\n'+rest,encoding='utf8')
    pointer=read(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json');pointer['plan_sha256']=sha(readme);replace_json(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json',pointer)
    print(json.dumps(dict(status=record['status'],actual_new_G_updates=g,last_complete_checkpoint_G=6000,artifacts_verified=24,automation_status='PAUSED')))

if __name__=='__main__':main()
