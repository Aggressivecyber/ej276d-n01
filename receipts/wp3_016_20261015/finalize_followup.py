"""Verify the saved one-time app followup and seal local startup delivery."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
THREAD='01a12bad-8839-7522-89a5-f028b107d7ff'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)


def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scheduled-iso',required=True);args=parser.parse_args()
    planned=dt.datetime.fromisoformat(args.scheduled_iso)
    now=dt.datetime.now(dt.timezone.utc)
    assert planned.tzinfo and planned>now
    local=planned.astimezone(dt.timezone(dt.timedelta(hours=8)))
    prompt=(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8')
    with Path('C:/Users/10629/.codex/automations/n01/automation.toml').open('rb') as stream:
        actual=tomllib.load(stream)
    assert actual['id']=='n01' and actual['kind']=='heartbeat' and actual['status']=='ACTIVE'
    assert actual['target_thread_id']==THREAD and actual['prompt']==prompt
    rule=dict(item.split('=',1) for item in actual['rrule'].removeprefix('RRULE:').split(';'))
    assert rule['FREQ']=='DAILY' and rule['COUNT']=='1' and int(rule['BYHOUR'])==local.hour
    assert int(rule['BYMINUTE'])==local.minute and int(rule.get('BYSECOND','0'))==local.second
    tool_path=OUT/'FOLLOWUP_TOOL_RESULT.json';tool=read(tool_path)
    assert not tool.get('isError',False)
    started=read(OUT/'EXECUTION_STARTED.json')
    assert started['future_wake_pending'] and started['status']=='actual_first_charge_update_and_current_same_snapshot_health_verified'
    followup_path=OUT/'FOLLOWUP_RECEIPT.json'
    followup=dict(schema='n01/wp3-016-one-future-followup/v1',UTC=now.isoformat(),status='app_update_and_saved_TOML_verified',
        automation_id='n01',kind='heartbeat',target_thread_id=THREAD,name=actual['name'],app_status='ACTIVE',
        scheduled_local_Asia_Singapore=local.isoformat(),scheduled_UTC=planned.astimezone(dt.timezone.utc).isoformat(),
        occurrences=1,prompt=ref(OUT/'FOLLOWUP_PROMPT.txt'),native_tool_result=ref(tool_path),
        saved_app_prompt_exactly_matches=True,model_policy='gpt-6.1-sol / max',
        further_remote_polling_performed_by_root=False)
    write(followup_path,followup,new=True)
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    snap=OUT/'metadata_before_followup';snap.mkdir(exist_ok=False)
    for path in paths:shutil.copy2(path,snap/path.name)
    master,program,jobs,current=[read(path) for path in (paths[0],paths[1],paths[2],CURRENT)]
    assert program['new_training_updates']==started['cumulative_training_updates']
    followup_ref=dict(**ref(followup_path),automation_id='n01',status='ACTIVE',
        scheduled_local_Asia_Singapore=local.isoformat(),single_occurrence=True)
    for value in (master,program,jobs,current):
        value.update(followup=followup_ref,followup_automation_id='n01',followup_automation_status='ACTIVE',
            followup_next_run_local=local.isoformat(),followup_receipt=followup_ref['path'],
            followup_receipt_sha256=followup_ref['sha256'],current_followup_status='one_future_n01_wake_saved_correct_start_verified',
            latest_subagent_model_preference='GPT-6.1-Sol Max',updated_UTC=now.isoformat())
    for value in (master,program,current):
        value['next_action']='Rest without polling; at the single saved wake collect016 and continue the master work packages.'
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015');job['followup']=followup_ref
    text=paths[3].read_text(encoding='utf-8-sig')
    text=text.replace('当前等待一次未来n01回收唤醒后休息。',f'已确认{local.isoformat()}（Asia/Singapore）同一n01单次回收唤醒，当前休息，不持续监控。',1)
    paths[3].write_text(text,encoding='utf-8')
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
    summary_path=OUT/'CORRECT_START_AND_FOLLOWUP.json'
    summary=dict(schema='n01/wp3-016-correct-start-and-followup/v1',UTC=now.isoformat(),
        status='correct_native_start_and_one_future_wake_verified',start=ref(OUT/'EXECUTION_STARTED.json'),
        followup=ref(followup_path),actual_G_last_snapshot=started['actual_counts_last_verified']['G_updates'],
        cumulative_G_last_snapshot=started['cumulative_training_updates'],
        unknown_inflight_ids=started['unknown_inflight_ids'],counts_after_saved_snapshot='unknown_not_polled',
        scientific_acceptance=False,whole_project_complete=False,rest_now=True)
    write(summary_path,summary,new=True)
    destination=Path('D:/GAN_backups/N01/wp3_016_20261015/startup_01');destination.mkdir(parents=True,exist_ok=False)
    suffix={'fix01':'_FIX01','fix02':'_FIX02'}.get(read(ROOT/'wp3_016_config.json').get('launch_revision'),'')
    source_path=OUT/('SOURCE_COMMIT_READBACK'+suffix+'.json')
    start_dir=OUT/started['start_receipt_directory'];verify_dir=OUT/started['startup_verification_directory']
    files=[OUT/'EXECUTION_STARTED.json',followup_path,tool_path,OUT/'FOLLOWUP_PROMPT.txt',summary_path,source_path,
        start_dir/'START_RECEIPT.json',verify_dir/'START_VALIDATION.json',verify_dir/'REMOTE_STATE.json',
        OUT/'NATIVE_FAILURE_RECONCILIATION.json',OUT/'D_FAILURE_BACKUP_RECEIPT.json',
        OUT/('D_SOURCE_BACKUP_RECEIPT'+suffix+'.json'),
        OUT/('native_execution_03/NATIVE_START_SUMMARY.json' if suffix=='_FIX02' else 'native_execution_02/NATIVE_START_SUMMARY.json')]+paths
    files.extend(OUT/name for name in ('FOLLOWUP_PROMPT_SENT.txt','FOLLOWUP_PROMPT_CANONICALIZATION.json') if (OUT/name).is_file())
    assert sum(path.stat().st_size for path in files)<8*1024**2
    inventory={}
    for path in files:
        name=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else 'workspace_selector/'+path.name
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        assert sha(path)==sha(target);inventory[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    write(destination/'MANIFEST.json',dict(UTC=now.isoformat(),files=inventory),new=True)
    (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8')
    write(OUT/'D_STARTUP_BACKUP_RECEIPT.json',dict(UTC=now.isoformat(),status='all_copied_files_SHA256_verified',
        destination=str(destination),files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
        manifest_sha256=sha(destination/'MANIFEST.json'),scientific_raw_payload_included=False),new=True)
    print(json.dumps(dict(status=summary['status'],wake_local=local.isoformat(),D_backup_files=len(inventory),
        cumulative_G_last_snapshot=started['cumulative_training_updates'],rest_now=True)))


if __name__=='__main__':main()
