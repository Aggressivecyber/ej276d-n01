"""Record one verified017 startup snapshot and prepare its single future wake."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
BASE_G=179667
THREAD='01a12bad-8839-7522-89a5-f028b107d7ff'
def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
def write(path,value,*,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def main():
    assert read(OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json')['remote_execution_authorized'] is True,'User deferred017 remote execution; this prospective native helper is inactive.'
    p=read(ROOT/'wp4_017_config.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
    assert source['configuration_sha256']==sha(ROOT/'wp4_017_config.json')
    assert source['source_manifest_sha256']==sha(ROOT/'WP4_017_SOURCE_MANIFEST.json')
    for name,row in source['files'].items():
        path=ROOT/name;assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],name
    summary_path=OUT/'native_execution_01/NATIVE_START_SUMMARY.json';summary=read(summary_path)
    state_path=OUT/'start_verification_01/REMOTE_STATE.json';state=read(state_path)
    validation_path=OUT/'start_verification_01/START_VALIDATION.json';validation=read(validation_path)
    start_path=OUT/'start_native_01/START_RECEIPT.json';start=read(start_path)
    assert summary['passed'] is True
    assert state['runs/primary/artifacts/FAILED.json'] is None
    assert state['runs/primary/artifacts/COMPLETE.json'] is None
    assert state['runs/primary/supervisor/exit_receipt.json'] is None
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('root017_startup_contract',ROOT/'scripts/wp4_017_controller.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proof=module.first_valid(state,p)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    durable=copy.deepcopy(state['durable_child_progress'])
    assert not durable['failed_ids'] and not durable['truncated_tail']
    counts=durable['completed_counts'];G=counts['G_updates'];AD=counts['new_AD_queries']
    assert 1<=G<=16000 and G<=AD<=16000
    status_G=validation['current_batch_completed_G_updates']
    assert type(status_G) is int and 1<=status_G<=G
    assert validation['same_snapshot_health']=='current_owned_process_chain_and_resources_verified'
    assert start['supervisor_pid']==validation['supervisor_pid']
    assert start['boot_id']==state['current_boot_id']
    assert state['current_resource_health']['snapshot_UTC']==state['UTC']
    for key,value in durable['entered_counts'].items():assert value<=p['budget_caps'][key],key
    stamp=dt.datetime.now(dt.timezone.utc).isoformat();total=BASE_G+G
    metadata=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(x) for x in (metadata[0],metadata[1],metadata[2],CURRENT)]
    assert all(x['active_scientific_jobs']==0 for x in (master,program,jobs,current))
    assert program['new_training_updates']==BASE_G
    snap=OUT/'metadata_before_started';snap.mkdir(exist_ok=False)
    for path in metadata:shutil.copy2(path,snap/path.name);assert sha(path)==sha(snap/path.name)
    entry=copy.deepcopy(program['wp4_017_execution'])
    assert entry['source_commit']==source['source_commit'] and not entry['source_commit_pending_local_readback']
    entry.update(status='running_first_update_and_same_snapshot_health_verified',
        correct_start_verified=True,source_commit=source['source_commit'],
        start_receipt=ref(start_path),startup_validation=ref(validation_path),startup_snapshot=ref(state_path),
        native_start_summary=ref(summary_path),supervisor_pid=start['supervisor_pid'],
        child_pid=validation['child_pid'],boot_id=state['current_boot_id'],last_verified_UTC=state['UTC'],
        first_update_validation=proof,actual_completed_counts_last_verified=counts,
        actual_entered_counts_last_verified=durable['entered_counts'],
        unknown_inflight_ids=durable['unknown_inflight_ids'],counts_are_durable_lower_bounds=True,
        native_child_exit_at_snapshot=None,native_supervisor_exit_at_snapshot=None,
        resource_snapshot=state['current_resource_health'],scientific_acceptance=False)
    for value in (master,program,current):
        value.update(status='WP4_017_running_first_update_verified',current_stage='WP4_017_gated_teacher_recovery',
            current_work_package='WP4',active_work_package='WP4',active_scientific_jobs=1,
            wp4_017_execution=copy.deepcopy(entry),wp4_017_G_updates=G,wp4_017_AD_queries=AD,
            active_run_id=p['run_id'],pending_run_id=None,last_verified_UTC=state['UTC'],
            cumulative_training_updates=total,current_run_training_updates=G,
            current_followup_status='verified_start_followup_preparation_pending_official_save',updated_UTC=stamp,
            scientific_status='response_match_not_established',
            next_action='Save exactly one future n01 heartbeat and rest; later recover017 without continuous monitoring.')
        if 'new_training_updates' in value:value['new_training_updates']=total
    program['new_training_updates']=total
    program.update(latest_run_id=p['run_id'],latest_training_source_commit=source['source_commit'],
        training_pid=validation['child_pid'],supervisor_pid=start['supervisor_pid'],
        training_update_accounting=f'179667 before017 + current durable017 G{G} = {total}; future only add the new durable difference, inflight unknown.')
    job=next(x for x in jobs['jobs'] if x['job_id']=='N01_WP4_017_20261015')
    job.update(status=entry['status'],execution=copy.deepcopy(entry),G_updates=G,new_AD_queries=AD,
        actual_counters_last_verified=counts,actual_entered_counts_last_verified=durable['entered_counts'],
        unknown_inflight_ids=durable['unknown_inflight_ids'],correct_start_verified=True,verified_start_UTC=state['UTC'])
    jobs.update(active_scientific_jobs=1,updated_UTC=stamp)
    wp4=next(w for w in master['work_packages'] if w['id']=='WP4')
    wp4.update(status='017_running_first_update_verified',execution=copy.deepcopy(entry))
    write(metadata[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(metadata[0])
    for path,value in ((metadata[1],program),(metadata[2],jobs),(CURRENT,current)):write(path,value)
    started=dict(schema='n01/wp4-017-correct-start/v1',UTC=stamp,snapshot_UTC=state['UTC'],
        execution=entry,actual_counts_last_verified=counts,actual_entered_counts=durable['entered_counts'],
        unknown_inflight_ids=durable['unknown_inflight_ids'],cumulative_training_updates=total,
        accounted017_G_updates=G,previous_cumulative=BASE_G,scientific_acceptance=False,
        continuous_monitoring=False,new_remote_queries_from_this_helper=0)
    write(OUT/'EXECUTION_STARTED.json',started,new=True)
    template=(OUT/'FOLLOWUP_PROMPT_TEMPLATE.txt').read_text(encoding='utf-8')
    values={'SOURCE_COMMIT':source['source_commit'],'CFG_SHA':source['configuration_sha256'],
        'MF_SHA':source['source_manifest_sha256'],'G_AT_SNAPSHOT':str(G),'TOTAL_AT_SNAPSHOT':str(total)}
    for key,value in values.items():template=template.replace('@@'+key+'@@',value)
    assert '@@' not in template
    prompt=template.strip()
    with (OUT/'FOLLOWUP_PROMPT.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(prompt)
    now=dt.datetime.now(dt.timezone.utc);due=(now.astimezone(dt.timezone(dt.timedelta(hours=8)))+dt.timedelta(minutes=30)).replace(second=0,microsecond=0)
    assert due>now
    plan=dict(automation_id='n01',mode='update',kind='heartbeat',destination='thread',
        targetThreadId=THREAD,name='N01 017回收与六工作包推进',status='ACTIVE',prompt=prompt,
        rrule=f'FREQ=DAILY;COUNT=1;BYHOUR={due.hour};BYMINUTE={due.minute};BYSECOND=0',
        next_wake_local=due.isoformat(),next_wake_UTC=due.astimezone(dt.timezone.utc).isoformat(),
        model_requested_in_prompt='gpt-6.1-sol',reasoning_effort_requested_in_prompt='max',
        continuous_monitoring=False)
    write(OUT/'FOLLOWUP_PLAN.json',plan,new=True)
    print(json.dumps(dict(status='017_running_snapshot_recorded_future_wake_ready_to_save',
        accounted017_G=G,cumulative_training_updates=total,next_wake_local=due.isoformat(),
        automation_plan=str(OUT/'FOLLOWUP_PLAN.json'))))

if __name__=='__main__':main()
