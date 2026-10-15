"""Verify the officially saved future wake, seal local017 startup, then stop."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
SETTINGS=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),**rec(p))
def write(p,d,*,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(d,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    assert read(OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json')['remote_execution_authorized'] is True,'User deferred017 remote execution; this prospective native helper is inactive.'
    plan=read(OUT/'FOLLOWUP_PLAN.json');started=read(OUT/'EXECUTION_STARTED.json')
    source=read(OUT/'SOURCE_COMMIT_READBACK.json');settings=tomllib.loads(SETTINGS.read_text(encoding='utf-8'))
    result_path=OUT/'FOLLOWUP_TOOL_RESULT.json';tool=read(result_path);assert not tool.get('isError',False)
    now=dt.datetime.now(dt.timezone.utc);stamp=now.isoformat()
    due=dt.datetime.fromisoformat(plan['next_wake_UTC']);assert due>now
    assert settings['id']=='n01' and settings['kind']=='heartbeat' and settings['status']=='ACTIVE'
    assert settings['rrule']==plan['rrule'] and 'COUNT=1' in settings['rrule']
    assert settings['target_thread_id']==plan['targetThreadId']
    assert settings['prompt'].rstrip('\r\n')==(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8')==plan['prompt']
    assert 'gpt-6.1-sol / max' in settings['prompt']
    for name,row in source['files'].items():assert rec(ROOT/name)==row,name
    protected=read(OUT/'engineering_01/PROTECTED_BEFORE.json')['files']
    assert len(protected)==124
    for name,row in protected.items():assert rec(ROOT/name)==row,name
    follow=dict(schema='n01/wp4-017-single-followup-confirmation/v1',UTC=stamp,automation_id='n01',
        kind='heartbeat',status='ACTIVE',count=1,target_thread_id=settings['target_thread_id'],
        next_wake_local=plan['next_wake_local'],next_wake_UTC=plan['next_wake_UTC'],
        actual_saved_settings_verified=True,settings_sha256=sha(SETTINGS),
        actual_prompt_sha256=hashlib.sha256(settings['prompt'].encode()).hexdigest(),
        prepared_prompt_sha256=sha(OUT/'FOLLOWUP_PROMPT.txt'),official_tool_result=ref(result_path),
        requested_model='gpt-6.1-sol',requested_reasoning_effort='max',
        source_commit=source['source_commit'],continuous_monitoring=False,
        no_remote_queries_after_correct_start_snapshot=True)
    follow_path=OUT/'FOLLOWUP_RECEIPT.json';write(follow_path,follow,new=True)
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(p) for p in (paths[0],paths[1],paths[2],CURRENT)]
    total=started['cumulative_training_updates'];G=started['accounted017_G_updates']
    assert program['new_training_updates']==total and program['wp4_017_G_updates']==G
    assert all(v['active_scientific_jobs']==1 for v in (master,program,jobs,current))
    for value in (master,program,current):
        value['wp4_017_execution']['followup']=follow
        value.update(current_followup_status='single_future_n01_wake_saved_resting',updated_UTC=stamp,
            next_action='Rest until the saved single n01 wake; recover017 then continue the six-work-package master plan.')
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP4_017_20261015')
    job['execution']['followup']=follow;job['followup']=follow;jobs['updated_UTC']=stamp
    wp4=next(w for w in master['work_packages'] if w['id']=='WP4');wp4['execution']['followup']=follow
    old=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in old
    prefix=f'''# N01 当前执行：WP4/017 已核首个真实更新，等待单次回收

017固定已知A/005/M0形状核，只训练两个电荷方差参数。启动快照 {started['snapshot_UTC']} 已核首个真实更新及同一快照的进程和资源健康；该快照完成G下界为{G}，项目累计下界为{total}。先两次T1配对，全部必需门通过才进入余下六次，最多16000G/AD。单次苏醒已保存为{plan['next_wake_local']}，随后休息。启动证据不代表未来完成或科学接受。

WP3/016训练、独立数组读回和报告已闭合，当前研究参考仍为016Q1 meanoff chargeon，weights_promoted=false、response_match_not_established。六工作包尚未全部完成。

[017启动与单次苏醒](receipts/wp4_017_20261015/CORRECT_START_AND_FOLLOWUP.json) · [017配置](wp4_017_config.json) · [017合同](docs/WP4_017_CONFIGURATION.md) · [016报告](reports/wp3_016_20261015/REPORT.md)

'''
    paths[3].write_text(prefix+old[old.index(marker):],encoding='utf-8')
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
    correct_path=OUT/'CORRECT_START_AND_FOLLOWUP.json'
    write(correct_path,dict(schema='n01/wp4-017-startup-and-future-wake/v1',UTC=stamp,
        status='first_real_update_same_snapshot_health_one_future_wake_verified_resting',
        source_commit=source['source_commit'],source_controls=source['source_controls'],
        configuration_sha256=source['configuration_sha256'],source_manifest_sha256=source['source_manifest_sha256'],
        startup_snapshot_UTC=started['snapshot_UTC'],startup=ref(OUT/'EXECUTION_STARTED.json'),followup=follow,
        accounted017_G_updates=G,cumulative_training_updates_lower_bound=total,
        actual_completed_counts=started['actual_counts_last_verified'],actual_entered_counts=started['actual_entered_counts'],
        unknown_inflight_ids=started['unknown_inflight_ids'],future_native_completion='unknown_not_polled',
        original124_controls_unchanged=True,scientific_acceptance=False,all_six_work_packages_complete=False,
        continuous_monitoring=False,new_remote_queries=0),new=True)
    files=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.md')]
    for name in ('native_execution_01','deploy_native_01','start_native_01','start_verification_01'):
        directory=OUT/name
        files += [p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.txt','.log')]
    files=list(dict.fromkeys(files));assert all(p.stat().st_size<8*1024**2 for p in files)
    assert sum(p.stat().st_size for p in files)+CURRENT.stat().st_size<16*1024**2
    assert not git(['diff','--cached','--name-only']).strip()
    names=[p.relative_to(ROOT).as_posix() for p in files]
    git(['add','-f','--',*names]);git(['diff','--cached','--check'])
    staged=git(['diff','--cached','--name-only']).decode().splitlines();assert set(staged)<=set(names)
    git(['commit','-m','Record verified WP4 first update and one future recovery wake'])
    commit=git(['rev-parse','HEAD']).decode().strip()
    for p in files:
        name=p.relative_to(ROOT).as_posix();blob=git(['cat-file','blob',commit+':'+name])
        assert blob==p.read_bytes() or (name=='README.md' and blob==p.read_bytes().replace(b'\r\n',b'\n')),name
    for name,row in source['files'].items():
        blob=git(['cat-file','blob',commit+':'+name]);assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'],name
    dest=Path('D:/GAN_backups/N01/wp4_017_20261015/startup_01')
    total_bytes=sum(p.stat().st_size for p in files)+CURRENT.stat().st_size
    assert shutil.disk_usage('D:/').free>total_bytes+64*1024**2 and not dest.exists();dest.mkdir(parents=True,exist_ok=False)
    inventory={}
    for p in files+[CURRENT]:
        name=p.relative_to(ROOT).as_posix() if p!=CURRENT else 'workspace_entry/CURRENT_SIM2REAL_PROGRAM.json'
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert rec(target)==rec(p)
        inventory[name]=rec(target)
    manifest_path=dest/'MANIFEST.json';write(manifest_path,dict(UTC=stamp,source_commit=source['source_commit'],startup_commit=commit,files=inventory),new=True)
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
    final_path=OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json'
    write(final_path,dict(schema='n01/wp4-017-startup-delivery/v1',UTC=stamp,source_commit=source['source_commit'],
        startup_commit=commit,correct_start_and_followup=ref(correct_path),followup=follow,
        all_original124_controls_unchanged=True,raw_Git_control_blobs_verified=source['source_controls'],
        D_startup_backup=dict(destination=str(dest),files=len(inventory),bytes=total_bytes,
            manifest_sha256=sha(manifest_path),all_files_SHA256_verified=True),
        accounted017_G_updates=G,cumulative_training_updates_lower_bound=total,
        scientific_acceptance=False,all_six_work_packages_complete=False,continuous_monitoring=False,
        future_native_completion='unknown_not_polled',no_remote_queries_after_snapshot=True,public_push=False),new=True)
    receipt_target=dest/'FINAL_STARTUP_DELIVERY_RECEIPT.json';shutil.copy2(final_path,receipt_target);assert rec(receipt_target)==rec(final_path)
    write(dest/'FINAL_DELIVERY_MANIFEST.json',dict(UTC=stamp,source_manifest=rec(manifest_path),final_receipt=rec(receipt_target)),new=True)
    print(json.dumps(dict(status='017_correct_start_future_wake_backup_and_commit_verified_resting',
        next_wake_local=plan['next_wake_local'],accounted017_G=G,cumulative_lower_bound=total,
        source_commit=source['source_commit'],startup_commit=commit,D_files=len(inventory)+1)))

if __name__=='__main__':main()
