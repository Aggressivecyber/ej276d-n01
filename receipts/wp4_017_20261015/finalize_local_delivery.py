"""Seal the reviewed local017 package. This helper has no server operations."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tomllib

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
AUTOMATION=Path('C:/Users/10629/.codex/automations/n01/automation.toml')
BASE_G=179667
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),**rec(p))
def write(p,d,*,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(d,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    boundary_path=OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json';boundary=read(boundary_path)
    assert boundary['remote_execution_authorized'] is False
    p=read(ROOT/'wp4_017_config.json');source=read(OUT/'SOURCE_COMMIT_READBACK.json')
    assert p['status']=='frozen' and p['native_launch_authorized'] is False and p['execution_scope']=='local_package_only'
    assert source['configuration_sha256']==sha(ROOT/'wp4_017_config.json')
    assert source['source_manifest_sha256']==sha(ROOT/'WP4_017_SOURCE_MANIFEST.json')
    for name,row in source['files'].items():assert rec(ROOT/name)==row,name
    protected=read(OUT/'engineering_01/PROTECTED_BEFORE.json')['files'];assert len(protected)==124
    for name,row in protected.items():assert rec(ROOT/name)==row,name
    for name,row in p['inputs'].items():
        path=Path(row['local_path']);assert path.drive.upper() in ('E:','D:')
        assert rec(path)=={k:row[k] for k in ('bytes','sha256')},name
    review_path=OUT/'independent_01/SOURCE_REVIEW.json';review=read(review_path)
    assert review['passed'] and not review['blockers']
    ready_path=OUT/'engineering_01/ENGINEERING_READY.json';ready=read(ready_path)
    assert ready['passed'] is True
    fr_path=OUT/'runtime_package_01/FREEZE_RECEIPT.json';fr=read(fr_path)
    package=Path(fr['source_package']);assert sha(package)==fr['source_package_sha256']
    controls=source['files'];names=set()
    with tarfile.open(package,'r:gz') as archive:
        for member in archive:
            name=member.name;path=PurePosixPath(name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts and name not in names
            assert name in controls and not name.lower().endswith(('.pt','.pth','.npz','.npy','.bin'))
            stream=archive.extractfile(member);assert stream is not None
            data=stream.read();row=controls[name]
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
            names.add(name)
    assert names==set(controls)
    settings=tomllib.loads(AUTOMATION.read_text(encoding='utf-8'));pause=read(OUT/'AUTOMATION_PAUSE_RECEIPT.json')
    assert settings['id']=='n01' and settings['status']=='PAUSED' and settings['kind']=='heartbeat'
    assert settings['target_thread_id']==pause['target_thread_id'] and pause['saved_local_only_prompt_verified']
    assert '不得SSH' in settings['prompt']
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    delivery_path=OUT/'LOCAL_DELIVERY.md'
    text=f'''# N01 017 本地交付

017的实现、人工工程检查和独立源码审核已完成，源码与配置已冻结。按用户最新要求，本次仅本地交付；未推送、部署或启动服务器任务。

实现固定已知A/005/M0下的两个电荷方差参数，包含paired/unpaired条件分布评分、SOURCE父隔离、独立随机流、硬观察、有限支持与阶段门控。首阶段两次T1配对和条件余下六次的预算保存在配置中，均未正式执行。

- [工程回执](engineering_01/ENGINEERING_READY.json)：人工两试验端到端、完整1024行三假设一次反传及两参数实际更新、原类人工权重缓存对照、保存重开和控制器有效/异常状态检查通过。
- [独立源码审核](independent_01/SOURCE_REVIEW.json)：本次同一审核已完成，blockers为空；[合同](planning_01/ENGINEERING_PROPOSAL.json)及[实现口径](planning_01/ROOT_IMPLEMENTATION_CLARIFICATIONS.json)保留。
- [冻结包](runtime_package_01/wp4_017_source.tar.gz)：包含源码、合同和配置，不含原始波形或checkpoint。源码提交 `{source['source_commit']}`，全部{source['source_controls']}个原始Git控制blob已逐项核验。
- [固定配置](../../wp4_017_config.json) · [数据角色](../../WP4_017_DATA_ROLES.json) · [执行说明](../../docs/WP4_017_CONFIGURATION.md)。
- [自动续跑暂停回执](AUTOMATION_PAUSE_RECEIPT.json)：同一个n01已暂停，未新增训练苏醒。新的服务器工作等待用户后续安排。

017正式科学训练G/AD、原checkpoint加载和SOURCE/REAL/DEV数值解码均为0；人工数据工程操作另记在对应回执。项目累计科学训练仍为{BASE_G}，124份旧源码和审核控制文件及7份本地输入原始SHA保持。教师恢复、REAL泛化和硬件验证尚未运行，科学状态为`response_match_not_established`，六工作包尚未全部完成。

此前[016报告](../../reports/wp3_016_20261015/REPORT.md)及其受限Q1研究参考保留；本地工程通过不构成新的科学晋级。
'''
    with delivery_path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(x) for x in (paths[0],paths[1],paths[2],CURRENT)]
    assert program['new_training_updates']==BASE_G and all(v['active_scientific_jobs']==0 for v in (master,program,jobs,current))
    assert program['wp4_017_G_updates']==program['wp4_017_AD_queries']==0
    snapshot=OUT/'metadata_before_local_closeout';snapshot.mkdir(exist_ok=False)
    for path in paths:shutil.copy2(path,snapshot/path.name);assert rec(path)==rec(snapshot/path.name)
    delivery=dict(document=ref(delivery_path),source_commit=source['source_commit'],source_review=ref(review_path),
        engineering_ready=ref(ready_path),freeze_receipt=ref(fr_path),source_package=ref(package),
        automation_pause=ref(OUT/'AUTOMATION_PAUSE_RECEIPT.json'),user_execution_boundary=ref(boundary_path),
        local_package_complete=True,remote_execution_authorized=False,formal_teacher_recovery_executed=False,
        science_G_AD_added=0,scientific_status='response_match_not_established')
    for value in (master,program,current):
        value.update(status='WP4_017_local_package_delivered_remote_deferred_by_user',
            current_stage='WP4_017_local_delivery_closed',current_work_package='WP4',active_work_package='WP4',
            active_scientific_jobs=0,active_run_id=None,pending_run_id=None,current_run_training_updates=0,
            wp4_017_local_delivery=delivery,wp4_017_native_run_executed=False,
            current_followup_status='n01_PAUSED_local_only_no_pending_long_job',updated_UTC=stamp,
            next_action='Local017 package is delivered. Await the user next arrangement; no automatic server execution.',
            scientific_status='response_match_not_established')
        value['wp4_017_execution'].update(local_delivery=delivery,status='local_package_delivered_native_not_run')
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP4_017_20261015_LOCAL_ENGINEERING')
    assert job['scientific_job'] is False and job['G_updates']==job['new_AD_queries']==0
    job.update(status='local_package_delivered_native_not_run',local_delivery=delivery,completed_UTC=stamp)
    job['execution']=program['wp4_017_execution'];jobs['updated_UTC']=stamp
    wp4=next(w for w in master['work_packages'] if w['id']=='WP4')
    wp4.update(status='017_local_implementation_delivered_known_teacher_recovery_not_run',execution=program['wp4_017_execution'])
    old=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in old
    prefix='''# N01 当前执行：WP4/017 本地交付完成，服务器执行暂缓

017实现、人工工程检查、独立源码审核和源码配置冻结已完成。按用户最新要求，仅本地交付；未推送或启动017服务器任务。017正式科学G/AD为0，累计179667，n01自动续跑已暂停。教师恢复尚未运行，科学状态仍为response_match_not_established，六工作包尚未全部完成。

[017本地交付](receipts/wp4_017_20261015/LOCAL_DELIVERY.md) · [017最终回执](receipts/wp4_017_20261015/FINAL_LOCAL_DELIVERY_RECEIPT.json) · [016报告](reports/wp3_016_20261015/REPORT.md)

'''
    paths[3].write_text(prefix+old[old.index(marker):],encoding='utf-8')
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
    files=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]
    files += [f for f in OUT.iterdir() if f.is_file() and f.suffix in ('.py','.json','.md','.txt')]
    files=list(dict.fromkeys(files));assert all(f.stat().st_size<8*1024**2 for f in files)
    assert not git(['diff','--cached','--name-only']).strip()
    listed=[f.relative_to(ROOT).as_posix() for f in files]
    git(['add','-f','--',*listed]);git(['diff','--cached','--check'])
    staged=git(['diff','--cached','--name-only']).decode().splitlines();assert set(staged)<=set(listed) and staged
    git(['commit','-m','Deliver locally verified WP4 package and defer server execution'])
    commit=git(['rev-parse','HEAD']).decode().strip()
    for name,row in controls.items():
        blob=git(['cat-file','blob',commit+':'+name]);assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'],name
    for path in files:
        name=path.relative_to(ROOT).as_posix();blob=git(['cat-file','blob',commit+':'+name])
        assert blob==path.read_bytes() or (name=='README.md' and blob==path.read_bytes().replace(b'\r\n',b'\n')),name
    copy_files=list(dict.fromkeys(files+[fr_path,package,ready_path,review_path,ROOT/'wp4_017_config.json',ROOT/'WP4_017_SOURCE_MANIFEST.json']))
    total=sum(f.stat().st_size for f in copy_files)+CURRENT.stat().st_size
    assert total<16*1024**2 and shutil.disk_usage('D:/').free>total+64*1024**2
    dest=Path('D:/GAN_backups/N01/wp4_017_20261015/local_delivery_01');assert not dest.exists();dest.mkdir(parents=True,exist_ok=False)
    inventory={}
    for path in copy_files+[CURRENT]:
        name=path.relative_to(ROOT).as_posix() if path!=CURRENT else 'workspace_entry/CURRENT_SIM2REAL_PROGRAM.json'
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);assert rec(target)==rec(path)
        inventory[name]=rec(target)
    manifest_path=dest/'MANIFEST.json';write(manifest_path,dict(UTC=stamp,source_commit=source['source_commit'],delivery_commit=commit,files=inventory),new=True)
    (dest/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+n+'\n' for n,r in sorted(inventory.items())),encoding='utf-8',newline='\n')
    final_path=OUT/'FINAL_LOCAL_DELIVERY_RECEIPT.json'
    write(final_path,dict(schema='n01/wp4-017-final-local-delivery/v1',UTC=stamp,status='local_delivery_complete_remote_deferred_by_user',
        delivery=delivery,source_commit=source['source_commit'],delivery_commit=commit,
        raw_Git_control_blobs_verified=len(controls),source_archive_all_members_SHA256_verified=True,
        source_archive_contains_original_waveforms_or_checkpoints=False,original124_controls_unchanged=True,
        local_original_input_raw_SHA_verified=7,actual017_original_CP_SOURCE_REAL_DEV_numeric_loads=0,
        actual017_scientific_G_AD=0,actual017_remote_actions=0,cumulative_scientific_training_updates=BASE_G,
        automation_status='PAUSED',automation_settings_sha256=sha(AUTOMATION),new_future_training_wake=False,
        source_backup=ref(OUT/'D_SOURCE_BACKUP_RECEIPT.json'),
        D_delivery_backup=dict(destination=str(dest),files=len(inventory),bytes=total,manifest_sha256=sha(manifest_path),all_SHA256_verified=True),
        known_teacher_recovery_executed=False,scientific_acceptance=False,scientific_status='response_match_not_established',
        all_six_work_packages_complete=False,public_push=False,server_push=False),new=True)
    target=dest/'FINAL_LOCAL_DELIVERY_RECEIPT.json';shutil.copy2(final_path,target);assert rec(target)==rec(final_path)
    write(dest/'FINAL_DELIVERY_MANIFEST.json',dict(UTC=stamp,manifest=rec(manifest_path),final_receipt=rec(target)),new=True)
    print(json.dumps(dict(status='017_local_delivery_complete_no_server_execution',source_commit=source['source_commit'],
        delivery_commit=commit,source_controls=len(controls),D_files=len(inventory)+1,D_bytes=total+target.stat().st_size,
        science_G_AD_added=0,automation='PAUSED',receipt=str(final_path))))

if __name__=='__main__':main()
