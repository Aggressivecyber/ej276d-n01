"""Bind reviewed017 controls and small D backup; never launch numeric work."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
CFG='wp4_017_config.json'
MF='WP4_017_SOURCE_MANIFEST.json'
BASE_G=179667
JOB='N01_WP4_017_20261015'
LOCAL_JOB=JOB+'_LOCAL_ENGINEERING'

def read(path): return json.loads(path.read_bytes())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path): return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
def write(path,value,*,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')
def checked(path,row):
    assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',required=True,choices=('local-ready','ready','commit-readback','backup-source','bind-committed-source'))
    ap.add_argument('--review',default='independent_01/SOURCE_REVIEW.json')
    ap.add_argument('--freeze-dir',default='runtime_package_01')
    args=ap.parse_args()
    boundary=read(OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json')
    if args.stage=='ready':assert boundary['remote_execution_authorized'] is True,'User deferred remote execution; use local-ready.'
    local_only=args.stage=='local-ready'
    freeze=OUT/args.freeze_dir
    p,mf,fr=[read(x) for x in (ROOT/CFG,ROOT/MF,freeze/'FREEZE_RECEIPT.json')]
    assert p['status']=='frozen' and p['run_id']=='wp4_017_20261015'
    assert fr['config_sha256']==sha(ROOT/CFG) and fr['source_manifest_sha256']==sha(ROOT/MF)
    for name,row in mf['files'].items(): checked(ROOT/name,row)
    protected=read(OUT/'engineering_01/PROTECTED_BEFORE.json')['files']
    assert len(protected)==124
    for name,row in protected.items(): checked(ROOT/name,row)
    review_path=OUT/args.review
    review=read(review_path)
    assert review['passed'] and not review['blockers']
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    commit_receipt=OUT/'SOURCE_COMMIT_READBACK.json'
    ready_receipt=OUT/'EXECUTION_READY.json'
    backup_receipt=OUT/'D_SOURCE_BACKUP_RECEIPT.json'
    controls={**mf['files'],CFG:dict(sha256=sha(ROOT/CFG),bytes=(ROOT/CFG).stat().st_size),
        MF:dict(sha256=sha(ROOT/MF),bytes=(ROOT/MF).stat().st_size)}
    if args.stage=='commit-readback':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
        for name,row in controls.items():
            blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
            assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'],name
        write(commit_receipt,dict(schema='n01/wp4-017-source-commit-readback/v1',UTC=stamp,
            source_commit=commit,source_controls=len(controls),files=controls,
            configuration_sha256=sha(ROOT/CFG),source_manifest_sha256=sha(ROOT/MF),
            all_raw_Git_blobs_match_frozen_bytes=True,original124_controls_unchanged=True,
            scientific_operations=0,public_push_performed=False),new=True)
        print(json.dumps(dict(status='all017_source_control_blobs_verified',source_commit=commit,source_controls=len(controls))))
        return
    if args.stage=='backup-source':
        source=read(commit_receipt)
        dest=Path('D:/GAN_backups/N01/wp4_017_20261015/source_01')
        paths=[ROOT/n for n in controls]+[freeze/'FREEZE_RECEIPT.json',commit_receipt,ready_receipt,review_path,
            OUT/'CONTROL_PREPARATION_RECEIPT.json',OUT/'engineering_01/ENGINEERING_READY.json',
            OUT/'planning_01/ROOT_CONTRACT_DECISION.json',OUT/'planning_01/INDEPENDENT_CONTRACT_REVIEW.json']
        paths+=list((OUT/'engineering_01').rglob('ENGINEERING_RECEIPT.json'))
        paths+=list((OUT/'engineering_01').rglob('*.py'))
        paths += [f for f in (OUT/'engineering_01/source_review_fix01').rglob('*')
            if f.is_file() and f.suffix in ('.json','.md','.txt')
            and 'artifacts' not in f.relative_to(OUT/'engineering_01/source_review_fix01').parts]
        paths+=[OUT/'independent_01/PREPARED_SEMANTIC_CHECKS.json',OUT/'independent_01/prepare_semantic_checks.py']
        paths=list(dict.fromkeys(paths)); total=sum(x.stat().st_size for x in paths)
        assert total<32*1024**2 and shutil.disk_usage('D:/').free>total+64*1024**2
        assert not dest.exists();dest.mkdir(parents=True,exist_ok=False)
        inventory={}
        for path in paths:
            assert path.suffix.lower() not in ('.pt','.npz','.tar','.gz','.npy')
            name=path.relative_to(ROOT).as_posix();target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);assert sha(path)==sha(target)
            inventory[name]=dict(bytes=target.stat().st_size,sha256=sha(target))
        write(dest/'MANIFEST.json',dict(UTC=stamp,source_commit=source['source_commit'],files=inventory),new=True)
        (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8')
        write(backup_receipt,dict(schema='n01/wp4-017-small-source-backup/v1',UTC=stamp,
            destination=str(dest),files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
            manifest_sha256=sha(dest/'MANIFEST.json'),all_copied_SHA256_verified=True,
            original124_controls_unchanged=True,no_private_waveforms_or_checkpoints=True),new=True)
        print(json.dumps(dict(status='D017_source_backup_verified',files=len(inventory))))
        return
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(x) for x in (paths[0],paths[1],paths[2],CURRENT)]
    assert all(v['active_scientific_jobs']==0 for v in (master,program,jobs,current))
    assert program['new_training_updates']==BASE_G and program['wp3_016_completed']
    if args.stage=='bind-committed-source':
        source=read(commit_receipt);backup=read(backup_receipt)
        assert source['configuration_sha256']==sha(ROOT/CFG) and source['source_manifest_sha256']==sha(ROOT/MF)
        assert backup['all_copied_SHA256_verified']
        for value in (master,program,current):
            execution=value['wp4_017_execution'];assert execution['status'] in ('frozen_ready_not_started','local_frozen_package_not_run')
            execution.update(source_commit=source['source_commit'],source_commit_pending_local_readback=False,
                source_commit_readback=ref(commit_receipt),D_source_backup=ref(backup_receipt))
            value['updated_UTC']=stamp
        job=next(x for x in jobs['jobs'] if x['job_id'] in (JOB,LOCAL_JOB))
        job['execution']=copy.deepcopy(program['wp4_017_execution']);job['source_commit']=source['source_commit']
        wp4=next(w for w in master['work_packages'] if w['id']=='WP4');wp4['execution']=copy.deepcopy(program['wp4_017_execution'])
        write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0])
        for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
        write(OUT/'SOURCE_BINDING_RECEIPT.json',dict(schema='n01/wp4-017-source-binding/v1',UTC=stamp,
            source_commit=source['source_commit'],source_commit_readback=ref(commit_receipt),D_source_backup=ref(backup_receipt),
            all_original124_controls_unchanged=True,configuration_sha256=sha(ROOT/CFG),
            source_manifest_sha256=sha(ROOT/MF),scientific_operations=0,actual017_G_AD=0),new=True)
        print(json.dumps(dict(status='committed017_source_bound_no_launch',source_commit=source['source_commit'])))
        return
    assert not any(x['job_id'] in (JOB,LOCAL_JOB) for x in jobs['jobs'])
    assert local_only and boundary['remote_execution_authorized'] is False
    snap=OUT/'metadata_before_ready';snap.mkdir(exist_ok=False)
    for path in paths:shutil.copy2(path,snap/path.name);assert sha(path)==sha(snap/path.name)
    execution=dict(configuration=CFG,configuration_sha256=sha(ROOT/CFG),source_manifest=MF,
        source_manifest_sha256=sha(ROOT/MF),source_files=len(mf['files']),source_controls=len(controls),
        original124_controls_preserved=True,freeze_receipt=ref(freeze/'FREEZE_RECEIPT.json'),
        independent_source_review=ref(review_path),contract_decision=ref(OUT/'planning_01/ROOT_CONTRACT_DECISION.json'),
        remote_root=p['remote_root'],protocol='docs/WP4_017_CONFIGURATION.md',roles='WP4_017_DATA_ROLES.json',
        budget_caps=p['budget_caps'],stage1_budget_caps=p['stage1_budget_caps'],
        resource_limits=p['resource_limits'],run_resource_limits=p['run_resource_limits'],
        status='local_frozen_package_not_run',new_training_updates=0,new_AD_queries=0,
        cumulative_training_before_run=BASE_G,source_commit_pending_local_readback=True,
        first_stage_gate_trials=p['trial_order'][:2],remaining_trials_only_if_all_stage1_required_pass=p['trial_order'][2:],
        whole_job_failure_checkpoint_cap=1,automatic_restart=False,remote_execution_authorized=False,
        execution_scope='local_package_only',actual_scientific_counters={k:0 for k in p['budget_caps']},
        user_execution_boundary=ref(OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json'))
    for value in (master,program,current):
        value.update(status='WP4_017_local_package_complete_remote_deferred_by_user',current_stage='WP4_017_local_delivery_no_native',
            current_work_package='WP4',active_work_package='WP4',active_scientific_jobs=0,
            pending_run_id=None,wp4_017_execution=copy.deepcopy(execution),wp4_017_G_updates=0,wp4_017_AD_queries=0,
            wp3_completed=True,wp3_independent_readback_complete=True,current_run_training_updates=0,
            latest_subagent_model_preference='GPT-6.1-Sol Max',updated_UTC=stamp,
            scientific_status='response_match_not_established',
            next_action='Finish local source binding and delivery, pause n01; remote work remains deferred by the latest user instruction.')
    job=dict(job_id=LOCAL_JOB,run_id=p['run_id'],scientific_job=False,status='local_engineering_complete_native_not_run',registered_UTC=stamp,
        user_authority='USER_MASTER_PLAN_20261015.txt',source_freeze_pending=False,exact_budget_pending=False,
        prospective_unexecuted_budgets=p['budget_caps'],prospective_unexecuted_stage1_budgets=p['stage1_budget_caps'],candidate_trials=p['trial_order'],
        first_stage_gate_trials=p['trial_order'][:2],prospective_teacher_trials=8,completed_teacher_trials=0,
        counts_as_WP3_short_candidate=False,G_updates=0,new_AD_queries=0,automatic_restart=False,
        REAL_numeric_read=False,DEV_numeric_read=False,execution=copy.deepcopy(execution),resource_limits=p['resource_limits'],
        scientific_status='response_match_not_established',remote_execution_authorized=False,
        user_execution_boundary=ref(OUT/'REMOTE_EXECUTION_DEFERRED_BY_USER.json'))
    jobs['jobs'].append(job);jobs['updated_UTC']=stamp
    wp4=next(w for w in master['work_packages'] if w['id']=='WP4')
    wp4.update(status='017_local_engineering_and_source_review_complete_native_not_run',execution=copy.deepcopy(execution))
    master['budgets']['WP4_017']=dict(budget_caps=p['budget_caps'],stage1_budget_caps=p['stage1_budget_caps'],
        resource_limits=p['resource_limits'],run_limits=p['run_resource_limits'],original_cumulative_training=BASE_G,
        teacher_trials_max=8,short_candidate_trials_added=0,actual_scientific_G_AD=0,remote_execution_authorized=False)
    text=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in text
    prefix='''# N01 当前执行：WP4/017 本地实现与源码包完成，服务器执行暂缓

017已完成本地实现、人工工程检查与源码独审，已知A/005/M0固定、只开放两个电荷方差参数。按用户最新要求不推送或启动服务器任务；教师恢复没有正式运行，017科学G/AD为0，项目累计179667。源码包保存了将来前两次T1配对和条件余下六次的固定合同，所有预算仅为未执行上限。科学状态仍为response_match_not_established，六工作包尚未全部完成。全部角色沿用GPT-6.1-Sol Max策略。

[017配置](wp4_017_config.json) · [017合同](docs/WP4_017_CONFIGURATION.md) · [016报告](reports/wp3_016_20261015/REPORT.md)

'''
    paths[3].write_text(prefix+text[text.index(marker):],encoding='utf-8')
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
    write(ready_receipt,dict(schema='n01/wp4-017-local-package-ready/v1',UTC=stamp,execution=execution,
        cumulative_training_updates=BASE_G,actual017_G_AD=0,source_commit_binding_pending=True,
        remote_execution_authorized=False,formal_teacher_recovery_executed=False),new=True)
    print(json.dumps(dict(status='017_local_frozen_package_not_run',source_controls=len(controls),cumulative=BASE_G)))

if __name__=='__main__':main()
