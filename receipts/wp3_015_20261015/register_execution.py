"""Root registration and exact-byte receipts for the single bounded015 job."""
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
CFG,MF='wp3_015_config.json','WP3_SOURCE_MANIFEST.json'
CURRENT=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:dt.datetime.now(dt.timezone.utc).isoformat()

def write(path,value,exclusive=False):
    with path.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def resolve(value):
    path=Path(value);return path.resolve() if path.is_absolute() else (ROOT/path).resolve()

def reference(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)

ap=argparse.ArgumentParser()
ap.add_argument('--stage',required=True,choices=['adopt','ready','commit-readback','backup-source','started','backup-startup'])
ap.add_argument('--freeze',default='receipts/wp3_015_20261015/runtime_package_01')
ap.add_argument('--review',default='receipts/wp3_015_20261015/independent_01/PRELAUNCH_REVIEW.json')
ap.add_argument('--start-dir');ap.add_argument('--verification-dir')
args=ap.parse_args()
freeze=resolve(args.freeze);fr=read(freeze/'FREEZE_RECEIPT.json')
assert fr['run_id']=='wp3_015_20261015' and fr['source_file_count']==105 and fr['protected_source_records']==368
assert sha(freeze/CFG)==fr['config_sha256'] and sha(freeze/MF)==fr['source_manifest_sha256']
assert fr['scientific_started'] is False

if args.stage=='adopt':
    assert not (ROOT/MF).exists()
    version=OUT/'control_versions/01_draft_before_frozen_adoption';version.mkdir(parents=True)
    shutil.copy2(ROOT/CFG,version/CFG)
    for name in (CFG,MF):shutil.copy2(freeze/name,ROOT/name)
    write(OUT/'FROZEN_CONTROL_ADOPTION.json',dict(UTC=now(),status='exact_frozen_bytes_adopted',
        config_sha256=sha(ROOT/CFG),source_manifest_sha256=sha(ROOT/MF),freeze=reference(freeze/'FREEZE_RECEIPT.json')),True)
    print(json.dumps(dict(status='frozen_controls_adopted',config_sha256=sha(ROOT/CFG))))
    raise SystemExit(0)

assert sha(ROOT/CFG)==fr['config_sha256'] and sha(ROOT/MF)==fr['source_manifest_sha256']
p,manifest=read(ROOT/CFG),read(ROOT/MF)
assert len(p['inputs'])==23 and len(p['contracts'])==11
for name,row in manifest['files'].items():
    assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes'],name
if args.stage=='commit-readback':
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();items={}
    for name in list(manifest['files'])+[CFG,MF]:
        blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
        assert blob==(ROOT/name).read_bytes(),'Frozen Git blob differs: '+name
        items[name]=dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
    old=read(ROOT/'WP2_SOURCE_MANIFEST.json')
    for name,row in old['files'].items():assert sha(ROOT/name)==row['sha256']
    assert sha(ROOT/'wp2_014_config.json')==old['config_sha256']
    write(OUT/'SOURCE_COMMIT_READBACK.json',dict(schema='n01/wp3-source-commit-readback/v1',UTC=now(),
        source_commit=commit,source_controls=len(items),all_raw_Git_blobs_match_frozen_bytes=True,
        config_sha256=fr['config_sha256'],source_manifest_sha256=fr['source_manifest_sha256'],files=items,
        original014_source95_plus_controls97_unchanged=True,public_push_performed=False),True)
    print(json.dumps(dict(status='all107_source_control_blobs_verified',commit=commit)))
    raise SystemExit(0)

if args.stage.startswith('backup-'):
    source=read(OUT/'SOURCE_COMMIT_READBACK.json');startup=args.stage=='backup-startup'
    destination=Path('D:/GAN_backups/N01/wp3_015_20261015')/('startup_01' if startup else 'source_01')
    assert not destination.exists();destination.mkdir(parents=True)
    if startup:
        assert args.start_dir and args.verification_dir
        files=[OUT/'EXECUTION_STARTED.json',OUT/'FOLLOWUP_RECEIPT.json',OUT/'SOURCE_COMMIT_READBACK.json',
            ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md',
            resolve(args.start_dir)/'START_RECEIPT.json',resolve(args.verification_dir)/'START_VALIDATION.json',
            resolve(args.verification_dir)/'REMOTE_STATE.json']
    else:
        files=[ROOT/name for name in manifest['files']]+[ROOT/CFG,ROOT/MF,freeze/'FREEZE_RECEIPT.json',
            OUT/'SOURCE_COMMIT_READBACK.json',OUT/'EXECUTION_READY.json',OUT/'FROZEN_CONTROL_ADOPTION.json',
            resolve(args.review),OUT/'independent_01/FROZEN_PACKAGE_REVIEW.json']
    assert sum(f.stat().st_size for f in files)<24*1024**2
    inventory={}
    for path in dict.fromkeys(files):
        name=path.relative_to(ROOT).as_posix();target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);assert sha(path)==sha(target)
        inventory[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
    write(destination/'MANIFEST.json',dict(UTC=now(),source_commit=source['source_commit'],files=inventory),True)
    (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8')
    write(OUT/('D_STARTUP_BACKUP_RECEIPT.json' if startup else 'D_SOURCE_BACKUP_RECEIPT.json'),
        dict(UTC=now(),status='all_copied_files_SHA256_verified',destination=str(destination),files=len(inventory),
            bytes=sum(v['bytes'] for v in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json'),
            no_raw_arrays_or_private_checkpoints=True),True)
    print(json.dumps(dict(status='small_D_backup_SHA256_verified',files=len(inventory))))
    raise SystemExit(0)

program,jobs,master,current=[read(path) for path in (ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'MASTER_EXECUTION_PLAN.json',CURRENT)]
paths=[ROOT/n for n in ('PROGRAM.json','jobs.json','MASTER_EXECUTION_PLAN.json','README.md')]+[CURRENT]
snapshot=OUT/('metadata_before_ready' if args.stage=='ready' else 'metadata_before_started')
snapshot.mkdir(exist_ok=False)
for path in paths:shutil.copy2(path,snapshot/path.name)
stamp=now()
if args.stage=='ready':
    review=resolve(args.review);rv=read(review)
    assert rv.get('passed') is True or rv.get('status')=='PASS'
    assert jobs['active_scientific_jobs']==0 and program['new_training_updates']==167667
    assert not any(j['job_id']=='N01_WP3_015_20261015' for j in jobs['jobs'])
    refs=dict(configuration=CFG,configuration_sha256=fr['config_sha256'],source_manifest=MF,
        source_manifest_sha256=fr['source_manifest_sha256'],source_files=105,source_controls=107,
        protected_source_records=368,original_protection271_preserved=True,freeze_receipt=reference(freeze/'FREEZE_RECEIPT.json'),
        independent_review=reference(review),remote_root=p['remote_root'],protocol='docs/WP3_015_CONFIGURATION.md',
        roles='WP3_DATA_ROLES.json',budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],
        run_resource_limits=p['run_resource_limits'],status='frozen_ready_not_started',new_training_updates=0,new_AD_queries=0,
        cumulative_training_before_run=167667,source_commit_pending_local_readback=True)
    program.update(status='WP3_015_frozen_ready',current_stage='WP3_015_paired_mean_path_trials',current_work_package='WP3',
        active_scientific_jobs=0,current_run_training_updates=0,wp3_execution=refs,wp3_G_updates=0,wp3_AD_queries=0,
        active_research=None,pending_run_id=p['run_id'],scientific_status='response_match_not_established')
    program['next_action']='Deploy the frozen015 batch and verify firstG1 or a correctly completed negative mean gate.'
    jobs['jobs'].append(dict(job_id='N01_WP3_015_20261015',run_id=p['run_id'],scientific_job=True,
        status='frozen_ready_not_started',registered_UTC=stamp,user_authority='USER_MASTER_PLAN_20261015.txt',
        source_freeze_pending=False,exact_budget_pending=False,planned_budgets=p['budget_caps'],
        candidate_trials=p['candidate_trial_order'],planned_short_candidate_runs=4,completed_short_candidate_runs=0,
        G_updates=0,new_AD_queries=0,automatic_restart=False,DEV_numeric_read=False,execution=refs,
        resource_limits=p['resource_limits'],scientific_status='response_match_not_established'))
    master.update(status='WP3_015_frozen_ready',active_work_package='WP3',current_work_package='WP3',active_scientific_jobs=0,
        current_action='Start one frozen015 batch; first run pre-check before all CP/NN; verify firstG1 or legal completed skip.',wp3_execution=refs)
    master.update(current_stage=program['current_stage'],next_action=program['next_action'],active_research=None)
    next(w for w in master['work_packages'] if w['id']=='WP3').update(status='frozen_ready',execution=refs)
    master['budgets']['short_candidate_runs']['registered_WP3_trials']=4
    master['budgets']['WP3_015']=dict(budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],
        run_limits=p['run_resource_limits'],original_cumulative_training=167667)
    current.update(status=program['status'],active_scientific_jobs=0,current_work_package='WP3',wp3_execution=refs,
        current_stage=program['current_stage'],next_action=program['next_action'])
    write(OUT/'EXECUTION_READY.json',dict(schema='n01/wp3-execution-ready/v1',UTC=stamp,refs=refs,
        cumulative_training_updates=167667,actual_WP3_training_updates=0,actual_WP3_AD_queries=0),True)
else:
    assert args.start_dir and args.verification_dir
    start,verification=resolve(args.start_dir),resolve(args.verification_dir)
    sv=read(verification/'START_VALIDATION.json');assert sv.get('status')=='actual_first_G1_and_mean_gate_verified'
    state=read(verification/'REMOTE_STATE.json');launch=read(start/'START_RECEIPT.json')
    progress=state['durable_child_progress'];counts=progress['completed_counts']
    assert not state.get('runs/primary/artifacts/FAILED.json') and not progress.get('failed_ids')
    assert 1<=counts['G_updates']<8000 and state.get('runs/primary/supervisor/exit_receipt.json') is None
    source=read(OUT/'SOURCE_COMMIT_READBACK.json')
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_015_20261015')
    assert jobs['active_scientific_jobs'] in (0,1) and program['new_training_updates']==167667
    refs=copy.deepcopy(program['wp3_execution'])
    refs.update(status='native_running_last_verified',source_commit=source['source_commit'],source_commit_pending_local_readback=False,
        source_commit_readback=reference(OUT/'SOURCE_COMMIT_READBACK.json'),start_receipt=reference(start/'START_RECEIPT.json'),
        startup_verification=reference(verification/'START_VALIDATION.json'),startup_snapshot=reference(verification/'REMOTE_STATE.json'),
        analysis_pid=sv['child_pid'],supervisor_pid=launch['supervisor_pid'],last_verified_completed_counts=counts,
        unknown_inflight_ids=progress.get('unknown_inflight_ids',[]),actual_costs_are_durable_lower_bounds_while_running=True,
        correct_start_verified=True,new_training_updates=counts['G_updates'],new_AD_queries=counts['new_AD_queries'])
    program.update(status='WP3_015_native_running_last_verified',active_scientific_jobs=1,active_research=p['run_id'],
        latest_source_commit=source['source_commit'],
        current_stage='WP3_015_native_paired_mean_path_trials',next_action='One scheduled wake, then collect015 and continue the six-workpackage plan.',
        current_run_training_updates=counts['G_updates'],new_training_updates=167667+counts['G_updates'],
        latest_run_id=p['run_id'],latest_training_run_id=p['run_id'],wp3_execution=refs,
        wp3_G_updates=counts['G_updates'],wp3_AD_queries=counts['new_AD_queries'],pending_run_id=None,
        training_update_accounting='Historical167667 + last verified015 durableG, accounted once. Snapshot counts are lower bounds; later add only newly verified difference.',
        active_scientific_jobs_status_basis='015 realG1/mean gate and same-snapshot health/source/PID verified; no continuous monitoring')
    job.update(status='native_running_last_verified',execution=refs,actual_counters_last_verified=counts,G_updates=counts['G_updates'],
        new_AD_queries=counts['new_AD_queries'],analysis_pid=sv['child_pid'],supervisor_pid=launch['supervisor_pid'],
        correct_start_verified=True,verified_start_UTC=stamp)
    jobs['active_scientific_jobs']=1
    master.update(status=program['status'],active_scientific_jobs=1,latest_run_id=p['run_id'],latest_training_run_id=p['run_id'],
        latest_source_commit=source['source_commit'],
        current_run_training_updates=counts['G_updates'],current_action='Four fixed015 short trials running last verified; one scheduled wake then rest.',wp3_execution=refs,
        current_stage=program['current_stage'],next_action=program['next_action'],active_research=p['run_id'])
    next(w for w in master['work_packages'] if w['id']=='WP3').update(status='native_running_last_verified',execution=refs)
    current.update(status=program['status'],active_scientific_jobs=1,latest_run_id=p['run_id'],latest_training_run_id=p['run_id'],
        latest_source_commit=source['source_commit'],
        current_run_training_updates=counts['G_updates'],new_training_updates=167667+counts['G_updates'],wp3_execution=refs,
        current_stage=program['current_stage'],next_action=program['next_action'])
    write(OUT/'EXECUTION_STARTED.json',dict(schema='n01/wp3-correct-start/v1',UTC=stamp,refs=refs,
        current_training_updates=counts['G_updates'],cumulative_training_last_verified=167667+counts['G_updates'],
        unknown_inflight_ids=progress.get('unknown_inflight_ids',[]),scientific_status='response_match_not_established',
        formal_response_acceptance=False),True)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig')
    lead='''# N01 当前执行：WP3/015 已正确启动

015已通过首次真实AdamW更新、ALL6固定人口均值残差门和同一快照健康检查。四次配对短试串行、每次2000G，最多8000G/AD；M0继续C1形状头，M1冻结相同头而训练6参数均值适配器。A/005保持冻结，原硬观测和013数字T不变，005基线DC限制单列。

实际费用按唯一持久化快照记录，当前在途操作完成数未知；累计为原167667加本次已核G，只入账一次。完整结果、科学验收和跨采集物理复用仍待对应证据。正确启动后设置同一n01一次未来唤醒并休息。

[015配置](wp3_015_config.json) · [015合同](docs/WP3_015_CONFIGURATION.md) · [015启动回执](receipts/wp3_015_20261015/EXECUTION_STARTED.json) · [014中文报告](reports/wp2_014_20261015/N01_WP2_014_REPORT.md)

以下保留此前阶段说明。

'''
    readme.write_text(lead+text,encoding='utf-8',newline='\n')

for value in (program,jobs,master,current):value['updated_UTC']=stamp
write(ROOT/'MASTER_EXECUTION_PLAN.json',master)
program['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(ROOT/'PROGRAM.json',program)
write(ROOT/'jobs.json',jobs)
current['plan_sha256']=sha(ROOT/'README.md');write(CURRENT,current)
print(json.dumps(dict(status=program['status'],active_scientific_jobs=program['active_scientific_jobs'],
    current_run_training_updates=program['current_run_training_updates'])))
