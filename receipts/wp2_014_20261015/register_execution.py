"""Root registration for the one authorized014 batch, using exact saved receipts."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp2_014_20261015'
CFG='wp2_014_config.json';MF='WP2_SOURCE_MANIFEST.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
now=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,obj,exclusive=False):
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def resolve(value):
    p=Path(value);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def relative(p):return p.resolve().relative_to(ROOT).as_posix()
def reference(p):return dict(path=relative(p),sha256=sha(p),bytes=p.stat().st_size)

parser=argparse.ArgumentParser()
parser.add_argument('--stage',required=True,choices=['adopt','ready','commit-readback','started','backup-source','backup-startup'])
parser.add_argument('--freeze',required=True)
parser.add_argument('--review')
parser.add_argument('--start-dir')
parser.add_argument('--verification-dir')
args=parser.parse_args()
freeze=resolve(args.freeze);fr=read(freeze/'FREEZE_RECEIPT.json')
assert fr['run_id']=='wp2_014_20261015' and fr['source_file_count']==95 and fr['protected_source_records']==271
assert sha(freeze/CFG)==fr['config_sha256'] and sha(freeze/MF)==fr['source_manifest_sha256']
assert fr['scientific_started'] is False

if args.stage=='adopt':
    assert not (ROOT/MF).exists()
    original=ROOT/CFG
    version=OUT/'control_versions/02_before_frozen_adoption';version.mkdir(parents=True)
    shutil.copy2(original,version/CFG)
    for n in [CFG,MF]:shutil.copy2(freeze/n,ROOT/n)
    assert read(ROOT/CFG)['status']=='frozen'
    write(OUT/'FROZEN_CONTROL_ADOPTION.json',dict(UTC=now(),status='exact_frozen_bytes_adopted',
        config_sha256=sha(ROOT/CFG),manifest_sha256=sha(ROOT/MF),freeze=reference(freeze/'FREEZE_RECEIPT.json')),
        True)
    print(json.dumps(dict(status='frozen_controls_adopted',config_sha256=sha(ROOT/CFG))))
    raise SystemExit(0)

assert sha(ROOT/CFG)==fr['config_sha256'] and sha(ROOT/MF)==fr['source_manifest_sha256']
p=read(ROOT/CFG);manifest=read(ROOT/MF)
for name,row in manifest['files'].items():
    assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
if args.stage=='commit-readback':
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    items={}
    for name in list(manifest['files'])+[CFG,MF]:
        blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
        assert blob==(ROOT/name).read_bytes(), 'Git source bytes differ: '+name
        items[name]=dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
    receipt=dict(schema='n01/wp2-source-commit-readback/v1',UTC=now(),source_commit=commit,
        source_controls=len(items),all_raw_Git_blobs_match_frozen_bytes=True,
        config_sha256=fr['config_sha256'],source_manifest_sha256=fr['source_manifest_sha256'],files=items,
        public_push_performed=False)
    write(OUT/'SOURCE_COMMIT_READBACK.json',receipt,True)
    print(json.dumps(dict(status='all97_source_control_blobs_verified',commit=commit)))
    raise SystemExit(0)

if args.stage.startswith('backup-'):
    source=read(OUT/'SOURCE_COMMIT_READBACK.json')
    start=args.stage=='backup-startup'
    destination=Path('D:/GAN_backups/N01/wp2_014_20261015')/('startup_01' if start else 'source_01')
    assert not destination.exists();destination.mkdir(parents=True)
    if start:
        files=[OUT/'EXECUTION_STARTED.json',OUT/'FOLLOWUP_RECEIPT.json',OUT/'SOURCE_COMMIT_READBACK.json',
            ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md']
        assert args.start_dir and args.verification_dir
        files+=[resolve(args.start_dir)/'START_RECEIPT.json',resolve(args.verification_dir)/'START_VALIDATION.json',
            resolve(args.verification_dir)/'REMOTE_STATE.json']
    else:
        files=[ROOT/n for n in manifest['files']]+[ROOT/CFG,ROOT/MF,freeze/'FREEZE_RECEIPT.json',
            OUT/'SOURCE_COMMIT_READBACK.json',OUT/'EXECUTION_READY.json',OUT/'FROZEN_CONTROL_ADOPTION.json']
    assert sum(f.stat().st_size for f in files)<16*1024**2
    inventory={}
    for f in dict.fromkeys(files):
        rel=f.relative_to(ROOT);q=destination/rel;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,q);assert sha(f)==sha(q);inventory[rel.as_posix()]=dict(sha256=sha(f),bytes=f.stat().st_size)
    write(destination/'MANIFEST.json',dict(UTC=now(),source_commit=source['source_commit'],files=inventory),True)
    (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(inventory.items())),encoding='utf-8')
    target=OUT/('D_STARTUP_BACKUP_RECEIPT.json' if start else 'D_SOURCE_BACKUP_RECEIPT.json')
    write(target,dict(UTC=now(),status='all_copied_files_SHA256_verified',destination=str(destination),
        files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),
        manifest_sha256=sha(destination/'MANIFEST.json'),no_raw_arrays_or_private_checkpoints=True),True)
    print(json.dumps(dict(status='small_D_backup_SHA256_verified',files=len(inventory))))
    raise SystemExit(0)

program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json');master=read(ROOT/'MASTER_EXECUTION_PLAN.json')
selector_path=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json');selector=read(selector_path)
meta=[ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'README.md',selector_path]
snapshot=OUT/('metadata_before_ready' if args.stage=='ready' else 'metadata_before_started')
assert not snapshot.exists();snapshot.mkdir()
for f in meta:shutil.copy2(f,snapshot/f.name)
stamp=now()
if args.stage=='ready':
    assert args.review and jobs['active_scientific_jobs']==0 and program['new_training_updates']==159667
    review=resolve(args.review);rv=read(review);assert rv.get('passed') is True or rv.get('status')=='PASS'
    assert not any(j.get('job_id')=='N01_WP2_014_20261015' for j in jobs['jobs'])
    refs=dict(configuration=CFG,configuration_sha256=fr['config_sha256'],source_manifest=MF,
        source_manifest_sha256=fr['source_manifest_sha256'],source_files=95,source_controls=97,protected_source_records=271,
        freeze_receipt=reference(freeze/'FREEZE_RECEIPT.json'),independent_review=reference(review),
        remote_root=p['remote_root'],protocol='docs/WP2_014_CONFIGURATION.md',roles='WP2_DATA_ROLES.json',
        budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],run_resource_limits=p['run_resource_limits'],
        status='frozen_ready_not_started',C2_enabled=False,new_training_updates=0,new_AD_queries=0,
        cumulative_training_before_run=159667,source_commit_pending_local_readback=True)
    program.update(status='WP2_014_frozen_ready',current_stage='WP2_014_paired_short_trials',
        current_work_package='WP2',active_scientific_jobs=0,current_run_training_updates=0,wp2_execution=refs,
        active_research=None,scientific_status='response_match_not_established')
    jobs['jobs'].append(dict(job_id='N01_WP2_014_20261015',run_id=p['run_id'],scientific_job=True,
        status='frozen_ready_not_started',registered_UTC=stamp,user_authority='USER_MASTER_PLAN_20261015.txt',
        source_freeze_pending=False,exact_budget_pending=False,planned_budgets=p['budget_caps'],
        candidate_trials=p['candidate_trial_order'],planned_short_candidate_runs=4,completed_short_candidate_runs=0,
        G_updates=0,new_AD_queries=0,automatic_restart=False,DEV_numeric_read=False,execution=refs,
        resource_limits=p['resource_limits'],scientific_status='response_match_not_established'))
    jobs['active_scientific_jobs']=0
    master.update(status='WP2_014_frozen_ready',active_work_package='WP2',active_scientific_jobs=0,
        current_action='Start one frozen serial C0/C1 batch; verify first realG1; set one wake then rest.')
    wp=next(w for w in master['work_packages'] if w['id']=='WP2');wp.update(status='frozen_ready',execution=refs)
    master['budgets']['short_candidate_runs']['registered_WP2_trials']=4
    master['budgets']['WP2_014']=dict(budget_caps=p['budget_caps'],resource_limits=p['resource_limits'],run_limits=p['run_resource_limits'],
        separate_from_WP1_zero_AD=True,original_cumulative_training=159667)
    selector.update(status='WP2_014_frozen_ready',active_scientific_jobs=0,current_work_package='WP2',wp2_execution=refs)
    receipt=OUT/'EXECUTION_READY.json'
    write(receipt,dict(schema='n01/wp2-execution-ready/v1',UTC=stamp,refs=refs,
        cumulative_training_updates=159667,actual_WP2_training_updates=0,actual_WP2_AD_queries=0),True)
else:
    assert args.start_dir and args.verification_dir and jobs['active_scientific_jobs']==0
    start=resolve(args.start_dir);verification=resolve(args.verification_dir)
    sv=read(verification/'START_VALIDATION.json');assert sv.get('status')=='actual_first_G1_source_and_initialization_verified'
    state=read(verification/'REMOTE_STATE.json');launch=read(start/'START_RECEIPT.json')
    assert not state.get('runs/primary/artifacts/FAILED.json')
    counts=state['durable_child_progress']['completed_counts']
    assert 1<=counts['G_updates']<=8000
    assert not state['durable_child_progress'].get('failed_ids')
    source=read(OUT/'SOURCE_COMMIT_READBACK.json')
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP2_014_20261015')
    refs=program['wp2_execution'].copy()
    refs.update(status='native_running',source_commit=source['source_commit'],source_commit_pending_local_readback=False,
        source_commit_readback=reference(OUT/'SOURCE_COMMIT_READBACK.json'),
        start_receipt=reference(start/'START_RECEIPT.json'),startup_verification=reference(verification/'START_VALIDATION.json'),
        analysis_pid=sv['child_pid'],supervisor_pid=launch['supervisor_pid'],
        last_verified_completed_counts=counts,actual_costs_are_durable_lower_bounds_while_running=True,
        correct_start_verified=True,new_training_updates=counts['G_updates'],new_AD_queries=counts['new_AD_queries'])
    program.update(status='WP2_014_native_running',active_scientific_jobs=1,active_research=p['run_id'],
        current_run_training_updates=counts['G_updates'],new_training_updates=159667+counts['G_updates'],
        wp2_execution=refs,wp2_G_updates=counts['G_updates'],wp2_AD_queries=counts['new_AD_queries'],
        training_update_accounting='Historical159667 + last verified014 durableG; running counts are lower bounds until collection.',
        active_scientific_jobs_status_basis='014 realG1 and same-snapshot health/source/worker PID verified; no continual monitoring')
    job.update(status='native_running',execution=refs,actual_counters_last_verified=counts,G_updates=counts['G_updates'],
        new_AD_queries=counts['new_AD_queries'],analysis_pid=sv['child_pid'],supervisor_pid=launch['supervisor_pid'],
        correct_start_verified=True,verified_start_UTC=stamp)
    jobs['active_scientific_jobs']=1
    master.update(status='WP2_014_native_running',active_scientific_jobs=1,current_action='Four fixed paired short trials running; one scheduled wake then rest.')
    next(w for w in master['work_packages'] if w['id']=='WP2').update(status='native_running',execution=refs)
    selector.update(status='WP2_014_native_running',active_scientific_jobs=1,latest_run_id=p['run_id'],wp2_execution=refs)
    write(OUT/'EXECUTION_STARTED.json',dict(schema='n01/wp2-correct-start/v1',UTC=stamp,refs=refs,
        current_training_updates=counts['G_updates'],cumulative_training_last_verified=159667+counts['G_updates'],
        scientific_status='response_match_not_established',formal_response_acceptance=False),True)

for obj in [program,jobs,master,selector]:obj['updated_UTC']=stamp
write(ROOT/'MASTER_EXECUTION_PLAN.json',master)
program['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(ROOT/'PROGRAM.json',program)
write(ROOT/'jobs.json',jobs)
if args.stage=='started':
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    lead='''<!-- WP2_014_STARTED_20261015 -->
## 当前作业：WP2/014 C0/C1两种子配对短试

014已通过首次真实AdamW更新与同一快照健康检查。固定4trial串行、每次2000G，最多8000G；新AD最多9872，已与013零AD成本分开登记。保护A/005，C2关闭，硬ADC/first-argmax和013冻结数字T保持不变。运行期计数以最后持久化核验为下界，完整结果待回收。

正确启动后设置同一n01的一次唤醒并休息。[014配置](wp2_014_config.json) · [014固定方案](docs/WP2_014_CONFIGURATION.md) · [启动回执](receipts/wp2_014_20261015/EXECUTION_STARTED.json) · [013报告](reports/wp1_013_20261015/N01_WP1_013_REPORT.md)

<!-- END_WP2_014_STARTED_20261015 -->

'''
    assert '<!-- WP2_014_STARTED_20261015 -->' not in text
    readme.write_text(lead+text,encoding='utf-8',newline='\n')
selector['plan_sha256']=sha(ROOT/'README.md');write(selector_path,selector)
print(json.dumps(dict(status=program['status'],active_scientific_jobs=program['active_scientific_jobs'],
    current_run_training_updates=program['current_run_training_updates'])))
