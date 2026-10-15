"""Close local startup administration using saved evidence only; no remote calls."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CURRENT=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['receipt','commit']);args=ap.parse_args()
source=read(OUT/'SOURCE_COMMIT_READBACK.json');mf=read(ROOT/'WP3_SOURCE_MANIFEST.json')
assert len(mf['files'])==105 and len(source['files'])==107
for name,row in source['files'].items():
    path=ROOT/name;assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],name
old=read(ROOT/'WP2_SOURCE_MANIFEST.json');assert len(old['files'])==95
for name,row in old['files'].items():assert sha(ROOT/name)==row['sha256'],name
assert sha(ROOT/'wp2_014_config.json')==old['config_sha256']
assert sha(ROOT/'WP2_SOURCE_MANIFEST.json')==source['files']['WP2_SOURCE_MANIFEST.json']['sha256']
for label,count in [('D_SOURCE_BACKUP_RECEIPT.json',113),('D_STARTUP_BACKUP_RECEIPT.json',10)]:
    backup=read(OUT/label);dest=Path(backup['destination']);inventory=read(dest/'MANIFEST.json')
    assert backup['files']==count and sha(dest/'MANIFEST.json')==backup['manifest_sha256']
    for name,row in inventory['files'].items():assert sha(dest/name)==row['sha256'],name

if args.stage=='receipt':
    for name,row in source['files'].items():
        blob=subprocess.check_output(['git','cat-file','blob',source['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==row['sha256'] and blob==(ROOT/name).read_bytes(),name
    state=read(OUT/'start_verification_01/REMOTE_STATE.json');progress=state['durable_child_progress']
    status=state['runs/primary/artifacts/status.json'];counts=progress['completed_counts']
    assert counts['G_updates']==counts['new_AD_queries']==counts['training_backward_queries']==120
    assert not progress['failed_ids'] and not progress['truncated_tail']
    assert state['runs/primary/artifacts/FAILED.json'] is None
    assert state['runs/primary/supervisor/exit_receipt.json'] is None
    assert state['runs/primary/artifacts/COMPLETE.json'] is None
    assert status['counters']['mean_adapter_rows']==0
    validation=read(OUT/'start_verification_01/START_VALIDATION.json')
    assert validation['status']=='actual_first_G1_and_mean_gate_verified'
    follow=read(OUT/'FOLLOWUP_RECEIPT.json');assert follow['actual_saved_settings_verified'] and follow['count']==1
    assert dt.datetime.fromisoformat(follow['next_wake_UTC'])>dt.datetime.now(dt.timezone.utc)
    program,master,jobs,current=[read(p) for p in (ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',CURRENT)]
    assert program['new_training_updates']==current['new_training_updates']==167787
    assert program['master_plan_sha256']==sha(ROOT/'MASTER_EXECUTION_PLAN.json')
    assert all(v['active_scientific_jobs']==1 for v in [program,master,jobs,current])
    assert program['latest_source_commit']==current['latest_source_commit']==source['source_commit']
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_015_20261015')
    assert job['G_updates']==job['new_AD_queries']==120
    if current.get('master_plan_sha256')!=sha(ROOT/'MASTER_EXECUTION_PLAN.json'):
        prior=OUT/'metadata_binding_fix01';prior.mkdir(exist_ok=False)
        shutil.copy2(CURRENT,prior/CURRENT.name)
        previous=current.get('master_plan_sha256');current['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
        with CURRENT.open('w',encoding='utf-8',newline='\n') as f:json.dump(current,f,ensure_ascii=False,indent=2);f.write('\n')
        write(OUT/'METADATA_BINDING_FIX.json',dict(UTC=now(),scope='administrative master-plan hash only',
            preserved_before=ref(prior/CURRENT.name),previous_master_plan_sha256=previous,
            corrected_master_plan_sha256=current['master_plan_sha256'],new_CP_NN_AD_G_queries=0))
    assert current['plan_sha256']==sha(ROOT/'README.md')
    current_copy=OUT/'CURRENT_PROGRAM_AT_DELIVERY.json'
    assert not current_copy.exists();shutil.copy2(CURRENT,current_copy)
    refs=[OUT/n for n in ['EXECUTION_STARTED.json','CORRECT_START_AND_FOLLOWUP.json','FOLLOWUP_RECEIPT.json',
        'SOURCE_COMMIT_READBACK.json','D_SOURCE_BACKUP_RECEIPT.json','D_STARTUP_BACKUP_RECEIPT.json',
        'native_execution_01/NATIVE_START_SUMMARY.json','start_verification_01/REMOTE_STATE.json',
        'start_verification_01/START_VALIDATION.json','independent_01/PRELAUNCH_REVIEW.json',
        'independent_01/FROZEN_PACKAGE_REVIEW.json','CURRENT_PROGRAM_AT_DELIVERY.json']]
    receipt=dict(schema='n01/wp3-startup-delivery/v1',UTC=now(),run_id='wp3_015_20261015',
        status='correct_start_and_single_future_wake_confirmed_rest',source_commit=source['source_commit'],
        source_files105_unchanged=True,source_and_control_Git_blobs107_verified=True,
        original014_source95_plus_controls97_unchanged=True,proofs=[ref(p) for p in refs],
        actual_first_G1_and_nonzero_parameter_update_verified=True,all6_mean_evidence_gate_passed=True,
        startup_snapshot_UTC=state['UTC'],last_verified_completed_counts=counts,
        mean_adapter_rows=dict(value=0,evidence='saved status explicit zero; durable journal omits zero charges'),
        earlier_status_view=dict(UTC=status['UTC'],completed_counters=status['counters'],
            attempted_counters=status['attempted_counters'],inflight=status['inflight']),
        later_durable_view=dict(completed_counts=counts,unknown_inflight_ids=progress['unknown_inflight_ids'],
            failed_ids=progress['failed_ids'],truncated_tail=progress['truncated_tail']),
        snapshot_views_not_synchronized=True,no_G121_or_unobserved_updates_inferred=True,
        cumulative_training_before015=167667,cumulative_training_last_verified=167787,
        startup_updates_counted_once=True,actual_costs_are_lower_bounds_while_job_runs=True,
        planned_G_AD_each8000_not_claimed_completed=True,full_batch_native_completion='unknown_until_next_wake',
        no_remote_calls_after_the_single_start_verification=True,continuous_agent_monitoring=False,
        followup=dict(id=follow['automation_id'],kind=follow['kind'],count=1,next_wake_local=follow['next_wake_local'],
            next_wake_UTC=follow['next_wake_UTC'],actual_saved_settings_verified=True),
        D_source_backup_files=113,D_startup_backup_files=10,all_backup_SHA256_verified=True,
        metadata={p.name:ref(p) for p in [ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md']},
        wp2_014_result=dict(receipt=ref(ROOT/'receipts/wp2_014_20261015/FINAL_RESULT_RECEIPT.json'),
            report=ref(ROOT/'reports/wp2_014_20261015/N01_WP2_014_REPORT.md'),
            saved_array_readback_complete=True,full_loss_CDF_or_median_refit=False,
            selected_code_path='C1',useful_improvement_passes=5,required_conditions=6,weights_promoted=False),
        response_match_established=False,formal_response_acceptance=False,
        whole_six_workpackage_project_complete=False,public_push_performed=False,
        all_new_artifacts_on_E_or_D=True)
    write(OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json',receipt)
    print(json.dumps(dict(status=receipt['status'],next_wake_local=follow['next_wake_local'],
        receipt_sha256=sha(OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json'),cumulative_training_last_verified=167787)))
    raise SystemExit(0)

assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
files=[ROOT/n for n in ['PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md']]
files += [OUT/n for n in ['register_execution.py','followup.py','finalize_startup.py','SOURCE_COMMIT_READBACK.json',
    'D_SOURCE_BACKUP_RECEIPT.json','D_STARTUP_BACKUP_RECEIPT.json','EXECUTION_READY.json','EXECUTION_STARTED.json',
    'FOLLOWUP_PLAN.json','FOLLOWUP_PROMPT.txt','FOLLOWUP_RECEIPT.json','CORRECT_START_AND_FOLLOWUP.json',
    'METADATA_BINDING_FIX.json','CURRENT_PROGRAM_AT_DELIVERY.json','FINAL_STARTUP_DELIVERY_RECEIPT.json']]
files += [OUT/n for n in ['deploy_native_01/install_NATIVE_RECEIPT.json','deploy_native_01/prepare_NATIVE_RECEIPT.json',
    'deploy_native_01/REMOTE_INSTALL_RECEIPT.json','deploy_native_01/SOURCE_TRANSFER.json',
    'native_execution_01/FIRST_EVENT_WAIT.json','native_execution_01/LOCAL_RELEASE_GATE.json',
    'native_execution_01/NATIVE_START_SUMMARY.json','start_native_01/start_NATIVE_RECEIPT.json','start_native_01/START_RECEIPT.json',
    'start_verification_01/REMOTE_STATE.json','start_verification_01/START_VALIDATION.json',
    'start_verification_01/status_NATIVE_RECEIPT.json','metadata_binding_fix01/CURRENT_SIM2REAL_PROGRAM.json']]
files += [ROOT/'receipts/wp2_014_20261015/postrun_root_02/CLOSEOUT_COMMIT_RECEIPT.json']
assert all(p.is_file() and p.stat().st_size<4*1024**2 for p in files)
names=[p.relative_to(ROOT).as_posix() for p in dict.fromkeys(files)]
subprocess.run(['git','add','--',*names],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert staged and set(staged)<=set(names)
subprocess.run(['git','commit','-m','Record WP3 015 correct start and single followup'],cwd=ROOT,check=True,capture_output=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for name in names:
    assert subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),name
write(OUT/'STARTUP_METADATA_COMMIT_RECEIPT.json',dict(UTC=now(),metadata_commit=commit,
    actual_executing_source_commit=source['source_commit'],files_verified=len(names),all_raw_Git_blobs_match=True,
    explicitly_staged_paths=names,public_push_performed=False))
destination=Path('D:/GAN_backups/N01/wp3_015_20261015/delivery_01');destination.mkdir(parents=True,exist_ok=False)
small=[OUT/n for n in ['FINAL_STARTUP_DELIVERY_RECEIPT.json','STARTUP_METADATA_COMMIT_RECEIPT.json',
    'CURRENT_PROGRAM_AT_DELIVERY.json','FOLLOWUP_PLAN.json','FOLLOWUP_PROMPT.txt','FOLLOWUP_RECEIPT.json',
    'CORRECT_START_AND_FOLLOWUP.json','METADATA_BINDING_FIX.json','followup.py','finalize_startup.py']]
inventory={}
for path in small:
    target=destination/path.name;shutil.copy2(path,target);assert sha(path)==sha(target)
    inventory[path.name]=dict(bytes=target.stat().st_size,sha256=sha(target))
write(destination/'MANIFEST.json',dict(UTC=now(),metadata_commit=commit,files=inventory))
(destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(inventory.items())),encoding='utf-8')
write(OUT/'D_DELIVERY_BACKUP_RECEIPT.json',dict(UTC=now(),destination=str(destination),files=len(inventory),
    manifest_sha256=sha(destination/'MANIFEST.json'),all_SHA256_verified=True))
print(json.dumps(dict(status='local_metadata_commit_and_delivery_backup_verified',metadata_commit=commit,
    executing_source_commit=source['source_commit'],Git_blobs_verified=len(names),D_delivery_files=len(inventory))))
