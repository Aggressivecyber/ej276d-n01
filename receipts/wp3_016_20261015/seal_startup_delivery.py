"""Seal already verified startup/followup; never calls the native host."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)


def main():
    summary=read(OUT/'CORRECT_START_AND_FOLLOWUP.json')
    followup=read(OUT/'FOLLOWUP_RECEIPT.json')
    started=read(OUT/'EXECUTION_STARTED.json')
    source=read(OUT/'SOURCE_COMMIT_READBACK_FIX02.json')
    backup=read(OUT/'D_STARTUP_BACKUP_RECEIPT.json')
    assert summary['status']=='correct_native_start_and_one_future_wake_verified' and summary['rest_now']
    assert dt.datetime.fromisoformat(followup['scheduled_UTC'])>dt.datetime.now(dt.timezone.utc)
    assert followup['occurrences']==1 and followup['saved_app_prompt_exactly_matches']
    assert backup['status']=='all_copied_files_SHA256_verified'
    assert source['source_controls']==118 and source['all_raw_Git_blobs_match_frozen_bytes']
    for name,row in source['files'].items():
        p=ROOT/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
    files=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]
    names=['SOURCE_COMMIT_READBACK_FIX02.json','D_SOURCE_BACKUP_RECEIPT_FIX02.json',
        'record_correct_start.py','mark_fix02_dispatch.py','build_followup_prompt.py','finalize_followup.py',
        'seal_startup_delivery.py','EXECUTION_STARTED.json','FOLLOWUP_PROMPT.txt','FOLLOWUP_TOOL_RESULT.json',
        'FOLLOWUP_RECEIPT.json','CORRECT_START_AND_FOLLOWUP.json','D_STARTUP_BACKUP_RECEIPT.json',
        'native_execution_03/NATIVE_START_SUMMARY.json','start_native_02/START_RECEIPT.json',
        'start_verification_02/START_VALIDATION.json','start_verification_02/REMOTE_STATE.json']
    files.extend(OUT/name for name in names)
    for name in ['native_execution_03/NATIVE_START_SUMMARY.md','native_execution_03/FIRST_EVENT_WAIT_RECEIPT.json']:
        if (OUT/name).is_file():files.append(OUT/name)
    files.extend(OUT/name for name in ('FOLLOWUP_PROMPT_SENT.txt','FOLLOWUP_PROMPT_CANONICALIZATION.json') if (OUT/name).is_file())
    assert all(p.is_file() and p.suffix.lower() not in ('.npz','.pt','.tar','.gz') for p in files)
    relative=[p.relative_to(ROOT).as_posix() for p in files]
    subprocess.run(['git','add','-f','--',*relative],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).decode().splitlines()
    assert set(staged)<=set(relative)
    subprocess.run(['git','commit','-m','Record verified WP3 016 startup and one scheduled recovery'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    for name in relative:
        blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
        actual=(ROOT/name).read_bytes()
        assert blob==actual or (name=='README.md' and blob.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n')),name
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    receipt=dict(schema='n01/wp3-016-startup-delivery/v1',UTC=stamp,
        status='correct_native_start_one_future_wake_and_E_D_delivery_verified',
        scientific_source_commit=source['source_commit'],startup_metadata_commit=commit,
        source_controls=118,all_frozen_local_source_controls_still_match=True,
        startup=ref(OUT/'EXECUTION_STARTED.json'),native_summary=ref(OUT/'native_execution_03/NATIVE_START_SUMMARY.json'),
        followup=ref(OUT/'FOLLOWUP_RECEIPT.json'),D_source_backup=ref(OUT/'D_SOURCE_BACKUP_RECEIPT_FIX02.json'),
        D_startup_backup=ref(OUT/'D_STARTUP_BACKUP_RECEIPT.json'),
        prior_failed_attempt=ref(OUT/'NATIVE_FAILURE_RECONCILIATION.json'),
        prior_failed_attempt_D_backup=ref(OUT/'D_FAILURE_BACKUP_RECEIPT.json'),
        actual_G_last_snapshot=started['actual_counts_last_verified']['G_updates'],
        cumulative_G_last_snapshot=started['cumulative_training_updates'],
        counts_after_saved_snapshot='unknown_not_polled',
        next_wake_local=followup['scheduled_local_Asia_Singapore'],single_occurrence=True,
        engineering_and_independent_review_model='gpt-6.1-sol / max',
        public_push_performed=False,scientific_acceptance=False,weights_promoted=False,
        whole_project_complete=False,rest_now=True)
    path=OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    destination=Path(backup['destination']);target=destination/path.name
    assert not target.exists();shutil.copy2(path,target);assert sha(path)==sha(target)
    delivery=dict(UTC=stamp,startup_metadata_commit=commit,files={path.name:dict(sha256=sha(target),bytes=target.stat().st_size)},
        original_backup_manifest_sha256=backup['manifest_sha256'])
    delivery_path=destination/'FINAL_DELIVERY_MANIFEST.json'
    with delivery_path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(delivery,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],source_commit=source['source_commit'],
        startup_metadata_commit=commit,final_receipt_sha256=sha(path),wake_local=receipt['next_wake_local'],
        snapshot_G=receipt['actual_G_last_snapshot'],cumulative_G=receipt['cumulative_G_last_snapshot'],rest_now=True)))


if __name__=='__main__':main()
