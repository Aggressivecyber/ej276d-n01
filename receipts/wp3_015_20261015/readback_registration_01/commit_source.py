"""Commit the six CPU readback controls and small local evidence, then verify bytes."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT)

cfg_path=OUT/'READBACK_EXECUTION.json';cfg=read(cfg_path)
assert read(BASE/'postrun_independent_01/ASTRA_PRELAUNCH_REVIEW.json')['passed'] is True
assert not git(['diff','--cached','--name-only']).strip()
assert not (OUT/'SOURCE_COMMIT_READBACK.json').exists()
controls=[Path(row['local_path']) for row in cfg['files'].values()]+[cfg_path]
assert len(controls)==6
for row in cfg['files'].values():
    assert record(Path(row['local_path']))=={k:row[k] for k in ['bytes','sha256']}
original=read(BASE/'SOURCE_COMMIT_READBACK.json')
for name,row in original['files'].items():assert record(ROOT/name)==row,name
assert read(ROOT/'PROGRAM.json')['new_training_updates']==175667
evidence=[
    OUT/'register_readback.py',Path(__file__).resolve(),OUT/'prepare_followup.py',OUT/'record_verified_start_and_followup.py',
    OUT/'FOLLOWUP_PROMPT_TEMPLATE.txt',OUT/'ROOT_REGISTRATION_RECEIPT.json',OUT/'AGENT_MODEL_POLICY.json',
    BASE/'postrun_independent_01/READBACK_WORKER_READY.json',BASE/'postrun_independent_01/READBACK_SCOPE_NOTE.json',
    BASE/'postrun_independent_01/BINDING_LITERAL_FIX.json',BASE/'postrun_independent_01/ASTRA_PRELAUNCH_REVIEW.json',
    BASE/'postrun_independent_01/DEVELOPMENT_SELECTION.json',BASE/'postrun_independent_01/DEVELOPMENT_SUMMARY.md',
    BASE/'postrun_independent_01/LOCAL_METADATA_READBACK.json',BASE/'readback_engineering_01/CONTROLLER_READY.json',
    BASE/'postrun_independent_01/prepare_saved_readback015.py',BASE/'postrun_independent_01/read_local015.py',
    BASE/'postrun_independent_01/WORKER_ADAPTATION.patch',BASE/'readback_engineering_01/CONTROLLER_SEAL.patch',
    BASE/'readback_engineering_01/check_readback_control.py',BASE/'readback_engineering_01/artificial_stdlib_01/ENGINEERING_RECEIPT.json',
    BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json',BASE/'postrun_root_01/COLLECTION_READBACK.json',
    BASE/'native_execution_01/WAKE_COMPLETION_SUMMARY.json',BASE/'native_execution_01/WAKE_COMPLETION_SUMMARY.md',
    BASE/'collection_01/COLLECTION_RECEIPT.json',ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md']
paths=list(dict.fromkeys(p.relative_to(ROOT).as_posix() for p in [*controls,*evidence]))
assert all((ROOT/name).is_file() for name in paths)
assert all(Path(name).suffix in ('.py','.json','.md','.txt','.patch') for name in paths)
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
staged=git(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
assert set(staged)<=set(paths) and staged
subprocess.run(['git','commit','-m','Register bounded WP3 saved-array audit and adopt Astra Max dispatch'],cwd=ROOT,check=True)
commit=git(['rev-parse','HEAD']).decode().strip()
records={}
for path in controls:
    name=path.relative_to(ROOT).as_posix();blob=git(['show',commit+':'+name])
    assert blob==path.read_bytes(),name
    records[name]=record(path)
receipt=dict(schema='n01/wp3-readback-source-commit/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
    source_commit=commit,source_file_count=5,source_controls=6,raw_Git_blobs_byte_exact=True,files=records,
    original_training_source_commit=original['source_commit'],original_training_control_count=len(original['files']),
    original_training_controls_unchanged=True,committed_small_evidence_paths=paths,private_numeric_payloads_committed=False,
    public_push=False,remote_execution_performed=False)
write(OUT/'SOURCE_COMMIT_READBACK.json',receipt)
destination=Path('D:/GAN_backups/N01/wp3_015_20261015/readback_source_01')
assert not destination.exists();destination.mkdir(parents=True)
backup_paths=[*controls,*evidence,OUT/'SOURCE_COMMIT_READBACK.json']
backup_paths=list(dict.fromkeys(backup_paths))
total=sum(p.stat().st_size for p in backup_paths)
assert shutil.disk_usage(destination).free>total+32*1024**2
inventory={}
for path in backup_paths:
    name=path.relative_to(ROOT).as_posix();target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target);assert record(target)==record(path);inventory[name]=record(path)
manifest=destination/'SHA256SUMS.txt'
manifest.write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
write(OUT/'D_SOURCE_BACKUP_RECEIPT.json',dict(status='all_copied_files_SHA256_verified',destination=str(destination),
    files=len(inventory),bytes=total,manifest_sha256=sha(manifest),source_commit=commit,source_controls=6))
print(json.dumps(dict(status='readback_source_committed_and_D_backed_up',source_commit=commit,
    raw_blobs_verified=6,original_training_controls_unchanged=len(original['files']),backup_files=len(inventory),bytes=total)))
