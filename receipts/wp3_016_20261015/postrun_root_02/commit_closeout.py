"""Commit only016 completion and report evidence, preserving WP4 preparation."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))
call=lambda args:subprocess.check_output(['git',*args],cwd=ROOT)

existing_staged=call(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
if existing_staged==['']:existing_staged=[]
assert read(ROOT/'PROGRAM.json')['new_training_updates']==179667
assert read(ROOT/'PROGRAM.json')['active_scientific_jobs']==0
assert read(BASE/'FINAL_RESULT_RECEIPT.json')['report_visual_QA']['sha256']==sha(HERE/'REPORT_VISUAL_QA.json')
old=read(BASE/'SOURCE_COMMIT_READBACK_FIX02.json')
cpu=read(BASE/'readback_registration_01/SOURCE_COMMIT_READBACK.json')
for rows in [old['files'],cpu['files']]:
    for name,row in rows.items():assert record(ROOT/name)==row,name
report=ROOT/'reports/wp3_016_20261015';manifest=read(report/'report_manifest.json')
files=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]
files += [report/name for name in manifest['outputs']]+[report/'report_manifest.json',report/'REPORT_QA.json']
files += [BASE/n for n in ['FINAL_RESULT_RECEIPT.json','D_CLOSEOUT_BACKUP_RECEIPT.json']]
for sub in ['postrun_root_02','postrun_independent_01','postrun_report_impl_01','readback_registration_01',
            'readback_execution_01','readback_prepare_01','readback_start_01','readback_start_verification_01','readback_collection_01']:
    for path in (BASE/sub).iterdir():
        if path.is_file() and path.suffix.lower() in ['.json','.py','.md','.txt']:files.append(path)
for path in (HERE/'html_qa_02').iterdir():
    if path.is_file() and path.suffix in ['.png','.html','.json']:files.append(path)
collected=ROOT/'runs/wp3_016_20261015/readback_collected_01'
for path in (collected/'artifacts').rglob('*'):
    if path.is_file() and path.suffix in ['.json','.jsonl']:files.append(path)
for name in ['START.json','DISPATCH_ATTEMPT.json','PREPARED.json','supervisor/exit_receipt.json','supervisor/launch.json','supervisor/resource_status.json']:
    path=collected/name
    if path.is_file():files.append(path)
files=list(dict.fromkeys(files))
assert all(path.is_file() and path.suffix.lower() not in ['.npz','.pt','.pth','.tar','.gz','.log'] for path in files)
relative=[path.relative_to(ROOT).as_posix() for path in files]
assert all('wp4_017' not in path for path in relative)
assert set(existing_staged)<=set(relative), 'Preserve unrelated staged work'
subprocess.run(['git','add','-f','--',*relative],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check','--','.',':(exclude)reports/wp3_016_20261015/figures/*.svg'],cwd=ROOT,check=True)
staged=call(['diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
assert set(staged)<=set(relative)
subprocess.run(['git','commit','-m','Close WP3 016 with independent array audit and rendered report'],cwd=ROOT,check=True)
commit=call(['rev-parse','HEAD']).decode().strip()
for path in files:
    name=path.relative_to(ROOT).as_posix();blob=call(['cat-file','blob',commit+':'+name])
    if name=='README.md':assert blob.replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n')
    else:assert blob==path.read_bytes(),name
for name,row in cpu['files'].items():assert call(['cat-file','blob',cpu['source_commit']+':'+name])==(ROOT/name).read_bytes()
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
receipt=dict(schema='n01/wp3-016-closeout-commit/v1',UTC=stamp,status='local016_closeout_commit_verified',commit=commit,
    changed_paths=len(staged),verified_artifact_paths=len(files),committed_artifact_paths=relative,
    README_Git_newline_normalization_verified=True,all_other_artifacts_exact_raw_blobs=True,
    original_training_controls_unchanged=118,CPU_raw_controls_unchanged=6,original_raw_or_checkpoints_committed=False,
    generated_SVG_original_bytes_preserved=True,generated_SVG_paths_excluded_from_source_whitespace_check=True,
    WP4_preparation_committed_by_this_action=False,public_push=False,cumulative_G=179667,whole_project_complete=False)
receipt_path=HERE/'CLOSEOUT_COMMIT_RECEIPT.json'
with receipt_path.open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
backup=Path('D:/GAN_backups/N01/wp3_016_20261015/closeout_01')
assert backup.is_dir()
delivery={}
for path in [receipt_path,Path(__file__).resolve()]:
    name=path.relative_to(ROOT).as_posix();target=backup/'commit_delivery'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target);assert record(target)==record(path);delivery[name]=record(target)
with (backup/'COMMIT_DELIVERY_MANIFEST.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(dict(UTC=stamp,commit=commit,files=delivery),stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(dict(status=receipt['status'],commit=commit,changed_paths=len(staged),verified_artifacts=len(files),cumulative_G=179667)))
