"""Seal local startup delivery without inspecting the running remote process."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))

def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

source=read(OUT/'SOURCE_COMMIT_READBACK.json')
old=read(BASE/'SOURCE_COMMIT_READBACK_FIX02.json')
for inventory in [source['files'],old['files']]:
    for name,row in inventory.items():assert record(ROOT/name)==row,name
follow=read(OUT/'FOLLOWUP_RECEIPT.json')
actual=tomllib.loads(Path('C:/Users/10629/.codex/automations/n01/automation.toml').read_text(encoding='utf-8'))
plan=read(OUT/'FOLLOWUP_PLAN.json')
assert actual['status']=='ACTIVE' and actual['rrule']==plan['rrule'] and actual['prompt']==(OUT/'FOLLOWUP_PROMPT.txt').read_text(encoding='utf-8').rstrip('\r\n')
assert actual['target_thread_id']==plan['target_thread_id']
assert dt.datetime.fromisoformat(follow['next_wake_UTC'])>dt.datetime.now(dt.timezone.utc)
assert read(ROOT/'PROGRAM.json')['new_training_updates']==179667
assert read(ROOT/'PROGRAM.json')['active_scientific_jobs']==1
assert read(ROOT/'PROGRAM.json')['master_plan_sha256']==sha(ROOT/'MASTER_EXECUTION_PLAN.json')
assert read(CURRENT)['master_plan_sha256']==sha(ROOT/'MASTER_EXECUTION_PLAN.json')
assert read(CURRENT)['plan_sha256']==sha(ROOT/'README.md')
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
files=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]
files += [p for p in OUT.glob('*') if p.is_file() and p.suffix in ['.json','.py','.txt']]
for directory in ['readback_prepare_01','readback_start_01','readback_start_verification_01','readback_execution_01']:
    for path in (BASE/directory).rglob('*'):
        if path.is_file() and path.suffix in ['.json','.py','.txt','.log']:
            assert path.stat().st_size<8*1024**2,path
            files.append(path)
files=list(dict.fromkeys(files))
relative=[p.relative_to(ROOT).as_posix() for p in files]
subprocess.run(['git','add','-f','--',*relative],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).decode().splitlines()
assert set(staged)<=set(relative)
subprocess.run(['git','commit','-m','Record verified WP3 016 CPU start and one future wake'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
for path in files:
    name=path.relative_to(ROOT).as_posix()
    blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
    if path.name=='README.md':assert blob.replace(b'\r\n',b'\n')==path.read_bytes().replace(b'\r\n',b'\n')
    else:assert blob==path.read_bytes(),name
for name,row in source['files'].items():
    assert subprocess.check_output(['git','cat-file','blob',source['source_commit']+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),name
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
receipt=dict(schema='n01/wp3-016-CPU-startup-delivery/v1',UTC=stamp,status='correct_start_one_future_wake_confirmed_rest',
    startup_metadata_commit=commit,CPU_source_commit=source['source_commit'],source_controls_raw_blobs_verified=6,
    original_training_controls_unchanged=118,committed_small_evidence_paths=relative,
    cumulative_training_updates=179667,new_CPU_training_updates=0,first_pair_same_snapshot_health_verified=True,
    future_single_wake=follow['next_wake_local'],automation_id='n01',continuous_remote_monitoring=False,
    actual_remote_state_after_single_saved_snapshot='unknown_not_polled',whole_project_complete=False,
    scientific_status='response_match_not_established',public_push=False)
write(OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json',receipt)
destination=Path('D:/GAN_backups/N01/wp3_016_20261015/readback_startup_01')
assert destination.is_dir()
target=destination/'FINAL_STARTUP_DELIVERY_RECEIPT.json';shutil.copy2(OUT/target.name,target)
assert record(target)==record(OUT/target.name)
final_files=[target,CURRENT,Path(__file__).resolve(),OUT/'record_verified_start_and_followup.py']
inventory={}
for index,path in enumerate(final_files):
    if path!=target:
        copied=destination/f'final_{index:02d}_{path.name}';shutil.copy2(path,copied);assert record(copied)==record(path)
    else:copied=target
    inventory[copied.name]=record(copied)
write(destination/'FINAL_DELIVERY_MANIFEST.json',dict(UTC=stamp,startup_metadata_commit=commit,files=inventory))
print(json.dumps(dict(status=receipt['status'],startup_metadata_commit=commit,cumulative_G=179667,
    next_wake_local=follow['next_wake_local'],final_receipt_sha256=sha(OUT/'FINAL_STARTUP_DELIVERY_RECEIPT.json'))))
