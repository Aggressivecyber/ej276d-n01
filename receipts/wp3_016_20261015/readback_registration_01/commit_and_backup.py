"""Bind the six reviewed CPU controls to raw Git blobs and a verified D backup."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p))


def write(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def main():
    config_path=OUT/'READBACK_EXECUTION.json';config=read(config_path)
    assert config['cumulative_training_before_and_after_this_audit']==179667
    assert all(v==0 for v in config['scientific_budget'].values())
    review_path=ROOT/config['prelaunch_independent_review']['path'];review=read(review_path)
    assert review['passed'] and not review['blockers']
    controls=[Path(r['local_path']) for r in config['files'].values()]+[config_path]
    assert len(controls)==6
    for row in config['files'].values():assert record(Path(row['local_path']))=={k:row[k] for k in ['bytes','sha256']}
    old=read(BASE/'SOURCE_COMMIT_READBACK_FIX02.json')
    for name,row in old['files'].items():assert record(ROOT/name)==row,name
    assert read(ROOT/'PROGRAM.json')['new_training_updates']==179667
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip()
    evidence=[Path(__file__).resolve(),OUT/'register_readback.py',OUT/'bind_committed_source.py',OUT/'prepare_followup.py',OUT/'record_verified_start_and_followup.py',OUT/'FOLLOWUP_PROMPT_TEMPLATE.txt',OUT/'ROOT_REGISTRATION_RECEIPT.json',review_path,
        ROOT/config['prelaunch_worker_ready']['path'],ROOT/config['prelaunch_controller_ready']['path'],
        ROOT/config['provisional_selection']['path'],BASE/'postrun_independent_01/SCOPE_NOTE.json',
        BASE/'postrun_independent_01/PREPARATION_RECEIPT.json',BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json',
        BASE/'postrun_root_01/SELECTED_COLLECTION_READBACK.json',BASE/'postrun_root_01/NATIVE_CLOSEOUT_COMMIT_RECEIPT.json',
        ROOT/config['agent_model_policy']['path']]+[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]
    for directory in [BASE/'readback_engineering_01',BASE/'postrun_independent_01']:
        for path in directory.glob('*.py'):evidence.append(path)
        for path in directory.glob('*RECEIPT.json'):evidence.append(path)
        for path in directory.glob('*/ENGINEERING_RECEIPT.json'):evidence.append(path)
    files=list(dict.fromkeys(controls+evidence))
    assert all(p.is_file() and p.suffix in ['.py','.json','.md','.txt'] for p in files)
    relative=[p.relative_to(ROOT).as_posix() for p in files]
    subprocess.run(['git','add','-f','--',*relative],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).decode().splitlines()
    assert set(staged)<=set(relative)
    subprocess.run(['git','commit','-m','Register one bounded zero-NN WP3 016 saved-array audit'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    records={}
    for path in controls:
        name=path.relative_to(ROOT).as_posix()
        blob=subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=ROOT)
        assert blob==path.read_bytes(),name
        records[name]=record(path)
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    source_path=OUT/'SOURCE_COMMIT_READBACK.json'
    write(source_path,dict(schema='n01/wp3-016-readback-source-commit/v1',UTC=stamp,source_commit=commit,
        source_file_count=5,source_controls=6,raw_Git_blobs_byte_exact=True,files=records,
        configuration_sha256=sha(config_path),original_training_source_commit=old['source_commit'],
        original_training_control_count=118,original_training_controls_unchanged=True,
        committed_small_evidence_paths=relative,private_numeric_payloads_committed=False,public_push=False,remote_execution_performed=False))
    destination=Path('D:/GAN_backups/N01/wp3_016_20261015/readback_source_01')
    destination.mkdir(parents=True,exist_ok=False)
    files=list(dict.fromkeys(files+[source_path]))
    assert shutil.disk_usage(destination).free>sum(p.stat().st_size for p in files)+32*1024**2
    inventory={}
    for path in files:
        name=path.relative_to(ROOT).as_posix();target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);assert record(target)==record(path)
        inventory[name]=record(target)
    write(destination/'MANIFEST.json',dict(UTC=stamp,source_commit=commit,files=inventory))
    (destination/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+n+'\n' for n,r in sorted(inventory.items())),encoding='utf-8',newline='\n')
    write(OUT/'D_SOURCE_BACKUP_RECEIPT.json',dict(UTC=stamp,status='all_copied_files_SHA256_verified',destination=str(destination),
        files=len(inventory),bytes=sum(r['bytes'] for r in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json'),
        source_commit=commit,source_controls=6,private_numeric_payloads_included=False))
    print(json.dumps(dict(status='readback_source_committed_and_D_backed_up',source_commit=commit,
        raw_blobs_verified=6,original_controls_unchanged=118,backup_files=len(inventory),bytes=sum(r['bytes'] for r in inventory.values()))))


if __name__=='__main__':main()
