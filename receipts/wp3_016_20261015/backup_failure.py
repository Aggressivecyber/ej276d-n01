"""Back up the selectively collected failed attempt, without decoding tensors."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    preservation=OUT/'failure_collection_01/FAILURE_PRESERVATION_RECEIPT.json'
    saved=read(preservation)
    source=ROOT/'runs/wp3_016_20261015/failed_collected_01'
    destination=Path('D:/GAN_backups/N01/wp3_016_20261015/native_failure_01')
    assert saved['collection']['all_selected_local_sizes_and_hashes_verified']
    selected=saved['collection']['selected_inventory']
    assert len(selected)==21 and saved['actual_completed_counters']['G_updates']==0
    files={}
    for name,row in selected.items():
        path=source/name
        assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
        files['failed_collected_01/'+name]=path
    for path in [preservation,OUT/'NATIVE_FAILURE_RECONCILIATION.json',
                 OUT/'failure_collection_01/COLLECTION_RECEIPT.json',OUT/'failure_collection_01/REMOTE_FULL_INVENTORY.json',
                 OUT/'launch_fix02/FAILURE_DIAGNOSIS_REVIEW.json',OUT/'native_execution_02/NATIVE_START_SUMMARY.json',
                 OUT/'start_verification_01/REMOTE_STATE.json']:
        files['receipts/'+path.relative_to(OUT).as_posix()]=path
    assert sum(p.stat().st_size for p in files.values())<8*1024**2
    destination.mkdir(parents=True,exist_ok=False)
    inventory={}
    for name,path in files.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        assert sha(path)==sha(target)
        inventory[name]=dict(bytes=target.stat().st_size,sha256=sha(target))
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    manifest=dict(schema='n01/wp3-016-failed-attempt-D-backup/v1',UTC=stamp,files=inventory,
        actual_scientific_training_updates=0,all_raw_collected=False,original_raw_retained_remote=True)
    (destination/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
    receipt=dict(UTC=stamp,status='all_failed_attempt_backup_files_SHA256_verified',destination=str(destination),
        files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json'),
        source_commit=saved['source_commit'],partial_evidence_only=True,all_raw_collected=False,
        checkpoint_and_array_decodes=0,scientific_operations=0)
    with (OUT/'D_FAILURE_BACKUP_RECEIPT.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
