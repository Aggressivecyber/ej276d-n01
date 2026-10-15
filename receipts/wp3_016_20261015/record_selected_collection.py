"""Reopen selected files and preserve a D backup without decoding scientific tensors."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HERE=OUT/'postrun_root_01'
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)


def write(p,value,*,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    summary_path=OUT/'wake_execution_01/WAKE_COMPLETION_SUMMARY.json';summary=read(summary_path)
    collection_path=OUT/'collection_01/COLLECTION_RECEIPT.json';collection=read(collection_path)
    selected_path=OUT/'wake_execution_01/SELECTED_INVENTORY.json';selected_file=read(selected_path)
    selected=selected_file.get('entries',selected_file.get('files'))
    assert isinstance(selected,dict),list(selected_file)
    assert len(selected)==99 and collection['selected_file_count']==99
    assert collection['whole_remote_file_count']==211 and collection['omitted_file_count']==112
    assert collection['whole_remote_inventory_verified'] and not collection['all_raw_artifacts_collected']
    assert not collection['partial_evidence_only'] and not collection['control_read_errors']
    assert summary['journal_counts_independently_replayed_locally'] and summary['operations_completed']==94249
    source=Path(collection['collected_root'])
    for name,row in selected.items():
        path=source/name
        assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
        assert path.suffix.lower()!='.npz'
    assert sha(Path(collection['archive']))==collection['archive_sha256']
    raw=read(OUT/'wake_execution_01/RAW_NPZ_INVENTORY.json')
    assert len(raw['entries'])==112 and raw['bytes']==2169655337
    destination=Path('D:/GAN_backups/N01/wp3_016_20261015/native_completion_01')
    destination.mkdir(parents=True,exist_ok=False)
    files={'selected_collected_01/'+name:source/name for name in selected}
    for path in [summary_path,collection_path,selected_path,OUT/'wake_execution_01/RAW_NPZ_INVENTORY.json',
                 OUT/'collection_01/REMOTE_FULL_INVENTORY.json',HERE/'NATIVE_COMPLETION_READBACK.json',
                 OUT/'wake_status_01/REMOTE_STATE.json']:
        files['receipts/'+path.relative_to(OUT).as_posix()]=path
    inventory={}
    for name,path in files.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        assert sha(target)==sha(path)
        inventory[name]=dict(bytes=target.stat().st_size,sha256=sha(target))
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    write(destination/'MANIFEST.json',dict(UTC=stamp,files=inventory,raw_NPZ_retained_remote=True,all_raw_collected=False),new=True)
    (destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(inventory.items())),encoding='utf-8',newline='\n')
    backup_path=OUT/'D_NATIVE_COMPLETION_BACKUP_RECEIPT.json'
    write(backup_path,dict(UTC=stamp,status='all_copied_files_SHA256_verified',destination=str(destination),
        files=len(inventory),bytes=sum(r['bytes'] for r in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json'),
        source_commit=summary['source_commit'],all_raw_collected=False,raw_NPZ_retained_remote=True,
        checkpoint_loads=0,array_decodes=0,NN_AD_G_RNG=0),new=True)
    receipt_path=HERE/'SELECTED_COLLECTION_READBACK.json'
    receipt=dict(schema='n01/wp3-016-selected-collection-readback/v1',UTC=stamp,
        status='selected99_reopened_SHA_verified_and_D_backed_up',collection=ref(collection_path),
        native_summary=ref(summary_path),selected_inventory=ref(selected_path),D_backup=ref(backup_path),
        selected_files=99,selected_bytes=sum(r['bytes'] for r in selected.values()),
        whole_inventory_files=211,raw_NPZ_files_retained_remote=112,raw_NPZ_bytes=2169655337,
        checkpoint_files_hashed_only=6,checkpoint_loads=0,array_decodes=0,scientific_operations=0,
        independent_numeric_QA_pending=True,scientific_acceptance=False)
    write(receipt_path,receipt,new=True)
    paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json']]+[CURRENT]
    master,program,jobs,current=[read(v) for v in paths]
    assert program['new_training_updates']==179667 and program['active_scientific_jobs']==0
    before=HERE/'metadata_before_collection_readback';before.mkdir(exist_ok=False)
    for path in paths:shutil.copy2(path,before/path.name)
    refs=copy.deepcopy(program['wp3_016_execution'])
    refs.update(status='completed_native0_selectively_collected_independent_readback_pending',
        collection_complete=True,collection=ref(collection_path),selected_collection_readback=ref(receipt_path),
        full_inventory=ref(OUT/'collection_01/REMOTE_FULL_INVENTORY.json'),D_native_completion_backup=ref(backup_path))
    for value in [master,program,current]:
        value.update(status='WP3_016_collected_independent_numeric_QA_pending',wp3_016_execution=refs,wp3_execution=refs,
            wp3_016_collection_complete=True,updated_UTC=stamp)
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015')
    job.update(status=refs['status'],execution=refs,collection=ref(collection_path))
    next(w for w in master['work_packages'] if w['id']=='WP3').update(status='016_collected_independent_numeric_QA_pending',execution=refs,execution016=refs)
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0])
    for path,value in [(paths[1],program),(paths[2],jobs),(CURRENT,current)]:write(path,value)
    print(json.dumps(dict(status=receipt['status'],selected_files=99,selected_bytes=receipt['selected_bytes'],
        D_files=len(inventory),D_bytes=sum(r['bytes'] for r in inventory.values()),cumulative_G=179667)))


if __name__=='__main__':main()
