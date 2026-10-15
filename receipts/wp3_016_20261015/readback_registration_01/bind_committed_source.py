"""Bind already-read-back CPU source commit into current root metadata."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

config=read(OUT/'READBACK_EXECUTION.json')
source=read(OUT/'SOURCE_COMMIT_READBACK.json')
backup=read(OUT/'D_SOURCE_BACKUP_RECEIPT.json')
assert source['raw_Git_blobs_byte_exact'] and source['source_controls']==6
assert backup['source_commit']==source['source_commit']
assert source['configuration_sha256']==sha(OUT/'READBACK_EXECUTION.json')
for name,row in source['files'].items():
    assert sha(ROOT/name)==row['sha256'] and (ROOT/name).stat().st_size==row['bytes']
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
refs={}
for key,name in [('source_commit_readback','SOURCE_COMMIT_READBACK.json'),('D_source_backup','D_SOURCE_BACKUP_RECEIPT.json')]:
    path=OUT/name
    refs[key]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)
paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json']]+[CURRENT]
for path in paths:
    value=read(path)
    assert value.get('active_scientific_jobs')==0
    if path.name=='jobs.json':
        row=next(j for j in value['jobs'] if j.get('run_id')==config['run_id'])
        execution=row['execution']
        assert row['status']=='registered_not_started'
        execution.update(source_commit=source['source_commit'],source_commit_pending=False,**refs)
    else:
        assert value['new_training_updates']==179667
        for key in ['wp3_readback_execution','wp3_016_readback_execution']:
            value[key].update(source_commit=source['source_commit'],source_commit_pending=False,**refs)
        if path.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            for key in ['saved_array_readback','readback016']:
                wp[key].update(source_commit=source['source_commit'],source_commit_pending=False,**refs)
    value['updated_UTC']=stamp
    write(path,value)
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for path in [ROOT/'PROGRAM.json',CURRENT]:
    value=read(path);value['master_plan_sha256']=master_sha;write(path,value)
write(OUT/'SOURCE_BINDING_RECEIPT.json',dict(UTC=stamp,source_commit=source['source_commit'],
    configuration_sha256=source['configuration_sha256'],raw_control_blobs_verified=6,
    cumulative_training_updates=179667,active_scientific_jobs=0,remote_action_performed=False,**refs),True)
print(json.dumps(dict(status='source_commit_bound_in_metadata',source_commit=source['source_commit'],cumulative_training_updates=179667)))
