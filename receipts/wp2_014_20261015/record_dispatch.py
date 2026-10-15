"""Record dispatch separately from the required proof of a correct first update."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

r=Path('E:/GAN/research/N01');o=r/'receipts/wp2_014_20261015'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
start=o/'start_native_01/START_RECEIPT.json';d=read(start);stamp=dt.datetime.now(dt.timezone.utc).isoformat()
paths=[r/'PROGRAM.json',r/'MASTER_EXECUTION_PLAN.json',r/'jobs.json',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
snapshot=o/'metadata_before_dispatch';snapshot.mkdir()
for p in paths:shutil.copy2(p,snapshot/p.name)
record=dict(UTC=stamp,status='native_initializing_verification_pending',start_receipt=str(start.relative_to(r)),
    start_receipt_sha256=sha(start),supervisor_pid=d['supervisor_pid'],
    actual_G1_verified=False,current_computation_counts='unknown_until_saved_progress_readback',
    training159667_is_last_pre014_verified_count=True)
assert read(r/'jobs.json')['active_scientific_jobs']==0
for p in paths:
    item=read(p);item.update(updated_UTC=stamp,active_scientific_jobs=1)
    if p.name=='jobs.json':
        j=next(j for j in item['jobs'] if j['job_id']=='N01_WP2_014_20261015');j.update(status=record['status'],dispatch=record,supervisor_pid=d['supervisor_pid'])
    else:
        item.update(status='WP2_014_native_initializing_verification_pending',wp2_dispatch=record)
        if p.name=='MASTER_EXECUTION_PLAN.json':next(w for w in item['work_packages'] if w['id']=='WP2')['status']='native_initializing_verification_pending'
    write(p,item)
program=read(r/'PROGRAM.json');program['master_plan_sha256']=sha(r/'MASTER_EXECUTION_PLAN.json');write(r/'PROGRAM.json',program)
write(o/'DISPATCH_READBACK.json',record)
print(json.dumps(dict(status=record['status'],supervisor_pid=d['supervisor_pid'],actual_G1_verified=False)))
