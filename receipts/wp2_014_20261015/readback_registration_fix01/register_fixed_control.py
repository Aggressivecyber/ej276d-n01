"""Adopt one POSIX path comparison repair; preserve the unstarted first attempt."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path('E:/GAN/research/N01')
BASE=ROOT/'receipts/wp2_014_20261015'
OUT=BASE/'readback_registration_fix01'
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

old=BASE/'readback_registration_01/READBACK_EXECUTION.json'
cfg=read(old)
assert sha(old)=='b75deba495da00bbbd27e0e9e2da4c202fa4da7538ebb3fd8a3afa5da968ed1b'
control=BASE/'readback_engineering_01/readback_control.py'
assert sha(control)=='552eb845c8b514f8838639703d2368bf399cddf2e09d41d5c3adca09fec1ae09'
assert sha(BASE/'readback_engineering_01/before_posix_path_fix01/readback_control.py')==cfg['files']['readback_control.py']['sha256']
assert read(ROOT/'PROGRAM.json')['new_training_updates']==167667
assert read(ROOT/'PROGRAM.json')['active_scientific_jobs']==0
now=dt.datetime.now(dt.timezone.utc).isoformat()
cfg['registered_UTC']=now
cfg['files']['readback_control.py'].update(bytes=control.stat().st_size,sha256=sha(control))
cfg['engineering_repair']=dict(reason='Literal POSIX incoming-path comparison on Windows; one-line control fix only',
    prior_configuration_sha256=sha(old),prior_source_commit='dfddb7a2eef45b4e7f4253fcb89ef2ec3b608b4a',
    failed_prepare_receipt_sha256=sha(BASE/'readback_execution_01/PREPARE_PATH_FAILURE.json'),
    engineering_receipt_sha256=sha(BASE/'readback_engineering_01/posix_path_fix01/ENGINEERING_RECEIPT.json'),
    old_empty_incoming_preserved=True,prior_worker_start_called=False,prior_raw_CP_NN_AD_training=0,
    finite_scope_and_resources_unchanged=True)
write(OUT/'READBACK_EXECUTION.json',cfg,True)
reduced=lambda v:{k:x for k,x in v.items() if k not in ['registered_UTC','files','engineering_repair']}
assert reduced(cfg)==reduced(read(old))
for name,value in cfg['files'].items():
    if name!='readback_control.py':assert value==read(old)['files'][name]
ref=dict(configuration=str((OUT/'READBACK_EXECUTION.json').relative_to(ROOT)),configuration_sha256=sha(OUT/'READBACK_EXECUTION.json'),
    run_id=cfg['run_id'],remote_root=cfg['remote_root'],resource_limits=cfg['resource_limits'],source_file_count=5,
    source_controls=6,source_commit_pending=True,status='registered_not_started_after_prepare_only_path_fix',
    new_training_updates=0,new_AD_queries=0,NN_forward_rows=0,scientific_status='response_match_not_established')
paths=[ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',ROOT/'README.md',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
snapshot=OUT/'metadata_before_adoption';snapshot.mkdir()
for p in paths:shutil.copy2(p,snapshot/p.name)
for p in [p for p in paths if p.suffix=='.json']:
    d=read(p);d['updated_UTC']=now;d['active_scientific_jobs']=0
    if p.name=='jobs.json':
        job=next(x for x in d['jobs'] if x.get('run_id')==cfg['run_id']);job['previous_source_registration']=job['execution'];job['execution']=ref;job['status']=ref['status']
    else:
        d['previous_readback_source_registration']=d['wp2_readback_execution'];d['wp2_readback_execution']=ref
        d.update(status='WP2_014_readback_path_fix_registered_not_started',current_stage='WP2_014_CPU_saved_array_readback_prepared',
            next_action='Start the same bounded CPU audit using the corrected POSIX comparison; verify one formal pair, schedule one wake, and rest.')
        if p.name=='MASTER_EXECUTION_PLAN.json':
            next(x for x in d['work_packages'] if x['id']=='WP2')['saved_array_readback']=ref;d['current_action']=d['next_action']
    write(p,d)
readme=ROOT/'README.md'
text=readme.read_text(encoding='utf-8').replace('receipts/wp2_014_20261015/readback_registration_01/READBACK_EXECUTION.json','receipts/wp2_014_20261015/readback_registration_fix01/READBACK_EXECUTION.json',1)
readme.write_text(text,encoding='utf-8',newline='\n')
selector=paths[-1];d=read(selector);d['plan_sha256']=sha(readme);write(selector,d)
write(OUT/'ROOT_FIX_REGISTRATION_RECEIPT.json',dict(schema='n01/wp2-readback-control-fix-registration/v1',UTC=now,
    configuration=ref,scope_and_resources_unchanged=True,worker_unchanged=True,
    old_prepare_only_failure_preserved=True,worker_attempts_started=0,cumulative_training_updates=167667,
    new_raw_CP_NN_AD_training=0,active_scientific_jobs=0),True)
print(json.dumps(dict(status=ref['status'],configuration_sha256=ref['configuration_sha256'],control_sha256=sha(control))))
