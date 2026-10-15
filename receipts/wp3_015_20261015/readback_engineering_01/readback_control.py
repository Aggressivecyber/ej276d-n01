"""Small CPU saved-array sidecar controller; original015 tree remains read-only.

prepare/start/snapshot/collect are explicit actions. No action retries, trains,
loads checkpoint tensors, or transfers the original raw NPZ population.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ORIGINAL='/home/rylan/projects/ej276d-n01/wp3_015_20261015'
SIDE='/home/rylan/projects/ej276d-n01/wp3_015_saved_readback_20261015'
RUN_ID='wp3_015_saved_readback_20261015'
HOST='rylan@100.82.231.102'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
CONFIG='READBACK_EXECUTION.json'
FILES=('readback_control.py','readback_saved_arrays.py','saved_array_kernels.py','READBACK_PLAN.json','BOUND_ORIGINAL_INVENTORY.json')
PLAN_SHA='32ee8e020d653491c8d97ae5e41f9aca18098824b11c6f6333b3635cc0d7dd50'
CONFIG015_SHA='54c096ad1a5e112493798e5c6c8db481cd3c79b78747200af0bbd47bccba5394'
MANIFEST015_SHA='f1255dbff0e47edb7ec20e76c7d90057201f9a56f1184b7896f0c8cb0a67825a'
LEGACY_SHA='3480825e94e954a46c2c633e5df6ddef514e2b7def1332bb2e7688e78ae041a3'
LIMITS=dict(CPU_threads=2,RSS_MiB=1536,virtual_memory_MiB=3072,wall_seconds=1800,output_MiB=64)
SOURCE_CAP=4*1024**2
COLLECT_CAP=80*1024**2
STATE_NAMES=('START.json','supervisor/launch.json','supervisor/exit_receipt.json','supervisor/resource_status.json',
    'artifacts/status.json','artifacts/FIRST_PAIR_PROOF.json','artifacts/FAILED.json','artifacts/COMPLETE.json')
_LEGACY=None


def require(test,message):
    if not test:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def legacy():
    global _LEGACY
    if _LEGACY is None:
        candidates=[Path.cwd()/'scripts/wp2_controller.py',Path(ORIGINAL)/'scripts/wp2_controller.py']
        if __file__ not in ('<stdin>','<string>'):
            candidates.extend(parent/'scripts/wp2_controller.py' for parent in Path(__file__).resolve().parents)
        p=next((p for p in candidates if p.is_file()),None)
        require(p is not None and sha(p)==LEGACY_SHA,'frozen014 stdlib controller unavailable/changed')
        spec=importlib.util.spec_from_file_location('saved_readback_reused014_stdlib',p)
        _LEGACY=importlib.util.module_from_spec(spec);spec.loader.exec_module(_LEGACY)
    return _LEGACY


def core():return legacy().core()
def read(path):return core().read_json(path)
def record(path):return legacy().named_record(path)
def new_json(path,value):core().write_new(path,value)
def progress():return legacy().journal_tools().durable_progress(Path(SIDE)/'artifacts')


def validate(p):
    require(p.get('schema')=='n01/wp3-independent-readback-execution/v1' and p.get('run_id')==RUN_ID
        and p.get('remote_root')==SIDE and p.get('original_root')==ORIGINAL,'wrong readback sidecar/identity')
    require(p.get('resource_limits')==LIMITS and set(p.get('files',{}))==set(FILES),'readback source/resource scope changed')
    require(p.get('automatic_retry') is False and p.get('original_files_read_only') is True,'readback cannot retry or write original artifacts')
    require(p['files']['READBACK_PLAN.json']['sha256']==PLAN_SHA,'root-registered finite readback plan changed')
    require(not Path(SIDE).is_relative_to(Path(ORIGINAL)),'sidecar output must be outside original run tree')
    for name,row in p['files'].items():
        require(row.get('path')==SIDE+'/'+name and legacy().valid_sha(row.get('sha256')) and type(row.get('bytes')) is int and 0<row['bytes']<=SOURCE_CAP,'invalid sidecar source record')


def gate(config_sha):
    root=core().unredirected(SIDE)
    require(sha(root/CONFIG)==config_sha,'readback registration SHA changed')
    p=read(root/CONFIG);validate(p)
    for name,row in p['files'].items():require(record(root/name)=={k:row[k] for k in ('bytes','sha256')},'sidecar source changed: '+name)
    plan=read(root/'READBACK_PLAN.json')
    require(all(v==0 for v in plan['scientific_budget'].values()),'nonzero science operation added to readback')
    finite=plan['finite_resource_budget']
    require((finite['CPU_threads'],finite['process_RSS_MiB'],finite['virtual_memory_backstop_MiB'],finite['wall_seconds'],finite['output_MiB'],finite['raw_file_SHA_passes'])==(2,1536,3072,1800,64,1),'plan finite limits changed')
    return p


def old_small_gate(p):
    """Only small controls/source. The raw161 single SHA pass (160 evaluation files decode; mean-evidence summary SHA only) belongs to worker."""
    root=core().unredirected(ORIGINAL)
    require(sha(root/'wp3_015_config.json')==CONFIG015_SHA and sha(root/'WP3_SOURCE_MANIFEST.json')==MANIFEST015_SHA,'original015 controls changed')
    manifest=read(root/'WP3_SOURCE_MANIFEST.json');require(len(manifest['files'])==105,'original105 inventory changed')
    for name,row in manifest['files'].items():require(record(root/name)==row,'original frozen source changed')
    inventory=read(Path(SIDE)/'BOUND_ORIGINAL_INVENTORY.json')
    require(inventory['whole_remote_inventory_verified'] is True and inventory['config_sha256']==CONFIG015_SHA
        and inventory['source_manifest_sha256']==MANIFEST015_SHA and len(inventory['whole_remote_inventory'])==300
        and len(inventory['omitted_remote_inventory'])==161 and all(n.endswith('.npz') for n in inventory['omitted_remote_inventory']),'original300/161 inventory binding differs')
    for name in ('runs/primary/artifacts/COMPLETE.json','runs/primary/supervisor/exit_receipt.json'):
        require(record(root/name)==inventory['whole_remote_inventory'][name],'original native completion control changed')
    reports=sorted(name[:-4]+'.json' for name in inventory['omitted_remote_inventory'] if name.endswith('_REPORT.npz'))
    require(len(reports)==72,'original saved REPORT JSON scope differs')
    for name in reports:
        require(name in inventory['whole_remote_inventory'] and record(root/name)==inventory['whole_remote_inventory'][name],
            'original saved REPORT JSON changed: '+name)
    complete=read(root/'runs/primary/artifacts/COMPLETE.json');native=read(root/'runs/primary/supervisor/exit_receipt.json')
    require(complete.get('all_four_trials_completed') is True and complete['counters']['G_updates']==8000
        and native['child_native_exit_code']==native['exit_code']==0 and native.get('reason') is None,'original015 not complete')
    return dict(original105_verified=True,original_native_complete_verified=True,saved_REPORT_JSON_verified=72,
        raw_npz_hash_passes_by_controller=0,checkpoint_reads_by_controller=0)


def process_identity(pid):
    """One read-only identity snapshot for an already-owned Linux PID."""
    require(os.name=='posix' and type(pid) is int and pid>0,'owned Linux PID required')
    base=Path('/proc')/str(pid)
    try:
        fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        start_ticks=int(fields[19]);command=(base/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        memory={r.split(':',1)[0]:int(r.split()[1])*1024 for r in (base/'status').read_text().splitlines()
            if r.startswith(('VmRSS:','VmSize:'))}
        result=dict(alive=fields[0] not in ('Z','X'),pid=pid,parent_pid=int(fields[1]),start_ticks=start_ticks,
            command=[v.decode('utf8',errors='replace') for v in command],cwd=os.readlink(base/'cwd'),
            CPU_affinity=sorted(os.sched_getaffinity(pid)),**memory)
        require(int((base/'stat').read_text().rsplit(')',1)[1].split()[19])==start_ticks,'PID changed while reading identity')
        return result
    except (OSError,ValueError,IndexError) as error:
        return dict(alive=False,pid=pid,error=str(error))


def first_valid(state,p,*,current_boot=True):
    failed=state.get('artifacts/FAILED.json');status=state.get('artifacts/status.json') or {};native=state.get('supervisor/exit_receipt.json') or {}
    require(failed is None and status.get('native_status')!='FAILED' and status.get('phase')!='failed'
        and native.get('child_native_exit_code') in (None,0) and native.get('exit_code') in (None,0),'same snapshot failed; historical pair is insufficient')
    proof=state.get('artifacts/FIRST_PAIR_PROOF.json');start=state.get('START.json');launch=state.get('supervisor/launch.json')
    require(all(isinstance(x,dict) for x in (proof,start,launch)),'first formal pair proof not yet available')
    require(proof.get('schema')=='n01/wp3-readback-first-formal-pair/v1' and proof.get('actual_saved_array_pair_completed') is True
        and proof.get('no_extra_tiny_actual_input') is True and proof.get('actual_new_CP_NN_AD')==0 and proof.get('artificial_only',False) is False,'not a formal saved-array first pair')
    require((not current_boot or start['boot_id']==state['current_boot_id']) and start['supervisor_pid']==launch['supervisor_pid']
        and launch['child_pid']==proof['pid'],'readback proof PID/boot chain differs')
    require(start['boot_id']==launch['boot_id'] and start['registration_sha256']==state['registration_sha256']
        and start['supervisor_start_ticks']==launch['supervisor_start_ticks'],'readback launch identity differs')
    require(proof['readback_script_sha256']==p['files']['readback_saved_arrays.py']['sha256']
        and proof['config_sha256']==CONFIG015_SHA and proof['source_manifest_sha256']==MANIFEST015_SHA,'readback proof source/control identities differ')
    require(proof['pure_kernel_sha256']==p['files']['saved_array_kernels.py']['sha256']
        and proof['inventory_sha256']==p['files']['BOUND_ORIGINAL_INVENTORY.json']['sha256'],'readback kernel/inventory identities differ')
    c=proof['counters']
    require(c['bank_pairs_verified']>=1 and c['hard_rows_files_decoded']>=2 and c['report_npz_files_decoded']>=2
        and c['waveform_rows_decoded']>=2*26372 and c['raw_files_SHA_checked']==161 and not proof.get('inflight')
        and c==proof['attempted_counters'] and proof['operations_entered']==proof['operations_completed'],
        'first pair entered but not completed')
    require(proof['scientific_CP_NN_AD_training']==0,'readback science counters not zero')
    require(status.get('run_id')==RUN_ID and status.get('pid')==proof['pid']
        and all(status.get('counters',{}).get(k,0)>=v for k,v in c.items()),'current status predates or differs from first pair')
    durable=state.get('durable_progress') or {}
    require(durable.get('failed_ids')==[] and durable.get('truncated_tail') is False
        and all(durable.get('completed_counts',{}).get(k,0)>=v for k,v in c.items()),'same snapshot durable progress is unhealthy or behind proof')
    if native:
        done=state.get('artifacts/COMPLETE.json') or {}
        require(native.get('child_native_exit_code')==native.get('exit_code')==0
            and native.get('reason') is None and native.get('error') is None
            and native.get('supervisor_pid')==start['supervisor_pid'] and native.get('child_pid')==proof['pid']
            and done.get('native_status')=='COMPLETE' and done.get('passed') is True and not done.get('inflight'),
            'terminal first-pair snapshot lacks native successful completion')
        require(native.get('elapsed_seconds',float('inf'))<=LIMITS['wall_seconds']
            and native.get('peak_owned_RSS_bytes',float('inf'))<=LIMITS['RSS_MiB']*1024**2
            and native.get('peak_worker_artifact_bytes',float('inf'))<=LIMITS['output_MiB']*1024**2,
            'terminal readback resource cap exceeded')
        health='native_completed_zero'
    else:
        require(start['boot_id']==state['current_boot_id'],'active first-pair proof belongs to another boot')
        live=state.get('live_processes') or {};supervisor=live.get('supervisor') or {};worker=live.get('worker') or {}
        require(supervisor.get('alive') is True and worker.get('alive') is True
            and supervisor.get('pid')==start['supervisor_pid'] and worker.get('pid')==proof['pid']
            and supervisor.get('start_ticks')==launch['supervisor_start_ticks']
            and worker.get('start_ticks')==launch['child_start_ticks'] and worker.get('parent_pid')==start['supervisor_pid'],
            'historical pair has no current owned process chain')
        command=supervisor.get('command',[])
        require(supervisor.get('cwd')==worker.get('cwd')==SIDE and SIDE+'/readback_control.py' in command
            and '--supervise' in command and '--config-sha256' in command
            and command[command.index('--config-sha256')+1]==state['registration_sha256']
            and worker.get('command')==launch['command'] and len(worker.get('CPU_affinity',[]))==2,
            'current process command/cwd/CPU identity differs')
        resource=state.get('supervisor/resource_status.json') or {}
        require(resource.get('resource_limits')==LIMITS
            and {start['supervisor_pid'],proof['pid']}<=set(resource.get('owned_process_tree',[]))
            and resource.get('elapsed_seconds',float('inf'))<=LIMITS['wall_seconds']
            and resource.get('RSS_bytes',float('inf'))<=LIMITS['RSS_MiB']*1024**2
            and resource.get('worker_artifact_bytes',float('inf'))<=LIMITS['output_MiB']*1024**2
            and supervisor.get('VmRSS',float('inf'))+worker.get('VmRSS',float('inf'))<=LIMITS['RSS_MiB']*1024**2
            and worker.get('VmSize',float('inf'))<=LIMITS['virtual_memory_MiB']*1024**2,
            'same snapshot resource health is missing or outside limits')
        health='current_owned_process_chain_and_resources_verified'
    return dict(first_formal_array_pair_verified=True,pairs_completed_at_proof=c['bank_pairs_verified'],worker_pid=proof['pid'],
        same_snapshot_health=health,current_inflight_not_counted_complete=durable.get('unknown_inflight_ids',[]),scientific_acceptance=False)


def snapshot(config_sha):
    p=gate(config_sha);root=Path(SIDE);state={}
    for name in STATE_NAMES:state[name]=read(root/name) if (root/name).is_file() else None
    try:durable=progress()
    except Exception as error:durable=dict(progress_error=repr(error),actual_inflight_completion='unknown')
    state.update(UTC=core().now(),current_boot_id=core().boot_id(),durable_progress=durable,registration_sha256=config_sha)
    launch=state.get('supervisor/launch.json') or {};start=state.get('START.json') or {}
    state['live_processes']={name:process_identity(pid) for name,pid in
        (('supervisor',start.get('supervisor_pid')),('worker',launch.get('child_pid'))) if type(pid) is int}
    try:state['first_pair_verification']=first_valid(state,p)
    except Exception as error:state['first_pair_verification']=dict(first_formal_array_pair_verified=False,reason=str(error))
    return p,state


def complete_valid(state,p):
    first_valid(state,p,current_boot=False)
    native=state.get('supervisor/exit_receipt.json') or {};done=state.get('artifacts/COMPLETE.json') or {}
    require(native.get('child_native_exit_code')==native.get('exit_code')==0 and native.get('reason') is None and native.get('error') is None,'readback native failed/resource-stopped')
    require(done.get('native_status')=='COMPLETE' and done.get('passed') is True and done.get('scientific_CP_NN_AD_training')==0
        and done.get('artificial_only',False) is False and not done.get('inflight'),'readback producer not complete')
    expected=dict(raw_files_SHA_checked=161,hard_rows_files_decoded=72,report_npz_files_decoded=72,baseline_npz_files_decoded=16,
        probe_raw_npz_files_decoded=0,probe_gradient_npz_files_decoded=0,bank_pairs_verified=36,probe_snapshots_verified=0,waveform_rows_decoded=72*26372)
    require(all(done['counters'].get(k)==v for k,v in expected.items()) and done['counters']==done['attempted_counters'],'incomplete finite saved-array scope')
    require(done['operations_entered']==done['operations_completed']==285,'readback finite operation count differs')
    raw=read(Path(SIDE)/'BOUND_ORIGINAL_INVENTORY.json')['omitted_remote_inventory']
    require(done['counters']['raw_bytes_SHA_read']==sum(row['bytes'] for row in raw.values()),'raw single-pass byte counter differs')
    durable=state['durable_progress']
    require(not durable['unknown_inflight_ids'] and not durable['failed_ids'] and not durable['truncated_tail'],'unknown readback operation remains')
    require(all(durable['completed_counts'].get(k,0)==v and durable['entered_counts'].get(k,0)==v for k,v in done['counters'].items()),'readback journal/producer count differs')
    require(sha(Path(SIDE)/'artifacts/ARRAY_READBACK.json')==done['ARRAY_READBACK_sha256'],'readback result SHA changed')
    return dict(status='bounded_saved_array_readback_complete',completed_pairs=36,scientific_CP_NN_AD_training=0,scientific_acceptance=False)


def supervise(config_sha,lock_fd):
    require(os.name=='posix','native CPU supervision only');import fcntl
    os.fstat(lock_fd);fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=gate(config_sha);root=Path(SIDE);sv=core().fresh_directory(root/'supervisor');helpers=legacy().supervisor_tools()
    clock=time.monotonic();interrupted={'signal':None};child=None;reason=error=None;code=1;peak_rss=peak_size=0
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda n,frame:interrupted.update(signal=n))
    limits=dict(job_tree_RSS_GiB=1.5,GPU_GiB=0,wall_seconds=1800,output_MiB=64)
    def child_bounds():
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(3072*1024**2,3072*1024**2))
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    command=[sys.executable,'-B',str(root/'readback_saved_arrays.py'),'--run-root',ORIGINAL,'--inventory',str(root/'BOUND_ORIGINAL_INVENTORY.json'),'--out',str(root/'artifacts')]
    try:
        with (root/'console.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_bounds)
            new_json(sv/'launch.json',dict(run_id=RUN_ID,supervisor_pid=os.getpid(),child_pid=child.pid,command=command,UTC=core().now(),
                boot_id=core().boot_id(),supervisor_start_ticks=process_identity(os.getpid())['start_ticks'],
                child_start_ticks=process_identity(child.pid)['start_ticks']))
            while child.poll() is None:
                pids,rss=helpers.process_tree(os.getpid());size=helpers.output_bytes(root/'artifacts');elapsed=time.monotonic()-clock
                peak_rss,peak_size=max(peak_rss,rss),max(peak_size,size)
                reason=helpers.limit_reason(limits,rss,0,size,elapsed,interrupted['signal'])
                try:durable=progress()
                except Exception as exc:durable=dict(progress_read_error=repr(exc),actual_inflight_completion='unknown')
                legacy().journal_tools().atomic_json(sv/'resource_status.json',dict(UTC=core().now(),owned_process_tree=sorted(pids),RSS_bytes=rss,worker_artifact_bytes=size,
                    elapsed_seconds=elapsed,resource_limits=LIMITS,durable_progress=durable))
                if reason:helpers.stop_owned(child,20);break
                time.sleep(2)
            code=child.wait();size=helpers.output_bytes(root/'artifacts');elapsed=time.monotonic()-clock;peak_size=max(peak_size,size)
            reason=reason or helpers.limit_reason(limits,0,0,size,elapsed,interrupted['signal'])
    except BaseException as exc:
        error,reason=repr(exc),reason or 'supervisor_error'
        if child is not None:helpers.stop_owned(child,20);code=child.returncode
    try:durable=progress()
    except Exception as exc:durable=dict(progress_read_error=repr(exc),actual_inflight_completion='unknown')
    receipt=dict(run_id=RUN_ID,UTC=core().now(),supervisor_pid=os.getpid(),child_pid=None if child is None else child.pid,child_native_exit_code=code,
        exit_code=0 if code==0 and reason is None else 1,reason=reason,error=error,elapsed_seconds=time.monotonic()-clock,peak_owned_RSS_bytes=peak_rss,
        peak_worker_artifact_bytes=peak_size,resource_limits=LIMITS,automatic_retry=False,**durable)
    legacy().journal_tools().atomic_json(sv/'exit_receipt.json',receipt);os.close(lock_fd)
    return receipt['exit_code']


def start(config_sha):
    require(os.name=='posix','native CPU start only');import fcntl
    p=gate(config_sha);root=Path(SIDE)
    require(not any(os.path.lexists(root/n) for n in ('START.json','DISPATCH_ATTEMPT.json','supervisor','artifacts','console.log','supervisor_console.log','tmp')),'preserve readback attempt; no restart')
    proof=old_small_gate(p);lock=core().unredirected(root.parent/'.n01_scientific_dispatch.lock')
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(len(os.sched_getaffinity(0))>=2,'CPU2 unavailable')
        available=int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:')))*1024
        disk=shutil.disk_usage(root).free
        require(available>=2*1024**3 and disk>=256*1024**2,'insufficient CPU readback RAM/disk')
        preflight=dict(available_RAM_bytes=available,free_disk_bytes=disk,CPU_affinity_count=len(os.sched_getaffinity(0)))
        new_json(root/'DISPATCH_ATTEMPT.json',dict(run_id=RUN_ID,UTC=core().now(),registration_sha256=config_sha,original_readonly_proof=proof,preflight=preflight,automatic_retry=False))
        (root/'tmp').mkdir(mode=0o700)
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'tmp'),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONPATH='')
        command=[PYTHON,'-B',str(root/'readback_control.py'),'--supervise','--config-sha256',config_sha,'--lock-fd',str(fd)]
        child=core().spawn_supervisor(command,root,fd,env)
        result=dict(run_id=RUN_ID,UTC=core().now(),supervisor_pid=child.pid,boot_id=core().boot_id(),registration_sha256=config_sha,
            supervisor_start_ticks=process_identity(child.pid)['start_ticks'],
            lock_inherited_by_supervisor=True,first_formal_pair_verified=False,automatic_retry=False,preflight=preflight,scientific_acceptance=False)
        new_json(root/'START.json',result);return result
    finally:os.close(fd)


def native(payload):
    require(os.name=='posix','native sidecar actions require Linux')
    mode=payload['mode'];config_sha=payload['config_sha256'];root=core().unredirected(SIDE)
    if mode in ('prepare','install'):
        require(re.fullmatch('[0-9a-f]{32}',payload['token']) and not os.path.lexists(root),'exclusive fresh sidecar required')
        incoming=root.parent/('.readback.incoming_'+payload['token'])
        if mode=='prepare':core().fresh_directory(incoming);return dict(incoming=str(incoming),archive=str(incoming/'readback_source.tar'))
        archive=incoming/'readback_source.tar';require(sha(archive)==payload['package_sha256'],'sidecar package SHA differs')
        inventory=legacy().index(archive,SOURCE_CAP)
        with tarfile.open(archive) as bundle:p=json.loads(bundle.extractfile(CONFIG).read())
        validate(p)
        require(inventory[CONFIG]['sha256']==config_sha and set(inventory)==set(FILES)|{CONFIG},'sidecar package controls/inventory differ')
        require(all(inventory[n]=={k:p['files'][n][k] for k in ('bytes','sha256')} for n in FILES),'sidecar package source differs')
        require(inventory['readback_control.py']['sha256']==payload['controller_sha256'],'sidecar installed controller pin differs')
        core().exclusive_extract(archive,root,inventory,SOURCE_CAP);gate(config_sha)
        result=dict(run_id=RUN_ID,status='sidecar_prepared_no_worker_started',registration_sha256=config_sha,package_sha256=payload['package_sha256'],raw_decodes=0,CP_NN_AD=0)
        new_json(root/'PREPARED.json',result);return result
    p=gate(config_sha)
    if mode=='start':return start(config_sha)
    p,state=snapshot(config_sha)
    if mode=='snapshot':return state
    require(mode=='collect','unknown sidecar action')
    native_exit=state.get('supervisor/exit_receipt.json')
    require(isinstance(native_exit,dict),'readback not terminated; snapshot only')
    try:verified=complete_valid(state,p);partial=False
    except Exception as error:verified=dict(status='partial_readback_evidence_only',reason=str(error),scientific_acceptance=False);partial=True
    inventory={}
    for q in root.rglob('*'):
        require(not q.is_symlink(),'redirected sidecar output')
        if q.is_file() and q.suffix in ('.json','.jsonl','.log','.py') and not q.is_relative_to(root/'tmp'):
            name=q.relative_to(root).as_posix();inventory[name]=record(q)
    require(sum(r['bytes'] for r in inventory.values())<=COLLECT_CAP,'sidecar small collection exceeds80MiB including controls')
    archive=root/'readback_output_01.tar'
    if payload.get('inspect'):require(archive.is_file(),'no prior sidecar archive')
    else:
        require(not os.path.lexists(archive),'preserve sidecar archive; use inspect')
        with tarfile.open(archive,'x:') as bundle:
            for n in sorted(inventory):legacy().add_regular(bundle,root/n,n)
    require(legacy().index(archive,COLLECT_CAP)==inventory,'sidecar archive bytes differ')
    return dict(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,inventory=inventory,partial_evidence_only=partial,job=verified,
        original_raw_NPZ_transferred=False,scientific_acceptance=False)


def ssh(payload,out):
    code=Path(__file__).read_bytes();require(hashlib.sha256(code).hexdigest()==payload['controller_sha256'],'readback controller changed')
    if payload['mode'] in ('prepare','install'):
        encoded=base64.urlsafe_b64encode(json.dumps(payload).encode()).decode();command=shlex.join([PYTHON,'-B','-','--bootstrap',encoded]);data=code
    else:command=shlex.join([PYTHON,'-B',SIDE+'/readback_control.py','--native']);data=json.dumps(payload).encode()
    argv=['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10',HOST,command]
    try:result=subprocess.run(argv,input=data,capture_output=True,timeout=300)
    except subprocess.TimeoutExpired as error:
        new_json(out/(payload['mode']+'_SSH_UNKNOWN.json'),dict(unknown_remote_result=True,automatic_retry=False,stdout=(error.stdout or b'').decode(errors='replace')));raise
    new_json(out/(payload['mode']+'_NATIVE.json'),dict(exit_code=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace'),automatic_retry=False))
    require(result.returncode==0,'sidecar action failed; no retry');return json.loads(result.stdout)


def action(mode,config,config_sha,out,*,collected=None,inspect=False):
    require(sha(config)==config_sha,'local registration changed');p=read(config);validate(p)
    for n,row in p['files'].items():
        local=core().unredirected(row['local_path']);require(os.name!='nt' or local.drive.upper() in ('E:','D:'),'local payload must stay E/D')
        require(record(local)=={k:row[k] for k in ('bytes','sha256')},'local sidecar source bytes changed')
    require(sha(Path(__file__))==p['files']['readback_control.py']['sha256'],'controller self identity differs')
    out=core().fresh_directory(out,local=True);payload=dict(mode=mode,config_sha256=config_sha,controller_sha256=p['files']['readback_control.py']['sha256'])
    if mode=='prepare':
        archive=out/'readback_source.tar'
        with tarfile.open(archive,'x:') as bundle:
            for n,row in p['files'].items():legacy().add_regular(bundle,row['local_path'],n)
            legacy().add_regular(bundle,config,CONFIG)
        require(sum(r['bytes'] for r in legacy().index(archive,SOURCE_CAP).values())<=SOURCE_CAP,'sidecar source cap')
        payload.update(token=uuid.uuid4().hex,package_sha256=sha(archive));incoming=ssh(payload,out)
        expected=SIDE.rsplit('/',1)[0]+'/.readback.incoming_'+payload['token']
        require(incoming['incoming']==expected and incoming['archive']==expected+'/readback_source.tar','sidecar incoming path differs')
        legacy().transfer(['scp','-q','-o','BatchMode=yes',str(archive),HOST+':'+incoming['archive']],out,'SOURCE_TRANSFER.json')
        payload['mode']='install';result=ssh(payload,out);new_json(out/'PREPARE_RECEIPT.json',result);return result
    result=ssh(dict(payload,inspect=inspect),out)
    if mode=='snapshot':
        new_json(out/'REMOTE_STATE.json',result)
        return dict(status='one_readback_snapshot_saved',first_pair_verification=result['first_pair_verification'],native_exit=result.get('supervisor/exit_receipt.json'),continuous_polling=False)
    if mode=='start':new_json(out/'START_RECEIPT.json',result);return result
    require(mode=='collect' and collected is not None,'fresh small output collection target required')
    collected=core().unredirected(collected);require(not os.path.lexists(collected),'preserve existing local readback collection')
    new_json(out/'REMOTE_ARCHIVE.json',result);archive=out/'readback_output_01.tar';part=out/'readback_output_01.tar.part'
    require(shutil.disk_usage(out).free>=2*result['archive_bytes']+32*1024**2,'insufficient E/D for small archive plus extraction')
    legacy().transfer(['scp','-q','-o','BatchMode=yes',HOST+':'+result['archive'],str(part)],out,'COLLECTION_TRANSFER.json')
    require(part.stat().st_size==result['archive_bytes'] and sha(part)==result['archive_sha256'] and legacy().index(part,COLLECT_CAP)==result['inventory'],'sidecar transferred bytes differ')
    part.replace(archive);core().exclusive_extract(archive,collected,result['inventory'],COLLECT_CAP)
    receipt=dict(status='sidecar_readback_outputs_verified',collected_root=str(collected),archive_sha256=sha(archive),files=len(result['inventory']),partial_evidence_only=result['partial_evidence_only'],job=result['job'],original_raw_NPZ_transferred=False,scientific_acceptance=False)
    new_json(out/'COLLECTION_RECEIPT.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('prepare','start','snapshot','collect'))
    parser.add_argument('--config',type=Path);parser.add_argument('--config-sha256');parser.add_argument('--out',type=Path)
    parser.add_argument('--collected',type=Path);parser.add_argument('--inspect',action='store_true');parser.add_argument('--supervise',action='store_true');parser.add_argument('--lock-fd',type=int)
    parser.add_argument('--native',action='store_true');parser.add_argument('--bootstrap');args=parser.parse_args()
    if args.supervise:
        require(legacy().valid_sha(args.config_sha256) and args.lock_fd is not None,'supervisor requires config SHA and inherited lock')
        return supervise(args.config_sha256,args.lock_fd)
    if args.native or args.bootstrap:
        payload=json.loads(base64.urlsafe_b64decode(args.bootstrap)) if args.bootstrap else json.load(sys.stdin)
        result=native(payload);print(json.dumps(result,allow_nan=False));return 0
    require(args.mode and args.config and args.out and legacy().valid_sha(args.config_sha256),'sidecar explicit mode/config/output/SHA required')
    result=action(args.mode,args.config,args.config_sha256,args.out,collected=args.collected,inspect=args.inspect)
    print(json.dumps({k:result[k] for k in ('status','supervisor_pid','first_pair_verification','partial_evidence_only','original_raw_NPZ_transferred') if k in result},allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
