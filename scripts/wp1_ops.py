"""Shared WP1 stdlib gate/journal and thin native operations.

Uses proven 012 archive and owned-process supervision primitives. Configuration
and input/source freezes are common to all WP1 phases; no scientific imports.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1] if not str(__file__).startswith('<') else Path.cwd()
CFG='wp1_013_config.json';MANIFEST='WP1_SOURCE_MANIFEST.json'
REMOTE='/home/rylan/projects/ej276d-n01/wp1_013_20261015'
HOST='rylan@100.82.231.102';PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
NEW_FILES=('scripts/wp1_ops.py','scripts/run_wp1_013.py','scripts/wp1_evaluation.py','scripts/wp1_parameter_probe.py',
    'scripts/wp1_input_information.py','scripts/wp1_boundary_saved.py','USER_MASTER_PLAN_20261015.txt',
    'OBSERVATION_CONTRACT.json','DATA_ROLES.json','docs/ACCEPTANCE_CONTRACT.md')

def require(value,message):
    if not value:raise ValueError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text('utf8'))
def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not (-float('inf')<value<float('inf')):return None
    return value
def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(clean(value),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic_json(path,value):
    p=Path(path);t=p.with_name(p.name+'.tmp');require(not t.exists(),'Preserve orphan temporary')
    write_new(t,value);t.replace(p)

def safe_path(path):
    p=Path(path).absolute();require(p.resolve()==p and all(not q.is_symlink() for q in (p,*p.parents)),'Redirected/noncanonical path')
    return p
def fresh(path):
    p=safe_path(path);require(os.name!='nt' or p.drive.upper() in ('E:','D:'),'All local output stays E/D')
    require(not os.path.lexists(p),'Preserve existing output');p.mkdir(parents=True,exist_ok=False);return p

def gate(config_path,expected_sha):
    cp=safe_path(config_path);require(sha(cp)==expected_sha,'Common config SHA changed');p=read_json(cp)
    require(p.get('schema')=='n01/wp1-013-config/v1' and p.get('run_id')=='wp1_013_20261015'
        and p.get('status')=='frozen' and p.get('remote_root')==REMOTE and p.get('scientific_training_updates')==0
        and p.get('automatic_restart') is False,'Common WP1 frozen zero-training configuration required')
    require(p.get('source_files') and 'wp1_013_config.json' not in p['source_files'],'No sourceSHA/config recursion')
    entries={**p['source_files'],**{'input:'+k:v for k,v in p['inputs'].items()},**{'contract:'+k:v for k,v in p['contracts'].items()},
        **{'protected:'+k:v for k,v in p.get('protection_files',{}).items()}}
    checked={str(cp):expected_sha};paths={}
    roots=[REMOTE,'/home/rylan/projects/ej276d-n01/window008_resume01_20261013','/home/rylan/projects/ej276d-n01/shape010_20261014',
        '/home/rylan/projects/ej276d-n01/mechanism012_20261015','/home/rylan/projects/ej276d-n01/stability011_20261014']
    for name,row in entries.items():
        q=safe_path(row['path']);key=str(q)
        require(os.name!='posix' or any(q.is_relative_to(Path(r)) for r in roots),'WP1 read escaped registered roots')
        require(len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']),'Bad frozen SHA')
        if key in checked:require(checked[key]==row['sha256'],'Conflicting alias SHA')
        else:require(sha(q)==row['sha256'],'WP1 frozen bytes changed: '+name);checked[key]=row['sha256']
        if 'bytes' in row:require(q.stat().st_size==row['bytes'],'Frozen byte size changed')
        if name.startswith('input:'):paths[name[6:]]=q
    require(set(NEW_FILES)<={n for n in p['source_files']},'Owned WP1/source contracts missing')
    old=read_json(paths['shape010_manifest']);binding=read_json(paths['shape010_bindings'])
    require(len(old['files'])==53 and binding['source_manifest']['sha256']==p['inputs']['shape010_manifest']['sha256']
        and old['bound_inputs']=={k:r['sha256'] for k,r in binding['files'].items()},'Original53/seven-input chain mismatch')
    for rel,d in old['files'].items():require(p['source_files'].get(rel,{}).get('sha256')==d,'Numerical source copy changed')
    return p,paths,checked

def unchanged(checked):
    for path,digest in checked.items():require(sha(safe_path(path))==digest,'Immutable byte changed during WP1: '+path)

class Store:
    """Generic per-operation caps and durable success, independent of 012 budgets."""
    def __init__(self,output,config,checked,artificial=False):
        self.root=fresh(output);self.config=config;self.checked=checked;self.artificial=artificial
        self.counts={k:0 for k in config['budget_caps']};self.attempted=self.counts.copy();self.inflight={};self.entered=self.completed=0
        self.started=time.monotonic();self.phase='frozen_inputs';self.error=None
        (self.root/'operations.jsonl').open('x').close()
    def base(self):return dict(schema='n01/wp1-state/v1',run_id=self.config['run_id'],UTC=now(),pid=os.getpid(),artificial_only=self.artificial,
        phase=self.phase,counters=self.counts.copy(),attempted_counters=self.attempted.copy(),operations_entered=self.entered,
        operations_completed=self.completed,inflight=self.inflight.copy(),unknown_inflight_completion='never_asserted_completed',
        immutable_files_sha256=self.checked,error=self.error,scientific_training_updates=0,scientific_status='response_match_not_established')
    def json(self,name,value):write_new(self.root/name,value)
    def status(self,extra=None):atomic_json(self.root/'status.json',{**self.base(),**(extra or {})})
    def npz(self,name,**arrays):
        import numpy as np
        with (self.root/name).open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    def journal(self,row):
        with (self.root/'operations.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(clean(row),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def call(self,name,charges,function,**detail):
        require(time.monotonic()-self.started<=self.config['resource_limits']['wall_seconds'],'WP1 wall cap exceeded')
        for k,v in charges.items():require(k in self.attempted and type(v) is int and 0<=v<=self.config['budget_caps'][k]-self.attempted[k],'WP1 budget exceeded: '+k)
        oid=self.entered;record=dict(operation_id=oid,event='entered',name=name,charges=charges,UTC=now(),details=detail)
        self.journal(record);self.entered+=1;self.inflight[str(oid)]=record
        for k,v in charges.items():self.attempted[k]+=v
        self.status()
        try:result=function()
        except BaseException as error:
            self.journal(dict(operation_id=oid,event='failed',name=name,charges=charges,error=repr(error),UTC=now()));raise
        self.journal(dict(operation_id=oid,event='completed',name=name,charges=charges,UTC=now()))
        for k,v in charges.items():self.counts[k]+=v
        self.completed+=1;self.inflight.pop(str(oid));self.status();return result

def durable_progress(output):
    p=Path(output)/'operations.jsonl';entered={};completed={};failed={};partial=False
    if p.exists():
        for line in p.read_text('utf8').splitlines(keepends=True):
            if not line.endswith('\n'):partial=True;break
            try:r=json.loads(line)
            except json.JSONDecodeError:partial=True;break
            oid=r['operation_id'];target={'entered':entered,'completed':completed,'failed':failed}[r['event']]
            require(oid not in target,'Duplicate operation event');target[oid]=r
    require(set(completed)<=set(entered) and not set(completed)&set(failed),'Invalid completion journal')
    keys={k for records in (entered,completed) for r in records.values() for k in r['charges']}
    return dict(entered_counts={k:sum(r['charges'].get(k,0) for r in entered.values()) for k in keys},
        completed_counts={k:sum(r['charges'].get(k,0) for r in completed.values()) for k in keys},
        unknown_inflight_ids=sorted(set(entered)-set(completed)),failed_ids=sorted(failed),truncated_tail=partial,
        counts_are_durable_lower_bounds=True)

def reused_supervision():
    import supervise_mechanism012 as s
    return s

def supervise(config_sha,lock_fd):
    require(os.name=='posix','Native Linux supervision only');import fcntl
    os.fstat(lock_fd);fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=safe_path(REMOTE);p,_,_=gate(root/CFG,config_sha);job=fresh(root/'runs/primary');sv=fresh(job/'supervisor');helpers=reused_supervision()
    import signal
    interrupted={'signal':None}
    for n in (signal.SIGINT,signal.SIGTERM):signal.signal(n,lambda n,frame:interrupted.update(signal=n))
    started=time.monotonic();child=None;reason=error=None;code=1;maxrss=maxgpu=maxsize=0
    cmd=[sys.executable,'-B',str(root/'scripts/run_wp1_013.py'),'--config',str(root/CFG),'--config-sha256',config_sha,'--output',str(job),'--device','cuda:0']
    try:
        with (job/'console.log').open('xb') as log:
            child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            atomic_json(sv/'launch.json',dict(run_id=p['run_id'],supervisor_pid=os.getpid(),child_pid=child.pid,command=cmd,UTC=now()))
            while child.poll() is None:
                ids,rss=helpers.process_tree(os.getpid());gpu=helpers.gpu_bytes(ids);size=helpers.output_bytes(job);elapsed=time.monotonic()-started
                maxrss,maxgpu,maxsize=max(maxrss,rss),max(maxgpu,gpu),max(maxsize,size)
                reason=helpers.limit_reason(p['resource_limits'],rss,gpu,size,elapsed,interrupted['signal'])
                atomic_json(sv/'resource_status.json',dict(UTC=now(),owned_process_tree=sorted(ids),RSS_bytes=rss,GPU_bytes=gpu,output_bytes=size,
                    elapsed_seconds=elapsed,**durable_progress(job/'artifacts')))
                if reason:helpers.stop_owned(child,p['resource_limits']['TERM_grace_seconds']);break
                time.sleep(p['resource_limits']['poll_seconds'])
            code=child.wait()
            size=helpers.output_bytes(job);elapsed=time.monotonic()-started;maxsize=max(maxsize,size)
            reason=reason or helpers.limit_reason(p['resource_limits'],0,0,size,elapsed,interrupted['signal'])
    except BaseException as exc:
        error,reason=repr(exc),reason or 'supervisor_error'
        if child is not None:helpers.stop_owned(child,p['resource_limits']['TERM_grace_seconds']);code=child.returncode
    try:progress=durable_progress(job/'artifacts')
    except BaseException as exc:progress=dict(progress_read_error=repr(exc),actual_inflight_completion='unknown')
    atomic_json(sv/'exit_receipt.json',dict(run_id=p['run_id'],child_native_exit_code=code,exit_code=0 if code==0 and reason is None else 1,
        reason=reason,error=error,supervisor_pid=os.getpid(),child_pid=None if child is None else child.pid,UTC=now(),
        peak_RSS_bytes=maxrss,peak_GPU_bytes=maxgpu,peak_output_bytes=maxsize,automatic_restart=False,
        native_zero_is_scientific_acceptance=False,**progress))
    os.close(lock_fd);return 0 if code==0 and reason is None else 1

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('supervise',));parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--lock-fd',required=True,type=int);args=parser.parse_args();return supervise(args.config_sha256,args.lock_fd)

if __name__=='__main__':raise SystemExit(main())
