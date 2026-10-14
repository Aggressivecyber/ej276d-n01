"""Supervise one owned, bounded 012 NN/AD diagnostic; no scientific imports."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mechanism012_ops import BUDGET,RESOURCES,REMOTE,POLICY,BINDINGS,validate_policy,operation_progress,boot_id,now

def atomic_json(path,value):
    p=Path(path);temporary=p.with_name(p.name+'.tmp')
    with temporary.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(p)

def process_tree(pid):
    rows=subprocess.check_output(['ps','-eo','pid,ppid,rss'],text=True,timeout=10).splitlines()[1:]
    entries=[tuple(map(int,r.split())) for r in rows if len(r.split())==3];owned={pid}
    while True:
        children={p for p,parent,_ in entries if parent in owned}
        if children<=owned:break
        owned|=children
    return owned,sum(rss*1024 for p,_,rss in entries if p in owned)

def gpu_bytes(pids):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    return sum(int(r.split(',')[1].strip())*1024**2 for r in rows.splitlines() if int(r.split(',')[0].strip()) in pids)

def output_bytes(root):
    size=0
    for directory,folders,files in os.walk(root,followlinks=False):
        if any((Path(directory)/n).is_symlink() for n in folders+files):raise ValueError('Redirected owned output path')
        for name in files:
            try:size+=(Path(directory)/name).stat().st_size
            except FileNotFoundError:pass
    return size

def limit_reason(limits,rss,gpu,size,elapsed,interrupted=None):
    for failed,name in ((rss>limits['job_tree_RSS_GiB']*1024**3,'RSS_limit'),(gpu>limits['GPU_GiB']*1024**3,'GPU_limit'),
        (size>limits['output_MiB']*1024**2,'output_limit'),(elapsed>limits['wall_seconds'],'wall_limit'),(interrupted is not None,'user_signal')):
        if failed:return name
    return None

def stop_owned(child,grace):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True);parser.add_argument('--bindings',required=True)
    parser.add_argument('--job-dir',required=True);parser.add_argument('--lock-fd',required=True,type=int);args=parser.parse_args()
    if os.name!='posix':raise ValueError('Native012 supervisor is Linux-only')
    import fcntl
    os.fstat(args.lock_fd);fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pp,bp,job=[Path(v).resolve() for v in (args.policy,args.bindings,args.job_dir)]
    p=json.loads(pp.read_text(encoding='utf8'));validate_policy(p);root=Path(REMOTE)
    if pp!=root/POLICY or bp!=root/BINDINGS or job!=root/'runs/primary' or os.path.lexists(job):raise ValueError('Only fresh registered012 job is allowed')
    job.mkdir(parents=True,exist_ok=False);supervisor=job/'supervisor';supervisor.mkdir();output=job/'artifacts'
    command=[sys.executable,'-B',str(root/'scripts/run_mechanism012.py'),'--policy',str(pp),'--bindings',str(bp),
        '--output',str(output),'--device','cuda:0']
    started,clock,started_boot=now(),time.monotonic(),boot_id();interrupted={'signal':None}
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda n,frame:interrupted.update(signal=n))
    child,reason,error,code=None,None,None,1;peak_rss=peak_gpu=peak_size=0
    try:
        with (job/'console.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(supervisor/'launch.json',dict(schema='n01/mechanism012-launch/v1',run_id=p['run_id'],boot_id=started_boot,
                supervisor_pid=os.getpid(),child_pid=child.pid,command=command,started_UTC=started,resource_limits=RESOURCES,
                registered_budgets=BUDGET,automatic_restart=False))
            while child.poll() is None:
                pids,rss=process_tree(os.getpid());gpu,size,elapsed=gpu_bytes(pids),output_bytes(job),time.monotonic()-clock
                peak_rss,peak_gpu,peak_size=max(peak_rss,rss),max(peak_gpu,gpu),max(peak_size,size)
                reason=limit_reason(RESOURCES,rss,gpu,size,elapsed,interrupted['signal'])
                atomic_json(supervisor/'resource_status.json',dict(UTC=now(),boot_id=started_boot,supervisor_pid=os.getpid(),child_pid=child.pid,
                    owned_process_tree=sorted(pids),RSS_bytes=rss,GPU_bytes=gpu,output_bytes=size,elapsed_seconds=elapsed,
                    peak_sampled_RSS_bytes=peak_rss,peak_sampled_GPU_bytes=peak_gpu,peak_output_bytes=peak_size,
                    resource_limits=RESOURCES,**operation_progress(output)))
                if reason:stop_owned(child,RESOURCES['TERM_grace_seconds']);break
                time.sleep(RESOURCES['poll_seconds'])
            code=child.wait();size,elapsed=output_bytes(job),time.monotonic()-clock;peak_size=max(peak_size,size)
            reason=reason or limit_reason(RESOURCES,0,0,size,elapsed,interrupted['signal'])
    except BaseException as exc:
        reason,error=reason or 'supervisor_error',repr(exc)
        if child is not None:stop_owned(child,RESOURCES['TERM_grace_seconds']);code=child.returncode if child.returncode is not None else 1
    try:progress=operation_progress(output)
    except BaseException as exc:progress=dict(progress_read_error=repr(exc),actual_inflight_completion='unknown')
    receipt=dict(schema='n01/mechanism012-supervisor-exit/v1',run_id=p['run_id'],boot_id=started_boot,started_UTC=started,finished_UTC=now(),
        supervisor_pid=os.getpid(),child_pid=None if child is None else child.pid,command=command,
        child_native_exit_code=None if child is None else code,exit_code=0 if code==0 and reason is None else 1,reason=reason,error=error,
        elapsed_seconds=time.monotonic()-clock,peak_sampled_RSS_bytes=peak_rss,peak_sampled_GPU_bytes=peak_gpu,peak_output_bytes=peak_size,
        registered_budgets=BUDGET,automatic_restart=False,native_zero_is_scientific_acceptance=False,**progress)
    atomic_json(supervisor/'exit_receipt.json',receipt);os.close(args.lock_fd);return receipt['exit_code']

if __name__=='__main__':raise SystemExit(main())
