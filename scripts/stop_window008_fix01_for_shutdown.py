"""Stop only the hash-bound owned trainer at the user's shutdown request."""
import json
import shlex
import subprocess
from window008_fix01_runtime import ROOT,read,write_new

PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def main():
    cfg=read(ROOT/'window008_fix01_config.json')
    code='''from pathlib import Path
import datetime as dt,json,os,signal,time,subprocess
r=Path(ROOT_VALUE);out=r/'runs';science=out/'science'
binding=json.loads((science/'BINDING.json').read_text());pid=binding['pid']
request=out/'USER_STOP_REQUEST.json'
if request.exists():raise RuntimeError('Prior stop request preserved; inspect before any repeated signal')
proc=Path('/proc')/str(pid)
before=json.loads((science/'status.json').read_text())
record=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),reason='Explicit user request: stop training before server shutdown; wait for user before resuming.',
 run_id=binding['run_id'],binding=binding,trainer_pid=pid,status_before=before,signal_sent=None)
if proc.exists():
 argv=(proc/'cmdline').read_bytes().decode().strip('\\0').split('\\0')
 if '-m' not in argv or argv[argv.index('-m')+1]!='n01.train_window008_fix01' or '--out' not in argv or argv[argv.index('--out')+1]!=str(science):raise RuntimeError('Owned trainer identity differs')
 if (proc/'cwd').resolve()!=r:raise RuntimeError('Owned trainer cwd differs')
 record['argv_before']=argv;record['proc_stat_before']=(proc/'stat').read_text()
 with request.open('x') as stream:json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 fd=os.pidfd_open(pid)
 try:signal.pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
 record['signal_sent']='SIGTERM_to_verified_owned_trainer_only'
else:
 with request.open('x') as stream:json.dump(record,stream,indent=2)
 record['already_absent_before_request']=True
deadline=time.monotonic()+25
while proc.exists() and time.monotonic()<deadline:time.sleep(.25)
if proc.exists():raise RuntimeError('Stop signal sent but trainer remains; preserve evidence and inspect, no automatic restart')
deadline=time.monotonic()+5
while not (out/'supervisor/exit_receipt.json').exists() and time.monotonic()<deadline:time.sleep(.25)
files={}
for name in ('science/FAILURE.json','supervisor/exit_receipt.json','supervisor/launch.json'):
 path=out/name
 if path.is_file():files[name]=json.loads(path.read_text())
owned=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=p.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 science_role='-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008_fix01' and str(science) in argv
 supervisor_role='scripts/supervise.py' in argv and str(out/'supervisor') in argv
 if science_role or supervisor_role:owned.append(dict(pid=int(p.parent.name),argv=argv))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
record.update(verified_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),trainer_absent=not proc.exists(),
 owned_processes_after=owned,GPU_compute_processes=apps,files=files,
 checkpoint_paths=[str(p.relative_to(r)) for p in science.rglob('*.pt')])
record['status']='user_requested_stop_verified' if not owned else 'trainer_stopped_supervisor_exit_pending'
with (out/'USER_STOP_VERIFICATION.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))
'''.replace('ROOT_VALUE',repr(cfg['remote_root']))
    command=' '.join([shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    result=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','-o','ConnectTimeout=8',
        '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','rylan@100.82.231.102',command],text=True,encoding='utf8')
    record=json.loads(result)
    write_new(ROOT/'receipts/window008_fix01_20261013/USER_STOP_VERIFICATION.json',record)
    failure=record['files'].get('science/FAILURE.json',{})
    print(json.dumps(dict(status=record['status'],trainer_absent=record['trainer_absent'],owned_processes_after=record['owned_processes_after'],
        native_exit=record['files'].get('supervisor/exit_receipt.json'),GPU_compute_processes=record['GPU_compute_processes'],
        new_G_updates=failure.get('new_execution_G_updates'),new_D_updates=failure.get('new_execution_D_updates'),
        arm=failure.get('current_arm'),arm_G_updates=failure.get('current_G_updates'),checkpoint_paths=record['checkpoint_paths'])))

if __name__=='__main__':main()
