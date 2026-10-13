"""Read one live remote snapshot, retain it on E, and reconcile job accounting."""
from pathlib import Path
import datetime as dt
import json
import os
import shlex
import subprocess
import hashlib

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    os.replace(temporary, path)

def main():
    remote_code = '''from pathlib import Path
import datetime as dt,json,subprocess,hashlib
r=Path(REMOTE_ROOT)
names=['REMOTE_START.json','PREFLIGHT.json','science/BINDING.json','science/RESUME_VERIFICATION.json','science/status.json','science/COMPLETE.json','science/STOP.json','supervisor/launch.json','supervisor/resource_status.json','supervisor/exit_receipt.json']
files={n:json.loads((r/'runs'/n).read_text()) for n in names if (r/'runs'/n).is_file()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
candidate_pids=set(files.get('supervisor/resource_status.json',{}).get('child_tree',[]))
candidate_pids.add(files['REMOTE_START.json']['supervisor_pid'])
if files.get('science/status.json',{}).get('pid'):candidate_pids.add(files['science/status.json']['pid'])
processes=[]
for pid in sorted(candidate_pids):
 try:argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().strip('\\0').split('\\0')
 except FileNotFoundError:continue
 scientific=('-m' in argv and argv[argv.index('-m')+1]=='n01.train_response007_deadline01' and str(r/'runs/science') in argv)
 supervisor=('scripts/supervise.py' in argv and str(r/'runs/supervisor') in argv)
 if scientific or supervisor:processes.append(dict(pid=pid,argv=argv))
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files=files,GPU=gpu,GPU_compute_processes=apps,owned_processes=processes)))
'''.replace('REMOTE_ROOT', repr(REMOTE))
    command = ' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(remote_code)])
    raw = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','rylan@100.82.231.102',command], text=True, encoding='utf8')
    snap = json.loads(raw)
    stamp = dt.datetime.fromisoformat(snap['UTC']).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'runs/response_007_20261013/deadline01/live'
    path = out/(stamp+'.json')
    write(path, snap)
    files = snap['files']
    status = files.get('science/status.json', {})
    native = files.get('supervisor/exit_receipt.json')
    complete = files.get('science/COMPLETE.json')
    active = bool(snap['owned_processes'])
    state = ('completed_bounded_TRAIN_stage_diagnostics' if complete and native and native.get('exit_code') == 0
             else 'bounded_training_finished_waiting_native_receipt' if complete
             else 'bounded_run_failed_or_stopped' if native or files.get('science/STOP.json')
             else 'response007_deadline01_running' if active else 'remote_state_requires_inspection')
    g = status.get('total_G_updates', status.get('G_updates', 1))
    d = status.get('total_D_updates', status.get('D_updates', 0))
    if isinstance(d, dict):
        d = sum(d.values())
    source_commit = subprocess.check_output(['git','rev-parse','f27521a'], cwd=ROOT, text=True).strip()
    jobs_path = ROOT/'jobs.json'
    jobs = json.loads(jobs_path.read_text(encoding='utf8'))
    job = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_RESPONSE007_DEADLINE01_20261013')
    job.update(status=state, source_commit=source_commit, started_UTC=files['REMOTE_START.json']['UTC'],
               pid=status.get('pid') if active else None, supervisor_pid=files['REMOTE_START.json']['supervisor_pid'] if active else None,
               historical_training_pid=status.get('pid',job.get('historical_training_pid')),
               historical_supervisor_pid=files['REMOTE_START.json']['supervisor_pid'],
               observed_cumulative_G_updates=g, observed_new_execution_G_updates=max(0,g-1),
               observed_D_updates=d, C2ST_updates=0, noise_updates=0,
               last_verified_UTC=snap['UTC'], live_snapshot=path.relative_to(ROOT).as_posix())
    write(jobs_path,jobs)
    for state_path in [ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']:
        p = json.loads(state_path.read_text(encoding='utf8'))
        p.update(status=state, active_scientific_jobs=int(active), response007_G_updates=g,
                 response007_D_updates=d,response007_C2ST_updates=0,response007_noise_updates=0,
                 response007_training_pid=status.get('pid') if active else None,
                 response007_supervisor_pid=files['REMOTE_START.json']['supervisor_pid'] if active else None,
                 response007_source_commit=source_commit,last_verified_UTC=snap['UTC'],updated_UTC=snap['UTC'],
                 scientific_status='response_match_not_established')
        if state_path.name=='PROGRAM.json':
            p.update(new_training_updates=92000+g,current_arm=status.get('arm'),
                     current_stage=status.get('stage',status.get('status')))
        write(state_path,p)
    receipts = ROOT/'receipts/response_007_20261013/deadline01'
    if not (receipts/'START_RECEIPT.json').exists():
        write(receipts/'START_RECEIPT.json',dict(UTC=snap['UTC'],source_commit=source_commit,
            remote_start=files['REMOTE_START.json'],preflight=files['PREFLIGHT.json'],
            resume=files.get('science/RESUME_VERIFICATION.json'),observed_G_updates=g,
            first_snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_snapshot=path.relative_to(ROOT).as_posix()))
    print(json.dumps(dict(UTC=snap['UTC'],status=state,arm=status.get('arm'),stage=status.get('stage'),
        G_updates=g,D_updates=d,weighted_total=status.get('weighted_total'),update_seconds=status.get('update_seconds'),
        coverage=status.get('coverage'),elapsed_seconds=status.get('elapsed_seconds'),GPU=snap['GPU'],
        resource=files.get('supervisor/resource_status.json'),native=native),ensure_ascii=False))

if __name__=='__main__':
    main()
