"""Retain one read-only remote snapshot and reconcile Window008 accounting."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/rylan/projects/ej276d-n01/window008_20261013'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def main():
    code = '''from pathlib import Path
import datetime as dt,json,subprocess
r=Path(REMOTE_ROOT)
names=['REMOTE_START.json','PREFLIGHT.json','science/BINDING.json','science/TRAINING_INPUTS.json','science/status.json','science/TRAINING_COMPLETE.json','science/FAILURE.json','supervisor/launch.json','supervisor/resource_status.json','supervisor/exit_receipt.json']
files={n:json.loads((r/'runs'/n).read_text()) for n in names if (r/'runs'/n).is_file()}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
pids=set(files.get('supervisor/resource_status.json',{}).get('child_tree',[]))
start=files.get('REMOTE_START.json',{})
if start.get('supervisor_pid'):pids.add(start['supervisor_pid'])
for key in ('science/status.json','science/BINDING.json'):
 if files.get(key,{}).get('pid'):pids.add(files[key]['pid'])
processes=[]
for pid in sorted(pids):
 try:argv=Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().strip('\\0').split('\\0')
 except FileNotFoundError:continue
 scientific=('-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008' and str(r/'runs/science') in argv)
 supervisor=('scripts/supervise.py' in argv and str(r/'runs/supervisor') in argv)
 if scientific or supervisor:processes.append(dict(pid=pid,argv=argv,role='scientific' if scientific else 'supervisor'))
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files=files,GPU=gpu,GPU_compute_processes=apps,owned_processes=processes)))
'''.replace('REMOTE_ROOT', repr(REMOTE))
    command = ' '.join(['env', 'PYTHONDONTWRITEBYTECODE=1', shlex.quote(PYTHON), '-B', '-c', shlex.quote(code)])
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ForwardX11=no', 'rylan@100.82.231.102', command], text=True, encoding='utf8')
    snapshot = json.loads(raw)
    stamp = dt.datetime.fromisoformat(snapshot['UTC']).strftime('%Y%m%dT%H%M%SZ')
    path = ROOT / 'runs/window008_20261013/live' / (stamp + '.json')
    if path.exists():
        raise FileExistsError(path)
    write(path, snapshot)
    files = snapshot['files']
    start = files.get('REMOTE_START.json')
    status = files.get('science/status.json', {})
    native = files.get('supervisor/exit_receipt.json')
    complete = files.get('science/TRAINING_COMPLETE.json')
    failure = files.get('science/FAILURE.json')
    science_alive = any(p['role'] == 'scientific' for p in snapshot['owned_processes'])
    supervisor_alive = any(p['role'] == 'supervisor' for p in snapshot['owned_processes'])
    active = science_alive or supervisor_alive
    state = ('window008_failed_or_bounded_stop_preserved' if failure or (native and native.get('exit_code') != 0)
             else 'fixed_window008_training_complete_pending_evaluation' if complete and native and native.get('exit_code') == 0
             else 'window008_training_finished_waiting_native_receipt' if complete
             else 'window008_native_success_missing_completion_receipt' if native
             else 'window008_training_running' if science_alive
             else 'window008_supervisor_waiting_scientific_process' if supervisor_alive
             else 'window008_dispatched_waiting_process_receipt' if start
             else 'window008_not_started')
    accounting = failure or complete or status
    g = accounting.get('total_G_updates', accounting.get('G_updates', 0))
    d = accounting.get('total_D_updates', accounting.get('D_updates', 0))
    if isinstance(d, dict):
        d = sum(d.values())
    if (failure or complete) and status.get('total_G_updates', 0) > g:
        raise ValueError('Terminal receipt has fewer updates than the earlier status')
    manifest = json.loads((ROOT / 'WINDOW008_SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    if start and start['source_manifest_sha256'] != hashlib.sha256((ROOT / 'WINDOW008_SOURCE_MANIFEST.json').read_bytes()).hexdigest():
        raise ValueError('Remote/local training manifest mismatch')
    jobs_path = ROOT / 'jobs.json'
    jobs = json.loads(jobs_path.read_text(encoding='utf8'))
    job = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_WINDOW008_20261013')
    pid = status.get('pid', files.get('science/BINDING.json', {}).get('pid'))
    job.update(status=state, source_commit=manifest['source_commit'],
               pid=pid if active else None, supervisor_pid=start['supervisor_pid'] if active and start else None,
               observed_G_updates=g, observed_D_updates=d, C2ST_updates=0, noise_updates=0, population_updates=0,
               last_verified_UTC=snapshot['UTC'], live_snapshot=path.relative_to(ROOT).as_posix())
    if start:
        job.update(started_UTC=start['UTC'], historical_supervisor_pid=start['supervisor_pid'])
    if pid:
        job['historical_training_pid'] = pid
    if native:
        job.update(native_exit_code=native['exit_code'],
                   native_exit_receipt=path.relative_to(ROOT).as_posix()+'#files/supervisor/exit_receipt.json')
    write(jobs_path, jobs)
    for state_path in [ROOT / 'PROGRAM.json', ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json']:
        value = json.loads(state_path.read_text(encoding='utf8'))
        value.update(status=state, active_scientific_jobs=int(science_alive and not (failure or complete or native)),
                     window008_supervisor_alive=supervisor_alive, latest_run_id='window008_20261013',
                     active_research='window008_20261013', window008_G_updates=g, window008_D_updates=d,
                     window008_C2ST_updates=0, window008_noise_updates=0, window008_population_updates=0,
                     window008_training_pid=pid if active else None,
                     window008_supervisor_pid=start['supervisor_pid'] if active and start else None,
                     window008_source_commit=manifest['source_commit'],
                     last_verified_UTC=snapshot['UTC'], updated_UTC=snapshot['UTC'],
                     scientific_status='response_match_not_established')
        value['next_action'] = ('Review the preserved failure and native receipt before any amended execution.'
                                if failure or (native and native.get('exit_code') != 0)
                                else 'Freeze and execute endpoint evaluation after the successful native training receipt.'
                                if complete else 'Continue the fixed three-arm19000-G schedule; review endpoint evaluation and exact replay independently.')
        if state_path.name == 'PROGRAM.json':
            value.update(new_training_updates=100000 + g, current_run_training_updates=g,
                         current_arm=status.get('arm'), current_stage=status.get('stage', status.get('status')),
                         training_update_accounting='Prior N01 stages 100000 + Window008 G updates; D and evaluation C2ST reported separately.')
        if native:
            value['window008_native_exit_code'] = native['exit_code']
        write(state_path, value)
    receipts = ROOT / 'receipts/window008_20261013'
    # A dispatch is not yet evidence of a scientific optimizer update.
    if start and g > 0 and not (receipts / 'START_RECEIPT.json').exists():
        write(receipts / 'START_RECEIPT.json', dict(UTC=snapshot['UTC'], source_commit=manifest['source_commit'],
            remote_start=start, preflight=files.get('PREFLIGHT.json'), binding=files.get('science/BINDING.json'),
            observed_G_updates=g, observed_D_updates=d,
            first_observed_update_snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            first_observed_update_snapshot=path.relative_to(ROOT).as_posix()))
    print(json.dumps(dict(UTC=snapshot['UTC'], status=state, arm=status.get('arm'), stage=status.get('stage'),
        G_updates=g, arm_G_updates=status.get('G_updates') if status.get('arm') else None, D_updates=d,
        weighted_total=status.get('weighted_total'), update_seconds=status.get('update_seconds'),
        coverage=status.get('coverage'), elapsed_seconds=status.get('elapsed_seconds'), GPU=snapshot['GPU'],
        resource=files.get('supervisor/resource_status.json'), native=native, failure=failure), ensure_ascii=True))


if __name__ == '__main__':
    main()
