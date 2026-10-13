"""One scheduled snapshot and truthful fix01/legacy update reconciliation."""
from pathlib import Path
import datetime as dt
import json
import os
import shlex
import subprocess
from window008_resume01_runtime import ROOT,read,sha,write_new

PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def write_state(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(temporary,path)


def main():
    cfg=read(ROOT/'window008_resume01_config.json');remote=cfg['remote_root']
    code='''from pathlib import Path
import datetime as dt,json,subprocess
r=Path(ROOT_VALUE)
names=['REMOTE_START.json','PREFLIGHT.json','science/BINDING.json','science/RESUME_VALIDATION.json','science/status.json','science/TRAINING_COMPLETE.json','science/FAILURE.json','supervisor/launch.json','supervisor/resource_status.json','supervisor/exit_receipt.json']
files={name:json.loads((r/'runs'/name).read_text()) for name in names if (r/'runs'/name).is_file()}
owned=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=path.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 science='-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008_resume01' and str(r/'runs/science') in argv
 supervisor='scripts/supervise.py' in argv and str(r/'runs/supervisor') in argv
 if science or supervisor:owned.append(dict(pid=int(path.parent.name),argv=argv,role='scientific' if science else 'supervisor'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),files=files,owned_processes=owned,GPU=gpu,GPU_compute_processes=apps)))
'''.replace('ROOT_VALUE',repr(remote))
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    snapshot=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','rylan@100.82.231.102',command],text=True,encoding='utf8'))
    stamp=dt.datetime.fromisoformat(snapshot['UTC']).strftime('%Y%m%dT%H%M%SZ')
    path=ROOT/'runs/window008_resume01_20261013/live'/(stamp+'.json');write_new(path,snapshot)
    files=snapshot['files'];status=files.get('science/status.json',{});complete=files.get('science/TRAINING_COMPLETE.json')
    failure=files.get('science/FAILURE.json');native=files.get('supervisor/exit_receipt.json');start=files.get('REMOTE_START.json')
    active=any(proc['role']=='scientific' for proc in snapshot['owned_processes'])
    accounting=failure or complete or status;g=accounting.get('new_execution_G_updates',0);d=accounting.get('new_execution_D_updates',0)
    state=('window008_resume01_failed_or_bounded_stop_preserved' if failure or (native and native.get('exit_code')!=0)
        else 'window008_resume01_complete_pending_endpoint_evaluation' if complete and native and native['exit_code']==0
        else 'window008_resume01_finished_waiting_native_receipt' if complete
        else 'window008_resume01_native_success_missing_completion' if native
        else 'window008_resume01_dispatched_pending_first_observed_update' if active and g==0
        else 'window008_resume01_running' if active
        else 'window008_resume01_process_missing_native_receipt_review_required' if g>0
        else 'window008_resume01_dispatched_pending_scientific_process' if start else 'window008_resume01_not_started')
    manifest=read(ROOT/cfg['source_manifest'])
    if start and start['source_manifest_sha256']!=sha(ROOT/cfg['source_manifest']):raise ValueError('Remote source manifest differs')
    jobs=read(ROOT/'jobs.json');job=next(row for row in jobs['jobs'] if row['job_id']=='N01_WINDOW008_RESUME01_20261013')
    job.update(status=state,observed_new_G_updates=g,observed_new_D_updates=d,source_commit=manifest['source_commit'],
        last_verified_UTC=snapshot['UTC'],live_snapshot=path.relative_to(ROOT).as_posix(),
        pid=status.get('pid',files.get('science/BINDING.json',{}).get('pid')) if active else None,
        supervisor_pid=start['supervisor_pid'] if start and active else None,
        effective_comparison_G_updates=accounting.get('effective_comparison_G_updates'),
        effective_comparison_D_updates=accounting.get('effective_comparison_D_updates'),
        historical_actual_G_updates=accounting.get('historical_actual_G_updates'),historical_actual_D_updates=accounting.get('historical_actual_D_updates'))
    if start:job.update(started_UTC=start['UTC'],historical_supervisor_pid=start['supervisor_pid'])
    if native:job['native_exit_code']=native['exit_code']
    write_state(ROOT/'jobs.json',jobs)
    for state_path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(state_path)
        value.update(status=state,latest_run_id=cfg['run_id'],active_research=cfg['run_id'],
            active_scientific_jobs=int(active and not (failure or complete or native)),
            window008_resume01_new_G_updates=g,window008_resume01_new_D_updates=d,
            window008_resume01_training_pid=job['pid'],window008_resume01_supervisor_pid=job['supervisor_pid'],
            window008_resume01_source_commit=manifest['source_commit'],last_verified_UTC=snapshot['UTC'],updated_UTC=snapshot['UTC'],
            current_arm=status.get('arm'),current_stage=status.get('stage',state),
            scientific_status='response_match_not_established',
            next_action='Review preserved fix01 failure before any further execution.' if failure or (native and native.get('exit_code')!=0)
            else 'Freeze the recovered-endpoint evaluation adapter and evaluate all three bound endpoints.' if complete and native and native['exit_code']==0
            else 'Obtain and bind the independent native exit receipt before evaluation.' if complete
            else 'Investigate native success without TRAINING_COMPLETE before further execution or evaluation.' if native
            else 'Investigate the missing scientific process and native receipt before any restart.' if not active and g>0
            else 'Verify correct startup once, then return at the scheduled estimated-completion follow-up.')
        if state_path.name=='PROGRAM.json':
            value.update(new_training_updates=125667+g,current_run_training_updates=g)
        if native:value['window008_resume01_native_exit_code']=native['exit_code']
        write_state(state_path,value)
    receipt=ROOT/'receipts/window008_resume01_20261013/START_RECEIPT.json'
    if start and g>0 and not receipt.exists():
        write_new(receipt,dict(UTC=snapshot['UTC'],remote_start=start,preflight=files.get('PREFLIGHT.json'),
            binding=files.get('science/BINDING.json'),resume=files.get('science/RESUME_VALIDATION.json'),
            first_observed_new_G_updates=g,first_observed_new_D_updates=d,
            first_observed_snapshot=path.relative_to(ROOT).as_posix(),snapshot_sha256=sha(path)))
    print(json.dumps(dict(UTC=snapshot['UTC'],status=state,arm=status.get('arm'),stage=status.get('stage'),
        new_G_updates=g,new_D_updates=d,arm_G_updates=status.get('G_updates'),update_seconds=status.get('update_seconds'),
        elapsed_seconds=accounting.get('elapsed_seconds'),GPU=snapshot['GPU'],native=native,failure=failure,
        resume_validated=files.get('science/RESUME_VALIDATION.json',{}).get('status'))))


if __name__=='__main__':main()
