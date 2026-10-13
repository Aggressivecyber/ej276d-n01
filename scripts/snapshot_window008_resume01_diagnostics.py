"""Take one evaluation startup/completion snapshot; never poll or restart a job."""
from pathlib import Path
import argparse
import datetime as dt
import json
import shlex
import subprocess
from window008_resume01_runtime import ROOT, read, write_new


def snapshot(kind):
    cfg=read(ROOT/'window008_resume01_config.json')
    code='''from pathlib import Path
import datetime as dt,json,subprocess
r=Path(ROOT_VALUE);kind=KIND_VALUE;j=r/'runs/diagnostic_jobs'/kind
names=['START_ATTEMPT.json','PREFLIGHT.json','REMOTE_START.json','WRAPPER_ATTEMPT.json','EVALUATOR_NATIVE.json','JOB_COMPLETE.json',
 'JOB_FAILED.json','LAUNCH_FAILURE.json','REPLAY_VERIFICATION.json','supervisor/launch.json','supervisor/resource_status.json',
 'supervisor/exit_receipt.json','artifacts/FROZEN_INPUTS.json','artifacts/FAILED.json','artifacts/SUMMARY.json']
files={name:json.loads((j/name).read_text()) for name in names if (j/name).is_file()}
processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 owned=('-m' in a and a[a.index('-m')+1]=='n01.diagnose_window008_resume01' and str(j/'artifacts') in a)
 owned=owned or ((str(r/'scripts/run_window008_resume01_diagnostics.py') in a or 'scripts/supervise_window008_resume01_diagnostics.py' in a) and kind in a)
 if owned:processes.append(dict(pid=int(p.parent.name),argv=a,cwd=str(p.parent.joinpath('cwd').resolve())))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),kind=kind,files=files,owned_processes=processes,GPU_compute_processes=gpu,
 artifact_files=[p.name for p in (j/'artifacts').iterdir()] if (j/'artifacts').is_dir() else [])))
'''.replace('ROOT_VALUE',repr(cfg['remote_root'])).replace('KIND_VALUE',repr(kind))
    py='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
    command=' '.join([shlex.quote(py),'-B','-c',shlex.quote(code)])
    result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','-o','ConnectTimeout=8',
        'rylan@100.82.231.102',command],text=True,encoding='utf8'))
    stamp=dt.datetime.fromisoformat(result['UTC']).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'runs'/cfg['run_id']/'live'/('diagnostics_'+stamp+'.json')
    write_new(output,result)
    files=result['files']
    return dict(UTC=result['UTC'],kind=kind,native=files.get('supervisor/exit_receipt.json'),
        core_native=files.get('EVALUATOR_NATIVE.json',{}).get('exit_code'),complete=files.get('JOB_COMPLETE.json',{}).get('status'),
        failure=files.get('JOB_FAILED.json',files.get('artifacts/FAILED.json',files.get('LAUNCH_FAILURE.json'))),
        frozen=files.get('artifacts/FROZEN_INPUTS.json',{}).get('status'),processes=result['owned_processes'],
        GPU=result['GPU_compute_processes'],artifacts=result['artifact_files'],snapshot=str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('primary',),required=True)
    print(json.dumps(snapshot(p.parse_args().kind)))
