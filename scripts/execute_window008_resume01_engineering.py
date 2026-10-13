"""Run the single, short native engineering check and recover its evidence."""
from pathlib import Path
import json
import shlex
import subprocess
from window008_resume01_runtime import ROOT,read,sha,write_new,validate_inputs

PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
HOST='rylan@100.82.231.102'

def main():
    cfg,_,_=validate_inputs();remote=cfg['remote_root']
    code='''from pathlib import Path
import json,os,subprocess
r=Path(ROOT_VALUE)
if (r/'runs/science').exists():raise RuntimeError('Scientific output already exists')
for name in ('engineering_cuda_01','engineering_supervisor_01'):
 if (r/'runs'/name).exists():raise RuntimeError('Preserve prior engineering attempt: '+name)
import hashlib
if hashlib.sha256((r/'scripts/check_window008_resume01_cuda.py').read_bytes()).hexdigest()!=QA_SHA_VALUE:raise RuntimeError('Staged QA source differs')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
if apps:raise RuntimeError('GPU computation already active: '+apps)
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=path.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if '-m' in argv and argv[argv.index('-m')+1].startswith('n01.train_'):raise RuntimeError('Scientific training process already active')
for name in ('tmp','cache'): (r/name).mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(r/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',
 TMPDIR=str(r/'tmp'),XDG_CACHE_HOME=str(r/'cache'),TORCH_HOME=str(r/'cache/torch'),
 CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12')
command=[PYTHON_VALUE,'-B','scripts/supervise.py','--output',str(r/'runs/engineering_supervisor_01'),
 '--rss-limit-gib','8','--','timeout','--signal=TERM','--kill-after=15s','300s',PYTHON_VALUE,'-B',
 'scripts/check_window008_resume01_cuda.py','--out',str(r/'runs/engineering_cuda_01')]
os.chdir(r);os.execve(PYTHON_VALUE,command,env)
'''.replace('ROOT_VALUE',repr(remote)).replace('PYTHON_VALUE',repr(PYTHON)).replace('QA_SHA_VALUE',repr(sha(ROOT/'scripts/check_window008_resume01_cuda.py')))
    command=' '.join([shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ForwardX11=no',HOST,command],text=True,encoding='utf8',capture_output=True)
    rec=ROOT/'receipts/window008_resume01_20261013'
    for directory in ('engineering_cuda_01','engineering_supervisor_01'):
        destination=rec/directory
        if destination.exists():raise RuntimeError('Preserve local engineering evidence: '+str(destination))
        copied=subprocess.run(['scp','-q','-r',HOST+':'+remote+'/runs/'+directory,str(destination)],text=True,capture_output=True)
        if copied.returncode:raise RuntimeError('Native check returned '+str(result.returncode)+'; evidence retrieval failed: '+copied.stderr)
    receipt=dict(status='native_evidence_recovered',SSH_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        files={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for directory in ('engineering_cuda_01','engineering_supervisor_01')
               for path in (rec/directory).rglob('*') if path.is_file()})
    write_new(rec/'ENGINEERING_NATIVE_COLLECTION.json',receipt)
    qa=read(rec/'engineering_cuda_01/ENGINEERING_RECEIPT.json');native=read(rec/'engineering_supervisor_01/exit_receipt.json')
    print(json.dumps(dict(native=native,status=qa['status'],failure_case=qa.get('failure_case'),steps=[dict(arm=x['arm'],stage=x['stage'],seconds=x['seconds'],peak_GPU_bytes=x['peak_GPU_bytes']) for x in qa['steps']])))
    if result.returncode!=0:raise RuntimeError('Native engineering check failed; preserved all returned evidence')

if __name__=='__main__':main()
