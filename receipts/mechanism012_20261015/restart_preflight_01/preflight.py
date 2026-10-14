from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

OUT = Path('E:/GAN/research/N01/receipts/mechanism012_20261015/restart_preflight_01')
REMOTE = r'''
from datetime import datetime,timezone
from pathlib import Path
import os,hashlib,json,shutil,subprocess,fcntl
root=Path('/home/rylan/projects/ej276d-n01')
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3 and line.split()[2]=='kB'}
rows=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True,timeout=10).splitlines()
project=[row for row in rows if str(root)+'/' in row and 'python' in row]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True,timeout=10).strip()
gpu_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=10).strip()
lock_path=root/'.n01_scientific_dispatch.lock'
fd=os.open(lock_path,os.O_RDWR|os.O_NOFOLLOW)
try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock_free=True
    fcntl.flock(fd,fcntl.LOCK_UN)
except BlockingIOError:
    lock_free=False
finally:
    os.close(fd)
paths=[root/'stability011_20261014/runs/primary/artifacts/COMPLETE.json',root/'shape010_20261014/runs/primary/artifacts/final.pt',root/'shape010_20261014/runs/primary/artifacts/EVALUATION_BANK.npz']
files={str(p):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]),RAM_available_bytes=mem['MemAvailable'],disk_free_bytes=shutil.disk_usage(root).free,CPU_affinity=sorted(os.sched_getaffinity(0)),GPU=gpu,GPU_compute_apps=gpu_apps,active_project_python_processes=project,exclusive_project_lock_free=lock_free,prior_artifacts=files,new_runtime_exists=(root/'mechanism012_20261015').exists(),numeric_array_decodes=0,checkpoint_loads=0,model_forwards=0,gradient_evaluations=0,optimizer_updates=0),allow_nan=False))
'''

result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
    input=REMOTE,text=True,capture_output=True,timeout=60)
receipt=dict(schema='n01/mechanism012-restart-preflight/v1',UTC=datetime.now(timezone.utc).isoformat(),
    SSH_native_exit_code=result.returncode,stderr=result.stderr,stdout=result.stdout)
if result.returncode==0:
    receipt['native']=json.loads(result.stdout)
with (OUT/'RESTART_PREFLIGHT.json').open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(receipt,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
if result.returncode:
    raise SystemExit(result.returncode)
