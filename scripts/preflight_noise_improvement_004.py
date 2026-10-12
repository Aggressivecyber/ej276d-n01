"""Remote read-only preflight. No waveform decoding or parameter fitting."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as s:
        for b in iter(lambda:s.read(1024**2),b''):h.update(b)
    return h.hexdigest()
container='/home/rylan/projects/ej276d-sim2real-two-papers-20261008-0305/R33_REMOTE_20261009_175239_UTC/inputs/TRAIN_PRIVATE.npz'
root=Path('/home/rylan/projects/ej276d-n01')
active=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cwd=(p/'cwd').resolve(strict=True)
        comm=(p/'comm').read_text().strip()
        if root in cwd.parents and comm.startswith('python'):
            active.append(dict(pid=int(p.name),cwd=str(cwd),executable=comm))
    except (OSError,PermissionError):pass
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),GPU=gpu,GPU_compute_processes=compute,
    active_N01_python=active,free_disk_bytes=shutil.disk_usage(root).free,torch=str(torch.__version__),cuda=torch.version.cuda,
    train_container_sha256=sha(container),cuda_available=torch.cuda.is_available(),scientific_arrays_decoded=False)
assert not active,'An N01 process is already active'
assert not compute,'GPU already has a compute process'
assert result['free_disk_bytes']>2*1024**3 and result['cuda_available']
assert result['train_container_sha256']=='9e2f2c3c18470f4b2f3374a39590a9c27b18222bd1b22b5d7467e561409ad854'
print(json.dumps(result))
