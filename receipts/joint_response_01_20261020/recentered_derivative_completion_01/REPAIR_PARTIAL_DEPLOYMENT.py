from pathlib import Path
import ast
import json
import subprocess

HERE = Path(__file__).resolve().parent
manifest_raw = (HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8')
additions = json.loads((HERE/'LOCAL_ADDITIONS.json').read_bytes())
payload = dict(manifest_raw=manifest_raw,
    additions={name: Path(path).read_text(encoding='utf-8') for name, path in additions.items()})
remote_script = """
from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json,os,sys,time
started=time.monotonic()
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_20261020')
manifest=json.loads(payload['manifest_raw'])
prior=Path(manifest['prior_NAS_root'])
assert not Path('/proc/215696').exists()
assert not any((root/name).exists() for name in ('run_01','control_01','DISPATCH_RECEIPT.json'))
root.mkdir(parents=True,exist_ok=True)
reused=[];copied=[]
for name,entry in manifest['files'].items():
    target=root/name
    if target.is_file():
        existing=target.read_bytes()
        if len(existing)==entry['bytes'] and hashlib.sha256(existing).hexdigest()==entry['sha256']:
            reused.append(name);continue
    raw=payload['additions'][name].encode('utf-8') if name in payload['additions'] else (prior/name).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);copied.append(name)
(root/'SOURCE_MANIFEST.json').write_bytes(payload['manifest_raw'].encode('utf-8'))
for name,entry in manifest['files'].items():
    raw=(root/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
    if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
assert len(manifest['files'])==190
for name,entry in manifest['external_prefix_files'].items():
    raw=Path(name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
cache=root/'cache';cache.mkdir()
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),CUDA_CACHE_PATH=str(cache/'cuda'))
sys.path[:0]=[str(root/'src'),str(root/'scripts')]
import torch,numpy,scipy
import complete_joint01_recentered_derivatives,supervise_joint01_nas
assert complete_joint01_recentered_derivatives.ROOT==root
assert torch.cuda.is_available()
record=dict(status='PASS_NAS_DEPLOYMENT',UTC=datetime.now(timezone.utc).isoformat(),source_files=190,source_pins=189,external_prefix_files_verified=12,config_sha256=manifest['config_sha256'],elapsed_seconds=time.monotonic()-started,remote_root=str(root),torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,native_driver_and_supervisor_imported=True,new_scientific_calls=0,new_local_or_compute_disk_payload_copies=False,reused_already_valid_files=len(reused),copied_missing_or_partial_files=len(copied),per_source_file_fsync=False)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(record))
"""
ast.parse(remote_script)
request = ('payload='+repr(payload)+'\n'+remote_script).encode('utf-8')
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
    'rylan@100.82.231.102', '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
    input=request, capture_output=True, timeout=180)
(HERE/'DEPLOY_STDERR.txt').write_bytes(r.stderr)
(HERE/'DEPLOY_STDOUT.txt').write_bytes(r.stdout)
r.check_returncode()
record = json.loads(r.stdout)
(HERE/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
