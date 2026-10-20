from pathlib import Path
import ast
import json
import subprocess

HERE = Path(__file__).resolve().parent
manifest_raw = (HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8')
additions = json.loads((HERE/'LOCAL_ADDITIONS.json').read_bytes())
payload = dict(manifest_raw=manifest_raw, additions={name: Path(path).read_text(encoding='utf-8') for name, path in additions.items()})
remote_script = """
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json,os,signal,sys,time
started=time.monotonic();signal.alarm(150)
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020')
manifest=json.loads(payload['manifest_raw']);prior=Path(manifest['prior_NAS_root'])
print(json.dumps(dict(stage='THIN_DEPLOYMENT_STARTED',UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],maximum_seconds=150)),flush=True)
assert not any((root/name).exists() for name in ('run_01','control_01','DISPATCH_RECEIPT.json'))
root.mkdir(parents=True,exist_ok=True)
for folder in sorted({str((root/name).parent) for name in payload['additions']}):Path(folder).mkdir(parents=True,exist_ok=True)
for name,text in payload['additions'].items():
    raw=text.encode('utf-8');entry=manifest['files'][name]
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
    (root/name).write_bytes(raw)
(root/'SOURCE_MANIFEST.json').write_bytes(payload['manifest_raw'].encode('utf-8'))
print(json.dumps(dict(stage='THIN_TEXT_FILES_WRITTEN',files=7,bytes=sum(len(v.encode('utf-8')) for v in payload['additions'].values()),elapsed_seconds=time.monotonic()-started)),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    pending=[(name,entry,pool.submit((root/name).read_bytes)) for name,entry in manifest['files'].items()]
    for name,entry,future in pending:
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
    pending=[(name,entry,pool.submit(Path(name).read_bytes)) for name,entry in manifest['external_prefix_files'].items()]
    for name,entry,future in pending:
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
assert len(manifest['files'])==192 and len(manifest['external_prefix_files'])==12
print(json.dumps(dict(stage='THIN_BOUND_SOURCES_VERIFIED',source_files=192,external_prefix_files=12,elapsed_seconds=time.monotonic()-started)),flush=True)
cache=root/'cache';cache.mkdir()
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),CUDA_CACHE_PATH=str(cache/'cuda'))
sys.path[:0]=[str(root/'scripts'),str(prior/'src'),str(prior/'scripts')]
import torch,numpy,scipy
import complete_joint01_recentered_derivatives,supervise_joint01_nas
assert complete_joint01_recentered_derivatives.ROOT==root and torch.cuda.is_available()
record=dict(status='PASS_NAS_DEPLOYMENT',UTC=datetime.now(timezone.utc).isoformat(),source_files=192,source_pins=191,external_immutable_sources=185,local_text_files=7,external_prefix_files_verified=12,config_sha256=manifest['config_sha256'],elapsed_seconds=time.monotonic()-started,remote_root=str(root),immutable_source_reuse_root=str(prior),torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,native_driver_and_supervisor_imported=True,new_scientific_calls=0,new_local_or_compute_disk_payload_copies=False)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(record),flush=True)
"""
ast.parse(remote_script)
request = ('payload='+repr(payload)+'\n'+remote_script).encode('utf-8')
try:
    r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, capture_output=True, timeout=180)
except subprocess.TimeoutExpired as error:
    (HERE/'THIN_DEPLOY_TIMEOUT_STDOUT.txt').write_bytes(error.stdout or b'')
    (HERE/'THIN_DEPLOY_TIMEOUT_STDERR.txt').write_bytes(error.stderr or b'')
    print((error.stdout or b'').decode('utf-8'))
    raise
(HERE/'THIN_DEPLOY_STDERR.txt').write_bytes(r.stderr)
(HERE/'THIN_DEPLOY_STDOUT.jsonl').write_bytes(r.stdout)
r.check_returncode()
rows = [json.loads(line) for line in r.stdout.decode('utf-8').splitlines()]
record = rows[-1]
assert record['status'] == 'PASS_NAS_DEPLOYMENT'
(HERE/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
