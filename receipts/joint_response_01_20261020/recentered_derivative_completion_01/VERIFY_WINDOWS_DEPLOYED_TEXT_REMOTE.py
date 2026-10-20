
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import ast,hashlib,json,os,signal,sys,time
started=time.monotonic();signal.alarm(150)
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020')
manifest=json.loads(payload['manifest_raw']);prior=Path(manifest['prior_NAS_root'])
print(json.dumps(dict(stage='THIN_DEPLOYMENT_STARTED',UTC=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],maximum_seconds=150)),flush=True)
print(json.dumps(dict(stage='VERIFY_WINDOWS_DEPLOYED_TEXT',UTC=datetime.now(timezone.utc).isoformat())),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    pending=[(name,entry,pool.submit((root/name).read_bytes)) for name,entry in manifest['files'].items()]
    for index,(name,entry,future) in enumerate(pending,1):
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
        if index%24==0:print(json.dumps(dict(stage='SOURCE_PREFIX_VERIFIED',files=index,elapsed_seconds=time.monotonic()-started)),flush=True)
    pending=[(name,entry,pool.submit(Path(name).read_bytes)) for name,entry in manifest['external_prefix_files'].items()]
    for name,entry,future in pending:
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
assert len(manifest['files'])==192 and len(manifest['external_prefix_files'])==12
print(json.dumps(dict(stage='THIN_BOUND_SOURCES_VERIFIED',source_files=192,external_prefix_files=12,elapsed_seconds=time.monotonic()-started)),flush=True)
cache=root/'cache';cache.mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),CUDA_CACHE_PATH=str(cache/'cuda'))
sys.path[:0]=[str(root/'scripts'),str(prior/'src'),str(prior/'scripts')]
import torch,numpy,scipy
import complete_joint01_recentered_derivatives,supervise_joint01_nas
assert complete_joint01_recentered_derivatives.ROOT==root and torch.cuda.is_available()
record=dict(status='PASS_NAS_DEPLOYMENT',UTC=datetime.now(timezone.utc).isoformat(),source_files=192,source_pins=191,external_immutable_sources=185,local_text_files=7,external_prefix_files_verified=12,config_sha256=manifest['config_sha256'],elapsed_seconds=time.monotonic()-started,remote_root=str(root),immutable_source_reuse_root=str(prior),torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,native_driver_and_supervisor_imported=True,new_scientific_calls=0,new_local_or_compute_disk_payload_copies=False)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
