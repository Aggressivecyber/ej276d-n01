from datetime import datetime, timedelta, timezone
from pathlib import Path
import ast
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
raw = (HERE/'PRIVATE_NATIVE_DEPLOYMENT_STDOUT.jsonl').read_bytes()
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
binding = next(row for row in rows if row.get('stage') == 'THIN_BOUND_SOURCES_VERIFIED')
assert binding['source_files'] == 192 and binding['external_prefix_files'] == 12
assert json.loads((HERE/'PRIVATE_NATIVE_DEPLOYMENT_EXIT.json').read_bytes())['returncode'] != 0
bootstrap = (HERE/'PRIVATE_NAS_SESSION_BOOTSTRAP.py').read_text(encoding='utf-8')
started = next(row for row in rows if row.get('stage') == 'THIN_DEPLOYMENT_STARTED')
review = dict(status='PASS_SOURCE_AND_PREFIX_BINDINGS_ONLY_NATIVE_IMPORT_PENDING',
    UTC=datetime.now(timezone.utc).isoformat(), source_files=192, source_pins=191,
    external_prefix_files_verified=12, source_verification_seconds=binding['elapsed_seconds'],
    binding_asof_UTC=(datetime.fromisoformat(started['UTC'])+timedelta(seconds=binding['elapsed_seconds'])).isoformat(),
    verifier_log_sha256=hashlib.sha256(raw).hexdigest(),
    manifest_sha256=hashlib.sha256((HERE/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    bootstrap_sha256=hashlib.sha256(bootstrap.encode('utf-8')).hexdigest(),
    prior_combined_preflight_timed_out=True, native_import_phase_maximum_seconds=150,
    scientific_operation_and_resource_budgets_unchanged=True, scientific_calls=0)
(HERE/'PRIVATE_SOURCE_BINDING_REVIEW.json').write_text(json.dumps(review, indent=2)+'\n', encoding='utf-8', newline='\n')
manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
body = '''
import hashlib,signal,sys
signal.alarm(150)
native_started=time.monotonic()
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020')
prior=Path(manifest['prior_NAS_root']);cache=root/'cache';cache.mkdir(exist_ok=True)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),CUDA_CACHE_PATH=str(cache/'cuda'))
sys.path[:0]=[str(root/'scripts'),str(prior/'src'),str(prior/'scripts')]
print(json.dumps(dict(stage='PRIVATE_NATIVE_IMPORTS_START',UTC=datetime.now(timezone.utc).isoformat())),flush=True)
import torch,numpy,scipy
print(json.dumps(dict(stage='TORCH_NUMPY_SCIPY_IMPORTED',elapsed_seconds=time.monotonic()-native_started)),flush=True)
import complete_joint01_recentered_derivatives
print(json.dumps(dict(stage='PINNED_DRIVER_IMPORTED',elapsed_seconds=time.monotonic()-native_started)),flush=True)
import supervise_joint01_nas
assert complete_joint01_recentered_derivatives.ROOT==root and torch.cuda.is_available()
config=root/'configs/joint_response_01_recentered_derivative_completion_thin_20261020.json'
assert hashlib.sha256(config.read_bytes()).hexdigest()==manifest['config_sha256']
record=dict(status='PASS_NAS_DEPLOYMENT',UTC=datetime.now(timezone.utc).isoformat(),source_files=192,source_pins=191,external_immutable_sources=185,local_text_files=7,external_prefix_files_verified=12,config_sha256=manifest['config_sha256'],elapsed_seconds=time.monotonic()-native_started,remote_root=str(root),immutable_source_reuse_root=str(prior),torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,native_driver_and_supervisor_imported=True,new_scientific_calls=0,new_local_or_compute_disk_payload_copies=False,source_binding_phase=source_review,private_NAS_session=True,global_mount_modified=False)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(record),flush=True)
'''
request = ('manifest='+repr(manifest)+'\nsource_review='+repr(review)+'\n'+bootstrap+'\n'+body).encode('utf-8')
ast.parse(request.decode('utf-8'))
with (HERE/'PRIVATE_NATIVE_IMPORTS_STDOUT.jsonl').open('xb') as stdout, (HERE/'PRIVATE_NATIVE_IMPORTS_STDERR.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', 'sudo -n unshare --mount --propagation private /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=180)
(HERE/'PRIVATE_NATIVE_IMPORTS_EXIT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), returncode=result.returncode), indent=2)+'\n', encoding='utf-8', newline='\n')
result.check_returncode()
rows = [json.loads(line) for line in (HERE/'PRIVATE_NATIVE_IMPORTS_STDOUT.jsonl').read_text(encoding='utf-8').splitlines()]
assert rows[-1]['status'] == 'PASS_NAS_DEPLOYMENT'
(HERE/'PRIVATE_DEPLOYMENT_SESSION.json').write_text(json.dumps(rows[0], indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(rows[-1], indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(rows[-1]))
