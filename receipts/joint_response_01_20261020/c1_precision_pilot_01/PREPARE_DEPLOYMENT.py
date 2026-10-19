from pathlib import Path
import ast
import base64
import hashlib
import json
import subprocess

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
config_name = 'configs/joint_response_01_c1_precision_pilot_20261020.json'
p = json.loads((ROOT / config_name).read_bytes())
config_sha = hashlib.sha256((ROOT / config_name).read_bytes()).hexdigest()
registration = json.loads((HERE / 'REGISTRATION.json').read_text())
assert config_sha == registration['config_sha256']
old = json.loads((ROOT / 'receipts/joint_response_01_20261020/c1_draw_completion_01/SOURCE_MANIFEST.json').read_text())
manifest = dict(schema='n01/joint-response01-c1-precision-pilot-source/v1', config_sha256=config_sha, files=dict(old['files']))
payload = {}
for name in (config_name, 'scripts/run_joint_response01_c1_precision_pilot.py', p['method_contract']):
    data = (ROOT / name).read_bytes()
    manifest['files'][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    payload[name] = base64.b64encode(data).decode()
for name, local_path in json.loads((HERE / 'INPUT_MAP.json').read_text()).items():
    data = Path(local_path).read_bytes()
    manifest['files'][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    payload[name] = base64.b64encode(data).decode()
for name, digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256'] == digest, name
assert len(manifest['files']) == registration['source_files_expected'] == 780
(HERE / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
request = ('from pathlib import Path\nimport base64,hashlib,json,os,shutil,subprocess\n'
    + 'manifest=' + repr(manifest) + '\npayload=' + repr(payload) + '\n' + '''
base=Path('/home/rylan/projects/ej276d-n01')
previous=base/'joint_response01_c1_draw_completion_20261020'
root=base/'joint_response01_c1_precision_pilot_20261020'
root.mkdir()
for name,record in manifest['files'].items():
 destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
 if name in payload:destination.write_bytes(base64.b64decode(payload[name]))
 else:shutil.copy2(previous/name,destination)
 raw=destination.read_bytes()
 if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
  raise ValueError('Precision pilot deployment hash mismatch: '+name)
(root/'cache').mkdir()
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\\n')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),TMPDIR=str(root/'cache'),XDG_CACHE_HOME=str(root/'cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
code='import run_joint_response01_c1_precision_pilot,torch,numpy,scipy,json; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,CUDA_available=torch.cuda.is_available())))'
check=subprocess.run([python,'-B','-c',code],cwd=root,env=env,capture_output=True,text=True)
(root/'IMPORT_STDOUT.txt').write_text(check.stdout);(root/'IMPORT_STDERR.txt').write_text(check.stderr)
check.check_returncode()
runtime=json.loads(check.stdout)
if not runtime['CUDA_available']:raise RuntimeError('Registered CUDA runtime is unavailable after restart')
receipt=dict(status='PASS',import_pass=True,files_verified=len(manifest['files']),file_bytes=sum(v['bytes'] for v in manifest['files'].values()),config_sha256=manifest['config_sha256'],remote_root=str(root),runtime=runtime)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
''')
ast.parse(request)
(HERE / 'REMOTE_INSTALL_REQUEST.py').write_text(request, encoding='utf-8', newline='\n')
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    'rylan@100.82.231.102', 'python3', '-'], input=request, text=True, capture_output=True, encoding='utf-8', timeout=120)
(HERE / 'INSTALL_STDOUT.txt').write_text(result.stdout, encoding='utf-8')
(HERE / 'INSTALL_STDERR.txt').write_text(result.stderr, encoding='utf-8')
result.check_returncode()
verified = json.loads(result.stdout)
assert verified['status'] == 'PASS' and verified['import_pass'] and verified['files_verified'] == 780
assert verified['config_sha256'] == config_sha
(HERE / 'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(verified))
