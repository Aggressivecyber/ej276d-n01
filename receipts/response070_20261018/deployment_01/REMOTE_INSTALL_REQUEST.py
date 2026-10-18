import hashlib,json,os,subprocess,tarfile,datetime
from pathlib import Path
base=Path('/home/rylan/projects/ej276d-n01')
root=base/'response070_split_area_crossing_level_warp_20261018'
archive=base/'response070_source_fd01baa0efc0.tar.gz'
manifest_path=base/'response070_source_manifest_01.json'
manifest=json.loads(manifest_path.read_text())
if root.exists():raise FileExistsError(root)
if hashlib.sha256(archive.read_bytes()).hexdigest()!='fd01baa0efc0a0200edf6a1abdcfb8588039718c72082ff5b9bf8a63b98dc71e':
 raise ValueError('070 source archive hash mismatch')
with tarfile.open(archive) as tf:
 if tf.getnames()!=list(manifest['files']):raise ValueError('070 archive inventory mismatch')
 for member in tf.getmembers():
  if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
   raise ValueError('070 archive path/type not permitted')
 root.mkdir()
 tf.extractall(root,filter='data')
for name,record in manifest['files'].items():
 b=(root/name).read_bytes()
 if len(b)!=record['bytes'] or hashlib.sha256(b).hexdigest()!=record['sha256']:
  raise ValueError('070 deployed file mismatch: '+name)
(root/'SOURCE_MANIFEST.json').write_bytes(manifest_path.read_bytes())
cache=root/'cache';cache.mkdir()
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src'),TMPDIR=str(cache),TMP=str(cache),TEMP=str(cache),
 TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),
 MPLCONFIGDIR=str(cache/'mpl'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 CUBLAS_WORKSPACE_CONFIG=':4096:8')
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
results={}
for name in ('039','069','070'):
 cmd=['taskset','-c','0,1',python,'-B',str(root/f'scripts/test_response{name}.py'),'--out',str(root/f'REUSED_{name}_TEST.json')]
 if name in ('069','070'):cmd+=['--device','cuda:0']
 test=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=60)
 (root/f'REUSED_{name}_STDOUT.txt').write_text(test.stdout);(root/f'REUSED_{name}_STDERR.txt').write_text(test.stderr)
 if test.returncode:raise RuntimeError('070 reused kernel check failed '+name+': '+test.stderr)
 results[name]=json.loads((root/f'REUSED_{name}_TEST.json').read_text())
smoke=subprocess.run([python,'-B',str(root/'scripts/run_response070.py'),'--help'],cwd=root,env=env,
 capture_output=True,text=True,timeout=30)
if smoke.returncode:raise RuntimeError('070 runner import failed: '+smoke.stderr)
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_root=str(root),
 source_files=len(manifest['files']),all_source_hashes_verified=True,
 config_sha256=manifest['config_sha256'],engineering_exit_code=test.returncode,
 engineering_result=results['039'],runner_import_exit_code=smoke.returncode,
 reused_kernel_result=results['039'],reused069_CUDA=results['069'],new070_CUDA=results['070'],CP_loads=0,TRAIN_decodes=0,AD_queries=0,scientific_started=False)
(root/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
