import datetime,fcntl,hashlib,json,os,shutil,subprocess
from pathlib import Path
base=Path('/home/rylan/projects/ej276d-n01')
root=base/'joint_response01_curvature_update_diagnostic_20261020'
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
assert manifest['config_sha256']=='f91086896da8c0f6fe14fd946551acd7eda5c9cbf91b96d2dad0c5757270d24e' and len(manifest['files'])==858
verification=json.loads((root/'DEPLOYMENT_VERIFICATION.json').read_text())
if not verification['import_pass'] or verification['status']!='PASS':
 raise ValueError('Original-generator draw precision deployment not verified')
for name,record in manifest['files'].items():
 b=(root/name).read_bytes()
 if len(b)!=record['bytes'] or hashlib.sha256(b).hexdigest()!=record['sha256']:
  raise ValueError('Original-generator draw precision file changed before dispatch: '+name)
for target in ('run_01','control_01','DISPATCH_RECEIPT.json'):
 if (root/target).exists():raise FileExistsError(root/target)
processes=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  argv=[part.decode(errors='replace') for part in (proc/'cmdline').read_bytes().split(bytes([0])) if part]
  cwd=os.readlink(proc/'cwd')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if str(base)+'/' in ' '.join(argv) or cwd==str(base) or cwd.startswith(str(base)+'/'):
  processes.append(dict(pid=int(proc.name),argv=argv,cwd=cwd))
if processes:raise RuntimeError('Other N01 processes present: '+str(processes))
with (base/'.n01_scientific_dispatch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
 '--format=csv,noheader,nounits'],text=True).strip()
gpu_free=int(gpu.split(',')[3].strip())
if gpu_free<6144:raise RuntimeError('Less than6GiB free GPU for4GiB allocator plus2GiB headroom')
memory={row.split(':')[0]:int(row.split()[1])*1024 for row in Path('/proc/meminfo').read_text().splitlines()
 if row.startswith(('MemTotal:','MemAvailable:'))}
disk=shutil.disk_usage(root)
if memory['MemAvailable']<14*1024**3 or disk.free<3*1024**3:raise RuntimeError('Original-generator draw precision RAM or disk headroom insufficient')
cores=[0,1,2,3]
if not set(cores)<=os.sched_getaffinity(0):raise RuntimeError('Original-generator draw precision registered CPU affinity unavailable')
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
command=['taskset','-c','0,1,2,3','flock','-n',str(base/'.n01_scientific_dispatch.lock'),python,'-B',
 str(root/'scripts/supervise.py'),'--output',str(root/'control_01'),'--rss-limit-gib','12','--',
 'timeout','--signal=TERM','--kill-after=20s','480s',python,'-B',str(root/'scripts/diagnose_joint01_curvature_update.py'),
 '--config',str(root/'configs/joint_response_01_curvature_update_diagnostic_20261020.json'),'--config-sha256',manifest['config_sha256'],
 '--out',str(root/'run_01'),'--device','cuda:0']
cache=root/'cache'
overrides=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),TMPDIR=str(cache),TEMP=str(cache),TMP=str(cache),
 XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),
 MPLCONFIGDIR=str(cache/'mpl'),CUDA_CACHE_PATH=str(cache/'cuda'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
 MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
env=os.environ.copy();env.update(overrides)
with (root/'dispatcher.log').open('x') as log:
 child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,close_fds=True)
ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
record=dict(schema='n01/joint-response01-curvature-update-diagnostic-dispatch/v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 dispatcher_pid=child.pid,dispatcher_start_ticks=ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 remote_root=str(root),command=command,environment_overrides=overrides,config_sha256=manifest['config_sha256'],
 GPU_snapshot=gpu,GPU_free_MiB=gpu_free,memory=memory,disk_free_bytes=disk.free,CPU_affinity=cores,
 project_processes_before_dispatch=processes,project_lock_verified_free=True,automatic_restart=False,
 science_result_yet=False)
(root/'DISPATCH_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
