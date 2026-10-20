from datetime import datetime,timezone
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess

base=Path('/home/rylan/projects/ej276d-n01')
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
assert m['config_sha256']=='1bb8ae27f1450350b393af374f638e10006f057ff158790e57b98e8943df5505' and len(m['files'])==185
assert json.loads((root/'DEPLOYMENT_VERIFICATION.json').read_bytes())['status']=='PASS_NAS_DEPLOYMENT'
for name,entry in m['files'].items():
    raw=(root/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
for name in ('run_01','control_01','DISPATCH_RECEIPT.json'):
    if (root/name).exists():raise FileExistsError(root/name)
mount=json.loads(subprocess.check_output(['findmnt','--json','--target',str(root),'-o','SOURCE,FSTYPE'],text=True))
assert any(v['fstype']=='cifs' and v['source']=='//192.168.31.38/我的文档-1430134245' for v in mount['filesystems'])
processes=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    command=' '.join(argv)
    if any(prefix in command for prefix in (str(base)+'/', '/mnt/MiNas/GAN/research/N01/')) and any('/scripts/'+stem in command for stem in ('run_','diagnose_','supervise.py')):
        processes.append(dict(pid=int(path.name),argv=argv))
if processes:raise RuntimeError('Another N01 scientific process is present: '+str(processes))
with (base/'.n01_scientific_dispatch.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.split(',')[3].strip())>=6144
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
assert memory['MemAvailable']>=14*1024**3 and shutil.disk_usage(root).free>=512*1024**2
assert {0,1,2,3}<=os.sched_getaffinity(0)
python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
command=['taskset','-c','0,1,2,3','flock','-n',str(base/'.n01_scientific_dispatch.lock'),python,'-B',
    str(root/'scripts/supervise.py'),'--output',str(root/'control_01'),'--rss-limit-gib','12','--',
    'timeout','--signal=TERM','--kill-after=20s','3900s',python,'-B',str(root/'scripts/diagnose_joint01_recentered_derivatives.py'),
    '--config',str(root/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json'),
    '--config-sha256',m['config_sha256'],'--out',str(root/'run_01'),'--device','cuda:0']
cache=root/'cache'
overrides=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),TMPDIR=str(cache),TEMP=str(cache),TMP=str(cache),
    XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),
    MPLCONFIGDIR=str(cache/'mpl'),CUDA_CACHE_PATH=str(cache/'cuda'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
env=os.environ.copy();env.update(overrides)
with (root/'dispatcher.log').open('x') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
record=dict(status='DISPATCHED',UTC=datetime.now(timezone.utc).isoformat(),dispatcher_pid=child.pid,
    dispatcher_start_ticks=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19],
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),remote_root=str(root),command=command,
    environment_overrides=overrides,config_sha256=m['config_sha256'],GPU_snapshot=gpu,memory=memory,
    NAS_free_bytes=shutil.disk_usage(root).free,NAS_mount=mount,project_processes_before_dispatch=processes,
    automatic_restart=False,no_compute_disk_output_copy=True,science_result_yet=False)
(root/'DISPATCH_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
