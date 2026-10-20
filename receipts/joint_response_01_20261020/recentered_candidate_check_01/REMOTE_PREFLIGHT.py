from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_candidate_check_20261020')
started=time.monotonic()
print(json.dumps(dict(stage='EXISTING_SCIENTIFIC_PATH_PREFLIGHT_STARTED',UTC=datetime.now(timezone.utc).isoformat(),
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())),flush=True)
m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
assert m['config_sha256']=='5755953098801a9b731ae4df6d69974b554fceb93fffa422769602e1566c9de3' and len(m['files'])==193
with ThreadPoolExecutor(max_workers=4) as pool:
    pending=[(name,entry,pool.submit((root/name).read_bytes)) for name,entry in m['files'].items()]
    for name,entry,future in pending:
        raw=future.result();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
print(json.dumps(dict(stage='193_SOURCE_BINDINGS_VERIFIED',elapsed_seconds=time.monotonic()-started)),flush=True)
review=json.loads((root/'PRE_DISPATCH_REVIEW.json').read_bytes())
assert review['status']=='PASS_FIXED_CANDIDATE_EXISTING_PATH_DEPLOYMENT_AND_STATIC_REVIEW' and review['terminal_auditor_prepared']
sys.path[:0]=[str(root/'scripts'),str(Path(m['prior_NAS_root'])/'src'),str(Path(m['prior_NAS_root'])/'scripts')]
import check_joint01_recentered_restoration
import torch,numpy
processes=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:argv=[v.decode(errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    command=' '.join(argv)
    if any(prefix in command for prefix in ('/home/rylan/projects/ej276d-n01/','/mnt/MiNas/GAN/research/N01/')) and any('/scripts/'+stem in command for stem in ('run_','diagnose_','complete_','check_','supervise.py','supervise_joint01_nas.py')):
        processes.append(dict(pid=int(path.name),argv=argv))
assert not processes,processes
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
assert int(gpu.split(',')[3].strip())>=6144 and memory['MemAvailable']>=14*1024**3
assert {0,1,2,3}<=os.sched_getaffinity(0)
record=dict(status='PASS_NATIVE_EXISTING_PATH_SCIENTIFIC_PREFLIGHT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=193,config_sha256=m['config_sha256'],torch=torch.__version__,numpy=numpy.__version__,
    cuda_available=torch.cuda.is_available(),GPU_snapshot=gpu,memory=memory,project_processes=processes,
    elapsed_seconds=time.monotonic()-started,NAS_mount_connection_transfer_or_migration_changes=False,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scientific_calls=0)
assert record['cuda_available']
(root/'NATIVE_PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
