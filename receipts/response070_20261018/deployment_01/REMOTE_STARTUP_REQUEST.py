import datetime,json,os,time,hashlib,base64
from pathlib import Path
root=Path('/home/rylan/projects/ej276d-n01/response070_split_area_crossing_level_warp_20261018')
started=time.monotonic()
while time.monotonic()-started<55:
    if (root/'run_01/FAILED.json').exists() or (root/'control_01/exit_receipt.json').exists():break
    path=root/'run_01/PROGRESS.json'
    if path.exists() and json.loads(path.read_text())['execution_counts']['candidate_head_evaluations']>=1:break
    time.sleep(1)
labels=['REAL','seed0_LEVEL069_fast','seed0_LEVEL069_intrinsic','seed0_SPLITWARP_fast','seed0_SPLITWARP_intrinsic']
names=['DISPATCH_RECEIPT.json','control_01/launch.json','control_01/resource_status.json','control_01/exit_receipt.json',
    'run_01/STARTED.json','run_01/PROGRESS.json','run_01/DEFINITIONS.json','run_01/FAILED.json','run_01/RESULT.json','run_01/COMPLETE.json',
    'run_01/seed0_M0.json','run_01/seed0_LEVEL069.json','run_01/seed0_SPLITWARP.json']
names += [f'run_01/{label}_PROFILE_DEFINITION.json' for label in labels]
files={name:json.loads((root/name).read_text()) for name in names if (root/name).exists()}
processes=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        argv=[part.decode(errors='replace') for part in (proc/'cmdline').read_bytes().split(bytes([0])) if part]
        if str(root)+'/' not in ' '.join(argv):continue
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        processes.append(dict(pid=int(proc.name),start_ticks=stat[19],state=stat[0],argv=argv,cwd=os.readlink(proc/'cwd')))
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
names=['run_01/REAL_ROWS.npz']+[f'run_01/{label}_PROFILES.npz' for label in labels]
for case in ('LEVEL069','SPLITWARP'):
    names += [f'run_01/seed0_{case}_{kind}.npz' for kind in ('COUPLING_ROWS','COUPLING_EXAMPLES','GEOMETRY_ROWS','EXAMPLES','INTRINSIC_MEAN_ROWS','fast_CARRIER_EXAMPLES')]
    names += [f'run_01/seed0_{case}_{h}_{kind}.npz' for h in ('fast','slow','common_level') for kind in ('NATIVE_ROWS','NATIVE_EXAMPLES','OBSERVATION_FLAGS')]
names.append('run_01/seed0_SPLITWARP_WARP_QUERIES.npz')
binary_files={}
for name in names:
    if (root/name).exists():
        payload=(root/name).read_bytes();binary_files[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),base64=base64.b64encode(payload).decode())
log=root/'control_01/console.log'
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    startup_wait_seconds=time.monotonic()-started,files=files,binary_files=binary_files,processes=processes,
    current_run_bytes=sum(p.stat().st_size for p in (root/'run_01').rglob('*') if p.is_file()),console_tail=log.read_text(errors='replace')[-4000:] if log.exists() else None)))
