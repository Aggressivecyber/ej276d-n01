from pathlib import Path
import datetime,fcntl,json,os,shutil,subprocess
base=Path('/home/rylan/projects/ej276d-n01')
with (base/'.n01_scientific_dispatch.lock').open('r+b') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 ram=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 disk=shutil.disk_usage(base).free;affinity=sorted(os.sched_getaffinity(0))
 lines=subprocess.check_output(['ps','-eo','pid,args'],text=True,timeout=10).splitlines()
 active=[line for line in lines if '/ej276d-n01/' in line and any(marker in line for marker in ('run_wp','REMOTE_SUPERVISOR.py','supervise_','-m n01.'))]
 assert ram>=10*1024**3 and disk>=4*1024**3 and len(affinity)>=12 and not active,(ram,disk,affinity,active)
 print(json.dumps(dict(run_id='wp4_023_20261016',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),RAM_available_bytes=ram,disk_free_bytes=disk,CPU_affinity=affinity,active_N01_science_processes=active,exclusive_project_lock_available=True,GPU_work_requested=False,scientific_started=False)))
