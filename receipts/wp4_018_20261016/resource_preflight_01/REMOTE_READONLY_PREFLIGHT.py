import json,os,shutil
from pathlib import Path
mem={p.split(':',1)[0]:p.split(':',1)[1].strip() for p in Path('/proc/meminfo').read_text().splitlines()}
processes=[]
for q in Path('/proc').iterdir():
 if not q.name.isdecimal():continue
 if (q/'cmdline').exists():
  try:cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except FileNotFoundError:continue
  if any(t in cmd for t in ['scripts/run_wp4_017.py','scripts/run_wp4_018.py','scripts/run_wp3_015.py','scripts/run_wp3_016.py']) and 'python -B -' not in cmd:processes.append({'pid':int(q.name),'command':cmd[:1200]})
print(json.dumps({'uid':os.getuid(),'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'CPU_affinity':sorted(os.sched_getaffinity(0)),'available_memory':mem['MemAvailable'],'disk_free_bytes':shutil.disk_usage('/home/rylan/projects/ej276d-n01').free,'matching_scientific_processes':processes},indent=2))
