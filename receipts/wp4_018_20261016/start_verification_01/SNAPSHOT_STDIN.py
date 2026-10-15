from pathlib import Path
from datetime import datetime,timezone
import json,os,time
r=Path('/home/rylan/projects/ej276d-n01/wp4_018_20261016');s=r/'runs/primary/supervisor';a=r/'runs/primary/artifacts'
d={'UTC':datetime.now(timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'root':str(r)}
for name,path in [('launch',s/'LAUNCH.json'),('started',a/'STARTED.json'),('phase',a/'PHASE.json'),('native_exit',s/'NATIVE_EXIT.json'),('failed',s/'FAILED.json'),('complete',a/'COMPLETE.json')]:
 d[name]=json.loads(path.read_text()) if path.exists() else None
processes={}
if d['launch']:
 for name in ['supervisor','child']:
  pid=d['launch'][name]['pid'];p=Path('/proc',str(pid));item={'pid':pid,'alive':p.exists()}
  if p.exists():
   stat=(p/'stat').read_text().rsplit(')',1)[1].split();status=(p/'status').read_text();rss=[line for line in status.splitlines() if line.startswith('VmRSS:')]
   item.update(parent_pid=int(stat[1]),start_ticks=int(stat[19]),command=[v.decode() for v in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')],cwd=os.readlink(p/'cwd'),CPU_affinity=sorted(os.sched_getaffinity(pid)),RSS_bytes=int(rss[0].split()[1])*1024 if rss else 0)
  processes[name]=item
 d['elapsed_seconds']=time.monotonic()-d['launch']['supervisor_started_monotonic']
d['processes']=processes;d['artifact_bytes']=sum(p.stat().st_size for p in a.rglob('*') if p.is_file())
d['driver_log_tail']=(s/'driver_console.log').read_text()[-6000:] if (s/'driver_console.log').exists() else None
d['supervisor_log_tail']=(r/'supervisor_console.log').read_text()[-3000:]
print(json.dumps(d,indent=2))
