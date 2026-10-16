from pathlib import Path
from datetime import datetime,timezone
import json,os,time,sys,hashlib
r=Path('/home/rylan/projects/ej276d-n01/wp3_025_20261016');s=r/'runs/primary/supervisor';a=r/'runs/primary/artifacts'
cfg_sha,mf_sha,ops_sha=sys.argv[1:]
cfgraw=(r/'wp3_025_config.json').read_bytes();assert hashlib.sha256(cfgraw).hexdigest()==cfg_sha
p=json.loads(cfgraw);assert hashlib.sha256((r/'WP3_025_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==mf_sha
assert hashlib.sha256((r/'REMOTE_SUPERVISOR.py').read_bytes()).hexdigest()==ops_sha
sys.path.insert(0,str(r/'scripts'))
assert hashlib.sha256((r/'scripts/wp1_ops.py').read_bytes()).hexdigest()=='f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5'
from wp1_ops import sha,durable_progress
for row in list(p['source_files'].values())+list(p['inputs'].values())+list(p['contracts'].values()):
 path=Path(row['path']);assert path.stat().st_size==row['bytes']
 assert sha(path)==row['sha256'],str(path)
d={'UTC':datetime.now(timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'root':str(r)}
for name,path in [('launch',s/'LAUNCH.json'),('started',a/'STARTED.json'),('phase',a/'PHASE.json'),('native_exit',s/'NATIVE_EXIT.json'),('failed',s/'FAILED.json'),('driver_failed',a/'FAILED.json'),('complete',a/'COMPLETE.json'),('status',a/'status.json'),('comparison',a/'COMPARISON.json'),('init_proof',a/'INIT_PROOF.json'),('first_CDF_proof',a/'FIRST_ACTUAL_CDF_PROOF.json'),('resource_usage',a/'RESOURCE_USAGE.json'),('inventory',a/'ARTIFACT_INVENTORY.json')]:
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
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
d['durable_child_progress']=durable_progress(a)
d['current_RSS_bytes']=sum(item.get('RSS_bytes',0) for item in processes.values())
d['verified_freeze']=dict(config_sha256=cfg_sha,source_manifest_sha256=mf_sha,supervisor_sha256=ops_sha,source_and_input_contract_raw_hashes_verified=True)
print(json.dumps(d,indent=2))
