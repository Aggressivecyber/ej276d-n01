from pathlib import Path
from datetime import datetime,timezone
import json,os,time,sys,hashlib
r=Path('/home/rylan/projects/ej276d-n01/wp4_023_20261016');s=r/'runs/primary/supervisor';a=r/'runs/primary/artifacts'
cfg_sha,mf_sha,ops_sha=sys.argv[1:]
cfgraw=(r/'wp4_023_config.json').read_bytes();assert hashlib.sha256(cfgraw).hexdigest()==cfg_sha
p=json.loads(cfgraw);assert hashlib.sha256((r/'WP4_023_SOURCE_MANIFEST.json').read_bytes()).hexdigest()==mf_sha
assert hashlib.sha256((r/'REMOTE_SUPERVISOR.py').read_bytes()).hexdigest()==ops_sha
sys.path.insert(0,str(r/'scripts'))
assert hashlib.sha256((r/'scripts/wp1_ops.py').read_bytes()).hexdigest()=='f8ddbbdb2f2f6df033e964f27bf03dcbda07055cc7378775591ef69b9caf85b5'
from wp1_ops import sha,durable_progress
for row in list(p['source_files'].values())+list(p['inputs'].values())+list(p['contracts'].values()):
 path=Path(row['path']);assert path.stat().st_size==row['bytes']
 assert sha(path)==row['sha256'],str(path)
d={'UTC':datetime.now(timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'root':str(r)}
for name,path in [('launch',s/'LAUNCH.json'),('started',a/'STARTED.json'),('phase',a/'PHASE.json'),('native_exit',s/'NATIVE_EXIT.json'),('failed',s/'FAILED.json'),('driver_failed',a/'FAILED.json'),('complete',a/'COMPLETE.json'),('status',a/'status.json'),('contrast',a/'CONTRAST023.json'),('inventory',a/'ARTIFACT_INVENTORY.json')]:
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

state=d;root=r;p=json.loads((root/'wp4_023_config.json').read_text());native=state['native_exit'];complete=state['complete'];dur=state['durable_child_progress']
assert native['child_native_returncode']==0 and native['reason'] is None and not native['resource_violations']
assert state['failed'] is None and state['driver_failed'] is None and not any(x['alive'] for x in state['processes'].values())
assert complete['run_id']==root.name and complete['native_status']=='COMPLETE' and complete['counters']==complete['attempted_counters']==p['budget_caps']
assert complete['operations_entered']==complete['operations_completed']==10 and not complete['inflight']
assert not dur['unknown_inflight_ids'] and not dur['failed_ids'] and not dur['truncated_tail']
for k,v in complete['counters'].items():assert dur['completed_counts'].get(k,0)==dur['entered_counts'].get(k,0)==v,k
assert complete['new_distance_counts']=={'19':94517248,'257':54009856} and complete['NN_CP_AD_G_RNG_ADC']==0
assert native['peak_RSS_KiB']*1024<=8*1024**3 and native['elapsed_seconds']<=7200 and native['output_bytes']<=1536*1024**2
old=root.parent/'wp4_017_20261015';sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();ops.COLLECTION_LIMIT=128*1024**2
whole={}
for path in sorted(root.rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  name=path.relative_to(root).as_posix();ops.core().safe_name(name);whole[name]=ops.named_record(path)
producer=state['inventory']
for name,row in producer['files'].items():
 ops.core().safe_name(name);assert whole['runs/primary/artifacts/'+name]==row,name
selected,omitted=ops.select_inventory(whole,include_heads=False)
compact=['runs/primary/artifacts/PARENT_AXES.npz','runs/primary/artifacts/CONTRAST023.npz']+['runs/primary/artifacts/PARENT_SCORES023_'+h+'.npz' for h in p['hypotheses']]
for name in compact:
 assert whole[name]['bytes']<=4*1024**2;selected[name]=whole[name];omitted.pop(name)
assert 'runs/primary/artifacts/operations.jsonl' in selected and 'runs/primary/supervisor/driver_console.log' in selected
assert sum(v['bytes'] for v in selected.values())<=128*1024**2
assert not any(n.endswith('_OBSERVABLE.npz') or 'HARD_ROWS' in n for n in selected)
import tarfile
archive=root.parent/'wp4_023_selected_20261016_1410_01.tar'
with tarfile.open(archive,'x') as tar:
 for name in sorted(selected):ops.add_regular(tar,root/name,name)
assert archive.stat().st_size<=128*1024**2
print(json.dumps(dict(whole=whole,selected=selected,omitted=omitted,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=ops.sha(archive),producer_inventory_entries=len(producer['files']),full_remote_inventory_SHA_verified=True,producer_inventory_all_entries_SHA_verified=True,full_operation_and_console_logs_included=True,selection_payload_bytes=sum(x['bytes'] for x in selected.values()),native=native,durable=dur,complete=complete,old017_scientific_validator_not_used=True,large_O19_arrays_retained_remote=True,scientific_acceptance=False)))
