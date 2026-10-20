
from pathlib import Path
import datetime,json,hashlib
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_original_precision_REAL_pilot_20261020')
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'processes':[],'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name in ['run_01/COMPLETE.json','run_01/FAILED.json','run_01/PROGRESS.json','run_01/RESULT.json','run_01/seed0_joint_MEMORY_ITER00.json','run_01/seed1_joint_MEMORY_ITER00.json','run_01/seed0_FEATURE_REPLAY.json','run_01/seed1_FEATURE_REPLAY.json','control_01/launch.json','control_01/exit_receipt.json','control_01/resource_status.json']:
 path=root/name
 if path.exists():result['files'][name]=json.loads(path.read_text())
for path in Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  args=[s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(bytes([0])) if s]
  if any(str(root) in arg for arg in args):result['processes'].append(dict(pid=int(path.name),argv=args,start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,item in manifest['files'].items():
 raw=(root/name).read_bytes()
 assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
result['source_files_verified']=len(manifest['files'])
result['config_sha256']=manifest['config_sha256']
result['tracks']={}
for path in sorted((root/'run_01').glob('*_ITER*.json')):
 d=json.loads(path.read_text())
 result['tracks'][path.stem]={k:d[k] for k in ('FD_columns_completed','selected_trial','stop_reason','current_solve')}
 result['tracks'][path.stem]['trials_completed']=len(d['trials'])
 result['tracks'][path.stem]['mtime_UTC']=datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat()
print(json.dumps(result))
