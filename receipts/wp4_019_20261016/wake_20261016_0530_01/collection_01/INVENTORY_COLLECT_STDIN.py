from pathlib import Path
import hashlib,json,sys
root=Path('/home/rylan/projects/ej276d-n01/wp4_019_20261016'); cfg_sha,mf_sha,ops_sha=sys.argv[1:]
raw=(root/'wp4_019_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw)
for row in p['protection_files'].values():
 data=Path(row['path']).read_bytes(); assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
old=root.parent/'wp4_017_20261015';sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();sys.path.insert(0,str(root/'scripts'))
import run_wp4_019 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
ops.__dict__.update(RUN_ID='wp4_019_20261016',REMOTE=str(root),CFG='wp4_019_config.json',MANIFEST='WP4_019_SOURCE_MANIFEST.json',DRIVER='scripts/run_wp4_019.py',CONTROLLER='REMOTE_SUPERVISOR.py',validate_config=driver.validate_settings019)
inherited.REMOTE=str(root)
ops.STATE_NAMES=ops.STATE_NAMES+('runs/primary/artifacts/UNPAIRED_GATE019.json',)
p,state=inherited.native_snapshot(dict(config_sha256=cfg_sha,source_manifest_sha256=mf_sha,controller_sha256=ops_sha))
state['supervisor_console_tail']=(root/'supervisor_console.log').read_text()[-8000:]
log=root/'runs/primary/console.log'
state['worker_console_tail']=log.read_text()[-16000:] if log.exists() else None
for name in ('runs/primary/artifacts/ARTIFACT_INVENTORY.json', 'runs/primary/artifacts/T1_unpaired_seed0/TRAINING_COMPLETE019.json', 'runs/primary/artifacts/T1_unpaired_seed0/COMPLETE.json', 'runs/primary/artifacts/T1_unpaired_seed1/TRAINING_COMPLETE019.json', 'runs/primary/artifacts/T1_unpaired_seed1/COMPLETE.json'):
 path=root/name;state[name]=json.loads(path.read_text()) if path.is_file() else None
assert state['runs/primary/supervisor/exit_receipt.json']['child_native_exit_code']==1
assert state['runs/primary/artifacts/FAILED.json']['current_trial_G_updates']==2000
assert state['runs/primary/artifacts/COMPLETE.json'] is None and state['runs/primary/artifacts/UNPAIRED_GATE019.json'] is None
assert not state['live_processes']['supervisor']['alive'] and not state['live_processes']['worker']['alive']
ops.COLLECTION_LIMIT=128*1024**2
inventory,selected,omitted,verified=ops.archive_inventory(state,p,partial=True,include_heads=True)
for path in sorted(root.rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  name=path.relative_to(root).as_posix();row=ops.named_record(path)
  if name in inventory:assert inventory[name]==row
  inventory[name]=row
selected,omitted=ops.select_inventory(inventory,include_heads=True)
assert all('runs/primary/artifacts/T1_unpaired_seed0/'+name in selected for name in ('step_1000.pt','final.pt','failure.pt'))
assert not any(name.startswith('runs/primary/artifacts/T1_unpaired_seed1/') for name in inventory)
import tarfile
archive=root.parent/'wp4_019_failed_selected_20261016_0530_01.tar'
with tarfile.open(archive,'x') as tar:
 for name in selected:ops.add_regular(tar,root/name,name)
assert archive.stat().st_size<=128*1024**2
print(json.dumps(dict(whole=inventory,selected=selected,omitted=omitted,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=ops.sha(archive),partial_evidence_only=True,producer_final_inventory_absent=True,complete_absent=True,gate_absent=True,all_raw_NPZ_retained_remote=True,native=state['runs/primary/supervisor/exit_receipt.json'],failure=state['runs/primary/artifacts/FAILED.json'],durable=state['durable_child_progress'],collection_primitives='unchanged017 archive_inventory partial + select_inventory + add_regular;128MiB bound',scientific_acceptance=False)))
