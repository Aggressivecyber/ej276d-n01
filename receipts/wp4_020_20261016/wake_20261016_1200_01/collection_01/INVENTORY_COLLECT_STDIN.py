from pathlib import Path
import hashlib,json,sys
root=Path('/home/rylan/projects/ej276d-n01/wp4_020_20261016'); cfg_sha,mf_sha,ops_sha=sys.argv[1:]
raw=(root/'wp4_020_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw)
for row in p['protection_files'].values():
 data=Path(row['path']).read_bytes(); assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
old=root.parent/'wp4_017_20261015';sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();sys.path.insert(0,str(root/'scripts'))
import run_wp4_020 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
ops.__dict__.update(RUN_ID='wp4_020_20261016',REMOTE=str(root),CFG='wp4_020_config.json',MANIFEST='WP4_020_SOURCE_MANIFEST.json',DRIVER='scripts/run_wp4_020.py',CONTROLLER='REMOTE_SUPERVISOR.py',validate_config=driver.validate_settings020)
inherited.REMOTE=str(root)
ops.STATE_NAMES=ops.STATE_NAMES+('runs/primary/artifacts/UNPAIRED_GATE020.json','runs/primary/artifacts/RESTORED_SEED0_PROOF.json',)
ops.STATE_NAMES=ops.STATE_NAMES+('runs/primary/artifacts/ARTIFACT_INVENTORY.json','runs/primary/artifacts/T1_unpaired_seed0/RESTORED_EVALUATION_COMPLETE020.json','runs/primary/artifacts/T1_unpaired_seed0/COMPLETE.json','runs/primary/artifacts/T1_unpaired_seed1/TRAINING_COMPLETE020.json','runs/primary/artifacts/T1_unpaired_seed1/COMPLETE.json',)
p,state=inherited.native_snapshot(dict(config_sha256=cfg_sha,source_manifest_sha256=mf_sha,controller_sha256=ops_sha))
state['supervisor_console_tail']=(root/'supervisor_console.log').read_text()[-8000:]
log=root/'runs/primary/console.log'
state['worker_console_tail']=log.read_text()[-16000:] if log.exists() else None

assert state['runs/primary/supervisor/exit_receipt.json']['child_native_exit_code']==state['runs/primary/supervisor/exit_receipt.json']['exit_code']==0
assert state['runs/primary/artifacts/FAILED.json'] is None
complete=state['runs/primary/artifacts/COMPLETE.json']; assert complete['counters']==complete['attempted_counters']==p['budget_caps']
assert complete['run_id']==p['run_id'] and complete['native_status']=='COMPLETE' and complete['seed0_new_G_AD']==0 and complete['seed1_new_G_AD']==2000
assert not state['live_processes']['supervisor']['alive'] and not state['live_processes']['worker']['alive']
ops.COLLECTION_LIMIT=128*1024**2
inventory,selected,omitted,transport=ops.archive_inventory(state,p,partial=True,include_heads=True)
for path in sorted(root.rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  name=path.relative_to(root).as_posix(); row=ops.named_record(path)
  if name in inventory: assert inventory[name]==row
  inventory[name]=row
manifest=state['runs/primary/artifacts/ARTIFACT_INVENTORY.json']
for name,row in manifest['files'].items():
 ops.core().safe_name(name)
 if name!='status.json': assert inventory['runs/primary/artifacts/'+name]==row, name
selected,omitted=ops.select_inventory(inventory,include_heads=True)
compact=['runs/primary/artifacts/ORIGINAL_PHYSICAL_WEIGHTS.npz']
for trial in p['trial_order']:
 prefix='runs/primary/artifacts/'+trial+'/'
 compact += [prefix+'COMPARISON020.npz',prefix+'evaluation/EVALUATION_AXES.npz',prefix+'evaluation/EVALUATION_RANDOM_STATES.npz']
 compact += [prefix+'PARENT_COMPARISON020_'+h+'.npz' for h in p['hypotheses']]
 compact += [prefix+'evaluation/bank'+str(i)+'_CHARGE.npz' for i in (0,1)]
compact += ['runs/primary/artifacts/T1_unpaired_seed1/TRAINING_INITIAL_STATES.npz','runs/primary/artifacts/T1_unpaired_seed1/TRAINING_END_STATES.npz']
for name in compact:
 assert inventory[name]['bytes']<=4*1024**2; selected[name]=inventory[name];omitted.pop(name)
assert sum(x['bytes'] for x in selected.values())<=128*1024**2
assert 'runs/primary/artifacts/operations.jsonl' in selected and 'runs/primary/artifacts/T1_unpaired_seed1/training.jsonl' in selected
assert 'runs/primary/artifacts/T1_unpaired_seed1/final.pt' in selected and 'runs/primary/artifacts/T1_unpaired_seed1/step_1000.pt' in selected
assert not any('HARD_ROWS.npz' in n or '_OBSERVABLE.npz' in n or '/DERIVED64_' in n or '_REPORT.npz' in n or '_BASELINE100.npz' in n for n in selected)
import tarfile
archive=root.parent/'wp4_020_selected_20261016_1200_01.tar'
with tarfile.open(archive,'x') as tar:
 for name in sorted(selected):ops.add_regular(tar,root/name,name)
assert archive.stat().st_size<=128*1024**2
print(json.dumps(dict(whole=inventory,selected=selected,omitted=omitted,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=ops.sha(archive),producer_complete_inventory_all_immutable_files_verified=True,terminal_native0_and_exact020_budget_verified=True,original_partial_collection_primitive_used_only_to_skip_old017_science_validator=True,all_large_raw_arrays_retained_remote=True,full_operation_and_training_logs_included=True,selection_payload_bytes=sum(x['bytes'] for x in selected.values()),native=state['runs/primary/supervisor/exit_receipt.json'],durable=state['durable_child_progress'],collection_primitives='unchanged017 archive_inventory partial transport + named_record + select_inventory + add_regular; ROOT verifies020 terminal scope and full producer inventory directly',scientific_acceptance=False)))
