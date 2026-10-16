from pathlib import Path
import hashlib,json,sys
root=Path('/home/rylan/projects/ej276d-n01/wp4_022_20261016'); cfg_sha,mf_sha,ops_sha=sys.argv[1:]
raw=(root/'wp4_022_config.json').read_bytes(); assert hashlib.sha256(raw).hexdigest()==cfg_sha
p=json.loads(raw)
for row in p['protection_files'].values():
 data=Path(row['path']).read_bytes(); assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
old=root.parent/'wp4_017_20261015';sys.path.insert(0,str(old/'scripts'))
import wp4_017_controller as inherited
ops=inherited.operations();sys.path.insert(0,str(root/'scripts'))
import run_wp4_022 as driver
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
ops.__dict__.update(RUN_ID='wp4_022_20261016',REMOTE=str(root),CFG='wp4_022_config.json',MANIFEST='WP4_022_SOURCE_MANIFEST.json',DRIVER='scripts/run_wp4_022.py',CONTROLLER='REMOTE_SUPERVISOR.py',validate_config=driver.validate_settings022)
inherited.REMOTE=str(root)
ops.STATE_NAMES=ops.STATE_NAMES+('runs/primary/artifacts/CONTRAST022.json','runs/primary/artifacts/ARTIFACT_INVENTORY.json','runs/primary/artifacts/T2/EVALUATION.json','runs/primary/artifacts/T2/EVALUATION_AXES.json',)
p,state=inherited.native_snapshot(dict(config_sha256=cfg_sha,source_manifest_sha256=mf_sha,controller_sha256=ops_sha))
state['supervisor_console_tail']=(root/'supervisor_console.log').read_text()[-8000:]
log=root/'runs/primary/console.log'
state['worker_console_tail']=log.read_text()[-16000:] if log.exists() else None

native=state['runs/primary/supervisor/exit_receipt.json'];failure=state['runs/primary/artifacts/FAILED.json'];dur=state['durable_child_progress']
assert native['child_native_exit_code']==native['exit_code']==1 and native['reason'] is None and native['error'] is None
assert state['runs/primary/artifacts/COMPLETE.json'] is None and failure['error_type']=='ImportError' and "Z95" in failure['error']
assert not state['live_processes']['supervisor']['alive'] and not state['live_processes']['worker']['alive']
assert not dur['failed_ids'] and not dur['unknown_inflight_ids'] and not dur['truncated_tail']
assert failure['counters']==failure['attempted_counters'] and failure['operations_entered']==failure['operations_completed']==26393
for key,amount in failure['counters'].items():assert dur['completed_counts'].get(key,0)==amount and amount<=p['budget_caps'][key],key
for key in ('evaluation_signal_rows','noise005_rows','T2_teacher_sign_uniform_requests','total_scalar_normal_requests','coverage_PIT_uniform_requests','hard_ADC_rows','evaluation_kernel_snapshots','new_teacher_kernels'):
 assert failure['counters'][key]==p['budget_caps'][key],key
assert failure['counters']['evaluation_energy_pair_distances']==27004928 and failure['counters']['evaluation_observable_pair_distances']==47258624
assert native['peak_RSS_bytes']<=8*1024**3 and native['peak_GPU_bytes']<=10*1024**3 and native['elapsed_seconds']<=7200 and native['peak_output_bytes']<=5120*1024**2
ops.COLLECTION_LIMIT=128*1024**2
inventory,selected,omitted,transport=ops.archive_inventory(state,p,partial=True,include_heads=False)
for path in sorted(root.rglob('*')):
 assert not path.is_symlink()
 if path.is_file():
  name=path.relative_to(root).as_posix();row=ops.named_record(path)
  if name in inventory:assert inventory[name]==row
  inventory[name]=row
selected,omitted=ops.select_inventory(inventory,include_heads=False)
compact=['runs/primary/artifacts/ORIGINAL_PHYSICAL_WEIGHTS.npz','runs/primary/artifacts/PARENT_SCORES022_fast.npz','runs/primary/artifacts/T2/EVALUATION_AXES.npz','runs/primary/artifacts/T2/EVALUATION_RANDOM_STATES.npz']
for name in compact:
 assert inventory[name]['bytes']<=8*1024**2;selected[name]=inventory[name];omitted.pop(name)
assert 'runs/primary/artifacts/operations.jsonl' in selected and 'runs/primary/console.log' in selected
assert sum(row['bytes'] for row in selected.values())<=128*1024**2
assert not any('HARD_ROWS.npz' in n or '_OBSERVABLE.npz' in n or '_REPORT.npz' in n or '_BASELINE100.npz' in n for n in selected)
import tarfile
archive=root.parent/'wp4_022_failed_selected_20261016_1325_01.tar'
with tarfile.open(archive,'x') as tar:
 for name in sorted(selected):ops.add_regular(tar,root/name,name)
assert archive.stat().st_size<=128*1024**2
teacher=ops.core().read_json(root/'runs/primary/artifacts/T2/EVALUATION.json')
assert teacher['teacher']=='T2'
print(json.dumps(dict(whole=inventory,selected=selected,omitted=omitted,archive_path=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=ops.sha(archive),full_remote_inventory_SHA_verified=True,producer_ARTIFACT_INVENTORY_exists=False,full_operation_log_included=True,selection_payload_bytes=sum(x['bytes'] for x in selected.values()),native=native,durable=dur,T2_completed_evaluation=teacher,old017_science_validator_skipped_for_new022_scope=True,all_large_raw_arrays_retained_remote=True,scientific_acceptance=False)))
