"""Local terminal JSON, hashes and journal only. Never decode NPZ/PT tensors."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as c
read=lambda p:json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def save(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def types(inv):
    result={}
    for name,row in inv.items():
        suffix=Path(name).suffix or '[no suffix]';part=result.setdefault(suffix,dict(files=0,bytes=0));part['files']+=1;part['bytes']+=row['bytes']
    return result

freeze=STAGE/'runtime_package_fix02';p=read(freeze/c.CFG);fr=read(freeze/'FREEZE_RECEIPT.json')
pre=read(STAGE/'native_execution_03/LOCAL_PRELAUNCH_VALIDATION.json')
assert all(fr[k]==v for k,v in pre['pins'].items()) and sha(ROOT/c.CONTROLLER)==pre['pins']['controller_sha256']
state_path=STAGE/'wake_status_01/REMOTE_STATE.json';state=read(state_path);verdict=c.complete_valid(state,p)
done=state['runs/primary/artifacts/COMPLETE.json'];native=state['runs/primary/supervisor/exit_receipt.json'];progress=state['durable_child_progress']
collection_path=STAGE/'collection_01/COLLECTION_RECEIPT.json';collection=read(collection_path)
inventory_path=STAGE/'collection_01/REMOTE_FULL_INVENTORY.json';inventory=read(inventory_path)
collected=Path(collection['collected_root']);art=collected/'runs/primary/artifacts';selected=inventory['inventory'];whole=inventory['whole_remote_inventory'];omitted=inventory['omitted_remote_inventory']
assert collection['status']=='native_inventory_and_selected_bytes_verified' and collection['partial_evidence_only'] is False
assert set(selected).isdisjoint(omitted) and set(selected)|set(omitted)==set(whole)
assert all(whole[k]==v for k,v in {**selected,**omitted}.items())
assert sha(Path(collection['archive']))==collection['archive_sha256'] and Path(collection['archive']).stat().st_size<=256*1024**2
for name,row in selected.items():
    path=collected/name;assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],name
assert not any(k.endswith('.npz') for k in selected)
raw={k:v for k,v in whole.items() if k.endswith('.npz')};assert raw==omitted
raw_kinds={suffix:dict(files=sum(k.endswith(suffix) for k in raw),bytes=sum(v['bytes'] for k,v in raw.items() if k.endswith(suffix)))
    for suffix in ['_HARD_ROWS.npz','_REPORT.npz','_BASELINE100.npz','_CHARGE_PREADC_MOMENTS.npz']}
assert sum(v['files'] for v in raw_kinds.values())==len(raw)
raw_path=HERE/'RAW_NPZ_INVENTORY.json'
save(raw_path,dict(schema='n01/wp3-016-raw-NPZ-inventory/v1',remote_root=c.REMOTE,source_inventory=ref(inventory_path),
    files=len(raw),bytes=sum(v['bytes'] for v in raw.values()),kinds=raw_kinds,entries=raw,
    locally_transferred=0,decoded_by_wake_agent=0,all_raw_retained_remote=True))
save(HERE/'SELECTED_INVENTORY.json',dict(schema='n01/wp3-016-selected-inventory/v1',source_inventory=ref(inventory_path),
    collected_root=str(collected),files=len(selected),bytes=sum(v['bytes'] for v in selected.values()),types=types(selected),entries=selected,all_local_size_SHA_verified=True))

entered=Counter();completed=Counter();active={};seen=set();records=0
with (art/'operations.jsonl').open(encoding='utf8') as f:
    for line in f:
        row=json.loads(line);records+=1;key=row['operation_id']
        if row['event']=='entered':
            assert key not in seen;seen.add(key);active[key]=row;entered.update(row['charges'])
        elif row['event']=='completed':
            assert key in active and active[key]['charges']==row['charges'];active.pop(key);completed.update(row['charges'])
        else:raise AssertionError('unexpected journal event '+row['event'])
assert not active and entered==completed and records==2*len(seen)
assert all(entered.get(k,0)==v==completed.get(k,0) for k,v in p['budget_caps'].items())
assert done['counters']==done['attempted_counters']==p['budget_caps']
assert progress['unknown_inflight_ids']==progress['failed_ids']==[] and progress['truncated_tail'] is False
assert sha(art/'COMPLETE.json')==sha(collected/'runs/primary/artifacts/status.json') or read(art/'status.json')['counters']==done['counters']
trials=[]
for trial in done['trials']:
    path=art/trial['directory']/'COMPLETE.json';assert sha(path)==trial['COMPLETE_sha256']
    record=dict(trial,receipt=ref(path),inventory_files=sum(k.startswith('runs/primary/artifacts/'+trial['directory']+'/') for k in whole),
        inventory_bytes=sum(v['bytes'] for k,v in whole.items() if k.startswith('runs/primary/artifacts/'+trial['directory']+'/')))
    assert record['inventory_bytes']<p['run_resource_limits']['output_MiB']*1024**2;trials.append(record)
old_failure=read(STAGE/'failure_collection_01/FAILURE_PRESERVATION_RECEIPT.json')
aggregate={k:done['counters'].get(k,0)+old_failure['actual_completed_counters'].get(k,0) for k in set(done['counters'])|set(old_failure['actual_completed_counters'])}
calpath=art/'CHARGE_INITIALIZATION.json';cal=read(calpath)
numeric_paths=sorted(art.glob('seed*/*_evaluation/*_A_NUMERIC_REPEATS.json'));assert len(numeric_paths)==8
numeric=[dict(source=ref(path),actual_A_max_abs_V=read(path)['actual_max_abs_residual_V'],actual_mu_repeat_absmax_Vns=read(path)['Q_mu_repeat_detail']['actual_absmax'],
    within_013_gate=read(path)['within_numerical_tolerance'],Q_CPU_GPU_within_bound=read(path)['Q_mu_CPU_GPU_within_derived_bound']) for path in numeric_paths]
assert all(row['within_013_gate'] and row['Q_CPU_GPU_within_bound'] for row in numeric)

oldstage=ROOT/'receipts/wp3_015_20261015'
reuse_paths={
    'worker':oldstage/'postrun_independent_01/readback_saved_arrays.py',
    'kernel':oldstage/'postrun_independent_01/saved_array_kernels.py',
    'controller':oldstage/'readback_engineering_01/readback_control.py',
    'controller_tests':oldstage/'readback_engineering_01/check_readback_control.py',
    'preparation_example':oldstage/'postrun_independent_01/prepare_saved_readback015.py',
    'registration_example':oldstage/'readback_registration_01/register_readback.py',
    'plan_example':oldstage/'postrun_independent_01/READBACK_PLAN.json',
    'native_completion_evidence':oldstage/'readback_execution_01/WAKE_COMPLETION_SUMMARY.json'}
reuse=dict(schema='n01/wp3-016-CPU-readback-reuse-entrypoints/v1',status='INFORMATION_ONLY_NOT_IMPLEMENTED_NOT_STARTED',
    mature015_sources={name:ref(path) for name,path in reuse_paths.items()},
    worker_APIs=['decode_window','lineage','wave_moments','Audit.member','Audit.identities','Audit.load_baselines','Audit.one_bank','Audit.pair','main'],
    controller_APIs=['validate','gate','old_small_gate','process_identity','first_valid','snapshot','complete_valid','supervise','start','action'],
    existing_scope='saved ADC float64 roundtrip and row identity/features/flags; native4draw and derived8draw 9x9 finite pair covariance; byte SHA input protection',
    required016_adaptation='Bind new frozen root/pins/actual112-file inventory and 24 hard-bank pairs. Add separately reviewed charge-array validation for actual A384/mu/v and saved per-draw statistics; existing015 worker does not yet validate new charge schema.',
    actual_raw_inventory=ref(raw_path),whole_inventory=ref(inventory_path),raw_files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values()),
    actual_hard_files=raw_kinds['_HARD_ROWS.npz']['files'],actual_report_files=raw_kinds['_REPORT.npz']['files'],
    actual_baseline_files=raw_kinds['_BASELINE100.npz']['files'],actual_charge_bank_files=raw_kinds['_CHARGE_PREADC_MOMENTS.npz']['files'],
    expected_hard_bank_pairs_from_exact_path_pairs=24,charge_z_or_full_stochastic_carrier_not_saved=True,
    no_original_raw_NPZ_transfer_or_tensor_checkpoint_load=True,no_NN_AD_random_training_loss_CDF_or_median_replay=True,
    CPU_source_modified_or_implemented=False,CPU_started=False,SSH_calls_for_this_information=0,root_registration_required_before_any_CPU_run=True)
save(HERE/'CPU_READBACK_REUSE_ENTRYPOINTS.json',reuse)
result=dict(schema='n01/wp3-016-wake-completion-summary/v1',recorded_UTC=datetime.now(timezone.utc).isoformat(),
    status='native0_two_trials_COMPLETE_selected_SHA_verified',run_id=c.RUN_ID,source_commit=pre['source_commit'],pins=pre['pins'],remote_root=c.REMOTE,
    source_files=116,source_controls=118,protected_records=475,wake_snapshot_UTC=state['UTC'],producer_complete_UTC=done['UTC'],native_exit_UTC=native['UTC'],
    collection_inventory_UTC=inventory['UTC'],terminal_validation=verdict,native_child_exit_code=native['child_native_exit_code'],native_supervisor_exit_code=native['exit_code'],
    native_reason=native['reason'],native_error=native['error'],supervisor_pid=native['supervisor_pid'],child_pid=native['child_pid'],
    launch_identity=state['runs/primary/supervisor/launch.json'],boot_id=state['current_boot_id'],live_processes_at_snapshot=state['live_processes'],
    actual_terminal_completed_counters=done['counters'],actual_terminal_entered_counters=done['attempted_counters'],
    operations_entered=len(seen),operations_completed=len(seen),journal_records=records,journal_counts_independently_replayed_locally=True,
    actual_inflight={},unknown_inflight_ids=[],failed_operation_ids=[],journal_truncated_tail=False,counts_are_exact_closed_terminal=True,
    native_durable_progress=progress,trial_records=trials,
    resource=dict(elapsed_seconds=native['elapsed_seconds'],peak_RSS_bytes=native['peak_RSS_bytes'],peak_GPU_bytes=native['peak_GPU_bytes'],
        peak_output_bytes=native['peak_output_bytes'],whole_inventory_bytes=sum(v['bytes'] for v in whole.values()),limits=p['resource_limits'],
        terminal_peak_measurements_distinct_from_later_inventory=True,current_live_metrics_after_exit_not_queried=True),
    protections=dict(errors=state['protection_errors'],producer_all_frozen_A_005_unchanged=done['all_frozen_A_005_unchanged'],
        producer_all_M0_heads_unchanged=done['all_M0_heads_unchanged'],all_counter_caps_exact=True,original_failure_evidence_preserved=True),
    producer_charge_initialization=dict(source=ref(calpath),parents=cal['parents'],Q_min_Vns=cal['Q_mu_min_Vns'],Q_max_Vns=cal['Q_mu_max_Vns'],
        all_Q_positive=cal['all_Q_mu_finite_strictly_positive'],actual_expected_RMS_V=cal['actual_A_expected_increment_RMS_V'],target_RMS_V=cal['target_RMS_V'],
        independent_array_recomputation_not_yet_performed=True),producer_saved_A_repeat_diagnostics=numeric,
    collection=dict(receipt=ref(collection_path),whole_inventory=ref(inventory_path),whole_files=len(whole),whole_bytes=sum(v['bytes'] for v in whole.values()),
        whole_types=types(whole),selected_files=len(selected),selected_bytes=sum(v['bytes'] for v in selected.values()),selected_types=types(selected),
        omitted_files=len(omitted),omitted_bytes=sum(v['bytes'] for v in omitted.values()),omitted_types=types(omitted),
        archive_bytes=Path(collection['archive']).stat().st_size,archive_sha256=collection['archive_sha256'],all_selected_local_sizes_SHA_verified=True,
        selected_inventory=ref(HERE/'SELECTED_INVENTORY.json'),raw_NPZ_inventory=ref(raw_path),raw_kinds=raw_kinds,
        raw_NPZ_selected_or_decoded=0,thin_checkpoint_files_hashed_only=sum(k.endswith('.pt') for k in selected),checkpoint_tensors_loaded=0,
        all_raw_retained_remote=True,all_raw_collected=False,independent_numerical_readback_complete=False,partial_evidence_only=False),
    old_failed_native_completed_counters=old_failure['actual_completed_counters'],
    combined_failed_plus_completed_native_counters=aggregate,old_failed_native_costs_not_erased=True,
    root_ledger_hint=dict(job_training_before=175667,already_booked_at_start=137,already_booked_cumulative=175804,
        full_job_completed_G=4000,add_this_wake_G=3863,expected_cumulative_after_root_update=179667,ledger_mutated_by_this_agent=False),
    calls_this_wake=dict(status_CLI_SSH=1,collect_CLI_SSH=1,collection_SCP=1,total_SSH=2,total_SCP=1,inspect=0,
        explicit_current_status_snapshots=1,collection_terminal_validation_snapshot=1,status_poll=0,start=0,deploy=0,scientific_probe=0,
        all_SSH_status_poll_stopped=True),
    all016_native_call_history=dict(deploy_CLI=3,start_CLI=2,presence_wait_SSH=2,verify_start_CLI=2,wake_status_CLI=1,
        collect_failed_CLI=1,collect_complete_CLI=1,total_SSH=14,total_SCP=4,automatic_scientific_retry=False),
    CPU_reuse_entrypoints=ref(HERE/'CPU_READBACK_REUSE_ENTRYPOINTS.json'),CPU_implementation_or_start_performed=False,
    main_metadata_or_source_changed=False,observations_after_collection='unknown; all remote calls stopped',
    scientific_acceptance=False,weights_promoted=False,independent_development_selection_pending=True)
save(HERE/'WAKE_COMPLETION_SUMMARY.json',result)
md=f'''# 016 原生完成与选收

唯一 wake 快照 {state['UTC']} 确认 child/supervisor 均 exit0、COMPLETE 齐全，两 trial 各2000G，共4000G/AD。{len(seen)}项操作 entered=completed，journal {records}行，inflight/failed均空，计数逐项等于冻结预算。

原生结束 {native['UTC']}，用时 {native['elapsed_seconds']:.3f}s；峰RSS {native['peak_RSS_bytes']}B、GPU {native['peak_GPU_bytes']}B、输出 {native['peak_output_bytes']}B，均在原cap。完整库存 {len(whole)}文件/{sum(v['bytes'] for v in whole.values())}B；选收 {len(selected)}文件/{sum(v['bytes'] for v in selected.values())}B，全SHA通过。遗漏的{len(raw)}个NPZ共{sum(v['bytes'] for v in raw.values())}B全部留远端，包括48 HARD、48 REPORT、8 baseline和8 charge bank；没有本地解码。6份薄CP仅字节核验。

旧失败G0仍保留A32965/head-noise-q26372/ADC237348/CP3等非零成本。根已登记137G，本次应增3863G，累计179667；本代理未改主台账。

本次仅status1、collect1，SSH2/SCP1，无启动、部署、探针或轮询。全部远端调用停止。CPU读回仅提供成熟015入口和精确112文件清单，尚未实现或启动；科学响应匹配和权重晋级待独审。
'''
with (HERE/'WAKE_COMPLETION_SUMMARY.md').open('x',encoding='utf8') as f:f.write(md)
print(json.dumps(dict(status=result['status'],summary_sha256=sha(HERE/'WAKE_COMPLETION_SUMMARY.json'),raw_inventory_sha256=sha(raw_path),
    raw_files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values()),selected_files=len(selected),operations=len(seen),journal_records=records,
    CPU_entrypoints_sha256=sha(HERE/'CPU_READBACK_REUSE_ENTRYPOINTS.json')),ensure_ascii=False))
