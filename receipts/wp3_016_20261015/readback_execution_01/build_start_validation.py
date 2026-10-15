"""Interpret the one saved CPU snapshot locally; no further host access."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;REG=STAGE/'readback_registration_01'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
p=read(REG/'READBACK_EXECUTION.json');pre=read(HERE/'LOCAL_PRELAUNCH_VALIDATION.json')
start=read(STAGE/'readback_start_01/START_RECEIPT.json');prepare=read(STAGE/'readback_prepare_01/PREPARE_RECEIPT.json')
state_path=STAGE/'readback_start_verification_01/REMOTE_STATE.json';state=read(state_path)
wait=read(HERE/'FIRST_PAIR_WAIT_RECEIPT.json');first=state['artifacts/FIRST_PAIR_PROOF.json'];done=state['artifacts/COMPLETE.json']
native=state['supervisor/exit_receipt.json'];durable=state['durable_progress'];verdict=state['first_pair_verification'];limits=p['resource_limits']
assert verdict['first_formal_array_pair_verified'] and verdict['same_snapshot_health']=='native_completed_zero'
assert native['child_native_exit_code']==native['exit_code']==0 and native['reason'] is native['error'] is None
assert state['artifacts/FAILED.json'] is None and done['native_status']=='COMPLETE' and done['passed'] is True
assert done['counters']==done['attempted_counters']==p['expected_complete_counters'] and not done['inflight']
assert done['operations_entered']==done['operations_completed']==p['expected_operations']==200
assert durable['unknown_inflight_ids']==durable['failed_ids']==[] and durable['truncated_tail'] is False
assert all(durable['entered_counts'].get(k,0)==v==durable['completed_counts'].get(k,0) for k,v in done['counters'].items())
assert all(native['entered_counts'].get(k,0)==v==native['completed_counts'].get(k,0) for k,v in done['counters'].items())
assert first['counters']['charge_npz_files_decoded']>=2 and first['counters']['bank_pairs_verified']>=1
assert first['counters']['raw_files_SHA_checked']==112 and first['scientific_CP_NN_AD_training']==0
assert native['elapsed_seconds']<=limits['wall_seconds'] and native['peak_owned_RSS_bytes']<=limits['RSS_MiB']*1024**2
assert native['peak_worker_artifact_bytes']<=limits['output_MiB']*1024**2
assert start['registration_sha256']==state['registration_sha256']==pre['config_sha256']
assert state['supervisor/launch.json']['child_pid']==first['pid']==native['child_pid']
result=dict(schema='n01/wp3-016-readback-start-validation-summary/v1',recorded_UTC=datetime.now(timezone.utc).isoformat(),passed=True,
    status='NATIVE0_COMPLETE_AT_ONLY_START_SNAPSHOT_NOT_YET_COLLECTED',run_id=p['run_id'],source_commit=pre['source_commit'],
    configuration_sha256=pre['config_sha256'],controller_sha256=pre['controller_sha256'],sources=p['files'],source_package_sha256=prepare['package_sha256'],
    remote_root=p['remote_root'],original_root=p['original_root'],original_training_source_commit=p['original_training_source_commit'],
    original_configuration_sha256=p['original_configuration_sha256'],original_source_manifest_sha256=p['original_source_manifest_sha256'],
    dispatch_UTC=start['UTC'],worker_launch_UTC=state['supervisor/launch.json']['UTC'],first_formal_pair_UTC=first['UTC'],
    presence_observed_UTC=wait['remote_wait_result']['observed_UTC'],only_snapshot_UTC=state['UTC'],worker_complete_UTC=done['UTC'],supervisor_exit_UTC=native['UTC'],
    supervisor_pid=start['supervisor_pid'],worker_pid=first['pid'],boot_id=state['current_boot_id'],
    supervisor_start_ticks=start['supervisor_start_ticks'],worker_start_ticks=state['supervisor/launch.json']['child_start_ticks'],
    same_snapshot_health=verdict['same_snapshot_health'],first_pair_verification=verdict,first_pair_proof=first,
    required_charge_checks_completed_at_first_pair=first['counters']['charge_npz_files_decoded'],
    terminal_completed_counters=done['counters'],terminal_entered_counters=done['attempted_counters'],
    operations_entered=done['operations_entered'],operations_completed=done['operations_completed'],expected_journal_records_after_collection=400,
    native_durable_progress=durable,actual_inflight={},unknown_inflight_ids=[],failed_ids=[],truncated_tail=False,
    exact_closed_terminal_counters_match_registered_scope=True,producer_COMPLETE=done,native_exit=native,
    resources=dict(supervisor_elapsed_seconds=native['elapsed_seconds'],supervisor_sampled_peak_owned_RSS_bytes=native['peak_owned_RSS_bytes'],
        peak_worker_artifact_bytes=native['peak_worker_artifact_bytes'],limits=limits,last_supervisor_sample=state['supervisor/resource_status.json'],
        worker_getrusage_peak_not_in_standard_snapshot=True,current_RSS_VM_CPU_affinity_after_exit_not_available=True,
        VM3072MiB_and_CPU2_enforced_by_registered_child_and_worker_code=True,GPU_work_budget=0,GPU_usage_not_queried=True),
    live_processes_at_terminal_snapshot=state['live_processes'],preflight=start['preflight'],
    completed_scope=dict(raw_SHA_files=112,raw_SHA_bytes=2169655337,raw_decode_files=112,hard_files=48,REPORT_files=48,baseline_files=8,charge_banks=8,
        hard_bank_pairs=24,hard_rows=1265856,actual_A384_rows=210976,actual_A_float64_cells=81014784,mu_draws=210976,q_draws=210976,parent_moments=52744),
    ARRAY_READBACK_declared_sha256=done['ARRAY_READBACK_sha256'],ARRAY_READBACK_locally_collected=False,
    result_content_full_SHA_readback_pending_collection=True,independent_saved_array_worker_complete=True,
    calls=dict(prepare_CLI=1,prepare_SSH=1,install_SSH=1,source_SCP=1,start_CLI_SSH=1,presence_wait_SSH=1,snapshot_CLI_SSH=1,
        total_SSH=5,total_SCP=1,explicit_current_snapshots=1,status_progress_polls=0,presence_content_reads=0,
        collect=0,automatic_retry=False,second_CPU_worker=0,training_start=0,extra_scientific_probe=0),
    scientific_budget=p['scientific_budget'],cumulative_training_before_after=179667,training_counter_delta=0,
    main_metadata_automation_or_sources_modified=False,all_remote_calls_stopped=True,observations_after_only_snapshot='unknown',
    root_next_action='Audit already terminated0. Do not schedule a running-job wake; root may authorize one small-output collection.',
    scientific_acceptance=False,weights_promoted=False,
    receipts={name:ref(path) for name,path in [('local_prelaunch',HERE/'LOCAL_PRELAUNCH_VALIDATION.json'),('prepare',STAGE/'readback_prepare_01/PREPARE_RECEIPT.json'),
        ('start',STAGE/'readback_start_01/START_RECEIPT.json'),('presence',HERE/'FIRST_PAIR_WAIT_RECEIPT.json'),('snapshot',state_path),
        ('source_commit',REG/'SOURCE_COMMIT_READBACK.json'),('D_source_backup',REG/'D_SOURCE_BACKUP_RECEIPT.json')]})
path=HERE/'START_VALIDATION_SUMMARY.json'
with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
md=f'''# 016 CPU审计在唯一启动快照已原生完成

首个正式pair {first['UTC']}；两bank charge检查在该证明前完成。唯一快照 {state['UTC']} 已是完整native0：COMPLETE {done['UTC']}，supervisor退出 {native['UTC']}，200项操作 entered=completed，全部16计数匹配，inflight/failed空。

112个原始NPZ共2,169,655,337B各SHA一次并decode一次：48HARD、48REPORT、8baseline、8charge；24双bankpair及210,976条实际A384记录完成。实际训练、模型、CP、AD、新随机、原TRAIN/DEV读取和GPU预算均0，累计G保持179667。

原生用时 {native['elapsed_seconds']:.3f}s，supervisor采样峰RSS {native['peak_owned_RSS_bytes']}B，输出 {native['peak_worker_artifact_bytes']}B，低于原cap。退出后无当前RSS/VM/CPU亲和性值；VM3072MiB和CPU2由登记子进程与worker约束，不把不存在的实时测量填0。

本次prepare/start/presence/snapshot各一次，SSH5/SCP1，全部远端调用已停止。ARRAY_READBACK声明SHA `{done['ARRAY_READBACK_sha256']}`，尚待根授权小文件回收；本地未回收原始NPZ。无需运行中唤醒。科学接受与权重晋级仍待独审闭合。
'''
with (HERE/'START_VALIDATION_SUMMARY.md').open('x',encoding='utf8') as f:f.write(md)
print(json.dumps(dict(passed=True,status=result['status'],summary_sha256=sha(path),completed_pairs=24,operations=200,collect_pending=True)))
