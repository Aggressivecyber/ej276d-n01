"""Reconcile the completed016 saved-array audit and reopen only small collected files."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))

def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),**record(path))

config_path=OUT/'READBACK_EXECUTION.json';config=read(config_path)
source=read(OUT/'SOURCE_COMMIT_READBACK.json')
state_path=BASE/'readback_start_verification_01/REMOTE_STATE.json';state=read(state_path)
spec=importlib.util.spec_from_file_location('root016_readback_complete',config['files']['readback_control.py']['local_path'])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.first_valid(state,config,current_boot=False)==state['first_pair_verification']
assert state['registration_sha256']==sha(config_path)==source['configuration_sha256']
complete=state['artifacts/COMPLETE.json'];native=state['supervisor/exit_receipt.json']
assert complete['native_status']=='COMPLETE' and complete['passed'] is True and not complete['inflight']
assert complete['scientific_CP_NN_AD_training']==0
assert complete['operations_entered']==complete['operations_completed']==200
assert complete['counters']==complete['attempted_counters']==config['expected_complete_counters']
assert state['durable_progress']['unknown_inflight_ids']==state['durable_progress']['failed_ids']==[]
assert state['durable_progress']['truncated_tail'] is False
assert all(v==0 for v in config['scientific_budget'].values())
assert native['child_native_exit_code']==native['exit_code']==0
for name,row in source['files'].items():assert record(ROOT/name)==row
collection_dir=BASE/'readback_collection_01'
collection_path=collection_dir/'COLLECTION_RECEIPT.json';collection=read(collection_path)
remote_archive=read(collection_dir/'REMOTE_ARCHIVE.json')
collected=Path(collection['collected_root'])
assert collected==ROOT/'runs/wp3_016_20261015/readback_collected_01'
assert collection['original_raw_NPZ_transferred'] is False and not collection['partial_evidence_only']
archive=collection_dir/'readback_output_01.tar'
assert sha(archive)==collection['archive_sha256']==remote_archive['archive_sha256']
inventory=remote_archive['inventory']
assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(inventory)
for name,row in inventory.items():
    path=collected/name
    assert path.is_relative_to(collected) and path.suffix not in ['.npz','.pt','.pth']
    assert record(path)==row,name
array_path=collected/'artifacts/ARRAY_READBACK.json';array=read(array_path)
assert sha(array_path)==complete['ARRAY_READBACK_sha256']
assert array['passed'] is True
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
receipt=dict(schema='n01/wp3-016-CPU-terminal-reconciliation/v1',UTC=stamp,
    native_snapshot=ref(state_path),array_result=ref(array_path),array_result_passed=True,
    native_child_exit_code=0,native_supervisor_exit_code=0,completed_UTC=native['UTC'],
    operations_entered=200,operations_completed=200,actual_counts=complete['counters'],attempted_counts=complete['attempted_counters'],
    unknown_inflight_ids=[],failed_ids=[],truncated_tail=False,
    first_pair_verification=state['first_pair_verification'],
    collection=dict(receipt=ref(collection_path),archive_sha256=sha(archive),files=len(inventory),
        bytes=sum(r['bytes'] for r in inventory.values()),all_files_SHA256_and_bytes_verified=True,
        raw_NPZ_transferred=False,collected_root=str(collected)),
    terminal_resources={k:native[k] for k in ['elapsed_seconds','peak_owned_RSS_bytes','peak_worker_artifact_bytes','resource_limits']},
    worker_self_reported_resources=array['resources'],
    scientific_budget=config['scientific_budget'],new_G_updates=0,new_AD_queries=0,cumulative_training_updates=179667,
    complete_CDF_loss_or_C1_median_replay=False,original_SOURCE_waveform_reads=0,checkpoint_tensor_loads=0,
    final_independent_review_pending=True,whole_project_complete=False,scientific_status='response_match_not_established')
receipt_path=OUT/'CPU_COMPLETION_RECONCILIATION.json';assert not receipt_path.exists()
paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
before=OUT/'metadata_before_cpu_completion';before.mkdir(exist_ok=False)
for path in paths:shutil.copy2(path,before/path.name)
write(receipt_path,receipt,True)

def finished(previous):
    value=copy.deepcopy(previous)
    value.update(status='completed_native0_collected_final_local_review_pending',source_commit=source['source_commit'],source_commit_pending=False,
        historical_analysis_pid=native['child_pid'],historical_supervisor_pid=native['supervisor_pid'],analysis_pid=None,supervisor_pid=None,
        last_verified_UTC=state['UTC'],actual_completed_counts_last_verified=complete['counters'],attempted_counts_last_verified=complete['attempted_counters'],
        unknown_inflight_ids=[],failed_ids=[],counts_are_durable_lower_bounds=False,
        actual_current_completion='native0_COMPLETE_all200_operations_completed',native_child_exit_at_snapshot=0,native_supervisor_exit_at_snapshot=0,
        native_exit_code=0,supervisor_exit_code=0,completed_UTC=native['UTC'],terminal_readback=ref(receipt_path),
        collection=receipt['collection'],collection_complete=True,bounded_saved_array_readback_complete=True,
        final_independent_local_review_complete=False,original_raw_NPZ_transferred=False,
        complete_distribution_loss_raw_replay=False,actual_terminal_resources=receipt['terminal_resources'],
        scientific_operations=config['scientific_budget'],CP_NN_AD_training=0,scientific_acceptance=False)
    return value

for path in [p for p in paths if p.suffix=='.json']:
    value=read(path)
    value.update(updated_UTC=stamp,active_scientific_jobs=0,
        active_scientific_jobs_status_basis='Collected native0 CPU016, all200 operations complete',
        followup_last_delivered_UTC='2026-10-15T08:45:06.316Z',current_followup_status='single_occurrence_delivered_this_turn_no_pending_long_job',
        scientific_status='response_match_not_established')
    if isinstance(value.get('followup'),dict):
        value['followup'].update(delivery_status='single_0845UTC_wake_delivered_this_turn',saved_app_status_is_historical=True)
    if path.name=='jobs.json':
        job=next(j for j in value['jobs'] if j.get('run_id')==config['run_id'])
        job.update(status='completed_native0_collected_final_local_review_pending',native_exit_code=0,supervisor_native_exit_code=0,
            completed_UTC=native['UTC'],historical_analysis_pid=native['child_pid'],historical_supervisor_pid=native['supervisor_pid'],
            analysis_pid=None,supervisor_pid=None,actual_array_counts=complete['counters'],collection_complete=True,
            terminal_readback=ref(receipt_path))
        job['execution']=finished(job['execution'])
    else:
        assert value['new_training_updates']==179667
        for key in ['wp3_readback_execution','wp3_016_readback_execution']:value[key]=finished(value[key])
        value.update(status='WP3_016_CPU_complete_final_review_report_in_progress',active_audit=None,
            active_research='wp3_016_20261015_closeout',current_work_package='WP3',
            current_stage='016_saved_array_complete_final_review_report_pending',current_run_training_updates=0,
            next_action='Finish016 independent review/report, then WP4 known-response recovery; no more WP3 candidates.')
        if path.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            wp.update(status='016_native0_array_readback_complete_final_review_report_pending')
            for key in ['saved_array_readback','readback016']:wp[key]=finished(wp[key])
    write(path,value)
readme=ROOT/'README.md';previous=readme.read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in previous
lead='''# N01 当前执行：WP3/016 数组复核完成，正在闭合报告

016两个seed各2000G，实际4000G/AD已闭合，累计179667；WP3短试6/6、全部短候选10/12。当前无运行中的科学作业。

保存数组CPU核验已原生退出0并回收：112份NPZ各SHA/解码一次、24组硬统计、8份电荷矩、200项操作完整；新增CP/NN/AD/G/RNG/GPU和原SOURCE/REAL TRAIN/DEV波形读入为0。小结果逐SHA重开核验，原2.17GB NPZ保留远端。

Q1仍为待最终独审闭合的研究参考选择：mean12 wave=0.15011008209012994，较M0改善约0.009976%；ALL6双参考非劣和相对A用途门通过，绝对T未全部通过。当前不宣称科学晋级；完成016报告后推进WP4，WP3不追加候选。

[CPU闭合回执](receipts/wp3_016_20261015/readback_registration_01/CPU_COMPLETION_RECONCILIATION.json) · [016原生终点](receipts/wp3_016_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json)

'''
readme.write_text(lead+previous[previous.index(marker):],encoding='utf-8',newline='\n')
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for path in [ROOT/'PROGRAM.json',CURRENT]:
    value=read(path);value['master_plan_sha256']=master_sha
    if path==CURRENT:value['plan_sha256']=sha(readme)
    write(path,value)
print(json.dumps(dict(status='CPU_native0_reconciled_no_new_G',selected_files=len(inventory),
    selected_bytes=sum(r['bytes'] for r in inventory.values()),operations=200,cumulative_G=179667,active_jobs=0)))
