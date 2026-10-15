"""Register one bounded015 saved-array CPU audit; no remote action."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[3];BASE=OUT_PARENT=ROOT/'receipts/wp3_015_20261015'
OUT=Path(__file__).resolve().parent
SIDE='/home/rylan/projects/ej276d-n01/wp3_015_saved_readback_20261015'
RUN='wp3_015_saved_readback_20261015'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),**record(p))
def write(p,v,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

native=read(BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json')
collection=read(BASE/'collection_01/COLLECTION_RECEIPT.json')
local=read(BASE/'postrun_root_01/COLLECTION_READBACK.json')
selection_path=BASE/'postrun_independent_01/DEVELOPMENT_SELECTION.json';selection=read(selection_path)
plan_path=BASE/'postrun_independent_01/READBACK_PLAN.json';plan=read(plan_path)
ready_path=BASE/'postrun_independent_01/READBACK_WORKER_READY.json';ready=read(ready_path)
control_ready_path=BASE/'readback_engineering_01/CONTROLLER_READY.json';control_ready=read(control_ready_path)
astra_review_path=BASE/'postrun_independent_01/ASTRA_PRELAUNCH_REVIEW.json';astra_review=read(astra_review_path)
policy_path=OUT/'AGENT_MODEL_POLICY.json';policy=read(policy_path)
assert native['cumulative_training_updates']==175667 and native['actual_counts']['G_updates']==8000
assert collection['whole_remote_inventory_verified'] and collection['selected_file_count']==139
assert not collection['all_raw_artifacts_collected'] and collection['raw_artifacts_retained_remote']
assert local['all_selected_local_SHA256_and_size_verified'] and local['raw_npz_downloaded']==0
assert selection['selected_research_reference']=='M0' and selection['weights_promoted'] is False
assert selection['winner_better_than_common_initial_by_more_than_epsilon'] is True
assert selection['independent_raw_saved_array_readback_complete'] is False
assert ready['status'].startswith('ready_') and control_ready['status'].startswith('ready_')
assert astra_review['passed'] is True
assert policy['requested_model']=='gpt-6-astra' and policy['requested_reasoning_effort']=='max'
assert all(v==0 for v in plan['scientific_budget'].values())
expected=plan['expected_raw_files']
assert expected['total_npz_SHA161']==161 and expected['evaluation_NPZ_decode160']==160 and expected['bank_pairs']==36
assert expected['hard_rows_npz']==expected['report_npz']==72 and expected['baseline100_npz']==16
assert plan['finite_resource_budget']['CPU_threads']==2
assert plan['finite_resource_budget']['process_RSS_MiB']==1536 and plan['finite_resource_budget']['wall_seconds']==1800
program=read(ROOT/'PROGRAM.json');assert program['new_training_updates']==175667 and program['active_scientific_jobs']==0
assert read(ROOT/'MASTER_EXECUTION_PLAN.json')['budgets']['short_candidate_runs']['completed']==8
original=read(BASE/'SOURCE_COMMIT_READBACK.json')
for name,row in original['files'].items():assert record(ROOT/name)==row,name
inventory=OUT/'BOUND_ORIGINAL_INVENTORY.json';assert not inventory.exists()
shutil.copy2(BASE/'collection_01/REMOTE_FULL_INVENTORY.json',inventory)
assert sha(inventory)=='9d253a57bb79d9f25b112b4f5e0009a239c21360766c89cef7357d040adebe42'
paths={'readback_control.py':BASE/'readback_engineering_01/readback_control.py',
    'readback_saved_arrays.py':BASE/'postrun_independent_01/readback_saved_arrays.py',
    'saved_array_kernels.py':BASE/'postrun_independent_01/saved_array_kernels.py',
    'READBACK_PLAN.json':plan_path,'BOUND_ORIGINAL_INVENTORY.json':inventory}
for label,name in [('worker','readback_saved_arrays.py'),('kernel','saved_array_kernels.py'),('plan','READBACK_PLAN.json')]:
    assert record(paths[name])=={k:ready[label][k] for k in ['bytes','sha256']},label
    assert record(paths[name])=={k:control_ready[label][k] for k in ['bytes','sha256']},label
assert record(paths['readback_control.py'])=={k:control_ready['controller'][k] for k in ['bytes','sha256']}
assert record(astra_review_path)=={k:control_ready['independent_review'][k] for k in ['bytes','sha256']}
now=dt.datetime.now(dt.timezone.utc).isoformat()
config=dict(schema='n01/wp3-independent-readback-execution/v1',run_id=RUN,registered_UTC=now,status='registered_not_started',
    remote_root=SIDE,original_root='/home/rylan/projects/ej276d-n01/wp3_015_20261015',host='rylan@100.82.231.102',
    python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
    files={name:dict(path=SIDE+'/'+name,local_path=str(path),**record(path)) for name,path in paths.items()},
    resource_limits=dict(CPU_threads=2,RSS_MiB=1536,virtual_memory_MiB=3072,wall_seconds=1800,output_MiB=64),
    selected_collection_MiB_cap_including_controls_and_logs=80,original_files_read_only=True,automatic_retry=False,
    exclusive_project_lock='/home/rylan/projects/ej276d-n01/.n01_scientific_dispatch.lock',scientific_budget=plan['scientific_budget'],
    finite_scope=dict(bank_pairs=36,raw_NPZ=161,evaluation_NPZ_decoded=160,metadata_only_NPZ_SHA_checked=1,
        hard_rows_NPZ=72,REPORT_NPZ=72,baseline100_NPZ=16,probe_raw_NPZ=0,probe_gradient_NPZ=0,SOURCE_parents=6593,
        rows_per_hard_file=26372,waveform_rows_decoded=1898784,raw_SHA_passes=1,raw_decode_passes=1,
        raw_compressed_bytes=3064431039,input_compressed_envelope_MiB=plan['finite_resource_budget']['fixed_input_compressed_envelope_MiB'],
        input_IO_envelope_MiB=plan['finite_resource_budget']['input_IO_upper_envelope_MiB']),
    root_authority=dict(user_master_plan='USER_MASTER_PLAN_20261015.txt',user_master_plan_sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt'),
        execution_instruction='2026-10-15T05:20:03.197Z n01 heartbeat',
        latest_model_instruction='2026-10-15T05:32:13.145512Z all roles Astra Max',authorizes_one_bounded_saved_array_audit=True,
        no_additional_user_permission_required=True),
    original_training_source_commit=original['source_commit'],original_configuration_sha256=sha(ROOT/'wp3_015_config.json'),
    original_source_manifest_sha256=sha(ROOT/'WP3_SOURCE_MANIFEST.json'),original_training_completed_G=8000,
    original_training_completed_AD=8000,cumulative_training_before_and_after_this_audit=175667,
    original_raw_NPZ_remain_remote=True,original_raw_local_collection_complete=False,new_original_TRAIN_or_DEV_waveform_reads=0,
    raw_complete_wave_QPSD_noise_loss_replay=False,C1_median_fit_independently_replayed=False,
    preADC_fullcarrier_independent_reconstruction=False,
    loss_evidence='Frozen producer plus independent saved-JSON aggregation; this audit independently recomputes saved row observables and r4/r8 statistics only.',
    provisional_development_path='M0',mean_adapter_enabled=False,weights_promoted=False,
    prelaunch_worker_ready=ref(ready_path),prelaunch_controller_ready=ref(control_ready_path),
    astra_prelaunch_review=ref(astra_review_path),agent_model_policy=ref(policy_path),
    provisional_selection=ref(selection_path),native_training_closure=ref(BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json'),
    long_job_policy='Verify first formal saved-array pair and one same-snapshot health proof, register one future wake in same n01 and rest.',
    scientific_status='response_match_not_established',scientific_acceptance=False)
write(OUT/'READBACK_EXECUTION.json',config,True)
execution=dict(configuration=(OUT/'READBACK_EXECUTION.json').relative_to(ROOT).as_posix(),
    configuration_sha256=sha(OUT/'READBACK_EXECUTION.json'),run_id=RUN,remote_root=SIDE,resource_limits=config['resource_limits'],
    source_file_count=5,source_controls=6,source_commit_pending=True,status='registered_not_started',
    new_training_updates=0,new_AD_queries=0,NN_forward_rows=0,scientific_status='response_match_not_established')
cref=dict(**ref(BASE/'collection_01/COLLECTION_RECEIPT.json'),selected_files=139,full_inventory_files=300,
    omitted_raw_files=161,all_raw_collected=False,raw_NPZ_retained_remote=True)
sref=dict(**ref(selection_path),development_path='M0',status='provisional_saved_JSON_audit_raw_readback_pending',
    mean_adapter_enabled=False,weights_promoted=False,scientific_acceptance=False)
metas=[ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']
backup=OUT/'metadata_before_registration';backup.mkdir(exist_ok=False)
for path in metas:shutil.copy2(path,backup/path.name)
for path in [p for p in metas if p.suffix=='.json']:
    value=read(path);value.update(updated_UTC=now,active_scientific_jobs=0,scientific_status='response_match_not_established',
        agent_model_policy=ref(policy_path),latest_subagent_model_preference='GPT-6-Astra Max')
    training_status='completed_native0_selectively_collected_independent_raw_readback_pending'
    if path.name=='jobs.json':
        job=next(j for j in value['jobs'] if j.get('run_id')=='wp3_015_20261015')
        job.update(status=training_status,collection_complete=True,collection=cref,independent_readback_complete=False,provisional_development_selection=sref)
        job['execution'].update(status=training_status,collection_complete=True,collection=cref,independent_readback_complete=False)
        assert not any(j.get('run_id')==RUN for j in value['jobs'])
        value['jobs'].append(dict(job_id='N01_WP3_015_SAVED_ARRAY_READBACK_20261015',run_id=RUN,scientific_job=True,
            kind='CPU_saved_array_audit',status='registered_not_started',registered_UTC=now,execution=execution,
            resource_limits=config['resource_limits'],G_updates=0,new_AD_queries=0,NN_forward_rows=0,checkpoint_loads=0,
            automatic_retry=False,original_training_before_and_after=175667,scientific_status='response_match_not_established'))
    else:
        value.update(status='WP3_015_saved_array_readback_prepared',current_work_package='WP3',
            current_stage='WP3_015_independent_saved_array_readback_prepared',active_research=RUN,
            wp3_collection=cref,wp3_readback_execution=execution,wp3_provisional_development_selection=sref,
            latest_training_run_id='wp3_015_20261015',latest_training_source_commit=original['source_commit'],
            current_run_training_updates=0,new_training_updates=175667,cumulative_training_updates=175667,
            next_action='Start the bounded zero-NN CPU saved-array audit; save first formal pair and one healthy snapshot, schedule one wake and rest. Then close015/report and continue all six work packages.')
        value['wp3_execution'].update(status=training_status,collection_complete=True,collection=cref,independent_readback_complete=False)
        if path.name=='PROGRAM.json':value.update(wp3_collection_complete=True,wp3_independent_readback_complete=False,active_audit=RUN)
        if path.name=='MASTER_EXECUTION_PLAN.json':
            value['active_work_package']='WP3';value['current_action']=value['next_action']
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            wp.update(status='native0_selectively_collected_independent_raw_readback_prepared',collection=cref,
                saved_array_readback=execution,provisional_development_selection=sref)
            wp['execution'].update(status=training_status,collection_complete=True,collection=cref,independent_readback_complete=False)
    write(path,value)
readme=ROOT/'README.md';lead='''# N01 当前执行：WP3/015 保存数组独立复核准备

015四trial各2000G，child/supervisor均0。实际8000G/AD已完整入账，累计175667，搜索短试8/12，WP3已4/6。选收139/300文件，161原NPZ仍在服务器，原A/005和冻结源保持不变。

保存JSON独立比较按冻结规则选择M0研究参考，均值模块关闭：mean12 final wave为M0 0.15012505860962605、M1 0.15243191496746586，共同初始0.15042006897712382。M0相对A的2%用途门6/6，M1为0/6；绝对T和完整响应匹配仍未通过。权重未作科学晋级。

下一步已登记一次36对、161份NPZ身份SHA/160份评价数组解码的CPU复核，新增CP/NN/AD/G和随机数均0。完整CDF/loss与median尺度不重拟合。当前仅已准备，启动以真实回执为准。

[原生终点](receipts/wp3_015_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [保存指标独审](receipts/wp3_015_20261015/postrun_independent_01/DEVELOPMENT_SELECTION.json) · [读回配置](receipts/wp3_015_20261015/readback_registration_01/READBACK_EXECUTION.json)

以下保留此前阶段说明。

''';readme.write_text(lead+readme.read_text(encoding='utf-8-sig'),encoding='utf-8',newline='\n')
program=read(ROOT/'PROGRAM.json');program['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(ROOT/'PROGRAM.json',program)
current=read(metas[-1]);current.update(plan_sha256=sha(readme),master_plan_sha256=sha(ROOT/'MASTER_EXECUTION_PLAN.json'));write(metas[-1],current)
write(OUT/'ROOT_REGISTRATION_RECEIPT.json',dict(schema='n01/wp3-readback-root-registration/v1',UTC=now,
    execution=execution,collection=cref,provisional_selection=sref,cumulative_training_updates=175667,
    active_scientific_jobs=0,remote_started=False,original107_source_controls_unchanged=True,scientific_status='response_match_not_established'),True)
print(json.dumps(dict(status='readback_registered_not_started',configuration_sha256=execution['configuration_sha256'],
    source_controls=6,cumulative_training_updates=175667,active_scientific_jobs=0)))
