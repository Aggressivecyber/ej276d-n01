"""Register one reviewed016 CPU saved-array audit; no native or tensor action."""
import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'receipts/wp3_016_20261015'
OUT=Path(__file__).resolve().parent
RUN='wp3_016_saved_readback_20261015'
SIDE='/home/rylan/projects/ej276d-n01/'+RUN
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),**record(p))


def write(p,value,*,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def local_ref(row):
    value=Path(row.get('local_path',row['path']))
    path=value if value.is_absolute() else ROOT/value
    assert path.is_file() and path.is_relative_to(ROOT),str(path)
    assert record(path)=={k:row[k] for k in ('bytes','sha256')},str(path)
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-ready',default='postrun_independent_01/READBACK_WORKER_READY.json')
    parser.add_argument('--controller-ready',default='readback_engineering_01/CONTROLLER_READY.json')
    parser.add_argument('--review',default='postrun_independent_01/PRELAUNCH_REVIEW.json')
    args=parser.parse_args()
    plan_path=BASE/'postrun_independent_01/READBACK_PLAN.json';plan=read(plan_path)
    selection_path=BASE/'postrun_independent_01/PROVISIONAL_DEVELOPMENT_SELECTION.json';selection=read(selection_path)
    worker_ready_path=BASE/args.worker_ready;worker_ready=read(worker_ready_path)
    controller_ready_path=BASE/args.controller_ready;controller_ready=read(controller_ready_path)
    review_path=BASE/args.review;review=read(review_path)
    assert review['passed'] and not review['blockers']
    assert plan['run_id']==RUN and plan['sidecar_root']==SIDE and plan['expected_operations']==200
    expected=plan['expected_raw_files'];counters=plan['expected_complete_counters'];finite=plan['finite_resource_budget']
    assert expected['total_npz_SHA']==expected['total_npz_decode']==112 and expected['bank_pairs']==24
    assert expected['hard_rows_npz']==expected['report_npz']==48
    assert expected['baseline100_npz']==expected['charge_moments_npz']==8
    assert expected['raw_compressed_bytes']==2169655337 and expected['total_hard_rows']==1265856
    assert expected['actual_A_float64_cells']==81014784 and expected['charge_parent_moment_records']==52744
    assert all(v==0 for v in plan['scientific_budget'].values())
    assert finite['CPU_threads']==2 and finite['process_RSS_MiB']==1536 and finite['wall_seconds']==1800
    assert finite['virtual_memory_backstop_MiB']==3072 and finite['output_MiB']==64
    assert selection['selected_research_reference']=='Q1' and not selection['weights_promoted']
    assert selection['independent_raw_saved_array_readback_complete'] is False
    policy_path=ROOT/'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json';policy=read(policy_path)
    assert policy['model']=='gpt-6.1-sol' and policy['reasoning_effort']=='max'
    native_path=BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json';native=read(native_path)
    assert native['cumulative_G']==179667 and native['actual_completed_counts']['G_updates']==4000
    local_path=BASE/'postrun_root_01/SELECTED_COLLECTION_READBACK.json';local=read(local_path)
    assert local['selected_files']==99 and local['array_decodes']==local['checkpoint_loads']==0
    assert local['whole_inventory_files']==211 and local['raw_NPZ_files_retained_remote']==112
    source=read(BASE/'SOURCE_COMMIT_READBACK_FIX02.json')
    for name,row in source['files'].items():assert record(ROOT/name)==row,name
    worker=local_ref(worker_ready['worker']);kernel=local_ref(worker_ready['kernel'])
    controller=local_ref(controller_ready['controller'])
    assert local_ref(controller_ready['worker'])==worker and local_ref(controller_ready['kernel'])==kernel
    for ready in [worker_ready,controller_ready]:
        assert local_ref(ready['plan'])==plan_path
        if 'expected_complete_counters' in ready:assert ready['expected_complete_counters']==counters
    for name,row in review['sources'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        assert record(path)=={k:row[k] for k in ('bytes','sha256')},name
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json')
    assert program['new_training_updates']==179667 and program['active_scientific_jobs']==jobs['active_scientific_jobs']==0
    assert program['short_candidate_runs_completed']==10 and program['wp3_short_candidate_runs_completed']==6
    assert not any(j.get('run_id')==RUN for j in jobs['jobs'])
    inventory=OUT/'BOUND_ORIGINAL_INVENTORY.json';assert not inventory.exists()
    original_inventory=BASE/'collection_01/REMOTE_FULL_INVENTORY.json'
    assert sha(original_inventory)=='e7cf8c67cdcb2dcadf006b3a8c060f706f9a1b64990c2e383288d79ce22097af'
    paths={'readback_control.py':controller,'readback_saved_arrays.py':worker,'saved_array_kernels.py':kernel,
        'READBACK_PLAN.json':plan_path,'BOUND_ORIGINAL_INVENTORY.json':inventory}
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    config=dict(schema='n01/wp3-independent-readback-execution/v1',run_id=RUN,registered_UTC=stamp,status='registered_not_started',
        remote_root=SIDE,original_root=plan['original_root'],host='rylan@100.82.231.102',
        python='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python',
        files={name:dict(path=SIDE+'/'+name,local_path=str(path),
            **record(original_inventory if name=='BOUND_ORIGINAL_INVENTORY.json' else path)) for name,path in paths.items()},
        resource_limits=dict(CPU_threads=2,RSS_MiB=1536,virtual_memory_MiB=3072,wall_seconds=1800,output_MiB=64),
        selected_collection_MiB_cap_including_controls_and_logs=80,original_files_read_only=True,automatic_retry=False,
        exclusive_project_lock='/home/rylan/projects/ej276d-n01/.n01_scientific_dispatch.lock',
        scientific_budget=plan['scientific_budget'],expected_complete_counters=counters,expected_operations=200,
        finite_scope=dict(bank_pairs=24,raw_NPZ=112,evaluation_NPZ_decoded=112,metadata_only_NPZ_SHA_checked=0,
            hard_rows_NPZ=48,REPORT_NPZ=48,baseline100_NPZ=8,charge_NPZ=8,probe_raw_NPZ=0,probe_gradient_NPZ=0,
            SOURCE_parents=6593,rows_per_hard_file=26372,waveform_rows_decoded=1265856,
            charge_A_rows=210976,charge_A_float64_cells=81014784,charge_mu_draws=210976,charge_q_draws=210976,
            charge_parent_moments=52744,raw_SHA_passes=1,raw_decode_passes=1,raw_compressed_bytes=2169655337,
            input_compressed_envelope_MiB=3072,input_IO_envelope_MiB=6144),
        root_authority=dict(user_master_plan='USER_MASTER_PLAN_20261015.txt',user_master_plan_sha256=sha(ROOT/'USER_MASTER_PLAN_20261015.txt'),
            execution_instruction='2026-10-15T08:45:06.316Z n01 heartbeat',
            latest_model_instruction='2026-10-15T06:16:30.767282Z 6.1solmax',
            authorizes_one_bounded_saved_array_audit=True,no_additional_user_permission_required=True),
        original_training_source_commit=source['source_commit'],original_configuration_sha256=source['configuration_sha256'],
        original_source_manifest_sha256=source['source_manifest_sha256'],original_training_completed_G=4000,
        original_training_completed_AD=4000,cumulative_training_before_and_after_this_audit=179667,
        original_raw_NPZ_remain_remote=True,original_raw_local_collection_complete=False,
        new_original_TRAIN_or_DEV_waveform_reads=0,raw_complete_wave_QPSD_noise_loss_replay=False,
        C1_median_fit_independently_replayed=False,preADC_fullcarrier_independent_reconstruction=False,
        full_frozen_A384_saved_array_verification=True,full_stochastic_preADC_carrier_reconstruction=False,
        original_SOURCE_input_byte_replay=False,charge_RNG_or_latent_replay=False,
        loss_evidence='Producer numerics plus independent saved-JSON decision; this audit recomputes saved row and charge/A statistics, not complete CDF/loss.',
        provisional_development_path='Q1',mean_adapter_enabled=False,charge_enabled_provisionally=True,weights_promoted=False,
        prelaunch_worker_ready=ref(worker_ready_path),prelaunch_controller_ready=ref(controller_ready_path),
        prelaunch_independent_review=ref(review_path),agent_model_policy=ref(policy_path),
        provisional_selection=ref(selection_path),native_training_closure=ref(native_path),
        long_job_policy='Verify first formal saved-array pair plus required charge checks and current same-snapshot health; set one future n01 wake and rest.',
        scientific_status='response_match_not_established',scientific_acceptance=False)
    spec=importlib.util.spec_from_file_location('root016_readback_control_validation',controller)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.validate(config)
    shutil.copy2(original_inventory,inventory)
    config_path=OUT/'READBACK_EXECUTION.json';write(config_path,config,new=True)
    execution=dict(configuration=config_path.relative_to(ROOT).as_posix(),configuration_sha256=sha(config_path),
        run_id=RUN,remote_root=SIDE,resource_limits=config['resource_limits'],source_file_count=5,source_controls=6,
        source_commit_pending=True,status='registered_not_started',new_training_updates=0,new_AD_queries=0,NN_forward_rows=0,
        exact_expected_counters=counters,expected_operations=200,scientific_status='response_match_not_established')
    selection_ref=dict(**ref(selection_path),development_path='Q1',status='provisional_raw_array_QA_pending',
        mean_adapter_enabled=False,charge_enabled=True,weights_promoted=False,scientific_acceptance=False)
    metas=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
    snapshot=OUT/'metadata_before_registration';snapshot.mkdir(exist_ok=False)
    for path in metas:shutil.copy2(path,snapshot/path.name)
    master,program,jobs,current=[read(v) for v in [metas[0],metas[1],metas[2],CURRENT]]
    for value in [master,program,current]:
        if 'wp3_readback_execution' in value and 'wp3_015_readback_execution' not in value:
            value['wp3_015_readback_execution']=copy.deepcopy(value['wp3_readback_execution'])
        value.update(status='WP3_016_CPU_readback_registered_not_started',current_work_package='WP3',
            current_stage='WP3_016_independent_saved_array_readback_prepared',active_scientific_jobs=0,active_research=RUN,
            active_audit=RUN,wp3_readback_execution=execution,wp3_016_readback_execution=execution,
            wp3_016_provisional_development_selection=selection_ref,current_run_training_updates=0,
            new_training_updates=179667,cumulative_training_updates=179667,
            next_action='Start the one bounded zero-NN CPU readback; verify first formal pair and same-snapshot health, set one future wake and rest. Close016/report then continue WP4-WP6.',
            agent_model_policy=ref(policy_path),latest_subagent_model_preference='GPT-6.1-Sol Max',updated_UTC=stamp)
    jobs['jobs'].append(dict(job_id='N01_WP3_016_SAVED_ARRAY_READBACK_20261015',run_id=RUN,scientific_job=True,
        kind='CPU_saved_array_audit',status='registered_not_started',registered_UTC=stamp,execution=execution,
        resource_limits=config['resource_limits'],G_updates=0,new_AD_queries=0,NN_forward_rows=0,checkpoint_loads=0,
        automatic_retry=False,original_training_before_and_after=179667,scientific_status='response_match_not_established'))
    jobs.update(updated_UTC=stamp,active_scientific_jobs=0)
    wp3=next(w for w in master['work_packages'] if w['id']=='WP3')
    wp3.update(status='016_CPU_readback_registered',saved_array_readback=execution,readback016=execution,
        provisional_development_selection=selection_ref)
    previous=metas[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in previous
    lead='''# N01 当前执行：WP3/016 保存数组独立复核已登记

016两个seed各2000G、child/supervisor均0，实际4000G/AD已闭合，累计179667；搜索短试10次、WP3用满6/6。选收99/211文件、112份原NPZ仍在服务器，首次失败的非训练费用保留。

保存JSON按冻结规则暂选Q1，mean关闭、charge开启；mean12波形距离0.15011008209012994，相对M0 0.15012505860962605改善约0.009976%。ALL6双参考非劣和相对A用途门通过，绝对T未全通过；等待保存数组QA，权重未科学晋级。

已登记112份NPZ各一次SHA/解码、24对硬统计和8份电荷矩，共200项CPU操作，新增CP/NN/AD/G/RNG/GPU及原TRAIN/DEV读取均0。实际A384与逐draw电荷矩将独立复核；完整随机载体、RNG、CDF/loss及C1median不重放。当前尚未启动，启动以真实回执为准。

[016原生终点](receipts/wp3_016_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [暂选结果](receipts/wp3_016_20261015/postrun_independent_01/PROVISIONAL_DEVELOPMENT_SELECTION.json) · [CPU配置](receipts/wp3_016_20261015/readback_registration_01/READBACK_EXECUTION.json)

'''
    metas[3].write_text(lead+previous[previous.index(marker):],encoding='utf-8',newline='\n')
    write(metas[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(metas[0]);current['plan_sha256']=sha(metas[3])
    for path,value in [(metas[1],program),(metas[2],jobs),(CURRENT,current)]:write(path,value)
    write(OUT/'ROOT_REGISTRATION_RECEIPT.json',dict(schema='n01/wp3-016-CPU-root-registration/v1',UTC=stamp,
        execution=execution,provisional_selection=selection_ref,cumulative_training_updates=179667,
        active_scientific_jobs=0,remote_started=False,original118_controls_unchanged=True,
        source_model_policy='gpt-6.1-sol / max',scientific_status='response_match_not_established'),new=True)
    print(json.dumps(dict(status='readback_registered_not_started',configuration_sha256=sha(config_path),
        source_controls=6,operations=200,cumulative_training_updates=179667)))


if __name__=='__main__':main()
