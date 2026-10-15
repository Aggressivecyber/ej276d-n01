"""Close the reviewed016 bounded experiment, report and E/D evidence delivery."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
REPORT=ROOT/'reports/wp3_016_20261015'
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))
ref=lambda p:dict(path=p.relative_to(ROOT).as_posix(),**record(p))

def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

stamp=dt.datetime.now(dt.timezone.utc).isoformat()
manifest_path=REPORT/'report_manifest.json';manifest=read(manifest_path)
for name,row in manifest['outputs'].items():assert record(REPORT/name)==row,name
qa_path=REPORT/'REPORT_QA.json';qa=read(qa_path);assert qa['passed'] is True
visual_path=HERE/'REPORT_VISUAL_QA.json';visual=read(visual_path)
assert visual['passed'] and visual['report']['sha256']==sha(REPORT/'REPORT.html')
review_path=BASE/'postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json';review=read(review_path)
selection_path=BASE/'postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json';selection=read(selection_path)
assert review['passed'] and review['blockers']==[] and not review['provisional']
assert selection['development_selection_final'] and not selection['provisional']
assert selection['selected_research_reference']=='Q1' and selection['selected_charge_enabled'] and not selection['selected_mean_adapter_enabled']
assert not selection['weights_promoted'] and not selection['scientific_acceptance']
cpu_path=BASE/'readback_registration_01/CPU_COMPLETION_RECONCILIATION.json';cpu=read(cpu_path)
assert cpu['operations_entered']==cpu['operations_completed']==200 and cpu['new_G_updates']==cpu['new_AD_queries']==0
assert cpu['collection']['all_files_SHA256_and_bytes_verified'] and cpu['collection']['files']==19
native_path=BASE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json';native=read(native_path)
assert native['actual_completed_counts']['G_updates']==native['actual_completed_counts']['new_AD_queries']==4000
assert native['cumulative_G']==179667
source=read(BASE/'SOURCE_COMMIT_READBACK_FIX02.json')
cpu_source=read(BASE/'readback_registration_01/SOURCE_COMMIT_READBACK.json')
for rows in [source['files'],cpu_source['files']]:
    for name,row in rows.items():assert record(ROOT/name)==row,name
assert not (BASE/'FINAL_RESULT_RECEIPT.json').exists()
paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
for path in [p for p in paths if p.suffix=='.json']:
    value=read(path);assert value['active_scientific_jobs']==0
    if path.name!='jobs.json':assert value['new_training_updates']==179667
before=HERE/'metadata_before_report_closeout';before.mkdir(exist_ok=False)
for path in paths:shutil.copy2(path,before/path.name)
backup=Path('D:/GAN_backups/N01/wp3_016_20261015/closeout_01')
backup.mkdir(parents=True,exist_ok=False)
copies=[REPORT/name for name in manifest['outputs']]+[manifest_path,qa_path,visual_path,review_path,selection_path,native_path,cpu_path,
    BASE/'readback_execution_01/COLLECTION_READBACK_SUMMARY.json',BASE/'readback_execution_01/START_VALIDATION_SUMMARY.json',
    BASE/'readback_collection_01/COLLECTION_RECEIPT.json',BASE/'collection_01/COLLECTION_RECEIPT.json',
    BASE/'collection_01/REMOTE_FULL_INVENTORY.json',BASE/'NATIVE_FAILURE_RECONCILIATION.json',
    BASE/'readback_registration_01/FINAL_SELECTION_BINDING_RECEIPT.json',
    ROOT/'receipts/wp3_015_20261015/postrun_root_02/AGENT_MODEL_POLICY_SOLMAX.json',Path(__file__).resolve()]
for directory in [HERE/'html_qa_02',BASE/'postrun_report_impl_01',ROOT/'runs/wp3_016_20261015/readback_collected_01']:
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in ['.json','.py','.md','.txt','.png','.html','.log'] and not any(part.startswith(('before_','versions','_cache')) for part in path.parts):
            copies.append(path)
copies=list(dict.fromkeys(copies))
assert all(p.is_file() and p.suffix.lower() not in ['.npz','.pt','.pth','.tar','.gz'] for p in copies)
assert sum(p.stat().st_size for p in copies)<48*1024**2
assert shutil.disk_usage(backup).free>sum(p.stat().st_size for p in copies)+32*1024**2
inventory={}
for path in copies:
    name=path.relative_to(ROOT).as_posix();target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target);assert record(target)==record(path)
    inventory[name]=record(target)
write(backup/'MANIFEST.json',dict(UTC=stamp,files=inventory,original_raw_or_checkpoint_included=False),True)
(backup/'SHA256SUMS.txt').write_text(''.join(row['sha256']+'  '+name+'\n' for name,row in sorted(inventory.items())),encoding='utf-8',newline='\n')
backup_path=BASE/'D_CLOSEOUT_BACKUP_RECEIPT.json'
write(backup_path,dict(UTC=stamp,status='all_copied_files_SHA256_verified',destination=str(backup),files=len(inventory),
    bytes=sum(row['bytes'] for row in inventory.values()),manifest_sha256=sha(backup/'MANIFEST.json'),original_raw_or_checkpoint_included=False),True)
result_path=BASE/'FINAL_RESULT_RECEIPT.json'
candidate=selection['candidates']['Q1']
result=dict(schema='n01/wp3-016-final-result/v1',UTC=stamp,status='016_bounded_training_array_audit_and_report_complete',
    training_source_commit=source['source_commit'],saved_array_CPU_source_commit=cpu_source['source_commit'],
    actual_completed_counts=native['actual_completed_counts'],cumulative_training_before016=175667,cumulative_training_after016=179667,
    additional_CPU_G_AD_NN_CP_RNG=0,global_short_trials_completed=10,WP3_short_trials_completed=6,WP3_remaining_slots=0,
    original_training_selected_files=99,original_training_inventory_files=211,raw112_NPZ_retained_remote=True,
    prior_preG_failure_nontraining_costs_preserved=ref(BASE/'NATIVE_FAILURE_RECONCILIATION.json'),
    bounded_CPU_saved_array_audit=ref(cpu_path),independent_review=ref(review_path),development_selection=ref(selection_path),
    selected_research_reference='Q1',mean_adapter_enabled=False,charge_enabled=True,
    Q1_mean12_wave=candidate['final_wave_mean12'],M0_mean12_wave=candidate['paired_M0_wave_mean12'],
    Q1_relative_improvement_vs_M0=candidate['mean12_relative_improvement_vs_M0'],
    dual_reference_positive_T_noninferior_cells=6,relative_wave_usefulness_vs_A_cells=6,absolute_T_ALL6_passed=False,
    weights_promoted=False,scientific_acceptance=False,scientific_status='response_match_not_established',
    independent_complete_CDF_loss_replay=False,independent_C1_median_refit=False,
    saved_actual_frozen_A384_verified=True,full_stochastic_preADC_carrier_replayed=False,charge_latent_RNG_replayed=False,
    original_SOURCE_input_byte_replay=False,original_failed_mu_difference_magnitude='unknown_not_saved',
    report_MD=ref(REPORT/'REPORT.md'),report_HTML=ref(REPORT/'REPORT.html'),report_manifest=ref(manifest_path),
    report_content_QA=ref(qa_path),report_visual_QA=ref(visual_path),D_closeout_backup=ref(backup_path),
    whole_WP3_bounded_candidate_decisions_complete=True,whole_six_workpackage_project_complete=False,
    next_stage='WP4_known_response_recovery',cross_acquisition_physical_reuse='not_verified')
write(result_path,result,True)
result_ref=dict(**ref(result_path),status=result['status'],weights_promoted=False,scientific_acceptance=False)
reports=dict(MD=ref(REPORT/'REPORT.md'),HTML=ref(REPORT/'REPORT.html'),visual_QA=ref(visual_path))
for path in [p for p in paths if p.suffix=='.json']:
    value=read(path);value.update(updated_UTC=stamp,active_scientific_jobs=0)
    if path.name=='jobs.json':
        for job in value['jobs']:
            if job.get('run_id')=='wp3_016_20261015':
                job.update(status='completed_native0_collected_independently_reviewed_reported_closed',result=result_ref,report=reports,report_pending=False)
                job['execution']['status']=job['status']
            elif job.get('run_id')=='wp3_016_saved_readback_20261015':job['status']='completed_native0_collected_independently_reviewed_closed'
    else:
        value.update(wp3_016_completed=True,wp3_016_result=result_ref,wp3_016_report=reports,wp3_016_report_visual_QA_complete=True,
            status='WP3_closed_WP4_preparation_in_progress',current_work_package='WP4',current_stage='WP4_known_response_contract_and_implementation',
            active_research=None,active_audit=None,current_research_reference='016_Q1_mean_off_charge_on',
            next_action='Prepare, review and register the bounded WP4 teacher protocol; firstT1 paired recovery gate precedes remaining teacher trials.',
            scientific_status='response_match_not_established',weights_promoted=False)
        if 'wp3_execution' in value:value['wp3_execution']['status']='completed_native0_collected_independently_reviewed_reported_closed'
        if path.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            wp.update(status='completed_bounded_candidate_decisions_response_match_not_established',result016=result_ref,report016=reports,
                selected_research_reference='Q1',mean_adapter_enabled=False,charge_enabled=True,remaining_short_trial_slots=0)
            if 'execution' in wp:wp['execution']['status']='completed_native0_collected_independently_reviewed_reported_closed'
            next(w for w in value['work_packages'] if w['id']=='WP4')['status']='protocol_preparation_in_progress_not_started'
    write(path,value)
readme=ROOT/'README.md';previous=readme.read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in previous
lead='''# N01 当前执行：WP3/016 已闭合，推进 WP4 教师恢复

016两seed共4000G/AD、112文件/200项保存数组独立核验、最终选择和报告均已闭合。累计179667G；WP3短试6/6、全部短候选10/12；当前无运行中的科学作业。

研究参考为Q1（mean off、charge on）。mean12 wave=0.15011008209012994，M0=0.15012505860962605，改善约0.009976%；6个seed/假设cell中3改善、3略差，ALL6双参考非劣及相对A用途通过，绝对T仍未全通。权重未科学晋级，response_match_not_established。

[016中文报告](reports/wp3_016_20261015/REPORT.md) · [HTML与英文图](reports/wp3_016_20261015/REPORT.html) · [最终回执](receipts/wp3_016_20261015/FINAL_RESULT_RECEIPT.json) · [D盘闭合备份](receipts/wp3_016_20261015/D_CLOSEOUT_BACKUP_RECEIPT.json)

WP3不追加候选。下一阶段为固定已知响应的WP4恢复检验；首批同族配对恢复门在其它教师试验之前，教师成功不代表实测或硬件物理复用。六工作包尚未全部完成。

'''
readme.write_text(lead+previous[previous.index(marker):],encoding='utf-8',newline='\n')
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for path in [ROOT/'PROGRAM.json',CURRENT]:
    value=read(path);value['master_plan_sha256']=master_sha
    if path==CURRENT:value['plan_sha256']=sha(readme)
    write(path,value)
state_path=HERE/'FINAL_REPORT_CLOSEOUT_STATE.json'
write(state_path,dict(UTC=stamp,final_result=ref(result_path),report_complete=True,cumulative_G=179667,
    active_scientific_jobs=0,whole_project_complete=False,new_scientific_operations=0,metadata={str(p):sha(p) for p in paths}),True)
supplement=[result_path,backup_path,state_path,*paths]
delivery={}
for path in supplement:
    name=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
    target=backup/'delivery'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    assert record(target)==record(path);delivery[name]=record(target)
write(backup/'DELIVERY_MANIFEST.json',dict(UTC=stamp,base_manifest_sha256=sha(backup/'MANIFEST.json'),files=delivery),True)
print(json.dumps(dict(status=result['status'],final_result_sha256=sha(result_path),D_backup_files=len(inventory),
    D_backup_bytes=sum(row['bytes'] for row in inventory.values()),cumulative_G=179667,whole_project_complete=False)))
