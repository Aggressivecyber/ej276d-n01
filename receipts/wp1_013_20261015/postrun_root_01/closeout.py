"""Close only the completed013 work package; keep the six-WP program active."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp1_013_20261015/postrun_root_01'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,obj,exclusive=False):
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

refs={
    'collection':'receipts/wp1_013_20261015/native_repeat_fix01/collection_03/COLLECTION_RECEIPT.json',
    'native_terminal':'receipts/wp1_013_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json',
    'numeric':'receipts/wp1_013_20261015/postrun_numeric_01/NATIVE_NUMERIC_READBACK.json',
    'independent':'receipts/wp1_013_20261015/postrun_independent_01/NUMERIC_READBACK.json',
    'decision':'receipts/wp1_013_20261015/postrun_independent_01/DECISION_READBACK.json',
    'arrays':'receipts/wp1_013_20261015/postrun_independent_01/REFERENCE_ARRAY_READBACK.json',
    'report_QA':'receipts/wp1_013_20261015/postrun_independent_01/REPORT_QA_FINAL.json',
    'visual_QA':'reports/wp1_013_20261015/visual_qa/VISUAL_QA.json',
    'report':'reports/wp1_013_20261015/N01_WP1_013_REPORT.md',
    'html':'reports/wp1_013_20261015/N01_WP1_013_REPORT.html',
    'T':'CALIBRATED_TOLERANCES_WP1_013.json',
    'T_freeze':'receipts/wp1_013_20261015/postrun_root_01/TOLERANCE_FREEZE_RECEIPT.json',
    'input_information':'INPUT_INFORMATION_AUDIT.json',
    'observation':'OBSERVATION_CONTRACT.json',
    'data_roles':'DATA_ROLES.json',
    'acceptance':'docs/ACCEPTANCE_CONTRACT.md',
}
for k in ['decision','arrays']:
    assert read(ROOT/refs[k])['passed'] is True
assert read(ROOT/refs['report_QA'])['status']=='PASS'
assert read(ROOT/refs['visual_QA'])['status']=='PASS'
assert read(ROOT/refs['decision'])['decision']=='adequate'
terminal=read(ROOT/refs['native_terminal'])
collection=read(ROOT/refs['collection'])
assert collection['job']['native_exit_code']==0 and collection['file_count']==265
assert collection['partial_evidence_only'] is False
assert read(ROOT/'PROGRAM.json')['new_training_updates']==159667
result=dict(schema='n01/wp1-final-result/v1',UTC=now,run_id='wp1_013_20261015',
    work_package='WP1',status='completed_native0_collected_independently_read_back',
    native_child_exit_code=0,native_supervisor_exit_code=0,
    files_collected=265,independent_inventory_verified=True,
    method_choice='adequate',adequate_direction='negative_saved012_total',
    method_scope='fixedG2000, registered positive nominal1LSB direction, all12 TRAIN bank/batch/hypothesis cells',
    current_head_useful_wave_gate_passed=False,
    full_response_acceptance=False,scientific_status='response_match_not_established',
    independent_acquisition_reuse='unverified',
    new_training_updates=0,new_AD_queries=0,cumulative_training_updates=159667,
    attempt3_actual_counters=terminal['attempt3_actual_counts'],
    two_prior_failed_attempt_actual_counters=terminal['prior_two_failed_attempt_actual_counts'],
    aggregate_actual_counters=terminal['aggregate_actual_counts'],
    source_commit='7be67b8d3809246a58bcb6f03cd531e8f7539ad3',
    references={k:dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for k,p in refs.items()},
    next_work_package='WP2',next_action='Freeze and start paired C0/C1 short trials; C2 disabled',
    next_scientific_job_started=False,whole_six_work_package_program_complete=False)
receipt=OUT/'FINAL_RESULT_RECEIPT.json'
write(receipt,result,True)
result_ref=dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt),
    status=result['status'],report=refs['report'],report_sha256=sha(ROOT/refs['report']),
    method_choice='adequate',new_training_updates=0,new_AD_queries=0,
    aggregate_actual_counters=terminal['aggregate_actual_counts'],
    formal_response_acceptance=False)

paths=[ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'PROGRAM.json',ROOT/'jobs.json',
    ROOT/'README.md',Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')]
backup=OUT/'metadata_before_final_closeout';backup.mkdir()
for p in paths:shutil.copy2(p,backup/p.name)
for p in paths:
    if p.suffix!='.json':continue
    d=read(p);d['updated_UTC']=now;d['active_scientific_jobs']=0
    if p.name=='jobs.json':
        job=next(j for j in d['jobs'] if j['job_id']=='N01_WP1_013_20261015')
        job.update(status=result['status'],collection_complete=True,independent_readback_complete=True,
            bounded_budget_closed=True,completion_receipt=result_ref,report=refs['report'],
            scientific_status='response_match_not_established',formal_response_acceptance=False,
            method_choice='adequate',actual_counters=terminal['attempt3_actual_counts'],
            total_actual_counters=terminal['aggregate_actual_counts'],active_attempt=None)
        if isinstance(job.get('execution'),dict):job['execution']['status']='completed_native0_independently_read_back'
    else:
        d.update(status='WP1_013_closed_WP2_014_preparation',wp1_result=result_ref,
            scientific_status='response_match_not_established')
        if isinstance(d.get('numerical_storage_repair'),dict):d['numerical_storage_repair']['status']='completed_native0_independently_read_back'
        if p.name=='MASTER_EXECUTION_PLAN.json':
            d['active_work_package']='WP2'
            wp=next(w for w in d['work_packages'] if w['id']=='WP1')
            wp.update(status=result['status'],completion=result_ref,method_choice='adequate')
            next(w for w in d['work_packages'] if w['id']=='WP2')['status']='preparation'
            d['current_action']='Prepare frozen C0/C1 paired short trials, two seeds each, at most2000G/run; C2 disabled after013 adequate.'
        if p.name=='PROGRAM.json':
            d.update(current_work_package='WP2',current_stage='WP2_014_preparation',
                active_research=None,wp1_completed=True,wp1_independent_readback_complete=True,
                wp1_report=refs['report'],wp1_method_choice='adequate',
                wp1_numeric_tolerance_table=refs['T'],wp1_numeric_tolerance_sha256=sha(ROOT/refs['T']),
                latest_run_id='wp1_013_20261015',
                active_scientific_jobs_status_basis='013 native child/supervisor0,265file collection and independent array/report readback;014 not dispatched')
        if p.name=='CURRENT_SIM2REAL_PROGRAM.json':
            d.update(latest_run_id='wp1_013_20261015',current_work_package='WP2',
                latest_report=str(ROOT/refs['report']),latest_completion_receipt=str(receipt))
    write(p,d)

readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
lead='''<!-- WP1_013_CLOSEOUT_20261015 -->
## 当前进展：WP1/013已独立闭合，推进WP2

013计算与监督均原生退出0，265文件库存及保存数组独立读回通过。固定总梯度方向通过全部12格有限步方法规则，选择adequate；数字容忍度已在014候选之前冻结。当前010头完整人口wave改善为0.9162%/0.9914%/1.6907%，均低于2%门槛，形态和基线均值仍有绝对偏差，科学状态保持response_match_not_established。

本轮训练与新AD均0，累计科学训练159667；前三次attempt全部消耗保留。下一步是C0/C1各2种子、每次≤2000G的配对短试，C2关闭。六工作包主计划继续执行。

[013中文报告](reports/wp1_013_20261015/N01_WP1_013_REPORT.md) · [数字T表](CALIBRATED_TOLERANCES_WP1_013.json) · [SOURCE审计](INPUT_INFORMATION_AUDIT.json) · [完成回执](receipts/wp1_013_20261015/postrun_root_01/FINAL_RESULT_RECEIPT.json)

<!-- END_WP1_013_CLOSEOUT_20261015 -->

'''
assert '<!-- WP1_013_CLOSEOUT_20261015 -->' not in text
readme.write_text(lead+text,encoding='utf-8',newline='\n')
selector=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json');s=read(selector);s['plan_sha256']=sha(readme);write(selector,s)
program=ROOT/'PROGRAM.json';s=read(program);s['master_plan_sha256']=sha(ROOT/'MASTER_EXECUTION_PLAN.json');write(program,s)

destination=Path('D:/GAN_backups/N01/wp1_013_20261015/closeout_01')
assert not destination.exists();destination.mkdir(parents=True)
files=[ROOT/p for p in refs.values()]+[receipt,ROOT/'PROGRAM.json',ROOT/'MASTER_EXECUTION_PLAN.json',ROOT/'jobs.json',readme,
    ROOT/'wp1_013_config.json',ROOT/'WP1_SOURCE_MANIFEST.json',ROOT/'USER_MASTER_PLAN_20261015.txt']
files += list((ROOT/'reports/wp1_013_20261015/visual_qa').glob('*.png'))
assert sum(p.stat().st_size for p in files)<16*1024**2
inventory={}
for p in dict.fromkeys(files):
    rel=p.relative_to(ROOT);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target);assert sha(p)==sha(target)
    inventory[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
shutil.copy2(selector,destination/'CURRENT_SIM2REAL_PROGRAM.json');assert sha(selector)==sha(destination/'CURRENT_SIM2REAL_PROGRAM.json')
inventory['CURRENT_SIM2REAL_PROGRAM.json']=dict(bytes=selector.stat().st_size,sha256=sha(selector))
write(destination/'MANIFEST.json',dict(UTC=now,files=inventory,raw_arrays_included=False),True)
(destination/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(inventory.items())),encoding='utf-8')
write(OUT/'D_CLOSEOUT_BACKUP_RECEIPT.json',dict(schema='n01/wp1-small-closeout-backup/v1',UTC=now,
    status='all_copied_files_SHA256_verified',destination=str(destination),files=len(inventory),
    bytes=sum(v['bytes'] for v in inventory.values()),manifest_sha256=sha(destination/'MANIFEST.json'),
    raw_arrays_included=False,source_data_remain_on_E_and_native_server=True),True)
print(json.dumps(dict(status=result['status'],final_result_sha256=sha(receipt),
    active_work_package='WP2',active_scientific_jobs=0,D_backup_files=len(inventory))))
