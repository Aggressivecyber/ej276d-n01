"""Bind the completed independent016 review without changing scientific acceptance."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))

def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

review_path=BASE/'postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json'
selection_path=BASE/'postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json'
review=read(review_path);selection=read(selection_path)
assert review['passed'] is True and review['blockers']==[] and not review['provisional']
assert review['selected_research_reference']==selection['selected_research_reference']=='Q1'
assert selection['independent_raw_saved_array_readback_complete'] and selection['development_selection_final']
assert not selection['selected_mean_adapter_enabled'] and selection['selected_charge_enabled']
assert not review['weights_promoted'] and not review['scientific_acceptance']
assert review['cumulative_training_updates']==179667 and review['expected_operations']==200
assert read(OUT/'CPU_COMPLETION_RECONCILIATION.json')['array_result_passed']
stamp=dt.datetime.now(dt.timezone.utc).isoformat()
selection_ref=dict(**ref(selection_path),selected_research_reference='Q1',mean_adapter_enabled=False,charge_enabled=True,
    status='independently_reviewed_development_research_reference',provisional=False,weights_promoted=False,scientific_acceptance=False)
review_ref=ref(review_path)
paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
before=OUT/'metadata_before_final_selection';before.mkdir(exist_ok=False)
for path in paths:shutil.copy2(path,before/path.name)
for path in [p for p in paths if p.suffix=='.json']:
    value=read(path);assert value['active_scientific_jobs']==0
    value['updated_UTC']=stamp
    if path.name=='jobs.json':
        for job in value['jobs']:
            if job.get('run_id')=='wp3_016_saved_readback_20261015':
                job.update(status='completed_native0_collected_independently_reviewed_closed',final_independent_review=review_ref)
                job['execution'].update(status=job['status'],final_independent_local_review_complete=True,final_independent_review=review_ref)
            elif job.get('run_id')=='wp3_016_20261015':
                job.update(final_independent_review=review_ref,development_selection=selection_ref,report_pending=True)
    else:
        assert value['new_training_updates']==179667
        provisional=value['wp3_016_provisional_development_selection']
        provisional.update(role='historical provisional record retained',superseded_by=selection_ref)
        value.update(status='WP3_016_independently_reviewed_report_in_progress',
            current_stage='016_array_and_selection_review_closed_report_pending',
            current_research_reference='016_Q1_mean_off_charge_on',wp3_016_development_selection=selection_ref,
            wp3_016_independent_review=review_ref,weights_promoted=False,scientific_status='response_match_not_established',
            next_action='Complete016 report and E/D delivery, then WP4; all6 WP3 short slots used.')
        for key in ['wp3_readback_execution','wp3_016_readback_execution']:
            value[key].update(status='completed_native0_collected_independently_reviewed_closed',
                final_independent_local_review_complete=True,final_independent_review=review_ref)
        if path.name=='MASTER_EXECUTION_PLAN.json':
            wp=next(w for w in value['work_packages'] if w['id']=='WP3')
            wp.update(status='016_independently_reviewed_report_pending',final_development_selection=selection_ref,final_independent_review=review_ref)
            for key in ['saved_array_readback','readback016']:
                wp[key].update(status='completed_native0_collected_independently_reviewed_closed',
                    final_independent_local_review_complete=True,final_independent_review=review_ref)
    write(path,value)
readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
text=text.replace('Q1仍为待最终独审闭合的研究参考选择：','最终独审已通过，研究参考选择Q1（mean off/charge on）：',1)
text=text.replace('当前不宣称科学晋级；完成016报告后推进WP4，WP3不追加候选。','权重未科学晋级，科学状态仍为response_match_not_established；完成016报告后推进WP4，WP3不追加候选。',1)
readme.write_text(text,encoding='utf-8',newline='\n')
master_sha=sha(ROOT/'MASTER_EXECUTION_PLAN.json')
for path in [ROOT/'PROGRAM.json',CURRENT]:
    value=read(path);value['master_plan_sha256']=master_sha
    if path==CURRENT:value['plan_sha256']=sha(readme)
    write(path,value)
write(OUT/'FINAL_SELECTION_BINDING_RECEIPT.json',dict(UTC=stamp,independent_review=review_ref,development_selection=selection_ref,
    cumulative_G=179667,active_scientific_jobs=0,scientific_acceptance=False,whole_project_complete=False),True)
print(json.dumps(dict(status='final016_research_selection_bound_report_pending',selected='Q1',cumulative_G=179667)))
