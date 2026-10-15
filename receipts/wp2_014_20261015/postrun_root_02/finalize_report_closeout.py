"""Close WP2 reporting without adding any scientific computation or updates."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import struct

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
REPORT=ROOT/'reports/wp2_014_20261015'
CURRENT=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=dt.datetime.now(dt.timezone.utc).isoformat()

def write(path,value):
    assert not path.exists(),str(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)

def png(path):
    data=path.read_bytes()
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    width,height=struct.unpack('>II',data[16:24])
    assert width>1000 and height>700
    return dict(**ref(path),width=width,height=height)

manifest=read(REPORT/'report_manifest.json')
qa_path=OUT/'postrun_report_impl_01/REPORT_QA.json'
assert read(qa_path)['passed'] is True
assert read(qa_path)['new_NPZ_CP_NN_AD_random_CDF_training_SSH']==0
files=manifest.get('outputs',manifest.get('files'))
assert isinstance(files,dict) and len(files)==15
for name,row in files.items():
    path=REPORT/name
    assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],name
visual=dict(schema='n01/wp2-root-representative-render-QA/v1',UTC=now,status='PASS_REPRESENTATIVE_VIEWS',
    report=ref(REPORT/'N01_WP2_014_REPORT.html'),
    original_HTML_top_viewport=png(HERE/'html_qa_01/top.png'),
    exact_later_section_previews={n:png(HERE/('html_qa_02/'+n+'.png')) for n in ('wave','conditional','readback')},
    later_section_bindings=ref(HERE/'html_qa_02/PREVIEW_BINDINGS.json'),
    original_English_figures={n:png(REPORT/('figures/'+n+'.png')) for n in ('01_wave_usefulness','02_absolute_tolerance','03_conditional_gradients')},
    observations=['Chinese text, full precision failed cell, all12 table rows and SVG labels are legible.',
        'Conditional-Q table and saved-row scope/covariance/MCSE table fit the original CSS without horizontal truncation.',
        'Three original English figures inspected; 1.988778858954% remains below the2% line; absolute-T failures visible.'],
    native_Edge_exit_codes=[0,0,0,0],
    scope='Original top HTML viewport plus exact later section subtrees with original CSS/assets; not an interactive full-page scroll test.',
    harness_limitation='Initial headless fragment captures blank/clipped later content; preserved inhtml_qa_01. Exact-subtree previews bypassed this capture problem without modifying the final report.',
    original_report_modified=False,new_scientific_CP_NN_AD_G=0)
visual_path=HERE/'REPORT_VISUAL_QA.json';write(visual_path,visual)

backup=Path('D:/GAN_backups/N01/wp2_014_20261015/closeout_01')
assert not backup.exists()
backup.mkdir(parents=True)
backup_files=[REPORT/name for name in files]+[REPORT/'report_manifest.json',qa_path,visual_path,
    HERE/'FINAL_DEVELOPMENT_SELECTION.json',HERE/'NATIVE_AND_INDEPENDENT_CLOSEOUT.json',
    OUT/'postrun_independent_02/CLOSEOUT_REVIEW.json',OUT/'readback_execution_01/WAKE_COMPLETION_SUMMARY.json',
    OUT/'postrun_root_01/NATIVE_COMPLETION_READBACK.json',OUT/'collection_01/COLLECTION_RECEIPT.json',
    OUT/'collection_01/REMOTE_FULL_INVENTORY.json',OUT/'readback_collection_01/COLLECTION_RECEIPT.json',
    OUT/'postrun_report_impl_01/build_report.py',OUT/'postrun_report_impl_01/REPORT_BUILD_RECEIPT.json']
backup_files += list((ROOT/'runs/wp2_014_20261015/independent_readback_collected_01').rglob('*'))
backup_files=[p for p in dict.fromkeys(backup_files) if p.is_file()]
assert sum(p.stat().st_size for p in backup_files)<32*1024**2
inventory={}
for path in backup_files:
    assert path.suffix.lower() not in ('.pt','.npz','.tar','.gz')
    name=path.relative_to(ROOT).as_posix();target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target);assert sha(path)==sha(target)
    inventory[name]=dict(sha256=sha(target),bytes=target.stat().st_size)
write(backup/'MANIFEST.json',dict(UTC=now,files=inventory,original_raw_or_head_weights_included=False))
(backup/'SHA256SUMS.txt').write_text(''.join(row['sha256']+'  '+name+'\n' for name,row in sorted(inventory.items())),encoding='utf-8')
backup_receipt=OUT/'D_CLOSEOUT_BACKUP_RECEIPT.json'
write(backup_receipt,dict(UTC=now,status='all_copied_files_SHA256_verified',destination=str(backup),
    files=len(inventory),bytes=sum(v['bytes'] for v in inventory.values()),manifest_sha256=sha(backup/'MANIFEST.json'),
    all_scientific_data_on_E_D=True,original_raw_or_head_weights_included=False))

native=read(OUT/'postrun_root_01/NATIVE_COMPLETION_READBACK.json')
counts=native.get('actual_counts',native.get('actual_counters'))
assert counts['G_updates']==8000 and counts['new_AD_queries']==9872
decision=read(HERE/'FINAL_DEVELOPMENT_SELECTION.json')
assert decision['development_path']=='C1' and decision['bounded_saved_row_readback_complete'] is True
assert decision['weights_promoted'] is False and decision['full_distance_replay_from_saved_rows'] is False
receipt_path=OUT/'FINAL_RESULT_RECEIPT.json'
receipt=dict(schema='n01/wp2-final-result/v1',UTC=now,status='engineering_experiment_readback_report_complete',
    run_id='wp2_014_20261015',science_source_commit='e116368c4f699243a60598c03865bcb4f601be35',
    saved_array_audit_source_commit='cde0b8db49c49d7beab23011cb5e95f5627cee6d',
    all_four_trials_native0=True,training_and_supervisor_native_exit=0,actual_counts=counts,
    cumulative_training_before=159667,cumulative_training_after=167667,additional_G_after_original014=0,
    completed_short_candidate_runs=4,maximum_short_candidate_runs=12,
    bounded_saved_array_audit_complete=True,raw_NPZ_files176_retained_remote=True,raw_NPZ_transferred=False,
    independent_complete_distribution_loss_replay=False,independent_C1_median_refit=False,
    independent_review=ref(OUT/'postrun_independent_02/CLOSEOUT_REVIEW.json'),
    development_selection=ref(HERE/'FINAL_DEVELOPMENT_SELECTION.json'),
    selected_development_codepath='C1',C0_wave_useful_cells=0,C1_wave_useful_cells=5,ALL6_wave_usefulness_pass=False,
    C1_seed1_fast_unrounded_improvement_fraction=0.019887788589535327,
    C0_mean12_final_wave=0.15255321144257886,C1_mean12_final_wave=0.15042006897712382,
    calibrated_positive_T_noninferiority='Both candidates15components x6cells pass',absolute_T_all_pass=False,
    uncalibrated_T='pending_not_automatic_PASS',weights_promoted=False,scientific_acceptance=False,
    scientific_status='response_match_not_established',cross_acquisition_physical_reuse='not_verified',
    report_MD=ref(REPORT/'N01_WP2_014_REPORT.md'),report_HTML=ref(REPORT/'N01_WP2_014_REPORT.html'),
    report_manifest=ref(REPORT/'report_manifest.json'),report_content_QA=ref(qa_path),report_visual_QA=ref(visual_path),
    D_closeout_backup=ref(backup_receipt),
    next_stage='WP3/015 implementation and independent review; mean gate before any model computation',
    whole_six_workpackage_project_complete=False)
write(receipt_path,receipt)
metadata=[ROOT/p for p in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
snapshot=HERE/'metadata_before_report_closeout';snapshot.mkdir(exist_ok=False)
for path in metadata:shutil.copy2(path,snapshot/path.name)
master,program,jobs,current=[read(p) for p in (metadata[0],metadata[1],metadata[2],CURRENT)]
assert program['new_training_updates']==167667
result_ref=dict(**ref(receipt_path),status=receipt['status'],scientific_acceptance=False,weights_promoted=False)
report_ref=dict(MD=ref(REPORT/'N01_WP2_014_REPORT.md'),HTML=ref(REPORT/'N01_WP2_014_REPORT.html'),visual_QA=ref(visual_path))
for value in (master,program,current):
    value.update(wp2_completed=True,wp2_result=result_ref,wp2_report=report_ref,wp2_report_visual_QA_complete=True,updated_UTC=now)
wp=next(v for v in master['work_packages'] if v['id']=='WP2')
wp.update(status='completed_optimizer_path_selected_weights_not_promoted',result=result_ref,report=report_ref)
wp['execution']['status']='completed_native0_selectively_collected_independently_reviewed_reported'
for job in jobs['jobs']:
    if job['job_id']=='N01_WP2_014_20261015':
        job.update(status='completed_native0_collected_independently_reviewed_reported_closed',result=result_ref,report=report_ref)
        job['execution']['status']=wp['execution']['status']
    elif job['job_id']=='N01_WP2_014_SAVED_ARRAY_READBACK_20261015':
        job['status']='completed_native0_collected_independently_reviewed_closed'
jobs['updated_UTC']=now
text=metadata[3].read_text(encoding='utf-8-sig')
text=text.replace('当前014正式报告正在编制。','014正式报告、三幅英文图和六份CSV已完成内容核验及代表性渲染检查。',1)
text=text.replace('[开发选择](receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json)',
    '[014中文报告](reports/wp2_014_20261015/N01_WP2_014_REPORT.md) · [014 HTML](reports/wp2_014_20261015/N01_WP2_014_REPORT.html) · [014最终回执](receipts/wp2_014_20261015/FINAL_RESULT_RECEIPT.json) · [开发选择](receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json)',1)
metadata[3].write_text(text,encoding='utf-8')
current['plan_sha256']=sha(metadata[3])
assert master['active_scientific_jobs']==program['active_scientific_jobs']==jobs['active_scientific_jobs']==current['active_scientific_jobs']==0
save(metadata[0],master)
for value in (program,current):
    if 'master_plan_sha256' in value:value['master_plan_sha256']=sha(metadata[0])
for path,value in ((metadata[1],program),(metadata[2],jobs),(CURRENT,current)):save(path,value)
write(HERE/'FINAL_REPORT_CLOSEOUT_STATE.json',dict(UTC=now,final_result=ref(receipt_path),
    metadata={str(p):sha(p) for p in metadata},cumulative_training_updates=167667,new_CP_NN_AD_G=0,
    report_complete=True,whole_project_complete=False))
print(json.dumps(dict(status=receipt['status'],final_result_sha256=sha(receipt_path),D_backup_files=len(inventory),
    report_MD_sha256=sha(REPORT/'N01_WP2_014_REPORT.md'),new_CP_NN_AD_G=0,cumulative_training_updates=167667)))
