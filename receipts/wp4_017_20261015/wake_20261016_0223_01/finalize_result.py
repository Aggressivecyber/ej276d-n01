"""Record the completed bounded017 result; no remote or training operations."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

root = Path('E:/GAN/research/N01')
out = Path(__file__).resolve().parent
current = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')
review = out / 'saved_support_review_01'
num = json.loads((review / 'NUMERIC_READBACK.json').read_bytes())
cpu = json.loads((review / 'EXECUTION_RECEIPT.json').read_bytes())
report = root / 'reports/wp4_017_20261015/REPORT.md'
refs = {}
for key, path in [('report', report), ('numeric', review / 'NUMERIC_READBACK.json'),
                  ('review', review / 'REVIEW.md'), ('review_summary', review / 'SUMMARY.json'),
                  ('CPU_execution', review / 'EXECUTION_RECEIPT.json'),
                  ('terminal', out / 'TERMINAL_RECONCILIATION.json'),
                  ('collection', out / 'collection_01/COLLECTION_RECEIPT.json'),
                  ('inventory', out / 'collection_01/REMOTE_FULL_INVENTORY.json'),
                  ('automation_pause', out / 'AUTOMATION_PAUSE_RECEIPT.json')]:
    refs[key] = dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
result = dict(schema='n01/wp4-017-bounded-stage1-result/v1', UTC=stamp,
    run_id='wp4_017_20261015', status='bounded_stage1_closed_primary_support_pending',
    source_commit=num['source_commit'], configuration_sha256=num['config_sha256'], native_exit_code=0,
    executed_trials=['T1_paired_seed0', 'T1_paired_seed1'], G_updates_per_trial=2000,
    actual_G_updates=4000, actual_AD_queries=4000, cumulative_training_updates=183667,
    new_G_difference_on_wake=3861, producer_required_cells=2688, producer_pass_cells=2616,
    pending_cells=72, failed_cells=0,
    pending_reason='common original-parent count and weight mass below fixed95percent support',
    low_Q_original_parent_denominator=435,
    teacher_supported_parents=dict(fast=407, slow=407, common_level=408),
    common_supported_parents=dict(fast=402, slow=403, common_level=404),
    saved_rise_undefined_flags_match_prepeak_censor_counts=True,
    other_eight_scalars_supported_parents=435, bounded_saved_support_review_complete=True,
    readback_scope=num['scope'], readback_npz_files=len(num['decoded_npz_files']),
    readback_cached_Q_npy_files=len(num['decoded_npy_files']), analytic_variance_cells_recomputed=24,
    all_original_parents_in_analytic_check=6593, full_primary_metrics_independently_recomputed=False,
    stage2_allowed=False, stage2_entered=False, remaining_six_trials_executed=False,
    unpaired_or_T2_recovery_evaluated=False,
    scientific_interpretation='Full conditional recovery remains inconclusive under the fixed support gate; no structural model-failure conclusion.',
    next_action='Keep the fixed256-point hard observation; define a prospective joint representation and score for prepeak-censored rise before a future protocol. Do not reopen017 or start real long training.',
    scientific_status='response_match_not_established', scientific_acceptance=False,
    T1_heads_promoted_to_real_deployment=False, all_six_work_packages_complete=False,
    active_scientific_jobs=0, automation_status='PAUSED', evidence=refs)
result_path = root / 'receipts/wp4_017_20261015/FINAL_RESULT_RECEIPT.json'
with result_path.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
result_ref = dict(path=result_path.relative_to(root).as_posix(), bytes=result_path.stat().st_size,
                  sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
paths = [root / n for n in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')] + [current]
master, program, jobs, entry = [json.loads(x.read_bytes()) for x in (paths[0], paths[1], paths[2], current)]
execution = copy.deepcopy(program['wp4_017_execution'])
execution['historical_startup_followup'] = execution.pop('followup')
execution.update(status='bounded_stage1_closed_primary_support_pending_reviewed',
    bounded_saved_support_review_complete=True, saved_support_review=refs['review'],
    saved_support_numeric_receipt=refs['numeric'], independent_numerical_readback_complete=False,
    full_primary_metrics_independently_recomputed=False,
    independent_numerical_readback_scope='saved derived masks/counts/groups/weights/coverage support and analytic charge variance only',
    result=result_ref, report=refs['report'],
    followup=dict(automation_id='n01', status='PAUSED', receipt=refs['automation_pause']))
for value in (master, program, entry):
    value.update(status='WP4_017_closed_support_pending_saved_array_reviewed',
        current_stage='WP4_017_closed_primary_support_pending', active_scientific_jobs=0,
        active_run_id=None, pending_run_id=None, wp4_017_execution=copy.deepcopy(execution),
        wp4_017_support_array_readback_complete=True, wp4_017_result=result_ref, wp4_017_report=refs['report'],
        new_training_updates=183667, cumulative_training_updates=183667,
        wp4_017_G_updates=4000, wp4_017_AD_queries=4000, current_run_training_updates=0,
        current_followup_status='n01_PAUSED_no_pending_long_job', followup_automation_id='n01',
        followup_automation_status='PAUSED', followup_next_run_local=None,
        followup_receipt=refs['automation_pause'], updated_UTC=stamp,
        next_action=result['next_action'], scientific_status='response_match_not_established')
master.update(current_action=result['next_action'], active_research=None, active_audit=None,
    active_scientific_jobs_status_basis='Collected native0 stage1:4000G/AD and bounded saved-support CPU readback exit0; no active scientific process.',
    followup=refs['automation_pause'])
job = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_WP4_017_20261015')
job.update(status=execution['status'], execution=copy.deepcopy(execution), result=result_ref,
           report=refs['report'], bounded_saved_support_review_complete=True,
           full_primary_metrics_independently_recomputed=False)
readjob = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_WP4_017_SUPPORT_ARRAY_READBACK_20261016')
readjob.update(status='completed_native0_bounded_saved_support_review', completed_recorded_UTC=stamp,
    native_exit_code=cpu['exit_code'], actual_saved_npz_files=len(num['decoded_npz_files']),
    actual_cached_Q_npy_files=len(num['decoded_npy_files']), actual_scope=num['scope'],
    remote_numpy_elapsed_seconds=num['elapsed_seconds'], SSH_round_trip_elapsed_seconds=cpu['elapsed_seconds'],
    peak_RSS_KiB=num['peak_RSS_KiB'], result=refs['numeric'], review=refs['review'])
jobs.update(active_scientific_jobs=0, status='no_active_scientific_job_WP4_stage1_support_pending_reviewed', updated_UTC=stamp)
wp4 = next(w for w in master['work_packages'] if w['id'] == 'WP4')
wp4.update(status='017_bounded_stage1_closed_support_pending_no_stage2', execution=copy.deepcopy(execution),
    result=result_ref, report=refs['report'], qualification='not_established_under_fixed_primary_support_gate')
wp5 = next(w for w in master['work_packages'] if w['id'] == 'WP5')
wp5['entry_condition_status'] = 'not_started; WP4 primary qualification remains pending, no real long-training continuation.'
old = paths[3].read_text(encoding='utf-8-sig')
marker = '以下保留此前阶段说明。'
prefix = '''# N01 当前执行：WP4/017 首阶段闭合，整体资格为支持不足 pending

017两次T1配对各完成2000 G/AD并native0退出；2616 producer pass、72 pending、0fail，余下六项未进入。低Q组原435父，教师自身rise支持仅407–408父，低于固定95%要求；保存rise缺失与峰前截尾标记对应。限定数组复核确认支持/权重/索引及24个解析方差单元，不是完整硬波形或损失重放。累计训练183667，当前无科学作业，n01已暂停；科学状态仍为response_match_not_established。

下一步保留256点硬观察口径，前瞻地定义峰前截尾时间量与可定义rise的联合评分，再另行冻结后续合同。017原门槛和pending结论保持，stage2及真实长训不自动启动，T1合成参数不作为已验证实测部署权重；六工作包尚未全部完成。

[017报告](reports/wp4_017_20261015/REPORT.md) · [最终结果回执](receipts/wp4_017_20261015/FINAL_RESULT_RECEIPT.json) · [限定数组复核](receipts/wp4_017_20261015/wake_20261016_0223_01/saved_support_review_01/REVIEW.md) · [016报告](reports/wp3_016_20261015/REPORT.md)

'''
with paths[3].open('w', encoding='utf-8', newline='\n') as stream:
    stream.write(prefix + old[old.index(marker):])
with paths[0].open('w', encoding='utf-8', newline='\n') as stream:
    json.dump(master, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
program['master_plan_sha256'] = entry['master_plan_sha256'] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
entry['plan_sha256'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
for path, value in ((paths[1], program), (paths[2], jobs), (current, entry)):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
print(json.dumps(dict(status=program['status'], result_receipt=str(result_path),
    result_sha256=result_ref['sha256'], training_total=183667, active_jobs=0,
    support_review_complete=True, full_primary_independently_recomputed=False)))
