"""Close015's bounded review and preserve the remaining project work."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
CURRENT = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    base = ROOT / 'receipts/wp3_015_20261015'
    review_path = base / 'postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json'
    selection_path = base / 'postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json'
    native_path = base / 'readback_execution_01/WAKE_COMPLETION_SUMMARY.json'
    review, selection, native = [read(p) for p in (review_path, selection_path, native_path)]
    assert review['passed'] and not review['blockers']
    assert selection['passed'] and not selection['provisional'] and not selection['readback_pending']
    assert selection['selected_research_reference'] == 'M0' and not selection['selected_mean_adapter_enabled']
    assert not selection['weights_promoted'] and not selection['scientific_acceptance']
    assert selection['independent_raw_saved_array_readback_complete']
    assert not selection['full_empirical_loss_independently_replayed']
    assert native['native_child_exit_code'] == native['native_supervisor_exit_code'] == 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    paths = [ROOT / p for p in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')] + [CURRENT]
    snapshot = HERE / 'metadata_before_independent_completion'
    snapshot.mkdir(exist_ok=False)
    for p in paths:
        shutil.copy2(p, snapshot / p.name)
    master, program, jobs, current = [read(p) for p in (paths[0], paths[1], paths[2], CURRENT)]
    assert program['new_training_updates'] == 175667
    assert all(v['active_scientific_jobs'] == 0 for v in (master, program, jobs, current))
    selection_ref = dict(**ref(selection_path), selected_research_reference='M0', mean_adapter_enabled=False,
        status=selection['status'], weights_promoted=False, scientific_acceptance=False)
    review_ref = ref(review_path)

    def readback(old):
        value = copy.deepcopy(old)
        value.update(status='completed_native0_collected_independently_reviewed',
            final_independent_local_review_complete=True, independent_raw_readback_complete=True,
            bounded_saved_array_readback_complete=True, independent_review=review_ref,
            scientific_acceptance=False)
        return value

    def execution(old):
        value = copy.deepcopy(old)
        value.update(status='completed_native0_selectively_collected_independently_reviewed_report_pending',
            collection_complete=True, independent_readback_complete=True,
            independent_saved_array_review=review_ref, development_selection=selection_ref)
        return value

    for value in (master, program, jobs, current):
        value.update(updated_UTC=now, latest_subagent_model_preference='GPT-6.1-Sol Max')
        if 'wp3_readback_execution' in value:
            value['wp3_readback_execution'] = readback(value['wp3_readback_execution'])
        if 'wp3_execution' in value:
            value['wp3_execution'] = execution(value['wp3_execution'])
        if 'wp3_terminal_readback' in value:
            value['wp3_terminal_readback'].update(collection_complete=True, independent_readback_complete=True,
                independent_saved_array_scope='36pairs/161SHA/160evaluationNPZ/1898784hardrows; no full distance replay')
        if 'wp3_provisional_development_selection' in value:
            value['wp3_provisional_development_selection'].update(role='historical provisional decision retained',
                superseded_by=ref(selection_path))
        value.update(wp3_015_development_selection=selection_ref, wp3_015_independent_review=review_ref)
    for value in (master, program, current):
        value.update(status='WP3_015_independently_closed_report_in_progress',
            current_stage='WP3_015_report_and_016_design_in_progress',
            wp3_collection_complete=True, wp3_independent_readback_complete=True,
            next_action='Finish015 report and representative render QA; implement and freeze the two-trial016 charge test.',
            wp3_015_scientific_status='response_match_not_established')
    wp3 = next(x for x in master['work_packages'] if x['id'] == 'WP3')
    wp3.update(status='015_independently_closed_report_pending_016_design',
        execution=execution(wp3['execution']), saved_array_readback=readback(wp3['saved_array_readback']),
        development_selection=selection_ref, independent_saved_array_review=review_ref)
    wp3['terminal_readback'].update(collection_complete=True, independent_readback_complete=True)
    wp3['provisional_development_selection'].update(role='historical provisional decision retained', superseded_by=ref(selection_path))
    for job in jobs['jobs']:
        if job['job_id'] == 'N01_WP3_015_20261015':
            job.update(status='completed_native0_selectively_collected_independently_reviewed_report_pending',
                execution=execution(job['execution']), collection_complete=True, independent_readback_complete=True,
                development_selection=selection_ref, independent_saved_array_review=review_ref)
            job['terminal_readback'].update(collection_complete=True, independent_readback_complete=True)
            job['provisional_development_selection'].update(role='historical provisional decision retained', superseded_by=ref(selection_path))
        elif job['job_id'] == 'N01_WP3_015_SAVED_ARRAY_READBACK_20261015':
            job.update(status='completed_native0_collected_independently_reviewed',
                execution=readback(job['execution']), independent_review=review_ref,
                independent_readback_complete=True)
    text = paths[3].read_text(encoding='utf-8-sig')
    marker = '以下保留此前阶段说明。'
    assert marker in text
    prefix = '''# N01 当前执行：WP3/015 独审闭合，报告与016设计正在完成

015独审后选择M0为后续开发研究参考，均值适配关闭。M0/M1的12格wave均值分别为0.15012505860962605 /0.15243191496746586，共同初始为0.15042006897712382；M0用途6/6、M1为0/6。原014 C1的5/6历史保持。

四trial共8000G/AD和保存数组CPU复核均native0。36对、161份SHA/160份评价NPZ、1,898,784行与285项操作完整，独立闭合通过；19个CPU小文件已核SHA。完整经验距离、C1 median和preADC384未独立重放；REAL均值摘要只核SHA。原NPZ留远端，累计训练175667不变。

绝对T仍有失败：M0峰高/rise/FWHM约6.7/7.4/4.8倍T，baseline mean约33倍T。科学状态response_match_not_established，权重未科学晋级。下一项仅用剩余两次WP3短试检验两参数正值均值校正电荷分支；原A/005/M0冻结，峰形和基线残差分别保留。当前无运行中的科学作业，后续统一GPT-6.1-Sol Max。

[015最终独审](receipts/wp3_015_20261015/postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json) · [研究参考选择](receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json) · [016设计](wp3_016_design.json)

'''
    paths[3].write_text(prefix + text[text.index(marker):], encoding='utf-8')
    write(paths[0], master)
    mh = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    program['master_plan_sha256'] = current['master_plan_sha256'] = mh
    current['plan_sha256'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    for p, value in ((paths[1], program), (paths[2], jobs), (CURRENT, current)):
        write(p, value)
    write(HERE / 'INDEPENDENT_COMPLETION_RECONCILIATION.json', dict(
        schema='n01/wp3-final-independent-reconciliation/v1', UTC=now,
        independent_review=review_ref, selection=selection_ref, native_completion=ref(native_path),
        active_scientific_jobs=0, cumulative_training_updates=175667, newly_added_scientific_updates=0,
        report_complete=False, whole_six_workpackage_project_complete=False,
        next_run_016_started=False, scientific_acceptance=False,
        updated_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    print(json.dumps(dict(status='015_independently_closed_report_pending', reference='M0', mean_enabled=False,
                         new_scientific_updates=0, cumulative=175667)))


if __name__ == '__main__':
    main()
