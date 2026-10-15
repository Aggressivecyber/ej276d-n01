"""Reconcile current 014 references with the immutable saved closeout receipts."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
CURRENT = ROOT.parents[1] / 'CURRENT_SIM2REAL_PROGRAM.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    snapshot = HERE / 'metadata_before_normalization'
    snapshot.mkdir(exist_ok=False)
    paths = [ROOT / p for p in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')]
    paths.append(CURRENT)
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    native_path = ROOT / 'receipts/wp2_014_20261015/readback_execution_01/WAKE_COMPLETION_SUMMARY.json'
    review_path = ROOT / 'receipts/wp2_014_20261015/postrun_independent_02/CLOSEOUT_REVIEW.json'
    decision_path = ROOT / 'receipts/wp2_014_20261015/postrun_engineering_01/DEVELOPMENT_SELECTION.json'
    native, review, decision = [read(p) for p in (native_path, review_path, decision_path)]
    assert native['child_native_exit_code'] == native['supervisor_native_exit_code'] == 0
    assert native['operations_entered'] == native['operations_completed'] == 308
    assert not native['unknown_inflight_ids'] and not native['failed_ids'] and not native['truncated_tail']
    assert review['passed'] is True and review['status'] == 'PASS'
    assert decision['development_path'] == 'C1' and decision['weights_promoted'] is False
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    final_decision_path = HERE / 'FINAL_DEVELOPMENT_SELECTION.json'
    assert not final_decision_path.exists()
    final_decision = copy.deepcopy(decision)
    final_decision.update(status='independently_reviewed_development_codepath_only',
        original_provisional_decision=ref(decision_path), independent_saved_array_review=ref(review_path),
        finalized_UTC=now, weights_promoted=False, scientific_acceptance=False,
        independent_raw_readback_complete=True, bounded_saved_row_readback_complete=True,
        full_distance_replay_from_saved_rows=False, C1_median_fit_independently_recomputed=False)
    write(final_decision_path, final_decision)
    decision_ref = dict(**ref(final_decision_path), development_path='C1',
        status=final_decision['status'], weights_promoted=False,
        both_candidates_ALL6_wave_usefulness_pass=False, scientific_acceptance=False)

    def completed_readback(old):
        value = copy.deepcopy(old)
        value.update(status='completed_native0_collected_independently_reviewed',
            historical_analysis_pid=native['worker_pid'], historical_supervisor_pid=native['supervisor_pid'],
            analysis_pid=None, supervisor_pid=None, last_verified_UTC=native['wake_snapshot_UTC'],
            actual_completed_counts_last_verified=native['counters'], attempted_counts_last_verified=native['attempted_counters'],
            unknown_inflight_ids=[], counts_are_durable_lower_bounds=False,
            actual_current_completion='native0_COMPLETE_all308_operations_completed',
            native_child_exit_at_snapshot=0, native_supervisor_exit_at_snapshot=0,
            native_exit_code=0, supervisor_exit_code=0, completed_UTC=native['finished_UTC'],
            terminal_readback=ref(native_path), independent_review=ref(review_path),
            collection_complete=True, independent_raw_readback_complete=True,
            original_raw_NPZ_transferred=False, complete_distribution_loss_raw_replay=False,
            resource_snapshot_role='historical startup snapshot; final resources in terminal_readback',
            CP_NN_AD_training=0, scientific_acceptance=False)
        return value

    master, program, jobs, current = [read(p) for p in (paths[0], paths[1], paths[2], CURRENT)]
    for value in (master, program, current):
        if 'wp2_readback_execution' in value:
            value['wp2_readback_execution'] = completed_readback(value['wp2_readback_execution'])
        value['wp2_development_selection'] = decision_ref
        if 'wp2_provisional_development_selection' in value:
            value['wp2_provisional_development_selection']['superseded_by'] = ref(final_decision_path)
            value['wp2_provisional_development_selection']['role'] = 'historical provisional record retained'
        if 'wp2_terminal_readback' in value:
            value['wp2_terminal_readback'].update(collection_complete=True, independent_readback_complete=True,
                collection_scope='selected142_of318; original176raw_NPZ_retained_remote',
                independent_saved_array_scope='36pairs/176NPZ/1898784hard_rows; not complete loss replay')
        value['updated_UTC'] = now
    wp2 = next(p for p in master['work_packages'] if p['id'] == 'WP2')
    wp2['saved_array_readback'] = completed_readback(wp2['saved_array_readback'])
    wp2['development_selection'] = decision_ref
    wp2['provisional_development_selection']['role'] = 'historical provisional record retained'
    wp2['provisional_development_selection']['superseded_by'] = ref(final_decision_path)
    wp2['terminal_readback'].update(collection_complete=True, independent_readback_complete=True)
    wp2['execution']['status'] = 'completed_native0_selectively_collected_independently_reviewed_report_pending'
    wp2['execution']['collection_complete'] = True
    wp2['execution']['independent_readback_complete'] = True
    for job in jobs['jobs']:
        if job['job_id'] == 'N01_WP2_014_20261015':
            job['execution']['status'] = wp2['execution']['status']
            job['development_selection'] = decision_ref
            job['terminal_readback'].update(collection_complete=True, independent_readback_complete=True)
        elif job['job_id'] == 'N01_WP2_014_SAVED_ARRAY_READBACK_20261015':
            job['execution'] = completed_readback(job['execution'])
    jobs['updated_UTC'] = now
    # Consume only the local current-state label; the app's saved automation is changed separately.
    for value in (master, program, jobs, current):
        if isinstance(value.get('followup'), dict):
            value['followup']['delivery_status'] = 'single_0317UTC_wake_delivered_0318UTC_this_turn'
            value['followup']['saved_app_status_is_historical'] = True
    text = paths[3].read_text(encoding='utf-8-sig')
    marker = '以下保留此前阶段说明。'
    assert marker in text
    prefix = '''# N01 当前执行：WP2/014 独立闭合，准备 WP3/015

014 四次配对短试全部完成，每次 2000G，总计 8000G / 9872AD，原生计算与监督均退出0。累计训练167667，搜索短试已完成4次；015正在实现，尚未开始新计算，WP3最多6次短试的预算保持不变。

独立复核后，选择 C1 作为后续开发代码路径。C0/C1均满足已标定分量ALL6非劣；12个final完整wave距离均值分别为0.15255321144257886 / 0.15042006897712382。C1的seed1-fast相对A改善1.9887788589535327%，全部6格至少2%的用途门未通过；权重未晋级，科学状态仍为response_match_not_established。

保存数组CPU独立读回已原生退出0并回收19个小文件：36对库、176个NPZ的SHA、1,898,784行硬输出、4/8次重复矩与标量协方差/MCSE及保存梯度向量核验通过。完整wave/QPSD/noise经验距离没有从raw重新计算，C1 median没有独立重拟合，176个原始NPZ仍保存在服务器。CPU复核新增CP/NN/AD/训练均0。

下一项015比较同预算继续C1形状头与冻结随机头后训练小型零初始化均值适配器；先检验固定人口假设下可辨均值残差，再按冻结门决定是否训练。均值残差不等于真实条件均值已识别，基线DC仍单列为电子学限制。当前014正式报告正在编制。

[开发选择](receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json) · [独立闭合](receipts/wp2_014_20261015/postrun_independent_02/CLOSEOUT_REVIEW.json) · [原生终点与计数](receipts/wp2_014_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [015准备设计](wp3_015_design.json)

'''
    paths[3].write_text(prefix + text[text.index(marker):], encoding='utf-8')
    current['plan_sha256'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    assert program['new_training_updates'] == 167667
    assert master['active_scientific_jobs'] == program['active_scientific_jobs'] == jobs['active_scientific_jobs'] == current['active_scientific_jobs'] == 0
    for path, value in ((paths[0], master), (paths[1], program), (paths[2], jobs), (CURRENT, current)):
        write(path, value)
    receipt = dict(schema='n01/wp2-current-state-normalization/v1', UTC=now,
        status='completed_current_state_consistent_report_pending', cumulative_training_updates=167667,
        new_G_AD_NN_CP=0, scientific_acceptance=False, whole_six_workpackage_project_complete=False,
        independent_native_closeout=ref(native_path), independent_review=ref(review_path),
        development_selection=ref(final_decision_path), updated_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    write(HERE / 'CURRENT_STATE_RECONCILIATION.json', receipt)
    print(json.dumps({k:receipt[k] for k in ('status','cumulative_training_updates','new_G_AD_NN_CP')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
