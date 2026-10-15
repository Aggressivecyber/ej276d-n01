"""Reconcile the collected, native-zero CPU audit without adding scientific cost."""
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
    native_path = ROOT / 'receipts/wp3_015_20261015/readback_execution_01/WAKE_COMPLETION_SUMMARY.json'
    native = read(native_path)
    assert native['native_child_exit_code'] == native['native_supervisor_exit_code'] == 0
    assert native['operations_entered'] == native['operations_completed'] == 285
    assert not native['unknown_inflight_ids'] and not native['failed_ids'] and not native['truncated_tail']
    assert native['full_finite_scope_complete'] and native['array_result']['passed']
    assert native['collection']['all_files_SHA256_and_bytes_verified']
    assert native['actual_counts'] == native['attempted_counts']
    assert all(v == 0 for v in native['scientific_operations'].values())
    for item in native['evidence'].values():
        p = ROOT / item['path']
        assert p.stat().st_size == item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
    paths = [ROOT / p for p in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')] + [CURRENT]
    snapshot = HERE / 'metadata_before_cpu_completion'
    snapshot.mkdir(exist_ok=False)
    for path in paths:
        shutil.copy2(path, snapshot / path.name)
    master, program, jobs, current = [read(p) for p in (paths[0], paths[1], paths[2], CURRENT)]
    assert program['new_training_updates'] == 175667
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    def completed(old):
        result = copy.deepcopy(old)
        result.update(status='completed_native0_collected_final_local_review_pending',
            historical_analysis_pid=native['worker_pid'], historical_supervisor_pid=native['supervisor_pid'],
            analysis_pid=None, supervisor_pid=None, last_verified_UTC=native['wake_snapshot_UTC'],
            actual_completed_counts_last_verified=native['actual_counts'], attempted_counts_last_verified=native['attempted_counts'],
            unknown_inflight_ids=[], failed_ids=[], counts_are_durable_lower_bounds=False,
            actual_current_completion='native0_COMPLETE_all285_operations_completed',
            native_child_exit_at_snapshot=0, native_supervisor_exit_at_snapshot=0,
            native_exit_code=0, supervisor_exit_code=0, completed_UTC=native['supervisor_exit_UTC'],
            terminal_readback=ref(native_path), collection=native['collection'], collection_complete=True,
            bounded_saved_array_readback_complete=True, final_independent_local_review_complete=False,
            original_raw_NPZ_transferred=False, complete_distribution_loss_raw_replay=False,
            resource_snapshot_role='historical startup only; final resources recorded in terminal_readback',
            actual_terminal_resources=native['actual_resources'], scientific_operations=native['scientific_operations'],
            CP_NN_AD_training=0, scientific_acceptance=False)
        return result

    for value in (master, program, jobs, current):
        value.update(updated_UTC=now, active_scientific_jobs=0,
            active_scientific_jobs_status_basis='collected native-zero CPU015 completion with 285 completed operations',
            followup_last_delivered_UTC='2026-10-15T06:03:03.848Z',
            current_followup_status='single_occurrence_delivered_this_turn_no_new_long_job',
            scientific_status='response_match_not_established')
        if isinstance(value.get('followup'), dict):
            value['followup'].update(delivery_status='single_0603UTC_wake_delivered_this_turn', saved_app_status_is_historical=True)
        if 'wp3_readback_execution' in value:
            value['wp3_readback_execution'] = completed(value['wp3_readback_execution'])
    for value in (master, program, current):
        value.update(status='WP3_015_CPU_complete_final_review_and_report_in_progress',
            active_audit=None, active_research='wp3_015_20261015_closeout', current_work_package='WP3',
            current_stage='015_saved_array_complete_final_review_report_pending',
            next_action='Finalize015 independent review/report, then the bounded remaining WP3 module decision.')
    wp3 = next(p for p in master['work_packages'] if p['id'] == 'WP3')
    wp3['status'] = '015_native0_array_readback_complete_final_local_review_report_pending'
    wp3['saved_array_readback'] = completed(wp3['saved_array_readback'])
    for job in jobs['jobs']:
        if job['job_id'] == 'N01_WP3_015_SAVED_ARRAY_READBACK_20261015':
            job.update(status='completed_native0_collected_final_local_review_pending', native_exit_code=0,
                supervisor_native_exit_code=0, completed_UTC=native['supervisor_exit_UTC'],
                historical_analysis_pid=native['worker_pid'], historical_supervisor_pid=native['supervisor_pid'],
                analysis_pid=None, supervisor_pid=None, actual_array_counts=native['actual_counts'],
                actual_array_count_basis='collected COMPLETE and complete285-operation journal',
                collection_complete=True, terminal_readback=ref(native_path))
            job['execution'] = completed(job['execution'])
    old = paths[3].read_text(encoding='utf-8-sig')
    marker = '以下保留此前阶段说明。'
    assert marker in old
    prefix = '''# N01 当前执行：WP3/015 保存数组复核完成，正在闭合报告

015四次短试均2000G，实际8000G/AD已入账，累计175667；搜索短试8/12，WP3已4/6。原A/005和冻结源保持不变。

保存数组CPU复核已原生退出0并回收：36对库、161份NPZ身份SHA、160份评价数组解码、1,898,784行硬输出、285项操作完整。19个小文件逐SHA核验，原NPZ保留远端，新增CP/NN/AD/G均0。完整距离与C1尺度未重新拟合或重放。

保存JSON比较选择M0研究参考，均值适配关闭；最终独立闭合和报告正在进行。M0用途门6/6、M1为0/6，绝对容忍度未全部通过，科学状态仍为response_match_not_established。当前无运行中的科学作业；后续按主计划推进剩余受控模块。

[CPU完成回执](receipts/wp3_015_20261015/readback_execution_01/WAKE_COMPLETION_SUMMARY.json) · [015训练终点](receipts/wp3_015_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json)

'''
    paths[3].write_text(prefix + old[old.index(marker):], encoding='utf-8')
    write(paths[0], master)
    mh = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    for value in (program, current):
        value['master_plan_sha256'] = mh
    current['plan_sha256'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    for path, value in ((paths[1], program), (paths[2], jobs), (CURRENT, current)):
        write(path, value)
    receipt = dict(schema='n01/wp3-cpu-terminal-reconciliation/v1', UTC=now,
        native_completion=ref(native_path), active_scientific_jobs=0, cumulative_training_updates=175667,
        newly_added_scientific_updates=0, scientific_acceptance=False,
        final_independent_local_review_pending=True, whole_six_workpackage_project_complete=False,
        updated_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    write(HERE / 'CPU_COMPLETION_RECONCILIATION.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('active_scientific_jobs', 'cumulative_training_updates', 'newly_added_scientific_updates')}))


if __name__ == '__main__':
    main()
