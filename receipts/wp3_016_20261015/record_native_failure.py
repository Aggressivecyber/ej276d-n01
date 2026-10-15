"""Reconcile the one saved terminal 016 snapshot; performs no remote calls."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CURRENT = Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def write(path, value, *, new=False):
    with path.open('x' if new else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    snapshot = OUT/'start_verification_01/REMOTE_STATE.json'
    summary = OUT/'native_execution_02/NATIVE_START_SUMMARY.json'
    source = OUT/'SOURCE_COMMIT_READBACK_FIX01.json'
    state, native, pinned = read(snapshot), read(summary), read(source)
    failed = state['runs/primary/artifacts/FAILED.json']
    terminal = state['runs/primary/supervisor/exit_receipt.json']
    progress = state['durable_child_progress']
    counts = failed['counters']
    assert failed['native_status'] == 'FAILED' and failed['artificial_only'] is False
    assert terminal['child_native_exit_code'] == terminal['exit_code'] == 1
    assert failed['error'] == '016 per-parent A canonical Q changed across draws'
    assert failed['error_type'] == 'ValueError'
    assert state['runs/primary/artifacts/FIRST_UPDATE_PROOF.json'] is None
    assert state['runs/primary/artifacts/COMPLETE.json'] is None
    assert all(v['alive'] is False for v in state['live_processes'].values())
    assert not state['protection_errors']
    assert failed['attempted_counters'] == counts
    assert failed['operations_entered'] == failed['operations_completed'] == 2929
    assert failed['inflight'] == {} and not progress['truncated_tail']
    assert not progress['unknown_inflight_ids'] and not progress['failed_ids']
    for key, value in counts.items():
        assert progress['completed_counts'].get(key, 0) == value, key
        assert progress['entered_counts'].get(key, 0) == value, key
        assert terminal['completed_counts'].get(key, 0) == value, key
    assert counts['G_updates'] == counts['new_AD_queries'] == counts['optimizer_steps'] == 0
    assert counts['old_A_rows'] == 32965 and counts['head_total_kernel_rows'] == 26372
    assert counts['charge_latent_scalar_draws'] == 26372 and counts['hard_ADC_rows'] == 237348
    assert counts['checkpoint_loads'] == 3 and counts['charge_initialization_fits'] == 1
    assert pinned['source_commit'] == native['source_commit']
    paths = [ROOT/n for n in ('MASTER_EXECUTION_PLAN.json', 'PROGRAM.json', 'jobs.json', 'README.md')] + [CURRENT]
    master, program, jobs, current = [read(p) for p in (paths[0], paths[1], paths[2], CURRENT)]
    assert program['new_training_updates'] == 175667
    job = next(j for j in jobs['jobs'] if j['job_id'] == 'N01_WP3_016_20261015')
    assert job['status'] == 'start_dispatched_first_update_verification_pending'
    backup = OUT/'metadata_before_native_failure_01'
    backup.mkdir(exist_ok=False)
    for path in paths:
        shutil.copy2(path, backup/path.name)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    receipt = dict(schema='n01/wp3-016-native-failure-reconciliation/v1', UTC=stamp,
        status='terminal_native1_before_first_G_update', source_commit=pinned['source_commit'],
        source_commit_readback=ref(source), only_snapshot=ref(snapshot), native_summary=ref(summary),
        producer_failed_UTC=failed['UTC'], native_exit_UTC=terminal['UTC'], snapshot_UTC=state['UTC'],
        error_type=failed['error_type'], error=failed['error'], actual_completed_counts=counts,
        attempted_counts=failed['attempted_counters'], operations_entered=2929, operations_completed=2929,
        unknown_inflight_ids=[], failed_operation_ids=[], truncated_journal=False,
        current_owned_supervisor_alive=False, current_owned_worker_alive=False, active_scientific_jobs=0,
        first_update_verified=False, automatic_restart=False, retry_authorized_by_this_receipt=False,
        no_scientific_training_candidate_completed=True, scientific_forward_and_RNG_costs_are_not_zero=True,
        cumulative_training_before016=175667, cumulative_training_updates=175667, newly_added_G_updates=0,
        scientific_status='response_match_not_established', formal_response_acceptance=False,
        diagnosis='Strict repeated-parent Q equality failed; numerical magnitude and root cause not captured by failed producer.',
        next_action='Collect preserved partial evidence, diagnose and independently review a minimal engineering repair before any replacement execution.')
    receipt_path = OUT/'NATIVE_FAILURE_RECONCILIATION.json'
    write(receipt_path, receipt, new=True)
    refs = copy.deepcopy(program['wp3_016_execution'])
    refs.update(status='terminal_native1_before_first_G_update', correct_start_verified=False,
        native_exit_code=1, supervisor_exit_code=1, failure_reconciliation=ref(receipt_path),
        last_verified_completed_counts=counts, unknown_inflight_ids=[],
        actual_costs_are_durable_lower_bounds_while_running=False, new_training_updates=0, new_AD_queries=0,
        analysis_pid=None, supervisor_pid=None, ended_UTC=terminal['UTC'],
        failure_remote_root=native['remote_root'], automatic_restart=False)
    for value in (master, program, current):
        value.update(status='WP3_016_native1_before_first_G_under_engineering_diagnosis',
            active_scientific_jobs=0, active_research=None, pending_run_id='wp3_016_20261015',
            current_stage='WP3_016_failure_preservation_and_local_diagnosis',
            wp3_016_execution=refs, wp3_execution=refs,
            wp3_016_G_updates=0, wp3_016_AD_queries=0,
            cumulative_training_updates=175667, current_run_training_updates=0,
            active_scientific_jobs_status_basis='terminal native1 plus both owned PIDs absent in the sole saved snapshot',
            next_action=receipt['next_action'], updated_UTC=stamp)
    program['new_training_updates'] = 175667
    program['training_update_accounting'] = '016 first native execution failed before G1; no G addition, all nontraining compute retained separately.'
    job.update(status='terminal_native1_before_first_G_update', execution=refs,
        actual_counters=counts, actual_counters_last_verified=counts, G_updates=0, new_AD_queries=0,
        correct_start_verified=False, analysis_pid=None, supervisor_pid=None,
        native_exit_code=1, supervisor_exit_code=1, failure_reconciliation=ref(receipt_path),
        native_execution_attempts=1, completed_short_candidate_runs=0)
    jobs.update(active_scientific_jobs=0, updated_UTC=stamp,
        active_scientific_jobs_status_basis=program['active_scientific_jobs_status_basis'])
    wp3 = next(w for w in master['work_packages'] if w['id'] == 'WP3')
    wp3.update(status='016_engineering_failure_before_training', execution=refs, execution016=refs)
    marker = '以下保留此前阶段说明。'
    previous = paths[3].read_text(encoding='utf-8-sig')
    assert marker in previous
    prefix = '''# N01 当前执行：WP3/016 首步前退出，保全与诊断中

016原生程序在初始评价保存前因同父事件Q逐位相等检查失败退出，首个训练更新尚未发生。唯一终点快照确认子进程和监督进程均已结束、native exit1、2929项进入与完成一致，无在途；当前活动科学作业0。累计训练仍为175667。已实际发生A前向32965行、形状/005/电荷各26372行、硬ADC237348行及3次权重加载，全部保留记账。

失败原目录与原始数据保留，正在一次回收partial证据并独立审查局部工程修复。未认定正确启动，未自动重启。015研究参考仍为均值适配关闭的M0，科学状态为response_match_not_established；后续使用GPT-6.1-Sol Max。

[016失败回执](receipts/wp3_016_20261015/NATIVE_FAILURE_RECONCILIATION.json) · [016合同](docs/WP3_016_CONFIGURATION.md) · [015报告](reports/wp3_015_20261015/REPORT.md)

'''
    paths[3].write_text(prefix+previous[previous.index(marker):], encoding='utf-8', newline='\n')
    write(paths[0], master)
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    program['master_plan_sha256'] = current['master_plan_sha256'] = digest
    current['plan_sha256'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    for path, value in ((paths[1], program), (paths[2], jobs), (CURRENT, current)):
        write(path, value)
    print(json.dumps(dict(status=receipt['status'], active_jobs=0, cumulative_G=175667,
        actual_A_rows=counts['old_A_rows'], actual_charge_draws=counts['charge_latent_scalar_draws'],
        receipt=ref(receipt_path))))


if __name__ == '__main__':
    main()
