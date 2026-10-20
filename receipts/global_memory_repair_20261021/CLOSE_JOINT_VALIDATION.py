from pathlib import Path
from datetime import datetime, timezone
import json

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
out = here / 'parent_audit_01'
stamp = datetime.now(timezone.utc).isoformat()
native = json.loads((out / 'RESULT.json').read_bytes())
exit_record = json.loads((out / 'exit_receipt.json').read_bytes())
analysis = json.loads((out / 'PARENT_POPULATION_ANALYSIS.json').read_bytes())
evaluation = json.loads((here / 'fixed_validation_fix01/EVALUATION_RESULT.json').read_bytes())
assert exit_record['exit_code'] == 0 and native['committed_optimizer_updates'] == 0
assert native['execution_counts']['response_rows'] == 843904
assert analysis['aggregate_reproduced'] and analysis['subsets'] == 2000
closure = dict(status='COMPLETE_POPULATION_SENSITIVITY_DIAGNOSED', UTC=stamp, native_exit_code=0,
    execution_counts=native['execution_counts'], new_optimizer_updates=0,
    FIT_subset_fraction_worsening=analysis['FIT_subset_fraction_worsening'],
    FIT_subset_fraction_wave_increase_above2pct=analysis['FIT_subset_fraction_above2pct'],
    report_percentile_in_subset_reference=analysis['REPORT_position_in_FIT_subset_reference'],
    conclusion='Global waveform improvement contains opposing source-Q population contributions; the observed REPORT degradation lies inside the128-parent FIT subsampling reference distribution.',
    limits='Conditional finite-population diagnostic, not an independent-acquisition result or formal p-value; does not establish overfitting or representation incapacity.',
    scientific_acceptance=False, no_more_optimizer_steps_registered=True)
(out / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
next_design = dict(status='DESIGN_ONLY_NOT_REGISTERED_OR_RUNNING', UTC=stamp,
    evidence=['9 new updates passed TRAIN hard joint guards; independent random-bank pilot improved PSD11.14%.',
        'All3 FIT noise hypotheses pass; fixed REPORT wave risks increase5.76%-6.50%.',
        'Changing REAL reference does not reverse the source-population effect.',
        '20.15% of2000 FIT128-parent subsets exceed2% wave degradation; macro-loss improvement does not protect every population.'],
    minimum_next_change='Add TRAIN-only joint local-Q/wave-feature memory to the existing16/32/64 soft-Q partitions; retain full-population objective and native hard observation.',
    purpose='Constrain conditional waveform response instead of relying solely on an aggregate waveform guard whose group contributions can cancel.',
    next_bounded_check='First validate one conditional-wave-aware direction against the existing Q/PSD/peak/ADC guards and fixed FIT-only population summaries; register exact calls before any forward.',
    prohibitions=['No additional steps appended to job158.', 'No parameter or architecture sweep.',
                  'REPORT rows never enter training, normalization, gradients or acceptance tuning.',
                  'No event IDs, chronological memory claims, physical-energy calibration claims, or declaration that247 parameters are incapable.'],
    preserved=['All6465 SOURCE FIT parents and18948 REAL FIT rows.', 'Existing immutable roles and source physics.',
               'Frozen original ADC/argmax/crop and endpoint thresholds.', 'Report full-Q coverage and sparse cells.'],
    unresolved='Stable conditional-shape improvement and future-acquisition transfer remain unestablished.',
    scientific_acceptance=False)
next_path = here / 'POPULATION_CONDITIONAL_NEXT_DESIGN.json'
next_path.write_bytes((json.dumps(next_design, indent=2) + '\n').encode())
summary = dict(UTC=stamp, new_training_updates=9, inherited_updates=1, science_accepted=False,
    roles={role: {h: dict(PSD_change_percent=v['changes_percent']['PSD_CDF'],
        wave_change_percent=v['changes_percent']['block_wave'], Q_change_percent=v['changes_percent']['Q_mass'],
        passes_all_consistency_checks=v['passes_consistency_check'],
        failed_checks=[k for k, passed in v['unchanged_pilot_checks'].items() if not passed])
        for h, v in entries.items()} for role, entries in evaluation['results'].items()},
    population_sensitivity=analysis, next_step_design=str(next_path.relative_to(root)))
figure = root / 'deliverables/global_memory_repair_20261021/joint_local_01'
(figure / 'VALIDATION_SUMMARY.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_REPORT_population_consistency_failed',
        active_scientific_jobs=0, active_job_id=None, active_run_id=None,
        pending_job_id=None, pending_registered_job_id=None, pending_next_run_id=None,
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0,
        current_job_ended_UTC=exit_record['ended_utc'], updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='Training and fixed evaluation completed;9 updates improved FIT but failed REPORT waveform consistency. Parent audit quantified population sensitivity.',
        next_action='Design-only conditional Q-wave memory/guard correction; no appended training steps or new scientific job registered.',
        followup_automation_id=None, followup_automation_status='DELETED', followup_status='DELETED_NO_WAITING_WORK',
        followup_next_run_local=None, followup_scheduled_local=None, next_wake_local=None)
    d['followup'] = dict(status='DELETED', automation_id='n01-psd', next_wake_local=None,
        receipt='receipts/global_memory_repair_20261021/joint_local_01/AUTOMATION_CLOSED.json')
    d['global_memory_repair'].update(status='REPORT_population_consistency_failed', active_scientific_jobs=0,
        current_committed_training_updates=0, most_recent_training_updates=9, inherited_checkpoint_updates=1,
        total_accepted_updates_across_pilots=17, current_native_execution_counts=native['execution_counts'],
        counts_asof_UTC=exit_record['ended_utc'], automatic_training_continuation=False,
        bounded_pilot_passed=True, fixed_validation_all_roles_pass=False, science_accepted=False,
        parent_audit_closure=str((out / 'CLOSURE_RECEIPT.json').relative_to(root)),
        next_step_design=str(next_path.relative_to(root)),
        latest_full_Q_figure=str((figure / 'GLOBAL_MEMORY_PSD_LINEAR.png').relative_to(root)),
        fixed_validation_summary=str((figure / 'VALIDATION_SUMMARY.json').relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 161
        d['jobs'][-1].update(status='completed_fixed_parent_population_diagnostic', native_exit_code=0,
            actual_execution_counts=native['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=str((out / 'CLOSURE_RECEIPT.json').relative_to(root)))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob161完成逐父事件复现与2000个FIT内128父事件子集分析，0更新。'
        '29.5%子集波形变差，20.15%超过2%；REPORT处于参考分布91.25百分位，仍在中间95%范围内。'
        '该条件人口诊断不能证明过拟合，但证实全局波形损失的不同Q组正负贡献会抵消。'
        '下一步仅形成TRAIN侧局部Q—波形记忆/约束设计，未自动扩步或登记新训练；'
        '全部当前任务结束，定时已删除，科学响应匹配仍未建立。\n')
print(json.dumps(dict(status='CLOSED_ALL_CURRENT_JOBS', active_scientific_jobs=0,
    most_recent_training_updates=9, report_consistency_passed=False, automation='DELETED', scientific_acceptance=False)))
