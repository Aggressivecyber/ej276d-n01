from pathlib import Path
from datetime import datetime, timezone
import json

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
out = here / 'conditional_wave_01'
native = json.loads((out / 'RESULT.json').read_bytes())
exit_record = json.loads((out / 'exit_receipt.json').read_bytes())
result = json.loads((out / 'TRAINING_RESULT.json').read_bytes())
direction = json.loads((out / 'DIRECTION_ANALYSIS.json').read_bytes())
config = json.loads((root / 'configs/global_memory_conditional_wave_20261021.json').read_bytes())
assert exit_record['exit_code'] == 0 and native['committed_optimizer_updates'] == 0
assert native['execution_counts']['optimizer_directions'] == 1
assert native['execution_counts']['response_rows'] <= config['operation_budget']['response_rows']
assert result['endpoint']['B']['positive_metrics'] == result['endpoint']['inherited']['positive_metrics']
assert result['endpoint']['B']['flags'] == result['endpoint']['inherited']['flags']
assert not result['endpoint']['B']['passes_conditional_probe']
stamp = datetime.now(timezone.utc).isoformat()
closure = dict(status='COMPLETE_CONDITIONAL_WAVE_DIRECTION_INFEASIBLE', UTC=stamp,
    native_exit_code=0, new_optimizer_updates=0, inherited_checkpoint_updates=10,
    execution_counts=native['execution_counts'], elapsed_seconds=native['elapsed_seconds'],
    memory_implemented_and_native_executed=True, supported_cells=direction['supported_cells'],
    total_cells=direction['total_cells'], independent_endpoint_matches_inherited=True,
    scientific_acceptance=False,
    conclusion='Within the5 measured native probe directions, no nonzero combination improves the objective while satisfying all original and supported conditional-wave constraints. No update committed.',
    limits='This rejects the measured finite direction set at the fixed radius, not conditional memory or the247-parameter representation.',
    decisive_constraints=direction['binding_dual_constraints'])
(out / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
design = dict(status='DESIGN_ONLY_NOT_REGISTERED_OR_RUNNING', UTC=stamp,
    evidence='Job162 native Q-wave descent improved aggregate Q-wave risk0.4496% but worsened15 supported cells and bank1 ADC margin. Full native linear program returned zero; two conditional-cell dual constraints support the obstruction.',
    active_cells=[v['name'] for v in direction['binding_dual_constraints']],
    minimum_next_change='If advancing the optimizer, add actual positive-risk gradients of the two identified active conditional cells as directions. Retain the same memory, parameters, hard readout and thresholds; do not remove the guards.',
    required_registration='A new bounded single-direction job with explicit extra gradient/probe accounting is required before new model forwards. No automatic addition to job162.',
    hypothesis='Targeted cell descent may supply missing locally feasible directions; saved secants do not establish success.',
    report_rows_used=0, scientific_acceptance=False)
next_path = here / 'CONDITIONAL_WAVE_ACTIVE_CELL_NEXT_DESIGN.json'
next_path.write_bytes((json.dumps(design, indent=2) + '\n').encode())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_conditional_wave_probe_rejected',
        active_scientific_jobs=0, active_job_id=None, active_run_id=None,
        pending_job_id=None, pending_registered_job_id=None, pending_next_run_id=None,
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0,
        current_job_ended_UTC=exit_record['ended_utc'], updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='Conditional wave memory implemented and one bounded native direction checked; no feasible nonzero combination,0 updates.',
        next_action='Design-only active-cell descent directions from saved dual constraints; no new scientific job registered.',
        followup_automation_id=None, followup_automation_status='DELETED', followup_status='DELETED_NO_WAITING_WORK',
        followup_next_run_local=None, followup_scheduled_local=None, next_wake_local=None)
    d['global_memory_repair'].update(status='conditional_wave_probe_rejected', active_scientific_jobs=0,
        current_committed_training_updates=0, most_recent_probe_updates=0, inherited_checkpoint_updates=10,
        current_native_execution_counts=native['execution_counts'], counts_asof_UTC=exit_record['ended_utc'],
        conditional_wave_probe_passed=False, automatic_training_continuation=False, science_accepted=False,
        conditional_wave_closure=str((out / 'CLOSURE_RECEIPT.json').relative_to(root)),
        next_step_design=str(next_path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 162
        d['jobs'][-1].update(status='completed_conditional_wave_feasibility_rejected', native_exit_code=0,
            actual_execution_counts=native['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=str((out / 'CLOSURE_RECEIPT.json').relative_to(root)))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob162原生结束，新增更新0。112个条件波形单元中92个满足预登记支撑规则；'
        '条件波形下降方向使总风险下降0.4496%，但15个受保护单元变差且一库ADC风险增大。'
        '5个实测方向的联合线性问题只返回零步；独立终点与继承状态一致。'
        '保存的对偶约束定位到L16_04与L64_26，仅形成增补这两单元下降方向的下一步设计，'
        '未放宽阈值、未扩展job162预算、未登记继续训练。\n')
print(json.dumps(closure))
