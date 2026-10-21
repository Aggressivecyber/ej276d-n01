from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('E:/GAN/research/N01')
r = root / 'receipts/global_memory_repair_20261021/charge_spline_restoration01'
out = root / 'deliverables/global_memory_repair_20261021/charge_spline_restoration01'
t = r / 'terminal/run_01'
result = json.loads((t / 'RESULT.json').read_bytes())
training = json.loads((t / 'TRAINING_RESULT.json').read_bytes())
binding = json.loads((t / 'BINDINGS.json').read_bytes())
exit_receipt = json.loads((r / 'terminal/control_01/exit_receipt.json').read_bytes())
config = json.loads((root / 'configs/charge_spline_restoration01_20261022.json').read_bytes())
steps = [json.loads(x) for x in (t / 'TRAINING_STEPS.jsonl').read_text().splitlines()]
assert exit_receipt['exit_code'] == 0 and result['status'] == training['status'] == 'COMPLETE'
assert len(steps) == result['optimizer_updates'] == training['optimizer_updates'] == 128
assert training['cumulative_optimizer_updates'] == 152
assert [s['iteration'] for s in steps] == list(range(1, 129))
assert binding['source_FIT'] == 6465 and binding['REAL_FIT'] == 18948
assert binding['REPORT_training_rows'] == training['report_role_rows_used'] == 0
assert binding['resumed_parameters_exactly_equal'] and binding['fixed_memory_prefix_equal']
assert all(result['execution_counts'][k] <= v for k, v in config['operation_budget'].items() if k != 'derivation')
assert hashlib.sha256((root / 'configs/charge_spline_restoration01_20261022.json').read_bytes()).hexdigest() == result['config_sha256']
assert result['peak_GPU_reserved_bytes'] < config['resources']['GPU_GiB'] * 1024**3
assert exit_receipt['peak_rss_bytes'] < config['resources']['RSS_GiB'] * 1024**3
assert result['elapsed_seconds'] < config['resources']['internal_seconds']
chunks = (6465 + config['parent_chunk'] - 1) // config['parent_chunk']
trained = {base + i * 1000000 + bank * 100000 + j * 1000 + offset
           for base, count in [(config['training_seed'], 128), (config['predecessor_training_seed'], 24)]
           for i in range(count) for bank in range(2) for j in range(chunks) for offset in range(3)}
endpoint = {config['endpoint_seed'] + bank * 100000 + j * 1000 + offset
            for bank in range(2) for j in range(chunks) for offset in range(3)}
normalization = {config['bank_seed'] + bank * 100000 + j * 1000 + offset
                 for bank in range(2) for j in range(chunks) for offset in range(3)}
assert not endpoint.intersection(trained | normalization)
arms = training['endpoint']
original, inherited, start, final = [arms[k]['positive_metrics'] for k in ['initial', 'inherited', 'stage_start', 'B']]
failed_cells = [k for k in binding['cell_soft_limits'] if final[k] > 1.02 * inherited[k]]
old = json.loads((root / 'receipts/global_memory_repair_20261021/charge_spline01/TRAINING_RESULT.json').read_bytes())
metrics = ['Q_mass', 'PSD_CDF', 'Q_wave', 'block_wave', 'block_peak_codes']
utc = datetime.now(timezone.utc).isoformat()
closure = dict(status='COMPLETE_WAVE_SUPPRESSED_JOINT_FIT_FAILED', UTC=utc, job_id=config['job_id'], job_number=170,
    native_exit_code=0, ended_UTC=exit_receipt['ended_utc'], elapsed_seconds=result['elapsed_seconds'],
    stage_optimizer_updates=128, cumulative_candidate_updates=152,
    failed_endpoint_gates=[k for k, v in arms['B']['pilot_checks'].items() if not v],
    failed_supported_cells=failed_cells, supported_cells=len(binding['cell_soft_limits']),
    error_ratio_to_original={k: final[k] / original[k] for k in metrics},
    error_ratio_to_stage_start={k: final[k] / start[k] for k in metrics},
    revalidated_stage_start_relative_difference_from_previous_endpoint={k: start[k] / old['endpoint']['B']['positive_metrics'][k] - 1 for k in metrics},
    ADC_out_of_range_fraction={k: [f['pre_ADC_out_of_range'] / (6465 * 32) for f in v['flags']] for k, v in arms.items()},
    mean_branch_excess_reduction_percent=(100 * (1 - np.array(training['fixed_window_excess']) / np.array(training['fixed_window_excess_reference']))).tolist(),
    fullwave_curvature_relative=training['trend_relative'],
    last32_log_ADC_slope_per_step=float(np.polyfit(np.arange(32), np.log([np.mean(s['ADC_margin_risk']) for s in steps[-32:]]), 1)[0]),
    actual_execution_counts=result['execution_counts'], all_operation_budgets_respected=True,
    endpoint_random_seed_independent=True, seed_intersection_size=0, independent_randomness_is_not_independent_data=True,
    peak_RSS_bytes=exit_receipt['peak_rss_bytes'], peak_GPU_reserved_bytes=result['peak_GPU_reserved_bytes'],
    REPORT_evaluation_started=False, scientific_acceptance=False, weights_promoted=False,
    automation_deleted='n01', next_action='Audit component gradients and saved Adam step at the fixed152-update endpoint before choosing any new training. No automatic extension of job170.')
(r / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())

with np.load(r / 'ENDPOINT_ANALYSIS_TEMP.npz') as a:
    data = {k: a[k].copy() for k in a.files}
with np.load(r / 'INHERITED_MEAN_TEMP.npz') as a:
    assert np.array_equal(a['source_parent_rows'], data['source_parent_rows'])
    data.update({k: a[k].copy() for k in a.files})
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 12})
fig, axes = plt.subplots(1, 4, figsize=(19, 5.5), sharex=True, sharey=True, layout='constrained')
for ax, key, title in zip(axes, ['inherited', 'anchored', 'stage_start', 'B'],
                         ['继承均值 · 原条件支路', '固定条件起点 · 未训练', '样条候选 · 24 步', '样条候选 · 累计 152 步']):
    ax.scatter(data[key + '_mean_Q'], data[key + '_mean_PSD'], s=3, alpha=.35, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('均值支路固定窗 Q（V·ns）')
    ax.grid(alpha=.2)
axes[0].set_ylabel('均值支路固定窗 PSD')
fig.suptitle('实际均值支路的 Q–PSD 关系 · 全部 6,465 个 SOURCE FIT 父事件', fontsize=18)
fig.supxlabel('固定窗均值支路单列；不把零潜变量输出等同于完整随机响应的期望。', fontsize=11)
for suffix in ['png', 'svg']:
    fig.savefig(out / ('MEAN_BRANCH_COMPARISON.' + suffix), dpi=150)
plt.close(fig)
q = data['B_mean_Q']
order = np.argsort(q, kind='stable')
groups = np.array_split(order, 16)
moment_report = dict(parent_count=6465, draws_per_parent_per_bank=32, banks=2,
    conditioning='16 equal-count groups ordered by fixed canonical mean-branch Q; no physical-energy or particle labels.',
    expectation='Actual random draws; never substituted by zero latent input.', models={})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, layout='constrained')
for arm, label in [('initial', '修复前'), ('inherited', '继承模型'), ('anchored', '固定条件起点'), ('stage_start', '24 步'), ('B', '累计 152 步')]:
    cells = []
    for indices in groups:
        cell = dict(parents=len(indices), mean_branch_Q_median=float(np.median(q[indices])))
        for observable in ['Q', 'tail', 'PSD_raw']:
            centers = (data[f'{arm}_bank0_parent_{observable}_mean'] + data[f'{arm}_bank1_parent_{observable}_mean']) / 2
            # Mean of two independent 32-draw variance estimates, no pooling correction needed.
            widths = np.sqrt((data[f'{arm}_bank0_parent_{observable}_std']**2 + data[f'{arm}_bank1_parent_{observable}_std']**2) / 2)
            cell[observable] = dict(MC_mean_quantiles=np.quantile(centers[indices], [.1, .5, .9]).tolist(),
                                    MC_within_parent_std_quantiles=np.quantile(widths[indices], [.1, .5, .9]).tolist())
        cells.append(cell)
    moment_report['models'][arm] = cells
    axes[0].plot([c['mean_branch_Q_median'] for c in cells], [c['PSD_raw']['MC_mean_quantiles'][1] for c in cells], label=label)
    axes[1].plot([c['mean_branch_Q_median'] for c in cells], [c['PSD_raw']['MC_within_parent_std_quantiles'][1] for c in cells], label=label)
for ax in axes:
    ax.set_xlabel('父事件均值支路 Q（V·ns；未作物理能量标定）')
    ax.grid(alpha=.2)
    ax.legend(fontsize=10)
axes[0].set_ylabel('父事件 MC 平均 PSD 的组内中位数')
axes[1].set_ylabel('父事件随机 PSD 标准差的组内中位数')
fig.suptitle('真实随机采样的条件中心与宽度 · 两库各 32 次/父事件', fontsize=17)
fig.supxlabel('每组保留全部父事件；组内 10%/50%/90% 分位数与 Q、tail 统计见 JSON。实测无逐父事件重复，未构造配对真值。', fontsize=10)
for suffix in ['png', 'svg']:
    fig.savefig(out / ('MC_CONDITIONAL_MOMENTS.' + suffix), dpi=150)
plt.close(fig)
(out / 'MC_CONDITIONAL_MOMENTS.json').write_bytes((json.dumps(moment_report, indent=2, ensure_ascii=False) + '\n').encode())
(out / 'MEAN_BRANCH_AUDIT.json').write_bytes((json.dumps(dict(rows_per_panel=6465, row_drop=0, inherited_from_job168=True,
    identical_parent_order=True, no_new_model_forward=True, mean_branch_excess_reduction_percent=closure['mean_branch_excess_reduction_percent']), indent=2) + '\n').encode())
for p in out.glob('*.svg'):
    p.write_bytes(('\n'.join(line.rstrip() for line in p.read_text(encoding='utf8').splitlines()) + '\n').encode())

backup = r / 'before_closure'
backup.mkdir(exist_ok=False)
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    path = root / name
    shutil.copy2(path, backup / name)
    raw = path.read_bytes()
    doc = json.loads(raw)
    doc.update(status='N01_CHARGE_SPLINE_restoration_complete_joint_failed', scientific_status='response_match_not_established',
        active_scientific_jobs=0, active_run_id=None, training_pid=None, supervisor_pid=None, current_arm=None,
        updated_UTC=utc, last_verified_UTC=utc, execution_state_asof_UTC=utc,
        next_action=closure['next_action'], followup_automation_id=None, followup_automation_status='DELETED',
        followup_status='DELETED_NO_WAITING_WORK', followup_next_run_local=None,
        charge_spline_restoration_closure=str((r / 'CLOSURE_RECEIPT.json').relative_to(root)).replace('\\', '/'))
    if name == 'jobs.json':
        job = doc['jobs'][-1]
        assert job['job_id'] == config['job_id']
        job.update(status='complete_joint_FIT_failed', ended_UTC=exit_receipt['ended_utc'], native_exit_code=0,
            actual_execution_counts=result['execution_counts'], verified_stage_optimizer_updates=128,
            verified_cumulative_candidate_updates=152, counts_asof_UTC=exit_receipt['ended_utc'],
            all_operation_budgets_respected=True, scientific_acceptance=False,
            closure_receipt=doc['charge_spline_restoration_closure'])
    doc['global_memory_repair'].update(status='charge_spline_restoration_complete_joint_failed', active_scientific_jobs=0,
        current_remote_root=config['storage']['remote_root'], most_recent_training_updates=128,
        current_native_execution_counts=result['execution_counts'], counts_asof_UTC=exit_receipt['ended_utc'],
        latest_full_Q_figure=str((out / 'GLOBAL_MEMORY_PSD_LINEAR.png').relative_to(root)).replace('\\', '/'),
        latest_mean_trend_figures=str(out.relative_to(root)).replace('\\', '/'),
        charge_spline_restoration_closure=doc['charge_spline_restoration_closure'], science_accepted=False,
        automatic_training_continuation=False)
    newline = '\r\n' if b'\r\n' in raw else '\n'
    path.write_bytes((json.dumps(doc, indent=2, ensure_ascii=False) + '\n').replace('\n', newline).encode())
print(json.dumps({k:v for k,v in closure.items() if k not in ['actual_execution_counts','failed_supported_cells']}, ensure_ascii=False))
