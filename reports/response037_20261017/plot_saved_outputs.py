"""Render saved037 scalar diagnostics only; no model, CP, waveform or AD calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / 'receipts/response037_20261017/completion_01'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mpl_cache')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['TMP'] = os.environ['TEMP'] = str(ROOT / 'temp')
(ROOT / 'temp').mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
    'font.size': 10.5, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.titlesize': 18,
    'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9FAAB4', 'xtick.color': '#35424E', 'ytick.color': '#35424E',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

COLORS = {'H': '#2878A6', 'S': '#D99B35', 'F': '#7953A1'}
MARKERS = {'H': 'o', 'S': 's', 'F': 'D'}
MODES = ['H', 'S', 'F']
SMALL = ['minus_1_16', 'minus_1_64', 'baseline', 'plus_1_64', 'plus_1_16']
POINTS = SMALL + ['full1']
LABELS = ['−1/16', '−1/64', '0', '+1/64', '+1/16']
manifest_path = SOURCE / 'COLLECTION_MANIFEST.json'
manifest_bytes = manifest_path.read_bytes()
manifest = {row['path']: row for row in json.loads(manifest_bytes)['files']}
inputs, definitions = [], []


def read(name, jsonl=False):
    relative = 'run_01/' + name
    path = SOURCE / relative
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == manifest[relative]['sha256'], str(path)
    assert len(payload) == manifest[relative]['bytes'], str(path)
    inputs.append({'path': str(path), 'bytes': len(payload), 'sha256': digest, 'collection_manifest_match': True})
    return [json.loads(line) for line in payload.splitlines()] if jsonl else json.loads(payload)


def definition(path):
    payload = path.read_bytes()
    definitions.append({'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
        'role': 'Frozen definition metadata, separately authorized by ROOT; outside completion manifest.'})
    return json.loads(payload)


def save(fig, name):
    path = ROOT / name
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    with Image.open(path) as bitmap:
        pixels = list(bitmap.size)
    return {'path': str(path), 'bytes': path.stat().st_size, 'pixels': pixels,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


result = read('RESULT.json')
rows = read('metrics.jsonl', jsonl=True)
direction = read('PARAMETER_DIRECTION.json')
first_step = read('FIRST_STEP_REPLAY.json')
config = definition(PROJECT / 'configs/response037_20261017.json')
plan = definition(PROJECT / 'receipts/response037_20261017/preparation_01/PLAN.json')
by_bank_point = {(row['bank_index'], row['point']): row for row in rows}
assert len(rows) == 54
assert first_step['all_passed'] and result['first_step_replay_passed']
baseline_checks = []
for bank in range(9):
    baseline = by_bank_point[(bank, 'baseline')]
    assert baseline['mode0_elementwise_equality_checked']
    assert baseline['modes']['H']['scalars'] == baseline['modes']['S']['scalars'] == baseline['modes']['F']['scalars']
    baseline_checks.append({'bank_index': bank, 'saved_elementwise_check_passed': True,
        'saved_scalars_exactly_equal_across_H_S_F': True, 'total': baseline['modes']['H']['total']})
first = result['first_training_bank']
first_deltas = {point: {mode: first[point]['modes'][mode]['total'] - first['baseline']['modes'][mode]['total']
    for mode in MODES} for point in POINTS}
for point in POINTS:
    for mode in MODES:
        assert first[point]['modes'][mode]['scalars'] == by_bank_point[(0, point)]['modes'][mode]['scalars']
risk = result['reused_TRAIN_paired_deltas']
assert len(risk) == 5
assert all(risk[point][mode]['total']['banks'] == 8 and risk[point][mode]['total']['descriptive_3MCSE_direction'] == 'unresolved'
    for point in risk for mode in MODES)
FD = result['first_training_centered_FD']
for scale, plus, minus in [('1_64', 'plus_1_64', 'minus_1_64'), ('1_16', 'plus_1_16', 'minus_1_16')]:
    epsilon = FD[scale]['epsilon']
    from_dots = (direction['points'][plus]['g_dot_effective_delta'] - direction['points'][minus]['g_dot_effective_delta']) / (2 * epsilon)
    assert np.isclose(from_dots, FD[scale]['effective_parameter_linear_prediction'], rtol=1e-12, atol=1e-14)

figures = []
fig, axes = plt.subplots(1, 3, figsize=(15.1, 6.5), gridspec_kw={'width_ratios': [1.25, .72, 1]})
fig.subplots_adjust(top=.73, bottom=.34, left=.06, right=.98, wspace=.36)
fig.suptitle('037 首训练 bank：固定六点与 F centered FD', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .913, 'H：hard ADC + dynamic crop；S：hard0 + Δclamp 后 dynamic crop；F：同 S，但冻结初态 crop', ha='center', fontsize=11.5)
fig.text(.5, .85, '仅首个 036 训练 bank；无批组间 MCSE。H 为原硬 ADC 观测；S/F 为数学代理。', ha='center', fontsize=11, color='#A32F37')
x = np.arange(5)
for index, mode in enumerate(MODES):
    values = [first_deltas[point][mode] * 1000 for point in SMALL]
    axes[0].scatter(x + (index - 1) * .13, values, color=COLORS[mode], marker=MARKERS[mode], s=48, zorder=3)
axes[0].axhline(0, color='#73818C', ls='--', lw=1)
axes[0].set(title='五个近零点：离散，不连线', xticks=x, xticklabels=LABELS,
    xlabel='固定 λ 点（分类等距）', ylabel='Δtotal（单位 1e-3，相对 λ=0）')
for index, mode in enumerate(MODES):
    value = first_deltas['full1'][mode] * 1000
    axes[1].scatter(index, value, color=COLORS[mode], marker=MARKERS[mode], s=60, zorder=3)
    axes[1].annotate(f'{value:.3f}', (index, value), xytext=(0, -19), textcoords='offset points', ha='center', fontsize=10)
axes[1].axhline(0, color='#73818C', ls='--', lw=1)
axes[1].set(title='λ = 1：独立纵轴', xticks=range(3), xticklabels=MODES, ylim=(-20.5, 1),
    xlabel='三种模式均下降（仅首 bank）', ylabel='Δtotal（单位 1e-3）')
scales = ['1_64', '1_16']
axes[2].scatter(np.arange(2) - .10, [FD[scale]['modes']['F']['total'] * 1000 for scale in scales],
    color=COLORS['F'], marker=MARKERS['F'], s=55, zorder=3)
axes[2].scatter(np.arange(2) + .10, [FD[scale]['effective_parameter_linear_prediction'] * 1000 for scale in scales],
    color='#283D4B', marker='x', s=65, linewidths=1.7, zorder=3)
axes[2].set(title='F centered FD 与有效位移线性预测', xticks=range(2), xticklabels=['ε = 1/64', 'ε = 1/16'],
    xlabel='固定双边尺度（局部纵轴）', ylabel='total 方向导数（单位 1e-3 / λ）', ylim=(-33.94, -33.20), xlim=(-.48, 1.48))
for ax in axes:
    ax.grid(axis='y', color='#E2E8EC', lw=.7)
    ax.set_axisbelow(True)
handles = [Line2D([0], [0], color=COLORS[mode], marker=MARKERS[mode], ls='none', markersize=7,
    label=mode if mode != 'F' else 'F；右图为 centered FD') for mode in MODES]
handles.append(Line2D([0], [0], color='#283D4B', marker='x', ls='none', markersize=8,
    label='右图：g · [δeff(+ε) − δeff(−ε)] / (2ε)'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .225), ncol=4, frameon=False, fontsize=10)
fig.text(.5, .178, 'λ=0：H/S/F 保存的逐元素检查通过；三模式 total 精确相同 = 0.02596907309128277，图中 Δ 为 0。', ha='center', fontsize=10.5)
fig.text(.5, .132, 'F FD − 线性预测：ε=1/64 为 +0.000373602；ε=1/16 为 +0.0000741143。未做额外 AD 查询。', ha='center', fontsize=10.5)
fig.text(.5, .086, 'FD = [F(+ε) − F(−ε)]/(2ε)；g 来自既有首步 preclip total。g·δ 是方向内积，不是各 branch 的真实损失贡献。', ha='center', fontsize=10)
fig.text(.5, .035, '六点仅作方向与数值诊断，不是候选 LR；没有 λ 选择、续训或新 CP。首 bank 的下降不能代表风险总体或解释后期失稳。', ha='center', fontsize=10)
figures.append(save(fig, 'initial_direction_and_fd.png'))

fig, axes = plt.subplots(1, 2, figsize=(13.9, 6.4), gridspec_kw={'width_ratios': [1.45, 1]})
fig.subplots_adjust(top=.76, bottom=.31, left=.075, right=.98, wspace=.28)
fig.suptitle('037 既有风险 bank：15 个 total Δ 的 3 MCSE 方向均 unresolved', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .912, '8 个既有 033 TRAIN diagnostic banks；不是 fresh holdout；均值 ±3 MCSE，非 95% CI 或实验误差', ha='center', fontsize=11.5, color='#A32F37')
for index, mode in enumerate(MODES):
    means = [0 if point == 'baseline' else risk[point][mode]['total']['mean'] * 1000 for point in SMALL]
    errors = [0 if point == 'baseline' else 3 * risk[point][mode]['total']['MCSE'] * 1000 for point in SMALL]
    axes[0].errorbar(x + (index - 1) * .14, means, yerr=errors, fmt=MARKERS[mode], ls='none',
        color=COLORS[mode], capsize=3, ms=6, lw=1.4, label=mode)
    item = risk['full1'][mode]['total']
    axes[1].errorbar(index, item['mean'] * 1000, yerr=3 * item['MCSE'] * 1000,
        fmt=MARKERS[mode], ls='none', color=COLORS[mode], capsize=5, ms=7, lw=1.6)
    axes[1].annotate(f"mean {item['mean'] * 1000:.3f}", (index, item['mean'] * 1000),
        xytext=(7, 2), textcoords='offset points', color=COLORS[mode], fontsize=9.5)
axes[0].set(title='近零点：每 bank 相对自身 λ=0 配对', xticks=x, xticklabels=LABELS, ylim=(-2.35, 1.85),
    xlabel='固定 λ 点（分类等距，不连线）', ylabel='配对 Δtotal（单位 1e-3）')
axes[1].set(title='λ = 1：独立纵轴，三条区间均跨 0', xticks=range(3), xticklabels=MODES,
    xlabel='相同 8 个既有风险 banks', ylabel='配对 Δtotal（单位 1e-3）', ylim=(-5.5, 16), xlim=(-.5, 2.65))
for ax in axes:
    ax.axhline(0, color='#5D6E7A', ls='--', lw=1.1)
    ax.grid(axis='y', color='#E2E8EC', lw=.7)
    ax.set_axisbelow(True)
fig.legend(handles=[Line2D([0], [0], color=COLORS[mode], marker=MARKERS[mode], ls='none', label=mode, markersize=7) for mode in MODES],
    loc='lower center', bbox_to_anchor=(.5, .205), ncol=3, frameon=False)
fig.text(.5, .17, '先计算每 bank 的 total(λ) − total(0)，再跨 8 bank 求均值与 MCSE；λ=0 是自差 0，三模式基线精确相同。', ha='center', fontsize=10.5)
fig.text(.5, .12, '“未分辨”仅指 5 个非零点 ×3 mode 的 15 个 total Δ；不涵盖全部 nested / component 统计。', ha='center', fontsize=10.5)
fig.text(.5, .074, '例如 λ=−1/16 的 H−S total：+0.000509206 ±0.000104205 MCSE，按 3 MCSE 为 positive（已分辨）。', ha='center', fontsize=10.5)
fig.text(.5, .025, 'H 为原硬 ADC 观测；S/F 为数学代理。仅诊断、未选 λ、未续训、未保存 CP；response_match_not_established。', ha='center', fontsize=10)
figures.append(save(fig, 'reused_risk_bank_effects.png'))

switches = {}
for point in POINTS:
    row = by_bank_point[(0, point)]
    hard = list(row['modes']['H']['hypotheses'].values())
    cells = sum(h['hard_ADC_cell_changes']['cells'] for h in hard)
    cells_denominator = sum(h['hard_ADC_cell_changes']['cells_denominator'] for h in hard)
    peak = {}
    for mode in MODES:
        hypotheses = row['modes'][mode]['hypotheses'].values()
        counts = sum(h['peak_changes_from_initial']['count'] for h in hypotheses)
        denominator = sum(h['peak_changes_from_initial']['denominator'] for h in row['modes'][mode]['hypotheses'].values())
        peak[mode] = {'count': counts, 'denominator': denominator, 'fraction': counts / denominator}
    switches[point] = {'lambda': row['lambda_value'], 'H_hard_ADC_changed_cells': cells,
        'H_hard_ADC_cells_denominator': cells_denominator, 'H_hard_ADC_changed_fraction': cells / cells_denominator,
        'peak_index_changes': peak, 'signal_displacement': row['signal_displacement']}

receipt = {'schema': 'n01/response037-saved-output-plots/v1', 'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'rendered_pending_visual_QA', 'generator': str(Path(__file__)),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_root': str(SOURCE / 'run_01'), 'collection_manifest': {'path': str(manifest_path),
        'sha256': hashlib.sha256(manifest_bytes).hexdigest()}, 'inputs': inputs, 'definition_inputs': definitions,
    'all_completion_inputs_match_manifest_SHA256': True, 'figures': figures,
    'mode_definitions': config['mode_definitions'], 'fixed_lambda_points': config['lambdas'],
    'first_training_bank_total_deltas_full_precision': first_deltas,
    'first_training_centered_FD_full_precision': FD,
    'risk_total_deltas_full_precision': {point: {mode: risk[point][mode]['total'] for mode in MODES} for point in risk},
    'risk_centered_FD_full_precision': result['reused_TRAIN_risk_centered_FD'],
    'nested_mode_differences_full_precision': result['reused_TRAIN_nested_mode_differences'],
    'parameter_direction_full_precision': direction, 'first_step_replay': first_step,
    'baseline_equality_checks': baseline_checks, 'first_bank_observation_switches': switches,
    'methods': {'point_display': 'Fixed six categorical points; five near zero andlambda1 on separate axes. No interpolation, connecting curve or LR selection.',
        'first_bank': 'Exact first036 training bank only; total minus its own baseline. No bank MCSE applies to these single-bank scalar points.',
        'FD': 'Centered F total FD [F(+epsilon)-F(-epsilon)]/(2epsilon), compared to g dot [effective_delta(+epsilon)-effective_delta(-epsilon)]/(2epsilon). Gradient is existing combined036 preclip total; no new AD.',
        'g_dot_delta': 'A parameter direction inner product, including per-group entries; not a true per-branch loss decomposition.',
        'risk': 'Only15 nonzero point-by-mode total deltas are all unresolved. Paired differences are computed within each of8 reused033 TRAIN diagnostic banks before MCSE. Error bars are plus/minus3 MCSE, not95-percent CI or experimental uncertainty.',
        'baseline': 'Saved elementwise mode0 check flag is true on all9 banks; current plotting read also confirms exact saved scalar equality across H/S/F. No original arrays were loaded.',
        'nested_scope': 'Nested and component effects are separate. In particular minus1/16 H-minus-S total is positive by the saved3MCSE rule, so the result is not globally inconclusive.',
        'observation_switches': 'Receipt-only exactfirstbank counts. ADC cell denominator sums3 hypothesis by128 generated row by384 cells=147456; peak-index row occurrences denominator384 is not384 independent source parents.'},
    'evidence_limits': {key: result[key] for key in ['fresh_confirmation', 'conditional_identification', 'lambda_selected',
        'continued_training', 'checkpoints_written', 'scientific_status', 'interpretation']},
    'frozen_plan_interpretation': {
        'scope': 'Copied pre-registered hypothetical if-then interpretation rules, not observed findings. Overshoot requires small positive steps to descend while full1 harms; sampled-direction mismatch requires first-bank F descent and resolved reused-bank F total harm.',
        'current_observed_outcome': 'Neither hypothetical trigger is established in037: the first-bank full1 total decreases in H/S/F, while all15 reused-bank point-mode total changes are unresolved at the registered descriptive3MCSE criterion.',
        'copied_preregistered_rules': plan['interpretation']},
    'source_audit_execution_counts': result['execution_counts'],
    'new_NN_calls': 0, 'new_AD_queries': 0, 'new_optimizer_steps': 0, 'checkpoint_reads': 0,
    'raw_SOURCE_REAL_reads': 0, 'array_reads': 0, 'new_observation_calls': 0, 'new_scientific_budget': 0,
    'visual_QA': 'pending_view_image'}
(ROOT / 'PLOT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'figures': figures, 'completion_inputs_verified': len(inputs), 'definition_inputs_hashed': len(definitions)}, ensure_ascii=False, indent=2))
