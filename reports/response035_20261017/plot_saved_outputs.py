"""Plot saved035 aggregate JSON evidence; no model, checkpoints or event arrays."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'receipts/response035_20261017/completion_01'
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

STATES = ['M0R0', 'M0Rf', 'MfR0', 'MfRf', 'A0_only', 'Af_only']
CORNERS = STATES[:4]
COLORS = {'M0R0': '#2878A6', 'M0Rf': '#D99B35', 'MfR0': '#7953A1', 'MfRf': '#C94747',
    'A0_only': '#4B8792', 'Af_only': '#B46779'}
manifest_path = SOURCE / 'COLLECTION_MANIFEST.json'
manifest_bytes = manifest_path.read_bytes()
manifest = {row['path']: row for row in json.loads(manifest_bytes)['files']}
inputs = []


def read(name, jsonl=False):
    relative = 'run_01/' + name
    path = SOURCE / relative
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == manifest[relative]['sha256'], str(path)
    assert len(payload) == manifest[relative]['bytes'], str(path)
    inputs.append({'path': str(path), 'bytes': len(payload), 'sha256': digest, 'collection_manifest_match': True})
    return [json.loads(line) for line in payload.splitlines()] if jsonl else json.loads(payload)


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'MCSE': float(values.std(ddof=1) / np.sqrt(len(values))),
        'banks': len(values), 'bank_values': values.tolist()}


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
banks = [read(f'BANK_{index:02d}_CONTRASTS.json') for index in range(1, 9)]
by_state = {state: sorted([row for row in rows if row['state'] == state], key=lambda row: row['bank_index']) for state in STATES}
assert len(rows) == 48
for state, records in by_state.items():
    assert [row['bank_index'] for row in records] == list(range(1, 9))
    assert all(row['bank_role'] == 'previously_examined033_TRAIN_diagnostic_bank' and not row['fresh_confirmation'] for row in records)
    assert [row['data'] for row in records] == result['six_state_data_energy'][state]['data']['bank_values']
assert not result['A_only_full_objective_reported']
assert all('total' not in row and 'regular' not in row for state in ['A0_only', 'Af_only'] for row in by_state[state])
for contrast, values in result['paired_data_contrasts'].items():
    assert [bank['data'][contrast]['data'] for bank in banks] == values['data']['bank_values']
assert all(bank['paired_contrasts_computed_before_MCSE'] for bank in banks)
assert all(bank['A_cross_R_absmax'] == {'M0': 0.0, 'Mf': 0.0} for bank in banks)
for index in range(8):
    assert by_state['A0_only'][index]['pre_ADC_mean'] == by_state['M0R0'][index]['pre_ADC_mean']
    assert by_state['Af_only'][index]['pre_ADC_mean'] == by_state['MfR0'][index]['pre_ADC_mean']
    assert by_state['M0R0'][index]['same_forward_feature_RMS'] == by_state['M0Rf'][index]['same_forward_feature_RMS']
    assert by_state['MfR0'][index]['same_forward_feature_RMS'] == by_state['MfRf'][index]['same_forward_feature_RMS']

figures = []
fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.7), gridspec_kw={'width_ratios': [1, 1.35]})
fig.subplots_adjust(top=.78, bottom=.34, left=.09, right=.98, wspace=.53)
fig.suptitle('035 参数块交换：data energy 与配对差值', y=.985, fontweight='bold', color='#223746')
fig.text(.5, .915, 'M：均值参数块；R：随机头参数块；0 / f：初态 / 末态。A-only 指响应只含确定性 A。', ha='center', fontsize=12)
fig.text(.5, .855, '8 个既有 033 TRAIN diagnostic banks；不是 fresh holdout；误差条为 ±1 MCSE，非 95% CI 或实验误差', ha='center', fontsize=11, color='#A32F37')
ax = axes[0]
for index, state in enumerate(STATES):
    item = result['six_state_data_energy'][state]['data']
    ax.errorbar(item['mean'], index, xerr=item['MCSE'], fmt='o', color=COLORS[state], capsize=4, ms=7, lw=1.6, zorder=3)
    ax.text(.77, index, f"{item['mean']:.4f} ± {item['MCSE']:.4f}", va='center', ha='left', fontsize=10)
ax.set(yticks=range(6), yticklabels=STATES, ylim=(5.55, -.55), xlim=(0, 1.06),
    title='六状态：三种 hypothesis 的平均 data energy', xlabel='Data energy（越低越好）')
ax.set_xticks([0, .2, .4, .6])
ax.grid(axis='x', color='#E2E8EC', lw=.7)
ax.set_axisbelow(True)

contrast_specs = [
    ('M0Rf_minus_M0R0', 'R @ M0\nM0Rf − M0R0'),
    ('MfR0_minus_M0R0', 'M @ R0\nMfR0 − M0R0'),
    ('MfRf_minus_M0R0', '共同交换\nMfRf − M0R0'),
    ('M_effect_at_Rf', 'M @ Rf\nMfRf − M0Rf'),
    ('R_effect_at_Mf', 'R @ Mf\nMfRf − MfR0'),
    ('interaction', 'Interaction\nMfRf − MfR0 − M0Rf + M0R0'),
    ('Af_only_minus_A0_only', '仅确定性 A\nAf_only − A0_only')]
ax = axes[1]
for index, (key, label) in enumerate(contrast_specs):
    item = result['paired_data_contrasts'][key]['data']
    direction = item['descriptive_3MCSE_direction']
    color = {'positive': '#C94747', 'negative': '#2878A6', 'unresolved': '#7B858D'}[direction]
    ax.errorbar(item['mean'], index, xerr=item['MCSE'], fmt='o', color=color, capsize=4, ms=7, lw=1.6)
    suffix = ' *' if direction == 'unresolved' else ''
    ax.text(.67, index, f"{item['mean']:+.4f} ± {item['MCSE']:.4f}{suffix}", va='center', fontsize=10)
ax.axvline(0, color='#7B858D', ls='--', lw=1)
ax.set(yticks=range(7), yticklabels=[item[1] for item in contrast_specs], ylim=(6.55, -.55),
    xlim=(-.65, 1.13), title='同 bank 先相减，再计算均值与 MCSE', xlabel='配对 Δ data energy（负值减小，正值增大）')
ax.set_xticks([-.6, -.3, 0, .3, .6])
ax.tick_params(axis='y', labelsize=9.3)
ax.grid(axis='x', color='#E2E8EC', lw=.7)
fig.text(.5, .245, 'A0 → Af：wave 0.02919 → 0.17585，carrier 0.29167 → 0.02542；两者方向冲突。以下是配对加权贡献：', ha='center', fontsize=11)
weighted_A_contrasts = {}
for position, metric, weight, label, color in [(.18, 'wave', 1, 'wave × 1', '#C94747'),
        (.41, 'integral', 2, 'integral × 2', '#C94747'), (.65, 'carrier', .25, 'carrier × 0.25', '#2878A6'),
        (.87, 'data', 1, '合计 Δ data', '#C94747')]:
    item = result['paired_data_contrasts']['Af_only_minus_A0_only'][metric]
    weighted_A_contrasts[metric] = {'weight': weight, 'mean': weight * item['mean'], 'MCSE': weight * item['MCSE'],
        'bank_values': [weight * value for value in item['bank_values']]}
    fig.text(position, .194, f"{label}\n{weight * item['mean']:+.4f} ± {weight * item['MCSE']:.4f}", ha='center', va='center', color=color, fontsize=11)
fig.text(.5, .12, 'data = wave + 2 × integral + 0.25 × carrier；先在 fast / slow / common_level 内计算，再取三者均值。', ha='center', fontsize=10.5)
fig.text(.5, .078, '* 按既定 3 MCSE 口径为 unresolved；Interaction 是差中差，负值不代表最终 fit 改善。A-only 无 full total。', ha='center', fontsize=10.5)
fig.text(.5, .035, '交换效应包含特征、幅度与非线性观测变化；不能据此归因训练历史。response_match_not_established。', ha='center', fontsize=10.5)
figures.append(save(fig, 'data_energy_and_paired_effects.png'))

geometry = {}
heads = {}
features = {}
for state in CORNERS:
    records = by_state[state]
    geometry[state] = {}
    for scope in ['pre_ADC_mean', 'pre_ADC_signal']:
        counts = [row[scope]['global_peak_outside_search']['count'] for row in records]
        denominators = [row[scope]['global_peak_outside_search']['denominator'] for row in records]
        assert denominators == [128] * 8
        geometry[state][scope] = {'outside_peak': summary([count / denominator for count, denominator in zip(counts, denominators)]),
            'count_sum': sum(counts), 'denominator_sum': sum(denominators),
            'counts_by_bank': counts, 'denominators_by_bank': denominators,
            'negative_area_Vns': summary([row[scope]['negative_area_carrier_Vns']['mean'] for row in records])}
    heads[state] = {key: summary([row['head_scales'][key]['RMS'] for row in records])
        for key in ['B_voltage_RMS', 'B_RMS_over_A_RMS', 'charge_v_norm', 'local_envelope_V']}
    heads[state]['charge_log_variance_row_mean'] = summary([row['head_scales']['charge_log_variance']['mean'] for row in records])
    heads[state]['charge_log_variance_row_RMS'] = summary([row['head_scales']['charge_log_variance']['RMS'] for row in records])
    heads[state]['charge_v_norm_row_mean'] = summary([row['head_scales']['charge_v_norm']['mean'] for row in records])
    features[state] = {key: summary([np.mean([forward[key] for forward in row['same_forward_feature_RMS']]) for row in records])
        for key in ['decoded_RMS', 'pooled_RMS']}

fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
fig.subplots_adjust(top=.79, bottom=.23, left=.075, right=.98, hspace=.58, wspace=.25)
fig.suptitle('035 保存几何与尺度：纯 A / signal、B 与共享电荷', y=.985, fontweight='bold', color='#223746')
fig.text(.5, .92, '纯 pre-ADC carrier384，未加电子噪声；几何分母每状态 = 8 bank × 128 生成行 = 1024 次', ha='center', fontsize=12)
fig.text(.5, .86, '跨 bank 父事件可重复，1024 次不等于 1024 个独立父；以下 MCSE 仅描述这 8 个既有诊断 bank', ha='center', fontsize=11, color='#A32F37')
x = np.arange(4)
width = .32
for ax in axes.flat:
    ax.grid(axis='y', color='#E2E8EC', lw=.7)
    ax.set_axisbelow(True)
for scope, offset, color, label in [('pre_ADC_mean', -width / 2, '#2878A6', '纯确定性 A'), ('pre_ADC_signal', width / 2, '#C94747', '纯完整 signal')]:
    values = np.array([geometry[state][scope]['outside_peak']['mean'] for state in CORNERS]) * 100
    errors = np.array([geometry[state][scope]['outside_peak']['MCSE'] for state in CORNERS]) * 100
    bars = axes[0, 0].bar(x + offset, values, width * .9, yerr=errors, capsize=3, color=color, label=label)
    if scope == 'pre_ADC_signal':
        axes[0, 0].bar_label(bars, labels=[f"{geometry[state][scope]['count_sum']}/1024" for state in CORNERS], padding=4, fontsize=9.5)
    values = [geometry[state][scope]['negative_area_Vns']['mean'] for state in CORNERS]
    errors = [geometry[state][scope]['negative_area_Vns']['MCSE'] for state in CORNERS]
    bars = axes[0, 1].bar(x + offset, values, width * .9, yerr=errors, capsize=3, color=color, label=label)
    axes[0, 1].bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=9.5)
axes[0, 0].set(title='全局峰位于搜索窗外 [global_peak_outside_search]', ylabel='生成行比例（%）',
    xticks=x, xticklabels=CORNERS, ylim=(0, 14))
axes[0, 1].set(title='负面积：每 bank 的行均值，再对 8 bank 均值', ylabel='负面积（V·ns，软件样点 1 ns）',
    xticks=x, xticklabels=CORNERS, ylim=(0, .57))
axes[0, 0].legend(frameon=False, loc='upper left')
for index, state in enumerate(CORNERS):
    item = heads[state]['B_voltage_RMS']
    bar = axes[1, 0].bar(index, item['mean'] * 1000, .55, yerr=item['MCSE'] * 1000, capsize=3, color=COLORS[state])
    axes[1, 0].bar_label(bar, labels=[f"{item['mean'] * 1000:.3f}"], padding=4, fontsize=10)
axes[1, 0].set(title='B 的物理 RMS；下行标签为 B/A 行比值 RMS', ylabel='银行内 B RMS 的均值（mV）',
    xticks=x, xticklabels=[f"{state}\nB/A = {heads[state]['B_RMS_over_A_RMS']['mean']:.3f}" for state in CORNERS], ylim=(0, 10.3))
for index, state in enumerate(CORNERS):
    item = heads[state]['charge_log_variance_row_mean']
    bars = axes[1, 1].bar(index, item['mean'], .55, yerr=item['MCSE'], capsize=3, color=COLORS[state])
    axes[1, 1].bar_label(bars, labels=[f"{item['mean']:.5f}"], padding=4, fontsize=10)
axes[1, 1].set(title='共享电荷：log α 条件方差的行均值', ylabel='每 bank mean ||v||²，再取 8 bank 均值',
    xticks=x, xticklabels=[f"{state}\nRMS ||v|| = {heads[state]['charge_v_norm']['mean']:.3f}" for state in CORNERS], ylim=(0, .53))
fig.text(.5, .142, 'B：每 bank 跨行/基/时刻 RMS，再取 8 bank 均值；B/A：先逐行求 RMS(B)/RMS(A)，再取行 RMS 与 bank 均值。', ha='center', fontsize=10)
fig.text(.5, .095, '误差条均为 ±1 MCSE，非实验误差；纯 A 的四角越窗计数均为 0/1024。几何只描述保存的 pre-ADC 输出。', ha='center', fontsize=10.5)
fig.text(.5, .035, '本图不建立条件均值真值、不解释长期失稳因果，也不替代响应匹配验收。保存结论：response_match_not_established。', ha='center', fontsize=10.5)
figures.append(save(fig, 'preADC_geometry_and_scales.png'))

receipt = {'schema': 'n01/response035-saved-output-plots/v1', 'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'rendered_pending_visual_QA', 'generator': str(Path(__file__)),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_root': str(SOURCE / 'run_01'), 'collection_manifest': {'path': str(manifest_path),
        'sha256': hashlib.sha256(manifest_bytes).hexdigest()}, 'inputs': inputs,
    'all_read_inputs_match_collection_SHA256': True, 'figures': figures,
    'six_state_data_energy_full_precision': result['six_state_data_energy'],
    'paired_data_contrasts_full_precision': result['paired_data_contrasts'],
    'weighted_Af_minus_A0_data_contributions': weighted_A_contrasts,
    'geometry_full_precision': geometry, 'head_scales_full_precision': heads,
    'same_forward_features_full_precision': features,
    'aggregation': {'energy': 'Unchanged RESULT mean and MCSE, data averaged over three hypotheses; data=wave+2*integral+0.25*carrier.',
        'paired': 'Per-bank contrasts saved in BANK_*.json; differences are taken before MCSE. Every bank value matched RESULT.',
        'geometry': '8 existing banks, 128 generated rows per state per bank. Flag count denominator is generated row occurrences, not distinct parents. Rate MCSE=sample_std(bank_rate,ddof=1)/sqrt(8).',
        'negative_area': 'Per-bank arithmetic mean over 128 row negative areas, then equal mean over 8 banks. MCSE across the 8 bank means.',
        'B_voltage_RMS': 'head_scales.B_voltage_RMS.RMS within each bank (across rows/bases/time), then mean and MCSE across 8 banks; plot converts volts to mV.',
        'B_over_A': 'head_scales.B_RMS_over_A_RMS.RMS within each bank: RMS across per-row RMS(B)/RMS(A) ratios; then equal bank mean. This is not ratio of the two aggregate RMS values.',
        'charge': 'Bar height is head_scales.charge_log_variance.mean per bank, then mean and MCSE over 8 banks; this is mean ||v||^2, the row-conditional log(alpha) variance. Tick labels use head_scales.charge_v_norm.RMS per bank, then bank mean. Logvariance RMS is separately saved and is not used as the bar height.',
        'features': 'Receipt-only: for each bank, arithmetic mean of its two saved same_forward_feature_RMS entries; then mean and MCSE across 8 banks. No global feature explosion claim.',
        'weighted_Af_minus_A0': 'Unchanged paired data component differences with original weights 1,2,0.25. Total data MCSE is taken from the saved paired total, not from an independent-error sum of component MCSEs.',
        'uncertainty': 'All error bars are plus/minus one MCSE, not 95-percent intervals or experimental uncertainty. Banks were examined in response033; not a fresh holdout.',
        'A_only': 'Data energy only; no full total or regular is imputed.'},
    'evidence_limits': {key: result[key] for key in ['primary', 'A_only_full_objective_reported',
        'previously_examined_TRAIN_banks', 'fresh_confirmation', 'conditional_identification', 'candidate_selected',
        'training_history_causal_claim', 'scientific_status', 'interpretation']},
    'raw_SOURCE_REAL_reads': 0, 'saved_NPZ_reads': 0, 'checkpoint_reads': 0,
    'new_NN_calls': 0, 'new_AD_queries': 0, 'new_optimizer_steps': 0,
    'visual_QA': 'pending_view_image'}
(ROOT / 'PLOT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'figures': figures, 'input_files_verified': len(inputs)}, ensure_ascii=False, indent=2))
