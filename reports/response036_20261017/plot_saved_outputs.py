"""Plot saved034/036 JSON and three bounded example NPZs; no model/raw inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import os

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mpl_cache')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['TMP'] = os.environ['TEMP'] = str(ROOT / 'temp')
(ROOT / 'temp').mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
    'font.size': 10.5, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.titlesize': 18,
    'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9FAAB4', 'xtick.color': '#35424E', 'ytick.color': '#35424E',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

RUNS = ['034', '036']
HYPOTHESES = ['fast', 'slow', 'common_level']
COLORS = {'A': '#2878A6', 'G0': '#D99B35', '034': '#7953A1', '036': '#C94747', 'actual036': '#5B6872'}
SOURCES = {run: PROJECT / f'receipts/response{run}_20261017/completion_01' for run in RUNS}
manifests, manifest_records, inputs = {}, [], []
for run in RUNS:
    path = SOURCES[run] / 'COLLECTION_MANIFEST.json'
    payload = path.read_bytes()
    document = json.loads(payload)
    entries = document['files'] if run == '034' else document
    manifests[run] = {entry['path']: entry for entry in entries}
    manifest_records.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})


def checked_bytes(run, name):
    relative = 'run_01/' + name
    path = SOURCES[run] / relative
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == manifests[run][relative]['sha256'], str(path)
    assert len(payload) == manifests[run][relative]['bytes'], str(path)
    inputs.append({'path': str(path), 'bytes': len(payload), 'sha256': digest, 'collection_manifest_match': True})
    return payload


def read(run, name, jsonl=False):
    payload = checked_bytes(run, name)
    return [json.loads(line) for line in payload.splitlines()] if jsonl else json.loads(payload)


def save(fig, name):
    path = ROOT / name
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    with Image.open(path) as bitmap:
        pixels = list(bitmap.size)
    return {'path': str(path), 'bytes': path.stat().st_size, 'pixels': pixels,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def trace(ax, steps, values, color, label, linestyle='-'):
    ax.plot(steps, values, color=color, alpha=.14, lw=.55, ls=linestyle)
    ax.plot(steps[49:], np.convolve(values, np.ones(50) / 50, mode='valid'), color=color, lw=1.8, ls=linestyle, label=label)


starts = {run: read(run, 'START.json')['config'] for run in RUNS}
results = {run: read(run, 'RESULT.json') for run in RUNS}
records = {(run, split, endpoint, hypothesis): read(run, f'{split}_{endpoint}_{hypothesis}.json')
    for run in RUNS for split in ['TRAIN', 'DEV'] for endpoint in ['A', 'final'] for hypothesis in HYPOTHESES}
training = {run: read(run, 'metrics.jsonl', jsonl=True) for run in RUNS}
assert starts['034']['evaluation'] == starts['036']['evaluation']
assert starts['034']['seed'] == starts['036']['seed']
assert {key: value for key, value in starts['036']['training'].items() if key != 'carrier_weight'} == starts['034']['training']
assert starts['036']['training']['carrier_weight'] == 0
for split in ['TRAIN', 'DEV']:
    for h in HYPOTHESES:
        assert records[('034', split, 'A', h)]['values'] == records[('036', split, 'A', h)]['values']
        for metric in ['wave', 'QPSD']:
            assert results['034']['comparisons'][split][h]['calibrated_components'][metric]['threshold'] == results['036']['comparisons'][split][h]['calibrated_components'][metric]['threshold']

figures = []
native_summary = {}
fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2))
fig.subplots_adjust(top=.80, bottom=.25, left=.075, right=.98, hspace=.43, wspace=.24)
fig.suptitle('034 / 036 原 native 距离：相对 A 与原冻结门', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .924, '036 仅移除 carrier 训练辅助；034 与 036 的评价定义、投影、A 和容差 T 相同', ha='center', fontsize=12)
fig.text(.5, .87, '两次 pilot_relative_progress 均为 false；response_match_not_established', ha='center', fontsize=12, color='#A32F37')
x = np.arange(3)
width = .23
for row, split in enumerate(['TRAIN', 'DEV']):
    for col, metric in enumerate(['wave', 'QPSD']):
        ax = axes[row, col]
        baseline = np.array([records[('036', split, 'A', h)]['values'][metric] for h in HYPOTHESES])
        thresholds = np.array([results['036']['comparisons'][split][h]['calibrated_components'][metric]['threshold'] for h in HYPOTHESES])
        native_summary.setdefault(split, {})[metric] = {'A': baseline.tolist(), 'T': thresholds.tolist(),
            'absolute_T_over_A': (thresholds / baseline).tolist(), 'noninferior_limit_over_A': (1 + thresholds / baseline).tolist()}
        maxima = []
        for index, series in enumerate(['A', '034', '036']):
            values = baseline if series == 'A' else np.array([records[(series, split, 'final', h)]['values'][metric] for h in HYPOTHESES])
            ratios = values / baseline
            native_summary[split][metric][series + '_native'] = values.tolist()
            native_summary[split][metric][series + '_over_A'] = ratios.tolist()
            maxima.extend(ratios)
            positions = x + (index - 1) * width
            bars = ax.bar(positions, ratios, width * .88, color=COLORS[series], zorder=3)
            ax.bar_label(bars, labels=[f'{value:.2f}×' for value in ratios], padding=4,
                rotation=90 if metric == 'wave' else 0, fontsize=9.5)
        for pos, absolute, noninferior in zip(x, thresholds / baseline, 1 + thresholds / baseline):
            ax.plot([pos - .43, pos + .43], [absolute, absolute], ls='--', lw=1.2, color='#3D4B56', zorder=4)
            ax.plot([pos - .43, pos + .43], [noninferior, noninferior], ls=':', lw=1.6, color='#87939D', zorder=4)
        if split == 'TRAIN' and metric == 'wave':
            ax.axhline(.98, ls='-.', color='#2E8F8A', lw=1.1, zorder=4)
        if metric == 'QPSD':
            ax.set_yscale('log')
            ax.set_ylim(.3, max(maxima) * 2.2)
        else:
            ax.set_ylim(0, 7)
        ax.set(xticks=x, xticklabels=HYPOTHESES, ylabel=f'{split} native / A（越低越好）',
            title=('Native waveform [wave]' if metric == 'wave' else 'Native Q/PSD [QPSD]；log 轴'))
        ax.grid(axis='y', color='#E2E8EC', lw=.7)
        ax.set_axisbelow(True)
handles = [Patch(color=COLORS[key], label=label) for key, label in [('A', 'A = 1'), ('034', '034 final / A'), ('036', '036 final / A')]]
handles += [Line2D([0], [0], color='#3D4B56', ls='--', label='绝对容差 T / A（另报 final/T）'),
    Line2D([0], [0], color='#87939D', ls=':', lw=1.6, label='原非劣界 (A + T) / A'),
    Line2D([0], [0], color='#2E8F8A', ls='-.', label='原 TRAIN wave 进步门：final/A ≤ 0.98')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .115), ncol=3, frameon=False, fontsize=10)
fig.text(.5, .074, '原 progress 还要求所有定义且正的校准分量非劣。图仅展示 wave / QPSD；精确 native、final/T 与全部原门见回执。', ha='center', fontsize=10)
fig.text(.5, .035, 'TRAIN：6593 父 × 4 draw，REAL 21047；DEV：512 父 × 4 draw，REAL 599，已暴露。每次评价只有 1 个 MC bank。', ha='center', fontsize=10)
fig.text(.5, .002, '四 draw 不是四个独立父；训练 energy ≠ native W1 类评价。图中没有独立重复误差条。', ha='center', fontsize=10)
figures.append(save(fig, 'native_034_036_relative_A.png'))

arrays = {}
training_summary = {}
for run in RUNS:
    rows = training[run]
    arrays[run] = {key: np.array([record[key] for record in rows]) for key in ['step', 'data', 'regular', 'saturation', 'objective', 'preclip_gradient_norm']}
    if run == '036':
        for component in ['wave', 'integral']:
            arrays[run][component] = np.array([np.mean([record['hypothesis_energy_parts_report_only'][h][component] for h in HYPOTHESES]) for record in rows])
        for key in ['carrier_energy_report_only', 'data_with_carrier_report_only', 'objective_with_carrier_report_only']:
            arrays[run][key] = np.array([record[key] for record in rows])
        arrays[run]['2integral'] = 2 * arrays[run]['integral']
        arrays[run]['0.25carrier_report_only'] = .25 * arrays[run]['carrier_energy_report_only']
        assert np.allclose(arrays[run]['data'], arrays[run]['wave'] + arrays[run]['2integral'], rtol=1e-12, atol=1e-14)
        assert np.allclose(arrays[run]['data_with_carrier_report_only'], arrays[run]['data'] + arrays[run]['0.25carrier_report_only'], rtol=1e-12, atol=1e-14)
    clips = arrays[run]['preclip_gradient_norm'] > 5
    training_summary[run] = {'steps': len(rows), 'seed': starts[run]['seed'], 'clipped_steps': int(clips.sum()),
        'clipped_fraction': float(clips.mean()), 'last100_clipped_fraction': float(clips[-100:].mean()),
        'metrics': {key: {'first': float(values[0]), 'last': float(values[-1]), 'last100_mean': float(values[-100:].mean()),
            'minimum': float(values.min()), 'maximum': float(values.max())} for key, values in arrays[run].items() if key != 'step'}}

fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.3))
fig.subplots_adjust(top=.80, bottom=.20, left=.065, right=.98, hspace=.43, wspace=.28)
fig.suptitle('036 训练 energy 分量与 034 可比总量：不替代 native 验收', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .924, '036 实际 data = wave + 2 × integral；carrier 训练权重 = 0，仅按原权重 0.25 报告', ha='center', fontsize=12)
fig.text(.5, .868, '细线：逐步值；粗线：50 步滑动均值。034 未保存 wave / integral 拆分，未反造。', ha='center', fontsize=11)
for ax, key, label in zip(axes[0], ['wave', '2integral', '0.25carrier_report_only'],
        ['036：wave（入训练）', '036：2 × integral（入训练）', '036：0.25 × carrier（仅报告）']):
    trace(ax, arrays['036']['step'], arrays['036'][key], COLORS['036'], '036')
    ax.set(title=label, ylabel='Energy 分量')
    ax.axhline(0, color='#9FAAB4', ls=':', lw=.7)
for run in RUNS:
    trace(axes[1, 0], arrays[run]['step'], arrays[run]['regular'], COLORS[run], run)
    trace(axes[1, 1], arrays[run]['step'], arrays[run]['preclip_gradient_norm'], COLORS[run], run)
trace(axes[1, 2], arrays['034']['step'], arrays['034']['objective'], COLORS['034'], '034 实际（含 carrier）')
trace(axes[1, 2], arrays['036']['step'], arrays['036']['objective_with_carrier_report_only'], COLORS['036'], '036 含 carrier 回报')
trace(axes[1, 2], arrays['036']['step'], arrays['036']['objective'], COLORS['actual036'], '036 实际（不含 carrier）', '--')
axes[1, 0].set(title='原始 regular R（目标权重 0.01）', ylabel='R（未加权）')
axes[1, 1].set(title='裁剪前梯度范数（log 轴）', ylabel='L2 norm')
axes[1, 1].set_yscale('log')
axes[1, 1].axhline(5, color='#475664', ls='--', lw=1)
axes[1, 1].text(.98, .97, f"clip > 5：034 {training_summary['034']['clipped_fraction']:.1%}\n036 {training_summary['036']['clipped_fraction']:.1%}",
    transform=axes[1, 1].transAxes, ha='right', va='top', fontsize=9,
    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 3})
axes[1, 2].set(title='共同含 carrier 口径及 036 实际目标', ylabel='Objective')
axes[1, 2].legend(frameon=False, fontsize=8.7, loc='upper right')
for ax in axes.flat:
    ax.set(xlabel='训练步', xlim=(0, 2000))
    ax.grid(axis='y', color='#E2E8EC', lw=.7)
fig.legend(handles=[Line2D([0], [0], color=COLORS['034'], lw=1.8, label='034'),
    Line2D([0], [0], color=COLORS['036'], lw=1.8, label='036'),
    Line2D([0], [0], color=COLORS['actual036'], lw=1.8, ls='--', label='036 实际目标（无 carrier）')],
    loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
fig.text(.5, .045, '共同回报口径：wave + 2 × integral + 0.25 × carrier + 0.01 × (R + saturation)；036 该量为 report-only。', ha='center', fontsize=10)
fig.text(.5, .01, '损失在该步参数更新前计算；记录在更新后写入。训练 energy ≠ native W1；各一个 seed，无新增训练、梯度或观测。', ha='center', fontsize=10)
figures.append(save(fig, 'training_components_034_036.png'))

# The first three saved positions are fixed before waveform values are read.
example_positions = [0, 1, 2]
examples = {}
for endpoint in ['A', 'G0', 'final']:
    payload = checked_bytes('036', f'TRAIN_{endpoint}_fast_examples.npz')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        waveforms = archive['waveform256']
        assert waveforms.shape == (32, 256)
        examples[endpoint] = {'waveform_V': waveforms[example_positions].copy(),
            'source_parent_rows': archive['source_parent_rows'][example_positions].tolist(),
            'example_row_indices': archive['example_row_indices'][example_positions].tolist()}
assert examples['A']['source_parent_rows'] == examples['G0']['source_parent_rows'] == examples['final']['source_parent_rows']
assert examples['A']['example_row_indices'] == examples['G0']['example_row_indices'] == examples['final']['example_row_indices']
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
fig.subplots_adjust(top=.73, bottom=.25, left=.065, right=.98, wspace=.25)
fig.suptitle('036 TRAIN / fast：固定保存顺序的三例 A / G0 / final', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .89, '取已保存 32 行示例的前 3 行；按保存顺序固定，未按数值筛选；未绘 REAL 波形', ha='center', fontsize=12)
for position, ax in enumerate(axes):
    for endpoint, color, label in [('A', COLORS['A'], 'A'), ('G0', COLORS['G0'], 'G0'), ('final', COLORS['036'], '036 final')]:
        ax.plot(np.arange(256), examples[endpoint]['waveform_V'][position] * 1000, color=color, lw=1.5, label=label)
    ax.set(title=f"保存位置 {example_positions[position]}；父行 {examples['A']['source_parent_rows'][position]}\n生成行 {examples['A']['example_row_indices'][position]}",
        xlabel='保存的 256 样点窗口索引', ylabel='保存电压（mV）', xlim=(0, 255), xticks=[0, 64, 128, 192, 255])
    ax.grid(color='#E2E8EC', lw=.7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .10), ncol=3, frameon=False)
fig.text(.5, .065, '直接绘制已保存 waveform256 数组；未重新观测、平移、归一化或减基线。不同父事件纵轴可不同。', ha='center', fontsize=10)
fig.text(.5, .02, '三个有限示例不代表全父总体或恶化频率；全评价失败结论来自原 JSON。MC draw 不是独立父事件。', ha='center', fontsize=10)
figures.append(save(fig, 'bounded_saved_waveform_examples.png'))

endpoint_numbers = {}
for (run, split, endpoint, h), record in records.items():
    endpoint_numbers.setdefault(run, {}).setdefault(split, {}).setdefault(h, {})[endpoint] = {
        'values': record['values'], 'source_parents': record['source_parents'], 'draws_per_parent': record['draws_per_parent'],
        'fake_rows': record['fake_rows'], 'real_rows': record['real_rows'], 'MC_banks': record['MC_banks'],
        'all_rows_retained': record['all_rows_retained'], 'previously_exposed_DEV': record['previously_exposed_DEV']}
receipt = {'schema': 'n01/response036-saved-output-plots/v1', 'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'rendered_pending_visual_QA', 'generator': str(Path(__file__)),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_roots': {run: str(path / 'run_01') for run, path in SOURCES.items()},
    'collection_manifests': manifest_records, 'inputs': inputs, 'all_read_inputs_match_collection_SHA256': True,
    'figures': figures, 'native_relative_A_full_precision': native_summary, 'endpoint_full_precision': endpoint_numbers,
    'calibrated_comparisons': {run: results[run]['comparisons'] for run in RUNS}, 'training_summary': training_summary,
    'bound_example_selection': {'split': 'TRAIN', 'hypothesis': 'fast', 'fixed_saved_positions': example_positions,
        'selection_before_waveform_values_read': True, 'selection': 'First three positions in saved 32-row arrays; no ranking or random best selection.',
        'source_parent_rows': examples['A']['source_parent_rows'], 'generated_row_indices': examples['A']['example_row_indices'],
        'NPZ_archives_read': 3, 'waveform_rows_decoded_per_archive': 32, 'waveforms_plotted_per_archive': 3,
        'members_decoded': ['waveform256', 'source_parent_rows', 'example_row_indices'],
        'no_REAL_waveform': True, 'not_population_representative': True,
        'transforms': 'Voltage unit V to mV only; original 256-sample indices, no baseline subtraction or alignment.'},
    'same_definition_checks': {'evaluation_config_equal': True, 'training_seed_equal': True,
        'training_config_equal_except_carrier_weight': True, 'A_values_exactly_equal': True, 'native_thresholds_equal': True,
        'training': {run: starts[run]['training'] for run in RUNS}, 'objectives': {run: starts[run]['objective'] for run in RUNS}},
    'methods': {'native': 'Saved native distances divided by matching A. Original TRAIN progress wave bound=0.98A. Calibrated noninferiority bound=A+T; absolute T shown separately; full final/T and original decisions unchanged in receipt.',
        'components': '036 wave and integral are arithmetic means over the three saved hypothesis parts; integral plotted with original weight 2; carrier plotted with old weight 0.25 but not used in036 training.',
        '034': 'Only saved data/objective/regular/gradient are used. No wave/integral/carrier decomposition is fabricated.',
        'common_objective': '034 objective compared to036 objective_with_carrier_report_only; both include0.25carrier. Actual036 objective is separately labeled with zero carrier weight.',
        'smoothing': 'Trailing 50-update arithmetic average; raw values also plotted without clamping.',
        'training_step_timing': 'Loss is computed before the optimizer update for that training step; the metrics record is written after the update. It is not a post-update loss evaluation.',
        'uncertainty': 'One evaluation MC bank per endpoint; four draws per source parent are not independent parents. DEV previously exposed.'},
    'evidence_limits': {run: {key: results[run][key] for key in ['pilot_relative_progress', 'scientific_status',
        'DEV_previously_exposed', 'DEV_used_for_training_or_selection', 'independent_acquisition_acceptance',
        'one_training_seed', 'one_evaluation_MC_bank', 'conditional_identification', 'source_physical_weights', 'noise005_frozen']} for run in RUNS},
    'raw_SOURCE_REAL_reads': 0, 'checkpoint_reads': 0, 'new_NN_calls': 0, 'new_AD_queries': 0,
    'new_optimizer_steps': 0, 'new_observation_calls': 0, 'new_scientific_budget': 0,
    'visual_QA': 'pending_view_image'}
(ROOT / 'PLOT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'figures': figures, 'inputs_verified': len(inputs), 'training_summary': training_summary}, ensure_ascii=False, indent=2))
