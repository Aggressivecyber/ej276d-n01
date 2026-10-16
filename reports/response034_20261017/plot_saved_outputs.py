"""Render saved032/034 aggregate evidence only; no model or event-array imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.titlesize': 18,
    'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9FAAB4', 'xtick.color': '#35424E', 'ytick.color': '#35424E',
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

HYPOTHESES = ['fast', 'slow', 'common_level']
RUNS = ['032', '034']
COLORS = {'A': '#2878A6', 'G0': '#D99B35', '032': '#7953A1', '034': '#C94747'}
SERIES_LABELS = {'A': 'A：原确定性响应 + 005', 'G0': 'G0：共同初态',
    '032': '032 final', '034': '034 final'}
SOURCES = {run: PROJECT / f'receipts/response{run}_20261017/completion_01' for run in RUNS}
MANIFEST_NAMES = {'032': 'COLLECTION_MANIFEST_01.json', '034': 'COLLECTION_MANIFEST.json'}
input_records = []
manifests = {}
manifest_records = []
for run in RUNS:
    path = SOURCES[run] / MANIFEST_NAMES[run]
    payload = path.read_bytes()
    manifests[run] = {row['path']: row for row in json.loads(payload)['files']}
    manifest_records.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()})


def read(run, name, jsonl=False):
    relative = f'run_01/{name}'
    path = SOURCES[run] / relative
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == manifests[run][relative]['sha256'], str(path)
    assert len(payload) == manifests[run][relative]['bytes'], str(path)
    input_records.append({'path': str(path), 'bytes': len(payload), 'sha256': digest,
        'collection_manifest_match': True})
    return [json.loads(line) for line in payload.splitlines()] if jsonl else json.loads(payload)


def save(fig, name):
    path = ROOT / name
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    with Image.open(path) as bitmap:
        pixels = list(bitmap.size)
    return {'path': str(path), 'bytes': path.stat().st_size, 'pixels': pixels,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


starts = {run: read(run, 'START.json')['config'] for run in RUNS}
results = {run: read(run, 'RESULT.json') for run in RUNS}
records = {(run, split, endpoint, hypothesis): read(run, f'{split}_{endpoint}_{hypothesis}.json')
    for run in RUNS
    for split, endpoints in [('TRAIN', ['A', 'G0', 'final']), ('DEV', ['A', 'final'])]
    for endpoint in endpoints for hypothesis in HYPOTHESES}
training = {run: read(run, 'metrics.jsonl', jsonl=True) for run in RUNS}

assert starts['032']['objective'] == starts['034']['objective'] == 'fixed_joint_energy_from031'
assert starts['032']['evaluation'] == starts['034']['evaluation']
assert starts['032']['seed'] == starts['034']['seed']
assert {key: value for key, value in starts['034']['training'].items() if key != 'shape_lr'} == starts['032']['training']
for split, endpoints in [('TRAIN', ['A', 'G0']), ('DEV', ['A'])]:
    for endpoint in endpoints:
        for hypothesis in HYPOTHESES:
            assert records[('032', split, endpoint, hypothesis)]['values'] == records[('034', split, endpoint, hypothesis)]['values']
for split in ['TRAIN', 'DEV']:
    for hypothesis in HYPOTHESES:
        for metric in ['wave', 'QPSD', 'noise']:
            assert results['032']['comparisons'][split][hypothesis]['calibrated_components'][metric]['threshold'] == results['034']['comparisons'][split][hypothesis]['calibrated_components'][metric]['threshold']

figures = []
fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.2))
fig.subplots_adjust(top=.80, bottom=.18, left=.065, right=.98, hspace=.42, wspace=.26)
fig.suptitle('034：原生 wave / QPSD 距离仍高于 A 与 032 末态', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .925, '032 / 034：同一目标、同一 seed、2000 步；034 的 shape 学习率为原来的 1/16', ha='center', fontsize=12)
fig.text(.5, .875, '034 pilot_relative_progress = false；response_match_not_established', ha='center', fontsize=12, color='#A32F37')
for row, split in enumerate(['TRAIN', 'DEV']):
    series = ['A', 'G0', '032', '034'] if split == 'TRAIN' else ['A', '032', '034']
    width = .18 if split == 'TRAIN' else .23
    x = np.arange(3)
    for col, (metric, metric_label) in enumerate([('wave', 'Native waveform [wave]'), ('QPSD', 'Native Q/PSD [QPSD]'), ('noise', 'Native noise [noise]')]):
        ax = axes[row, col]
        top_values = []
        for index, series_name in enumerate(series):
            run, endpoint = ('034', series_name) if series_name in ['A', 'G0'] else (series_name, 'final')
            values = [records[(run, split, endpoint, h)]['values'][metric] for h in HYPOTHESES]
            top_values.extend(values)
            positions = x + (index - (len(series) - 1) / 2) * width
            bars = ax.bar(positions, values, width * .90, color=COLORS[series_name], zorder=3, label=SERIES_LABELS[series_name])
            ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=3, rotation=90, fontsize=9)
        thresholds = [results['034']['comparisons'][split][h]['calibrated_components'][metric]['threshold'] for h in HYPOTHESES]
        for position, threshold in zip(x, thresholds):
            ax.plot([position - .43, position + .43], [threshold, threshold], color='#4D5259', ls='--', lw=1.2, zorder=4)
        top = max(top_values + thresholds)
        ax.set(xticks=x, xticklabels=HYPOTHESES, ylabel=f'{split} 距离（越低越好）', title=metric_label, ylim=(0, top * 1.24))
        ax.grid(axis='y', color='#E2E8EC', lw=.7, zorder=0)
        ax.set_axisbelow(True)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color='#4D5259', ls='--', lw=1.2))
labels.append('各 hypothesis 冻结容差 T')
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=5, frameon=False)
fig.text(.5, .055, 'TRAIN：6593 父 × 4 draw / REAL 21047；DEV：512 父 × 4 draw / REAL 599。DEV G0 未保存，未补算。', ha='center', fontsize=10)
fig.text(.5, .015, '每次评价均为单一 MC bank；四 draw 共享父身份。DEV 已暴露；无独立采集验收。noise 冻结且各状态相同。', ha='center', fontsize=10)
figures.append(save(fig, 'native_distances_032_034.png'))

scalar_metrics = ['Q_Vns', 'PSD_raw', 'PSD_guarded', 'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns']
scalar_labels = ['Q_Vns\nV·ns', 'PSD_raw\n无量纲', 'PSD_guarded\n无量纲', 'peak_voltage_V\nV', 'rise10_90_ns\nns', 'FWHM_ns\nns']
fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.4))
fig.subplots_adjust(top=.76, bottom=.22, left=.10, right=.96, wspace=.30)
fig.suptitle('标量分布距离：A / 032 final / 034 final', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .918, '三种 hypothesis 全部展示；格内为保存的距离数值，非 Q、峰值或时间的物理均值', ha='center', fontsize=12)
fig.text(.5, .865, '颜色仅表示同 split、同 hypothesis 的距离 / A；不表示科学验收通过', ha='center', fontsize=11, color='#A32F37')
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=6)
for ax, split in zip(axes, ['TRAIN', 'DEV']):
    table, ratios, row_labels = [], [], []
    for hypothesis in HYPOTHESES:
        baseline = np.array([records[('034', split, 'A', hypothesis)]['values'][m] for m in scalar_metrics])
        for series_name in ['A', '032', '034']:
            run, endpoint = ('034', 'A') if series_name == 'A' else (series_name, 'final')
            values = np.array([records[(run, split, endpoint, hypothesis)]['values'][m] for m in scalar_metrics])
            table.append(values)
            ratios.append(values / baseline)
            row_labels.append(f'{hypothesis} / {series_name}')
    table, ratios = np.array(table), np.array(ratios)
    heatmap = ax.imshow(np.log2(ratios), cmap='RdBu_r', norm=norm, aspect='auto')
    for row in range(9):
        for col in range(6):
            ax.text(col, row, f'{table[row, col]:.4g}', ha='center', va='center', fontsize=10.2,
                color='white' if ratios[row, col] >= 14 else '#223746', fontweight='bold' if row % 3 == 2 else 'normal')
    ax.set(xticks=np.arange(6), xticklabels=scalar_labels, yticks=np.arange(9), yticklabels=row_labels,
        title='TRAIN' if split == 'TRAIN' else 'DEV：已暴露')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, axis='both', pad=8)
    ax.tick_params(axis='x', labelsize=9.5)
    for line in [2.5, 5.5]:
        ax.axhline(line, color='white', lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
color_axis = fig.add_axes([.34, .15, .36, .022])
bar = fig.colorbar(heatmap, cax=color_axis, orientation='horizontal', extend='both', ticks=[-1, 0, 2, 4, 6])
bar.ax.set_xticklabels(['0.5×', '1× A', '4×', '16×', '64×'])
bar.set_label('距离 / A；颜色在 0.5× 至 64× 饱和，格内数值不截断', fontsize=10)
fig.text(.5, .052, 'rise / FWHM 使用显式有定义子总体；未定义行仍留在 flags，未设为 0。034 TRAIN fast 的 rise 未定义率差：14.994 个百分点。', ha='center', fontsize=10)
fig.text(.5, .015, '单一 MC bank，无独立重复误差条；032 与 034 的 A 数值完全一致。最终响应匹配均未建立。', ha='center', fontsize=10)
figures.append(save(fig, 'scalar_distances_032_034.png'))

fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.0))
fig.subplots_adjust(top=.81, bottom=.16, left=.075, right=.97, hspace=.40, wspace=.26)
fig.suptitle('032 / 034 同定义训练轨迹：降低 shape 学习率未带来响应验收通过', y=.985, color='#223746', fontweight='bold')
fig.text(.5, .922, '固定 joint energy 目标；细线为逐步值，粗线为 50 步滑动均值；data 可为负，未截断', ha='center', fontsize=12)
trace_keys = ['data', 'regular', 'preclip_gradient_norm', 'mean_residual_saturation_fraction']
trace_titles = ['训练 data：joint energy', '原始正则项 R（目标权重 0.01）', '裁剪前梯度范数 [preclip_gradient_norm]', '均值残差饱和比例 [mean_residual_saturation_fraction]']
training_summary = {}
for run in RUNS:
    rows = training[run]
    steps = np.array([row['step'] for row in rows])
    arrays = {key: np.array([row[key] for row in rows]) for key in trace_keys + ['objective', 'saturation']}
    clips = arrays['preclip_gradient_norm'] > starts[run]['training']['clip_norm']
    training_summary[run] = {'steps': len(rows), 'seed': starts[run]['seed'],
        'clipped_steps': int(clips.sum()), 'clipped_fraction': float(clips.mean()),
        'last100_clipped_steps': int(clips[-100:].sum()), 'last100_clipped_fraction': float(clips[-100:].mean()),
        'metrics': {key: {'first': float(value[0]), 'last': float(value[-1]), 'last100_mean': float(value[-100:].mean()),
            'minimum': float(value.min()), 'maximum': float(value.max())} for key, value in arrays.items()}}
    for ax, key, title in zip(axes.flat, trace_keys, trace_titles):
        ax.plot(steps, arrays[key], color=COLORS[run], alpha=.16, lw=.55)
        ax.plot(steps[49:], np.convolve(arrays[key], np.ones(50) / 50, mode='valid'), color=COLORS[run], lw=1.8, label=f'{run}：50 步均值')
        ax.set(title=title, xlabel='已完成更新次数', xlim=(0, 2000))
        ax.grid(axis='y', color='#E2E8EC', lw=.7)
axes[0, 0].axhline(0, color='#77838C', ls=':', lw=.8)
axes[0, 0].set_ylabel('data')
axes[0, 1].set_ylabel('R（未加权）')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel('L2 norm（log 轴）')
axes[1, 0].axhline(5, color='#475664', ls='--', lw=1.2)
axes[1, 0].text(.02, .97, 'clip > 5：032 80.3%；034 98.7%\n末 100 步：032 65%；034 100%', transform=axes[1, 0].transAxes, va='top', fontsize=10,
    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 3})
axes[1, 1].set_ylabel('比例')
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color='#475664', ls='--', lw=1.2))
labels.append('clip 阈值 5')
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
fig.text(.5, .015, '每个 run 一个训练 seed；优化轨迹不能替代独立评价或响应匹配门。全部取自保存 JSONL，无新增模型或梯度调用。', ha='center', fontsize=10)
figures.append(save(fig, 'training_traces_032_034.png'))

endpoint_numbers = {}
for (run, split, endpoint, hypothesis), record in records.items():
    endpoint_numbers.setdefault(run, {}).setdefault(split, {}).setdefault(hypothesis, {})[endpoint] = {
        'values': record['values'], 'defined_populations': record['defined_populations'],
        'source_parents': record['source_parents'], 'draws_per_parent': record['draws_per_parent'],
        'fake_rows': record['fake_rows'], 'real_rows': record['real_rows'], 'MC_banks': record['MC_banks'],
        'all_rows_retained': record['all_rows_retained'], 'previously_exposed_DEV': record['previously_exposed_DEV']}
receipt = {'schema': 'n01/response034-saved-output-plots/v1', 'UTC': datetime.now(timezone.utc).isoformat(),
    'status': 'rendered_pending_visual_QA', 'generator': str(Path(__file__)),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_roots': {run: str(path / 'run_01') for run, path in SOURCES.items()},
    'collection_manifests': manifest_records, 'inputs': input_records, 'all_read_inputs_match_collection_SHA256': True,
    'figures': figures, 'endpoint_full_precision': endpoint_numbers,
    'training_summary': training_summary,
    'same_definition_checks': {'objective_equal': True, 'evaluation_config_equal': True, 'training_seed_equal': True,
        'training_config_equal_except_shape_lr': True, 'A_and_TRAIN_G0_values_exactly_equal': True,
        'native_metric_thresholds_equal': True, 'objective': starts['034']['objective'],
        'training': {run: starts[run]['training'] for run in RUNS}, 'evaluation': starts['034']['evaluation']},
    'final034_over_final032': {split: {h: {m: records[('034', split, 'final', h)]['values'][m] / records[('032', split, 'final', h)]['values'][m]
        for m in ['wave', 'QPSD', 'noise'] + scalar_metrics} for h in HYPOTHESES} for split in ['TRAIN', 'DEV']},
    'calibrated_comparisons': {run: results[run]['comparisons'] for run in RUNS},
    'methods': {'native_figures': 'Unchanged saved native distances and frozen thresholds; lower is better.',
        'scalar_figure': 'Saved distances, not physical means; rows are explicit defined subpopulations. Color is log2(distance/A) with range [-1,6]; text is untruncated 4-significant-digit display.',
        'trajectories': 'Same saved fields; raw values plus 50-update trailing arithmetic means; gradient log axis; R is unweighted.',
        'uncertainty': 'One training seed and one evaluation MC bank per run; no independent replicate error bars.',
        'missing_DEV_G0': 'Not saved in either run; not recomputed or imputed.'},
    'evidence_limits': {run: {key: results[run][key] for key in ['pilot_relative_progress', 'scientific_status',
        'DEV_previously_exposed', 'DEV_used_for_training_or_selection', 'independent_acquisition_acceptance',
        'one_training_seed', 'one_evaluation_MC_bank', 'conditional_identification', 'source_physical_weights', 'noise005_frozen']} for run in RUNS},
    'saved_NPZ_archives_read': 0, 'saved_event_waveform_members_decoded': 0, 'raw_SOURCE_REAL_reads': 0,
    'checkpoint_reads': 0, 'new_NN_calls': 0, 'new_AD_queries': 0, 'new_optimizer_steps': 0,
    'visual_QA': 'pending_view_image'}
(ROOT / 'PLOT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({'figures': figures, 'training_summary': training_summary}, ensure_ascii=False, indent=2))
