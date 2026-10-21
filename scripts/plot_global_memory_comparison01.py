"""Compare saved descriptive conditional reports without loading waveform data."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--first', type=Path)
parser.add_argument('--second', type=Path)
parser.add_argument('--latest', type=Path)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
latest = json.loads(args.latest.read_bytes()) if args.latest else None
if latest and latest['native_pilot_result']['mode'].startswith('charge_spline'):
    entries = [('initial', '修复前'), ('inherited', '继承模型'), ('anchored', '固定条件起点')]
    if latest['native_pilot_result']['mode'] == 'charge_spline_restoration':
        entries.append(('stage_start', '上一阶段 · 24步候选'))
    entries.append(('B', '样条候选 · 固定终点'))
    series = {label: latest['models'][key]['64'] for key, label in entries}
else:
    first, second = [json.loads(p.read_bytes()) for p in (args.first, args.second)]
    series = {
        '修复前': first['models']['initial']['64'],
        '第一轮 A': first['models']['A']['64'],
        '第一轮 B': first['models']['B']['64'],
        '第二轮 B · 波形方向约束': second['models']['B']['64'],
    }
if latest and not latest['native_pilot_result']['mode'].startswith('charge_spline'):
    if 'inherited' in latest['models']:
        series['上一轮 B · 联合约束'] = latest['models']['inherited']['64']
        label = '本轮 B · 条件波形记忆'
        if latest['native_pilot_result']['mode'] == 'mean_trend_repair':
            label = f"本轮 B · 趋势约束（{latest['native_pilot_result']['new_committed_optimizer_updates']} 更新）"
        series[label] = latest['models']['B']['64']
    else:
        series['本轮 B · 联合约束'] = latest['models']['B']['64']
initial = series['修复前']['cells']
centers = np.array([(c['Q_left'] + c['Q_right']) / 2 for c in initial])
assert all([c['Q_left'] for c in entry['cells']] == [c['Q_left'] for c in initial]
           for entry in series.values())
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 12})
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, layout='constrained')
colors = ['#686868', '#4575b4', '#d55e00', '#009e73', '#8e44ad', '#c68d00']
axes[1].plot(centers, np.array([c['real_rows'] for c in initial]) / 18948,
             color='black', lw=2, label='REAL · 合法 FIT')
for (name, entry), color in zip(series.items(), colors):
    cells = entry['cells']
    w1 = np.array([np.nan if c['empirical_PSD_W1'] is None else c['empirical_PSD_W1'] for c in cells])
    sparse = np.array([c['sparse_support'] for c in cells])
    axes[0].plot(centers, w1, color=color, label=name)
    axes[0].scatter(centers[sparse], w1[sparse], facecolors='none', edgecolors=color, s=30)
    axes[1].plot(centers, np.array([c['generated_rows'] for c in cells]) / entry['denominator'],
                 color=color, label=name)
    axes[2].plot(centers, [c['unique_source_parents'] for c in cells], color=color, label=name)
axes[0].set_ylabel('条件 PSD 的经验 W1')
axes[0].set_title('64 个等宽 Q 区间；W1 越低越接近 REAL')
axes[1].set_ylabel('区间人口概率')
axes[2].set_ylabel('独立 SOURCE 父事件数')
axes[2].axhline(30, ls=':', lw=1, color='black')
axes[2].set_xlabel('原生读出 Q（V·ns；未作物理能量标定）')
for ax in axes:
    ax.grid(alpha=.2)
    ax.set_xlim(initial[0]['Q_left'], initial[-1]['Q_right'])
    ax.set_ylim(bottom=0)
axes[0].legend(ncol=4, fontsize=11, loc='upper right')
axes[1].legend(ncol=5, fontsize=10, loc='upper right')
fig.suptitle('全域记忆修复 · 细分 Q 区间对比', fontsize=19)
caption = ('空心点：REAL 少于 30 行或 SOURCE 少于 30 个父事件；缺失区间留空，不计作零误差。'
           '\n32 次响应/父事件；报告为 TRAIN 分布描述，未作显著性检验。')
if latest and latest['native_pilot_result'].get('independent_final_random_banks') is False:
    caption += '\n终点 seed 与第21步训练重合，不是独立随机复核。'
fig.supxlabel(caption, fontsize=11)
args.out.mkdir(parents=True, exist_ok=True)
for suffix in ('png', 'svg'):
    path = args.out / f'CONDITIONAL_PSD_64_COMPARISON.{suffix}'
    fig.savefig(path, dpi=150)
    if suffix == 'svg':
        path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines()) + '\n').encode())
plt.close(fig)
audit = {name: dict(missing_W1_cells=sum(c['empirical_PSD_W1'] is None for c in entry['cells']),
                   sparse_cells=sum(c['sparse_support'] for c in entry['cells']),
                   underflow_rows=entry['underflow_rows'], overflow_rows=entry['overflow_rows'])
         for name, entry in series.items()}
(args.out / 'CONDITIONAL_COMPARISON_AUDIT.json').write_bytes(
    (json.dumps(audit, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps(audit, ensure_ascii=False))
