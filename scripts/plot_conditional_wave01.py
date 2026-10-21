"""Saved conditional-wave risks and native-direction conflicts; no model forward."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
memory = json.loads((args.receipt / 'CONDITIONAL_WAVE_MEMORY.json').read_bytes())
result = json.loads((args.receipt / 'TRAINING_RESULT.json').read_bytes())
steps = [json.loads(line) for line in (args.receipt / 'TRAINING_STEPS.jsonl').read_text().splitlines()]
step = steps[0]
supported = np.array(memory['supported'])
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 11})
fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
at = 0
for j, level in enumerate((16, 32, 64)):
    names = memory['names'][at:at + level]
    keep = supported[at:at + level]
    x = np.arange(level)
    for label, title, color, style in [('initial', '原始', '#777777', '-'),
            ('inherited', '上一轮 B', '#4575b4', '-'), ('B', '本轮端点', '#d55e00', '--')]:
        y = np.array([result['endpoint'][label]['positive_metrics'][k] for k in names])
        axes[0, j].plot(x, y, style, color=color, label=title)
        axes[0, j].scatter(x[~keep], y[~keep], facecolors='none', edgecolors=color, s=22)
    baseline = np.array([step['previous_metrics'][k] for k in names])
    for probe in step['probes']:
        changes = np.array([probe['positive_metrics'][k] for k in names]) - baseline
        axes[1, j].plot(x, changes, label=probe['direction'])
    axes[1, j].axhline(0, color='black', lw=.8)
    axes[0, j].set_title(f'{level} 个重叠软 Q 区间')
    axes[0, j].set_yscale('symlog', linthresh=1e-3)
    axes[1, j].set_xlabel('软 Q 区间序号（Q 从左向右增大）')
    for ax in axes[:, j]:
        ax.grid(alpha=.2)
        ax.set_xlim(0, level - 1)
    at += level
axes[0, 0].set_ylabel('固定真实人口尺度下的局部波形风险')
axes[1, 0].set_ylabel('原生方向相对继承状态的风险变化')
axes[0, 2].legend(fontsize=10)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(.5, -.22), fontsize=9, ncol=2)
fig.suptitle('条件波形记忆：独立终点与单步方向检查', fontsize=18)
fig.supxlabel(f'新增提交 {result["new_committed_optimizer_updates"]} 次；逐区间约束覆盖 {supported.sum()}/{len(supported)} 个单元。'
    '\n空心点为支撑不足的单元，仍参与目标；下排正值表示该方向使局部风险增大。', fontsize=11)
args.out.mkdir(parents=True, exist_ok=True)
for suffix in ('png', 'svg'):
    target = args.out / f'CONDITIONAL_WAVE_RISK.{suffix}'
    fig.savefig(target, dpi=150)
    if suffix == 'svg':
        target.write_bytes(('\n'.join(line.rstrip() for line in target.read_text(encoding='utf8').splitlines()) + '\n').encode())
plt.close(fig)
print(json.dumps(dict(cells=len(supported), supported=int(supported.sum()), new_updates=result['new_committed_optimizer_updates'])))
