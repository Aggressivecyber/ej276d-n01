"""Descriptive full-Q endpoint figures and conditional empirical distributions."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('--arrays', type=Path, required=True)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
with np.load(args.arrays, allow_pickle=False) as archive:
    data = {key: archive[key].copy() for key in archive.files}
result = json.loads(args.result.read_bytes())
labels = list(result['endpoint'])
if 'inherited' in labels:
    labels = ['initial', 'inherited', 'B']
if result['mode'] == 'charge_spline_mean_repair':
    labels = ['initial', 'inherited', 'anchored', 'B']
groups = {'REAL': (data['real_Q'], data['real_PSD_raw'])}
for label in labels:
    groups[label] = (data[f'{label}_bank0_Q'], data[f'{label}_bank0_PSD_raw'])
rows = {k: len(v[0]) for k, v in groups.items()}
assert rows['REAL'] == 18948
assert all(rows[k] == 6465 * 32 for k in labels)
qrange = (min(v[0].min() for v in groups.values()), max(v[0].max() for v in groups.values()))
prange = (min(v[1][np.isfinite(v[1])].min() for v in groups.values()),
          max(v[1][np.isfinite(v[1])].max() for v in groups.values()))
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 12})
titles = {'REAL': 'REAL · 合法 FIT', 'initial': '修复前 · 固定端点',
          'A': 'A · 全域目标 + 53 参数', 'B': 'B · 全域目标 + 末端头（247 参数）'}
if 'A' not in labels:
    titles['B'] = 'B · 全域目标 + 波形方向约束'
if result['mode'] == 'joint_secant_pilot':
    titles['B'] = 'B · 联合 Q / 波形 / ADC 约束'
if 'inherited' in labels:
    titles.update(inherited='上一轮 B · 累计 10 次更新', B='条件波形记忆 · 本轮端点')
if result['mode'] == 'mean_trend_repair':
    titles['B'] = ('条件路径解冻 + 波形趋势约束\n'
                   f"本轮端点 · 新增 {result['new_committed_optimizer_updates']} 次更新")
if result['mode'] == 'charge_spline_mean_repair':
    verdict = '联合验收通过' if result['endpoint']['B']['passes_bounded_pilot'] else '联合验收未通过'
    titles.update(anchored='固定条件起点 · 未训练',
                  B=f"样条均值支路 · {result['optimizer_updates']} 次更新\n{verdict}")
audit = dict(populations=rows, actual_Q_range=list(map(float, qrange)),
             actual_PSD_range=list(map(float, prange)), hypothesis='fast', seed_head=0,
             draws_per_parent=32, inference_bank='independent endpoint bank0',
             full_population_weights_preserved=True, physical_energy_calibration=False)
for mode, yrange in (('LINEAR', (-.25, .55)), ('FULL_RANGE', prange)):
    qedges = np.linspace(*qrange, 281)
    pedges = np.linspace(*yrange, 261)
    histograms, outside = {}, {}
    for key, (q, psd) in groups.items():
        finite = np.isfinite(psd)
        histograms[key] = np.histogram2d(q[finite], psd[finite], bins=(qedges, pedges))[0] / len(q)
        outside[key] = dict(undefined=int((~finite).sum()),
                            below_view=int((psd < yrange[0]).sum()),
                            above_view=int((psd > yrange[1]).sum()),
                            plotted=int(round(histograms[key].sum() * len(q))))
        if mode == 'FULL_RANGE':
            assert outside[key]['plotted'] + outside[key]['undefined'] == len(q)
    norm = LogNorm(vmin=1 / max(rows.values()), vmax=max(h.max() for h in histograms.values()))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.875 * len(groups), 6.2), sharex=True, sharey=True,
                             layout='constrained')
    for ax, (key, h) in zip(axes, histograms.items()):
        image = ax.pcolormesh(qedges, pedges, np.ma.masked_equal(h.T, 0),
                             norm=norm, cmap='viridis', rasterized=True)
        ax.set_title(titles[key], fontsize=13)
        ax.set_xlabel('原生读出 Q（V·ns）')
        ax.set_xlim(qrange)
        ax.set_ylim(yrange)
        ax.grid(alpha=.15)
        removed = outside[key]['below_view'] + outside[key]['above_view']
        ax.text(.02, .02, f'N={rows[key]:,}；视窗外 {removed}', transform=ax.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    if mode == 'FULL_RANGE':
        axes[0].set_yscale('symlog', linthresh=.05)
    axes[0].set_ylabel('PSD = tail / Q')
    fig.colorbar(image, ax=axes, shrink=.86, label='每格人口概率（统一归一）')
    fig.suptitle('全 Q 范围 PSD 修复小试验 · 同一参考电子学假设 fast', fontsize=18)
    fig.supxlabel('32 次独立响应/父事件；Q 未作物理能量标定。'
                  '图中未剔除任何 Q 区间；主视图的 PSD 视窗外记录另见 FULL_RANGE 图。', fontsize=11)
    for suffix in ('png', 'svg'):
        path = args.out / f'GLOBAL_MEMORY_PSD_{mode}.{suffix}'
        fig.savefig(path, dpi=150)
        if suffix == 'svg':
            path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines()) + '\n').encode())
    plt.close(fig)
    audit[mode] = outside

summary = {}
parent_rows = np.repeat(data['source_parent_rows'], 32)
for name in labels:
    entry = {}
    for bins in (16, 32, 64):
        # Equal-width observable Q bins; endpoint overflow mass remains explicit.
        edges = np.linspace(data['real_Q'].min(), data['real_Q'].max(), bins + 1)
        iq = np.clip(np.searchsorted(edges, data['real_Q'], side='right') - 1, 0, bins - 1)
        gq, gp = groups[name]
        gi = np.searchsorted(edges, gq, side='right') - 1
        gi[gq == edges[-1]] = bins - 1
        cells = []
        for k in range(bins):
            rm, gm = iq == k, gi == k
            rr = data['real_PSD_raw'][rm]
            gg = gp[gm]
            rf, gf = rr[np.isfinite(rr)], gg[np.isfinite(gg)]
            cell = dict(Q_left=float(edges[k]), Q_right=float(edges[k + 1]),
                        real_rows=int(rm.sum()), generated_rows=int(gm.sum()),
                        unique_source_parents=int(np.unique(parent_rows[gm]).size),
                        real_PSD_undefined=int((~np.isfinite(rr)).sum()),
                        generated_PSD_undefined=int((~np.isfinite(gg)).sum()))
            cell['empirical_PSD_W1'] = float(wasserstein_distance(rf, gf)) if len(rf) and len(gf) else None
            cell['sparse_support'] = len(rf) < 30 or cell['unique_source_parents'] < 30
            cells.append(cell)
        entry[str(bins)] = dict(cells=cells, underflow_rows=int((gi < 0).sum()),
                                overflow_rows=int((gi >= bins).sum()),
                                denominator=len(gq), missing_bins_not_scored_as_zero=True)
    psd_by_parent = gp.reshape(6465, 32)
    entry['conditional_sampling'] = dict(
        mean_within_parent_PSD_variance=float(np.nanvar(psd_by_parent, axis=1, ddof=1).mean()),
        between_parent_PSD_mean_variance=float(np.nanmean(psd_by_parent, axis=1).var()))
    summary[name] = entry
(args.out / 'PLOT_DATA_AUDIT.json').write_bytes((json.dumps(audit, indent=2, ensure_ascii=False) + '\n').encode())
(args.out / 'CONDITIONAL_PSD_REPORT.json').write_bytes((json.dumps(
    dict(scope='Descriptive TRAIN-role endpoint slices, not paired physical-energy truth or scientific acceptance.',
         models=summary, native_pilot_result=result), indent=2, ensure_ascii=False) + '\n').encode())
print(json.dumps(dict(status='PLOTS_AND_DESCRIPTIVE_SLICES_WRITTEN', populations=rows,
                      visual_QA_required=True, scientific_acceptance=False)))
