"""Plot the requested latest fixed-candidate PSD census without energy cuts."""
from pathlib import Path
from datetime import datetime, timezone
import io
import json
import os

ROOT = Path('E:/GAN/research/N01')
HERE = ROOT / 'receipts/joint_response_01_20261021/full_energy_psd_01'
OUT = ROOT / 'deliverables/joint_response01_full_energy_psd_20261021'
OUT.mkdir(exist_ok=True)
cache = ROOT / 'cache/full_energy_psd_plot_20261021'
cache.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.scale import SymmetricalLogTransform

p = json.loads((ROOT / 'configs/joint_response_01_full_energy_psd_20261021.json').read_bytes())
terminal = json.loads((HERE / 'terminal_01/TERMINAL_VERIFICATION.json').read_bytes())
result = json.loads((HERE / 'terminal_01/RESULT.json').read_bytes())
assert terminal['native_exit']['exit_code'] == 0 and not terminal['processes']
assert terminal['complete_present'] and not terminal['failed_present']
for key, expected in p['operation_budget'].items():
    assert result['execution_counts'].get(key, 0) == expected
assert result['peak_GPU_reserved_bytes'] <= 4 * 1024**3
assert terminal['native_exit']['peak_rss_bytes'] <= 12 * 1024**3
assert result['elapsed_seconds'] <= 600
nas = Path(p['storage']['windows_root']) / 'run_01'
with np.load(io.BytesIO((nas / 'QPSD_SCALARS.npz').read_bytes()), allow_pickle=False) as z:
    arrays = dict(z)
with np.load(io.BytesIO(Path(p['roles_file'].replace('/mnt/MiNas/', 'Z:/')).read_bytes()), allow_pickle=False) as z:
    np.testing.assert_array_equal(arrays['source_parent_rows'], z['source_train_rows'])
    np.testing.assert_array_equal(arrays['real_rows'], z['real_train_rows'])
    assert not np.isin(arrays['source_parent_rows'], z['source_heldout_rows']).any()
    assert not np.isin(arrays['real_rows'], z['real_heldout_rows']).any()
for seed in range(2):
    path = Path(p['candidate_arrays'][seed].replace('/mnt/MiNas/', 'Z:/'))
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
        np.testing.assert_array_equal(arrays[f'seed{seed}_fixed_candidate_theta'], z['candidate'])
hypotheses = p['hypotheses']
groups = ['real'] + [f'seed{seed}_{h}' for seed in range(2) for h in hypotheses]
samples = {}
summary = {}
for group in groups:
    q, tail, psd = [arrays[group + '_' + key] for key in ('Q', 'tail', 'PSD_raw')]
    n = p['real_train_rows'] if group == 'real' else p['source_train_parents'] * p['draws_per_parent']
    assert len(q) == len(psd) == n
    nonzero = q != 0
    np.testing.assert_allclose(psd[nonzero], tail[nonzero] / q[nonzero], rtol=1e-12, atol=1e-12)
    assert np.isnan(psd[~nonzero]).all()
    finite = np.isfinite(q) & np.isfinite(psd)
    samples[group] = (q[finite], psd[finite])
    summary[group] = dict(total_rows=n, plotted_rows=int(finite.sum()), undefined_rows=int((~finite).sum()),
        Q_min=float(q.min()), Q_max=float(q.max()), PSD_min=float(psd[finite].min()),
        PSD_max=float(psd[finite].max()), flags=result['summaries'][group]['flags'])
xmin = min(0., min(v[0].min() for v in samples.values()))
xmax = max(v[0].max() for v in samples.values())
ymin = min(0., min(v[1].min() for v in samples.values()))
ymax = max(v[1].max() for v in samples.values())
dx, dy = xmax - xmin, ymax - ymin
xlim = (xmin - .008 * dx, xmax + .008 * dx)
ylim = (ymin - .015 * dy, ymax + .015 * dy)
xedges = np.linspace(*xlim, 281)
symlog = ylim[0] < -.25 or ylim[1] > 1.25
if symlog:
    transform = SymmetricalLogTransform(base=10, linthresh=.5, linscale=1.)
    yedges = transform.inverted().transform(np.linspace(*transform.transform(np.array(ylim)), 261))
else:
    yedges = np.linspace(*ylim, 261)
histograms = {}
for group, (q, psd) in samples.items():
    hist, _, _ = np.histogram2d(q, psd, bins=(xedges, yedges))
    assert int(hist.sum()) == len(q)
    histograms[group] = hist / summary[group]['total_rows']
    summary[group]['histogram_rows'] = int(hist.sum())
vmin = min(float(v[v > 0].min()) for v in histograms.values())
vmax = max(float(v.max()) for v in histograms.values())
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none'})
fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), sharex=True, sharey=True)
fig.subplots_adjust(left=.082, right=.88, bottom=.125, top=.87, hspace=.23, wspace=.12)
cmap = plt.get_cmap('magma').copy()
cmap.set_bad('#faf9f6')
norm = LogNorm(vmin=vmin, vmax=vmax)
for row, h in enumerate(hypotheses):
    for col, group in enumerate(('real', f'seed0_{h}', f'seed1_{h}')):
        ax = axes[row, col]
        mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(histograms[group].T, 0),
                            cmap=cmap, norm=norm, shading='flat', rasterized=True)
        if symlog:
            ax.set_yscale('symlog', base=10, linthresh=.5, linscale=1.)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=9)
        ax.text(.97, .96, f"N = {summary[group]['plotted_rows']:,}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=3))
        if col == 0:
            ax.set_ylabel(f'{h}\nPSD = Qtail / Q', fontsize=11, labelpad=9)
        if row == 0:
            ax.set_title(('实测 TRAIN 参照', '最新固定候选 · seed 0', '最新固定候选 · seed 1')[col],
                         fontsize=13, pad=12)
        if row == 2:
            ax.set_xlabel('读出电荷 Q / (V·ns)', fontsize=11, labelpad=8)
cax = fig.add_axes([.91, .225, .016, .55])
bar = fig.colorbar(mesh, cax=cax)
bar.set_label('每格事件比例（共同对数色标）', fontsize=11, labelpad=13)
fig.suptitle('最新 JOINT_RESPONSE_01 固定恢复候选｜全能区 PSD',
             x=.47, y=.972, fontsize=20, fontweight='bold')
fig.text(.47, .925, '原 ADC、基线、峰搜索和积分窗；两头候选及三种噪声假设全部保留',
         ha='center', fontsize=12, color='#333333')
undefined = sum(summary[g]['undefined_rows'] for g in groups)
foot = ('训练拟合侧完整人口：6,465 个源父事件 × 32 次/头；实测 18,948 条。无能区、PSD 或异常标志筛选，原留出集保留。\n'
        f'横轴 Q 为未标定能量代理；第一列实测在三行重复用于比较。Q=0 等未定义 PSD：{undefined} 条，单列计数。')
if symlog:
    foot += ' 纵轴使用对称对数，保留全部有限 PSD。'
footer = fig.text(.082, .055, foot, ha='left', va='center', fontsize=10, linespacing=1.9)
png = OUT / 'LATEST_FULL_ENERGY_PSD.png'
svg = OUT / 'LATEST_FULL_ENERGY_PSD.svg'
fig.savefig(png, dpi=200, facecolor='white')
fig.savefig(svg, facecolor='white')
outside_main = {group: int(((samples[group][1] < -.25) | (samples[group][1] > .55)).sum())
                for group in groups}
for ax in axes.flat:
    ax.set_yscale('linear')
    ax.set_ylim(-.25, .55)
footer.set_text('训练拟合侧完整人口：6,465 个源父事件 × 32 次/头；实测 18,948 条。横轴保留全部 Q 范围，原留出集保留。\n'
    f'纵轴放大至 −0.25–0.55；窗外实测 {outside_main["real"]} 条、候选 {sum(outside_main[g] for g in groups[1:])} 条，全部保留在全范围图。Q 尚无 MeV 标定。')
linear_png = OUT / 'LATEST_FULL_ENERGY_PSD_LINEAR.png'
linear_svg = OUT / 'LATEST_FULL_ENERGY_PSD_LINEAR.svg'
fig.savefig(linear_png, dpi=200, facecolor='white')
fig.savefig(linear_svg, facecolor='white')
plt.close(fig)
review = dict(status='PASS_FULL_ENERGY_PSD_DATA_AND_PLOT_COUNTS_AUDIT',
    UTC=datetime.now(timezone.utc).isoformat(), panels=9, unique_populations=7,
    source_parents=6465, draws_per_parent=32, real_rows=18948,
    heldout_rows_plotted=0, energy_cuts=0, PSD_or_flag_cuts=0, raw_PSD_arithmetic_verified=True,
    same_two_fixed_candidate_vectors=True, x_limits=list(map(float, xlim)),
    y_limits=list(map(float, ylim)), y_scale='symlog' if symlog else 'linear',
    histogram_bins=[280, 260], colorbar_quantity='events in cell / all rows in that population',
    summaries=summary, PNG=str(png), SVG=str(svg), new_candidates=0, parameter_updates=0,
    linear_main_PNG=str(linear_png), linear_main_SVG=str(linear_svg),
    linear_main_PSD_limits=[-.25, .55], rows_outside_linear_main=outside_main,
    scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False)
(HERE / 'PLOT_DATA_AUDIT.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
print(json.dumps(review))
