"""Attribute the saved frozen PSD comb to mean-path components and readout."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--arrays', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
with np.load(args.arrays, allow_pickle=False) as z:
    a = {k: z[k].copy() for k in z.files}
p = json.loads(args.config.read_bytes())
args.out.mkdir(parents=True, exist_ok=True)
assert len(a['source_parent_rows']) == 6465
plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'font.size': 11})
source_q = a['source_Q']
source_psd = a['SOURCE_clean_dynamic_PSD_raw']
summary = dict(source_rows=6465, real_rows=len(a['real_Q']), optimizer_updates=0,
    original_weights_frozen=True, groups={}, all_rows={}, controls_not_deliverable_models=True)
for stage in p['stages']:
    for mode in p['modes']:
        q = a[f'{stage}_{mode}_Q']
        y = a[f'{stage}_{mode}_PSD_raw']
        finite = np.isfinite(y)
        summary['all_rows'][stage + '_' + mode] = dict(rows=len(y), undefined=int((~finite).sum()),
            Q_nonpositive=int((q <= 0).sum()), PSD_min=float(y[finite].min()), PSD_max=float(y[finite].max()))
        assert np.allclose(y[finite], a[f'{stage}_{mode}_tail'][finite] / q[finite], atol=1e-12, rtol=1e-12)
for key in ('A0_reconstructed', 'filter_only', 'time_only', 'residual_no_time',
            'A0_fixed_residual_condition', 'A0_fixed_time_condition'):
    q = a[key + '_raw_canonical_Q']
    summary.setdefault('raw_shape_integrals', {})[key] = dict(minimum=float(q.min()), nonpositive=int((q <= 0).sum()))

curves = {}
memberships = []
for target in (.207919414, .312575162):
    mask = np.abs(source_psd - target) < 1e-6
    order = np.flatnonzero(mask)[np.argsort(source_q[mask], kind='stable')]
    slices = np.array_split(order, 128)
    memberships.append(order)
    x = np.array([np.median(source_q[s]) for s in slices])
    feature = a['unit_source'][order]
    shape_rms = np.sqrt(((feature - feature.mean(0))**2).mean(1))
    group = dict(rows=len(order), source_PSD_range=[float(source_psd[order].min()), float(source_psd[order].max())],
        normalized_source_shape_RMS_spread_quantiles=np.quantile(shape_rms, [0, .5, .95, 1]).tolist(),
        time_scale_range=[float(a['time_scale'][order].min()), float(a['time_scale'][order].max())],
        residual_to_filtered_integral_ratio_quantiles=np.quantile(a['R_Q'][order] / a['W_Q'][order], [0, .5, 1]).tolist(),
        stages={})
    curves[str(target)] = dict(source_Q=x.tolist(), curves={})
    for stage in p['stages']:
        for mode in p['modes']:
            y = a[f'{stage}_{mode}_PSD_raw'][order]
            medians = np.array([np.nanmedian(a[f'{stage}_{mode}_PSD_raw'][s]) for s in slices])
            tv = float(abs(np.diff(medians)).sum())
            group['stages'][stage + '_' + mode] = dict(PSD_range=[float(np.nanmin(y)), float(np.nanmax(y))],
                slice_median_total_variation=tv,
                slice_median_excess_variation=tv - float(abs(medians[-1] - medians[0])))
            curves[str(target)]['curves'][stage + '_' + mode] = medians.tolist()
    direct = a['A0_clean_dynamic_PSD_raw'][order]
    fixed = a['A0_clean_fixed126_PSD_raw'][order]
    adc = a['A0_ADC_dynamic_PSD_raw'][order]
    correction = a['residual_PSD_correction'][order]
    w = a['time_only_clean_fixed126_PSD_raw'][order]
    assert np.allclose(fixed, w + correction, atol=1e-12, rtol=1e-12)
    group.update(ADC_PSD_RMSE=float(np.sqrt(np.mean((adc - direct)**2))),
        dynamic_vs_fixed_PSD_RMSE=float(np.sqrt(np.mean((direct - fixed)**2))),
        A0_peak_index_counts={str(k): int(v) for k, v in zip(*np.unique(a['A0_clean_dynamic_peak_index'][order], return_counts=True))},
        current_mean_vs_A0_PSD_RMSE=float(np.sqrt(np.mean((a['current_mean_clean_dynamic_PSD_raw'][order] - direct)**2))),
        fixed_PSD_variance_decomposition=dict(full=float(np.var(fixed)), warped_filter=float(np.var(w)),
            residual_correction=float(np.var(correction)), twice_covariance=float(2 * np.mean((w - w.mean()) * (correction - correction.mean())))))
    base_tv = group['stages']['A0_clean_fixed126']['slice_median_total_variation']
    group['fixed_window_TV_relative_to_A0'] = {stage: group['stages'][stage + '_clean_fixed126']['slice_median_total_variation'] / base_tv
        for stage in ('SOURCE', 'filter_only', 'time_only', 'residual_no_time', 'A0_fixed_residual_condition', 'A0_fixed_time_condition', 'current_mean')}
    summary['groups'][str(target)] = group
summary['dominant_group_rows'] = sum(len(o) for o in memberships)
summary['other_rows_retained'] = 6465 - summary['dominant_group_rows']
summary['descriptive_resolution'] = '128 equal-count source-Q slices, medians and total variation; not uncertainty intervals or physical particle labels.'
(args.out / 'COMPONENT_ANALYSIS.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode())
(args.out / 'CONDITIONAL_TREND_CURVES.json').write_bytes((json.dumps(curves, indent=2) + '\n').encode())

def save(fig, name):
    for suffix in ('png', 'svg'):
        path = args.out / (name + '.' + suffix)
        fig.savefig(path, dpi=160)
        if suffix == 'svg':
            path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines()) + '\n').encode())
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
panels = [
    [('SOURCE', '输入波形', '#777777'), ('filter_only', '仅滤波', '#009e73'),
     ('time_only', '滤波＋时间变换', '#4477aa'), ('A0', '完整 A0', '#cc3311')],
    [('A0', '完整 A0', '#cc3311'), ('residual_no_time', '保留残差、取消时间变换', '#eeaa33'),
     ('A0_fixed_residual_condition', '固定残差的 Q / 幅度条件', '#009e73'),
     ('A0_fixed_time_condition', '固定时间变换条件', '#4477aa')]]
for i, (key, entry) in enumerate(curves.items()):
    x = np.array(entry['source_Q'])
    for j, panel in enumerate(panels):
        for stage, title, color in panel:
            axes[i, j].plot(x, entry['curves'][stage + '_clean_fixed126'], label=title, color=color, lw=1.8)
    for mode, title, color in [('clean_fixed126', '固定126窗、无ADC', '#009e73'),
            ('clean_dynamic', '动态寻峰窗、无ADC', '#4477aa'), ('ADC_dynamic', '动态窗＋ADC', '#cc3311')]:
        axes[i, 2].plot(x, entry['curves']['A0_' + mode], label=title, color=color, lw=1.8)
    for ax in axes[i]:
        ax.grid(alpha=.2)
        ax.set_xlabel('输入 SOURCE Q（V·ns）')
        ax.legend(fontsize=9)
    axes[i, 0].set_ylabel(f'源 PSD≈{float(key):.6f}，N={len(memberships[i])}\n输出 PSD 的分区中位数')
for ax, title in zip(axes[0], ['平均响应逐项加入（固定窗）', '能量条件路径对照（固定窗）', '同一 A0 波形的读出对照']):
    ax.set_title(title)
fig.suptitle('PSD 波浪来源：固定权重的组成与条件对照', fontsize=18)
fig.supxlabel('两条主要源形状分支；每条分支沿源 Q 分128个等人口片段。固定窗和条件冻结仅用于归因，未修改正式模型。', fontsize=11)
save(fig, 'COMB_CAUSAL_COMPONENTS')

fig, axes = plt.subplots(2, 4, figsize=(18, 8), layout='constrained', sharey=True)
chain = [('SOURCE', 'clean_dynamic', '原始 SOURCE'), ('filter_only', 'clean_dynamic', '仅滤波'),
    ('time_only', 'clean_dynamic', '滤波＋均值时间变换'), ('A0', 'clean_dynamic', '完整 A0 均值'),
    ('current_mean', 'clean_dynamic', '当前均值头'), ('current_z0', 'clean_dynamic', '当前响应 · z=0'),
    ('current_random', 'clean_dynamic', '加入随机响应'), ('current_plus_noise', 'ADC_dynamic', '再加电子学＋原ADC')]
colors = np.full(6465, '#999999', dtype=object)
colors[memberships[0]], colors[memberships[1]] = '#0077bb', '#ee7733'
for ax, (stage, mode, title) in zip(axes.flat, chain):
    x, y = a[f'{stage}_{mode}_Q'], a[f'{stage}_{mode}_PSD_raw']
    finite = np.isfinite(y)
    ax.scatter(x[finite], y[finite], c=colors[finite], s=2, alpha=.5, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('该阶段读出 Q（V·ns）')
    ax.grid(alpha=.15)
    ax.text(.02, .02, f'N={len(y)}；未定义={int((~finite).sum())}', transform=ax.transAxes, fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel('PSD = tail / Q')
fig.suptitle('同一组6,465个源父事件：波浪从哪一步出现', fontsize=18)
fig.supxlabel('蓝／橙：两条主要源形状分支；灰：其余源事件。每父事件一次；前七图无电子学、无ADC。Q未作物理能量标定。', fontsize=11)
save(fig, 'COMB_STAGE_CHAIN')
print(json.dumps(dict(dominant_rows=summary['dominant_group_rows'], other_rows=summary['other_rows_retained'],
    groups={k: {field: v[field] for field in ('rows', 'time_scale_range', 'normalized_source_shape_RMS_spread_quantiles',
        'fixed_window_TV_relative_to_A0', 'ADC_PSD_RMSE', 'dynamic_vs_fixed_PSD_RMSE', 'fixed_PSD_variance_decomposition')}
        for k, v in summary['groups'].items()})))
