from pathlib import Path
from datetime import datetime, timezone
import os, io, json
import numpy as np

r = Path('E:/GAN/research/N01')
h = Path(__file__).resolve().parent
p = json.loads((r/'configs/joint_response_01_energy_trend_origin_20261021.json').read_bytes())
terminal = json.loads((h/'terminal_01/TERMINAL_VERIFICATION.json').read_bytes())
result = json.loads((h/'terminal_01/RESULT.json').read_bytes())
assert terminal['native_exit']['exit_code'] == 0 and not terminal['processes']
for k, v in p['operation_budget'].items():
    assert result['execution_counts'].get(k, 0) == v
with np.load(io.BytesIO((Path(p['storage']['windows_root'])/'run_01/STAGE_QPSD_SCALARS.npz').read_bytes()), allow_pickle=False) as z:
    a = dict(z)
assert len(a['source_parent_rows']) == 6465
summary = {}
for stage in p['stages']:
    for mode in ('clean', 'ADC'):
        q, t, y = [a[f'{stage}_{mode}_{key}'] for key in ('Q', 'tail', 'PSD_raw')]
        np.testing.assert_allclose(y, t/q, rtol=1e-12, atol=1e-12)
        summary[stage+'_'+mode] = dict(rows=len(q), Q_mean=float(q.mean()), Q_min=float(q.min()), Q_max=float(q.max()),
            PSD_mean=float(y.mean()), PSD_min=float(y.min()), PSD_max=float(y.max()))
    summary[stage+'_ADC_effect'] = dict(PSD_RMSE=float(np.mean((a[stage+'_ADC_PSD_raw']-a[stage+'_clean_PSD_raw'])**2)**.5),
        peak_index_changes=int(np.sum(a[stage+'_ADC_peak_index'] != a[stage+'_clean_peak_index'])))
qsrc, ysrc = a['SOURCE_clean_Q'], a['SOURCE_clean_PSD_raw']
branches = {}
for value in (.207919414, .312575162):
    take = np.abs(ysrc-value) < 1e-6
    order = np.flatnonzero(take)[np.argsort(qsrc[take])]
    by = {}
    for stage in ('frozen_A0', 'latest_seed0', 'latest_seed1'):
        q = a[stage+'_clean_Q'][order]
        delta = np.diff(q)
        by[stage] = dict(negative_Q_step_fraction=float(np.mean(delta < -1e-5)),
            sum_negative_Q_steps=float(-delta[delta < -1e-5].sum()))
    branches[str(value)] = dict(rows=int(take.sum()), source_Q_min=float(qsrc[take].min()),
        source_Q_max=float(qsrc[take].max()), mappings=by)
receipt = dict(status='PASS_NO_TRAINING_ENERGY_TREND_ORIGIN_ARRAY_AUDIT', UTC=datetime.now(timezone.utc).isoformat(),
    native_exit=terminal['native_exit'], execution_counts=result['execution_counts'], summary=summary,
    source_template_branches=branches, new_candidates=0, updates=0, NAS_file_IO_checks=False,
    NAS_file_hash_verification=False, scientific_acceptance=False)
(h/'ORIGIN_ARRAY_AUDIT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
os.environ['MPLCONFIGDIR'] = str(r/'cache/full_energy_psd_plot_20261021')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
fig, axes = plt.subplots(2, 4, figsize=(17, 7.8), sharey=True)
for row, mode in enumerate(('clean', 'ADC')):
    for col, stage in enumerate(p['stages']):
        ax = axes[row, col]
        q, y = a[f'{stage}_{mode}_Q'], a[f'{stage}_{mode}_PSD_raw']
        ax.scatter(q, y, s=1.5, alpha=.45, color='#2e5190', rasterized=True)
        ax.set_ylim(-.15, .55)
        ax.set_xlabel('该阶段读出 Q / (V·ns)')
        if col == 0:
            ax.set_ylabel(('无 ADC、无电子学', '仅原 ADC、无电子学')[row]+'\nPSD = Qtail / Q')
        if row == 0:
            ax.set_title(('原 SOURCE', '冻结 A0 均值路径', '最新修正 seed 0 · z=0', '最新修正 seed 1 · z=0')[col])
fig.suptitle('PSD 条纹逐段定位：相同 6,465 个源父事件，无新增拟合', fontsize=17)
fig.tight_layout()
fig.savefig(r/'deliverables/joint_response01_full_energy_psd_20261021/ORIGIN_CHAIN.png', dpi=180)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
colors = ('#246a9b', '#c37332')
for row, stage in enumerate(('frozen_A0', 'latest_seed0')):
    for ib, value in enumerate((.207919414, .312575162)):
        take = np.abs(ysrc-value) < 1e-6
        for col, key in enumerate(('Q', 'PSD_raw')):
            axes[row, col].scatter(qsrc[take], a[f'{stage}_clean_{key}'][take], s=2, alpha=.4,
                color=colors[ib], rasterized=True, label=f'SOURCE PSD≈{value:.4f}')
    for col in range(2):
        axes[row, col].set_xlabel('SOURCE Q / (V·ns)')
        axes[row, col].set_ylabel(('冻结 A0', '最新 seed0')[row]+(' 的输出 Q' if col == 0 else ' 的 PSD'))
        axes[row, col].legend(markerscale=3)
fig.suptitle('冻结均值响应的能量映射：仅区分输入模板带，不赋予实测粒子真值标签', fontsize=15)
fig.tight_layout()
fig.savefig(r/'deliverables/joint_response01_full_energy_psd_20261021/ENERGY_RESPONSE_MAP.png', dpi=180)
plt.close(fig)
print(json.dumps(receipt))
