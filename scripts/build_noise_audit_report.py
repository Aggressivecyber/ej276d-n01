"""Render frozen N01 diagnostic outputs; does not fit or generate responses."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime/mpl_noise_audit'))
sys.path.insert(0, str(ROOT / 'runtime/plotting_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AUDIT = ROOT / 'runs/noise_audit_20261012'
OUT = ROOT / 'reports/noise_audit_20261012'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save(fig, name):
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=170, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def fmt(value, digits=5):
    return '未定义' if value is None or not np.isfinite(value) else f'{value:.{digits}g}'


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    windows = load(AUDIT / 'windows/WINDOW_DIAGNOSTIC_COMPLETE.json')
    controls = load(AUDIT / 'windows_stratified_control/WINDOW_DIAGNOSTIC_COMPLETE.json')
    projection = load(AUDIT / 'projection/PROJECTION_AUDIT.json')
    slope_root = AUDIT / 'slope/paired_cuda'
    slope = {arm: load(slope_root / arm / 'report.json') for arm in ('full', 'deterministic')}
    theory = load(slope_root / 'algebra_and_variance.json')
    slope_complete = load(slope_root / 'COMPLETE.json')
    view_names = ['full', 'baseline_raw', 'baseline_demean', 'baseline_detrend', 'pulse_demean', 'pulse_shape']
    labels = ['Full voltage (480)', 'Raw baseline (100)', 'Demeaned baseline (100)',
              'Detrended baseline (100)', 'Baseline-subtracted pulse (380)', 'Peak-normalized pulse (380)']
    zh = ['完整电压', '原始基线', '去均值基线', '去趋势基线', '减基线脉冲', '峰值归一化形状']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    colors = ['#126DA7', '#D66A29', '#248D79']
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    rows = []
    for i, view in enumerate(view_names):
        matched = windows['results'][view]['matched_SOURCE_REAL']
        null = controls['results'][view]['stratified_REAL_REAL']
        full = windows['results'][view]['all_SOURCE_REAL']
        rows.append(dict(view=view, label=zh[i], matched=matched, negative_control=null,
                         all_support=full, original_group_shift_control=windows['results'][view]['matched_REAL_REAL']))
        for j, (r, name) in enumerate(zip([matched, null, full], ['Generated-REAL: matched n', 'REAL-REAL: random labels within group', 'Generated-REAL: all saved rows'])):
            lo, hi = r['uncertainty']['raw']['percentile_95_interval']
            y = len(view_names)-1-i+(j-1)*.20
            ax.errorbar(r['AUC'], y, xerr=[[r['AUC']-lo], [hi-r['AUC']]], fmt='o', color=colors[j],
                        capsize=3, label=name if i == 0 else None)
    ax.axvline(.5, ls='--', color='#555555', lw=1)
    ax.set(yticks=np.arange(6), yticklabels=labels[::-1], xlim=(.30, 1.01),
           xlabel='Held-out domain ROC AUC (label direction retained)',
           title='Frozen-model window diagnostics: baseline and pulse gaps both remain')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), fontsize=9, frameon=True)
    ax.grid(axis='x', alpha=.18)
    save(fig, '01_window_classifiers')

    medians = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for j, arm in enumerate(('full', 'deterministic')):
        with np.load(slope_root / arm / 'per_parent_metrics.npz', allow_pickle=False) as z:
            medians[arm] = {}
            for col, metric in enumerate(('variance_Q', 'variance_PSD_raw')):
                on, off = z['on_'+metric], z['off_'+metric]
                median_on, median_off = float(np.median(on)), float(np.median(off))
                medians[arm][metric] = dict(on_median=median_on, off_median=median_off,
                    median_ratio_off_over_on=median_off/median_on,
                    mean_on=float(on.mean()), mean_off=float(off.mean()),
                    median_within_parent_ratio=float(np.median(off/on)))
                ax = axes[0, col]
                x = j*3+np.array([0,1])
                ax.bar(x, [median_on, median_off], color=[colors[0], colors[1]], width=.72)
                for xx, yy in zip(x, [median_on, median_off]):
                    ax.text(xx, yy*1.12, f'{yy:.4g}', ha='center', fontsize=10)
                ax.set_yscale('log')
                ax.set_xticks([0,.0+1,3,4], ['full\non', 'full\noff', 'det.\non', 'det.\noff'])
                ax.set_title('Median conditional '+('Q variance' if col == 0 else 'PSD variance'))
                ax.set_ylabel('(V ns)$^2$' if col == 0 else 'PSD$^2$')
                ax.grid(axis='y', alpha=.18)
    with np.load(slope_root/'full/paired_outputs.npz', allow_pickle=False) as z:
        on, off = z['on_baseline_mean_V']*1000, z['off_baseline_mean_V']*1000
    endpoint = ROOT/'runs/run_001_20261012/final_review_20261012_0900/train/full/evaluation/endpoint_populations.npz'
    with np.load(endpoint, allow_pickle=False) as z:
        real_mean = z['REAL_waveform'][:, :100].mean(1)*1000
    bins = np.linspace(-2, 2, 65)
    for values, name, c in [(on,'Original random slope',colors[0]), (off,'Random slope off',colors[1]),
                             (real_mean,'REAL (599)',colors[2])]:
        axes[1,0].hist(values, bins=bins, density=True, histtype='step', lw=1.6, label=name, color=c)
    axes[1,0].set(title='Baseline mean: slope removal leaves offset mismatch', xlabel='Mean of first 100 samples (mV)', ylabel='Density')
    axes[1,0].legend(fontsize=9)
    ax = axes[1,1]
    for label, name, c in [('on','Original random slope',colors[0]),('off','Random slope off',colors[1]),
                            ('REAL_saved_validation','REAL',colors[2])]:
        r=slope['full']['populations'][label]['baseline']
        ax.plot(r['ACF_lags'], r['ACF_mean'], marker='o', ms=3, label=name, color=c)
    ax.set(title='Mean per-event baseline ACF', xlabel='Lag (ns)', ylabel='ACF after event demeaning')
    ax.axhline(0, color='#555555', lw=.7); ax.legend(fontsize=9)
    save(fig, '02_paired_slope_intervention')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    with np.load(AUDIT/'projection/artificial_counterexample.npz', allow_pickle=False) as z:
        axes[0].plot(np.arange(480), z['original'][0]*1000, lw=1.4, label='Artificial original')
        axes[0].plot(np.arange(480), z['altered'][0]*1000, lw=1.1, alpha=.8, label='Fixed-projection null-space change')
    axes[0].axvline(180, ls='--', color='#555555', lw=1)
    axes[0].set(title='Artificial counterexample (not an N01 output)', xlabel='Time sample (1 ns)', ylabel='Voltage (mV)')
    axes[0].legend(fontsize=8)
    values=[projection['current_implementation_with_float32_input']['wave'],
            projection['independent_frozen_evaluation']['wave'], projection['mean_refreshed_wave_distance']]
    axes[1].bar(['Training fixed\ndirections', 'Independent\nfrozen directions', 'Refreshed banks\n(mean of four)'], values,
                color=[colors[0],colors[1],colors[2]])
    axes[1].set_yscale('log');axes[1].set(title='Same 1 mV RMS perturbation, different detection', ylabel='Standardized projected wave distance')
    for x,y in enumerate(values):axes[1].text(x,y*1.25,f'{y:.3g}',ha='center')
    axes[1].set_ylim(1e-7, 2);axes[1].grid(axis='y',alpha=.18)
    save(fig, '03_projection_counterexample')

    baseline={label: slope['full']['populations'][label]['baseline']['baseline_mean_V']['std_population']*1000
              for label in ('on','off','REAL_saved_validation')}
    conditional = load(slope_root / 'CONDITIONAL_PSD_WIDTHS.json')
    summary=dict(scope='Frozen N01 diagnosis; not a repaired or accepted model',response_model_training_updates=0,
        diagnostic_classifier_updates=4800, windows=rows, slope_conditional_medians_ddof1=medians,
        baseline_mean_std_mV=baseline, slope_source_unchanged=slope_complete['source_unchanged'],
        replay={a:r['old_replay'] for a,r in slope.items()}, projection={k:projection[k] for k in (
            'rank','nullity','maximum_absolute_perturbation_V','current_implementation_with_float32_input',
            'independent_frozen_evaluation','mean_refreshed_wave_distance','FIR_rho1_analysis')},
        next_action='Separate electronics calibration before any further pulse training',
        limitations=['One acquisition', 'Only five REAL held-out groups in window C2ST',
                    'Transform-specific C2ST is not original discriminator attribution',
                    'Removing a latent draw is an intervention, not a repaired model',
                    'Artificial projection nullspace does not prove model exploitation'])
    write(OUT/'REVIEW_SUMMARY.json',summary)
    classifier_rows=''.join(f'<tr><td>{r["label"]}</td><td>{r["matched"]["AUC"]:.3f}</td><td>{r["negative_control"]["AUC"]:.3f}</td><td>{r["all_support"]["AUC"]:.3f}</td></tr>' for r in rows)
    slope_rows=''.join(f'<tr><td>{a}</td><td>{fmt(v["variance_Q"]["on_median"])}</td><td>{fmt(v["variance_Q"]["off_median"])}</td><td>{fmt(v["variance_PSD_raw"]["on_median"])}</td><td>{fmt(v["variance_PSD_raw"]["off_median"])}</td></tr>' for a,v in medians.items())
    tp=theory['arms']['deterministic']['theory']
    title='N01 冻结模型机制复核'
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#152b3b;background:#f1f5f8;margin:0}}main{{max-width:1080px;margin:28px auto;padding:30px 42px;background:white}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:34px;border-bottom:2px solid #1678ac}}p{{margin:12px 0}}.lead{{border-left:5px solid #d66a29;padding:14px 20px;background:#fff4e8;font-size:18px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{padding:9px;border-bottom:1px solid #d4dfe6;text-align:left}}th{{background:#e8f2f9}}img{{display:block;width:100%;height:auto;margin:18px 0}}.meta{{color:#365266}}code{{background:#edf2f7;padding:2px 4px}}a{{color:#086b9f}}@media(max-width:700px){{main{{margin:0;padding:20px}}table{{font-size:13px}}}}</style>
<main><h1>{title}</h1><p class="meta">2026-10-12 · 冻结基准 2c501e9 · 新增响应模型训练更新 0 · 诊断分类器更新 4800</p>
<p class="lead">电子学统计分解与固定投影盲区均获复核。基线偏置足以形成域差异；去除该信息后，脉冲及归一化形状仍明显可分。当前结果支持先独立修复电子学，同时保留对脉冲模型和损失的检查。</p>
<h2>1. 分窗口分类器：基线与脉冲都存在差距</h2>
<p>下表SOURCE–REAL指full模型生成响应与实测的比较，不是清洁源波形与实测的比较。匹配样本量的两域各用128个拟合、96个测试事件；SOURCE每父只取一个预先固定响应。拟合/测试父事件和REAL原组隔离。补充阴性对照在同一角色的每个REAL组内随机分配标签，事件不复用；不确定度共同重采样原组。测试仅5个REAL组。完整支持列复用原保存总体和划分；三种列均为独立拟合的固定200更新分类器。</p>
<table><tr><th>视图</th><th>SOURCE–REAL 匹配样本 AUC</th><th>REAL–REAL 随机标签 AUC</th><th>SOURCE–REAL 全支持 AUC</th></tr>{classifier_rows}</table>
<p>AUC标签方向保留，接近0.5意味着本分类器难以区分。去均值/趋势后的弱区分不证明完整相关噪声已经匹配。独立窗口分类器定位可用的域信息，不证明原训练判别器实际使用了哪种特征；窗口长度不同时全连接层参数量也不同，不能把AUC差异视为严格等容量的因果分解。24个保存分类器的缩放、测试预测与AUC已重开核对。</p>
<img src="figures/01_window_classifiers.png" alt="分窗口分类器与阴性对照">
<p>原预定四向原组分域的REAL–REAL结果为0.397–0.423，原结果完整保留为组移位压力对照。上述组内随机标签控制是看到这个问题后的公开补充分析；六个区间均包含0.5，未重选网络、预算或种子。</p>
<h2>2. 仅关闭随机斜率：同噪声配对干预</h2>
<p>full和deterministic均固定128个父事件×64个噪声响应。恢复原CPU噪声流并保存全部显式随机数，只将<code>electronics_slope</code>置零，保留斜率均值、其它随机量、模型权重和ADC。下表为跨父事件的条件方差中位数，统一使用ddof=1。原审查full≈0.703来自ddof=0；同一原数组在ddof=1下为0.714665，不是不同训练结果。</p>
<table><tr><th>模型</th><th>Q方差：原斜率</th><th>Q方差：斜率关</th><th>PSD方差：原斜率</th><th>PSD方差：斜率关</th></tr>{slope_rows}</table>
<p>斜率的电荷系数为120000/479 = 250.52192067。冻结参数给出的随机斜率Q方差为{fmt(tp['Q_random_slope_variance'])}，滤波白噪声Q方差为{fmt(tp['Q_filtered_white_variance'])}（单位均为(V·ns)²）。配对方差差额同时含有限样本交叉协方差，不能直接把两个方差相减当作独立因果百分比。</p>
<p>基线均值标准差由{baseline['on']:.4f} mV变为{baseline['off']:.4f} mV；REAL为{baseline['REAL_saved_validation']:.4f} mV。关闭斜率后仍有偏置分解和相关结构残差。这是机制干预，不能作为修复模型发布。</p>
<p>确定性模型的条件Q方差中位数降低{100*(1-medians['deterministic']['variance_Q']['median_ratio_off_over_on']):.1f}%；full的条件PSD方差中位数降低{100*(1-medians['full']['variance_PSD_raw']['median_ratio_off_over_on']):.1f}%。这些是固定模型中的干预变化，未校准物理方差。</p>
<img src="figures/02_paired_slope_intervention.png" alt="同噪声斜率开关的条件方差与基线统计">
<p>另按原TRAIN的8个输出Q箱保存PSD标准差与描述性双高斯结果。关闭斜率后，full有{conditional['arms']['full']['rows_migrating_output_charge_bin']}/8192行迁移Q箱，确定性模型有{conditional['arms']['deterministic']['rows_migrating_output_charge_bin']}/8192行迁移；因此同时保留重新分箱与固定原箱成员两种口径。两高斯成分不等同于真实粒子标签，重复父事件也不等于8192个独立实测事件。</p>
<p>重放与量化差异、所有非正Q/PSD数值保护/饱和计数、preADC恒等式残差和每父指标保存在相邻机器可读报告中。所有输入行保留；跨设备位级重放仅按实际检查结果报告。</p>
<h2>3. 固定投影视角的人工盲区</h2>
<p>96条人工波形，仅修改第180点以后；每条全480点扰动RMS=1 mV，最大单点改变{projection['maximum_absolute_perturbation_V']*1000:.3f} mV。194条线性约束秩为{projection['rank']}，零空间维数为{projection['nullity']}。基线、峰位、Q和PSD保持到数值误差。当前float32实现的wave距离为{projection['current_implementation_with_float32_input']['wave']:.3g}；独立冻结方向库为{projection['independent_frozen_evaluation']['wave']:.4f}。</p>
<img src="figures/03_projection_counterexample.png" alt="人工固定投影盲区及独立方向检出">
<p>这复现了盲区机制，没有强求原人工包的完全相同数值。测试不含ADC、对抗项和正则项，也不证明旧模型利用了该零空间。受限FIR族的一阶总体自相关绝对值上确界为192/16705≈0.0114936，支持原审查给出的保守界。它单独针对白噪声经过受限平稳滤波器，不能套到叠加漂移并每事件去均值后的观测ACF。</p>
<h2>4. 下一修复边界</h2>
<p>保留U-Net主干，先实现平稳相关噪声＋事件基线偏置的独立候选；采用自由FIR系数与L2归一化，暂不默认加入逐事件随机斜率。完整基线联合验证未通过就停在噪声模块。随后刷新训练投影、复用一次双遍的方向，采用独立评价方向与逐坐标/短延迟检查，再做统一计数权重下必要的确定性和随机响应对照。</p>
<p>本轮未改响应模型、未训练新的噪声候选，也未启动脉冲重训。验证集已用于本次诊断，后续不能再称为未查看的独立终验。所有数据仍来自同次采集，真实物理权重与绝对硬件校准没有新增证据。</p>
<p><a href="REVIEW_SUMMARY.json">汇总数值</a> · <a href="WINDOW_DIAGNOSTIC_COMPLETE.json">预定窗口诊断</a> · <a href="STRATIFIED_CONTROL_COMPLETE.json">补充阴性对照</a> · <a href="CLASSIFIER_REOPEN_VERIFICATION.json">24个分类器重开核验</a> · <a href="SLOPE_full.json">full斜率干预</a> · <a href="SLOPE_deterministic.json">deterministic斜率干预</a> · <a href="CONDITIONAL_PSD_WIDTHS.json">输出Q分箱峰宽</a> · <a href="PROJECTION_AUDIT.json">投影与FIR核验</a></p></main></html>'''
    (OUT/'N01_NOISE_MECHANISM_REVIEW.html').write_text(document, encoding='utf-8')
    write(OUT/'WINDOW_DIAGNOSTIC_COMPLETE.json',windows)
    write(OUT/'STRATIFIED_CONTROL_COMPLETE.json',controls)
    write(OUT/'PROJECTION_AUDIT.json',projection)
    write(OUT/'SLOPE_ALGEBRA.json',theory)
    write(OUT/'CONDITIONAL_PSD_WIDTHS.json', conditional)
    write(OUT/'CLASSIFIER_REOPEN_VERIFICATION.json',load(AUDIT/'CLASSIFIER_REOPEN_VERIFICATION.json'))
    for arm,r in slope.items():write(OUT/f'SLOPE_{arm}.json',r)
    print(json.dumps(dict(output=str(OUT), figures=3, response_updates=0, classifier_updates=4800)))


if __name__ == '__main__':main()
