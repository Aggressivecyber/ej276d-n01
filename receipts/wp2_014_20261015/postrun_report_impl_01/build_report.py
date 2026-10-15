"""Build WP2/014 delivery from saved JSON only; no arrays, checkpoints or models."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
import os
import re
import sys

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reports/wp2_014_20261015'
SELECTED = ROOT / 'runs/wp2_014_20261015/selected_collected_01'
ART = SELECTED / 'runs/primary/artifacts'
READBACK = ROOT / 'runs/wp2_014_20261015/independent_readback_collected_01'
TMP = Path('E:/GAN/runtime_tmp/wp2_014_report')
for name in ['TEMP', 'TMP', 'TMPDIR']:
    os.environ[name] = str(TMP)
os.environ['MPLCONFIGDIR'] = str(TMP / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
TMP.mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

INPUTS = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(rel):
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    assert path.suffix == '.json', 'JSON-only numerical evidence'
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT)).replace('\\', '/')] = dict(bytes=len(data), sha256=digest(data))
    return json.loads(data)


def saved_text_evidence(rel):
    path = ROOT / rel
    assert path.suffix in {'.txt', '.md', '.py'}
    data = path.read_bytes()
    INPUTS[rel] = dict(bytes=len(data), sha256=digest(data))
    return data.decode('utf-8')


def write_new(path, data):
    with path.open('xb') as f:
        f.write(data if isinstance(data, bytes) else data.encode('utf-8'))


def csv_file(name, rows):
    with (OUT / name).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean2(values):
    assert len(values) == 2
    return (values[0] + values[1]) / 2


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def figure_outputs(fig, stem):
    fig.savefig(OUT / 'figures' / (stem + '.png'), dpi=200, facecolor='white')
    fig.savefig(OUT / 'figures' / (stem + '.svg'), facecolor='white', metadata={'Date': None})
    plt.close(fig)


def render_html(lines):
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    render = []
    i = 0
    section = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i])
                i += 1
            cells = lambda row: [x.strip() for x in row.strip('|').split('|')]
            render.append('<div class="tablewrap"><table><thead><tr>' + ''.join('<th>' + inline(x) + '</th>' for x in cells(block[0])) + '</tr></thead><tbody>')
            render.extend('<tr>' + ''.join('<td>' + inline(x) + '</td>' for x in cells(row)) + '</tr>' for row in block[2:])
            render.append('</tbody></table></div>')
            continue
        image = re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)', line)
        if image:
            render.append('<figure><img src="' + html.escape(image.group(2), quote=True) + '" alt="' + html.escape(image.group(1), quote=True) + '"></figure>')
        elif line.startswith('## '):
            section += 1
            render.append(f'<h2 id="section{section}">' + inline(line[3:]) + '</h2>')
        elif line.startswith('# '):
            render.append('<h1>' + inline(line[2:]) + '</h1>')
        elif line:
            render.append('<p>' + inline(line) + '</p>')
        i += 1
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>N01 WP2 014 损失表示对照报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf3f8;color:#172638;font:17px/1.7 "Microsoft YaHei",Arial,sans-serif}main{width:min(1320px,96%);margin:24px auto 80px;padding:38px 46px;background:white;border-top:7px solid #1763a8;box-shadow:0 6px 28px #18355012}h1{font-size:31px;line-height:1.4;margin:0 0 24px;color:#103e6b}h2{font-size:23px;margin:40px 0 18px;border-bottom:2px solid #e0eaf4;padding-bottom:9px;color:#174a78;scroll-margin-top:24px}p{margin:13px 0}strong{color:#0e4e7c}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin:16px 0}th{background:#eaf2f9;color:#16456d;text-align:left}td,th{padding:10px 11px;border:1px solid #dce5ed;vertical-align:top}tbody tr:nth-child(even){background:#f8fbfe}td:first-child{font-weight:600}code{font:13px/1.5 Consolas,monospace;background:#f0f4f7;padding:2px 4px;overflow-wrap:anywhere}a{color:#1263a3;text-decoration:none}.tablewrap{overflow-x:auto}figure{margin:24px 0 10px}figure img{display:block;width:100%;height:auto;border:1px solid #e3eaf1;padding:8px;background:white}@media print{body{background:white;font-size:11pt}main{width:100%;margin:0;padding:10px;box-shadow:none}h2{break-after:avoid}tr,figure{break-inside:avoid}table{font-size:9pt}}
</style><main>''' + ''.join(render) + '</main></html>\n'


def main():
    assert not OUT.exists(), 'Preserve earlier report outputs'
    OUT.mkdir(parents=True)
    (OUT / 'figures').mkdir()
    cfg = read_json('wp2_014_config.json')
    tolerance = read_json('CALIBRATED_TOLERANCES_WP1_013.json')
    selection = read_json('receipts/wp2_014_20261015/postrun_engineering_01/DEVELOPMENT_SELECTION.json')
    final_selection = read_json('receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json')
    assert final_selection['development_path'] == 'C1' and final_selection['independent_raw_readback_complete'] is True
    assert final_selection['candidates'] == selection['candidates'] and final_selection['cells'] == selection['cells']
    local = read_json('receipts/wp2_014_20261015/postrun_independent_01/LOCAL_METADATA_READBACK.json')
    closeout = read_json('receipts/wp2_014_20261015/postrun_independent_02/CLOSEOUT_REVIEW.json')
    array = read_json(READBACK / 'artifacts/ARRAY_READBACK.json')
    raw_exit = read_json(READBACK / 'supervisor/exit_receipt.json')
    native = read_json(SELECTED / 'runs/primary/supervisor/exit_receipt.json')
    fit = read_json(ART / 'INTEGRAL_TRAIN_STATISTICS.json')
    science_collection = read_json('receipts/wp2_014_20261015/collection_01/COLLECTION_RECEIPT.json')
    raw_collection = read_json('receipts/wp2_014_20261015/readback_collection_01/COLLECTION_RECEIPT.json')
    full_inventory = read_json('receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json')
    information = read_json('INPUT_INFORMATION_AUDIT.json')
    for rel in ['USER_MASTER_PLAN_20261015.txt', 'docs/WP2_ACCEPTANCE_CONTRACT.md',
                'docs/WP2_014_CONFIGURATION.md', 'receipts/wp2_014_20261015/postrun_independent_01/READBACK_SCOPE_NOTE.json']:
        if rel.endswith('.json'):
            read_json(rel)
        else:
            saved_text_evidence(rel)
    read_json('WP2_DATA_ROLES.json')
    read_json('OBSERVATION_CONTRACT.json')
    read_json('WP2_SOURCE_MANIFEST.json')
    assert closeout['passed'] and array['passed'] and native['exit_code'] == raw_exit['exit_code'] == 0
    assert local['scientific_acceptance'] is False and selection['weights_promoted'] is False
    assert fit['fit_rows'] == 21047 and fit['all_rows_retained'] and not fit['candidate_outputs_used']
    hypotheses = ['fast', 'slow', 'common_level']
    T = tolerance['T_by_hypothesis']
    cells = []
    components = []
    for candidate in ['C0', 'C1']:
        for seed in [0, 1]:
            trial = f'seed{seed}_{candidate}'
            initial = read_json(ART / trial / 'initial_evaluation/METRICS.json')
            final = read_json(ART / trial / 'final_evaluation/METRICS.json')
            for snapshot in ['initial_evaluation', 'final_evaluation']:
                key = f'runs/primary/artifacts/{trial}/{snapshot}/METRICS.json'
                evidence = full_inventory['whole_remote_inventory'][key]
                assert INPUTS[str((ART / trial / snapshot / 'METRICS.json').relative_to(ROOT)).replace('\\', '/')]['sha256'] == evidence['sha256']
            for hyp in hypotheses:
                by_arm = {}
                for arm, doc in [('A', initial), ('initial', initial), ('final', final)]:
                    banks = [doc['metrics'][f'bank{b}_{arm}_{hyp}'] for b in [0, 1]]
                    assert all(b['coverage']['conditional_coverage_complete'] and b['coverage']['conditional_bins_used'] == 8 for b in banks)
                    by_arm[arm] = {k: mean2([b['values'][k] for b in banks]) for k in banks[0]['values']}
                    by_arm[arm + '_banks'] = [b['values'] for b in banks]
                A, ini, fin = (by_arm[x] for x in ['A', 'initial', 'final'])
                improvement = (A['wave'] - fin['wave']) / A['wave']
                row = dict(candidate=candidate, seed=seed, hypothesis=hyp, A_wave=A['wave'], initial_wave=ini['wave'],
                           final_wave=fin['wave'], relative_wave_improvement=improvement,
                           wave_improvement_percent=100 * improvement, usefulness_at_least_2percent=improvement >= .02,
                           final_bank0_wave=by_arm['final_banks'][0]['wave'], final_bank1_wave=by_arm['final_banks'][1]['wave'],
                           A=A, initial=ini, final=fin)
                cells.append(row)
                for metric, threshold in T[hyp].items():
                    calibrated = isinstance(threshold, (int, float)) and math.isfinite(threshold) and threshold > 0
                    components.append(dict(candidate=candidate, seed=seed, hypothesis=hyp, component=metric,
                        A=A[metric], initial=ini[metric], final=fin[metric], T=threshold,
                        final_minus_A=fin[metric]-A[metric], final_over_T=fin[metric]/threshold if calibrated else None,
                        noninferiority='PASS' if calibrated and fin[metric]-A[metric] <= threshold else 'FAIL' if calibrated else 'pending_not_calibrated',
                        absolute='PASS' if calibrated and fin[metric] <= threshold else 'FAIL' if calibrated else 'pending_not_calibrated'))
    selected = closeout['development_selection']
    assert selected['development_codepath'] == 'C1'
    assert sum(r['usefulness_at_least_2percent'] for r in cells if r['candidate'] == 'C1') == 5
    seed1_fast = next(r for r in cells if (r['candidate'], r['seed'], r['hypothesis']) == ('C1', 1, 'fast'))
    assert seed1_fast['relative_wave_improvement'] < .02
    gradient_rows = []
    for trial, training in selection['training_and_conditional_gradient_summary'].items():
        for probe in training['conditional_gradient_diagnostics']:
            for row in probe['strata']:
                ratio = row['weighted_distribution_to_wave_parameter_gradient_norm_ratio']
                cosine = row['wave_distribution_cosine']
                gradient_rows.append(dict(trial=trial, G_updates=probe['G_updates'], stratum=row['stratum'],
                    cells=row['cells'], computed_cells=row['computed_cells'], AD_queries=row['AD_queries'],
                    parameter_gradient_distribution_to_wave_median=ratio['median'],
                    ratio_min=ratio['minimum'], ratio_max=ratio['maximum'],
                    parameter_gradient_cosine_median=cosine['median'], cosine_min=cosine['minimum'], cosine_max=cosine['maximum'],
                    peak_switch_vs_initial_row_occurrences=row['peak_switch_rows_sum'],
                    Q_nonpositive_row_occurrences=row['Q_nonpositive_rows_sum'],
                    PSD_undefined_row_occurrences=row['PSD_undefined_rows_sum'],
                    additive_event_contribution=False))
    csv_file('WP2_ALL6_WAVE.csv', [{k: v for k, v in row.items() if k not in ['A', 'initial', 'final']} for row in cells])
    csv_file('WP2_ALL6_COMPONENTS.csv', components)
    csv_file('WP2_GRADIENT_STRATA.csv', gradient_rows)
    csv_file('WP2_R8_MOMENTS.csv', [dict(trial=p['trial'], snapshot=p['snapshot'], arm=p['arm'], hypothesis=p['hypothesis'],
        parents=p['parents'], draws=8, denominator=7, sqrt_mean_sample_variance_V=p['hard_wave_sqrt_mean_sample_variance_V'],
        conditional_mean_MCSE_RMS_V=p['hard_wave_RMS_conditional_mean_MCSE_V'],
        scalar_covariance_SHA256=p['r8_scalar_covariance_sha256'], waveform_factor_SHA256=p['centered_wave_factor_sha256']) for p in array['pairs']])

    # Two additional suites use already saved REPORT JSON; no metric replay.
    conditional_rows, source_rows = [], []
    conditional_features = ['Q_Vns', 'front_Vns', 'tail_Vns', 'PSD_raw', 'PSD_guarded',
                            'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns']
    for pair in array['pairs']:
        trial, snapshot, arm, hyp = (pair[k] for k in ['trial', 'snapshot', 'arm', 'hypothesis'])
        for bank in [0, 1]:
            report_path = ART / trial / snapshot / f'bank{bank}_{arm}_{hyp}_REPORT.json'
            report = read_json(report_path)
            key = str(report_path.relative_to(SELECTED)).replace('\\', '/')
            record = full_inventory['whole_remote_inventory'][key]
            assert INPUTS[str(report_path.relative_to(ROOT)).replace('\\', '/')]['sha256'] == record['sha256']
            conditions = report['output_Q_conditions']
            assert conditions['minimum_support'] == 16 and conditions['all_rows_in_exactly_one_bin']
            assert sum(b['fake_rows'] for b in conditions['bins']) == 26372
            assert sum(b['real_rows'] for b in conditions['bins']) == 21047
            for block in conditions['bins']:
                row = dict(trial=trial, candidate=trial.split('_')[1], seed=int(trial[4]), snapshot=snapshot,
                    arm=arm, hypothesis=hyp, bank=bank, output_Q_bin=block['bin'],
                    fake_rows=block['fake_rows'], real_rows=block['real_rows'], supported=block['supported'],
                    PSD_W1_defined=block['defined_PSD_W1']['value'], conditional_T='not_calibrated')
                for feature in conditional_features:
                    for domain in ['fake', 'real']:
                        stats = block[domain][feature]
                        for statistic in ['defined_rows', 'undefined_rows', 'mean', 'std_population']:
                            row[f'{domain}_{feature}_{statistic}'] = stats[statistic]
                row['rise_mean_difference_ns'] = row['fake_rise10_90_ns_mean'] - row['real_rise10_90_ns_mean']
                row['FWHM_mean_difference_ns'] = row['fake_FWHM_ns_mean'] - row['real_FWHM_ns_mean']
                conditional_rows.append(row)
            trends = report['source_input_trends']
            assert trends['parent_count'] == 6593 and trends['descriptors_are_audit_only_not_network_inputs']
            assert trends['unpaired_REAL_conditional_target'] == 'unavailable'
            for descriptor, responses in trends['correlations'].items():
                for response_statistic, values in responses.items():
                    for response, correlation in values.items():
                        source_rows.append(dict(trial=trial, candidate=trial.split('_')[1], seed=int(trial[4]),
                            snapshot=snapshot, arm=arm, hypothesis=hyp, bank=bank, descriptor=descriptor,
                            response_statistic=response_statistic, response_feature=response,
                            Pearson=correlation['value'], defined_parents=correlation['parents'],
                            kind=correlation['kind'], physical_conditional_response_identified=False))
    assert len(conditional_rows) == 72 * 8 and len(source_rows) == 72 * 18 * 2 * 4
    assert all(r['supported'] and r['fake_rows'] >= 16 and r['real_rows'] >= 16 for r in conditional_rows)
    csv_file('WP2_OUTPUT_Q_CONDITIONS.csv', conditional_rows)
    csv_file('WP2_SOURCE_INPUT_TRENDS.csv', source_rows)
    conditional_summary, source_summary = [], []
    for cell in cells:
        trial = f"seed{cell['seed']}_{cell['candidate']}"
        group = [r for r in conditional_rows if r['trial'] == trial and r['arm'] == 'final' and r['hypothesis'] == cell['hypothesis']]
        assert len(group) == 16
        worst_PSD = max(group, key=lambda r:r['PSD_W1_defined'])
        worst_rise = max(group, key=lambda r:abs(r['rise_mean_difference_ns']))
        worst_FWHM = max(group, key=lambda r:abs(r['FWHM_mean_difference_ns']))
        conditional_summary.append(dict(trial=trial, hypothesis=cell['hypothesis'],
            supported=sum(r['supported'] for r in group), minimum_fake_bin_rows=min(r['fake_rows'] for r in group),
            minimum_real_bin_rows=min(r['real_rows'] for r in group),
            worst_PSD={k:worst_PSD[k] for k in ['bank','output_Q_bin','PSD_W1_defined']},
            worst_rise={k:worst_rise[k] for k in ['bank','output_Q_bin','rise_mean_difference_ns']},
            worst_FWHM={k:worst_FWHM[k] for k in ['bank','output_Q_bin','FWHM_mean_difference_ns']}))
        summary = dict(trial=trial, hypothesis=cell['hypothesis'])
        for descriptor, statistic, feature, label in [
            ('canonical_Q_Vns','response_mean','Q_Vns','inputQ_meanQ'),
            ('canonical_Q_Vns','response_variance','Q_Vns','inputQ_varianceQ'),
            ('shape_PC_00','response_mean','Q_Vns','PC0_meanQ'),
            ('shape_PC_00','response_mean','tail_Vns','PC0_meanTail')]:
            selected_source = [r for r in source_rows if r['trial'] == trial and r['arm'] == 'final'
                and r['hypothesis'] == cell['hypothesis'] and r['descriptor'] == descriptor
                and r['response_statistic'] == statistic and r['response_feature'] == feature]
            assert len(selected_source) == 2
            summary[label] = mean2([r['Pearson'] for r in selected_source])
        source_summary.append(summary)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'N01-WP2-014', 'axes.axisbelow': True})
    colors = {'C0': '#2866a6', 'C1': '#d07532'}
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(11.8, 5.1), layout='constrained')
    for i, candidate in enumerate(['C0', 'C1']):
        values = [r['wave_improvement_percent'] for r in cells if r['candidate'] == candidate]
        bars = ax.bar(x + (i-.5)*.35, values, width=.33, color=colors[candidate], label=candidate)
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in values], fontsize=9, padding=4)
    ax.axhline(2, color='#b62d36', linestyle='--', linewidth=1.5, label='Predeclared usefulness gate: 2% in ALL6')
    ax.set_xticks(x, ['S0 fast', 'S0 slow', 'S0 common', 'S1 fast', 'S1 slow', 'S1 common'])
    ax.set_ylabel('Wave-distance improvement over paired A (%)')
    ax.set_title('WP2/014: C1 improves all six cells; the ALL6 usefulness gate fails')
    ax.set_ylim(0, 3.8)
    ax.grid(axis='y', color='#dbe3ec')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.annotate('1.988778858954% < 2%', xy=(3+.175, seed1_fast['wave_improvement_percent']),
                xytext=(3.5, 3.40), arrowprops=dict(arrowstyle='->', color='#b62d36'), color='#a2212b', fontsize=10)
    figure_outputs(fig, '01_wave_usefulness')

    metric_keys = ['wave', 'QPSD', 'noise', 'peak_voltage_V', 'rise10_90_ns', 'FWHM_ns', 'baseline_mean_V']
    metric_labels = ['Wave', 'QPSD', 'Noise', 'Peak', 'Rise', 'FWHM', 'Baseline DC']
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), layout='constrained')
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=6)
    for ax, candidate in zip(axes, ['C0', 'C1']):
        rows = [r for r in cells if r['candidate'] == candidate]
        ratios = np.array([[r['final'][key] / T[r['hypothesis']][key] for key in metric_keys] for r in rows])
        image = ax.imshow(np.log2(ratios), cmap='RdBu_r', norm=norm, aspect='auto')
        for row_index in range(6):
            for col_index in range(len(metric_keys)):
                value = ratios[row_index, col_index]
                label = f'{value:.3f}' if .9 < value < 1.1 else f'{value:.2f}'
                ax.text(col_index, row_index, label, va='center', ha='center',
                        color='white' if abs(math.log2(value)) > 3 else '#172638', fontsize=10)
        ax.set_xticks(np.arange(len(metric_keys)), metric_labels)
        ax.set_yticks(np.arange(6), ['S0 fast', 'S0 slow', 'S0 common', 'S1 fast', 'S1 slow', 'S1 common'])
        ax.set_title(f'{candidate}: absolute final hard-distance / frozen T')
    cbar = fig.colorbar(image, ax=axes, pad=.02, fraction=.025, ticks=[-4, -2, 0, 2, 4, 6])
    cbar.ax.set_yticklabels(['1/16', '1/4', '1', '4', '16', '64'])
    cbar.set_label('Distance / T (log2 color scale; >1 fails)')
    figure_outputs(fig, '02_absolute_tolerance')

    names = ['all'] + [f'SOURCE_Q_quartile_{i}' for i in range(4)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), layout='constrained')
    for candidate in ['C0', 'C1']:
        for seed in [0, 1]:
            rows = {r['stratum']: r for r in gradient_rows if r['trial'] == f'seed{seed}_{candidate}' and r['G_updates'] == 2000}
            style = '-' if seed == 0 else '--'
            axes[0].plot(range(5), [rows[n]['parameter_gradient_distribution_to_wave_median'] for n in names],
                         color=colors[candidate], linestyle=style, marker='o', label=f'{candidate}, seed {seed}')
            axes[1].plot(range(5), [rows[n]['parameter_gradient_cosine_median'] for n in names],
                         color=colors[candidate], linestyle=style, marker='o')
    for ax in axes:
        ax.set_xticks(range(5), ['All', 'Source Q1', 'Source Q2', 'Source Q3', 'Source Q4'])
        ax.grid(color='#e0e7ee', linewidth=.6)
        ax.tick_params(axis='x', labelrotation=18)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Median ||g_distribution|| / ||g_wave||')
    axes[0].set_title('Weighted parameter-gradient norm ratio at G2000')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_ylabel('Median cosine(g_wave, g_distribution)')
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].axhline(0, color='#707b86', linestyle=':', linewidth=1)
    axes[1].set_title('Alignment varies by seed and conditional subset')
    figure_outputs(fig, '03_conditional_gradients')

    lines = []
    def para(text):
        lines.extend([text, ''])
    def heading(text):
        para('## ' + text)
    def table(headers, rows):
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join('---' for _ in headers) + '|')
        lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        lines.append('')
    f9 = lambda x: f'{x:.9f}'
    para('# N01 WP2 / 014：损失表示对照与独立读回报告')
    para('日期：2026-10-15（Asia/Singapore）。按六工作包主实施计划完成 C0/C1 两种子、四次各2000G的固定预算对照；本报告数据来自保存JSON与已完成的独立保存数组读回。')
    para('**结果：选择C1作为后续开发代码路径；两候选的当前权重均未晋级。** C1的12项最终wave距离均值为0.15042006897712382，低于C0的0.15255321144257886。C1通过已冻结的ALL6关键组件非劣性规则，用途改善达到5/6格；seed1-fast为1.9887788589535327%，未达到2%门槛。形态与基线均值的绝对T门仍未通过，科学状态保持 `response_match_not_established`。')
    heading('1. 固定实验与数据角色')
    table(['项目', '实际范围', '比较含义'], [
        ['训练', 'seed0 C0→C1，seed1 C0→C1；每次2000G，共8000G', '四次按固定顺序串行完成，C2关闭'],
        ['模型', 'A(det19k)及005冻结；同结构8维随机形状头，1288可训练参数', 'C0/C1单因素为分布损失表示'],
        ['每步训练', '512 SOURCE父×2draw；1024 REAL行；三电子学假设', '全部合法TRAIN循环，SOURCE6593/REAL21047均覆盖'],
        ['完整硬评价', '6593 SOURCE父×4draw×2库；REAL全部21047行；三假设', '每格先平均两库各自距离，父事件是重复响应的源单位'],
        ['梯度探针', 'G0/G2000；64父×2draw×2库×3假设；13类子集', '每个子集对固定128条全局REAL参考计算条件经验损失'],
        ['独立保存数组审核', '36 pair、176 NPZ、1898784硬波形行、8梯度快照', '176个原NPZ留在服务器，本地回收审核结果19文件']])
    para('本轮C0沿用010的原损失代码，并重新初始化、校准随机头；C1在同seed复用同一校准初始化。它们均不以010的G2000头作为本轮初始权重。128 TRAIN父×4draw×2库的校准RMS分别为0.9994843046300347和0.9999251348242464 LSB，采用预登记16次lower-bracket二分；0.05相对幅度增益和头容量保持相同。')
    para('硬观察保持14 bit、384点载体、原100点基线、first-argmax搜索[112,144)、p−16:p+240裁窗、256点输出及tail起点46，时间单位为1 ns软件采样间隔。REAL来自已确认的单次采集、422个旧逻辑组；旧DEV已经暴露。本轮响应训练、损失拟合与候选比较均使用TRAIN。独立跨采集物理复用仍待确认。')
    heading('2. C1改变的目标与配对控制')
    para('C0训练目标保留 wave + 2×原QPSD + 0.01×regular；C1将其中的分布项替换为无分母积分距离，wave与regular及其权重保持相同。六维特征为Q、T及[0,16)、[16,46)、[46,96)、[96,256)四段积分。每维按 `max(REAL TRAIN median(abs(integral)), ADC_LSB×sqrt(segment_length))` 固定尺度做signed-asinh变换，距离为0.5×逐坐标经验W1加0.5×16方向联合投影W1，外层权重为2。全部有限行保留，非正Q不触发删除。')
    para('积分统计在候选输出之前仅拟合一次，覆盖21047条REAL TRAIN；六个尺度为' + '、'.join(f'{v:.9g}' for v in fit['scale_Vns']) + ' V·软件ns。统计JSON的SHA256为 `' + INPUTS[str((ART / 'INTEGRAL_TRAIN_STATISTICS.json').relative_to(ROOT)).replace('\\', '/')]['sha256'] + '`。硬评价继续使用原native QPSD、raw/guarded PSD、形态和基线指标。训练日志中C1的distribution/QPSD别名表示积分目标，本文表格的QPSD均取硬评价native指标。')
    para('每seed的C0/C1初始头哈希一致、初始完整人口指标一致；全部21个记录批次的SOURCE父序列与wave投影上下文相同。SOURCE/REAL、shape/005/extension和投影使用配对的显式流地址；配置指纹是流地址证据。原生冻结参数/缓冲守卫通过，校准初始状态和配对日志的独立核验以保存哈希与JSON为依据。')
    heading('3. 全六格wave：C1更低，用途门仍为5/6')
    table(['候选', 'seed', '假设', 'A', '共同初始', 'final', '相对A改善/%', '≥2%用途门'], [
        [r['candidate'], r['seed'], r['hypothesis'], f9(r['A_wave']), f9(r['initial_wave']), f9(r['final_wave']),
         f"{r['wave_improvement_percent']:.12f}", 'PASS' if r['usefulness_at_least_2percent'] else 'FAIL'] for r in cells])
    para('判定使用保存的未舍入浮点数。C1的seed1-fast改善为1.9887788589535327%，与2%门槛差−0.011221141046467362个百分点；该格保留FAIL。C0的6格用途门全部失败，C1为5格通过。表内两库均值是两个独立经验距离的均值，合并8draw的经验距离未用于这些判定。')
    para('![Unrounded usefulness comparison](figures/01_wave_usefulness.svg)')
    para('图1：英文图按两个种子×三假设展示wave改善，虚线为预登记2%用途门。精确数值另存 [ALL6 wave表](WP2_ALL6_WAVE.csv)。')
    table(['候选', 'ALL6正T组件非劣', 'mean12 final wave', '用途门', '绝对T全过', '开发选择', '权重/科学验收'], [
        [c, 'PASS（每格15项，共90项）', repr(selected['candidates'][c]['mean12_final_wave']),
         f"{selected['candidates'][c]['useful_cells']}/6", 'FAIL', 'C1' if c == 'C1' else '—', 'false / false'] for c in ['C0', 'C1']])
    para('开发选择按已冻结规则执行：同一候选全部6格、每个正且有定义T组件的 `mean2banks(final−A)≤T`，再比较全部12项final wave均值；绝对差≤1e−12时选C0。C0/C1均eligible，C1取得较低分数，因此开发代码路径选择C1。用途、绝对容忍度及科学确认分别保留自己的判定。')
    heading('4. 原QPSD、噪声和绝对T缺口')
    table(['候选', 'seed', '假设', 'A QPSD', 'initial QPSD', 'final QPSD', 'final相对A变化/%', 'noise'], [
        [r['candidate'], r['seed'], r['hypothesis'], f9(r['A']['QPSD']), f9(r['initial']['QPSD']), f9(r['final']['QPSD']),
         f"{100*(r['final']['QPSD']/r['A']['QPSD']-1):+.5f}", f9(r['final']['noise'])] for r in cells])
    para('QPSD距离越小越好。C1的slow QPSD较A增加，wave收益与原比值/条件PSD目标存在具体取舍；逐格数值完整保留。noise_prefix100在所有候选、种子和假设下均为0.124825000，形状头的原100点前缀保持冻结基线噪声。')
    table(['候选', 'seed', '假设', '峰幅D/T', '上升D/T', 'FWHM D/T', '基线DC D/T', '上升未定义率差D/T'], [
        [r['candidate'], r['seed'], r['hypothesis'], *[f"{r['final'][key]/T[r['hypothesis']][key]:.4f}" for key in
         ['peak_voltage_V', 'rise10_90_ns', 'FWHM_ns', 'baseline_mean_V', 'rise_undefined_rate']]] for r in cells])
    para('![Absolute tolerance comparison](figures/02_absolute_tolerance.svg)')
    para('图2：最终硬评价距离除以WP1冻结T；红色区域大于1。两候选的峰幅、上升时间、FWHM与基线DC绝对距离持续超过T，基线DC约33.27T。T来自预登记gain/tail/电子学幅度/展宽的人工自距离，作为本计划的规范性灵敏度标尺，`hardware_calibrated=false`。')
    uncalibrated = [k for k, v in T['fast'].items() if not isinstance(v, (int, float)) or v <= 0]
    table(['未标定组件', '本轮状态'], [[name, 'pending_not_calibrated；零/未定义T不自动PASS'] for name in uncalibrated])
    para('Q_nonpositive、PSD undefined/guarded等已记录的零率差仍保留上述pending角色。所有低Q、非正Q与失效路径按原合同保留；已完成库中的零发生数不替代该路径的物理验收。全部组件的A/initial/final、final−A、T及绝对判定见 [完整组件表](WP2_ALL6_COMPONENTS.csv)。')
    heading('5. 输出Q条件：保留每格最差库与箱')
    para('输出Q条件沿用原TRAIN统计的8个logQ箱；每行按观测Q落入且仅落入一箱，支持门为fake与REAL各至少16行。72套REPORT中的576个库×臂×假设×箱均有支持。箱边界按保存REPORT中的冻结数组指针引用，本报告未读取边界NPZ；输出Q不是能量或粒子真值。')
    def worst_label(block, field, digits):
        return f"{block[field]:.{digits}f}（b{block['bank']}/q{block['output_Q_bin']}）"
    table(['trial', '假设', '支持箱', '最小fake/REAL箱行数', '最大PSD W1（库/箱）', '最大绝对rise均值差/ns', '最大绝对FWHM均值差/ns'], [
        [r['trial'], r['hypothesis'], f"{r['supported']}/16", f"{r['minimum_fake_bin_rows']}/{r['minimum_real_bin_rows']}",
         worst_label(r['worst_PSD'],'PSD_W1_defined',6),
         worst_label(dict(r['worst_rise'], rise_mean_difference_ns=abs(r['worst_rise']['rise_mean_difference_ns'])),'rise_mean_difference_ns',4),
         worst_label(dict(r['worst_FWHM'], FWHM_mean_difference_ns=abs(r['worst_FWHM']['FWHM_mean_difference_ns'])),'FWHM_mean_difference_ns',4)]
        for r in conditional_summary])
    para('表内逐一保留两候选×两seed×三假设的final结果；每列选择该格两库×8箱中的实际最差值，列间可能来自不同箱。PSD W1为保存producer距离，rise/FWHM为保存条件均值的差；它们属于不同统计量。条件T未标定，这些数值只作描述，不套用全人口T作条件PASS判定。')
    para('[输出Q条件完整表](WP2_OUTPUT_Q_CONDITIONS.csv)保留A/initial/final共576条箱记录，包含fake/REAL行数与支持、PSD W1及Q/front/tail/PSD/峰幅/rise/FWHM的均值、总体标准差和定义/未定义计数；各seed、hypothesis和bank均单列。')
    heading('6. SOURCE输入趋势与输入信息边界')
    table(['trial', '假设', 'r(inputQ,父meanQ)', 'r(inputQ,父varQ)', 'r(PC0,父meanQ)', 'r(PC0,父meanTail)'], [
        [r['trial'],r['hypothesis'],*[f"{r[key]:+.6f}" for key in ['inputQ_meanQ','inputQ_varianceQ','PC0_meanQ','PC0_meanTail']]]
        for r in source_summary])
    para('表中为每库4draw父统计上的描述性Pearson，再算两个库r值的算术均值；全部12个候选×seed×假设格均列出。canonical-Q、384点幅度和16个形状PC共18个审计描述量，对Q/tail/峰幅/基线RMS的父mean与variance的每库结果，共10368条，保存于 [SOURCE趋势完整表](WP2_SOURCE_INPUT_TRENDS.csv)。这些描述量未新增为网络输入。')
    pca = information['PCA']['TRAIN_carrier384_canonical_Q']
    contrast = information['same_Q_real_input_shape_feasibility']
    para('原WP1信息审计显示，TRAIN canonical-Q归一化384点SOURCE形状的95%、99%、99.9%解释方差维数均为1，总方差约' + f"{pca['total_variance']:.9g}" + '。它来自已登记的EE/hadron双模板混合；高阶数值PC继续作为审计项保存，不把float32舍入图样解释为新的物理自由度。')
    para('保存的全8192父同原角色输入普查，在相对Q差≤1%、480点峰归一化形状RMS差>0.001的规则下找到' + str(contrast['rows_with_testable_shape_contrast']) + '个有可测形状对比的父，最大强度' + f"{contrast['largest_verified_strength']:.8f}" + '。这是旧SOURCE输入可区分性的证据；本轮未新增同Q配对响应干预。SOURCE到生成响应的相关性描述当前生成器的输入依赖；未配对REAL没有相应父真值，真实条件均值和物理源权重继续保持未识别/unknown。')
    heading('7. 两种子的描述性差异')
    table(['候选', '假设', 'seed1−seed0 final wave', 'seed1−seed0 final QPSD', 'seed1−seed0 final峰幅', 'seed1−seed0 final FWHM'], [
        [row['candidate'], row['hypothesis'], *[f"{row['seed1_minus_seed0_final_distance'][key]:+.9g}" for key in
          ['wave', 'QPSD', 'peak_voltage_V', 'FWHM_ns']]] for row in selection['seed_differences']])
    para('两候选的seed1 final wave均高于seed0；C1在fast的种子差使该格落到2%线下。这些是两个拟合种子的实际差值，本轮没有追加bootstrap、置信区间或近阈值补训。两库重复控制Monte Carlo变化，统计源单位仍是6593个SOURCE父。')
    heading('8. 分层梯度：强度与方向，按子集解释')
    all_gradient = [r for r in gradient_rows if r['stratum'] == 'all']
    table(['trial', 'G', '分布/wave范数比中位数', '六格范围', 'wave/分布cos中位数', '峰位较自身初始变化/768次'], [
        [r['trial'], r['G_updates'], f"{r['parameter_gradient_distribution_to_wave_median']:.5f}",
         f"{r['ratio_min']:.4f}..{r['ratio_max']:.4f}", f"{r['parameter_gradient_cosine_median']:+.5f}",
         r['peak_switch_vs_initial_row_occurrences']] for r in all_gradient])
    para('![Conditional parameter gradient diagnostics](figures/03_conditional_gradients.svg)')
    para('图3：G2000固定探针中，按all及SOURCE Q四分位展示六个库×假设格的中位数。C1的加权分布/wave参数梯度范数比低于C0；wave与分布梯度的夹角随种子和子集变化，C1 seed0的all中位cos为负而seed1为正。')
    para('13类子集包括all、SOURCE Q四分位、生成输出Q四分位和4个预指定单父位置。每个子集均针对同一全局128行REAL参考计算非加性经验损失；范数比是梯度强度比，子集梯度不能相加解释为事件贡献、损失比例或因果占比。weighted total向量由wave、2×distribution和0.01×regular三向量相加，保存向量的逐项加和误差为0，范数最大核验差约2.13e−14。')
    para('峰位变化以各trial自身G0探针为参考，表内768次为2库×3假设×128行，包含同父重复。保存probe没有完整窗或peak_index；独立读回核对其已保存布尔值及行/父计数。完整13子集的中位数、范围与计数见 [分层梯度表](WP2_GRADIENT_STRATA.csv)。')
    heading('9. 保存数组读回与4/8draw统计')
    para('受监督CPU任务原生退出0，耗时' + f"{raw_exit['elapsed_seconds']:.3f}" + '秒，CPU2线程；worker记录峰值RSS约' + f"{array['resources']['peak_RSS_bytes']/2**20:.2f}" + ' MiB，限制为RSS1536 MiB、虚拟内存3072 MiB、wall1800秒、输出64 MiB。176个原NPZ共3077281496字节逐项核SHA并解码一次；308项durable操作entered=completed，无失败、缺口或在途。首个完整pair是正式预算内任务。')
    table(['层级', '独立核验内容', '范围'], [
        ['硬波形编码', 'uint16裁窗ADC码＋原double基线均值＋可选IEEE XOR，解码与编码回写', '72 HARD_ROWS，父/seed路径/bank/draw身份与可重建flags'],
        ['逐行量', 'Q、T、raw/guarded PSD、峰幅、rise、FWHM、基线均值/RMS；另核front/crossing辅助值', '1898784行；16 BASELINE100文件'],
        ['r4原报告', '均值、无偏方差、9×9标量协方差、有限交集计数及MCSE', '72个原生REPORT NPZ；协方差分母3'],
        ['r8派生', '实际4+4行的中心化波形因子、方差、9×9标量协方差及MCSE；与merge式交叉核验', '36 pair；协方差分母7；波形因子经验秩≤7'],
        ['梯度保存向量', '1288维三向量加和、范数、余弦和分层成员/掩码计数', '8快照、624条件记录，原有1872次AD查询的输出']])
    para('r8硬波形方差使用 `[3(s²_bank0+s²_bank1)+2(μ_bank0−μ_bank1)²]/7`；硬波形r4、r8条件均值MCSE分别为sqrt(s²/4)、sqrt(s²/8)。标量MCSE使用各维有限样本数n，按sqrt(s²/n)计算；标量协方差使用逐对共同有限样本及其计数。带不同缺失支持的完整标量矩阵不强加统一半正定或秩7声明。完整256×256波形协方差由中心化因子表示，本轮未输出6593个稠密矩阵。')
    table(['候选', 'seed', '假设', 'final硬波形8draw散布/mV', '条件均值MCSE/mV'], [
        [p['trial'].split('_')[1], p['trial'][4], p['hypothesis'],
         f"{1000*p['hard_wave_sqrt_mean_sample_variance_V']:.6f}", f"{1000*p['hard_wave_RMS_conditional_mean_MCSE_V']:.6f}"]
        for p in array['pairs'] if p['arm'] == 'final'])
    para('上表散布为sqrt(mean_parent,time(s²_8draw))，MCSE为sqrt(mean_parent,time(s²_8draw/8))，涵盖电子学、ADC和裁窗后的总硬输出。014未保存纯信号完整384点库；上表不能用作纯形状头RMS。36 pair的摘要和因子/协方差SHA见 [r8派生矩表](WP2_R8_MOMENTS.csv)。')
    para('独立核验边界集中记录如下：完整wave/QPSD/noise经验距离沿用冻结producer输出并独立复核JSON聚合与选择逻辑；原REAL TRAIN median未重拟合，核验其保存SHA、fit21047、floor与使用链；原始Y没有第二份独立副本，producer曾在内存逐字节比较编码回放，独立任务核编码回写与统计；完整preADC384、纯载体及probe完整窗/peak_index未保存，相应原生诊断维持producer或保存布尔证据。')
    heading('10. 原生费用、保护与交付位置')
    table(['计数', '实际完成', '含义'], [
        ['G/optimizer steps', 8000, '4trial×2000；累计训练159667→167667'],
        ['training backward AD', 8000, '每G一次'],
        ['probe AD', 1872, '4trial×2快照×2库×3假设×13子集×3项'],
        ['全部新AD', 9872, '8000+1872'],
        ['old_A_rows', local['native_actual_cost']['old_A_rows'], '冻结A前向'],
        ['head实际参数前向', local['native_actual_cost']['head_actual_parameter_forward_rows'], '含校准实际评估'],
        ['head校准map', local['native_actual_cost']['calibration_map_rows'], '解析校准映射另列'],
        ['head总kernel行', local['native_actual_cost']['head_total_kernel_rows'], '实际参数行+校准map行'],
        ['005前向行', local['native_actual_cost']['noise005_rows'], '冻结电子学噪声模型'],
        ['hard ADC / gather', local['native_actual_cost']['hard_ADC_rows'], '两项各26480928'],
        ['原科学checkpoint loads', 2, 'A与005；保存头CP未在审核中解码'],
        ['optimizer创建 / 积分统计fit', '4 / 1', '每trial一个优化器；一次完整REAL TRAIN统计'],
        ['D/noise/pop/C2ST更新 / DEV模型行', '全部0', '本轮noGAN且DEV模型未使用'],
        ['独立读回新CP/NN/AD/更新/随机/CDF', '全部0', '176 NPZ CPU读回与本报告不新增科学训练成本']])
    para(f"科学原生child={native['child_pid']}、supervisor={native['supervisor_pid']}，两者退出0；总wall {native['elapsed_seconds']:.3f}秒，采样峰值RSS {native['peak_RSS_bytes']/2**30:.3f} GiB、GPU {native['peak_GPU_bytes']/2**30:.3f} GiB，整批输出{native['peak_output_bytes']/2**20:.2f} MiB。95个源码与271条旧保护记录通过冻结/原生保护链；独立closeout另核侧车6个实际Git blob及全部19个回收文件。")
    para('首次选择性回收142/318个科学文件，176个原NPZ留在原服务器只读目录 `/home/rylan/projects/ej276d-n01/wp2_014_20261015`；14个新头CP已随选择性包保存，审核未加载。随后在同级隔离目录执行一次保存数组CPU审核，回收19文件到 `runs/wp2_014_20261015/independent_readback_collected_01`。原142文件的collection和provisional selection保留原始时点字段；现行 `FINAL_DEVELOPMENT_SELECTION.json` 已引用独立CLOSEOUT_REVIEW，将C1代码路径选择登记为independently_reviewed，原科学分数及阈值保持相同。')
    para('科学选收归档SHA256：`' + science_collection['archive_sha256'] + '`。保存数组审核归档SHA256：`' + raw_collection['archive_sha256'] + '`。审核ARRAY_READBACK SHA256：`' + digest((READBACK / 'artifacts/ARRAY_READBACK.json').read_bytes()) + '`。本报告及图表仅由保存JSON生成，图中没有重播模型、AD、完整CDF或损失计算。')
    heading('11. 决策与下一工作包')
    para('WP2完成四次固定短试验及保存数组独立读回。后续开发沿用C1无分母积分表示；当前C0/C1权重仍为未晋级候选，科学响应匹配未建立。下一WP3由根流程按主计划登记固定人口的平均残差定位和配对均值路径对照，先完成实现与人工QA；014不补训、不追加候选，也不修改2%或T判据。')
    para('可交付的软件、冻结源码、负结果与审核证据均已保存。独立采集信息、完整系统外部泛化与最终确认按主计划另行保持自己的状态。')
    heading('12. 证据与可复用表')
    references = [
        ('用户主实施计划', 'USER_MASTER_PLAN_20261015.txt'), ('当前六工作包登记', 'MASTER_EXECUTION_PLAN.json'),
        ('014冻结配置', 'wp2_014_config.json'), ('014源码manifest', 'WP2_SOURCE_MANIFEST.json'),
        ('014验收合同', 'docs/WP2_ACCEPTANCE_CONTRACT.md'), ('014数据角色', 'WP2_DATA_ROLES.json'),
        ('观察合同', 'OBSERVATION_CONTRACT.json'), ('WP1冻结数字T', 'CALIBRATED_TOLERANCES_WP1_013.json'),
        ('已保存SOURCE信息审计', 'INPUT_INFORMATION_AUDIT.json'),
        ('科学选择性回收', 'receipts/wp2_014_20261015/collection_01/COLLECTION_RECEIPT.json'),
        ('原科学完整库存', 'receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json'),
        ('原始开发选择JSON（保留provisional时点）', 'receipts/wp2_014_20261015/postrun_engineering_01/DEVELOPMENT_SELECTION.json'),
        ('现行独审后开发选择', 'receipts/wp2_014_20261015/postrun_root_02/FINAL_DEVELOPMENT_SELECTION.json'),
        ('独立小JSON/日志审核', 'receipts/wp2_014_20261015/postrun_independent_01/LOCAL_METADATA_READBACK.json'),
        ('保存数组审核范围', 'receipts/wp2_014_20261015/postrun_independent_01/READBACK_SCOPE_NOTE.json'),
        ('已执行保存数组读回', 'runs/wp2_014_20261015/independent_readback_collected_01/artifacts/ARRAY_READBACK.json'),
        ('CPU读回回收', 'receipts/wp2_014_20261015/readback_collection_01/COLLECTION_RECEIPT.json'),
        ('独立收尾审核', 'receipts/wp2_014_20261015/postrun_independent_02/CLOSEOUT_REVIEW.json')]
    for label, path in references:
        para(f'[{label}](../../{path})')
    para('[完整wave表](WP2_ALL6_WAVE.csv) · [全部组件表](WP2_ALL6_COMPONENTS.csv) · [输出Q条件表](WP2_OUTPUT_Q_CONDITIONS.csv) · [SOURCE输入趋势表](WP2_SOURCE_INPUT_TRENDS.csv) · [全部分层梯度表](WP2_GRADIENT_STRATA.csv) · [r8矩摘要](WP2_R8_MOMENTS.csv) · [机器可读报告数据](REPORT_DATA.json) · [报告文件SHA清单](report_manifest.json)')
    md = '\n'.join(lines).rstrip() + '\n'
    write_new(OUT / 'N01_WP2_014_REPORT.md', md)
    write_new(OUT / 'N01_WP2_014_REPORT.html', render_html(lines))
    report_data = dict(schema='n01/wp2-report-data/v1', source='saved JSON only', cells=cells,
        all_components=components, gradient_rows=gradient_rows, r8_pair_summaries=array['pairs'],
        output_Q_conditions=conditional_rows, output_Q_final_worst_by_cell=conditional_summary,
        source_input_trends=source_rows, source_input_final_summary=source_summary,
        development_choice=selected, native_cost=local['native_actual_cost'], native_exit=native,
        raw_CPU_exit=raw_exit, scientific_acceptance=False, weights_promoted=False,
        new_NPZ_CP_NN_AD_random_CDF_training=0)
    write_new(OUT / 'REPORT_DATA.json', json_text(report_data))
    outputs = {str(p.relative_to(OUT)).replace('\\', '/'): dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes()))
               for p in sorted(OUT.rglob('*')) if p.is_file()}
    manifest = dict(schema='n01/wp2-report-manifest/v1', UTC=datetime.now(timezone.utc).isoformat(),
        status='built_saved_JSON_numeric_QA_passed_root_visual_review_pending', report_root=str(OUT),
        source=dict(path=str(Path(__file__)), bytes=Path(__file__).stat().st_size, sha256=digest(Path(__file__).read_bytes())),
        inputs=INPUTS, outputs=outputs, English_scientific_figures=3, figure_formats=['SVG', 'PNG'],
        skills=['anti-defensive-writing'], software=dict(Python=sys.version, Matplotlib=matplotlib.__version__, NumPy=np.__version__),
        source_readback_artifacts_unchanged=True, new_NPZ_reads=0, new_CP_loads=0, new_NN_rows=0,
        new_AD_queries=0, new_random_draws=0, new_CDF_or_loss_replays=0, new_training_updates=0,
        output_and_cache_E_D_only=True, browser_or_remote_calls=0, scientific_status='response_match_not_established',
        development_codepath='C1', usefulness='5/6; ALL6 failed', weights_promoted=False, scientific_acceptance=False,
        visual_QA_pending=True)
    write_new(OUT / 'report_manifest.json', json_text(manifest))
    receipt = dict(schema='n01/wp2-report-build/v1', UTC=datetime.now(timezone.utc).isoformat(), passed=True,
        report_root=str(OUT), outputs=outputs, manifest_sha256=digest((OUT / 'report_manifest.json').read_bytes()),
        source_sha256=manifest['source']['sha256'], inputs=INPUTS,
        numeric_construction_checks=dict(twelve_wave_cells=True, C1_unrounded5of6=True,
            all_saved_metrics_SHA_bound=True, all_three_hypotheses_retained=True,
            old_selection_unchanged=True, zero_NPZ_CP_NN_AD_random_CDF_training=True),
        root_visual_QA_pending=True, scientific_acceptance=False, weights_promoted=False)
    write_new(HERE / 'REPORT_BUILD_RECEIPT.json', json_text(receipt))
    print(json.dumps(dict(passed=True, report_root=str(OUT), outputs=len(outputs),
        MD_sha256=outputs['N01_WP2_014_REPORT.md']['sha256'], HTML_sha256=outputs['N01_WP2_014_REPORT.html']['sha256'],
        manifest_sha256=receipt['manifest_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
