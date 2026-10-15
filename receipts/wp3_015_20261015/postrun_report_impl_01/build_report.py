"""WP3/015 report from local saved JSON; no scientific arrays, models or SSH.

Reuses the WP2/014 JSON/CSV/HTML/figure helpers. Input numerical work is
limited to saved-summary arithmetic and standard plotting.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import sys

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reports/wp3_015_20261015'
SELECTED = ROOT / 'runs/wp3_015_20261015/selected_collected_01'
ART = SELECTED / 'runs/primary/artifacts'
READBACK = ROOT / 'runs/wp3_015_20261015/independent_readback_collected_01'
TMP = Path('E:/GAN/runtime_tmp/wp3_015_report')
TMP.mkdir(parents=True, exist_ok=True)
for name in ['TEMP', 'TMP', 'TMPDIR']:
    os.environ[name] = str(TMP)
os.environ['MPLCONFIGDIR'] = str(TMP / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
sys.dont_write_bytecode = True

HELPER = ROOT / 'receipts/wp2_014_20261015/postrun_report_impl_01/build_report.py'
spec = importlib.util.spec_from_file_location('wp2_report_helpers', HELPER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
# The imported helper has no main side effects. Restore this report's E-drive runtime.
base.OUT, base.TMP = OUT, TMP
for name in ['TEMP', 'TMP', 'TMPDIR']:
    os.environ[name] = str(TMP)
os.environ['MPLCONFIGDIR'] = str(TMP / 'matplotlib')
plt, np = base.plt, base.np
read_json, write_new, json_text, figure_outputs = base.read_json, base.write_new, base.json_text, base.figure_outputs
mean2 = base.mean2


def sha(path):
    assert path.suffix.lower() not in {'.npz', '.npy', '.pt', '.pth'}
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(name, rows):
    base.csv_file('tables/' + name, rows)


def native_json(path, inventory):
    value = read_json(path)
    key = path.relative_to(SELECTED).as_posix()
    meta = inventory['whole_remote_inventory'][key]
    observed = base.INPUTS[path.relative_to(ROOT).as_posix()]
    assert observed['sha256'] == meta['sha256'] and observed['bytes'] == meta['bytes']
    return value


def arithmetic_close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


def main():
    # Both final independent receipts must exist before any output is created.
    review = read_json('receipts/wp3_015_20261015/postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json')
    final = read_json('receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json')
    assert review.get('passed') is True
    assert final['selected_research_reference'] == 'M0'
    assert final['selected_mean_adapter_enabled'] is False
    assert final['weights_promoted'] is False and final['scientific_acceptance'] is False
    assert final['independent_raw_saved_array_readback_complete'] is True
    assert not OUT.exists(), 'Preserve all earlier report outputs; use an E-drive version snapshot before rebuilding.'

    cfg = read_json('wp3_015_config.json')
    tolerance = read_json('CALIBRATED_TOLERANCES_WP1_013.json')
    assert base.INPUTS['CALIBRATED_TOLERANCES_WP1_013.json']['sha256'] == 'f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
    selection = read_json('receipts/wp3_015_20261015/postrun_independent_01/DEVELOPMENT_SELECTION.json')
    local = read_json('receipts/wp3_015_20261015/postrun_independent_01/LOCAL_METADATA_READBACK.json')
    inventory = read_json('receipts/wp3_015_20261015/collection_01/REMOTE_FULL_INVENTORY.json')
    collection = read_json('receipts/wp3_015_20261015/collection_01/COLLECTION_RECEIPT.json')
    readback_collection = read_json('receipts/wp3_015_20261015/readback_collection_01/COLLECTION_RECEIPT.json')
    wake = read_json('receipts/wp3_015_20261015/readback_execution_01/WAKE_COMPLETION_SUMMARY.json')
    array = read_json(READBACK / 'artifacts/ARRAY_READBACK.json')
    assert base.INPUTS[(READBACK / 'artifacts/ARRAY_READBACK.json').relative_to(ROOT).as_posix()]['sha256'] == 'd0ed2adfba7eb193647dba54eb99bcc029cf02f31f206c699b3a7854545df2c3'
    readback_exit = read_json(READBACK / 'supervisor/exit_receipt.json')
    native = read_json(SELECTED / 'runs/primary/supervisor/exit_receipt.json')
    complete = native_json(ART / 'COMPLETE.json', inventory)
    gate = native_json(ART / 'MEAN_RESIDUAL_GATE.json', inventory)
    fit = native_json(ART / 'INTEGRAL_TRAIN_STATISTICS_REFERENCE.json', inventory)
    info = read_json('INPUT_INFORMATION_AUDIT.json')
    read_json('OBSERVATION_CONTRACT.json')
    read_json('WP3_DATA_ROLES.json')
    read_json('WP3_SOURCE_MANIFEST.json')
    assert final['candidates'] == selection['candidates']
    for a in final['cells']:
        b=next(c for c in selection['cells'] if (c['candidate'],c['seed'],c['hypothesis'])==(a['candidate'],a['seed'],a['hypothesis']))
        assert all(a[k]==b[k] for k in ['candidate','seed','hypothesis','eligible_cell','wave_usefulness_2percent','pending_uncalibrated','absolute_T_all_positive_defined'])
        assert all(arithmetic_close(a[k],b[k]) for k in ['wave_relative_improvement_vs_A','wave_relative_improvement_vs_common_initial'])
        for metric,details in a['components'].items():
            for k,v in details.items():
                old=b['components'][metric][k]
                assert arithmetic_close(v,old) if isinstance(v,(int,float)) and not isinstance(v,bool) else v==old
    assert array['passed'] and array['status'] == 'PASS' and native['exit_code'] == readback_exit['exit_code'] == 0
    assert gate['all6_passed'] and complete['all_four_trials_completed'] and complete['counters']['G_updates'] == 8000
    assert complete['operations_entered'] == complete['operations_completed'] == 154518 and not complete['inflight']
    assert wake['operations_entered'] == wake['operations_completed'] == 285 and wake['journal_events'] == 570
    assert wake['original_training_counts']['cumulative_training_unchanged'] == 175667
    assert local['whole_remote_inventory_count'] == 300 and local['selected_count'] == 139

    cells, components = [], []
    hypotheses = ['fast', 'slow', 'common_level']
    for candidate in ['M0', 'M1']:
        for seed in [0, 1]:
            trial = f'seed{seed}_{candidate}'
            initial = native_json(ART / trial / 'initial_evaluation/METRICS.json', inventory)
            end = native_json(ART / trial / 'final_evaluation/METRICS.json', inventory)
            for hyp in hypotheses:
                by_arm = {}
                for arm, doc in [('A', initial), ('initial', initial), ('final', end)]:
                    banks = [doc['metrics'][f'bank{bank}_{arm}_{hyp}'] for bank in [0, 1]]
                    assert all(b['all_rows_retained'] and b['coverage']['conditional_coverage_complete'] and b['coverage']['conditional_bins_used'] == 8 for b in banks)
                    by_arm[arm] = {k:mean2([b['values'][k] for b in banks]) for k in banks[0]['values']}
                    by_arm[arm + '_banks'] = [b['values'] for b in banks]
                A, initial_values, final_values = [by_arm[a] for a in ['A','initial','final']]
                s = next(c for c in final['cells'] if (c['candidate'],c['seed'],c['hypothesis']) == (candidate,seed,hyp))
                rel = (A['wave']-final_values['wave'])/A['wave']
                assert arithmetic_close(rel, s['wave_relative_improvement_vs_A'])
                row = dict(candidate=candidate, seed=seed, hypothesis=hyp, A_wave=A['wave'],
                    common_initial_wave=initial_values['wave'], final_wave=final_values['wave'],
                    final_bank0_wave=by_arm['final_banks'][0]['wave'], final_bank1_wave=by_arm['final_banks'][1]['wave'],
                    wave_improvement_vs_A_percent=100*rel,
                    wave_improvement_vs_common_initial_percent=100*(initial_values['wave']-final_values['wave'])/initial_values['wave'],
                    usefulness_ALL6_cell=rel >= .02, dual_reference_eligible_cell=s['eligible_cell'],
                    absolute_T_all_positive_defined=s['absolute_T_all_positive_defined'], A=A, initial=initial_values, final=final_values)
                cells.append(row)
                for metric, t in tolerance['T_by_hypothesis'][hyp].items():
                    detail = s['components'][metric]
                    assert all(arithmetic_close(detail[k], v[metric]) for k,v in [('A',A),('common_initial',initial_values),('final',final_values)])
                    positive_defined = isinstance(t,(int,float)) and math.isfinite(t) and t > 0
                    if positive_defined:
                        assert arithmetic_close(detail['T'], t)
                        assert detail['status'] == 'noninferior_to_both_references'
                    else:
                        assert detail['status'] == 'pending_not_calibrated'
                    components.append(dict(candidate=candidate,seed=seed,hypothesis=hyp,component=metric,
                        A=A[metric], common_initial=initial_values[metric], final=final_values[metric], T=t,
                        final_minus_A=final_values[metric]-A[metric], final_minus_common_initial=final_values[metric]-initial_values[metric],
                        status=detail['status'], absolute_final_over_T=detail['absolute_distance_over_T'],
                        absolute_within_T=detail['absolute_within_T']))
    assert len(cells)==12 and len(components)==264
    assert sum(c['usefulness_ALL6_cell'] for c in cells if c['candidate']=='M0')==6
    assert sum(c['usefulness_ALL6_cell'] for c in cells if c['candidate']=='M1')==0
    assert sum(c['status']=='pending_not_calibrated' for c in components)==84

    conditional_rows, source_rows, flags, population = [], [], [], []
    features = ['Q_Vns','front_Vns','tail_Vns','PSD_raw','PSD_guarded','peak_voltage_V','rise10_90_ns','FWHM_ns']
    for pair in array['pairs']:
        trial,snapshot,arm,hyp = [pair[k] for k in ['trial','snapshot','arm','hypothesis']]
        for bank in [0,1]:
            report = native_json(ART/trial/snapshot/f'bank{bank}_{arm}_{hyp}_REPORT.json', inventory)
            identity = dict(trial=trial,candidate=trial.split('_')[1],seed=int(trial[4]),snapshot=snapshot,arm=arm,hypothesis=hyp,bank=bank)
            conditions=report['output_Q_conditions']
            assert conditions['minimum_support']==16 and conditions['all_rows_in_exactly_one_bin']
            assert sum(b['fake_rows'] for b in conditions['bins'])==26372
            assert sum(b['real_rows'] for b in conditions['bins'])==21047
            for b in conditions['bins']:
                row=dict(**identity,output_Q_bin=b['bin'],fake_rows=b['fake_rows'],real_rows=b['real_rows'],supported=b['supported'],
                    PSD_W1_defined=b['defined_PSD_W1']['value'],conditional_T='not_calibrated')
                for feature in features:
                    for domain in ['fake','real']:
                        stats=b[domain][feature]
                        for statistic in ['defined_rows','undefined_rows','mean','std_population']:
                            row[f'{domain}_{feature}_{statistic}']=stats[statistic]
                        for quantile,value in stats['quantiles'].items():
                            row[f'{domain}_{feature}_{quantile}']=value
                row['rise_mean_difference_ns']=row['fake_rise10_90_ns_mean']-row['real_rise10_90_ns_mean']
                row['FWHM_mean_difference_ns']=row['fake_FWHM_ns_mean']-row['real_FWHM_ns_mean']
                conditional_rows.append(row)
            trends=report['source_input_trends']
            assert trends['parent_count']==6593 and trends['descriptors_are_audit_only_not_network_inputs']
            for descriptor,responses in trends['correlations'].items():
                for statistic,values in responses.items():
                    for feature,corr in values.items():
                        source_rows.append(dict(**identity,descriptor=descriptor,response_statistic=statistic,response_feature=feature,
                            Pearson=corr['value'],defined_parents=corr['parents'],kind=corr['kind'],physical_conditional_response_identified=False))
            p=report['population']
            for domain in ['fake','real']:
                for flag,counts in p[domain+'_flags'].items():
                    flags.append(dict(**identity,domain=domain,flag=flag,**counts,status='saved_report_value'))
            for flag in p['real_unknown_flags']:
                flags.append(dict(**identity,domain='real',flag=flag,count=None,rows=21047,rate=None,status='unknown_not_zero'))
            for category in ['integrals','morphology']:
                for feature,domains in p[category].items():
                    for domain in ['fake','real']:
                        stats=domains[domain]
                        row=dict(**identity,category=category,feature=feature,domain=domain,
                            rows=stats['rows'],defined_rows=stats['defined_rows'],undefined_rows=stats['undefined_rows'],
                            mean=stats['mean'],std_population=stats['std_population'])
                        row.update(stats['quantiles'])
                        population.append(row)
    assert len(conditional_rows)==576 and len(source_rows)==10368
    assert all(r['supported'] for r in conditional_rows)
    # Retain the fake and REAL flag schemas exactly; all CSV rows share the union of fields.
    flag_fields=list(dict.fromkeys(k for row in flags for k in row))
    flags=[{k:row.get(k) for k in flag_fields} for row in flags]
    condition_summary, source_summary=[],[]
    for c in cells:
        trial=f"seed{c['seed']}_{c['candidate']}"
        group=[r for r in conditional_rows if (r['trial'],r['hypothesis'],r['arm'])==(trial,c['hypothesis'],'final')]
        assert len(group)==16
        worst_psd=max(group,key=lambda r:r['PSD_W1_defined'])
        worst_rise=max(group,key=lambda r:abs(r['rise_mean_difference_ns']))
        worst_width=max(group,key=lambda r:abs(r['FWHM_mean_difference_ns']))
        condition_summary.append(dict(trial=trial,hypothesis=c['hypothesis'],supported_bins=16,
            minimum_fake_rows=min(r['fake_rows'] for r in group),minimum_real_rows=min(r['real_rows'] for r in group),
            PSD_W1_max=worst_psd['PSD_W1_defined'],PSD_bank=worst_psd['bank'],PSD_bin=worst_psd['output_Q_bin'],
            rise_signed_worst=worst_rise['rise_mean_difference_ns'],rise_bank=worst_rise['bank'],rise_bin=worst_rise['output_Q_bin'],
            FWHM_signed_worst=worst_width['FWHM_mean_difference_ns'],FWHM_bank=worst_width['bank'],FWHM_bin=worst_width['output_Q_bin']))
        summary=dict(trial=trial,hypothesis=c['hypothesis'])
        for descriptor,statistic,feature,label in [('canonical_Q_Vns','response_mean','Q_Vns','inputQ_meanQ'),
            ('canonical_Q_Vns','response_variance','Q_Vns','inputQ_varianceQ'),('shape_PC_00','response_mean','Q_Vns','PC0_meanQ'),
            ('shape_PC_00','response_mean','tail_Vns','PC0_meanTail')]:
            group_source=[r for r in source_rows if (r['trial'],r['hypothesis'],r['arm'],r['descriptor'],r['response_statistic'],r['response_feature'])==
                (trial,c['hypothesis'],'final',descriptor,statistic,feature)]
            assert len(group_source)==2
            summary[label]=mean2([r['Pearson'] for r in group_source])
        source_summary.append(summary)
    moment_rows=[dict(trial=p['trial'],snapshot=p['snapshot'],arm=p['arm'],hypothesis=p['hypothesis'],parents=p['parents'],
        native_bank0_draws=4,native_bank1_draws=4,derived_draws=8,covariance_denominator=7,waveform_empirical_rank_upper_bound=7,
        spread_RMS_V=p['hard_wave_sqrt_mean_sample_variance_V'],conditional_mean_MCSE_RMS_V=p['hard_wave_RMS_conditional_mean_MCSE_V'],
        native_bank0_spread_RMS_V=p['r4_by_bank'][0]['waveform_spread_RMS_V'],native_bank1_spread_RMS_V=p['r4_by_bank'][1]['waveform_spread_RMS_V'],
        complete_case_scalar_parents=p['complete_case_scalar_parents'],scalar_covariance_SHA256=p['r8_scalar_covariance_sha256'],
        waveform_centered_factor_SHA256=p['centered_wave_factor_sha256']) for p in array['pairs']]
    assert len(moment_rows)==36

    OUT.mkdir(parents=True)
    (OUT/'figures').mkdir()
    (OUT/'tables').mkdir()
    csv_file('ALL6_WAVE.csv',[{k:v for k,v in r.items() if k not in ['A','initial','final']} for r in cells])
    csv_file('ALL6_COMPONENTS.csv',components)
    csv_file('OUTPUT_Q_CONDITIONS.csv',conditional_rows)
    csv_file('SOURCE_INPUT_TRENDS.csv',source_rows)
    csv_file('ALL_FLAGS.csv',flags)
    csv_file('POPULATION_FEATURES.csv',population)
    csv_file('DERIVED8_MOMENTS.csv',moment_rows)
    csv_file('MEAN_RESIDUAL_GATE.csv',gate['cells'])
    csv_file('OUTPUT_Q_FINAL_SUMMARY.csv',condition_summary)
    csv_file('SOURCE_INPUT_FINAL_SUMMARY.csv',source_summary)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'N01-WP3-015','axes.axisbelow':True})
    colors={'M0':'#246ba5','M1':'#cf7a3b'}
    fig,(ax,score)=plt.subplots(1,2,figsize=(13.4,5.1),gridspec_kw={'width_ratios':[2.25,1]},layout='constrained')
    x=np.arange(6)
    for k,candidate in enumerate(['M0','M1']):
        values=[r['wave_improvement_vs_A_percent'] for r in cells if r['candidate']==candidate]
        bars=ax.bar(x+(k-.5)*.36,values,width=.34,color=colors[candidate],label=candidate)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8.5,padding=4)
    ax.axhline(2,color='#ac2c35',linestyle='--',label='Usefulness: 2% in ALL6')
    ax.set_xticks(x,['S0 fast','S0 slow','S0 common','S1 fast','S1 slow','S1 common'])
    ax.set_ylabel('Wave-distance improvement over A (%)')
    ax.set_ylim(0,4.3)
    ax.set_title('M0: 6/6 useful cells; M1: 0/6')
    ax.grid(axis='y',color='#e0e6ee')
    ax.legend(loc='upper left',frameon=False,ncol=3,fontsize=9)
    wave_means=[selection['candidates'][c]['final_wave_mean12'] for c in ['M0','M1']]
    bars=score.bar([0,1],wave_means,color=[colors[c] for c in ['M0','M1']],width=.60)
    score.bar_label(bars,labels=[f'{v:.9f}' for v in wave_means],fontsize=9,padding=5)
    common=selection['candidates']['M0']['common_initial_wave_mean12']
    score.axhline(common,color='#333d4b',linestyle='--',label=f'Common initial\n{common:.9f}')
    score.set_xticks([0,1],['M0','M1'])
    score.set_ylim(.1485,.1540)
    score.set_ylabel('Mean of 12 final bank distances (lower is better)')
    score.set_title('Frozen development selection')
    score.legend(frameon=False,loc='upper left',fontsize=8.5)
    figure_outputs(fig,'01_wave_selection')

    calibrated=[k for k,t in tolerance['T_by_hypothesis']['fast'].items() if isinstance(t,(int,float)) and t>0]
    ratio=np.array([[next(r['absolute_final_over_T'] for r in components if (r['candidate'],r['seed'],r['hypothesis'],r['component'])==
        (c['candidate'],c['seed'],c['hypothesis'],k)) for k in calibrated] for c in cells])
    fig,ax=plt.subplots(figsize=(13.8,8.0),layout='constrained')
    im=ax.imshow(np.log10(ratio),cmap='RdBu_r',norm=base.TwoSlopeNorm(vmin=-1.6,vcenter=0,vmax=1.6),aspect='auto')
    labels={'wave':'wave','QPSD':'QPSD','noise':'noise','Q_Vns':'Q','front_Vns':'front','tail_Vns':'tail','PSD_raw':'PSD raw',
        'PSD_guarded':'PSD guarded','peak_voltage_V':'peak','rise10_90_ns':'rise','FWHM_ns':'FWHM',
        'baseline_mean_V':'baseline DC','baseline_RMS_V':'baseline RMS','rise_undefined_rate':'rise undef.','rise_prepeak_censored_rate':'rise censored'}
    ax.set_xticks(range(len(calibrated)),[labels[k] for k in calibrated],rotation=45,ha='right')
    ax.set_yticks(range(12),[f"{c['candidate']} S{c['seed']} {c['hypothesis'].replace('common_level','common')}" for c in cells])
    for iy in range(12):
        for ix in range(len(calibrated)):
            v=ratio[iy,ix]
            label=f'{v:.5f}' if .995<v<1.005 else f'{v:.3f}' if .80<v<1.20 else f'{v:.2f}'
            ax.text(ix,iy,label,ha='center',va='center',fontsize=8.3,color='white' if abs(math.log10(v))>1.05 else '#172b40')
    ax.axhline(5.5,color='white',linewidth=3)
    ax.set_title('Absolute distance / frozen T: values above 1 fail; both candidates fail ALL6')
    fig.colorbar(im,ax=ax,label='log10(distance / T)',shrink=.72)
    figure_outputs(fig,'02_absolute_tolerance')

    fig,ax=plt.subplots(figsize=(11.6,4.5),layout='constrained')
    residual=[1000*c['residual_RMS_V'] for c in gate['cells']]
    required=[1000*c['required_residual_RMS_V'] for c in gate['cells']]
    bars=ax.bar(x-.18,residual,width=.34,color='#246ba5',label='Saved finite-population residual RMS')
    ax.bar_label(bars,fmt='%.4f',fontsize=9,padding=4)
    bars=ax.bar(x+.18,required,width=.34,color='#cf7a3b',label='Predeclared required residual RMS')
    ax.bar_label(bars,fmt='%.4f',fontsize=9,padding=4)
    ax.set_xticks(x,['S0 fast','S0 slow','S0 common','S1 fast','S1 slow','S1 common'])
    ax.set_ylabel('Hard-output population-mean RMS (mV)')
    ax.set_ylim(0,9.1)
    ax.set_title('Mean residual gate: 6/6 passed before checkpoint loads; conditional mean truth remains unknown')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax.grid(axis='y',color='#e0e6ee')
    figure_outputs(fig,'03_mean_residual_gate')

    fig,(ax,mc)=plt.subplots(1,2,figsize=(12.6,4.7),layout='constrained')
    for k,candidate in enumerate(['M0','M1']):
        vals=[next(p for p in moment_rows if (p['trial'],p['arm'],p['hypothesis'])==(f"seed{c['seed']}_{candidate}",'final',c['hypothesis']))
            for c in cells if c['candidate']==candidate]
        ax.bar(x+(k-.5)*.36,[1000*p['spread_RMS_V'] for p in vals],width=.34,color=colors[candidate],label=candidate)
        mc.bar(x+(k-.5)*.36,[1000*p['conditional_mean_MCSE_RMS_V'] for p in vals],width=.34,color=colors[candidate],label=candidate)
    for axis in [ax,mc]:
        axis.set_xticks(x,['S0 fast','S0 slow','S0 common','S1 fast','S1 slow','S1 common'],rotation=25,ha='right')
        axis.grid(axis='y',color='#e0e6ee')
        axis.set_ylim(0,axis.get_ylim()[1]*1.18)
        axis.legend(frameon=False,ncol=2,loc='upper left')
    ax.set_ylabel('sqrt(mean sample variance), mV')
    mc.set_ylabel('sqrt(mean conditional-mean MCSE squared), mV')
    ax.set_title('Final hard-output spread: derived 4+4 draws')
    mc.set_title('Finite Monte Carlo precision: derived 8 draws')
    figure_outputs(fig,'04_repeated_parent_moments')

    lines=[]
    def para(t):
        lines.extend([t,''])
    def heading(t):
        para('## '+t)
    def table(headers,rows):
        lines.append('| '+' | '.join(headers)+' |')
        lines.append('| '+' | '.join(['---']*len(headers))+' |')
        lines.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows)
        lines.append('')
    para('# N01 · WP3 / 015 有限均值路径对照报告')
    para('**开发结论：采用 M0 继续形状头作为下一阶段研究参考，均值适配器关闭。** 四次2000G短试和一次保存数组CPU独立读回已完成。M0相对A的完整wave用途门为6/6，M1为0/6；两个候选均通过冻结双参考非劣条件，均未通过全部绝对T。权重未晋级，科学状态仍为 `response_match_not_established`。')
    heading('1. 固定对照与数据口径')
    para('每个seed从原014 C1 final头的相同字节起步，使用新的AdamW。M0只继续1288参数形状头；M1冻结该头，只训练6个零初始化均值参数。每trial为2000G/AD，共8000G/AD；原A和005保持冻结。M1均值由增益、软件σ2 ns平滑差、软件τ16 ns慢尾差三个基和连续输入控制构成，prefix100严格为0，增量RMS上限为0.10×RMS(A)。随机形状幅度仍由原A控制。')
    para('完整6593条SOURCE TRAIN和21047条REAL TRAIN参与各trial；每seed/候选/假设包含2个独立随机库，每库每父4 draw，单库26372条硬输出。fast、slow、common_level分别报告；mean2bank是两个库各自经验距离的均值，开发排序使用12个final bank距离的均值。MC draw的统计单位仍是父事件，422个REAL逻辑组属于既有采集内分组。旧DEV已暴露，本轮DEV模型行和人口调整均为0。')
    para('观察链沿用14 bit ADC、384点内部carrier、原100点基线、搜索[112:144)的first-argmax、裁窗[p−16:p+240)和256点输出；采样间隔为1软件ns，tail从输出第46点起。T是WP1预登记的人工灵敏度标尺，SHA256为 `f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b`。')
    heading('2. 均值残差入口')
    rmin,rmax=min(c['residual_RMS_V'] for c in gate['cells']),max(c['residual_RMS_V'] for c in gate['cells'])
    tmin,tmax=min(c['required_residual_RMS_V'] for c in gate['cells']),max(c['required_residual_RMS_V'] for c in gate['cells'])
    para(f'checkpoint、NN、优化器和新随机数之前的入口检查为6/6阳性：保存总体均值残差RMS为{rmin*1000:.5f}–{rmax*1000:.5f} mV，要求值为{tmin*1000:.5f}–{tmax*1000:.5f} mV；两库残差cosine均大于0.99998。它支持执行预登记的有限均值对照；REAL与SOURCE未配对，真实条件均值保持unknown。hard256均值DC与电子学baseline100 DC分别解释。')
    para('![Mean residual gate](figures/03_mean_residual_gate.svg)')
    heading('3. 开发选择与完整wave')
    para('冻结规则要求同一候选ALL6的每个正且已定义T分量，同时相对A与共同C1初始非劣：两项mean2bank距离差均≤T。必需已标定分量缺失或undefined使候选pending；零或undefined T继续pending。合格候选按mean12排序，1e−12内tie选M0；胜者还须比共同初始低超过1e−12。')
    table(['候选','final wave mean12','共同初始 mean12','双参考非劣','≥2%用途','绝对T ALL6'],
        [[c,str(selection['candidates'][c]['final_wave_mean12']),str(common),'6/6',f"{selection['candidates'][c]['relative_wave_usefulness_cells']}/6",'FAIL'] for c in ['M0','M1']])
    para(f'M0相对共同初始的mean12改善为{100*(common-wave_means[0])/common:.6f}%；M1相对共同初始变化为{100*(wave_means[1]-common)/common:+.6f}%（距离增大）。015的M0用途6/6属于本次继续训练结果；原014 C1的历史用途5/6保持原记录。')
    para('![Wave selection](figures/01_wave_selection.svg)')
    table(['候选','seed','噪声假设','A wave','共同初始','final wave','相对A改善/%','用途'],
        [[c['candidate'],c['seed'],c['hypothesis'],f"{c['A_wave']:.9f}",f"{c['common_initial_wave']:.9f}",f"{c['final_wave']:.9f}",f"{c['wave_improvement_vs_A_percent']:.9f}",'PASS' if c['usefulness_ALL6_cell'] else 'FAIL'] for c in cells])
    heading('4. 绝对容忍度与保留失败')
    para('每个候选90个已标定分量判定均满足双参考非劣，42个未标定分量继续pending；合计180个非劣、84个pending。绝对distance≤T另行判断。峰高、rise、FWHM及baseline DC仍有明显绝对差距；其中baseline DC为005电子学限制，prefix100为0的signal均值适配器不覆盖该段。完整QPSD、raw/guarded PSD、Q/front/tail、噪声和失效率见全部组件表。')
    para('![Absolute tolerance](figures/02_absolute_tolerance.svg)')
    fail_ranges=[]
    for k in ['QPSD','peak_voltage_V','rise10_90_ns','FWHM_ns','baseline_mean_V']:
        v=[r['absolute_final_over_T'] for r in components if r['component']==k]
        fail_ranges.append([k,f'{min(v):.5f}–{max(v):.5f}',sum(x>1 for x in v),'12'])
    table(['分量','final distance/T范围','绝对失败格','总格'],fail_ranges)
    heading('5. 输出Q条件与低Q')
    para('四套评价中的输出Q条件部分覆盖8个冻结箱；全部72个REPORT、576个条件箱均有至少16行支持，Q≤0、分母保护和undefined行均保留在flags及分母中。下表每项分别取2库×8箱的最差值，括号为库/箱；条件T未标定。完整表保留两域各特征的defined/undefined数、均值、宽度和p01/p10/p50/p90/p99分位数。')
    table(['trial','假设','最大PSD W1','rise均值差/ns（绝对最差）','FWHM均值差/ns（绝对最差）'],
        [[r['trial'],r['hypothesis'],f"{r['PSD_W1_max']:.6f} (b{r['PSD_bank']}/q{r['PSD_bin']})",
          f"{r['rise_signed_worst']:+.6f} (b{r['rise_bank']}/q{r['rise_bin']})",f"{r['FWHM_signed_worst']:+.6f} (b{r['FWHM_bank']}/q{r['FWHM_bin']})"] for r in condition_summary])
    heading('6. SOURCE输入趋势与可辨识范围')
    para('SOURCE输入趋势使用6593个父事件的18个审计描述量，对Q、tail、peak和baseline RMS的父内均值/方差记录相关性；描述量和事件标识均不加入网络。下表为两个库相关系数的均值，仅描述生成器内部输入关联。真实条件响应、真实源人口权重以及跨采集物理复用继续保持未识别/unknown。')
    table(['trial','假设','inputQ→meanQ','inputQ→varQ','PC0→meanQ','PC0→meanTail'],
        [[r['trial'],r['hypothesis']]+[f"{r[k]:+.6f}" for k in ['inputQ_meanQ','inputQ_varianceQ','PC0_meanQ','PC0_meanTail']] for r in source_summary])
    feasibility=info['same_Q_real_input_shape_feasibility']
    para(f'既有输入审计覆盖全部8192条合法SOURCE（TRAIN6593、INTERNAL_VAL1599；固定DEV512已暴露），分别报告full480、carrier384、clean256的归一形状PCA及数值舍入。验证过的沉积元数据h=.15×edep_hadron/(edep_ee+.15×edep_hadron)属于SOURCE代理。相对Q差≤1%的既有同角色配对中，{feasibility["rows_with_testable_shape_contrast"]}条输入具有达到0.001峰归一RMS的形状对照，最大强度{feasibility["largest_verified_strength"]:.9f}；响应形状干预尚未执行。全人口Q宽差及输入关联不构成条件charge涨落识别。')
    heading('7. 同父重复响应：native4与derived8')
    para('保存数组CPU审核验证36组bank pair；每组6593父、每库4 draw。native4的行特征、flags、有限支持、均值/方差、MCSE及9×9标量协方差已核验。derived8由实际4+4保存行重新合并，协方差分母7，MCSE按有效n计算；它是新派生摘要，原producer没有native r8报告。')
    para('硬波形derived8方差使用 [3(s²₀+s²₁)+2(μ₀−μ₁)²]/7；r4/r8条件均值MCSE分别为sqrt(s²/4)、sqrt(s²/8)。中心化8draw波形因子的经验秩≤7；这与preADC形状分支8维潜变量的理论协方差秩≤8属于不同层次。标量协方差使用逐对共同有限支持与计数；缺失支持不同的矩阵不自动保证半正定。完整256×256波形协方差通过中心化因子表示，未物化每父稠密矩阵。')
    para('![Repeated parent moments](figures/04_repeated_parent_moments.svg)')
    para('图中散布为sqrt(mean_parent,time(s²_8draw))，MCSE为sqrt(mean_parent,time(s²_8draw/8))，包括电子学、ADC和裁窗后的总硬输出；纯信号完整preADC384未保存为独立审核库，图中数值不作纯形状分支RMS。')
    heading('8. flags与数值边界')
    para('全部fake/REAL flags及其原分母在 [ALL_FLAGS.csv](tables/ALL_FLAGS.csv) 保留，包括低Q、PSD undefined/guarded、rise/FWHM失效与截断、峰位边界/并列、裁窗全局峰关系、ADC rail/preADC越界、half-grid、形状峰floor和Q floor。REAL缺少的硬件域标记保持unknown。未标定flag的pending状态与观测到的0发生率分开保存。')
    heading('9. 独立审核分层')
    table(['证据层','完成内容','边界'],[
        ['原producer','完整wave/QPSD/noise经验距离和四套REPORT','数值距离沿用冻结producer'],
        ['独立JSON聚合/决策','原日志、库存、双参考ALL6、mean12、用途和绝对T','未重播完整经验CDF或loss'],
        ['保存数组CPU','ADC roundtrip、身份/行映射、行features/flags、native4矩、derived8和有限支持','原Y无第二份独立副本；CP tensor未解码'],
        ['原统计与均值入口','冻结C1统计与median使用链、mean gate标量公式','原REAL TRAIN median未重拟合；REAL mean摘要NPZ仅SHA未decode'],
        ['物理响应','科学状态response_match_not_established','绝对T和跨条件确认仍按主计划独立判定']])
    para('CPU完成161个原NPZ SHA检查，共3064431039字节；解码160个评价NPZ（72 HARD_ROWS、72 REPORT、16 BASELINE100），核验1898784条硬输出。285个operation entered=completed、570条journal事件，无inflight/failed。两个原生退出均为0；用时60.446秒。worker getrusage峰值489336832字节、supervisor采样owned-tree峰值402255872字节是不同测量，均在1536 MiB上限内。19个小文件共513764字节已回收并核SHA。')
    para('本报告仅使用已回收JSON聚合和标准绘图；新增NPZ/CP/NN/AD/G、随机、原TRAIN/DEV波形读取、GPU、CDF/loss重播和SSH均为0。独立审核完成不授予物理响应匹配成功。')
    heading('10. 实际成本与可追溯性')
    counters=complete['counters']
    table(['计数','实际值','解释'],[
        ['G / AD / optimizer step','8000 / 8000 / 8000','4 trial×2000；累计训练已含015，为175667'],
        ['CP load / optimizer创建','4 / 4','A、005、每seed一个共享C1头；每trial新优化器'],
        ['A / 实际head / 总head kernel / 005','各8613952','训练行及initial/final评价'],
        ['mean adapter行',4306976,'仅两个M1 trial'],
        ['hard ADC / gather','各26474784','所有路径分别记账'],
        ['科学operation entered / completed','154518 / 154518','无failed或unknown inflight'],
        ['mean入口 REAL / REPORT父行','21047 / 79116','12 REPORT、6 gate测试；checkpoint前'],
        ['map / 新统计fit / probeAD / D-noise-pop-C2ST / DEV','全部0','冻结目标、冻结005、原角色']])
    para(f'训练原生child与supervisor退出0，总wall {native["elapsed_seconds"]:.3f}秒，采样峰值RSS {native["peak_RSS_bytes"]/2**30:.3f} GiB、GPU {native["peak_GPU_bytes"]/2**30:.3f} GiB。首次选择性回收139/300文件，共76908267字节，包含12个未解码薄checkpoint且不含NPZ；161个原NPZ留在服务器原目录。CPU审核随后在独立目录完成；本报告阶段无远程操作。')
    para('原 provisional DEVELOPMENT_SELECTION 和 LOCAL_METADATA_READBACK 保留其生成时的pending字段；现行FINAL_DEVELOPMENT_SELECTION引用最终独立审核，将保存数组读回登记为complete。原分数、T、用途及科学状态保持原值。')
    heading('11. 下一步')
    para('以015 M0为下一阶段研究参考，均值适配器关闭。WP3既定6次短试已用4次，剩余最多2次只检验单因素q_sig电荷随机性；采用正值、均值校正且可完全关闭的低自由度形式。该试验检验信号电荷涨落，不承诺同时修复peak、rise或电子学DC。教师恢复、稳健验证和最终确认按用户主计划继续。')
    heading('12. 交付与来源')
    para('[机器可读报告数据](REPORT_DATA.json) · [来源SHA256](SOURCES_SHA256.json) · [输出清单](report_manifest.json) · [内容验证](VALIDATION.json)')
    for filename,label in [('ALL6_WAVE.csv','12格wave'),('ALL6_COMPONENTS.csv','264组件'),('OUTPUT_Q_CONDITIONS.csv','576条件Q箱'),
        ('SOURCE_INPUT_TRENDS.csv','10368输入趋势'),('ALL_FLAGS.csv','全部flags'),('POPULATION_FEATURES.csv','全人口特征'),
        ('DERIVED8_MOMENTS.csv','36 pair矩'),('MEAN_RESIDUAL_GATE.csv','6格均值入口')]:
        para(f'[{label}](tables/{filename})')
    for label,path in [('015冻结配置','wp3_015_config.json'),('015验收合同','docs/WP3_ACCEPTANCE_CONTRACT.md'),
        ('用户主实施计划','USER_MASTER_PLAN_20261015.txt'),('既有SOURCE审计','INPUT_INFORMATION_AUDIT.json'),
        ('最终独立审核','receipts/wp3_015_20261015/postrun_independent_02/FINAL_INDEPENDENT_REVIEW.json'),
        ('最终研究参考选择','receipts/wp3_015_20261015/postrun_independent_02/FINAL_DEVELOPMENT_SELECTION.json'),
        ('原始JSON独立决策','receipts/wp3_015_20261015/postrun_independent_01/DEVELOPMENT_SELECTION.json'),
        ('CPU完成回执','receipts/wp3_015_20261015/readback_execution_01/WAKE_COMPLETION_SUMMARY.json'),
        ('保存数组审核结果','runs/wp3_015_20261015/independent_readback_collected_01/artifacts/ARRAY_READBACK.json')]:
        para(f'[{label}](../../{path})')
    md='\n'.join(lines).rstrip()+'\n'
    write_new(OUT/'REPORT.md',md)
    html=base.render_html(lines).replace('N01 WP2 014 损失表示对照报告','N01 WP3 015 有限均值路径对照报告')
    write_new(OUT/'REPORT.html',html)
    report_data=dict(schema='n01/wp3-015-report-data/v1',UTC=datetime.now(timezone.utc).isoformat(),cells=cells,
        all_components=components,development_choice=final,mean_residual_gate=gate,output_Q_conditions=conditional_rows,
        output_Q_final_worst_by_cell=condition_summary,source_input_trends=source_rows,source_input_final_summary=source_summary,
        all_flags=flags,population_features=population,derived8_moments=moment_rows,native_cost=counters,native_exit=native,
        readback_actual_counts=wake['actual_counts'],readback_resources=wake['actual_resources'],readback_operations=285,
        source_input_audit_scope=info['counts'],new_scientific_or_remote_operations=0,weights_promoted=False,scientific_acceptance=False,
        scientific_status='response_match_not_established',cumulative_training_unchanged=175667)
    write_new(OUT/'REPORT_DATA.json',json_text(report_data))
    sources=dict(schema='n01/wp3-015-report-sources/v1',UTC=datetime.now(timezone.utc).isoformat(),
        numerical_evidence='saved local JSON only; no binary scientific files opened',inputs=base.INPUTS,
        reused_helper=dict(path=HELPER.relative_to(ROOT).as_posix(),bytes=HELPER.stat().st_size,sha256=sha(HELPER)),
        report_source=dict(path=Path(__file__).relative_to(ROOT).as_posix(),bytes=Path(__file__).stat().st_size,sha256=sha(Path(__file__))))
    write_new(OUT/'SOURCES_SHA256.json',json_text(sources))
    outputs={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()}
    manifest=dict(schema='n01/wp3-015-report-manifest/v1',UTC=datetime.now(timezone.utc).isoformat(),inputs=base.INPUTS,outputs=outputs,
        figures=4,figure_formats=['PNG','SVG'],CSV_encoding='UTF-8 with BOM',report_source=sources['report_source'],
        validation_separate_file='VALIDATION.json',render_visual_QA='pending root browser rendering')
    write_new(OUT/'report_manifest.json',json_text(manifest))
    receipt=dict(schema='n01/wp3-015-report-build/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
        source_sha256=sha(Path(__file__)),outputs=outputs,manifest_sha256=sha(OUT/'report_manifest.json'),inputs=base.INPUTS,
        new_NPZ_CP_NN_AD_G_random_TRAINDEV_GPU_CDF_loss_SSH=0,scientific_acceptance=False,weights_promoted=False,
        HTML_visual_QA='root representative render pending',report=str(OUT/'REPORT.html'))
    write_new(HERE/'REPORT_BUILD_RECEIPT.json',json_text(receipt))
    print(json.dumps(dict(passed=True,outputs=len(outputs),source_JSON=len(base.INPUTS),report=str(OUT/'REPORT.html')),ensure_ascii=False))


if __name__=='__main__':
    main()
