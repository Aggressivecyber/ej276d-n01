"""016 Chinese report and English figures from accepted local JSON only."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
OUT=ROOT/'reports/wp3_016_20261015';STAGE=ROOT/'receipts/wp3_016_20261015'
ART=ROOT/'runs/wp3_016_20261015/selected_collected_01/runs/primary/artifacts'
CPU=ROOT/'runs/wp3_016_20261015/readback_collected_01'
TMP=Path('E:/GAN/runtime_tmp/wp3_016_report');TMP.mkdir(parents=True,exist_ok=True)
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(TMP)
os.environ.update(MPLCONFIGDIR=str(TMP/'matplotlib'),PYTHONDONTWRITEBYTECODE='1')
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True
HELPER=ROOT/'receipts/wp2_014_20261015/postrun_report_impl_01/build_report.py'
spec=importlib.util.spec_from_file_location('mature_JSON_report_helpers',HELPER);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=OUT;base.TMP=TMP
for k in ['TEMP','TMP','TMPDIR']:os.environ[k]=str(TMP)
os.environ['MPLCONFIGDIR']=str(TMP/'matplotlib')
plt,np=base.plt,base.np
read,write,json_text,figure=base.read_json,base.write_new,base.json_text,base.figure_outputs
def sha(p):
 assert p.suffix.lower() not in {'.npz','.npy','.pt','.pth'}
 return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p))
def native(path,inventory):
 j=read(path);key=path.relative_to(ART.parents[2]).as_posix()
 assert record(path)==inventory['whole_remote_inventory'][key]
 return j
def tablefile(name,rows):
 assert rows;keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/'tables'/name).open('x',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def close(a,b):return math.isclose(a,b,rel_tol=2e-11,abs_tol=2e-13)

def main():
 final=read(STAGE/'postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json')
 selection=read(STAGE/'postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json')
 assert final['passed'] and not final['blockers'] and not final['provisional'] and final['selected_research_reference']=='Q1'
 assert selection['development_selection_final'] and selection['independent_raw_saved_array_readback_complete']
 assert not selection['scientific_acceptance'] and not selection['weights_promoted'] and selection['selected_charge_enabled'] and not selection['selected_mean_adapter_enabled']
 assert not OUT.exists(),'Preserve existing reports; no overwrite'
 cfg=read('wp3_016_config.json');T=read('CALIBRATED_TOLERANCES_WP1_013.json')
 inventory=read(STAGE/'collection_01/REMOTE_FULL_INVENTORY.json');collection=read(STAGE/'collection_01/COLLECTION_RECEIPT.json')
 cpu=read(CPU/'artifacts/ARRAY_READBACK.json');cpu_exit=read(CPU/'supervisor/exit_receipt.json')
 native_exit=read(ART.parent/'supervisor/exit_receipt.json');complete=native(ART/'COMPLETE.json',inventory)
 native_closure=read(STAGE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json')
 cpu_collection=read(STAGE/'readback_collection_01/COLLECTION_RECEIPT.json');cpu_recon=read(STAGE/'readback_registration_01/CPU_COMPLETION_RECONCILIATION.json')
 calibration=native(ART/'CHARGE_INITIALIZATION.json',inventory);roles=read('WP3_016_DATA_ROLES.json');info=read('INPUT_INFORMATION_AUDIT.json')
 read('WP3_016_SOURCE_MANIFEST.json');read('docs/WP3_016_RUNTIME_FIX02.json')
 assert cpu['passed'] and cpu['counters']==final['actual_CPU_counts'] and cpu['operations_completed']==200
 assert complete['counters']==cfg['budget_caps'] and complete['operations_completed']==94249 and native_exit['exit_code']==cpu_exit['exit_code']==0
 assert native_closure['cumulative_G']==179667
 cells=selection['cells'];assert len(cells)==6
 wave_rows=[];components=[]
 for c in cells:
  wave=c['components']['wave'];row=dict(seed=c['seed'],hypothesis=c['hypothesis'])
  for arm in ['A','M0','initial','final']:
   row[arm+'_wave_mean2bank']=wave[arm]
   for b in [0,1]:row[arm+f'_bank{b}_wave']=wave['bank_values'][arm][b]
  row.update(improvement_vs_A_percent=100*c['wave_relative_improvement_vs_A'],improvement_vs_M0_percent=100*c['wave_relative_improvement_vs_M0'],
   dual_reference_eligible=c['eligible_cell'],usefulness_vs_A_2percent=c['wave_usefulness_2percent'],absolute_T_all_positive_defined=c['absolute_T_all_positive_defined'])
  wave_rows.append(row)
  for name,v in c['components'].items():
   r=dict(seed=c['seed'],hypothesis=c['hypothesis'],component=name,T=v['T'],status=v['status'],A=v['A'],M0=v['M0'],charge_on_initial=v['initial'],Q1_final=v['final'],
    final_minus_A=v['delta_final_minus_A'],final_minus_M0=v['delta_final_minus_M0'],final_over_T=v['absolute_distance_over_T'],absolute_within_T=v['absolute_within_T'])
   for arm,values in v['bank_values'].items():
    for b,x in enumerate(values):r[f'{arm}_bank{b}']=x
   components.append(r)
 assert len(components)==132 and sum(r['status']=='pending_not_calibrated' for r in components)==42
 assert all(r['status']=='noninferior_to_both_references' for r in components if r['T'] is not None and r['T']>0)

 # Preserve the mature summary organization for all four actual016 arms.
 conditions=[];trends=[];flags=[];population=[]
 features=['Q_Vns','front_Vns','tail_Vns','PSD_raw','PSD_guarded','peak_voltage_V','rise10_90_ns','FWHM_ns']
 for pair in cpu['pairs']:
  trial,snapshot,arm,h=[pair[k] for k in ['trial','snapshot','arm','hypothesis']]
  for b in [0,1]:
   report=native(ART/trial/snapshot/f'bank{b}_{arm}_{h}_REPORT.json',inventory)
   ident=dict(trial=trial,seed=int(trial[4]),snapshot=snapshot,arm=arm,hypothesis=h,bank=b)
   q=report['output_Q_conditions'];assert q['all_rows_in_exactly_one_bin'] and q['minimum_support']==16
   assert sum(z['fake_rows'] for z in q['bins'])==26372 and sum(z['real_rows'] for z in q['bins'])==21047
   for z in q['bins']:
    r=dict(**ident,output_Q_bin=z['bin'],fake_rows=z['fake_rows'],real_rows=z['real_rows'],supported=z['supported'],PSD_W1=z['defined_PSD_W1']['value'],conditional_T='not_calibrated')
    for f in features:
     for domain in ['fake','real']:
      values=z[domain][f]
      for statistic in ['defined_rows','undefined_rows','mean','std_population']:r[f'{domain}_{f}_{statistic}']=values[statistic]
      for quantile,value in values['quantiles'].items():r[f'{domain}_{f}_{quantile}']=value
    r['rise_mean_difference_ns']=r['fake_rise10_90_ns_mean']-r['real_rise10_90_ns_mean']
    r['FWHM_mean_difference_ns']=r['fake_FWHM_ns_mean']-r['real_FWHM_ns_mean'];conditions.append(r)
   tr=report['source_input_trends'];assert tr['parent_count']==6593 and tr['descriptors_are_audit_only_not_network_inputs']
   for descriptor,responses in tr['correlations'].items():
    for statistic,values in responses.items():
     for f,corr in values.items():trends.append(dict(**ident,descriptor=descriptor,response_statistic=statistic,response_feature=f,Pearson=corr['value'],defined_parents=corr['parents'],kind=corr['kind'],physical_response_identified=False))
   p=report['population'];assert p['fake_rows']==26372 and p['real_rows']==21047 and p['all_population_rows_retained']
   for domain in ['fake','real']:
    for flag,counts in p[domain+'_flags'].items():flags.append(dict(**ident,domain=domain,flag=flag,**counts,status='saved_report_value'))
   for flag in p['real_unknown_flags']:flags.append(dict(**ident,domain='real',flag=flag,count=None,denominator=21047,rate=None,status='unknown_not_zero'))
   for category in ['integrals','morphology']:
    for f,domains in p[category].items():
     for domain in ['fake','real']:
      s=domains[domain];population.append(dict(**ident,category=category,feature=f,domain=domain,rows=s['rows'],defined_rows=s['defined_rows'],undefined_rows=s['undefined_rows'],mean=s['mean'],std_population=s['std_population'],**s['quantiles']))
 assert len(conditions)==384 and len(trends)==6912 and all(r['supported'] for r in conditions)
 charge=[]
 for c in cpu['charge_banks']:
  sem=native(ART/c['trial']/c['snapshot']/'CHARGE_SEMANTICS.json',inventory)['banks'][c['bank']]
  assert close(sem['theory_sample_mean_MCSE_RMS_Vns'],c['independent_random_mean_theory_MCSE_RMS_Vns'])
  charge.append(dict(**c,a_Vns=sem['a_Vns'],b=sem['b'],conditional_CV_mean=sem['conditional_CV_mean'],
   expected_A_increment_RMS_ADC_LSB=c['independent_expected_A_increment_RMS_V']*16383,
   producer_realized_increment_RMS_V=sem['realized_increment_RMS_V'],empirical_q_mean_MCSE_RMS_Vns=sem['finite_MC_mean_MCSE_RMS_Vns'],
   centered_empirical_mean_MCSE_RMS_Vns=sem['draw_centered_mean_MCSE_RMS_Vns']))
 moments=[dict(trial=p['trial'],snapshot=p['snapshot'],arm=p['arm'],hypothesis=p['hypothesis'],parents=p['parents'],
  native_draws_per_bank=4,derived_draws=8,covariance_denominator=7,empirical_hard_wave_rank_bound=7,
  spread_RMS_V=p['hard_wave_sqrt_mean_sample_variance_V'],derived8_mean_MCSE_RMS_V=p['hard_wave_RMS_conditional_mean_MCSE_V'],
  native4_bank0_MCSE_RMS_V=p['r4_by_bank'][0]['waveform_MCSE_RMS_V'],native4_bank1_MCSE_RMS_V=p['r4_by_bank'][1]['waveform_MCSE_RMS_V'],
  complete_case_scalar_parents=p['complete_case_scalar_parents'],covariance_SHA256=p['r8_scalar_covariance_sha256'],centered_factor_SHA256=p['centered_wave_factor_sha256']) for p in cpu['pairs']]
 params=[]
 for trial,snapshot,label in [('seed0_Q1','initial_evaluation','Shared initial'),('seed0_Q1','final_evaluation','Final seed0'),('seed1_Q1','final_evaluation','Final seed1')]:
  g=[c for c in charge if c['trial']==trial and c['snapshot']==snapshot];assert len(g)==2
  params.append(dict(label=label,a_Vns=g[0]['a_Vns'],b=g[0]['b'],expected_increment_RMS_ADC_LSB=g[0]['expected_A_increment_RMS_ADC_LSB'],
   mean_conditional_CV_percent=100*g[0]['conditional_CV_mean'],theory_mean_MCSE_RMS_Vns=g[0]['independent_random_mean_theory_MCSE_RMS_Vns'],
   empirical_mean_MCSE_RMS_Vns=sum(c['empirical_q_mean_MCSE_RMS_Vns'] for c in g)/2))
 ratios=final['key_summary']['absolute_T_ratio_ranges']
 absolute_rows=[dict(component=k,**v) for k,v in ratios.items()]
 condition_summary=[]
 for c in cells:
  g=[r for r in conditions if r['trial']==f"seed{c['seed']}_Q1" and r['arm']=='final' and r['hypothesis']==c['hypothesis']]
  assert len(g)==16
  condition_summary.append(dict(seed=c['seed'],hypothesis=c['hypothesis'],supported_bins=16,minimum_fake_rows=min(r['fake_rows'] for r in g),minimum_real_rows=min(r['real_rows'] for r in g),
   PSD_W1_max=max(r['PSD_W1'] for r in g),rise_signed_worst_ns=max(g,key=lambda r:abs(r['rise_mean_difference_ns']))['rise_mean_difference_ns'],
   FWHM_signed_worst_ns=max(g,key=lambda r:abs(r['FWHM_mean_difference_ns']))['FWHM_mean_difference_ns']))
 OUT.mkdir(parents=True);(OUT/'figures').mkdir();(OUT/'tables').mkdir()
 tables={'ALL6_WAVE.csv':wave_rows,'ALL6_COMPONENTS.csv':components,'ABSOLUTE_T.csv':absolute_rows,'CHARGE_BANKS.csv':charge,'CHARGE_PARAMETERS.csv':params,
  'OUTPUT_Q_CONDITIONS.csv':conditions,'OUTPUT_Q_FINAL_SUMMARY.csv':condition_summary,'SOURCE_INPUT_TRENDS.csv':trends,'ALL_FLAGS.csv':flags,'POPULATION_FEATURES.csv':population,'DERIVED8_MOMENTS.csv':moments}
 for name,rows in tables.items():tablefile(name,rows)

 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
  'svg.fonttype':'none','svg.hashsalt':'N01-WP3-016','axes.axisbelow':True,'axes.unicode_minus':False})
 labels=['S0 fast','S0 slow','S0 common','S1 fast','S1 slow','S1 common'];x=np.arange(6)
 colors={'A':'#778da4','M0':'#246ba5','initial':'#c98b28','final':'#20836a'}
 fig,axs=plt.subplots(1,2,figsize=(13.8,5.4),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
 for k,arm in enumerate(['A','M0','initial','final']):axs[0].bar(x+(k-1.5)*.19,[r[arm+'_wave_mean2bank'] for r in wave_rows],width=.18,color=colors[arm],label={'initial':'Charge-on initial','final':'Q1 final'}.get(arm,arm))
 axs[0].set(xticks=x,xticklabels=labels,ylabel='Hard-wave distance (mean of two banks)',ylim=(0,.175),title='Four paired arms on the fixed libraries')
 axs[0].legend(ncol=2,frameon=False,fontsize=9);axs[0].tick_params(axis='x',labelrotation=15);axs[0].grid(axis='y',color='#e5eaf0')
 gains=[r['improvement_vs_M0_percent'] for r in wave_rows];bars=axs[1].bar(x,gains,color=['#20836a' if v>=0 else '#b45f43' for v in gains])
 axs[1].bar_label(bars,labels=[f'{v:.4f}' for v in gains],fontsize=9,padding=4);axs[1].axhline(0,color='#435062',linewidth=1)
 axs[1].set(xticks=x,xticklabels=labels,ylabel='Wave-distance improvement vs M0 (%)',title='3 improved cells; 3 slightly worse')
 axs[1].set_ylim(-.065,.115);axs[1].tick_params(axis='x',labelrotation=20);axs[1].grid(axis='y',color='#e5eaf0')
 fig.suptitle('WP3/016: mean12 improvement vs M0 = 0.009976% (fixed-library development result)',fontsize=13)
 figure(fig,'01_paired_wave_and_tiny_gain')

 calibrated=list(ratios);matrix=np.array([[c['components'][k]['absolute_distance_over_T'] for c in cells] for k in calibrated])
 fig,ax=plt.subplots(figsize=(12.2,9),layout='constrained')
 image=ax.imshow(np.log10(matrix),aspect='auto',cmap='RdYlGn_r',norm=base.TwoSlopeNorm(vmin=-.6,vcenter=0,vmax=1.6))
 ax.set(xticks=x,xticklabels=labels,yticks=np.arange(len(calibrated)),yticklabels=calibrated,title='Absolute T: 7 of 15 calibrated components fail in every cell')
 for i in range(15):
  for j in range(6):ax.text(j,i,f'{matrix[i,j]:.5g}',ha='center',va='center',fontsize=9,color='#172638')
 fig.colorbar(image,ax=ax,label='Log10(final distance / frozen T)');ax.set_xlabel('Ratios above 1 fail; seven uncalibrated flag components remain unresolved')
 figure(fig,'02_absolute_T_ratios')

 fig,axes=plt.subplots(2,2,figsize=(12.2,8.8),layout='constrained');px=np.arange(3);pl=[p['label'] for p in params]
 for ax,key,title,ylabel in [(axes[0,0],'a_Vns','Learned a','a (V software ns)'),(axes[0,1],'b','Learned b','b (dimensionless)'),
  (axes[1,0],'expected_increment_RMS_ADC_LSB','Expected increment over all 384 coordinates','RMS / ADC LSB')]:
  bars=ax.bar(px,[p[key] for p in params],color=['#c98b28','#246ba5','#20836a']);ax.bar_label(bars,labels=[f'{p[key]:.4g}' for p in params],padding=4,fontsize=10)
  ax.set(xticks=px,xticklabels=pl,title=title,ylabel=ylabel);ax.margins(y=.22);ax.grid(axis='y',color='#e5eaf0')
 axes[1,0].axhline(1,color='#606b79',linestyle='--',linewidth=1)
 ax=axes[1,1]
 ax.bar(px-.18,[p['theory_mean_MCSE_RMS_Vns'] for p in params],width=.34,color='#246ba5',label='Theory: sqrt(sum v)/4')
 ax.bar(px+.18,[p['empirical_mean_MCSE_RMS_Vns'] for p in params],width=.34,color='#c98b28',label='Empirical four-draw MCSE')
 ax.set(xticks=px,xticklabels=pl,title='Charge sample-mean precision',ylabel='RMS MCSE (V software ns)');ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',color='#e5eaf0')
 fig.suptitle('Positive mean-corrected charge: fixed-A saved-row quantities',fontsize=14);figure(fig,'03_charge_parameters_and_precision')

 fm=[next(p for p in moments if p['trial']==f"seed{c['seed']}_Q1" and p['arm']=='final' and p['hypothesis']==c['hypothesis']) for c in cells]
 fig,axes=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
 axes[0].bar(x-.18,[math.sqrt((p['native4_bank0_MCSE_RMS_V']**2+p['native4_bank1_MCSE_RMS_V']**2)/2) for p in fm],width=.34,label='Native 4 draws (bank RMS average)',color='#c98b28')
 axes[0].bar(x+.18,[p['derived8_mean_MCSE_RMS_V'] for p in fm],width=.34,label='Derived 8 draws (4 + 4)',color='#246ba5')
 axes[0].set(xticks=x,xticklabels=labels,title='Final hard-wave Monte Carlo precision',ylabel='Conditional-mean RMS MCSE (V)');axes[0].tick_params(axis='x',labelrotation=20);axes[0].legend(frameon=False,fontsize=9)
 values=[100*p['complete_case_scalar_parents']/6593 for p in fm];bars=axes[1].bar(x,values,color='#20836a')
 axes[1].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9);axes[1].set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Parents with all 9 scalars finite (%)',title='Finite support is retained explicitly');axes[1].tick_params(axis='x',labelrotation=20)
 for ax in axes:ax.grid(axis='y',color='#e5eaf0')
 fig.suptitle('Derived8 uses denominator 7; empirical hard rank <= 7 is not a preADC rank claim',fontsize=13)
 figure(fig,'04_saved_hard_wave_precision')

 lines=[]
 def heading(s):lines.extend(['## '+s,''])
 def para(s):lines.extend([s,''])
 def table(headers,rows):
  lines.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
  lines.extend('| '+' | '.join('—' if x is None else str(x) for x in row)+' |' for row in rows);lines.append('')
 def image(name,caption):para(f'![{name}](figures/{name}.png)');para(caption)
 scores=selection['reference_scores_mean12'];candidate=selection['candidates']['Q1'];prior=cfg['runtime_repair']['prior_actual_cost'];aggregate=cfg['runtime_repair']['aggregate_budget_caps']
 lines.extend(['# N01 WP3 / 016：冻结 M0 后的正值电荷涨落短试',''])
 para('**结论：最终研究参考选 Q1，均值适配关闭、电荷分支开启；权重未作科学晋级，响应匹配仍为 response_match_not_established。** 两个 seed 各完成 2000G，保存数组独立 CPU QA 已通过。016 消耗 WP3 最后两次短试，WP3 短试预算为 6/6、总短试为 10/12；六个工作包尚未完成，下一步进入 WP4 已知响应恢复。')
 heading('1. 固定规则与小幅开发增益')
 para(f'共 2 seed×3 假设×2 库=12 个原 wave 距离；每个 seed×假设先对两库取均值得到 6 个单元，再对 6 单元等权平均（与 12 个原距离等权平均相同）；不 pool8 重算距离。A / M0 / charge-on initial / Q1 final 的 mean12 依次为 {scores["A"]:.17g} / {scores["M0"]:.17g} / {scores["initial"]:.17g} / {scores["final"]:.17g}。Q1 比配对 M0 低 {candidate["mean12_absolute_improvement_vs_M0"]:.17g}，相对改善 {100*candidate["mean12_relative_improvement_vs_M0"]:.9f}%。')
 para('Q1 在 ALL6 中对每个必需且正值有定义 T 的组件，均满足 final−A≤T 且 final−M0≤T；90 个组件单元、180 个参考比较全部非劣。mean12 改善超过预先冻结的 1e-12，故按原规则替换研究参考。六个单元中对 M0 为 3 个改善、3 个略差；该小幅固定库效应没有新增重采样或置信区间，不能解释为跨采集稳定收益。')
 table(['seed / 假设','A','M0','初始电荷开启','Q1 final','相对 M0 改善 %','相对 A 改善 %'],[[f"{r['seed']} / {r['hypothesis']}",*[f"{r[a+'_wave_mean2bank']:.9f}" for a in ['A','M0','initial','final']],f"{r['improvement_vs_M0_percent']:.6f}",f"{r['improvement_vs_A_percent']:.6f}"] for r in wave_rows])
 para('ALL6 相对 A 至少 2% 的用途门为 6/6。用途、研究参考替换、绝对 T 和科学响应确认分别判定；当前选择不增加科学接受结论。两个库原始分数与全部组件见 ALL6_WAVE.csv 和 ALL6_COMPONENTS.csv。')
 image('01_paired_wave_and_tiny_gain','图 1：四个配对 arm 与相对 M0 的微小变化。柱图使用保存 JSON 的 mean2bank；右图逐单元变化和总体 mean12 不是新的统计检验。')
 heading('2. 冻结实验与数据角色')
 para('只训练两个全局参数 theta_a / theta_b。原 A、005、每 seed 的 015 M0 final 形状头、C1 积分尺度、观察流程及 013 T 冻结；均值适配器关闭，不继承 015 优化器。新 AdamW lr=.01、weight_decay=0、clip5，每步一次 backward，共 4000G/AD，无新增 AD 探针。形状、005、扩展和投影前缀沿用015，电荷流单独登记；模型内部不创建随机数。')
 para('SOURCE TRAIN 6593 个父事件、REAL TRAIN 21047 行、422 个逻辑组；重复 MC draw 不增加物理事件数。每库 4 draw、两库、fast / slow / common_level 三假设。初始 A、M0 关闭电荷、初始开启电荷共享一次 forward 和电子学；结束后只评价 final。48 份 HARD/REPORT 保留全部行及低 Q、缺失和 flags，未读原 DEV 波形。SOURCE Q 是软件积分描述量，不是已标定入射能量；SOURCE 趋势与输出 Q 分箱不提供独立物理标签。')
 para('电荷规律：s²=log1p(a/mu+b)，q_sig=mu·exp(s·Z−s²/2)，a=q_reference·softplus(theta_a)、b=softplus(theta_b)，固定 q_reference=3.7103998046755806 V软件ns。mu 来自每一实际 A 输出的 canonical [110:366) 积分，全部严格正，不设 floor。E[q_d]=mu_d、v_d=a·mu_d+b·mu_d²。信号组装为原 M0 加 A·expm1(log_multiplier)，B(x)z_shape 的幅度仍取原 A，不乘 q_sig；电荷关闭分支直接返回原 M0 tensor 字节。')
 heading('3. 绝对 T 与未标定组件')
 para('每个假设有 15 个正值 T 和 7 个未标定 flag 组件。未标定状态、观测到零发生率和 PASS 分开记录；缺失必需正 T 组件会使候选 pending。绝对 T 全部通过的单元数为 0/6：每个单元均有同样 7 个失败组件。T 来自已冻结的人工自距离规则，hardware_calibrated=false。')
 table(['组件','final/T 最小','final/T 最大','绝对失败单元 / 6'],[[k,f"{v['min']:.6g}",f"{v['max']:.6g}",v['failing_cells']] for k,v in ratios.items()])
 para('持续失败集中于 QPSD、peak_voltage、rise10_90、FWHM、baseline_mean 以及 rise_undefined / rise_prepeak_censored 两个 rate。baseline_mean 约 33.27T，峰高约 6.69T、rise 约 7.40T、FWHM 约 4.82T。不能把这些残差共同归因为电荷方差。PSD_raw / guarded 的最大 ratio 为 0.9998749634，保留该结果及原停止规则，不据近阈值另加短试。')
 image('02_absolute_T_ratios','图 2：六单元×十五正 T 组件。显示值作数值格式化，精确值在表格与 REPORT_DATA.json；七个未标定 flag 不进入 ratio 热图。')
 heading('4. 学到的电荷参数与校准')
 para(f'一次初始化覆盖全部 6593 SOURCE 父事件及 384 坐标，含严格零 prefix100。producer 保存 S0={calibration["S0_V2"]:.17g} V²，Sminus={calibration["Sminus_V2_per_Vns"]:.17g} V²/(V软件ns)；a0={params[0]["a_Vns"]:.17g} V软件ns，b0={params[0]["b"]:.17g}，目标为 1 ADC LSB=1/16383 V。解析 a0=δ²/(2Sminus)、b0=δ²/(2S0)，未按有限 MC 重新调参。CPU 在已保存的实际重复 A 上独立复算预期增量 RMS，不另作 A 前向或校准重拟合。')
 table(['状态','a / V软件ns','b','预期 A 增量 RMS / LSB','平均条件 CV %','理论四 draw 均值 MCSE RMS / V软件ns'],[[p['label'],f"{p['a_Vns']:.10g}",f"{p['b']:.10g}",f"{p['expected_increment_RMS_ADC_LSB']:.6f}",f"{p['mean_conditional_CV_percent']:.6f}",f"{p['theory_mean_MCSE_RMS_Vns']:.9g}"] for p in params])
 para('实际每 draw 的 mu_d / v_d 全部保留。异均值条件下 E[samplevar(q,ddof1)]=mean(v_d)+samplevar(mu_d,ddof1)；理论均值 MCSE=sqrt(sum(v_d))/4。经验 sqrt(samplevar(q)/4)、逐 draw q−mu 的二阶矩 mean((q−mu)²)（分母4）及样本方差（分母3）分别核对，不用均值或第一 draw 替代模型分母。数值 mu spread 单列，不解释为物理方差。有限四 draw 矩不是精确总体实现。')
 image('03_charge_parameters_and_precision','图 3：保存 JSON 与 CPU 已验收摘要。经验 MCSE 为两个库 RMS 摘要的平均；理论与经验栏分别命名，不作为响应匹配门。')
 heading('5. 独立 CPU 保存数组核验')
 para('112 个原 NPZ 逐文件 SHA 一次、解码一次：48 HARD、48 REPORT、8 baseline100、8 charge moments，共 2169655337 字节。核验 1265856 条硬输出和 210976 条实际 A384 行（81014784 个 float64 坐标）、210976 个 mu/q draw、52744 条父矩记录。24 个两库 hard pair、8 个 charge bank 全部完成；200 entered=completed、400 条 journal，无 inflight/failed/truncated tail。')
 para('独立 QA 包括：savedADC+double baseline+可选 IEEE XOR 逆变换与精确 code roundtrip；父/draw/bank/ID；Q/front/tail/PSD、峰高、rise/FWHM、baseline 与可重建 flags；native4 均值/样本方差/MCSE、9×9 pairwise-finite covariance 与 joint counts；从 4+4 保存行派生8draw并核对分母7合并恒等式。经验硬波形 rank≤7 不代表理论 preADC 秩；不同缺失支持下的 pairwise covariance 不自动保证 PSD。')
 para(f'实际 A 最大同父重复差 {max(c["actual_A_repeat_max_abs_V"] for c in charge):.17g} V，在既有013 1e-7 V 工程门内；mu 最大重复差 {max(c["actual_mu_repeat_max_abs_Vns"] for c in charge):.17g} V软件ns。GPU mu 与独立 CPU canonical sum 的最大差 {max(c["Q_CPU_GPU_absmax_Vns"] for c in charge):.17g}，逐行在预先 γ255 双归约界内；保存 residual 位级等于保存 mu−保存 CPU sum。所有 mu/q 严格正。旧失败运行未保存的差值幅度仍 unknown，不能用本次结果回填旧幅度。')
 table(['证据','独立完成','保留边界'],[
  ['完整经验距离','冻结 producer JSON 的独立选择代数','完整 CDF/loss 未重播'],
  ['hard 保存行','逆变换、行 features/flags、native4/derived8 矩与有限支持','无第二份原 Y 浮点副本'],
  ['charge / A384','实际 A、mu/q/v、数值门、方差和 MCSE 恒等式','原 SOURCE 字节未重 hash；仅保存 digest 重复与跨库一致性'],
  ['随机和模型','来源/随机状态字节 schema 与冻结源码绑定','未重放 Z / RNG、CP tensors、NN、完整随机 preADC signal/carrier'],
  ['统计与物理','冻结 C1 使用链；条件和 flags 完整保留','C1 median 未重拟合；不授予物理响应匹配']])
 image('04_saved_hard_wave_precision','图 4：final 六单元 native4 与 derived8 的 MCSE 和9标量完整支持比例。所有缺失行仍在原分母中，pairwise-finite 矩按各自 joint counts 核验。')
 heading('6. 全人口、条件分箱与 flags')
 para('OUTPUT_Q_CONDITIONS.csv 保留 48 REPORT×8箱=384 条条件记录，全部箱支持完整；SOURCE_INPUT_TRENDS.csv 保留 6912 条输入描述量相关摘要；POPULATION_FEATURES.csv 保留全部积分/形态总体摘要。它们来自已回收 REPORT JSON，未重算原 REAL 或 SOURCE 波形。输出 Q 条件和 Pearson 趋势只作诊断，不将总体 Q 宽度认作真实条件方差，也不从单采集识别全部物理参数。')
 table(['seed / 假设','16库箱最少fake行','最少REAL行','最大条件PSD W1','最差有符号rise差 / ns','最差有符号FWHM差 / ns'],[[f"{r['seed']} / {r['hypothesis']}",r['minimum_fake_rows'],r['minimum_real_rows'],f"{r['PSD_W1_max']:.6g}",f"{r['rise_signed_worst_ns']:.6g}",f"{r['FWHM_signed_worst_ns']:.6g}"] for r in condition_summary])
 para('ALL_FLAGS.csv 保留所有 fake / REAL flags 的 count、原 denominator、rate 和 unknown 状态，包括 Q≤0、PSD undefined/guarded、rise/FWHM 失效或截断、峰位边界/并列、crop 与全局峰、ADC rail / preADC 越界、half-grid 和 floor。不能从 crop256 重建的 full-carrier flags 只核验保存行与计数的一致性。REAL 未知硬件域标记保持 unknown，不写成零。')
 heading('7. 实际执行成本与资源')
 para('最初 bootstrap 准备失败未安装/启动科学计算；随后一次原生运行在首 G 前严格 Q 相等断言失败，G/AD=0 但 NN/随机/CP 成本非零且保留。fix02 使用已有 A 点门并先保存实际逐 draw 证据，在新目录作一次替代执行，两个科学 seed 槽位不增加。没有自动重启或第三个训练候选。')
 table(['计数','旧失败实际','成功替代实际','合计'],[[k,prior[k],complete['counters'][k],aggregate[k]] for k in complete['counters']])
 para(f'成功替代 94249 个 operation 完整，child/supervisor 均退出0；wall {native_exit["elapsed_seconds"]:.3f}s，采样峰值RSS {native_exit["peak_RSS_bytes"]}B、GPU {native_exit["peak_GPU_bytes"]}B、输出 {native_exit["peak_output_bytes"]}B。累计G保持179667；先前已记137，本次root只补3863，未重复加4000。')
 para(f'独立CPU child/supervisor退出0，supervisor wall {cpu_exit["elapsed_seconds"]:.3f}s；supervisor采样owned-tree峰值RSS {cpu_exit["peak_owned_RSS_bytes"]}B，与worker getrusage峰值 {cpu["resources"]["peak_RSS_bytes"]}B是不同口径，分别保留且均低于1536MiB。CPU输出峰值 {cpu_exit["peak_worker_artifact_bytes"]}B；19个小文件共470489B回收并核SHA，CPU没有新增G/AD。')
 para('训练选择性回收99/211文件、47224353B内容，包括6个仅作不透明hash核验的薄checkpoint，0个NPZ；112个raw仍在原远端目录。CPU另在独立sidecar只读原文件并回收小JSON/journal。报告生成只读本地已接受JSON和标准绘图，新增原NPZ/CP/NN/AD/G/RNG、原TRAIN/DEV波形、GPU、CDF/loss和SSH均为0。')
 heading('8. 来源、交付与下一步')
 table(['关键文件','SHA256'],[['冻结 config',sha(ROOT/'wp3_016_config.json')],['冻结 source manifest',sha(ROOT/'WP3_016_SOURCE_MANIFEST.json')],['独立CPU ARRAY_READBACK',sha(CPU/'artifacts/ARRAY_READBACK.json')],['最终独立审核',sha(STAGE/'postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json')],['最终开发选择',sha(STAGE/'postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json')]])
 para('保留 Q1 作为当前开发研究参考，并按主计划进入 WP4 已知响应恢复与可识别性验证。WP3 有限搜索闭合不等于六工作包完成；绝对响应残差、跨条件稳定性和物理响应匹配仍按主计划推进，不因本次小幅分数变化改门或追加短试。')
 para('[机器可读报告](REPORT_DATA.json) · [来源SHA](SOURCES_SHA256.json) · [输出manifest](report_manifest.json) · [结构QA](REPORT_QA.json)')
 for name in tables:para(f'[{name}](tables/{name})')
 for label,path in [('最终独立审核','receipts/wp3_016_20261015/postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json'),('最终开发选择','receipts/wp3_016_20261015/postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json'),('用户主计划','USER_MASTER_PLAN_20261015.txt'),('016配置合同','docs/WP3_016_CONFIGURATION.md')]:para(f'[{label}](../../{path})')
 write(OUT/'REPORT.md','\n'.join(lines).rstrip()+'\n')
 write(OUT/'REPORT.html',base.render_html(lines).replace('N01 WP2 014 损失表示对照报告','N01 WP3 016 正值电荷涨落短试报告'))
 data=dict(schema='n01/wp3-016-report-data/v1',UTC=datetime.now(timezone.utc).isoformat(),development_choice=selection,final_independent_review=final,
  wave_cells=wave_rows,all_components=components,absolute_T=absolute_rows,charge_parameters=params,charge_banks=charge,calibration=calibration,
  output_Q_conditions=conditions,output_Q_final_summary=condition_summary,source_input_trends=trends,all_flags=flags,population_features=population,derived8_moments=moments,
  successful_native_counts=complete['counters'],prior_failed_nonG_cost=prior,aggregate_native_counts=aggregate,native_resources=native_exit,
  CPU_counts=cpu['counters'],CPU_worker_resources=cpu['resources'],CPU_supervisor_resources=cpu_exit,
  collection=collection,CPU_collection=cpu_collection,source_input_audit_counts=info['counts'],cumulative_training_updates=179667,
  new_scientific_operations=0,weights_promoted=False,scientific_acceptance=False,scientific_status='response_match_not_established',six_work_packages_complete=False)
 write(OUT/'REPORT_DATA.json',json_text(data))
 sources=dict(schema='n01/wp3-016-report-sources/v1',inputs=base.INPUTS,report_source=dict(path=Path(__file__).relative_to(ROOT).as_posix(),**record(Path(__file__))),
  mature_helper=dict(path=HELPER.relative_to(ROOT).as_posix(),**record(HELPER)),numerical_evidence='accepted local JSON only; no scientific binary opened')
 write(OUT/'SOURCES_SHA256.json',json_text(sources))

 # Structural QA only. Root owns representative Edge and figure visual QA.
 qa=[]
 def passed(name,**detail):qa.append(dict(name=name,passed=True,**detail))
 assert read(OUT/'REPORT_DATA.json')['development_choice']['selected_research_reference']=='Q1';passed('accepted_final_choice_and_negative_scientific_promotion_boundary')
 assert len(cells)==6 and len(components)==132 and len(conditions)==384 and len(trends)==6912;passed('exact_cell_component_conditional_and_source_trend_dimensions')
 for name,rows in tables.items():
  with (OUT/'tables'/name).open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
  assert len(actual)==len(rows);passed('CSV_'+name,rows=len(rows))
 for p in sorted((OUT/'figures').glob('*.png')):
  b=p.read_bytes();assert b[:8]==b'\x89PNG\r\n\x1a\n';width,height=struct.unpack('>II',b[16:24]);assert width>=2000 and height>=900
  passed('PNG_'+p.name,width=width,height=height)
 for p in sorted((OUT/'figures').glob('*.svg')):
  xml=ET.fromstring(p.read_text('utf8'));texts=[e.text or '' for e in xml.iter() if e.tag.endswith('}text')]
  assert texts and all(not re.search('[\u4e00-\u9fff]',s) for s in texts);passed('SVG_'+p.name,English_text_nodes=len(texts))
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.links=[];self.images=0
  def handle_starttag(self,tag,attrs):
   d=dict(attrs)
   if tag=='a' and 'href' in d:self.links.append(d['href'])
   if tag=='img':self.links.append(d['src']);self.images+=1
 parser=Links();parser.feed((OUT/'REPORT.html').read_text('utf8'));assert parser.images==4
 for link in parser.links:
  if link in ['report_manifest.json','REPORT_QA.json']:continue
  assert (OUT/link).is_file(),link
 passed('HTML_UTF8_images_and_all_local_links',images=4,links=len(parser.links))
 assert '179667' in (OUT/'REPORT.md').read_text('utf8') and '0.009976' in (OUT/'REPORT.md').read_text('utf8');passed('Chinese_report_key_numbers_reopened')
 qa_result=dict(schema='n01/wp3-016-report-structural-QA/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,checks=qa,
  new_scientific_operations=0,real_NPZ_CP_NN_AD_G_RNG_TRAINDEV_GPU_CDF_loss_SSH=0,
  HTML_representative_render='pending root Edge check',figure_visual_QA='pending root actual image inspection',scientific_acceptance=False,weights_promoted=False)
 write(OUT/'REPORT_QA.json',json_text(qa_result))
 outputs={p.relative_to(OUT).as_posix():record(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
 manifest=dict(schema='n01/wp3-016-report-manifest/v1',UTC=datetime.now(timezone.utc).isoformat(),outputs=outputs,inputs=base.INPUTS,
  figures=4,figure_formats=['PNG','SVG'],CSV_encoding='UTF-8 BOM',report_source=sources['report_source'],visual_QA='root pending')
 write(OUT/'report_manifest.json',json_text(manifest))
 receipt=dict(schema='n01/wp3-016-report-build/v1',passed=True,report=str(OUT/'REPORT.html'),manifest_sha256=sha(OUT/'report_manifest.json'),
  QA_sha256=sha(OUT/'REPORT_QA.json'),outputs=outputs,new_scientific_operations=0,scientific_acceptance=False,weights_promoted=False)
 write(HERE/'REPORT_BUILD_RECEIPT.json',json_text(receipt))
 print(json.dumps(dict(passed=True,report=str(OUT/'REPORT.html'),outputs=len(outputs),tables=len(tables),figures=4,QA_checks=len(qa),manifest_sha256=receipt['manifest_sha256']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
