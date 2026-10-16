"""One033 evidence panel from saved JSON/JSONL only."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]/'receipts/response033_20261017/completion_01/run_01'
os.environ['MPLCONFIGDIR'] = str(ROOT/'mpl_cache')
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np
from PIL import Image

font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
    'axes.titlesize':14,'axes.labelsize':12,'figure.titlesize':19,'axes.spines.top':False,
    'axes.spines.right':False,'axes.edgecolor':'#9FAAB4','figure.facecolor':'white','axes.facecolor':'white'})

result_path = SOURCE/'RESULT.json'
metrics_path = SOURCE/'metrics.jsonl'
result = json.loads(result_path.read_text(encoding='utf8'))
rows = [json.loads(line) for line in metrics_path.read_text(encoding='utf8').splitlines()]
risk_rows = [row for row in rows if row['bank_role'] == 'independent_TRAIN_risk']
states = ['full','shape_only','quarter','sixteenth','mean_only','charge_only','local_only']
labels = {'full':'full（全组）','shape_only':'shape_only','quarter':'quarter（1/4 步）',
    'sixteenth':'sixteenth（1/16 步）','mean_only':'mean_only','charge_only':'charge_only','local_only':'local_only'}
baseline = {row['bank_index']:row for row in risk_rows if row['state'] == 'baseline'}
assert sorted(baseline) == list(range(1,9))
totals = {}
B_summary = {}
for state in ['baseline']+states:
    selected = [row for row in risk_rows if row['state'] == state]
    assert len(selected) == 8
    rms = np.array([row['head_scales']['B_voltage_RMS']['RMS'] for row in selected])
    B_summary[state] = dict(bank_indices=[row['bank_index'] for row in selected],bank_RMS_V=rms.tolist(),
        mean_bank_RMS_V=float(rms.mean()),MCSE_across_bank_RMS_V=float(rms.std(ddof=1)/np.sqrt(8)))
    if state != 'baseline':
        saved = result['independent_risk_paired_deltas'][state]['total']
        delta = np.array([row['risk_scalars']['total']-baseline[row['bank_index']]['risk_scalars']['total'] for row in selected])
        assert np.isclose(delta.mean(),saved['mean_delta'],rtol=0,atol=1e-12)
        assert np.isclose(delta.std(ddof=1)/np.sqrt(8),saved['MCSE'],rtol=0,atol=1e-12)
        totals[state] = saved
base_RMS = B_summary['baseline']['mean_bank_RMS_V']
for item in B_summary.values():
    item['ratio_of_bank_means_to_baseline'] = item['mean_bank_RMS_V']/base_RMS

fig,(left,right) = plt.subplots(1,2,figsize=(13.3,6.4),gridspec_kw={'width_ratios':[1.40,1.]})
fig.subplots_adjust(left=.035,right=.975,top=.73,bottom=.24,wspace=.19)
fig.suptitle('033：首步 shape 参数组 Adam 更新出现过冲',y=.98,fontweight='bold',color='#223746')
fig.text(.5,.895,'shape_only：独立总风险 +0.21855；B 电压 RMS 为 baseline 的 2.46×',
    ha='center',fontsize=13,color='#AE343D')

left.axis('off')
left.set_title('独立银行 Δ 总风险（state − baseline，越低越好）',pad=18)
table_rows = [[labels[state],f"{totals[state]['mean_delta']:+.5f} ± {totals[state]['MCSE']:.5f}",
               '增加' if totals[state]['descriptive_3MCSE_direction']=='increase' else '未确定'] for state in states]
table = left.table(cellText=table_rows,colLabels=['干预状态','均值 ± MCSE','3MCSE 方向'],
    cellLoc='center',colWidths=[.34,.45,.21],bbox=[0.,.015,1.,.98])
table.auto_set_font_size(False);table.set_fontsize(12)
for (row,column),cell in table.get_celld().items():
    cell.set_edgecolor('#DCE3E8');cell.set_linewidth(.6)
    if row == 0:
        cell.set_facecolor('#EAF0F5');cell.get_text().set_fontweight('bold');cell.get_text().set_color('#223746')
    elif row in (1,2):
        cell.set_facecolor('#FCECEC');cell.get_text().set_color('#AE343D');cell.get_text().set_fontweight('bold')
    else:
        cell.set_facecolor('#F7F9FA' if row%2 else 'white');cell.get_text().set_color('#45535E')
left.text(.5,-.15,'8 个独立风险银行；MCSE 不是 95% 置信区间。\n首个训练银行已排除；“未确定”即 unresolved。',
    transform=left.transAxes,ha='center',va='top',fontsize=11,color='#35424E',linespacing=1.6)

right_states = ['baseline','shape_only','full','mean_only']
colors = ['#2878A6','#C94747','#D8796D','#8B9BA8']
x = np.arange(4)
means = np.array([B_summary[state]['mean_bank_RMS_V'] for state in right_states])*1000
errors = np.array([B_summary[state]['MCSE_across_bank_RMS_V'] for state in right_states])*1000
right.bar(x,means,.63,color=colors,yerr=errors,capsize=3,error_kw={'lw':1.,'color':'#223746'},zorder=2)
for index,state in enumerate(right_states):
    values = np.array(B_summary[state]['bank_RMS_V'])*1000
    right.scatter(index+np.linspace(-.10,.10,8),values,s=13,color='#283943',alpha=.65,zorder=3)
    right.text(index,means[index]+.20,f'{means[index]:.3f} mV\n{B_summary[state]["ratio_of_bank_means_to_baseline"]:.2f}×',
        ha='center',va='bottom',fontsize=11,color='#223746',linespacing=1.45)
right.set(title='B 物理电压 RMS：8 银行等权均值',xticks=x,xticklabels=right_states,
    ylabel='B 电压 RMS（mV）',ylim=(0,6.15))
right.tick_params(axis='x',labelsize=10)
right.set_axisbelow(True);right.grid(axis='y',color='#E2E8EC',lw=.7)
right.text(.5,-.15,'每银行先跨父事件 / 基 / 时刻求 RMS，再取银行均值。\n点为各银行 RMS；误差线为银行均值 ± MCSE。',
    transform=right.transAxes,ha='center',va='top',fontsize=10.5,color='#35424E',linespacing=1.6)
fig.text(.5,.035,'一次 Adam 步的参数组干预；3MCSE 方向仅作描述，不等于解释晚期失稳或长期拟合。\n银行是 MC 抽样组，非独立采集；mean_only 指参数组干预，不是纯输出均值干预。',
    ha='center',fontsize=10.5,color='#35424E',linespacing=1.6)
figure = ROOT/'first_step_shape_overshoot.png'
fig.savefig(figure,dpi=180,bbox_inches='tight',pad_inches=.2)
plt.close(fig)
with Image.open(figure) as im:
    pixels = list(im.size)
receipt = dict(schema='n01/response033-saved-evidence-plot/v1',status='rendered_pending_visual_QA',
    figure=str(figure),figure_bytes=figure.stat().st_size,figure_pixels=pixels,figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
    generator=str(Path(__file__)),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    inputs=[dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (result_path,metrics_path)],
    total_risk_deltas=totals,B_physical_RMS=B_summary,MCSE_is_not_95percent_CI=True,
    statistical_scope=dict(risk_banks=list(range(1,9)),first_training_bank_excluded=True,
        risk_delta='Saved total risk state-minus-baseline paired by bank; mean and sample_SD/sqrt8 crosschecked against RESULT.',
        risk_direction='Saved descriptive3MCSE direction, not a scientific acceptance test.',
        B_field='head_scales.B_voltage_RMS.RMS',
        B_statistic='Each bank RMS is across parents/bases/time. Plot arithmetic mean across8 banks; this is not the head_scales mean field or a fresh waveform calculation.',
        RMS_ratio='Ratio of the two arithmetic means across the same8 banks.',
        intervention='One saved Adam parameter-group step; no continued training or explanation of late instability/long-term fit.'),
    raw_or_CP_reads=0,new_NN_calls=0,new_AD_queries=0,new_optimizer_steps=0,new_observation_calls=0,
    visual_QA='pending_view_image')
(ROOT/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(figure=str(figure),pixels=pixels,bytes=figure.stat().st_size,
    full_total_delta=totals['full']['mean_delta'],shape_total_delta=totals['shape_only']['mean_delta'],
    baseline_B_RMS_V=base_RMS,shape_B_RMS_V=B_summary['shape_only']['mean_bank_RMS_V'],
    shape_B_ratio=B_summary['shape_only']['ratio_of_bank_means_to_baseline']),indent=2))
