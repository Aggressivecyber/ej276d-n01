from pathlib import Path
import json,os
import numpy as np
from scipy.stats import wasserstein_distance
w=Path(__file__).parent;r=w.parents[2];run=w/'collected/run_01';prior=r/'receipts/response060_20261018/wake_20261018_1605_01/collected/run_01'
load=lambda p:dict(np.load(p,allow_pickle=False));read=lambda p:json.loads(p.read_text());real=load(run/'REAL_ROWS.npz');qfloor=read(run/'DEFINITIONS.json')['q_floor']
rp=real['tail_Vns']/np.maximum(real['Q_Vns'],qfloor);groups=[];population=[];baseline=[]
for seed in range(2):
 original=load(prior/f'seed{seed}_BASE054_INTRINSIC_MEAN_ROWS.npz');oldreport=read(prior/f'seed{seed}_BASE054.json')
 for case in ('WARP068','LEVELWARP'):
  intrinsic=load(run/f'seed{seed}_{case}_INTRINSIC_MEAN_ROWS.npz');assert np.array_equal(original['parent_row'],intrinsic['parent_row']) and np.array_equal(original['intrinsic_peak_index'],intrinsic['intrinsic_peak_index'])
  assert np.max(np.abs(intrinsic['Q_Vns']-original['Q_Vns']))<1e-12
  for h in ('fast','slow','common_level'):
   n=load(run/f'seed{seed}_{case}_{h}_NATIVE_ROWS.npz');b=load(prior/f'seed{seed}_BASE054_{h}_NATIVE_ROWS.npz');fixed=load(run/f'seed{seed}_WARP068_{h}_NATIVE_ROWS.npz')['native_output_Q_bin']
   assert np.array_equal(n['parent_row'],b['parent_row']) and np.array_equal(n['input_Q_Vns'],b['input_Q_Vns'])
   native_delta=n['tail_Vns']-b['tail_Vns'];mean_delta=intrinsic['tail_Vns']-original['tail_Vns'];other_delta=(n['tail_Vns']-intrinsic['tail_Vns'])-(b['tail_Vns']-original['tail_Vns'])
   assert np.max(np.abs(native_delta-mean_delta-other_delta))<1e-12
   psd=n['tail_Vns']/np.maximum(n['Q_Vns'],qfloor);ipsd=intrinsic['tail_Vns']/np.maximum(intrinsic['Q_Vns'],qfloor)
   population.append(dict(seed=seed,case=case,hypothesis=h,rows=len(n['Q_Vns']),native_tail_mean=float(n['tail_Vns'].mean()),intrinsic_tail_mean=float(intrinsic['tail_Vns'].mean()),original_BASE054_intrinsic_tail_mean=float(original['tail_Vns'].mean()),REAL_tail_mean=float(real['tail_Vns'].mean()),
     mean_tail_change_from_BASE054=float(mean_delta.mean()),native_tail_change_from_BASE054=float(native_delta.mean()),remaining_readout_change_from_BASE054=float(other_delta.mean()),mean_tail_decreased_rows=int((mean_delta<0).sum()),mean_tail_delta_quantiles=np.quantile(mean_delta,[0,.01,.1,.5,.9,.99,1]).tolist(),
     intrinsic_tail_W1_REAL=float(wasserstein_distance(intrinsic['tail_Vns'],real['tail_Vns'])),original_BASE054_intrinsic_tail_W1_REAL=float(wasserstein_distance(original['tail_Vns'],real['tail_Vns'])),native_tail_W1_REAL=float(wasserstein_distance(n['tail_Vns'],real['tail_Vns'])),
     native_PSD_W1_REAL=float(wasserstein_distance(psd,rp)),native_Q_PSD_covariance=float(np.cov(n['Q_Vns'],psd,ddof=0)[0,1]),REAL_Q_PSD_covariance=float(np.cov(real['Q_Vns'],rp,ddof=0)[0,1])))
   for grouping,bins in [('input_Q',n['input_Q_bin']),('fixed_WARP068_Q',fixed),('own_output_Q',n['native_output_Q_bin'])]:
    for j in range(8):
     m=bins==j;rm=real['output_Q_bin']==j
     groups.append(dict(seed=seed,case=case,hypothesis=h,group=grouping,bin=j,SOURCE_rows=int(m.sum()),REAL_rows=int(rm.sum()),
       native_tail_mean=float(n['tail_Vns'][m].mean()),intrinsic_tail_mean=float(intrinsic['tail_Vns'][m].mean()),BASE054_intrinsic_tail_mean=float(original['tail_Vns'][m].mean()),REAL_tail_mean=float(real['tail_Vns'][rm].mean()),
       mean_tail_change_from_BASE054=float(mean_delta[m].mean()),native_tail_change_from_BASE054=float(native_delta[m].mean()),remaining_readout_change_from_BASE054=float(other_delta[m].mean()),
       native_PSD_mean=float(psd[m].mean()),intrinsic_PSD_mean=float(ipsd[m].mean()),REAL_PSD_mean=float(rp[rm].mean()),native_PSD_variance=float(psd[m].var()),REAL_PSD_variance=float(rp[rm].var()),
       unstandardized_conditional_PSD_W1=float(wasserstein_distance(psd[m],rp[rm]))))
 for h in ('fast','slow','common_level'):
  m0=read(run/f'seed{seed}_M0.json')[h]['values'];before=oldreport['metrics'][h]['values'];T=read(run/'DEFINITIONS.json')['T_by_hypothesis'][h]
  baseline.append(dict(seed=seed,hypothesis=h,BASE054_QPSD=before['QPSD'],BASE054_tail_W1=before['tail_Vns'],M0_QPSD=m0['QPSD'],M0_tail_W1=m0['tail_Vns'],QPSD_limit=m0['QPSD']+.05*T['QPSD'],tail_limit=m0['tail_Vns']+.05*T['tail_Vns']))
result=dict(scope='Only existing saved scalar rows. Native minus intrinsic combines stochastic response, electronics, baseline subtraction, ADC and peak-based windowing; not a causal noise decomposition. SOURCE/REAL Q groups are unpaired. No proposed candidate evaluated.',population=population,groups=groups,BASE054_native_reference=baseline)
(w/'TAIL_AND_CONDITIONAL_PSD_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
os.environ['MPLCONFIGDIR']=str(w/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True);x=np.arange(1,9)
for case,c in [('WARP068','#0072b2'),('LEVELWARP','#009e73')]:
 z=[a for a in groups if a['seed']==0 and a['case']==case and a['hypothesis']=='fast' and a['group']=='fixed_WARP068_Q']
 axes[0].plot(x,[a['mean_tail_change_from_BASE054'] for a in z],'o-',color=c,label=case+' mean')
 axes[0].plot(x,[a['native_tail_change_from_BASE054'] for a in z],':',color=c,label=case+' native')
 z=[a for a in groups if a['seed']==0 and a['case']==case and a['hypothesis']=='fast' and a['group']=='own_output_Q']
 axes[1].plot(x,[a['native_PSD_mean']-a['REAL_PSD_mean'] for a in z],'o-',color=c,label=case)
 axes[2].plot(x,[a['unstandardized_conditional_PSD_W1'] for a in z],'o-',color=c,label=case)
for ax,title,ylabel in zip(axes,['Tail change from BASE054','Conditional PSD mean residual','Conditional PSD distribution distance'],['Tail integral change (V ns)','Mean PSD minus REAL','Unstandardized PSD W1']):
 ax.axhline(0,color='black',lw=.7);ax.set(title=title,ylabel=ylabel,xlabel='Fixed WARP068 Q bin' if ax is axes[0] else 'Own output Q bin');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('069 tail and conditional PSD | seed0 / fast | existing scalar rows only');fig.savefig(r/'reports/response069_20261018/TAIL_DIAGNOSTIC.png',dpi=160);plt.close(fig)
print(json.dumps(dict(population=[a for a in population if a['case']=='LEVELWARP'],BASE054_native_reference=baseline),indent=2))
