from pathlib import Path
import json,os
import numpy as np
from scipy.stats import wasserstein_distance
w=Path(__file__).parent;r=w.parents[2];run=w/'collected/run_01';base=r/'receipts/response060_20261018/wake_20261018_1605_01/collected/run_01'
load=lambda p:dict(np.load(p,allow_pickle=False));real=load(run/'REAL_ROWS.npz');audit=[];examples=[]
for seed in range(2):
 g=load(run/f'seed{seed}_SPLITWARP_GEOMETRY_ROWS.npz');ce=load(run/f'seed{seed}_SPLITWARP_COUPLING_EXAMPLES.npz');p=g['old_mean_peak_index']
 for h in ('fast','slow','common_level'):
  n=load(run/f'seed{seed}_SPLITWARP_{h}_NATIVE_ROWS.npz');b=load(base/f'seed{seed}_BASE054_{h}_NATIVE_ROWS.npz');d=n['tail_Vns']-b['tail_Vns'];ns=n['native_peak_index']-p;bs=b['native_peak_index']-p
  cells=[]
  for x,y in sorted(set(zip(bs.tolist(),ns.tolist()))):
   m=(bs==x)&(ns==y);cells.append(dict(BASE054_peak_minus_mean=x,SPLITWARP_peak_minus_mean=y,rows=int(m.sum()),mean_tail_delta_Vns=float(d[m].mean()),sum_tail_delta_Vns=float(d[m].sum()),sum_abs_tail_delta_Vns=float(np.abs(d[m]).sum()),max_abs_tail_delta_Vns=float(np.abs(d[m]).max())))
  groups={}
  for name,m in [('both_at_mean',(ns==0)&(bs==0)),('same_native_peak',ns==bs),('different_native_peak',ns!=bs),('candidate_at_mean',ns==0),('candidate_off_mean',ns!=0)]:
   groups[name]=dict(rows=int(m.sum()),mean_delta_Vns=float(d[m].mean()),mean_abs_delta_Vns=float(np.abs(d[m]).mean()),max_abs_delta_Vns=float(np.abs(d[m]).max()),sum_abs_delta_Vns=float(np.abs(d[m]).sum()))
  # This replacement isolates the contribution of a group in the saved marginal distance; it is not a realizable waveform candidate.
  impact=[]
  for name,mask in [('both_at_mean',(ns==0)&(bs==0)),('same_native_peak',ns==bs),('different_native_peak',ns!=bs),('candidate_off_mean',ns!=0)]:
   vals=n['tail_Vns'].copy();vals[mask]=b['tail_Vns'][mask]
   impact.append(dict(group=name,diagnostic_replaced_tail_W1=float(wasserstein_distance(vals,real['tail_Vns']))))
  audit.append(dict(seed=seed,hypothesis=h,native_tail_W1=float(wasserstein_distance(n['tail_Vns'],real['tail_Vns'])),BASE054_tail_W1=float(wasserstein_distance(b['tail_Vns'],real['tail_Vns'])),peak_shift_cells=cells,groups=groups,scalar_replacement_diagnostics=impact))
  if h!='fast':continue
  for i,row in enumerate(ce['parent_row']):
   row=int(row);fixed=ce['fixed_mean_pre_V'][i];new=ce['adapted_mean_pre_V'][i];rr=ce['old_random_V'][i];dr=ce['random_correction_V'][i];noise=ce['noise_fast_V'][i]
   oldpre=fixed+rr+noise;newpre=ce['coupled_signal_V'][i]+noise
   oldadc=np.round((np.clip(oldpre,-.5,.5)+.5)*16383)/16383-.5;newadc=np.round((np.clip(newpre,-.5,.5)+.5)*16383)/16383-.5
   op=int(np.argmax(oldadc[112:144])+112);npk=int(np.argmax(newadc[112:144])+112);ref=int(p[row]);assert op==b['native_peak_index'][row] and npk==n['native_peak_index'][row]
   assert np.array_equal(oldadc[:100],newadc[:100]);baseline=oldadc[:100].mean()
   oldtail=oldadc[op+30:op+240].sum()-210*baseline;newtail=newadc[npk+30:npk+240].sum()-210*baseline
   at_ref=(newadc[ref+30:ref+240]-oldadc[ref+30:ref+240]).sum()
   same_new_window=(newadc[npk+30:npk+240]-oldadc[npk+30:npk+240]).sum()
   window_shift=oldadc[npk+30:npk+240].sum()-oldadc[op+30:op+240].sum()
   assert max(abs(oldtail-b['tail_Vns'][row]),abs(newtail-n['tail_Vns'][row]),abs(newtail-oldtail-same_new_window-window_shift))<1e-12
   examples.append(dict(seed=seed,parent_row=row,reference_peak=ref,BASE054_peak=op,candidate_peak=npk,native_tail_delta=newtail-oldtail,ADC_same_new_window_delta=same_new_window,BASE054_window_shift_delta=window_shift,ADC_fixed_reference_delta=at_ref,preADC_fixed_reference_mean_delta=float((new-fixed)[ref+30:ref+240].sum()),preADC_fixed_reference_random_delta=float(dr[ref+30:ref+240].sum()),mean_delta_at_boundary_V=float((new-fixed)[ref+30]),random_delta_at_boundary_V=float(dr[ref+30]),seam_delta_V=float(g['correction_split_step_V'][row])))
result=dict(scope='All 6593 saved scalar rows per head/H and 78 saved fast waveform examples. Diagnostic scalar replacements are attribution, not candidate evaluations. Exact two-term example decomposition is same-output-window waveform change plus BASE054 window shift; it includes ADC. No full-population unsaved waveforms reconstructed.',population=audit,examples=examples)
(w/'NATIVE_TAIL_DECOMPOSITION.json').write_text(json.dumps(result,indent=2)+'\n')
os.environ['MPLCONFIGDIR']=str(w/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for seed,color in [(0,'#0072b2'),(1,'#d55e00')]:
 e=[z for z in examples if z['seed']==seed];axes[0].scatter([z['ADC_same_new_window_delta'] for z in e],[z['BASE054_window_shift_delta'] for z in e],s=18,alpha=.7,label=f'head{seed}')
 g=load(run/f'seed{seed}_SPLITWARP_GEOMETRY_ROWS.npz')
 if seed==0:axes[1].hist(g['correction_split_step_V']*1000,bins=70,color='#009e73');axes[2].scatter(g['input_Q_Vns'],g['correction_split_step_V']*1000,s=3,alpha=.25,color='#009e73')
axes[0].set(xlabel='Same new window waveform change (V ns)',ylabel='BASE054 window shift (V ns)',title='78 saved examples: exact tail decomposition');axes[0].legend()
axes[1].set(xlabel='Correction step at p+30 (mV)',ylabel='Rows',title='All 6593 deterministic seam changes')
axes[2].set(xlabel='Input Q (V ns)',ylabel='Correction step (mV)',title='Seam change versus input Q')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(r/'reports/response070_20261018/NATIVE_TAIL_DIAGNOSTIC.png',dpi=160);plt.close(fig)
print(json.dumps([{k:v for k,v in z.items() if k!='peak_shift_cells'} for z in audit],indent=2))
