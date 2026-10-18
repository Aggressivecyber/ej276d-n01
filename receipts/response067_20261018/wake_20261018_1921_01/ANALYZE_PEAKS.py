from pathlib import Path
import json,os,numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;run=w/'collected/run_01';out=r/'reports/response067_20261018';old66=r/'receipts/response066_20261018/wake_20261018_1901_01/collected/run_01';load=lambda path:dict(np.load(path,allow_pickle=False));records=[];examples=[]
for seed in range(2):
 c=load(run/f'seed{seed}_COUPLED_COUPLING_ROWS.npz');e=load(run/f'seed{seed}_COUPLED_COUPLING_EXAMPLES.npz');g=load(run/f'seed{seed}_COUPLED_GEOMETRY_ROWS.npz');intr=load(run/f'seed{seed}_COUPLED_INTRINSIC_MEAN_ROWS.npz');oldex=load(old66/f'seed{seed}_MASS_fast_DIAGNOSTIC_EXAMPLES.npz')
 for h in ('fast','slow','common_level'):
  b=load(run/f'seed{seed}_MASS065_{h}_NATIVE_ROWS.npz');n=load(run/f'seed{seed}_COUPLED_{h}_NATIVE_ROWS.npz');flags=load(run/f'seed{seed}_COUPLED_{h}_OBSERVATION_FLAGS.npz');oldbad=b['FWHM_undefined_flag'];newbad=n['FWHM_undefined_flag'];shift=n['native_peak_index']-b['native_peak_index'];cells=[]
  for group,bins in [('input_Q',n['input_Q_bin']),('fixed_MASS065_output_Q',b['native_output_Q_bin']),('own_output_Q',n['native_output_Q_bin'])]:
   for k in range(8):
    m=bins==k
    cells.append(dict(group=group,bin=k,rows=int(m.sum()),old_width_undefined=int(oldbad[m].sum()),new_width_undefined=int(newbad[m].sum()),native_peak_changes=int((shift[m]!=0).sum()),signal_peak_changes=int((c['new_signal_peak_index'][m]!=c['old_signal_peak_index'][m]).sum()),random_delta_max_V=float(c['maximum_absolute_delta_V'][m].max()),random_delta_mean_of_max_V=float(c['maximum_absolute_delta_V'][m].mean()),old_random_L2_mean_V=float(c['old_random_L2_V'][m].mean()),new_random_L2_mean_V=float(c['new_random_L2_V'][m].mean()),observation_flags={key:int(val[m].sum()) for key,val in flags.items() if val.dtype==bool},paired_area_changes={key:dict(mean=float((n[key]-b[key])[m].mean()),SD=float((n[key]-b[key])[m].std())) for key in ('Q_Vns','front_Vns','tail_Vns')}))
  rec=dict(seed=seed,hypothesis=h,old_bad=np.flatnonzero(oldbad).tolist(),remaining=np.flatnonzero(oldbad&newbad).tolist(),resolved=np.flatnonzero(oldbad&~newbad).tolist(),new_bad=np.flatnonzero(~oldbad&newbad).tolist(),native_peak_changed_rows=int((shift!=0).sum()),peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))},groups=cells)
  records.append(rec)
  if h=='fast':
   for row in np.flatnonzero(oldbad):
    i=int(np.flatnonzero(e['parent_row']==row)[0]);j=int(np.flatnonzero(oldex['parent_row']==row)[0]);mean=oldex['carriers'][0,j];fixed=e['fixed_mean_pre_V'][i];signal=e['coupled_signal_V'][i];pre=signal+e['noise_fast_V'][i];adc=np.round((np.clip(pre,-.5,.5)+.5)*16383)/16383-.5
    peaks=[int(np.argmax(x[112:144])+112) for x in (fixed,mean,e['base_signal_V'][i],signal,pre,adc)];native=peaks[-1];mp=peaks[1]
    assert native==n['native_peak_index'][row] and mp==intr['intrinsic_peak_index'][row]
    adv=np.array([x[native]-x[mp] for x in (mean,signal,pre,adc)]);components=np.r_[adv[0],np.diff(adv)]
    local=np.arange(112,144);fixedpeaks=local[(fixed[local]>=fixed[local-1])&(fixed[local]>fixed[local+1])];meanpeaks=local[(mean[local]>=mean[local-1])&(mean[local]>mean[local+1])]
    examples.append(dict(seed=seed,parent_row=int(row),remaining_bad=bool(newbad[row]),input_Q_Vns=float(n['input_Q_Vns'][row]),intrinsic_mean_width_ns=float(intr['FWHM_ns'][row]),native_width_ns=float(n['FWHM_ns'][row]),ordered_peaks=peaks,final_vs_mean_peak_advantage_mV=(adv*1000).tolist(),ordered_peak_components_mV=(components*1000).tolist(),fixed_BASE054_local_peak_positions=fixedpeaks.tolist(),fixed_BASE054_local_peak_voltages_V=fixed[fixedpeaks].tolist(),MASS_mean_local_peak_positions=meanpeaks.tolist(),MASS_mean_local_peak_voltages_V=mean[meanpeaks].tolist(),fixed_BASE054_samples112to144_V=fixed[112:144].tolist(),MASS_mean_samples112to144_V=mean[112:144].tolist(),COUPLED_signal_samples112to144_V=signal[112:144].tolist(),COUPLED_ADC_samples112to144_V=adc[112:144].tolist()))
a=dict(scope='All row paired comparisons and old-seven selected fast examples; BASE054 reference mean is not raw deposited waveform or proof of two physical interactions.',records=records,examples=examples,peaks_order=['BASE054_mean','MASS_mean','MASS065_signal','COUPLED_signal','plus_fast_noise','ADC'],peak_component_order=['mean','coupled_random','electronics','quantization'],scientific_goal_achieved=False)
(w/'COUPLING_PEAK_AND_FLAG_ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
os.environ['MPLCONFIGDIR']=str(r/'runtime/mpl_noise_audit')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,(seed,row) in zip(axs.ravel(),[(0,567),(0,1801),(0,3674),(1,362),(1,567),(1,5571)]):
 x=next(z for z in examples if z['seed']==seed and z['parent_row']==row)
 for key,label,color in [('fixed_BASE054_samples112to144_V','BASE054 mean','#777777'),('MASS_mean_samples112to144_V','065/067 mean','#0072b2'),('COUPLED_signal_samples112to144_V','067 signal','#d55e00')]:ax.plot(np.arange(112,144),np.array(x[key])*1000,label=label,color=color)
 ax.set(title=f"Head{seed}, parent{row}: {'still undefined' if x['remaining_bad'] else 'resolved'}",xlabel='Absolute carrier sample',ylabel='Voltage (mV)');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('067 reference multi-crest structure and signed random coupling',fontsize=14);fig.savefig(out/'PEAK_EXAMPLES.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in a.items() if k not in ('records','examples')}));print(json.dumps([{k:v for k,v in z.items() if k!='groups'} for z in records]));print(json.dumps([dict(seed=z['seed'],row=z['parent_row'],remaining=z['remaining_bad'],peaks=z['ordered_peaks'],components_mV=z['ordered_peak_components_mV']) for z in examples]))
