from pathlib import Path
import json,os,sys,numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;run=w/'collected/run_01';out=r/'reports/response068_20261018';load=lambda path:dict(np.load(path,allow_pickle=False));read=lambda p:json.loads(p.read_text());records=[];examples=[];geometry=[]
sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from run_response039 import morphology018
p=read(r/'configs/response068_20261018.json')
for seed in range(2):
 c=load(run/f'seed{seed}_VOLTAGEWARP_COUPLING_ROWS.npz');e=load(run/f'seed{seed}_VOLTAGEWARP_COUPLING_EXAMPLES.npz');g=load(run/f'seed{seed}_VOLTAGEWARP_GEOMETRY_ROWS.npz');intr=load(run/f'seed{seed}_VOLTAGEWARP_INTRINSIC_MEAN_ROWS.npz');be=load(run/f'seed{seed}_COUPLED067_COUPLING_EXAMPLES.npz');qmap=load(run/f'seed{seed}_VOLTAGEWARP_WARP_QUERIES.npz')['inverse_query_absolute_sample']
 for h in ('fast','slow','common_level'):
  b=load(run/f'seed{seed}_COUPLED067_{h}_NATIVE_ROWS.npz');n=load(run/f'seed{seed}_VOLTAGEWARP_{h}_NATIVE_ROWS.npz');flags=load(run/f'seed{seed}_VOLTAGEWARP_{h}_OBSERVATION_FLAGS.npz');oldbad=b['FWHM_undefined_flag'];newbad=n['FWHM_undefined_flag'];shift=n['native_peak_index']-b['native_peak_index'];cells=[]
  for group,bins in [('input_Q',n['input_Q_bin']),('fixed_COUPLED067_output_Q',b['native_output_Q_bin']),('own_output_Q',n['native_output_Q_bin'])]:
   for k in range(8):
    m=bins==k
    cells.append(dict(group=group,bin=k,rows=int(m.sum()),old_width_undefined=int(oldbad[m].sum()),new_width_undefined=int(newbad[m].sum()),native_peak_changes=int((shift[m]!=0).sum()),mean_to_signal_peak_changes=int((c['new_signal_peak_index'][m]!=g['new_mean_peak_index'][m]).sum()),mean_to_native_peak_changes=int((n['native_peak_index'][m]!=g['new_mean_peak_index'][m]).sum()),random_delta_max_V=float(c['maximum_absolute_delta_V'][m].max()),random_delta_mean_of_max_V=float(c['maximum_absolute_delta_V'][m].mean()),old_random_L2_mean_V=float(c['old_random_L2_V'][m].mean()),new_random_L2_mean_V=float(c['new_random_L2_V'][m].mean()),amplitude_scale_mean=float(g['amplitude_scale'][m].mean()),amplitude_scale_range=[float(g['amplitude_scale'][m].min()),float(g['amplitude_scale'][m].max())],query_displacement_mean=float(g['query_maximum_displacement_samples'][m].mean()),observation_flags={key:int(val[m].sum()) for key,val in flags.items() if val.dtype==bool},paired_area_changes={key:dict(mean=float((n[key]-b[key])[m].mean()),SD=float((n[key]-b[key])[m].std())) for key in ('Q_Vns','front_Vns','tail_Vns')}))
  records.append(dict(seed=seed,hypothesis=h,old_bad=np.flatnonzero(oldbad).tolist(),remaining=np.flatnonzero(oldbad&newbad).tolist(),resolved=np.flatnonzero(oldbad&~newbad).tolist(),new_bad=np.flatnonzero(~oldbad&newbad).tolist(),native_peak_changed_rows=int((shift!=0).sum()),mean_to_signal_peak_changed_rows=int(np.sum(c['new_signal_peak_index']!=g['new_mean_peak_index'])),mean_to_native_peak_changed_rows=int(np.sum(n['native_peak_index']!=g['new_mean_peak_index'])),all_observation_flags={key:int(val.sum()) for key,val in flags.items() if val.dtype==bool},all_feature_flags={key:int(val.sum()) for key,val in n.items() if key.endswith('_flag')},groups=cells))
  if h=='fast':
   for row in p['coupling_example_parent_rows'][str(seed)][32:]:
    i=int(np.flatnonzero(e['parent_row']==row)[0]);old=be['adapted_mean_pre_V'][i];fixed=e['fixed_mean_pre_V'][i];mean=e['adapted_mean_pre_V'][i];signal=e['coupled_signal_V'][i];pre=signal+e['noise_fast_V'][i];adc=np.round((np.clip(pre,-.5,.5)+.5)*16383)/16383-.5
    peaks=[int(np.argmax(x[112:144])+112) for x in (fixed,old,mean,e['base_signal_V'][i],signal,pre,adc)];native=peaks[-1];mp=peaks[2];assert native==n['native_peak_index'][row] and mp==intr['intrinsic_peak_index'][row]
    adv=np.array([x[native]-x[mp] for x in (mean,signal,pre,adc)]);components=np.r_[adv[0],np.diff(adv)]
    examples.append(dict(seed=seed,parent_row=row,prior067_bad=bool(oldbad[row]),new_bad=bool(newbad[row]),input_Q_Vns=float(n['input_Q_Vns'][row]),intrinsic_mean_width_ns=float(intr['FWHM_ns'][row]),native_width_ns=float(n['FWHM_ns'][row]),ordered_peaks=peaks,final_vs_mean_peak_advantage_mV=(adv*1000).tolist(),ordered_peak_components_mV=(components*1000).tolist(),fixed_BASE054_samples112to160_V=fixed[112:160].tolist(),COUPLED067_mean_samples112to160_V=old[112:160].tolist(),VOLTAGEWARP_mean_samples112to160_V=mean[112:160].tolist(),VOLTAGEWARP_signal_samples112to160_V=signal[112:160].tolist(),VOLTAGEWARP_ADC_samples112to160_V=adc[112:160].tolist()))
 geo=dict(seed=seed,original_peak_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(g['old_mean_peak_index'],return_counts=True))},mean_peak_changed_rows=int(np.sum(g['old_mean_peak_index']!=g['new_mean_peak_index'])),amplitude_scale_range=[float(g['amplitude_scale'].min()),float(g['amplitude_scale'].max())],query_max_shift=float(g['query_maximum_displacement_samples'].max()),query_min_spacing=float(g['query_minimum_spacing_samples'].min()))
 geometry.append(geo)
audit=dict(scope='All paired source scalar/flag rows and all original066 seven failure examples per head. Native quantum/noise decisions are kept. SOURCE/REAL Q groups are unpaired; conditional profile covariance is not repeated-input MC covariance.',records=records,examples=examples,geometry=geometry,peaks_order=['BASE054_mean','COUPLED067_mean','VOLTAGEWARP_mean','new_mean_plus_original_random','coupled_signal','plus_fast_noise','ADC'],scientific_goal_achieved=False)
(w/'VOLTAGE_WARP_TOPOLOGY.json').write_text(json.dumps(audit,indent=2)+'\n')
template=read(r/p['template_file']);template_metrics={k:{n:v.tolist() for n,v in morphology018(np.asarray(template[k+'_unit_Q_profiles'])).items()} for k in ('source','target')}
# Existing068 formula applied analytically to its own frozen conditional input templates, not a new candidate or a decoded population.
template_comparison=[];sk=np.log(template['source_Q_centers_Vns']);tk=np.log(template['target_Q_centers_Vns']);sp=np.asarray(template['source_unit_Q_profiles']);tp=np.asarray(template['target_unit_Q_profiles'])
for j,Q in enumerate(template['source_Q_centers_Vns']):
 ss=sp[j].copy();tt=np.array([np.interp(np.log(Q),tk,tp[:,i]) for i in range(256)]);peak=122;locations=np.arange(110,176)-peak+16;a=0;pidx=12
 local_s=ss[locations];local_t=tt[locations];sc=np.r_[0.,np.cumsum(.5*(local_s[:-1]+local_s[1:]))];tc=np.r_[0.,np.cumsum(.5*(local_t[:-1]+local_t[1:]))];query=np.arange(66,dtype=float)
 for left,right in ((a,pidx),(pidx,65)):
  f=(tc[left:right+1]-tc[left])/(tc[right]-tc[left]);query[left:right+1]=np.interp(sc[left]+f*(sc[right]-sc[left]),sc,np.arange(66))
 query[[a,pidx,65]]=[a,pidx,65];mapped=np.interp(query,np.arange(66),local_s);mapped*=local_s.sum()/mapped.sum();new=ss.copy();new[locations]=mapped
 mm=morphology018(np.array([ss,new,tt]));template_comparison.append(dict(knot=j,input_Q_Vns=Q,source_width_ns=float(mm['FWHM_ns'][0]),mapped068_width_ns=float(mm['FWHM_ns'][1]),target_width_ns=float(mm['FWHM_ns'][2]),source_rise_ns=float(mm['rise10_90_ns'][0]),mapped068_rise_ns=float(mm['rise10_90_ns'][1]),target_rise_ns=float(mm['rise10_90_ns'][2]),source_peak_V=float(mm['peak_voltage_V'][0]),mapped068_peak_V=float(mm['peak_voltage_V'][1]),target_peak_V=float(mm['peak_voltage_V'][2])))
(w/'TEMPLATE_GEOMETRY_AUDIT.json').write_text(json.dumps(dict(scope='Read-only analysis of already-frozen068 operator and062 conditional profiles; no native evaluation or new candidate.',template_metrics=template_metrics,template_map_comparison=template_comparison,actual_mean_peak_geometry=geometry),indent=2)+'\n')
os.environ['MPLCONFIGDIR']=str(r/'runtime/mpl_noise_audit')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,(seed,row) in zip(axs.ravel(),[(0,567),(0,1615),(0,3674),(1,362),(1,567),(1,5571)]):
 x=next(z for z in examples if z['seed']==seed and z['parent_row']==row)
 for key,label,color in [('COUPLED067_mean_samples112to160_V','067 mean','#0072b2'),('VOLTAGEWARP_mean_samples112to160_V','068 mean','#009e73'),('VOLTAGEWARP_signal_samples112to160_V','068 signal','#d55e00')]:ax.plot(np.arange(112,160),np.array(x[key])*1000,label=label,color=color)
 ax.set(title=f"Head{seed}, parent{row}: width {x['native_width_ns']:.2f} ns",xlabel='Absolute carrier sample',ylabel='Voltage (mV)');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('068 preserved mean peak and remaining shape tradeoff',fontsize=14);fig.savefig(out/'PEAK_EXAMPLES.png',dpi=160);plt.close(fig)
print(json.dumps(dict(geometry=geometry,records=[{k:v for k,v in z.items() if k!='groups'} for z in records],template_comparison=template_comparison),indent=2))
