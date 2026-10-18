from pathlib import Path
import os,json,numpy as np
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;out=w/'collected/run_01';prior=r/'receipts/response065_20261018/wake_20261018_1842_01/collected/run_01';dest=r/'reports/response066_20261018';dest.mkdir()
os.environ['MPLCONFIGDIR']=str(r/'runtime/mpl_noise_audit')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text());stages=read(out/'DEFINITIONS.json')['stages'];pairs=[];cells=[];population=[]
for seed in range(2):
 g=dict(np.load(prior/f'seed{seed}_MASS_GEOMETRY_ROWS.npz'));intr=dict(np.load(prior/f'seed{seed}_MASS_INTRINSIC_MEAN_ROWS.npz'))
 for h in ('fast','slow','common_level'):
  p=read(out/f'seed{seed}_{h}_PAIRED_CONTRASTS.json');b=dict(np.load(out/f'seed{seed}_BASE054_{h}_STAGE_ROWS.npz'));n=dict(np.load(out/f'seed{seed}_MASS_{h}_STAGE_ROWS.npz'))
  widths=n['FWHM_undefined_flag_by_stage'];on=widths[-1]&~widths[-2];off=widths[-2]&~widths[-1];peaks=n['selected_peak_index_by_carrier_stage'];adv=n['native_minus_mean_peak_advantage_V_by_carrier'];parts=np.concatenate((adv[:1],np.diff(adv,axis=0)));shift=n['native_peak_index']-b['native_peak_index']
  q=n['Q_Vns_by_stage'];n['PSD_guarded_by_stage']=n['tail_Vns_by_stage']/np.maximum(q,read(out/'DEFINITIONS.json')['q_floor']);b['PSD_guarded_by_stage']=b['tail_Vns_by_stage']/np.maximum(b['Q_Vns_by_stage'],read(out/'DEFINITIONS.json')['q_floor'])
  for group,bins in [('input_Q',n['input_Q_bin']),('fixed_BASE054_output_Q',n['BASE054_output_Q_bin'])]:
   for k in range(8):
    sel=bins==k
    cells.append(dict(seed=seed,hypothesis=h,group=group,bin=k,rows=int(sel.sum()),FWHM_new_on=int((sel&on).sum()),FWHM_resolved=int((sel&off).sum()),native_peak_moved_from_BASE=int((sel&(shift!=0)).sum()),native_peak_differs_from_mean=int((sel&(peaks[-1]!=peaks[0])).sum()),features={field:dict(BASE_mean=np.nanmean(b[field+'_by_stage'][:,sel],axis=1).tolist(),MASS_mean=np.nanmean(n[field+'_by_stage'][:,sel],axis=1).tolist(),BASE_defined=np.isfinite(b[field+'_by_stage'][:,sel]).sum(1).tolist(),MASS_defined=np.isfinite(n[field+'_by_stage'][:,sel]).sum(1).tolist()) for field in ('Q_Vns','front_Vns','tail_Vns','peak_voltage_V','rise10_90_ns','FWHM_ns','PSD_guarded')}))
  pop=dict(seed=seed,hypothesis=h,rows=len(shift),native_peak_differs_from_mean=int((peaks[-1]!=peaks[0]).sum()),carrier_adjacent_peak_changes=[int((peaks[i]!=peaks[i-1]).sum()) for i in range(1,4)],FWHM_counts=widths.sum(1).tolist(),recrop_new_bad_parent_rows=np.flatnonzero(on).tolist(),recrop_resolved_parent_rows=np.flatnonzero(off).tolist(),bad_rows=[],peak_groups=[])
  for selname,sel in [('all',np.ones(len(shift),dtype=bool)),('native_peak_differs_from_mean',peaks[-1]!=peaks[0]),('native_FWHM_undefined',widths[-1])]:
   pop['peak_groups'].append(dict(group=selname,rows=int(sel.sum()),advantage_mean_mV=(1000*adv[:,sel].mean(1)).tolist(),component_mean_mV=(1000*parts[:,sel].mean(1)).tolist(),component_min_mV=(1000*parts[:,sel].min(1)).tolist(),component_max_mV=(1000*parts[:,sel].max(1)).tolist()))
  for row in np.flatnonzero(widths[-1]):
   pop['bad_rows'].append(dict(parent_row=int(row),input_Q_Vns=float(n['input_Q_Vns'][row]),carrier_selected_peaks=peaks[:,row].tolist(),advantage_mV=(1000*adv[:,row]).tolist(),components_mV=(1000*parts[:,row]).tolist(),FWHM_by_stage=n['FWHM_ns_by_stage'][:,row].tolist(),intrinsic_mean_FWHM=float(intr['FWHM_ns'][row]),intrinsic_mean_undefined=bool(intr['FWHM_undefined_flag'][row])))
  assert len(pop['bad_rows'])==7 and on.sum()==7 and not (widths[-1]&widths[-2]).any()
  assert all(x['carrier_selected_peaks'][0]==123 and x['carrier_selected_peaks'][1] in (133,134) and x['carrier_selected_peaks'][1:]==[x['carrier_selected_peaks'][1]]*3 and not x['intrinsic_mean_undefined'] for x in pop['bad_rows'])
  p['new_FWHM_censor_rows']=np.flatnonzero(on).tolist();p['resolved_FWHM_censor_rows']=np.flatnonzero(off).tolist();pairs.append(p);population.append(pop)
a=dict(stage_order=stages,pairs=pairs,cells=cells,scientific_acceptance=False,interpretation='Ordered path-dependent contrasts; fixed-index16 morphology is not intrinsic-peak morphology. All populations and flags retained; final-native acceptance unchanged.')
(w/'ORDERED_READOUT_ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n');(w/'PEAK_COMPETITION_ANALYSIS.json').write_text(json.dumps(dict(population=population,component_order=['mean','M0_random','electronics','ADC_quantization'],selection='Pair of intrinsic mean peak and final ADC selected peak; diagnostic not independent causal attribution.'),indent=2)+'\n')
s=dict(scientific_goal_achieved=False,all_six_native_FWHM_failures_new_at_recrop=True,all_seven_per_head_peak_flip_at_M0_random=True,population=[{k:v for k,v in x.items() if k not in ('bad_rows','peak_groups')} for x in population],score_ranges={})
for field in ('wave','QPSD','front_Vns','tail_Vns','rise10_90_ns','FWHM_ns'):
 s['score_ranges'][field]={key:np.array([p['score_contrasts'][field][key] for p in pairs]).T[[0,1,2,3,4]].tolist() for key in ('stage_delta','ordered_contributions')}
(w/'ANALYSIS_SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(2,3,figsize=(16,9));short=['Mean','+Random','+Noise','ADC fixed','ADC native'];colors=plt.cm.tab10(np.arange(6))
for i,p in enumerate(pairs):
 for ax,k in [(axs[0,0],'wave'),(axs[0,1],'QPSD'),(axs[0,2],'FWHM_ns')]:ax.plot(range(5),p['score_contrasts'][k]['stage_delta'],marker='o',lw=1.3,color=colors[i],label=f"head{p['seed']} {p['hypothesis']}")
for ax,title in zip(axs[0],['Wave distance: MASS minus BASE','QPSD distance: MASS minus BASE','FWHM W1 delta (fixed-stage caveat)']):
 ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(5),short,rotation=20);ax.set_title(title);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
for row in population[0]['bad_rows']:axs[1,0].plot(range(4),row['advantage_mV'],'o-',label=str(row['parent_row']))
axs[1,0].axhline(0,color='black',lw=.7);axs[1,0].set_xticks(range(4),['Mean','+Random','+Noise','ADC'],rotation=20);axs[1,0].set(title='Final peak advantage over mean peak: 7 failures',ylabel='Voltage difference (mV)');axs[1,0].legend(title='Parent row',fontsize=7,ncol=2)
for i,p in enumerate(population):axs[1,1].plot(range(5),p['FWHM_counts'],'o-',color=colors[i])
axs[1,1].set_xticks(range(5),short,rotation=20);axs[1,1].set(title='FWHM undefined: new 7 at final recrop',ylabel='Rows /6593')
for case,col in [('BASE054','#4c78a8'),('MASS','#e45756')]:
 z=dict(np.load(out/f'seed0_{case}_fast_STAGE_ROWS.npz'));axs[1,2].plot(np.arange(64)-16,z['population_wave_mean_V'][-1,:64],label=case,color=col)
axs[1,2].set(title='Native mean waveform: head0 fast',xlabel='Time relative to selected peak (ns)',ylabel='Voltage (V)');axs[1,2].legend()
fig.suptitle('066 unchanged 065 replay: ordered readout and peak-competition audit',fontsize=16);fig.tight_layout();fig.savefig(dest/'DIAGNOSTIC.png',dpi=160);plt.close(fig)
print(json.dumps(s['population']))
