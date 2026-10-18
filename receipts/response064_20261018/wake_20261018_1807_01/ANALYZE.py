from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('E:/GAN/research/N01');w=r/'receipts/response064_20261018/wake_20261018_1807_01';out=w/'collected/run_01';dest=r/'reports/response064_20261018';dest.mkdir()
read=lambda p:json.loads(p.read_text());pairs=[];cells=[];limits=[];stages=read(out/'DEFINITIONS.json')['stages']
for seed in range(2):
 g=dict(np.load(out/f'seed{seed}_RESIDUAL_GEOMETRY_ROWS.npz'));lim=dict(np.load(out/f'seed{seed}_RESIDUAL_POSITIVITY_ROWS.npz'))
 for h in ('fast','slow','common_level'):
  p=read(out/f'seed{seed}_{h}_PAIRED_CONTRASTS.json');b=dict(np.load(out/f'seed{seed}_BASE054_{h}_STAGE_ROWS.npz'));n=dict(np.load(out/f'seed{seed}_RESIDUAL_{h}_STAGE_ROWS.npz'))
  widths=n['FWHM_undefined_flag_by_stage'];on=widths[-1]&~widths[-2];off=widths[-2]&~widths[-1];shift=n['native_peak_index']-b['native_peak_index'];relative={}
  for group,bins in [('input_Q',n['input_Q_bin']),('fixed_BASE054_output_Q',n['BASE054_output_Q_bin'])]:
   for k in range(8):
    sel=bins==k;new_bad=sel&on
    cell=dict(seed=seed,hypothesis=h,group=group,bin=k,rows=int(sel.sum()),FWHM_new_on=int(new_bad.sum()),FWHM_resolved=int((sel&off).sum()),mean_peak_moved=int(((g['new_mean_peak_index']!=g['old_mean_peak_index'])&sel).sum()),native_peak_moved=int((sel&(shift!=0)).sum()),capped=int((sel&g['positivity_boundary_reached']).sum()),new_bad_capped=int((new_bad&g['positivity_boundary_reached']).sum()),new_bad_shift_values=shift[new_bad].tolist(),new_bad_parent_rows=np.flatnonzero(new_bad).tolist(),features={})
    for field in ('Q_Vns','front_Vns','tail_Vns','peak_voltage_V','rise10_90_ns','FWHM_ns','PSD_guarded'):
     x=n[field+'_by_stage'][:,sel];y=b[field+'_by_stage'][:,sel];cell['features'][field]=dict(BASE_mean=np.nanmean(y,axis=1).tolist(),RESIDUAL_mean=np.nanmean(x,axis=1).tolist(),BASE_defined=np.isfinite(y).sum(1).tolist(),RESIDUAL_defined=np.isfinite(x).sum(1).tolist())
    cells.append(cell)
  parts=p['wave_component_contrasts']['coordinate_by_16sample_block'];p['coordinate_first16_native_delta']={k:z['stage_delta'][-1][0] for k,z in parts.items()};p['coordinate_16to32_fixed_ADC_delta']={k:z['stage_delta'][-2][1] for k,z in parts.items()};p['new_FWHM_censor_rows']=np.flatnonzero(on).tolist();p['new_FWHM_censor_capped_rows']=int((on&g['positivity_boundary_reached']).sum());p['new_FWHM_censor_native_shifts']=shift[on].tolist();pairs.append(p)
 for group,bins in [('input_Q',lim['input_Q_bin']),('fixed_BASE054_output_Q',dict(np.load(out/f'seed{seed}_BASE054_fast_STAGE_ROWS.npz'))['BASE054_output_Q_bin'])]:
  for k in range(8):
   sel=bins==k;cap=sel&g['positivity_boundary_reached'];absx=lim['limiting_absolute_index'][cap];relx=lim['limiting_relative_index'][cap]
   limits.append(dict(seed=seed,group=group,bin=k,rows=int(sel.sum()),capped=int(cap.sum()),scale_quantiles=np.quantile(g['applied_scale'][sel],[0,.25,.5,.75,1]).tolist(),absolute_hist={str(int(a)):int(c) for a,c in zip(*np.unique(absx,return_counts=True))},relative_hist={str(int(a)):int(c) for a,c in zip(*np.unique(relx,return_counts=True))},capped_at110to113=int(((absx>=110)&(absx<=113)).sum())))
a=dict(stage_order=stages,pairs=pairs,cells=cells,positivity_groups=limits,scientific_acceptance=False,interpretation='Ordered path-dependent contrasts. Fixed-anchor shape view is normalized by its actual crop maximum; index16 can be below it. Native metric and thresholds unchanged. Final deep freeze postconditions unavailable after output-cap failure.')
(w/'ORDERED_READOUT_ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n')
plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(2,3,figsize=(16,9));short=['Mean','+Random','+Noise','ADC fixed','ADC native'];colors=plt.cm.tab10(np.arange(6))
for i,p in enumerate(pairs):
 label=f"seed{p['seed']} {p['hypothesis']}"
 for ax,k in [(axs[0,0],'wave'),(axs[0,1],'QPSD'),(axs[0,2],'FWHM_ns')]:ax.plot(range(5),p['score_contrasts'][k]['stage_delta'],marker='o',lw=1.3,color=colors[i],label=label)
for ax,title in zip(axs[0],['Wave distance: RESIDUAL minus BASE','QPSD distance: RESIDUAL minus BASE','FWHM W1 delta (fixed-stage caveat)']):
 ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(5),short,rotation=20);ax.set_title(title);ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
lim=dict(np.load(out/'seed0_RESIDUAL_POSITIVITY_ROWS.npz'));g=dict(np.load(out/'seed0_RESIDUAL_GEOMETRY_ROWS.npz'));cap=g['positivity_boundary_reached'];axs[1,0].hist(lim['limiting_relative_index'][cap],bins=np.arange(1.5,80.5,1),color='#995fbc');axs[1,0].set(title='Positivity-limiting positions: seed0',xlabel='Sample relative to original mean peak +16',ylabel='Capped parent rows')
p=pairs[0];x=np.arange(5);f=p['flag_transitions']['FWHM_undefined'];axs[1,1].plot(x,f['RESIDUAL_counts'],'o-',label='RESIDUAL');axs[1,1].plot(x,f['BASE_counts'],'o-',label='BASE');axs[1,1].set_xticks(x,short,rotation=20);axs[1,1].set(title='FWHM undefined rows: seed0 fast',ylabel='Rows /6593');axs[1,1].legend()
for case,col in [('BASE054','#4c78a8'),('RESIDUAL','#e45756')]:
 z=dict(np.load(out/f'seed0_{case}_fast_STAGE_ROWS.npz'));axs[1,2].plot(np.arange(64)-16,z['population_wave_mean_V'][-1,:64],label=case,color=col)
axs[1,2].set(title='Native mean waveform: seed0 fast',xlabel='Time relative to selected peak (ns)',ylabel='Voltage (V)');axs[1,2].legend()
fig.suptitle('064 ordered readout audit | all tables verified; native run failed output cap',fontsize=16);fig.tight_layout();fig.savefig(dest/'DIAGNOSTIC.png',dpi=160);plt.close(fig)
summary=dict(wave_native_delta_range=[min(p['score_contrasts']['wave']['stage_delta'][-1] for p in pairs),max(p['score_contrasts']['wave']['stage_delta'][-1] for p in pairs)],QPSD_recrop_contribution_range=[min(p['score_contrasts']['QPSD']['ordered_contributions'][-1] for p in pairs),max(p['score_contrasts']['QPSD']['ordered_contributions'][-1] for p in pairs)],FWHM_recrop_contribution_range=[min(p['score_contrasts']['FWHM_ns']['ordered_contributions'][-1] for p in pairs),max(p['score_contrasts']['FWHM_ns']['ordered_contributions'][-1] for p in pairs)],new_censor_counts=[len(p['new_FWHM_censor_rows']) for p in pairs],new_censor_capped_counts=[p['new_FWHM_censor_capped_rows'] for p in pairs],seed0_fixed_BASE_limits=[z for z in limits if z['seed']==0 and z['group']=='fixed_BASE054_output_Q'])
(w/'ANALYSIS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
