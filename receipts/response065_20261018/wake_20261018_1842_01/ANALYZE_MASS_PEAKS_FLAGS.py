from pathlib import Path
import json,numpy as np
w=Path(__file__).parent;run=w/'collected/run_01';load=lambda n:dict(np.load(run/(n+'.npz'),allow_pickle=False));read=lambda p:json.loads(p.read_text())
a=[]
for seed in range(2):
    geo=load(f'seed{seed}_MASS_GEOMETRY_ROWS');intrinsic=load(f'seed{seed}_MASS_INTRINSIC_MEAN_ROWS');rec=read(run/f'seed{seed}_MASS.json');gm=geo['new_mean_peak_index']-geo['old_mean_peak_index']
    for h in ('fast','slow','common_level'):
        new=load(f'seed{seed}_MASS_{h}_NATIVE_ROWS');old=load(f'seed{seed}_BASE054_{h}_NATIVE_ROWS');flags=load(f'seed{seed}_MASS_{h}_OBSERVATION_FLAGS');oldflags=load(f'seed{seed}_BASE054_{h}_OBSERVATION_FLAGS');shift=new['native_peak_index']-old['native_peak_index'];undef=np.isnan(new['FWHM_ns']);cells=[]
        for group,bins in [('input',new['input_Q_bin']),('fixed_BASE',old['native_output_Q_bin']),('own_output',new['native_output_Q_bin'])]:
            for b in range(8):
                mask=bins==b
                cells.append(dict(group=group,bin=b,rows=int(mask.sum()),mass_displacement_quantiles=np.quantile(geo['mean_absolute_mass_displacement_samples'][mask],[0,.1,.5,.9,1]).tolist(),
                    mean_peak_moves=int(np.sum(mask&(gm!=0))),native_peak_moves=int(np.sum(mask&(shift!=0))),mean_peak_max_V=float(geo['new_peak_voltage_V'][mask].max()),mean_peak_above_positive_ADC_rail=int(np.sum(mask&(geo['new_peak_voltage_V']>.5))),
                    delta_max_V=float(geo['maximum_absolute_delta_V'][mask].max()),delta_mean_V=float(geo['maximum_absolute_delta_V'][mask].mean()),native_width_undefined=int(np.sum(mask&undef)),intrinsic_width_undefined=int(np.sum(mask&np.isnan(intrinsic['FWHM_ns']))),
                    observation_counts={k:int(v[mask].sum()) for k,v in flags.items() if v.dtype==bool},baseline_observation_counts={k:int(v[mask].sum()) for k,v in oldflags.items() if v.dtype==bool}))
        failed=[]
        for i in np.flatnonzero(undef):
            failed.append(dict(row=int(i),inputQ=float(new['input_Q_Vns'][i]),fixed_BASE_bin=int(old['native_output_Q_bin'][i]),own_bin=int(new['native_output_Q_bin'][i]),mean_shift=int(gm[i]),native_shift=int(shift[i]),native_peak=int(new['native_peak_index'][i]),intrinsic_undefined=bool(np.isnan(intrinsic['FWHM_ns'][i])),peak_voltage=float(new['peak_voltage_V'][i]),mass_displacement=float(geo['mean_absolute_mass_displacement_samples'][i])))
        same=shift==0;effects={}
        for name,mask in [('same_peak',same),('shifted_peak',~same),('all',np.ones(6593,bool))]:
            effects[name]=dict(rows=int(mask.sum()),deltas={key:dict(mean=float((new[key]-old[key])[mask].mean()),std=float((new[key]-old[key])[mask].std()),quantiles=np.quantile((new[key]-old[key])[mask],[0,.1,.5,.9,1]).tolist()) for key in ['Q_Vns','front_Vns','tail_Vns']})
        a.append(dict(seed=seed,hypothesis=h,mean_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(gm,return_counts=True))},native_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))},observation_flags=rec['metrics'][h]['observation_flags'],groups=cells,failed_width_rows=failed,paired_peak_effects=effects))
(w/'MASS_PEAK_AND_FLAG_ANALYSIS.json').write_text(json.dumps(dict(scope='All saved rows, paired relations only; equal peaks do not prove causation.',cells=a),indent=2)+'\n')
print('065 all-head/H mass, peak and flag analysis saved')
