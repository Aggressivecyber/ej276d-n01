from pathlib import Path
import json,numpy as np
w=Path(__file__).parent;run=w/'collected/run_01';load=lambda n:dict(np.load(run/(n+'.npz'),allow_pickle=False));read=lambda p:json.loads(p.read_text())
a=[]
for seed in range(2):
    geo=load(f'seed{seed}_RESIDUAL_GEOMETRY_ROWS');intrinsic=load(f'seed{seed}_RESIDUAL_INTRINSIC_MEAN_ROWS');rec=read(run/f'seed{seed}_RESIDUAL.json')
    gm=geo['new_mean_peak_index']-geo['old_mean_peak_index']
    for h in ('fast','slow','common_level'):
        new=load(f'seed{seed}_RESIDUAL_{h}_NATIVE_ROWS');old=load(f'seed{seed}_BASE054_{h}_NATIVE_ROWS');flags=load(f'seed{seed}_RESIDUAL_{h}_OBSERVATION_FLAGS');oldflags=load(f'seed{seed}_BASE054_{h}_OBSERVATION_FLAGS')
        shift=new['native_peak_index']-old['native_peak_index'];cells=[]
        for group,bins in [('input',new['input_Q_bin']),('fixed_BASE',old['native_output_Q_bin']),('own_output',new['native_output_Q_bin'])]:
            for b in range(8):
                mask=bins==b;scale=geo['applied_scale'][mask];n=int(mask.sum());rail=flags['ADC_rail_hit'];undef=np.isnan(new['FWHM_ns']);capped=geo['positivity_boundary_reached']
                cells.append(dict(group=group,bin=b,rows=n,scale_quantiles=np.quantile(scale,[0,.1,.5,.9,1]).tolist(),capped_rows=int(np.sum(mask&capped)),
                    mean_peak_moves=int(np.sum(mask&(gm!=0))),native_peak_moves=int(np.sum(mask&(shift!=0))),mean_peak_max_V=float(geo['new_peak_voltage_V'][mask].max()),mean_peak_above_positive_ADC_rail=int(np.sum(mask&(geo['new_peak_voltage_V']>.5))),
                    raw_delta_max_V=float(geo['raw_direction_maximum_absolute_V'][mask].max()),applied_delta_max_V=float(geo['maximum_absolute_delta_V'][mask].max()),applied_delta_mean_V=float(geo['maximum_absolute_delta_V'][mask].mean()),
                    native_width_undefined=int(np.sum(mask&undef)),intrinsic_width_undefined=int(np.sum(mask&np.isnan(intrinsic['FWHM_ns']))),
                    observation_counts={k:int(v[mask].sum()) for k,v in flags.items() if v.dtype==bool},baseline_observation_counts={k:int(v[mask].sum()) for k,v in oldflags.items() if v.dtype==bool},
                    associations=dict(rail_and_capped=int(np.sum(mask&rail&capped)),rail_and_mean_move=int(np.sum(mask&rail&(gm!=0))),rail_and_native_move=int(np.sum(mask&rail&(shift!=0))),width_undefined_and_rail=int(np.sum(mask&undef&rail)),width_undefined_and_capped=int(np.sum(mask&undef&capped)))))
        a.append(dict(seed=seed,hypothesis=h,mean_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(gm,return_counts=True))},native_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))},observation_flags=rec['metrics'][h]['observation_flags'],groups=cells))
record=dict(scope='All-row paired geometry, native features and observation extrema/flags; associations do not isolate causation. Capped means positivity_boundary_reached.',cells=a)
(w/'SCALES_PEAK_AND_FLAG_ANALYSIS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(a[0],indent=2))
