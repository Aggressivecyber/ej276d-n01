from pathlib import Path
import json,numpy as np
w=Path(__file__).parent;run=w/'collected/run_01';read=lambda p:json.loads(p.read_text());load=lambda n:dict(np.load(run/(n+'.npz')))
a=[]
for seed in range(2):
    geo=load(f'seed{seed}_RATIO_GEOMETRY_ROWS');intrinsic=load(f'seed{seed}_RATIO_INTRINSIC_MEAN_ROWS');rec=read(run/f'seed{seed}_RATIO.json')
    for h in ('fast','slow','common_level'):
        new=load(f'seed{seed}_RATIO_{h}_NATIVE_ROWS');old=load(f'seed{seed}_BASE054_{h}_NATIVE_ROWS');bins=old['native_output_Q_bin'];shift=new['native_peak_index']-old['native_peak_index'];gm=geo['new_mean_peak_index']-geo['old_mean_peak_index']
        a.append(dict(seed=seed,hypothesis=h,old_mean_peak_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(geo['old_mean_peak_index'],return_counts=True))},mean_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(gm,return_counts=True))},native_peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))},observation_flags=rec['metrics'][h]['observation_flags'],bins=[dict(bin=b,rows=int(np.sum(bins==b)),mean_peak_moves=int(np.sum((bins==b)&(gm!=0))),native_peak_moves=int(np.sum((bins==b)&(shift!=0))),mean_peak_max_V=float(geo['new_peak_voltage_V'][bins==b].max()),mean_peak_above_positive_ADC_rail=int(np.sum((bins==b)&(geo['new_peak_voltage_V']>.5))),native_width_undefined=int(np.sum((bins==b)&np.isnan(new['FWHM_ns']))),intrinsic_width_undefined=int(np.sum((bins==b)&np.isnan(intrinsic['FWHM_ns']))),ratio_max_p99=float(np.quantile(geo['ratio_maximum'][bins==b],.99))) for b in range(8)]))
(w/'PEAK_AND_FLAG_ANALYSIS.json').write_text(json.dumps(dict(scope='Saved all-row geometry and native feature association; not unique causal attribution, per-row ADC rail flags were not exported.',cells=a),indent=2)+'\n')
print('Peak and flag concentration analysis saved:6H')
