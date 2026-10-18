"""Known triangle crossings under different fixed and native anchors; no scientific data."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from run_response059 import stage_measurements059,STAGES059
from run_response053 import intrinsic_mean053
from n01.observation008 import observe_generated_carrier

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64);amp=torch.linspace(.05,.12,32,device=device,dtype=torch.float64)[:,None]
carriers=[amp*torch.minimum(1+(t-center)/12,1-(t-center)/24).clamp_min(0) for center in (126,127,128)]
mean,signal,pre_adc=carriers;signal=signal+.003;pre_adc=pre_adc+.007
base=observe_generated_carrier(mean,ste=False,q_floor=.001,return_carrier=True)
native=observe_generated_carrier(pre_adc,ste=False,q_floor=.001,return_carrier=True)
rows,stages=stage_measurements059(mean,signal,pre_adc,native,base['crop_indices'])
assert len(stages)==5 and len(STAGES059)==5 and torch.equal(stages[-1],native['window_V'])
assert np.array_equal(rows['stage_crop_start'][:4],np.full((4,32),110)) and np.array_equal(rows['stage_crop_start'][-1],np.full(32,112))
expected_cross=np.array([[115.2,120.,124.8,138.],[116.1,120.5,124.9,140.],[117.,121.,125.,142.]])
cross_names=('left_crossing10_ns','left_crossing50_ns','left_crossing90_ns','right_crossing50_ns')
maximum_error=0.
for i in range(3):
    for j,key in enumerate(cross_names):maximum_error=max(maximum_error,float(np.max(np.abs(rows[key+'_absolute_ns_by_stage'][i]-expected_cross[i,j]))))
assert maximum_error<1e-12
assert np.max(np.abs(rows['FWHM_ns_by_stage'][:3]-np.array([18.,19.5,21.])[:,None]))<1e-12
assert np.max(np.abs(rows['rise10_90_ns_by_stage'][:3]-np.array([9.6,8.8,8.])[:,None]))<1e-12
assert not rows['selected_peak_below_global_crop_max_flag_by_stage'][0].any()
assert rows['selected_peak_below_global_crop_max_flag_by_stage'][1:4].all() and not rows['selected_peak_below_global_crop_max_flag_by_stage'][-1].any()
intrinsic=intrinsic_mean053(mean);assert np.max(np.abs(intrinsic['features']['FWHM_ns']-18.))<1e-12
assert np.array_equal(intrinsic['peak_index'],np.full(32,126))
direct_error=0.
for stage,tensor in enumerate((mean,signal,pre_adc,native['carrier_adc'])):
    x=tensor.cpu().numpy();centered=x-x[:,:100].mean(1)[:,None];expected=centered[np.arange(32)[:,None],base['crop_indices'].cpu().numpy()]
    direct_error=max(direct_error,float(np.max(np.abs(expected-stages[stage].cpu().numpy()))))
assert direct_error<1e-12
permutation=np.arange(31,-1,-1);perm=torch.as_tensor(permutation.copy(),device=device)
new_obs=observe_generated_carrier(pre_adc[perm],ste=False,q_floor=.001,return_carrier=True)
permuted,_=stage_measurements059(mean[perm],signal[perm],pre_adc[perm],new_obs,base['crop_indices'][perm])
assert all(np.array_equal(v[:,permutation],permuted[k],equal_nan=True) for k,v in rows.items())
zero=torch.zeros_like(mean);zero_obs=observe_generated_carrier(zero,ste=False,q_floor=.001,return_carrier=True)
missing,_=stage_measurements059(zero,zero,zero,zero_obs,zero_obs['crop_indices'])
assert np.isnan(missing['FWHM_ns_by_stage']).all() and missing['FWHM_undefined_flag_by_stage'].all()
assert all(np.isnan(missing[key+'_absolute_ns_by_stage'][:3]).all() for key in cross_names)
receipt=dict(status='PASS',device=str(device),manufactured_rows=32,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,
    known_absolute_crossings_max_error_ns=maximum_error,known_fixed_anchor_FWHM_ns=[18.,19.5,21.],
    known_fixed_anchor_rise_ns=[9.6,8.8,8.],mean_intrinsic_peak126_and_width18=True,
    baseline_and_fixed_windows_independent_numpy_max_error_V=direct_error,
    fixed_anchor_nonmax_flags_preserved=True,native_selected_peak128_distinguished_from_anchor126=True,
    all_rows_and_undefined_crossings_preserved=True,row_permutation_exact=True,scientific_acceptance=False)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
