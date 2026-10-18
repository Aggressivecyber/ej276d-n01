"""Global and bounded peaks retain first-argmax, baseline and ADC semantics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.observation008 import observe_generated_carrier
from run_response046 import carrier_peaks046

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu')
args=parser.parse_args()
torch.set_num_threads(2)
carrier=torch.zeros((5,384),device=args.device,dtype=torch.float64)
carrier[0,126]=.03;carrier[0,158]=.05
carrier[1,143]=.04
carrier[2,126]=.04;carrier[2,158]=.04
carrier[3,111]=.04;carrier[3,112]=.04
expected_global=np.array([158,143,126,111,0])
expected_selected=np.array([126,143,126,112,112])
expected_outside=np.array([True,False,False,True,True])
original=carrier_peaks046(carrier)
assert np.array_equal(original['global_peak_index'],expected_global)
assert np.array_equal(original['selected_peak_index'],expected_selected)
assert np.array_equal(original['global_peak_outside_search'],expected_outside)
shifted=carrier_peaks046(carrier+.02)
assert all(np.array_equal(original[k],shifted[k]) for k in ('global_peak_index','selected_peak_index','global_peak_outside_search'))
assert np.allclose(original['global_peak_V'],shifted['global_peak_V'],rtol=0,atol=1e-15)
obs=observe_generated_carrier(carrier+.02,ste=False,q_floor=.001,return_carrier=True)
adc=carrier_peaks046(obs['carrier_adc'])
assert np.array_equal(adc['global_peak_index'],expected_global)
assert np.array_equal(adc['selected_peak_index'],expected_selected)
assert np.array_equal(adc['global_peak_outside_search'],expected_outside)
assert np.array_equal(adc['global_peak_index'],obs['global_peak_index'].cpu().numpy())
assert np.array_equal(adc['selected_peak_index'],obs['peak_index'].cpu().numpy())
assert np.array_equal(adc['global_peak_outside_search'],obs['flags']['global_peak_outside_search'].cpu().numpy())
permutation=torch.arange(4,-1,-1,device=args.device)
permuted=carrier_peaks046(obs['carrier_adc'][permutation])
assert all(np.array_equal(v[::-1],permuted[k]) for k,v in adc.items())
write_json(args.out,dict(status='PASS',device=args.device,manufactured_rows=5,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    outside_peak_retained=True,search_boundary_retained=True,first_global_and_bounded_argmax_ties_verified=True,
    flat_row_retained=True,baseline_shift_invariance=True,ADC_peak_and_flag_replay_exact=True,
    row_permutation_verified=True,scientific_acceptance=False))
print('response046 manufactured carrier-peak checks: PASS')
