"""Manufactured conservative conditional-width semantics; no scientific data."""
import argparse
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.stats018 import morphology018
from run_response045 import conditional_width045

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu')
args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device)
t=torch.arange(384,device=device,dtype=torch.float64)
shape=torch.exp(-.5*((t-126.)/8.).square());shape[:100]=0.
input_q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=input_q[:,None]*shape/shape[110:366].sum()
qref=3.680695072375209
delta,coefficient,mapped=conditional_width045(mean,input_q,.5,qref)
adapted=mean+delta
canonical_error=float(canonical_charge008(delta).abs().max())
full_error=float(delta.sum(1).abs().max())
assert max(canonical_error,full_error)<1e-12
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,159:],mean[:,159:])
assert float(mapped.diff(dim=1).min())>0.
assert bool((adapted>=-1e-14).all())
assert torch.equal(mapped[:,0],torch.full_like(input_q,109.5))
assert torch.equal(mapped[:,-1],torch.full_like(input_q,158.5))
assert bool((coefficient[input_q<qref]<0).all() and (coefficient[input_q>qref]>0).all())
baseline=observe_generated_carrier(mean,ste=False,q_floor=.001,return_indices=True)
changed=observe_generated_carrier(adapted,ste=False,q_floor=.001,return_indices=True)
assert not bool(baseline['flags']['pre_ADC_out_of_range'].any() or changed['flags']['pre_ADC_out_of_range'].any())
old_width=morphology018(baseline['window_V'].cpu().numpy())['FWHM_ns']
new_width=morphology018(changed['window_V'].cpu().numpy())['FWHM_ns']
assert new_width[0]<old_width[0] and new_width[-1]>old_width[-1]
assert float(adapted[0].max())>float(mean[0].max()) and float(adapted[-1].max())<float(mean[-1].max())
zero,_,_=conditional_width045(mean,input_q,0.,qref)
assert torch.equal(zero,torch.zeros_like(mean))
identity=observe_generated_carrier(mean+zero,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(baseline[k],identity[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(baseline['flags'][k],identity['flags'][k]) for k in baseline['flags'])
reverse=torch.arange(31,-1,-1,device=device)
permuted,_,_=conditional_width045(mean[reverse],input_q[reverse],.5,qref)
assert torch.equal(permuted,delta[reverse])
single=torch.cat([conditional_width045(mean[i:i+1],input_q[i:i+1],.5,qref)[0] for i in range(32)])
batching_error=float((single-delta).abs().max())
assert batching_error<1e-12
negative,_,_=conditional_width045(-mean,input_q,.5,qref)
assert torch.equal(negative,-delta)
scaled,_,_=conditional_width045(3.*mean,input_q,.5,qref)
assert torch.allclose(scaled,3.*delta,atol=1e-12,rtol=1e-12)
write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=32,checkpoint_loads=0,dataset_loads=0,
    AD_queries=0,full_integral_residual_Vns=full_error,canonical_Q_residual_Vns=canonical_error,
    support_exact=True,minimum_mapped_interval_ns=float(mapped.diff(dim=1).min()),
    low_Q_width_before_ns=float(old_width[0]),low_Q_width_after_ns=float(new_width[0]),
    high_Q_width_before_ns=float(old_width[-1]),high_Q_width_after_ns=float(new_width[-1]),
    low_Q_peak_increased=True,high_Q_peak_decreased=True,manufactured_no_ADC_clipping=True,zero_strength_exact_native_observation=True,
    batching_max_absolute_error_V=batching_error,batching_tolerance_V=1e-12,
    signed_linearity_and_row_permutation_and_batching_verified=True,scientific_acceptance=False))
print('response045 manufactured conditional-width checks: PASS')
