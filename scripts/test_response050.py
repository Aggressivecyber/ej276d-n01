"""Manufactured voltage-resampling semantics; no scientific population or checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.stats018 import morphology018
from run_response047 import conditional_width047
from run_response049 import conditional_width049
from run_response050 import conditional_width050

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64)
centers=120+torch.arange(32,device=device)%13
shape=torch.exp(-.5*((t[None,:]-centers[:,None])/8.).square());shape[:,:100]=0.
q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True);qref=3.680695072375209
delta,coefficient,mapped,norm,derived_centers=conditional_width050(mean,q,.5,qref);adapted=mean+delta
full_error=float(delta.sum(1).abs().max());canonical_error=float(canonical_charge008(delta).abs().max())
assert max(full_error,canonical_error)<1e-12
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,159:],mean[:,159:])
assert float(mapped.diff(dim=1).min())>.5 and bool((norm>0).all())
assert torch.equal(mapped[:,0],torch.full_like(q,109.5)) and torch.equal(mapped[:,-1],torch.full_like(q,158.5))
assert bool((coefficient[q<qref]<0).all() and (coefficient[q>qref]>0).all())
assert bool((adapted>=0).all()) and bool((adapted.argmax(1)==centers).all()) and torch.equal(derived_centers,centers.to(torch.float64))
flanks=adapted[:,110:159].diff(dim=1);left=torch.arange(110,158,device=device)[None,:]<centers[:,None]
assert bool(torch.where(left,flanks>=0,flanks<=0).all())
old=observe_generated_carrier(mean,ste=False,q_floor=.001,return_indices=True)
new=observe_generated_carrier(adapted,ste=False,q_floor=.001,return_indices=True)
assert not bool(old['flags']['pre_ADC_out_of_range'].any() or new['flags']['pre_ADC_out_of_range'].any())
before=morphology018(old['window_V'].cpu().numpy())['FWHM_ns'];after=morphology018(new['window_V'].cpu().numpy())['FWHM_ns']
assert after[0]<before[0] and after[-1]>before[-1]
assert float(adapted[0].max())>float(mean[0].max()) and float(adapted[-1].max())<float(mean[-1].max())
zero,_,_,identity_norm,_=conditional_width050(mean,q,0.,qref)
assert torch.equal(zero,torch.zeros_like(mean)) and torch.equal(identity_norm,torch.ones_like(q))
identity=observe_generated_carrier(mean+zero,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(old[k],identity[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(old['flags'][k],identity['flags'][k]) for k in old['flags'])
reverse=torch.arange(31,-1,-1,device=device)
permuted=conditional_width050(mean[reverse],q[reverse],.5,qref)[0]
assert torch.equal(permuted,delta[reverse])
single=torch.cat([conditional_width050(mean[i:i+1],q[i:i+1],.5,qref)[0] for i in range(32)])
batch_error=float((single-delta).abs().max());assert batch_error<1e-12
scaled=conditional_width050(3.*mean,q,.5,qref)[0]
assert torch.allclose(scaled,3.*delta,rtol=1e-12,atol=1e-12)
signed_mean=mean.clone();signed_mean[:,147]-=.003
signed_delta=conditional_width050(signed_mean,q,.5,qref)[0]
assert float(signed_delta.sum(1).abs().max())<1e-12 and float(canonical_charge008(signed_delta).abs().max())<1e-12
assert torch.equal(signed_delta[:,:110],torch.zeros_like(signed_delta[:,:110])) and torch.equal(signed_delta[:,159:],torch.zeros_like(signed_delta[:,159:]))
plateau=torch.full((1,384),.25,device=device,dtype=torch.float64);pq=canonical_charge008(plateau)
old_delta,_,old_map=conditional_width047(plateau,pq,.5,qref)
flat_delta,_,new_map,flat_norm,flat_center=conditional_width050(plateau,pq,.5,qref)
assert torch.equal(flat_delta,torch.zeros_like(plateau)) and float(flat_center[0])==112.
assert torch.equal(flat_norm,torch.ones_like(pq))
old_spread=float((plateau+old_delta)[:,110:159].max()-(plateau+old_delta)[:,110:159].min())
assert old_spread>.1
fixed_delta=conditional_width049(mean,q,.5,qref)[0]
fixed_obs=observe_generated_carrier(mean+fixed_delta,ste=False,q_floor=.001,return_indices=True)
fixed_width=morphology018(fixed_obs['window_V'].cpu().numpy())['FWHM_ns']
assert after[0]<fixed_width[0]
write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=65,width_test_rows=32,flat_rows=1,signed_shape_rows=32,
    checkpoint_loads=0,dataset_loads=0,AD_queries=0,full_integral_residual_Vns=full_error,canonical_Q_residual_Vns=canonical_error,
    support_exact=True,minimum_mapped_interval_ns=float(mapped.diff(dim=1).min()),
    amplitude_normalization_min=float(norm.min()),amplitude_normalization_max=float(norm.max()),
    low_Q_width_before_ns=float(before[0]),low_Q_width_after_ns=float(after[0]),
    high_Q_width_before_ns=float(before[-1]),high_Q_width_after_ns=float(after[-1]),
    Gaussian_peaks120_through132_and_monotone_flanks_preserved=True,flat_voltage_exactly_unchanged=True,
    flat_old047_peak_to_peak_V=old_spread,fixed_support_and_derived_centers_verified=True,low_Q_fixed126_width_ns=float(fixed_width[0]),zero_strength_exact_native_observation=True,
    batching_max_absolute_error_V=batch_error,batching_tolerance_V=1e-12,
    positive_homogeneity_and_row_permutation_verified=True,signed_lobe_integral_and_support_verified=True,whole_sign_reversal_not_claimed=True,manufactured_no_ADC_clipping=True,scientific_acceptance=False))
print('response050 manufactured event-centered voltage-resampling checks: PASS')
