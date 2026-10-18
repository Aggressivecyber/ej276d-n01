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

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64)
shape=torch.exp(-.5*((t-126.)/8.).square());shape[:100]=0.
q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=q[:,None]*shape/shape[110:366].sum();qref=3.680695072375209
delta,coefficient,mapped,norm=conditional_width049(mean,q,.5,qref);adapted=mean+delta
full_error=float(delta.sum(1).abs().max());canonical_error=float(canonical_charge008(delta).abs().max())
assert max(full_error,canonical_error)<1e-12
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,159:],mean[:,159:])
assert float(mapped.diff(dim=1).min())>.5 and bool((norm>0).all())
assert torch.equal(mapped[:,0],torch.full_like(q,109.5)) and torch.equal(mapped[:,-1],torch.full_like(q,158.5))
assert bool((coefficient[q<qref]<0).all() and (coefficient[q>qref]>0).all())
assert bool((adapted>=0).all()) and bool((adapted.argmax(1)==126).all())
assert bool((adapted[:,110:126].diff(dim=1)>=0).all()) and bool((adapted[:,126:159].diff(dim=1)<=0).all())
old=observe_generated_carrier(mean,ste=False,q_floor=.001,return_indices=True)
new=observe_generated_carrier(adapted,ste=False,q_floor=.001,return_indices=True)
assert not bool(old['flags']['pre_ADC_out_of_range'].any() or new['flags']['pre_ADC_out_of_range'].any())
before=morphology018(old['window_V'].cpu().numpy())['FWHM_ns'];after=morphology018(new['window_V'].cpu().numpy())['FWHM_ns']
assert after[0]<before[0] and after[-1]>before[-1]
assert float(adapted[0].max())>float(mean[0].max()) and float(adapted[-1].max())<float(mean[-1].max())
zero,_,_,identity_norm=conditional_width049(mean,q,0.,qref)
assert torch.equal(zero,torch.zeros_like(mean)) and torch.equal(identity_norm,torch.ones_like(q))
identity=observe_generated_carrier(mean+zero,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(old[k],identity[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(old['flags'][k],identity['flags'][k]) for k in old['flags'])
reverse=torch.arange(31,-1,-1,device=device)
permuted=conditional_width049(mean[reverse],q[reverse],.5,qref)[0]
assert torch.equal(permuted,delta[reverse])
single=torch.cat([conditional_width049(mean[i:i+1],q[i:i+1],.5,qref)[0] for i in range(32)])
batch_error=float((single-delta).abs().max());assert batch_error<1e-12
negative=conditional_width049(-mean,q,.5,qref)[0]
scaled=conditional_width049(3.*mean,q,.5,qref)[0]
assert torch.allclose(negative,-delta,rtol=0,atol=1e-12) and torch.allclose(scaled,3.*delta,rtol=1e-12,atol=1e-12)
plateau=torch.full((1,384),.25,device=device,dtype=torch.float64);pq=canonical_charge008(plateau)
old_delta,_,old_map=conditional_width047(plateau,pq,.5,qref)
flat_delta,_,new_map,flat_norm=conditional_width049(plateau,pq,.5,qref)
assert torch.equal(old_map,new_map) and torch.equal(flat_delta,torch.zeros_like(plateau))
assert torch.equal(flat_norm,torch.ones_like(pq))
old_spread=float((plateau+old_delta)[:,110:159].max()-(plateau+old_delta)[:,110:159].min())
assert old_spread>.1
write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=33,width_test_rows=32,flat_rows=1,
    checkpoint_loads=0,dataset_loads=0,AD_queries=0,full_integral_residual_Vns=full_error,canonical_Q_residual_Vns=canonical_error,
    support_exact=True,minimum_mapped_interval_ns=float(mapped.diff(dim=1).min()),
    amplitude_normalization_min=float(norm.min()),amplitude_normalization_max=float(norm.max()),
    low_Q_width_before_ns=float(before[0]),low_Q_width_after_ns=float(after[0]),
    high_Q_width_before_ns=float(before[-1]),high_Q_width_after_ns=float(after[-1]),
    Gaussian_peak126_and_monotone_flanks_preserved=True,flat_voltage_exactly_unchanged=True,
    flat_old047_peak_to_peak_V=old_spread,same_edge_map_verified=True,zero_strength_exact_native_observation=True,
    batching_max_absolute_error_V=batch_error,batching_tolerance_V=1e-12,
    signed_homogeneity_and_row_permutation_verified=True,manufactured_no_ADC_clipping=True,scientific_acceptance=False))
print('response049 manufactured voltage-resampling checks: PASS')
