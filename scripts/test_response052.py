"""Manufactured half-height knot semantics, independently known crossings; no data/CP."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from n01.stats018 import morphology018
from run_response052 import conditional_sides052

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64)
qref=3.680695072375209
triangle=torch.where(t<=126.,(t-114.)/12.,(150.-t)/24.).clamp(min=0.)[None,:]*.1
q=canonical_charge008(triangle)
delta,displacement,mapped,norm,center,knots=conditional_sides052(triangle,q,1.,qref,[-2.,0.,3.,0.])
assert torch.equal(knots,torch.tensor([[109.5,120.,126.,138.,158.5]],device=device))
assert torch.equal(mapped,torch.tensor([[109.5,118.,126.,141.,158.5]],device=device))
adapted=triangle+delta
observed=morphology018(adapted[:,110:366].cpu().numpy())
assert abs(float(observed['FWHM_ns'][0])-23.)<1e-12
assert abs(float(observed['left_crossing50_ns'][0])-8.)<1e-12
assert abs(float(observed['right_crossing50_ns'][0])-31.)<1e-12
assert int(adapted.argmax(1)[0])==126
leading=conditional_sides052(triangle,q,1.,qref,[-2.,0.,0.,0.])[0]
trailing=conditional_sides052(triangle,q,1.,qref,[0.,0.,3.,0.])[0]
lm=morphology018((triangle+leading)[:,110:366].cpu().numpy())
tm=morphology018((triangle+trailing)[:,110:366].cpu().numpy())
assert abs(float(lm['right_crossing50_ns'][0])-28.)<1e-12
assert abs(float(tm['left_crossing50_ns'][0])-10.)<1e-12
assert abs(float(lm['FWHM_ns'][0])-20.)<1e-12 and abs(float(tm['FWHM_ns'][0])-21.)<1e-12

centers=120+torch.arange(32,device=device)%13
shape=torch.exp(-.5*((t[None,:]-centers[:,None])/8.).square());shape[:,:100]=0.
q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True)
fit=json.loads((Path(__file__).resolve().parents[1]/'receipts/response052_20261018/preparation_01/HALF_HEIGHT_SIDES_FIT.json').read_text())
theta=fit['coefficients_ns']
delta,displacement,mapped,norm,center,knots=conditional_sides052(mean,q,.05,qref,theta)
adapted=mean+delta
full_error=float(delta.sum(1).abs().max());canonical_error=float(canonical_charge008(delta).abs().max())
assert max(full_error,canonical_error)<1e-12
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,159:],mean[:,159:])
assert bool((mapped.diff(dim=1)>0).all() and (norm>0).all())
assert torch.equal(mapped[:,[0,2,4]],knots[:,[0,2,4]])
assert bool((adapted>=0).all()) and torch.equal(adapted.argmax(1),centers)
old=observe_generated_carrier(mean,ste=False,q_floor=.001,return_indices=True)
new=observe_generated_carrier(adapted,ste=False,q_floor=.001,return_indices=True)
assert not bool(old['flags']['pre_ADC_out_of_range'].any() or new['flags']['pre_ADC_out_of_range'].any())
before=morphology018(old['window_V'].cpu().numpy())['FWHM_ns'];after=morphology018(new['window_V'].cpu().numpy())['FWHM_ns']
assert after[0]<before[0] and after[-1]>before[-1]
zero,_,_,identity_norm,_,_=conditional_sides052(mean,q,0.,qref,theta)
assert torch.equal(zero,torch.zeros_like(mean)) and torch.equal(identity_norm,torch.ones_like(q))
identity=observe_generated_carrier(mean+zero,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(old[k],identity[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(old['flags'][k],identity['flags'][k]) for k in old['flags'])
reverse=torch.arange(31,-1,-1,device=device)
permuted=conditional_sides052(mean[reverse],q[reverse],.05,qref,theta)[0]
assert torch.equal(permuted,delta[reverse])
single=torch.cat([conditional_sides052(mean[i:i+1],q[i:i+1],.05,qref,theta)[0] for i in range(32)])
batch_error=float((single-delta).abs().max());assert batch_error<1e-12
scaled=conditional_sides052(3.*mean,q,.05,qref,theta)[0]
assert torch.allclose(scaled,3.*delta,rtol=1e-12,atol=1e-12)
signed=mean.clone();signed[:,155]-=.003
sd=conditional_sides052(signed,q,.05,qref,theta)[0]
assert float(sd.sum(1).abs().max())<1e-12 and float(canonical_charge008(sd).abs().max())<1e-12
try:
    conditional_sides052(torch.ones_like(mean[:1]),q[:1],.05,qref,theta)
except RuntimeError as exc:
    assert str(exc)=='052 half-height knot domain failed'
else:
    raise AssertionError('Unmeasurable half-height must fail, never replace or discard a row')
try:
    conditional_sides052(triangle,canonical_charge008(triangle),1.,qref,[-20.,0.,0.,0.])
except RuntimeError as exc:
    assert str(exc)=='052 mapped knots nonmonotone'
else:
    raise AssertionError('Nonmonotone map must fail, never clip a fitted displacement')
write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=66,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    independently_known_triangle_left50_ns=118.,independently_known_triangle_right50_ns=141.,triangle_FWHM_ns=float(observed['FWHM_ns'][0]),
    leading_only_and_trailing_only_crossings_independent=True,full_integral_residual_Vns=full_error,canonical_Q_residual_Vns=canonical_error,
    support_exact=True,peak_knot_exact=True,Gaussian_peaks120_through132_preserved=True,
    minimum_mapped_interval_ns=float(mapped.diff(dim=1).min()),minimum_interval_ratio=float((mapped.diff(dim=1)/knots.diff(dim=1)).min()),
    amplitude_normalization_min=float(norm.min()),amplitude_normalization_max=float(norm.max()),
    low_Q_width_before_ns=float(before[0]),low_Q_width_after_ns=float(after[0]),high_Q_width_before_ns=float(before[-1]),high_Q_width_after_ns=float(after[-1]),
    zero_strength_exact_native_observation=True,batching_max_absolute_error_V=batch_error,positive_homogeneity_and_row_permutation_verified=True,
    signed_lobe_integral_and_support_verified=True,undefined_knot_and_nonmonotone_domain_rejected_without_imputation=True,
    manufactured_no_ADC_clipping=True,scientific_acceptance=False))
print('response052 independent half-height knot checks: PASS')
