"""Analytic area mixture, competing peaks, event contrasts and unchanged observation."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response061 import shape_mixture061

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
knots=np.geomspace(.5,12.,8);simple=np.zeros((8,256));simple[:,15]=.2;simple[:,16]=.8
template=dict(Q_centers_Vns=knots.tolist(),positive_unit_Q_profiles=simple.tolist())
x=torch.zeros((1,384),device=device,dtype=torch.float64);x[0,125]=.08;x[0,126]=.10;x[0,140]=-.003;x[0,180]=.02;x[0,240]=-.001
d,z=shape_mixture061(x,r.canonical_charge008(x),.125,template);expected=x.cpu().numpy().copy();expected[0,125]=.0745;expected[0,126]=.1055
analytic_error=float(np.max(np.abs((x+d).cpu().numpy()-expected)));assert analytic_error<1e-14
t=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device)
shape=torch.exp(-.5*((t[None,:]-centers[:,None])/6.).square());shape[:,:100]=0.
q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True);mean[:,110]=-.001;mean[:,175]=-.002;mean[:,240]=-.0005
# Near-equal and exactly equal competing crests were the failure mechanism in057.
double=mean.clone()
for row in range(31):double[row,centers[row]+1]=double[row,centers[row]]*(1.-(0. if row%2==0 else 1e-10))
u=np.arange(256);profiles=np.array([np.exp(-.5*((u-16)/s)**2) for s in np.linspace(3.,10.,8)])
profiles/=profiles.sum(1,keepdims=True);template=dict(Q_centers_Vns=knots.tolist(),positive_unit_Q_profiles=profiles.tolist())
d,z=shape_mixture061(double,q,.125,template);adapted=double+d
assert torch.equal(adapted[:,:110],double[:,:110]) and torch.equal(adapted[:,176:],double[:,176:])
assert torch.equal(adapted[double<=0.],double[double<=0.]) and bool((adapted[double>0.]>0.).all())
assert torch.equal(z['old_mean_peak_index'],z['new_mean_peak_index'])
area_error=float(d.sum(1).abs().max());assert area_error<1e-12 and float(r.canonical_charge008(d).abs().max())<1e-12
assert float(z['negative_area_change_Vns'].abs().max())==0.
assert z['lookup_clamped_low'].any() and z['lookup_clamped_high'].any()
numeric=[]
for row in range(32):
    old=double[row].cpu().numpy();peak=int(np.argmax(old[112:144]))+112
    query=float(np.log(float(q[row])));s=np.array([np.interp(query,np.log(knots),profiles[:,j]) for j in range(256)])
    local=old[110:176];position=np.arange(110,176)-peak+16;weights=np.array([s[k] if k>=0 and value>0 else 0. for k,value in zip(position,local)])
    target=np.maximum(local,0.).sum()*weights/weights.sum()
    one=np.zeros(384);one[110:176]=.125*(target-np.maximum(local,0.));numeric.append(one)
numpy_error=float(np.max(np.abs(np.array(numeric)-d.cpu().numpy())));assert numpy_error<1e-12
rev=torch.arange(31,-1,-1,device=device);assert torch.equal(shape_mixture061(double[rev],q[rev],.125,template)[0],d[rev])
one=torch.cat([shape_mixture061(double[i:i+1],q[i:i+1],.125,template)[0] for i in range(32)]);batch_error=float((one-d).abs().max());assert batch_error<1e-12
pair=mean[10:11].repeat(2,1);pair[1,116]+=.0001;pair[1,117]-=.0001;pairq=q[10:11].repeat(2)
pd,_=shape_mixture061(pair,pairq,.125,template)
content_error=float(((pair+pd)[1]-(pair+pd)[0]-.875*(pair[1]-pair[0])).abs().max());assert content_error<1e-13
zero,_=shape_mixture061(double,q,0.,template);assert torch.equal(zero,torch.zeros_like(zero))
carrier=double+.0001*torch.sin(t/3)[None,:];new=r._ExactZeroMeanAddition015.apply(carrier,zero);assert torch.equal(new,carrier)
before=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True);after=r.observe_generated_carrier(new,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(before[k],after[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(before['flags'][k],after['flags'][k]) for k in before['flags'])
for f in (-.01,.126):
    try:shape_mixture061(double,q,f,template)
    except ValueError as exc:assert 'fraction' in str(exc)
    else:raise AssertionError('Out-of-contract fraction must fail')
r.write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=32,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,
    known_two_sample_partition_max_error_V=analytic_error,full_and_canonical_area_max_error_Vns=area_error,independent_numpy_interpolation_max_error_V=numpy_error,
    negative_samples_exact=True,all_bounded_peaks_preserved=True,equal_and_near_equal_crest_cases=True,identity_permutation_exact=True,batch_max_error_V=batch_error,
    same_Q_same_area_event_contrast_retained_fraction=.875,event_contrast_error_V=content_error,zero_operator_and_native_observation_exact=True,
    clamped_Q_domain_exercised=True,scientific_acceptance=False))
print('response061 peak-preserving positive-area mixture semantics: PASS')
