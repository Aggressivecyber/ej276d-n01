"""Manufactured conservation, signed-lobe, response-domain and LP semantics; no data/CP."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response057 import core_redistribution057,solve057

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64)
qref=3.680695072375209

# Two positive core samples admit an independently evaluated scalar area partition.
two=torch.zeros((1,384),device=device,dtype=torch.float64)
two[0,125]=.08;two[0,126]=.10;two[0,150]=.02;two[0,200]=-.001
q=r.canonical_charge008(two)
delta,rows=core_redistribution057(two,q,[.25,0.],qref)
c1=.08*math.sin(math.pi*15/31)**2;c2=.10*math.sin(math.pi*16/31)**2
odds=(c1/c2)*math.exp(.25*(math.exp(-1/18)-1))
new_c2=math.fsum([c1,c2])/(1+odds);new_c1=new_c2*odds
expected=two.cpu().numpy().copy();expected[0,125]+=(new_c1-c1);expected[0,126]+=(new_c2-c2)
scalar_error=float(np.max(np.abs((two+delta).cpu().numpy()-expected)))
assert scalar_error<1e-14
assert float(delta[0,126])>0 and float(delta[0,125])<0

centers=112+torch.arange(32,device=device)
shape=torch.exp(-.5*((t[None,:]-centers[:,None])/8.).square());shape[:,:100]=0.
input_q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=input_q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True)
mean[:,110]=-.002;mean[:,140]=-.003;mean[:,150]=-.001;mean[:,240]=-.0005
L=float(torch.log(input_q/qref).abs().max());scale=np.array([.25,.25/L])
integral_error=0.;negative_error=0.;moment_violations=0;maximum_tilt=0.
for signs in ((-1,-1),(-1,1),(1,-1),(1,1)):
    coeff=scale*np.array(signs)
    d,z=core_redistribution057(mean,input_q,coeff,qref);adapted=mean+d
    assert torch.equal(d[:,:111],torch.zeros_like(d[:,:111]))
    assert torch.equal(d[:,141:],torch.zeros_like(d[:,141:]))
    assert torch.equal(adapted[mean<=0],mean[mean<=0])
    assert bool((adapted[mean>0]>=0).all())
    for native_peak in range(112,144):
        assert torch.equal(adapted[:,native_peak+30:],mean[:,native_peak+30:])
    neg=float(((-adapted).clamp_min(0).sum(1)-(-mean).clamp_min(0).sum(1)).abs().max())
    err=max(float(d.sum(1).abs().max()),float(r.canonical_charge008(d).abs().max()))
    integral_error=max(integral_error,err);negative_error=max(negative_error,neg)
    assert err<1e-12 and neg==0.
    maximum_tilt=max(maximum_tilt,float(z['log_tilt'].abs().max()))
    assert maximum_tilt<=.5+1e-12
    peak=mean[:,112:144].argmax(1)+112
    weights=torch.exp(-.5*((t[None,:]-peak[:,None])/3.).square())
    signed_moment=(d*weights).sum(1)*z['log_tilt']
    moment_violations+=int((signed_moment < -1e-14).sum())
assert moment_violations==0

coeff=[0.,-.25/L];d,z=core_redistribution057(mean,input_q,coeff,qref)
reverse=torch.arange(31,-1,-1,device=device)
permuted=core_redistribution057(mean[reverse],input_q[reverse],coeff,qref)[0]
assert torch.equal(permuted,d[reverse])
single=torch.cat([core_redistribution057(mean[i:i+1],input_q[i:i+1],coeff,qref)[0] for i in range(32)])
batch_error=float((single-d).abs().max());assert batch_error<1e-12
scaled=core_redistribution057(3*mean,input_q,coeff,qref)[0]
assert torch.allclose(scaled,3*d,atol=1e-12,rtol=1e-12)
zero,identity_rows=core_redistribution057(mean,input_q,[0.,0.],qref)
assert torch.equal(zero,torch.zeros_like(mean)) and torch.equal(identity_rows['normalization'],torch.ones_like(input_q))
carrier=mean+.0001*torch.sin(t/3)[None,:]
newcarrier=r._ExactZeroMeanAddition015.apply(carrier,zero)
assert torch.equal(newcarrier,carrier)
old=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True)
new=r.observe_generated_carrier(newcarrier,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(old[k],new[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(old['flags'][k],new['flags'][k]) for k in old['flags'])
for testmean,testcoeff,message in ((-torch.ones_like(mean),[0.,0.],'no positive movable core area'),(mean,[.51,0.],'fixed log-tilt domain exceeded')):
    try:core_redistribution057(testmean,input_q,testcoeff,qref)
    except RuntimeError as exc:assert message in str(exc)
    else:raise AssertionError('Declared scientific domain failure must stop without dropping rows')

# Analytic optimum: each of six morphology gains is .5+.2*x0+.1*x1.
labels=[[dict(hypothesis=h,family='values',metric=k) for h in r.HYPOTHESES for k in ('wave','joint_morphology')+r.MORPH] for _ in (0,1)]
base=np.array([[-.05 if x['metric'] in ('wave','joint_morphology') else -.5 for x in labels[0]]]*2)
upper=np.zeros_like(base);J=np.array([[0.,0.] if x['metric'] in ('wave','joint_morphology') else [-.2,-.1] for x in labels[0]])
probes=np.array([[np.array([base[s]-J[:,j],base[s]+J[:,j]]) for j in (0,1)] for s in (0,1)])
sol=solve057(base,probes,upper,labels,dict(useful_wave_relative_gain=.02,useful_joint_relative_gain=.02))
assert np.allclose(sol['x'],[1.,1.,.8],atol=1e-12,rtol=0)
assert max(abs(v) for v in sol['KKT'].values())<1e-12
assert float(np.max(np.abs(sol['central_second_difference'])))<1e-12
r.write_json(args.out,dict(status='PASS',device=str(device),checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    two_sample_scalar_partition_max_error_V=scalar_error,full_and_canonical_integral_max_error_Vns=integral_error,
    signed_negative_area_max_change_Vns=negative_error,nonpositive_samples_exact=True,
    all_possible_bounded_native_anchor_tails_exact=True,zero_exact_full_carrier_and_native_observation=True,
    row_permutation_exact=True,batch_max_error_V=batch_error,positive_homogeneity=True,
    peak_weighted_moment_violations=moment_violations,maximum_absolute_log_tilt=maximum_tilt,
    invalid_scientific_domains_fail=True,manufactured_LP_calls=1,analytic_LP_solution=sol['x'],KKT=sol['KKT'],
    scientific_acceptance=False))
print('response057 manufactured conservation and LP semantics: PASS')
