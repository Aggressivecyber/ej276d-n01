"""Manufactured amplitude order, signed areas and exact-zero native response; no data/CP."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response058 import monotone_core058
from run_response057 import solve057

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
device=torch.device(args.device);t=torch.arange(384,device=device,dtype=torch.float64);qref=3.680695072375209
two=torch.zeros((1,384),device=device,dtype=torch.float64);two[0,125]=.08;two[0,126]=.10;two[0,140]=-.003;two[0,160]=.02;two[0,200]=-.001
q=r.canonical_charge008(two);delta,rows=monotone_core058(two,q,[.25,0.],qref)
gamma=math.exp(.25);new_peak=.18/(1+.8**gamma);expected=two.cpu().numpy().copy();expected[0,126]=new_peak;expected[0,125]=.18-new_peak
scalar_error=float(np.max(np.abs((two+delta).cpu().numpy()-expected)));assert scalar_error<1e-14
assert float(delta[0,126])>0 and float(delta[0,125])<0

centers=112+torch.arange(32,device=device)
shape=torch.exp(-.5*((t[None,:]-centers[:,None])/6.).square());shape[:,:100]=0.
input_q=torch.logspace(np.log10(.8),np.log10(8.),32,device=device,dtype=torch.float64)
mean=input_q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True)
mean[:,110]=-.002;mean[:,140]=-.003;mean[:,150]=-.001;mean[:,240]=-.0005
L=float(torch.log(input_q/qref).abs().max());scale=np.array([.25,.25/L]);integral_error=0.;negative_error=0.;minimum_rank_gap=0.;maximum_boundary_delta=0.
for signs in ((-1,-1),(-1,1),(1,-1),(1,1)):
    d,z=monotone_core058(mean,input_q,scale*np.array(signs),qref);adapted=mean+d
    assert torch.equal(d[:,:110],torch.zeros_like(d[:,:110])) and torch.equal(d[:,142:],torch.zeros_like(d[:,142:]))
    assert torch.equal(adapted[mean<=0],mean[mean<=0]) and bool((adapted[mean>0]>=0).all())
    for native_peak in range(112,144):assert torch.equal(adapted[:,native_peak+30:],mean[:,native_peak+30:])
    neg=float(((-adapted).clamp_min(0).sum(1)-(-mean).clamp_min(0).sum(1)).abs().max())
    err=max(float(d.sum(1).abs().max()),float(r.canonical_charge008(d).abs().max()))
    integral_error=max(integral_error,err);negative_error=max(negative_error,neg);assert err<1e-12 and neg==0.
    assert float(z['log_tilt'].abs().max())<=.5+1e-12
    order=mean[:,110:142].argsort(dim=1,stable=True);gap=adapted[:,110:142].gather(1,order).diff(dim=1)
    minimum_rank_gap=min(minimum_rank_gap,float(gap.min()));assert minimum_rank_gap>=-1e-14
    maximum_boundary_delta=max(maximum_boundary_delta,float(d[:,141].abs().max()))

# Position does not distinguish equal crests; near-equal competing crests retain their original order.
double=torch.zeros((2,384),device=device,dtype=torch.float64);double[:,110:142]=.04
double[:,121]=.1;double[0,124]=.1;double[1,124]=.1*(1-1e-4);double[:,142:180]=.015
minimum_crest_margin=1.
for u in (-.5,.5):
    d,z=monotone_core058(double,r.canonical_charge008(double),[u,0.],qref);adapted=double+d
    assert float(adapted[0,121])==float(adapted[0,124])
    assert float(adapted[1,121])>float(adapted[1,124]);minimum_crest_margin=min(minimum_crest_margin,float(adapted[1,121]-adapted[1,124]))
    assert torch.equal(adapted[:,112:144].argmax(1),torch.tensor([9,9],device=device))

gaussian=torch.exp(-.5*((t-126.)/6.).square())[None,:].repeat(2,1);gaussian[:,:100]=0.
gq=torch.tensor([.8,8.],device=device,dtype=torch.float64);gaussian*=gq[:,None]/gaussian[:,110:366].sum(1,keepdim=True)
gd,_=monotone_core058(gaussian,gq,[0.,-.25/L],qref)
before=r.morphology018(gaussian[:,110:366].cpu().numpy());after=r.morphology018((gaussian+gd)[:,110:366].cpu().numpy())
assert after['FWHM_ns'][0]<before['FWHM_ns'][0] and after['FWHM_ns'][1]>before['FWHM_ns'][1]
assert after['peak_voltage_V'][0]>before['peak_voltage_V'][0] and after['peak_voltage_V'][1]<before['peak_voltage_V'][1]

coeff=[0.,-.25/L];d,z=monotone_core058(mean,input_q,coeff,qref);reverse=torch.arange(31,-1,-1,device=device)
assert torch.equal(monotone_core058(mean[reverse],input_q[reverse],coeff,qref)[0],d[reverse])
single=torch.cat([monotone_core058(mean[i:i+1],input_q[i:i+1],coeff,qref)[0] for i in range(32)])
batch_error=float((single-d).abs().max());assert batch_error<1e-12
assert torch.allclose(monotone_core058(3*mean,input_q,coeff,qref)[0],3*d,rtol=1e-12,atol=1e-12)
zero,identity=monotone_core058(mean,input_q,[0.,0.],qref)
assert torch.equal(zero,torch.zeros_like(mean)) and torch.equal(identity['normalization'],torch.ones_like(input_q))
carrier=mean+.0001*torch.sin(t/3)[None,:];newcarrier=r._ExactZeroMeanAddition015.apply(carrier,zero);assert torch.equal(newcarrier,carrier)
old=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True);new=r.observe_generated_carrier(newcarrier,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(old[k],new[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(old['flags'][k],new['flags'][k]) for k in old['flags'])
for m,c,message in ((-torch.ones_like(mean),[0.,0.],'no positive core area'),(mean,[.51,0.],'fixed log-power domain exceeded')):
    try:monotone_core058(m,input_q,c,qref)
    except RuntimeError as exc:assert message in str(exc)
    else:raise AssertionError('Invalid scientific domain must fail without row replacement')

labels=[[dict(hypothesis=h,family='values',metric=k) for h in r.HYPOTHESES for k in ('wave','joint_morphology')+r.MORPH] for _ in (0,1)]
base=np.array([[-.05 if x['metric'] in ('wave','joint_morphology') else -.5 for x in labels[0]]]*2)
J=np.array([[0.,0.] if x['metric'] in ('wave','joint_morphology') else [-.2,-.1] for x in labels[0]])
probes=np.array([[np.array([base[s]-J[:,j],base[s]+J[:,j]]) for j in (0,1)] for s in (0,1)])
sol=solve057(base,probes,np.zeros_like(base),labels,dict(useful_wave_relative_gain=.02,useful_joint_relative_gain=.02))
assert np.allclose(sol['x'],[1.,1.,.8],atol=1e-12,rtol=0) and max(abs(v) for v in sol['KKT'].values())<1e-12
r.write_json(args.out,dict(status='PASS',device=str(device),checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    two_sample_known_power_partition_max_error_V=scalar_error,full_and_canonical_integral_max_error_Vns=integral_error,negative_area_change_Vns=negative_error,
    minimum_rank_difference_V=minimum_rank_gap,equal_crests_exactly_equal=True,near_equal_crests_order_preserved=True,minimum_near_equal_crest_margin_V=minimum_crest_margin,
    maximum_manufactured_boundary_correction_V=maximum_boundary_delta,edge_corrections_are_allowed_and_recorded=True,
    low_Q_FWHM_before_after_ns=[float(before['FWHM_ns'][0]),float(after['FWHM_ns'][0])],high_Q_FWHM_before_after_ns=[float(before['FWHM_ns'][1]),float(after['FWHM_ns'][1])],
    all_possible_fixed_native_anchor_tails_exact=True,zero_exact_full_carrier_and_native_readout=True,permutation_exact=True,batch_max_error_V=batch_error,positive_homogeneity=True,
    invalid_scientific_domains_fail=True,reused057_LP_analytic_solution=sol['x'],KKT=sol['KKT'],scientific_acceptance=False))
print('response058 manufactured monotone amplitude and area semantics: PASS')
