"""Known overlap map, template transport, event contrasts and native identity."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response065 import shape_transport065

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device);knots=np.geomspace(.5,12.,8)
s=np.ones((8,256));t=s.copy();s[:,15:17]=[.5,.5];t[:,15:17]=[.25,.75]
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=knots.tolist(),source_unit_Q_profiles=s.tolist(),target_unit_Q_profiles=t.tolist())
x=torch.zeros((2,384),device=device,dtype=torch.float64);x[:,125:127]=torch.tensor([[.08,.10],[.01,.17]],device=device,dtype=torch.float64);x[:,140]=-.003;x[:,180]=.02
q=r.canonical_charge008(x);d,z=shape_transport065(x,q,True,template);expected=x.cpu().numpy().copy();expected[:,125:127]=[[.04,.14],[.005,.175]]
analytic_error=float(np.max(np.abs((x+d).cpu().numpy()-expected)));assert analytic_error<1e-14
pair=x[:1].repeat(2,1);pair[1,125]+=.001;pair[1,126]-=.001;pd,pz=shape_transport065(pair,q[:1].repeat(2),True,template)
pair_error=float(torch.max(torch.abs(((pair+pd)[1]-(pair+pd)[0])[125:127]-torch.tensor([.0005,-.0005],device=device,dtype=torch.float64))));assert pair_error<1e-14
axis=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device);q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
shape=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());shape[:,:100]=0.;mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True);mean[:,110]=-.001;mean[:,175]=-.002;mean[:,240]=-.0005
for row in range(31):mean[row,centers[row]+1]=mean[row,centers[row]]*(1.-(0. if row%2==0 else 1e-10))
u=np.arange(256);s=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(4.,11.,8)]);t=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(3.,14.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=s.tolist(),target_unit_Q_profiles=t.tolist());d,z=shape_transport065(mean,q,True,template);adapted=mean+d
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,176:],mean[:,176:]) and torch.equal(adapted[mean<=0.],mean[mean<=0.]) and bool((adapted[mean>0.]>=0.).all())
area_error=float(d.sum(1).abs().max());assert area_error<1e-12 and float(r.canonical_charge008(d).abs().max())<1e-12 and float(z['negative_area_change_Vns'].abs().max())==0.
numeric=[];column_error=0.;template_error=0.;identity_error=0.
for row in range(32):
    old=mean[row].cpu().numpy();peak=int(np.argmax(old[112:144]))+112;loc=np.arange(110,176)-peak+16;query=float(np.log(float(q[row])));mask=(loc>=2)&(old[110:176]>0.);positive=np.where(mask,old[110:176],0.)
    ss=np.array([np.interp(query,np.log(knots),s[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask;tt=np.array([np.interp(query,np.log(knots*1.01),t[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask
    sc=np.cumsum(ss);sc=np.r_[0.,sc/sc[-1]];tc=np.cumsum(tt);tc=np.r_[0.,tc/tc[-1]];width=np.diff(sc);target=np.diff(tc)
    K=np.zeros((66,66))
    for i in range(66):
        for j in range(66):
            if width[j]>0.:K[i,j]=max(0.,min(tc[i+1],sc[j+1])-max(tc[i],sc[j]))/width[j]
    out=old.copy();out[110:176]+=K@positive-positive;numeric.append(out-old);column_error=max(column_error,float(np.max(np.abs(K.sum(0)[mask]-1.))));template_error=max(template_error,float(np.max(np.abs(K@width-target))))
    assert np.array_equal(out[110:176][loc<2],old[110:176][loc<2])
numpy_error=float(np.max(np.abs(np.array(numeric)-d.cpu().numpy())));assert max(numpy_error,column_error,template_error)<1e-12
rev=torch.arange(31,-1,-1,device=device);assert torch.equal(shape_transport065(mean[rev],q[rev],True,template)[0],d[rev]);one=torch.cat([shape_transport065(mean[i:i+1],q[i:i+1],True,template)[0] for i in range(32)]);batch_error=float((one-d).abs().max());assert batch_error<1e-12
many=shape_transport065(mean.repeat(5,1),q.repeat(5),True,template)[0];chunk_error=float((many-d.repeat(5,1)).abs().max());assert chunk_error<1e-12
identity=dict(template,target_unit_Q_profiles=s.tolist(),target_Q_centers_Vns=knots.tolist());di,_=shape_transport065(mean,q,True,identity);identity_error=float(di.abs().max());assert identity_error<1e-14
zero,_=shape_transport065(mean,q,False,template);assert torch.equal(zero,torch.zeros_like(zero));carrier=mean+.0001*torch.sin(axis/3)[None,:];new=r._ExactZeroMeanAddition015.apply(carrier,zero);assert torch.equal(new,carrier)
before=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True);after=r.observe_generated_carrier(new,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(before[k],after[k]) for k in ('window_V','baseline100','peak_index')) and all(torch.equal(before['flags'][k],after['flags'][k]) for k in before['flags'])
assert all(bool(z[k].any()) for k in ('source_clamped_low','source_clamped_high','target_clamped_low','target_clamped_high'))
r.write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=32,chunk_boundary_rows=160,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,known_two_sample_error_V=analytic_error,full_and_canonical_area_max_error_Vns=area_error,independent_numpy_formula_error_V=numpy_error,column_stochastic_error=column_error,template_pushforward_error=template_error,same_condition_event_contrast_transport_error_V=pair_error,negative_relative_prefix_and_outer_samples_exact=True,no_new_negative_samples=True,all_bounded_old_peak_positions_and_competing_crests_exercised=True,identity_template_error_V=identity_error,zero_native_readout_exact=True,permutation_exact=True,batch_error_V=batch_error,chunk_boundary_error_V=chunk_error,separate_knot_clamps_exercised=True,scientific_acceptance=False))
print('response065 fixed conditional positive-mass transport semantics: PASS')
