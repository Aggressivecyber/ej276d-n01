"""Known additive residual, positivity boundary, event contrasts and readout identity."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response063 import shape_residual063
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device);knots=np.geomspace(.5,12.,8)
s=np.ones((8,256));t=s.copy();s[:,15:17]=[.5,.5];t[:,15:17]=[.25,.75];template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=knots.tolist(),source_unit_Q_profiles=s.tolist(),target_unit_Q_profiles=t.tolist())
x=torch.zeros((2,384),device=device,dtype=torch.float64);x[:,125:127]=torch.tensor([[.08,.10],[.01,.17]],device=device);x[:,140]=-.003;x[:,180]=.02
q=r.canonical_charge008(x);d,z=shape_residual063(x,q,True,template);expected=x.cpu().numpy().copy();C=expected[:,125:127].sum(1);direction=np.stack([-.25*C,.25*C],1);scale=np.minimum(1.,expected[:,125]/(.25*C));scale[1]=np.nextafter(scale[1],0.);expected[:,125:127]+=scale[:,None]*direction
analytic_error=float(np.max(np.abs((x+d).cpu().numpy()-expected)));assert analytic_error<1e-14 and not bool(z['positivity_boundary_reached'][0]) and bool(z['positivity_boundary_reached'][1]);assert bool(((x+d)[:,125:127]>=0.).all())
pair=x[:1].repeat(2,1);pair[1,125]+=.001;pair[1,126]-=.001;pd,pz=shape_residual063(pair,q[:1].repeat(2),True,template);assert bool((pz['applied_scale']==1.).all());pair_error=float(((pair+pd)[1]-(pair+pd)[0]-(pair[1]-pair[0])).abs().max());assert pair_error<1e-14
axis=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device);q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
shape=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());shape[:,:100]=0.;mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True);mean[:,110]=-.001;mean[:,175]=-.002;mean[:,240]=-.0005
for row in range(31):mean[row,centers[row]+1]=mean[row,centers[row]]*(1.-(0. if row%2==0 else 1e-10))
u=np.arange(256);s=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(4.,11.,8)]);t=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(3.,14.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=s.tolist(),target_unit_Q_profiles=t.tolist());d,z=shape_residual063(mean,q,True,template);adapted=mean+d
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,176:],mean[:,176:]) and torch.equal(adapted[mean<=0.],mean[mean<=0.]) and bool((adapted[mean>0.]>=0.).all())
area_error=float(d.sum(1).abs().max());assert area_error<1e-12 and float(r.canonical_charge008(d).abs().max())<1e-12 and float(z['negative_area_change_Vns'].abs().max())==0.
numeric=[];scale_error=0.
for row in range(32):
    old=mean[row].cpu().numpy();peak=int(np.argmax(old[112:144]))+112;loc=np.arange(110,176)-peak+16;query=float(np.log(float(q[row])));mask=(loc>=2)&(old[110:176]>0.);positive=np.where(mask,old[110:176],0.);C=math.fsum(positive)
    ss=np.array([np.interp(query,np.log(knots),s[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask;tt=np.array([np.interp(query,np.log(knots*1.01),t[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask
    dd=C*tt/math.fsum(tt)-C*ss/math.fsum(ss);raw=min((positive[j]/(-dd[j]) for j in range(66) if dd[j]<0.),default=math.inf);scale=min(1.,raw)
    if raw<=1.:scale=np.nextafter(scale,0.)
    expected=np.zeros(384);expected[110:176]=scale*dd;numeric.append(expected);scale_error=max(scale_error,abs(scale-float(z['applied_scale'][row])))
    assert np.array_equal(adapted[row,110:176].cpu().numpy()[loc<2],old[110:176][loc<2])
numpy_error=float(np.max(np.abs(np.array(numeric)-d.cpu().numpy())));assert max(numpy_error,scale_error)<1e-12
rev=torch.arange(31,-1,-1,device=device);assert torch.equal(shape_residual063(mean[rev],q[rev],True,template)[0],d[rev]);one=torch.cat([shape_residual063(mean[i:i+1],q[i:i+1],True,template)[0] for i in range(32)]);batch_error=float((one-d).abs().max());assert batch_error<1e-12
identity=dict(template,target_unit_Q_profiles=s.tolist(),target_Q_centers_Vns=knots.tolist());di,_=shape_residual063(mean,q,True,identity);assert torch.equal(di,torch.zeros_like(di))
zero,_=shape_residual063(mean,q,False,template);assert torch.equal(zero,torch.zeros_like(zero));carrier=mean+.0001*torch.sin(axis/3)[None,:];new=r._ExactZeroMeanAddition015.apply(carrier,zero);assert torch.equal(new,carrier)
before=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True);after=r.observe_generated_carrier(new,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(before[k],after[k]) for k in ('window_V','baseline100','peak_index')) and all(torch.equal(before['flags'][k],after['flags'][k]) for k in before['flags'])
assert all(bool(z[k].any()) for k in ('source_clamped_low','source_clamped_high','target_clamped_low','target_clamped_high'))
r.write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=32,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,known_uncapped_and_capped_two_sample_error_V=analytic_error,full_and_canonical_area_max_error_Vns=area_error,independent_numpy_formula_error_V=numpy_error,independent_feasible_scale_error=scale_error,uncapped_same_condition_event_contrast_error_V=pair_error,negative_relative_prefix_and_outer_samples_exact=True,no_new_negative_samples=True,all_bounded_old_peak_positions_and_competing_crests_exercised=True,identity_and_zero_native_readout_exact=True,permutation_exact=True,batch_error_V=batch_error,separate_knot_clamps_exercised=True,scientific_acceptance=False))
print('response063 positivity-limited additive residual semantics: PASS')
