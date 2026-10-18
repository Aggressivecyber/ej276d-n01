"""Analytic conditional odds transport, inverse, ties, support and native readout."""
import argparse,math
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response062 import shape_ratio062
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
knots=np.geomspace(.5,12.,8);source=np.ones((8,256));target=source.copy();target[:,15]=2.;target[:,16]=.5
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=knots.tolist(),source_unit_Q_profiles=source.tolist(),target_unit_Q_profiles=target.tolist())
x=torch.zeros((1,384),device=device,dtype=torch.float64);x[0,125]=.08;x[0,126]=.10;x[0,140]=-.003;x[0,180]=.02
q=r.canonical_charge008(x);d,z=shape_ratio062(x,q,True,template);expect=x.cpu().numpy().copy();expect[0,125]=.18*.16/.21;expect[0,126]=.18*.05/.21
analytic_error=float(np.max(np.abs((x+d).cpu().numpy()-expect)));assert analytic_error<1e-14 and int(z['old_mean_peak_index'][0])==126 and int(z['new_mean_peak_index'][0])==125
obs=r.observe_generated_carrier(x+d,ste=False,q_floor=.001,return_indices=True);assert int(obs['peak_index'][0])==125
axis=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device);q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
shape=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());shape[:,:100]=0.;mean=q[:,None]*shape/shape[:,110:366].sum(1,keepdim=True);mean[:,110]=-.001;mean[:,175]=-.002;mean[:,240]=-.0005
for row in range(31):mean[row,centers[row]+1]=mean[row,centers[row]]*(1.-(0. if row%2==0 else 1e-10))
u=np.arange(256);source=np.array([np.exp(-.5*((u-16)/s)**2)+.001 for s in np.linspace(4.,11.,8)]);target=np.array([np.exp(-.5*((u-16)/s)**2)+.001 for s in np.linspace(3.,14.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=source.tolist(),target_unit_Q_profiles=target.tolist())
d,z=shape_ratio062(mean,q,True,template);adapted=mean+d
assert torch.equal(adapted[:,:110],mean[:,:110]) and torch.equal(adapted[:,176:],mean[:,176:]) and torch.equal(adapted[mean<=0.],mean[mean<=0.])
area_error=float(d.sum(1).abs().max());assert area_error<1e-12 and float(r.canonical_charge008(d).abs().max())<1e-12 and float(z['negative_area_change_Vns'].abs().max())==0.
assert all(bool(z[k].any()) for k in ('source_clamped_low','source_clamped_high','target_clamped_low','target_clamped_high'))
numeric=[];inverse_error=0.;odds_error=0.
for row in range(32):
    old=mean[row].cpu().numpy();peak=int(np.argmax(old[112:144]))+112;query=float(np.log(float(q[row])));loc=np.arange(110,176)-peak+16
    s=np.array([np.interp(query,np.log(knots),source[:,j]) for j in range(256)]);t=np.array([np.interp(query,np.log(knots*1.01),target[:,j]) for j in range(256)])
    mask=(loc>=2)&(old[110:176]>0.);weights=t[np.maximum(loc,2)]/s[np.maximum(loc,2)];positive=np.where(mask,old[110:176],0.);area=math.fsum(positive);weighted=positive*weights;new=area*weighted/math.fsum(weighted)
    one=np.zeros(384);one[110:176]=new-positive;numeric.append(one)
    actual=adapted[row,110:176].cpu().numpy()[mask];back=actual/weights[mask];recovered=area*back/math.fsum(back);inverse_error=max(inverse_error,float(np.max(np.abs(recovered-positive[mask]))))
    odds_error=max(odds_error,float(np.max(np.abs(np.log(actual/actual[0])-np.log(positive[mask]/positive[mask][0])-np.log(weights[mask]/weights[mask][0])))))
    assert np.array_equal(adapted[row,110:176].cpu().numpy()[loc<2],old[110:176][loc<2])
numpy_error=float(np.max(np.abs(np.array(numeric)-d.cpu().numpy())));assert max(numpy_error,inverse_error)<1e-12 and odds_error<1e-11
pair=mean[10:11].repeat(2,1);pair[1,120]+=.0001;pair[1,121]-=.0001;pairq=q[10:11].repeat(2);pd,_=shape_ratio062(pair,pairq,True,template)
before=torch.log(pair[:,120]/pair[:,121]);after=torch.log((pair+pd)[:,120]/(pair+pd)[:,121]);pair_error=float(((after[1]-after[0])-(before[1]-before[0])).abs());assert pair_error<1e-12
rev=torch.arange(31,-1,-1,device=device);assert torch.equal(shape_ratio062(mean[rev],q[rev],True,template)[0],d[rev])
one=torch.cat([shape_ratio062(mean[i:i+1],q[i:i+1],True,template)[0] for i in range(32)]);batch_error=float((one-d).abs().max());assert batch_error<1e-12
identity=dict(template,source_unit_Q_profiles=source.tolist(),target_unit_Q_profiles=source.tolist(),target_Q_centers_Vns=knots.tolist());zero_identity,_=shape_ratio062(mean,q,True,identity);assert torch.equal(zero_identity,torch.zeros_like(zero_identity))
zero,_=shape_ratio062(mean,q,False,template);assert torch.equal(zero,torch.zeros_like(zero));carrier=mean+.0001*torch.sin(axis/3)[None,:];native=r._ExactZeroMeanAddition015.apply(carrier,zero);assert torch.equal(native,carrier)
before=r.observe_generated_carrier(carrier,ste=False,q_floor=.001,return_indices=True);after=r.observe_generated_carrier(native,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(before[k],after[k]) for k in ('window_V','baseline100','peak_index')) and all(torch.equal(before['flags'][k],after['flags'][k]) for k in before['flags'])
for setting in (.125,1):
    try:shape_ratio062(mean,q,setting,template)
    except ValueError as exc:assert 'boolean' in str(exc)
    else:raise AssertionError('Only fixed disabled/enabled setting admitted')
r.write_json(args.out,dict(status='PASS',device=str(device),manufactured_rows=32,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0,known_two_sample_partition_max_error_V=analytic_error,full_and_canonical_area_max_error_Vns=area_error,independent_numpy_max_error_V=numpy_error,conditional_inverse_max_error_V=inverse_error,log_shape_odds_max_error=odds_error,same_condition_pair_log_odds_contrast_error=pair_error,negative_and_relative_prefix_samples_exact=True,all_bounded_old_peak_positions_exercised=True,competing_crests_and_explicit_native_peak_migration_exercised=True,identity_and_zero_observation_exact=True,permutation_exact=True,batch_max_error_V=batch_error,separate_Q_knot_clamps_exercised=True,scientific_acceptance=False))
print('response062 conditional shape odds semantics: PASS')
