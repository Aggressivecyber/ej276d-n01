"""Independent manufactured crossing queries, split mean and residual-integral checks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response070 import split_warp070
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
knots=np.geomspace(.5,12.,8);u=np.arange(256)
sp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(4.,7.5,8)])
tp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(3.,8.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=sp.tolist(),target_unit_Q_profiles=tp.tolist())
axis=torch.arange(384,device=device,dtype=torch.float64);centers=120+torch.arange(32,device=device)%7;q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
m=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());m=q[:,None]*m/m[:,110:366].sum(1,keepdim=True);m[:,:100]=0.;m[:,151:176]+=.001;m[:,110]=-.001;m[:,175]=-.002;m[:,150]=-.001
z=.002*torch.sin(axis[None,:]/3.+torch.arange(32,device=device)[:,None])
d,dr,rows,query,source_knots,target_knots=split_warp070(m,z,q,template);mean_err=random_err=query_err=scale_err=knot_err=0.
for i in range(32):
 old=m[i].cpu().numpy();rnd=z[i].cpu().numpy();peak=np.argmax(old[112:144])+112;loc=np.arange(110,176)-peak+16;mask=(loc>=2)&(old[110:176]>0.);logq=float(q[i].log())
 ss=np.array([np.interp(logq,np.log(knots),sp[:,j]) for j in range(256)])
 tt=np.array([np.interp(logq,np.log(knots*1.01),tp[:,j]) for j in range(256)])
 kk=[];a=max(110,peak-14)
 for profile in (ss,tt):
  left=[];right=[]
  for fraction in (.5,.9):
   level=fraction*profile[16];lo=np.flatnonzero(profile[:16]<level)[-1];hi=np.flatnonzero(profile[17:]<level)[0]+17
   left.append(lo+(level-profile[lo])/(profile[lo+1]-profile[lo]));right.append(hi-1+(level-profile[hi-1])/(profile[hi]-profile[hi-1]))
  kk.append(np.array([a,left[0]+peak-16,left[1]+peak-16,peak,right[1]+peak-16,right[0]+peak-16,175.]))
 expected_query=np.interp(np.arange(110,176),kk[1],kk[0]);expected_query[:a-110+1]=np.arange(110,a+1);expected_query[[peak-110,65]]=[peak,175.]
 knot_err=max(knot_err,float(np.abs(kk[0]-source_knots[i].cpu().numpy()).max()),float(np.abs(kk[1]-target_knots[i].cpu().numpy()).max()))
 positive=old[110:176]*mask;v=np.interp(expected_query,np.arange(110,176),positive)*mask;active=rnd[110:176]*mask;raw_random=np.interp(expected_query,np.arange(110,176),active)*mask
 mapped=np.zeros(66);rr=np.zeros(66);front=np.arange(110,176)<peak+30
 for side,select in [('front',front),('tail',~front)]:
  alpha=positive[select].sum()/v[select].sum();mapped[select]=alpha*v[select];part=alpha*raw_random[select]
  rr[select]=part-(part.sum()-active[select].sum())*mapped[select]/positive[select].sum()-active[select]
  scale_err=max(scale_err,abs(alpha-float(rows[side+'_amplitude_scale'][i])))
 mean_err=max(mean_err,float(np.max(np.abs(mapped-positive-d[i,110:176].cpu().numpy()))));random_err=max(random_err,float(np.max(np.abs(rr-dr[i,110:176].cpu().numpy()))))
 query_err=max(query_err,float(np.max(np.abs(expected_query-query[i].cpu().numpy()))))
 assert int(rows['reference_split_index'][i])==peak+30
 for select in (front,~front):assert abs(d[i,110:176].cpu().numpy()[select].sum())<1e-12 and abs(dr[i,110:176].cpu().numpy()[select].sum())<1e-12
 assert abs(float(rows['correction_split_step_V'][i])-(d[i,peak+30]-d[i,peak+29]).item())<1e-15
 assert np.array_equal(d[i,110:176].cpu().numpy()[~mask],np.zeros((~mask).sum())) and np.array_equal(dr[i,110:176].cpu().numpy()[~mask],np.zeros((~mask).sum()))
assert mean_err<1e-12 and random_err<1e-12 and query_err<1e-10 and scale_err<1e-12 and knot_err<1e-12
assert d.sum(1).abs().max()<1e-12 and dr.sum(1).abs().max()<1e-12 and r.canonical_charge008(d).abs().max()<1e-12 and r.canonical_charge008(dr).abs().max()<1e-12
assert query.diff(dim=1).min()>=0. and query.min()>=110. and query.max()<=175.
assert torch.equal((m+d)[m<=0.],m[m<=0.]) and torch.equal(rows['old_mean_peak_index'],rows['new_mean_peak_index'])
for value in (d,dr):assert torch.equal(value[:,:110],torch.zeros_like(value[:,:110])) and torch.equal(value[:,176:],torch.zeros_like(value[:,176:]))
_,opposite,_,_,_,_=split_warp070(m,-z,q,template);assert torch.equal(opposite,-dr)
_,zero,_,_,_,_=split_warp070(m,torch.zeros_like(z),q,template);assert torch.equal(zero,torch.zeros_like(zero))
_,scaled,_,_,_,_=split_warp070(m,2*z,q,template);linearity=float((scaled-2*dr).abs().max());assert linearity<1e-14
rev=torch.arange(31,-1,-1,device=device);rd,rr,_,rq,_,_=split_warp070(m[rev],z[rev],q[rev],template);assert torch.equal(rd,d[rev]) and torch.equal(rr,dr[rev]) and torch.equal(rq,query[rev])
cd,cr,_,cq,_,_=split_warp070(m.repeat(5,1),z.repeat(5,1),q.repeat(5),template);chunk_voltage=max(float((cd-d.repeat(5,1)).abs().max()),float((cr-dr.repeat(5,1)).abs().max()));chunk_query=float((cq-query.repeat(5,1)).abs().max());assert chunk_voltage<1e-12 and chunk_query<1e-10
identity=dict(template,target_unit_Q_profiles=sp.tolist(),target_Q_centers_Vns=knots.tolist());iz,ir,_,iq,_,_=split_warp070(m,z,q,identity);identity_err=max(float(iz.abs().max()),float(ir.abs().max()));assert identity_err<1e-12 and (iq-(torch.arange(66,device=device)+110)).abs().max()<1e-10
dual=torch.zeros((1,384),device=device,dtype=torch.float64);dual[0,118:129]=torch.tensor([.01,.03,.08,.16,.23,.20,.14,.10,.06,.03,.01],device=device);dual[0,130:140]=torch.tensor([.03,.06,.11,.18,.21,.17,.12,.07,.03,.01],device=device)
dual[0,150:175]=.001
dd,_,ds,dq,_,_=split_warp070(dual,torch.zeros_like(dual),torch.tensor([3.],device=device,dtype=torch.float64),template);dual_new=dual+dd
assert int(ds['old_mean_peak_index'][0])==122 and int(ds['new_mean_peak_index'][0])==122 and float(dq[0,12])==122.
crests=((dual_new[0,1:-1]>dual_new[0,:-2])&(dual_new[0,1:-1]>=dual_new[0,2:])&(dual_new[0,1:-1]>0.)).sum();assert int(crests)>=2
sptri=np.tile(np.maximum(1.-np.abs(u-16)/8.,0.),(8,1));tptri=np.tile(np.maximum(1.-np.abs(u-16)/12.,0.),(8,1))
triangle=dict(template,source_unit_Q_profiles=sptri.tolist(),target_unit_Q_profiles=tptri.tolist());tri=(1.-(axis-126).abs()/8.).clamp_min(0.)[None,:]
tri[0,150:175]=.001
td,_,_,tq,_,_=split_warp070(tri,torch.zeros_like(tri),r.canonical_charge008(tri),triangle);newtri=tri+td
halfheight_error=max(abs(float(newtri[0,120]/newtri[0,126])-.5),abs(float(newtri[0,132]/newtri[0,126])-.5));assert halfheight_error<1e-14
assert float(tq[0,10])==122. and float(tq[0,22])==130.
unsupported=torch.zeros_like(tri);unsupported[0,112]=1.;caught=False
try:split_warp070(unsupported,torch.zeros_like(unsupported),torch.tensor([1.],device=device,dtype=torch.float64),triangle)
except ValueError as exc:assert 'ordered support domain' in str(exc);caught=True
assert caught
empty=torch.zeros_like(tri);empty[0,126]=1.;caught_empty=False
try:split_warp070(empty,torch.zeros_like(empty),torch.tensor([1.],device=device,dtype=torch.float64),triangle)
except ValueError as exc:assert 'empty positive front/tail area' in str(exc);caught_empty=True
assert caught_empty
r.write_json(args.out,dict(status='PASS',device=str(device),independent_mean_error_V=mean_err,independent_random_error_V=random_err,independent_query_error_samples=query_err,independent_amplitude_scale_error=scale_err,independent_knot_error_samples=knot_err,
 mean_split_integral_max_error_Vns=max(float(rows[k].abs().max()) for k in ('reference_front_integral_residual_Vns','reference_tail_integral_residual_Vns')),random_split_integral_max_error_Vns=max(float(rows[k].abs().max()) for k in ('random_reference_front_integral_residual_Vns','random_reference_tail_integral_residual_Vns')),empty_partition_fails_explicitly=True,
 mean_full_integral_error_Vns=float(d.sum(1).abs().max()),random_full_integral_error_Vns=float(dr.sum(1).abs().max()),canonical_mean_error_Vns=float(r.canonical_charge008(d).abs().max()),canonical_random_error_Vns=float(r.canonical_charge008(dr).abs().max()),
 monotone_bounded_queries=True,peak_anchor_exact=True,negative_and_outer_support_exact=True,antithetic_exact=True,zero_residual_exact=True,linearity_error_V=linearity,permutation_exact=True,chunk_voltage_error_V=chunk_voltage,chunk_query_error_samples=chunk_query,identity_error_V=identity_err,
 dual_crest_survives=True,dual_crest_primary_peak_retained=True,analytic_triangle_halfheight_ratio_error=halfheight_error,unsupported_knot_domain_fails_explicitly=True,manufactured_rows=32,chunk_rows=160,TRAIN_decodes=0,checkpoint_loads=0,scientific_acceptance=False))
print('070 conditional crossing-level voltage warp semantics: PASS')
