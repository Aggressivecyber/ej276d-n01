"""Manufactured independent voltage-warp and linear residual-projection checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response068 import voltage_warp068
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
knots=np.geomspace(.5,12.,8);u=np.arange(256)
sp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(4.,11.,8)])
tp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(3.,14.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=sp.tolist(),target_unit_Q_profiles=tp.tolist())
axis=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device);q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
m=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());m=q[:,None]*m/m[:,110:366].sum(1,keepdim=True);m[:,:100]=0.;m[:,110]=-.001;m[:,175]=-.002;m[:,150]=-.001
z=.002*torch.sin(axis[None,:]/3.+torch.arange(32,device=device)[:,None])
d,dr,rows,query=voltage_warp068(m,z,q,template);mean_err=random_err=query_err=scale_err=0.
for i in range(32):
 old=m[i].cpu().numpy();rnd=z[i].cpu().numpy();peak=np.argmax(old[112:144])+112;loc=np.arange(110,176)-peak+16;mask=(loc>=2)&(old[110:176]>0.);logq=float(q[i].log())
 ss=np.array([np.interp(logq,np.log(knots),sp[:,j]) for j in range(256)])[np.maximum(loc,2)]
 tt=np.array([np.interp(logq,np.log(knots*1.01),tp[:,j]) for j in range(256)])[np.maximum(loc,2)]
 sc=np.r_[0.,np.cumsum(.5*(ss[:-1]+ss[1:]))];tc=np.r_[0.,np.cumsum(.5*(tt[:-1]+tt[1:]))]
 p=peak-110;a=max(0,peak-124);expected_query=np.arange(66,dtype=float)
 for left,right in ((a,p),(p,65)):
  fraction=(tc[left:right+1]-tc[left])/(tc[right]-tc[left]);desired=sc[left]+fraction*(sc[right]-sc[left])
  expected_query[left:right+1]=np.interp(desired,sc,np.arange(66))
 expected_query[[a,p,65]]=[a,p,65]
 positive=old[110:176]*mask;v=np.interp(expected_query,np.arange(66),positive)*mask;alpha=positive.sum()/v.sum();mapped=alpha*v
 active=rnd[110:176]*mask;provisional=alpha*np.interp(expected_query,np.arange(66),active)*mask
 rr=provisional-(provisional.sum()-active.sum())*mapped/positive.sum()-active
 mean_err=max(mean_err,float(np.max(np.abs(mapped-positive-d[i,110:176].cpu().numpy()))))
 random_err=max(random_err,float(np.max(np.abs(rr-dr[i,110:176].cpu().numpy()))))
 query_err=max(query_err,float(np.max(np.abs(expected_query+110-query[i].cpu().numpy()))));scale_err=max(scale_err,abs(alpha-float(rows['amplitude_scale'][i])))
 assert np.array_equal(d[i,110:176].cpu().numpy()[~mask],np.zeros((~mask).sum()))
 assert np.array_equal(dr[i,110:176].cpu().numpy()[~mask],np.zeros((~mask).sum()))
assert mean_err<1e-12 and random_err<1e-12 and query_err<1e-10 and scale_err<1e-12
assert d.sum(1).abs().max()<1e-12 and dr.sum(1).abs().max()<1e-12
assert r.canonical_charge008(d).abs().max()<1e-12 and r.canonical_charge008(dr).abs().max()<1e-12
assert query.diff(dim=1).min()>=0. and query.min()>=110. and query.max()<=175.
assert torch.equal((m+d)[m<=0.],m[m<=0.]) and torch.equal(rows['old_mean_peak_index'],rows['new_mean_peak_index'])
for value in (d,dr):assert torch.equal(value[:,:110],torch.zeros_like(value[:,:110])) and torch.equal(value[:,176:],torch.zeros_like(value[:,176:]))
_,opposite,_,_=voltage_warp068(m,-z,q,template);assert torch.equal(opposite,-dr)
_,zero,_,_=voltage_warp068(m,torch.zeros_like(z),q,template);assert torch.equal(zero,torch.zeros_like(zero))
_,scaled,_,_=voltage_warp068(m,2*z,q,template);linearity=float((scaled-2*dr).abs().max());assert linearity<1e-14
rev=torch.arange(31,-1,-1,device=device);rd,rr,_,rq=voltage_warp068(m[rev],z[rev],q[rev],template);assert torch.equal(rd,d[rev]) and torch.equal(rr,dr[rev]) and torch.equal(rq,query[rev])
cd,cr,_,cq=voltage_warp068(m.repeat(5,1),z.repeat(5,1),q.repeat(5),template);chunk_voltage=max(float((cd-d.repeat(5,1)).abs().max()),float((cr-dr.repeat(5,1)).abs().max()));chunk_query=float((cq-query.repeat(5,1)).abs().max());assert chunk_voltage<1e-12 and chunk_query<1e-10
identity=dict(template,target_unit_Q_profiles=sp.tolist(),target_Q_centers_Vns=knots.tolist());iz,ir,_,iq=voltage_warp068(m,z,q,identity);identity_err=max(float(iz.abs().max()),float(ir.abs().max()));assert identity_err<1e-12 and (iq-(torch.arange(66,device=device)+110)).abs().max()<1e-10
dual=torch.zeros((1,384),device=device,dtype=torch.float64);dual[0,118:129]=torch.tensor([.01,.03,.08,.16,.23,.20,.14,.10,.06,.03,.01],device=device);dual[0,130:140]=torch.tensor([.03,.06,.11,.18,.21,.17,.12,.07,.03,.01],device=device)
dd,_,ds,dq=voltage_warp068(dual,torch.zeros_like(dual),torch.tensor([3.],device=device,dtype=torch.float64),template);dual_new=dual+dd
assert int(ds['old_mean_peak_index'][0])==122 and int(ds['new_mean_peak_index'][0])==122 and float(dq[0,12])==122.
crests=((dual_new[0,1:-1]>dual_new[0,:-2])&(dual_new[0,1:-1]>=dual_new[0,2:])&(dual_new[0,1:-1]>0.)).sum();assert int(crests)>=2
actual=__import__('json').loads((Path(__file__).resolve().parents[1]/'receipts/response062_20261018/preparation_01/TEMPLATE.json').read_text())
template_min=min(np.asarray(actual[k])[:,2:80].min() for k in ('source_unit_Q_profiles','target_unit_Q_profiles'));assert template_min>0.
r.write_json(args.out,dict(status='PASS',device=str(device),independent_mean_error_V=mean_err,independent_random_error_V=random_err,independent_query_error_samples=query_err,independent_amplitude_scale_error=scale_err,
 mean_full_integral_error_Vns=float(d.sum(1).abs().max()),random_full_integral_error_Vns=float(dr.sum(1).abs().max()),canonical_mean_error_Vns=float(r.canonical_charge008(d).abs().max()),canonical_random_error_Vns=float(r.canonical_charge008(dr).abs().max()),
 monotone_bounded_queries=True,peak_anchor_exact=True,negative_and_outer_support_exact=True,antithetic_exact=True,zero_residual_exact=True,linearity_error_V=linearity,permutation_exact=True,chunk_voltage_error_V=chunk_voltage,chunk_query_error_samples=chunk_query,identity_error_V=identity_err,
 dual_crest_survives=True,dual_crest_primary_peak_retained=True,frozen_template_minimum_positive=template_min,manufactured_rows=32,chunk_rows=160,TRAIN_decodes=0,checkpoint_loads=0,scientific_acceptance=False))
print('068 peak-anchored voltage warp semantics: PASS')
