"""Signed coupling semantics: analytic map, conserved charge, conditional linearity."""
import argparse
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response067 import random_transport067
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device);knots=np.geomspace(.5,12.,8)
sp=np.ones((8,256));tp=sp.copy();sp[:,15:17]=[.5,.5];tp[:,15:17]=[.25,.75]
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=knots.tolist(),source_unit_Q_profiles=sp.tolist(),target_unit_Q_profiles=tp.tolist())
m=torch.zeros((2,384),device=device,dtype=torch.float64);m[:,125:127]=torch.tensor([.08,.1],device=device,dtype=torch.float64);m[:,140]=-.003
z=torch.zeros_like(m);z[:,125:127]=torch.tensor([[.002,-.004],[-.002,.004]],device=device,dtype=torch.float64);z[:,140]=.001;z[:,180]=-.001
q=r.canonical_charge008(m);d=random_transport067(m,z,q,True,template);expected=z.clone();expected[:,125:127]=torch.tensor([[.001,-.003],[-.001,.003]],device=device,dtype=torch.float64)
analytic=float((z+d-expected).abs().max());assert analytic<1e-14 and torch.equal(d[0],-d[1])
axis=torch.arange(384,device=device,dtype=torch.float64);centers=112+torch.arange(32,device=device);q=torch.logspace(np.log10(.25),np.log10(24.),32,device=device,dtype=torch.float64)
m=torch.exp(-.5*((axis[None,:]-centers[:,None])/6.).square());m=q[:,None]*m/m[:,110:366].sum(1,keepdim=True);m[:,:100]=0.;m[:,110]=-.001;m[:,175]=-.002
u=np.arange(256);sp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(4.,11.,8)]);tp=np.array([np.exp(-.5*((u-16)/v)**2)+.001 for v in np.linspace(3.,14.,8)])
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=(knots*1.01).tolist(),source_unit_Q_profiles=sp.tolist(),target_unit_Q_profiles=tp.tolist())
z=.002*torch.sin(axis[None,:]/3.+torch.arange(32,device=device)[:,None]);d=random_transport067(m,z,q,True,template);err=0.
for i in range(32):
 old=m[i].cpu().numpy();rnd=z[i].cpu().numpy();peak=np.argmax(old[112:144])+112;loc=np.arange(110,176)-peak+16;mask=(loc>=2)&(old[110:176]>0.);query=float(q[i].log())
 ss=np.array([np.interp(query,np.log(knots),sp[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask;tt=np.array([np.interp(query,np.log(knots*1.01),tp[:,j]) for j in range(256)])[np.maximum(loc,2)]*mask
 sc=np.cumsum(ss);sc=np.r_[0.,sc/sc[-1]];tc=np.cumsum(tt);tc=np.r_[0.,tc/tc[-1]];width=np.diff(sc);K=np.zeros((66,66))
 for a in range(66):
  for b in range(66):
   if width[b]>0.:K[a,b]=max(0.,min(tc[a+1],sc[b+1])-max(tc[a],sc[b]))/width[b]
 active=rnd[110:176]*mask;ex=np.zeros(384);ex[110:176]=K@active-active;err=max(err,float(np.max(np.abs(ex-d[i].cpu().numpy()))))
 assert np.array_equal(d[i].cpu().numpy()[110:176][~mask],np.zeros((~mask).sum()))
assert err<1e-12 and d.sum(1).abs().max()<1e-12 and r.canonical_charge008(d).abs().max()<1e-12
assert torch.equal(random_transport067(m,-z,q,True,template),-d) and torch.equal(random_transport067(m,z,q,False,template),torch.zeros_like(d))
assert torch.equal(random_transport067(m,torch.zeros_like(z),q,True,template),torch.zeros_like(d))
rev=torch.arange(31,-1,-1,device=device);assert torch.equal(random_transport067(m[rev],z[rev],q[rev],True,template),d[rev])
chunk=float((random_transport067(m.repeat(5,1),z.repeat(5,1),q.repeat(5),True,template)-d.repeat(5,1)).abs().max());assert chunk<1e-12
identity=dict(template,target_unit_Q_profiles=sp.tolist(),target_Q_centers_Vns=knots.tolist());identity_error=float(random_transport067(m,z,q,True,identity).abs().max());assert identity_error<1e-14
assert torch.equal(d[:,:110],torch.zeros_like(d[:,:110])) and torch.equal(d[:,176:],torch.zeros_like(d[:,176:]))
r.write_json(args.out,dict(status='PASS',device=str(device),analytic_signed_map_error_V=analytic,independent_numpy_error_V=err,full_integral_error_Vns=float(d.sum(1).abs().max()),canonical_Q_error_Vns=float(r.canonical_charge008(d).abs().max()),negative_and_outer_support_exact=True,antithetic_zero_mean_exact=True,zero_residual_exact=True,disabled_coupling_exact=True,permutation_exact=True,chunk_error_V=chunk,identity_error_V=identity_error,manufactured_rows=32,chunk_boundary_rows=160,TRAIN_decodes=0,checkpoint_loads=0,scientific_acceptance=False))
print('067 signed deterministic mass coupling semantics: PASS')
