"""Manufactured local-restitution semantics; no checkpoints or datasets."""
import argparse
from pathlib import Path
import torch
from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from run_response042 import local_restitution042

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(2)
t=torch.arange(384,dtype=torch.float64)
mean=torch.linspace(.03,.1,32)[:,None]*torch.exp(-.5*((t-126)/5).square());mean[:,:100]=0
donor=torch.exp(-.5*((t-140)/6).square());donor[:100]=0;donor/=canonical_charge008(donor)
front=torch.exp(-.5*((t-108)/3).square());front[:100]=0;front-=canonical_charge008(front)*donor
delta=torch.linspace(-.0005,.0005,32)[:,None]*front
new,correction,lost=local_restitution042(delta,2.)
assert bool((lost<0).any() and (lost>0).any())
charge_error=float(canonical_charge008(correction).abs().max());assert charge_error<1e-12
assert torch.equal(correction[:,:100],torch.zeros(32,100,dtype=torch.float64))
assert torch.equal(correction[:,158:],torch.zeros(32,226,dtype=torch.float64))
assert torch.equal(new[:,158:],delta[:,158:]) and torch.equal(new[:,:100],delta[:,:100])
zero,c0,l0=local_restitution042(torch.zeros_like(delta),2.);assert torch.equal(zero,torch.zeros_like(zero))
original,c0,l0=local_restitution042(delta,0.);assert torch.equal(original,delta) and torch.count_nonzero(c0)==0
scaled,_,_=local_restitution042(3*delta,2.);assert torch.allclose(scaled,3*new,atol=1e-16,rtol=1e-14)
permutation=torch.arange(31,-1,-1);reordered,_,_=local_restitution042(delta[permutation],2.)
assert torch.equal(reordered,new[permutation])
single=torch.cat([local_restitution042(delta[i:i+1],2.)[0] for i in range(32)])
assert torch.equal(single,new)
noise=.0002*torch.cos(t/7).expand(32,-1)
a=observe_generated_carrier(mean+delta+noise,ste=False,q_floor=.001,return_indices=True)
b=observe_generated_carrier(mean+original+noise,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(a[k],b[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(a['flags'][k],b['flags'][k]) for k in a['flags'])
write_json(args.out,dict(status='PASS',manufactured_rows=32,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    canonical_local_Q_max_residual_Vns=charge_error,prefix_and_tail_support_exact=True,
    signed_lost_area_retained=True,zero_fraction_exact_old_observation=True,zero_front_exact_identity=True,
    input_amplitude_scaling_verified=True,row_permutation_and_batching_invariant=True,scientific_acceptance=False))
print('response042 local restitution manufactured checks: PASS')
