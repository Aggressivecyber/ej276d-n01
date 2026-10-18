"""Manufactured conservative peak diffusion; no dataset or checkpoint reads."""
import argparse
from pathlib import Path
import torch
from n01.data import write_json
from n01.model008 import canonical_charge008
from n01.observation008 import observe_generated_carrier
from run_response043 import peak_diffusion043

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(2)
t=torch.arange(384,dtype=torch.float64)
width=torch.linspace(3.,6.,32)[:,None];amp=torch.linspace(.03,.1,32)[:,None]
positive=amp*torch.exp(-.5*((t-126)/width).square());positive[:,:100]=0
mean=positive.clone();mean[16:]=-mean[16:]
delta=peak_diffusion043(mean,.5);adapted=mean+delta
assert torch.equal(delta[:,:112],torch.zeros_like(delta[:,:112]))
assert torch.equal(delta[:,145:],torch.zeros_like(delta[:,145:]))
full_error=float(delta.sum(1).abs().max());canonical_error=float(canonical_charge008(delta).abs().max())
assert max(full_error,canonical_error)<1e-12
neighbors=torch.stack((torch.cat((mean[:,:1],mean[:,:-1]),1),mean,torch.cat((mean[:,1:],mean[:,-1:]),1)))
assert bool((adapted>=neighbors.amin(0)-1e-16).all() and (adapted<=neighbors.amax(0)+1e-16).all())
assert bool((adapted[:16].amax(1)<mean[:16].amax(1)).all())
edge=torch.arange(383,dtype=torch.float64)+.5
g=torch.where((edge>112)&(edge<144),torch.sin(torch.pi*(edge-112)/32).square(),0.)
center=1-.5*(torch.cat((g,torch.zeros(1)))+torch.cat((torch.zeros(1),g)))
assert float(center.min())>=0
assert torch.equal(peak_diffusion043(torch.ones_like(mean),.5),torch.zeros_like(mean))
zero=peak_diffusion043(mean,0.);assert torch.equal(mean+zero,mean)
permutation=torch.arange(31,-1,-1)
assert torch.equal(peak_diffusion043(mean[permutation],.5),delta[permutation])
assert torch.equal(torch.cat([peak_diffusion043(mean[i:i+1],.5) for i in range(32)]),delta)
assert torch.allclose(peak_diffusion043(3*mean,.5),3*delta,atol=1e-16,rtol=1e-13)
noise=.0002*torch.cos(t/7).expand(32,-1)
a=observe_generated_carrier(positive+noise,ste=False,q_floor=.001,return_indices=True)
b=observe_generated_carrier(positive+peak_diffusion043(positive,0.)+noise,ste=False,q_floor=.001,return_indices=True)
assert all(torch.equal(a[k],b[k]) for k in ('window_V','baseline100','peak_index'))
assert all(torch.equal(a['flags'][k],b['flags'][k]) for k in a['flags'])
write_json(args.out,dict(status='PASS',manufactured_rows=32,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
 full_integral_residual_Vns=full_error,canonical_Q_residual_Vns=canonical_error,peak_support_exact=True,
 minimum_center_coefficient=float(center.min()),signed_input_neighbor_range_preserved=True,
 manufactured_positive_peak_reduced=True,constant_input_exact_invariant=True,zero_strength_exact_old_observation=True,
 scaling_and_row_permutation_and_batching_verified=True,scientific_acceptance=False))
print('response043 manufactured peak diffusion checks: PASS')
