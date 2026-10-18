"""A local conserved peak change can move a tail gate while the carrier tail is identical."""
import argparse
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.observation008 import observe_generated_carrier
from run_response041 import fixed_anchor_stages, ordered_contrasts
from run_response044 import readouts044, STAGES044

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(2)
t=torch.arange(384,dtype=torch.float64);amp=torch.linspace(.03,.1,32)[:,None]
mean=amp*torch.exp(-.5*((t-126)/6).square())+.005*torch.exp(-(t-126).clamp_min(0)/30)*(t>=126)
mean[:,:100]=0
delta=torch.zeros_like(mean);delta[:,126]=-.15*amp[:,0];delta[:,127]=.15*amp[:,0]
changed=mean+delta
random=.0003*torch.sin(t/3).expand(32,-1).clone();random[:,:100]=0
noise=.0002*torch.cos(t/7).expand(32,-1)
signal0,signal1=mean+random,changed+random
obs0=observe_generated_carrier(signal0+noise,ste=False,q_floor=.001,return_carrier=True)
obs1=observe_generated_carrier(signal1+noise,ste=False,q_floor=.001,return_carrier=True)
assert torch.equal(mean[:,145:],changed[:,145:]) and float(delta.sum(1).abs().max())==0
assert bool((obs0['peak_index']!=obs1['peak_index']).all())
a=fixed_anchor_stages(mean,signal0,signal0+noise,obs0,obs0['crop_indices'])
b=fixed_anchor_stages(changed,signal1,signal1+noise,obs1,obs0['crop_indices'])
fa=[readouts044(x.numpy())[0] for x in a];fb=[readouts044(x.numpy())[0] for x in b]
tail0=np.array([x['tail_Vns'] for x in fa]);tail1=np.array([x['tail_Vns'] for x in fb])
assert len(a)==len(b)==len(STAGES044)==5
assert np.array_equal(tail0[:4],tail1[:4])
assert np.all(tail0[-1]!=tail1[-1])
contrast=ordered_contrasts(tail1,tail0)
assert contrast['closure_max_error']<1e-12
assert np.array_equal(contrast['ordered_contributions'][:4],np.zeros((4,32)))
assert np.allclose(contrast['ordered_contributions'][-1],tail1[-1]-tail0[-1],atol=1e-15,rtol=0)
permutation=np.arange(31,-1,-1)
features,flags=readouts044(obs1['window_V'].numpy()[permutation])
original,original_flags=readouts044(obs1['window_V'].numpy())
assert all(np.array_equal(features[k],v[permutation],equal_nan=True) for k,v in original.items())
assert all(np.array_equal(flags[k],v[permutation]) for k,v in original_flags.items())
zeros,zero_flags=readouts044(np.zeros((32,256)))
assert len(zeros['FWHM_ns'])==32 and np.isnan(zeros['FWHM_ns']).all() and zero_flags['FWHM_undefined'].all()
write_json(args.out,dict(status='PASS',manufactured_rows=32,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    conserved_local_change_carrier_tail_identical=True,all_fixed_anchor_tail_stages_exactly_equal=True,
    native_tail_change_entirely_from_reselection=True,tail_closure_max_error=contrast['closure_max_error'],
    undefined_width_rows_preserved=True,row_permutation_preserved=True,scientific_acceptance=False))
print('response044 local-tail versus selected-gate semantic checks: PASS')
