"""Manufactured attribution semantics; zero data/CP loads."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from n01.data import write_json
from n01.losses008_v2 import create_evaluation_context, WAVE_WEIGHTS
from n01.observation008 import observe_generated_carrier
from run_response036 import NativeMetricLoss036
from run_response041 import wave_decomposition, fixed_anchor_stages, ordered_contrasts, STAGES

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(2)
t=torch.arange(384,dtype=torch.float64)
amp=torch.linspace(.03,.1,32)[:,None]; width=torch.linspace(3.,6.,32)[:,None]
mean0=amp*torch.exp(-.5*((t-126)/width).square());mean1=amp*torch.exp(-.5*((t-128)/width).square())
mean0[:,:100]=0;mean1[:,:100]=0
random=.0003*torch.sin(t/3).expand(32,-1).clone();random[:,:100]=0
noise=.0002*torch.cos(t/7).expand(32,-1)
signal0,signal1=mean0+random,mean1+random
obs0=observe_generated_carrier(signal0+noise,ste=False,q_floor=.001,return_carrier=True)
obs1=observe_generated_carrier(signal1+noise,ste=False,q_floor=.001,return_carrier=True)
s0=fixed_anchor_stages(mean0,signal0,signal0+noise,obs0,obs0['crop_indices'])
s1=fixed_anchor_stages(mean1,signal1,signal1+noise,obs1,obs0['crop_indices'])
assert len(s0)==len(s1)==len(STAGES)==5 and all(x.shape==(32,256) for x in s0+s1)
assert torch.equal(s0[-2],s0[-1])
assert not torch.equal(s1[-2],s1[-1]) and bool((obs0['peak_index']!=obs1['peak_index']).any())
a0=np.array([(-x).clamp_min(0).sum(1).numpy() for x in s0]);a1=np.array([(-x).clamp_min(0).sum(1).numpy() for x in s1])
contrast=ordered_contrasts(a1,a0)
assert contrast['closure_max_error']<1e-12
null=ordered_contrasts(a0,a0);assert np.array_equal(null['ordered_contributions'],np.zeros_like(a0))
stub=SimpleNamespace(peak_floor=torch.tensor(.001),column_chunk=16,_z=lambda name,x:x)
context=create_evaluation_context(41001)
fake,real=obs1['window_V'][:23],obs0['window_V'][:29]
native=NativeMetricLoss036.wave_components(stub,fake,real,None,context)
expected=float(sum(WAVE_WEIGHTS[key]*value for key,value in native.items()))
pieces=wave_decomposition(fake,real,stub,context)
error=abs(pieces['reconstructed_wave']-expected)
assert error<1e-12 and len(pieces['pieces'])==16
assert abs(sum(v['weight'] for v in pieces['pieces'].values())-1)<1e-12
write_json(args.out,dict(status='PASS',manufactured_rows=32,checkpoint_loads=0,dataset_loads=0,AD_queries=0,
    old_anchor_equals_native_for_M0=True,peak_reselection_control_is_distinct=True,
    ordered_area_closure_max_error=contrast['closure_max_error'],null_effect_exact_zero=True,
    wave_16_piece_reconstruction_error=error,unequal_population_counts=[23,29],all_rows_retained=True,
    scientific_acceptance=False))
print('response041 manufactured attribution checks: PASS')
