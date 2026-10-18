"""053 engineering checks on manufactured unequal populations; no data/CP."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from n01.losses008_v2 import create_evaluation_context
from run_response036 import NativeMetricLoss036, WAVE_WEIGHTS, _wave_views
from run_response053 import wave_views053,mapped_positions053,intrinsic_mean053

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);device=torch.device(args.device);torch.manual_seed(5301)
x=torch.arange(256,device=device,dtype=torch.float64)
fake=torch.stack([(.05+.003*i)*torch.exp(-.5*((x-16)/(3.+.13*i))**2) for i in range(11)])
real=torch.stack([(.055+.002*i)*torch.exp(-.5*((x-16)/(3.2+.11*i))**2) for i in range(13)])
fake+=torch.randn_like(fake)*.0003;real+=torch.randn_like(real)*.0003
context=create_evaluation_context(202610185301)
loss=SimpleNamespace(peak_floor=.001,column_chunk=16,_z=lambda name,x:(x-.003)/.02)
a=wave_views053(fake,real,loss,context)
b=NativeMetricLoss036.wave_components(loss,fake,real,None,context)
errors={k:abs(a['aggregate_parts'][k]-float(v)) for k,v in b.items()};assert max(errors.values())<1e-12
assert abs(a['aggregate_wave']-sum(WAVE_WEIGHTS[k]*float(v) for k,v in b.items()))<1e-12
zero=wave_views053(fake,fake,loss,context);assert zero['aggregate_wave']==0
permuted=wave_views053(fake.flip(0),real.flip(0),loss,context);assert abs(permuted['aggregate_wave']-a['aggregate_wave'])<1e-12
f=loss._z('voltage',_wave_views(fake,loss.peak_floor)['voltage']).cpu().numpy();rr=loss._z('voltage',_wave_views(real,loss.peak_floor)['voltage']).cpu().numpy()
independent=[float(np.mean([wasserstein_distance(f[:,i],rr[:,i]) for i in range(start,start+16)])) for start in range(0,256,16)]
assert np.max(np.abs(np.array(independent)-a['coordinate_by_16sample_block']['voltage']))<1e-12
knots=np.array([[109.5,120.,126.,138.,158.5]]);mapped=np.array([[109.5,118.,126.,141.,158.5]])
points=np.array([[np.nan,108.,109.5,114.75,120.,123.,126.,132.,138.,148.25,158.5,160.]])
expected=np.array([[np.nan,108.,109.5,113.75,118.,122.,126.,133.5,141.,149.75,158.5,160.]])
actual=mapped_positions053(points,knots,mapped);assert np.allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=True)
assert np.array_equal(mapped_positions053(points,knots,knots),points,equal_nan=True)
t=torch.arange(384,device=device,dtype=torch.float64);mean=torch.maximum(torch.zeros_like(t),torch.minimum(1+(t-126)/12,1-(t-126)/24))[None,:]
intrinsic=intrinsic_mean053(mean);assert intrinsic['peak_index'][0]==126 and abs(intrinsic['features']['FWHM_ns'][0]-18)<1e-12
assert abs(intrinsic['crossings']['left_crossing50_ns'][0]-120)<1e-12 and abs(intrinsic['crossings']['right_crossing50_ns'][0]-138)<1e-12
result=dict(status='PASS',device=str(device),manufactured_fake_rows=11,manufactured_real_rows=13,checkpoint_loads=0,TRAIN_decodes=0,DEV_decodes=0,AD_queries=0,
 aggregate_matches_original036_max_error=max(errors.values()),aggregate_errors=errors,independent_scipy_coordinate_blocks_max_error=float(np.max(np.abs(np.array(independent)-a['coordinate_by_16sample_block']['voltage']))),
 identical_population_zero=True,row_permutation_invariance=True,five_knot_piecewise_mapping_known_exact=True,undefined_and_outside_support_preserved=True,triangle_intrinsic_width_18ns=True)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
