"""Manufactured intrinsic-mean crossing and actual-edge-map measurements; no data or CP."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from run_response051 import intrinsic_mean051,mapped_positions051,CROSSINGS051

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
device=torch.device(args.device);centers=np.arange(120,133);x=np.arange(384,dtype=np.float64)
gaussians=np.exp(-.5*((x[None,:]-centers[:,None])/6.)**2)
triangles=np.maximum(0.,np.minimum(1+(x[None,:]-centers[:,None])/8.,1-(x[None,:]-centers[:,None])/20.))
values=np.r_[gaussians,triangles,np.ones((1,384)),-np.ones((1,384))]
mean=torch.as_tensor(values,dtype=torch.float64,device=device);v=intrinsic_mean051(mean)
assert np.array_equal(v['peak_index'][:26],np.r_[centers,centers])
assert np.max(np.abs(v['features']['FWHM_ns'][:13]-12*np.sqrt(2*np.log(2))))<.03
assert np.max(np.abs(v['features']['FWHM_ns'][13:26]-14.))<1e-12
assert np.max(np.abs(v['features']['rise10_90_ns'][13:26]-6.4))<1e-12
assert np.max(np.abs(v['crossings']['left_crossing50_ns'][13:26]-(centers-4.)))<1e-12
assert np.max(np.abs(v['crossings']['right_crossing50_ns'][13:26]-(centers+10.)))<1e-12
assert np.allclose(v['features']['FWHM_ns'],
    v['crossings']['right_crossing50_ns']-v['crossings']['left_crossing50_ns'],rtol=0,atol=1e-12,equal_nan=True)
assert v['flags']['FWHM_undefined'][-2:].all() and v['flags']['rise_undefined'][-2:].all()
assert v['flags']['FWHM_prepeak_censored'][-2] and v['flags']['FWHM_postpeak_censored'][-2]
assert v['flags']['temporal_nonpositive_peak'][-1]
edges=np.arange(110,160,dtype=np.float64)-.5
points=np.array([[np.nan,108.,109.5,116.25,126.,146.75,158.5,160.]])
identity=mapped_positions051(points,edges[None,:])
assert np.array_equal(identity,points,equal_nan=True)
affine=mapped_positions051(points,(1.1*(edges-126.)+126.)[None,:])
inside=(points>=109.5)&(points<=158.5);expected=points.copy();expected[inside]=1.1*(points[inside]-126.)+126.
assert np.allclose(affine,expected,rtol=0,atol=1e-12,equal_nan=True)
permutation=np.arange(len(values))[::-1].copy();permuted=intrinsic_mean051(mean[torch.as_tensor(permutation.copy(),device=device)])
for family in ('features','flags','crossings'):
 for key,original in v[family].items():assert np.array_equal(permuted[family][key],original[permutation],equal_nan=True)
parts=[intrinsic_mean051(mean[:9]),intrinsic_mean051(mean[9:])]
for family in ('features','flags','crossings'):
 for key,original in v[family].items():assert np.array_equal(np.concatenate([a[family][key] for a in parts]),original,equal_nan=True)
args.out.parent.mkdir(parents=True,exist_ok=True)
result=dict(status='PASS',device=str(device),manufactured_rows=len(values),moving_gaussian_rows=13,
 moving_asymmetric_triangles=13,flat_and_nonpositive_rows=2,peak_selection_exact=True,
 absolute_crossings_and_original_morphology_identity=True,Gaussian_width_error_max_ns=float(np.max(np.abs(v['features']['FWHM_ns'][:13]-12*np.sqrt(2*np.log(2))))),
 no_ADC_or_random_or_original_data=True,flat_and_nonpositive_undefined_preserved=True,
 actual_piecewise_edge_map_identity_and_linear_cases=True,outside_support_identity_and_NaN_preserved=True,
 permutation_and_batch_exact=True,checkpoint_loads=0,TRAIN_decodes=0,AD_queries=0)
args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
