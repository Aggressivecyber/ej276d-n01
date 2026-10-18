"""Manufactured positivity locations, ordered fixed-anchor stages and ADC replay."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response063 import shape_residual063
from run_response064 import positivity_audit064,readouts064,STAGES064
from run_response041 import fixed_anchor_stages,ordered_contrasts

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
knots=np.geomspace(.5,12.,8);s=np.ones((8,256));t=s.copy();s[:,15:17]=[.5,.5];t[:,15:17]=[.25,.75]
template=dict(source_Q_centers_Vns=knots.tolist(),target_Q_centers_Vns=knots.tolist(),source_unit_Q_profiles=s.tolist(),target_unit_Q_profiles=t.tolist())
mean=torch.zeros((2,384),dtype=torch.float64,device=device);mean[:,125:127]=torch.tensor([[.08,.10],[.01,.17]],dtype=torch.float64,device=device);q=r.canonical_charge008(mean)
delta,rows=shape_residual063(mean,q,True,template);direction,lim=positivity_audit064(mean,delta,rows,template)
assert torch.equal(lim['limiting_absolute_index'],torch.full((2,),125,device=device)) and torch.equal(lim['limiting_relative_index'],torch.full((2,),15,device=device))
assert np.max(np.abs(lim['raw_positivity_limit'].cpu().numpy()-np.array([.08/.045,.01/.045])))<1e-14
assert np.max(np.abs(direction[:,15:17].cpu().numpy()-np.array([[-.045,.045],[-.045,.045]])))<1e-14
identity=dict(template,target_unit_Q_profiles=s.tolist());di,ri=shape_residual063(mean,q,True,identity);_,li=positivity_audit064(mean,di,ri,identity)
assert torch.isinf(li['raw_positivity_limit']).all() and (li['limiting_absolute_index']==-1).all() and torch.isnan(li['limiting_old_amplitude_V']).all()

axis=torch.arange(384,dtype=torch.float64,device=device)
old=.12*torch.clamp(1.-(axis-126.).abs()/8.,min=0.)[None,:].repeat(2,1)
new=.12*torch.clamp(1.-(axis-128.).abs()/8.,min=0.)[None,:].repeat(2,1)
random=.00012*torch.sin(axis/4.)[None,:].repeat(2,1);electronic=.00021*torch.cos(axis/7.)[None,:].repeat(2,1)+.001
oldobs=r.observe_generated_carrier(old+random+electronic,ste=False,q_floor=.001,return_carrier=True)
obs=r.observe_generated_carrier(new+random+electronic,ste=False,q_floor=.001,return_carrier=True)
anchor=oldobs['crop_indices'];stages=fixed_anchor_stages(new,new+random,new+random+electronic,obs,anchor)
assert torch.equal(oldobs['peak_index'],torch.full((2,),126,device=device)) and torch.equal(obs['peak_index'],torch.full((2,),128,device=device))
carriers=[x.cpu().numpy() for x in (new,new+random,new+random+electronic,obs['carrier_adc'])]
independent=[];indices=anchor.cpu().numpy();native=obs['crop_indices'].cpu().numpy()
for value in carriers:independent.append(np.take_along_axis(value-value[:,:100].mean(1,keepdims=True),indices,axis=1))
independent.append(np.take_along_axis(carriers[-1]-carriers[-1][:,:100].mean(1,keepdims=True),native,axis=1))
stage_error=float(np.max(np.abs(np.array(independent)-np.array([x.cpu().numpy() for x in stages]))));assert stage_error<1e-15
codes=np.round((np.clip(carriers[2],-.5,.5)+.5)*16383);adc=codes/16383-.5
assert np.array_equal(codes,np.round((carriers[3]+.5)*16383)) and np.max(np.abs(adc-carriers[3]))<1e-15
baseline_stages=fixed_anchor_stages(old,old+random,old+random+electronic,oldobs,anchor);assert torch.equal(baseline_stages[3],baseline_stages[4])
f0,flags0=readouts064(stages[0].cpu().numpy(),.001);fn,flagsn=readouts064(stages[-1].cpu().numpy(),.001)
assert flags0['selected_peak_below_global_crop_max'].all() and not flagsn['selected_peak_below_global_crop_max'].any()
zero,zflag=readouts064(np.zeros((2,256)),.001);assert np.isnan(zero['FWHM_ns']).all() and np.isnan(zero['PSD_raw']).all() and zflag['PSD_undefined'].all()
known=ordered_contrasts(np.array([12.,12.,17.,15.,18.]),np.full(5,10.));assert np.array_equal(known['ordered_contributions'],np.array([2.,0.,5.,-2.,3.])) and known['closure_max_error']==0.
features=[readouts064(x.cpu().numpy(),.001)[0] for x in stages];base=[readouts064(x.cpu().numpy(),.001)[0] for x in baseline_stages]
closure=max(ordered_contrasts(np.array([x[k] for x in features]),np.array([x[k] for x in base]))['closure_max_error'] for k in ('Q_Vns','front_Vns','tail_Vns'))
assert closure<1e-12
record=dict(status='PASS',device=str(device),checkpoint_loads=0,TRAIN_decodes=0,DEV_decodes=0,AD_queries=0,known_positivity_limiting_position125_and_scale=True,
    unconstrained_infinite_bound_explicit=True,known_native_peak_shift_2_samples=True,fixed_stage_reference_not_reselected=True,stage_order=STAGES064,
    independent_numpy_stage_error_V=stage_error,ADC_integer_codes_exact=True,BASE_fixed_ADC_equals_native=True,index16_below_global_flag_retained=True,
    undefined_width_and_PSD_retained=True,known_ordered_contributions_exact=True,row_contribution_closure=closure,scientific_acceptance=False)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
