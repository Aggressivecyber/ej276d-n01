"""Manufactured NumPy versus Torch audit of paired readout attribution."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from response071_diagnostics import native_decomposition071

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
x=np.arange(384);N=32;center=120+np.arange(N)%7
m=np.array([(.04+.004*i)*np.exp(-.5*((x-p)/5.)**2)+.015*np.exp(-np.maximum(x-p,0)/30.)*(x>=p) for i,p in enumerate(center)]);m[:,:100]=0.
dm=np.zeros_like(m);dm[:,110:176]=.003*np.sin(np.arange(66)[None,:]/5.+np.arange(N)[:,None])
rr=.002*np.sin(x[None,:]/3.+np.arange(N)[:,None]);dr=np.zeros_like(m);dr[:,110:176]=.003*np.cos(np.arange(66)[None,:]/4.+np.arange(N)[:,None])
noise=.0007*np.cos(x[None,:]/5.+np.arange(N)[:,None])
# A genuine ADC tie, and a separate clipped waveform, exercise native ordering and quantization attribution.
m[0]=0.;m[0,122:124]=.2;dm[0]=0.;rr[0]=0.;dr[0]=0.;noise[0]=0.;dm[0,123]=.000001
m[1]*=16.
mt,dt,rt,drt,nt=[torch.as_tensor(z,device=device,dtype=torch.float64) for z in (m,dm,rr,dr,noise)]
mean=mt+dt;base=mt+rt;meanonly=mean+rt;signal=meanonly+drt
arguments=(mt,mean,rt,drt,base,meanonly,signal,nt);saved_arguments=[value.clone() for value in arguments]
obs=r.observe_generated_carrier(signal+nt,ste=False,q_floor=.001,return_carrier=True)
rows,bobs=native_decomposition071(mt,mean,rt,drt,base,meanonly,signal,nt,obs,.001)
assert len(rows)==51 and sum(a.dtype==bool for a in rows.values())==9
max_error=0.;native_error=0.;row_count=0
for i in range(N):
 old=base[i].cpu().numpy()+noise[i];new=signal[i].cpu().numpy()+noise[i];mc=mean[i].cpu().numpy()-m[i];reassoc=(new-old)-mc-dr[i]
 oldadc=np.round((np.clip(old,-.5,.5)+.5)*16383)/16383-.5;newadc=np.round((np.clip(new,-.5,.5)+.5)*16383)/16383-.5
 op=int(np.argmax(oldadc[112:144])+112);npk=int(np.argmax(newadc[112:144])+112);ref=int(np.argmax(m[i,112:144])+112)
 for key,val in [('reference_peak',ref),('BASE054_native_peak_index',op),('candidate_native_peak_index',npk),('BASE054_signal_peak_index',int(np.argmax(base[i,112:144].cpu().numpy())+112)),('mean_only_signal_peak_index',int(np.argmax(meanonly[i,112:144].cpu().numpy())+112)),('coupled_signal_peak_index',int(np.argmax(signal[i,112:144].cpu().numpy())+112)),('BASE054_preADC_peak_index',int(np.argmax(old[112:144])+112)),('candidate_preADC_peak_index',int(np.argmax(new[112:144])+112))]:assert rows[key][i]==val
 for name,lo,length in [('tail',30,210),('Q',-16,256)]:
  oi=slice(op+lo,op+lo+length);ni=slice(npk+lo,npk+lo+length);ri=slice(ref+lo,ref+lo+length)
  expected={name+'_native_delta_Vns':math.fsum(newadc[ni])-math.fsum(oldadc[oi]),name+'_BASE054_window_shift_Vns':math.fsum(oldadc[ni])-math.fsum(oldadc[oi])}
  for part,val in [('mean',mc),('random',dr[i]),('ADC',(newadc-new)-(oldadc-old)),('reassociation',reassoc)]:expected[name+'_same_new_window_'+part+'_Vns']=math.fsum(val[ni])
  expected[name+'_closure_Vns']=0.
  if name=='tail':
   for part,val in [('mean',mc),('random',dr[i]),('ADC',(newadc-new)-(oldadc-old))]:expected['tail_reference_'+part+'_Vns']=math.fsum(val[ri])
  for key,val in expected.items():max_error=max(max_error,abs(rows[key][i]-val))
 for name,val in [('BASE054_mean',m[i]),('mean_change',mc),('original_random',rr[i]),('random_change',dr[i]),('electronic_noise',noise[i]),('candidate_ADC_error',newadc-new),('candidate_reassociation',new-mean[i].cpu().numpy()-rr[i]-dr[i]-noise[i]),('BASE054_ADC',oldadc),('candidate_ADC',newadc)]:max_error=max(max_error,abs(rows['peak_advantage_'+name+'_V'][i]-(val[npk]-val[op])))
 window=oldadc[op-16:op+240]-oldadc[:100].mean();native_error=max(native_error,abs(rows['BASE054_Q_Vns'][i]-window.sum()),abs(rows['BASE054_tail_Vns'][i]-window[46:].sum()),abs(rows['BASE054_front_Vns'][i]-window[:46].sum()))
 row_count+=1
assert max_error<1e-12 and native_error<1e-12
assert rows['candidate_preADC_peak_index'][0]==123 and rows['candidate_native_peak_index'][0]==122
assert bool(obs['flags']['ADC_rail_hit'][1]) and bool(bobs['flags']['ADC_rail_hit'][1])
identity_obs=r.observe_generated_carrier(base+nt,ste=False,q_floor=.001,return_carrier=True)
identity,_=native_decomposition071(mt,mt,rt,torch.zeros_like(rt),base,base,base,nt,identity_obs,.001)
for key in ('Q_native_delta_Vns','tail_native_delta_Vns','Q_BASE054_window_shift_Vns','tail_BASE054_window_shift_Vns','peak_advantage_candidate_ADC_V'):assert np.array_equal(identity[key],np.zeros(N)),key
reverse=torch.arange(N-1,-1,-1,device=device);robs=r.observe_generated_carrier(signal[reverse]+nt[reverse],ste=False,q_floor=.001,return_carrier=True)
permuted,_=native_decomposition071(mt[reverse],mean[reverse],rt[reverse],drt[reverse],base[reverse],meanonly[reverse],signal[reverse],nt[reverse],robs,.001)
for key in rows:
 if key!='parent_row':assert np.array_equal(permuted[key],rows[key][::-1],equal_nan=True),key
assert all(torch.equal(value,before) for value,before in zip(arguments,saved_arguments))
record=dict(status='PASS',device=str(device),manufactured_rows=row_count,numeric_columns=sum(z.dtype!=bool for z in rows.values()),boolean_columns=sum(z.dtype==bool for z in rows.values()),independent_decomposition_max_error=max_error,native_integral_max_error=native_error,identity_exact=True,permutation_exact=True,ADC_tie_first_argmax_verified=True,ADC_saturation_preserved=True,original_arrays_unchanged=True,CP_loads=0,TRAIN_decodes=0,AD_queries=0,optimizer_steps=0)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
