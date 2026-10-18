"""Paired frozen-response diagnostics, with no fitted or changed response."""
import numpy as np
import torch
import run_response039 as r
from run_response044 import readouts044


@torch.no_grad()
def native_decomposition071(fixed_mean,mean,random,random_delta,baseline_signal,mean_only_signal,signal,noise,obs,q_floor):
    before=baseline_signal+noise;after=signal+noise
    baseline=r.observe_generated_carrier(before,ste=False,q_floor=q_floor,return_carrier=True)
    assert torch.equal(baseline['baseline100'],obs['baseline100'])
    old_adc=baseline['carrier_adc'];new_adc=obs['carrier_adc'];old_peak=baseline['peak_index'];new_peak=obs['peak_index']
    ref=fixed_mean[:,112:144].argmax(1)+112
    mean_delta=mean-fixed_mean
    reassociation=(after-before)-mean_delta-random_delta
    adc_delta=(new_adc-after)-(old_adc-before)
    f,flags=readouts044(baseline['window_V'].cpu().numpy())
    rows=dict(parent_row=np.arange(len(mean)),reference_peak=ref.cpu().numpy(),BASE054_native_peak_index=old_peak.cpu().numpy(),candidate_native_peak_index=new_peak.cpu().numpy(),
        BASE054_signal_peak_index=(baseline_signal[:,112:144].argmax(1)+112).cpu().numpy(),mean_only_signal_peak_index=(mean_only_signal[:,112:144].argmax(1)+112).cpu().numpy(),
        coupled_signal_peak_index=(signal[:,112:144].argmax(1)+112).cpu().numpy(),BASE054_preADC_peak_index=(before[:,112:144].argmax(1)+112).cpu().numpy(),candidate_preADC_peak_index=(after[:,112:144].argmax(1)+112).cpu().numpy(),
        **{'BASE054_'+k:v for k,v in f.items()},**{'BASE054_'+k+'_flag':v for k,v in flags.items()})
    for name,lo,length in [('tail',30,210),('Q',-16,256)]:
        old_index=old_peak[:,None]+lo+torch.arange(length,device=mean.device)[None,:]
        new_index=new_peak[:,None]+lo+torch.arange(length,device=mean.device)[None,:]
        ref_index=ref[:,None]+lo+torch.arange(length,device=mean.device)[None,:]
        native_delta=new_adc.gather(1,new_index).sum(1)-old_adc.gather(1,old_index).sum(1)
        parts={key:value.gather(1,new_index).sum(1) for key,value in [('mean',mean_delta),('random',random_delta),('ADC',adc_delta),('reassociation',reassociation)]}
        shift=old_adc.gather(1,new_index).sum(1)-old_adc.gather(1,old_index).sum(1)
        closure=native_delta-shift-sum(parts.values())
        rows[name+'_native_delta_Vns']=native_delta.cpu().numpy();rows[name+'_BASE054_window_shift_Vns']=shift.cpu().numpy();rows[name+'_closure_Vns']=closure.cpu().numpy()
        for key,value in parts.items():rows[name+'_same_new_window_'+key+'_Vns']=value.cpu().numpy()
        if name=='tail':
            for key,value in [('mean',mean_delta),('random',random_delta),('ADC',adc_delta)]:rows['tail_reference_'+key+'_Vns']=value.gather(1,ref_index).sum(1).cpu().numpy()
    for name,value in [('BASE054_mean',fixed_mean),('mean_change',mean_delta),('original_random',random),('random_change',random_delta),('electronic_noise',noise),('candidate_ADC_error',new_adc-after),('candidate_reassociation',after-mean-random-random_delta-noise),('BASE054_ADC',old_adc),('candidate_ADC',new_adc)]:
        rows['peak_advantage_'+name+'_V']=(value.gather(1,new_peak[:,None])-value.gather(1,old_peak[:,None])).squeeze(1).cpu().numpy()
    return rows,baseline
