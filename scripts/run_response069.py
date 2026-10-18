"""One conditional crossing-level voltage warp with area-preserving random coupling."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import run_response039 as r
from run_response042 import local_restitution042
from run_response052 import conditional_sides052,evaluate052
from run_response053 import intrinsic_mean053,CROSSINGS053
from run_response060 import save_profiles060
from run_response044 import readouts044
from wp1_evaluation import summary,output_q_bins

ROOT=Path(__file__).resolve().parents[1]


from run_response068 import voltage_warp068

def level_warp069(mean,random,input_q,template):
    """Conditional50/90% crossing queries, one area scale, linear residual projection."""
    if not bool((input_q>0.).all()):raise ValueError('069 nonpositive input-Q domain')
    peak=mean[:,112:144].argmax(1)+112;logq=input_q.log()
    sk=torch.as_tensor(template['source_Q_centers_Vns'],device=mean.device,dtype=torch.float64).log()
    tk=torch.as_tensor(template['target_Q_centers_Vns'],device=mean.device,dtype=torch.float64).log()
    sp=torch.as_tensor(template['source_unit_Q_profiles'],device=mean.device,dtype=torch.float64)
    tp=torch.as_tensor(template['target_unit_Q_profiles'],device=mean.device,dtype=torch.float64)
    sl=logq.clamp(sk[0],sk[-1]);tl=logq.clamp(tk[0],tk[-1])
    si=(torch.searchsorted(sk,sl,right=True)-1).clamp(0,6);ti=(torch.searchsorted(tk,tl,right=True)-1).clamp(0,6)
    sw=(sl-sk[si])/(sk[si+1]-sk[si]);tw=(tl-tk[ti])/(tk[ti+1]-tk[ti])
    source=(1.-sw[:,None])*sp[si]+sw[:,None]*sp[si+1];target=(1.-tw[:,None])*tp[ti]+tw[:,None]*tp[ti+1]
    positions=torch.arange(66,device=mean.device)[None,:];locations=positions+110-peak[:,None]+16
    a=(peak-124).clamp_min(0)[:,None];p=(peak-110)[:,None];knots=[]
    for profile in (source,target):
        left=[];right=[]
        for fraction in (.5,.9):
            level=fraction*profile[:,16]
            li=torch.where(profile[:,:16]<level[:,None],torch.arange(16,device=mean.device)[None,:],-1).max(1).values
            below=profile[:,17:]<level[:,None]
            if not bool((li>=0).all() and below.any(1).all()):raise ValueError('069 conditional profile lacks nearest-peak crossing')
            ri=below.long().argmax(1)+17
            lv=profile.gather(1,li[:,None]).squeeze(1);ln=profile.gather(1,(li+1)[:,None]).squeeze(1)
            rv=profile.gather(1,ri[:,None]).squeeze(1);rp=profile.gather(1,(ri-1)[:,None]).squeeze(1)
            left.append(li+(level-lv)/(ln-lv));right.append(ri-1+(level-rp)/(rv-rp))
        center=peak.to(mean.dtype)
        k=torch.stack((a.squeeze(1)+110,left[0]+center-16,left[1]+center-16,center,right[1]+center-16,right[0]+center-16,torch.full_like(center,175.)),1)
        if not bool(torch.isfinite(k).all() and (k.diff(dim=1)>0.).all()):raise ValueError('069 crossing knots outside ordered support domain')
        knots.append(k)
    source_knots,target_knots=knots
    absolute=(positions+110).expand(len(mean),-1).to(mean.dtype).contiguous()
    segment=(torch.searchsorted(target_knots.contiguous(),absolute,right=True)-1).clamp(0,5)
    fraction=(absolute-target_knots.gather(1,segment))/(target_knots.gather(1,segment+1)-target_knots.gather(1,segment))
    inverse=source_knots.gather(1,segment)+fraction*(source_knots.gather(1,segment+1)-source_knots.gather(1,segment))
    query=torch.where((positions<=a)|(positions==p)|(positions==65),positions.to(mean.dtype),inverse-110)
    lo=query.floor().long().clamp(0,64);weight=query-lo
    local=mean[:,110:176];mask=(locations>=2)&(local>0.);positive=torch.where(mask,local,torch.zeros_like(local));area=positive.sum(1)
    interpolated=torch.where(mask,(1.-weight)*positive.gather(1,lo)+weight*positive.gather(1,lo+1),torch.zeros_like(local))
    scale=area/interpolated.sum(1);mapped=scale[:,None]*interpolated
    delta=torch.zeros_like(mean);delta[:,110:176]=mapped-positive;adapted=mean+delta
    active=torch.where(mask,random[:,110:176],torch.zeros_like(local))
    provisional=scale[:,None]*torch.where(mask,(1.-weight)*active.gather(1,lo)+weight*active.gather(1,lo+1),torch.zeros_like(local))
    leakage=provisional.sum(1)-active.sum(1);unit=mapped/area[:,None]
    random_delta=torch.zeros_like(random);random_delta[:,110:176]=provisional-leakage[:,None]*unit-active
    rows=dict(input_Q_Vns=input_q,source_lookup_logQ=sl,target_lookup_logQ=tl,source_interval=si,target_interval=ti,source_right_weight=sw,target_right_weight=tw,
        source_clamped_low=logq<sk[0],source_clamped_high=logq>sk[-1],target_clamped_low=logq<tk[0],target_clamped_high=logq>tk[-1],
        positive_area_Vns=area,active_positive_samples=mask.sum(1),amplitude_scale=scale,interpolated_positive_area_Vns=interpolated.sum(1),
        query_maximum_displacement_samples=(query-positions).abs().max(1).values,query_minimum_spacing_samples=query.diff(dim=1).min(1).values,
        peak_query_error_samples=(query.gather(1,p)-p).abs().squeeze(1),left_anchor_index=a.squeeze(1)+110,
        minimum_mapped_active_V=torch.where(mask,mapped,torch.full_like(local,float('inf'))).min(1).values,
        old_mean_peak_index=peak,new_mean_peak_index=adapted[:,112:144].argmax(1)+112,
        old_peak_voltage_V=mean.gather(1,peak[:,None]).squeeze(1),new_peak_voltage_V=adapted[:,112:144].max(1).values,
        full_integral_residual_Vns=delta.sum(1),canonical_Q_residual_Vns=r.canonical_charge008(delta),
        negative_area_change_Vns=(-adapted).clamp_min(0.).sum(1)-(-mean).clamp_min(0.).sum(1),
        transferred_absolute_area_Vns=delta.abs().sum(1),maximum_absolute_delta_V=delta.abs().max(1).values,
        boundary_delta_110_V=delta[:,110],boundary_delta_175_V=delta[:,175],random_provisional_integral_leakage_Vns=leakage)
    for prefix,k in [('source',source_knots),('target',target_knots)]:
        for name,index in [('left50',1),('left90',2),('right90',4),('right50',5)]:rows[prefix+'_'+name+'_absolute_sample']=k[:,index]
    return delta,random_delta,rows,query+110,source_knots,target_knots


@torch.no_grad()
def evaluate069(setup,index,bank,T,projection,real_joint,p,out,progress,enabled,label,template,base_by_h):
    model=setup['models'][index];device=setup['source'].device
    model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],device=device,dtype=model.mean_adapter[-1].bias.dtype))
    progress['mean_operator_calls']+=1;progress['mean_operator_rows']+=len(setup['source'])
    adapter=model._mean_response(setup['encoding'],enabled=True)
    delta,_,_=local_restitution042(adapter['delta_mean_pre_V'],p['base_restitution_fraction'])
    A0=setup['encoding']['original']['signal_pre'];q=r.canonical_charge008(A0)
    transport,_,_,_,_,_=conditional_sides052(A0+delta,q,1.,p['input_Q_reference_Vns'],p['side_coefficients_ns'])
    fixed_delta=delta+transport;fixed_mean=A0+fixed_delta
    random=bank['M0']-A0
    if enabled:
        correction,random_delta,rows,queries,source_knots,target_knots=level_warp069(fixed_mean,random,q,template)
        np.savez_compressed(out/(label+'_WARP_QUERIES.npz'),parent_row=np.arange(len(q)),inverse_query_absolute_sample=queries.cpu().numpy(),source_knots_absolute_sample=source_knots.cpu().numpy(),target_knots_absolute_sample=target_knots.cpu().numpy())
        progress['level_warp_records']+=1;progress['level_warp_rows']+=len(q)
        if float(rows['query_minimum_spacing_samples'].min())<0. or float(rows['peak_query_error_samples'].max())!=0.:raise RuntimeError('069 monotone/peak-anchor invariant failed')
    else:
        correction,random_delta,rows,_=voltage_warp068(fixed_mean,random,q,template)
        progress['voltage_warp_records']+=1;progress['voltage_warp_rows']+=len(q)
    delta=fixed_delta+correction;mean=fixed_mean+correction;rows['new_mean_peak_index']=mean[:,112:144].argmax(1)+112;rows['new_peak_voltage_V']=mean[:,112:144].max(1).values;base_signal=r._ExactZeroMeanAddition015.apply(bank['M0'],delta)
    signal=r._ExactZeroMeanAddition015.apply(base_signal,random_delta)
    random_rows=dict(full_integral_residual_Vns=random_delta.sum(1),canonical_Q_residual_Vns=r.canonical_charge008(random_delta),
        maximum_absolute_delta_V=random_delta.abs().max(1).values,old_random_L2_V=random.square().sum(1).sqrt(),new_random_L2_V=(random+random_delta).square().sum(1).sqrt(),
        old_signal_peak_index=base_signal[:,112:144].argmax(1)+112,new_signal_peak_index=signal[:,112:144].argmax(1)+112)
    if max(float(random_rows['full_integral_residual_Vns'].abs().max()),float(random_rows['canonical_Q_residual_Vns'].abs().max()))>1e-12:raise RuntimeError('069 random mass conservation failed')
    if not torch.equal(signal[:,:110],base_signal[:,:110]) or not torch.equal(signal[:,176:],base_signal[:,176:]):raise RuntimeError('069 random support changed')
    if not torch.isfinite(signal).all():raise RuntimeError('069 nonfinite coupled signal')
    np.savez_compressed(out/(label+'_COUPLING_ROWS.npz'),parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),**{k:v.cpu().numpy() for k,v in random_rows.items()})
    ix=torch.as_tensor(p['coupling_example_parent_rows'][str(index)],device=device)
    np.savez_compressed(out/(label+'_COUPLING_EXAMPLES.npz'),parent_row=ix.cpu().numpy(),input_Q_Vns=q[ix].cpu().numpy(),fixed_mean_pre_V=fixed_mean[ix].cpu().numpy(),adapted_mean_pre_V=mean[ix].cpu().numpy(),
        old_random_V=random[ix].cpu().numpy(),random_correction_V=random_delta[ix].cpu().numpy(),base_signal_V=base_signal[ix].cpu().numpy(),coupled_signal_V=signal[ix].cpu().numpy(),noise_fast_V=bank['noise']['fast'][ix].cpu().numpy())
    progress['random_operator_calls']+=1;progress['random_operator_rows']+=len(q);progress['coupling_records']+=1;progress['coupling_rows']+=len(q)
    progress['coupling_example_records']+=1;progress['coupling_example_rows']+=len(ix)
    geometry=dict(enabled=enabled,support=[110,176],support_exact=bool((correction[:,:110]==0.).all() and (correction[:,176:]==0.).all()),
        frozen_prefix_bitwise_unchanged=torch.equal(signal[:,:100],bank['M0'][:,:100]),mean_tail_from176_bitwise_unchanged=torch.equal(mean[:,176:],fixed_mean[:,176:]),
        nonpositive_mean_samples_unchanged=torch.equal(mean[fixed_mean<=0.],fixed_mean[fixed_mean<=0.]),
        mean_peak_changed_rows=int(((mean[:,112:144].argmax(1)+112)!=rows['old_mean_peak_index']).sum()),
        maximum_canonical_Q_residual_Vns=float((r.canonical_charge008(mean)-q).abs().max()),
        maximum_full_integral_residual_Vns=float(correction.sum(1).abs().max()),mean_negative_area_max_change_Vns=float(rows['negative_area_change_Vns'].abs().max()),
        maximum_absolute_correction_V=float(correction.abs().max()),mean_addition_reassociation_max_error_V=float((mean-(A0+delta)).abs().max()),boundary_delta_max_absolute_V=float(torch.maximum(correction[:,110].abs(),correction[:,175].abs()).max()),
        row_summaries={k:summary(v.cpu().numpy()) for k,v in rows.items()},random_row_summaries={k:summary(v.cpu().numpy()) for k,v in random_rows.items()},random_coupling_enabled=True,peak_anchored_voltage_warp_enabled=True,crossing_level_warp_enabled=enabled)
    if not all(geometry[k] for k in ('support_exact','frozen_prefix_bitwise_unchanged','mean_tail_from176_bitwise_unchanged','nonpositive_mean_samples_unchanged')):raise RuntimeError('069 exact support/prefix/tail/negative invariant failed')
    if geometry['mean_negative_area_max_change_Vns']!=0.:raise RuntimeError('069 negative area changed')
    if geometry['maximum_canonical_Q_residual_Vns']>1e-10 or geometry['maximum_full_integral_residual_Vns']>1e-12:raise RuntimeError('069 charge/area invariant failed')
    if not bool(torch.isfinite(mean).all()):raise RuntimeError('069 nonfinite mean')
    if float(rows['minimum_mapped_active_V'].min())<0.:raise RuntimeError('069 nonnegative mapped voltage invariant failed')
    np.savez_compressed(out/(label+'_GEOMETRY_ROWS.npz'),parent_row=np.arange(len(q)),**{k:v.cpu().numpy() for k,v in rows.items()})
    np.savez_compressed(out/(label+'_EXAMPLES.npz'),parent_row=np.arange(32),input_Q_Vns=q[:32].cpu().numpy(),fixed_mean_pre_V=fixed_mean[:32].cpu().numpy(),adapted_mean_pre_V=mean[:32].cpu().numpy(),correction_pre_V=correction[:32].cpu().numpy())
    progress['geometry_records']+=1;progress['geometry_rows']+=len(q)
    intrinsic=intrinsic_mean053(mean)
    np.savez_compressed(out/(label+'_INTRINSIC_MEAN_ROWS.npz'),parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),input_Q_bin=p['_input_bins'],
        intrinsic_peak_index=intrinsic['peak_index'],**intrinsic['features'],**intrinsic['crossings'],**{k+'_flag':v for k,v in intrinsic['flags'].items()})
    progress['intrinsic_mean_records']+=1;progress['intrinsic_mean_rows']+=len(q)
    intrinsic_wave=mean.gather(1,torch.as_tensor(intrinsic['peak_index'],device=device)[:,None]-16+torch.arange(256,device=device)[None,:]).cpu().numpy()
    reports={}
    for h in r.HYPOTHESES:
        obs=r.observe_generated_carrier(signal+bank['noise'][h],ste=False,q_floor=float(setup['loss'].q_floor),return_carrier=True)
        progress['generated_observation_rows']+=len(signal)
        wave,baseline,peaks=[obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
        report=r.hard_metrics(wave,baseline,setup['data']['real'],setup['data']['real_baseline100'],setup['loss'],setup['context'])
        z=r.joint_embedding(wave,baseline,peaks,dict(T[h],q_floor=float(setup['loss'].q_floor)))
        report['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=device)@projection,real_joint[h],16))
        report['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),np.maximum(-setup['data']['real'],0).sum(1))['value']
        report['observation_flags']={k:dict(count=int(v.sum()),denominator=len(signal)) for k,v in obs['flags'].items()}
        progress['full_native_hypothesis_evaluations']+=1;reports[h]=report
        r.write_json(out/(label+'_'+h+'_METRICS.json'),report)
        features,flags=readouts044(wave);morph=r.morphology018(wave);bins=output_q_bins(features['Q_Vns'],setup['loss'])
        if not enabled:base_by_h[h]=dict(peak_index=peaks.copy(),Q_bin=bins.copy())
        base=base_by_h[h]
        native=dict(parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),input_Q_bin=p['_input_bins'],native_peak_index=peaks,
            WARP068_peak_index=base['peak_index'],WARP068_output_Q_bin=base['Q_bin'],native_output_Q_bin=bins,
            **features,**{k+'_flag':v for k,v in flags.items()},**{k+'_absolute_ns':morph[k]+peaks-16 for k in CROSSINGS053})
        np.savez_compressed(out/(label+'_'+h+'_OBSERVATION_FLAGS.npz'),parent_row=np.arange(len(q)),
            pre_ADC_minimum_V=obs['carrier_input'].min(1).values.cpu().numpy(),pre_ADC_maximum_V=obs['carrier_input'].max(1).values.cpu().numpy(),
            ADC_minimum_V=obs['carrier_adc'].min(1).values.cpu().numpy(),ADC_maximum_V=obs['carrier_adc'].max(1).values.cpu().numpy(),
            global_peak_index=obs['global_peak_index'].cpu().numpy(),**{k:v.cpu().numpy() for k,v in obs['flags'].items()})
        progress['observation_flag_records']+=1;progress['observation_flag_rows']+=len(q)
        np.savez_compressed(out/(label+'_'+h+'_NATIVE_ROWS.npz'),**native)
        np.savez_compressed(out/(label+'_'+h+'_NATIVE_EXAMPLES.npz'),parent_row=np.arange(32),wave=wave[:32],baseline100=baseline[:32],peak_index=peaks[:32])
        progress['feature_records']+=1;progress['feature_rows']+=len(q)
        if h=='fast':
            np.savez_compressed(out/(label+'_fast_CARRIER_EXAMPLES.npz'),parent_row=np.arange(32),mean_pre_V=mean[:32].cpu().numpy(),signal_pre_V=signal[:32].cpu().numpy(),pre_ADC_V=(signal+bank['noise'][h])[:32].cpu().numpy(),ADC_pre_V=obs['carrier_adc'][:32].cpu().numpy(),native_crop_indices=obs['crop_indices'][:32].cpu().numpy())
            save_profiles060(wave,np.array([p['_input_bins'],bins]),np.arange(len(q)),float(setup['loss'].q_floor),out,label+'_fast',
                ['A0_input_Q_quantile','own_native_output_Q'],progress)
            save_profiles060(intrinsic_wave,np.array([p['_input_bins'],base['Q_bin']]),np.arange(len(q)),float(setup['loss'].q_floor),out,label+'_intrinsic',
                ['A0_input_Q_quantile','fixed_WARP068_fast_native_output_Q'],progress)
        r.check_resources(p,out,p['_started'])
        r.write_json(out/'PROGRESS.json',dict(phase='fixed_crossing_level_voltage_warp',seed_index=index,case=label,hypothesis=h,elapsed_seconds=time.monotonic()-p['_started'],execution_counts=dict(progress)))
    return reports,geometry


@torch.no_grad()
def run069(p,out,device,progress):
    started=time.monotonic();p['_started']=started;out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],config_sha256=p['_config_sha256'],pid=os.getpid(),resources=p['resources']))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis'];q=r.canonical_charge008(setup['encoding']['original']['signal_pre']).cpu().numpy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=p['input_Q_array_sha256']:raise RuntimeError('069 input Q bytes differ')
    reference=np.load(ROOT/p['input_reference_file']);p['_input_bins']=reference['input_Q_bin']
    if not np.array_equal(q,reference['A0_canonical_Q_Vns']):raise RuntimeError('069 original input population differs')
    template=json.loads((ROOT/p['template_file']).read_text());replay=json.loads((ROOT/p['replay_file']).read_text())['reports']
    real_wave,real_baseline=setup['data']['real'],setup['data']['real_baseline100'];features,flags=readouts044(real_wave);morph=r.morphology018(real_wave)
    np.savez_compressed(out/'REAL_ROWS.npz',row=np.arange(len(real_wave)),output_Q_bin=output_q_bins(features['Q_Vns'],setup['loss']),**features,**{k+'_flag':v for k,v in flags.items()},**{k+'_from_selected_peak_ns':morph[k]-16 for k in CROSSINGS053})
    save_profiles060(real_wave,np.array([output_q_bins(features['Q_Vns'],setup['loss'])]),np.arange(len(real_wave)),float(setup['loss'].q_floor),out,'REAL',['own_native_output_Q'],progress)
    progress['REAL_feature_rows']+=len(real_wave)
    projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real_wave,real_baseline,np.full(len(real_wave),16),dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device)@projection for h in r.HYPOTHESES}
    r.write_json(out/'DEFINITIONS.json',dict(T_by_hypothesis=T,q_floor=float(setup['loss'].q_floor),mean_transport_enabled=True,random_coupling_enabled=True,crossing_level_warp_enabled=p['level_warp_enabled'],
        support=[110,176],template_file=p['template_file'],template_is_preADC_truth=False,SOURCE_REAL_event_pairing=False,all_rows_retained=True,
        full_profile_hypothesis='fast',all_native_hypotheses=r.HYPOTHESES,original_contract=p['contract']))
    results=[]
    for index in range(2):
        bank=r.make_population_bank(setup,index,'FIT',0,p,progress);base_by_h={}
        m0,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
        r.write_json(out/f'seed{index}_M0.json',m0)
        if json.loads((out/f'seed{index}_M0.json').read_text())!=replay[str(index)]['M0']:raise RuntimeError('069 M0 replay changed')
        for label,enabled in [('WARP068',False),('LEVELWARP',p['level_warp_enabled'])]:
            if index==0 and enabled:progress['new_candidate_settings']+=1
            report,geometry=evaluate069(setup,index,bank,T,projection,real_joint,p,out,progress,enabled,f'seed{index}_{label}',template,base_by_h)
            decision=r.objective039(report,m0,T,p['contract'])
            usefulness={h:dict(wave_relative_gain=(m0[h]['values']['wave']-report[h]['values']['wave'])/m0[h]['values']['wave'],
                joint_relative_gain=(m0[h]['values']['joint_morphology']-report[h]['values']['joint_morphology'])/m0[h]['values']['joint_morphology'],
                morphology_mean_T_gain=float(np.mean([(m0[h]['values'][k]-report[h]['values'][k])/T[h][k] for k in r.MORPH]))) for h in r.HYPOTHESES}
            for h,u in usefulness.items():u['passes_point_targets']=u['wave_relative_gain']>=p['contract']['useful_wave_relative_gain'] and u['joint_relative_gain']>=p['contract']['useful_joint_relative_gain'] and u['morphology_mean_T_gain']>=p['contract']['useful_morphology_mean_T_gain']
            record=dict(seed_index=index,enabled=enabled,metrics=report,geometry=geometry,decision=decision,point_usefulness=usefulness)
            path=out/f'seed{index}_{label}.json';r.write_json(path,record)
            if not enabled:
                if json.loads(path.read_text())['metrics']!=replay[str(index)]['VOLTAGEWARP']['metrics']:raise RuntimeError('069 WARP068 full report differs')
                progress['fixed_case_replays']+=1
            else:results.append(record);progress['candidate_head_evaluations']+=1
        r.write_json(out/'PROGRESS.json',dict(phase='first_head_crossing_level_warp_complete' if index==0 else 'both_heads_complete',seed_index=index,elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
        del bank,base_by_h
    if dict(progress)!=p['operation_bounds']:raise RuntimeError('069 exact operation counts differ')
    r.check_resources(p,out,started)
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]):raise RuntimeError('069 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']:raise RuntimeError('069 A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('069 005 changed')
    passed=all(x['decision']['feasible'] and all(u['passes_point_targets'] for u in x['point_usefulness'].values()) for x in results)
    result=dict(schema='n01/response069-crossing-level-voltage-warp-result/v1',run_id=p['run_id'],results=results,
        stage_outcome='FIT_USEFUL_MEAN_PENDING_CONFIRMATION' if passed else 'CLOSED_NEGATIVE_CROSSING_LEVEL_VOLTAGE_WARP',original_point_targets_passed=passed,
        execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,
        all_parent_count_weights_preserved=True,confirmation_banks_opened=0,DEV_opened=False,mean_stage_accepted=False,full_random_response_acceptance=False,
        scientific_status='response_match_not_established',uncertainty_scope='Conditional50/90% nearest-peak voltage crossing warp anchored at BASE054 mean peak and support ends; one positive-area scale. The target observed profile is not preADC truth; its crossings need not equal mean per-event crossings. Residual uses the same interpolation and deterministic linear integral projection. FIT only; covariance changes; conditional zero expectation is preserved only if it held in the original residual. Multiple crests and input event detail are retained, not replaced by a template.')
    r.write_json(out/'RESULT.json',result);r.check_resources(p,out,started);r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('069 config hash mismatch')
    p=json.loads(args.config.read_text())
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest:raise ValueError('069 source pin mismatch: '+name)
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(p['operation_bounds'],0))
    try:run069(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
