"""Frozen063 replay with ordered waveform/QPSD/crossing and crop diagnostics."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import numpy as np
import torch
import run_response039 as r
from run_response041 import fixed_anchor_stages,ordered_contrasts
from run_response042 import local_restitution042
from run_response044 import readouts044,FIELDS044
from run_response046 import carrier_peaks046
from run_response052 import conditional_sides052
from run_response053 import intrinsic_mean053,wave_views053,CROSSINGS053,write_compact053
from run_response063 import shape_residual063
from wp1_evaluation import summary,output_q_bins

ROOT=Path(__file__).resolve().parents[1]
STAGES064=('mean_at_BASE054_anchor','plus_M0_random','plus_electronic_noise','ADC_at_BASE054_anchor','native_ADC_reselected')
FIELDS064=FIELDS044+('PSD_raw','PSD_guarded')


def readouts064(wave,qfloor):
    features,flags=readouts044(wave);q=features['Q_Vns'];tail=features['tail_Vns'];psd=np.full(len(q),np.nan)
    np.divide(tail,q,out=psd,where=q!=0.)
    features.update(PSD_raw=psd,PSD_guarded=tail/(np.where(q<0.,-1.,1.)*np.maximum(np.abs(q),qfloor)))
    flags.update(Q_nonpositive=q<=0.,PSD_undefined=q==0.,PSD_guarded=np.abs(q)<qfloor)
    return features,flags


def verify_arrays064(name,arrays,reference):
    for key,meta in reference[name].items():
        value=np.asarray(arrays[key])
        if list(value.shape)!=meta['shape'] or str(value.dtype)!=meta['dtype'] or hashlib.sha256(value.tobytes()).hexdigest()!=meta['sha256']:
            raise RuntimeError('064 fixed063 array changed: '+name+'/'+key)


def positivity_audit064(mean,correction,rows,template):
    device=mean.device;sp=torch.as_tensor(template['source_unit_Q_profiles'],device=device,dtype=torch.float64);tp=torch.as_tensor(template['target_unit_Q_profiles'],device=device,dtype=torch.float64)
    si,ti,sw,tw=(rows[k] for k in ('source_interval','target_interval','source_right_weight','target_right_weight'))
    source=(1.-sw[:,None])*sp[si]+sw[:,None]*sp[si+1];target=(1.-tw[:,None])*tp[ti]+tw[:,None]*tp[ti+1]
    locations=torch.arange(110,176,device=device)[None,:]-rows['old_mean_peak_index'][:,None]+16;local=mean[:,110:176];mask=(locations>=2)&(local>0.)
    direction=rows['target_normalization'][:,None]*target.gather(1,locations.clamp_min(2))*mask-rows['source_normalization'][:,None]*source.gather(1,locations.clamp_min(2))*mask
    bound=torch.where(direction<0.,local/torch.where(direction<0.,-direction,torch.ones_like(direction)),torch.full_like(local,float('inf')))
    limit,index=bound.min(1);bounded=torch.isfinite(limit);selected=index[:,None]
    if float((correction[:,110:176]-rows['applied_scale'][:,None]*direction).abs().max())>1e-12:raise RuntimeError('064 direction differs from frozen063')
    if not torch.equal(torch.minimum(torch.ones_like(limit),limit),rows['feasible_scale']):raise RuntimeError('064 raw positivity bound differs')
    result=dict(raw_positivity_limit=limit,has_negative_direction=bounded,limiting_absolute_index=torch.where(bounded,index+110,torch.full_like(index,-1)),
        limiting_relative_index=torch.where(bounded,locations.gather(1,selected).squeeze(1),torch.full_like(index,-1)),
        limiting_old_amplitude_V=torch.where(bounded,local.gather(1,selected).squeeze(1),torch.full_like(limit,float('nan'))),
        limiting_raw_direction_V=torch.where(bounded,direction.gather(1,selected).squeeze(1),torch.full_like(limit,float('nan'))))
    return direction,result


@torch.no_grad()
def run064(p,out,device,progress):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],pid=os.getpid(),config_sha256=p['_config_sha256'],resources=p['resources']))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    reference=json.loads((ROOT/p['diagnostic_reference_file']).read_text());examples=json.loads((ROOT/p['example_selection_file']).read_text())['rows']
    template=json.loads((ROOT/p['template_file']).read_text());T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    A0=setup['encoding']['original']['signal_pre'];q=r.canonical_charge008(A0);qfloor=float(setup['loss'].q_floor);N=len(A0)
    input_reference=np.load(ROOT/p['input_reference_file']);input_bins=input_reference['input_Q_bin'];qn=q.cpu().numpy()
    if not np.array_equal(qn,input_reference['A0_canonical_Q_Vns']) or hashlib.sha256(qn.tobytes()).hexdigest()!=p['input_Q_array_sha256']:raise RuntimeError('064 input identity changed')
    real_wave,real_baseline=setup['data']['real'],setup['data']['real_baseline100'];real_tensor=torch.as_tensor(real_wave,device=device,dtype=torch.float64)
    rf,rflags=readouts064(real_wave,qfloor);rm=r.morphology018(real_wave)
    real_rows=dict(row=np.arange(len(real_wave)),output_Q_bin=output_q_bins(rf['Q_Vns'],setup['loss']),**{k:rf[k] for k in FIELDS044},
        **{k+'_flag':v for k,v in readouts044(real_wave)[1].items()},**{k+'_from_selected_peak_ns':rm[k]-16 for k in CROSSINGS053})
    verify_arrays064('REAL_ROWS.npz',real_rows,reference['arrays']);progress['reference_array_records_verified']+=1
    np.savez_compressed(out/'REAL_ROWS.npz',**real_rows);progress['REAL_feature_rows']+=len(real_wave)
    projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real_wave,real_baseline,np.full(len(real_wave),16),dict(T[h],q_floor=qfloor)),device=device)@projection for h in r.HYPOTHESES}
    write_compact053(out/'DEFINITIONS.json',dict(stages=STAGES064,features=FIELDS064,q_floor=qfloor,T_by_hypothesis=T,
        fixed_anchor='Each same-parent BASE054 native ADC peak and crop, separate by head and H',
        fixed_stage_morphology='Original index16 reference retained even if it is not the stage maximum; flag retained, no native-acceptance substitution',
        scalar_W1_defined_population=True,all_rows_retained=True,SOURCE_rows=N,REAL_rows=len(real_wave),REAL_preADC_unknown=True,
        ordered_contributions_are_order_dependent=True,original_native_acceptance_unchanged=True,new_candidate_settings=0,
        original063_profile_files_reused_not_recomputed=True,example_selection_file=p['example_selection_file'],input_Q_bins=input_bins,
        output_Q_reference_Vns=float(setup['loss'].q_reference),output_Q_boundaries=setup['loss'].charge_bin_boundaries.cpu().numpy(),
        REAL_features={k:summary(rf[k]) for k in FIELDS064},REAL_flags={k:dict(count=int(v.sum()),denominator=len(real_wave)) for k,v in rflags.items()}))
    replay_checks=[];pair_summaries=[]
    for seed in range(2):
        bank=r.make_population_bank(setup,seed,'FIT',0,p,progress);cases={}
        for case,enabled in [('BASE054',False),('RESIDUAL',True)]:
            label=f'seed{seed}_{case}';model=setup['models'][seed]
            model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],device=device,dtype=model.mean_adapter[-1].bias.dtype))
            adapter=model._mean_response(setup['encoding'],enabled=True);progress['mean_operator_calls']+=1;progress['mean_operator_rows']+=N
            delta,_,_=local_restitution042(adapter['delta_mean_pre_V'],p['base_restitution_fraction'])
            transport,_,_,_,_,_=conditional_sides052(A0+delta,q,1.,p['input_Q_reference_Vns'],p['side_coefficients_ns'])
            fixed_delta=delta+transport;fixed_mean=A0+fixed_delta;correction,geo=shape_residual063(fixed_mean,q,enabled,template)
            delta=fixed_delta+correction;mean=fixed_mean+correction;geo['new_mean_peak_index']=mean[:,112:144].argmax(1)+112;geo['new_peak_voltage_V']=mean[:,112:144].max(1).values
            signal=r._ExactZeroMeanAddition015.apply(bank['M0'],delta)
            geometry=dict(parent_row=np.arange(N),**{k:v.cpu().numpy() for k,v in geo.items()})
            mean_examples=dict(parent_row=np.arange(32),input_Q_Vns=qn[:32],fixed_mean_pre_V=fixed_mean[:32].cpu().numpy(),adapted_mean_pre_V=mean[:32].cpu().numpy(),correction_pre_V=correction[:32].cpu().numpy())
            intrinsic=intrinsic_mean053(mean);intrinsic_rows=dict(parent_row=np.arange(N),input_Q_Vns=qn,input_Q_bin=input_bins,intrinsic_peak_index=intrinsic['peak_index'],**intrinsic['features'],**intrinsic['crossings'],**{k+'_flag':v for k,v in intrinsic['flags'].items()})
            for suffix,data in [('GEOMETRY_ROWS',geometry),('EXAMPLES',mean_examples),('INTRINSIC_MEAN_ROWS',intrinsic_rows)]:
                verify_arrays064(label+'_'+suffix+'.npz',data,reference['arrays']);progress['reference_array_records_verified']+=1
                np.savez_compressed(out/(label+'_'+suffix+'.npz'),**data)
            progress['geometry_records']+=1;progress['geometry_rows']+=N;progress['intrinsic_mean_records']+=1;progress['intrinsic_mean_rows']+=N
            direction,limits=positivity_audit064(fixed_mean,correction,geo,template)
            np.savez_compressed(out/(label+'_POSITIVITY_ROWS.npz'),parent_row=np.arange(N),input_Q_Vns=qn,input_Q_bin=input_bins,**{k:v.cpu().numpy() for k,v in limits.items()})
            progress['positivity_audit_records']+=1;progress['positivity_audit_rows']+=N
            cases[case]=dict(mean=mean,signal=signal,direction=direction,geometry=geometry)
        for h in r.HYPOTHESES:
            paired={}
            for case in ('BASE054','RESIDUAL'):
                r.check_resources(p,out,started);label=f'seed{seed}_{case}_{h}';state=cases[case];mean,signal=state['mean'],state['signal'];pre=signal+bank['noise'][h]
                obs=r.observe_generated_carrier(pre,ste=False,q_floor=qfloor,return_carrier=True);progress['generated_observation_rows']+=N
                wave,baseline,peaks=[obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
                metrics=r.hard_metrics(wave,baseline,real_wave,real_baseline,setup['loss'],setup['context'])
                z=r.joint_embedding(wave,baseline,peaks,dict(T[h],q_floor=qfloor));metrics['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=device)@projection,real_joint[h],16))
                metrics['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),rf['negative_lobe_area_Vns'])['value']
                metrics['observation_flags']={k:dict(count=int(v.sum()),denominator=N) for k,v in obs['flags'].items()}
                write_compact053(out/(label+'_METRICS.json'),metrics);progress['full_native_hypothesis_evaluations']+=1
                if json.loads((out/(label+'_METRICS.json')).read_text())!=reference['reports'][str(seed)][case]['metrics'][h]:raise RuntimeError('064 full native063 replay changed: '+label)
                replay_checks.append(dict(seed=seed,case=case,hypothesis=h,full_report_exact063=True));r.write_json(out/'REPLAY_CHECKS.json',replay_checks)
                native,flags=readouts044(wave);morph=r.morphology018(wave);own_bins=output_q_bins(native['Q_Vns'],setup['loss'])
                if case=='BASE054':anchor=obs['crop_indices'];base_peaks=peaks.copy();base_bins=own_bins.copy()
                native_rows=dict(parent_row=np.arange(N),input_Q_Vns=qn,input_Q_bin=input_bins,native_peak_index=peaks,BASE054_peak_index=base_peaks,BASE054_output_Q_bin=base_bins,native_output_Q_bin=own_bins,
                    **native,**{k+'_flag':v for k,v in flags.items()},**{k+'_absolute_ns':morph[k]+peaks-16 for k in CROSSINGS053})
                observation=dict(parent_row=np.arange(N),pre_ADC_minimum_V=obs['carrier_input'].min(1).values.cpu().numpy(),pre_ADC_maximum_V=obs['carrier_input'].max(1).values.cpu().numpy(),
                    ADC_minimum_V=obs['carrier_adc'].min(1).values.cpu().numpy(),ADC_maximum_V=obs['carrier_adc'].max(1).values.cpu().numpy(),global_peak_index=obs['global_peak_index'].cpu().numpy(),**{k:v.cpu().numpy() for k,v in obs['flags'].items()})
                verify_arrays064(label+'_NATIVE_ROWS.npz',native_rows,reference['arrays']);verify_arrays064(label+'_OBSERVATION_FLAGS.npz',observation,reference['arrays']);progress['reference_array_records_verified']+=2
                np.savez_compressed(out/(label+'_OBSERVATION_FLAGS.npz'),**observation)
                carriers=(mean,signal,pre,obs['carrier_adc']);carrier_values=[carrier_peaks046(x) for x in carriers]
                if not np.array_equal(carrier_values[-1]['selected_peak_index'],peaks):raise RuntimeError('064 ADC carrier peak differs')
                progress['carrier_stage_summaries']+=4;progress['carrier_stage_rows']+=4*N
                stages=fixed_anchor_stages(mean,signal,pre,obs,anchor);values={k:[] for k in FIELDS064};flag_values={};crossings={k:[] for k in CROSSINGS053};stage_reports={};means=[];SDs=[];group_means=[];group_SDs=[]
                for stage,tensor in zip(STAGES064,stages):
                    r.check_resources(p,out,started);array=tensor.cpu().numpy();features,sflags=readouts064(array,qfloor);morph=r.morphology018(array)
                    for k in FIELDS064:values[k].append(features[k])
                    for k,val in sflags.items():flag_values.setdefault(k,[]).append(val)
                    start=base_peaks-16 if stage!=STAGES064[-1] else peaks-16
                    for k in CROSSINGS053:crossings[k].append(morph[k]+start)
                    views=wave_views053(tensor,real_tensor,setup['loss'],setup['context']);qpsd=float(setup['loss']._qpsd_with_context(tensor.double(),real_tensor,None,setup['context'],mean_response=False));coverage=copy.deepcopy(setup['loss'].last_diagnostics)
                    progress['wave_view_stage_summaries']+=1;progress['wave_view_stage_rows']+=N;progress['qpsd_stage_evaluations']+=1;progress['qpsd_stage_rows']+=N
                    means.append(array.mean(0));SDs.append(array.std(0));groups={};gm=[];gs=[]
                    for group,bins in [('input_Q',input_bins),('fixed_BASE054_output_Q',base_bins)]:
                        gm.append(np.array([array[bins==b].mean(0) for b in range(8)]));gs.append(np.array([array[bins==b].std(0) for b in range(8)]))
                        groups[group]=[dict(bin=b,rows=int(np.sum(bins==b)),features={k:summary(features[k][bins==b]) for k in FIELDS064},flags={k:int(v[bins==b].sum()) for k,v in sflags.items()}) for b in range(8)]
                    group_means.append(gm);group_SDs.append(gs)
                    stage_reports[stage]=dict(wave_views=views,QPSD=qpsd,QPSD_coverage=coverage,features={k:summary(features[k]) for k in FIELDS064},
                        W1_to_REAL={k:r.raw_distance(features[k],rf[k]) for k in FIELDS064},flags={k:dict(count=int(v.sum()),denominator=N) for k,v in sflags.items()},groups=groups)
                    progress['counterfactual_stage_summaries']+=1;progress['counterfactual_stage_rows']+=N
                native_stage=stage_reports[STAGES064[-1]]
                if max(abs(native_stage['wave_views']['aggregate_parts'][k]-value) for k,value in metrics['wave_components'].items())>1e-12:raise RuntimeError('064 original wave components do not close')
                if abs(native_stage['QPSD']-metrics['values']['QPSD'])>1e-12:raise RuntimeError('064 original QPSD differs')
                for k in FIELDS064:
                    if abs(native_stage['W1_to_REAL'][k]['value']-metrics['values'][k])>1e-11:raise RuntimeError('064 native scalar W1 differs: '+k)
                if case=='BASE054' and not torch.equal(stages[-2],stages[-1]):raise RuntimeError('064 BASE anchor is not native')
                values={k:np.array(v) for k,v in values.items()};flag_values={k:np.array(v) for k,v in flag_values.items()}
                stage_rows=dict(parent_row=np.arange(N),input_Q_Vns=qn,input_Q_bin=input_bins,BASE054_output_Q_bin=base_bins,native_output_Q_bin=own_bins,BASE054_peak_index=base_peaks,native_peak_index=peaks,
                    population_wave_mean_V=np.array(means),population_wave_SD_V=np.array(SDs),group_wave_mean_V=np.array(group_means),group_wave_SD_V=np.array(group_SDs),
                    group_counts=np.array([[np.sum(bins==b) for b in range(8)] for bins in (input_bins,base_bins)]),
                    **{k+'_by_stage':v for k,v in values.items()},**{k+'_flag_by_stage':v for k,v in flag_values.items()},**{k+'_absolute_ns_by_stage':np.array(v) for k,v in crossings.items()},
                    **{k+'_by_carrier_stage':np.array([v[k] for v in carrier_values]) for k in carrier_values[0]})
                np.savez_compressed(out/(label+'_STAGE_ROWS.npz'),**stage_rows)
                selected=np.asarray(examples[label],dtype=np.int64);sel=torch.as_tensor(selected,device=device)
                np.savez_compressed(out/(label+'_DIAGNOSTIC_EXAMPLES.npz'),parent_row=selected,carriers=np.array([x[sel].cpu().numpy() for x in carriers]),waves=np.array([x[sel].cpu().numpy() for x in stages]),
                    BASE054_crop_indices=anchor[sel].cpu().numpy(),native_crop_indices=obs['crop_indices'][sel].cpu().numpy(),raw_direction110_176=state['direction'][sel].cpu().numpy(),applied_scale=state['geometry']['applied_scale'][selected])
                progress['diagnostic_example_records']+=1;progress['diagnostic_example_rows']+=len(selected)
                write_compact053(out/(label+'_STAGE_REPORT.json'),dict(seed=seed,case=case,hypothesis=h,stages=stage_reports))
                paired[case]=dict(values=values,flags=flag_values,reports=stage_reports,peaks=peaks)
            base,new=paired['BASE054'],paired['RESIDUAL'];contrasts={}
            for k in ('Q_Vns','front_Vns','tail_Vns'):
                z=ordered_contrasts(new['values'][k],base['values'][k])
                if z['closure_max_error']>1e-12:raise RuntimeError('064 row contribution closure failed')
                contrasts[k]=dict(closure_max_error=z['closure_max_error'],stage_delta=[summary(x) for x in z['stage_delta']],ordered_contributions=[summary(x) for x in z['ordered_contributions']])
            score_contrasts={}
            for k in ('wave','QPSD')+FIELDS064:
                old=[(base['reports'][s]['wave_views']['aggregate_wave'] if k=='wave' else base['reports'][s]['QPSD'] if k=='QPSD' else base['reports'][s]['W1_to_REAL'][k]['value']) for s in STAGES064]
                actual=[(new['reports'][s]['wave_views']['aggregate_wave'] if k=='wave' else new['reports'][s]['QPSD'] if k=='QPSD' else new['reports'][s]['W1_to_REAL'][k]['value']) for s in STAGES064]
                score_contrasts[k]=dict(BASE=old,RESIDUAL=actual,**ordered_contrasts(actual,old))
            components={}
            for key in ('swd_by_view','coordinate_by_16sample_block','short_lag_by_view_and_lag','aggregate_parts'):
                components[key]={}
                for name in base['reports'][STAGES064[0]]['wave_views'][key]:
                    old=[base['reports'][s]['wave_views'][key][name] for s in STAGES064];actual=[new['reports'][s]['wave_views'][key][name] for s in STAGES064]
                    components[key][name]=ordered_contrasts(actual,old)
            transitions={k:dict(BASE_counts=base['flags'][k].sum(1),RESIDUAL_counts=new['flags'][k].sum(1),
                new_relative_to_BASE=(new['flags'][k]&~base['flags'][k]).sum(1),resolved_relative_to_BASE=(~new['flags'][k]&base['flags'][k]).sum(1),
                RESIDUAL_adjacent_on=(new['flags'][k][1:]&~new['flags'][k][:-1]).sum(1),RESIDUAL_adjacent_off=(~new['flags'][k][1:]&new['flags'][k][:-1]).sum(1)) for k in new['flags']}
            shift=new['peaks']-base['peaks'];record=dict(seed=seed,hypothesis=h,stage_order=STAGES064,order_dependent=True,row_contrasts=contrasts,score_contrasts=score_contrasts,wave_component_contrasts=components,flag_transitions=transitions,peak_shift_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(shift,return_counts=True))})
            write_compact053(out/f'seed{seed}_{h}_PAIRED_CONTRASTS.json',record);pair_summaries.append(record);progress['diagnostic_pair_evaluations']+=1
            r.write_json(out/'PROGRESS.json',dict(phase='paired_readout_diagnostic_complete',seed_index=seed,hypothesis=h,elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
        progress['fixed_case_replays']+=2
        del bank,cases,paired
    if dict(progress)!=p['operation_bounds']:raise RuntimeError('064 exact operation counts differ')
    r.check_resources(p,out,started)
    for seed,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][seed]):raise RuntimeError('064 M0 changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest'] or r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('064 frozen A0/005 changed')
    result=dict(schema='n01/response064-ordered-readout-audit-result/v1',run_id=p['run_id'],stage_outcome='DIAGNOSTIC_COMPLETE_PENDING_ROOT_REVIEW',replay_checks=replay_checks,execution_counts=dict(progress),
        elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,all_TRAIN_SOURCE_parents=N,all_TRAIN_REAL_rows=len(real_wave),all_parent_count_weights_preserved=True,
        confirmation_banks_opened=0,DEV_opened=False,new_candidate_settings=0,mean_stage_accepted=False,full_random_response_acceptance=False,scientific_status='response_match_not_established',
        uncertainty_scope='Unchanged063 endpoint only; ordered fixed-anchor diagnostics are not native acceptance or a unique causal attribution; no full-population wave export.')
    r.write_json(out/'RESULT.json',result);r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('064 config hash mismatch')
    p=json.loads(args.config.read_text())
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest:raise ValueError('064 source pin mismatch: '+name)
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(p['operation_bounds'],0))
    try:run064(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
