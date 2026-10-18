"""Fixed058 Q-slope probes; full native widths and focused fast stage/crossing diagnosis."""
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
from run_response058 import monotone_core058
from run_response041 import fixed_anchor_stages
from run_response044 import readouts044,FIELDS044
from run_response053 import intrinsic_mean053,CROSSINGS053,carrier_peaks053,CARRIERS053
from wp1_evaluation import summary,output_q_bins

ROOT=Path(__file__).resolve().parents[1]
STAGES059=('mean_at_BASE054_anchor','plus_M0_random','plus_electronic_noise','ADC_at_BASE054_anchor','native_ADC_reselected')


def stage_measurements059(mean,signal,pre_adc,observation,anchor):
    """Existing five-stage windows; explicit absolute software crossings and every flag."""
    stages=fixed_anchor_stages(mean,signal,pre_adc,observation,anchor)
    starts=np.array([anchor[:,0].cpu().numpy()]*4+[observation['crop_indices'][:,0].cpu().numpy()])
    features=[];flags=[];crossings=[]
    for i,tensor in enumerate(stages):
        array=tensor.cpu().numpy();f,g=readouts044(array);m=r.morphology018(array)
        features.append(f);flags.append(g);crossings.append({k:m[k]+starts[i] for k in CROSSINGS053})
    rows=dict(stage_crop_start=starts,**{k+'_by_stage':np.array([v[k] for v in features]) for k in FIELDS044},
        **{k+'_flag_by_stage':np.array([v[k] for v in flags]) for k in flags[0]},
        **{k+'_absolute_ns_by_stage':np.array([v[k] for v in crossings]) for k in CROSSINGS053})
    return rows,stages


@torch.no_grad()
def evaluate059(setup,index,bank,T,projection,real_joint,p,out,progress,coefficients,label,base_by_h,expected,started):
    model=setup['models'][index];device=setup['source'].device
    model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],device=device,dtype=model.mean_adapter[-1].bias.dtype))
    progress['mean_operator_calls']+=1;progress['mean_operator_rows']+=len(setup['source'])
    adapter=model._mean_response(setup['encoding'],enabled=True)
    delta,_,_=local_restitution042(adapter['delta_mean_pre_V'],p['base_restitution_fraction'])
    A0=setup['encoding']['original']['signal_pre'];q=r.canonical_charge008(A0)
    transport,_,_,_,_,_=conditional_sides052(A0+delta,q,1.,p['input_Q_reference_Vns'],p['side_coefficients_ns'])
    fixed_delta=delta+transport;fixed_mean=A0+fixed_delta
    correction,rows=monotone_core058(fixed_mean,q,coefficients,p['input_Q_reference_Vns'])
    delta=fixed_delta+correction;mean=A0+delta
    signal=r._ExactZeroMeanAddition015.apply(bank['M0'],delta)
    geometry=dict(coefficients=coefficients,maximum_absolute_log_tilt=float(rows['log_tilt'].abs().max()),
        core_support_exact=bool((correction[:,:110]==0.).all() and (correction[:,142:]==0.).all()),
        core_rank_preserved_1e14V=bool((rows['core_rank_minimum_adjacent_difference_V']>=-1e-14).all()),
        core_peak_changed_rows=int((rows['old_core_peak_index']!=rows['new_core_peak_index']).sum()),
        boundary_delta_max_absolute_V=float(torch.maximum(rows['boundary_delta_110_V'].abs(),rows['boundary_delta_141_V'].abs()).max()),
        frozen_prefix_bitwise_unchanged=torch.equal(signal[:,:100],bank['M0'][:,:100]),
        mean_tail_from142_bitwise_unchanged=torch.equal(mean[:,142:],fixed_mean[:,142:]),
        nonpositive_mean_samples_unchanged=torch.equal(mean[fixed_mean<=0.],fixed_mean[fixed_mean<=0.]),
        mean_negative_area_max_change_Vns=float(rows['negative_area_change_Vns'].abs().max()),
        maximum_full_integral_residual_Vns=float(rows['full_integral_residual_Vns'].abs().max()),
        maximum_canonical_Q_residual_Vns=float((r.canonical_charge008(mean)-q).abs().max()),
        maximum_correction_canonical_Q_residual_Vns=float(rows['canonical_Q_residual_Vns'].abs().max()),
        maximum_absolute_correction_V=float(correction.abs().max()),
        mean_peak_changed_rows=int((rows['old_mean_peak_index']!=rows['new_mean_peak_index']).sum()),
        redistribution_summaries={k:summary(z.cpu().numpy()) for k,z in rows.items()})
    if not all(geometry[k] for k in ('core_support_exact','core_rank_preserved_1e14V','frozen_prefix_bitwise_unchanged','mean_tail_from142_bitwise_unchanged','nonpositive_mean_samples_unchanged')):
        raise RuntimeError('058 support/core-order/prefix/tail/nonpositive invariant failed')
    if geometry['mean_negative_area_max_change_Vns']!=0. or max(geometry['maximum_full_integral_residual_Vns'],geometry['maximum_correction_canonical_Q_residual_Vns'])>1e-12 or geometry['maximum_canonical_Q_residual_Vns']>1e-10:
        raise RuntimeError('058 area or negative-part invariant failed')
    if not bool(torch.isfinite(mean).all()):raise RuntimeError('058 nonfinite mean')
    np.savez_compressed(out/(label+'_GEOMETRY_ROWS.npz'),parent_row=np.arange(len(q)),**{k:z.cpu().numpy() for k,z in rows.items()})
    np.savez_compressed(out/(label+'_EXAMPLES.npz'),parent_row=np.arange(32),input_Q_Vns=q[:32].cpu().numpy(),
        fixed_mean_pre_V=fixed_mean[:32].cpu().numpy(),adapted_mean_pre_V=mean[:32].cpu().numpy(),correction_pre_V=correction[:32].cpu().numpy())
    progress['geometry_records']+=1;progress['geometry_rows']+=len(q)
    intrinsic=intrinsic_mean053(mean)
    intrinsic_rows=dict(parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),input_Q_bin=p['_input_bins'],
        intrinsic_peak_index=intrinsic['peak_index'],**intrinsic['features'],**intrinsic['crossings'],
        **{k+'_flag':v for k,v in intrinsic['flags'].items()})
    np.savez_compressed(out/(label+'_INTRINSIC_MEAN_ROWS.npz'),**intrinsic_rows)
    r.write_json(out/(label+'_INTRINSIC_MEAN.json'),dict(features={k:summary(v) for k,v in intrinsic['features'].items()},
        crossings_absolute_ns={k:summary(v) for k,v in intrinsic['crossings'].items()},
        flags={k:dict(count=int(v.sum()),denominator=len(q)) for k,v in intrinsic['flags'].items()},own_bounded_peak=True,no_ADC_or_random=True))
    progress['intrinsic_mean_records']+=1;progress['intrinsic_mean_rows']+=len(q)
    reports={}
    for h in r.HYPOTHESES:
        obs=r.observe_generated_carrier(signal+bank['noise'][h],ste=False,q_floor=float(setup['loss'].q_floor),return_carrier=True)
        progress['generated_observation_rows']+=len(signal)
        wave,baseline,peaks=[obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
        report=r.hard_metrics(wave,baseline,setup['data']['real'],setup['data']['real_baseline100'],setup['loss'],setup['context'])
        z=r.joint_embedding(wave,baseline,peaks,dict(T[h],q_floor=float(setup['loss'].q_floor)))
        report['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=signal.device)@projection,real_joint[h],16))
        report['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),np.maximum(-setup['data']['real'],0).sum(1))['value']
        report['observation_flags']={k:dict(count=int(v.sum()),denominator=len(signal)) for k,v in obs['flags'].items()}
        progress['full_native_hypothesis_evaluations']+=1;reports[h]=report
        metric_path=out/(label+'_'+h+'_METRICS.json');r.write_json(metric_path,report)
        if json.loads(metric_path.read_text())!=expected[h]:raise RuntimeError('059 full native report differs from058: '+label+'/'+h)
        features,flags=readouts044(wave);morph=r.morphology018(wave);native_bins=output_q_bins(features['Q_Vns'],setup['loss'])
        if label.endswith('BASE054'):
            base_by_h[h]=dict(indices=obs['crop_indices'].clone(),peak_index=peaks.copy(),Q_bin=native_bins.copy())
        base=base_by_h[h]
        rows=dict(parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),input_Q_bin=p['_input_bins'],
            native_peak_index=peaks,BASE054_peak_index=base['peak_index'],BASE054_output_Q_bin=base['Q_bin'],native_output_Q_bin=native_bins,
            **features,**{k+'_flag':v for k,v in flags.items()},
            **{k+'_absolute_ns':morph[k]+peaks-16 for k in CROSSINGS053})
        np.savez_compressed(out/(label+'_'+h+'_NATIVE_ROWS.npz'),**rows)
        progress['feature_records']+=1;progress['feature_rows']+=len(q)
        if h=='fast':
            stage_rows,stages=stage_measurements059(mean,signal,signal+bank['noise'][h],obs,base['indices'])
            stage_rows.update(parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),input_Q_bin=p['_input_bins'],BASE054_output_Q_bin=base['Q_bin'])
            stage_rows['output_Q_bin_by_stage']=np.array([output_q_bins(v,setup['loss']) for v in stage_rows['Q_Vns_by_stage']])
            carriers=[carrier_peaks053(value) for value in (mean,signal,signal+bank['noise'][h],obs['carrier_adc'])]
            for k in carriers[0]:stage_rows[k+'_by_carrier_stage']=np.array([v[k] for v in carriers])
            if not np.array_equal(stage_rows['selected_peak_index_by_carrier_stage'][-1],peaks):raise RuntimeError('059 ADC bounded peak mismatch')
            if not np.array_equal(stage_rows['global_peak_index_by_carrier_stage'][-1],obs['global_peak_index'].cpu().numpy()):raise RuntimeError('059 ADC global peak mismatch')
            for k,value in features.items():
                if not np.array_equal(stage_rows[k+'_by_stage'][-1],value,equal_nan=True):raise RuntimeError('059 final stage feature differs: '+k)
            for k,value in flags.items():
                if not np.array_equal(stage_rows[k+'_flag_by_stage'][-1],value):raise RuntimeError('059 final stage flag differs: '+k)
            np.savez_compressed(out/(label+'_fast_STAGE_ROWS.npz'),**stage_rows)
            np.savez_compressed(out/(label+'_fast_STAGE_EXAMPLES.npz'),parent_row=np.arange(32),mean_pre_V=mean[:32].cpu().numpy(),
                signal_pre_V=signal[:32].cpu().numpy(),pre_ADC_V=(signal+bank['noise'][h])[:32].cpu().numpy(),ADC_pre_V=obs['carrier_adc'][:32].cpu().numpy(),
                BASE054_crop_indices=base['indices'][:32].cpu().numpy(),native_crop_indices=obs['crop_indices'][:32].cpu().numpy(),
                stage_windows_V=np.array([value[:32].cpu().numpy() for value in stages]))
            r.write_json(out/(label+'_fast_STAGE_SUMMARY.json'),dict(stage_order=STAGES059,carrier_stage_order=CARRIERS053,all_rows=len(q),
                stage_features={k:[summary(v) for v in stage_rows[k+'_by_stage']] for k in FIELDS044},
                stage_flags={k:[dict(count=int(v.sum()),denominator=len(q)) for v in stage_rows[k+'_flag_by_stage']] for k in flags},
                full_stage_scope='fast only; slow/common_level native rows and crossings preserved',ordered_not_unique_causal_effects=True,
                core_support=[110,142],support_is_software_prior=True))
            progress['stage_feature_records']+=1;progress['stage_feature_observations']+=5;progress['stage_feature_rows']+=5*len(q)
            progress['carrier_stage_summaries']+=4;progress['carrier_stage_rows']+=4*len(q)
            if not label.endswith('BASE054'):progress['fixed_nonzero_fast_replays']+=1
        r.check_resources(p,out,started)
        r.write_json(out/'PROGRESS.json',dict(phase='fixed058_hypothesis_and_width_replay',seed_index=index,case=label,hypothesis=h,
            elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
    return reports,geometry



@torch.no_grad()
def run059(p,out,device,progress):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],config_sha256=p['_config_sha256'],pid=os.getpid(),resources=p['resources']))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis'];q=r.canonical_charge008(setup['encoding']['original']['signal_pre']).cpu().numpy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=p['input_Q_array_sha256']:raise RuntimeError('059 input Q bytes differ')
    input_reference=np.load(ROOT/p['input_reference_file']);p['_input_bins']=input_reference['input_Q_bin']
    if not np.array_equal(q,input_reference['A0_canonical_Q_Vns']):raise RuntimeError('059 input population differs')
    real_wave,real_baseline=setup['data']['real'],setup['data']['real_baseline100'];real_features,real_flags=readouts044(real_wave);real_morph=r.morphology018(real_wave)
    np.savez_compressed(out/'REAL_ROWS.npz',row=np.arange(len(real_wave)),output_Q_bin=output_q_bins(real_features['Q_Vns'],setup['loss']),
        **real_features,**{k+'_flag':v for k,v in real_flags.items()},**{k+'_from_selected_peak_ns':real_morph[k]-16. for k in CROSSINGS053})
    progress['REAL_feature_rows']+=len(real_wave)
    r.write_json(out/'DEFINITIONS.json',dict(stage_order=STAGES059,carrier_stage_order=CARRIERS053,full_stage_hypotheses=['fast'],
        native_hypotheses=r.HYPOTHESES,input_reference_file=p['input_reference_file'],input_Q_sha256=p['input_Q_array_sha256'],
        crossing_coordinates='SOURCE absolute software carrier sample; REAL offset from existing selected peak. ns is fixed software sample unit, not hardware calibration.',
        REAL_pre_ADC='unknown',SOURCE_REAL_event_pairing=False,output_Q_is_energy_truth=False,all_rows_retained=True,
        fixed_anchor_peak_not_always_current_max=True,ordered_contrasts_not_unique_causal_effects=True))
    projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real_wave,real_baseline,np.full(len(real_wave),16),dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device)@projection for h in r.HYPOTHESES}
    replay=json.loads((ROOT/p['replay_file']).read_text())['reports'];checks=[]
    for index in (0,1):
        r.check_resources(p,out,started);bank=r.make_population_bank(setup,index,'FIT',0,p,progress)
        m0,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
        path=out/f'seed{index}_M0.json';r.write_json(path,m0)
        if json.loads(path.read_text())!=replay[str(index)]['M0']:raise RuntimeError('059 full M0 replay differs')
        checks.append(dict(seed=index,case='M0',hypotheses=3,full_JSON_exact=True));base_by_h={}
        for case in p['cases']:
            name=case['name'];expected=replay[str(index)][name];r.check_resources(p,out,started)
            report,geo=evaluate059(setup,index,bank,T,projection,real_joint,p,out,progress,case['coefficients'],f'seed{index}_{name}',base_by_h,expected['metrics'],started)
            record=dict(metrics=report,geometry=geo,decision=r.objective039(report,m0,T,p['contract']))
            path=out/f'seed{index}_{name}.json';r.write_json(path,record);saved=json.loads(path.read_text())
            if saved!=expected:raise RuntimeError('059 full fixed058 case metrics/geometry/decision differ: '+name)
            progress['fixed_case_replays']+=1
            if name!='BASE054':progress['fixed_probe_replays']+=1
            checks.append(dict(seed=index,case=name,hypotheses=3,full_JSON_exact=True));r.write_json(out/'REPLAY_CHECKS.json',checks)
            r.write_json(out/'PROGRESS.json',dict(phase='fixed058_full_case_complete',seed_index=index,case=name,
                elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
        del bank,base_by_h
    if dict(progress)!=p['operation_bounds']:raise RuntimeError('059 exact operation counts differ')
    r.check_resources(p,out,started)
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]):raise RuntimeError('059 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']:raise RuntimeError('059 A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('059 005 changed')
    result=dict(schema='n01/response059-diagnostic-result/v1',run_id=p['run_id'],stage_outcome='CLOSED_FIXED058_WIDTH_DIAGNOSTIC_VERIFIED',
        diagnostic_complete=True,replay_checks=checks,execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,
        frozen_A0_M0_005_verified=True,all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        full_stage_hypotheses=['fast'],full_native_hypotheses=r.HYPOTHESES,model_selected=False,new_candidate_settings=0,
        confirmation_banks_opened=0,DEV_opened=False,mean_stage_accepted=False,full_random_response_acceptance=False,
        scientific_status='response_match_not_established',next_step='Analyze paired conditional CDFs, crossing/support locations and stage variance; then choose the next evidence-backed mean representation.')
    r.write_json(out/'RESULT.json',result);r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('059 config hash mismatch')
    p=json.loads(args.config.read_text())
    for n,digest in p['source_pins'].items():
        if r.sha256(ROOT/n)!=digest:raise ValueError('059 source pin mismatch: '+n)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']:raise ValueError('059 setup hash mismatch')
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(p['operation_bounds'],0))
    try:run059(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
