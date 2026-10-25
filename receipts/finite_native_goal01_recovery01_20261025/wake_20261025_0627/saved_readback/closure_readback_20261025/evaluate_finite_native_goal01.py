"""Native readback of the single frozen actual-hard-objective fit endpoint."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,time
import numpy as np
import torch
from n01.data import write_json
from n01.global_memory01 import GlobalDistributionMemory01,MeanHeadRepair01
from n01.joint01_model import JointResponse01
from n01.model008 import canonical_charge008,relative_time_warp008
from n01.losses008_v2 import ProjectionContext
from n01.response008 import loss_tensor_statistics
from n01.observation008 import observe_existing_carrier,observe_generated_carrier
from run_joint_response01 import load_joint_setup,random_bank
from run_response036 import NativeMetricLoss036,hard_metrics
from stability011_metrics import CDFCache,VIEWS,COORDINATE_VIEWS,LAGS,wave_views,q_features

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--config-sha256',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert hashlib.sha256(args.config.read_bytes()).hexdigest()==args.config_sha256
p=json.loads(args.config.read_bytes());assert p['registration_status']=='REGISTERED_FOR_EXECUTION'
args.out.mkdir(parents=True,exist_ok=False)
began=time.monotonic();counts=Counter({key:0 for key in p['operation_budget']})
torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.use_deterministic_algorithms(True)
device=torch.device('cuda:0')
torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory)


def write(name,value):write_json(args.out/name,value)


def progress(stage,**extra):
    if time.monotonic()-began>p['resources']['internal_seconds']:raise TimeoutError('Registered one frozen candidate hard readback time exhausted')
    for key,limit in p['operation_budget'].items():
        if counts[key]>limit:raise RuntimeError('Registered candidate count exceeded: '+key)
    write('PROGRESS.json',dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,elapsed_seconds=time.monotonic()-began,execution_counts=dict(counts),**extra))


def memory_metrics(means):
    means=means.to(device)
    avg=means.mean(0);metrics={key:float(value) for key,value in memory.losses(avg,avg).items()}
    native=memory.sections['native_joint'];raw=memory.joint.cholesky@(avg-memory.target)[native]
    variance=memory.joint.cholesky.square().sum(1)
    for name,selection in memory.joint.block_slices.items():metrics['block_'+name]=float((raw[selection].square()/variance[selection]).mean())
    metrics.update(zip(memory.wave_cell_names,[float(v) for v in memory.conditional_wave_risks(*means)]))
    real_mass=memory.target[memory.sections['Q_mass']];mass=avg[memory.sections['Q_mass']]
    s=memory.sections['PSD_CDF'];width=memory.threshold_count
    residual=real_mass[:,None]*(avg[s].reshape(-1,width)/mass[:,None]-memory.target[s].reshape(-1,width)/real_mass[:,None])
    metrics['fixed_REAL_conditional_PSD']=float((memory.weights[s].reshape(-1,width)*residual.square()).sum())
    return metrics


def independent_metrics(wave,real,state,label):
    started=time.monotonic();stats={key:float(state['loss:'+key]) for key in ['peak_floor','q_floor','q_reference']}
    fv,rv=wave_views(wave,stats['peak_floor']),wave_views(real,stats['peak_floor'])
    f={name:(v-state['loss:'+name+'_center'])/state['loss:'+name+'_scale'] for name,v in fv.items()}
    r={name:(v-state['loss:'+name+'_center'])/state['loss:'+name+'_scale'] for name,v in rv.items()}
    del fv,rv
    vectors=[];weights=[];slices={};offset=0
    def columns(x,y,name,weight,category):
        nonlocal offset
        values=[];tick=time.monotonic()
        for first in range(0,x.shape[1],16):
            cache=CDFCache.create(x[:,first:first+16],y[:,first:first+16])
            mass=np.r_[np.full(len(x),1/len(x)),np.full(len(y),-1/len(y))]
            cumulative=mass[cache.order];np.cumsum(cumulative,axis=1,out=cumulative)
            values.extend((abs(cumulative[:,:-1])*cache.gap).sum(1))
            progress('independent_CDF_columns',case=label,group=name,completed_group_columns=min(first+16,x.shape[1]))
        values=np.array(values);vectors.append(values);weights.append(np.full(len(values),weight/len(values)))
        slices[name]=[offset,offset+len(values)];offset+=len(values);counts[category]+=len(values)
        row=dict(case=label,group=name,columns=len(values),seconds=time.monotonic()-tick)
        with (args.out/'CDF_GROUPS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    for name in VIEWS:
        d=state['projection:wave:'+name].astype(np.float64);columns(f[name]@d,r[name]@d,'swd:'+name,.5/6,'independent_wave_CDF_columns')
    for name in COORDINATE_VIEWS:
        columns(f[name],r[name],'coordinate:'+name,.25/2,'independent_wave_CDF_columns')
        for lag in LAGS:
            d=state[f'projection:lag:{name}:{lag}'].astype(np.float64)
            x=(f[name][:,:-lag,None]*d[0]+f[name][:,lag:,None]*d[1]).reshape(len(wave),-1)
            y=(r[name][:,:-lag,None]*d[0]+r[name][:,lag:,None]*d[1]).reshape(len(real),-1)
            columns(x,y,f'lag:{name}:{lag}',.25/8,'independent_wave_CDF_columns')
    wave_columns=np.concatenate(vectors);wave_weights=np.concatenate(weights);wave_slices=slices.copy()
    del f,r,x,y
    fr,_=q_features(wave,stats);rr,_=q_features(real,stats)
    fq=(fr-state['loss:qpsd_center'])/state['loss:qpsd_scale'];rq=(rr-state['loss:qpsd_center'])/state['loss:qpsd_scale']
    fb=np.searchsorted(state['loss:charge_bin_boundaries'],fr[:,0],side='left');rb=np.searchsorted(state['loss:charge_bin_boundaries'],rr[:,0],side='left')
    vectors=[];weights=[];slices={};offset=0
    columns(fq,rq,'QPSD:coordinate',.25,'independent_QPSD_CDF_columns')
    d=state['projection:qpsd'].astype(np.float64);columns(fq@d,rq@d,'QPSD:projected',.25,'independent_QPSD_CDF_columns')
    for index in range(8):
        fm,rm=fb==index,rb==index
        assert fm.sum()>=16 and rm.sum()>=16
        columns(fq[fm,1:2],rq[rm,1:2],f'QPSD:bin:{index}',.5/8,'independent_QPSD_CDF_columns')
    q=np.concatenate(vectors);qw=np.concatenate(weights)
    np.savez_compressed(args.out/f'INDEPENDENT_CDF_{label}.npz',wave_columns=wave_columns,wave_weights=wave_weights,QPSD_columns=q,QPSD_weights=qw)
    counts['independent_CDF_cases']+=1
    return dict(wave=float(wave_columns@wave_weights),QPSD=float(q@qw),wave_slices=wave_slices,QPSD_slices=slices,seconds=time.monotonic()-started)


try:
    write('FROZEN_POINT_READBACK_STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),parameter_fits=0,candidate_selections=0,source_fit_job=p['source_fit_job'],study_id=p['study_id'],trial_id=p['trial_id'],study_coefficient_fits_this_trial=0,study_candidate_choices_this_trial=0,no_new_optimizer_updates=True))
    for name,item in {**p['active_inputs'],**p['frozen_source_pins'],**p['new_source_pins']}.items():
        data=Path(name).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
    fit_result=json.loads(Path(p['source_fit_result']).read_bytes());counts['source_fit_result_loads']=1
    assert fit_result['status']=='COMPLETE_ONE_ACTUAL_FINITE_NATIVE_FIT' and fit_result['eligible_for_one_native_readback']
    assert fit_result['job_number']==p['source_fit_job'] and fit_result['parameter_fits']==1 and fit_result['candidate_selections']==1
    with np.load(p['source_memory'],allow_pickle=False) as z:saved={key:z[key].copy() for key in z.files}
    with np.load(p['source_parameters'],allow_pickle=False) as z:
        old_theta=z['original_theta'].copy();fit=z['SOURCE_FIT_rows'].copy();real_fit=z['REAL_FIT_rows'].copy()
    with np.load(p['source_joint_solution'],allow_pickle=False) as z:
        joint_coefficients=z['primal'].copy();steps=z['coordinate_steps'].copy();predicted=z['native']@joint_coefficients
    assert len(joint_coefficients)==163 and np.sum(abs(joint_coefficients))<=1+1e-8
    with np.load(p['source_mean_tangent'],allow_pickle=False) as z:original_head=z['initial_head'].copy()
    with np.load(p['source_additive_head'],allow_pickle=False) as z:head_saved={key:z[key].copy() for key in z.files}
    np.testing.assert_array_equal(head_saved['source_FIT_rows'],fit);np.testing.assert_array_equal(head_saved['REAL_FIT_rows'],real_fit);np.testing.assert_array_equal(head_saved['original_theta'],old_theta)
    head_delta=joint_coefficients[:161]*steps[:161];NN_coefficients=joint_coefficients[161:]*steps[161:]
    point=old_theta.copy();frozen=np.arange(54);active=np.empty(0,dtype=np.int64)
    np.savez_compressed(args.out/'FIXED_CANDIDATE_POINT.npz',theta=point,original_theta=old_theta,NN_coefficients=NN_coefficients,active_coordinates=active,frozen_coordinates=frozen,
        head_delta=head_delta,original_head=original_head,joint_coefficients=joint_coefficients,coordinate_steps=steps,predicted_relative_changes=predicted,factor=np.array(1.),source_NN_head_SHA256=np.array(p['active_inputs'][p['source_additive_head']]['sha256']))
    counts['candidate_parameter_assignments']=1;counts['temporary_mean_head_points']=1;counts['candidate_array_writes']=1
    write('POINT_FROZEN.json',dict(UTC=datetime.now(timezone.utc).isoformat(),status='ONE_RETURNED_ACTUAL_HARD_FIT_ENDPOINT_FIXED_BEFORE_NATIVE_READBACK',factor=1,source_fit_job=p['source_fit_job'],predicted_relative_changes=predicted,
        source_point_SHA256=hashlib.sha256((args.out/'FIXED_CANDIDATE_POINT.npz').read_bytes()).hexdigest(),parameter_fits=0,candidate_selections=0,study_id=p['study_id'],trial_id=p['trial_id'],study_head_fits_this_trial=0,study_coefficient_calibrations_this_trial=0,study_candidate_choices_this_trial=0,scientific_acceptance=False,endpoint_adopted=False))
    with np.load(p['original_readout'],allow_pickle=False) as z:original={key:z[key].copy() for key in z.files}
    with np.load(p['metric_state'],allow_pickle=False) as z:state={key:z[key].copy() for key in z.files}
    definition=json.loads(Path(p['metric_definition']).read_bytes())
    native=dict(original=json.loads(Path(p['original_native_metrics']).read_bytes()))
    with np.load(p['original_CDF'],allow_pickle=False) as z:
        independent=dict(original=dict(wave=float(z['wave_columns']@z['wave_weights']),QPSD=float(z['QPSD_columns']@z['QPSD_weights'])))
    counts.update(source_memory_loads=1,source_parameter_array_loads=1,source_joint_array_loads=1,source_mean_tangent_loads=1,source_additive_head_array_loads=1,saved_readout_array_loads=1,metric_state_loads=1,reused_original_native_cases=1,reopened_original_CDF_cases=1)
    generated=[];indices_all=[];means={key:torch.zeros(2,35331,dtype=torch.float64) for key in ['original','candidate']}
    flags={key:[Counter(),Counter()] for key in means}
    with torch.no_grad():
        setup=load_joint_setup(p,device,counts);response=setup['joint_models'][0];q_floor=float(setup['loss'].q_floor)
        real_cpu=observe_existing_carrier(torch.as_tensor(setup['data']['real_carrier384'][real_fit]),q_floor=q_floor,return_carrier=True)
        real={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in real_cpu.items()};real['flags']={key:value.to(device) for key,value in real_cpu['flags'].items()}
        memory=GlobalDistributionMemory01(real,q_floor=q_floor,**p['memory'])
        target_error=float(abs(memory.target.cpu().numpy()-saved['target']).max())
        cholesky_error=float(abs(memory.joint.cholesky.cpu().numpy()-saved['joint_cholesky']).max())
        np.testing.assert_allclose(memory.target.cpu().numpy(),saved['target'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(memory.joint.cholesky.cpu().numpy(),saved['joint_cholesky'],atol=1e-12,rtol=0)
        for key in ['target','weights','wave_denominator']:getattr(memory,key).copy_(torch.as_tensor(saved[key],device=device))
        memory.joint.cholesky.copy_(torch.as_tensor(saved['joint_cholesky'],device=device))
        theta0=torch.as_tensor(old_theta,device=device);theta1=torch.as_tensor(point,device=device)
        enc=setup['encoding'];baseline=setup['baseline'];centered=enc['source']-enc['source'][:,:100].mean(1,keepdim=True);unit=centered/centered.amax(1,keepdim=True)
        filtered=torch.cat([baseline.response_filter(unit[start:start+128,None]) for start in range(0,len(unit),128)]);counts['mean_filter_parent_rows']=6593
        mean=MeanHeadRepair01(baseline,enc,filtered);mean.mean_time.requires_grad_(False);mean.mean_head=mean.mean_head.double().requires_grad_(False)
        np.testing.assert_array_equal(torch.cat((mean.mean_head.weight.flatten(),mean.mean_head.bias)).cpu().numpy(),original_head)
        head_shift=torch.as_tensor(head_delta,device=device);point_A=response.cache['A'].clone();max_mean_Q_error=0.;max_mean_RMS=0.;fit_t=torch.as_tensor(fit,device=device)
        mean_radii=response.cache['A'][fit_t].norm(dim=1);np.testing.assert_allclose(mean_radii.cpu().numpy(),head_saved['mean_radii'],atol=1e-12,rtol=0)
        weights={key:torch.as_tensor(head_saved[key],device=device) for key in ['conv0_weight','conv0_bias','conv2_weight','conv2_bias','input_scale','target_scale','gate']};NN_live=[]
        for first in range(0,len(fit),128):
            rows=fit_t[first:first+128];value=enc['decoded'][rows].double()/weights['input_scale'][None,:,None]
            value=torch.nn.functional.silu(torch.nn.functional.conv1d(value,weights['conv0_weight'],weights['conv0_bias'],padding=2));value=torch.nn.functional.conv1d(value,weights['conv2_weight'],weights['conv2_bias'],padding=2)
            value*=weights['gate'][None,None]*weights['target_scale'][None,:,None];NN_live.append(value);counts['NN_head_live_inference_rows']+=len(rows);counts['NN_head_live_inference_calls']+=1
        NN_fields=torch.cat(NN_live).permute(1,0,2);live_error=float(abs(NN_fields.cpu().numpy()-head_saved['direction_fields']).max());assert live_error<1e-12
        NN_delta=mean_radii[:,None]*torch.einsum('d,dpt->pt',torch.as_tensor(NN_coefficients*head_saved['coordinate_steps'],device=device),NN_fields)
        for first in range(0,len(fit),128):
            rows=torch.as_tensor(fit[first:first+128],device=device);decoded=enc['decoded'][rows].double();logits=mean.mean_head(decoded)
            residual_logits=torch.nn.functional.conv1d(decoded,head_shift[:160].reshape(1,32,5),head_shift[160:],padding=2)
            scale=torch.exp(np.log(2)*mean.mean_time(mean.scalars[rows]).squeeze(1).tanh().double());warped,_=relative_time_warp008(filtered[rows],scale)
            raw0=warped*mean.gate+.20*logits.tanh()*mean.gate;raw1=warped*mean.gate+.20*(logits+residual_logits).tanh()*mean.gate
            charge0=canonical_charge008(raw0);charge1=canonical_charge008(raw1);assert bool((charge0>1e-6).all() and (charge1>1e-6).all())
            qbar=enc['original']['qbar'][rows,None,None];anchor=(qbar*raw0/charge0[...,None])[:,0];changed=(qbar*raw1/charge1[...,None])[:,0]
            point_A[rows]=response.cache['A'][rows]+(changed-anchor)+NN_delta[first:first+len(rows)]
            expected_Q=canonical_charge008(response.cache['A'][rows])+canonical_charge008(NN_delta[first:first+len(rows)]);assert bool((canonical_charge008(point_A[rows])>1e-6).all())
            max_mean_Q_error=max(max_mean_Q_error,float(abs(canonical_charge008(point_A[rows])-expected_Q).max()))
            max_mean_RMS=max(max_mean_RMS,float(((point_A[rows]-response.cache['A'][rows]).square().mean(1)/response.cache['A'][rows].square().mean(1)).sqrt().max()))
            counts['mean_head_anchor_parent_rows']+=len(rows);counts['mean_head_delta_parent_rows']+=len(rows);counts['NN_additive_parent_rows']+=len(rows)
        assert max_mean_Q_error<1e-12 and torch.equal(point_A[:,:100],response.cache['A'][:,:100])
        assert max_mean_RMS<=.01+1e-12
        point_response=JointResponse01(dict(response.cache,A=point_A))
        write('MEAN_POINT_SEMANTICS.json',dict(status='PASS_ORIGINAL_SHAPE_FUNCTION_DIFFERENCE_PLUS_FIXED_ADDITIVE_NN_AND_PEAK_CACHE',canonical_mean_Q_expected_arithmetic_error_max=max_mean_Q_error,mean_relative_RMS_max=max_mean_RMS,mean_prefix_exact=True,
            all54_response_coordinates_frozen=True,frozen_response_coordinates=frozen,NN_additive_coefficients=NN_coefficients,NN_live_field_error=live_error,canonical_mean_charge_intentionally_variable=True,stochastic_L_and_B_scales_frozen=True,original_peak_cache_fixed=True,parameter_fits=0,candidate_selections=0))
        for bank_index in range(2):
            for chunk_index,parents in enumerate(torch.arange(6593,device=device).split(128)):
                bank=random_bank(setup,parents,8,p['full_population_seed']+100000*bank_index+1000*chunk_index,counts)
                take=np.flatnonzero(np.isin(bank['rows'].cpu().numpy(),fit))
                for first in range(0,len(take),256):
                    choice=take[first:first+256];ii=bank_index*52744+chunk_index*1024+choice
                    rows,z,noise=bank['rows'][choice],bank['z'][choice],bank['noise']['fast'][choice]
                    a=response(theta0,rows,z,return_details=True);b=point_response(theta1,rows,z,return_details=True)
                    assert torch.equal(a['queries'],b['queries']);counts['time_query_exact_replay_rows']+=len(rows)
                    for label,details,current in [('original',a,theta0),('candidate',b,theta1)]:
                        obs=observe_generated_carrier(details['signal']+response.noise_scale(current)*noise,q_floor=q_floor,return_carrier=True)
                        counts['response_rows']+=len(rows);counts['hard_observation_rows']+=len(rows)
                        means[label][bank_index]+=memory.sums(obs).cpu();counts['memory_feature_rows']+=len(rows)
                        flags[label][bank_index].update({key:int(value.sum()) for key,value in obs['flags'].items()})
                        arrays=dict(window_adc_codes=((obs['carrier_adc'].gather(1,obs['crop_indices'])+.5)*16383).round().cpu().numpy().astype(np.uint16),
                            prefix_adc_codes=((obs['baseline100']+.5)*16383).round().cpu().numpy().astype(np.uint16),baseline_mean_V=obs['baseline_mean_V'].cpu().numpy(),
                            crop_start=obs['crop_start'].cpu().numpy(),**{'flag_'+key:value.cpu().numpy() for key,value in obs['flags'].items()})
                        if label=='original':
                            for key,value in arrays.items():np.testing.assert_array_equal(value,original[key][ii])
                            counts['center_exact_replayed_observation_rows']+=len(rows)
                        else:
                            np.testing.assert_array_equal(arrays['prefix_adc_codes'],original['prefix_adc_codes'][ii]);np.testing.assert_array_equal(arrays['baseline_mean_V'],original['baseline_mean_V'][ii])
                            generated.append(arrays);indices_all.append(ii)
                counts['paired_chunks']+=1;progress('fixed_joint_mean_point_chunk_complete',completed_chunks=counts['paired_chunks'],total_chunks=104)
        for value in means.values():value/=51720
        np.testing.assert_allclose(means['original'].numpy(),saved['original'],atol=1e-12,rtol=0)
        means['inherited']=torch.as_tensor(saved['inherited'])
        records={label:dict(metrics=memory_metrics(value),metrics_by_bank=[memory_metrics(torch.stack((v,v))) for v in value]) for label,value in means.items()}
        for label in ['original','candidate']:records[label]['flags']=[dict(v) for v in flags[label]]
        supported=[name for name,ok in zip(memory.wave_cell_names,saved['wave_supported']) if ok]
        common={}
        for bank_index in range(2):
            origin=records['original']['metrics_by_bank'][bank_index];new=records['candidate']['metrics_by_bank'][bank_index];old=records['inherited']['metrics_by_bank'][bank_index]
            for record in [origin,new]:record['local_excess_penalty']=float(np.mean([max(record[name]/(1.02*old[name])-1,0)**2 for name in supported]))
            for name in p['risk_names']:common[f'bank{bank_index}/{name}']=new[name]<=origin[name]+1e-10*max(1.,abs(origin[name]))
            common[f'bank{bank_index}/flags']=all(records['candidate']['flags'][bank_index][key]<=records['original']['flags'][bank_index][key] for key in records['original']['flags'][bank_index])
        write('PRE_HARD_PROTECTION.json',dict(status='ACTUAL_NATIVE_FEATURE_GUARDS_BEFORE_COMPLETE_HARD_READBACK',common_checks=common,all_original18_and_flags_pass=all(common.values()),records=records,parameter_fits=0,candidate_selections=0,scientific_acceptance=False))
        arrays={key:np.concatenate([v[key] for v in generated]) for key in generated[0]};indices_all=np.concatenate(indices_all)
        with np.load(p['source_fit_cached_readout'],allow_pickle=False) as z:
            assert set(z.files)==set(arrays)|{'global_generated_indices'}
            np.testing.assert_array_equal(indices_all,z['global_generated_indices'])
            for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
        counts['fit_cache_readout_loads']=1;counts['fit_cache_exact_rows']=len(indices_all)
        write('FIT_CACHE_NATIVE_REPLAY.json',dict(status='PASS_ALL_FROZEN_FIT_ADC_PREFIX_BASELINE_CROP_FLAGS_AND_INDICES_EXACT',rows=len(indices_all),all_arrays=list(arrays),source_fit_job=p['source_fit_job'],parameter_fits=0,candidate_selections=0))
        np.savez_compressed(args.out/'CANDIDATE_HARD_READOUT.npz',**arrays,global_generated_indices=indices_all)
        np.savez_compressed(args.out/'MEMORY_AUDIT_ARRAYS.npz',**{key:value.numpy() for key,value in means.items()},**{key:saved[key] for key in ['target','weights','joint_cholesky','wave_denominator','wave_supported']})
        counts['candidate_readout_writes']=1;counts['memory_array_writes']=1;counts['candidate_point_evaluations']=1
        peak_GPU=torch.cuda.max_memory_reserved(device)
    write('START_REPLAY.json',dict(status='PASS_M0_FULL_FEATURES_ADC_PREFIX_BASELINE_CROP_ALL_FLAGS_AND_SAME_TIME_QUERIES',max_abs_feature_error=float(abs(means['original'].numpy()-saved['original']).max()),
        max_abs_target_error=target_error,max_abs_cholesky_error=cholesky_error,exact_M0_rows=103440,time_queries_exact_rows=103440,canonical_mean_Q_expected_arithmetic_error_max=max_mean_Q_error,mean_relative_RMS_max=max_mean_RMS,parameter_fits=0,candidate_selections=0))
    del setup,response,point_response,point_A,mean,memory,real,bank,a,b,details,obs,enc,baseline,filtered,decoded,logits,residual_logits,warped,raw0,raw1,anchor,changed,NN_live,NN_fields,NN_delta,weights,value,mean_radii,fit_t
    torch.cuda.empty_cache()
    with np.load(p['REAL_readout'],allow_pickle=False) as z:
        real_wave=z['window_adc_codes'][real_fit].astype(np.float64)/16383-.5-z['baseline_mean_V'][real_fit,None]
        real_prefix=z['prefix_adc_codes'][real_fit].astype(np.float64)/16383-.5
    np.testing.assert_array_equal(real_wave,real_cpu['window_V'].numpy());np.testing.assert_array_equal(real_prefix,real_cpu['baseline100'].numpy())
    loss=NativeMetricLoss036(loss_tensor_statistics(definition['statistics']),definition['integral_statistics'],seed=2026101429,evaluation_seed=definition['projection_context']['seed'],integral_seed=2026101431)
    loss.load_state_dict({key[5:]:torch.from_numpy(value) for key,value in state.items() if key.startswith('loss:')},strict=True)
    context=ProjectionContext('evaluation',definition['projection_context']['seed'],None,{key[11:]:torch.from_numpy(value) for key,value in state.items() if key.startswith('projection:')})
    assert context.metadata()==definition['projection_context']
    wave=arrays['window_adc_codes'].astype(np.float64)/16383-.5-arrays['baseline_mean_V'][:,None];prefix=arrays['prefix_adc_codes'].astype(np.float64)/16383-.5
    progress('native_full_hard_start',case='candidate');tick=time.monotonic()
    native['candidate']=hard_metrics(wave,prefix,real_wave,real_prefix,loss,context);counts['native_full_hard_cases']=1
    write('NATIVE_candidate.json',native['candidate'])
    with (args.out/'HARD_CASES.jsonl').open('a') as stream:stream.write(json.dumps(dict(case='candidate',seconds=time.monotonic()-tick))+'\n')
    progress('independent_full_CDF_start',case='candidate');independent['candidate']=independent_metrics(wave,real_wave,state,'candidate')
    independent['original']['wave_slices']=independent['candidate']['wave_slices']
    for label in ['original','candidate']:
        for key in ['wave','QPSD']:assert abs(independent[label][key]-native[label]['values'][key])<1e-9
    changes={key:native['candidate']['values'][key]/native['original']['values'][key]-1 for key in ['wave','QPSD','noise']}
    for key in ['wave','QPSD']:assert abs(changes[key]-fit_result['actual_relative_changes'][key])<1e-9
    frozen_tolerances=json.loads(Path(p['tolerance_file']).read_bytes());write('FROZEN_TOLERANCES.json',frozen_tolerances);guards={}
    for key,value in frozen_tolerances['T_by_hypothesis']['fast'].items():
        if value is not None and value>0:
            new,old=native['candidate']['values'][key],native['original']['values'][key]
            guards[key]=dict(status='undefined' if new is None or old is None else 'defined',threshold=value,noninferior=None if new is None or old is None else new-old<=value)
    protected=all(common.values()) and all(v['status']=='defined' and v['noninferior'] for v in guards.values())
    native_common=changes['wave']<0 and changes['QPSD']<0 and protected
    write('NATIVE_CANDIDATE_RESULT.json',dict(status='COMPLETE_ONE_FIXED_SHAPE_ADDITIVE_CANDIDATE_NATIVE_READBACK',records=records,common_checks=common,native=native,independent=independent,
        native_relative_changes=changes,predicted_relative_changes=predicted,original_calibrated_guard_checks=guards,native_common_step_pass=native_common,
        original_native_wave_useful_2percent=changes['wave']<=-.02,original_entry_conditions_satisfied=changes['wave']<=-.02 and changes['QPSD']<=0 and protected,
        parameter_fits=0,candidate_selections=0,optimizer_updates=0,accepted_response_updates=0,REPORT_rows_used=0,scientific_acceptance=False,endpoint_adopted=False,
        diagnostic_only=False,fresh_dispatch_authorized=False,study_id=p['study_id'],trial_id=p['trial_id'],canonical_mean_Q_expected_arithmetic_error_max=max_mean_Q_error,mean_relative_RMS_max=max_mean_RMS))
    progress('complete');write('RESULT.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-began,execution_counts=dict(counts),config_sha256=args.config_sha256,
        peak_GPU_reserved_bytes=peak_GPU,scientific_acceptance=False,NAS_checks=False,NAS_hashes=False))
    assert sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())<p['resources']['output_MiB']*1024**2
    write('COMPLETE.json',dict(status='COMPLETE'))
finally:
    error=sys.exc_info()[1]
    if error is not None:write('FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(counts),elapsed_seconds=time.monotonic()-began,automatic_restart=False))
