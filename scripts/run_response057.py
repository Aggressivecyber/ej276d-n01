"""Bounded input-Q-conditioned positive core-area redistribution on frozen054."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import torch
import run_response039 as r
from run_response042 import local_restitution042
from run_response052 import conditional_sides052,evaluate052
from run_response054 import constraint_vector054
from run_response044 import readouts044
from wp1_evaluation import summary,output_q_bins

ROOT=Path(__file__).resolve().parents[1]


def core_redistribution057(mean,input_q,coefficients,q_reference):
    """Tilt only a smooth nonnegative part of the core, preserving its exact area."""
    t=torch.arange(110,142,device=mean.device,dtype=torch.float64)
    peak=mean[:,112:144].argmax(1)+112
    taper=torch.sin(torch.pi*(t-110.)/31.).square();taper[0]=0.;taper[-1]=0.
    movable=mean[:,110:142].clamp_min(0.)*taper
    area=movable.sum(1)
    if not bool((area>0.).all()):raise RuntimeError('057 no positive movable core area; no row may be dropped')
    coefficients=torch.as_tensor(coefficients,device=mean.device,dtype=torch.float64)
    u=coefficients[0]+coefficients[1]*torch.log(input_q/float(q_reference))
    if not bool(torch.isfinite(u).all()) or bool((u.abs()>.5+1e-12).any()):raise RuntimeError('057 fixed log-tilt domain exceeded')
    weight=torch.exp(-.5*((t[None,:]-peak[:,None])/3.).square())
    if bool((coefficients==0.).all()):
        normalization=torch.ones_like(area);delta=torch.zeros_like(mean)
    else:
        unnormalized=movable*torch.exp(u[:,None]*weight)
        normalization=area/unnormalized.sum(1)
        local=unnormalized*normalization[:,None]-movable
        delta=torch.cat((torch.zeros_like(mean[:,:110]),local,torch.zeros_like(mean[:,142:])),1)
    adapted=mean+delta
    rows=dict(input_Q_Vns=input_q,log_tilt=u,movable_area_Vns=area,normalization=normalization,
        old_mean_peak_index=peak,new_mean_peak_index=adapted[:,112:144].argmax(1)+112,
        old_mean_peak_voltage_V=mean.gather(1,peak[:,None]).squeeze(1),
        new_mean_peak_voltage_V=adapted[:,112:144].max(1).values,
        transferred_absolute_area_Vns=delta.abs().sum(1),maximum_absolute_delta_V=delta.abs().max(1).values,
        full_integral_residual_Vns=delta.sum(1),canonical_Q_residual_Vns=r.canonical_charge008(delta),
        tail_from142_residual_Vns=delta[:,142:].sum(1),
        negative_area_change_Vns=(-adapted).clamp_min(0.).sum(1)-(-mean).clamp_min(0.).sum(1))
    return delta,rows


@torch.no_grad()
def evaluate057(setup,index,bank,T,projection,real_joint,p,out,progress,coefficients,label,save_features=False):
    model=setup['models'][index];device=setup['source'].device
    model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],device=device,dtype=model.mean_adapter[-1].bias.dtype))
    progress['mean_operator_calls']+=1;progress['mean_operator_rows']+=len(setup['source'])
    adapter=model._mean_response(setup['encoding'],enabled=True)
    delta,_,_=local_restitution042(adapter['delta_mean_pre_V'],p['base_restitution_fraction'])
    A0=setup['encoding']['original']['signal_pre'];q=r.canonical_charge008(A0)
    transport,_,_,_,_,_=conditional_sides052(A0+delta,q,1.,p['input_Q_reference_Vns'],p['side_coefficients_ns'])
    fixed_delta=delta+transport;fixed_mean=A0+fixed_delta
    correction,rows=core_redistribution057(fixed_mean,q,coefficients,p['input_Q_reference_Vns'])
    delta=fixed_delta+correction;mean=A0+delta
    signal=r._ExactZeroMeanAddition015.apply(bank['M0'],delta)
    geometry=dict(coefficients=coefficients,maximum_absolute_log_tilt=float(rows['log_tilt'].abs().max()),
        core_support_and_endpoints_exact=bool((correction[:,:111]==0.).all() and (correction[:,141:]==0.).all()),
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
    if not all(geometry[k] for k in ('core_support_and_endpoints_exact','frozen_prefix_bitwise_unchanged','mean_tail_from142_bitwise_unchanged','nonpositive_mean_samples_unchanged')):
        raise RuntimeError('057 exact support/prefix/tail/nonpositive invariant failed')
    if geometry['mean_negative_area_max_change_Vns']!=0. or max(geometry['maximum_full_integral_residual_Vns'],geometry['maximum_correction_canonical_Q_residual_Vns'])>1e-12 or geometry['maximum_canonical_Q_residual_Vns']>1e-10:
        raise RuntimeError('057 area or negative-part invariant failed')
    if not bool(torch.isfinite(mean).all()):raise RuntimeError('057 nonfinite mean')
    np.savez_compressed(out/(label+'_GEOMETRY_ROWS.npz'),parent_row=np.arange(len(q)),**{k:z.cpu().numpy() for k,z in rows.items()})
    np.savez_compressed(out/(label+'_EXAMPLES.npz'),parent_row=np.arange(32),input_Q_Vns=q[:32].cpu().numpy(),
        fixed_mean_pre_V=fixed_mean[:32].cpu().numpy(),adapted_mean_pre_V=mean[:32].cpu().numpy(),correction_pre_V=correction[:32].cpu().numpy())
    progress['geometry_records']+=1;progress['geometry_rows']+=len(q)
    reports={}
    for h in r.HYPOTHESES:
        obs=r.observe_generated_carrier(signal+bank['noise'][h],ste=False,q_floor=float(setup['loss'].q_floor),return_indices=True)
        progress['generated_observation_rows']+=len(signal)
        wave,baseline,peaks=[obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
        report=r.hard_metrics(wave,baseline,setup['data']['real'],setup['data']['real_baseline100'],setup['loss'],setup['context'])
        z=r.joint_embedding(wave,baseline,peaks,dict(T[h],q_floor=float(setup['loss'].q_floor)))
        report['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=signal.device)@projection,real_joint[h],16))
        report['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),np.maximum(-setup['data']['real'],0).sum(1))['value']
        report['observation_flags']={k:dict(count=int(v.sum()),denominator=len(signal)) for k,v in obs['flags'].items()}
        progress['full_native_hypothesis_evaluations']+=1;reports[h]=report
        if save_features:
            features,flags=readouts044(wave)
            np.savez_compressed(out/(label+'_'+h+'_FEATURE_ROWS.npz'),parent_row=np.arange(len(q)),input_Q_Vns=q.cpu().numpy(),
                peak_index=peaks,native_output_Q_bin=output_q_bins(features['Q_Vns'],setup['loss']),**features,
                **{k+'_flag':z for k,z in flags.items()})
            progress['feature_records']+=1;progress['feature_rows']+=len(q)
    return reports,geometry


def solve057(base,probes,upper,labels,contract):
    """One local minimax morphology solve, using original guards and useful primary targets."""
    J=((probes[:,:,1]-probes[:,:,0])/2).transpose(0,2,1)
    curvature=(probes[:,:,1]+probes[:,:,0]-2*base[:,None,:]).transpose(0,2,1)
    limits=upper.copy()
    for seed in (0,1):
        for i,label in enumerate(labels[seed]):
            if label['family']=='values' and label['metric'] in ('wave','joint_morphology'):
                limits[seed,i]=-contract['useful_wave_relative_gain' if label['metric']=='wave' else 'useful_joint_relative_gain']
    A=np.column_stack((J.reshape(-1,2),np.zeros(J.shape[0]*J.shape[1])));b=(limits-base).ravel();targets=[]
    for seed in (0,1):
        for h in r.HYPOTHESES:
            indices=[i for i,z in enumerate(labels[seed]) if z['family']=='values' and z['hypothesis']==h and z['metric'] in r.MORPH]
            A=np.vstack((A,np.r_[J[seed,indices].mean(0),1.]));b=np.r_[b,-base[seed,indices].mean()]
            targets.append(dict(seed=seed,hypothesis=h,row_indices=indices,base_morphology_gain_T=float(-base[seed,indices].mean())))
    c=np.array([0.,0.,-1.]);z=linprog(c,A_ub=A,b_ub=b,bounds=[(-1.,1.)]*2+[(0.,None)],method='highs',
        options=dict(time_limit=10.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not z.success:raise RuntimeError('057 bounded LP failed: '+str(z.status)+' '+z.message)
    x=z.x;y=z.ineqlin.marginals;lo=z.lower.marginals;hi=z.upper.marginals;slack=b-A@x
    primal=max(0.,float(-slack.min()),float(np.max(np.abs(x[:2])-1)),-x[2])
    stationarity=float(np.max(np.abs(c-A.T@y-lo-hi)))
    comp=max(float(np.max(np.abs(slack*y))),float(np.max(np.abs(lo*np.r_[x[:2]+1,x[2]]))),float(np.max(np.abs(hi[:2]*(1-x[:2])))))
    dual=float(b@y-lo[:2].sum()+hi[:2].sum());gap=float(c@x-dual)
    if max(primal,stationarity,comp,abs(gap))>1e-8:raise RuntimeError('057 LP certificate failed')
    return dict(jacobian=J,central_second_difference=curvature,positive_one_sided=(probes[:,:,1]-base[:,None,:]).transpose(0,2,1),
        negative_one_sided=(base[:,None,:]-probes[:,:,0]).transpose(0,2,1),scaled_base=base,scaled_probes=probes,
        original_scaled_upper=upper,point_target_scaled_upper=limits,A_ub=A,b_ub=b,c=c,x=x,normalized_coefficients=x[:2],
        predicted_worst_morphology_gain_T=float(x[2]),base_worst_morphology_gain_T=min(t['base_morphology_gain_T'] for t in targets),
        predicted_scaled_endpoint=base+np.einsum('smj,j->sm',J,x[:2]),morphology_targets=targets,solver_status=int(z.status),solver_message=z.message,
        iterations=int(z.nit),inequality_slack=slack,inequality_marginals=y,lower_marginals=lo,upper_marginals=hi,
        KKT=dict(primal_residual=primal,stationarity_residual=stationarity,complementarity_residual=comp,duality_gap=gap),one_solver_call=True)


@torch.no_grad()
def run057(p,out,device,progress):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],config_sha256=p['_config_sha256'],pid=os.getpid(),resources=p['resources']))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    q=r.canonical_charge008(setup['encoding']['original']['signal_pre']).cpu().numpy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=p['input_Q_array_sha256']:raise RuntimeError('057 original full input-Q array differs')
    L=float(np.max(np.abs(np.log(q/p['input_Q_reference_Vns']))));scale=np.asarray(p['coefficient_scales'])
    if scale.tolist()!=[.25,.25/L]:raise RuntimeError('057 frozen coefficient scales differ from input-Q bound')
    real_wave,real_baseline=setup['data']['real'],setup['data']['real_baseline100'];projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real_wave,real_baseline,np.full(len(real_wave),16),dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device)@projection for h in r.HYPOTHESES}
    replay=json.loads((ROOT/p['replay_file']).read_text())['reports'];banks=[];references=[];bases=[];vectors=[];probes=[];limits=[];labels=[]
    for index in (0,1):
        r.check_resources(p,out,started);bank=r.make_population_bank(setup,index,'FIT',0,p,progress);banks.append(bank)
        m0,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
        path=out/f'seed{index}_M0.json';r.write_json(path,m0)
        if json.loads(path.read_text())!=replay[str(index)]['M0']:raise RuntimeError('057 M0 full replay differs')
        references.append(m0)
        base,geo=evaluate057(setup,index,bank,T,projection,real_joint,p,out,progress,[0.,0.],f'seed{index}_BASE054',True)
        path=out/f'seed{index}_BASE054.json';r.write_json(path,dict(metrics=base,geometry=geo,decision=r.objective039(base,m0,T,p['contract'])))
        if json.loads(path.read_text())['metrics']!=replay[str(index)]['ENDPOINT054']:raise RuntimeError('057 zero redistribution does not exactly replay054')
        bases.append(base);value,upper,row_labels=constraint_vector054(base,m0,T,p['contract']);vectors.append(value);limits.append(upper);labels.append(row_labels);head_probes=[]
        for j in range(2):
            directions=[]
            for sign in (-1,1):
                r.check_resources(p,out,started);coeff=np.zeros(2);coeff[j]=sign*scale[j]
                label=f'seed{index}_PROBE_{j}_'+('neg' if sign<0 else 'pos')
                report,geo=evaluate057(setup,index,bank,T,projection,real_joint,p,out,progress,coeff.tolist(),label)
                progress['finite_difference_probes']+=1;v,u,l=constraint_vector054(report,m0,T,p['contract'])
                if not np.array_equal(u,upper) or l!=row_labels:raise RuntimeError('057 constraint schema changed')
                directions.append(v)
                r.write_json(out/(label+'.json'),dict(seed_index=index,coordinate=j,sign=sign,coefficients=coeff,scaled_constraints=v,decision=r.objective039(report,m0,T,p['contract']),geometry=geo,metrics=report))
                r.write_json(out/'PROGRESS.json',dict(phase='core_redistribution_fixed_probes',seed_index=index,coordinate=j,sign=sign,elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
            head_probes.append(directions)
        probes.append(head_probes)
    r.check_resources(p,out,started);progress['linear_program_solves']+=1
    solution=solve057(np.asarray(vectors),np.asarray(probes),np.asarray(limits),labels,p['contract'])
    r.write_json(out/'LINEARIZATION_AND_SOLUTION.json',dict(**solution,constraint_row_labels_by_seed=labels,coefficient_scales=scale,original_contract=p['contract'],T_by_hypothesis=T))
    positive=solution['predicted_worst_morphology_gain_T']>solution['base_worst_morphology_gain_T']+1e-8
    coeff=scale*solution['normalized_coefficients']
    endpoint=dict(coefficients=coeff.tolist(),predicted_worst_morphology_gain_T=solution['predicted_worst_morphology_gain_T'],base_worst_morphology_gain_T=solution['base_worst_morphology_gain_T']) if positive else None
    r.write_json(out/'ENDPOINT_FROZEN.json',dict(UTC=datetime.now(timezone.utc).isoformat(),endpoint=endpoint,no_reselection=True,no_additional_probes=True,confirmation_banks=0))
    results=[]
    if positive:
        for index in (0,1):
            r.check_resources(p,out,started)
            report,geo=evaluate057(setup,index,banks[index],T,projection,real_joint,p,out,progress,coeff.tolist(),f'seed{index}_ENDPOINT',True)
            progress['hard_endpoint_head_evaluations']+=1;decision=r.objective039(report,references[index],T,p['contract'])
            actual,_,_=constraint_vector054(report,references[index],T,p['contract']);useful={}
            for h in r.HYPOTHESES:
                a,b=report[h]['values'],references[index][h]['values']
                wave=(b['wave']-a['wave'])/b['wave'];joint=(b['joint_morphology']-a['joint_morphology'])/b['joint_morphology']
                morphology=float(np.mean([(b[k]-a[k])/T[h][k] for k in r.MORPH]))
                useful[h]=dict(wave_gain=wave,joint_gain=joint,morphology_gain_T=morphology,passes_point_targets=wave>=p['contract']['useful_wave_relative_gain'] and joint>=p['contract']['useful_joint_relative_gain'] and morphology>=p['contract']['useful_morphology_mean_T_gain'])
            record=dict(seed_index=index,decision=decision,point_usefulness=useful,actual_scaled_constraints=actual,predicted_scaled_constraints=solution['predicted_scaled_endpoint'][index],prediction_residual=actual-solution['predicted_scaled_endpoint'][index],geometry=geo,metrics=report)
            r.write_json(out/f'seed{index}_ENDPOINT.json',record);results.append({k:v for k,v in record.items() if k not in ('metrics','geometry')})
            r.write_json(out/'PROGRESS.json',dict(phase='one_frozen_core_endpoint',seed_index=index,elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
    expected=dict(p['operation_bounds'])
    if not positive:expected.update(mean_operator_calls=12,mean_operator_rows=79116,generated_observation_rows=237348,full_native_hypothesis_evaluations=36,geometry_records=10,geometry_rows=65930,hard_endpoint_head_evaluations=0,feature_records=6,feature_rows=39558)
    if dict(progress)!=expected:raise RuntimeError('057 exact operation counts differ from contract')
    r.check_resources(p,out,started)
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]):raise RuntimeError('057 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']:raise RuntimeError('057 A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('057 005 changed')
    passed=positive and all(z['decision']['feasible'] and all(v['passes_point_targets'] for v in z['point_usefulness'].values()) for z in results)
    result=dict(schema='n01/response057-result/v1',run_id=p['run_id'],endpoint=endpoint,results=results,stage_outcome='FIT_USEFUL_MEAN_PENDING_CONFIRMATION' if passed else 'CLOSED_NEGATIVE_HARD_ENDPOINT' if positive else 'CLOSED_NEGATIVE_LOCAL_LINEAR_MODEL_NO_GAIN',
        original_point_targets_passed=passed,scientific_status='response_match_not_established',mean_stage_accepted=False,full_random_response_acceptance=False,
        confirmation_banks_opened=0,DEV_opened=False,all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,hardware_calibrated=False,
        uncertainty_scope='Same TRAIN/FIT banks calibrated; no independent confirmation')
    r.write_json(out/'RESULT.json',result);r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('057 config hash mismatch')
    p=json.loads(args.config.read_text())
    for n,digest in p['source_pins'].items():
        if r.sha256(ROOT/n)!=digest:raise ValueError('057 source pin mismatch: '+n)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']:raise ValueError('057 setup hash mismatch')
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True);device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(('checkpoint_loads','TRAIN_archives_decoded','PCA_archives_decoded','DEV_archives_decoded','AD_queries','optimizer_steps',
        'old_A_calls','old_A_rows','response_calls','response_rows','noise005_calls','noise005_rows','extension_rows','full_population_banks',
        'mean_operator_calls','mean_operator_rows','generated_observation_rows','full_native_hypothesis_evaluations','geometry_records','geometry_rows',
        'finite_difference_probes','linear_program_solves','hard_endpoint_head_evaluations','feature_records','feature_rows'),0))
    try:run057(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
