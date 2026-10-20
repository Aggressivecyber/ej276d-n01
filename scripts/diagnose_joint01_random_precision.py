"""Paired IID/antithetic latent precision at fixed endpoints; no directions or updates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from n01.data import write_json
from n01.joint01_features import JointFeatureMap01,observable_blocks
from n01.observation008 import observe_existing_carrier
from diagnose_joint01_original_draw_precision_fix01 import prediction,native_values,hard_check,NATIVE_TIMING
from run_joint_response01 import load_joint_setup,progress_file,random_bank,feature_means,residual_vector,risks,HYPOTHESES
from run_joint_response01_real import select_real,REAL_KEYS
from run_joint_response01_teacher import check_budget

ROOT=Path(__file__).resolve().parents[1]


def run(p,setup,out,counts,started):
    device=setup['source'].device
    with np.load(ROOT/p['roles_file'],allow_pickle=False) as z:roles={k:z[k].copy() for k in z.files}
    np.savez(out/'REAL_CALIBRATION_ROLES.npz',**roles)
    real_pool=torch.as_tensor(roles['real_train_rows'],device=device)
    cpu=observe_existing_carrier(setup['data']['real_carrier384'],already_digitized=True,q_floor=float(setup['loss'].q_floor),return_carrier=True,return_indices=False)
    for key in ('window_V','baseline100'):np.testing.assert_array_equal(cpu[key],setup['real'][key].cpu().numpy())
    real=dict(**{key:torch.as_tensor(cpu[key],device=device) for key in REAL_KEYS},flags={key:torch.as_tensor(v,device=device) for key,v in cpu['flags'].items()})
    counts['REAL_observable_replay_calls']+=1;counts['REAL_observable_replay_rows']+=len(real['Q'])
    del setup['real'],cpu
    blocks=observable_blocks(select_real(real,real_pool));fmap=JointFeatureMap01(blocks,seed=p['feature_seed']);maps={h:fmap for h in HYPOTHESES};del blocks
    context=setup['loss'].create_evaluation_context(seed=p['native_projection_seed']);local=dict(setup,context=context)
    counts['feature_maps_constructed']+=1;counts['new_native_projection_contexts']+=1
    all_geometry={};all_errors={}
    for seed,model in enumerate(setup['joint_models']):
        with np.load(ROOT/p['saved_arrays'][seed],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
        with np.load(ROOT/p['fixed_candidates'][seed],allow_pickle=False) as z:fixed={k:z[k].copy() for k in z.files}
        theta=torch.as_tensor(saved['theta'],device=device);initial=model.initial()
        candidates={name:torch.as_tensor(fixed[key],device=device) for name,key in p['candidate_keys'].items()}
        parents=torch.as_tensor(saved['FIT0_source_parents'],device=device);rows=torch.as_tensor(saved['FIT0_real_rows'],device=device)
        assert np.isin(parents.cpu().numpy(),roles['source_train_rows']).all() and np.isin(rows.cpu().numpy(),roles['real_train_rows']).all()
        np.testing.assert_array_equal(fmap.cholesky.cpu().numpy(),saved['FIT0_cholesky'])
        target=select_real(real,rows);features=feature_means({h:target for h in HYPOTHESES},maps)
        counts['REAL_fit_feature_rows']+=len(rows)
        geometry={scheme:[] for scheme in ('IID','ANTITHETIC_Z')};errors={scheme:[] for scheme in geometry}
        for replicate in range(-1,p['replicates']):
            bank_seed=int(saved['FIT0_random_seed']) if replicate==-1 else p['random_seed']+seed*p['head_seed_stride']+replicate*p['replicate_seed_stride']
            bank0=random_bank(setup,parents,32,bank_seed,counts)
            noise={h:bank0['noise'][h].cpu() for h in HYPOTHESES};bank_rows=bank0['rows'].cpu();iid_z=bank0['z'].cpu();del bank0
            noise_hashes={h:hashlib.sha256(v.numpy().tobytes()).hexdigest() for h,v in noise.items()}
            if replicate>=0:np.savez(out/f'seed{seed}_rep{replicate}_BANK.npz',rows=bank_rows.numpy(),IID_z=iid_z.numpy(),source_parents=parents.cpu().numpy(),real_rows=rows.cpu().numpy(),bank_seed=np.array(bank_seed,dtype=np.int64))
            schemes=('REPLAY',) if replicate==-1 else ('IID','ANTITHETIC_Z')
            for scheme in schemes:
                check_budget(p,counts,started)
                z=iid_z
                if scheme=='ANTITHETIC_Z':
                    z=iid_z.reshape(len(parents),32,8).clone();z[:,16:]=-z[:,:16];z=z.reshape(-1,8)
                    assert torch.equal(z.reshape(len(parents),32,8)[:,16:],-z.reshape(len(parents),32,8)[:,:16])
                    counts['derived_antithetic_banks']+=1
                bank=dict(rows=bank_rows,z=z,noise=noise)
                stem=f'seed{seed}_rep{replicate}_{scheme}'
                arrays=dict(theta=theta.cpu().numpy(),initial=initial.cpu().numpy(),lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy(),cholesky=fmap.cholesky.cpu().numpy(),source_parents=parents.cpu().numpy(),real_rows=rows.cpu().numpy(),random_seed=np.array(bank_seed,dtype=np.int64),**{name:value.cpu().numpy() for name,value in candidates.items()})
                report=dict(seed=seed,replicate=replicate,scheme=scheme,noise_hashes=noise_hashes,noise_shared_with_paired_scheme=replicate>=0,shape_latent_mean=z.reshape(len(parents),32,8).double().mean(1).mean(0).tolist(),maximum_parent_latent_mean_absolute=float(z.reshape(len(parents),32,8).double().mean(1).abs().max()),projection_context=context.metadata(),FD_columns_completed=0,checks={})
                current=prediction(model,theta,bank,setup['loss'].q_floor,counts);means=feature_means(current,maps);r=residual_vector(means,features,maps);risk=risks(means,features,maps);n=native_values(current,target,local,counts)
                baseline={h:dict(carrier=current[h]['carrier_input'].cpu(),flags={k:int(v.sum()) for k,v in current[h]['flags'].items()}) for h in HYPOTHESES};del current
                anchor_obs=prediction(model,initial,bank,setup['loss'].q_floor,counts);anchor=native_values(anchor_obs,target,local,counts);del anchor_obs
                arrays.update(baseline_residual=r.cpu().numpy(),baseline_native=n.cpu().numpy(),initial_native=anchor.cpu().numpy())
                report.update(baseline_risks=risk,baseline_native=n.tolist(),initial_native=anchor.tolist())
                if scheme=='REPLAY':
                    for key in ('baseline_residual','baseline_native','initial_native'):np.testing.assert_allclose(arrays[key],saved['FIT0_'+key],rtol=1e-12,atol=1e-12)
                    counts['saved_center_replays_verified']+=1;counts['saved_anchor_replays_verified']+=1
                else:
                    plus_r=[];minus_r=[];plus_n=[];minus_n=[];plus_theta=[];minus_theta=[]
                    for coordinate in range(model.parameter_count):
                        check_budget(p,counts,started)
                        plus,minus=theta.clone(),theta.clone()
                        plus[coordinate]=min(float(model.upper[coordinate]),float(theta[coordinate])+p['finite_difference_step'])
                        minus[coordinate]=max(float(model.lower[coordinate]),float(theta[coordinate])-p['finite_difference_step'])
                        rr=[];nn=[]
                        for proposal in (plus,minus):
                            obs=prediction(model,proposal,bank,setup['loss'].q_floor,counts);means=feature_means(obs,maps)
                            rr.append(residual_vector(means,features,maps));nn.append(native_values(obs,target,local,counts));del obs
                        plus_r.append(rr[0]);minus_r.append(rr[1]);plus_n.append(nn[0]);minus_n.append(nn[1]);plus_theta.append(plus);minus_theta.append(minus)
                        counts['finite_difference_columns']+=1;report['FD_columns_completed']=coordinate+1
                        if (coordinate+1)%9==0 or coordinate+1==model.parameter_count:
                            for key,values in dict(FD_plus_residual=plus_r,FD_minus_residual=minus_r,FD_plus_native=plus_n,FD_minus_native=minus_n,FD_plus_theta=plus_theta,FD_minus_theta=minus_theta).items():arrays[key]=torch.stack(values,1).cpu().numpy()
                            np.savez(out/(stem+'.npz'),**arrays);write_json(out/(stem+'.json'),report)
                            progress_file(out,counts,'random_precision_Jacobian',seed=seed,replicate=replicate,scheme=scheme,column=coordinate+1,native_timing=dict(NATIVE_TIMING))
                    difference=np.diag(arrays['FD_plus_theta'])-np.diag(arrays['FD_minus_theta'])
                    J=(arrays['FD_plus_residual']-arrays['FD_minus_residual'])/difference;N=(arrays['FD_plus_native']-arrays['FD_minus_native'])/difference
                    H=np.maximum((arrays['FD_plus_native']+arrays['FD_minus_native']-2*arrays['baseline_native'][:,None])/p['finite_difference_step']**2,0);H[:,-1]=0
                    arrays.update(jacobian=J,native_jacobian=N,positive_diagonal_native_curvature=H)
                    geometry[scheme].append(dict(J=J,N=N,n=arrays['baseline_native']))
                for name,candidate in candidates.items():
                    ar,an,check=hard_check(p,model,candidate,bank,target,features,maps,baseline,risk,n,anchor,local,counts)
                    arrays[name+'_actual_residual']=ar.cpu().numpy();arrays[name+'_actual_native']=an.cpu().numpy();report['checks'][name]=check;counts['fixed_candidate_checks']+=1
                    if scheme=='REPLAY':
                        key=p['candidate_keys'][name]
                        for suffix,value in [('residual',ar),('native',an)]:np.testing.assert_allclose(value.cpu().numpy(),fixed[key+'_FIT0_actual_'+suffix],rtol=1e-12,atol=1e-12)
                        counts['saved_control_candidate_replays_verified']+=1
                    else:
                        delta=arrays[name]-arrays['theta'];pr=arrays['baseline_residual']+J@delta;pn=arrays['baseline_native']+N@delta+.5*H@np.square(delta)
                        arrays[name+'_predicted_residual']=pr;arrays[name+'_predicted_native']=pn
                        error=(arrays[name+'_actual_native']-pn)/arrays['baseline_native'];errors[scheme].append(error.reshape(3,3)[:,:2])
                        report['checks'][name].update(linear_joint_vector_relative_error=float(np.linalg.norm(arrays[name+'_actual_residual']-pr)/np.linalg.norm(arrays['baseline_residual'])),diagonal_native_relative_error=error.reshape(3,3).tolist())
                np.savez(out/(stem+'.npz'),**arrays);write_json(out/(stem+'.json'),report)
                progress_file(out,counts,'random_precision_bank_complete',seed=seed,replicate=replicate,scheme=scheme,native_timing=dict(NATIVE_TIMING))
                del bank,baseline
            del noise,bank_rows,iid_z
        all_geometry[seed]=geometry;all_errors[seed]=errors
    summary={}
    for seed,geometry in all_geometry.items():
        native_scale=np.mean([v['n'] for values in geometry.values() for v in values],axis=0);by_scheme={}
        for scheme,values in geometry.items():
            jj=np.stack([v['J'] for v in values]);nn=np.stack([v['N']/native_scale[:,None] for v in values]).reshape(p['replicates'],3,3,54)[:,:,:2].reshape(p['replicates'],6,54)
            by_scheme[scheme]=dict(joint_derivative_dispersion=float(np.square(jj-jj.mean(0)).sum()/(p['replicates']-1)),native_wave_QPSD_derivative_dispersion=float(np.square(nn-nn.mean(0)).sum()/(p['replicates']-1)),mean_joint_J_norm=float(np.linalg.norm(jj.mean(0))),mean_native_N_norm=float(np.linalg.norm(nn.mean(0))),mean_absolute_native_Taylor_error=float(np.abs(all_errors[seed][scheme]).mean()))
        iid,anti=by_scheme['IID'],by_scheme['ANTITHETIC_Z']
        variance_ratios={key:anti[key]/iid[key] for key in ('joint_derivative_dispersion','native_wave_QPSD_derivative_dispersion')}
        norm_ratio=anti['mean_joint_J_norm']/iid['mean_joint_J_norm']
        supported=all(v<=p['diagnostic_gate']['maximum_variance_ratio'] for v in variance_ratios.values()) and p['diagnostic_gate']['mean_J_norm_ratio_low']<=norm_ratio<=p['diagnostic_gate']['mean_J_norm_ratio_high'] and anti['mean_absolute_native_Taylor_error']<=iid['mean_absolute_native_Taylor_error']
        summary[str(seed)]=dict(by_scheme=by_scheme,variance_ratios=variance_ratios,mean_J_norm_ratio=norm_ratio,diagnostic_gate_pass=bool(supported))
    return dict(status='COMPLETE',summaries=summary,two_head_precision_gate_pass=all(v['diagnostic_gate_pass'] for v in summary.values()),committed_parameter_updates=0,scientific_acceptance=False,automatic_training=False,new_parameter_candidates=0,native_timing=dict(NATIVE_TIMING),scope='Four paired random libraries per head at the frozen FIT0 records and projections; antithetic shape latents only, unchanged IID electronic noise. No generalization or unbiased empirical-distance claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256:raise ValueError('Registered random precision config changed')
    p=json.loads(raw)
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned random precision input changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False);counts,started=Counter(),time.monotonic();torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    try:
        with torch.no_grad():
            setup=load_joint_setup(p,device,counts)
            for key in ('encoding','references','models','baseline','heads'):del setup[key]
            progress_file(args.out,counts,'random_precision_setup_loaded');result=run(p,setup,args.out,counts,started)
        check_budget(p,counts,started);result.update(config_sha256=args.config_sha256,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device));write_json(args.out/'RESULT.json',result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())>p['resources']['output_MiB']*1024**2:raise ValueError('Registered random precision output bound exceeded')
        write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error=sys.exc_info()[1]
        if error is not None:write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,committed_parameter_updates=0,scientific_acceptance=False,automatic_restart=False))
