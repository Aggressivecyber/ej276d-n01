"""Fixed goal-scaled joint/peak/native candidate versus native-gain control; no updates."""
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
from run_joint_response01_teacher import check_budget,draw_parents

ROOT=Path(__file__).resolve().parents[1]
LABELS=('FIT0','FIT1','FIT2','CHECK','FRESH')

def run(p,setup,out,counts,started):
    device=setup['source'].device
    with np.load(ROOT/p['roles_file'],allow_pickle=False) as z:roles={k:z[k].copy() for k in z.files}
    np.savez(out/'REAL_CALIBRATION_ROLES.npz',**roles)
    source_pool,real_pool=[torch.as_tensor(roles[k],device=device) for k in ('source_train_rows','real_train_rows')]
    cpu=observe_existing_carrier(setup['data']['real_carrier384'],already_digitized=True,q_floor=float(setup['loss'].q_floor),return_carrier=True,return_indices=False)
    for key in ('window_V','baseline100'):np.testing.assert_array_equal(cpu[key],setup['real'][key].cpu().numpy())
    real=dict(**{key:torch.as_tensor(cpu[key],device=device) for key in REAL_KEYS},flags={key:torch.as_tensor(v,device=device) for key,v in cpu['flags'].items()})
    counts['REAL_observable_replay_calls']+=1;counts['REAL_observable_replay_rows']+=len(real['Q'])
    del setup['real'],cpu
    blocks=observable_blocks(select_real(real,real_pool));maps={};contexts={}
    for label in LABELS:
        fmap=JointFeatureMap01(blocks,seed=p['feature_seeds'][label]);maps[label]={h:fmap for h in HYPOTHESES}
        contexts[label]=setup['loss'].create_evaluation_context(seed=p['native_projection_seeds'][label])
        counts['feature_maps_constructed']+=1;counts['new_native_projection_contexts']+=1
    del blocks
    with np.load(ROOT/p['fixed_proposals'],allow_pickle=False) as z:proposals={k:z[k].copy() for k in z.files}
    summaries={}
    for seed,model in enumerate(setup['joint_models']):
        with np.load(ROOT/p['saved_arrays'][seed],allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
        theta=torch.as_tensor(saved['theta'],device=device);initial=model.initial()
        np.testing.assert_array_equal(saved['endpoint'],saved['theta'])
        with np.load(ROOT/p['control_arrays'][seed],allow_pickle=False) as z:control={k:z[k].copy() for k in z.files}
        candidates=dict(native_gain_control=torch.as_tensor(control['native_gain'],device=device),
            scientific_progress=torch.as_tensor(proposals[f'seed{seed}_scientific_progress'],device=device))
        arrays=dict(theta=saved['theta'],initial=initial.cpu().numpy(),lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy(),
            **{name:value.cpu().numpy() for name,value in candidates.items()})
        report=dict(seed=seed,baseline_risks={},baseline_native={},initial_native={},checks={name:{} for name in candidates},
            projection_contexts={label:contexts[label].metadata() for label in LABELS},replay_errors={})
        for label in LABELS:
            check_budget(p,counts,started)
            if label=='FRESH':
                rs=p['fresh_seed']+100000*seed
                parents=draw_parents(source_pool,1024,rs+100);rows=draw_parents(real_pool,4096,rs+200);bank_seed=rs+1000
            else:
                parents=torch.as_tensor(saved[label+'_source_parents'],device=device);rows=torch.as_tensor(saved[label+'_real_rows'],device=device)
                bank_seed=int(saved[label+'_random_seed'])
                np.testing.assert_array_equal(maps[label][HYPOTHESES[0]].cholesky.cpu().numpy(),saved[label+'_cholesky'])
            assert np.isin(parents.cpu().numpy(),roles['source_train_rows']).all() and np.isin(rows.cpu().numpy(),roles['real_train_rows']).all()
            bank0=random_bank(setup,parents,32,bank_seed,counts)
            bank=dict(rows=bank0['rows'].cpu(),z=bank0['z'].cpu(),noise={h:bank0['noise'][h].cpu() for h in HYPOTHESES});del bank0
            target=select_real(real,rows);features=feature_means({h:target for h in HYPOTHESES},maps[label])
            counts['REAL_fit_feature_rows' if label.startswith('FIT') else 'REAL_CHECK_feature_rows']+=len(rows)
            local=dict(setup,context=contexts[label]);current=prediction(model,theta,bank,setup['loss'].q_floor,counts)
            means=feature_means(current,maps[label]);old_risks=risks(means,features,maps[label]);r=residual_vector(means,features,maps[label]);n=native_values(current,target,local,counts)
            baseline={h:dict(carrier=current[h]['carrier_input'].cpu(),flags={key:int(v.sum()) for key,v in current[h]['flags'].items()}) for h in HYPOTHESES}
            del current
            anchor_observation=prediction(model,initial,bank,setup['loss'].q_floor,counts);anchor=native_values(anchor_observation,target,local,counts);del anchor_observation
            for key,value in dict(source_parents=parents,real_rows=rows,cholesky=maps[label][HYPOTHESES[0]].cholesky,baseline_residual=r,baseline_native=n,initial_native=anchor).items():arrays[label+'_'+key]=value.cpu().numpy()
            arrays[label+'_random_seed']=np.array(bank_seed,dtype=np.int64)
            report['baseline_risks'][label]=old_risks;report['baseline_native'][label]=n.tolist();report['initial_native'][label]=anchor.tolist()
            if label!='FRESH':
                errors={}
                for key in ('baseline_residual','baseline_native','initial_native'):
                    np.testing.assert_allclose(arrays[label+'_'+key],saved[label+'_'+key],rtol=1e-12,atol=1e-12)
                    errors[key]=float(np.max(np.abs(arrays[label+'_'+key]-saved[label+'_'+key])))
                report['replay_errors'][label]=errors
                counts['saved_center_replays_verified']+=1;counts['saved_anchor_replays_verified']+=1
            for name,candidate in candidates.items():
                ar,an,check=hard_check(p,model,candidate,bank,target,features,maps[label],baseline,old_risks,n,anchor,local,counts)
                arrays[name+'_'+label+'_actual_residual']=ar.cpu().numpy();arrays[name+'_'+label+'_actual_native']=an.cpu().numpy()
                report['checks'][name][label]=check;counts['fixed_candidate_checks']+=1
                if label!='FRESH' and name=='native_gain_control':
                    for suffix,value in [('residual',ar),('native',an)]:np.testing.assert_allclose(value.cpu().numpy(),control['native_gain_'+label+'_actual_'+suffix],rtol=1e-12,atol=1e-12)
                    counts['saved_control_candidate_replays_verified']+=1
            np.savez(out/f'seed{seed}_SCIENTIFIC_PROGRESS.npz',**arrays);write_json(out/f'seed{seed}_SCIENTIFIC_PROGRESS.json',report)
            progress_file(out,counts,'scientific_progress_fixed_candidate_checks',seed=seed,role=label,native_timing=dict(NATIVE_TIMING))
            del bank,baseline,target,features
        report['all_local_gates']={name:all(check['accepted'] for check in checks.values()) for name,checks in report['checks'].items()}
        write_json(out/f'seed{seed}_SCIENTIFIC_PROGRESS.json',report)
        summaries[str(seed)]=report['all_local_gates']
    return dict(status='COMPLETE',summaries=summaries,two_seed_scientific_progress_all_local_pass=all(v['scientific_progress'] for v in summaries.values()),
        committed_parameter_updates=0,scientific_acceptance=False,automatic_training=False,original_final_gates_unchanged=True,
        native_timing=dict(NATIVE_TIMING),scope='Fixed FIT-derived candidate comparison with original hard observation and new fitting-role FRESH sample and projections.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256:raise ValueError('Registered scientific-progress diagnostic config changed')
    p=json.loads(raw)
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned scientific-progress diagnostic input changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False);counts,started=Counter(),time.monotonic();torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    try:
        with torch.no_grad():
            setup=load_joint_setup(p,device,counts)
            for key in ('encoding','references','models','baseline','heads'):del setup[key]
            progress_file(args.out,counts,'scientific_progress_setup_loaded');result=run(p,setup,args.out,counts,started)
        check_budget(p,counts,started)
        result.update(config_sha256=args.config_sha256,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out/'RESULT.json',result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())>p['resources']['output_MiB']*1024**2:raise ValueError('Registered diagnostic output bound exceeded')
        write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error=sys.exc_info()[1]
        if error is not None:write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,committed_parameter_updates=0,scientific_acceptance=False,automatic_restart=False))
