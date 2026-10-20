"""Check only the two fixed KKT-certified restoration candidates with the frozen original hard gates."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,os,sys,time
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
LABELS=('FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3')

def run(p,setup,out,counts,started):
    device=setup['source'].device
    with np.load(ROOT/p['roles_file'],allow_pickle=False) as z:roles=dict(z)
    np.savez(out/'REAL_CALIBRATION_ROLES.npz',**roles)
    real_pool=torch.as_tensor(roles['real_train_rows'],device=device)
    cpu=observe_existing_carrier(setup['data']['real_carrier384'],already_digitized=True,q_floor=float(setup['loss'].q_floor),return_carrier=True,return_indices=False)
    for key in ('window_V','baseline100'):np.testing.assert_array_equal(cpu[key],setup['real'][key].cpu().numpy())
    real=dict(**{key:torch.as_tensor(cpu[key],device=device) for key in REAL_KEYS},flags={key:torch.as_tensor(value,device=device) for key,value in cpu['flags'].items()})
    counts['REAL_observable_replay_calls']+=1;counts['REAL_observable_replay_rows']+=len(real['Q'])
    del setup['real'],cpu
    blocks=observable_blocks(select_real(real,real_pool));maps={};contexts={}
    for label in ('FIT0','FIT1','FIT2'):
        feature=JointFeatureMap01(blocks,seed=p['feature_seeds'][label])
        maps[label]={h:feature for h in HYPOTHESES}
        contexts[label]=setup['loss'].create_evaluation_context(seed=p['native_projection_seeds'][label])
        counts['feature_maps_constructed']+=1;counts['new_native_projection_contexts']+=1
    del blocks
    summaries={}
    for seed,model in enumerate(setup['joint_models']):
        with np.load(ROOT/p['saved_arrays'][seed],allow_pickle=False) as z:saved=dict(z)
        previous=torch.as_tensor(saved['theta'],device=device)
        center=torch.as_tensor(saved['curvature_update_control'],device=device)
        with np.load(ROOT/p['candidate_arrays'][seed],allow_pickle=False) as proposal_file:
            np.testing.assert_array_equal(proposal_file['center'],saved['curvature_update_control'])
            proposal=torch.as_tensor(proposal_file['candidate'],device=device)
        assert proposal.shape==center.shape and proposal[53]==center[53]
        assert float(torch.linalg.vector_norm(proposal[:53]-center[:53]))<=.003125+1e-10
        counts['fixed_parameter_candidates']+=1
        summaries[str(seed)]={}
        for label in LABELS:
            check_budget(p,counts,started)
            map_label='FIT0' if label.startswith('IID') else label
            parents=torch.as_tensor(saved[label+'_source_parents'],device=device)
            rows=torch.as_tensor(saved[label+'_real_rows'],device=device)
            np.testing.assert_array_equal(maps[map_label][HYPOTHESES[0]].cholesky.cpu().numpy(),saved[label+'_cholesky'])
            assert np.isin(parents.cpu().numpy(),roles['source_train_rows']).all() and np.isin(rows.cpu().numpy(),roles['real_train_rows']).all()
            bank_seed=int(saved[label+'_random_seed']);bank0=random_bank(setup,parents,32,bank_seed,counts)
            bank=dict(rows=bank0['rows'].cpu(),z=bank0['z'].cpu(),noise={h:bank0['noise'][h].cpu() for h in HYPOTHESES});del bank0
            target=select_real(real,rows);features=feature_means({h:target for h in HYPOTHESES},maps[map_label])
            counts['REAL_fit_feature_rows']+=len(rows)
            local=dict(setup,context=contexts[map_label])
            original=prediction(model,previous,bank,setup['loss'].q_floor,counts)
            means=feature_means(original,maps[map_label]);original_r=residual_vector(means,features,maps[map_label])
            original_n=native_values(original,target,local,counts);original_risk=risks(means,features,maps[map_label])
            baseline={h:dict(carrier=original[h]['carrier_input'].cpu(),flags={key:int(value.sum()) for key,value in original[h]['flags'].items()}) for h in HYPOTHESES}
            del original
            for key,value in [('baseline_residual',original_r),('baseline_native',original_n)]:
                np.testing.assert_allclose(value.cpu().numpy(),saved[label+'_'+key],rtol=1e-12,atol=1e-12)
            counts['saved_previous_center_replays_verified']+=1
            anchor=torch.as_tensor(saved[label+'_initial_native'],device=device);counts['saved_initial_anchors_reused']+=1
            r,n,control_check=hard_check(p,model,center,bank,target,features,maps[map_label],baseline,original_risk,original_n,anchor,local,counts)
            for key,value in [('residual',r),('native',n)]:
                np.testing.assert_allclose(value.cpu().numpy(),saved['curvature_update_control_'+label+'_actual_'+key],rtol=1e-12,atol=1e-12)
            counts['saved_new_center_replays_verified']+=1;counts['fixed_control_checks']+=1
            arrays=dict(previous_theta=previous.cpu().numpy(),theta=center.cpu().numpy(),initial=saved['initial'],lower=saved['lower'],upper=saved['upper'],
                source_parents=parents.cpu().numpy(),real_rows=rows.cpu().numpy(),random_seed=np.array(bank_seed,dtype=np.int64),
                cholesky=saved[label+'_cholesky'],previous_residual=original_r.cpu().numpy(),previous_native=original_n.cpu().numpy(),
                initial_native=anchor.cpu().numpy(),baseline_residual=r.cpu().numpy(),baseline_native=n.cpu().numpy())
            report=dict(seed=seed,label=label,projection_context=contexts[map_label].metadata(),original_risks=original_risk,
                center_check=control_check,fixed_candidate=True,initial_anchor_reused_from_pinned_output=True,noise_coordinate_frozen=True,
                shape_latent_sha256=hashlib.sha256(bank['z'].numpy().tobytes()).hexdigest(),
                noise_sha256={h:hashlib.sha256(value.numpy().tobytes()).hexdigest() for h,value in bank['noise'].items()})
            candidate_r,candidate_n,candidate_check=hard_check(p,model,proposal,bank,target,features,maps[map_label],baseline,original_risk,original_n,anchor,local,counts)
            counts['fixed_candidate_library_checks']+=1
            arrays.update(candidate_theta=proposal.cpu().numpy(),candidate_residual=candidate_r.cpu().numpy(),candidate_native=candidate_n.cpu().numpy())
            report['candidate_check']=candidate_check
            stem=f'seed{seed}_{label}_FIXED_CANDIDATE'
            np.savez(out/(stem+'.npz'),**arrays);write_json(out/(stem+'.json'),report)
            summaries[str(seed)][label]=dict(candidate_accepted=candidate_check['accepted'],control_accepted=control_check['accepted'])
            progress_file(out,counts,'fixed_candidate_library_complete',seed=seed,role=label,native_timing=dict(NATIVE_TIMING))
            del bank,target,features,baseline
    return dict(status='COMPLETE',summaries=summaries,native_timing=dict(NATIVE_TIMING),scientific_acceptance=False,
        committed_parameter_updates=0,new_parameter_candidates=2,automatic_training=False,all_fixed_candidate_gates_pass=all(v['candidate_accepted'] for by_seed in summaries.values() for v in by_seed.values()),
        scope='Fixed two-candidate hard-readout diagnostic on seven already exposed libraries per head; no new solve or committed update.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256:raise ValueError('Registered fixed candidate configuration changed')
    p=json.loads(raw)
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned fixed candidate input changed: '+name)
    certificate=json.loads((ROOT/p['restoration_evidence_file']).read_bytes())
    assert certificate['two_seed_local_feasible'] and certificate['status']=='PASS_INDEPENDENT_FRESH_RESTORATION_MATRIX_AND_CERTIFICATE_AUDIT'
    args.out.mkdir(parents=True,exist_ok=False);counts,started=Counter(),time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device);torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    try:
        with torch.no_grad():
            setup=load_joint_setup(p,device,counts)
            for key in ('encoding','references','models','baseline','heads'):del setup[key]
            progress_file(args.out,counts,'fixed_candidate_setup_loaded');result=run(p,setup,args.out,counts,started)
        check_budget(p,counts,started)
        result.update(config_sha256=args.config_sha256,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out/'RESULT.json',result)
        if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file())>p['resources']['output_MiB']*1024**2:raise ValueError('Registered output limit exceeded')
        write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error=sys.exc_info()[1]
        if error is not None:write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,scientific_acceptance=False,automatic_restart=False))
