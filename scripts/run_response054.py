"""One bounded four-coefficient hard-metric linearization and one common endpoint."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys,time

import numpy as np
from scipy.optimize import linprog
import torch
import run_response039 as r
from run_response052 import evaluate052

ROOT=Path(__file__).resolve().parents[1]


def constraint_vector054(report,reference,tolerances,contract):
    """Original039 limits, in fixed numerical units; integer flags remain counts."""
    values=[];upper=[];labels=[]
    for h in r.HYPOTHESES:
        a,b=report[h]['values'],reference[h]['values']
        for key in ('wave','joint_morphology')+r.PROTECTED+r.MORPH+('negative_lobe_area_Vns',):
            if key in ('wave','joint_morphology'):
                scale=b[key];limit=contract['fit_primary_relative_worsening_max']
            elif key in r.PROTECTED+r.MORPH:
                scale=tolerances[h][key]
                limit=contract['protected_allowance_T'] if key in r.PROTECTED else contract['morphology_individual_allowance_T']
            else:
                scale=tolerances[h]['Q_Vns'];limit=0.
            values.append((a[key]-b[key])/scale);upper.append(limit)
            labels.append(dict(hypothesis=h,family='values',metric=key,scale=scale,M0_value=b[key],scaled_upper=limit))
        for family in ('fake_flags','observation_flags'):
            for key,row in report[h][family].items():
                values.append(row['count']-reference[h][family][key]['count']);upper.append(0.)
                labels.append(dict(hypothesis=h,family=family,metric=key,scale=1.,M0_value=reference[h][family][key]['count'],scaled_upper=0.))
        values.append(float(not report[h]['conditional_coverage']['conditional_coverage_complete']));upper.append(0.)
        labels.append(dict(hypothesis=h,family='coverage',metric='incomplete_output_Q_coverage',scale=1.,M0_value=0.,scaled_upper=0.))
    return np.asarray(values),np.asarray(upper),labels


def solve054(base,probes,upper,score_rows):
    """Maximize the smaller original039 score improvement, once in a fixed box."""
    # probes[seed, coordinate, sign(-,+), metric]
    jacobian=((probes[:,:,1]-probes[:,:,0])/2).transpose(0,2,1)
    curvature=(probes[:,:,1]+probes[:,:,0]-2*base[:,None,:]).transpose(0,2,1)
    guard_matrix=jacobian.reshape(-1,4)
    A=np.column_stack((guard_matrix,np.zeros(len(guard_matrix))))
    b=(upper-base).ravel()
    common=np.column_stack((jacobian[:,score_rows,:].mean(1),np.ones(len(base))))
    A=np.vstack((A,common));b=np.r_[b,np.zeros(len(base))]
    c=np.array([0.,0.,0.,0.,-1.])
    result=linprog(c,A_ub=A,b_ub=b,bounds=[(-1.,1.)]*4+[(0.,None)],method='highs',
        options=dict(time_limit=10.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    if not result.success:
        raise RuntimeError('054 bounded LP failed: '+str(result.status)+' '+result.message)
    primal=float(max(0.,np.max(A@result.x-b),np.max(np.abs(result.x[:4])-1),-result.x[4]))
    stationarity=c-A.T@result.ineqlin.marginals-result.lower.marginals-result.upper.marginals
    if primal>1e-8 or np.max(np.abs(stationarity))>1e-8:
        raise RuntimeError('054 LP primal/dual residual exceeds declared numerical tolerance')
    return dict(jacobian=jacobian,central_second_difference=curvature,
        positive_one_sided=(probes[:,:,1]-base[:,None,:]).transpose(0,2,1),
        negative_one_sided=(base[:,None,:]-probes[:,:,0]).transpose(0,2,1),
        scaled_base=base,scaled_upper=upper,scaled_probes=probes,A_ub=A,b_ub=b,c=c,
        x=result.x,normalized_coefficients=result.x[:4],predicted_common_score_gain=float(result.x[4]),
        predicted_scaled_endpoint=base+np.einsum('smj,j->sm',jacobian,result.x[:4]),
        solver_status=int(result.status),solver_message=result.message,iterations=int(result.nit),
        inequality_slack=result.ineqlin.residual,inequality_marginals=result.ineqlin.marginals,
        lower_marginals=result.lower.marginals,upper_marginals=result.upper.marginals,
        maximum_primal_violation=primal,maximum_stationarity_residual=float(np.max(np.abs(stationarity))),
        singular_values=np.linalg.svd(guard_matrix,compute_uv=False),one_solver_call=True)


@torch.no_grad()
def run054(p,out,device,progress):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    r.write_json(out/'STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),run_id=p['run_id'],
        config_sha256=p['_config_sha256'],pid=os.getpid(),resources=p['resources'],phase='setup'))
    setup=r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()),device,progress)
    for model in setup['models']:model.set_phase('frozen').eval()
    T=json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    q=r.canonical_charge008(setup['encoding']['original']['signal_pre']).cpu().numpy()
    if hashlib.sha256(q.tobytes()).hexdigest()!=p['input_Q_array_sha256']:
        raise RuntimeError('054 original full input-Q array differs')
    real_wave,real_baseline=setup['data']['real'],setup['data']['real_baseline100']
    projection=r.projection039(p['joint_projection_fit_seed'],device)
    real_joint={h:torch.as_tensor(r.joint_embedding(real_wave,real_baseline,np.full(len(real_wave),16),
        dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device)@projection for h in r.HYPOTHESES}
    banks=[];references=[];bases=[];base_vectors=[];probe_vectors=[];limits=[];row_labels=[];base_decisions=[]
    probe_scale=np.asarray(p['finite_difference_coefficient_scales_ns'])
    for index in range(2):
        bank=r.make_population_bank(setup,index,'FIT',0,p,progress);banks.append(bank)
        reference,_=evaluate052(setup,index,np.zeros(8),bank,T,projection,real_joint,p,out,progress)
        path=out/f'seed{index}_FIT_M0.json';r.write_json(path,reference)
        if json.loads(path.read_text())!=json.loads((ROOT/p['FIT_replay_files'][index]).read_text()):
            raise RuntimeError('054 full M0 report does not replay')
        references.append(reference)
        base,geometry=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,p,out,progress,
            restitution_fraction=p['base_restitution_fraction'],geometry_rows=f'seed{index}_BASE042')
        path=out/f'seed{index}_BASE042.json';r.write_json(path,base)
        if json.loads(path.read_text())!=json.loads((ROOT/p['base_replay_files'][index]).read_text()):
            raise RuntimeError('054 full BASE042 report does not replay')
        decision=r.objective039(base,reference,T,p['contract'])
        if not decision['feasible']:raise RuntimeError('054 registered base is not feasible')
        r.write_json(out/f'seed{index}_BASE_GEOMETRY.json',dict(geometry=geometry,decision=decision))
        bases.append(base);base_decisions.append(decision)
        value,upper,labels=constraint_vector054(base,reference,T,p['contract'])
        base_vectors.append(value);limits.append(upper);row_labels.append(labels)
        probes=[]
        for j in range(4):
            directions=[]
            for sign in (-1,1):
                r.check_resources(p,out,started)
                theta=np.zeros(4);theta[j]=sign*probe_scale[j]
                trial_p=dict(p,side_coefficients_ns=theta.tolist())
                label=f'seed{index}_PROBE_{j}_'+('neg' if sign<0 else 'pos')
                report,geo=evaluate052(setup,index,p['base_bias'],bank,T,projection,real_joint,trial_p,out,progress,
                    restitution_fraction=p['base_restitution_fraction'],step_size=1.,geometry_rows=label)
                progress['finite_difference_probes']+=1
                values,new_upper,new_labels=constraint_vector054(report,reference,T,p['contract'])
                if not np.array_equal(new_upper,upper) or new_labels!=labels:
                    raise RuntimeError('054 numerical constraint schema changed')
                directions.append(values)
                r.write_json(out/(label+'.json'),dict(seed_index=index,coordinate=j,sign=sign,side_coefficients_ns=theta,
                    scaled_constraints=values,decision=r.objective039(report,reference,T,p['contract']),geometry=geo,metrics=report))
                r.write_json(out/'PROGRESS.json',dict(phase='fixed_central_differences',seed_index=index,coordinate=j,sign=sign,
                    elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
            probes.append(directions)
        probe_vectors.append(probes)
    score_rows=[i for i,x in enumerate(row_labels[0]) if x['family']=='values' and x['metric'] in ('wave','joint_morphology')]
    r.check_resources(p,out,started)
    solution=solve054(np.asarray(base_vectors),np.asarray(probe_vectors),np.asarray(limits),score_rows)
    progress['linear_program_solves']+=1
    r.write_json(out/'LINEARIZATION_AND_SOLUTION.json',dict(**solution,constraint_row_labels_by_seed=row_labels,
        score_rows=score_rows,coefficient_scales_ns=probe_scale,original_contract=p['contract'],T_by_hypothesis=T))
    theta=probe_scale*solution['normalized_coefficients']
    positive=solution['predicted_common_score_gain']>1e-12
    endpoint=dict(side_coefficients_ns=theta.tolist(),normalized_coefficients=solution['normalized_coefficients'].tolist(),
        predicted_common_score_gain=solution['predicted_common_score_gain'],base_bias=p['base_bias'],
        restitution_fraction=p['base_restitution_fraction']) if positive else None
    r.write_json(out/'ENDPOINT_FROZEN.json',dict(UTC=datetime.now(timezone.utc).isoformat(),endpoint=endpoint,
        unopened_confirmation_banks=0,no_endpoint_reselection=True,no_additional_probe_or_scale=True))
    results=[]
    if positive:
        for index in range(2):
            r.check_resources(p,out,started)
            trial_p=dict(p,side_coefficients_ns=theta.tolist())
            report,geo=evaluate052(setup,index,p['base_bias'],banks[index],T,projection,real_joint,trial_p,out,progress,
                restitution_fraction=p['base_restitution_fraction'],step_size=1.,geometry_rows=f'seed{index}_ENDPOINT')
            progress['hard_endpoint_head_evaluations']+=1
            decision=r.objective039(report,references[index],T,p['contract'])
            actual,_,_=constraint_vector054(report,references[index],T,p['contract'])
            predicted=solution['predicted_scaled_endpoint'][index]
            record=dict(seed_index=index,decision=decision,BASE042_decision=base_decisions[index],
                actual_score_gain_vs_BASE042=base_decisions[index]['score']-decision['score'],
                actual_scaled_constraints=actual,predicted_scaled_constraints=predicted,prediction_residual=actual-predicted,
                geometry=geo,metrics=report)
            r.write_json(out/f'seed{index}_ENDPOINT.json',record)
            results.append({k:v for k,v in record.items() if k not in ('geometry','metrics')})
            r.write_json(out/'PROGRESS.json',dict(phase='single_frozen_endpoint',seed_index=index,
                elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
    accepted=positive and all(x['decision']['feasible'] and x['actual_score_gain_vs_BASE042']>0 for x in results)
    r.check_resources(p,out,started)
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]):raise RuntimeError('054 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']:raise RuntimeError('054 A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']:raise RuntimeError('054 005 changed')
    result=dict(schema='n01/response054-result/v1',run_id=p['run_id'],endpoint=endpoint,results=results,
        stage_outcome='FIT_COMMON_IMPROVEMENT_PENDING_CONFIRMATION' if accepted else
            'CLOSED_NEGATIVE_HARD_ENDPOINT' if positive else 'CLOSED_NEGATIVE_LOCAL_LINEAR_MODEL_NO_IMPROVEMENT',
        fit_common_improvement=accepted,scientific_status='response_match_not_established',full_random_response_acceptance=False,
        useful_mean_stage_not_established=True,confirmation_banks_opened=0,DEV_opened=False,
        all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,
        hardware_calibrated=False,uncertainty_scope='Same TRAIN/FIT banks used to calibrate; no independent confirmation',
        next_step='Verify every probe, LP primal/dual and hard endpoint; confirm only a feasible common improvement, otherwise analyze the limiting constraints without automatic expansion.')
    r.write_json(out/'RESULT.json',result)
    r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
        result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256:raise ValueError('054 config hash mismatch')
    p=json.loads(args.config.read_text())
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest:raise ValueError('054 source pin mismatch: '+name)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']:raise ValueError('054 setup hash mismatch')
    p['_config_sha256']=args.config_sha256;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(('checkpoint_loads','TRAIN_archives_decoded','PCA_archives_decoded','DEV_archives_decoded',
        'AD_queries','optimizer_steps','old_A_calls','old_A_rows','response_calls','response_rows','noise005_calls','noise005_rows',
        'extension_rows','full_population_banks','mean_operator_calls','mean_operator_rows','generated_observation_rows',
        'full_native_hypothesis_evaluations','geometry_records','geometry_rows','finite_difference_probes','linear_program_solves',
        'hard_endpoint_head_evaluations'),0))
    try:run054(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),
                automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__':main()
