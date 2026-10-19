"""Bounded high-precision joint calibration after independently repeated directions."""
from __future__ import annotations

import argparse
import ctypes
import resource
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.joint01_model_c1 import JointResponse01C1
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction, teacher_parameters, HYPOTHESES, feature_means, residual_vector, risks, conditional_report)
from run_joint_response01_teacher import hard_metrics
from run_joint_response01_teacher import check_budget, TeacherBudget01, recovery_decision, draw_parents, population_weights, accept_joint
from run_joint_response01_c1_teacher import precision_check
from run_joint_response01_c1_fullpool import cpu_chunks, pooled_mean, pooled_candidate

ROOT = Path(__file__).resolve().parents[1]
libc=ctypes.CDLL(None)
libc.malloc_trim.argtypes=[ctypes.c_size_t]
libc.malloc_trim.restype=ctypes.c_int


def fit_track_precision(p,setup,teacher,teacher_theta,model,initial,pool,maps,check_maps,
                fit_population,target_population,iterations,seed,out,counts,started,name):
    q_floor,theta=float(setup['loss'].q_floor),initial.clone()
    fit_weights,target_weights=population_weights(setup,fit_population),population_weights(setup,target_population)
    coverage=torch.zeros(len(setup['source']),dtype=torch.long,device=pool.device)
    target_coverage=torch.zeros_like(coverage)
    trace=[]
    for iteration in range(iterations):
        check_budget(p,counts,started)
        orders=[draw_parents(pool,len(pool),p['order_seed']+1000*seed+100*iteration+role) for role in range(4)]
        streams=[p['bank_seed']+1000000*seed+10000*iteration+1000*role for role in range(4)]
        # Hold only one full random bank at a time; the summation order is unchanged.
        target_bank=cpu_chunks(p,setup,orders[1],p['batch']['draws_per_parent'],streams[1],counts)
        target,_=pooled_mean(teacher,teacher_theta,target_bank,maps,target_weights,q_floor,counts)
        del target_bank
        libc.malloc_trim(0)
        fit=cpu_chunks(p,setup,orders[0],p['batch']['draws_per_parent'],streams[0],counts)
        predicted,_=pooled_mean(model,theta,fit,maps,fit_weights,q_floor,counts)
        residual=residual_vector(predicted,target,maps)
        arrays=dict(theta=theta.cpu().numpy().copy(),initial=initial.cpu().numpy(),lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy(),
            fit_parents=orders[0].cpu().numpy(),target_parents=orders[1].cpu().numpy(),check_parents=orders[2].cpu().numpy(),check_target_parents=orders[3].cpu().numpy(),
            bank_stream_base_seeds=np.array(streams,dtype=np.int64),residual=residual.cpu().numpy())
        columns,highs,lows,spans=[],[],[],[]
        for j in range(model.parameter_count):
            check_budget(p,counts,started)
            plus,minus=theta.clone(),theta.clone()
            plus[j]=min(float(model.upper[j]),float(theta[j])+p['solver']['finite_difference_step'])
            minus[j]=max(float(model.lower[j]),float(theta[j])-p['solver']['finite_difference_step'])
            high,_=pooled_mean(model,plus,fit,maps,fit_weights,q_floor,counts)
            low,_=pooled_mean(model,minus,fit,maps,fit_weights,q_floor,counts)
            high_vector,low_vector=residual_vector(high,target,maps),residual_vector(low,target,maps)
            span=float(plus[j]-minus[j])
            columns.append((high_vector-low_vector)/span)
            highs.append(high_vector.cpu().numpy());lows.append(low_vector.cpu().numpy());spans.append(span)
            counts['finite_difference_columns']+=1
            counts['fullpool_FIT_chunk_columns']+=len(fit)
            if (j+1)%9==0 or j+1==model.parameter_count:
                arrays.update(FD_high=np.stack(highs),FD_low=np.stack(lows),FD_parameter_spans=np.array(spans))
                np.savez(out/f'{name}_ITER{iteration:02d}.npz',**arrays)
                progress_file(out,counts,'precision_pilot_FD',track=name,iteration=iteration+1,column=j+1,parameters=model.parameter_count)
        J=torch.stack(columns,1)
        counts['joint_direction_solves']+=1
        delta,geometry=joint_step(residual,J,ridge_fraction=p['solver']['ridge_fraction'],radius=p['solver']['radius'])
        arrays.update(jacobian=J.cpu().numpy(),delta=delta.cpu().numpy())
        del fit,high,low,columns,highs,lows
        libc.malloc_trim(0)
        check_target_bank=cpu_chunks(p,setup,orders[3],p['batch']['draws_per_parent'],streams[3],counts)
        check_target,_=pooled_mean(teacher,teacher_theta,check_target_bank,check_maps,target_weights,q_floor,counts)
        del check_target_bank
        libc.malloc_trim(0)
        check=cpu_chunks(p,setup,orders[2],p['batch']['draws_per_parent'],streams[2],counts)
        old_mean,references=pooled_mean(model,theta,check,check_maps,fit_weights,q_floor,counts,save_reference=True)
        old_risk=risks(old_mean,check_target,check_maps)
        arrays['old_CHECK_residual']=residual_vector(old_mean,check_target,check_maps).cpu().numpy()
        attempts,accepted=[],False
        for fraction in p['solver']['backtracking_fractions']:
            counts['joint_candidate_trials']+=1
            proposal=torch.maximum(model.lower,torch.minimum(model.upper,theta+fraction*delta))
            candidate_mean,trust,relative=pooled_candidate(p,model,proposal,check,check_maps,fit_weights,q_floor,references,counts)
            candidate_risk=risks(candidate_mean,check_target,check_maps)
            accepted=accept_joint(old_risk,candidate_risk,trust,len(pool)*p['batch']['draws_per_parent'],p)
            index=len(attempts)
            arrays[f'trial{index}_proposal']=proposal.cpu().numpy()
            arrays[f'trial{index}_CHECK_residual']=residual_vector(candidate_mean,check_target,check_maps).cpu().numpy()
            arrays[f'trial{index}_relative_RMS']=relative
            attempts.append(dict(fraction=fraction,accepted=accepted,risks=candidate_risk,trust=trust))
            np.savez(out/f'{name}_ITER{iteration:02d}.npz',**arrays)
            if accepted:
                theta=proposal;counts['teacher_accepted_joint_updates']+=1
                break
        arrays['endpoint']=theta.cpu().numpy().copy()
        np.savez(out/f'{name}_ITER{iteration:02d}.npz',**arrays)
        coverage.index_add_(0,orders[0],torch.ones_like(orders[0]));target_coverage.index_add_(0,orders[1],torch.ones_like(orders[1]))
        trace.append(dict(iteration=iteration,old_CHECK_risks=old_risk,attempts=attempts,accepted=accepted,geometry=geometry,parameter_norm=float(theta.norm())))
        write_json(out/(name+'_TRACE.json'),dict(track=name,iterations=trace,response_population=fit_population,target_population=target_population,
            pairing_used_for_fit=False,conditional_teacher_labels_used_for_fit=False,full_fit_parent_count=len(pool),draws_per_parent=p['batch']['draws_per_parent']))
        np.savez(out/(name+'_STATE.npz'),theta=theta.cpu().numpy(),source_coverage=coverage.cpu().numpy(),target_coverage=target_coverage.cpu().numpy())
        progress_file(out,counts,'precision_pilot_training',track=name,iteration=iteration+1,iterations=iterations,accepted_updates=sum(row['accepted'] for row in trace))
        del check,references,arrays,relative
        libc.malloc_trim(0)
        write_json(out/f'{name}_MEMORY_ITER{iteration:02d}.json',dict(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,full_banks_live=0,allocator_trim_requested=True))
    return theta,dict(track=name,iterations=iterations,accepted_updates=sum(row['accepted'] for row in trace),
        source_parents_visited=int((coverage>0).sum()),target_parents_visited=int((target_coverage>0).sum()),allowed_parent_count=len(pool),
        fit_population=fit_population,target_population=target_population,source_ESS=float(fit_weights[pool].sum().square()/fit_weights[pool].square().sum()),
        conditional_pairing_for_fit=False,all_fit_parents_in_every_FIT_and_CHECK=True)



def evaluate_fresh_seed(p, setup, seed_index, teacher, truth_theta, candidates, heldout, maps, out, counts):
    q_floor = float(setup['loss'].q_floor)
    result = {}
    for library in range(2):
        bank = random_bank(setup, heldout, p['final']['draws_per_parent'], p['evaluation_bank_seed'] + seed_index * 1000000 + library * 10000, counts)
        target_bank = random_bank(setup, heldout, p['final']['draws_per_parent'], p['evaluation_bank_seed'] + seed_index * 1000000 + library * 10000 + 1000, counts)
        target_obs = prediction(teacher, truth_theta, target_bank, q_floor, counts)
        target = feature_means(target_obs, maps)
        counts['joint_response_calls'] += 1
        counts['joint_response_rows'] += len(bank['rows'])
        truth_signal = teacher(truth_theta, bank['rows'], bank['z'])
        comparisons = dict(candidates)
        comparisons['teacher_oracle'] = (teacher, truth_theta)
        result[str(library)] = {}
        for name, (model, theta) in comparisons.items():
            obs = prediction(model, theta, bank, q_floor, counts)
            means = feature_means(obs, maps)
            conditional = conditional_report(model, theta, bank, truth_signal, counts)
            native = {}
            # The original metrics remain separately defined and reported.
            for h in HYPOTHESES:
                counts['native_metric_rows'] += 2 * len(bank['rows'])
                native[h] = hard_metrics(obs[h]['window_V'].cpu().numpy(), obs[h]['baseline100'].cpu().numpy(),
                    target_obs[h]['window_V'].cpu().numpy(), target_obs[h]['baseline100'].cpu().numpy(),
                    setup['loss'], setup['context'])
            result[str(library)][name] = dict(risks=risks(means, target, maps), conditional=conditional,
                native=native, observation_flags={h: {key: dict(count=int(value.sum()), denominator=len(value))
                    for key, value in obs[h]['flags'].items()} for h in HYPOTHESES})
        progress_file(out, counts, 'teacher_new_library_evaluation', seed=seed_index, library=library)
    return result



def run(p, setup, out, counts, started):
    device = setup['source'].device
    roles = np.load(ROOT/p['roles'],allow_pickle=False)
    pool, heldout = (torch.as_tensor(roles[name],device=device) for name in ('train_rows','heldout_rows'))
    np.savez(out/'TEACHER_ROLES.npz',train_rows=pool.cpu().numpy(),heldout_rows=heldout.cpu().numpy())
    tracks, decisions = {}, {}
    for seed, original in enumerate(setup['joint_models']):
        truth = teacher_parameters(original)
        matched, student = JointResponse01C1(original.cache), JointResponse01C1(original.cache)
        bank = random_bank(setup,pool,1,1940000+10000*seed,counts)
        obs = prediction(original,truth,bank,float(setup['loss'].q_floor),counts)
        blocks = {h:observable_blocks(obs[h]) for h in HYPOTHESES}
        maps = {role:{h:JointFeatureMap01(blocks[h],seed=p['feature_seeds'][role]) for h in HYPOTHESES}
            for role in ('FIT','CHECK','FINAL')}
        counts['feature_maps_constructed'] += 9
        old_factors = np.load(ROOT/p['feature_factors'][seed],allow_pickle=False)
        factors = {}
        for role in maps:
            for h in HYPOTHESES:
                key=role+'_'+h+'_cholesky'
                factors[key]=maps[role][h].cholesky.cpu().numpy()
                np.testing.assert_array_equal(factors[key],old_factors[key])
        np.savez(out/f'seed{seed}_FEATURE_MAPS.npz',**factors)
        write_json(out/f'seed{seed}_FEATURE_SLICES.json',{name:[s.start,s.stop] for name,s in maps['FIT'][HYPOTHESES[0]].block_slices.items()})
        write_json(out/f'seed{seed}_FEATURE_REPLAY.json',dict(pass_all=True,factors=9,statistics_target='original_teacher'))
        del bank,obs,blocks,old_factors
        label=f'seed{seed}_joint'
        saved=np.load(ROOT/p['endpoint_iterations'][seed],allow_pickle=False)
        initial=torch.as_tensor(saved['endpoint'],device=device)
        endpoint,summary = fit_track_precision(p,setup,matched,truth,student,initial,pool,maps['FIT'],maps['CHECK'],
            'nominal','nominal',p['iterations'],seed,out,counts,started,label)
        tracks[label]=summary
        candidates={'M0':(student,student.initial()),'joint':(student,endpoint)}
        decisions[str(seed)]={}
        for family,teacher in (('matched_C1',matched),('original',original)):
            target_out=out/family
            target_out.mkdir(exist_ok=True)
            evaluation=evaluate_fresh_seed(p,setup,seed,teacher,truth,candidates,heldout,maps['FINAL'],target_out,counts)
            low_draw=recovery_decision(p,evaluation)
            hard=[gate for gate in low_draw['gates'] if len(gate['name'].split('/'))==3]
            if len(hard)!=18: raise ValueError('Matched-control hard gate count differs')
            precise=precision_check(p,teacher,truth,student,endpoint,heldout,target_out,counts,started,seed)
            decision=dict(pass_all=all(item['pass_gate'] for item in hard) and precise['pass_all'],
                hard_observation_gates=hard,conditional_precision_gates=precise['gates'],diagnostic_64draw_decision=low_draw,
                target_family=family,positive_control_only=True,automatic_REAL_promotion=False)
            decisions[str(seed)][family]=decision
            write_json(target_out/f'seed{seed}_EVALUATION.json',evaluation)
            write_json(target_out/f'seed{seed}_PRECISION.json',precise)
            write_json(target_out/f'seed{seed}_DECISION.json',decision)
        progress_file(out,counts,'C1_precision_pilot_head_complete',seed=seed)
    return dict(status='COMPLETE',tracks=tracks,decisions=decisions,estimator_draws_per_parent=32,additional_iterations_per_head=p['iterations'],warm_start_from_frozen_matched_endpoints=True,
        feature_statistics_from_original_teacher=True,known_parameters_supplied_to_student=False,
        report_parents_used_for_fitting=False,paired_labels_used_for_fitting=False,
        original_cross_family_failure_preserved=True,positive_control_not_stage3_replacement=True,
        automatic_REAL_promotion=False,REAL_response_updates=0,scientific_acceptance=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256: raise ValueError('Registered precision-pilot config changed')
    p=json.loads(raw)
    for name,digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest: raise ValueError('Pinned input changed: '+name)
    args.out.mkdir(parents=True,exist_ok=False)
    counts,started=Counter(),time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    try:
        with torch.no_grad():
            setup=load_joint_setup(p,device,counts)
            for key in ('encoding','references','models','baseline','heads','real'): del setup[key]
            result=run(p,setup,args.out,counts,started)
        check_budget(p,counts,started)
    except TeacherBudget01 as error:
        write_json(args.out/'FAILED.json',dict(status='BUDGET_STOP',error=str(error),execution_counts=dict(counts),
            elapsed_seconds=time.monotonic()-started,UTC=datetime.now(timezone.utc).isoformat()))
        raise
    result.update(config_sha256=args.config_sha256,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.out/'RESULT.json',result)
    if sum(path.stat().st_size for path in args.out.rglob('*') if path.is_file())>p['resources']['output_MiB']*1024**2:
        raise RuntimeError('Precision-pilot output budget exceeded')
    write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest(),scientific_acceptance=False))
