"""Complete the missing second-head diagnostic without full-bank copies."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import resource

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.joint01_model_c1 import JointResponse01C1
from run_joint_response01 import (load_joint_setup, random_bank, prediction, residual_vector,
    risks, teacher_parameters, progress_file, HYPOTHESES)
from run_joint_response01_teacher import (check_budget, TeacherBudget01, draw_parents,
    population_weights, accept_joint)
from run_joint_response01_c1_fullpool import cpu_chunks, pooled_mean, pooled_candidate

ROOT = Path(__file__).resolve().parents[1]


def run(p, setup, out, counts, started):
    device, q_floor = setup['source'].device, float(setup['loss'].q_floor)
    roles = np.load(ROOT/p['roles'], allow_pickle=False)
    pool = torch.as_tensor(roles['train_rows'], device=device)
    weights = population_weights(setup, 'nominal')
    reports = {}
    for seed in (1,):
        original=setup['joint_models'][seed]
        truth = teacher_parameters(original)
        model = JointResponse01C1(original.cache)
        saved = np.load(ROOT/p['endpoint_iterations'][seed], allow_pickle=False)
        theta = torch.as_tensor(saved['endpoint'], device=device)
        low_saved=np.load(ROOT/p['low_direction'],allow_pickle=False)
        (out/'seed1_draw4_DIRECTION.npz').write_bytes((ROOT/p['low_direction']).read_bytes())
        replay_checks=0
        stats = random_bank(setup, pool, 1, 1940000+10000*seed, counts)
        obs = prediction(original, truth, stats, q_floor, counts)
        blocks = {h:observable_blocks(obs[h]) for h in HYPOTHESES}
        maps = {role:{h:JointFeatureMap01(blocks[h],seed=p['feature_seeds'][role]) for h in HYPOTHESES}
            for role in ('FIT','CHECK','FINAL')}
        counts['feature_maps_constructed'] += 9
        factors = np.load(ROOT/p['feature_factors'][seed], allow_pickle=False)
        for role in maps:
            for h in HYPOTHESES:
                np.testing.assert_array_equal(maps[role][h].cholesky.cpu().numpy(),factors[role+'_'+h+'_cholesky'])
        np.savez(out/f'seed{seed}_FEATURE_MAPS.npz',**{key:factors[key] for key in factors.files})
        del stats,obs,blocks,factors
        orders = [draw_parents(pool,len(pool),p['order_seed']+100*seed+role) for role in range(6)]
        streams = [p['bank_seed']+1000000*seed+1000*role for role in range(6)]
        bank_arrays=dict(pool=pool.cpu().numpy(),theta=theta.cpu().numpy(),parent_orders=np.stack([a.cpu().numpy() for a in orders]),bank_base_seeds=np.array(streams,dtype=np.int64))
        old_banks=np.load(ROOT/p['saved_banks'],allow_pickle=False)
        for key,value in bank_arrays.items(): np.testing.assert_array_equal(value,old_banks[key])
        np.savez(out/f'seed{seed}_BANKS.npz',**bank_arrays)
        fit,target = [cpu_chunks(p,setup,orders[role],32,streams[role],counts) for role in range(2)]
        write_json(out/'MEMORY_FIT_BANKS.json',dict(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,fit_target_tensor_bytes=sum(v.numel()*v.element_size() for bank in (fit,target) for c in bank for v in (c['rows'],c['z'],*c['noise'].values())),full_bank_copies=0))
        directions, report = {}, {'directions':{},'validation':{}}
        for draws in (4,32):
            # Each low-precision bank is the first four draws of each parent in
            # the same high-precision bank; no additional random stream is used.
            if draws==4:
                banks=[]
                for chunks in (fit,target):
                    nested=[]
                    for chunk in chunks:
                        take=torch.arange(len(chunk['rows'])).reshape(-1,32)[:,:4].reshape(-1)
                        nested.append(dict(rows=chunk['rows'][take],z=chunk['z'][take],
                            noise={h:chunk['noise'][h][take] for h in HYPOTHESES},seed=chunk['seed']))
                    banks.append(nested)
                fit_view,target_view=banks
            else:
                fit_view,target_view=fit,target
            direction_path=out/(f'seed{seed}_draw4_REPLAY.npz' if draws==4 else f'seed{seed}_draw32_DIRECTION.npz')
            target_mean,_=pooled_mean(model,truth,target_view,maps['FIT'],weights,q_floor,counts)
            predicted,_=pooled_mean(model,theta,fit_view,maps['FIT'],weights,q_floor,counts)
            residual=residual_vector(predicted,target_mean,maps['FIT'])
            arrays=dict(theta=theta.cpu().numpy(),lower=model.lower.cpu().numpy(),upper=model.upper.cpu().numpy(),
                residual=residual.cpu().numpy(),draws=np.array(draws))
            if draws==4:
                for key in ('theta','lower','upper','residual'):
                    np.testing.assert_array_equal(arrays[key],low_saved[key]);replay_checks+=1
            columns,highs,lows,spans=[],[],[],[]
            indices=p['replay_columns'] if draws==4 else list(range(model.parameter_count))
            for column in indices:
                check_budget(p,counts,started)
                plus,minus=theta.clone(),theta.clone()
                plus[column]=min(float(model.upper[column]),float(theta[column])+p['solver']['finite_difference_step'])
                minus[column]=max(float(model.lower[column]),float(theta[column])-p['solver']['finite_difference_step'])
                high,_=pooled_mean(model,plus,fit_view,maps['FIT'],weights,q_floor,counts)
                low,_=pooled_mean(model,minus,fit_view,maps['FIT'],weights,q_floor,counts)
                hi,lo=residual_vector(high,target_mean,maps['FIT']),residual_vector(low,target_mean,maps['FIT'])
                span=float(plus[column]-minus[column])
                columns.append((hi-lo)/span);highs.append(hi.cpu().numpy());lows.append(lo.cpu().numpy());spans.append(span)
                if draws==4:
                    for key,value in (('FD_high',highs[-1]),('FD_low',lows[-1]),('FD_parameter_spans',span)):
                        np.testing.assert_array_equal(value,low_saved[key][column]);replay_checks+=1
                    counts['replayed_FD_columns']+=1
                counts['finite_difference_columns']+=1
                counts['fullpool_FIT_chunk_columns']+=len(fit_view)
                if (column+1)%9==0:
                    arrays.update(FD_high=np.stack(highs),FD_low=np.stack(lows),FD_parameter_spans=np.array(spans))
                    np.savez(direction_path,**arrays)
                    progress_file(out,counts,'draw_precision_restore' if draws==4 else 'draw_precision_FD',
                        seed=seed,draws=draws,column=column+1,completed_columns=len(highs),parameters=len(indices))
            J=torch.as_tensor(low_saved['jacobian'],device=device) if draws==4 else torch.stack(columns,1)
            delta,geometry=joint_step(residual,J,ridge_fraction=p['solver']['ridge_fraction'],radius=p['solver']['radius'])
            counts['joint_direction_solves']+=1
            if draws==4:
                np.testing.assert_array_equal(delta.cpu().numpy(),low_saved['delta']);replay_checks+=1
                write_json(out/'RESTORE_REPLAY.json',dict(pass_all=True,exact_array_checks=replay_checks,parent_bank_arrays_exact=4,feature_factors_exact=9,columns=p['replay_columns'],old_direction_reused=True,replayed_work_charged=True))
            directions[draws]=delta
            arrays['FD_parameter_indices']=np.array(indices)
            arrays.update(jacobian=J.cpu().numpy(),delta=delta.cpu().numpy())
            np.savez(direction_path,**arrays)
            report['directions'][str(draws)]=geometry
            del fit_view,target_view,arrays,J,columns,highs,lows,high,low
            if draws==4: del banks,nested,chunks,chunk,take
        del fit,target
        write_json(out/'MEMORY_AFTER_DIRECTIONS.json',dict(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,full_bank_copies=0))
        for validation,role in (('CHECK',2),('VERIFY',4)):
            mapping=maps['CHECK' if validation=='CHECK' else 'FINAL']
            source_bank,target_bank=[cpu_chunks(p,setup,orders[r],32,streams[r],counts) for r in (role,role+1)]
            target_mean,_=pooled_mean(model,truth,target_bank,mapping,weights,q_floor,counts)
            del target_bank
            base_mean,references=pooled_mean(model,theta,source_bank,mapping,weights,q_floor,counts,save_reference=True)
            base_risk=risks(base_mean,target_mean,mapping)
            arrays=dict(theta=theta.cpu().numpy(),old_residual=residual_vector(base_mean,target_mean,mapping).cpu().numpy())
            trials=[]
            for draws in (4,32):
                for index,fraction in enumerate(p['solver']['backtracking_fractions']):
                    check_budget(p,counts,started)
                    proposal=torch.maximum(model.lower,torch.minimum(model.upper,theta+fraction*directions[draws]))
                    value,trust,relative=pooled_candidate(p,model,proposal,source_bank,mapping,weights,q_floor,references,counts)
                    candidate_risk=risks(value,target_mean,mapping)
                    passed=accept_joint(base_risk,candidate_risk,trust,len(pool)*32,p)
                    key=f'draw{draws}_trial{index}'
                    arrays[key+'_proposal']=proposal.cpu().numpy()
                    arrays[key+'_residual']=residual_vector(value,target_mean,mapping).cpu().numpy()
                    arrays[key+'_relative_RMS']=relative
                    trials.append(dict(draws=draws,fraction=fraction,passes_original_candidate_rule=passed,
                        risks=candidate_risk,trust=trust))
                    counts['joint_candidate_trials']+=1
                    np.savez(out/f'seed{seed}_{validation}.npz',**arrays)
            report['validation'][validation]=dict(old_risks=base_risk,trials=trials)
            del source_bank,references,arrays
        report['fractions_passing_both_independent_banks']={str(draws):[
            fraction for fraction in p['solver']['backtracking_fractions'] if all(any(
                trial['draws']==draws and trial['fraction']==fraction and trial['passes_original_candidate_rule']
                for trial in report['validation'][view]['trials']) for view in ('CHECK','VERIFY'))] for draws in (4,32)}
        reports[str(seed)]=report
        write_json(out/f'seed{seed}_REPORT.json',report)
        progress_file(out,counts,'draw_precision_head_complete',seed=seed)
    return dict(status='COMPLETE',reports=reports,completion_only_seed1=True,old_first_head_frozen=True,replay_array_checks=replay_checks,target_family='C1',parameter_updates=0,
        frozen_endpoint_unchanged=True,nested_four_of_thirtytwo_draws=True,report_parents_used=False,
        known_teacher_parameter_labels_used_for_direction=False,all_candidates_reported_without_endpoint_selection=True,
        REAL_response_updates=0,scientific_acceptance=False,automatic_REAL_promotion=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256: raise ValueError('Registered draw-precision completion config changed')
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
        raise RuntimeError('Draw-precision output budget exceeded')
    write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
