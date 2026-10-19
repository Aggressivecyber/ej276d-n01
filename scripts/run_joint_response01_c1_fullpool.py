"""C1 unpaired teacher recovery using every fit parent in each empirical mean."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks, joint_step
from n01.joint01_model_c1 import JointResponse01C1
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction,
    feature_means, residual_vector, risks, teacher_parameters, conditional_report, HYPOTHESES)
from run_joint_response01_teacher import (check_budget, draw_parents, population_weights,
    accept_joint, evaluate_seed, recovery_decision)
from run_joint_response01_c1_teacher import precision_check

ROOT = Path(__file__).resolve().parents[1]


def cpu_chunks(p, setup, parents, draws, seed, counts):
    """Generate each fixed CRN bank once; retain it in CPU memory across FD calls."""
    chunks = []
    counts['full_fit_parent_banks'] += 1
    for index, start in enumerate(range(0, len(parents), p['parents_per_chunk'])):
        bank = random_bank(setup, parents[start:start+p['parents_per_chunk']], draws, seed+3*index, counts)
        chunks.append(dict(rows=bank['rows'].cpu(), z=bank['z'].cpu(),
            noise={h: bank['noise'][h].cpu() for h in HYPOTHESES}, seed=bank['seed']))
    return chunks


def pooled_mean(model, theta, chunks, maps, weights, q_floor, counts, *, save_reference=False):
    sums, total_weight, references = {}, 0., []
    for chunk in chunks:
        bank = dict(rows=chunk['rows'].to(theta.device), z=chunk['z'].to(theta.device),
            noise={h:chunk['noise'][h].to(theta.device) for h in HYPOTHESES})
        observations = prediction(model,theta,bank,q_floor,counts)
        weight = weights[bank['rows']]
        mass = float(weight.sum())
        means = feature_means(observations,maps,weight)
        for h in HYPOTHESES:
            sums[h] = sums[h]+mass*means[h] if h in sums else mass*means[h]
        total_weight += mass
        if save_reference:
            references.append({h:dict(carrier=observations[h]['carrier_input'].cpu(),
                flags={name:int(value.sum()) for name,value in observations[h]['flags'].items()}) for h in HYPOTHESES})
    return {h:sums[h]/total_weight for h in HYPOTHESES},references


def pooled_candidate(p, model, theta, chunks, maps, weights, q_floor, references, counts):
    sums,total_weight,relative = {},0.,[]
    changes = {h:Counter() for h in HYPOTHESES}
    denominator = 0
    for chunk,old in zip(chunks,references,strict=True):
        bank = dict(rows=chunk['rows'].to(theta.device),z=chunk['z'].to(theta.device),
            noise={h:chunk['noise'][h].to(theta.device) for h in HYPOTHESES})
        observations = prediction(model,theta,bank,q_floor,counts)
        weight=weights[bank['rows']]
        mass=float(weight.sum())
        means=feature_means(observations,maps,weight)
        scale=model.cache['A'][bank['rows']].abs().amax(1)
        for h in HYPOTHESES:
            sums[h]=sums[h]+mass*means[h] if h in sums else mass*means[h]
            relative.append(((observations[h]['carrier_input']-old[h]['carrier'].to(theta.device)).square().mean(1).sqrt()/scale).cpu())
            for name,value in observations[h]['flags'].items():
                changes[h][name]+=int(value.sum())-old[h]['flags'][name]
        total_weight+=mass
        denominator+=len(bank['rows'])
    relative=torch.cat(relative)
    trust=dict(mean_relative_RMS=float(relative.mean()),p99_relative_RMS=float(torch.quantile(relative,.99)),
        max_relative_RMS=float(relative.max()),flag_count_changes={h:dict(changes[h]) for h in HYPOTHESES},denominator=denominator)
    trust['pass']=(trust['mean_relative_RMS']<=p['solver']['output_TR_mean_RMS_fraction']
        and trust['p99_relative_RMS']<=p['solver']['output_TR_p99_RMS_fraction']
        and max(v for d in changes.values() for v in d.values())<=max(4,p['solver']['flag_increase_fraction']*denominator))
    return {h:sums[h]/total_weight for h in HYPOTHESES},trust,relative.numpy()


def fit_track_c1(p,setup,teacher,teacher_theta,model,initial,pool,maps,check_maps,
                fit_population,target_population,iterations,seed,out,counts,started,name):
    q_floor,theta=float(setup['loss'].q_floor),initial.clone()
    fit_weights,target_weights=population_weights(setup,fit_population),population_weights(setup,target_population)
    coverage=torch.zeros(len(setup['source']),dtype=torch.long,device=pool.device)
    target_coverage=torch.zeros_like(coverage)
    trace=[]
    for iteration in range(iterations):
        check_budget(p,counts,started)
        orders=[draw_parents(pool,len(pool),seed+10000*(role+1)+iteration) for role in range(4)]
        streams=[seed*100000+1000*iteration+100*role for role in range(4)]
        fit,target_bank,check,check_target_bank=[cpu_chunks(p,setup,order,p['batch']['draws_per_parent'],stream,counts)
            for order,stream in zip(orders,streams,strict=True)]
        # Check the changed aggregation against a single native 4096-row call.
        if name=='seed0_joint' and iteration==0:
            parts=fit[:2]
            whole=dict(rows=torch.cat([c['rows'] for c in parts]).to(theta.device),z=torch.cat([c['z'] for c in parts]).to(theta.device),
                noise={h:torch.cat([c['noise'][h] for c in parts]).to(theta.device) for h in HYPOTHESES})
            observed=prediction(model,theta,whole,q_floor,counts)
            contrasts=[]
            for profile in ('nominal','high_h'):
                weights=population_weights(setup,profile)
                direct=feature_means(observed,maps,weights[whole['rows']])
                streamed,_=pooled_mean(model,theta,parts,maps,weights,q_floor,counts)
                maximum=max(float((direct[h]-streamed[h]).abs().max()) for h in HYPOTHESES)
                for h in HYPOTHESES:
                    torch.testing.assert_close(direct[h],streamed[h],atol=1e-7,rtol=1e-6)
                contrasts.append(dict(profile=profile,maximum_absolute_feature_difference=maximum))
                counts['stream_equivalence_checks']+=1
            write_json(out/'STREAMING_EQUIVALENCE.json',dict(pass_all=True,contrasts=contrasts,atol=1e-7,rtol=1e-6,
                denominator=4096,feature_dtype='float32 encoding, float64 means',scientific_gate_changed=False))
            del observed,whole,direct,streamed,parts
        target,_=pooled_mean(teacher,teacher_theta,target_bank,maps,target_weights,q_floor,counts)
        del target_bank
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
                progress_file(out,counts,'fullpool_teacher_FD',track=name,iteration=iteration+1,column=j+1,parameters=model.parameter_count)
        J=torch.stack(columns,1)
        counts['joint_direction_solves']+=1
        delta,geometry=joint_step(residual,J,ridge_fraction=p['solver']['ridge_fraction'],radius=p['solver']['radius'])
        arrays.update(jacobian=J.cpu().numpy(),delta=delta.cpu().numpy())
        del fit,high,low,columns,highs,lows
        check_target,_=pooled_mean(teacher,teacher_theta,check_target_bank,check_maps,target_weights,q_floor,counts)
        del check_target_bank
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
        progress_file(out,counts,'fullpool_teacher_training',track=name,iteration=iteration+1,iterations=iterations,accepted_updates=sum(row['accepted'] for row in trace))
        del check,references,arrays
    return theta,dict(track=name,iterations=iterations,accepted_updates=sum(row['accepted'] for row in trace),
        source_parents_visited=int((coverage>0).sum()),target_parents_visited=int((target_coverage>0).sum()),allowed_parent_count=len(pool),
        fit_population=fit_population,target_population=target_population,source_ESS=float(fit_weights[pool].sum().square()/fit_weights[pool].square().sum()),
        conditional_pairing_for_fit=False,all_fit_parents_in_every_FIT_and_CHECK=True)


def run_teacher_c1(p, setup, out, counts, started):
    device, N = setup['source'].device, len(setup['source'])
    order = torch.randperm(N, generator=torch.Generator().manual_seed(p['role_seed'])).to(device)
    heldout, pool = order[:p['final']['parents']], order[p['final']['parents']:]
    np.savez(out / 'TEACHER_ROLES.npz', train_rows=pool.cpu().numpy(), heldout_rows=heldout.cpu().numpy())
    tracks, evaluations, decisions, saved, fixed_population, precision_runs = {}, {}, {}, {}, {}, {}
    students = [JointResponse01C1(model.cache) for model in setup['joint_models']]
    setup['physical_model'] = JointResponse01C1(setup['physical_model'].cache, physical=True)
    for seed_index, teacher in enumerate(setup['joint_models']):
        check_budget(p, counts, started)
        truth = teacher_parameters(teacher)
        student = students[seed_index]
        stats_bank = random_bank(setup, pool, 1, 1940000 + 10000 * seed_index, counts)
        stats_obs = prediction(teacher, truth, stats_bank, float(setup['loss'].q_floor), counts)
        blocks = {h: observable_blocks(stats_obs[h]) for h in HYPOTHESES}
        maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FIT']) for h in HYPOTHESES}
        check_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['CHECK']) for h in HYPOTHESES}
        final_maps = {h: JointFeatureMap01(blocks[h], seed=p['feature_seeds']['FINAL']) for h in HYPOTHESES}
        counts['feature_maps_constructed'] += 9
        np.savez(out / f'seed{seed_index}_FEATURE_MAPS.npz', **{
            role + '_' + h + '_cholesky': mapping[h].cholesky.cpu().numpy()
            for role, mapping in (('FIT', maps), ('CHECK', check_maps), ('FINAL', final_maps)) for h in HYPOTHESES})
        write_json(out / f'seed{seed_index}_FEATURE_SLICES.json', {
            block: [take.start, take.stop] for block, take in maps[HYPOTHESES[0]].block_slices.items()})
        fixed_population[str(seed_index)] = {}
        nominal = feature_means(stats_obs, maps)
        for scenario in ('nominal', 'low_h', 'high_h'):
            weights = population_weights(setup, scenario)[stats_bank['rows']]
            means = feature_means(stats_obs, maps, weights)
            fixed_population[str(seed_index)][scenario] = dict(
                ESS=float(weights.sum().square() / weights.square().sum()),
                weight_min=float(weights.min()), weight_max=float(weights.max()),
                mean_observed_Q={h: float((stats_obs[h]['Q'] * weights).sum() / weights.sum()) for h in HYPOTHESES},
                distance_from_nominal_same_fixed_response=risks(means, nominal, maps))
        candidates = {'M0': (student, student.initial()), 'physical_initial': (setup['physical_model'], setup['physical_model'].initial())}
        for name, learner in (('joint', student), ('physical', setup['physical_model'])):
            label = f'seed{seed_index}_{name}'
            endpoint, summary = fit_track_c1(p, setup, teacher, truth, learner, learner.initial(), pool, maps, check_maps,
                'nominal', 'nominal', p['iterations'], 1950000 + seed_index * 100000,
                out, counts, started, label)
            candidates[name] = (learner, endpoint)
            tracks[label] = summary
        evaluations[str(seed_index)] = evaluate_seed(p, setup, seed_index, teacher, truth, candidates, heldout,
                                                    final_maps, out, counts)
        diagnostic64 = recovery_decision(p, evaluations[str(seed_index)])
        hard_gates = [gate for gate in diagnostic64['gates'] if len(gate['name'].split('/')) == 3]
        if len(hard_gates) != 18:
            raise ValueError('C1 teacher hard-observation gate set differs')
        precise = precision_check(p, teacher, truth, student, candidates['joint'][1], heldout,
            out, counts, started, seed_index)
        precision_runs[str(seed_index)] = precise
        decisions[str(seed_index)] = dict(pass_all=all(item['pass_gate'] for item in hard_gates) and precise['pass_all'],
            hard_observation_gates=hard_gates, conditional_precision_gates=precise['gates'],
            diagnostic_64draw_decision=diagnostic64, conditional_acceptance_uses_prespecified_2048_draws=True)
        write_json(out / f'seed{seed_index}_PRECISION.json', precise)
        write_json(out / f'seed{seed_index}_EVALUATION.json', evaluations[str(seed_index)])
        write_json(out / f'seed{seed_index}_DECISION.json', decisions[str(seed_index)])
        saved[seed_index] = (teacher, truth, student, maps, check_maps, final_maps)
    # Deliberately warm-start at the known response. With a changed population,
    # compare correct fixed profile with a wrong nominal assumption; no free weights.
    original_teacher, truth, teacher, maps, check_maps, final_maps = saved[0]
    stress = {}
    for profile in ('nominal', 'high_h'):
        label = 'population_high_h_fit_' + profile
        endpoint, summary = fit_track_c1(p, setup, teacher, truth, teacher, truth, pool, maps, check_maps,
            profile, 'high_h', p['population_stress_iterations'], 2150000,
            out, counts, started, label)
        bank = random_bank(setup, heldout, p['final']['draws_per_parent'], 2160000, counts)
        counts['joint_response_calls'] += 1
        counts['joint_response_rows'] += len(bank['rows'])
        truth_signal = teacher(truth, bank['rows'], bank['z'])
        summary['conditional_drift_from_known_response'] = conditional_report(teacher, endpoint, bank, truth_signal, counts)
        summary['parameter_shift_norm'] = float((endpoint - truth).norm())
        summary['known_response_warm_start_not_recovery_evidence'] = True
        stress[profile] = summary
    return dict(status='COMPLETE', coupled_teacher_pass_all=all(d['pass_all'] for d in decisions.values()),
        tracks=tracks, recovery_decisions=decisions, population_stress=stress,
        fixed_response_population_scenarios=fixed_population, precision_runs=precision_runs,
        recovery_target='unchanged original joint01 coupled teachers', recovery_learners='C1 neural and C1 physical',
        population_stress_target_and_learner='same known C1 response, fixed high_h target population',
        all_scientific_teacher_gates_preregistered=True,
        teacher_training_parent_count=len(pool), teacher_holdout_parent_count=len(heldout),
        validation_scope='post-hoc heldout parents of frozen A0/M0; not full-pipeline unseen validation',
        true_teacher_parameter_labels_supplied_to_recovery_solver=False,
        three_noise_hypotheses_retained=True, population_parameters_fitted=0,
        REAL_response_updates=0, DEV_waveforms_decoded=0, scientific_acceptance=False,
        next_action='register_REAL_joint_calibration' if all(d['pass_all'] for d in decisions.values()) else
                    'diagnose_teacher_failure_before_REAL_fit; do not enlarge network or open_REAL_calibration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered C1 coupled-teacher config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned C1 teacher input changed: ' + name)
    review = json.loads((ROOT / p['engineering_review']).read_text())
    if not review['engineering_gate_pass'] or review['status'] != 'PASS_INDEPENDENT_C1_ENGINEERING_AUDIT':
        raise ValueError('C1 engineering prerequisite was not independently confirmed')
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        for key in ('encoding', 'references', 'models', 'baseline', 'heads', 'real'):
            del setup[key]
        progress_file(args.out, counts, 'C1_teacher_setup_loaded')
        result = run_teacher_c1(p, setup, args.out, counts, started)
    check_budget(p, counts, started)
    result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
        elapsed_seconds=time.monotonic() - started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
    write_json(args.out / 'RESULT.json', result)
    if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()) > p['resources']['output_MiB'] * 1024**2:
        raise RuntimeError('C1 teacher output budget exceeded')
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE',
        result_sha256=hashlib.sha256((args.out / 'RESULT.json').read_bytes()).hexdigest(),
        coupled_teacher_pass_all=result['coupled_teacher_pass_all']))
