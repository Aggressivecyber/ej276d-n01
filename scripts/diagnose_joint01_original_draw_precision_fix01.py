"""Nested 4-to-32-draw original-generator direction check; no updates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from n01.data import write_json
from n01.joint01_features import JointFeatureMap01, observable_blocks
from n01.joint01_native_step import native_joint_step
from n01.observation008 import observe_existing_carrier
from diagnose_joint01_native_direction import native_values as original_native_values
from run_joint_response01 import (load_joint_setup, progress_file, random_bank, prediction as raw_prediction,
    feature_means, residual_vector, risks, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents, accept_joint

# Bind the exact 8:1 operator in this isolated scientific process only.
# NativeMetricLoss036.wave_components uses its module's imported W1 symbol;
# inherited QPSD/noise methods resolve the losses008_v2 module symbol.
import run_response036 as native_wave_module
from n01 import losses008_v2 as native_distribution_module
from n01.joint01_integer_w1 import empirical_wasserstein_1d
native_wave_module.empirical_wasserstein_1d = empirical_wasserstein_1d
native_distribution_module.empirical_wasserstein_1d = empirical_wasserstein_1d

NATIVE_TIMING = dict(calls=0,seconds=0.)


def native_values(observations,target,setup,counts):
    started=time.monotonic()
    values=original_native_values(observations,target,setup,counts)
    NATIVE_TIMING['calls']+=1
    NATIVE_TIMING['seconds']+=time.monotonic()-started
    return values


ROOT = Path(__file__).resolve().parents[1]
FIT_LABELS = ('FIT0', 'FIT1', 'FIT2')


def prediction(model, theta, bank, q_floor, counts):
    active = dict(rows=bank['rows'].to(theta.device), z=bank['z'].to(theta.device),
        noise={h: bank['noise'][h].to(theta.device) for h in HYPOTHESES})
    return raw_prediction(model, theta, active, q_floor, counts)


def hard_check(p, model, proposal, bank, target, features, maps, baseline,
               old_risk, old_native, anchor, setup, counts):
    obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
    means = feature_means(obs, maps)
    residual, risk = residual_vector(means, features, maps), risks(means, features, maps)
    native = native_values(obs, target, setup, counts)
    scale = model.cache['A'][bank['rows'].to(proposal.device)].abs().amax(1)
    relative = torch.cat([((obs[h]['carrier_input'] - baseline[h]['carrier'].to(proposal.device)).square().mean(1).sqrt()/scale)
        for h in HYPOTHESES])
    changes = {h: {key: int(value.sum())-baseline[h]['flags'][key] for key,value in obs[h]['flags'].items()} for h in HYPOTHESES}
    trust = dict(mean_relative_RMS=float(relative.mean()), p99_relative_RMS=float(torch.quantile(relative,.99)),
        max_relative_RMS=float(relative.max()), flag_count_changes=changes, denominator=len(bank['rows']))
    trust['pass'] = (trust['mean_relative_RMS'] <= p['solver']['output_TR_mean_RMS_fraction']
        and trust['p99_relative_RMS'] <= p['solver']['output_TR_p99_RMS_fraction']
        and max(v for d in changes.values() for v in d.values()) <= max(4,p['solver']['flag_increase_fraction']*len(bank['rows'])))
    ratios, anchored = (native/old_native).reshape(3,3), (native/anchor).reshape(3,3)
    native_pass = all(bool((value[:,:2] <= p['native_guard']['wave_QPSD_factor']).all()
        and (value[:,2] <= p['native_guard']['noise_factor']).all()) for value in (ratios,anchored))
    joint_pass = accept_joint(old_risk,risk,trust,p['batch']['target_rows'],p)
    return residual,native,dict(risks=risk,native=native.tolist(),native_ratios=ratios.tolist(),
        initial_anchor_ratios=anchored.tolist(),native_pass=native_pass,trust=trust,
        original_joint_accept=joint_pass,accepted=native_pass and joint_pass)



def run(p, setup, out, counts, started):
    device = setup['source'].device
    with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
        source_pool, real_pool = [torch.as_tensor(archive[key], device=device)
            for key in ('source_train_rows', 'real_train_rows')]
    zero_arrays = {}
    for seed, model in enumerate(setup['joint_models']):
        rows = source_pool[:128]
        z = torch.randn(128, 8, generator=torch.Generator().manual_seed(
            p['zero_identity_seed'] + seed * 100000), dtype=torch.float32).to(device)
        counts['shape_standard_normal_values'] += z.numel()
        selected = setup['references'][seed].select_encoding(setup['encoding'], rows)
        counts['M0_head_calls'] += 1
        counts['M0_head_rows'] += len(rows)
        original = setup['references'][seed](selected['source'], z, cache=selected,
            mean_enabled=False, charge_enabled=False)['signal_pre']
        theta = model.initial()
        if not bool((theta == 0).all()) or float(model.noise_scale(theta)) != 1.:
            raise ValueError('Registered M0 start is not zero increment')
        actual = model(theta, rows, z)
        counts['joint_response_calls'] += 1
        counts['joint_response_rows'] += len(rows)
        if not torch.equal(original.view(torch.uint8), actual.view(torch.uint8)):
            raise ValueError('M0 zero-increment signal identity failed')
        counts['M0_zero_signal_replays_verified'] += 1
        counts['M0_zero_signal_rows_verified'] += len(rows)
        for key, value in dict(rows=rows, z=z, original_signal=original, joint_zero_signal=actual).items():
            zero_arrays[f'seed{seed}_' + key] = value.cpu().numpy()
    np.savez(out / 'M0_ZERO_IDENTITY.npz', **zero_arrays)
    del zero_arrays, original, actual, selected, z
    for key in ('encoding', 'references', 'models', 'baseline', 'heads'):
        del setup[key]
    progress_file(out, counts, 'multibank_M0_zero_identity_verified')
    cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
        q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
    for key in ('window_V', 'baseline100'):
        if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
            raise ValueError('REAL readout differs from frozen loader: ' + key)
    real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
        flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
    counts['REAL_observable_replay_calls'] += 1
    counts['REAL_observable_replay_rows'] += len(real['Q'])
    del setup['real'], cpu_real
    blocks = observable_blocks(select_real(real, real_pool))
    maps, contexts = {}, {}
    for label, seed in p['feature_seeds'].items():
        fmap = JointFeatureMap01(blocks, seed=seed)
        maps[label] = {h: fmap for h in HYPOTHESES}
        contexts[label] = setup['loss'].create_evaluation_context(seed=p['native_projection_seeds'][label])
        counts['feature_maps_constructed'] += 1
        counts['new_native_projection_contexts'] += 1
    del blocks
    role_setup = {label: dict(setup, context=context) for label, context in contexts.items()}
    joint_maps = {label + '/' + h: maps[label][h] for label in FIT_LABELS for h in HYPOTHESES}
    summaries = {}
    for seed_index, model in enumerate(setup['joint_models']):
        check_budget(p, counts, started)
        theta = model.initial()
        saved = np.load(ROOT/p['low_draw_arrays'][seed_index],allow_pickle=False)
        if seed_index==0:
            predecessor=np.load(ROOT/p['stopped_prefix_arrays'],allow_pickle=False)
            prefix_native_errors=[]
        banks, targets, features, baseline, old_risks, old_native, residuals = {}, {}, {}, {}, {}, {}, {}
        arrays = dict(theta=theta.cpu().numpy(), lower=model.lower.cpu().numpy(), upper=model.upper.cpu().numpy())
        for label, seed_base in p['role_seed_bases'].items():
            seed = seed_base + seed_index * 100000
            parents = draw_parents(source_pool, 1024, seed + 100)
            real_rows = draw_parents(real_pool, 4096, seed + 200)
            if not np.isin(parents.cpu().numpy(), source_pool.cpu().numpy()).all() or not np.isin(real_rows.cpu().numpy(), real_pool.cpu().numpy()).all():
                raise ValueError('Multibank rows are outside fitting roles')
            arrays[label + '_source_parents'], arrays[label + '_real_rows'] = parents.cpu().numpy(), real_rows.cpu().numpy()
            arrays[label + '_cholesky'] = maps[label][HYPOTHESES[0]].cholesky.cpu().numpy()
            counts['REAL_fit_feature_rows' if label in FIT_LABELS else 'REAL_CHECK_feature_rows'] += len(real_rows)
            np.testing.assert_array_equal(arrays[label+'_source_parents'],saved[label+'_source_parents'])
            np.testing.assert_array_equal(arrays[label+'_real_rows'],saved[label+'_real_rows'])
            np.testing.assert_array_equal(arrays[label+'_cholesky'],saved[label+'_cholesky'])
            targets[label] = select_real(real,real_rows)
            features[label] = feature_means({h:targets[label] for h in HYPOTHESES},maps[label])
            low = random_bank(setup,parents,4,seed+1000,counts)
            low_obs = prediction(model,theta,low,setup['loss'].q_floor,counts)
            low_residual = residual_vector(feature_means(low_obs,maps[label]),features[label],maps[label])
            low_native = native_values(low_obs,targets[label],role_setup[label],counts)
            np.testing.assert_allclose(low_residual.cpu().numpy(),saved[label+'_baseline_residual'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(low_native.cpu().numpy(),saved[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
            arrays[label+'_replayed_low_residual'],arrays[label+'_replayed_low_native'] = low_residual.cpu().numpy(),low_native.cpu().numpy()
            counts['low_draw_center_replays_verified'] += 1
            del low_obs
            extension_seed = p['extension_bank_seed']+1000000*seed_index+10000*list(p['role_seed_bases']).index(label)
            extra = random_bank(setup,parents,p['batch']['draws_per_parent']-4,extension_seed,counts)
            draws = p['batch']['draws_per_parent']
            banks[label] = dict(rows=parents.repeat_interleave(draws).cpu(),
                z=torch.cat((low['z'].reshape(1024,4,8),extra['z'].reshape(1024,draws-4,8)),1).reshape(-1,8).cpu(),
                noise={h:torch.cat((low['noise'][h].reshape(1024,4,384),extra['noise'][h].reshape(1024,draws-4,384)),1).reshape(-1,384).cpu() for h in HYPOTHESES})
            np.testing.assert_array_equal(banks[label]['z'].reshape(1024,draws,8)[:,:4].numpy(),low['z'].cpu().numpy().reshape(1024,4,8))
            for h in HYPOTHESES:
                np.testing.assert_array_equal(banks[label]['noise'][h].reshape(1024,draws,384)[:,:4].numpy(),low['noise'][h].cpu().numpy().reshape(1024,4,384))
            arrays[label+'_random_seeds'] = np.array([seed+1000,extension_seed],dtype=np.int64)
            counts['nested_banks_verified'] += 1
            del low,extra
            current = prediction(model,theta,banks[label],setup['loss'].q_floor,counts)
            means = feature_means(current,maps[label])
            old_risks[label] = risks(means,features[label],maps[label])
            residuals[label] = residual_vector(means,features[label],maps[label])
            old_native[label] = native_values(current,targets[label],role_setup[label],counts)
            baseline[label] = {h:dict(carrier=current[h]['carrier_input'].cpu(),
                flags={key:int(value.sum()) for key,value in current[h]['flags'].items()}) for h in HYPOTHESES}
            del current
            arrays[label + '_baseline_residual'] = residuals[label].cpu().numpy()
            arrays[label + '_baseline_native'] = old_native[label].cpu().numpy()
            if seed_index==0:
                np.testing.assert_array_equal(arrays[label+'_baseline_residual'],predecessor[label+'_baseline_residual'])
                np.testing.assert_allclose(arrays[label+'_baseline_native'],predecessor[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
                prefix_native_errors.append(float(np.max(np.abs(arrays[label+'_baseline_native']-predecessor[label+'_baseline_native']))))
                counts['same_32draw_centers_verified']+=1
            progress_file(out,counts,'original_precision_fix01_bank_ready',seed=seed_index,role=label,native_timing=dict(NATIVE_TIMING))
        r = torch.cat([residuals[label] for label in FIT_LABELS]) / np.sqrt(len(FIT_LABELS))
        n = torch.cat([old_native[label] for label in FIT_LABELS])
        arrays['stacked_baseline_residual'], arrays['stacked_baseline_native'] = r.cpu().numpy(), n.cpu().numpy()
        report = dict(seed=seed_index, M0_anchor_equals_center=True, fit_bank_count=3,
            baseline_risks=old_risks, baseline_native={k: v.tolist() for k, v in old_native.items()},
            native_projection_contexts={k: v.metadata() for k, v in contexts.items()},
            FD_columns_completed=0, trials=[], selected_trial=None, fresh_pass=False,
            stop_reason='trial_budget', current_solve=None)
        np.savez(out / f'seed{seed_index}_MULTIBANK_M0.npz', **arrays)
        write_json(out / f'seed{seed_index}_MULTIBANK_M0.json', report)
        plus_r, minus_r, plus_n, minus_n, plus_theta, minus_theta = [], [], [], [], [], []
        for coordinate in range(model.parameter_count):
            check_budget(p, counts, started)
            plus, minus = theta.clone(), theta.clone()
            plus[coordinate] = min(float(model.upper[coordinate]), float(theta[coordinate]) + p['finite_difference_step'])
            minus[coordinate] = max(float(model.lower[coordinate]), float(theta[coordinate]) - p['finite_difference_step'])
            pairs_r, pairs_n = [], []
            for proposal in (plus, minus):
                bank_r, bank_n = [], []
                for label in FIT_LABELS:
                    obs = prediction(model, proposal, banks[label], setup['loss'].q_floor, counts)
                    means = feature_means(obs, maps[label])
                    bank_r.append(residual_vector(means, features[label], maps[label]))
                    bank_n.append(native_values(obs, targets[label], role_setup[label], counts))
                    del obs
                pairs_r.append(torch.cat(bank_r) / np.sqrt(len(FIT_LABELS)))
                pairs_n.append(torch.cat(bank_n))
            plus_r.append(pairs_r[0]); minus_r.append(pairs_r[1])
            plus_n.append(pairs_n[0]); minus_n.append(pairs_n[1])
            plus_theta.append(plus); minus_theta.append(minus)
            counts['finite_difference_columns'] += 1
            counts['FIT_bank_finite_difference_columns'] += len(FIT_LABELS)
            if (coordinate + 1) % 9 == 0:
                for key, values in dict(FD_plus_residual=plus_r, FD_minus_residual=minus_r,
                        FD_plus_native=plus_n, FD_minus_native=minus_n,
                        FD_plus_theta=plus_theta, FD_minus_theta=minus_theta).items():
                    arrays[key] = torch.stack(values, 1).cpu().numpy()
                if seed_index==0 and coordinate+1==p['stopped_prefix_columns']:
                    for key in ('FD_plus_residual','FD_minus_residual','FD_plus_theta','FD_minus_theta'):
                        np.testing.assert_array_equal(arrays[key],predecessor[key])
                    for key in ('FD_plus_native','FD_minus_native'):
                        np.testing.assert_allclose(arrays[key],predecessor[key],rtol=1e-12,atol=1e-12)
                        prefix_native_errors.append(float(np.max(np.abs(arrays[key]-predecessor[key]))))
                    counts['predecessor_FD_columns_verified']+=p['stopped_prefix_columns']
                    report['same_32draw_prefix_replay']=dict(columns=p['stopped_prefix_columns'],centers=5,
                        joint_and_parameter_arrays_exact=True,max_native_absolute_difference=max(prefix_native_errors))
                    predecessor.close()
                report['FD_columns_completed'] = coordinate + 1
                report['native_timing']=dict(NATIVE_TIMING)
                np.savez(out / f'seed{seed_index}_MULTIBANK_M0.npz', **arrays)
                write_json(out / f'seed{seed_index}_MULTIBANK_M0.json', report)
                progress_file(out, counts, 'original_precision_fix01_joint_Jacobian', seed=seed_index, column=coordinate + 1,native_timing=dict(NATIVE_TIMING))
        difference = torch.stack(plus_theta, 1).diag() - torch.stack(minus_theta, 1).diag()
        J = (torch.stack(plus_r, 1) - torch.stack(minus_r, 1)) / difference
        N = (torch.stack(plus_n, 1) - torch.stack(minus_n, 1)) / difference
        arrays['original_jacobian'], arrays['original_native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
        report['draws_per_parent'] = p['batch']['draws_per_parent']
        report['nested_low_draw_replays'] = 5
        report['low_to_high_jacobian'] = {}
        for label,key,value in (('joint','original_jacobian',J),('native','original_native_jacobian',N)):
            low_j = torch.as_tensor(saved[key],device=device)
            report['low_to_high_jacobian'][label] = dict(relative_difference=float((value-low_j).norm()/low_j.norm()),
                cosine=float((value.flatten()@low_j.flatten())/(value.norm()*low_j.norm())))
        report['frozen_low_candidate_checks'] = {}
        low_candidate = torch.as_tensor(saved['uncommitted_candidate'],device=device)
        arrays['frozen_low_candidate'] = low_candidate.cpu().numpy()
        for label in ('CHECK','FRESH'):
            ar,an,ck = hard_check(p,model,low_candidate,banks[label],targets[label],features[label],maps[label],baseline[label],
                old_risks[label],old_native[label],old_native[label],role_setup[label],counts)
            arrays['low_candidate_'+label+'_actual_residual'],arrays['low_candidate_'+label+'_actual_native'] = ar.cpu().numpy(),an.cpu().numpy()
            report['frozen_low_candidate_checks'][label] = ck
            counts['frozen_low_candidate_checks'] += 1
        saved.close()

        radius, selected = p['initial_radius'], None
        for trial_index in range(p['max_trials_per_seed']):
            check_budget(p, counts, started)
            prefix = f'trial{trial_index}_'
            arrays[prefix + 'jacobian'], arrays[prefix + 'native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
            counts['joint_direction_solve_attempts'] += 1
            report['current_solve'] = dict(trial=trial_index, radius=radius)
            np.savez(out / f'seed{seed_index}_MULTIBANK_M0.npz', **arrays)
            write_json(out / f'seed{seed_index}_MULTIBANK_M0.json', report)
            progress_file(out, counts, 'multibank_M0_joint_solve', seed=seed_index, trial=trial_index + 1, radius=radius)
            unit_delta, geometry = native_joint_step(r, J * radius, joint_maps, n, N * radius,
                theta / radius, model.lower / radius, model.upper / radius,
                ridge_fraction=p['solver']['ridge_fraction'], radius=1.,
                macro_factor=p['solver']['macro_degradation_factor'], sample_count=4096,
                max_iterations=p['constrained_solver']['max_iterations'], ftol=p['constrained_solver']['ftol'])
            delta = unit_delta * radius
            geometry.update(solver='SLSQP_multibank_full_joint_native_nonincrease_radius_scaled_variable',
                variable_scale=radius, unit_step_norm=geometry['step_norm'], unit_ridge=geometry['ridge'],
                step_norm=float(delta.norm()), ridge=geometry['ridge'] / radius**2)
            counts['joint_direction_solves'] += 1
            report['current_solve'] = None
            if float(delta.square().sum()) == 0:
                report['stop_reason'], report['zero_direction_geometry'] = 'zero_joint_direction', geometry
                break
            proposal = torch.maximum(model.lower, torch.minimum(model.upper, theta + delta))
            step = proposal - theta
            counts['joint_candidate_trials'] += 1
            predicted_r, predicted_n = r + J @ step, n + N @ step
            arrays[prefix + 'proposal'] = proposal.cpu().numpy()
            arrays[prefix + 'predicted_residual'], arrays[prefix + 'predicted_native'] = predicted_r.cpu().numpy(), predicted_n.cpu().numpy()
            trial = dict(radius=radius, geometry=geometry, checks={})
            bank_r, bank_n = [], []
            for label in (*FIT_LABELS, 'CHECK'):
                actual_r, actual_n, check = hard_check(p, model, proposal, banks[label], targets[label],
                    features[label], maps[label], baseline[label], old_risks[label], old_native[label],
                    old_native[label], role_setup[label], counts)
                arrays[prefix + label + '_actual_residual'], arrays[prefix + label + '_actual_native'] = actual_r.cpu().numpy(), actual_n.cpu().numpy()
                trial['checks'][label] = check
                if label in FIT_LABELS:
                    bank_r.append(actual_r); bank_n.append(actual_n)
            fit_r, fit_n = torch.cat(bank_r) / np.sqrt(len(FIT_LABELS)), torch.cat(bank_n)
            trial['accepted'] = all(check['accepted'] for check in trial['checks'].values())
            predicted_gain, actual_gain = float(r.square().sum() - predicted_r.square().sum()), float(r.square().sum() - fit_r.square().sum())
            trial.update(predicted_joint_reduction=predicted_gain, actual_joint_reduction=actual_gain,
                reduction_ratio=None if predicted_gain <= 0 else actual_gain / predicted_gain,
                FIT_joint_vector_relative_error=float((fit_r - predicted_r).norm() / r.norm()),
                FIT_native_relative_error=((fit_n - predicted_n) / n).reshape(3, 3, 3).tolist())
            report['trials'].append(trial)
            np.savez(out / f'seed{seed_index}_MULTIBANK_M0.npz', **arrays)
            write_json(out / f'seed{seed_index}_MULTIBANK_M0.json', report)
            progress_file(out, counts, 'multibank_M0_candidate', seed=seed_index, trial=trial_index + 1,
                radius=radius, accepted=trial['accepted'])
            if trial['accepted']:
                selected = proposal
                report['selected_trial'], report['stop_reason'] = trial_index, 'first_all_FIT_CHECK_pass'
                arrays['uncommitted_candidate'] = proposal.cpu().numpy()
                break
            if trial_index + 1 < p['max_trials_per_seed']:
                J += torch.outer(fit_r - r - J @ step, step) / step.square().sum()
                N += torch.outer(fit_n - n - N @ step, step) / step.square().sum()
                counts['secant_matrix_corrections'] += 1
                if max(float((J @ step - (fit_r - r)).abs().max()), float((N @ step - (fit_n - n)).abs().max())) > 1e-11:
                    raise ValueError('Stacked multibank secant identity failed')
                radius *= p['rejection_radius_factor']
        if selected is not None:
            actual_r, actual_n, check = hard_check(p, model, selected, banks['FRESH'], targets['FRESH'],
                features['FRESH'], maps['FRESH'], baseline['FRESH'], old_risks['FRESH'], old_native['FRESH'],
                old_native['FRESH'], role_setup['FRESH'], counts)
            counts['FRESH_candidate_checks'] += 1
            arrays['FRESH_actual_residual'], arrays['FRESH_actual_native'] = actual_r.cpu().numpy(), actual_n.cpu().numpy()
            report['fresh_confirmation'], report['fresh_pass'] = check, check['accepted']
        np.savez(out / f'seed{seed_index}_MULTIBANK_M0.npz', **arrays)
        write_json(out / f'seed{seed_index}_MULTIBANK_M0.json', report)
        summaries[str(seed_index)] = dict(selected_trial=report['selected_trial'], fresh_pass=report['fresh_pass'],
            trials=len(report['trials']), stop_reason=report['stop_reason'])
    return dict(status='COMPLETE', summaries=summaries,
        two_seed_FRESH_pass=all(value['fresh_pass'] for value in summaries.values()),
        committed_parameter_updates=0, scientific_acceptance=False, automatic_training=False,
        original_final_acceptance_gates_unchanged=True, generator_and_readout_code_unchanged=True,native_distance_definition_unchanged=True,native_8_to_1_implementation_changed=True,native_timing=dict(NATIVE_TIMING),
        scope='Nested 4-to-32-draw comparison at frozen M0 with unchanged source/REAL rows, joint/native projections, generator and solver.',
        FRESH_role='Original fitting-role records/projections with 28 new draws per parent plus four replayed draws; no full-pipeline unseen claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered multibank-M0 config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned multibank-M0 input changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            progress_file(args.out, counts, 'multibank_M0_setup_loaded')
            result = run(p, setup, args.out, counts, started)
        check_budget(p, counts, started)
        result.update(config_sha256=args.config_sha256, execution_counts=dict(counts),
            elapsed_seconds=time.monotonic() - started, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out / 'RESULT.json', result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file()) > p['resources']['output_MiB'] * 1024**2:
            raise ValueError('Registered multibank-M0 output bound exceeded')
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_sha256=hashlib.sha256(
            (args.out / 'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
                committed_parameter_updates=0, scientific_acceptance=False, automatic_restart=False))
