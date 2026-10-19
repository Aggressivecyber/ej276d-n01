"""One registered S FIT evaluation; all original parents, banks and readouts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import osqp
import torch

import run_response039 as r
from run_response042 import local_restitution042
from run_response052 import conditional_sides052
from run_response053 import intrinsic_mean053, CROSSINGS053
from run_response060 import save_profiles060
from run_response069 import level_warp069
from run_response070 import split_warp070
from run_response044 import readouts044
from response071_diagnostics import native_decomposition071
from wp1_evaluation import output_q_bins, summary
from n01.finalization.smooth_mean import constraint_space, fit_mean, SOLVER_SETTINGS
from n01.finalization.linear_random_transport import raw_warp_delta, project_random
from n01.finalization.native_storage import save_native

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_case(setup, index, bank, mean, signal, T, projection, real_joint,
                  p, out, counts, name, base_arrays):
    label = f'seed{index}_{name}'
    device = signal.device
    intrinsic = intrinsic_mean053(mean)
    np.savez_compressed(out/(label+'_INTRINSIC_MEAN_ROWS.npz'),
        parent_row=np.arange(len(mean)), input_Q_bin=p['_input_bins'],
        intrinsic_peak_index=intrinsic['peak_index'], **intrinsic['features'],
        **intrinsic['crossings'], **{k+'_flag':v for k,v in intrinsic['flags'].items()})
    counts['intrinsic_mean_records'] += 1
    counts['intrinsic_mean_rows'] += len(mean)
    intrinsic_wave = mean.gather(1, torch.as_tensor(intrinsic['peak_index'], device=device)[:, None]
                                -16+torch.arange(256, device=device)[None, :]).cpu().numpy()
    reports, storage = {}, {}
    for h in r.HYPOTHESES:
        obs = r.observe_generated_carrier(signal+bank['noise'][h], ste=False,
                     q_floor=float(setup['loss'].q_floor), return_carrier=True)
        counts['generated_observation_rows'] += len(signal)
        counts['full_native_hypothesis_evaluations'] += 1
        wave, baseline, peaks = [obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
        report = r.hard_metrics(wave, baseline, setup['data']['real'], setup['data']['real_baseline100'],
                                setup['loss'], setup['context'])
        z = r.joint_embedding(wave, baseline, peaks, dict(T[h], q_floor=float(setup['loss'].q_floor)))
        report['values']['joint_morphology'] = float(r.column_W1(torch.as_tensor(z, device=device)@projection, real_joint[h], 16))
        report['values']['negative_lobe_area_Vns'] = r.raw_distance(np.maximum(-wave,0).sum(1), np.maximum(-setup['data']['real'],0).sum(1))['value']
        report['observation_flags'] = {k:dict(count=int(v.sum()), denominator=len(signal)) for k,v in obs['flags'].items()}
        reports[h] = report
        r.write_json(out/(label+'_'+h+'_METRICS.json'), report)
        storage[h] = save_native(out/(label+'_'+h+'_NATIVE.npz'), obs)
        counts['lossless_native_records'] += 1
        counts['lossless_native_rows'] += len(mean)
        features, flags = readouts044(wave)
        morph = r.morphology018(wave)
        bins = output_q_bins(features['Q_Vns'], setup['loss'])
        if name == 'BASE054':
            base_arrays[h] = dict(native_peak_index=peaks.copy(), output_Q_bin=bins.copy(),
                                   features={k:v.copy() for k,v in features.items()})
        np.savez_compressed(out/(label+'_'+h+'_NATIVE_ROWS.npz'), parent_row=np.arange(len(mean)),
            input_Q_bin=p['_input_bins'], native_peak_index=peaks, native_output_Q_bin=bins,
            **features, **{k+'_flag':v for k,v in flags.items()},
            **{k+'_absolute_ns':morph[k]+peaks-16 for k in CROSSINGS053})
        counts['feature_records'] += 1
        counts['feature_rows'] += len(mean)
        if name == 'S':
            fixed, random, random_delta, base_signal, mean_signal = p['_S_decomposition']
            diagnostic, check_base = native_decomposition071(fixed, mean, random, random_delta,
                base_signal, mean_signal, signal, bank['noise'][h], obs, float(setup['loss'].q_floor))
            counts['diagnostic_BASE054_observation_calls'] += 1
            counts['diagnostic_BASE054_observation_rows'] += len(mean)
            if not np.array_equal(diagnostic['BASE054_native_peak_index'], base_arrays[h]['native_peak_index']):
                raise RuntimeError('S decomposition BASE054 peak replay differs')
            for feature in ('Q_Vns','tail_Vns'):
                if not np.array_equal(diagnostic['BASE054_'+feature], base_arrays[h]['features'][feature]):
                    raise RuntimeError('S decomposition BASE054 integral replay differs')
            if max(np.max(np.abs(diagnostic[k])) for k in ('Q_closure_Vns','tail_closure_Vns')) > 1e-12:
                raise RuntimeError('S native window decomposition does not close')
            np.savez_compressed(out/(label+'_'+h+'_DECOMPOSITION_ROWS.npz'), **diagnostic)
            r.write_json(out/(label+'_'+h+'_DECOMPOSITION.json'),
                dict(rows=len(mean), row_summaries={k:summary(v) for k,v in diagnostic.items() if k!='parent_row'}))
            counts['diagnostic_decomposition_records'] += 1
            counts['diagnostic_decomposition_rows'] += len(mean)
            del diagnostic, check_base
        if h == 'fast':
            save_profiles060(wave, np.array([p['_input_bins'],bins]), np.arange(len(mean)),
                             float(setup['loss'].q_floor), out, label+'_fast',
                             ['A0_input_Q_quantile','own_native_output_Q'], counts)
            save_profiles060(intrinsic_wave, np.array([p['_input_bins']]), np.arange(len(mean)),
                             float(setup['loss'].q_floor), out, label+'_intrinsic',
                             ['A0_input_Q_quantile'], counts)
        r.check_resources(p, out, p['_started'])
        r.write_json(out/'PROGRESS.json', dict(phase='native_evaluation', seed_index=index,
            case=name, hypothesis=h, elapsed_seconds=time.monotonic()-p['_started'], execution_counts=dict(counts)))
    r.write_json(out/(label+'_STORAGE.json'), storage)
    return reports


@torch.no_grad()
def run(p, out, device, counts):
    p['_started'] = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    r.write_json(out/'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        run_id=p['run_id'], config_sha256=p['_config_sha256'], resources=p['resources']))
    setup = r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()), device, counts)
    for model in setup['models']: model.set_phase('frozen').eval()
    T = json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    A0 = setup['encoding']['original']['signal_pre']
    q = r.canonical_charge008(A0)
    if hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != p['input_Q_array_sha256']:
        raise RuntimeError('S input Q differs from original full TRAIN')
    with np.load(ROOT/p['input_reference_file']) as reference:
        p['_input_bins'] = reference['input_Q_bin'].copy()
        if not np.array_equal(q.cpu().numpy(), reference['A0_canonical_Q_Vns']):
            raise RuntimeError('S original input population differs')
    template = json.loads((ROOT/p['template_file']).read_text())
    replay = json.loads((ROOT/p['replay_file']).read_text())['reports']
    replay054 = json.loads((ROOT/p['baseline054_replay_file']).read_text())
    real, real_baseline = setup['data']['real'], setup['data']['real_baseline100']
    features, flags = readouts044(real)
    np.savez_compressed(out/'REAL_ROWS.npz', row=np.arange(len(real)), **features,
                        **{k+'_flag':v for k,v in flags.items()})
    counts['REAL_feature_rows'] += len(real)
    save_profiles060(real, np.array([output_q_bins(features['Q_Vns'],setup['loss'])]), np.arange(len(real)),
                     float(setup['loss'].q_floor),out,'REAL',['own_native_output_Q'],counts)
    projection = r.projection039(p['joint_projection_fit_seed'], device)
    real_joint = {h:torch.as_tensor(r.joint_embedding(real, real_baseline, np.full(len(real),16),
                  dict(T[h],q_floor=float(setup['loss'].q_floor))),device=device)@projection for h in r.HYPOTHESES}
    r.write_json(out/'DEFINITIONS.json', dict(T_by_hypothesis=T, q_floor=float(setup['loss'].q_floor),
        settings=p['S'], original_contract=p['contract'], all_rows_retained=True,
        SOURCE_REAL_event_pairing=False, final_noisy_peak_frozen=False,
        random_linearity_scope='fixed input/mean, preADC; no per-draw clipping',
        full_native_scientific_outputs='all four systems, both heads, all H; ADC codes plus float64 XOR residuals'))
    results = []
    for index in range(2):
        bank = r.make_population_bank(setup,index,'FIT',0,p,counts)
        model = setup['models'][index]
        model.mean_adapter[-1].bias.copy_(torch.as_tensor(p['base_bias'],device=device,dtype=model.mean_adapter[-1].bias.dtype))
        counts['mean_operator_calls'] += 1
        counts['mean_operator_rows'] += len(q)
        adapter = model._mean_response(setup['encoding'],enabled=True)
        delta, _, _ = local_restitution042(adapter['delta_mean_pre_V'],p['base_restitution_fraction'])
        transport, _, _, _, _, _ = conditional_sides052(A0+delta,q,1.,p['input_Q_reference_Vns'],p['side_coefficients_ns'])
        fixed_delta = delta+transport
        fixed = A0+fixed_delta
        random = bank['M0']-A0
        base_signal = r._ExactZeroMeanAddition015.apply(bank['M0'],fixed_delta)
        base_arrays = {}
        m0 = evaluate_case(setup,index,bank,A0,bank['M0'],T,projection,real_joint,p,out,counts,'M0',base_arrays)
        if m0 != replay[str(index)]['M0']: raise RuntimeError('S M0 exact full metric replay differs')
        counts['reference_case_replays'] += 1
        base = evaluate_case(setup,index,bank,fixed,base_signal,T,projection,real_joint,p,out,counts,'BASE054',base_arrays)
        if base != replay054[str(index)]: raise RuntimeError('S BASE054 exact full metric replay differs')
        counts['reference_case_replays'] += 1
        d070, r070, _, _, _, _ = split_warp070(fixed,random,q,template)
        signal070 = r._ExactZeroMeanAddition015.apply(r._ExactZeroMeanAddition015.apply(bank['M0'],fixed_delta+d070),r070)
        report070 = evaluate_case(setup,index,bank,fixed+d070,signal070,T,projection,real_joint,p,out,counts,'070',base_arrays)
        if report070 != replay[str(index)]['SPLITWARP']['metrics']: raise RuntimeError('S070 exact full metric replay differs')
        counts['reference_case_replays'] += 1
        target, _, warp_rows, queries, source_knots, target_knots = level_warp069(fixed,random,q,template)
        counts['level_warp_target_rows'] += len(q)
        means, target_np, random_np = fixed.cpu().numpy(), target[:,110:176].cpu().numpy(), random[:,110:176].cpu().numpy()
        query_np = queries.cpu().numpy()
        scale_np = warp_rows['amplitude_scale'].cpu().numpy()
        scales = p['S']['normalization_by_head'][str(index)]['values']
        intrinsic_base = intrinsic_mean053(fixed)
        actual_scales = [float(np.median(intrinsic_base['features'][k])) for k in ('peak_voltage_V','Q_Vns','tail_Vns')]
        if actual_scales != scales: raise RuntimeError('S full TRAIN scale binding changed')
        mean_delta = np.zeros((len(q),66)); random_delta = np.zeros_like(mean_delta)
        row_info = []
        r.write_json(out/'PROGRESS.json',dict(phase='S_QP',seed_index=index,completed_rows=0,
            reference_replays_complete=True,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-p['_started']))
        for row in range(len(q)):
            space = constraint_space(means[row],scales,p['S']['radius'],p['S']['lambda_s'],p['S']['lambda_r'])
            # Persist current parent on failure; no discarded row or automatic retry.
            try:
                dm, coefficient, info = fit_mean(means[row],target_np[row],space)
                raw = raw_warp_delta(random_np[row],query_np[row],scale_np[row],space['allowed'])
                dr = project_random(raw,space)
                equalities = [abs(dm[space['front']].sum()),abs(dm[space['tail']].sum()),
                              abs(dr[space['front']].sum()),abs(dr[space['tail']].sum())]
                if max(equalities)>1e-12 or not np.isfinite(dr).all(): raise RuntimeError('S random projection equality/nonfinite failure')
            except (RuntimeError,np.linalg.LinAlgError) as exc:
                r.write_json(out/'FAILED_PARENT.json',dict(seed_index=index,parent_row=row,error=str(exc),
                    equality_rank=space['equality_rank'],free_dimensions=space['basis'].shape[1],
                    parents_processed=len(row_info),parent_removed=False,automatic_retry=False))
                np.savez_compressed(out/'FAILED_PARENT_ARRAYS.npz',mean=means[row],target_delta=target_np[row],
                    random_local=random_np[row],query=query_np[row],basis=space['basis'],hessian=space['hessian'],
                    inequality=space['inequality'],lower=space['lower'])
                raise
            mean_delta[row],random_delta[row] = dm,dr
            info.update(parent_row=row,reference_peak_index=space['peak'],maximum_equality_residual_Vns=max(equalities),
                        maximum_mean_delta_V=float(np.max(np.abs(dm))),maximum_random_delta_V=float(np.max(np.abs(dr))))
            row_info.append(info)
            counts['S_parent_rows'] += 1
            if info['status']=='solved': counts['event_QP_solves'] += 1
            else: counts['zero_DOF_parent_rows'] += 1
            counts['linear_random_projection_rows'] += 1
            if (row+1)%128==0 or row+1==len(q):
                r.check_resources(p,out,p['_started'])
                r.write_json(out/'PROGRESS.json',dict(phase='S_QP',seed_index=index,completed_rows=row+1,
                    reference_replays_complete=True,execution_counts=dict(counts),elapsed_seconds=time.monotonic()-p['_started']))
        r.write_json(out/f'seed{index}_S_QP_ROWS.json',row_info)
        correction = torch.zeros_like(fixed); correction[:,110:176] = torch.as_tensor(mean_delta,device=device)
        dr_tensor = torch.zeros_like(random); dr_tensor[:,110:176] = torch.as_tensor(random_delta,device=device)
        mean = fixed+correction
        mean_signal = r._ExactZeroMeanAddition015.apply(bank['M0'],fixed_delta+correction)
        signal = r._ExactZeroMeanAddition015.apply(mean_signal,dr_tensor)
        p['_S_decomposition'] = (fixed,random,dr_tensor,base_signal,mean_signal)
        geometry = dict(support_exact=bool((correction[:,:110]==0.).all() and (correction[:,176:]==0.).all()),
            boundary_values_and_first_differences_exact=bool((correction[:,[110,111,174,175]]==0.).all()),
            deterministic_peak_exact=torch.equal(mean[:,112:144].argmax(1),fixed[:,112:144].argmax(1)),
            nonpositive_mean_samples_exact=torch.equal(mean[fixed<=0.],fixed[fixed<=0.]),
            prefix_exact=torch.equal(signal[:,:100],bank['M0'][:,:100]),
            random_support_exact=bool((dr_tensor[:,:110]==0.).all() and (dr_tensor[:,176:]==0.).all()),
            max_reference_area_residual_Vns=max(item['maximum_equality_residual_Vns'] for item in row_info),
            canonical_charge_residual_Vns=float((r.canonical_charge008(mean)-q).abs().max()),
            random_canonical_charge_residual_Vns=float(r.canonical_charge008(dr_tensor).abs().max()),
            free_dimensions=dict(Counter(item['free_dimensions'] for item in row_info)),
            full_population_rows=len(q),zero_DOF_rows=sum(item['free_dimensions']==0 for item in row_info))
        if not all(geometry[k] for k in ('support_exact','boundary_values_and_first_differences_exact',
            'deterministic_peak_exact','nonpositive_mean_samples_exact','prefix_exact','random_support_exact')):
            raise RuntimeError('S assembled geometry invariant failed')
        if geometry['canonical_charge_residual_Vns']>1e-10 or geometry['random_canonical_charge_residual_Vns']>1e-12:
            raise RuntimeError('S canonical charge invariant failed')
        np.savez_compressed(out/f'seed{index}_S_OPERATOR.npz',parent_row=np.arange(len(q)),fixed_mean_V=means,
            mean_correction_local_V=mean_delta,original_random_local_V=random_np,random_correction_local_V=random_delta,
            inverse_query_absolute=query_np,warp_scale=scale_np,source_knots=source_knots.cpu().numpy(),target_knots=target_knots.cpu().numpy())
        report = evaluate_case(setup,index,bank,mean,signal,T,projection,real_joint,p,out,counts,'S',base_arrays)
        decision = r.objective039(report,m0,T,p['contract'])
        usefulness = {h:dict(wave_relative_gain=(m0[h]['values']['wave']-report[h]['values']['wave'])/m0[h]['values']['wave'],
            joint_relative_gain=(m0[h]['values']['joint_morphology']-report[h]['values']['joint_morphology'])/m0[h]['values']['joint_morphology'],
            morphology_mean_T_gain=float(np.mean([(m0[h]['values'][k]-report[h]['values'][k])/T[h][k] for k in r.MORPH]))) for h in r.HYPOTHESES}
        for u in usefulness.values():
            u['passes_point_targets']=(u['wave_relative_gain']>=p['contract']['useful_wave_relative_gain']
                and u['joint_relative_gain']>=p['contract']['useful_joint_relative_gain']
                and u['morphology_mean_T_gain']>=p['contract']['useful_morphology_mean_T_gain'])
        record=dict(seed_index=index,metrics=report,geometry=geometry,decision=decision,point_usefulness=usefulness)
        r.write_json(out/f'seed{index}_S.json',record)
        results.append(record)
        counts['candidate_head_evaluations'] += 1
        del p['_S_decomposition']
        del bank, base_arrays
    exact=p['operation_counts_exact']
    if any(counts[k]!=v for k,v in exact.items()): raise RuntimeError('S exact operation accounting mismatch')
    if any(counts[k]>v for k,v in p['operation_bounds'].items()): raise RuntimeError('S operation budget exceeded')
    if counts['event_QP_solves']+counts['zero_DOF_parent_rows']!=13186: raise RuntimeError('S parent disposition incomplete')
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict())!=r.state_digest(setup['heads'][index]): raise RuntimeError('S M0 changed')
    if r.state_digest(setup['baseline'].state_dict())!=setup['frozen_A_digest']: raise RuntimeError('S A0 changed')
    if r.state_digest(setup['extension'].state_dict())!=setup['frozen005_digest']: raise RuntimeError('S005 changed')
    passed=all(x['decision']['feasible'] and all(u['passes_point_targets'] for u in x['point_usefulness'].values()) for x in results)
    result=dict(schema='n01/finalization-S-fit/v1',run_id=p['run_id'],results=results,execution_counts=dict(counts),
        stage_outcome='FIT_USEFUL_S_PENDING_CONFIRMATION' if passed else 'S_FIT_COMPLETE_REQUIRES_DECISION',
        original_point_targets_passed=passed,confirmation_banks_opened=0,DEV_opened=False,
        all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        frozen_A0_M0_005_verified=True,scientific_status='response_match_not_established',
        mean_stage_accepted=False,full_random_response_acceptance=False,elapsed_seconds=time.monotonic()-p['_started'])
    r.write_json(out/'RESULT.json',result)
    r.check_resources(p,out,p['_started'])
    r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
        result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-p['_started']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256: raise ValueError('S config hash mismatch')
    p=json.loads(args.config.read_text())
    for name,digest in p['source_pins'].items():
        if r.sha256(ROOT/name)!=digest: raise ValueError('S source changed: '+name)
    if osqp.__version__!=p['S']['solver_version'] or SOLVER_SETTINGS!=p['S']['solver_settings']:
        raise ValueError('S solver lock changed')
    for package,version in p['runtime_dependencies'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('S runtime dependency changed: '+package)
    p['_config_sha256']=args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    counts=Counter(dict.fromkeys(p['operation_bounds'],0))
    try: run(p,args.out,device,counts)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),
                execution_counts=dict(counts),automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__': main()
