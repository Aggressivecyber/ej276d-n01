"""Frozen front/compensator factorial and ordered observation attribution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import run_response039 as r
from run_response036 import _wave_views, _pair_projection, VIEWS, COORDINATE_VIEWS, SHORT_LAGS, WAVE_WEIGHTS
from wp1_evaluation import summary

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('mean_only', 'mean_plus_M0_random', 'plus_electronic_noise', 'ADC_at_M0_anchor', 'native_ADC_reselected')
CROP_EDGES = (0, 16, 46, 112, 192, 256)
CARRIER_EDGES = (0, 100, 112, 144, 190, 254, 318, 384)


@torch.no_grad()
def wave_decomposition(wave, real, loss, context):
    fv, rv = [_wave_views(x.double(), loss.peak_floor) for x in (wave, real)]
    f = {name: loss._z(name, fv[name]) for name in VIEWS}
    b = {name: loss._z(name, rv[name]) for name in VIEWS}
    pieces, profiles = {}, {}
    for name in VIEWS:
        score = float(r.column_W1(f[name]@context.banks['wave:'+name].to(f[name]),
                                  b[name]@context.banks['wave:'+name].to(b[name]), 16))
        weight = WAVE_WEIGHTS['swd']/len(VIEWS)
        pieces['swd/'+name] = dict(distance=score, weight=weight, weighted_score=score*weight)
        profiles[name] = dict(fake_mean=fv[name].mean(0).cpu().numpy(), fake_std=fv[name].std(0, correction=0).cpu().numpy(),
                              real_mean=rv[name].mean(0).cpu().numpy(), real_std=rv[name].std(0, correction=0).cpu().numpy())
    for name in COORDINATE_VIEWS:
        score = float(r.column_W1(f[name], b[name], 16)); weight = WAVE_WEIGHTS['coordinate']/len(COORDINATE_VIEWS)
        pieces['coordinate/'+name] = dict(distance=score, weight=weight, weighted_score=score*weight)
        for lag in SHORT_LAGS:
            directions = context.banks[f'lag:{name}:{lag}']
            score = float(r.column_W1(_pair_projection(f[name], lag, directions), _pair_projection(b[name], lag, directions), 16))
            weight = WAVE_WEIGHTS['short_lag']/(len(COORDINATE_VIEWS)*len(SHORT_LAGS))
            pieces[f'short_lag/{name}/{lag}'] = dict(distance=score, weight=weight, weighted_score=score*weight)
    return dict(pieces=pieces, reconstructed_wave=sum(x['weighted_score'] for x in pieces.values()), raw_view_profiles=profiles)


def fixed_anchor_stages(mean, signal, pre_adc, observation, anchor_indices):
    carriers = (mean, signal, pre_adc, observation['carrier_adc'])
    return [(value-value[:, :100].mean(1, keepdim=True)).gather(1, anchor_indices) for value in carriers]+[observation['window_V']]


def ordered_contrasts(values, reference):
    delta = np.asarray(values)-np.asarray(reference)
    parts = np.concatenate((delta[:1], np.diff(delta, axis=0)), axis=0)
    error = float(np.max(np.abs(parts.sum(axis=0)-delta[-1])))
    return dict(stage_delta=delta, ordered_contributions=parts, closure_max_error=error)


@torch.no_grad()
def run041(p, out, device, progress):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    r.write_json(out/'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), run_id=p['run_id'],
                 config_sha256=p['_config_sha256'], pid=os.getpid(), resources=p['resources'], phase='setup'))
    setup = r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()), device, progress)
    T = json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    for model in setup['models']: model.set_phase('frozen').eval()
    real_wave, real_baseline = setup['data']['real'], setup['data']['real_baseline100']
    real_tensor = torch.as_tensor(real_wave, device=device)
    real_negative = np.maximum(-real_wave, 0).sum(1)
    projection = r.projection039(p['joint_projection_fit_seed'], device)
    real_joint = {h:torch.as_tensor(r.joint_embedding(real_wave, real_baseline, np.full(len(real_wave),16),
                    dict(T[h], q_floor=float(setup['loss'].q_floor))), device=device)@projection for h in r.HYPOTHESES}
    replays = json.loads((ROOT/p['replay_file']).read_text())['reports']
    all_metrics, decomposed, replay_checks = {}, {}, []
    A0 = setup['encoding']['original']['signal_pre']
    r.write_json(out/'REAL_PROFILE.json', dict(rows=len(real_wave), mean=real_wave.mean(0), std=real_wave.std(0),
                 negative_area=summary(real_negative), flags={k:int(v.sum()) for k,v in r.morphology018(real_wave).items() if v.dtype == bool}))
    for index in range(2):
        bank = r.make_population_bank(setup, index, 'FIT', 0, p, progress)
        base = {}; all_metrics[index] = {}; decomposed[index] = {}
        for case in p['cases']:
            r.check_resources(p, out, started)
            name = case['name']; model = setup['models'][index]
            model.mean_adapter[-1].bias.copy_(torch.as_tensor(case['bias'], device=device, dtype=model.mean_adapter[-1].bias.dtype))
            adapter = model._mean_response(setup['encoding'], enabled=True)
            progress['mean_operator_calls'] += 1; progress['mean_operator_rows'] += len(A0)
            mean = adapter['mean_signal_pre_V']
            signal = r._ExactZeroMeanAddition015.apply(bank['M0'], adapter['delta_mean_pre_V'])
            geometry = dict(max_time_displacement_ns=float((adapter['time_queries']-torch.arange(384,device=device)).abs().max()),
                max_canonical_Q_residual_Vns=float((r.canonical_charge008(mean)-r.canonical_charge008(A0)).abs().max()),
                prefix_bitwise_unchanged=torch.equal(signal[:, :100], bank['M0'][:, :100]),
                mean_delta_by_absolute_region={f'{a}:{b}':summary(adapter['delta_mean_pre_V'][:,a:b].sum(1).cpu().numpy())
                                               for a,b in zip(CARRIER_EDGES[:-1],CARRIER_EDGES[1:])})
            all_metrics[index][name] = {}; decomposed[index][name] = {}
            for h in r.HYPOTHESES:
                pre_adc = signal+bank['noise'][h]
                obs = r.observe_generated_carrier(pre_adc, ste=False, q_floor=float(setup['loss'].q_floor), return_carrier=True)
                progress['generated_observation_rows'] += len(A0)
                wave, baseline, peaks = [obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
                metrics = r.hard_metrics(wave,baseline,real_wave,real_baseline,setup['loss'],setup['context'])
                z = r.joint_embedding(wave,baseline,peaks,dict(T[h],q_floor=float(setup['loss'].q_floor)))
                metrics['values']['joint_morphology'] = float(r.column_W1(torch.as_tensor(z,device=device)@projection,real_joint[h],16))
                metrics['values']['negative_lobe_area_Vns'] = r.raw_distance(np.maximum(-wave,0).sum(1),real_negative)['value']
                metrics['observation_flags'] = {k:dict(count=int(v.sum()),denominator=len(A0)) for k,v in obs['flags'].items()}
                progress['full_native_hypothesis_evaluations'] += 1
                parts = wave_decomposition(obs['window_V'],real_tensor,setup['loss'],setup['context'])
                progress['wave_decomposition_evaluations'] += 1
                reconstruction_error = abs(parts['reconstructed_wave']-metrics['values']['wave'])
                if reconstruction_error > 1e-12: raise RuntimeError('041 native wave decomposition does not close')
                anchor = obs['crop_indices'] if name == 'M0' else base[h]['indices']
                stages = fixed_anchor_stages(mean,signal,pre_adc,obs,anchor)
                details, negative, censored, widths, w1s = {}, [], [], [], []
                for stage_name, stage_wave in zip(STAGES,stages):
                    array = stage_wave.cpu().numpy(); area = np.maximum(-array,0).sum(1); morph = r.morphology018(array)
                    negative.append(area); censored.append(morph['rise_undefined']); widths.append(morph['FWHM_undefined'])
                    w1 = r.raw_distance(area,real_negative)['value']; w1s.append(w1)
                    details[stage_name] = dict(negative_area=summary(area),negative_area_W1_to_REAL=w1,
                        rise_undefined_count=int(morph['rise_undefined'].sum()),FWHM_undefined_count=int(morph['FWHM_undefined'].sum()),
                        rise_prepeak_censored_count=int(morph['rise_prepeak_censored'].sum()),
                        population_mean_V=array.mean(0),population_std_V=array.std(0),
                        negative_area_by_crop_region={f'{a}:{b}':summary(np.maximum(-array[:,a:b],0).sum(1))
                                                     for a,b in zip(CROP_EDGES[:-1],CROP_EDGES[1:])})
                    progress['counterfactual_stage_summaries'] += 1
                    progress['counterfactual_stage_rows'] += len(A0)
                negative, censored, widths = np.asarray(negative), np.asarray(censored), np.asarray(widths)
                if name == 'M0':
                    base[h] = dict(indices=anchor,negative=negative,censored=censored,w1s=w1s,wave=wave.copy(),peaks=peaks.copy())
                area_contrasts = ordered_contrasts(negative,base[h]['negative'])
                w1_contrasts = ordered_contrasts(w1s,base[h]['w1s'])
                if max(area_contrasts['closure_max_error'],w1_contrasts['closure_max_error']) > 1e-12:
                    raise RuntimeError('041 ordered attribution does not close')
                new = censored[-1]&~base[h]['censored'][-1]
                resolved = ~censored[-1]&base[h]['censored'][-1]
                peak_delta = peaks-base[h]['peaks']; positions,counts = np.unique(peak_delta,return_counts=True)
                selected = np.unique(np.r_[np.flatnonzero(new)[:16],np.argsort(-(negative[-1]-base[h]['negative'][-1]),kind='stable')[:16]])
                np.savez_compressed(out/f'seed{index}_{name}_{h}_rows.npz',parent_row=np.arange(len(A0)),
                    negative_area_by_stage_Vns=negative,rise_undefined_by_stage=censored,FWHM_undefined_by_stage=widths,
                    native_peak_index=peaks,M0_peak_index=base[h]['peaks'])
                np.savez_compressed(out/f'seed{index}_{name}_{h}_examples.npz',parent_row=selected,
                    M0_wave_V=base[h]['wave'][selected],native_wave_V=wave[selected],
                    mean_pre_V=mean[selected].cpu().numpy(),M0_random_pre_V=(bank['M0']-A0)[selected].cpu().numpy(),
                    noise_pre_V=bank['noise'][h][selected].cpu().numpy())
                report = dict(seed_index=index,case=name,hypothesis=h,metrics=metrics,geometry=geometry,
                    wave_decomposition=parts,wave_reconstruction_absolute_error=reconstruction_error,
                    stage_definitions=list(STAGES),stages=details,
                    ordered_negative_area_effects={stage:summary(values) for stage,values in zip(STAGES,area_contrasts['ordered_contributions'])},
                    ordered_negative_area_W1_effects=dict(zip(STAGES,w1_contrasts['ordered_contributions'].tolist())),
                    area_decomposition_closure_max_error=area_contrasts['closure_max_error'],
                    W1_decomposition_closure_error=w1_contrasts['closure_max_error'],
                    native_rise_transition=dict(new_undefined=int(new.sum()),resolved_undefined=int(resolved.sum()),denominator=len(A0)),
                    peak_shift_counts=dict(zip(map(str,positions),map(int,counts))),
                    diagnostic_only=True,physical_acceptance=False)
                r.write_json(out/f'seed{index}_cases'/f'{name}_{h}.json',report)
                all_metrics[index][name][h] = metrics; decomposed[index][name][h] = parts['pieces']
            if name != 'compensator':
                expected = replays[str(index)][name]; diffs = []
                for h in r.HYPOTHESES:
                    actual = all_metrics[index][name][h]
                    for key,value in actual['values'].items():
                        before = expected[h]['values'][key]
                        if value is None or before is None:
                            if value != before: raise RuntimeError('041 replay undefined population changed')
                        else: diffs.append(abs(value-before))
                    for family in ('fake_flags','observation_flags'):
                        if actual[family] != expected[h][family]: raise RuntimeError('041 replay flags changed')
                if max(diffs)>1e-12: raise RuntimeError('041 native FIT replay mismatch')
                replay_checks.append(dict(seed=index,case=name,max_scalar_absolute_difference=max(diffs),flags_exact=True))
                r.write_json(out/'REPLAY_CHECKS.json',replay_checks)
            progress['completed_full_population_cases'] += 1
            r.write_json(out/'PROGRESS.json',dict(phase='mechanism_attribution',seed_index=index,case=name,
                         elapsed_seconds=time.monotonic()-started,execution_counts=dict(progress)))
        del bank,base
    factorial = []
    for index in range(2):
        for h in r.HYPOTHESES:
            values = {name:all_metrics[index][name][h]['values'] for name in ('M0','front','compensator','combined')}
            effects = {}
            for key in ('wave','joint_morphology','negative_lobe_area_Vns'):
                a,f,c,b = [values[n][key] for n in ('M0','front','compensator','combined')]
                effects[key] = dict(M0=a,front=f,compensator=c,combined=b,front_effect=f-a,
                                    compensator_effect=c-a,interaction=b-f-c+a,combined_effect=b-a)
            piece_effects = {}
            for key in decomposed[index]['M0'][h]:
                a,f,c,b = [decomposed[index][n][h][key]['weighted_score'] for n in ('M0','front','compensator','combined')]
                piece_effects[key] = dict(front_effect=f-a,compensator_effect=c-a,interaction=b-f-c+a,combined_effect=b-a)
            factorial.append(dict(seed=index,hypothesis=h,metric_effects=effects,wave_piece_effects=piece_effects))
    for index,model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict()) != r.state_digest(setup['heads'][index]): raise RuntimeError('041 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest']: raise RuntimeError('041 A0 changed')
    if r.state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']: raise RuntimeError('041 005 changed')
    result = dict(schema='n01/response041-mechanism-result/v1',run_id=p['run_id'],factorial=factorial,replay_checks=replay_checks,
        execution_counts=dict(progress),elapsed_seconds=time.monotonic()-started,frozen_A0_M0_005_verified=True,
        all_TRAIN_SOURCE_parents=6593,all_TRAIN_REAL_rows=21047,all_parent_count_weights_preserved=True,
        diagnostic_complete=True,model_selected=False,scientific_status='response_match_not_established',
        DEV_opened=False,physical_acceptance=False,full_random_response_acceptance=False,
        next_step='Use measured component costs and ordered physical effects to choose the next minimal response correction; no automatic search extension.')
    r.write_json(out/'RESULT.json',result)
    r.write_json(out/'COMPLETE.json',dict(UTC=datetime.now(timezone.utc).isoformat(),native_exit_expected=0,
                 result_sha256=r.sha256(out/'RESULT.json'),elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256: raise ValueError('041 config hash mismatch')
    p=json.loads(args.config.read_text())
    for path,digest in p['source_pins'].items():
        if r.sha256(ROOT/path)!=digest: raise ValueError('041 source pin mismatch: '+path)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']: raise ValueError('041 setup pin mismatch')
    p['_config_sha256']=args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(('checkpoint_loads','TRAIN_archives_decoded','PCA_archives_decoded','DEV_archives_decoded',
        'AD_queries','optimizer_steps','old_A_calls','old_A_rows','response_calls','response_rows','noise005_calls','noise005_rows',
        'extension_rows','full_population_banks','mean_operator_calls','mean_operator_rows','generated_observation_rows',
        'full_native_hypothesis_evaluations','wave_decomposition_evaluations','counterfactual_stage_summaries',
        'counterfactual_stage_rows','completed_full_population_cases'),0))
    try: run041(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),
                         automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__': main()
