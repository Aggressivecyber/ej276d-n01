"""Fixed045 replay; transport area and stage-wise global-peak/readout attribution."""
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
from run_response041 import fixed_anchor_stages, ordered_contrasts
from run_response042 import local_restitution042
from run_response045 import conditional_width045
from run_response044 import readouts044, STAGES044, FIELDS044
from wp1_evaluation import summary

ROOT = Path(__file__).resolve().parents[1]
CARRIERS046 = ('mean', 'plus_M0_random', 'plus_electronic_noise', 'ADC')


def carrier_peaks046(value):
    centered = value-value[:, :100].mean(1, keepdim=True)
    global_peak = centered.argmax(1)
    selected_peak = centered[:, 112:144].argmax(1)+112
    return dict(global_peak_index=global_peak.cpu().numpy(),
        selected_peak_index=selected_peak.cpu().numpy(),
        global_peak_V=centered.gather(1, global_peak[:, None])[:, 0].cpu().numpy(),
        global_peak_outside_search=((global_peak<112)|(global_peak>=144)).cpu().numpy())


@torch.no_grad()
def run046(p, out, device, progress):
    started = time.monotonic(); out.mkdir(parents=True, exist_ok=False)
    r.write_json(out/'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), run_id=p['run_id'],
        config_sha256=p['_config_sha256'], pid=os.getpid(), resources=p['resources'], phase='setup'))
    setup = r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()), device, progress)
    for model in setup['models']: model.set_phase('frozen').eval()
    T = json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    real_wave, real_baseline = setup['data']['real'], setup['data']['real_baseline100']
    real_features, real_flags = readouts044(real_wave)
    projection = r.projection039(p['joint_projection_fit_seed'], device)
    real_joint = {h:torch.as_tensor(r.joint_embedding(real_wave, real_baseline, np.full(len(real_wave), 16),
        dict(T[h], q_floor=float(setup['loss'].q_floor))), device=device)@projection for h in r.HYPOTHESES}
    replays = json.loads((ROOT/p['replay_file']).read_text())['reports']
    A0 = setup['encoding']['original']['signal_pre']
    input_q = r.canonical_charge008(A0).cpu().numpy()
    input_edges = np.quantile(input_q, np.arange(1, 8)/8)
    input_bins = np.searchsorted(input_edges, input_q, side='right')
    np.savez_compressed(out/'INPUT_CONTENT.npz', parent_row=np.arange(len(A0)),
        A0_canonical_Q_Vns=input_q, input_Q_quantile_edges=input_edges, input_Q_bin=input_bins)
    r.write_json(out/'INPUT_AND_REAL_DEFINITIONS.json', dict(SOURCE_rows=len(A0), REAL_rows=len(real_wave),
        input_Q_definition='Canonical[110,366) integral of frozen A0 mean; SOURCE-only quantiles, not deposited energy truth',
        input_Q_edges=input_edges, input_Q_bin_counts=np.bincount(input_bins, minlength=8),
        input_bins_do_not_define_REAL_labels=True, output_Q_conditions_remain_descriptive=True,
        REAL_pre_ADC_unknown=True, stage_order=list(STAGES044), carrier_stage_order=list(CARRIERS046), ordered_contrasts_not_unique_causal_effects=True,
        REAL_features={key:summary(value) for key,value in real_features.items()},
        REAL_flags={key:dict(count=int(value.sum()),denominator=len(real_wave)) for key,value in real_flags.items()}))
    replay_checks, case_summaries = [], []
    for index in range(2):
        bank = r.make_population_bank(setup, index, 'FIT', 0, p, progress)
        base = {}
        for case in p['cases']:
            r.check_resources(p, out, started)
            name = case['name']; model = setup['models'][index]
            model.mean_adapter[-1].bias.copy_(torch.as_tensor(case['bias'], device=device,
                dtype=model.mean_adapter[-1].bias.dtype))
            adapter = model._mean_response(setup['encoding'], enabled=True)
            delta, correction, _ = local_restitution042(adapter['delta_mean_pre_V'], case['restitution_fraction'])
            transport, coefficient, mapped = conditional_width045(A0+delta, r.canonical_charge008(A0),
                case['width_strength'], p['input_Q_reference_Vns'])
            delta = delta+transport; mean = A0+delta
            signal = r._ExactZeroMeanAddition015.apply(bank['M0'], delta)
            progress['mean_operator_calls'] += 1; progress['mean_operator_rows'] += len(A0)
            segment_areas = torch.stack([transport[:, a:b].sum(1) for a,b in ((110,126),(126,144),(144,159))],1).cpu().numpy()
            minimum_intervals = mapped.diff(dim=1).min(1).values.cpu().numpy()
            geometry = dict(max_canonical_Q_residual_Vns=float((r.canonical_charge008(mean)-r.canonical_charge008(A0)).abs().max()),
                prefix_bitwise_unchanged=torch.equal(signal[:, :100], bank['M0'][:, :100]),
                transport_support_exact=bool((transport[:, :110]==0).all() and (transport[:, 159:]==0).all()),
                transport_integral_max_residual_Vns=float(transport.sum(1).abs().max()),
                segment_area_closure_max_error_Vns=float(np.abs(segment_areas.sum(1)-transport.sum(1).cpu().numpy()).max()),
                minimum_mapped_interval_ns=float(minimum_intervals.min()),
                maximum_absolute_transport_V=float(transport.abs().max()))
            if not geometry['prefix_bitwise_unchanged'] or not geometry['transport_support_exact']:
                raise RuntimeError('046 replay support changed')
            if geometry['max_canonical_Q_residual_Vns'] > 1e-10 or geometry['transport_integral_max_residual_Vns'] > 1e-12:
                raise RuntimeError('046 replay charge changed')
            if geometry['segment_area_closure_max_error_Vns']>1e-12 or geometry['minimum_mapped_interval_ns']<=0:
                raise RuntimeError('046 transport area or monotonicity invariant failed')
            for h in r.HYPOTHESES:
                pre_adc = signal+bank['noise'][h]
                obs = r.observe_generated_carrier(pre_adc, ste=False, q_floor=float(setup['loss'].q_floor), return_carrier=True)
                progress['generated_observation_rows'] += len(A0)
                wave, baseline, peaks = [obs[key].cpu().numpy() for key in ('window_V', 'baseline100', 'peak_index')]
                metrics = r.hard_metrics(wave, baseline, real_wave, real_baseline, setup['loss'], setup['context'])
                z = r.joint_embedding(wave, baseline, peaks, dict(T[h], q_floor=float(setup['loss'].q_floor)))
                metrics['values']['joint_morphology'] = float(r.column_W1(torch.as_tensor(z, device=device)@projection, real_joint[h], 16))
                metrics['values']['negative_lobe_area_Vns'] = r.raw_distance(np.maximum(-wave, 0).sum(1), real_features['negative_lobe_area_Vns'])['value']
                metrics['observation_flags'] = {key:dict(count=int(value.sum()),denominator=len(A0)) for key,value in obs['flags'].items()}
                progress['full_native_hypothesis_evaluations'] += 1
                metric_path = out/f'seed{index}_{name}_{h}_METRICS.json'
                r.write_json(metric_path, metrics)
                if json.loads(metric_path.read_text()) != replays[str(index)][name][h]:
                    raise RuntimeError('046 full native report does not exactly replay045: '+name+'/'+h)
                replay_checks.append(dict(seed=index, case=name, hypothesis=h, full_report_exact=True))
                r.write_json(out/'REPLAY_CHECKS.json', replay_checks)
                if name == 'M0': continue
                carrier_values = [carrier_peaks046(value) for value in (mean, signal, pre_adc, obs['carrier_adc'])]
                carrier_rows = {key:np.asarray([value[key] for value in carrier_values]) for key in carrier_values[0]}
                if not np.array_equal(carrier_rows['global_peak_index'][-1], obs['global_peak_index'].cpu().numpy()):
                    raise RuntimeError('046 ADC global peak differs from native observation')
                if not np.array_equal(carrier_rows['selected_peak_index'][-1], peaks):
                    raise RuntimeError('046 ADC selected peak differs from native observation')
                if not np.array_equal(carrier_rows['global_peak_outside_search'][-1], obs['flags']['global_peak_outside_search'].cpu().numpy()):
                    raise RuntimeError('046 ADC outside-search flag differs from native observation')
                carrier_reports = {}
                for stage_name, row in zip(CARRIERS046, carrier_values):
                    carrier_reports[stage_name] = dict(rows=len(A0),
                        global_peak_outside_search_count=int(row['global_peak_outside_search'].sum()),
                        global_peak_V=summary(row['global_peak_V']),
                        global_peak_index_counts=dict(zip(*[v.tolist() for v in np.unique(row['global_peak_index'],return_counts=True)])),
                        selected_peak_index_counts=dict(zip(*[v.tolist() for v in np.unique(row['selected_peak_index'],return_counts=True)])),
                        by_input_Q_bin=[dict(bin=b,rows=int((input_bins==b).sum()),
                            global_peak_outside_search_count=int(row['global_peak_outside_search'][input_bins==b].sum()),
                            global_peak_V=summary(row['global_peak_V'][input_bins==b]),
                            global_peak_index_counts=dict(zip(*[v.tolist() for v in np.unique(row['global_peak_index'][input_bins==b],return_counts=True)])),
                            selected_peak_index_counts=dict(zip(*[v.tolist() for v in np.unique(row['selected_peak_index'][input_bins==b],return_counts=True)])))
                            for b in range(8)])
                    progress['carrier_stage_summaries'] += 1; progress['carrier_stage_rows'] += len(A0)
                anchor = obs['crop_indices'] if name == 'BASE042' else base[h]['indices']
                stages = fixed_anchor_stages(mean, signal, pre_adc, obs, anchor)
                values = {key:[] for key in FIELDS044}; stage_reports = {}; width_flags = []
                for stage_name, tensor in zip(STAGES044, stages):
                    array = tensor.cpu().numpy(); features, flags = readouts044(array)
                    for key in FIELDS044: values[key].append(features[key])
                    width_flags.append(flags['FWHM_undefined'])
                    stage_reports[stage_name] = dict(features={key:summary(value) for key,value in features.items()},
                        W1_to_REAL={key:r.raw_distance(value, real_features[key]) for key,value in features.items()},
                        flags={key:dict(count=int(value.sum()),denominator=len(A0)) for key,value in flags.items()},
                        population_mean_V=array.mean(0), population_std_V=array.std(0),
                        by_input_Q_bin=[dict(bin=b, rows=int((input_bins==b).sum()),
                            features={key:summary(features[key][input_bins==b]) for key in FIELDS044}) for b in range(8)])
                    progress['counterfactual_stage_summaries'] += 1; progress['counterfactual_stage_rows'] += len(A0)
                values = {key:np.asarray(value) for key,value in values.items()}; width_flags=np.asarray(width_flags)
                if name == 'BASE042':
                    if not torch.equal(stages[-2], stages[-1]): raise RuntimeError('046 base anchor differs from native crop')
                    base[h] = dict(indices=anchor, values=values, peaks=peaks.copy(), wave=wave.copy(),
                        width_flags=width_flags, stage_reports=stage_reports, mean=mean)
                tail = ordered_contrasts(values['tail_Vns'], base[h]['values']['tail_Vns'])
                q = ordered_contrasts(values['Q_Vns'], base[h]['values']['Q_Vns'])
                w1_effects = {}
                for key in ('tail_Vns', 'FWHM_ns'):
                    distances = [stage_reports[s]['W1_to_REAL'][key]['value'] for s in STAGES044]
                    old = [base[h]['stage_reports'][s]['W1_to_REAL'][key]['value'] for s in STAGES044]
                    w1_effects[key] = ordered_contrasts(distances, old)
                    if abs(distances[-1]-metrics['values'][key]) > 1e-12: raise RuntimeError('046 native feature distance mismatch')
                if max(tail['closure_max_error'], q['closure_max_error'],
                       *(v['closure_max_error'] for v in w1_effects.values())) > 1e-12:
                    raise RuntimeError('046 ordered contributions do not close')
                width_delta = values['FWHM_ns'][-1]-base[h]['values']['FWHM_ns'][-1]
                new_width = width_flags[-1]&~base[h]['width_flags'][-1]
                resolved_width = ~width_flags[-1]&base[h]['width_flags'][-1]
                shift, counts = np.unique(peaks-base[h]['peaks'], return_counts=True)
                selected = np.unique(np.r_[np.flatnonzero(new_width)[:8],
                    np.argsort(-np.abs(tail['stage_delta'][-1]),kind='stable')[:12],
                    np.argsort(np.where(np.isfinite(width_delta),-np.abs(width_delta),np.inf),kind='stable')[:12],
                    np.flatnonzero(carrier_rows['global_peak_outside_search'][-1])[:8]])
                np.savez_compressed(out/f'seed{index}_{name}_{h}_ROWS.npz', parent_row=np.arange(len(A0)),
                    A0_canonical_Q_Vns=input_q, input_Q_bin=input_bins, native_peak_index=peaks, BASE042_peak_index=base[h]['peaks'],
                    FWHM_undefined_by_stage=width_flags, transport_segment_area_Vns=segment_areas,
                    transport_coefficient=coefficient.cpu().numpy(), minimum_mapped_interval_ns=minimum_intervals,
                    **{key+'_by_carrier_stage':value for key,value in carrier_rows.items()},
                    **{key+'_by_stage':value for key,value in values.items()})
                np.savez_compressed(out/f'seed{index}_{name}_{h}_EXAMPLES.npz', parent_row=selected,
                    BASE042_wave_V=base[h]['wave'][selected], native_wave_V=wave[selected],
                    ADC_at_BASE042_anchor_V=stages[-2][selected].cpu().numpy(), mean_pre_V=mean[selected].cpu().numpy(),
                    BASE042_mean_pre_V=base[h]['mean'][selected].cpu().numpy(),
                    transport_pre_V=transport[selected].cpu().numpy(), mapped_edges_ns=mapped[selected].cpu().numpy(),
                    transport_coefficient=coefficient[selected].cpu().numpy(), M0_random_pre_V=(bank['M0']-A0)[selected].cpu().numpy(),
                    noise_pre_V=bank['noise'][h][selected].cpu().numpy())
                diagnostic = dict(seed_index=index, case=name, hypothesis=h, geometry=geometry, stages=stage_reports,
                    carrier_stages=carrier_reports, ADC_global_peak_and_flag_replay_exact=True,
                    transport_segments_inclusive=[[110,125],[126,143],[144,158]],
                    transport_segment_area_Vns=[summary(segment_areas[:,i]) for i in range(3)],
                    transport_by_input_Q_bin=[dict(bin=b,rows=int((input_bins==b).sum()),
                        segment_area_Vns=[summary(segment_areas[input_bins==b,i]) for i in range(3)],
                        coefficient=summary(coefficient.cpu().numpy()[input_bins==b]),
                        minimum_mapped_interval_ns=summary(minimum_intervals[input_bins==b])) for b in range(8)],
                    tail_ordered_effects={s:summary(v) for s,v in zip(STAGES044,tail['ordered_contributions'])},
                    observed_Q_ordered_effects={s:summary(v) for s,v in zip(STAGES044,q['ordered_contributions'])},
                    ordered_W1_effects=w1_effects, tail_closure_max_error=tail['closure_max_error'], Q_closure_max_error=q['closure_max_error'],
                    native_width_delta_common_defined=summary(width_delta),
                    native_width_transitions=dict(new_undefined=int(new_width.sum()),resolved_undefined=int(resolved_width.sum()),denominator=len(A0)),
                    peak_shift_counts=dict(zip(map(str,shift),map(int,counts))),
                    BASE042_tail_start_min=int((base[h]['peaks']+30).min()),
                    BASE042_tail_support_overlap_rows=int((base[h]['peaks']+30<=158).sum()),
                    example_parent_rows=selected, all_rows_retained=True, diagnostic_only=True,
                    ordered_effects_not_unique_causal_effects=True, physical_acceptance=False)
                r.write_json(out/f'seed{index}_{name}_{h}_DIAG.json', diagnostic)
                case_summaries.append({key:diagnostic[key] for key in ('seed_index','case','hypothesis','tail_ordered_effects',
                    'ordered_W1_effects','native_width_transitions','peak_shift_counts','BASE042_tail_support_overlap_rows')})
            progress['completed_full_population_cases'] += 1
            r.write_json(out/'PROGRESS.json', dict(phase='frozen_mechanism_diagnostic', seed_index=index, case=name,
                elapsed_seconds=time.monotonic()-started, execution_counts=dict(progress)))
        del bank, base
    r.check_resources(p, out, started)
    for index, model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict()) != r.state_digest(setup['heads'][index]): raise RuntimeError('046 M0 head changed')
    if r.state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest']: raise RuntimeError('046 A0 changed')
    if r.state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']: raise RuntimeError('046 005 changed')
    result = dict(schema='n01/response046-diagnostic-result/v1', run_id=p['run_id'], diagnostic_complete=True,
        replay_checks=replay_checks, case_summaries=case_summaries, execution_counts=dict(progress),
        elapsed_seconds=time.monotonic()-started, frozen_A0_M0_005_verified=True,
        all_TRAIN_SOURCE_parents=6593, all_TRAIN_REAL_rows=21047, all_parent_count_weights_preserved=True,
        model_selected=False, new_candidate_settings=0, confirmation_banks_opened=0, DEV_opened=False,
        scientific_status='response_match_not_established', full_random_response_acceptance=False,
        next_step='Use transport area and staged peak/crop attribution to register the next minimal response change; no automatic parameter expansion.')
    r.write_json(out/'RESULT.json', result)
    r.write_json(out/'COMPLETE.json', dict(UTC=datetime.now(timezone.utc).isoformat(), native_exit_expected=0,
        result_sha256=r.sha256(out/'RESULT.json'), elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if r.sha256(args.config)!=args.config_sha256: raise ValueError('046 config hash mismatch')
    p=json.loads(args.config.read_text())
    for path,digest in p['source_pins'].items():
        if r.sha256(ROOT/path)!=digest: raise ValueError('046 source pin mismatch: '+path)
    if r.sha256(ROOT/p['setup038']['path'])!=p['setup038']['sha256']: raise ValueError('046 setup pin mismatch')
    p['_config_sha256']=args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads']); torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    progress=Counter(dict.fromkeys(('checkpoint_loads','TRAIN_archives_decoded','PCA_archives_decoded','DEV_archives_decoded',
        'AD_queries','optimizer_steps','old_A_calls','old_A_rows','response_calls','response_rows','noise005_calls','noise005_rows',
        'extension_rows','full_population_banks','mean_operator_calls','mean_operator_rows','generated_observation_rows',
        'full_native_hypothesis_evaluations','counterfactual_stage_summaries','counterfactual_stage_rows','completed_full_population_cases','carrier_stage_summaries','carrier_stage_rows'),0))
    try: run046(p,args.out,device,progress)
    finally:
        error=sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),execution_counts=dict(progress),
                automatic_restart=False,scientific_status='response_match_not_established'))


if __name__=='__main__': main()
