"""One conservative local peak diffusion on the fixed042 mean; unchanged hard metrics."""
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
from wp1_evaluation import summary
from run_response042 import local_restitution042
from run_response040 import select_common_candidate

ROOT = Path(__file__).resolve().parents[1]


def peak_diffusion043(mean,strength):
    """One weighted nearest-neighbor diffusion; frozen support, no noise input."""
    edge=torch.arange(383,device=mean.device,dtype=torch.float64)+.5
    conductance=torch.where((edge>112)&(edge<144),torch.sin(torch.pi*(edge-112)/32).square(),0.)
    flux=(mean[:,1:]-mean[:,:-1])*conductance
    zero=torch.zeros_like(mean[:,:1])
    return float(strength)*(torch.cat((flux,zero),1)-torch.cat((zero,flux),1))


@torch.no_grad()
def evaluate043(setup,index,bias,bank,tolerances,projections,real_joint,p,out,progress,
                *,restitution_fraction=0.,diffusion_strength=0.,examples=None):
    model=setup['models'][index]
    model.mean_adapter[-1].bias.copy_(torch.as_tensor(bias,device=setup['source'].device,dtype=model.mean_adapter[-1].bias.dtype))
    progress['mean_operator_calls']+=1;progress['mean_operator_rows']+=len(setup['source'])
    adapter=model._mean_response(setup['encoding'],enabled=True)
    delta,correction,lost=local_restitution042(adapter['delta_mean_pre_V'],restitution_fraction)
    A0=setup['encoding']['original']['signal_pre']
    diffusion=peak_diffusion043(A0+delta,diffusion_strength)
    delta=delta+diffusion
    signal=r._ExactZeroMeanAddition015.apply(bank['M0'],delta)
    mean=setup['encoding']['original']['signal_pre']+delta
    geometry=dict(max_time_displacement_ns=float((adapter['time_queries']-torch.arange(384,device=signal.device)).abs().max()),
        max_canonical_Q_residual_Vns=float((r.canonical_charge008(mean)-r.canonical_charge008(setup['encoding']['original']['signal_pre'])).abs().max()),
        local_canonical_Q_max_residual_Vns=float(r.canonical_charge008(correction).abs().max()),
        prefix_bitwise_unchanged=torch.equal(signal[:,:100],bank['M0'][:,:100]),
        local_tail_bitwise_unchanged=torch.equal(delta[:,158:],adapter['delta_mean_pre_V'][:,158:]),
        local_prefix_exact_zero=bool((correction[:,:100]==0).all()),
        local_tail_exact_zero=bool((correction[:,158:]==0).all()),
        restitution_fraction=restitution_fraction,early_lost_area_Vns=summary(lost.cpu().numpy()),
        maximum_absolute_local_voltage_V=float(correction.abs().max()),
        diffusion_strength=diffusion_strength,maximum_absolute_peak_diffusion_V=float(diffusion.abs().max()),
        peak_diffusion_full_integral_max_residual_Vns=float(diffusion.sum(1).abs().max()),
        peak_diffusion_canonical_Q_max_residual_Vns=float(r.canonical_charge008(diffusion).abs().max()),
        peak_diffusion_support_exact=bool((diffusion[:,:112]==0).all() and (diffusion[:,145:]==0).all()))
    if not all(geometry[k] for k in ('prefix_bitwise_unchanged','local_tail_bitwise_unchanged','local_prefix_exact_zero','local_tail_exact_zero')):
        raise RuntimeError('043 local support invariant failed')
    if not geometry['peak_diffusion_support_exact'] or geometry['peak_diffusion_full_integral_max_residual_Vns']>1e-12:
        raise RuntimeError('043 peak diffusion support or integral invariant failed')
    if geometry['max_canonical_Q_residual_Vns']>1e-10 or geometry['local_canonical_Q_max_residual_Vns']>1e-12:
        raise RuntimeError('043 canonical charge identity failed')
    reports={}
    for h in r.HYPOTHESES:
        obs=r.observe_generated_carrier(signal+bank['noise'][h],ste=False,q_floor=float(setup['loss'].q_floor),return_indices=True)
        progress['generated_observation_rows']+=len(signal)
        wave,baseline,peaks=[obs[k].cpu().numpy() for k in ('window_V','baseline100','peak_index')]
        report=r.hard_metrics(wave,baseline,setup['data']['real'],setup['data']['real_baseline100'],setup['loss'],setup['context'])
        z=r.joint_embedding(wave,baseline,peaks,dict(tolerances[h],q_floor=float(setup['loss'].q_floor)))
        report['values']['joint_morphology']=float(r.column_W1(torch.as_tensor(z,device=signal.device)@projections,real_joint[h],16))
        report['values']['negative_lobe_area_Vns']=r.raw_distance(np.maximum(-wave,0).sum(1),np.maximum(-setup['data']['real'],0).sum(1))['value']
        report['observation_flags']={k:dict(count=int(v.sum()),denominator=len(signal)) for k,v in obs['flags'].items()}
        progress['full_native_hypothesis_evaluations']+=1;reports[h]=report
        if examples is not None:
            np.savez_compressed(out/f'{examples}_{h}.npz',wave=wave[:64],baseline=baseline[:64],parent_row=np.arange(min(64,len(signal))),peak_index=peaks[:64])
    return reports,geometry


@torch.no_grad()
def run043(p, out, device, progress):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    r.write_json(out/'STARTED.json', dict(UTC=datetime.now(timezone.utc).isoformat(), run_id=p['run_id'],
                 config_sha256=p['_config_sha256'], pid=os.getpid(), resources=p['resources'], phase='setup'))
    setup = r.load_setup(json.loads((ROOT/p['setup038']['path']).read_text()), device, progress)
    for model in setup['models']:
        model.set_phase('frozen').eval()
    tolerances = json.loads(setup['paths']['tolerances'].read_text())['T_by_hypothesis']
    real_wave, real_baseline = setup['data']['real'], setup['data']['real_baseline100']
    real_embedding = {h: torch.as_tensor(r.joint_embedding(real_wave, real_baseline, np.full(len(real_wave), 16),
                      dict(tolerances[h], q_floor=float(setup['loss'].q_floor))), device=device) for h in r.HYPOTHESES}
    projection = r.projection039(p['joint_projection_fit_seed'], device)
    real_joint = {h: value@projection for h, value in real_embedding.items()}
    proposals = json.loads((ROOT/p['proposal_file']).read_text())['candidates']
    records = [dict(candidate_id=row['candidate_id'], bias=row['bias'], restitution_fraction=row['restitution_fraction'], diffusion_strength=row['diffusion_strength'], seeds=[]) for row in proposals]
    for index in range(2):
        r.check_resources(p, out, started)
        bank = r.make_population_bank(setup, index, 'FIT', 0, p, progress)
        reference, geometry = evaluate043(setup, index, np.zeros(8), bank, tolerances,
                                            projection, real_joint, p, out, progress)
        expected = json.loads((ROOT/p['FIT_replay_files'][index]).read_text())
        differences = []
        for h in r.HYPOTHESES:
            for key, value in reference[h]['values'].items():
                old = expected[h]['values'][key]
                if value is None or old is None:
                    if value != old: raise RuntimeError('043 FIT replay undefined value changed')
                else: differences.append(abs(value-old))
            for family in ('fake_flags', 'observation_flags'):
                if reference[h][family] != expected[h][family]:
                    raise RuntimeError('043 FIT replay flag count or denominator changed')
        if max(differences) > 1e-12:
            raise RuntimeError('043 FIT M0 replay scalar difference exceeds 1e-12')
        r.write_json(out/f'seed{index}_FIT_M0.json', reference)
        r.write_json(out/f'seed{index}_REPLAY.json', dict(max_scalar_absolute_difference=max(differences),
                     flags_exactly_reproduced=True, reference_file=p['FIT_replay_files'][index], geometry=geometry))
        for row in records:
            r.check_resources(p, out, started)
            progress['parameter_proposals'] += 1
            before = time.monotonic()
            report, geometry = evaluate043(setup, index, row['bias'], bank, tolerances,
                                          projection, real_joint, p, out, progress,
                                          restitution_fraction=row['restitution_fraction'],
                                          diffusion_strength=row['diffusion_strength'])
            entry = dict(seed_index=index, decision=r.objective039(report, reference, tolerances, p['contract']),
                         geometry=geometry, metrics=report)
            entry.update(evaluation_seconds=time.monotonic()-before, execution_counts=dict(progress))
            if row['candidate_id'] == 0:
                prior = json.loads((ROOT/p['base_replay_files'][index]).read_text())
                r.write_json(out/f'seed{index}_BASE042_CURRENT.json', report)
                current = json.loads((out/f'seed{index}_BASE042_CURRENT.json').read_text())
                if current != prior:
                    raise RuntimeError('043 zero diffusion does not exactly replay042')
                r.write_json(out/f'seed{index}_BASE042_REPLAY.json',dict(max_scalar_absolute_difference=0.,flags_exact=True,full_report_exact=True))
            row['seeds'].append(entry)
            r.write_json(out/f'seed{index}_trials'/f"{row['candidate_id']:03d}.json",
                         dict(candidate_id=row['candidate_id'], bias=row['bias'], restitution_fraction=row['restitution_fraction'], diffusion_strength=row['diffusion_strength'], **entry))
            r.write_json(out/'PROGRESS.json', dict(phase='fixed_candidates', seed_index=index,
                         completed_candidate_evaluations=progress['parameter_proposals'],
                         elapsed_seconds=time.monotonic()-started, execution_counts=dict(progress)))
        del bank
    selected = select_common_candidate([row for row in records if row['diffusion_strength']>0])
    endpoint = None if selected is None else dict(candidate_id=selected['candidate_id'], bias=selected['bias'],
                     restitution_fraction=selected['restitution_fraction'], diffusion_strength=selected['diffusion_strength'],
                     per_seed_FIT_decisions=[row['decision'] for row in selected['seeds']])
    if selected is not None:
        progress['retained_parameter_updates'] = 2
    r.write_json(out/'CANDIDATE_SELECTION.json', dict(records=records, common_endpoint=endpoint))
    r.write_json(out/'ENDPOINTS_FROZEN.json', dict(UTC=datetime.now(timezone.utc).isoformat(), common_endpoint=endpoint,
                 unopened_confirmation_banks_per_seed=p['confirmation_banks'] if selected is not None else 0,
                 no_endpoint_reselection=True, no_additional_candidates=True))
    results = []
    if selected is not None:
        projection = r.projection039(p['joint_projection_confirmation_seed'], device)
        real_joint = {h: value@projection for h, value in real_embedding.items()}
        for index in range(2):
            pairs = []
            for bank_id in range(p['confirmation_banks']):
                r.check_resources(p, out, started)
                bank = r.make_population_bank(setup, index, p['confirmation_purpose'], bank_id, p, progress)
                reference, _ = evaluate043(setup, index, np.zeros(8), bank, tolerances, projection,
                             real_joint, p, out, progress,
                             examples=f'seed{index}_confirm{bank_id}_M0' if bank_id == 0 else None)
                report, geometry = evaluate043(setup, index, selected['bias'], bank, tolerances, projection,
                             real_joint, p, out, progress,
                             examples=f'seed{index}_confirm{bank_id}' if bank_id == 0 else None,
                             restitution_fraction=selected['restitution_fraction'],
                             diffusion_strength=selected['diffusion_strength'])
                pairs.append((report, reference))
                r.write_json(out/f'seed{index}_CONFIRM_{bank_id}.json', dict(candidate=report, M0=reference, geometry=geometry))
                r.write_json(out/'PROGRESS.json', dict(phase='fresh_confirmation', seed_index=index,
                             completed_banks=bank_id+1, elapsed_seconds=time.monotonic()-started,
                             execution_counts=dict(progress)))
                del bank
            results.append(dict(seed_index=index, decision=r.confirmation039(pairs, tolerances, p['contract'])))
    for index, model in enumerate(setup['models']):
        model.assert_frozen_unchanged(deep=True)
        if r.state_digest(model.basis_head.state_dict()) != r.state_digest(setup['heads'][index]):
            raise RuntimeError('043 frozen M0 basis changed')
    if r.state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest']:
        raise RuntimeError('043 frozen A0 changed')
    if r.state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']:
        raise RuntimeError('043 frozen005 changed')
    useful = selected is not None and all(row['decision']['useful_mean_correction'] for row in results)
    result = dict(schema='n01/response043-result/v1', run_id=p['run_id'], common_endpoint=endpoint, results=results,
                  useful_mean_correction_both_seeds=useful,
                  stage_outcome='USEFUL_MEAN_STAGE' if useful else 'CLOSED_NEGATIVE_CONFIRMATION' if selected is not None
                                else 'CLOSED_NEGATIVE_NO_COMMON_FEASIBLE_CANDIDATE',
                  scientific_status='response_match_not_established', full_random_response_acceptance=False,
                  execution_counts=dict(progress), unique_frozen_candidates=len(records),eligible_nonzero_strengths=3,zero_control_excluded_from_selection=True,
                  elapsed_seconds=time.monotonic()-started, frozen_A0_M0_005_verified=True,
                  all_TRAIN_SOURCE_parents=6593, all_TRAIN_REAL_rows=21047, all_parent_count_weights_preserved=True,
                  DEV_opened=False, hardware_calibrated=False,
                  next_step='Review signed failures and physical gains, then advance the necessary bounded next stage.')
    r.write_json(out/'RESULT.json', result)
    r.write_json(out/'COMPLETE.json', dict(UTC=datetime.now(timezone.utc).isoformat(), native_exit_expected=0,
                 result_sha256=r.sha256(out/'RESULT.json'), elapsed_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if r.sha256(args.config) != args.config_sha256:
        raise ValueError('043 config hash mismatch')
    p = json.loads(args.config.read_text())
    for name, digest in p['source_pins'].items():
        if r.sha256(ROOT/name) != digest:
            raise ValueError('043 source pin mismatch: '+name)
    if r.sha256(ROOT/p['setup038']['path']) != p['setup038']['sha256']:
        raise ValueError('043 setup038 pin mismatch')
    p['_config_sha256'] = args.config_sha256
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3/
                                                   torch.cuda.get_device_properties(device).total_memory, device)
    progress = Counter(dict.fromkeys(('checkpoint_loads', 'TRAIN_archives_decoded', 'PCA_archives_decoded',
        'DEV_archives_decoded', 'AD_queries', 'optimizer_steps', 'old_A_calls', 'old_A_rows',
        'response_calls', 'response_rows', 'noise005_calls', 'noise005_rows', 'extension_rows',
        'full_population_banks', 'mean_operator_calls', 'mean_operator_rows', 'generated_observation_rows',
        'full_native_hypothesis_evaluations', 'parameter_proposals', 'retained_parameter_updates'), 0))
    try:
        run043(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                         execution_counts=dict(progress), automatic_restart=False,
                         scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
