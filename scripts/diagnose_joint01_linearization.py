"""Actual hard-readout behavior along two saved directions; no solves or fitting."""
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
from n01.observation008 import observe_existing_carrier
from run_joint_response01 import load_joint_setup, progress_file, random_bank, prediction, feature_means, residual_vector, risks, HYPOTHESES
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget
from run_response036 import hard_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered linearization diagnostic config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned linearization input changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB']*1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            for key in ('encoding','references','models','baseline','heads'):
                del setup[key]
            with np.load(ROOT/p['roles_file'], allow_pickle=False) as z:
                real_pool = torch.as_tensor(z['real_train_rows'], device=device)
            cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
                q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
            for key in ('window_V','baseline100'):
                if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
                    raise ValueError('REAL readout differs: ' + key)
            real = dict(**{key:torch.as_tensor(cpu_real[key],device=device) for key in REAL_KEYS},
                flags={key:torch.as_tensor(value,device=device) for key,value in cpu_real['flags'].items()})
            del setup['real'], cpu_real
            counts['REAL_observable_replay_calls'] += 1
            counts['REAL_observable_replay_rows'] += len(real['Q'])
            blocks = observable_blocks(select_real(real, real_pool))
            maps = {}
            for label in ('FIT','CHECK'):
                fmap = JointFeatureMap01(blocks,seed=p['feature_seeds'][label])
                maps[label] = {h:fmap for h in HYPOTHESES}
                counts['feature_maps_constructed'] += 1
            del blocks
            summaries = {}
            for seed_index, model in enumerate(setup['joint_models']):
                with np.load(ROOT/p['direction_files'][seed_index],allow_pickle=False) as z:
                    saved = {key:z[key].copy() for key in z.files}
                parents = torch.as_tensor(saved['fit_source_parents'],device=device)
                rows = torch.as_tensor(saved['fit_real_rows'],device=device)
                target_obs = select_real(real,rows)
                counts['REAL_fit_feature_rows'] += len(rows)
                targets = {label:feature_means({h:target_obs for h in HYPOTHESES},mp) for label,mp in maps.items()}
                bank = random_bank(setup,parents,4,p['training_seed']+seed_index*100000+1000,counts)
                theta = model.initial()
                old = prediction(model,theta,bank,setup['loss'].q_floor,counts)
                old_means = {label:feature_means(old,mp) for label,mp in maps.items()}
                base = residual_vector(old_means['FIT'],targets['FIT'],maps['FIT'])
                saved_r = torch.as_tensor(saved['residual'],device=device)
                J = torch.as_tensor(saved['jacobian'],device=device)
                parity = float((base-saved_r).abs().max())
                if parity > 1e-12:
                    raise ValueError('Saved FIT residual reproduction differs: '+str(parity))
                reports = []
                steps = [('baseline',0.,theta)] + [(name,fraction,torch.maximum(model.lower,torch.minimum(model.upper,
                    theta+fraction*torch.as_tensor(saved[name+'_delta'],device=device))))
                    for name in ('original','constrained') for fraction in p['fractions']]
                for name,fraction,proposal in steps:
                    check_budget(p,counts,started)
                    obs = old if name=='baseline' else prediction(model,proposal,bank,setup['loss'].q_floor,counts)
                    predicted_r = base+J@(proposal-theta)
                    actual = {label:feature_means(obs,mp) for label,mp in maps.items()}
                    actual_r = residual_vector(actual['FIT'],targets['FIT'],maps['FIT'])
                    change, predicted_change = actual_r-base,predicted_r-base
                    norm_product = float(change.norm()*predicted_change.norm())
                    predicted_risks = {}
                    for hi,h in enumerate(HYPOTHESES):
                        raw_residual = maps['FIT'][h].cholesky@(predicted_r[hi*272:(hi+1)*272]*3.**.5)
                        predicted_risks[h] = maps['FIT'][h].risks(raw_residual)
                    entry = dict(direction=name,fraction=fraction,
                        actual_risks={label:risks(means,targets[label],maps[label]) for label,means in actual.items()},
                        predicted_FIT_risks=predicted_risks,
                        whitened_residual_absolute_prediction_error=float((actual_r-predicted_r).norm()),
                        whitened_residual_relative_prediction_error=float((actual_r-predicted_r).norm()/base.norm()),
                        change_alignment_cosine=None if norm_product==0 else float(change@predicted_change)/norm_product,
                        predicted_change_norm=float(predicted_change.norm()),actual_change_norm=float(change.norm()),
                        hard_readout={},native={})
                    for h in HYPOTHESES:
                        peak_delta = obs[h]['peak_index']-old[h]['peak_index']
                        entry['hard_readout'][h] = dict(denominator=len(peak_delta),
                            changed_peak_count=int((peak_delta!=0).sum()),peak_delta_RMS=float(peak_delta.double().square().mean().sqrt()),
                            preADC_input_change_RMS_V=float((obs[h]['carrier_input']-old[h]['carrier_input']).square().mean().sqrt()),
                            aligned_window_change_RMS_V=float((obs[h]['window_V']-old[h]['window_V']).square().mean().sqrt()))
                        counts['native_metric_rows'] += 8192
                        entry['native'][h] = hard_metrics(obs[h]['window_V'].cpu().numpy(),obs[h]['baseline100'].cpu().numpy(),
                            target_obs['window_V'].cpu().numpy(),target_obs['baseline100'].cpu().numpy(),setup['loss'],setup['context'])
                    reports.append(entry)
                write_json(args.out/f'seed{seed_index}_LINEARIZATION.json',dict(FIT_residual_reproduction_max_difference=parity,
                    sample_count=4096,roles='original research FIT only',reports=reports))
                summaries[str(seed_index)] = dict(FIT_residual_reproduction_max_difference=parity,evaluated_directions=2,
                                                 fractions=p['fractions'],committed_updates=0)
                progress_file(args.out,counts,'local_model_agreement',seed=seed_index)
        check_budget(p,counts,started)
        result = dict(status='COMPLETE',config_sha256=args.config_sha256,execution_counts=dict(counts),
            elapsed_seconds=time.monotonic()-started,summaries=summaries,committed_parameter_updates=0,
            new_Jacobians_or_direction_solves=0,scientific_acceptance=False,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(args.out/'RESULT.json',result)
        if sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())>p['resources']['output_MiB']*1024**2:
            raise ValueError('Linearization output bound exceeded')
        write_json(args.out/'COMPLETE.json',dict(status='COMPLETE',result_sha256=hashlib.sha256((args.out/'RESULT.json').read_bytes()).hexdigest()))
    finally:
        error=sys.exc_info()[1]
        if error is not None:
            write_json(args.out/'FAILED.json',dict(error_type=type(error).__name__,error=str(error),
                execution_counts=dict(counts),elapsed_seconds=time.monotonic()-started,automatic_restart=False))


if __name__=='__main__':
    main()
