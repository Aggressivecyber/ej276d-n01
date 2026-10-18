"""Frozen coupled candidates, unchanged 039 hard metrics, fresh confirmation."""
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

ROOT = Path(__file__).resolve().parents[1]


def select_common_candidate(records):
    eligible = [row for row in records if all(
        seed['decision']['feasible'] and seed['decision']['score'] < 1.
        for seed in row['seeds'])]
    return min(eligible, key=lambda row: np.mean([seed['decision']['score'] for seed in row['seeds']])) if eligible else None


@torch.no_grad()
def run040(p, out, device, progress):
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
    records = [dict(candidate_id=row['candidate_id'], bias=row['bias'], seeds=[]) for row in proposals]
    for index in range(2):
        r.check_resources(p, out, started)
        bank = r.make_population_bank(setup, index, 'FIT', 0, p, progress)
        reference, geometry = r.evaluate039(setup, index, np.zeros(8), bank, tolerances,
                                            projection, real_joint, p, out, progress)
        expected = json.loads((ROOT/p['FIT_replay_files'][index]).read_text())
        differences = []
        for h in r.HYPOTHESES:
            for key, value in reference[h]['values'].items():
                old = expected[h]['values'][key]
                if value is None or old is None:
                    if value != old: raise RuntimeError('040 FIT replay undefined value changed')
                else: differences.append(abs(value-old))
            for family in ('fake_flags', 'observation_flags'):
                if reference[h][family] != expected[h][family]:
                    raise RuntimeError('040 FIT replay flag count or denominator changed')
        if max(differences) > 1e-12:
            raise RuntimeError('040 FIT M0 replay scalar difference exceeds 1e-12')
        r.write_json(out/f'seed{index}_FIT_M0.json', reference)
        r.write_json(out/f'seed{index}_REPLAY.json', dict(max_scalar_absolute_difference=max(differences),
                     flags_exactly_reproduced=True, reference_file=p['FIT_replay_files'][index], geometry=geometry))
        for row in records:
            r.check_resources(p, out, started)
            progress['parameter_proposals'] += 1
            before = time.monotonic()
            try:
                report, geometry = r.evaluate039(setup, index, row['bias'], bank, tolerances,
                                                 projection, real_joint, p, out, progress)
                entry = dict(seed_index=index, decision=r.objective039(report, reference, tolerances, p['contract']),
                             geometry=geometry, metrics=r.compact039(report))
            except r.MeanProposalOutsideDomain038 as exc:
                entry = dict(seed_index=index, decision=dict(feasible=False, score=None,
                             violations=['mean_geometry_domain']), rejected_domain=str(exc))
            entry.update(evaluation_seconds=time.monotonic()-before, execution_counts=dict(progress))
            row['seeds'].append(entry)
            r.write_json(out/f'seed{index}_trials'/f"{row['candidate_id']:03d}.json",
                         dict(candidate_id=row['candidate_id'], bias=row['bias'], **entry))
            r.write_json(out/'PROGRESS.json', dict(phase='fixed_candidates', seed_index=index,
                         completed_candidate_evaluations=progress['parameter_proposals'],
                         elapsed_seconds=time.monotonic()-started, execution_counts=dict(progress)))
        del bank
    selected = select_common_candidate(records)
    endpoint = None if selected is None else dict(candidate_id=selected['candidate_id'], bias=selected['bias'],
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
                reference, _ = r.evaluate039(setup, index, np.zeros(8), bank, tolerances, projection,
                             real_joint, p, out, progress,
                             examples=f'seed{index}_confirm{bank_id}_M0' if bank_id == 0 else None)
                report, geometry = r.evaluate039(setup, index, selected['bias'], bank, tolerances, projection,
                             real_joint, p, out, progress,
                             examples=f'seed{index}_confirm{bank_id}' if bank_id == 0 else None)
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
            raise RuntimeError('040 frozen M0 basis changed')
    if r.state_digest(setup['baseline'].state_dict()) != setup['frozen_A_digest']:
        raise RuntimeError('040 frozen A0 changed')
    if r.state_digest(setup['extension'].state_dict()) != setup['frozen005_digest']:
        raise RuntimeError('040 frozen005 changed')
    useful = selected is not None and all(row['decision']['useful_mean_correction'] for row in results)
    result = dict(schema='n01/response040-result/v1', run_id=p['run_id'], common_endpoint=endpoint, results=results,
                  useful_mean_correction_both_seeds=useful,
                  stage_outcome='USEFUL_MEAN_STAGE' if useful else 'CLOSED_NEGATIVE_CONFIRMATION' if selected is not None
                                else 'CLOSED_NEGATIVE_NO_COMMON_FEASIBLE_CANDIDATE',
                  scientific_status='response_match_not_established', full_random_response_acceptance=False,
                  execution_counts=dict(progress), unique_frozen_candidates=len(records),
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
        raise ValueError('040 config hash mismatch')
    p = json.loads(args.config.read_text())
    for name, digest in p['source_pins'].items():
        if r.sha256(ROOT/name) != digest:
            raise ValueError('040 source pin mismatch: '+name)
    if r.sha256(ROOT/p['setup038']['path']) != p['setup038']['sha256']:
        raise ValueError('040 setup038 pin mismatch')
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
        run040(p, args.out, device, progress)
    finally:
        error = sys.exc_info()[1]
        if error is not None and args.out.exists():
            r.write_json(args.out/'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                         execution_counts=dict(progress), automatic_restart=False,
                         scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
