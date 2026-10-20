"""Frozen global-memory candidate: fixed-role and noise-sensitivity reporting."""
from pathlib import Path
from collections import Counter
import argparse
import json
import sys
import time

import numpy as np
import torch
import n01.inference028 as inference028

# Actual data reads are authorized; NAS path/size/hash checks are not.
inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from n01.data import write_json
from n01.global_memory01 import GlobalDistributionMemory01, MeanHeadRepair01, GlobalRepairResponse01
from n01.joint01_features import observable_blocks
from n01.observation008 import observe_existing_carrier, observe_generated_carrier
from n01.response008 import HYPOTHESES
from run_joint_response01 import load_joint_setup, random_bank, progress_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    p = json.loads(args.config.read_bytes())
    hypotheses = tuple(p['hypotheses'])
    assert set(hypotheses) <= set(HYPOTHESES)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    device = torch.device(args.device)
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(
        p['resources']['GPU_GiB'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)

    def check_budget(rows=0):
        if time.monotonic() - started > p['resources']['internal_seconds']:
            raise TimeoutError('Registered evaluation wall budget exhausted')
        if counts['response_rows'] + rows > p['operation_budget']['response_rows']:
            raise RuntimeError('Registered evaluation response budget exhausted')

    def metrics(a, b, target):
        residual = (a + b) / 2 - target
        result = {k: float((memory.weights[s] * residual[s].square()).sum())
                  for k, s in memory.sections.items()}
        raw = memory.joint.cholesky @ residual[memory.sections['native_joint']]
        variance = memory.joint.cholesky.square().sum(1)
        for k, s in memory.joint.block_slices.items():
            result['block_' + k] = float((raw[s].square() / variance[s]).mean())
        return result

    try:
        with torch.no_grad():
            setup = load_joint_setup(p, device, counts)
            with np.load(p['roles_file'], allow_pickle=False) as z:
                roles = {role: (z[source].copy(), z[real].copy()) for role, source, real in
                         [('FIT', 'source_train_rows', 'real_train_rows'),
                          ('REPORT', 'source_heldout_rows', 'real_heldout_rows')]}
            assert [len(v) for v in roles['FIT']] == [6465, 18948]
            assert [len(v) for v in roles['REPORT']] == [128, 2099]
            assert all(not np.intersect1d(roles['FIT'][i], roles['REPORT'][i]).size for i in range(2))
            q_floor = float(setup['loss'].q_floor)
            real_observations = {}
            for role, (_, rows) in roles.items():
                obs = observe_existing_carrier(torch.as_tensor(setup['data']['real_carrier384'][rows]),
                                               q_floor=q_floor, return_carrier=True)
                real_observations[role] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
                real_observations[role]['flags'] = {k: v.to(device) for k, v in obs['flags'].items()}
            memory = GlobalDistributionMemory01(real_observations['FIT'], q_floor=q_floor, **p['memory'])
            saved_memory = torch.load(p['frozen_memory'], map_location=device, weights_only=False)
            counts['memory_checkpoint_loads'] += 1
            assert torch.equal(memory.target, saved_memory['target'])
            assert torch.equal(memory.joint.cholesky, saved_memory['joint_cholesky'])
            memory.weights = saved_memory['weights'].clone()
            memory.normalization = saved_memory['normalization']
            # REPORT creates an evaluation reference only. It never enters the
            # memory constructor, normalization, optimizer, or model selection.
            targets = {role: memory.mean(obs) for role, obs in real_observations.items()}
            encoding, baseline = setup['encoding'], setup['baseline']
            source = encoding['source']
            centered = source - source[:, :100].mean(1, keepdim=True)
            amplitude = centered.amax(1)
            filtered = torch.cat([baseline.response_filter((centered[s:s+128] / amplitude[s:s+128, None])[:, None])
                                  for s in range(0, len(source), 128)])
            with np.load(p['candidate_arrays'][0], allow_pickle=False) as z:
                theta = torch.as_tensor(z['candidate'].copy(), device=device)
            candidate = GlobalRepairResponse01(setup['joint_models'][0], theta,
                                               MeanHeadRepair01(baseline, encoding, filtered))
            saved = torch.load(p['frozen_candidate'], map_location=device, weights_only=False)
            counts['repair_checkpoint_loads'] += 1
            assert saved['accepted_updates'] == 10 and saved['new_committed_updates'] == 9
            candidate.load_state_dict(saved['model'], strict=True)
            candidate.requires_grad_(False).eval()
            reference = GlobalRepairResponse01(setup['joint_models'][0], theta).requires_grad_(False).eval()
            models = dict(initial=reference, B=candidate)
            results, arrays = {}, {}
            write_json(args.out / 'BINDINGS.json', dict(
                populations={r: dict(source=len(s), REAL=len(t)) for r, (s, t) in roles.items()},
                inherited_updates=1, candidate_new_training_updates=9, evaluation_optimizer_updates=0,
                seed_head=0, hypotheses=hypotheses, memory_fit_rows=18948, memory_report_rows=0,
                REPORT_is_same_acquisition_not_future_acquisition=True, inference_ID_input=False))
            progress_file(args.out, counts, 'fixed_candidate_and_FIT_only_memory_loaded')
            for role_index, (role, (parent_rows, real_rows)) in enumerate(roles.items()):
                parents = torch.as_tensor(parent_rows, device=device)
                means = {name: {h: [] for h in hypotheses} for name in models}
                flags = {name: {h: [] for h in hypotheses} for name in models}
                if p.get('parent_wave_audit', False):
                    parent_wave = {name: torch.zeros(len(parents), 56, device=device, dtype=torch.float64) for name in models}
                    arrays[f'{role}_source_Q'] = encoding['original']['q_source'][parents].cpu().numpy()
                    arrays[f'{role}_source_peak'] = amplitude[parents].cpu().numpy()
                    arrays[f'{role}_parent_rows'] = parent_rows
                for bank_index in range(2):
                    totals = {name: {h: torch.zeros_like(memory.target) for h in hypotheses} for name in models}
                    counters = {name: {h: Counter() for h in hypotheses} for name in models}
                    for j, start in enumerate(range(0, len(parents), p['parent_chunk'])):
                        bank = random_bank(setup, parents[start:start+p['parent_chunk']], p['draws_per_parent'],
                            p['evaluation_seed'] + 10000000 * role_index + 100000 * bank_index + 1000 * j, counts)
                        for name, model in models.items():
                            for offset in range(0, len(bank['rows']), p['micro_rows']):
                                select = slice(offset, offset+p['micro_rows'])
                                rows, z = bank['rows'][select], bank['z'][select]
                                check_budget(len(rows))
                                signal = model(rows, z)
                                counts['response_rows'] += len(rows)
                                counts['response_calls'] += 1
                                for h in hypotheses:
                                    obs = observe_generated_carrier(signal + model.noise_scale() * bank['noise'][h][select],
                                                                   q_floor=q_floor, return_carrier=True)
                                    counts['hard_observation_rows'] += len(rows)
                                    totals[name][h] += memory.sums(obs)
                                    counters[name][h].update({k: int(v.sum()) for k, v in obs['flags'].items()})
                                    if p.get('parent_wave_audit', False):
                                        assert hypotheses == ('fast',) and p['micro_rows'] % p['draws_per_parent'] == 0
                                        block = memory.joint.features(observable_blocks(obs))[:, :56]
                                        per_parent = block.reshape(-1, p['draws_per_parent'], 56).mean(1)
                                        at = start + offset // p['draws_per_parent']
                                        parent_wave[name][at:at+len(per_parent)] += per_parent / 2
                    for name in models:
                        for h in hypotheses:
                            avg = totals[name][h] / (len(parents) * p['draws_per_parent'])
                            means[name][h].append(avg)
                            flags[name][h].append(dict(counters[name][h]))
                            arrays[f'{role}_{name}_{h}_bank{bank_index}_mean'] = avg.cpu().numpy()
                    progress_file(args.out, counts, 'fixed_candidate_bank_complete', role=role, bank=bank_index)
                results[role] = {}
                arrays[f'{role}_evaluation_target'] = targets[role].cpu().numpy()
                if p.get('parent_wave_audit', False):
                    for name in models:
                        arrays[f'{role}_{name}_parent_wave'] = parent_wave[name].cpu().numpy()
                        raw = memory.joint.cholesky @ ((means[name]['fast'][0] + means[name]['fast'][1]) / 2)[-272:]
                        assert float((raw[:56] - parent_wave[name].mean(0)).abs().max()) < 1e-9
                for h in hypotheses:
                    scores = {name: metrics(*means[name][h], targets[role]) for name in models}
                    initial, current = scores['initial'], scores['B']
                    checks = dict(PSD_CDF_at_least_10pct_improvement=current['PSD_CDF'] <= .9 * initial['PSD_CDF'],
                        Q_mass_not_worse_2pct=current['Q_mass'] <= 1.02 * initial['Q_mass'],
                        wave_not_worse_2pct=current['block_wave'] <= 1.02 * initial['block_wave'],
                        peak_codes_not_worse_2pct=current['block_peak_codes'] <= 1.02 * initial['block_peak_codes'],
                        flags_not_increased=all(flags['B'][h][b][k] <= flags['initial'][h][b][k]
                                               for b in range(2) for k in flags['initial'][h][b]))
                    results[role][h] = dict(metrics=scores, flags={name: flags[name][h] for name in models},
                        changes_percent={k: 100 * (current[k] / initial[k] - 1) for k in initial if initial[k] != 0.},
                        unchanged_pilot_checks=checks, passes_consistency_check=all(checks.values()))
            assert torch.equal(memory.target, saved_memory['target'])
            assert all(torch.equal(v, saved['model'][k]) for k, v in candidate.state_dict().items())
            setup['references'][0].assert_frozen_unchanged(deep=True)
            check_budget()
            assert counts['hard_observation_rows'] == p['operation_budget']['hard_observation_rows']
            write_json(args.out / 'EVALUATION_RESULT.json', dict(results=results,
                all_role_hypothesis_checks_pass=all(v['passes_consistency_check'] for r in results.values() for v in r.values()),
                evaluation_optimizer_updates=0, frozen_candidate_unchanged=True, report_rows_added_to_training_memory=0,
                scope=f'Fixed seed0 candidate, two registered random banks, same-acquisition FIT/REPORT and {len(hypotheses)} noise hypotheses. Not independent acquisition or physical-energy truth.',
                scientific_acceptance=False))
            np.savez(args.out / 'EVALUATION_FEATURE_MEANS.npz', **arrays)
            write_json(args.out / 'RESULT.json', dict(status='COMPLETE', elapsed_seconds=time.monotonic()-started,
                execution_counts=dict(counts), config_sha256=args.config_sha256,
                peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), committed_optimizer_updates=0,
                NAS_file_IO_checks=False, NAS_file_hash_verification=False, scientific_acceptance=False))
            write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
    finally:
        if sys.exc_info()[1] is not None:
            error = sys.exc_info()[1]
            write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
                execution_counts=dict(counts), elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
