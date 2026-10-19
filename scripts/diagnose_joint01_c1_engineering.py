"""Zero-update original/C1 generator parity and fixed-direction approximation."""
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
from n01.joint01_model_c1 import JointResponse01C1, pchip_uniform
from run_joint_response01 import load_joint_setup, progress_file
from run_joint_response01_teacher import check_budget

ROOT = Path(__file__).resolve().parents[1]


def signal_call(model, theta, rows, z, counts, branch, details=False):
    counts[branch + '_generator_calls'] += 1
    counts[branch + '_generator_rows'] += len(rows)
    return model(theta, rows, z, return_details=details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered C1 engineering config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned C1 engineering input changed: ' + name)
    audit = json.loads((ROOT / p['bank_projection_review']).read_text())
    if not audit['independent_hard_output_replay_pass']:
        raise ValueError('Bank/projection audit must be complete before this new engineering diagnostic')
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
        originals = setup['joint_models'] + [setup['physical_model']]
        alternatives = [JointResponse01C1(model.cache, physical=model.physical) for model in originals]
        identities = []
        for mi, (old, new) in enumerate(zip(originals, alternatives, strict=True)):
            seed = p['identity_seed'] + mi
            random = torch.randn(len(setup['source']), 8, generator=torch.Generator().manual_seed(seed)).to(device)
            counts['shape_standard_normal_values'] += random.numel()
            for label, z in (('zero_latents', torch.zeros_like(random)), ('shared_random_latents', random)):
                digests = {branch: hashlib.sha256() for branch in ('original', 'C1')}
                for start in range(0, len(z), 128):
                    check_budget(p, counts, started)
                    rows = torch.arange(start, min(start + 128, len(z)), device=device)
                    values = [signal_call(model, model.initial().zero_(), rows, z[rows], counts, branch)
                        for model, branch in ((old, 'original'), (new, 'C1'))]
                    if not torch.equal(values[0], values[1]):
                        raise ValueError('C1 zero-coordinate full-row identity failed')
                    for branch, value in zip(digests, values, strict=True):
                        digests[branch].update(value.cpu().numpy().tobytes())
                    counts['identity_rows_compared'] += len(rows)
                identities.append(dict(model=mi, physical=old.physical, latents=label, seed=seed,
                    rows=len(z), exact=True, signal_sha256={k: v.hexdigest() for k, v in digests.items()}))
        write_json(args.out / 'ZERO_IDENTITY.json', dict(cases=identities, parameter_counts=[m.parameter_count for m in alternatives],
            identity_is_zero_coordinates=True, physical_nonzero_initialization_not_claimed_identical=True))
        progress_file(args.out, counts, 'C1_full_SOURCE_identity_pass')
        comparisons = []
        for case in p['cases']:
            si = case['seed']
            with np.load(ROOT / case['arrays'], allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            parents = torch.as_tensor(saved['FIT0_source_parents'], device=device)
            rows = parents.repeat_interleave(4)
            seed = p['training_seed'] + si * 100000 + 1000
            z = torch.randn(len(rows), 8, generator=torch.Generator().manual_seed(seed)).to(device)
            counts['shape_standard_normal_values'] += z.numel()
            counts['signal_banks'] += 1
            proposals = torch.stack([torch.as_tensor(saved[name + '_proposal'], device=device) for name in ('trial0', 'trial1')])
            arrays = dict(source_parents=parents.cpu().numpy(), z=z.cpu().numpy(), proposals=proposals.cpu().numpy(),
                peak=originals[si].cache['peak'][rows].cpu().numpy(), bank_seed=np.array(seed))
            reports = {}
            for branch, model in (('original', originals[si]), ('C1', alternatives[si])):
                theta = model.initial()
                baseline = signal_call(model, theta, rows, z, counts, branch)
                tangent = torch.zeros(2, len(rows), 384, dtype=torch.float64, device=device)
                first8, plus_theta, minus_theta = [], [], []
                for coordinate in range(model.parameter_count):
                    check_budget(p, counts, started)
                    plus, minus = theta.clone(), theta.clone()
                    plus[coordinate] = min(float(model.upper[coordinate]), float(theta[coordinate]) + p['finite_difference_step'])
                    minus[coordinate] = max(float(model.lower[coordinate]), float(theta[coordinate]) - p['finite_difference_step'])
                    high, low = [signal_call(model, point, rows, z, counts, branch) for point in (plus, minus)]
                    column = (high - low) / (plus[coordinate] - minus[coordinate])
                    tangent += (proposals[:, coordinate] - theta[coordinate])[:, None, None] * column[None]
                    first8.append(torch.stack((high[:8], low[:8])).cpu().numpy())
                    plus_theta.append(plus.cpu().numpy()); minus_theta.append(minus.cpu().numpy())
                    counts['signal_secant_columns'] += 1
                    if (coordinate + 1) % 18 == 0:
                        progress_file(args.out, counts, 'C1_signal_secants', seed=si, branch=branch, column=coordinate + 1)
                arrays[branch + '_baseline_signal'] = baseline.cpu().numpy()
                arrays[branch + '_signal_tangent'] = tangent.cpu().numpy()
                arrays[branch + '_FD_pairs_first8'] = np.stack(first8)
                arrays[branch + '_FD_plus_theta'] = np.stack(plus_theta)
                arrays[branch + '_FD_minus_theta'] = np.stack(minus_theta)
                reports[branch] = []
                for pi, name in enumerate(('trial0', 'trial1')):
                    details = signal_call(model, proposals[pi], rows, z, counts, branch, details=True)
                    actual = details['signal']
                    difference = actual - baseline - tangent[pi]
                    movement = actual - baseline
                    finite = bool(torch.isfinite(actual).all())
                    nonpositive = int((actual[:, 100:].sum(1) <= 0).sum())
                    summary = dict(point=name, baseline_norm=float(baseline.norm()), movement_norm=float(movement.norm()),
                        error_norm=float(difference.norm()), relative_baseline_error=float(difference.norm() / baseline.norm()),
                        relative_movement_error=float(difference.norm() / movement.norm()), finite=finite,
                        nonpositive_canonical_charge_rows=nonpositive, rows=len(rows))
                    arrays[branch + '_' + name + '_actual_signal'] = actual.cpu().numpy()
                    if branch == 'C1':
                        for key in ('base', 'queries', 'coordinates'):
                            arrays[branch + '_' + name + '_' + key] = details[key].cpu().numpy()
                        query = details['queries']
                        left = query.floor().long().clamp_max(382)
                        low, high = details['base'].gather(1, left), details['base'].gather(1, left + 1)
                        interpolated = pchip_uniform(details['base'], query)
                        below = torch.minimum(low, high) - interpolated
                        above = interpolated - torch.maximum(low, high)
                        overshoot = float(torch.maximum(below, above).clamp_min(0).max())
                        knots = torch.arange(384, device=device, dtype=torch.float64)[None].expand(8, -1)
                        exact_knots = torch.equal(pchip_uniform(details['base'][:8], knots), details['base'][:8])
                        summary.update(interpolation_max_overshoot_V=overshoot, integer_nodes_exact=exact_knots,
                            query_min=float(query.min()), query_max=float(query.max()),
                            query_strictly_increasing=bool((query[:, 1:] > query[:, :-1]).all()))
                        counts['interpolation_identity_rows'] += 8
                        counts['interpolation_check_calls'] += 2
                        counts['interpolation_check_rows'] += len(rows) + 8
                    reports[branch].append(summary)
                    counts['fixed_signal_points'] += 1
                del baseline, tangent, high, low, details, actual
            for pi, point in enumerate(('trial0', 'trial1')):
                old, new = reports['original'][pi], reports['C1'][pi]
                ratio = new['relative_baseline_error'] / old['relative_baseline_error']
                movement_ratio = new['movement_norm'] / old['movement_norm']
                passed = (ratio <= p['success']['maximum_signal_error_ratio']
                    and new['relative_baseline_error'] <= p['success']['maximum_baseline_error']
                    and p['success']['movement_ratio_low'] <= movement_ratio <= p['success']['movement_ratio_high']
                    and new['finite'] and new['nonpositive_canonical_charge_rows'] == 0
                    and new['interpolation_max_overshoot_V'] <= 1e-12 and new['integer_nodes_exact']
                    and new['query_strictly_increasing'] and new['query_min'] == 0 and abs(new['query_max'] - 383) <= 1e-12)
                comparisons.append(dict(seed=si, point=point, original=old, C1=new,
                    signal_error_ratio=ratio, movement_ratio=movement_ratio, pass_for_teacher_validation=passed))
            np.savez(args.out / f'seed{si}_C1_ENGINEERING.npz', **arrays)
            write_json(args.out / f'seed{si}_C1_ENGINEERING.json', reports)
            del arrays, saved
            if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()) > p['resources']['output_MiB'] * 1024**2:
                raise RuntimeError('C1 engineering output budget exceeded')
            progress_file(args.out, counts, 'C1_seed_complete', seed=si)
    check_budget(p, counts, started)
    passed = all(item['pass_for_teacher_validation'] for item in comparisons)
    result = dict(status='COMPLETE', config_sha256=args.config_sha256, comparisons=comparisons,
        original_zero_identity_pass=True, engineering_gate_pass=passed,
        execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        new_candidate_directions=0, parameter_updates=0, scientific_acceptance=False,
        next_action='Independently audit PCHIP against SciPy and saved full-row errors; if confirmed, register renewed coupled teacher validation before REAL.'
            if passed else 'Do not adopt C1 or start teacher/REAL fitting; diagnose this fixed engineering failure.',
        scope='Generator interpolation test at fixed original parameter points. No REAL metrics, new response calibration or conditional-kernel acceptance.')
    write_json(args.out / 'RESULT.json', result)
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', config_sha256=args.config_sha256))


if __name__ == '__main__':
    main()
