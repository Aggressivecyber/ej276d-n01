"""Cross frozen FIT banks and projection contexts at three original points."""
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
from n01.joint01_features import JointFeatureMap01, observable_blocks
from n01.observation008 import observe_existing_carrier
from diagnose_joint01_native_direction import native_values, NATIVE_NAMES
from run_joint_response01 import (load_joint_setup, random_bank, prediction,
    feature_means, residual_vector, progress_file, HYPOTHESES)
from run_joint_response01_real import select_real, REAL_KEYS
from run_joint_response01_teacher import check_budget, draw_parents

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('FIT0', 'FIT1', 'FIT2')
POINTS = ('M0', 'trial0', 'trial1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256:
        raise ValueError('Registered bank/projection config changed')
    p = json.loads(raw)
    for name, digest in p['source_pins'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned bank/projection input changed: ' + name)
    args.out.mkdir(parents=True, exist_ok=False)
    counts, started = Counter(), time.monotonic()
    torch.set_num_threads(p['resources']['CPU_threads'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(p['resources']['GPU_GiB'] * 1024**3 /
        torch.cuda.get_device_properties(device).total_memory, device)
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        for key in ('encoding', 'references', 'models', 'baseline', 'heads'):
            del setup[key]
        with np.load(ROOT / p['roles_file'], allow_pickle=False) as archive:
            roles = {key: torch.as_tensor(archive[key], device=device) for key in
                ('source_train_rows', 'source_heldout_rows', 'real_train_rows', 'real_heldout_rows')}
        cpu_real = observe_existing_carrier(setup['data']['real_carrier384'], already_digitized=True,
            q_floor=float(setup['loss'].q_floor), return_carrier=True, return_indices=False)
        for key in ('window_V', 'baseline100'):
            if not np.array_equal(cpu_real[key], setup['real'][key].cpu().numpy()):
                raise ValueError('REAL readout differs: ' + key)
        real = dict(**{key: torch.as_tensor(cpu_real[key], device=device) for key in REAL_KEYS},
            flags={key: torch.as_tensor(value, device=device) for key, value in cpu_real['flags'].items()})
        del setup['real'], cpu_real
        counts['REAL_observable_replay_calls'] += 1
        counts['REAL_observable_replay_rows'] += len(real['Q'])
        blocks = observable_blocks(select_real(real, roles['real_train_rows']))
        maps, contexts = {}, {}
        for label in LABELS:
            fmap = JointFeatureMap01(blocks, seed=p['feature_seeds'][label])
            maps[label] = {h: fmap for h in HYPOTHESES}
            contexts[label] = setup['loss'].create_evaluation_context(seed=p['native_projection_seeds'][label])
            counts['feature_maps_constructed'] += 1
            counts['new_native_projection_contexts'] += 1
        del blocks
        state = {'loss/' + key: value.cpu().numpy() for key, value in setup['loss'].state_dict().items()}
        state.update({label + '/' + key: value.numpy() for label, context in contexts.items()
            for key, value in context.banks.items()})
        np.savez(args.out / 'NATIVE_METRIC_STATE.npz', **state)
        write_json(args.out / 'NATIVE_METRIC_STATE.json', dict(
            loss_class=type(setup['loss']).__name__, minimum_support=setup['loss'].minimum_support,
            contexts={key: value.metadata() for key, value in contexts.items()},
            hypothesis_order=list(HYPOTHESES), metric_order=list(NATIVE_NAMES)))
        results, parity = {}, []
        progress_file(args.out, counts, 'bank_projection_setup_loaded')
        for case, model in zip(p['cases'], setup['joint_models'], strict=True):
            seed_index = case['seed']
            seed = p['training_seed'] + seed_index * 100000
            with np.load(ROOT / case['arrays'], allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            report = json.loads((ROOT / case['report']).read_text())
            if report['selected_trial'] != 1 or not np.array_equal(saved['theta'], model.initial().cpu().numpy()):
                raise ValueError('The fixed first-iteration M0/trial0/trial1 case differs')
            order = {}
            for name, pool, number, offset in (
                    ('source', roles['source_train_rows'], 1024, 100),
                    ('real', roles['real_train_rows'], 4096, 200)):
                epochs = (3 * number + len(pool) - 1) // len(pool)
                order[name] = torch.cat([draw_parents(pool, len(pool), seed + offset + epoch) for epoch in range(epochs)])
            native_grid = np.empty((3, 3, 3, 3, 3), dtype=np.float64)
            for bi, label in enumerate(LABELS):
                check_budget(p, counts, started)
                parents = order['source'][bi * 1024:(bi + 1) * 1024]
                rows = order['real'][bi * 4096:(bi + 1) * 4096]
                for kind, actual in (('source_parents', parents), ('real_rows', rows)):
                    if not np.array_equal(saved[label + '_' + kind], actual.cpu().numpy()):
                        raise ValueError('Original fixed row stream differs: ' + label + '/' + kind)
                bank = random_bank(setup, parents, 4, seed + 1000 + bi * 10, counts)
                target = select_real(real, rows)
                target_features = feature_means({h: target for h in HYPOTHESES}, maps[label])
                counts['REAL_fit_feature_rows'] += len(rows)
                arrays = dict(source_parents=parents.cpu().numpy(), real_rows=rows.cpu().numpy(),
                    bank_seed=np.array(bank['seed']), target_window_V=target['window_V'].cpu().numpy(),
                    target_baseline100=target['baseline100'].cpu().numpy())
                for pi, point in enumerate(POINTS):
                    check_budget(p, counts, started)
                    theta = model.initial() if point == 'M0' else torch.as_tensor(saved[point + '_proposal'], device=device)
                    obs = prediction(model, theta, bank, setup['loss'].q_floor, counts)
                    residual = residual_vector(feature_means(obs, maps[label]), target_features, maps[label])
                    original_r = saved[label + '_baseline_residual'] if point == 'M0' else saved[point + '_' + label + '_actual_residual']
                    residual_difference = float(np.max(np.abs(residual.cpu().numpy() - original_r)))
                    original_n = saved[label + '_baseline_native'] if point == 'M0' else saved[point + '_' + label + '_actual_native']
                    for ci, context_label in enumerate(LABELS):
                        native = native_values(obs, target, dict(setup, context=contexts[context_label]), counts)
                        native_grid[pi, bi, ci] = native.cpu().numpy().reshape(3, 3)
                    native_difference = float(np.max(np.abs(native_grid[pi, bi, bi].reshape(-1) - original_n)))
                    if max(residual_difference, native_difference) > p['original_replay_atol']:
                        raise ValueError('Original same-bank same-projection replay differs')
                    parity.append(dict(seed=seed_index, bank=label, point=point,
                        residual_max_abs_difference=residual_difference, native_max_abs_difference=native_difference))
                    arrays[point + '_theta'] = theta.cpu().numpy()
                    arrays[point + '_residual'] = residual.cpu().numpy()
                    arrays[point + '_native_cross'] = native_grid[pi, bi].copy()
                    for hi, h in enumerate(HYPOTHESES):
                        arrays[point + f'_h{hi}_window_V'] = obs[h]['window_V'].cpu().numpy()
                        arrays[point + f'_h{hi}_baseline100'] = obs[h]['baseline100'].cpu().numpy()
                    counts['fixed_parameter_points_evaluated'] += 1
                np.savez(args.out / f'seed{seed_index}_{label}.npz', **arrays)
                del arrays, obs, bank, target
                if sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()) > p['resources']['output_MiB'] * 1024**2:
                    raise RuntimeError('Bank/projection output budget exceeded')
                progress_file(args.out, counts, 'fixed_bank_projection_complete', seed=seed_index, bank=label)
            ratios = native_grid[1:] / native_grid[0]
            descriptions = []
            for pi, point in enumerate(POINTS[1:]):
                for hi, h in enumerate(HYPOTHESES):
                    for mi, metric in enumerate(NATIVE_NAMES):
                        values = ratios[pi, :, :, hi, mi]
                        grand = values.mean()
                        bank_effect = values.mean(1) - grand
                        projection_effect = values.mean(0) - grand
                        interaction = values - grand - bank_effect[:, None] - projection_effect[None, :]
                        ss = np.array([3 * np.square(bank_effect).sum(),
                            3 * np.square(projection_effect).sum(), np.square(interaction).sum()])
                        descriptions.append(dict(point=point, hypothesis=h, metric=metric,
                            ratios=values.tolist(), minimum=float(values.min()), maximum=float(values.max()),
                            grand_mean=float(grand), bank_main_effect=bank_effect.tolist(),
                            projection_main_effect=projection_effect.tolist(), interaction=interaction.tolist(),
                            sum_squares_bank_projection_interaction=ss.tolist(),
                            sum_squares_fractions=None if ss.sum() == 0 else (ss / ss.sum()).tolist()))
            results[str(seed_index)] = dict(native_grid=native_grid.tolist(), contrasts=descriptions)
    check_budget(p, counts, started)
    result = dict(status='COMPLETE', config_sha256=args.config_sha256, seeds=results,
        original_replay=parity, execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
        interpretation='Descriptive fixed-level crossing; bank includes SOURCE, REAL and response/noise draws. '
            'Not a causal decomposition of FD gradients, a new candidate selection or an independent validation.',
        parameter_updates=0, new_directions=0, heldout_rows_evaluated=0, scientific_acceptance=False)
    write_json(args.out / 'RESULT.json', result)
    write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', config_sha256=args.config_sha256))


if __name__ == '__main__':
    main()
