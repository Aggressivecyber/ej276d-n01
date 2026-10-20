"""Requested full-Q PSD census of the two existing fixed response candidates."""
from collections import Counter
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch
import n01.inference028 as inference028
inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from n01.data import write_json
from n01.observation008 import observe_existing_carrier
from run_joint_response01 import load_joint_setup, random_bank, prediction, progress_file, HYPOTHESES

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--config-sha256', required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
p = json.loads(args.config.read_bytes())
args.out.mkdir(parents=True, exist_ok=False)
counts, started = Counter(), time.monotonic()
torch.set_num_threads(p['resources']['CPU_threads'])
torch.use_deterministic_algorithms(True)
device = torch.device(args.device)
torch.cuda.set_per_process_memory_fraction(
    p['resources']['GPU_GiB'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
try:
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        for key in ('encoding', 'references', 'models', 'baseline', 'heads', 'real'):
            del setup[key]
        with np.load(p['roles_file'], allow_pickle=False) as z:
            parents = z['source_train_rows'].copy()
            real_rows = z['real_train_rows'].copy()
        q_floor = float(setup['loss'].q_floor)
        real = observe_existing_carrier(setup['data']['real_carrier384'],
            already_digitized=True, q_floor=q_floor)
        counts['REAL_observable_replay_calls'] += 1
        counts['REAL_observable_replay_rows'] += len(real['Q'])
        keys = ('Q', 'tail', 'PSD', 'PSD_raw')
        arrays = {'source_parent_rows': parents, 'real_rows': real_rows}
        for key in keys:
            arrays['real_' + key] = real[key][real_rows].copy()
        flags = {'real': {k: int(v[real_rows].sum()) for k, v in real['flags'].items()}}
        del real
        counts['real_rows_plotted'] = len(real_rows)
        counts['source_parents_plotted'] = len(parents)
        theta = []
        for path in p['candidate_arrays']:
            with np.load(path, allow_pickle=False) as z:
                theta.append(torch.as_tensor(z['candidate'].copy(), device=device))
            counts['fixed_candidates_loaded'] += 1
        parts = {f'seed{seed}_{h}_{key}': [] for seed in range(2) for h in HYPOTHESES for key in keys}
        for seed in range(2):
            for h in HYPOTHESES:
                flags[f'seed{seed}_{h}'] = Counter()
        progress_file(args.out, counts, 'full_energy_PSD_setup_loaded')
        for chunk, start in enumerate(range(0, len(parents), p['parent_chunk'])):
            if time.monotonic() - started > p['resources']['internal_seconds']:
                raise TimeoutError('Registered PSD inference wall budget exhausted')
            selected = torch.as_tensor(parents[start:start + p['parent_chunk']], device=device)
            bank = random_bank(setup, selected, p['draws_per_parent'],
                               p['chunk_bank_seed_base'] + chunk, counts)
            for seed, model in enumerate(setup['joint_models']):
                obs = prediction(model, theta[seed], bank, q_floor, counts)
                for h in HYPOTHESES:
                    group = f'seed{seed}_{h}'
                    for key in keys:
                        parts[group + '_' + key].append(obs[h][key].cpu().numpy().copy())
                    flags[group].update({key: int(value.sum()) for key, value in obs[h]['flags'].items()})
                del obs
            del bank
            progress_file(args.out, counts, 'full_energy_PSD_chunk_complete',
                          completed_parent_rows=min(start + p['parent_chunk'], len(parents)),
                          total_parent_rows=len(parents))
        arrays.update({key: np.concatenate(values) for key, values in parts.items()})
        del parts
        for seed, value in enumerate(theta):
            arrays[f'seed{seed}_fixed_candidate_theta'] = value.cpu().numpy()
        np.savez(args.out / 'QPSD_SCALARS.npz', **arrays)
        summaries = {}
        for group in flags:
            q, psd = arrays[group + '_Q'], arrays[group + '_PSD_raw']
            finite = np.isfinite(q) & np.isfinite(psd)
            summaries[group] = dict(rows=len(q), finite_Q_PSD_rows=int(finite.sum()),
                undefined_Q_PSD_rows=int((~finite).sum()), Q_min=float(q.min()), Q_max=float(q.max()),
                PSD_raw_min=float(psd[finite].min()), PSD_raw_max=float(psd[finite].max()),
                flags=dict(flags[group]))
        for key, maximum in p['operation_budget'].items():
            assert counts[key] == maximum, (key, counts[key], maximum)
        assert set(counts) <= set(p['operation_budget'])
        result = dict(status='COMPLETE', config_sha256=args.config_sha256,
            execution_counts=dict(counts), summaries=summaries, q_floor=q_floor,
            scalar_arrays_file='QPSD_SCALARS.npz', scalar_arrays_uncompressed_bytes=sum(v.nbytes for v in arrays.values()),
            elapsed_seconds=time.monotonic() - started,
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
            new_parameter_candidates=0, committed_parameter_updates=0, scientific_acceptance=False,
            purpose='Requested TRAIN-role full-Q PSD visualization; no energy cut, fitting or model selection.',
            heldout_rows_plotted=0, NAS_file_IO_checks=False, NAS_file_hash_verification=False)
        write_json(args.out / 'RESULT.json', result)
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_file='RESULT.json',
                                                  NAS_file_hash_verification=False))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False))
