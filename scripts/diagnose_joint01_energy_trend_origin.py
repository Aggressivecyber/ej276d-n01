"""Locate PSD trend structure with frozen SOURCE/A0/current zero-latent paths."""
from pathlib import Path
from collections import Counter
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
from n01.observation008 import crop_existing_values, observe_generated_carrier
from run_joint_response01 import load_joint_setup, progress_file

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
torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
try:
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        for key in ('encoding', 'references', 'models', 'baseline', 'heads', 'real'):
            del setup[key]
        with np.load(p['roles_file'], allow_pickle=False) as z:
            parents = z['source_train_rows'].copy()
        theta = []
        for path in p['candidate_arrays']:
            with np.load(path, allow_pickle=False) as z:
                theta.append(torch.as_tensor(z['candidate'].copy(), device=device))
            counts['fixed_candidates_loaded'] += 1
        parts = {f'{stage}_{mode}_{key}': [] for stage in p['stages']
                 for mode in ('clean', 'ADC') for key in ('Q', 'tail', 'PSD_raw', 'peak_index')}
        progress_file(args.out, counts, 'origin_setup_loaded')
        for start in range(0, len(parents), p['parent_chunk']):
            if time.monotonic() - started > p['resources']['internal_seconds']:
                raise TimeoutError('Registered trend-origin wall limit exhausted')
            rows = torch.as_tensor(parents[start:start + p['parent_chunk']], device=device)
            z = torch.zeros(len(rows), 8, device=device)
            signals = dict(SOURCE=setup['source'][rows], frozen_A0=setup['joint_models'][0].cache['A'][rows])
            for seed, model in enumerate(setup['joint_models']):
                signals[f'latest_seed{seed}'] = model(theta[seed], rows, z)
                counts['joint_response_calls'] += 1
                counts['joint_response_rows'] += len(rows)
            for stage, signal in signals.items():
                for mode in ('clean', 'ADC'):
                    if mode == 'clean':
                        obs = crop_existing_values(signal, q_floor=None)
                        counts['clean_observation_calls'] += 1
                        counts['clean_observation_rows'] += len(rows)
                    else:
                        obs = observe_generated_carrier(signal, q_floor=None)
                        counts['ADC_observation_calls'] += 1
                        counts['ADC_observation_rows'] += len(rows)
                    for key in ('Q', 'tail', 'PSD_raw', 'peak_index'):
                        parts[f'{stage}_{mode}_{key}'].append(obs[key].cpu().numpy().copy())
            progress_file(args.out, counts, 'origin_chunk_complete',
                          completed_parent_rows=min(start + p['parent_chunk'], len(parents)))
        arrays = {key: np.concatenate(values) for key, values in parts.items()}
        arrays['source_parent_rows'] = parents
        np.savez(args.out / 'STAGE_QPSD_SCALARS.npz', **arrays)
        for key, expected in p['operation_budget'].items():
            assert counts[key] == expected, (key, counts[key], expected)
        assert set(counts) <= set(p['operation_budget'])
        write_json(args.out / 'RESULT.json', dict(status='COMPLETE', config_sha256=args.config_sha256,
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device), scalar_arrays='STAGE_QPSD_SCALARS.npz',
            new_parameter_candidates=0, committed_parameter_updates=0, scientific_acceptance=False,
            no_noise_draws=True, NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE', result_file='RESULT.json'))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False))
