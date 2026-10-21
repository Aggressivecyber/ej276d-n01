"""Frozen mean-path ablations for the observed PSD comb; no fitting or updates."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
import n01.inference028 as inference028

inference028.verified_path028 = lambda record: Path(record['path'])
import run_response038
run_response038.verified_path028 = inference028.verified_path028
from n01.data import write_json
from n01.global_memory01 import MeanHeadRepair01, GlobalRepairResponse01
from n01.model008 import canonical_charge008, relative_time_warp008
from n01.observation008 import crop_existing_values, observe_generated_carrier
from run_joint_response01 import load_joint_setup, random_bank, progress_file

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--config-sha256', required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
p = json.loads(args.config.read_bytes())
args.out.mkdir(parents=True, exist_ok=False)
counts, started, parts = Counter(), time.monotonic(), defaultdict(list)
torch.set_num_threads(p['resources']['CPU_threads'])
torch.use_deterministic_algorithms(True)
device = torch.device(args.device)
torch.cuda.set_per_process_memory_fraction(
    p['resources']['GPU_GiB'] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)

try:
    with torch.no_grad():
        setup = load_joint_setup(p, device, counts)
        with np.load(p['roles_file'], allow_pickle=False) as z:
            parents, held = z['source_train_rows'].copy(), z['source_heldout_rows'].copy()
            real_rows = z['real_train_rows'].copy()
        assert len(parents) == 6465 and len(real_rows) == 18948 and not np.intersect1d(parents, held).size
        source, enc, base = setup['source'], setup['encoding'], setup['baseline']
        centered = source - source[:, :100].mean(1, keepdim=True)
        unit = centered / centered.amax(1, keepdim=True)
        filtered = torch.cat([base.response_filter(unit[s:s + 128, None]) for s in range(0, len(source), 128)])
        with np.load(p['candidate_arrays'][0], allow_pickle=False) as z:
            theta = torch.as_tensor(z['candidate'].copy(), device=device)
        current = GlobalRepairResponse01(setup['joint_models'][0], theta, MeanHeadRepair01(base, enc, filtered))
        saved = torch.load(p['frozen_candidate'], map_location=device, weights_only=False)
        counts['repair_checkpoint_loads'] += 1
        assert saved['accepted_updates'] == 10
        current.load_state_dict(saved['model'], strict=True)
        current.requires_grad_(False).eval()
        frozen = {k: v.clone() for k, v in current.state_dict().items()}
        original = enc['original']
        constant_scalar = torch.tensor([[0., 0., base.stats['detector_condition']]], device=device, dtype=torch.float32)
        constant_time = torch.exp(math.log(2.) * base.mean_time(constant_scalar).squeeze(1).tanh().double())
        write_json(args.out / 'BINDINGS.json', dict(source_fit_rows=6465, real_fit_rows=18948,
            report_rows_used=0, inherited_updates=10, updates=0,
            fixed_condition='log(Q/q_ref)=0, log(amplitude/a_ref)=0; original detector condition',
            constant_time_scale=float(constant_time[0]), stages=p['stages'],
            modes=p['modes'], counterfactuals_are_diagnostics_only=True,
            NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        progress_file(args.out, counts, 'frozen_component_setup_ready')
        reconstruction_error, decomposition_error = 0., 0.
        examples = []
        for chunk, start in enumerate(range(0, len(parents), p['parent_chunk'])):
            if time.monotonic() - started > p['resources']['internal_seconds']:
                raise TimeoutError('Registered frozen diagnostic wall limit exhausted')
            rows = torch.as_tensor(parents[start:start + p['parent_chunk']], device=device)
            qbar = original['qbar'][rows]
            residual = original['mean_residual'][rows, None]
            scale = original['time_scale'][rows]
            warped, _ = relative_time_warp008(filtered[rows], scale)
            w = warped * base.gate
            unwarped = filtered[rows] * base.gate
            features = base._features(unit[rows, None])
            decoded_constant, _ = base.unet(features, constant_scalar.expand(len(rows), -1))
            counts['fixed_condition_UNet_calls'] += 1
            counts['fixed_condition_UNet_rows'] += len(rows)
            residual_constant = .20 * base.mean_head(decoded_constant).tanh().double() * base.gate
            warped_constant, _ = relative_time_warp008(filtered[rows], constant_time.expand(len(rows)))
            raw = dict(filter_only=unwarped, time_only=w, residual_no_time=unwarped + residual,
                A0_reconstructed=w + residual, A0_fixed_residual_condition=w + residual_constant,
                A0_fixed_time_condition=warped_constant * base.gate + residual)
            signals = dict(SOURCE=source[rows])
            for label, value in raw.items():
                charge = canonical_charge008(value)
                signals[label] = (qbar[:, None, None] * value / charge[..., None])[:, 0]
                parts[label + '_raw_canonical_Q'].append(charge[:, 0].cpu().numpy())
            error = float((signals['A0_reconstructed'] - original['signal_pre'][rows]).abs().max())
            reconstruction_error = max(reconstruction_error, error)
            assert error < 1e-7
            signals['A0'] = signals.pop('A0_reconstructed')
            signals['current_mean'] = current.mean(rows)
            counts['current_mean_rows'] += len(rows)
            signals['current_z0'] = current(rows, torch.zeros(len(rows), 8, device=device))
            bank = random_bank(setup, rows, 1, p['bank_seed'] + 1000 * chunk, counts)
            signals['current_random'] = current(rows, bank['z'])
            signals['current_plus_noise'] = signals['current_random'] + current.noise_scale() * bank['noise']['fast']
            counts['current_response_calls'] += 2
            counts['current_response_rows'] += 2 * len(rows)
            assert set(signals) == set(p['stages'])
            for stage, signal in signals.items():
                for mode in p['modes']:
                    if mode == 'clean_dynamic':
                        obs = crop_existing_values(signal, q_floor=None)
                    elif mode == 'ADC_dynamic':
                        obs = observe_generated_carrier(signal, q_floor=None)
                    else:
                        assert mode == 'clean_fixed126'
                        centered_signal = signal - signal[:, :100].mean(1, keepdim=True)
                        window = centered_signal[:, 110:366]
                        q, tail = window.sum(1), window[:, 46:].sum(1)
                        obs = dict(Q=q, tail=tail, PSD_raw=tail / q,
                                   peak_index=torch.full_like(q, 126, dtype=torch.long))
                    counts['observation_rows'] += len(rows)
                    for key in ('Q', 'tail', 'PSD_raw', 'peak_index'):
                        parts[f'{stage}_{mode}_{key}'].append(obs[key].cpu().numpy().copy())
            qw, qr = canonical_charge008(w)[:, 0], canonical_charge008(residual)[:, 0]
            tw, tr = w[:, 0, 156:366].sum(1), residual[:, 0, 156:366].sum(1)
            # Exact fixed-window PSD identity: W term plus signed residual correction.
            base_psd = tw / qw
            correction = (tr - base_psd * qr) / (qw + qr)
            full_psd = (tw + tr) / (qw + qr)
            decomposition_error = max(decomposition_error, float((full_psd - base_psd - correction).abs().max()))
            for key, value in dict(source_Q=original['q_source'][rows], qbar=qbar, time_scale=scale,
                W_Q=qw, R_Q=qr, W_tail=tw, R_tail=tr, residual_PSD_correction=correction,
                unit_source=unit[rows]).items():
                parts[key].append(value.cpu().numpy().copy())
            if chunk == 0:
                examples = [source[rows].cpu().numpy(), w[:, 0].cpu().numpy(), residual[:, 0].cpu().numpy()]
            progress_file(args.out, counts, 'component_chunk_complete', completed_parents=start + len(rows))
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        arrays.update(source_parent_rows=parents, real_Q=setup['data']['real'][real_rows].sum(1),
            real_PSD=setup['data']['real'][real_rows, 46:].sum(1) / setup['data']['real'][real_rows].sum(1),
            example_source=examples[0], example_W=examples[1], example_R=examples[2])
        np.savez(args.out / 'COMPONENT_SCALARS.npz', **arrays)
        assert all(torch.equal(current.state_dict()[k], v) for k, v in frozen.items())
        setup['references'][0].assert_frozen_unchanged(deep=True)
        for key, expected in p['operation_budget'].items():
            assert counts[key] == expected, (key, counts[key], expected)
        assert time.monotonic() - started <= p['resources']['internal_seconds']
        write_json(args.out / 'RESULT.json', dict(status='COMPLETE', config_sha256=args.config_sha256,
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started,
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
            A0_reconstruction_max_abs_V=reconstruction_error, fixed_window_identity_max_abs=decomposition_error,
            original_and_current_parameters_unchanged=True, optimizer_updates=0,
            scalar_arrays=str(args.out / 'COMPONENT_SCALARS.npz'), scientific_acceptance=False,
            NAS_file_IO_checks=False, NAS_file_hash_verification=False))
        write_json(args.out / 'COMPLETE.json', dict(status='COMPLETE'))
finally:
    error = sys.exc_info()[1]
    if error is not None:
        write_json(args.out / 'FAILED.json', dict(error_type=type(error).__name__, error=str(error),
            execution_counts=dict(counts), elapsed_seconds=time.monotonic() - started, automatic_restart=False))
