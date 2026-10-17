"""Strict saved-ADC arithmetic check, then the unchanged038 C2ST supplement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from n01.data import sha256, write_json
from n01.inference028 import verified_path028

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def engineering_probe(p, out, device, progress, started, config_sha, environment):
    code = np.arange(16384, dtype=np.float64)
    cpu_grid = code/16383
    cpu_grid = cpu_grid+(-.5)
    lut = torch.arange(16384, dtype=torch.float64, device=device)/16383
    lut = lut+(-.5)
    gpu_grid = lut.cpu().numpy()
    mismatch = cpu_grid.view(np.uint64) != gpu_grid.view(np.uint64)
    np.savez_compressed(out/'PRIVATE_ADC_GRID_CPU_GPU.npz', code=code.astype(np.uint16),
        CPU_grid=cpu_grid, original_device_grid=gpu_grid)
    progress['code_grid_values'] = 16384
    with np.load(verified_path028(p['common_prefix']), allow_pickle=False) as archive:
        prefix_codes = archive['prefix_adc_codes'].copy()
        saved_mean = archive['observed_baseline_mean_V'].copy()
    progress['saved_NPZ_archives_decoded'] += 1
    if prefix_codes.shape != (26372, 100) or prefix_codes.dtype != np.uint16 or saved_mean.dtype != np.float64:
        raise ValueError('038 saved prefix binding differs')
    prefix = np.empty((26372, 100), dtype=np.float64)
    for start in range(0, 26372, 128):
        if time.monotonic()-started >= p['engineering_probe']['budget_max']['wall_seconds']:
            raise TimeoutError('038 engineering probe180s boundary')
        values = lut[torch.as_tensor(prefix_codes[start:start+128], dtype=torch.long, device=device)]
        mean = values.mean(1).cpu().numpy()
        prefix[start:start+128] = values.cpu().numpy()
        progress['prefix_rows'] += len(values)
        unequal = mean.view(np.uint64) != saved_mean[start:start+128].view(np.uint64)
        if unequal.any():
            write_json(out/'PREFIX_MISMATCH.json', dict(chunk_start=start,
                row_indices=(np.flatnonzero(unequal)+start).tolist(), comparison='exact float64 bits'))
            raise RuntimeError('038 original-device prefix128-row mean reconstruction mismatch')
    cells = []
    for cell in p['cells']:
        if time.monotonic()-started >= p['engineering_probe']['budget_max']['wall_seconds']:
            raise TimeoutError('038 engineering probe180s boundary')
        with np.load(verified_path028(cell['ADC']), allow_pickle=False) as archive:
            codes = archive['raw_crop_adc_codes'].copy()
        progress['saved_NPZ_archives_decoded'] += 1
        with np.load(verified_path028(cell['examples']), allow_pickle=False) as archive:
            indices = archive['example_row_indices'].copy()
            expected_wave = archive['waveform256'].copy()
            expected_prefix = archive['baseline100'].copy()
        progress['saved_NPZ_archives_decoded'] += 1
        if (codes.shape != (26372, 256) or codes.dtype != np.uint16 or indices.shape != (32,) or
                expected_wave.dtype != np.float64 or expected_prefix.dtype != np.float64):
            raise ValueError('038 saved example/code binding differs: '+cell['label']+'/'+cell['hypothesis'])
        reconstructed = lut[torch.as_tensor(codes[indices], dtype=torch.long, device=device)]
        reconstructed = reconstructed-torch.as_tensor(saved_mean[indices, None], device=device)
        wave = reconstructed.cpu().numpy()
        same64 = np.array_equal(wave.view(np.uint64), expected_wave.view(np.uint64))
        same32 = np.array_equal(wave.astype(np.float32).view(np.uint32), expected_wave.astype(np.float32).view(np.uint32))
        same_prefix = np.array_equal(prefix[indices].view(np.uint64), expected_prefix.view(np.uint64))
        record = dict(label=cell['label'], hypothesis=cell['hypothesis'], rows=32,
            exact_wave_float64=same64, exact_classifier_float32=same32, exact_baseline100=same_prefix,
            waveform_absmax_V=float(np.max(np.abs(wave-expected_wave))))
        cells.append(record)
        progress['saved_example_waveform_rows'] += 32
        write_json(out/'CELL_CHECKS.json', cells)
        if not (same64 and same32 and same_prefix):
            raise RuntimeError('038 strict persisted example reconstruction mismatch: '+cell['label']+'/'+cell['hypothesis'])
    receipt = dict(schema='n01/response038-engineering-ADC-reconstruction-check/v1', passed=True,
        UTC=datetime.now(timezone.utc).isoformat(), config_sha256=config_sha, script_sha256=sha256(Path(__file__)),
        ROOT_design=p['root_design'], original_source_pins=p['original_source_pins'], environment=environment,
        environment_sha256=hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
        code_grid=dict(values=16384, CPU_vs_original_device_float64_mismatches=int(mismatch.sum()),
            max_abs_error_V=float(np.max(np.abs(cpu_grid-gpu_grid))), private_values='PRIVATE_ADC_GRID_CPU_GPU.npz'),
        prefix=dict(rows=26372, original_chunk_rows=128, final_chunk_rows=4, all_saved_mean_float64_bits_equal=True),
        cells=cells, actual_operations=dict(progress), all_saved_examples_and_prefix_exact=True,
        operation_provenance='Original torch eager float64 division by16383, then separate add(-.5); LUT lookup then separate subtraction of saved original baseline mean.',
        full_population_original_waveform_bytes_saved=False, direct_exhaustive_original_wave_replay_claim=False,
        proof_scope='Complete finite ADC grid operation provenance, all saved prefix means and all1056saved example waveforms; full original population waves were not persisted.',
        original_false_reconstruction_flags_modified=False, CP_NN_AD_optimizer=0)
    write_json(out/'ENGINEERING_RECONSTRUCTION_CHECK.json', receipt)


def complete_classifiers(p, out, device, progress, started, check, environment):
    from run_response038 import classifier256, stream_seed
    if (not check['passed'] or not check['all_saved_examples_and_prefix_exact'] or
            check['script_sha256'] != sha256(Path(__file__)) or check['environment'] != environment):
        raise ValueError('038 classifier supplement requires exact bound engineering PASS')
    with np.load(verified_path028(p['common_prefix']), allow_pickle=False) as archive:
        baseline_mean = archive['observed_baseline_mean_V'].copy()
    progress['common_prefix_NPZ_decodes'] += 1
    with np.load(verified_path028(p['TRAIN_dataset']), allow_pickle=False) as archive:
        real = archive['real'].copy()
        source_ids = archive['source_ids'].copy()
        real_groups = archive['real_groups'].copy()
    progress['TRAIN_dataset_NPZ_decodes'] += 1
    statistics = json.loads(verified_path028(p['TRAIN_statistics']).read_text(encoding='utf8'))
    primary_path = verified_path028(p['PRIMARY_FINAL'])
    primary = json.loads(primary_path.read_text(encoding='utf8'))
    shutil.copy2(primary_path, out/'ORIGINAL_PRIMARY_FINAL.json')
    parent_ids = np.repeat(source_ids, 4)
    lut = torch.arange(16384, dtype=torch.float64, device=device)/16383
    lut = lut+(-.5)
    (out/'C2ST').mkdir()
    results = {}
    classifier_p = dict(seeds=p['seeds'], resources=p['classifier_completion']['resources'])
    for cell in p['cells']:
        with np.load(verified_path028(cell['ADC']), allow_pickle=False) as archive:
            codes = archive['raw_crop_adc_codes'].copy()
        progress['saved_ADC_NPZ_decodes'] += 1
        fake = np.empty((26372, 256), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, 26372, 128):
                values = lut[torch.as_tensor(codes[start:start+128], dtype=torch.long, device=device)]
                values = values-torch.as_tensor(baseline_mean[start:start+128, None], device=device)
                fake[start:start+128] = values.cpu().numpy()
        label, h = cell['label'], cell['hypothesis']
        result = classifier256(fake, real, parent_ids, real_groups, statistics['loss'], device,
            stream_seed(p['seeds'][0], 'C2ST', label, h), out/'C2ST'/f'{label}_{h}', progress, classifier_p, started)
        results.setdefault(label, {})[h] = result
        write_json(out/'CLASSIFIER_STATUS.json', dict(completed=progress['classifiers_completed'],
            label=label, hypothesis=h, execution_counts=dict(progress), original_PRIMARY_FINAL_sha256=p['PRIMARY_FINAL']['sha256']))
    write_json(out/'C2ST_RESULTS.json', results)
    supplement = dict(primary)
    supplement.update(native_status='C2ST_SUPPLEMENT_COMPLETE', C2ST=results,
        classifier_execution_counts=dict(progress), original_response_execution_counts=primary['execution_counts'],
        original_native_exit_code=1, original_RESULT_native_status='INCOMPLETE_AFTER_PRIMARY_FINAL',
        original_FAILED_preserved=True, original_false_reconstruction_flags_preserved=True,
        original_A0_005_terminal_guard='not reached in original execution; not rerun or claimed passed',
        response_training_repeated=False, FINAL_recomputed=False,
        engineering_check_sha256=p['engineering_check_sha256'], classifier256_unchanged=True)
    write_json(out/'RESULT.json', supplement)
    write_json(out/'COMPLETE.json', dict(native_status='C2ST_SUPPLEMENT_COMPLETE', RESULT_sha256=sha256(out/'RESULT.json'),
        classifier_execution_counts=dict(progress), response_Adam_added=0, response_NN_added=0,
        original_native_exit_code=1, all_four_tracks_complete=False, teacher1_interrupted=True,
        full_teacher_qualification_established=False, unified_scientific_acceptance_established=False,
        scientific_status='response_match_not_established'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('probe', 'classify'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--engineering-check', type=Path)
    parser.add_argument('--engineering-check-sha256')
    args = parser.parse_args()
    if sha256(args.config) != args.config_sha256:
        raise ValueError('038 C2ST supplement config SHA differs')
    p = json.loads(args.config.read_text(encoding='utf8'))
    if p['status'] != 'frozen' or p['schema'] != 'n01/response038-C2ST-storage-completion-config/v1':
        raise ValueError('038 C2ST supplement requires ROOT-frozen config')
    for name, digest in p['source_pins'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('038 C2ST supplement source differs: '+name)
    if sha256(ROOT/p['root_design']['path']) != p['root_design']['sha256']:
        raise ValueError('038 C2ST supplement ROOT design differs')
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.device != 'cuda:0' or torch.__version__ != p['runtime']['torch_version'] or sys.executable != p['runtime']['python_executable']:
        raise ValueError('038 original device/Python/Torch environment differs')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    if properties.name != p['runtime']['GPU_name']:
        raise ValueError('038 original GPU model differs')
    budget = p['engineering_probe']['budget_max'] if args.phase == 'probe' else p['classifier_completion']['resources']
    torch.cuda.set_per_process_memory_fraction(budget['GPU_GiB']*1024**3/properties.total_memory, device)
    environment = dict(Python=sys.version, executable=sys.executable, torch_version=torch.__version__,
        torch_CUDA=torch.version.cuda, device=str(device), GPU_name=properties.name,
        GPU_total_bytes=properties.total_memory, GPU_capability=[properties.major, properties.minor])
    args.out.mkdir(parents=True)
    started = time.monotonic()
    progress = dict(saved_NPZ_archives_decoded=0, code_grid_values=0, prefix_rows=0, saved_example_waveform_rows=0,
        common_prefix_NPZ_decodes=0, saved_ADC_NPZ_decodes=0, TRAIN_dataset_NPZ_decodes=0,
        classifier_forward_calls=0, classifier_forward_rows=0, classifier_AD_queries=0,
        classifier_optimizer_steps=0, classifier_bootstrap_resamples=0, classifiers_completed=0)
    write_json(args.out/'START.json', dict(phase=args.phase, UTC=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), config_sha256=args.config_sha256, environment=environment,
        original_native_exit_preserved=1, response_training_allowed=False, FINAL_recomputation_allowed=False))
    try:
        if args.phase == 'probe':
            engineering_probe(p, args.out, device, progress, started, args.config_sha256, environment)
        else:
            if args.engineering_check is None or args.engineering_check_sha256 is None:
                raise ValueError('038 classifier completion requires a bound engineering receipt')
            if sha256(args.engineering_check) != args.engineering_check_sha256:
                raise ValueError('038 engineering receipt SHA differs')
            check = json.loads(args.engineering_check.read_text(encoding='utf8'))
            if check['config_sha256'] != args.config_sha256:
                raise ValueError('038 engineering receipt config differs')
            p['engineering_check_sha256'] = args.engineering_check_sha256
            complete_classifiers(p, args.out, device, progress, started, check, environment)
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            write_json(args.out/'FAILED.json', dict(phase=args.phase, error_type=type(error).__name__,
                error=str(error), actual_operations=dict(progress), elapsed_seconds=time.monotonic()-started,
                original_failed_run_preserved=True, automatic_restart=False,
                response_NN_AD_Adam_added=0, scientific_status='response_match_not_established'))


if __name__ == '__main__':
    main()
