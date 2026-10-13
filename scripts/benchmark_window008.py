"""Formal-batch resource check: artificial pulses, fixed005, full mean1+joint1.

No TRAIN/DEV container or prepared scientific data is opened. No configuration,
model or engine source is changed. These two optimizer steps are engineering
work only, and their losses cannot select scientific settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from check_window008_engine import fixture, file_digest, write_json
from supervise import rss_tree
from n01.response008 import Response008Engine, common_initial_state, state_digest
from n01.train import utc


class RSSSampler:
    """50 ms samples of this process and its descendants; not other jobs."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        rss, _ = rss_tree(os.getpid())
        self.peak = max(self.peak, rss)
        self.samples += 1

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self._sample()
                self.stop_event.wait(.05)
        except Exception as error:
            self.error = repr(error)

    def start(self):
        self._sample()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.)
        self._sample()
        if self.thread.is_alive() or self.error is not None:
            raise RuntimeError('RSS sampler did not complete: ' + str(self.error))


def source_hashes():
    sources = [ROOT / 'src/n01' / name for name in ('model.py', 'model008.py', 'response008.py',
        'observation008.py', 'losses008.py', 'losses008_v2.py', 'electronics_v3.py',
        'extension006.py', 'data.py', 'train.py')]
    sources += [ROOT / 'scripts' / name for name in ('check_window008_engine.py', 'benchmark_window008.py', 'supervise.py')]
    return {path.relative_to(ROOT).as_posix(): file_digest(path) for path in sources}


def run(args, output, report):
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This capacity check requires native CUDA; no CPU substitution')
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    dimensions = (config['distribution_microbatches'], config['source_parents_per_microbatch'],
                  config['responses_per_parent'], config['real_rows_per_microbatch'])
    if dimensions != (2, 256, 2, 512):
        raise ValueError('Formal Window008 batch binding changed: ' + repr(dimensions))
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(2)
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:config['cpu_threads']])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total_memory = torch.cuda.get_device_properties(device).total_memory
    memory_limit = int(config['gpu_memory_limit_GiB'] * 1024**3)
    if not (0 < memory_limit <= total_memory):
        raise ValueError('Registered GPU memory cap cannot be represented on this device')
    torch.cuda.set_per_process_memory_fraction(memory_limit / total_memory, device)
    noise_path = Path(config['noise_reference']['path'])
    if not noise_path.is_absolute():
        noise_path = config_path.parent / noise_path
    noise_hash = file_digest(noise_path)
    if noise_hash != config['noise_reference']['sha256']:
        raise ValueError('Frozen005 checkpoint hash mismatch')
    noise_checkpoint = torch.load(noise_path, map_location='cpu', weights_only=False)
    data, statistics, fixture_hash = fixture(rows=1024)
    common = common_initial_state(statistics['model'], config['seeds']['initialization'])
    engine = Response008Engine(data, statistics, config, noise_checkpoint, common, arm='full', device=device)
    frozen_noise_hash = state_digest(engine.extension.model005.state_dict())
    report.update(device=str(device), device_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        formal_batch=dict(microbatches=2, source_parents_per_microbatch=256,
            responses_per_parent=2, real_rows_per_microbatch=512,
            distribution_rows_per_hypothesis=1024),
        fixture_rows=1024, fixture_sha256=fixture_hash,
        config_path=str(config_path), config_sha256=file_digest(config_path),
        noise005_path=str(noise_path), noise005_sha256=noise_hash,
        noise005_state_sha256=frozen_noise_hash, common_initial_state_sha256=state_digest(common),
        gpu_memory_limit_bytes=memory_limit, physical_gpu_memory_bytes=total_memory,
        cpu_threads=config['cpu_threads'], deterministic_algorithms=True,
        setup_elapsed_seconds=time.monotonic() - args.started,
        method_settings_changed=False, real_or_source_data_decoded=False)
    write_json(output / 'ARTIFICIAL_STATISTICS.json', statistics)
    for stage in ('mean_response', 'joint_stochastic'):
        engine.set_stage(stage)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        rss = RSSSampler()
        rss.start()
        started = time.monotonic()
        metrics = None
        try:
            metrics = engine.step(audit=True, verify_replay=True)
            torch.cuda.synchronize(device)
        finally:
            elapsed = time.monotonic() - started
            rss.stop()
            row = dict(stage=stage, step_completed=metrics is not None,
                seconds_synchronized=elapsed, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                rss_tree_sampled_peak_bytes=rss.peak, rss_sampling_interval_seconds=.05,
                rss_samples=rss.samples, engineering_G_updates=engine.update,
                engineering_D_updates=dict(engine.d_updates))
            if metrics is not None:
                row['engine_metrics'] = metrics
            report['steps'].append(row)
            report.update(engineering_G_updates=engine.update,
                engineering_D_updates=dict(engine.d_updates),
                engineering_D_updates_total=sum(engine.d_updates.values()))
            write_json(output / 'BENCHMARK.json', report)
            print(json.dumps({key:value for key,value in row.items() if key != 'engine_metrics'}), flush=True)
        if row['gpu_peak_allocated_bytes'] > memory_limit or row['gpu_peak_reserved_bytes'] > memory_limit:
            raise RuntimeError('Formal-batch GPU peak exceeded registered cap')
        if row['rss_tree_sampled_peak_bytes'] > 8 * 1024**3:
            raise RuntimeError('Formal-batch RSS sample exceeded 8 GiB')
        if not metrics['same_version_replay_verified']:
            raise AssertionError('Explicit replay was not verified')
    if engine.update != 2 or any(count != 1 for count in engine.d_updates.values()):
        raise AssertionError('Expected exactly engineering G2 and D3')
    engine.extension.assert_frozen()
    if state_digest(engine.extension.model005.state_dict()) != frozen_noise_hash:
        raise AssertionError('Frozen005 parameters changed')
    if source_hashes() != report['source_sha256'] or file_digest(config_path) != report['config_sha256']:
        raise RuntimeError('Bound code/config changed during benchmark')
    report.update(status='passed_native_fullbatch_capacity_only', scientific_G_updates=0,
                  scientific_D_updates=0, noise_updates=0, population_updates=0,
                  hyperparameters_selected=False, low_rate_stage_tested=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'window008_config.json'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.started = time.monotonic()
    output = Path(args.out).resolve()
    if os.name == 'nt' and output.drive.upper() not in ('E:', 'D:'):
        parser.error('Windows output must be on E: or D:')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='window008/fullbatch-engineering/v1', status='running', started_UTC=utc(),
        scope='Capacity-only formal-batch test on artificial pulses/baseline plus frozen005; never scientific training.',
        source_sha256=source_hashes(), TRAIN_rows_read=0, DEV_rows_read=0,
        scientific_G_updates=0, scientific_D_updates=0, engineering_G_updates=0,
        engineering_D_updates_total=0, steps=[])
    code = 0
    try:
        run(args, output, report)
    except Exception as error:
        report.update(status='failed_native_fullbatch_capacity_only', error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        code = 1
    finally:
        report.update(completed_UTC=utc(), elapsed_seconds=time.monotonic() - args.started)
        write_json(output / 'BENCHMARK.json', report)
    print(json.dumps(dict(status=report['status'], receipt=str(output / 'BENCHMARK.json'),
        engineering_G_updates=report['engineering_G_updates'], engineering_D_updates=report['engineering_D_updates_total'],
        scientific_G_updates=0, error=report.get('error'))), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
