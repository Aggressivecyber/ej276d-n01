"""Twenty artificial full-batch engineering updates for steady throughput only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics as py_statistics
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import torch

from benchmark_window008 import RSSSampler, source_hashes
from check_window008_engine import fixture, file_digest, write_json
from n01.response008 import Response008Engine, common_initial_state, state_digest
from n01.train import utc


def bound_sources():
    result = source_hashes()
    result['scripts/benchmark_window008_steady.py'] = file_digest(Path(__file__))
    result['src/n01/train_window008.py'] = file_digest(ROOT/'src/n01/train_window008.py')
    return result


def run(args, output, report):
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Native CUDA required; no substitution')
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    dims = tuple(config[key] for key in ('distribution_microbatches','source_parents_per_microbatch','responses_per_parent','real_rows_per_microbatch'))
    if dims != (2,256,2,512):
        raise ValueError('Formal batch binding changed')
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:config['cpu_threads']])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    limit = int(config['gpu_memory_limit_GiB']*1024**3)
    total = torch.cuda.get_device_properties(device).total_memory
    if not 0 < limit <= total:
        raise ValueError('Invalid native GPU memory cap')
    torch.cuda.set_per_process_memory_fraction(limit/total,device)
    noise_path = Path(config['noise_reference']['path'])
    if not noise_path.is_absolute():
        noise_path = config_path.parent/noise_path
    noise_hash = file_digest(noise_path)
    if noise_hash != config['noise_reference']['sha256']:
        raise ValueError('Frozen005 reference changed')
    data, stats, fixture_hash = fixture(rows=1024)
    checkpoint = torch.load(noise_path,map_location='cpu',weights_only=False)
    common = common_initial_state(stats['model'],config['seeds']['initialization'])
    engine = Response008Engine(data,stats,config,checkpoint,common,arm='full',device=device)
    frozen_noise = state_digest(engine.extension.model005.state_dict())
    report.update(device=str(device),device_name=torch.cuda.get_device_name(device),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        config_path=str(config_path),config_sha256=file_digest(config_path),fixture_rows=1024,fixture_sha256=fixture_hash,
        noise005_path=str(noise_path),noise005_sha256=noise_hash,noise005_state_sha256=frozen_noise,
        formal_batch=dict(microbatches=2,source_parents_per_microbatch=256,responses_per_parent=2,real_rows_per_microbatch=512,distribution_rows_per_hypothesis=1024),
        deterministic_algorithms=True,gpu_memory_limit_bytes=limit,cpu_threads=config['cpu_threads'],
        setup_elapsed_seconds=time.monotonic()-args.started,
        audit_cadence='same as train_window008: first2, stage change, or update modulo configured audit interval',
        replay_cadence='same as train_window008: first configured2 or stage change',
        schedule_scope='manual engineering mean10+joint10; shared scientific schedule/config is unchanged')
    write_json(output/'ARTIFICIAL_STATISTICS.json',stats)
    for stage in ('mean_response','joint_stochastic'):
        engine.set_stage(stage)
        for stage_step in range(1,11):
            update = engine.update+1
            changed = stage_step == 1
            audit = update <= 2 or changed or update%config['engineering']['gradient_audit_interval'] == 0
            replay = update <= config['engineering']['waveform_replay_first_steps'] or changed
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            rss = RSSSampler()
            rss.start()
            started = time.monotonic()
            metrics = None
            try:
                metrics = engine.step(audit=audit,verify_replay=replay)
                torch.cuda.synchronize(device)
            finally:
                elapsed = time.monotonic()-started
                rss.stop()
                row = dict(stage=stage,stage_step=stage_step,seconds_synchronized=elapsed,
                    step_completed=metrics is not None,audit=audit,verify_replay=replay,
                    included_in_steady_summary=stage_step >= 3,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    rss_tree_sampled_peak_bytes=rss.peak,rss_sampling_interval_seconds=.05,
                    engineering_G_updates=engine.update,engineering_D_updates=dict(engine.d_updates))
                if metrics is not None:
                    row['engine_metrics'] = metrics
                report['steps'].append(row)
                report.update(engineering_G_updates=engine.update,engineering_D_updates=dict(engine.d_updates),engineering_D_updates_total=sum(engine.d_updates.values()))
                write_json(output/'BENCHMARK.json',report)
                print(json.dumps({k:v for k,v in row.items() if k!='engine_metrics'}),flush=True)
            if max(row['gpu_peak_allocated_bytes'],row['gpu_peak_reserved_bytes']) > limit or row['rss_tree_sampled_peak_bytes'] > 8*1024**3:
                raise RuntimeError('Engineering resource limit exceeded')
    if engine.update != 20 or any(value != 10 for value in engine.d_updates.values()):
        raise AssertionError('Expected engineering G20/D30')
    engine.extension.assert_frozen()
    if state_digest(engine.extension.model005.state_dict()) != frozen_noise:
        raise AssertionError('Frozen005 state changed')
    if bound_sources() != report['source_sha256'] or file_digest(config_path) != report['config_sha256']:
        raise RuntimeError('Source/config changed during throughput check')
    summary = {}
    for stage in ('mean_response','joint_stochastic'):
        values = [row['seconds_synchronized'] for row in report['steps'] if row['stage']==stage and row['included_in_steady_summary']]
        if len(values)!=8:
            raise AssertionError('Expected exactly last8 timings per stage')
        summary[stage] = dict(samples=8,seconds=values,median_seconds=py_statistics.median(values),max_seconds=max(values),min_seconds=min(values))
    report['steady_last8'] = summary
    report['rough_57000_G_estimate'] = dict(
        phase_counts=dict(mean_response=9000,joint_plus_low=48000),
        median_based_seconds=9000*summary['mean_response']['median_seconds']+48000*summary['joint_stochastic']['median_seconds'],
        observed_max_based_seconds=9000*summary['mean_response']['max_seconds']+48000*summary['joint_stochastic']['max_seconds'],
        assumption='Apply full-arm timings to all3 arms and joint timing to low-rate; checkpoint/evaluation/load variability are excluded. Not a completion guarantee.')
    report.update(status='passed_native_fullbatch_steady_engineering_only',noise_updates=0,population_updates=0,scientific_G_updates=0,scientific_D_updates=0,method_settings_changed=False,hyperparameters_selected=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--config',default=str(ROOT/'window008_config.json'))
    args = parser.parse_args()
    args.started = time.monotonic()
    out = Path(args.out).resolve()
    if os.name=='nt' and out.drive.upper() not in ('E:','D:'):
        parser.error('Windows output must remain on E:/D:')
    out.mkdir(parents=True,exist_ok=False)
    report = dict(schema='window008/fullbatch-steady-engineering/v1',status='running',started_UTC=utc(),
        scope='Steady throughput on artificial pulses/baseline with fixed005; no science data or signal checkpoint read.',
        TRAIN_rows_read=0,DEV_rows_read=0,scientific_G_updates=0,scientific_D_updates=0,
        source_sha256=bound_sources(),engineering_G_updates=0,engineering_D_updates_total=0,steps=[])
    code = 0
    try:
        run(args,out,report)
    except Exception as error:
        report.update(status='failed_native_fullbatch_steady_engineering_only',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        code = 1
    finally:
        report.update(completed_UTC=utc(),elapsed_seconds=time.monotonic()-args.started)
        write_json(out/'BENCHMARK.json',report)
    print(json.dumps(dict(status=report['status'],receipt=str(out/'BENCHMARK.json'),engineering_G_updates=report['engineering_G_updates'],engineering_D_updates=report['engineering_D_updates_total'],scientific_G_updates=0,error=report.get('error'))),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
