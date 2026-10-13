"""Capacity-only test on fabricated waveforms, never scientific data."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_response007 import fabricated
from n01.response007 import Response007Engine,common_initial_state
from n01.data import write_json
from n01.train import utc

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'response007_config.json').read_text(encoding='utf8'))
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    data,stats,noise,_=fabricated();common=common_initial_state(stats['model'],cfg['seeds']['initialization'])
    engine=Response007Engine(data,stats,cfg,noise,common,arm='full',device='cuda:0');results=[]
    for stage in ['mean_response','joint_stochastic','low_learning_rate']:
        engine.set_stage(stage)
        for i in range(2):
            torch.cuda.synchronize();started=time.monotonic()
            result=engine.step(audit=i==0,verify_replay=i==0)
            torch.cuda.synchronize()
            item=dict(stage=stage,stage_step=i+1,seconds=time.monotonic()-started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                G_preclip_gradient_norm=result['G_preclip_gradient_norm'],signal_prefix_absolute_max_V=result['signal_prefix_absolute_max_V'])
            results.append(item);print(json.dumps(item),flush=True)
    report=dict(UTC=utc(),status='artificial_capacity_test_passed',rows_are_fabricated=True,scientific_data_decoded=False,
        artificial_G_updates=engine.update,artificial_D_updates=sum(engine.d_updates.values()),scientific_G_updates=0,
        full_distribution_rows_per_hypothesis=1024,results=results,device=torch.cuda.get_device_name(0))
    write_json(out/'BENCHMARK.json',report)
if __name__=='__main__':main()
