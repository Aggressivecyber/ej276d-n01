"""Native boundary-state import and artificial formal-batch continuation QA."""
from pathlib import Path
import argparse
import copy
import datetime as dt
import gc
import json
import os
import sys
import time
import types
from contextlib import ExitStack
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from check_window008_engine import fixture,serial
from window008_resume01_runtime import read,sha,identity,fix01_identity,write_new,validate_inputs
from n01.response008 import common_initial_state
from n01.response008_resume01 import Response008Resume01Engine

def sources():
    names=['src/n01/response008_resume01.py','src/n01/train_window008_resume01.py','scripts/check_window008_resume01_cuda.py',
        'scripts/check_window008_engine.py','scripts/window008_resume01_runtime.py']
    names.append('scripts/check_window008_resume01_engine.py')
    return {name:sha(ROOT/name) for name in names}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out)
    if os.name=='nt' and out.resolve().drive.upper() not in ('E:','D:'):raise ValueError('Outputs must stay on E/D')
    out.mkdir(parents=True,exist_ok=False)
    cfg,base,_=validate_inputs();report=dict(status='running_native_engineering_only',schema='n01/window008-resume01-native-engineering/v1',
        pid=os.getpid(),parent_pid=os.getppid(),started_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),device='cuda:0',
        source_sha256=sources(),config_sha256=sha(ROOT/'window008_resume01_config.json'),scientific_G_updates=0,scientific_D_updates=0,
        actual_SOURCE_forward_rows=0,actual_REAL_forward_rows=0,actual_SOURCE_numeric_decodes=0,actual_REAL_numeric_decodes=0,
        DEV_numeric_decodes=0,engineering_G_updates=0,engineering_D_updates=0,artificial_forward_calls=0,artificial_forward_rows=0,steps=[])
    engine=None;base_g=base_d=0
    torch.set_num_threads(12);torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:12])
    if not torch.cuda.is_available():raise RuntimeError('Native CUDA required')
    device=torch.device('cuda:0');torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    limit=10*1024**3;torch.cuda.set_per_process_memory_fraction(limit/torch.cuda.get_device_properties(device).total_memory,device)
    report.update(torch_version=torch.__version__,CUDA_build=torch.version.cuda,GPU=torch.cuda.get_device_name(device))
    try:
        refs=cfg['reference_inputs']
        statistics=torch.load(ROOT/refs['fix01_full_periodic_06000']['path'],map_location='cpu',weights_only=False)['statistics']
        common=torch.load(ROOT/refs['common_initial_state']['path'],map_location='cpu',weights_only=False)['model']
        noise=torch.load(ROOT/base['noise_reference']['path'],map_location='cpu',weights_only=False)
        placeholders=dict(source_carrier384=np.zeros((6593,384),dtype=np.float64),real=np.zeros((21047,256),dtype=np.float64),
            real_baseline100=np.zeros((21047,100),dtype=np.float64),source_ids=np.arange(6593),real_ids=np.arange(21047))
        rid=identity(cfg,'e'*64)
        engine=Response008Resume01Engine(placeholders,statistics,base,noise,common,arm='full',recovery_identity=rid,device=device)
        ref=cfg['reference_inputs']['fix01_full_periodic_06000']
        with ExitStack() as guards:
            def forbidden(*args,**kwargs):raise AssertionError('Actual checkpoint import called a forward, optimizer, or sampler/stream draw')
            for owner,name in ((torch.nn.Module,'_call_impl'),(engine.model,'signal_forward'),(torch.optim.AdamW,'step'),
                (engine.source_sampler,'draw'),(engine.real_sampler,'draw'),(engine.streams,'draw')):
                guards.enter_context(patch.object(owner,name,side_effect=forbidden))
            report['periodic_import']=engine.import_fix01_full_periodic6000(ROOT/ref['path'],expected_sha256=ref['sha256'],fix01_identity=fix01_identity(cfg))
        if engine.update!=6000 or engine.new_G_updates!=0 or engine.d_updates!=dict(fast=3000,slow=3000,common_level=3000):raise RuntimeError('Native import counters differ')
        del engine;engine=None;del placeholders,statistics,common;gc.collect();torch.cuda.empty_cache()
        saved_env={name:os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES','TMP','TEMP')}
        try:
            from check_window008_resume01_engine import engineering_engine_from_bundle
        finally:
            for name,value in saved_env.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
        bundle_path=ROOT/'receipts/window008_resume01_20261013/engine_cpu_01/ENGINEERING_FIXTURE_INPUTS_FORMAL_BATCH.pt'
        parser_path=bundle_path.parent/'fabricated_fix01_periodic6000.formal_batch.engineering.pt'
        if sha(bundle_path)!='73727307ff88b9a78d5f25d1d15f6a1acbfea09c465c99f8395a3a89f61eb330' or sha(parser_path)!='080ce349ae714c3318d636fd4200b59c09b6df4a21caff746256a78a3bab5233':raise RuntimeError('Frozen artificial bundle differs')
        bundle=torch.load(bundle_path,map_location='cpu',weights_only=False)
        report.update(fixture_bundle_sha256=sha(bundle_path),artificial_parser_sha256=sha(parser_path),
            artificial_initial_D_installed_from_declared_bundle_only=True,artificial_assigned6000_counters_are_not_executed_updates=True)
        for arm in ('full','simple'):
            base_g=report['engineering_G_updates'];base_d=report['engineering_D_updates']
            engine=engineering_engine_from_bundle(bundle,arm=arm,device=device)
            if arm=='full':engine.import_fix01_full_periodic6000(parser_path,expected_sha256=sha(parser_path),fix01_identity=bundle['fix01_identity'])
            forward=type(engine.model).signal_forward
            def meter(self,source,response):
                report['artificial_forward_calls']+=1;report['artificial_forward_rows']+=len(source)
                return forward(self,source,response)
            engine.model.signal_forward=types.MethodType(meter,engine.model)
            for stage in (('joint_stochastic',) if arm=='full' else ('mean_response','joint_stochastic')):
                engine.set_stage(stage);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                began=time.monotonic();metrics=engine.step(audit=True,verify_replay=True);torch.cuda.synchronize()
                peak=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
                if peak>limit or not metrics['same_version_replay_verified']:raise RuntimeError('Native synthetic replay/capacity differs')
                report['steps'].append(dict(arm=arm,stage=stage,seconds=time.monotonic()-began,peak_GPU_bytes=peak,metrics=metrics))
                report['engineering_G_updates']=base_g+engine.new_G_updates;report['engineering_D_updates']=base_d+sum(engine.new_D_updates.values())
            engine.enforce_frozen_references();del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        if report['engineering_G_updates']!=3 or report['engineering_D_updates']!=6 or report['artificial_forward_rows']!=6144:raise RuntimeError('Native engineering accounting differs')
        if sources()!=report['source_sha256']:raise RuntimeError('Native QA source changed')
        report.update(status='passed_native_resume01_engineering_and_import',finished_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
        write_new(out/'ENGINEERING_RECEIPT.json',serial(report))
        print(json.dumps(dict(status=report['status'],engineering_G=3,engineering_D=6,science_G=0,native_import_G=6000)))
    except BaseException as error:
        if engine is not None and engine.engineering_fixture:
            report['engineering_G_updates']=base_g+engine.new_G_updates;report['engineering_D_updates']=base_d+sum(engine.new_D_updates.values())
        report.update(status='failed_native_engineering_preserved',error=repr(error));write_new(out/'ENGINEERING_RECEIPT.json',serial(report));raise

if __name__=='__main__':main()
