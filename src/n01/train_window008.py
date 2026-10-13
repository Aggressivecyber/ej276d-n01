"""Frozen three-arm19000-step window008 training; endpoint evaluation is separate."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch

from .data import sha256,write_json
from .response008 import ARMS,HYPOTHESES,Response008Engine,common_initial_state,response_stage,state_digest
from .train import atomic_torch_save,utc

ROOT=Path(__file__).resolve().parents[2]
RUN_ID='window008_20261013'


def bind(config_path,manifest_path,dataset,statistics,container,identities):
    cfg=json.loads(Path(config_path).read_text(encoding='utf8'))
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if cfg['run_id']!=RUN_ID or manifest['run_id']!=RUN_ID:raise ValueError('Run ID differs')
    if sha256(config_path)!=manifest['config_sha256']:raise ValueError('Frozen configuration differs')
    if tuple(cfg['arms'])!=ARMS or cfg['response']['updates_per_arm']!=19000:raise ValueError('Arm endpoint differs')
    if cfg['response']['total_G_updates']!=57000 or cfg['response']['total_D_updates']!=144000:raise ValueError('Budget differs')
    if cfg['noise_updates']!=0 or cfg['population_updates']!=0:raise ValueError('Unregistered update type')
    for name,digest in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=digest:raise ValueError('Frozen source/input differs: '+name)
    checks=[(dataset,cfg['dataset_sha256']),(statistics,cfg['statistics_sha256']),
            (container,cfg['train_container_sha256']),(identities,cfg['identities_sha256'])]
    for path,digest in checks:
        if sha256(path)!=digest:raise ValueError('Bound input differs: '+str(path))
    if not manifest['before_scientific_model_forward']:raise ValueError('Missing prior training freeze')
    return cfg,manifest


def configure_runtime(cfg,device):
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Science requires native CUDA')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('config','source-manifest','dataset','statistics','container','identities','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);out=Path(args.out).resolve()
    if os.name=='nt' and out.drive.upper() not in ('E:','D:'):raise ValueError('Output must stay on E/D')
    if out.exists():raise FileExistsError('Fresh run directory required; historical failures are preserved')
    out.mkdir(parents=True)
    engine=None;completed=[];began=time.monotonic();cfg=None
    def stop(signum,frame):raise TimeoutError('Owned training process received termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        cfg,manifest=bind(args.config,args.source_manifest,args.dataset,args.statistics,args.container,args.identities)
        device=torch.device(args.device);configure_runtime(cfg,device)
        write_json(out/'BINDING.json',dict(UTC=utc(),run_id=RUN_ID,pid=os.getpid(),
            config_sha256=sha256(args.config),source_manifest_sha256=sha256(args.source_manifest),
            dataset_sha256=sha256(args.dataset),statistics_sha256=sha256(args.statistics),
            raw_container_sha256=cfg['train_container_sha256'],identities_sha256=cfg['identities_sha256'],
            before_prepared_numeric_decode=True,DEV_waveforms_read=False))
        with np.load(args.dataset,allow_pickle=False) as z:data={name:z[name].copy() for name in z.files}
        if data['source'].shape!=(6593,256) or data['real'].shape!=(21047,256):raise ValueError('All-TRAIN effective dimensions changed')
        if data['source_carrier384'].shape!=(6593,384) or data['real_baseline100'].shape!=(21047,100):raise ValueError('Carrier/baseline dimensions changed')
        if set(np.unique(data['source_role']))!={'TRAIN'} or set(np.unique(data['real_role']))!={'TRAIN'}:raise ValueError('Non-TRAIN waveforms in prepared input')
        if np.intersect1d(data['source_ids'],data['source_val_ids']).size or np.intersect1d(data['real_ids'],data['real_val_original_index']).size:
            raise ValueError('Parent split overlap')
        statistics=json.loads(Path(args.statistics).read_text(encoding='utf8'))
        if statistics['prepared_dataset_sha256']!=cfg['dataset_sha256'] or statistics['statistics_split']!='TRAIN':raise ValueError('Statistics source/split differs')
        ref=cfg['noise_reference']
        if sha256(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Frozen005 differs')
        noise=torch.load(ROOT/ref['path'],map_location='cpu',weights_only=False)
        if noise.get('update')!=5000 or noise.get('schema_version')!=5 or noise.get('model_type')!='ObservedBaselineNoiseV3' or noise.get('engineering') is not False:
            raise ValueError('Wrong frozen005 endpoint')
        common=common_initial_state(statistics['model'],cfg['seeds']['initialization'])
        atomic_torch_save(dict(model=common,seed=cfg['seeds']['initialization'],statistics_sha256=cfg['statistics_sha256']),out/'common_initial_state.pt')
        write_json(out/'TRAINING_INPUTS.json',dict(UTC=utc(),SOURCE_rows=6593,REAL_rows=21047,effective_samples=256,internal_carrier_samples=384,
            source_ADC=False,source_unique=len(np.unique(data['source_ids'])),real_unique=len(np.unique(data['real_ids'])),
            common_initial_state_sha256=state_digest(common),DEV_waveforms_read=False,source_count_weights='all_one',
            original_source_count_weights_preserved=True,original_physical_weights='unknown',new_noise_updates=0,new_population_updates=0))
        for arm in ARMS:
            folder=out/'train'/arm;folder.mkdir(parents=True,exist_ok=False)
            engine=Response008Engine(data,statistics,cfg,noise,common,arm=arm,device=device)
            engine.set_stage('mean_response');engine.checkpoint(folder/'initial.pt')
            while engine.update<cfg['response']['updates_per_arm']:
                if time.monotonic()-began>=cfg['wall_limit_s']-30:raise TimeoutError('Registered wall bound reached before next update')
                stage=response_stage(engine.update,cfg);changed=stage!=engine.stage
                if changed:engine.set_stage(stage)
                update=engine.update+1
                audit=update<=2 or changed or update%cfg['engineering']['gradient_audit_interval']==0
                metrics=engine.step(audit=audit,verify_replay=update<=cfg['engineering']['waveform_replay_first_steps'] or changed)
                with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
                total_g=sum(row['G_updates'] for row in completed)+engine.update
                total_d=sum(sum(row['D_updates'].values()) for row in completed)+sum(engine.d_updates.values())
                if audit or engine.update%cfg['status_interval']==0:
                    status=dict(status='training_window008',run_id=RUN_ID,pid=os.getpid(),
                        total_G_updates=total_g,total_D_updates=total_d,target_G_updates=57000,target_D_updates=144000,
                        elapsed_seconds=time.monotonic()-began,DEV_waveforms_read=False,**metrics)
                    write_json(out/'status.json',status)
                    print(json.dumps({k:status[k] for k in ('utc','status','arm','stage','G_updates','total_G_updates','total_D_updates','weighted_total','update_seconds','coverage')}),flush=True)
                if engine.update%cfg['checkpoint_interval']==0:engine.checkpoint(folder/'latest.pt')
                if engine.update in (3000,15000,19000):
                    path=folder/('final.pt' if engine.update==19000 else f'stage_{engine.update:05d}.pt')
                    if path.exists():raise FileExistsError('Do not overwrite a fixed endpoint')
                    engine.checkpoint(path)
                if engine.update%1000==0:
                    if sum(path.stat().st_size for path in out.rglob('*') if path.is_file())>cfg['output_limit_MiB']*1024**2:
                        raise RuntimeError('Registered output bound exceeded')
            if engine.update!=19000 or engine.d_updates!={h:16000 for h in HYPOTHESES}:raise RuntimeError('Stage update accounting differs')
            if engine.last_metrics['coverage']['source_unique']!=6593 or engine.last_metrics['coverage']['real_unique']!=21047:
                raise RuntimeError('All-parent coverage incomplete')
            done=dict(arm=arm,G_updates=engine.update,D_updates=dict(engine.d_updates),coverage=engine.last_metrics['coverage'],
                final_sha256=sha256(folder/'final.pt'),UTC=utc())
            completed.append(done);write_json(folder/'TRAINING_COMPLETE.json',done)
            del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        files={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('status.json','TRAINING_COMPLETE.json')}
        result=dict(UTC=utc(),status='fixed_window008_training_complete_pending_evaluation',run_id=RUN_ID,
            G_updates=57000,D_updates=144000,C2ST_updates=0,noise_updates=0,population_updates=0,DEV_waveforms_read=False,
            arms=completed,files=files,config_sha256=sha256(args.config),source_manifest_sha256=sha256(args.source_manifest),
            elapsed_seconds=time.monotonic()-began,scientific_status='response_match_not_established',evaluation_complete=False)
        write_json(out/'TRAINING_COMPLETE.json',result);write_json(out/'status.json',result)
    except BaseException as error:
        if engine is not None:
            try:engine.checkpoint(out/'FAILURE_CHECKPOINT.pt')
            except Exception as saved_error:write_json(out/'CHECKPOINT_SAVE_ERROR.json',dict(error=repr(saved_error)))
            try:atomic_torch_save(engine.last_context,out/'FAILURE_CONTEXT.pt')
            except Exception as saved_error:write_json(out/'CONTEXT_SAVE_ERROR.json',dict(error=repr(saved_error)))
        prior=[row for row in completed if engine is None or row['arm']!=engine.arm]
        result=dict(UTC=utc(),status='failed_or_bounded_stop_preserved',run_id=RUN_ID,error=repr(error),traceback=traceback.format_exc(),
            G_updates=sum(row['G_updates'] for row in prior)+(engine.update if engine else 0),
            D_updates=sum(sum(row['D_updates'].values()) for row in prior)+(sum(engine.d_updates.values()) if engine else 0),
            completed_arms=completed,current_arm=engine.arm if engine else None,DEV_waveforms_read=False,
            C2ST_updates=0,noise_updates=0,population_updates=0,elapsed_seconds=time.monotonic()-began,
            scientific_status='response_match_not_established',evaluation_complete=False)
        write_json(out/'FAILURE.json',result);write_json(out/'status.json',result);raise


if __name__=='__main__':main()
