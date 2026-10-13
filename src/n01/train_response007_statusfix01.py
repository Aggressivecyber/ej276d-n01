"""Bounded native007 runner; complete fixed training before descriptive eval."""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import shutil
from pathlib import Path
import time
import traceback

import torch

from .data import load_training,sha256,write_json
from .extension006 import HYPOTHESES006
from .response007 import ARMS,HYPOTHESES,Response007Engine,common_initial_state,response_stage,state_digest
from .train import atomic_torch_save,utc

ROOT=Path(__file__).resolve().parents[2]
ID='response_007_20261013'


def training_status(metrics,total_G,total_D,elapsed):
    result=dict(metrics)
    result.update(status='training_response007',pid=os.getpid(),run_id=ID,total_G_updates=total_G,total_D_updates=total_D,
        total_C2ST_updates=0,elapsed_seconds=elapsed,new_execution_G_updates=total_G-1,
        previous_completed_G_updates=1,attempt_id='status_fix01')
    return result


def restore_first_update(engine,checkpoint,config):
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (value['arm']!='deterministic' or value['G_updates']!=1 or any(value['D_updates'].values())
            or value['stage']!='mean_response' or value['config']!=config):
        raise ValueError('Only the bound completed first update may be resumed')
    engine.restore(checkpoint)
    if state_digest(engine.model.state_dict())!=state_digest(value['model']):raise RuntimeError('Resume model state changed')
    if not all(torch.equal(engine.streams.state_dict()[k],v.cpu()) for k,v in value['streams'].items()):
        raise RuntimeError('Resume stochastic streams changed')
    return value



def bind(config_path,manifest_path,container,identities):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if cfg['run_id']!=ID or manifest['run_id']!=ID:raise ValueError('Run ID changed')
    if sha256(config_path)!=manifest['config_sha256']:raise ValueError('Configuration changed')
    if cfg['hypotheses']!=list(HYPOTHESES006) or tuple(cfg['arms'])!=ARMS:raise ValueError('Hypotheses or arms changed')
    if cfg['response']['total_G_updates']!=38000 or cfg['response']['total_D_updates']!=96000:raise ValueError('Budget changed')
    if not manifest['before_new_scientific_forward'] or not manifest['before_new_scientific_array_decode']:
        raise ValueError('Missing prior source freeze')
    required={'src/n01/response007.py','src/n01/train_response007.py','src/n01/evaluate_response007.py',
        'src/n01/model.py','src/n01/extension006.py','src/n01/losses.py','src/n01/losses_v2.py',
        'src/n01/electronics_v3.py','src/n01/data.py','scripts/supervise.py','scripts/launch_response007.sh',
        'response007_config.json','docs/RESPONSE_007_PROTOCOL.md','receipts/'+ID+'/PREREGISTRATION.json'}
    if not required<=set(manifest['files']):raise ValueError('Source binding omits required files')
    for name,digest in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or sha256(p)!=digest:raise ValueError('Frozen source mismatch: '+name)
    if sha256(container)!=cfg['train_container_sha256'] or sha256(identities)!=cfg['identities_sha256']:
        raise ValueError('Raw input changed before decode')
    for name,ref in cfg['references'].items():
        if sha256(ROOT/ref['path'])!=ref['sha256'] or manifest['files'].get(ref['path'])!=ref['sha256']:
            raise ValueError('Reference changed/unbound: '+name)
    registration=json.loads((ROOT/'receipts'/ID/'PREREGISTRATION.json').read_text(encoding='utf8'))
    if registration['config_sha256']!=sha256(config_path) or registration['protocol_sha256']!=sha256(ROOT/'docs/RESPONSE_007_PROTOCOL.md'):
        raise ValueError('Preregistered config/protocol changed')
    return cfg,manifest


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--source-manifest',required=True)
    p.add_argument('--container',required=True);p.add_argument('--identities',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--resume-checkpoint',required=True)
    args=p.parse_args(argv);out=Path(args.out).resolve()
    if os.name=='nt' and out.drive.upper() not in ('E:','D:'):raise ValueError('Output must be E/D')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Science output must be fresh; no implicit retry')
    out.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):raise TimeoutError('Owned job received termination signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    engine=None;completed=[];started=time.monotonic();cfg=None
    try:
        cfg,manifest=bind(args.config,args.source_manifest,args.container,args.identities)
        amendment=json.loads((ROOT/'receipts'/ID/'status_fix01/PREREGISTRATION.json').read_text(encoding='utf8'))
        required={'src/n01/train_response007_statusfix01.py','scripts/launch_response007_statusfix01.sh',
            'docs/RESPONSE_007_STATUS_FIX01.md','receipts/'+ID+'/status_fix01/PREREGISTRATION.json'}
        if not required<=set(manifest['files']):raise ValueError('Missing amendment sources')
        if sha256(args.resume_checkpoint)!=amendment['resume_checkpoint_sha256']:
            raise ValueError('Resume checkpoint differs from the preserved first update')
        if manifest.get('previous_G_updates')!=1 or manifest.get('new_execution_G_budget')!=37999:
            raise ValueError('Resume update accounting changed')
        torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
        if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        device=torch.device(args.device)
        if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Registered science requires CUDA')
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        write_json(out/'BINDING.json',dict(run_id=ID,UTC=utc(),config_sha256=sha256(args.config),
            source_manifest_sha256=sha256(args.source_manifest),source_files=len(manifest['files']),
            container_sha256=cfg['train_container_sha256'],identities_sha256=cfg['identities_sha256'],before_scientific_decode=True))
        data=load_training(args.container,args.identities)
        stats=json.loads((ROOT/cfg['references']['statistics001']['path']).read_text(encoding='utf8'))
        noise=torch.load(ROOT/cfg['references']['noise005']['path'],map_location='cpu',weights_only=False)
        cfg005=json.loads((ROOT/cfg['references']['config005']['path']).read_text(encoding='utf8'))
        if (noise.get('schema_version')!=5 or noise.get('engineering') is not False or noise.get('update')!=5000
                or noise.get('model_type')!='ObservedBaselineNoiseV3' or noise['config']!=cfg005
                or noise.get('response_integration_allowed') is not False):
            raise ValueError('005 endpoint or its unchanged historical support scope differs')
        common=common_initial_state(stats['model'],cfg['seeds']['initialization'])
        atomic_torch_save(dict(model=common,seed=cfg['seeds']['initialization'],statistics_sha256=cfg['references']['statistics001']['sha256']),out/'common_initial_state.pt')
        write_json(out/'TRAINING_INPUTS.json',dict(SOURCE_rows=len(data['source']),REAL_rows=len(data['real']),
            source_unique=len(set(data['source_ids'].tolist())),real_unique=len(set(data['real_ids'].tolist())),
            common_initial_state_sha256=state_digest(common),noise_weight_file_sha256=cfg['references']['noise005']['sha256'],
            statistics_file_sha256=cfg['references']['statistics001']['sha256'],DEV_waveforms_decoded_for_training=False,
            source_count_weights='all_one',unknown_physical_weights='unknown',subagents=False))
        for arm in ARMS:
            directory=out/'train'/arm;directory.mkdir(parents=True,exist_ok=False)
            engine=Response007Engine(data,stats,cfg,noise,common,arm=arm,device=device)
            engine.set_stage('mean_response')
            if arm=='deterministic':
                restored=restore_first_update(engine,args.resume_checkpoint,cfg)
                journal=ROOT/'provenance/response007_statusfix01/step00001_training.jsonl'
                lines=journal.read_text(encoding='utf8').splitlines()
                if len(lines)!=1 or json.loads(lines[0])['G_updates']!=1:raise ValueError('First update journal differs')
                shutil.copy2(journal,directory/'training.jsonl')
                engine.checkpoint(directory/'resume_start.pt')
                write_json(out/'RESUME_VERIFICATION.json',dict(UTC=utc(),resumed_G_updates=engine.update,
                    resumed_D_updates=engine.d_updates,checkpoint_sha256=sha256(args.resume_checkpoint),
                    model_state_sha256=state_digest(engine.model.state_dict()),
                    common_initial_state_sha256=engine.common_state_digest,noise005_state_sha256=engine.noise_state_digest,
                    explicit_streams_verified=True,new_scientific_forward_for_restore=0,new_G_updates_for_restore=0))
                del restored
            else:engine.checkpoint(directory/'initial.pt')
            while engine.update<cfg['response']['updates_per_arm']:
                stage=response_stage(engine.update,cfg);changed=stage!=engine.stage
                if changed:engine.set_stage(stage)
                next_update=engine.update+1
                audit=next_update<=2 or changed or next_update%cfg['engineering']['gradient_audit_interval']==0
                metrics=engine.step(audit=audit,verify_replay=next_update<=cfg['engineering']['waveform_replay_first_steps'] or changed)
                with (directory/'training.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
                total_G=sum(x['G_updates'] for x in completed)+engine.update
                total_D=sum(sum(x['D_updates'].values()) for x in completed)+sum(engine.d_updates.values())
                if audit or engine.update%cfg['status_interval']==0:
                    status=training_status(metrics,total_G,total_D,time.monotonic()-started)
                    write_json(out/'status.json',status)
                    print(json.dumps({k:status[k] for k in ['utc','status','arm','stage','G_updates','total_G_updates','total_D_updates','weighted_total','update_seconds','coverage']}),flush=True)
                if engine.update%cfg['checkpoint_interval']==0:engine.checkpoint(directory/'latest.pt')
                if engine.update in (3000,15000,19000):
                    path=directory/('final.pt' if engine.update==19000 else f'stage_{engine.update:05d}.pt')
                    if path.exists():raise FileExistsError('Stage endpoint must not be overwritten')
                    engine.checkpoint(path)
                if engine.update%1000==0:
                    size=sum(x.stat().st_size for x in out.rglob('*') if x.is_file())
                    if size>cfg['output_limit_MiB']*1024**2:raise RuntimeError('Science output size limit exceeded')
            completed.append(dict(arm=arm,G_updates=engine.update,D_updates=engine.d_updates,
                coverage=engine.last_metrics['coverage'],final_sha256=sha256(directory/'final.pt')))
            write_json(directory/'TRAINING_COMPLETE.json',completed[-1])
            del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        write_json(out/'TRAINING_COMPLETE.json',dict(UTC=utc(),status='fixed_response_training_complete',arms=completed,
            G_updates=38000,D_updates=96000,noise_updates=0,population_updates=0,scientific_status='response_match_not_established'))
        from .evaluate_response007 import evaluate_run
        evaluation=evaluate_run(out,cfg,args.container,args.identities,device=device)
        files={x.relative_to(out).as_posix():sha256(x) for x in sorted(out.rglob('*')) if x.is_file() and x.name not in ('status.json','COMPLETE.json')}
        write_json(out/'COMPLETE.json',dict(UTC=utc(),status='fixed_response007_training_and_evaluation_complete',
            run_id=ID,G_updates=38000,new_execution_G_updates=37999,previous_G_updates=1,D_updates=96000,C2ST_updates=evaluation['C2ST_updates'],noise_updates=0,
            source_manifest_sha256=sha256(args.source_manifest),config_sha256=sha256(args.config),files=files,
            scientific_status='response_match_not_established',elapsed_seconds=time.monotonic()-started))
        write_json(out/'status.json',dict(status='completed_fixed_response007',UTC=utc(),pid=None,G_updates=38000,D_updates=96000,
            C2ST_updates=evaluation['C2ST_updates'],scientific_status='response_match_not_established'))
    except BaseException as error:
        current_G=engine.update if engine is not None else 0
        current_D=sum(engine.d_updates.values()) if engine is not None else 0
        if engine is not None:
            try:engine.checkpoint(out/'FAILURE_CHECKPOINT.pt')
            except Exception:pass
            atomic_torch_save(engine.last_context,out/'FAILURE_CONTEXT.pt')
        c2st_updates=sum(json.loads(p.read_text(encoding='utf8'))['completed_updates'] for p in (out/'evaluation'/'classifiers').rglob('PROGRESS.json')) if (out/'evaluation'/'classifiers').exists() else 0
        finished=[x for x in completed if engine is None or x['arm']!=engine.arm]
        receipt=dict(UTC=utc(),status='failed_preserved_no_automatic_retry',error=repr(error),traceback=traceback.format_exc(),
            completed_arms=completed,current_arm=engine.arm if engine is not None else None,
            G_updates=sum(x['G_updates'] for x in finished)+current_G,
            D_updates=sum(sum(x['D_updates'].values()) for x in finished)+current_D,C2ST_updates=c2st_updates,previous_G_updates=1,new_execution_G_updates=max(0,sum(x['G_updates'] for x in finished)+current_G-1),
            noise_updates=0,scientific_status='response_match_not_established',elapsed_seconds=time.monotonic()-started)
        write_json(out/'FAILURE.json',receipt);write_json(out/'status.json',receipt)
        raise

if __name__=='__main__':main()
