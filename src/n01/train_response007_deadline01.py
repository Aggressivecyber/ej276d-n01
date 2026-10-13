"""Time-bounded partial007 stages, preserving the later fixed endpoint role."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
import numpy as np
import torch

from .data import load_training,sha256,write_json
from .evaluate_integration006 import clean
from .evaluate_response007 import generate_split,paired_width_bootstrap,distance_receipt,save_npz
from .losses_v2 import N01LossV2
from .noise_validation import balanced_real_domains
from .response007 import (ARMS,HYPOTHESES,Response007Engine,SignalResponse007,common_initial_state,
    response_stage,state_digest,make_extension,loss_tensor_statistics)
from .train import atomic_torch_save,utc
from .train_response007_statusfix01 import bind,restore_first_update,training_status

ROOT=Path(__file__).resolve().parents[2]


def deadline_reached(bound,started):
    return (datetime.now(timezone.utc).timestamp()>=datetime.fromisoformat(bound['stop_updates_UTC']).timestamp()
            or time.monotonic()-started>=bound['wall_limit_s']-60)


@torch.no_grad()
def diagnose(data,cfg,bound,noise,stats,out,device):
    folder=out/'TRAIN_diagnostics';folder.mkdir(exist_ok=False)
    diag=copy.deepcopy(cfg);diag['seeds'].update(bound['diagnostic_seeds'])
    models={};hashes={}
    for arm in ARMS:
        checkpoint=torch.load(out/'train'/arm/'bounded_04000.pt',map_location='cpu',weights_only=False)
        if checkpoint['G_updates']!=4000 or checkpoint['D_updates']!={h:1000 for h in HYPOTHESES}:
            raise ValueError('Wrong partial checkpoint')
        model=SignalResponse007(stats['model'],arm);model.load_state_dict(checkpoint['model']);model.set_stage('joint_stochastic')
        model.eval().requires_grad_(False);model.to(device);models[arm]=model;hashes[arm]=state_digest(model.state_dict())
        del checkpoint
    extension=make_extension(noise,device)
    selected=np.random.default_rng(diag['seeds']['selected_TRAIN']).permutation(len(data['source']))[:2048]
    real_a,real_b=balanced_real_domains(data['real_groups'],2048,diag['seeds']['real_TRAIN'])
    generated=generate_split('TRAIN',data,models,extension,stats,diag,folder,selected,device)
    sensitivity,bootstrap=paired_width_bootstrap(generated,data,stats,diag,'TRAIN')
    save_npz(folder/'TRAIN_bootstrap.npz',statistics=bootstrap)
    objective=N01LossV2(loss_tensor_statistics(stats['loss']),seed=diag['seeds']['train_projection'],
        evaluation_seed=diag['seeds']['evaluation_projection']).to(device)
    context=objective.create_evaluation_context()
    distances={a:{h:distance_receipt(objective,generated['matched'][a][h],data['real'][real_a],context) for h in HYPOTHESES} for a in ARMS}
    distances['REAL_REAL']=distance_receipt(objective,data['real'][real_b],data['real'][real_a],context)
    for a,m in models.items():
        if state_digest(m.state_dict())!=hashes[a]:raise RuntimeError('Diagnostic changed response')
    extension.assert_frozen()
    report=dict(UTC=utc(),status='partial_stage_TRAIN_diagnostics_complete',updates_per_arm=4000,
        original_endpoint_updates_per_arm=19000,original_endpoint_reached=False,DEV_waveforms_read=False,C2ST_updates=0,
        matched_distances=distances,worst_hypothesis_distances={a:{k:max(distances[a][h]['components'][k] for h in HYPOTHESES) for k in ['wave','QPSD','noise']} for a in ARMS},
        sensitivity=sensitivity,engineering=generated['engineering'],population_summary=generated['summary'],
        matched_selection=dict(source_rows=selected,real_A=real_a,real_B=real_b,real_rule='balanced_original_group',original_population_separately_reported=True),
        scientific_status='response_match_not_established',scientific_pass_threshold=None,independent_acquisition_available=False)
    write_json(folder/'SUMMARY.json',clean(report));return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bound-config',required=True)
    p.add_argument('--source-manifest',required=True);p.add_argument('--container',required=True);p.add_argument('--identities',required=True)
    p.add_argument('--out',required=True);p.add_argument('--resume-checkpoint',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    engine=None;completed=[];started=time.monotonic()
    def stop(signum,frame):raise TimeoutError('Owned bounded job termination signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        cfg,manifest=bind(args.config,args.source_manifest,args.container,args.identities)
        bound=json.loads(Path(args.bound_config).read_text(encoding='utf8'))
        if sha256(args.bound_config)!=manifest['bound_config_sha256'] or bound['updates_per_arm']!=4000 or bound['cumulative_G_budget']!=8000:
            raise ValueError('Time-bound protocol mismatch')
        if sha256(args.resume_checkpoint)!=bound['resume_checkpoint_sha256']:raise ValueError('First update checkpoint changed')
        if deadline_reached(bound,started):raise TimeoutError('Deadline already reached; no new science')
        torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(2)
        if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:cfg['cpu_threads']])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        device=torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_limit_GiB']*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        data=load_training(args.container,args.identities)
        stats=json.loads((ROOT/cfg['references']['statistics001']['path']).read_text(encoding='utf8'))
        noise=torch.load(ROOT/cfg['references']['noise005']['path'],map_location='cpu',weights_only=False)
        common=common_initial_state(stats['model'],cfg['seeds']['initialization'])
        atomic_torch_save(dict(model=common,seed=cfg['seeds']['initialization']),out/'common_initial_state.pt')
        write_json(out/'BINDING.json',dict(UTC=utc(),config_sha256=sha256(args.config),bound_config_sha256=sha256(args.bound_config),
            source_manifest_sha256=sha256(args.source_manifest),previous_G_updates=1,new_G_budget=7999,cumulative_G_budget=8000,
            planned_D_updates=6000,planned_C2ST_updates=0,noise_updates=0,DEV_waveforms_read=False,
            SOURCE_TRAIN_rows=len(data['source']),REAL_TRAIN_rows=len(data['real'])))
        for arm in ARMS:
            directory=out/'train'/arm;directory.mkdir(parents=True,exist_ok=False)
            engine=Response007Engine(data,stats,cfg,noise,common,arm=arm,device=device);engine.set_stage('mean_response')
            if arm=='deterministic':
                value=restore_first_update(engine,args.resume_checkpoint,cfg)
                shutil.copy2(ROOT/'provenance/response007_statusfix01/step00001_training.jsonl',directory/'training.jsonl')
                write_json(out/'RESUME_VERIFICATION.json',dict(UTC=utc(),G_updates=engine.update,D_updates=engine.d_updates,
                    model_state_sha256=state_digest(engine.model.state_dict()),checkpoint_sha256=sha256(args.resume_checkpoint),
                    common_initial_state_sha256=engine.common_state_digest,noise005_state_sha256=engine.noise_state_digest,
                    exact_explicit_stream_restore=True,new_forward_or_update_for_restore=0));del value
            engine.checkpoint(directory/('resume_start.pt' if arm=='deterministic' else 'initial.pt'))
            while engine.update<bound['updates_per_arm']:
                if deadline_reached(bound,started):raise TimeoutError('Pre-update deadline reached; save and stop')
                stage=response_stage(engine.update,cfg);changed=stage!=engine.stage
                if changed:engine.set_stage(stage)
                next_update=engine.update+1;audit=next_update<=2 or changed or next_update%1000==0
                metrics=engine.step(audit=audit,verify_replay=next_update<=2 or changed)
                with (directory/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
                total_G=sum(x['G_updates'] for x in completed)+engine.update
                total_D=sum(sum(x['D_updates'].values()) for x in completed)+sum(engine.d_updates.values())
                if audit or engine.update%25==0:
                    status=training_status(metrics,total_G,total_D,time.monotonic()-started)
                    status.update(attempt_id='deadline01',target_G_updates=8000,target_D_updates=6000,stop_updates_UTC=bound['stop_updates_UTC'],original_full_endpoint_reached=False)
                    write_json(out/'status.json',status)
                    print(json.dumps({k:status[k] for k in ['utc','arm','stage','G_updates','total_G_updates','total_D_updates','weighted_total','update_seconds','coverage']}),flush=True)
                if engine.update%250==0:engine.checkpoint(directory/'latest.pt')
                if engine.update in (3000,4000):engine.checkpoint(directory/('stage_03000.pt' if engine.update==3000 else 'bounded_04000.pt'))
            completed.append(dict(arm=arm,G_updates=engine.update,D_updates=dict(engine.d_updates),coverage=engine.last_metrics['coverage'],
                endpoint_sha256=sha256(directory/'bounded_04000.pt')))
            write_json(directory/'PARTIAL_STAGE_COMPLETE.json',completed[-1])
            del engine;engine=None;gc.collect();torch.cuda.empty_cache()
        if deadline_reached(bound,started):raise TimeoutError('Training saved; no diagnostic starts after deadline')
        report=diagnose(data,cfg,bound,noise,stats,out,device)
        files={x.relative_to(out).as_posix():sha256(x) for x in sorted(out.rglob('*')) if x.is_file() and x.name not in ['COMPLETE.json','status.json']}
        result=dict(UTC=utc(),status='bounded_stage_comparison_complete',G_updates=8000,new_execution_G_updates=7999,D_updates=6000,
            C2ST_updates=0,noise_updates=0,arms=completed,files=files,DEV_waveforms_read=False,
            source_manifest_sha256=sha256(args.source_manifest),bound_config_sha256=sha256(args.bound_config),
            original_19000_per_arm_endpoint_reached=False,scientific_status='response_match_not_established',elapsed_seconds=time.monotonic()-started)
        write_json(out/'COMPLETE.json',result);write_json(out/'status.json',result)
    except BaseException as error:
        if engine is not None:
            engine.checkpoint(out/'STOP_CHECKPOINT.pt');atomic_torch_save(engine.last_context,out/'STOP_CONTEXT.pt')
        prior=[x for x in completed if engine is None or x['arm']!=engine.arm]
        total_G=sum(x['G_updates'] for x in prior)+(engine.update if engine is not None else 0)
        total_D=sum(sum(x['D_updates'].values()) for x in prior)+(sum(engine.d_updates.values()) if engine is not None else 0)
        result=dict(UTC=utc(),status='saved_at_time_boundary' if isinstance(error,TimeoutError) else 'failed_preserved',error=repr(error),
            traceback=traceback.format_exc(),current_arm=engine.arm if engine is not None else None,
            completed_arms=completed,G_updates=max(1,total_G),previous_completed_G_updates=1,new_execution_G_updates=max(0,total_G-1),D_updates=total_D,C2ST_updates=0,noise_updates=0,
            elapsed_seconds=time.monotonic()-started,DEV_waveforms_read=False,original_endpoint_reached=False,scientific_status='response_match_not_established')
        write_json(out/'STOP.json',result);write_json(out/'status.json',result);raise

if __name__=='__main__':main()
