"""N01 repair_002: independent noise fit, binding gate, then two response arms."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .data import (FullCoverageSampler,fit_model_statistics,load_training,load_validation,sha256,write_json)
from .electronics_v2 import ElectronicsNoiseV2,fit_electronics_initialization,make_n01_response_v2
from .losses import fit_loss_statistics
from .losses_v2 import BaselineDistributionLoss,fit_baseline_statistics,N01LossV2
from .model import digitize
from .noise_validation import evaluate_noise
from .train import Engine,atomic_torch_save,seed_all,utc


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(v.shape)).encode());h.update(str(v.dtype).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


class BoundLossV2(N01LossV2):
    """Compatibility adapter: caller explicitly binds one context per update.

    No forward increments or samples it. The inherited v1 Engine performs one
    joint endpoint loss and replays the resulting waveform derivative exactly.
    """
    def bind(self,context):
        context.validate()
        self.bound_projection_context=context
        return self

    def _context(self,supplied):
        context=supplied if supplied is not None else getattr(self,'bound_projection_context',None)
        if context is None:raise RuntimeError('Caller must bind an explicit projection context')
        return context.validate()

    def components(self,outputs,real,comparison_weights=None,stage='joint_stochastic',*,projection_context=None):
        return super().components(outputs,real,comparison_weights,stage,projection_context=self._context(projection_context))

    def noise_loss(self,fake,real,weights=None,*,projection_context=None):
        return super().noise_loss(fake,real,weights,projection_context=self._context(projection_context))

    def population_objective(self,fake,real,weights,*,projection_context=None):
        return super().population_objective(fake,real,weights,projection_context=self._context(projection_context))


class RepairEngine(Engine):
    def __init__(self,data,model_stats,loss_stats,config,noise_state,noise_initialization,shared_initial_state,
                 *,variant,device='cuda',noise_gate_sha256):
        super().__init__(data,model_stats,loss_stats,config,variant=variant,arm=variant,
                         learn_population=False,device=device)
        seed_all(self.seed)
        self.model=make_n01_response_v2(model_stats,variant,
            electronics_initialization=noise_initialization,stage='mean_response').to(self.device)
        current=self.model.state_dict()
        # Both variants begin with identical values for every common parameter.
        for name in current:
            if name in shared_initial_state:
                if current[name].shape!=shared_initial_state[name].shape:raise ValueError('Common initialization shape mismatch: '+name)
                current[name]=shared_initial_state[name].to(current[name])
        self.model.load_state_dict(current,strict=True)
        self.model.electronics.load_state_dict(noise_state,strict=True)
        self.model.electronics.requires_grad_(False)
        self.noise_initialization=copy.deepcopy(noise_initialization)
        self.noise_state_hash=state_digest(self.model.electronics.state_dict())
        self.noise_gate_sha256=noise_gate_sha256
        self.loss=BoundLossV2(loss_stats,seed=config['projection_seed'],
                             evaluation_seed=config['response']['evaluation_projection_seed']).to(self.device)
        self.g_optimizer=torch.optim.AdamW(self.model.parameters(),lr=2e-4,weight_decay=1e-4)
        self.population.requires_grad_(False)
        self.stage=None

    def set_stage(self,stage):
        if stage=='baseline':raise ValueError('Validated electronics must never enter old baseline refit')
        super().set_stage(stage)
        self.model.electronics.requires_grad_(False)
        if state_digest(self.model.electronics.state_dict())!=self.noise_state_hash:
            raise RuntimeError('Validated electronics changed')

    def step(self,**kwargs):
        context=self.loss.create_projection_context(self.update+1)
        self.loss.bind(context)
        try:
            result=super().step(**kwargs)
        finally:
            self.last_context['projection_context']=context.state_dict()
        context.validate()
        result['projection_context']=context.metadata()
        result['wave_subcomponents']={k:float(v) for k,v in self.loss.last_wave_components.items()}
        result['electronics_state_sha256']=self.noise_state_hash
        result['population_learning']=False
        if kwargs.get('audit') and state_digest(self.model.electronics.state_dict())!=self.noise_state_hash:
            raise RuntimeError('Electronics mutated during response update')
        self.last_metrics=result
        return result

    def checkpoint(self,path):
        super().checkpoint(path)
        value=torch.load(path,map_location='cpu',weights_only=False)
        value.update(schema_version=2,noise_initialization=self.noise_initialization,
                     electronics_state_sha256=self.noise_state_hash,noise_gate_sha256=self.noise_gate_sha256,
                     loss_version=2,shared_baseline_training_updates=0,
                     last_projection_context=self.loss.bound_projection_context.state_dict())
        atomic_torch_save(value,path)


def _append(path,value):
    with Path(path).open('a',encoding='utf8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def validate_noise_endpoint(state,config):
    if state['config']!=config or state.get('engineering') is not False or state['update']!=config['noise']['updates']:
        raise ValueError('Noise endpoint configuration, budget, or scientific role mismatch')
    if state_digest(state['model'])!=state['electronics_state_sha256']:
        raise ValueError('Noise endpoint state digest mismatch')
    return state


def fit_noise(data,config,out,device,*,engineering=False,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=config['noise'];seed_all(cfg['seed'])
    real=torch.as_tensor(data['real'][:,:100],dtype=torch.float64,device=device)
    initialization=fit_electronics_initialization(data['real'][:,:100])
    statistics=fit_baseline_statistics(torch.from_numpy(data['real'][:,:100]))
    model=ElectronicsNoiseV2(initialization).to(device)
    objective=BaselineDistributionLoss(statistics,seed=cfg['projection_seed'],
                                      evaluation_seed=cfg['evaluation_projection_seed']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['initial_learning_rate'])
    sampler=FullCoverageSampler(len(real),cfg['seed']+101)
    rng=torch.Generator(device=device).manual_seed(cfg['seed']+102)
    update=0;started=time.monotonic()
    if (out/'final.pt').exists():
        if not resume:raise FileExistsError('Preserve prior noise endpoint')
        return validate_noise_endpoint(torch.load(out/'final.pt',map_location='cpu',weights_only=False),config)
    if resume and (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Noise resume configuration mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        sampler.load_state_dict(state['sampler']);rng.set_state(state['noise_rng'].cpu());update=state['update']
    write_json(out/'fit_statistics.json',dict(initialization=initialization,loss=statistics,
        real_TRAIN_rows=len(real),TRAIN_original_groups=len(np.unique(data['real_groups'])),validation_read=False))
    end=3 if engineering else int(cfg['updates'])
    last_noise=None;last_rows=None;context=None
    def checkpoint():
        return dict(schema_version=2,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    initialization=initialization,loss_statistics=statistics,config=config,
                    optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),noise_rng=rng.get_state(),
                    update=update,projection_context=context.state_dict() if context else None,
                    electronics_state_sha256=state_digest(model.state_dict()),saved_UTC=utc(),
                    response_training_updates=0,engineering=engineering)
    try:
        while update<end:
            then=time.monotonic()
            lr=cfg['low_learning_rate'] if update>=cfg['low_learning_rate_start'] else cfg['initial_learning_rate']
            for group in optimizer.param_groups:group['lr']=lr
            last_rows=sampler.draw(cfg['batch_size'])
            target=real[torch.as_tensor(last_rows,device=device)]
            last_noise=model.sample_noise(cfg['batch_size'],device=device,generator=rng,length=100)
            context=objective.create_projection_context(update+1)
            optimizer.zero_grad(set_to_none=True)
            generated=digitize(model(last_noise,length=100))[:,0]
            components=objective.components(generated,target,projection_context=context)
            total=objective.total(components)
            if not torch.isfinite(total):raise FloatingPointError('Noise objective nonfinite')
            total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();context.validate();update+=1
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Noise parameter nonfinite')
            metrics=dict(utc=utc(),update=update,target_updates=end,phase='noise_engineering' if engineering else 'noise_fit',
                learning_rate=lr,total=float(total.detach()),components={k:float(v.detach()) for k,v in components.items()},
                gradient_preclip_norm=float(norm),projection_context=context.metadata(),
                real_unique=int((sampler.visits>0).sum()),real_total=len(real),real_epochs=sampler.completed_epochs,
                update_seconds=time.monotonic()-then,elapsed_seconds=time.monotonic()-started,
                gpu_allocated_bytes=torch.cuda.memory_allocated() if str(device).startswith('cuda') else 0,
                gpu_reserved_bytes=torch.cuda.memory_reserved() if str(device).startswith('cuda') else 0)
            _append(out/'training.jsonl',metrics)
            if update<=5 or update%25==0:
                write_json(out.parent/'status.json',dict(status='training_noise',pid=os.getpid(),**metrics))
                print(json.dumps(metrics),flush=True)
            if update%1000==0:atomic_torch_save(checkpoint(),out/'latest.pt')
        state=checkpoint();atomic_torch_save(state,out/'engineering.pt' if engineering else out/'final.pt')
        write_json(out/'ENGINEERING.json' if engineering else out/'TRAINING_COMPLETE.json',
            dict(status='engineering_discarded' if engineering else 'fixed_noise_endpoint_reached',
                 updates=update,real_unique=int((sampler.visits>0).sum()),real_total=len(real),
                 state_sha256=state['electronics_state_sha256'],elapsed_seconds=time.monotonic()-started,
                 scientific_acceptance_evaluated=False,validation_read=False,response_training_updates=0))
        return state
    except BaseException:
        atomic_torch_save(dict(state=checkpoint(),rows=last_rows,noise=last_noise),out/'failure_state.pt')
        raise


def response_stage(update,config):
    cfg=config['response']
    if update<cfg['mean_response_updates']:return 'mean_response'
    if update<cfg['mean_response_updates']+cfg['joint_stochastic_updates']:return 'joint_stochastic'
    return 'low_learning_rate'


def train_responses(data,config,noise_state,gate_path,out,device,*,engineering=False,resume=False,container=None,identities=None):
    gate=json.loads(Path(gate_path).read_text(encoding='utf8'))
    if not gate['passed']:raise RuntimeError('Noise gate has not passed; pulse training is prohibited')
    if gate.get('electronics_state_sha256')!=noise_state['electronics_state_sha256']:
        raise ValueError('Gate and frozen electronics checkpoint differ')
    model_stats=fit_model_statistics(data)
    loss_stats=fit_loss_statistics(torch.from_numpy(data['real']),torch.from_numpy(data['source']))
    seed_all(config['seed'])
    reference=make_n01_response_v2(model_stats,'full',electronics_initialization=noise_state['initialization'],stage='mean_response')
    reference.electronics.load_state_dict(noise_state['model'])
    shared=reference.state_dict();atomic_torch_save(shared,out/'shared_response_initialization.pt')
    gate_sha=sha256(gate_path)
    for variant in config['response']['arms']:
        directory=out/variant;directory.mkdir(parents=True,exist_ok=True)
        complete=directory/'TRAINING_COMPLETE.json'
        if complete.exists() and not resume:raise FileExistsError('Preserve completed response arm')
        engine=RepairEngine(data,model_stats,loss_stats,config,noise_state['model'],noise_state['initialization'],shared,
                            variant=variant,device=device,noise_gate_sha256=gate_sha)
        if not complete.exists():
            if resume and (directory/'latest.pt').exists():engine.restore(directory/'latest.pt')
            end=3 if engineering else config['response']['updates_per_arm']
            while engine.update<end:
                stage=(['mean_response','joint_stochastic','low_learning_rate'][engine.update] if engineering else response_stage(engine.update,config))
                changed=stage!=engine.stage
                if changed:engine.set_stage(stage)
                u=engine.update+1
                audit=u<=5 or u%50==0 or changed
                try:
                    metrics=engine.step(audit=audit,parameter_audit=changed or u%4000==0,
                                        verify_replay=u<=3 or changed)
                except BaseException:
                    atomic_torch_save(engine.last_context,directory/'failure_batch.pt')
                    engine.checkpoint(directory/'failure_state.pt')
                    raise
                _append(directory/'training.jsonl',metrics)
                if audit or u%10==0:write_json(out/'status.json',dict(status='training_response',pid=os.getpid(),**metrics))
                if audit:print(json.dumps({k:metrics[k] for k in ['arm','stage','update','weighted_total','update_seconds']}),flush=True)
                if u%1000==0:engine.checkpoint(directory/'latest.pt')
                if u in (3000,15000,end):engine.checkpoint(directory/f'stage_{u:05d}.pt')
            if state_digest(engine.model.electronics.state_dict())!=noise_state['electronics_state_sha256']:
                raise RuntimeError('Validated electronics changed during a response arm')
            write_json(complete,dict(status='engineering_discarded' if engineering else 'fixed_endpoint_reached',
                updates=engine.update,coverage=engine.last_metrics['coverage'],electronics_state_sha256=engine.noise_state_hash,
                noise_gate_sha256=gate_sha,common_initialization_sha256=state_digest(shared),
                population_weights='uniform_ones',trained_old_model_used=False))
        del engine
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
        if not engineering:
            from .evaluate_repair import evaluate_repair_checkpoint
            evaluate_repair_checkpoint(directory/f"stage_{config['response']['updates_per_arm']:05d}.pt",
                container,identities,directory/'evaluation',device=device)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--container',required=True);parser.add_argument('--identities',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=['noise','responses','all'],default='all')
    parser.add_argument('--engineering',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cuda');args=parser.parse_args(argv)
    if args.engineering and args.phase!='noise':
        raise ValueError('CLI engineering is noise-only; response engineering requires its own isolated output directory')
    config=json.loads(Path(args.config).read_text(encoding='utf8'));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(2)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:config['cpu_threads']])
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_limit_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
    try:
        if sha256(args.container)!=config['train_container_sha256'] or sha256(args.identities)!=config['identities_sha256']:
            raise ValueError('Fixed input hash mismatch')
        data=load_training(args.container,args.identities)
        if args.phase in ('noise','all'):
            state=fit_noise(data,config,out/'noise',args.device,engineering=args.engineering,resume=args.resume)
            if args.engineering:
                write_json(out/'status.json',dict(status='noise_engineering_complete',updates=3,scientific_updates=0,utc=utc()));return
        else:
            state=validate_noise_endpoint(torch.load(out/'noise'/'final.pt',map_location='cpu',weights_only=False),config)
        gate_path=out/'noise_validation'/'NOISE_GATE.json'
        if not gate_path.exists():
            write_json(out/'status.json',dict(status='evaluating_frozen_noise',pid=os.getpid(),utc=utc()))
            validation=load_validation(args.container,args.identities)
            model=ElectronicsNoiseV2(state['initialization']).to(args.device);model.load_state_dict(state['model'])
            gate=evaluate_noise(model,data['real'][:,:100],validation['real'][:,:100],data['real_groups'],
                                validation['real_groups'],config['noise'],gate_path.parent,args.device)
            gate['electronics_state_sha256']=state['electronics_state_sha256'];gate['noise_checkpoint_sha256']=sha256(out/'noise'/'final.pt')
            gate['config_sha256']=sha256(args.config);write_json(gate_path,gate)
        else:
            gate=json.loads(gate_path.read_text(encoding='utf8'))
            if gate['noise_checkpoint_sha256']!=sha256(out/'noise'/'final.pt') or gate['config_sha256']!=sha256(args.config):
                raise ValueError('Existing gate binding mismatch')
        if not gate['passed']:
            write_json(out/'status.json',dict(status='stopped_at_noise_gate',noise_updates=state['update'],response_updates=0,
                gate_status=gate['status'],failed_checks=gate['failed_checks'],utc=utc(),pid=os.getpid()))
            return
        if args.phase in ('responses','all'):
            train_responses(data,config,state,gate_path,out,args.device,engineering=args.engineering,resume=args.resume,
                            container=args.container,identities=args.identities)
            write_json(out/'status.json',dict(status='two_response_arms_complete',noise_updates=state['update'],response_updates=38000,
                utc=utc(),pid=os.getpid(),scientific_status='requires_complete_endpoint_review'))
        else:
            write_json(out/'status.json',dict(status='noise_gate_passed',noise_updates=state['update'],response_updates=0,utc=utc(),pid=os.getpid()))
    except BaseException as error:
        record=dict(status='failed',error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),utc=utc(),pid=os.getpid())
        write_json(out/'failure.json',record);write_json(out/'status.json',record);raise


if __name__=='__main__':main()
