"""One shared signal response under three fixed effective-noise hypotheses.

Only this module is new. Historical signal/loss/noise/006 sources stay intact.
There is no hypothesis, identity or population metadata input to the signal.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import FullCoverageSampler
from .electronics_v3 import ObservedBaselineNoiseV3
from .extension006 import ConditionalExtension006, HYPOTHESES006, signal_components
from .losses import WaveformDiscriminator, d_hinge, g_hinge
from .losses_v2 import N01LossV2
from .model import N01Response, ADC_MIN, ADC_MAX, charge, digitize
from .train import atomic_torch_save, gradient_summary, seed_all, utc

HYPOTHESES=tuple(x['name'] for x in HYPOTHESES006)
ARMS=('deterministic','full')
WEIGHTS=dict(wave=1.,QPSD=2.,noise=1.,adv=.1,regular=.01,trend=.1)


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tuple(v.shape)).encode())
        digest.update(str(v.dtype).encode());digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def cpu_state(state):
    return {key:value.detach().cpu().clone() for key,value in state.items()}


def loss_tensor_statistics(statistics):
    """Restore frozen JSON numbers exactly into the historical tensor API.

    Float64 keeps every stored binary32 or binary64 value; V2 already performs
    standardization in float64 and D converts its buffers to its float32 input.
    No statistic is fitted, rounded or changed here.
    """
    result=copy.deepcopy(statistics)
    for key in ('q_reference','q_floor','peak_floor','charge_bin_boundaries'):
        result[key]=torch.as_tensor(result[key],dtype=torch.float64).clone()
    for record in list(result['views'].values())+[result['qpsd'],result['noise']]:
        for key in ('center','scale'):record[key]=torch.as_tensor(record[key],dtype=torch.float64).clone()
    return result


class ZeroElectronics(nn.Module):
    """No learned/frozen old electronics is present in the signal model."""
    def forward(self,noise):
        return torch.zeros((len(noise['phys']),1,480),dtype=torch.float64,device=noise['phys'].device)


def saturation_penalty(pre):
    excess=(F.relu(pre-ADC_MAX)+F.relu(ADC_MIN-pre))/(ADC_MAX-ADC_MIN)
    return excess.square().mean()


class SignalResponse007(N01Response):
    def __init__(self,stats,variant):
        super().__init__(stats,variant=variant)
        self.electronics=ZeroElectronics()
        self.set_stage('mean_response')

    def signal_forward(self,source,noise):
        out=super().forward(source,noise=noise,adc_ste=False)
        signal=signal_components(out)
        # The historical forward contains one signal-only ADC diagnostic. Its
        # waveform is unused, and its saturation regularizer must be replaced.
        out['base_regular']=out['regular']-saturation_penalty(out['pre_adc'])
        out['signal_pre']=signal
        return out


def common_initial_state(statistics,seed):
    seed_all(seed)
    return cpu_state(SignalResponse007(statistics,'full').state_dict())


def make_signal(statistics,variant,common,seed,device):
    seed_all(seed)
    model=SignalResponse007(statistics,variant)
    state=model.state_dict()
    for name in state:
        if name not in common or state[name].shape!=common[name].shape or state[name].dtype!=common[name].dtype:
            raise ValueError('Common initialization mismatch: '+name)
        state[name]=common[name].clone()
    model.load_state_dict(state,strict=True)
    for name,value in model.state_dict().items():
        if not torch.equal(value,common[name]):raise RuntimeError('Common initializer not copied: '+name)
    return model.to(device)


def make_extension(checkpoint,device):
    noise=ObservedBaselineNoiseV3(checkpoint['initialization'])
    noise.load_state_dict(checkpoint['model'],strict=True)
    noise.requires_grad_(False);noise.eval();noise.to(device)
    result=ConditionalExtension006(noise,HYPOTHESES006)
    result.requires_grad_(False);result.eval()
    return result


class PairedStreams:
    """Explicit CPU float32 streams; same starts and draw shapes in both arms."""
    def __init__(self,response_seed,noise_seed,extension_seed):
        self.generators={k:torch.Generator(device='cpu').manual_seed(s) for k,s in
            [('response',response_seed),('noise005',noise_seed),('extension',extension_seed)]}

    def draw(self,count,device):
        rg=self.generators['response'];ng=self.generators['noise005'];eg=self.generators['extension']
        response={name:torch.zeros((count,)+shape,dtype=torch.float32,device=device)
                  for name,shape in N01Response.noise_shapes.items()}
        for name in ('phys','local_white'):
            response[name]=torch.randn((count,)+N01Response.noise_shapes[name],generator=rg,dtype=torch.float32).to(device)
        noise={name:torch.randn((count,)+shape,generator=ng,dtype=torch.float32).to(device)
               for name,shape in ObservedBaselineNoiseV3.noise_shapes.items()}
        epsilon=torch.randn((count,1,380),generator=eg,dtype=torch.float32).to(device)
        return response,noise,epsilon

    def state_dict(self):return {k:g.get_state().clone() for k,g in self.generators.items()}
    def load_state_dict(self,state):
        for name,generator in self.generators.items():generator.set_state(state[name].cpu())


@torch.no_grad()
def fixed_noises(extension,noise_latents,epsilon):
    extension.assert_frozen()
    prefix=extension.model005(noise_latents,length=100)
    scale=torch.exp(extension.eta*noise_latents['baseline_scale'].double()-extension.eta.square())
    noises={h:extension.extend_prefix(prefix,scale,epsilon,h)['noise_pre'][:,0] for h in HYPOTHESES}
    for values in noises.values():
        if not torch.equal(values[:,:100],prefix[:,0]):raise RuntimeError('005 prefix changed')
    return noises


def candidates(signal,noises,ste=True):
    return {h:digitize(signal+noises[h],ste=ste) for h in HYPOTHESES}


def replay_backward(model,batches,gradients,*,expected=None,regular_weight=.01,trend_weight=.1):
    """Exact VJP of separately reduced hypothesis losses and one base penalty."""
    start=0
    for source,response,noises in batches:
        output=model.signal_forward(source,response)
        signal=output['signal_pre'];count=len(source)
        waves=candidates(signal,noises,ste=True)
        if expected is not None:
            for h in HYPOTHESES:
                if not torch.equal(waves[h].detach(),expected[h][start:start+count]):
                    raise RuntimeError('Same-parameter/same-latent replay changed: '+h)
        regular=output['base_regular']+sum(saturation_penalty(signal+noises[h]) for h in HYPOTHESES)/3
        penalty=(regular_weight*regular+trend_weight*output['trend'])/len(batches)
        torch.autograd.backward(tuple(waves[h] for h in HYPOTHESES)+(penalty,),
            tuple(gradients[h][start:start+count] for h in HYPOTHESES)+(None,))
        start+=count
    if any(start!=len(value) for value in gradients.values()):raise ValueError('VJP row mismatch')


def response_stage(update,cfg):
    a=cfg['response']['mean_response_updates'];b=a+cfg['response']['joint_stochastic_updates']
    return 'mean_response' if update<a else 'joint_stochastic' if update<b else 'low_learning_rate'


class Response007Engine:
    def __init__(self,data,statistics,config,noise_checkpoint,common,*,arm,device='cuda:0'):
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.arm,self.device,self.config=arm,torch.device(device),config
        self.data,self.statistics=data,statistics
        seeds=config['seeds']
        self.model=make_signal(statistics['model'],arm,common,seeds['initialization'],self.device)
        self.initial_state_digest=state_digest(self.model.state_dict())
        self.common_state_digest=state_digest(common)
        self.extension=make_extension(noise_checkpoint,self.device)
        self.noise_state_digest=state_digest(self.extension.model005.state_dict())
        loss_stats=loss_tensor_statistics(statistics['loss'])
        self.loss=N01LossV2(loss_stats,seed=seeds['train_projection'],
            evaluation_seed=seeds['evaluation_projection']).to(self.device)
        self.discriminators=nn.ModuleDict()
        for h in HYPOTHESES:
            seed_all(seeds['discriminator'])
            self.discriminators[h]=WaveformDiscriminator(loss_stats).to(self.device)
        hashes=[state_digest(d.state_dict()) for d in self.discriminators.values()]
        if len(set(hashes))!=1:raise RuntimeError('D initializations differ')
        self.initial_D_digest=hashes[0]
        self.g_optimizer=torch.optim.AdamW(self.model.parameters(),lr=config['loss']['G_lr'],weight_decay=config['loss']['G_weight_decay'])
        self.d_optimizers={h:torch.optim.AdamW(d.parameters(),lr=config['loss']['D_lr'],weight_decay=0.) for h,d in self.discriminators.items()}
        self.source=torch.as_tensor(data['source'],dtype=torch.float32,device=self.device)
        self.real=torch.as_tensor(data['real'],dtype=torch.float64,device=self.device)
        self.source_sampler=FullCoverageSampler(len(self.source),seeds['source_sampler'])
        self.real_sampler=FullCoverageSampler(len(self.real),seeds['real_sampler'])
        self.streams=PairedStreams(seeds['train_response'],seeds['train_noise005'],seeds['train_extension'])
        self.update=0;self.d_updates={h:0 for h in HYPOTHESES};self.stage=None
        self.last_context={};self.last_metrics={}

    def set_stage(self,stage):
        if stage=='baseline':raise ValueError('007 never refits the baseline')
        self.stage=stage;self.model.set_stage(stage);self.model.train()
        low=stage=='low_learning_rate';cfg=self.config['loss']
        for group in self.g_optimizer.param_groups:group['lr']=cfg['G_low_lr'] if low else cfg['G_lr']
        for optimizer in self.d_optimizers.values():
            for group in optimizer.param_groups:group['lr']=cfg['D_low_lr'] if low else cfg['D_lr']
        self.extension.assert_frozen()

    def step(self,*,audit=False,verify_replay=False):
        import time
        started=time.monotonic()
        if self.stage is None:raise RuntimeError('Stage not set')
        joint=self.stage!='mean_response';cfg=self.config
        groups=cfg['distribution_microbatches'];parents=cfg['source_parents_per_microbatch'];repeats=cfg['responses_per_parent']
        context=self.loss.create_projection_context(self.update+1)
        self.last_context=dict(arm=self.arm,stage=self.stage,attempted_G_update=self.update+1,
            streams_before=self.streams.state_dict(),source_sampler_before=self.source_sampler.state_dict(),
            real_sampler_before=self.real_sampler.state_dict(),projection_context=context.state_dict(),batches=[])
        batches=[];snapshots={h:[] for h in HYPOTHESES};real_batches=[];all_si=[];all_ri=[]
        engineering={h:dict(nonpositive_Q=0,guarded_PSD=0,saturation_fraction=0.,charge_identity_max=0.) for h in HYPOTHESES}
        prefix_max=0.;regulars=[];trends=[]
        self.g_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for _ in range(groups):
                si=np.repeat(self.source_sampler.draw(parents),repeats)
                ri=self.real_sampler.draw(cfg['real_rows_per_microbatch'])
                all_si.append(si);all_ri.append(ri)
                response,nz,epsilon=self.streams.draw(len(si),self.device)
                self.last_context['batches'].append(dict(source_rows=si,real_rows=ri,
                    source_parent_ids=self.data['source_ids'][si],real_original_ids=self.data['real_ids'][ri],
                    response=cpu_state(response),noise005=cpu_state(nz),extension=epsilon.detach().cpu()))
                source=self.source[torch.as_tensor(si,device=self.device)]
                noises=fixed_noises(self.extension,nz,epsilon)
                output=self.model.signal_forward(source,response);signal=output['signal_pre']
                prefix_max=max(prefix_max,float(signal[:,:100].abs().max()))
                if prefix_max>cfg['engineering']['signal_prefix_max_V']:
                    raise RuntimeError('Signal prefix leakage exceeds registered one ADC step')
                waves=candidates(signal,noises,ste=False)
                for h in HYPOTHESES:
                    pre=signal+noises[h];q=charge(waves[h])
                    engineering[h]['nonpositive_Q']+=int((q<=0).sum())
                    engineering[h]['guarded_PSD']+=int((q.abs()<self.loss.q_floor).sum())
                    engineering[h]['saturation_fraction']+=float(((pre<ADC_MIN)|(pre>ADC_MAX)).double().mean())/groups
                    residual=float((charge(pre)-charge(signal)-charge(noises[h])).abs().max())
                    engineering[h]['charge_identity_max']=max(engineering[h]['charge_identity_max'],residual)
                    if not bool(torch.isfinite(waves[h]).all()) or bool((q<=0).any()) or bool((q.abs()<self.loss.q_floor).any()):
                        raise RuntimeError('Invalid candidate Q/PSD or nonfinite output: '+h)
                    snapshots[h].append(waves[h])
                regulars.append(output['base_regular']+sum(saturation_penalty(signal+noises[h]) for h in HYPOTHESES)/3)
                trends.append(output['trend'])
                batches.append((source,response,noises));real_batches.append(self.real[torch.as_tensor(ri,device=self.device)])
        real=torch.cat(real_batches);fake={h:torch.cat(values) for h,values in snapshots.items()}
        dmetrics={};gradients={};metrics={};audit_gradients={}
        for h in HYPOTHESES:
            d=self.discriminators[h];optimizer=self.d_optimizers[h]
            if joint:
                d.requires_grad_(True);d.train();optimizer.zero_grad(set_to_none=True)
                value=d_hinge(d(fake[h].float()),d(real.float()))
                if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite D loss')
                value.backward();norm=torch.nn.utils.clip_grad_norm_(d.parameters(),cfg['loss']['clip_norm'],error_if_nonfinite=True)
                optimizer.step();self.d_updates[h]+=1
                if any(not bool(torch.isfinite(p).all()) for p in d.parameters()):raise FloatingPointError('Nonfinite D parameter')
                dmetrics[h]=dict(loss=float(value.detach()),preclip_gradient_norm=float(norm))
            d.requires_grad_(False);d.eval()
            endpoint=fake[h].detach().requires_grad_(True)
            components=self.loss.components(dict(waveform=endpoint),real,stage=self.stage,projection_context=context)
            diagnostics=copy.deepcopy(self.loss.last_diagnostics)
            wave_parts={k:float(v) for k,v in self.loss.last_wave_components.items()}
            components['adv']=g_hinge(d(endpoint.float())) if joint else endpoint.sum()*0
            keys=('wave','QPSD','noise','adv')
            if audit:
                audit_gradients[h]=gradient_summary({k:torch.autograd.grad(components[k]*WEIGHTS[k]/3,endpoint,retain_graph=True)[0].detach() for k in keys})
            objective=sum(components[k]*WEIGHTS[k] for k in keys)/3
            if not bool(torch.isfinite(objective)):raise FloatingPointError('Nonfinite G loss')
            gradients[h]=torch.autograd.grad(objective,endpoint)[0].detach()
            if not bool(torch.isfinite(gradients[h]).all()):raise FloatingPointError('Nonfinite endpoint gradient')
            metrics[h]=dict(components={k:float(components[k].detach()) for k in keys},wave_subcomponents=wave_parts,
                conditional_diagnostics=diagnostics,engineering=engineering[h])
        replay_backward(self.model,batches,gradients,expected=fake if verify_replay else None)
        gnorm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),cfg['loss']['clip_norm'],error_if_nonfinite=True)
        branch_gradients={}
        if audit:
            for name,p in self.model.named_parameters():
                if p.grad is not None:
                    key=name.split('.')[0];branch_gradients[key]=branch_gradients.get(key,0.)+float(p.grad.double().square().sum())
            branch_gradients={k:v**.5 for k,v in branch_gradients.items()}
        self.g_optimizer.step();self.update+=1
        for name,p in self.model.named_parameters():
            if not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite G parameter: '+name)
        context.validate();self.extension.assert_frozen()
        regular=float(torch.stack(regulars).mean());trend=float(torch.stack(trends).mean())
        total=sum(sum(WEIGHTS[k]*row['components'][k] for k in ('wave','QPSD','noise','adv')) for row in metrics.values())/3+WEIGHTS['regular']*regular+WEIGHTS['trend']*trend
        result=dict(utc=utc(),arm=self.arm,stage=self.stage,G_updates=self.update,D_updates=dict(self.d_updates),
            weighted_total=total,regular_once=regular,trend_once=trend,hypotheses=metrics,discriminators=dmetrics,
            G_preclip_gradient_norm=float(gnorm),signal_prefix_absolute_max_V=prefix_max,
            projection_context=context.metadata(),same_version_replay_verified=verify_replay,
            coverage=dict(source_unique=int((self.source_sampler.visits>0).sum()),source_total=len(self.source),
                real_unique=int((self.real_sampler.visits>0).sum()),real_total=len(self.real),
                source_epochs=self.source_sampler.completed_epochs,real_epochs=self.real_sampler.completed_epochs),
            noise_updates=0,population_updates=0,update_seconds=time.monotonic()-started,
            gpu_allocated_bytes=torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else 0,
            gpu_reserved_bytes=torch.cuda.memory_reserved(self.device) if self.device.type=='cuda' else 0)
        if audit:result.update(endpoint_gradients=audit_gradients,gradient_scope='per-hypothesis weighted waveform endpoints; base regular excluded',
            G_branch_postclip_gradient_norms=branch_gradients)
        if result['gpu_allocated_bytes']>cfg['gpu_memory_limit_GiB']*1024**3 or result['gpu_reserved_bytes']>cfg['gpu_memory_limit_GiB']*1024**3:
            raise RuntimeError('Registered GPU memory limit exceeded')
        self.last_metrics=result
        return result

    def checkpoint(self,path):
        self.extension.assert_frozen()
        if state_digest(self.extension.model005.state_dict())!=self.noise_state_digest:raise RuntimeError('005 changed')
        state=dict(schema='response007/v1',arm=self.arm,config=self.config,model=cpu_state(self.model.state_dict()),
            discriminators=cpu_state(self.discriminators.state_dict()),g_optimizer=self.g_optimizer.state_dict(),
            d_optimizers={h:o.state_dict() for h,o in self.d_optimizers.items()},
            source_sampler=self.source_sampler.state_dict(),real_sampler=self.real_sampler.state_dict(),streams=self.streams.state_dict(),
            G_updates=self.update,D_updates=self.d_updates,stage=self.stage,statistics=self.statistics,last_metrics=self.last_metrics,
            initial_state_sha256=self.initial_state_digest,common_initial_state_sha256=self.common_state_digest,
            initial_D_sha256=self.initial_D_digest,noise005_state_sha256=self.noise_state_digest,saved_UTC=utc(),
            population_learning=False,noise_updates=0)
        atomic_torch_save(state,path)

    def restore(self,path):
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state['config']!=self.config or state['arm']!=self.arm:raise ValueError('Resume identity/config mismatch')
        if state['common_initial_state_sha256']!=self.common_state_digest or state['noise005_state_sha256']!=self.noise_state_digest:
            raise ValueError('Resume frozen reference mismatch')
        self.model.load_state_dict(state['model']);self.discriminators.load_state_dict(state['discriminators'])
        self.g_optimizer.load_state_dict(state['g_optimizer'])
        for h,o in self.d_optimizers.items():o.load_state_dict(state['d_optimizers'][h])
        self.source_sampler.load_state_dict(state['source_sampler']);self.real_sampler.load_state_dict(state['real_sampler'])
        self.streams.load_state_dict(state['streams']);self.update=state['G_updates'];self.d_updates=state['D_updates']
        self.last_metrics=state['last_metrics'];self.set_stage(state['stage'])
