"""Frozen006 conditional hypotheses; no fit, file IO or internal randomness.

The first100 values are the original005 effective continuous latent, not
identified hardware analog noise. A valid conditional covariance does not
validate unknown380-point behavior. These are fixed sensitivity hypotheses.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping,Sequence

import torch
from torch import Tensor,nn

from .electronics_v3 import ObservedBaselineNoiseV3
from .model import charge,digitize,ADC_MIN,ADC_MAX

PREFIX=100
LENGTH=480
TAIL=LENGTH-PREFIX
HYPOTHESES006=(
    dict(name='fast',short_weight=1.,short_length=4.,long_weight=0.,long_length=None,long_type='none'),
    dict(name='slow',short_weight=.75,short_length=4.,long_weight=.25,long_length=80.,long_type='exponential'),
    dict(name='common_level',short_weight=.75,short_length=4.,long_weight=.25,long_length=None,long_type='constant'))


def _require(condition,message):
    if not bool(condition):raise ValueError(message)


def _finite_tensor(value,shape,*,name,device=None,float64=False):
    _require(isinstance(value,Tensor) and tuple(value.shape)==tuple(shape),name+' has invalid shape')
    _require(value.is_floating_point() and (not float64 or value.dtype==torch.float64),name+' must have the prescribed floating dtype')
    _require(device is None or value.device==device,name+' must share the model device')
    _require(torch.isfinite(value).all(),name+' is nonfinite; no rows are removed')


class ConditionalExtension006(nn.Module):
    """Fixed full480 conditional law around a frozen strict100 V3 model.

    No new parameters, scaler fit or RNG. ``extend`` invokes005 original
    forward. ``extend_prefix`` reuses the caller's same005 prefix/S, permitting
    all hypotheses to share one005 forward. All outputs/calculations float64.
    Return shapes: noise_pre Bx1x480,prefix_pre Bx1x100,scale Bx1x1.
    """
    def __init__(self,model005:ObservedBaselineNoiseV3,hypotheses:Sequence[Mapping],ridge_relative:float=1e-6):
        super().__init__()
        _require(isinstance(model005,ObservedBaselineNoiseV3),'Requires the original strict100 ObservedBaselineNoiseV3')
        _require(ridge_relative==1e-6,'006 fixes full-kernel relative ridge at1e-6')
        specs=copy.deepcopy(list(hypotheses))
        _require(specs==list(HYPOTHESES006),'006 requires exactly the three registered hypothesis specifications/order')
        _require(all(p.dtype==torch.float64 for p in model005.parameters()),'005 parameters must remain float64')
        self.model005=model005
        self.hypotheses=tuple(spec['name'] for spec in specs)
        self._model_snapshot={key:value.detach().cpu().clone() for key,value in model005.state_dict().items()}
        with torch.no_grad():
            mu=model005.mean_vector().detach().cpu().clone()
            lower=model005.cholesky_factor().detach().cpu().clone()
            eta=model005.log_scale_std.detach().cpu().clone()
        _finite_tensor(mu,(100,),name='frozen005 mean',float64=True)
        _finite_tensor(lower,(100,100),name='frozen005 Cholesky',float64=True)
        _finite_tensor(eta,(),name='frozen005 eta',float64=True)
        C=lower@lower.T;v=C.diagonal().mean();mu0=mu.mean()
        _require(v>0,'Frozen005 covariance needs positive mean diagonal variance')
        self.register_buffer('mean_prefix',mu)
        self.register_buffer('covariance_prefix',C)
        self.register_buffer('eta',eta)
        self.register_buffer('kernel_variance',v)
        self.register_buffer('mean_reference',mu0)
        t=torch.arange(LENGTH,dtype=torch.float64)
        distance=abs(t[:,None]-t[None,:])
        identity=torch.eye(LENGTH,dtype=torch.float64)
        for spec in specs:
            kernel=spec['short_weight']*torch.exp(-distance/spec['short_length'])
            if spec['long_type']=='exponential':kernel=kernel+spec['long_weight']*torch.exp(-distance/spec['long_length'])
            elif spec['long_type']=='constant':kernel=kernel+spec['long_weight']
            kernel=v*(kernel+ridge_relative*identity)
            KBB=kernel[:PREFIX,:PREFIX];KBT=kernel[:PREFIX,PREFIX:]
            # Solve both reference blocks in float64; no inverse, adaptive ridge
            # or eigenvalue clipping. Full-kernel ridge precedes block extraction.
            KBB_lower=torch.linalg.cholesky(KBB)
            A=torch.cholesky_solve(KBT,KBB_lower).T
            R=kernel[PREFIX:,PREFIX:]-A@KBT
            R=(R+R.T)*.5
            B=torch.linalg.cholesky(R)
            tail_mean=mu0+A@(mu-mu0)
            cross=C@A.T
            covariance=torch.cat((torch.cat((C,cross),1),torch.cat((cross.T,A@C@A.T+R),1)),0)
            mean=torch.cat((mu,tail_mean))
            _require(torch.isfinite(covariance).all() and torch.isfinite(mean).all(),'Nonfinite conditional hypothesis')
            for key,value in dict(A=A,R=R,cholesky_R=B,mean=mean,covariance=covariance,kernel_reference=kernel).items():
                self.register_buffer(spec['name']+'_'+key,value)
        self.to(device=model005.voltage_scale.device)
        self._buffer_versions={name:value._version for name,value in self.named_buffers() if not name.startswith('model005.')}
        self.assert_frozen()

    def assert_frozen(self):
        """Fail on005 state mutation or ordinary mutation of cached fixed laws."""
        state=self.model005.state_dict()
        _require(set(state)==set(self._model_snapshot),'Frozen005 state schema changed')
        for name,value in state.items():
            saved=self._model_snapshot[name]
            _require(value.dtype==saved.dtype and value.shape==saved.shape and torch.equal(value.detach().cpu(),saved),
                     'Frozen005 state changed: '+name)
        _require(self.mean_prefix.device==self.model005.voltage_scale.device,'Move the whole extension/model to the same device')
        for name,value in self.named_buffers():
            if name.startswith('model005.'):continue
            _require(value.dtype==torch.float64 and value._version==self._buffer_versions[name],
                     'Frozen006 conditional buffer changed: '+name)

    def _law(self,hypothesis,key):
        _require(isinstance(hypothesis,str) and hypothesis in self.hypotheses,'Unknown006 hypothesis')
        return getattr(self,hypothesis+'_'+key)

    @torch.no_grad()
    def extend(self,noise005:Mapping[str,Tensor],extension_gaussian:Tensor,hypothesis:str)->dict[str,Tensor]:
        self.assert_frozen()
        _require(isinstance(noise005,Mapping) and set(noise005)=={'baseline_gaussian','baseline_scale'},
                 '005 requires exactly baseline_gaussian and baseline_scale; no sources/IDs')
        prefix=self.model005(noise005,length=100)
        scale=torch.exp(self.eta*noise005['baseline_scale'].to(torch.float64)-self.eta.square())
        return self.extend_prefix(prefix,scale,extension_gaussian,hypothesis)

    @torch.no_grad()
    def extend_prefix(self,prefix_pre:Tensor,scale:Tensor,extension_gaussian:Tensor,hypothesis:str)->dict[str,Tensor]:
        """Caller supplies unchanged original005 prefix and its exact shared S.

        This lower-level helper cannot identify the origin of supplied values;
        the driver must assert its one original005 forward and exact S binding.
        No division by S, prefix transformation or new scale draw occurs here.
        """
        self.assert_frozen()
        _require(isinstance(prefix_pre,Tensor) and prefix_pre.ndim==3 and prefix_pre.shape[0]>0,'prefix_pre must be nonemptyBx1x100')
        batch=prefix_pre.shape[0];device=self.mean_prefix.device
        _finite_tensor(prefix_pre,(batch,1,PREFIX),name='prefix_pre',device=device,float64=True)
        _finite_tensor(scale,(batch,1,1),name='shared005 scale',device=device,float64=True)
        _require((scale>0).all(),'Shared005 scale must be strictly positive')
        _finite_tensor(extension_gaussian,(batch,1,TAIL),name='independent extension_gaussian',device=device)
        A=self._law(hypothesis,'A');B=self._law(hypothesis,'cholesky_R')
        tail_mean=self._law(hypothesis,'mean')[PREFIX:]
        tail=tail_mean[None,None,:]+(prefix_pre-self.mean_prefix[None,None,:])@A.T
        tail=tail+scale*(extension_gaussian.to(torch.float64)@B.T)
        result=torch.cat((prefix_pre,tail),-1)
        _require(torch.isfinite(result).all(),'Nonfinite006 output; no row deletion')
        self.assert_frozen()
        return dict(noise_pre=result,prefix_pre=prefix_pre,scale=scale)

    @torch.no_grad()
    def theoretical(self,hypothesis:str)->dict[str,Tensor]:
        """Exact unquantized scale-mixture moments; no480 observation claim."""
        self.assert_frozen()
        result={key:self._law(hypothesis,key).clone() for key in
                ('mean','covariance','A','R','cholesky_R','kernel_reference')}
        covariance=result['covariance'];mean=result['mean']
        q=mean.new_zeros(LENGTH);q[:100]=-3.;q[100:400]=1.
        tail=q.new_zeros(LENGTH);tail[:100]=-2.5;tail[150:400]=1.
        seam=q.new_zeros(LENGTH);seam[99]=-1.;seam[100]=1.
        starts=torch.arange(90,99,device=mean.device)
        prefix_variances=self.covariance_prefix.diagonal()[starts]+self.covariance_prefix.diagonal()[starts+1]-2*self.covariance_prefix[starts,starts+1]
        seam_var=seam@covariance@seam
        result.update(C100=self.covariance_prefix.clone(),mean_reference=self.mean_reference.clone(),
            kernel_variance=self.kernel_variance.clone(),eta=self.eta.clone(),
            scale_mean=torch.exp(-.5*self.eta.square()),scale_second_moment=torch.ones_like(self.eta),
            scale_fourth_moment=torch.exp(4*self.eta.square()),
            mean_Q=q@mean,variance_Q=q@covariance@q,
            mean_tail_charge=tail@mean,variance_tail_charge=tail@covariance@tail,
            covariance_Q_tail=q@covariance@tail,
            seam_increment_mean=seam@mean,seam_increment_variance=seam_var,
            prefix_increment_start=starts,prefix_increment_variance90_99=prefix_variances,
            seam_variance_over_prefix_increment_mean=seam_var/prefix_variances.mean(),
            first_tail_last10_prefix_correlation=covariance[100,90:100]/torch.sqrt(covariance[100,100]*covariance.diagonal()[90:100]),
            tail_sample_variance=covariance.diagonal()[100:].clone(),
            tail400480_variance_mean=covariance.diagonal()[400:480].mean())
        return result


def signal_components(outputs:Mapping[str,Tensor])->Tensor:
    """Reconstruct old signal directly; ignore its electronics/ADC outputs."""
    _require(isinstance(outputs,Mapping) and {'qsig','unit_shape','signal_noise'}<=set(outputs),'Missing old signal components')
    q=outputs['qsig'];shape=outputs['unit_shape'];local=outputs['signal_noise']
    _require(isinstance(q,Tensor) and q.ndim==1 and len(q)>0,'qsig must be nonemptyB')
    batch=len(q)
    _finite_tensor(q,(batch,),name='qsig',float64=True)
    _require((q>0).all(),'Signal charge must be positive')
    _finite_tensor(shape,(batch,LENGTH),name='unit_shape',device=q.device,float64=True)
    _finite_tensor(local,(batch,LENGTH),name='signal_noise',device=q.device,float64=True)
    signal=q[:,None]*shape+local
    _require(torch.isfinite(signal).all(),'Nonfinite reconstructed signal')
    return signal


def assemble_response(outputs:Mapping[str,Tensor],noise_pre:Tensor)->dict[str,Tensor]:
    """Candidate=dataflow signal+new noise, followed by exactly one ADC call.

    An unused legacy waveform may exist in outputs; it never enters this chain.
    No noise/Q projection, baseline correction, clipping beyond canonical ADC,
    or second quantization of candidate values is applied.
    """
    signal=signal_components(outputs)
    if isinstance(noise_pre,Tensor) and noise_pre.ndim==3 and noise_pre.shape[1]==1:noise_pre=noise_pre[:,0]
    _finite_tensor(noise_pre,signal.shape,name='noise_pre',device=signal.device,float64=True)
    pre=signal+noise_pre
    _require(torch.isfinite(pre).all(),'Nonfinite assembled006 candidate')
    waveform=digitize(pre,ste=False)
    qsignal=charge(signal);qnoise=charge(noise_pre);qpre=charge(pre)
    saturated=(pre<ADC_MIN)|(pre>ADC_MAX)
    return dict(signal_pre=signal,noise_pre=noise_pre,pre_adc=pre,waveform=waveform,
        signal_charge=qsignal,noise_charge=qnoise,pre_adc_charge=qpre,
        charge_identity_residual=qpre-qsignal-qnoise,adc_charge_change=charge(waveform)-qpre,
        saturation_mask=saturated,saturation_fraction=saturated.to(torch.float64).mean(),
        saturation_fraction_per_event=saturated.to(torch.float64).mean(1),
        signal_baseline_absolute_max=signal[:,:100].abs().max(),
        signal_baseline_max_per_event=signal[:,:100].abs().amax(1))
