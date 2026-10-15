"""Paired C1 continuation or a six-parameter deterministic mean adapter.

The original A and 005 are never reopened. Shape voltage is still computed
from the original A, before the mean adapter. No RNG or observation lives here.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

from .model008 import canonical_charge008
from .model010 import tensor_state_digest010
from .model014 import FrozenDeterministicVoltageShape014


class _ExactZeroMeanAddition015(torch.autograd.Function):
    """Keep zero-adapter IEEE bytes while retaining its correct unit derivative."""
    @staticmethod
    def forward(ctx, original, delta):
        return original.clone() if not bool(delta.ne(0).any()) else original+delta

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class FrozenC1MeanResponse015(FrozenDeterministicVoltageShape014):
    """M0 trains the original 1288 shape parameters; M1 only six mean controls."""
    def __init__(self, baseline, initial_head, *, arm, seed, q_reference, q_floor):
        if arm not in ('M0', 'M1'):
            raise ValueError('015 only permits M0 and M1')
        if not (math.isfinite(float(q_reference)) and float(q_reference)>0
                and math.isfinite(float(q_floor)) and float(q_floor)>0):
            raise ValueError('015 requires frozen positive TRAIN q_reference/q_floor')
        super().__init__(baseline, seed=seed)
        self.basis_head.load_state_dict(initial_head, strict=True)
        self.arm=arm
        ref=self.baseline.mean_head.weight
        self.mean_theta=nn.Parameter(torch.zeros(6,device=ref.device,dtype=torch.float64),requires_grad=arm=='M1')
        t=torch.arange(-6,7,device=ref.device,dtype=torch.float64)
        gaussian=torch.exp(-.5*(t/2.)**2);gaussian=gaussian/gaussian.sum()
        causal=torch.exp(-torch.arange(64,device=ref.device,dtype=torch.float64)/16.)
        self.register_buffer('mean_gaussian_kernel',gaussian)
        self.register_buffer('mean_causal_kernel',causal/causal.sum())
        self.register_buffer('mean_q_reference',torch.tensor(float(q_reference),device=ref.device,dtype=torch.float64))
        self.register_buffer('mean_q_floor',torch.tensor(float(q_floor),device=ref.device,dtype=torch.float64))
        self.basis_head.requires_grad_(arm=='M0')
        self.basis_head.train(arm=='M0')
        self.reference_C1_head_sha256=tensor_state_digest010(self.basis_head)
        self._mean_parameter_id=id(self.mean_theta)
        self._mean_buffer_reference={n:v.detach().clone() for n,v in self.named_buffers() if n.startswith('mean_')}
        self.assert_frozen_unchanged(deep=True)

    def train(self,mode=True):
        if type(mode) is not bool:raise TypeError('015 train mode must be bool')
        self.assert_frozen_unchanged();self.training=mode
        self.basis_head.train(mode and self.arm=='M0')
        self.assert_frozen_unchanged();return self

    def requires_grad_(self,requires_grad=True):
        if type(requires_grad) is not bool:raise TypeError('015 requires_grad flag must be bool')
        self.assert_frozen_unchanged()
        self.basis_head.requires_grad_(requires_grad and self.arm=='M0')
        self.mean_theta.requires_grad_(requires_grad and self.arm=='M1')
        return self

    def trainable_parameter_names(self):
        return ('basis_head.weight','basis_head.bias') if self.arm=='M0' else ('mean_theta',)

    def trainable_parameters(self):
        if self.arm=='M0':return super().trainable_parameters()
        if id(self.mean_theta)!=self._mean_parameter_id:raise RuntimeError('015 mean parameter replaced')
        return (self.mean_theta,)

    def assert_optimizer_membership(self,optimizer):
        self.assert_frozen_unchanged();expected=self.trainable_parameters()
        actual=tuple(v for group in optimizer.param_groups for v in group['params'])
        if len(actual)!=len(expected) or {id(v) for v in actual}!={id(v) for v in expected}:
            raise ValueError('015 optimizer must contain exactly its arm whitelist')
        if not all(v.requires_grad for v in expected):raise ValueError('015 trainable whitelist disabled')

    def assert_frozen_unchanged(self,*,deep=False):
        result=super().assert_frozen_unchanged(deep=deep)
        if not hasattr(self,'mean_theta'):return result  # parent construction only
        if self.arm=='M1':
            if any(p.requires_grad or p.grad is not None for p in self.basis_head.parameters()):
                raise RuntimeError('015 M1 opened its protected C1 shape head')
            if deep and tensor_state_digest010(self.basis_head)!=self.reference_C1_head_sha256:
                raise RuntimeError('015 M1 changed the protected C1 shape head')
        elif self.mean_theta.requires_grad or self.mean_theta.grad is not None or bool(self.mean_theta.ne(0).any()):
            raise RuntimeError('015 M0 changed or opened its disabled mean adapter')
        if deep and hasattr(self,'_mean_buffer_reference'):
            for name,value in self._mean_buffer_reference.items():
                if not torch.equal(dict(self.named_buffers())[name],value):raise RuntimeError('015 fixed mean buffer changed')
        return result

    def mean_adapter(self,source,base_signal,*,enabled=True):
        if source.ndim==3 and source.shape[1]==1:source=source[:,0]
        if source.shape!=base_signal.shape or base_signal.shape[1:]!=(384,):
            raise ValueError('015 mean adapter requires matching original Bx384 source and A')
        if not bool(torch.isfinite(source).all() and torch.isfinite(base_signal).all()):
            raise ValueError('015 nonfinite mean input; no rows discarded')
        a=base_signal.double();rms_a=a.square().mean(1).sqrt()
        if not enabled:
            zero=torch.zeros_like(a)
            return dict(delta_mean_pre_V=zero,mean_adapter_regular=a.sum()*0.,
                mean_RMS_V=rms_a*0.,mean_RMS_relative_A=rms_a*0.,mean_trust_scale=torch.ones_like(rms_a),
                mean_canonical_charge=rms_a*0.,mean_coefficients=torch.zeros(len(a),3,device=a.device,dtype=a.dtype),
                mean_controls=torch.zeros(len(a),2,device=a.device,dtype=a.dtype))
        x=source.double()-source.double()[:,:100].mean(1,keepdim=True)
        q=canonical_charge008(x);sign=torch.where(q<0,-torch.ones_like(q),torch.ones_like(q))
        cq=torch.tanh(torch.log1p(q.abs()/self.mean_q_reference)-math.log(2.))
        cs=torch.tanh(2*(x[:,156:366].sum(1)/(sign*q.abs().clamp_min(self.mean_q_floor))-.5))
        coefficients=.1*torch.tanh(torch.stack((self.mean_theta[0]+self.mean_theta[1]*cq,
            self.mean_theta[2]+self.mean_theta[3]*cs,self.mean_theta[4]+self.mean_theta[5]*cs),1))
        smooth=F.conv1d(F.pad(a[:,None],(6,6)),self.mean_gaussian_kernel[None,None])[:,0]
        tail=F.conv1d(F.pad(a[:,None],(63,0)),self.mean_causal_kernel.flip(0)[None,None])[:,0]
        gate=self.baseline.gate.reshape(1,384);projection=self.baseline.projection.reshape(1,384)
        bases=[]
        for difference in (smooth-a,tail-a):
            raw=difference*gate
            projected=raw-canonical_charge008(raw)[:,None]*projection
            bases.append(projected*(rms_a/projected.square().mean(1).sqrt().clamp_min(1e-12))[:,None])
        raw=coefficients[:,0,None]*a+coefficients[:,1,None]*bases[0]+coefficients[:,2,None]*bases[1]
        raw_rms=torch.linalg.vector_norm(raw,dim=1)/math.sqrt(384.)
        scale=torch.minimum(torch.ones_like(raw_rms),.1*rms_a/raw_rms.clamp_min(1e-12))
        delta=raw*scale[:,None]
        rms=torch.linalg.vector_norm(delta,dim=1)/math.sqrt(384.);relative=rms/rms_a.clamp_min(1e-12)
        if bool((delta[:,:100]!=0).any()) or not bool(torch.isfinite(delta).all()):
            raise ValueError('015 adapter changed prefix100 or became nonfinite')
        if bool((rms>.1*rms_a+1e-14).any()):raise ValueError('015 mean trust domain exceeded')
        return dict(delta_mean_pre_V=delta,mean_adapter_regular=relative.square().mean(),
            mean_RMS_V=rms,mean_RMS_relative_A=relative,mean_trust_scale=scale,
            mean_canonical_charge=canonical_charge008(delta),mean_coefficients=coefficients,
            mean_controls=torch.stack((cq,cs),1),mean_zeroQ_basis_charge=torch.stack([canonical_charge008(b) for b in bases],1))

    def random_forward(self,source_carrier_V,z_shape,*,enabled=True,mean_enabled=None):
        if mean_enabled is None:mean_enabled=self.arm=='M1'
        if type(mean_enabled) is not bool or (mean_enabled and self.arm!='M1'):
            raise ValueError('015 M0 mean adapter cannot be enabled')
        out=super().random_forward(source_carrier_V,z_shape,enabled=enabled)
        adapter=self.mean_adapter(source_carrier_V,out['base_signal_pre_V'],enabled=mean_enabled)
        c1=out['signal_pre'];signal=_ExactZeroMeanAddition015.apply(c1,adapter['delta_mean_pre_V']) if mean_enabled else c1
        original_q=out['qsig'];original_qbar=out['qbar'];expected_q=original_q+adapter['mean_canonical_charge']
        out.update(adapter)
        out.update(signal_pre=signal,pure_signal_pre_V=signal,C1_reference_signal_pre_V=c1,
            deterministic_mean_signal_pre_V=out['base_signal_pre_V']+adapter['delta_mean_pre_V'],
            stochastic_shape_regular=out['regular'],regular=out['regular']+adapter['mean_adapter_regular'],
            canonical_charge=canonical_charge008(signal),canonical_charge_expected_Vns=expected_q,
            canonical_charge_identity_residual=canonical_charge008(signal)-expected_q,
            reference_A_qbar=original_qbar,reference_A_qsig=original_q,Q_mu_Vns=expected_q,
            qbar=expected_q if mean_enabled else original_qbar,qsig=expected_q if mean_enabled else original_q,
            mean_adapter_enabled=mean_enabled,mean_adapter_parameter_count=6,
            mean015=dict(schema='n01/frozen-C1-mean015/v1',arm=self.arm,control_Q_slice=[110,366],
                control_tail_slice=[156,366],basis_canonical_Q_slice=[110,366],prefix_samples=100,
                gaussian_sigma_ns=2.,gaussian_length=13,causal_tau_ns=16.,causal_length=64,
                coefficient_limit=.1,per_event_RMS_trust_fraction=.1,normalization_epsilon_V=1e-12,
                inherited_unit_shape_and_shape_Q='original A reference fields; not the adapted mean shape',
                stochastic_basis_amplitude='original A only; mean adapter and stochastic Q never multiply it',
                C1_reference_head_sha256=self.reference_C1_head_sha256))
        self.assert_frozen_unchanged();return out

    forward=random_forward
