"""Frozen M0 plus a positive, mean-corrected two-parameter charge law.

The original014 voltage shape arithmetic is reused, with the015 M0 head bytes.
No RNG, observation, data or checkpoint access occurs in this module.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .model008 import canonical_charge008
from .model010 import tensor_state_digest010
from .model014 import FrozenDeterministicVoltageShape014

Q_REFERENCE016=3.7103998046755806


def inverse_softplus016(value):
    value=torch.as_tensor(value,dtype=torch.float64)
    if not bool(torch.isfinite(value).all() and (value>0).all()):
        raise ValueError('016 inverse-softplus needs finite strictly positive values')
    return value+torch.log(-torch.expm1(-value))


def charge_coefficients016(theta_a,theta_b,q_reference):
    a=q_reference*F.softplus(theta_a);b=F.softplus(theta_b)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all() and (a>0).all() and (b>0).all()):
        raise ValueError('016 nonpositive or nonfinite charge coefficients; no clipping')
    return a,b


def calibration_statistics016(A,*,adc_step,q_reference=Q_REFERENCE016):
    """Pure actual-A summary; all parents and all384 samples are retained."""
    if A.ndim!=2 or A.shape[1]!=384 or not len(A) or A.requires_grad:
        raise ValueError('016 calibration requires detached actual A Bx384')
    x=A.double();Q=canonical_charge008(x)
    if not bool(torch.isfinite(x).all() and torch.isfinite(Q).all() and (Q>0).all()):
        raise ValueError('016 calibration actual canonical Q must be finite strictly positive for every parent')
    if not bool((x[:,:100]==0).all()):raise ValueError('016 calibration prefix100 changed')
    if not math.isfinite(float(adc_step)) or float(adc_step)<=0 or q_reference!=Q_REFERENCE016:
        raise ValueError('016 fixed ADC step and q_reference required')
    S0=x.square().mean();Sm=(x.square()/Q[:,None]).mean()
    if not bool(torch.isfinite(S0) and torch.isfinite(Sm) and S0>0 and Sm>0):
        raise ValueError('016 nonpositive or nonfinite actual-A RMS statistics')
    a=.5*float(adc_step)**2/Sm;b=.5*float(adc_step)**2/S0
    theta_a=inverse_softplus016(a/q_reference);theta_b=inverse_softplus016(b)
    actual_a,actual_b=charge_coefficients016(theta_a,theta_b,torch.as_tensor(q_reference,device=x.device,dtype=torch.float64))
    cv2=actual_a/Q+actual_b
    return dict(parents=len(x),coordinates=384,zero_prefix100_in_RMS_denominator=True,
        S0_V2=float(S0),Sminus_V2_per_Vns=float(Sm),a_initial_Vns=float(actual_a),b_initial=float(actual_b),
        theta_a_initial=float(theta_a),theta_b_initial=float(theta_b),q_reference_Vns=q_reference,
        actual_A_expected_increment_RMS_V=float((actual_a*Sm+actual_b*S0).sqrt()),target_RMS_V=float(adc_step),
        Q_mu_min_Vns=float(Q.min()),Q_mu_max_Vns=float(Q.max()),all_Q_mu_finite_strictly_positive=True,
        conditional_CV_min=float(cv2.sqrt().min()),conditional_CV_mean=float(cv2.sqrt().mean()),
        conditional_CV_max=float(cv2.sqrt().max()),calibration_uses_no_finite_MC=True)


class FrozenM0ChargeResponse016(FrozenDeterministicVoltageShape014):
    """Exactly theta_a/theta_b are trainable; original A and M0 stay frozen."""
    def __init__(self,baseline,initial_head,*,seed,calibration):
        super().__init__(baseline,seed=seed)
        self.basis_head.load_state_dict(initial_head,strict=True)
        self.basis_head.requires_grad_(False);self.basis_head.eval()
        self.reference_M0_head_sha256=tensor_state_digest010(self.basis_head)
        ref=self.baseline.mean_head.weight
        self.theta_a=nn.Parameter(torch.tensor(calibration['theta_a_initial'],dtype=torch.float64,device=ref.device))
        self.theta_b=nn.Parameter(torch.tensor(calibration['theta_b_initial'],dtype=torch.float64,device=ref.device))
        self.register_buffer('charge_q_reference',torch.tensor(Q_REFERENCE016,dtype=torch.float64,device=ref.device))
        self._charge_parameter_ids=(id(self.theta_a),id(self.theta_b))
        self.assert_frozen_unchanged(deep=True)

    def train(self,mode=True):
        if type(mode) is not bool:raise TypeError('016 train mode must be bool')
        self.assert_frozen_unchanged();self.training=mode;self.basis_head.eval();return self

    def requires_grad_(self,requires_grad=True):
        if type(requires_grad) is not bool:raise TypeError('016 requires_grad flag must be bool')
        self.assert_frozen_unchanged()
        for p in self.trainable_parameters():p.requires_grad_(requires_grad)
        return self

    def trainable_parameter_names(self):return ('theta_a','theta_b')

    def trainable_parameters(self):
        p=(self.theta_a,self.theta_b)
        if tuple(id(v) for v in p)!=self._charge_parameter_ids:raise RuntimeError('016 charge parameter replaced')
        return p

    def assert_optimizer_membership(self,optimizer):
        self.assert_frozen_unchanged()
        wanted=self.trainable_parameters();actual=tuple(p for g in optimizer.param_groups for p in g['params'])
        if len(actual)!=2 or {id(p) for p in actual}!={id(p) for p in wanted} or not all(p.requires_grad for p in wanted):
            raise ValueError('016 optimizer must contain exactly theta_a and theta_b')
        for group in optimizer.param_groups:
            if group['lr']!=.01 or group['weight_decay']!=0:raise ValueError('016 fixed AdamW lr=.01 wd=0 required')

    def assert_frozen_unchanged(self,*,deep=False):
        result=super().assert_frozen_unchanged(deep=deep)
        if not hasattr(self,'theta_a'):return result
        if self.basis_head.training or any(p.requires_grad or p.grad is not None for p in self.basis_head.parameters()):
            raise RuntimeError('016 opened protected M0 shape head')
        if deep and tensor_state_digest010(self.basis_head)!=self.reference_M0_head_sha256:
            raise RuntimeError('016 changed protected M0 shape bytes')
        if float(self.charge_q_reference)!=Q_REFERENCE016:raise RuntimeError('016 fixed charge reference changed')
        return result

    def apply_charge(self,out,z_charge,*,enabled=True):
        if type(enabled) is not bool:raise TypeError('016 charge switch must be bool')
        original=out['signal_pre'];A=out['base_signal_pre_V'].double()
        # The off branch returns the original tensor object, including signed zeros.
        if not enabled:
            out.update(M0_reference_signal_pre_V=original,charge_enabled=False)
            return out
        if not isinstance(z_charge,torch.Tensor) or z_charge.shape!=(len(A),) or z_charge.device!=A.device or not z_charge.is_floating_point():
            raise ValueError('016 active branch needs explicit same-device finite B charge latent')
        Q=canonical_charge008(A)
        if not bool(torch.isfinite(Q).all() and (Q>0).all() and torch.isfinite(z_charge).all()):
            raise ValueError('016 actual canonical Q must be finite strictly positive; latent must be finite')
        a,b=charge_coefficients016(self.theta_a,self.theta_b,self.charge_q_reference)
        cv2=a/Q+b;s2=torch.log1p(cv2);log_multiplier=s2.sqrt()*z_charge.double()-.5*s2
        increment=A*torch.expm1(log_multiplier)[:,None]
        signal=original+increment
        q=Q*torch.exp(log_multiplier)
        if not bool(torch.isfinite(signal).all() and torch.isfinite(q).all() and (q>0).all() and torch.isfinite(cv2).all()):
            raise ValueError('016 nonfinite/underflow charge law or signal; no rows clipped or dropped')
        if bool((signal[:,:100]!=0).any()):raise ValueError('016 charge changed exact zero prefix100')
        observed=canonical_charge008(signal);inherited=canonical_charge008(original)-Q
        out.update(signal_pre=signal,pure_signal_pre_V=signal,M0_reference_signal_pre_V=original,
            charge_enabled=True,Q_mu_Vns=Q,qsig=q,qbar=Q,canonical_charge=observed,
            canonical_charge_expected_Vns=q+inherited,canonical_charge_identity_residual=observed-(q+inherited),
            inherited_M0_canonical_projection_residual_Vns=inherited,charge_increment_pre_V=increment,
            charge_increment_RMS_V=increment.square().mean(1).sqrt(),charge_expected_increment_RMS_V=(A.square().mean(1)*cv2).sqrt(),
            charge_conditional_CV=cv2.sqrt(),charge_log_variance=s2,charge_latent=z_charge,
            charge_a_Vns=a,charge_b=b,charge_variance_regular=cv2.mean(),stochastic_shape_regular=out['regular'],
            charge016=dict(schema='n01/positive-mean-corrected-charge016/v1',parameters=2,
                law='Q*exp(sqrt(log1p(a/Q+b))*z-0.5*log1p(a/Q+b))',denominator_floor=False,
                condition='actual frozen A canonical Q',shape_amplitude='original A only',mean_protection='preADC conditional expectation',
                M0_head_sha256=self.reference_M0_head_sha256))
        return out

    def random_forward(self,source_carrier_V,z_shape,z_charge=None,*,enabled=True,charge_enabled=True):
        out=super().random_forward(source_carrier_V,z_shape,enabled=enabled)
        out=self.apply_charge(out,z_charge,enabled=charge_enabled)
        self.assert_frozen_unchanged();return out

    forward=random_forward
