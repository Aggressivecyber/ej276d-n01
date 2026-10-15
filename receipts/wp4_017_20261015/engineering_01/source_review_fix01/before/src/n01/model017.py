"""Known fixed A/B cache with exactly two learned charge scalars.

No file access, random draw, optimizer, observation or checkpoint load occurs
here. The caller supplies independent latent values and the frozen cache rows.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import torch
from torch import nn

from .model016 import charge_coefficients016, inverse_softplus016, Q_REFERENCE016
from .model008 import canonical_charge008


def require(value, message):
    if not bool(value):
        raise ValueError(message)


@contextmanager
def deterministic_loading017():
    """Zero temporary initialization, then caller strictly loads original state.

    Used only while constructing new local objects, in the main Python thread.
    Unlike seed/restore, this produces no placeholder random values. All init
    functions are restored even on failure. A hidden CPU RNG draw is an error.
    """
    require(threading.current_thread() is threading.main_thread(), '017 loader main thread required')
    require(not getattr(deterministic_loading017, 'active', False), '017 nested loader forbidden')
    names=('uniform_','normal_','trunc_normal_','kaiming_uniform_','kaiming_normal_',
           'xavier_uniform_','xavier_normal_','orthogonal_','sparse_')
    saved={name:getattr(nn.init,name) for name in names}
    before=torch.get_rng_state().clone()
    def zero(tensor,*args,**kwargs):
        with torch.no_grad(): return tensor.zero_()
    deterministic_loading017.active=True
    try:
        for name in names: setattr(nn.init,name,zero)
        yield
    finally:
        for name,value in saved.items(): setattr(nn.init,name,value)
        deterministic_loading017.active=False
        require(torch.equal(before,torch.get_rng_state()), '017 constructor advanced hidden CPU RNG')


def teacher_shape_coefficients017(z_shape, *, teacher, sign=None):
    require(z_shape.ndim==2 and z_shape.shape[1]==8 and bool(torch.isfinite(z_shape).all()), '017 shape latent Bx8')
    require(teacher in ('T1','T2'), '017 unknown teacher')
    if teacher=='T1':
        require(sign is None, '017 T1 has no mixture sign')
        return z_shape.double()
    require(sign is not None and sign.shape==(len(z_shape),) and sign.device==z_shape.device
        and bool(((sign==-1)|(sign==1)).all()), '017 T2 explicit independent Rademacher required')
    out=z_shape.double().clone()
    out[:,0]=.9*sign.double()+(1-.9**2)**.5*out[:,0]
    return out


def assemble_cached017(A, B, Q, z_shape, z_charge, a, b, *, enabled=True):
    require(A.ndim==2 and A.shape[1]==384 and A.dtype==torch.float64, '017 actual A float64 Bx384')
    require(B.shape==(len(A),8,384) and B.dtype==torch.float64 and B.device==A.device, '017 actual B float64 Bx8x384')
    require(Q.shape==(len(A),) and Q.dtype==torch.float64 and Q.device==A.device, '017 actual Q float64 B')
    require(bool(torch.isfinite(A).all() and torch.isfinite(B).all() and torch.isfinite(Q).all() and (Q>0).all()), '017 actual cache finite and positive Q')
    require(not A.requires_grad and not B.requires_grad and not Q.requires_grad, '017 cache must be frozen')
    require(z_shape.shape==(len(A),8) and z_shape.device==A.device and bool(torch.isfinite(z_shape).all()), '017 explicit shape latent')
    require(bool((A[:,:100]==0).all() and (B[:,:,:100]==0).all()), '017 exact zero signal prefix')
    require(type(enabled) is bool, '017 charge switch must be bool')
    shape=(B*z_shape.double()[:,:,None]).sum(1)
    active=z_shape.ne(0).any(1)
    M0=torch.where(active[:,None],A+shape,A)
    if not enabled:
        return dict(signal_pre=M0,M0_reference_signal_pre_V=M0,charge_enabled=False,
            Q_mu_Vns=Q,qsig=Q,charge_increment_pre_V=torch.zeros_like(A))
    require(z_charge is not None and z_charge.shape==(len(A),) and z_charge.device==A.device
        and bool(torch.isfinite(z_charge).all()), '017 explicit charge latent')
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all() and (a>0).all() and (b>0).all()), '017 a,b strictly positive')
    cv2=a/Q+b;s2=torch.log1p(cv2)
    log_multiplier=s2.sqrt()*z_charge.double()-.5*s2
    increment=A*torch.expm1(log_multiplier)[:,None]
    signal=M0+increment;q=Q*torch.exp(log_multiplier)
    require(bool(torch.isfinite(signal).all() and torch.isfinite(q).all() and (q>0).all()), '017 invalid positive charge realization')
    require(bool((signal[:,:100]==0).all()), '017 prefix changed')
    return dict(signal_pre=signal,M0_reference_signal_pre_V=M0,charge_enabled=True,
        Q_mu_Vns=Q,qsig=q,theory_variance_Vns2=a*Q+b*Q.square(),
        charge_increment_pre_V=increment,charge_conditional_CV=cv2.sqrt())


class CachedFrozenQ1Charge017(nn.Module):
    """Rows are supplied by the caller; no huge cache or original NN is copied."""
    def __init__(self, *, a_initial=.001*Q_REFERENCE016, b_initial=.00025, device='cpu'):
        super().__init__()
        self.theta_a=nn.Parameter(inverse_softplus016(a_initial/Q_REFERENCE016).to(device).clone())
        self.theta_b=nn.Parameter(inverse_softplus016(b_initial).to(device).clone())
        self.register_buffer('q_reference',torch.tensor(Q_REFERENCE016,dtype=torch.float64,device=device))
        self._ids=(id(self.theta_a),id(self.theta_b))

    def coefficients(self):
        require(float(self.q_reference)==Q_REFERENCE016,'017 fixed qref changed')
        return charge_coefficients016(self.theta_a,self.theta_b,self.q_reference)

    def trainable_parameters(self):
        p=(self.theta_a,self.theta_b)
        require(tuple(id(x) for x in p)==self._ids,'017 parameters replaced')
        return p

    def assert_optimizer_membership(self, optimizer):
        p=tuple(x for group in optimizer.param_groups for x in group['params'])
        require(len(p)==2 and {id(x) for x in p}==set(self._ids)
            and all(x.requires_grad for x in p),'017 optimizer exactly2theta')
        require(all(g['lr']==.01 and g['weight_decay']==0 for g in optimizer.param_groups),'017 registered AdamW settings')

    def forward(self,A,B,Q,z_shape,z_charge=None,*,enabled=True):
        a,b=self.coefficients()
        return assemble_cached017(A,B,Q,z_shape,z_charge,a,b,enabled=enabled)


def metric_features017(wave):
    require(wave.ndim>=2 and wave.shape[-1]==256 and bool(torch.isfinite(wave).all()), '017 full hard256 metric input')
    delta=1/16383
    return torch.cat((wave.double()/(16*delta),wave.double().sum(-1,keepdim=True)/(256*delta)),dim=-1)


def energy_score_U017(student, teacher, *, paired=False, block_rows=64):
    """Unbiased ES, with explicit sample self diagonal removal.

    paired: P x m x256 and P x n x256; marginal: m x256/n x256.
    Both cross sets must be independent, as enforced by the driver streams.
    Returns one value per parent for paired, scalar otherwise. No clipping.
    """
    require(type(block_rows) is int and block_rows>0,'017 distance block')
    S,T=metric_features017(student),metric_features017(teacher)
    require(S.ndim==T.ndim==(3 if paired else 2),'017 energy layout')
    if paired:require(len(S)==len(T),'017 paired parent identities differ')
    m,n=S.shape[-2],T.shape[-2]
    require(m>=2 and n>=1,'017 energy U-statistic sample counts')
    if paired and m==n==2:
        # Exactly four cross distances and one unique off-diagonal distance.
        # The two ordered student self terms are identical; neither diagonal
        # is computed in this registered paired training branch.
        cross=torch.cdist(S,T,p=2,compute_mode='donot_use_mm_for_euclid_dist').mean((-2,-1))
        unique=torch.linalg.vector_norm(S[...,0,:]-S[...,1,:],dim=-1)
        return cross-.5*unique
    cross=S.sum(dim=(-2,-1))*0.;self_sum=cross.clone()
    for start in range(0,m,block_rows):
        part=S[...,start:start+block_rows,:]
        cross=cross+torch.cdist(part,T,p=2,compute_mode='donot_use_mm_for_euclid_dist').sum(dim=(-2,-1))
        distances=torch.cdist(part,S,p=2,compute_mode='donot_use_mm_for_euclid_dist')
        rows=torch.arange(start,start+part.shape[-2],device=S.device)
        columns=torch.arange(m,device=S.device)
        mask=rows[:,None]!=columns[None,:]
        self_sum=self_sum+torch.where(mask,distances,torch.zeros_like(distances)).sum(dim=(-2,-1))
    return cross/(m*n)-self_sum/(2*m*(m-1))
