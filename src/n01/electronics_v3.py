"""100-point effective observed-baseline prior, not upstream electronics.

No pulse integration, source conditioning, file reads, or length extrapolation.
The full covariance and mean describe the fixed acquisition/selection/alignment
observation interface; ADC remains the caller's unchanged canonical operator.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor,nn
import torch.nn.functional as F


LENGTH=100
PARAMETER_COUNT=5151
NOISE_SHAPES={"baseline_gaussian":(1,LENGTH),"baseline_scale":(1,1)}


def _require_length(length):
    if isinstance(length,bool) or not isinstance(length,int) or length!=LENGTH:
        raise ValueError("ObservedBaselineNoiseV3 supports exactly100 samples;480 extrapolation/integration is prohibited")


def _inverse_softplus(value:Tensor)->Tensor:
    return value+torch.log(-torch.expm1(-value))


def fit_baseline_initialization_v3(train,*,length=100,covariance_ridge_relative=1e-6,
    normalized_diagonal_floor=1e-6,initial_log_scale_std=.02,scale_second_moment=1,
    init_covariance_ddof=1,full_mean_and_cholesky_trainable=True,**bindings)->dict:
    """All supplied TRAIN observations, full empirical mean/covariance.

    Accepts every registered config['model'] key (including type via bindings).
    Caller owns TRAIN lineage/canonicalization; this function performs no ADC,
    source selection, event demeaning/detrending, OLS fitting or dequantization.
    Only numerical ridge is added; no physical latent covariance is identified.
    """
    kind=bindings.pop("type","ObservedBaselineNoiseV3")
    if bindings or kind!="ObservedBaselineNoiseV3":raise ValueError("Unknown observed-baseline model binding")
    _require_length(length)
    if init_covariance_ddof!=1 or scale_second_moment!=1 or full_mean_and_cholesky_trainable is not True:
        raise ValueError("V3 requires ddof1 covariance,E[S^2]=1,and fully trainable mean/Cholesky")
    for name,value in (("covariance_ridge_relative",covariance_ridge_relative),
                       ("normalized_diagonal_floor",normalized_diagonal_floor),
                       ("initial_log_scale_std",initial_log_scale_std)):
        if not math.isfinite(float(value)) or float(value)<=0:raise ValueError(f"{name} must be finite and positive")
    x=torch.as_tensor(train).detach().to(device="cpu",dtype=torch.float64)
    if x.ndim==3 and x.shape[1]==1:x=x[:,0]
    if x.ndim!=2 or x.shape[1]!=100 or len(x)<2 or not bool(torch.isfinite(x).all()):
        raise ValueError("All supplied TRAIN baselines must be finite Nx100,N>=2;no rows removed")
    mean=x.mean(0);centered=x-mean
    empirical=centered.T@centered/(len(x)-1)
    empirical=(empirical+empirical.T)*.5
    variance=empirical.diagonal().mean()
    if not bool(torch.isfinite(variance)) or bool(variance<=0):raise ValueError("TRAIN covariance has no positive finite voltage normalization")
    voltage_scale=variance.sqrt()
    ridge=variance*float(covariance_ridge_relative)
    covariance=empirical+ridge*torch.eye(100,dtype=torch.float64)
    factor=torch.linalg.cholesky(covariance)
    if bool((factor.diagonal()/voltage_scale<=float(normalized_diagonal_floor)).any()):
        raise ValueError("Registered diagonal floor incompatible with empirical initialization;do not silently change it")
    return dict(length=100,mean_V=mean.tolist(),cholesky_V=factor.tolist(),voltage_scale=float(voltage_scale),
        normalized_diagonal_floor=float(normalized_diagonal_floor),initial_log_scale_std=float(initial_log_scale_std),
        metadata=dict(role="caller_supplied_all_legal_TRAIN_observed_baselines",events=len(x),samples=100,
            covariance_ddof=1,covariance_ridge_relative=float(covariance_ridge_relative),ridge_added_V2=float(ridge),
            empirical_covariance_trace_V2=float(torch.trace(empirical)),
            regularized_covariance_trace_V2=float(torch.trace(covariance)),
            initial_covariance_eigenvalue_min_V2=float(torch.linalg.eigvalsh(covariance).min()),
            no_event_demeaning=True,no_OLS_latent_initialization=True,no_ADC_noise_subtraction=True,
            preADC_scale_second_moment=1.,supported_lengths=[100],response480_integration_allowed=False,
            interpretation="numerical initialization from observedADC covariance;not latent hardware dequantization"))


class ObservedBaselineNoiseV3(nn.Module):
    """mean100 + S*L*z;S=exp(eta*u-eta^2),E[S^2]=1.

    Exactly5151 float64 trainable scalars. Independent explicit standard normals
    have no source/event ID input. Output is Bx1x100 float64 before canonical ADC.
    All other lengths fail; the model cannot replace480-point N01 electronics.
    """
    length=100
    supports_response480=False
    noise_shapes=NOISE_SHAPES

    def __init__(self,initialization:Mapping):
        super().__init__()
        required={"length","mean_V","cholesky_V","voltage_scale","normalized_diagonal_floor","initial_log_scale_std"}
        if not required<=set(initialization):raise ValueError("Explicit TRAIN or fabricated V3 initialization is required")
        _require_length(initialization["length"])
        mean=torch.as_tensor(initialization["mean_V"],dtype=torch.float64)
        lower=torch.as_tensor(initialization["cholesky_V"],dtype=torch.float64)
        scale=float(initialization["voltage_scale"])
        floor=float(initialization["normalized_diagonal_floor"])
        eta=float(initialization["initial_log_scale_std"])
        if mean.shape!=(100,) or lower.shape!=(100,100) or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(lower).all()):
            raise ValueError("Malformed/nonfinite100-point initialization")
        if not torch.equal(lower,torch.tril(lower)) or bool((lower.diagonal()<=0).any()):
            raise ValueError("Initialization must contain a lower triangular positive-diagonal Cholesky factor")
        if any(not math.isfinite(v) or v<=0 for v in (scale,floor,eta)):raise ValueError("V3 normalization/floor/spread must be positive and finite")
        self.initialization_metadata=dict(initialization.get("metadata",{}))
        self.register_buffer("voltage_scale",torch.tensor(scale,dtype=torch.float64))
        self.register_buffer("normalized_diagonal_floor",torch.tensor(floor,dtype=torch.float64))
        indices=torch.tril_indices(100,100)
        self.register_buffer("lower_rows",indices[0]);self.register_buffer("lower_columns",indices[1])
        self.register_buffer("lower_diagonal",indices[0]==indices[1])
        units=lower/scale
        adjusted=units.diagonal()-floor
        if bool((adjusted<=0).any()):raise ValueError("Positive diagonal floor exceeds initializer")
        packed=units[indices[0],indices[1]].clone()
        packed[self.lower_diagonal]=_inverse_softplus(adjusted)
        self.mean_units=nn.Parameter(mean/scale)
        self.raw_lower=nn.Parameter(packed)
        self.raw_log_scale_std=nn.Parameter(_inverse_softplus(torch.tensor(eta,dtype=torch.float64)))
        if sum(p.numel() for p in self.parameters())!=PARAMETER_COUNT:raise AssertionError("V3 trainable scalar count must be5151")

    @property
    def log_scale_std(self)->Tensor:
        return F.softplus(self.raw_log_scale_std)

    def mean_vector(self)->Tensor:
        return self.mean_units*self.voltage_scale

    def cholesky_factor(self)->Tensor:
        entries=torch.where(self.lower_diagonal,F.softplus(self.raw_lower)+self.normalized_diagonal_floor,self.raw_lower)
        factor=entries.new_zeros((100,100))
        factor[self.lower_rows,self.lower_columns]=entries
        return factor*self.voltage_scale

    def covariance(self)->Tensor:
        lower=self.cholesky_factor()
        return lower@lower.T

    def theoretical_stats(self)->dict[str,Tensor]:
        eta=self.log_scale_std;lower=self.cholesky_factor();covariance=lower@lower.T
        return dict(mean_V=self.mean_vector(),cholesky_V=lower,covariance_V2=covariance,
            log_scale_std=eta,scale_mean=torch.exp(-.5*eta.square()),
            scale_second_moment=torch.ones_like(eta),scale_fourth_moment=torch.exp(4*eta.square()),
            covariance_trace_V2=torch.trace(covariance),sample_std_V=covariance.diagonal().sqrt(),
            voltage_scale=self.voltage_scale,normalized_diagonal_floor=self.normalized_diagonal_floor)

    def theoretical_metadata(self)->dict:
        return dict(type="ObservedBaselineNoiseV3",length=100,trainable_scalars=5151,parameter_dtype="float64",
            output="preADC_Bx1x100",scale_rule="exp(eta*u-eta^2)",E_scale_squared=1,
            covariance="L L^T",Gaussian_global_scale_mixture=True,centrally_symmetric_preADC=True,
            supports_arbitrary_skew_or_multimodality=False,source_or_ID_conditioning=False,
            supported_lengths=[100],response480_integration_allowed=False,
            hardware_noise_identified=False,initialization=self.initialization_metadata)

    def sample_noise(self,batch:int,device=None,generator:torch.Generator|None=None,*,length:int=100)->dict[str,Tensor]:
        _require_length(length)
        if isinstance(batch,bool) or not isinstance(batch,int) or batch<1:raise ValueError("batch must be a positive integer")
        device=self.voltage_scale.device if device is None else device
        return {key:torch.randn((batch,)+shape,device=device,dtype=torch.float32,generator=generator)
                for key,shape in self.noise_shapes.items()}

    def forward(self,noise:Mapping[str,Tensor],*,length:int=100)->Tensor:
        _require_length(length)
        if set(noise)!=set(self.noise_shapes):raise ValueError("V3 requires exactly baseline_gaussian and baseline_scale latents;no ID/source fields")
        z=noise["baseline_gaussian"];u=noise["baseline_scale"]
        if not isinstance(z,Tensor) or z.ndim!=3 or z.shape[1:]!=(1,100) or z.shape[0]<1 or not isinstance(u,Tensor) or u.shape!=(z.shape[0],1,1):
            raise ValueError("Latents must have shapes Bx1x100 and Bx1x1")
        if not z.is_floating_point() or not u.is_floating_point() or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(u).all()):
            raise ValueError("Nonfinite/nonfloating explicit latents;no rows dropped")
        if z.device!=u.device or z.device!=self.voltage_scale.device:raise ValueError("Model and both explicit latents must share a device")
        eta=self.log_scale_std
        lower=self.cholesky_factor();mean=self.mean_vector()
        if not bool(torch.isfinite(eta)) or not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(mean).all()):
            raise ValueError("Nonfinite observed-baseline parameters")
        scale=torch.exp(eta*u.to(torch.float64)-eta.square())
        residual=z[:,0].to(torch.float64)@lower.T
        result=mean[None,None,:]+scale*residual[:,None,:]
        if not bool(torch.isfinite(result).all()):raise ValueError("Nonfinite observed-baseline output;no row deletion")
        return result
