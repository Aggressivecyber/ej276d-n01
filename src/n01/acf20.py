"""Independent, differentiable all20-lag observed-baseline ACF target.

This does not change BaselineDistributionLoss or its existing32 features.
Inputs are finite floating Nx100 or Nx1x100. Reductions use float64, with
backpropagation retained to the input's original dtype. No data are read here.
"""
from __future__ import annotations

import torch
from torch import Tensor,nn


LENGTH=100
LAGS=20
ADC_STEP=1./16383
VARIANCE_FLOOR=ADC_STEP**2/12
LOSS_SCALE=.015
LOSS_COEFFICIENT=1.


def _baseline(x:Tensor)->Tensor:
    if not isinstance(x,Tensor) or not x.is_floating_point():
        raise ValueError("ACF20 input must be a floating tensor")
    if x.ndim==3 and x.shape[1]==1:x=x[:,0]
    if x.ndim!=2 or x.shape[1]!=LENGTH or x.shape[0]<1:
        raise ValueError("ACF20 requires nonempty Nx100 or Nx1x100;other lengths/channels are rejected")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite ACF20 input;no rows are removed or clipped")
    return x.to(torch.float64)


def acf20(x:Tensor)->Tensor:
    """Per-event demeaned lag1..20 covariance/variance, shape Nx20.

    Each lag uses its own100-lag overlap mean. The per-event population variance
    uses divisor100 and the same quantization floor as noise_validation. Finite
    window demeaning implies white-noise sample ACF need not have zero mean.
    """
    y=_baseline(x)
    b=y-y.mean(-1,keepdim=True)
    variance=b.square().mean(-1).clamp_min(VARIANCE_FLOOR)
    return torch.stack([(b[:,:-lag]*b[:,lag:]).mean(-1)/variance
                        for lag in range(1,LAGS+1)],dim=-1)


def mean_acf20(x:Tensor)->Tensor:
    """Unweighted mean of individual event ACFs, never a pooled ratio."""
    return acf20(x).mean(0)


class ACF20Loss(nn.Module):
    """Fixed detached TRAIN target; .5*(mean(e^2)+max(e^2)).

    e=(mean_acf20(fake)-target_acf20)/.015; coefficient is fixed1.
    No implicit target fitting, random state, data read, or mutable diagnostics.
    """
    def __init__(self,train_target20):
        super().__init__()
        target=torch.as_tensor(train_target20,dtype=torch.float64).detach().clone()
        if target.shape!=(LAGS,) or not bool(torch.isfinite(target).all()):
            raise ValueError("TRAIN ACF20 target must be an explicit finite20-vector")
        self.register_buffer("target_acf20",target)
        self.register_buffer("scale",target.new_tensor(LOSS_SCALE))
        self.register_buffer("coefficient",target.new_tensor(LOSS_COEFFICIENT))

    def forward(self,fake:Tensor)->Tensor:
        current=mean_acf20(fake)
        if current.device!=self.target_acf20.device:
            raise ValueError("Move ACF20Loss buffers to the fake waveform device")
        squared=((current-self.target_acf20)/self.scale).square()
        return self.coefficient*.5*(squared.mean()+squared.max())
