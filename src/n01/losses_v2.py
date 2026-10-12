"""User-authorized N01 loss repair. The original losses.py remains immutable.

Projection banks are explicit values, refreshed by the caller's scientific
update number; forward calls never advance an RNG or hidden update counter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .losses import (ADC_STEP, N01Loss, _directions, _noise_features, _probabilities,
                     _qpsd, _robust_statistics, _wave_views, _waveform,
                     weighted_wasserstein_1d as _weighted_wasserstein_1d, window_charge)

VIEWS=("voltage","shape","difference","curvature","difference_smooth_5","difference_smooth_17")
VIEW_DIMENSIONS=(480,480,479,478,479,479)
COORDINATE_VIEWS=("voltage","shape")
SHORT_LAGS=(1,2,4,8)
WAVE_WEIGHTS=MappingProxyType(dict(swd=.50,coordinate=.25,short_lag=.25))
BASELINE_WEIGHTS=MappingProxyType(dict(swd=.50,coordinate=.25,
    feature_marginal=.125,feature_joint=.0625,ols_joint=.0625))


def _bank_digest(banks):
    digest=hashlib.sha256()
    for key in sorted(banks):
        value=banks[key].detach().cpu().contiguous()
        digest.update(key.encode("utf8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectionContext:
    """Immutable-by-contract CPU tensors, with in-place version checks.

    Tensor storage cannot be made read-only by PyTorch. Exposing it allows
    efficient device conversion; any in-place mutation is detected before use.
    """
    kind: str
    seed: int
    update: int | None
    banks: Mapping[str,Tensor]
    fingerprint: str=field(init=False)
    _versions: tuple=field(init=False,repr=False)

    def __post_init__(self):
        if self.kind not in ("training","evaluation"):
            raise ValueError("Projection context kind must be training/evaluation")
        if self.kind=="training" and (self.update is None or self.update<0):
            raise ValueError("Training context requires a nonnegative explicit scientific update")
        if self.kind=="evaluation" and self.update is not None:
            raise ValueError("Frozen evaluation context has no training update")
        banks={key:value.detach().cpu().contiguous().clone() for key,value in self.banks.items()}
        object.__setattr__(self,"banks",MappingProxyType(banks))
        object.__setattr__(self,"fingerprint",_bank_digest(banks))
        object.__setattr__(self,"_versions",tuple((key,value._version) for key,value in banks.items()))

    def validate(self):
        if any(self.banks[key]._version!=version for key,version in self._versions):
            raise RuntimeError("Projection context was mutated in-place; reuse its immutable original bank")
        return self

    def metadata(self):
        return dict(kind=self.kind,seed=self.seed,update=self.update,fingerprint=self.fingerprint)

    def state_dict(self):
        self.validate()
        return dict(kind=self.kind,seed=self.seed,update=self.update,
                    banks={key:value.clone() for key,value in self.banks.items()},
                    fingerprint=self.fingerprint)

    @classmethod
    def from_state_dict(cls,state):
        result=cls(state["kind"],int(state["seed"]),state["update"],state["banks"])
        if result.fingerprint!=state["fingerprint"]:
            raise ValueError("Serialized projection-bank fingerprint mismatch")
        return result


def _make_context(kind,seed,update):
    text=f"N01-loss-v2|{kind}|{int(seed)}|{update}".encode("ascii")
    mixed=int.from_bytes(hashlib.sha256(text).digest()[:8],"little") & ((1<<63)-1)
    generator=torch.Generator(device="cpu").manual_seed(mixed)
    banks={}
    for name,dimension in zip(VIEWS,VIEW_DIMENSIONS):
        banks["wave:"+name]=_directions(dimension,32,generator)
    for name in COORDINATE_VIEWS:
        for lag in SHORT_LAGS:
            banks[f"lag:{name}:{lag}"]=_directions(2,4,generator)
    banks["qpsd"]=_directions(2,16,generator)
    banks["noise"]=_directions(32,32,generator)
    banks["baseline_wave"]=_directions(100,32,generator)
    banks["baseline_features"]=_directions(32,16,generator)
    banks["baseline_ols"]=_directions(4,16,generator)
    return ProjectionContext(kind,int(seed),update,banks)


def create_projection_context(update:int,seed:int) -> ProjectionContext:
    """Call ONCE per scientific optimizer update, outside microbatch/replay loops."""
    if isinstance(update,bool) or int(update)!=update or update<0:
        raise ValueError("Scientific update must be a nonnegative integer")
    return _make_context("training",seed,int(update))


def create_evaluation_context(seed:int) -> ProjectionContext:
    """One independently seeded frozen bank for all endpoint/control comparisons."""
    return _make_context("evaluation",seed,None)


def _require_context(context,training_seed=None,evaluation_seed=None):
    if not isinstance(context,ProjectionContext):
        raise ValueError("An explicit ProjectionContext is required; no hidden forward RNG is used")
    context.validate()
    if context.kind=="evaluation" and training_seed is not None and context.seed==training_seed:
        raise ValueError("Evaluation seed must be independent of the training projection seed")
    if context.kind=="training" and evaluation_seed is not None and context.seed==evaluation_seed:
        raise ValueError("Training seed must not equal the frozen evaluation seed")
    return context


def _constant_untrained_weights(weights,count):
    if weights is None:
        return True
    weights=weights.reshape(-1)
    if weights.requires_grad or len(weights)!=count or len(weights)==0:
        return False
    return bool(torch.isfinite(weights).all() and weights[0]>0 and (weights==weights[0]).all())


def empirical_wasserstein_1d(x,y,weights=None,real_weights=None):
    """Exact uniform equal-N fast path; learnable weights always use CDF W1.

    Strictly constant detached weights define the same uniform empirical measure.
    No approximate-uniform tolerance is applied, and no weight gradient is lost.
    """
    if x.ndim==1:
        x=x[:,None]
    if y.ndim==1:
        y=y[:,None]
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or min(len(x),len(y))==0:
        raise ValueError("Empirical W1 requires nonempty (N,D)/(M,D) with identical D")
    if (len(x)==len(y) and _constant_untrained_weights(weights,len(x))
            and _constant_untrained_weights(real_weights,len(y))):
        y=y.to(x)
        return (x.sort(dim=0).values-y.sort(dim=0).values).abs().mean()
    return _weighted_wasserstein_1d(x,y,weights,real_weights)

def _standard64(x,center,scale):
    x=x.to(torch.float64)
    return (x-center.to(x))/scale.to(x)


def _weighted_mean(x,weights):
    p=_probabilities(weights,len(x),x)
    return (x*p[:,None]).sum(0)


def _pair_projection(x,lag,directions):
    d=directions.to(x)
    return (x[:,:-lag,None]*d[0]+x[:,lag:,None]*d[1]).flatten(1)


class N01LossV2(N01Loss):
    """Refreshed SWD + voltage/shape coordinates + time-resolved short-lag joint.

    Returns the same five generator components as v1. QPSD, weight detachment,
    output-Q bins, support16 and all-row global semantics are preserved.
    """
    def __init__(self,stats:dict[str,Any],seed:int=2026101211,evaluation_seed:int=2026101212):
        if seed==evaluation_seed:
            raise ValueError("Training and frozen evaluation projection seeds must differ")
        super().__init__(stats,seed=seed)
        if tuple(self.view_names)!=VIEWS:
            raise ValueError("Loss v2 requires the documented six480-point N01 views")
        self.training_projection_seed=int(seed)
        self.evaluation_projection_seed=int(evaluation_seed)
        self.last_wave_components={}

    def create_projection_context(self,update,seed=None):
        return create_projection_context(update,self.training_projection_seed if seed is None else seed)

    def create_evaluation_context(self,seed=None):
        actual=self.evaluation_projection_seed if seed is None else int(seed)
        if actual==self.training_projection_seed:
            raise ValueError("Frozen evaluation seed equals training seed")
        return create_evaluation_context(actual)

    def _z(self,name,x):
        return _standard64(x,getattr(self,name+"_center"),getattr(self,name+"_scale"))

    def wave_components(self,fake,real,weights,projection_context,mean_response=False):
        context=_require_context(projection_context,self.training_projection_seed,self.evaluation_projection_seed)
        fv=_wave_views(_waveform(fake).double(),self.peak_floor)
        rv=_wave_views(_waveform(real).double(),self.peak_floor)
        f={name:self._z(name,fv[name]) for name in VIEWS}
        r={name:self._z(name,rv[name]) for name in VIEWS}
        swd=[]
        for name in VIEWS:
            if mean_response:
                term=(_weighted_mean(f[name],weights)-r[name].mean(0)).abs().mean()
            else:
                d=context.banks["wave:"+name].to(f[name])
                term=empirical_wasserstein_1d(f[name]@d,r[name]@d,weights)
            swd.append(term)
        coordinate=[]
        joint=[]
        for name in COORDINATE_VIEWS:
            if mean_response:
                term=(_weighted_mean(f[name],weights)-r[name].mean(0)).abs().mean()
            else:
                term=empirical_wasserstein_1d(f[name],r[name],weights)
            coordinate.append(term)
            for lag in SHORT_LAGS:
                d=context.banks[f"lag:{name}:{lag}"]
                fp,rp=_pair_projection(f[name],lag,d),_pair_projection(r[name],lag,d)
                if mean_response:
                    term=(_weighted_mean(fp,weights)-rp.mean(0)).abs().mean()
                else:
                    term=empirical_wasserstein_1d(fp,rp,weights)
                joint.append(term)
        return dict(swd=torch.stack(swd).mean(),coordinate=torch.stack(coordinate).mean(),
                    short_lag=torch.stack(joint).mean())

    def noise_loss(self,fake,real,weights=None,*,projection_context=None):
        context=_require_context(projection_context,self.training_projection_seed,self.evaluation_projection_seed)
        f,r=self._z("noise",self.noise_features(fake)),self._z("noise",self.noise_features(real))
        d=context.banks["noise"].to(f)
        return .5*(empirical_wasserstein_1d(f,r,weights)+empirical_wasserstein_1d(f@d,r@d,weights))

    def _qpsd_with_context(self,fake,real,weights,context,mean_response=False):
        fraw,rraw=_qpsd(fake,self.q_reference,self.q_floor),_qpsd(real,self.q_reference,self.q_floor)
        f,r=self._z("qpsd",fraw),self._z("qpsd",rraw)
        if mean_response:
            global_term=(_weighted_mean(f,weights)-r.mean(0)).abs().mean()
        else:
            d=context.banks["qpsd"].to(f)
            global_term=.5*(empirical_wasserstein_1d(f,r,weights)+empirical_wasserstein_1d(f@d,r@d,weights))
        fb=torch.bucketize(fraw[:,0].detach().contiguous(),self.charge_bin_boundaries)
        rb=torch.bucketize(rraw[:,0].detach().contiguous(),self.charge_bin_boundaries)
        conditional,diagnostics=[],[]
        for index in range(8):
            fm,rm=fb==index,rb==index
            nf,nr=int(fm.sum()),int(rm.sum())
            supported=nf>=self.minimum_support and nr>=self.minimum_support
            diagnostics.append(dict(bin=index,fake_count=nf,real_count=nr,supported=supported,
                status="used" if supported else "insufficient_current_batch_support"))
            if supported:
                fw=None if weights is None else weights.reshape(-1)[fm]
                term=(_weighted_mean(f[fm,1:2],fw)-r[rm,1].mean()).abs().mean() if mean_response else empirical_wasserstein_1d(f[fm,1],r[rm,1],fw)
                conditional.append(term)
        fq,rq=window_charge(fake),window_charge(real)
        self.last_diagnostics=dict(fake_global_rows=len(fake),real_global_rows=len(real),
            conditional_bins=diagnostics,minimum_conditional_support=self.minimum_support,
            conditional_bins_used=len(conditional),conditional_bins_total=8,
            conditional_coverage_complete=len(conditional)==8,conditional_rebatch_needed=len(conditional)!=8,
            fake_nonpositive_q_rows=int((fq<=0).sum()),real_nonpositive_q_rows=int((rq<=0).sum()),
            fake_guarded_psd_rows=int((fq.abs()<self.q_floor).sum()),real_guarded_psd_rows=int((rq.abs()<self.q_floor).sum()),
            global_term_includes_all_rows=True,conditional_term="mean_PSD" if mean_response else "PSD_empirical_W1",
            no_supported_bin_behavior="global_term_only_with_incomplete_coverage_diagnostic",
            projection_context=context.metadata(),loss_version=2,standardization_split="TRAIN")
        return .5*(global_term+torch.stack(conditional).mean()) if conditional else global_term

    def population_objective(self,fake,real,weights,*,projection_context=None):
        context=_require_context(projection_context,self.training_projection_seed,self.evaluation_projection_seed)
        return self._qpsd_with_context(_waveform(fake).detach(),_waveform(real).detach(),weights,context)

    def components(self,outputs,real,comparison_weights=None,stage="joint_stochastic",*,projection_context=None):
        context=_require_context(projection_context,self.training_projection_seed,self.evaluation_projection_seed)
        if stage not in ("baseline","mean_response","joint_stochastic","low_learning_rate"):
            raise ValueError(f"Unknown N01 stage {stage}")
        fake,real=_waveform(outputs["waveform"]),_waveform(real)
        weights=None if comparison_weights is None else comparison_weights.detach().reshape(-1).to(fake)
        mean_stage=stage=="mean_response"
        parts=self.wave_components(fake,real,weights,context,mean_stage)
        self.last_wave_components={key:value.detach() for key,value in parts.items()}
        wave=sum(WAVE_WEIGHTS[key]*value for key,value in parts.items())
        qpsd=self._qpsd_with_context(fake,real,weights,context,mean_stage)
        zero=fake.sum()*0
        regular,trend=outputs.get("regular",zero),outputs.get("trend",zero)
        if regular.numel()!=1 or trend.numel()!=1:
            raise ValueError("regular/trend must be scalar generator outputs")
        return dict(wave=wave,QPSD=qpsd,
            noise=self.noise_loss(fake,real,weights,projection_context=context),
            regular=regular.reshape(()),trend=trend.reshape(()))

    def forward(self,outputs,real,comparison_weights=None,stage="joint_stochastic",*,projection_context=None):
        return self.components(outputs,real,comparison_weights,stage,projection_context=projection_context)


def _baseline(x):
    if x.ndim==3 and x.shape[1]==1:
        x=x[:,0]
    if x.ndim!=2 or x.shape[1] not in (100,480) or len(x)==0:
        raise ValueError("Baseline loss requires nonempty (N,100)/(N,480) waveforms")
    return x[:,:100].to(torch.float64)


def baseline_ols_features(x):
    """Mean, global-coordinate OLS slope/intercept, and detrended logRMS."""
    b=_baseline(x)
    t=2*torch.arange(100,device=b.device,dtype=b.dtype)/479-1
    tc=t-t.mean()
    mean=b.mean(1)
    slope=((b-mean[:,None])*tc).sum(1)/tc.square().sum()
    intercept=mean-slope*t.mean()
    residual=b-mean[:,None]-slope[:,None]*tc
    log_rms=.5*residual.square().mean(1).clamp_min(ADC_STEP**2/12).log()
    return torch.stack((mean,slope,intercept,log_rms),1)


@torch.no_grad()
def fit_baseline_statistics(real_train:Tensor) -> dict[str,Any]:
    """Fit legal REAL TRAIN first100 only. Caller owns the split provenance audit."""
    b=_baseline(real_train).detach().cpu()
    invalid=~torch.isfinite(b).all(1)
    if invalid.any():
        raise ValueError(f"REAL TRAIN baseline has {int(invalid.sum())} nonfinite rows; no rows dropped")
    if len(b)<2:
        raise ValueError("At least two TRAIN baseline rows are required")
    feature_floor=torch.tensor([ADC_STEP]*2+[.02]+[.02]*16+[ADC_STEP]*3+[.01]*5+[.04]*5)
    return dict(schema_version=2,fit_split="TRAIN",fit_rows=len(b),baseline_samples=100,
        waveform=_robust_statistics(b,ADC_STEP),features=_robust_statistics(_noise_features(b),feature_floor),
        ols=_robust_statistics(baseline_ols_features(b),torch.tensor([ADC_STEP,ADC_STEP,ADC_STEP,.02])),
        metadata=dict(row_selection="none",full_pulse_features_used=False,
            OLS_time_coordinate="2*sample/479-1_on_first100",intercept_reference="full_window_coordinate0",
            RMS_scope="OLS-detrended_first100",latent_OLS_coefficients_assumed_independent=False,
            noise_features="mean,slope,logRMS,ACF1..16,amplitude_quantiles,power_fraction5,local_logvariance5"))


class BaselineDistributionLoss(nn.Module):
    """Independent electronics objective, no SOURCE/pulse/particle features or GAN."""
    weights=BASELINE_WEIGHTS

    def __init__(self,stats,seed=2026101221,evaluation_seed=2026101222):
        super().__init__()
        if stats.get("fit_split")!="TRAIN" or stats.get("baseline_samples")!=100:
            raise ValueError("Baseline distribution statistics must be fitted to legal TRAIN first100")
        if seed==evaluation_seed:
            raise ValueError("Training/evaluation direction seeds must differ")
        self.training_projection_seed,self.evaluation_projection_seed=int(seed),int(evaluation_seed)
        self.statistics_metadata=stats.get("metadata",{})
        for name,stats_name in (("waveform","waveform"),("features","features"),("ols","ols")):
            for key in ("center","scale"):
                self.register_buffer(name+"_"+key,torch.as_tensor(stats[stats_name][key]).clone())

    def create_projection_context(self,update,seed=None):
        return create_projection_context(update,self.training_projection_seed if seed is None else seed)

    def create_evaluation_context(self,seed=None):
        actual=self.evaluation_projection_seed if seed is None else int(seed)
        if actual==self.training_projection_seed:
            raise ValueError("Frozen evaluation seed equals training seed")
        return create_evaluation_context(actual)

    def _z(self,name,x):
        return _standard64(x,getattr(self,name+"_center"),getattr(self,name+"_scale"))

    def components(self,fake,real,comparison_weights=None,*,projection_context=None):
        context=_require_context(projection_context,self.training_projection_seed,self.evaluation_projection_seed)
        fake,real=_baseline(fake),_baseline(real)
        # This independent fit targets the REAL electronics population uniformly;
        # optional explicit comparison weights apply only to simulated rows.
        weights=None if comparison_weights is None else comparison_weights.detach().reshape(-1).to(fake)
        f,r=self._z("waveform",fake),self._z("waveform",real)
        nf,nr=self._z("features",_noise_features(fake)),self._z("features",_noise_features(real))
        of,orr=self._z("ols",baseline_ols_features(fake)),self._z("ols",baseline_ols_features(real))
        wd=context.banks["baseline_wave"].to(f)
        nd=context.banks["baseline_features"].to(nf)
        od=context.banks["baseline_ols"].to(of)
        return dict(swd=empirical_wasserstein_1d(f@wd,r@wd,weights),
                    coordinate=empirical_wasserstein_1d(f,r,weights),
                    feature_marginal=empirical_wasserstein_1d(nf,nr,weights),
                    feature_joint=empirical_wasserstein_1d(nf@nd,nr@nd,weights),
                    ols_joint=empirical_wasserstein_1d(of@od,orr@od,weights))

    def total(self,components):
        if set(components)!=set(self.weights):
            raise ValueError("Baseline component keys differ from fixed registered weights")
        return sum(self.weights[key]*value for key,value in components.items())

    def forward(self,fake,real,comparison_weights=None,*,projection_context=None):
        return self.components(fake,real,comparison_weights,projection_context=projection_context)
