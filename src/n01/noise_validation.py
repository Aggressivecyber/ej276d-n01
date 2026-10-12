"""Frozen, TRAIN/VAL group-isolated validation of the independent noise module.

This is a development gate on one acquisition, not hardware identification.
No function in this file changes the candidate electronics parameters.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import write_json, sha256
from .evaluate import _weighted_auc, _clean
from .model import digitize


def baseline_statistics(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 100 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 100 baseline; no row selection')
    mean = x.mean(1)
    b = x - mean[:, None]
    variance = np.maximum((b*b).mean(1), (1/16383)**2/12)
    t = np.linspace(-1, 1, 100)
    slope = b @ t / (t @ t)
    acf = np.array([(b[:, :-k]*b[:, k:]).mean(1)/variance for k in range(1, 21)]).T
    power = abs(np.fft.rfft(b, axis=1))**2
    bands = np.array([power[:, lo:hi].sum(1) for lo, hi in [(1,4),(4,9),(9,17),(17,33),(33,51)]]).T
    bands /= np.maximum(power[:, 1:].sum(1, keepdims=True), 1e-30)
    return dict(rows=len(x), mean_mean_V=float(mean.mean()), mean_std_V=float(mean.std(ddof=1)),
                rms_quantiles_V=np.quantile(np.sqrt(variance), [.1,.5,.9]),
                acf_mean=acf.mean(0), psd_band_fraction_mean=bands.mean(0),
                ols_slope_mean_V=float(slope.mean()),ols_slope_std_V=float(slope.std(ddof=1)), ols_covariance_V2=np.cov(np.array([mean,slope])),
                time_mean_V=x.mean(0), time_std_V=x.std(0,ddof=1))


def balanced_real_domains(groups, per_class, seed):
    """Disjoint random labels within every original group, exact class counts."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    quota = np.full(len(unique), per_class//len(unique), dtype=int)
    # Randomize which groups get the one-row remainder before any outcomes.
    quota[rng.permutation(len(unique))[:per_class%len(unique)]] += 1
    a, b = [], []
    for g,n in zip(unique,quota):
        rows = rng.permutation(np.flatnonzero(groups==g))
        if len(rows) < 2*n:
            raise ValueError('Insufficient group support for prespecified quota')
        a.extend(rows[:n]); b.extend(rows[n:2*n])
    a,b = np.asarray(a,dtype=np.int64),np.asarray(b,dtype=np.int64)
    assert len(a)==len(b)==per_class and not np.intersect1d(a,b).size
    return a,b


class BaselineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1,8,9,stride=4,padding=4),nn.SiLU(),
                                 nn.Conv1d(8,16,9,stride=4,padding=4),nn.SiLU(),nn.Flatten(),
                                 nn.Linear(16*7,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):
        return self.net(x[:,None])[:,0]


def _view(x, name):
    x=np.asarray(x,dtype=np.float64)
    if name=='raw': return x
    b=x-x.mean(1,keepdims=True)
    if name=='demean': return b
    if name=='detrend':
        t=np.linspace(-1,1,100)
        return b-(b@t/(t@t))[:,None]*t
    raise ValueError(name)


def _classifier(train0,train1,test0,test1,groups0,groups1,config,device,out,shared_groups):
    seed=int(config['classifier_seed'])
    torch.manual_seed(seed)
    x=np.concatenate([train0,train1]).astype(np.float32)
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(x.std(0,dtype=np.float64),1/16383).astype(np.float32)
    x=torch.as_tensor((x-center)/scale,device=device)
    y=torch.as_tensor(np.r_[np.zeros(len(train0)),np.ones(len(train1))],device=device,dtype=torch.float32)
    tx=torch.as_tensor((np.concatenate([test0,test1]).astype(np.float32)-center)/scale,device=device)
    labels=np.r_[np.zeros(len(test0)),np.ones(len(test1))]
    model=BaselineClassifier().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+1)
    losses=[]
    for _ in range(int(config['classifier_updates'])):
        rows=rng.choice(len(x),128,replace=False)
        loss=F.binary_cross_entropy_with_logits(model(x[rows]),y[rows])
        if not torch.isfinite(loss): raise FloatingPointError('classifier loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad(): scores=model(tx).cpu().numpy()
    if shared_groups:
        allgroups=np.r_[groups0,groups1]
        unique,index=np.unique(allgroups,return_inverse=True)
        brng=np.random.default_rng(seed+2)
        values=[]
        for _ in range(int(config['bootstrap_resamples'])):
            weights=np.bincount(brng.integers(0,len(unique),len(unique)),minlength=len(unique))[index]
            values.append(_weighted_auc(labels,scores,weights))
        interval=np.quantile(values,[.025,.975])
        uncertainty=dict(method='joint_original_group_across_random_labels',resamples=config['bootstrap_resamples'],percentile_95_interval=interval)
    else:
        u0,i0=np.unique(groups0,return_inverse=True);u1,i1=np.unique(groups1,return_inverse=True)
        brng=np.random.default_rng(seed+2);values=[]
        for _ in range(int(config['bootstrap_resamples'])):
            w0=np.bincount(brng.integers(0,len(u0),len(u0)),minlength=len(u0))[i0]
            w1=np.bincount(brng.integers(0,len(u1),len(u1)),minlength=len(u1))[i1]
            values.append(_weighted_auc(labels,scores,np.r_[w0,w1]))
        interval=np.quantile(values,[.025,.975])
        uncertainty=dict(method='independent_synthetic_parent_and_REAL_group_clusters',resamples=config['bootstrap_resamples'],percentile_95_interval=interval)
    leave_one=[]
    for group in np.unique(groups1):
        keep=np.r_[groups0!=group if shared_groups else np.ones(len(groups0),dtype=bool),groups1!=group]
        leave_one.append(_weighted_auc(labels[keep],scores[keep]))
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.cpu().state_dict(),center=center,scale=scale,seed=seed,
                    updates=config['classifier_updates']),out/'classifier.pt')
    np.savez_compressed(out/'predictions.npz',labels=labels,scores=scores,groups0=groups0,groups1=groups1,
                        loss=np.asarray(losses))
    result=dict(AUC=_weighted_auc(labels,scores),percentile_95_interval=interval,
                uncertainty=uncertainty,n_fit=[len(train0),len(train1)],n_test=[len(test0),len(test1)],
                test_group_count=[len(np.unique(groups0)),len(np.unique(groups1))],
                updates=config['classifier_updates'],orientation_flipped=False,scaler_fit_only=True,
                leave_one_REAL_group_out_AUC_range=[min(leave_one),max(leave_one)])
    write_json(out/'result.json',_clean(result))
    return result


def _swd(a,b,center,scale,directions):
    # All distance comparisons use equal sample count; exact uniform 1D W1.
    if len(a)!=len(b):raise ValueError('Matched SWD requires equal counts')
    x=(np.asarray(a)-center)/scale; y=(np.asarray(b)-center)/scale
    return float(np.abs(np.sort(x@directions,axis=0)-np.sort(y@directions,axis=0)).mean())


def evaluate_noise(model,train,validation,train_groups,val_groups,config,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if len(train)!=len(train_groups) or len(validation)!=len(val_groups):
        raise ValueError('One original group identity is required for every baseline')
    if np.intersect1d(train_groups,val_groups).size:raise ValueError('TRAIN VAL group overlap')
    if (out/'NOISE_GATE.json').exists():raise FileExistsError('Preserve frozen gate')
    model.eval().requires_grad_(False)
    rng=torch.Generator(device=device).manual_seed(int(config['generation_seed']))
    def generate(n):
        values=[]
        with torch.no_grad():
            for s in range(0,n,256):
                noise=model.sample_noise(min(256,n-s),device=device,generator=rng,length=100)
                values.append(digitize(model(noise,length=100),ste=False)[:,0].cpu().numpy())
        return np.concatenate(values)
    fake_fit=generate(int(config['classifier_fit_per_class']))
    fake=generate(int(config['generated_statistics_rows']))
    train=np.asarray(train,dtype=np.float64);validation=np.asarray(validation,dtype=np.float64)
    fit0,fit1=balanced_real_domains(train_groups,int(config['classifier_fit_per_class']),config['split_seed'])
    test0,test1=balanced_real_domains(val_groups,int(config['classifier_test_per_class']),config['split_seed']+1)
    np.savez_compressed(out/'baseline_populations.npz',generated=fake,generated_classifier_fit=fake_fit,
                        REAL_validation=validation,real_validation_groups=val_groups,
                        real_train_fit0=fit0,real_train_fit1=fit1,real_validation_test0=test0,real_validation_test1=test1)
    stats={name:baseline_statistics(x) for name,x in [('generated',fake),('REAL',validation),('TRAIN',train)]}
    classifiers={}
    n=int(config['classifier_test_per_class'])
    for view in ('raw','demean','detrend'):
        classifiers[view]={}
        classifiers[view]['generated_REAL']=_classifier(_view(fake_fit,view),_view(train[fit0],view),
            _view(fake[:n],view),_view(validation[test0],view),np.arange(n),val_groups[test0],
            config,device,out/'classifiers'/view/'generated_REAL',False)
        classifiers[view]['REAL_REAL']=_classifier(_view(train[fit0],view),_view(train[fit1],view),
            _view(validation[test0],view),_view(validation[test1],view),val_groups[test0],val_groups[test1],
            config,device,out/'classifiers'/view/'REAL_REAL',True)
    center=train.mean(0);scale=np.maximum(train.std(0),1/16383)
    nrng=np.random.default_rng(config['evaluation_projection_seed'])
    directions=nrng.normal(size=(100,256));directions/=np.linalg.norm(directions,axis=0)
    rr=[];gr=[]
    for k in range(int(config['distance_partitions'])):
        r0,r1=balanced_real_domains(val_groups,n,config['distance_seed']+k)
        fi=np.random.default_rng(config['distance_seed']+100+k).choice(len(fake),n,replace=False)
        rr.append(_swd(validation[r0],validation[r1],center,scale,directions))
        gr.append(_swd(fake[fi],validation[r0],center,scale,directions))
    rr_median=float(np.median(rr))
    if not math.isfinite(rr_median) or rr_median<=0:
        raise ValueError('REAL-REAL full100 reference median must be finite and strictly positive; gate invalid')
    joint=dict(direction_count=256,seed=config['evaluation_projection_seed'],matched_rows=n,
               REAL_REAL_distances=rr,generated_REAL_distances=gr,
               median_ratio=float(np.median(gr)/rr_median),TRAIN_scales_only=True)
    f,r=stats['generated'],stats['REAL']
    ols_scale=np.sqrt(np.diag(stats['TRAIN']['ols_covariance_V2']))
    scale_matrix=ols_scale[:,None]*ols_scale[None,:]
    scaled_f=np.asarray(f['ols_covariance_V2'])/scale_matrix
    scaled_r=np.asarray(r['ols_covariance_V2'])/scale_matrix
    metrics=dict(mean_shift_V=abs(f['mean_mean_V']-r['mean_mean_V']),
                 mean_shift_over_REAL_std=abs(f['mean_mean_V']-r['mean_mean_V'])/r['mean_std_V'],
                 mean_std_ratio=f['mean_std_V']/r['mean_std_V'],
                 rms_quantile_ratios=np.asarray(f['rms_quantiles_V'])/r['rms_quantiles_V'],
                 max_mean_acf_difference=float(np.max(np.abs(np.asarray(f['acf_mean'])-r['acf_mean']))),
                 psd_band_fraction_ratios=np.asarray(f['psd_band_fraction_mean'])/r['psd_band_fraction_mean'],
                 psd_band_fraction_absolute_difference=np.abs(np.asarray(f['psd_band_fraction_mean'])-r['psd_band_fraction_mean']),
                 ols_slope_std_ratio=f['ols_slope_std_V']/r['ols_slope_std_V'],
                 ols_covariance_relative_error=float(np.linalg.norm(scaled_f-scaled_r)/np.linalg.norm(scaled_r)),
                 ols_covariance_scaling='fixed_TRAIN_marginal_SD_of_centered_time_OLS',
                 full100_independent_SWD_ratio=joint['median_ratio'])
    # All numerical tolerances are read from the preregistered run JSON.
    g=config['gate']
    within=lambda x,b: bool(np.all((np.asarray(x)>=b[0])&(np.asarray(x)<=b[1])))
    raw=classifiers['raw']['generated_REAL'];control=classifiers['raw']['REAL_REAL']
    checks=dict(mean_location=metrics['mean_shift_V']<=max(g['mean_shift_ADC_steps_max']/16383,g['mean_shift_over_REAL_std_max']*r['mean_std_V']),
                mean_scatter=within(metrics['mean_std_ratio'],g['mean_std_ratio']),
                rms_quantiles=all(within(metrics['rms_quantile_ratios'][i],g['rms_quantile_ratios'][i]) for i in range(3)),
                acf=metrics['max_mean_acf_difference']<=g['mean_acf_absolute_max'],
                spectrum=all(within(ratio,g['psd_band_fraction_ratios']) or delta<=g['psd_band_fraction_absolute_max'] for ratio,delta in zip(metrics['psd_band_fraction_ratios'],metrics['psd_band_fraction_absolute_difference'])),
                ols_slope=within(metrics['ols_slope_std_ratio'],g['ols_slope_std_ratio']),
                ols_joint=metrics['ols_covariance_relative_error']<=g['ols_covariance_relative_error_max'],
                full100=metrics['full100_independent_SWD_ratio']<=g['full100_SWD_ratio_max'],
                classifier_point=within(raw['AUC'],g['classifier_point_interval']),
                classifier_interval=within(raw['percentile_95_interval'],g['classifier_CI_containment']),
                negative_control_point=within(control['AUC'],g['negative_control_point_interval']),
                negative_control_interval_contains_half=bool(control['percentile_95_interval'][0]<=.5<=control['percentile_95_interval'][1]))
    passed=all(checks.values())
    control_inconclusive=not(checks['negative_control_point'] and checks['negative_control_interval_contains_half'])
    status=('development_noise_gate_inconclusive_control' if control_inconclusive else
            'development_noise_gate_passed' if passed else 'development_noise_gate_not_passed')
    report=dict(status=status,
                passed=passed,checks=checks,failed_checks=[k for k,v in checks.items() if not v],
                metrics=metrics,statistics=stats,independent_full100=joint,classifiers=classifiers,
                train_original_groups=len(np.unique(train_groups)),validation_original_groups=len(np.unique(val_groups)),
                group_overlap=0,validation_previously_used_for_diagnosis=True,physical_calibration_established=False,
                independent_acquisitions=1,gate_type='predeclared_practical_development_screen_not_hardware_equivalence',
                classifier_updates=6*config['classifier_updates'],electronics_updates=0,
                next_action='two_fixed_uniform_population_response_arms' if passed else 'stop_before_pulse_training',
                validation_rows_used_for_descriptive_statistics=len(validation),generated_statistics_rows=len(fake),
                generated_real_test_rows=n,negative_control_test_rows=2*n,
                control_inconclusive=control_inconclusive,
                uncertainty_scope='12_original_groups_one_previously_viewed_acquisition;not_independent_acquisition_uncertainty',
                limited_original_group_count=True)
    write_json(out/'NOISE_GATE.json',_clean(report))
    return report
