"""Four separate hard-observation evaluations; numerical imports are lazy.

Callers must run the shared frozen input/source gate before calling these
functions. This module never loads datasets/checkpoints, draws randomness,
fits statistics, selects rows, creates models, or decides scientific acceptance.
"""
from __future__ import annotations
import copy


def require(flag, message):
    if not flag:
        raise ValueError(message)


def summary(values):
    """Defined values are described with their full denominator and missingness."""
    import numpy as np
    x=np.asarray(values,dtype=np.float64).reshape(-1);defined=np.isfinite(x);v=x[defined]
    return dict(rows=len(x),defined_rows=int(defined.sum()),undefined_rows=int((~defined).sum()),
        mean=None if len(v)==0 else float(v.mean()),
        std_population=None if len(v)==0 else float(v.std()),
        quantiles=None if len(v)==0 else dict(zip(('p01','p10','p50','p90','p99'),map(float,np.quantile(v,[.01,.1,.5,.9,.99])))),
        undefined_is_not_zero=True)


def observable_features(wave256, baseline100, q_floor):
    import numpy as np
    from n01.diagnose_mean_random009 import morphology
    y,b=np.asarray(wave256,np.float64),np.asarray(baseline100,np.float64)
    require(y.ndim==2 and y.shape[1]==256 and len(y)>0 and b.shape==(len(y),100)
        and np.isfinite(y).all() and np.isfinite(b).all(),'Finite hard256 plus actual prefix100 required; no row dropping')
    require(q_floor>0,'Explicit frozen positive q_floor required')
    q,t=y.sum(1),y[:,46:].sum(1);psd=np.full(len(y),np.nan)
    np.divide(t,q,out=psd,where=q!=0)
    denominator=np.where(q<0,-1.,1.)*np.maximum(np.abs(q),float(q_floor))
    centered=b-b.mean(1,keepdims=True)
    features=dict(Q_Vns=q,tail_Vns=t,front_Vns=q-t,PSD_raw=psd,PSD_guarded=t/denominator,
        baseline_mean_V=b.mean(1),baseline_RMS_V=np.sqrt(np.mean(centered**2,axis=1)),**morphology(y))
    flags=dict(Q_nonpositive=q<=0,PSD_undefined=q==0,PSD_guarded=np.abs(q)<q_floor,
        **{n:v for n,v in features.items() if np.asarray(v).dtype.kind=='b'})
    return features,flags


def raw_distance(x,y):
    """Diagnostic W1 on explicitly defined values; never a whole-row replacement."""
    import numpy as np
    from stability011_metrics import CDFCache
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64);fx,ry=np.isfinite(x),np.isfinite(y)
    result=dict(fake_rows=len(x),real_rows=len(y),fake_defined_rows=int(fx.sum()),real_defined_rows=int(ry.sum()),
        domain='explicit_defined_subpopulation',undefined_rows_retained_in_flags=True)
    if not fx.any() or not ry.any():
        return dict(result,value=None,reason='no_defined_population_on_one_side')
    cache=CDFCache.create(x[fx],y[ry]);value=cache.distance(np.ones(int(fx.sum()),np.int64),np.ones(int(ry.sum()),np.int64),column_chunk=16)
    return dict(result,value=float(value))


def flag_rates(flags):
    import numpy as np
    result={}
    for name,value in flags.items():
        x=np.asarray(value);require(x.ndim==1 and x.dtype.kind=='b','Rowwise boolean flag required')
        result[name]=dict(count=int(x.sum()),denominator=len(x),fraction=float(x.mean()))
    return result


def native_metrics(fake256,fake_baseline100,real256,real_baseline100,loss,context):
    """Original frozen statistics, projections, support/fallback; no scalar AD."""
    import torch
    from n01.losses008_v2 import WAVE_WEIGHTS
    device=loss.q_reference.device
    convert=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
    with torch.no_grad():
        f,r,fb,rb=map(convert,(fake256,real256,fake_baseline100,real_baseline100))
        parts=loss.wave_components(f,r,None,context,mean_response=False)
        wave=sum(WAVE_WEIGHTS[k]*v for k,v in parts.items())
        qpsd=loss._qpsd_with_context(f,r,None,context,mean_response=False)
        coverage=copy.deepcopy(loss.last_diagnostics)
        noise=loss.noise_loss(fb,rb,None,projection_context=context)
    return dict(wave=float(wave),wave_components={k:float(v) for k,v in parts.items()},
        QPSD_native=float(qpsd),noise_prefix100=float(noise),coverage=coverage,
        auxiliary_legacy_wave_plus_2QPSD=float(wave+2*qpsd),projection_context=context.metadata(),
        primary_classes_reported_separately=True,statistic_refitting=False,comparison_weights=None)


def component_distances(wave256,baseline100,reference256,reference_baseline100,loss,context,*,cached_reference=None):
    """Same frozen scale on REAL comparisons and synthetic self comparisons.

Self reference is an empirical model response, never a new measurement. Keep
native coverage and missingness beside each scalar used by the root contract.
"""
    import numpy as np
    fake,ff=observable_features(wave256,baseline100,float(loss.q_floor))
    real,rf=observable_features(reference256,reference_baseline100,float(loss.q_floor)) if cached_reference is None else cached_reference
    require(all(len(v)==len(reference256) for v in real.values()),'Cached reference feature lineage mismatch')
    native=native_metrics(wave256,baseline100,reference256,reference_baseline100,loss,context)
    values=dict(wave=native['wave'],QPSD=native['QPSD_native'],noise=native['noise_prefix100'])
    diagnostics={}
    for name in ('Q_Vns','front_Vns','tail_Vns','PSD_raw','PSD_guarded','peak_voltage_V','rise10_90_ns','FWHM_ns','baseline_mean_V','baseline_RMS_V'):
        detail=raw_distance(fake[name],real[name]);values[name]=detail['value'];diagnostics[name]=detail
    for name in ('Q_nonpositive','PSD_undefined','PSD_guarded','rise_undefined','FWHM_undefined',
        'temporal_nonpositive_peak','rise_prepeak_censored','FWHM_prepeak_censored','FWHM_postpeak_censored'):
        values[name+'_rate']=float(abs(np.asarray(ff[name]).mean()-np.asarray(rf[name]).mean()))
    return dict(values=values,coverage=native['coverage'],defined_population=diagnostics,
        wave_components=native['wave_components'],legacy_objective=native['auxiliary_legacy_wave_plus_2QPSD'],
        reference_rows=len(reference256),fake_rows=len(wave256),all_rows_retained=True,
        standardization='original_frozen_TRAIN',projection_context=context.metadata())


def output_q_bins(q,loss):
    import numpy as np
    # Frozen extended-log coordinate; no new bin fit.
    qref,qfloor=float(loss.q_reference),float(loss.q_floor)
    bounds=loss.charge_bin_boundaries.detach().cpu().double().numpy()
    coordinate=np.where(q>=qfloor,np.log(np.maximum(q,qfloor)/qref),np.log(qfloor/qref)+np.arcsinh((q-qfloor)/qfloor))
    return np.searchsorted(bounds,coordinate,side='left')


def output_q_conditions(fake_features,real_features,loss):
    import numpy as np
    qref,qfloor=float(loss.q_reference),float(loss.q_floor);bounds=loss.charge_bin_boundaries.detach().cpu().double().numpy()
    fm,rm=output_q_bins(fake_features['Q_Vns'],loss),output_q_bins(real_features['Q_Vns'],loss);result=[]
    names=('Q_Vns','front_Vns','tail_Vns','PSD_raw','PSD_guarded','peak_voltage_V','rise10_90_ns','FWHM_ns')
    for index in range(8):
        a,b=fm==index,rm==index;nf,nr=int(a.sum()),int(b.sum())
        result.append(dict(bin=index,fake_rows=nf,real_rows=nr,supported=nf>=int(loss.minimum_support) and nr>=int(loss.minimum_support),
            fake={n:summary(fake_features[n][a]) for n in names},real={n:summary(real_features[n][b]) for n in names},
            defined_PSD_W1=raw_distance(fake_features['PSD_raw'][a],real_features['PSD_raw'][b])))
    return dict(bins=result,fake_bin_per_row=fm,real_bin_per_row=rm,boundaries=bounds,
        q_floor=qfloor,q_reference=qref,minimum_support=int(loss.minimum_support),
        all_rows_in_exactly_one_bin=True,output_Q_is_not_energy_or_particle_truth=True,
        missing_bins_not_zero_distance=True)


def repeat_grid(parent_row,draw_index,bank_index):
    """Reject duplicates or unequal draws, keeping parent clusters as the unit."""
    import numpy as np
    p,d,b=map(np.asarray,(parent_row,draw_index,bank_index))
    require(p.ndim==d.ndim==b.ndim==1 and len(p)==len(d)==len(b)>0
        and all(x.dtype.kind in 'iu' for x in (p,d,b)),'Integer parent/draw/bank lineage required')
    require(all(np.all(x>=0) for x in (p,d,b)),'Negative lineage index')
    order=np.lexsort((d,b,p));lineage=np.stack((p[order],b[order],d[order]),1)
    require(len(lineage)==len(np.unique(lineage,axis=0)),'Duplicate parent-bank-draw cannot be treated as fresh response')
    parents,counts=np.unique(p,return_counts=True);require(np.all(counts==counts[0]),'Unequal repeat count; no implicit reweighting')
    repeats=int(counts[0]);shape=(len(parents),repeats)
    return dict(parents=parents,repeats=repeats,order=order,
        bank_grid=b[order].reshape(shape),draw_grid=d[order].reshape(shape))


def repeated_parent_statistics(wave256,features,parent_row,draw_index,bank_index,signal_pre384=None):
    import numpy as np
    from n01.diagnose_mean_random009 import covariance_factor,scalar_covariance
    grid=repeat_grid(parent_row,draw_index,bank_index);order=grid['order'];p,r=len(grid['parents']),grid['repeats']
    require(r>=2,'At least two explicit draws required to estimate repeat covariance')
    y=np.asarray(wave256,np.float64)[order].reshape(p,r,256)
    names=('Q_Vns','tail_Vns','PSD_raw','PSD_guarded','peak_voltage_V','baseline_mean_V','baseline_RMS_V','rise10_90_ns','FWHM_ns')
    scalars=np.stack([features[n] for n in names],1)[order].reshape(p,r,len(names))
    wave=covariance_factor(y);scalar=scalar_covariance(scalars)
    result=dict(parent_rows=grid['parents'],bank_grid=grid['bank_grid'],draw_grid=grid['draw_grid'],parent_count=p,draws_per_parent=r,
        statistical_unit='SOURCE_parent_not_MC_draw',scalar_names=list(names),scalars=scalar,
        hard_wave_mean_V=wave['mean'],hard_wave_covariance_diagonal_V2=wave['sample_variance'],
        hard_wave_mean_MCSE_V=dict(reconstruction='sqrt(hard_wave_covariance_diagonal_V2/draws_per_parent)',dtype='float64'),
        complete_covariance_storage='reconstruct_centered_draw_factor_from_retained_observed_rows',
        covariance_denominator=r-1,covariance_rank_upper_bound=r-1,
        input_sampling_uncertainty='not_estimated_here',conditional_mean_truth='unpaired_REAL_unknown')
    if signal_pre384 is not None:
        s=np.asarray(signal_pre384,np.float64);require(s.shape==(len(order),384) and np.isfinite(s).all(),'Invalid preADC signal lineage')
        signal=covariance_factor(s[order].reshape(p,r,384))
        result.update(preADC_signal_mean_V=signal['mean'],preADC_signal_covariance_diagonal_V2=signal['sample_variance'],
            preADC_signal_mean_MCSE_V=dict(reconstruction='sqrt(preADC_signal_covariance_diagonal_V2/draws_per_parent)',dtype='float64'))
    else:result['preADC_signal_reason']='not_provided_by_caller'
    return result


def source_input_trends(features,parent_row,source_information,draw_index,bank_index):
    import numpy as np
    grid=repeat_grid(parent_row,draw_index,bank_index);parents=grid['parents'];order=grid['order'];r=grid['repeats']
    require(source_information is not None,'Declared SOURCE information artifact required; no fabricated descriptors')
    info=source_information;ids=np.asarray(info['source_id']);roles=np.asarray(info['role'])
    require(len(ids)==len(np.unique(ids)) and len(ids)==len(roles) and parents.max()<len(ids),'Invalid prepared-TRAIN information join')
    require(np.all(roles[parents]=='TRAIN'),'Internal_val/exposedDEV cannot enter model evaluation')
    q=np.asarray(info['canonical_Q_Vns'],np.float64)[parents]
    amp=np.asarray(info['amplitude_384_V'],np.float64)[parents]
    pcs=np.asarray(info['PC_scores_384_Qnorm'],np.float64)[parents]
    valid=np.asarray(info['PC_valid_384_Qnorm'],bool)[parents]
    outputs={n:features[n][order].reshape(len(parents),r) for n in ('Q_Vns','tail_Vns','peak_voltage_V','baseline_RMS_V')}
    mean={n:x.mean(1) for n,x in outputs.items()};variance={n:x.var(1,ddof=1) for n,x in outputs.items()}
    def correlation(a,b):
        ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok]
        if len(a)<2 or a.std()==0 or b.std()==0:return dict(value=None,reason='insufficient_defined_variation',parents=int(ok.sum()))
        return dict(value=float(np.corrcoef(a,b)[0,1]),parents=int(ok.sum()),kind='descriptive_parent_Pearson')
    predictors=dict(canonical_Q_Vns=q,amplitude_384_V=amp)
    for k in range(pcs.shape[1]):predictors['shape_PC_%02d'%k]=np.where(valid,pcs[:,k],np.nan)
    return dict(parent_rows=parents,source_ids=ids[parents],parent_count=len(parents),
        correlations={name:{'response_mean':{n:correlation(x,v) for n,v in mean.items()},
            'response_variance':{n:correlation(x,v) for n,v in variance.items()}} for name,x in predictors.items()},
        response_parent_mean=mean,response_parent_variance=variance,
        physical_source_weights='unknown_unless_separately_verified',original_parent_count_weight=1,
        descriptors_are_audit_only_not_network_inputs=True,unpaired_REAL_conditional_target='unavailable',
        inference='generator_input_association_not_identified_physical_conditional_response')


def evaluate_hard_population(wave256,baseline100,*,real_wave256,real_baseline100,parent_row,draw_index,bank_index,
    source_information,loss,context,observation_flags=None,real_observation_flags=None,real_unknown_flags=(),real_groups=None,signal_pre384=None,cached_real=None):
    import numpy as np
    fake,ff=observable_features(wave256,baseline100,float(loss.q_floor))
    real,rf=observable_features(real_wave256,real_baseline100,float(loss.q_floor)) if cached_real is None else cached_real
    rf=dict(rf);require(all(len(v)==len(real_wave256) for v in real.values()),'Cached REAL lineage mismatch')
    if observation_flags is not None:
        for n,value in observation_flags.items():
            v=np.asarray(value);require(v.shape==(len(wave256),) and v.dtype.kind=='b','Observation flag lineage mismatch');ff[n]=v
    if real_observation_flags is not None:
        for n,value in real_observation_flags.items():
            v=np.asarray(value);require(v.shape==(len(real_wave256),) and v.dtype.kind=='b','REAL observation flag lineage mismatch');rf[n]=v
    native=native_metrics(wave256,baseline100,real_wave256,real_baseline100,loss,context)
    quantities=('Q_Vns','front_Vns','tail_Vns','PSD_raw','PSD_guarded')
    morph=('peak_voltage_V','rise10_90_ns','FWHM_ns')
    population=dict(fake_rows=len(wave256),real_rows=len(real_wave256),all_population_rows_retained=True,
        waveform=native,integrals={n:dict(fake=summary(fake[n]),real=summary(real[n]),W1=raw_distance(fake[n],real[n])) for n in quantities},
        noise=dict(frozen_prefix100_distance=native['noise_prefix100'],fake={n:summary(fake[n]) for n in ('baseline_mean_V','baseline_RMS_V')},
            real={n:summary(real[n]) for n in ('baseline_mean_V','baseline_RMS_V')},pulse_first100_is_not_baseline=True),
        morphology={n:dict(fake=summary(fake[n]),real=summary(real[n]),W1_defined=raw_distance(fake[n],real[n])) for n in morph},
        fake_flags=flag_rates(ff),real_flags=flag_rates(rf),real_unknown_flags=list(real_unknown_flags),
        REAL_group_count=None if real_groups is None else int(len(np.unique(real_groups))),
        group_is_independent_acquisition=False,scientific_acceptance=False)
    return dict(schema='n01/wp1-hard-evaluation/v1',population=population,
        output_Q_conditions=output_q_conditions(fake,real,loss),
        source_input_trends=source_input_trends(fake,parent_row,source_information,draw_index,bank_index),
        repeated_parent=repeated_parent_statistics(wave256,fake,parent_row,draw_index,bank_index,signal_pre384),
        metric_classes=['wave','Q_tail_PSD','noise','morphology','input_dependence'],
        feature_rows=fake,observation_flags=ff,threshold_decision='owned_by_frozen_ACCEPTANCE_CONTRACT')
