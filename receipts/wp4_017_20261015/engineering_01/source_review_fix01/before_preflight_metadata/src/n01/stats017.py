"""NumPy-only fixed WP4 finite-support, weighting and conditional gates."""
from __future__ import annotations
import hashlib
import numpy as np
from .saved_array_kernels017 import scalar_moments, morphology

FEATURES017=('Q','tail','PSD_raw','baseline_mean','baseline_RMS','peak_voltage','rise10_90','FWHM','peak_index')
TIME017=(8,6,7)
COVERAGE017=(0,1,5,8,6,7)
FLOORS017=np.asarray([256/16383,256/16383,1e-6,1/16383,1/16383,1/16383,1.,1.,1.])
REQUIRED_MISSING_FLAGS017=('flag__PSD_undefined','flag__rise_undefined','flag__FWHM_undefined',
    'flag__temporal_nonpositive_peak','flag__rise_prepeak_censored','flag__rise_other_undefined',
    'flag__FWHM_prepeak_censored','flag__FWHM_postpeak_censored','flag__FWHM_other_undefined')
REQUIRED_MISSING017=tuple('undefined_'+f for f in FEATURES017)+REQUIRED_MISSING_FLAGS017

def require(value,message):
    if not bool(value):raise ValueError(message)

def partition017(ids, heldout=1319):
    ids=np.asarray(ids)
    require(ids.ndim==1 and len(np.unique(ids))==len(ids),'017 unique parent IDs')
    require(0<heldout<len(ids),'017 nonempty parent isolation')
    def canonical(x):return str(int(x)) if isinstance(x,(int,np.integer)) else str(x)
    order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256(('N01-WP4-017-parent-split-v1|'+canonical(ids[i])).encode()).digest(),canonical(ids[i])))
    return np.sort(order[heldout:]),np.sort(order[:heldout])

def groups_weights017(fit_Q,test_Q,ids):
    fit_Q,test_Q=np.asarray(fit_Q),np.asarray(test_Q)
    require(np.isfinite(fit_Q).all() and np.isfinite(test_Q).all() and (fit_Q>0).all() and (test_Q>0).all(),'017 group actual positive Q')
    cuts=np.quantile(fit_Q,[1/3,2/3],method='linear')
    label=np.searchsorted(cuts,test_Q,side='right')
    # Stable parent-ID tie order, never a model input.
    order=np.lexsort((np.asarray(ids).astype(str),test_Q));rank=np.empty(len(order),np.int64);rank[order]=np.arange(len(order))
    u=2*(rank+.5)/len(rank)-1
    groups={'all_heldout':np.ones(len(rank),bool),**{f'fit_Q_tertile{k}_heldout':label==k for k in range(3)}}
    weights={'nominal':np.ones(len(rank)),'low':np.exp(-np.log(2)*u),'high':np.exp(np.log(2)*u)}
    return groups,weights,cuts

def weighted017(values, weights):
    values,weights=np.asarray(values,np.float64),np.asarray(weights,np.float64)
    require(values.ndim==weights.ndim==1 and len(values)==len(weights)>1 and np.isfinite(values).all()
        and np.isfinite(weights).all() and (weights>0).all(),'017 finite weighted parent values')
    alpha=weights/weights.sum();mean=float(alpha@values)
    return dict(mean=mean,SE=float(np.sqrt(len(values)/(len(values)-1)*np.sum(alpha**2*(values-mean)**2))),
        ESS=float(1/np.sum(alpha**2)),parents=len(values))

def support_cell017(group,weights,supported):
    group,weights,supported=np.asarray(group,bool),np.asarray(weights,np.float64),np.asarray(supported,bool)
    require(group.shape==weights.shape==supported.shape and (weights>0).all() and np.isfinite(weights).all(),'017 group weight schema')
    selected=group&supported;N=int(group.sum());n=int(selected.sum());W=float(weights[group].sum());Wi=float(weights[selected].sum())
    mass=Wi/W if W>0 else 0.;ESS=Wi**2/float(np.square(weights[selected]).sum()) if n else 0.
    reasons=[]
    if n<64:reasons.append('fewer_than64_common_parents')
    if N==0 or n/N<.95:reasons.append('common_count_below95percent_original')
    if mass<.95:reasons.append('common_weight_mass_below95percent_original')
    if ESS<.5*N:reasons.append('ESS_below_half_original_parent_count')
    return dict(status='pending' if reasons else 'supported',reasons=reasons,original_parents=N,
        supported_parents=n,original_weight=W,supported_weight=Wi,supported_weight_fraction=mass,
        ESS=ESS,indices=np.flatnonzero(selected))

def scalar_banks017(value):
    x=np.asarray(value,np.float64)
    require(x.ndim==4 and x.shape[1:]==(2,32,9),'017 scalar P,2,32,9')
    native=[scalar_moments(x[:,b]) for b in range(2)]
    combined=scalar_moments(x.reshape(len(x),64,9))
    feature_support=(np.isfinite(x).sum(2)>=24).all(1)&(combined['defined_draws']>=48)
    pair_support=(native[0]['jointly_defined_draws']>=24)&(native[1]['jointly_defined_draws']>=24)&(combined['jointly_defined_draws']>=48)
    return dict(native=native,combined=combined,feature_support=feature_support,pair_support=pair_support,
        full_slot_undefined_fraction=(~np.isfinite(x)).mean((1,2)))

def coverage017(student,teacher,uniforms):
    S,T,U=[np.asarray(x,np.float64) for x in (student,teacher,uniforms)]
    require(S.shape==T.shape and S.ndim==4 and S.shape[1:]==(2,32,9),'017 PIT scalar banks')
    require(U.shape==(len(S),2,32,6) and np.isfinite(U).all() and ((U>=0)&(U<1)).all(),'017 explicit PIT uniforms')
    sm,tm=scalar_banks017(S),scalar_banks017(T)
    rates=np.full((len(S),6,2),np.nan);hits=np.full((len(S),6,2),-1,np.int64)
    nS=np.isfinite(S.reshape(len(S),64,9)).sum(1)[:,COVERAGE017]
    nT=np.isfinite(T.reshape(len(T),64,9)).sum(1)[:,COVERAGE017]
    support=(sm['feature_support']&tm['feature_support'])[:,COVERAGE017]
    for i in range(len(S)):
        for f,j in enumerate(COVERAGE017):
            if not support[i,f]:continue
            s=np.sort(S[i,:,:,j][np.isfinite(S[i,:,:,j])]);t=T[i,:,:,j].reshape(-1);u=U[i,:,:,f].reshape(-1)
            defined=np.isfinite(t);t=t[defined];u=u[defined]
            lo=np.searchsorted(s,t,side='left');hi=np.searchsorted(s,t,side='right')
            pit=(lo+u*(hi-lo+1))/(len(s)+1)
            for level,(a,b) in enumerate(((.25,.75),(.05,.95))):
                hits[i,f,level]=int(((pit>=a)&(pit<=b)).sum());rates[i,f,level]=hits[i,f,level]/len(t)
    return dict(rates=rates,hits=hits,n_student_finite=nS,n_teacher_defined=nT,supported=support,
        unsupported_hits_sentinel=-1,unsupported_rate='NaN_with_support_reason',
        teacher_original_slots_per_parent=64)

def covariance_cell017(S,T,group,weights,*,moments=None):
    sm,tm=moments if moments is not None else (scalar_banks017(S),scalar_banks017(T))
    support=(sm['feature_support']&tm['feature_support']).all(1)&(sm['pair_support']&tm['pair_support']).all((1,2))
    cell=support_cell017(group,weights,support)
    if cell['status']=='pending':return cell
    ix=cell.pop('indices');w=weights[ix];a=w/w.sum()
    scale=np.maximum(np.sqrt(tm['combined']['sample_variance'][ix]),FLOORS017)
    denom=scale[:,:,None]*scale[:,None,:]
    S_norm=sm['combined']['sample_covariance'][ix]/denom;T_norm=tm['combined']['sample_covariance'][ix]/denom
    C_S=np.einsum('p,pij->ij',a,S_norm);C_T=np.einsum('p,pij->ij',a,T_norm)
    C0=np.einsum('p,pij->ij',a,tm['native'][0]['sample_covariance'][ix]/denom)
    C1=np.einsum('p,pij->ij',a,tm['native'][1]['sample_covariance'][ix]/denom)
    reference=float(np.linalg.norm(C_T));split_ref=float(np.linalg.norm(.5*(C0+C1)))
    if reference==0 or split_ref==0:return dict(cell,status='pending',reasons=['zero_covariance_reference'])
    error=float(np.linalg.norm(C_S-C_T)/reference);resolution=float(np.linalg.norm(C0-C1)/split_ref)
    status='pending' if resolution>.075 else 'pass' if error<=.15 else 'fail'
    parent_reference=np.linalg.norm(T_norm,axis=(1,2));parent_errors=np.full(len(ix),np.nan)
    np.divide(np.linalg.norm(S_norm-T_norm,axis=(1,2)),parent_reference,out=parent_errors,where=parent_reference>0)
    return dict(cell,status=status,reasons=['MC_unresolved'] if status=='pending' else [],
        group_relative_Frobenius=error,teacher32_split_resolution=resolution,
        student_normalized_conditional_covariance=C_S,teacher_normalized_conditional_covariance=C_T,
        parent_errors=parent_errors,parent_error_zero_reference=parent_reference==0,
        supported_parent_indices=ix,teacher_scale=scale,no_between_parent_scatter=True)

def timing_cell017(S,T,feature,group,weights,*,moments=None):
    require(feature in TIME017,'017 registered timing feature')
    sm,tm=moments if moments is not None else (scalar_banks017(S),scalar_banks017(T))
    support=(sm['feature_support']&tm['feature_support'])[:,feature]
    cell=support_cell017(group,weights,support)
    if cell['status']=='pending':return cell
    ix=cell.pop('indices');a=weights[ix]/weights[ix].sum()
    d=sm['combined']['mean'][ix,feature]-tm['combined']['mean'][ix,feature]
    r=tm['native'][0]['mean'][ix,feature]-tm['native'][1]['mean'][ix,feature]
    err=abs(float(a@d));rms=float(np.sqrt(a@(d*d)));ref=abs(float(a@r));ref_rms=float(np.sqrt(a@(r*r)))
    status='pending' if ref>.5 or ref_rms>1 else 'pass' if err<=1 and rms<=2 else 'fail'
    return dict(cell,status=status,reasons=['MC_unresolved'] if status=='pending' else [],
        aggregate_abs_error_ns=err,parent_mean_RMSE_ns=rms,
        teacher32_split_aggregate_abs_ns=ref,teacher32_split_parent_RMSE_ns=ref_rms,
        supported_parent_indices=ix,parent_mean_differences=d)

def energy_cell017(final,initial,reference,group,weights):
    final,initial,reference=[np.asarray(x,np.float64) for x in (final,initial,reference)]
    require(final.shape==initial.shape==reference.shape and final.ndim==1,'017 parent energy vectors')
    support=np.isfinite(final)&np.isfinite(initial)&np.isfinite(reference)
    require(support.all(),'017 nonfinite fullwave energy is engineering failure')
    cell=support_cell017(group,weights,support)
    if cell['status']=='pending':return cell
    ix=cell.pop('indices');w=weights[ix]
    d=weighted017(final[ix]-initial[ix],w);z=weighted017(final[ix]-reference[ix],w);r=weighted017(reference[ix],w)
    upper=d['mean']+1.959963984540054*d['SE'];allowed=max(.02*abs(r['mean']),3*z['SE'])
    return dict(cell,status='pass' if upper<0 and z['mean']<=allowed else 'fail',reasons=[],
        improvement=d,upper95=upper,excess_reference=z,reference=r,excess_allowed=allowed)

def scalar_features017(wave,baseline,peak_indices):
    y,b=np.asarray(wave,np.float64),np.asarray(baseline,np.float64)
    require(y.ndim==2 and y.shape[1]==256 and b.shape==(len(y),100),'017 hard and actual baseline shapes')
    m=morphology(y);q=y.sum(1);tail=y[:,46:].sum(1)
    psd=np.full(len(y),np.nan);np.divide(tail,q,out=psd,where=q!=0)
    rms=np.sqrt(np.mean((b-b.mean(1,keepdims=True))**2,1))
    return np.column_stack((q,tail,psd,b.mean(1),rms,m['peak_voltage_V'],m['rise10_90_ns'],m['FWHM_ns'],peak_indices)),m

def variance_curve_cell017(Q,a,b,group,weights):
    Q=np.asarray(Q,np.float64);group=np.asarray(group,bool);weights=np.asarray(weights,np.float64)
    require(np.isfinite(Q).all() and (Q>0).all(),'017 analytic variance actual Q')
    require(np.ndim(a)==np.ndim(b)==0 and np.isfinite(a) and np.isfinite(b) and a>0 and b>0,'017 analytic a,b finite positive scalars')
    with np.errstate(over='ignore',invalid='ignore'):
        v=a*Q+b*Q**2;t=.01*3.7103998046755806*Q+.0025*Q**2
    # This analytic max is over ALL original group parents. The empirical
    # 95% finite-support rule must never delete a model/reference failure.
    cell=support_cell017(group,weights,np.ones(len(Q),bool))
    invalid=group&(~np.isfinite(v)|~np.isfinite(t)|(v<=0)|(t<=0))
    if invalid.any():return dict(cell,status='pending',reasons=['reference_undefined'],invalid_parent_indices=np.flatnonzero(invalid),
        all_original_group_parents_required=True,invalid_parents_deleted=False)
    if cell['status']=='pending':return cell
    ix=cell.pop('indices');alpha=weights[ix]/weights[ix].sum()
    with np.errstate(over='ignore',invalid='ignore'):
        denominator=float(np.sum(alpha*t[ix]**2));numerator=float(np.sum(alpha*(v[ix]-t[ix])**2))
    if not np.isfinite(denominator) or denominator<=0:return dict(cell,status='pending',reasons=['reference_undefined'],
        all_original_group_parents_required=True,invalid_parents_deleted=False)
    require(np.isfinite(numerator),'017 nonfinite analytic variance discrepancy')
    relative_RMS=float(np.sqrt(numerator/denominator))
    relative_max=float(np.max(np.abs(v[ix]-t[ix])/t[ix]))
    return dict(cell,status='pass' if relative_RMS<=.10 and relative_max<=.20 else 'fail',reasons=[],
        relative_RMS=relative_RMS,relative_max=relative_max,supported_parent_indices=ix,
        formula='sqrt(sum alpha*(vS-vT)^2 / sum alpha*vT^2); max abs(vS-vT)/vT')

def charge_identifiability017(fit_Q,test_Q,a,b):
    """Secondary only: fixed dimensionless columns, no column L2 refitting."""
    qref=3.7103998046755806;designs={}
    for role,values in (('fit',fit_Q),('test',test_Q)):
        q=np.asarray(values,np.float64)
        require(q.ndim==1 and len(q)>1 and np.isfinite(q).all() and (q>0).all(),'017 fit/test Gram actual positive Q')
        x=q/qref;X=np.column_stack((x,x*x));gram=X.T@X/len(q)
        require(np.isfinite(gram).all(),'017 nonfinite fixed design Gram')
        eigenvalues=np.linalg.eigvalsh(gram);require(eigenvalues[-1]>0,'017 zero fixed design Gram')
        ratio=float(eigenvalues[0]/eigenvalues[-1]);condition=float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0]>0 else None
        designs[role]=dict(parents=len(q),columns=['Q/qref','(Q/qref)^2'],normalization='X.T@X/N; no per-column normalization',
            Gram=gram,eigenvalues=eigenvalues,eigenvalue_ratio=ratio,condition_number=condition,ratio_at_least1e_4=ratio>=1e-4)
    identifiable=all(v['ratio_at_least1e_4'] for v in designs.values());aT=.01*qref;bT=.0025
    aerr=abs(a-aT)/aT;berr=abs(b-bT)/bT
    return dict(a=a,b=b,a_relative_error=aerr,b_relative_error=berr,q_reference_Vns=qref,design_by_role=designs,
        individually_identifiable=identifiable,identifiability_rule='fit AND test fixed-scaled Gram ratio>=1e-4',
        secondary_25percent_pass=(aerr<=.25 and berr<=.25) if identifiable else None,
        secondary_only_not_primary_gate=True)

def pooled_wave017(mean0,var0,mean1,var1):
    m0,v0,m1,v1=[np.asarray(x,np.float64) for x in (mean0,var0,mean1,var1)]
    require(m0.shape==v0.shape==m1.shape==v1.shape and m0.ndim==2 and m0.shape[1]==256
        and all(np.isfinite(x).all() for x in (m0,v0,m1,v1)),'017 finite complete native32 waveform moments')
    mean=.5*(m0+m1)
    variance=(31*v0+31*v1+32*(m0-mean)**2+32*(m1-mean)**2)/63
    return mean,variance

def waveform_cells017(student_mean,student_variance,teacher_mean,teacher_variance,group,weights):
    S,V,T,W=[np.asarray(x,np.float64) for x in (student_mean,student_variance,teacher_mean,teacher_variance)]
    require(S.shape==V.shape==T.shape==W.shape and S.ndim==2 and S.shape[1]==256
        and all(np.isfinite(x).all() for x in (S,V,T,W)),'017 nonfinite fullwave moments are engineering failure')
    cell=support_cell017(group,weights,np.ones(len(S),bool))
    if cell['status']=='pending':return dict(mean=cell.copy(),variance=cell.copy())
    ix=cell.pop('indices');alpha=weights[ix]/weights[ix].sum();trace_T=np.sum(W[ix],axis=1)
    denominator=float(alpha@trace_T)
    if denominator<=0:return dict(mean=dict(cell,status='pending',reasons=['zero_teacher_trace']),variance=dict(cell,status='pending',reasons=['zero_teacher_trace']))
    mean_error=float(np.sqrt(np.sum(alpha*np.sum((S[ix]-T[ix])**2,axis=1))/denominator))
    ratio=float(np.sum(alpha*np.sum(V[ix],axis=1))/denominator)
    return dict(mean=dict(cell,status='pass' if mean_error<=.30 else 'fail',reasons=[],normalized_mean_RMS=mean_error),
        variance=dict(cell,status='pass' if abs(ratio-1)<=.10 else 'fail',reasons=[],pooled_conditional_trace_ratio=ratio,
            no_between_parent_scatter=True))

def coverage_cells017(coverage,group,weights):
    cells=[]
    for f,name in enumerate(np.asarray(FEATURES017)[list(COVERAGE017)]):
        for level,target,tolerance in ((0,.5,.04),(1,.9,.03)):
            cell=support_cell017(group,weights,coverage['supported'][:,f])
            if cell['status']!='pending':
                ix=cell.pop('indices');summary=weighted017(coverage['rates'][ix,f,level],weights[ix])
                cell.update(status='pass' if abs(summary['mean']-target)<=tolerance else 'fail',reasons=[],
                    coverage=summary,target=target,tolerance=tolerance,supported_parent_indices=ix)
            cells.append(dict(cell,feature=str(name),level=target))
    return cells

def missingness_cells017(sm,tm,student_flags,teacher_flags,group,weights):
    require(student_flags.keys()==teacher_flags.keys(),'017 retained flag schemas differ')
    require(set(REQUIRED_MISSING_FLAGS017)<=set(student_flags),'017 required undefined/censor causes missing')
    cell=support_cell017(group,weights,np.ones(len(group),bool))
    values={**{'undefined_'+name:(sm['full_slot_undefined_fraction'][:,j],tm['full_slot_undefined_fraction'][:,j])
        for j,name in enumerate(FEATURES017)},**{name:(np.asarray(student_flags[name],np.float64),np.asarray(teacher_flags[name],np.float64)) for name in student_flags}}
    result=[]
    for name,(s,t) in values.items():
        required=name in REQUIRED_MISSING017;item=cell.copy()
        if item['status']!='pending':
            ix=item.pop('indices');require(s.shape==t.shape==group.shape and np.isfinite(s).all() and np.isfinite(t).all(),'017 full-slot flag fractions')
            a=weights[ix]/weights[ix].sum();S=float(a@s[ix]);T=float(a@t[ix])
            item.update(status=('pass' if abs(S-T)<=.01 else 'fail') if required else 'diagnostic',reasons=[],student_fraction=S,teacher_fraction=T,
                absolute_difference=abs(S-T),original_draw_slots_per_parent=64)
        if not required:item.update(status='diagnostic',support_status=cell['status'],scientific_threshold=None)
        result.append(dict(item,feature=name,required_for_gate=required))
    return result

def stage_gate017(trials):
    require(len(trials)==2 and [x['trial'] for x in trials]==['T1_paired_seed0','T1_paired_seed1'],'017 stage1 exact trial identity')
    groups=('all_heldout','fit_Q_tertile0_heldout','fit_Q_tertile1_heldout','fit_Q_tertile2_heldout')
    expected=set()
    for g in groups:
        for w in ('nominal','low','high'):
            expected.add(('analytic_charge_variance',g,w,None,None,None))
            for h in ('fast','slow','common_level'):
                for metric in ('hard_wave_mean','hard_wave_variance','conditional_cov9','conditional_energy',
                    'time_peak_index','time_rise10_90','time_FWHM'):
                    expected.add((metric,g,w,h,None,None))
                for f in np.asarray(FEATURES017)[list(COVERAGE017)]:
                    for level in (.5,.9):expected.add(('PIT_coverage',g,w,h,str(f),level))
    for trial in trials:
        primary=[(c['metric'],c['group'],c['weights'],c.get('hypothesis'),c.get('feature'),c.get('level'))
            for c in trial.get('cells',[]) if c['metric']!='missingness_or_censor']
        require(len(primary)==len(expected) and set(primary)==expected,'017 stage1 missing/duplicate required primary cells')
        missing=[c for c in trial['cells'] if c['metric']=='missingness_or_censor']
        for g in groups:
            for w in ('nominal','low','high'):
                for h in ('fast','slow','common_level'):
                    features=[c['feature'] for c in missing if (c['group'],c['weights'],c.get('hypothesis'))==(g,w,h)]
                    require(len(features)==len(set(features)) and set(features)==set(REQUIRED_MISSING017),
                        '017 missing/extra mandatory full-slot undefined/censor cause gates')
        require(all(c.get('status') in ('pass','pending','fail') for c in trial['cells']),'017 unresolved gate status')
    bad=[dict(trial=x['trial'],cell=i,status=c['status'],metric=c['metric'],group=c['group'],weights=c['weights'],
        hypothesis=c.get('hypothesis'),reasons=c.get('reasons',[])) for x in trials for i,c in enumerate(x['cells']) if c['status']!='pass']
    return dict(status='pass' if not bad else 'closed',stage2_allowed=not bad,
        required_trials=[x['trial'] for x in trials],required_cell_count=sum(len(x['cells']) for x in trials),
        nonpass=bad,pending_is_pass=False,scientific_acceptance=False,
        scope='fixed known A/005/M0 shape conditional charge recovery; not hardware or REAL response acceptance')
