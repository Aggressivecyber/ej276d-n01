"""Prespecified descriptive007 endpoints, never checkpoint/hypothesis selection."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import charge_np,load_training,load_validation,sha256,write_json
from .evaluate import _weighted_auc
from .evaluate_integration006 import (Moments,array_sha,charge_bins,clean,extended_quantile,
    fixed_bin_support,readouts,scalar_summary,sd_ratio)
from .losses_v2 import N01LossV2
from .model import ADC_MIN,ADC_MAX,charge,digitize
from .noise_validation import balanced_real_domains
from .response007 import (ARMS,HYPOTHESES,PairedStreams,SignalResponse007,candidates,
    fixed_noises,make_extension,state_digest,loss_tensor_statistics)
from .train import atomic_torch_save,utc

ROOT=Path(__file__).resolve().parents[2]
VIEWS=('signal_only',)+HYPOTHESES


def save_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve endpoint artifact: '+str(path))
    with path.with_suffix(path.suffix+'.tmp').open('xb') as f:np.savez_compressed(f,**values)
    path.with_suffix(path.suffix+'.tmp').replace(path)


def summary_readouts(obs):
    return {k:(dict(count=int(v.sum()),rows=len(v)) if v.dtype.kind=='b' else scalar_summary(v)) for k,v in obs.items()}


def conditional_rows(obs,bins,parents=None):
    rows=[]
    for b in range(8):
        mask=bins==b
        rows.append(dict(bin=b,rows=int(mask.sum()),parents=None if parents is None else len(np.unique(parents[mask])),
            Q=scalar_summary(obs['Q'][mask]),PSD=scalar_summary(obs['PSD'][mask])))
    return rows


@torch.no_grad()
def generate_split(split,data,models,extension,stats,cfg,out,matched_rows,device):
    ev=cfg['evaluation'];draws=ev['draws_per_parent'];chunk=ev['chunk_rows'];n=len(data['source'])
    parents=np.repeat(np.arange(n),draws);draw=np.tile(np.arange(draws),n)
    source_bins=charge_bins(charge_np(data['source']),stats['loss'])
    seeds=cfg['seeds'];streams=PairedStreams(seeds[split+'_response'],seeds[split+'_noise005'],seeds[split+'_extension'])
    stream_initial=streams.state_dict()
    pieces={a:{v:[] for v in VIEWS} for a in ARMS};signal_q={a:[] for a in ARMS}
    moments={a:{v:Moments((480,)) for v in VIEWS} for a in ARMS}
    bin_moments={a:{v:[Moments((480,)) for _ in range(8)] for v in VIEWS} for a in ARMS}
    matched={a:{h:np.empty((len(matched_rows),480),np.float64) for h in HYPOTHESES} for a in ARMS}
    selection_position={int(row):i for i,row in enumerate(matched_rows)}
    written=np.zeros(len(matched_rows),bool);samples={};chunks=[]
    engineering={a:dict(signal_prefix_absolute_max_V=0.,charge_identity_absolute_max=0.,
        nonpositive_Q={h:0 for h in HYPOTHESES},guarded_PSD={h:0 for h in HYPOTHESES},prefix_same=True) for a in ARMS}
    for start in range(0,len(parents),chunk):
        end=min(start+chunk,len(parents));pr=parents[start:end]
        response,nz,epsilon=streams.draw(end-start,device)
        noises=fixed_noises(extension,nz,epsilon)
        if start==0:
            for h in HYPOTHESES:samples['noise_'+h]=noises[h].cpu().numpy()[:ev['sample_waveform_rows']]
        source=torch.as_tensor(data['source'][pr],dtype=torch.float32,device=device)
        this=dict(start=start,end=end,response_sha256={k:array_sha(v) for k,v in response.items() if k in ('phys','local_white')},
            noise_latents_sha256={k:array_sha(v) for k,v in nz.items()},epsilon_sha256=array_sha(epsilon),outputs={})
        selected=[(j,selection_position[int(parent)]) for j,parent in enumerate(pr) if draw[start+j]==0 and int(parent) in selection_position]
        for arm,model in models.items():
            output=model.signal_forward(source,response);signal=output['signal_pre'];sq=charge(signal).cpu().numpy()
            fixed_bins=charge_bins(sq,stats['loss']);signal_q[arm].append(sq)
            waves=candidates(signal,noises,ste=False);waves['signal_only']=digitize(signal,ste=False)
            en=engineering[arm]
            en['signal_prefix_absolute_max_V']=max(en['signal_prefix_absolute_max_V'],float(signal[:,:100].abs().max()))
            for h in HYPOTHESES:
                pre=signal+noises[h];res=float((charge(pre)-charge(signal)-charge(noises[h])).abs().max())
                en['charge_identity_absolute_max']=max(en['charge_identity_absolute_max'],res)
                en['prefix_same'] &= torch.equal(waves[h][:,:100],waves['fast'][:,:100])
            this['outputs'][arm]={}
            for view in VIEWS:
                y=waves[view].cpu().numpy();pre=signal if view=='signal_only' else signal+noises[view]
                obs=readouts(y,stats['loss']['q_floor'],pre_adc=pre.cpu().numpy())
                pieces[arm][view].append(obs);moments[arm][view].add(y)
                for b in range(8):bin_moments[arm][view][b].add(y[fixed_bins==b])
                this['outputs'][arm][view]=array_sha(y)
                if view in HYPOTHESES:
                    en['nonpositive_Q'][view]+=int(obs['Q_nonpositive'].sum());en['guarded_PSD'][view]+=int(obs['PSD_guarded'].sum())
                    for local,destination in selected:matched[arm][view][destination]=y[local]
                if start==0:
                    samples[arm+'_'+view]=y[:ev['sample_waveform_rows']]
            if start==0:samples[arm+'_signal_pre']=signal.cpu().numpy()[:ev['sample_waveform_rows']]
        for _,destination in selected:written[destination]=True
        chunks.append(this)
    if not written.all():raise RuntimeError('Matched SOURCE population incomplete')
    allobs={};curves={};summary={}
    for arm in ARMS:
        allobs[arm]={v:{k:np.concatenate([piece[k] for piece in pieces[arm][v]]) for k in pieces[arm][v][0]} for v in VIEWS}
        allobs[arm]['signal_Q']=np.concatenate(signal_q[arm]);allobs[arm]['fixed_bins']=charge_bins(allobs[arm]['signal_Q'],stats['loss'])
        summary[arm]={}
        for view in VIEWS:
            ob=allobs[arm][view];mean,var=moments[arm][view].values()
            curves[arm+'_'+view+'_mean']=mean;curves[arm+'_'+view+'_variance']=var
            curves[arm+'_'+view+'_fixed_bin_mean']=np.array([m.values()[0] for m in bin_moments[arm][view]])
            curves[arm+'_'+view+'_fixed_bin_variance']=np.array([m.values()[1] for m in bin_moments[arm][view]])
            output_bins=charge_bins(ob['Q'],stats['loss'])
            matrix=np.bincount(allobs[arm]['fixed_bins']*8+output_bins,minlength=64).reshape(8,8)
            summary[arm][view]=dict(population=summary_readouts(ob),
                fixed_signal_Q_bins=conditional_rows(ob,allobs[arm]['fixed_bins'],parents),
                original_SOURCE_Q_bins=conditional_rows(ob,np.repeat(source_bins,draws),parents),
                output_Q_bins=conditional_rows(ob,output_bins,parents),fixed_to_output_migration=matrix)
    real_obs=readouts(data['real'],stats['loss']['q_floor'])
    curves['REAL_mean']=data['real'].mean(0);curves['REAL_variance']=data['real'].var(0)
    summary['REAL']=dict(population=summary_readouts(real_obs),output_Q_bins=conditional_rows(real_obs,charge_bins(real_obs['Q'],stats['loss'])))
    flat=dict(source_parent_ids=data['source_ids'],source_parent_row=parents,draw=draw,source_Q=charge_np(data['source']),
        source_h=data['h'],source_bins=source_bins,real_original_ids=data['real_ids'],real_original_groups=data['real_groups'])
    for k,v in real_obs.items():flat['REAL_'+k]=v
    for arm in ARMS:
        flat[arm+'_signal_Q']=allobs[arm]['signal_Q'];flat[arm+'_fixed_bins']=allobs[arm]['fixed_bins']
        for view in VIEWS:
            for k,v in allobs[arm][view].items():flat[arm+'_'+view+'_'+k]=v
    samples['source_parent_ids']=data['source_ids'][parents[:min(ev['sample_waveform_rows'],chunk)]]
    save_npz(out/(split+'_observables.npz'),**flat);save_npz(out/(split+'_curves.npz'),**curves)
    save_npz(out/(split+'_sample_waveforms.npz'),**samples)
    atomic_torch_save(dict(initial=stream_initial,final=streams.state_dict(),chunk_rows=chunk),out/(split+'_streams.pt'))
    write_json(out/(split+'_generation.json'),clean(dict(parent_count=n,draws_per_parent=draws,rows=len(parents),
        chunks=chunks,engineering=engineering,summary=summary,matched_source_rows=matched_rows,
        extra_sample_noise_replay_rows=0,new_unique_random_draws_in_sample_replay=0)))
    if any(e['signal_prefix_absolute_max_V']>cfg['engineering']['signal_prefix_max_V'] or not e['prefix_same']
            or sum(e['nonpositive_Q'].values()) or sum(e['guarded_PSD'].values()) for e in engineering.values()):
        raise RuntimeError('Endpoint interface failed; split outputs preserved and later evaluation stopped')
    return dict(observables=allobs,real=real_obs,parents=parents,summary=summary,matched=matched,engineering=engineering)


def paired_width_bootstrap(generated,data,stats,cfg,split):
    parents=generated['parents'];n=len(data['source']);draws=cfg['evaluation']['draws_per_parent']
    records={};support={};real_scales={k:generated['real'][k].std(ddof=0) for k in ['Q','PSD']}
    for arm in ARMS:
        ob=generated['observables'][arm];bins=ob['fixed_bins'];psd=np.array([ob[h]['PSD'] for h in HYPOTHESES])
        rows,counts,sufficient=fixed_bin_support(bins,parents,cfg['evaluation']['minimum_parents_per_fixed_bin'],cfg['evaluation']['minimum_rows_per_fixed_bin'])
        count=np.zeros((8,n));total=np.zeros((8,3,n));square=np.zeros_like(total)
        for b in range(8):
            mask=bins==b;count[b]=np.bincount(parents[mask],minlength=n)
            for h in range(3):
                center=psd[h,np.flatnonzero(mask)[0]] if mask.any() else 0.
                delta=psd[h,mask]-center
                total[b,h]=np.bincount(parents[mask],weights=delta,minlength=n)
                square[b,h]=np.bincount(parents[mask],weights=delta**2,minlength=n)
        records[arm]=dict(count=count,total=total,square=square,
            Q=np.array([ob[h]['Q'].reshape(n,draws).mean(1) for h in HYPOTHESES]),
            PSD=psd.reshape(3,n,draws).mean(2))
        support[arm]=dict(rows=rows,parents=counts,supported=sufficient)
    def statistic(rec,w):
        result=[]
        for key in ['Q','PSD']:
            means=(rec[key]@w)/w.sum();result.append(np.ptp(means)/real_scales[key])
        counts=rec['count']@w
        sums=rec['total']@w;squares=rec['square']@w
        for b in range(8):
            if counts[b]<=0:result.append(np.nan);continue
            variance=np.maximum(squares[b]/counts[b]-(sums[b]/counts[b])**2,0.)
            result.append(sd_ratio(np.sqrt(variance)))
        return result
    point=np.array([statistic(records[arm],np.ones(n)) for arm in ARMS])
    rng=np.random.default_rng(cfg['seeds']['bootstrap_'+split]);samples=[]
    for _ in range(cfg['evaluation']['bootstrap_draws']):
        w=np.bincount(rng.integers(0,n,n),minlength=n)
        samples.append([statistic(records[arm],w) for arm in ARMS])
    samples=np.asarray(samples);interval=extended_quantile(samples.reshape(len(samples),-1),[.025,.975]).reshape(2,10,2)
    report=dict(arms=ARMS,statistic_names=['Q_mean_range_over_REAL_sd','PSD_mean_range_over_REAL_sd']+[f'fixed_bin_{b}_PSD_SD_ratio' for b in range(8)],
        point=point,percentile_95_interval=interval,support=support,bootstrap_draws=len(samples),
        method='SOURCE_parent_paired_across_hypotheses_and_arms',same_acquisition_only=True,
        note='Arm-specific fixed signal bins can contain different parents; ratios are sensitivity measures, not arm superiority.',scientific_pass_threshold=None)
    return report,samples


@torch.no_grad()
def distance_receipt(objective,fake,real,context):
    device=next(objective.buffers()).device
    parts=objective.components(dict(waveform=torch.as_tensor(fake,dtype=torch.float64,device=device)),
        torch.as_tensor(real,dtype=torch.float64,device=device),stage='joint_stochastic',projection_context=context)
    return dict(components={k:float(parts[k]) for k in ('wave','QPSD','noise')},
        wave_subcomponents={k:float(v) for k,v in objective.last_wave_components.items()},
        conditional_diagnostics=objective.last_diagnostics,matched_rows=[len(fake),len(real)],
        projection_context=context.metadata(),scientific_pass_threshold=None)


class EndpointClassifier(nn.Module):
    """Fixed independent CNN, supports all three declared views."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(1,8,9,stride=2,padding=4),nn.SiLU(),
            nn.Conv1d(8,16,9,stride=2,padding=4),nn.SiLU(),nn.Conv1d(16,32,9,stride=2,padding=4),nn.SiLU(),
            nn.AdaptiveAvgPool1d(8),nn.Flatten(),nn.Linear(256,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x[:,None])[:,0]


def classifier_view(x,view):
    return x if view=='full480' else x[:,:100] if view=='prefix100' else x[:,100:480] if view=='pulse380' else None


def auc_uncertainty(labels,scores,g0,g1,seed,resamples,shared):
    rng=np.random.default_rng(seed);values=[]
    if shared:
        unique,index=np.unique(np.r_[g0,g1],return_inverse=True)
        for _ in range(resamples):
            w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[index]
            values.append(_weighted_auc(labels,scores,w))
    else:
        u0,i0=np.unique(g0,return_inverse=True);u1,i1=np.unique(g1,return_inverse=True)
        for _ in range(resamples):
            w0=np.bincount(rng.integers(0,len(u0),len(u0)),minlength=len(u0))[i0]
            w1=np.bincount(rng.integers(0,len(u1),len(u1)),minlength=len(u1))[i1]
            values.append(_weighted_auc(labels,scores,np.r_[w0,w1]))
    return dict(method='joint_REAL_group_across_labels' if shared else 'separate_SOURCE_parent_and_REAL_group',
        resamples=resamples,percentile_95_interval=np.quantile(values,[.025,.975]),
        groups0=len(np.unique(g0)),groups1=len(np.unique(g1)),independent_acquisition_uncertainty=False)


@torch.enable_grad()
def fit_classifier(train0,train1,test0,test1,g0,g1,cfg,out,device,shared=False):
    seed=cfg['seeds']['classifier'];ev=cfg['evaluation'];torch.manual_seed(seed)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'PROGRESS.json',dict(completed_updates=0,UTC=utc()))
    x=np.concatenate([train0,train1]).astype(np.float32)
    center=x.mean(0,dtype=np.float64).astype(np.float32);scale=np.maximum(x.std(0,dtype=np.float64),1/16383).astype(np.float32)
    tx=torch.as_tensor((np.concatenate([test0,test1]).astype(np.float32)-center)/scale,device=device)
    x=torch.as_tensor((x-center)/scale,device=device);y=torch.cat((torch.zeros(len(train0)),torch.ones(len(train1)))).to(device)
    model=EndpointClassifier().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=ev['classifier_lr'])
    rng=np.random.default_rng(seed+1);losses=[]
    try:
        for _ in range(ev['updates_per_classifier']):
            rows=np.r_[rng.choice(len(train0),ev['classifier_batch_per_class'],replace=False),
                len(train0)+rng.choice(len(train1),ev['classifier_batch_per_class'],replace=False)]
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(x[rows]),y[rows])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite C2ST loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
            write_json(out/'PROGRESS.json',dict(completed_updates=len(losses),UTC=utc()))
    except BaseException:
        atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=len(losses)),out/'FAILED_CLASSIFIER.pt')
        raise
    model.eval()
    with torch.no_grad():scores=model(tx).cpu().numpy()
    labels=np.r_[np.zeros(len(test0)),np.ones(len(test1))]
    uncertainty=auc_uncertainty(labels,scores,g0,g1,cfg['seeds']['classifier_bootstrap'],ev['bootstrap_draws'],shared)
    result=dict(AUC=_weighted_auc(labels,scores),uncertainty=uncertainty,fit_counts=[len(train0),len(train1)],
        test_counts=[len(test0),len(test1)],updates=ev['updates_per_classifier'],seed=seed,orientation_flipped=False,
        normalization_fit='this_classifier_TRAIN_only',training_discriminator_used=False,scientific_pass_threshold=None)
    atomic_torch_save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},center=center,scale=scale,
        config=ev,seed=seed,updates=ev['updates_per_classifier']),out/'classifier.pt')
    save_npz(out/'predictions.npz',labels=labels,scores=scores,groups0=g0,groups1=g1,training_loss=np.array(losses))
    write_json(out/'RESULT.json',clean(result))
    return result


@torch.no_grad()
def repeats(data,models,extension,stats,cfg,out,device):
    ev=cfg['evaluation'];seeds=cfg['seeds'];rows=np.random.default_rng(seeds['repeat_selection']).permutation(len(data['source']))[:ev['repeat_parents']]
    streams=PairedStreams(seeds['repeat_response'],seeds['repeat_noise005'],seeds['repeat_extension'])
    initial=streams.state_dict();values={};covariances={a:{v:np.zeros((480,480)) for v in VIEWS} for a in ARMS}
    checks={a:dict(signal_prefix_absolute_max_V=0.,nonpositive_Q=0,guarded_PSD=0,saturated_samples=0) for a in ARMS}
    for ai,arm in enumerate(ARMS):
        for view in VIEWS:
            for key in ['Q','PSD','half_rise_ns','peak_time_ns']:
                values[arm+'_'+view+'_'+key]=np.empty((len(rows),ev['repeat_draws']))
            values[arm+'_'+view+'_mean_waveform']=np.empty((len(rows),480))
            values[arm+'_'+view+'_variance_waveform']=np.empty((len(rows),480))
    for i,row in enumerate(rows):
        response,nz,epsilon=streams.draw(ev['repeat_draws'],device);noises=fixed_noises(extension,nz,epsilon)
        source=torch.as_tensor(np.repeat(data['source'][row:row+1],ev['repeat_draws'],axis=0),device=device,dtype=torch.float32)
        for arm,model in models.items():
            signal=model.signal_forward(source,response)['signal_pre'];waves=candidates(signal,noises,ste=False)
            checks[arm]['signal_prefix_absolute_max_V']=max(checks[arm]['signal_prefix_absolute_max_V'],float(signal[:,:100].abs().max()))
            waves['signal_only']=digitize(signal,ste=False)
            for view,w in waves.items():
                y=w.cpu().numpy();pre=signal if view=='signal_only' else signal+noises[view]
                obs=readouts(y,stats['loss']['q_floor'],pre_adc=pre.cpu().numpy())
                if view in HYPOTHESES:
                    checks[arm]['nonpositive_Q']+=int(obs['Q_nonpositive'].sum());checks[arm]['guarded_PSD']+=int(obs['PSD_guarded'].sum())
                    checks[arm]['saturated_samples']+=int(obs['saturation_samples'].sum())
                for key in ['Q','PSD','half_rise_ns','peak_time_ns']:values[arm+'_'+view+'_'+key][i]=obs[key]
                mean=y.mean(0);center=y-mean
                values[arm+'_'+view+'_mean_waveform'][i]=mean;values[arm+'_'+view+'_variance_waveform'][i]=y.var(0)
                covariances[arm][view]+=center.T@center/ev['repeat_draws']/len(rows)
    report={}
    for arm in ARMS:
        report[arm]={}
        for view in VIEWS:
            q=values[arm+'_'+view+'_Q'];p=values[arm+'_'+view+'_PSD']
            values[arm+'_'+view+'_VarQ']=q.var(1);values[arm+'_'+view+'_VarPSD']=p.var(1)
            values[arm+'_'+view+'_CovQPSD']=((q-q.mean(1,keepdims=True))*(p-p.mean(1,keepdims=True))).mean(1)
            values[arm+'_'+view+'_mean_conditional_covariance']=covariances[arm][view]
            report[arm][view]={key:scalar_summary(values[arm+'_'+view+'_'+key]) for key in ['VarQ','VarPSD','CovQPSD']}
    values['source_parent_ids']=data['source_ids'][rows];values['source_rows']=rows
    save_npz(out/'repeated_parents.npz',**values)
    atomic_torch_save(dict(initial=initial,final=streams.state_dict()),out/'repeated_streams.pt')
    write_json(out/'REPEATED_SUMMARY.json',clean(dict(parents=len(rows),draws=ev['repeat_draws'],summary=report,engineering=checks,
        real_repeated_parent_ground_truth_available=False,conditional_distribution_uniquely_identified=False)))
    if any(e['signal_prefix_absolute_max_V']>cfg['engineering']['signal_prefix_max_V'] or e['nonpositive_Q'] or e['guarded_PSD'] for e in checks.values()):
        raise RuntimeError('Repeated-parent interface failed; evidence preserved')
    return dict(summary=report,engineering=checks)


@torch.no_grad()
def intervention(data,models,extension,stats,cfg,out,device):
    q=charge_np(data['source']);ordered=np.argsort(q,kind='stable');pairs=ordered[:len(ordered)//2*2].reshape(-1,2)
    gap=np.abs(np.log(q[pairs[:,0]]/q[pairs[:,1]]));pairs=pairs[np.argsort(gap,kind='stable')[:cfg['evaluation']['intervention_pairs']]]
    rows=pairs.flatten();partner=pairs[:,::-1].flatten();draws=cfg['evaluation']['intervention_draws']
    centered=data['source'].astype(np.float64)-data['source'][:,:100].mean(1,keepdims=True)
    swapped=centered[partner]/charge_np(centered[partner])[:,None]*q[rows,None]
    if not np.allclose(charge_np(swapped),q[rows],rtol=2e-14,atol=2e-14):raise RuntimeError('Intervention changed source Q')
    seeds=cfg['seeds'];streams=PairedStreams(seeds['intervention_response'],seeds['intervention_noise005'],seeds['intervention_extension'])
    initial=streams.state_dict();response,nz,epsilon=streams.draw(len(rows)*draws,device);noises=fixed_noises(extension,nz,epsilon)
    source=torch.as_tensor(np.repeat(data['source'][rows],draws,axis=0),dtype=torch.float32,device=device)
    changed=torch.as_tensor(np.repeat(swapped,draws,axis=0),dtype=torch.float32,device=device)
    values=dict(source_parent_ids=data['source_ids'][rows],shape_partner_ids=data['source_ids'][partner],
        source_Q=q[rows],partner_Q=q[partner],relative_Q_gap=np.abs(q[rows]/q[partner]-1),
        float32_input_charge_roundoff=charge_np(swapped.astype(np.float32))-q[rows])
    report={}
    # A fixed reverse permutation changes storage order only; explicit z follows
    # each event. No ID enters either call.
    permutation=torch.arange(len(source)-1,-1,-1,device=device);restore=torch.argsort(permutation)
    for arm,model in models.items():
        original=model.signal_forward(source,response)['signal_pre'];altered=model.signal_forward(changed,response)['signal_pre']
        permuted=model.signal_forward(source[permutation],{k:v[permutation] for k,v in response.items()})['signal_pre'][restore]
        error=float((original-permuted).abs().max())
        tolerance=64*np.finfo(np.float32).eps*max(float(original.abs().max()),float(permuted.abs().max()),1/16383)
        if error>tolerance:raise RuntimeError('Permutation equivariance failed beyond float32 arithmetic bound')
        report[arm]=dict(permutation_maximum_absolute_error_V=error,hypotheses={},
            permutation_float32_absolute_tolerance_V=tolerance,
            signal_prefix_absolute_max_V=max(float(original[:,:100].abs().max()),float(altered[:,:100].abs().max())),
            nonpositive_Q=0,guarded_PSD=0)
        for h in HYPOTHESES:
            a=readouts(digitize(original+noises[h],ste=False).cpu().numpy(),stats['loss']['q_floor'])
            b=readouts(digitize(altered+noises[h],ste=False).cpu().numpy(),stats['loss']['q_floor'])
            report[arm]['nonpositive_Q']+=int(a['Q_nonpositive'].sum()+b['Q_nonpositive'].sum())
            report[arm]['guarded_PSD']+=int(a['PSD_guarded'].sum()+b['PSD_guarded'].sum())
            report[arm]['hypotheses'][h]={}
            for key in ['Q','PSD','tail','half_rise_ns','peak_voltage']:
                delta=(b[key]-a[key]).reshape(len(rows),draws)
                values[arm+'_'+h+'_delta_'+key]=delta
                report[arm]['hypotheses'][h][key]=dict(paired_parent_mean_delta=scalar_summary(delta.mean(1)),
                    mean_absolute_parent_delta=float(np.mean(np.abs(delta.mean(1)))))
    save_npz(out/'input_intervention.npz',**values)
    atomic_torch_save(dict(initial=initial,final=streams.state_dict()),out/'intervention_streams.pt')
    write_json(out/'INPUT_DEPENDENCY.json',clean(dict(pairs=pairs,draws=draws,summary=report,
        fixed_z_and_electronics=True,identities_never_neural_inputs=True,
        no_physical_source_preservation_acceptance_threshold=True)))
    if any(e['signal_prefix_absolute_max_V']>cfg['engineering']['signal_prefix_max_V'] or e['nonpositive_Q'] or e['guarded_PSD'] for e in report.values()):
        raise RuntimeError('Input-intervention interface failed; evidence preserved')
    return report


def population_sensitivity(generated,data,cfg):
    q=charge_np(data['source']);draws=cfg['evaluation']['draws_per_parent']
    cases=dict(original_count=np.ones(len(q)),SOURCE_Q_tilt=np.where(q>=np.median(q),1.5,.5),composition_h_tilt=.5+data['h'])
    report={}
    for name,weights in cases.items():
        w=np.repeat(weights,draws);w=w/w.sum();report[name]={}
        for arm in ARMS:
            report[name][arm]={}
            for h in HYPOTHESES:
                ob=generated['observables'][arm][h]
                terms={}
                for key in ['Q','PSD']:
                    mean=float(np.sum(w*ob[key]));variance=float(np.sum(w*(ob[key]-mean)**2))
                    terms[key]=dict(mean=mean,SD=variance**.5)
                report[name][arm][h]=terms
    return dict(cases=report,response_parameters_frozen=True,additional_generation_rows=0,
        physical_population_weights_known=False,cross_acquisition_generalization_established=False)


def evaluate_run(run,cfg,container,identities,device):
    run=Path(run);out=run/'evaluation';out.mkdir(parents=True,exist_ok=False)
    if sha256(container)!=cfg['train_container_sha256'] or sha256(identities)!=cfg['identities_sha256']:raise ValueError('Evaluation inputs changed')
    data={'TRAIN':load_training(container,identities),'DEV':load_validation(container,identities)}
    models={};frozen_hashes={};states={}
    for arm in ARMS:
        path=run/'train'/arm/'final.pt';value=torch.load(path,map_location='cpu',weights_only=False)
        if value['G_updates']!=19000 or value['D_updates']!={h:16000 for h in HYPOTHESES} or value['config']!=cfg:
            raise ValueError('Wrong fixed response endpoint')
        model=SignalResponse007(value['statistics']['model'],arm);model.load_state_dict(value['model'],strict=True)
        model.set_stage('low_learning_rate');model.requires_grad_(False);model.eval();model.to(device)
        models[arm]=model;frozen_hashes[arm]=state_digest(model.state_dict());states[arm]=sha256(path)
        stats=value['statistics']
    noise=torch.load(ROOT/cfg['references']['noise005']['path'],map_location='cpu',weights_only=False)
    extension=make_extension(noise,device);selections={};results={};generated={}
    objective=N01LossV2(loss_tensor_statistics(stats['loss']),seed=cfg['seeds']['train_projection'],evaluation_seed=cfg['seeds']['evaluation_projection']).to(device)
    context=objective.create_evaluation_context();distance={}
    for split in ['TRAIN','DEV']:
        n=cfg['evaluation']['matched_'+split+'_per_class'];d=data[split]
        selected=np.random.default_rng(cfg['seeds']['selected_'+split]).permutation(len(d['source']))[:n]
        a,b=balanced_real_domains(d['real_groups'],n,cfg['seeds']['real_'+split])
        selections[split]=dict(source=selected,REAL_A=a,REAL_B=b)
        generated[split]=generate_split(split,d,models,extension,stats,cfg,out,selected,device)
        boot,draws=paired_width_bootstrap(generated[split],d,stats,cfg,split)
        save_npz(out/(split+'_bootstrap.npz'),statistics=draws)
        write_json(out/(split+'_SENSITIVITY.json'),clean(boot))
        distance[split]={arm:{h:distance_receipt(objective,generated[split]['matched'][arm][h],d['real'][a],context) for h in HYPOTHESES} for arm in ARMS}
        distance[split]['REAL_REAL']=distance_receipt(objective,d['real'][b],d['real'][a],context)
        results[split]=dict(sensitivity=boot,engineering=generated[split]['engineering'])
    write_json(out/'MATCHED_SELECTIONS.json',clean(dict(selection=selections,REAL_rule='balanced_equal_quota_per_original_group',
        original_population_statistics_saved_separately=True,one_SOURCE_draw_per_parent=True,TRAIN_DEV_roles_preserved=True)))
    write_json(out/'DISTRIBUTION_DISTANCES.json',clean(distance))
    matched_payload={}
    for split in ['TRAIN','DEV']:
        for arm in ARMS:
            for h in HYPOTHESES:matched_payload[split+'_'+arm+'_'+h]=generated[split]['matched'][arm][h]
    save_npz(out/'matched_generated_waveforms.npz',**matched_payload)
    c2st={};updates=0
    for view in cfg['evaluation']['classifier_views']:
        c2st[view]={}
        tr,va=data['TRAIN'],data['DEV'];st,sv=selections['TRAIN'],selections['DEV']
        for arm in ARMS:
            c2st[view][arm]={}
            for h in HYPOTHESES:
                result=fit_classifier(classifier_view(generated['TRAIN']['matched'][arm][h],view),classifier_view(tr['real'][st['REAL_A']],view),
                    classifier_view(generated['DEV']['matched'][arm][h],view),classifier_view(va['real'][sv['REAL_A']],view),
                    va['source_ids'][sv['source']],va['real_groups'][sv['REAL_A']],cfg,out/'classifiers'/view/arm/h,device)
                c2st[view][arm][h]=result;updates+=result['updates']
                write_json(out/'C2ST_PROGRESS.json',dict(UTC=utc(),completed_updates=updates,last_view=view,last_arm=arm,last_hypothesis=h))
        result=fit_classifier(classifier_view(tr['real'][st['REAL_B']],view),classifier_view(tr['real'][st['REAL_A']],view),
            classifier_view(va['real'][sv['REAL_B']],view),classifier_view(va['real'][sv['REAL_A']],view),
            va['real_groups'][sv['REAL_B']],va['real_groups'][sv['REAL_A']],cfg,out/'classifiers'/view/'REAL_REAL',device,shared=True)
        c2st[view]['REAL_REAL']=result;updates+=result['updates']
        write_json(out/'C2ST_PROGRESS.json',dict(UTC=utc(),completed_updates=updates,last_view=view,last_arm='REAL_REAL'))
    if updates!=cfg['evaluation']['classifier_total_updates']:raise RuntimeError('C2ST budget mismatch')
    write_json(out/'C2ST.json',clean(c2st))
    repeat_result=repeats(data['DEV'],models,extension,stats,cfg,out,device)
    input_result=intervention(data['DEV'],models,extension,stats,cfg,out,device)
    population=population_sensitivity(generated['TRAIN'],data['TRAIN'],cfg);write_json(out/'POPULATION_SENSITIVITY.json',clean(population))
    extension.assert_frozen()
    for arm,m in models.items():
        if state_digest(m.state_dict())!=frozen_hashes[arm]:raise RuntimeError('Evaluation mutated response')
    worst={split:{arm:{key:max(distance[split][arm][h]['components'][key] for h in HYPOTHESES) for key in ['wave','QPSD','noise']} for arm in ARMS} for split in ['TRAIN','DEV']}
    engineering_pass=all(e['signal_prefix_absolute_max_V']<=cfg['engineering']['signal_prefix_max_V'] and e['prefix_same']
        and sum(e['nonpositive_Q'].values())==0 and sum(e['guarded_PSD'].values())==0 for sp in results.values() for e in sp['engineering'].values())
    engineering_pass &= all(e['signal_prefix_absolute_max_V']<=cfg['engineering']['signal_prefix_max_V']
        and e['nonpositive_Q']==0 and e['guarded_PSD']==0 for e in list(repeat_result['engineering'].values())+list(input_result.values()))
    summary=dict(UTC=utc(),status='fixed_endpoint_evaluation_complete',G_updates=38000,D_updates=96000,C2ST_updates=updates,noise_updates=0,
        fixed_endpoint_files_sha256=states,engineering_passed=engineering_pass,worst_hypothesis_distances=worst,
        split_results=results,classifiers=c2st,repeated_parent_summary=repeat_result,input_dependency_summary=input_result,
        scientific_status='response_match_not_established',scientific_pass_threshold=None,
        independent_acquisition_available=False,DEV_previously_exposed=True,
        hypotheses_are_finite_assumptions_not_a_confidence_set=True,source_population_learning=False,
        evaluation_updates_response=0,evaluation_updates_noise=0,all_original_rows_preserved=True)
    write_json(out/'SUMMARY.json',clean(summary))
    return summary
