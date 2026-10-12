"""Independent 005 mechanism/statistical reconstruction from frozen artifacts."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

from n01.data import sha256
from n01.electronics_v3 import ObservedBaselineNoiseV3,fit_baseline_initialization_v3
from n01.model import digitize
from n01.train_noise_v3 import state_digest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def close(a,b,label,atol=2e-13,rtol=2e-12):
    if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False):
        raise AssertionError(label+': max difference '+str(np.max(np.abs(np.asarray(a)-np.asarray(b)))))


def event_acf20(x):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==100 and np.isfinite(x).all()
    b=x-x.mean(axis=1,keepdims=True)
    v=np.maximum(np.sum(b*b,axis=1)/100,(1/16383)**2/12)
    return np.stack([np.sum(b[:,:100-k]*b[:,k:],axis=1)/(100-k)/v for k in range(1,21)],axis=1)


def acf_loss(mean,target):
    e=(np.asarray(mean)-np.asarray(target))/.015
    return float(.5*(np.mean(e*e)+np.max(e*e)))


def verify_step_records(science,state,initial,target):
    frozen=torch.as_tensor(state['acf20_target']).to(torch.float64).cpu().numpy()
    initial_frozen=torch.as_tensor(initial['acf20_target']).to(torch.float64).cpu().numpy()
    assert frozen.shape==(20,) and np.array_equal(frozen,initial_frozen)
    digest=hashlib.sha256(np.asarray(frozen,dtype='<f8').tobytes(order='C')).hexdigest()
    assert digest==state['acf20_target_sha256']==initial['acf20_target_sha256']
    assert state['acf20_target_rows']==initial['acf20_target_rows']==21047
    close(frozen,target,'frozen target independently reconstructed from full TRAIN')
    records=[json.loads(line) for line in (Path(science)/'noise/training.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    assert len(records)==5000
    weights=dict(swd=.5,coordinate=.25,feature_marginal=.125,feature_joint=.0625,ols_joint=.0625)
    for u,row in enumerate(records,1):
        assert row['update']==u and row['forward_state_update']==u-1
        assert set(row['components'])==set(weights)
        old=sum(weights[key]*row['components'][key] for key in weights)
        errors=np.asarray(row['acf20_signed_errors'],dtype=np.float64)
        assert errors.shape==(20,) and np.isfinite(errors).all()
        extra=.5*(np.mean((errors/.015)**2)+np.max((errors/.015)**2))
        close(row['old_total'],old,'old step objective')
        close(row['acf20_loss'],extra,'added ACF20 step objective')
        close(row['total'],old+extra,'total step objective')
        a,b,c=row['old_gradient_norm'],row['acf20_gradient_norm'],row['total_gradient_preclip_norm']
        dot=row['old_vs_acf']['dot']
        close(c*c,a*a+b*b+2*dot,'exact total gradient norm',atol=2e-9,rtol=2e-10)
        close(row['old_vs_total']['dot'],a*a+dot,'old versus total gradient dot',atol=2e-9,rtol=2e-10)
        for name,norm in [('old_vs_acf',b),('old_vs_total',c)]:
            if a==0 or norm==0:
                assert row[name]['cosine'] is None and row[name]['angle_degrees'] is None
            else:
                cosine=float(np.clip(row[name]['dot']/(a*norm),-1,1))
                close(row[name]['cosine'],cosine,'gradient cosine',atol=2e-10)
                close(row[name]['angle_degrees'],math.degrees(math.acos(cosine)),'gradient angle',atol=2e-7)
    return dict(rows=5000,all_old_and_added_objectives_recomputed=True,
                gradient_norm_dot_and_angle_identities_verified=True,frozen_target_sha256=digest)


def check_bootstrap(path,real,groups,generated,seed,repetitions):
    """Reconstruct every draw from the registered seed without calling the evaluator."""
    with np.load(path,allow_pickle=False) as archive:
        saved={key:archive[key] for key in archive.files}
    ar,ag=event_acf20(real),event_acf20(generated)
    ids,inverse=np.unique(groups,return_inverse=True)
    counts=np.bincount(inverse,minlength=len(ids))
    sums=np.zeros((len(ids),20),dtype=np.float64)
    for i in range(len(ids)):sums[i]=ar[inverse==i].sum(axis=0)
    assert np.array_equal(ids,saved['group_ids']) and np.array_equal(counts,saved['group_counts'])
    close(sums,saved['group_sums'],'bootstrap group sums')
    point=ag.mean(axis=0)-ar.mean(axis=0)
    close(point,saved['difference_point'],'bootstrap point difference')
    rng=np.random.default_rng(seed);diffs=[]
    for rep in range(repetitions):
        group_draw=rng.integers(0,len(ids),len(ids))
        multiplicity=np.bincount(group_draw,minlength=len(ids))
        assert np.array_equal(multiplicity,saved['REAL_group_multiplicity'][rep])
        real_mean=(multiplicity@sums)/(multiplicity@counts)
        gen_draw=rng.integers(0,len(ag),len(ag))
        gen_mean=ag[gen_draw].mean(axis=0)
        close(real_mean,saved['REAL_bootstrap_means'][rep],'bootstrap REAL mean')
        close(gen_mean,saved['generated_bootstrap_means'][rep],'bootstrap generated mean')
        diffs.append(gen_mean-real_mean)
    diffs=np.asarray(diffs)
    close(diffs,saved['differences'],'bootstrap all differences')
    marginal=np.quantile(diffs,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(np.abs(diffs-point),axis=1),.95))
    return dict(rows=repetitions,point=point,marginal=marginal,radius=radius,
                simultaneous=np.stack([point-radius,point+radius]))


def check_permutations(path,seed,draws):
    with np.load(path,allow_pickle=False) as archive:saved={key:archive[key] for key in archive.files}
    labels=saved['labels'].astype(int);scores=saved['scores'];groups=saved['groups']
    assert len(labels)==len(scores)==len(groups)==512
    order=np.argsort(scores,kind='stable');ranks=np.empty(len(scores),dtype=np.float64)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and scores[order[end]]==scores[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2
        start=end
    ids=np.unique(groups);rows=[np.flatnonzero(groups==g) for g in ids]
    quotas=[int(labels[row].sum()) for row in rows]
    assert all(q>0 and 2*q==len(row) for row,q in zip(rows,quotas))
    assert np.array_equal(ids,saved['group_ids']) and np.array_equal(quotas,saved['positive_quota'])
    assert int(saved['seed'])==seed
    def auc(y):
        n1=int(y.sum());n0=len(y)-n1
        return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n0*n1))
    observed=auc(labels);assert observed==float(saved['observed_AUC'])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        y=np.zeros(len(labels),dtype=np.int8)
        for row,q in zip(rows,quotas):y[rng.permutation(row)[:q]]=1
        values.append(auc(y))
    values=np.asarray(values)
    assert np.array_equal(values,saved['permutation_AUC'])
    tail=int(np.sum(np.abs(values-.5)>=abs(observed-.5)))
    phat=tail/draws;z=1.959963984540054;den=1+z*z/draws
    center=(phat+z*z/(2*draws))/den
    half=z*math.sqrt(phat*(1-phat)/draws+z*z/(4*draws*draws))/den
    digests={}
    for name in ('labels','scores','groups','validation_test0_rows','validation_test1_rows'):
        value=np.ascontiguousarray(saved[name]);h=hashlib.sha256()
        h.update(str(value.shape).encode());h.update(str(value.dtype).encode());h.update(value.tobytes())
        digests[name]=h.hexdigest()
    return dict(draws=draws,tail_count=tail,plus_one_p_value=(tail+1)/(draws+1),all_values_exact=True,
                observed_AUC=observed,conditional_tail_fraction_K_over_draws=phat,
                conditional_tail_probability_MC_Wilson95=[max(0,center-half),min(1,center+half)],
                independently_recomputed_array_digests=digests)


def verify_details(root,science,out,device,train,validation,train_groups,val_groups,cfg,state,initial,model):
    diagnostic=Path(science)/'diagnostics';evaluation=Path(science)/'noise_validation'
    for item in cfg['references'].values():assert sha256(Path(root)/item['path'])==item['sha256']
    old_initial=torch.load(Path(root)/cfg['references']['initial004']['path'],map_location='cpu',weights_only=False)
    old_final=torch.load(Path(root)/cfg['references']['final004']['path'],map_location='cpu',weights_only=False)
    reconstructed=ObservedBaselineNoiseV3(fit_baseline_initialization_v3(train,**cfg['model']))
    assert state_digest(reconstructed.state_dict())==state_digest(old_initial['model'])==state_digest(initial['model'])
    assert all(torch.equal(state['initial_model_state'][key],old_initial['model'][key]) for key in old_initial['model'])
    latent=torch.load(diagnostic/'latents.pt',map_location='cpu',weights_only=False)
    assert latent['seed']==cfg['diagnostics']['seed'] and latent['rows']==8192 and latent['chunk_size']==256
    assert len(latent['chunks'])==32
    rng=torch.Generator(device=device).manual_seed(cfg['diagnostics']['seed'])
    for chunk in latent['chunks']:
        fresh=model.sample_noise(256,device=device,generator=rng,length=100)
        assert all(torch.equal(fresh[key].cpu(),chunk['noise'][key]) for key in fresh)
    with np.load(diagnostic/'paired_populations.npz',allow_pickle=False) as archive:
        pairs={key:archive[key] for key in archive.files}
    assert set(pairs)=={prefix+'_'+suffix for prefix in ('initial','v004','v005') for suffix in ('pre_adc','post_adc')}
    all_samples=[]
    with torch.no_grad():
        for prefix,checkpoint in [('initial',old_initial),('v004',old_final),('v005',state)]:
            m=ObservedBaselineNoiseV3(checkpoint['initialization']).to(device)
            m.load_state_dict(checkpoint['model'],strict=True);m.eval().requires_grad_(False)
            pres=[];posts=[]
            for chunk in latent['chunks']:
                noise={key:value.to(device) for key,value in chunk['noise'].items()}
                pre=m(noise,length=100);post=digitize(pre,ste=False)
                pres.append(pre[:,0].cpu().numpy());posts.append(post[:,0].cpu().numpy())
            pre=np.concatenate(pres);post=np.concatenate(posts)
            assert np.array_equal(pre,pairs[prefix+'_pre_adc']) and np.array_equal(post,pairs[prefix+'_post_adc'])
            all_samples.append(dict(model=prefix,rows=len(pre),pre_ADC_bitwise=True,post_ADC_bitwise=True))
    target=event_acf20(train).mean(axis=0)
    step_records=verify_step_records(science,state,initial,target)
    acf_pairs={key:event_acf20(value).mean(axis=0) for key,value in pairs.items()}
    means_old,means_new=acf_pairs['v004_post_adc'],acf_pairs['v005_post_adc']
    error_old,error_new=np.abs(means_old-target),np.abs(means_new-target)
    maximum_old,maximum_new=float(error_old.max()),float(error_new.max())
    loss_old,loss_new=acf_loss(means_old,target),acf_loss(means_new,target)
    train_match=maximum_new<=.015
    improved=loss_new<loss_old and maximum_new<=maximum_old
    gate=read(evaluation/'LEGACY_GATE.json');receipt=read(evaluation/'EVALUATION_RECEIPT.json')
    assert receipt['legacy_gate']==gate
    assert receipt['target_ACF20_sha256']==step_records['frozen_target_sha256']
    assert receipt['implementation_checks']['status']=='passed'
    assert all(value is True for value in receipt['implementation_checks']['checks'].values())
    assert all(value is True for value in receipt['implementation_checks']['runner'].values())
    candidate={key:val for key,val in gate['checks'].items() if key not in ('negative_control_point','negative_control_interval_contains_half')}
    assert len(candidate)==10 and candidate==receipt['candidate_checks']['checks']
    assert bool(all(candidate.values()))==receipt['candidate_checks']['passed']
    assert receipt['implementation_checks']['pending_runner_binding'] is False
    decision=receipt['decision']
    assert decision['TRAIN_match']==train_match and decision['TRAIN_improved']==improved
    assert decision['read_only_480_design_check_allowed']==bool(train_match and improved and all(candidate.values()))
    assert decision['response_training_authorized'] is False and decision['scientifically_qualified'] is False
    assert receipt['support_scope']['supported_samples']==100
    assert receipt['support_scope']['full480_integration_supported'] is False
    assert receipt['support_scope']['independent_acquisition_validations']==0
    assert receipt['control_calibration']['status']=='measurement_uncertain_single_exposed_acquisition'
    for name,value in [('TRAIN_current_max_ACF_error',maximum_new),('TRAIN004_max_ACF_error',maximum_old),
                       ('TRAIN_current_acf20_loss',loss_new),('TRAIN004_acf20_loss',loss_old)]:
        close(decision[name],value,'independent TRAIN decision '+name)
    table=read(evaluation/'ACF_PER_LAG.json')
    close(table['series']['TRAIN_target']['mean'],target,'full TRAIN ACF target')
    for key,value in acf_pairs.items():close(table['series'][key]['mean'],value,'paired table '+key)
    with np.load(diagnostic/'trajectory.npz',allow_pickle=False) as archive:
        trajectory={key:archive[key] for key in archive.files}
    assert np.array_equal(trajectory['updates'],np.arange(0,5001,250))
    close(trajectory['TRAIN_covariance_V2'],np.cov(train,rowvar=False,ddof=1),'trajectory TRAIN covariance',atol=2e-20)
    with np.load(diagnostic/'005A_covariances.npz',allow_pickle=False) as archive:
        close(archive['TRAIN_covariance_V2'],trajectory['TRAIN_covariance_V2'],'005A TRAIN covariance',atol=2e-20)
        for prefix in ('initial','v004'):
            for adc in ('pre_adc','post_adc'):
                close(archive[prefix+'_'+adc+'_covariance_V2'],np.cov(pairs[prefix+'_'+adc],rowvar=False,ddof=1),
                      '005A full covariance '+prefix+' '+adc,atol=2e-20)
    for row,prefix in [(0,'initial'),(-1,'v005')]:
        for adc,suffix in [('pre','pre_adc'),('post','post_adc')]:
            cov=np.cov(pairs[prefix+'_'+suffix],rowvar=False,ddof=1)
            close(trajectory[adc+'_covariance_V2'][row],cov,'trajectory endpoint covariance '+adc,atol=2e-20)
            close(trajectory[adc+'_ACF20'][row],acf_pairs[prefix+'_'+suffix],'trajectory endpoint ACF '+adc)
    trajectory_records=[json.loads(line) for line in (diagnostic/'trajectory.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    assert [row['update'] for row in trajectory_records]==list(range(0,5001,250))
    assert all(row['rng_isolation_verified'] for row in trajectory_records)
    with np.load(evaluation/'baseline_populations.npz',allow_pickle=False) as archive:legacy_gen=archive['generated']
    close(table['series']['legacy_generated_post_adc']['mean'],event_acf20(legacy_gen).mean(0),'legacy generated table')
    close(table['series']['REAL_validation_post_adc']['mean'],event_acf20(validation).mean(0),'development table')
    close(gate['statistics']['generated']['acf_mean'],event_acf20(legacy_gen).mean(0),'legacy generated ACF definition')
    close(gate['statistics']['REAL']['acf_mean'],event_acf20(validation).mean(0),'legacy REAL ACF definition')
    train_boot=check_bootstrap(evaluation/'bootstrap_TRAIN.npz',train,train_groups,pairs['v005_post_adc'],cfg['uncertainty']['TRAIN_seed'],1000)
    dev_boot=check_bootstrap(evaluation/'bootstrap_DEVELOPMENT.npz',validation,val_groups,legacy_gen,cfg['uncertainty']['development_seed'],1000)
    uncertainty=read(evaluation/'ACF_UNCERTAINTY.json')
    assert uncertainty['acceptance_gate'] is False
    for key,value in [('TRAIN',train_boot),('development',dev_boot)]:
        current=uncertainty[key]
        close(current['difference_point'],value['point'],'bootstrap point '+key)
        close(current['marginal95'],value['marginal'],'bootstrap marginal '+key)
        close(current['simultaneous95_centered_max'],value['simultaneous'],'bootstrap simultaneous '+key)
        close(current['bound_radius95'],value['radius'],'bootstrap radius '+key)
    permutation=check_permutations(evaluation/'RAW_CONTROL_CONDITIONAL.npz',cfg['controls']['conditional_seed'],9999)
    control=read(evaluation/'CONTROL_CALIBRATION.json')
    assert control==receipt['control_calibration'] and control['measurement_certified'] is False
    current=control['current_raw']
    for name in ('draws','tail_count','plus_one_p_value','observed_AUC','conditional_tail_fraction_K_over_draws',
                 'conditional_tail_probability_MC_Wilson95'):
        close(current[name],permutation[name],'control conditional '+name)
    for name in ('labels','scores','groups'):
        assert current['arrays_sha256'][name]==permutation['independently_recomputed_array_digests'][name]
    for side in (0,1):
        assert current['selected_test_rows_sha256']['class'+str(side)]==permutation['independently_recomputed_array_digests']['validation_test'+str(side)+'_rows']
    old_calibration=read(Path(root)/cfg['references']['calibration003']['path'])
    for name in ('legacy_joint_bootstrap_false_positive','conditional_permutation_false_positive'):
        assert control['reference003'][name]==old_calibration[name]
    assert control['reference003']['refits']==128 and control['reference003']['method_draws']==255
    for view in ('raw','demean','detrend'):
        with np.load(evaluation/'classifiers'/view/'REAL_REAL/predictions.npz',allow_pickle=False) as current_npz:
            with np.load(Path(root)/cfg['references']['control004_'+view]['path'],allow_pickle=False) as old_npz:
                for key in ('labels','scores','groups0','groups1'):
                    assert current_npz[key].dtype==old_npz[key].dtype and np.array_equal(current_npz[key],old_npz[key])
    return dict(initialization_reconstructed_from_full_TRAIN_and_matches004=True,
        paired_generated_samples_replayed=all_samples,all_24576_paired_rows_bitwise=True,
        step_records=step_records,all_21_trajectory_records_present=True,
        initial_and_final_trajectory_covariance_ACF_recomputed=True,all_005A_covariances_recomputed=True,
        TRAIN_target_and_all_ACF_table_rows_independently_recomputed=True,
        all_2000_group_bootstrap_draws_reconstructed=True,conditional_permutation=permutation,
        candidate_and_control_decisions_independently_recomputed=True,TRAIN_match=train_match,
        TRAIN_improved=improved,read_only_480_design_check_allowed=decision['read_only_480_design_check_allowed'],
        noise_training_updates_added=0,response_training_updates_added=0,evaluation_classifier_updates_added=0)
