"""One common WP1 run: information, saved boundary, hard evaluation and probe.

Draft configuration cannot execute. All files are hashed by stdlib before any
numeric import. No optimizer, training, new AD, DEV-model use or restart.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from wp1_ops import Store,gate,unchanged,require,sha,write_new,read_json,clean

HYPOTHESES=('fast','slow','common_level')
TRAIN_KEYS=('source','source_carrier384','real','real_baseline100','source_ids','real_ids','real_groups','source_role','real_role')
REAL_METADATA=('real_peak_tied_samples','real_peak_search_edge','real_global_peak_outside_search','real_peak_index480','real_crop_start480')
REAL_UNKNOWN_FLAGS=('pre_ADC_out_of_range','ADC_halfgrid_tied','ADC_rail_hit','ADC_halfgrid_min_distance','ADC_halfgrid_tie_cells')

def arr(value):return value.detach().cpu().numpy()

def observation_rows(observation,carrier,loss,selected=None):
    """Reuse the one actual observation; these diagnostics perform no ADC."""
    from mechanism012_losses import numeric_flags,peak_diagnostics
    fields={n:arr(observation[n]) for n in ('window_V','baseline100','peak_index','adc_carrier')}
    fields['diagnostic__baseline_mean_V']=arr(observation['baseline_mean_V'])
    diagnostics=numeric_flags(observation,carrier,loss)
    from n01.model import ADC_MIN,ADC_MAX
    diagnostics['pre_ADC_out_of_range']=((carrier<ADC_MIN)|(carrier>ADC_MAX)).any(1)
    diagnostics.update(peak_diagnostics(observation['adc_carrier']))
    for n,value in diagnostics.items():
        v=arr(value) if hasattr(value,'detach') else value
        fields[('flag__' if v.dtype.kind=='b' else 'diagnostic__')+n]=v
    return fields if selected is None else {n:v[selected] for n,v in fields.items()}

def observation_booleans(fields):return {n[6:]:v for n,v in fields.items() if n.startswith('flag__')}

def signed_changes(values,reference):
    return {k:None if v is None or reference[k] is None else v-reference[k] for k,v in values.items()}

def lossless_window_codes(observation):
    """Reversible storage, without another ADC or changing numerical precision.

Optional IEEE XOR bits preserve any platform division/subtraction differences.
Actual prefix means are retained, rather than recomputed with a different sum.
"""
    import numpy as np
    y,a,p=observation['window_V'],observation['adc_carrier'],observation['peak_index']
    indices=p[:,None]-16+np.arange(256);cells=np.take_along_axis(a,indices,axis=1)
    codes=np.rint((cells+.5)*16383).astype(np.uint16);mu=observation['baseline100'].mean(1)
    # The baseline itself was computed in Torch, whose reduction order can differ.
    exact_mu=observation.get('diagnostic__baseline_mean_V',mu)
    reconstructed=codes.astype(np.float64)/16383-.5-exact_mu[:,None]
    xor=reconstructed.view(np.uint64)^np.ascontiguousarray(y).view(np.uint64)
    decoded=(reconstructed.view(np.uint64)^xor).view(np.float64)
    require(np.array_equal(decoded.view(np.uint64),np.ascontiguousarray(y).view(np.uint64)),'Lossless window reconstruction failed')
    result=dict(ADC_crop_codes=codes,baseline_mean_V=exact_mu,peak_index=p)
    if xor.any():result['IEEE754_reconstruction_XOR_uint64']=xor
    return result

def save_report(directory,name,report):
    """Compact JSON references dense arrays in a private compressed NPZ."""
    import numpy as np
    arrays={}
    def collect(value,path='root'):
        if isinstance(value,np.ndarray):arrays[path]=value;return dict(array=path,file=name+'.npz',shape=list(value.shape),dtype=str(value.dtype))
        if isinstance(value,dict):return {str(k):collect(v,path+'/'+str(k)) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [collect(v,path+'/'+str(i)) for i,v in enumerate(value)]
        return clean(value)
    reduced=collect(report)
    if arrays:
        with (directory/(name+'.npz')).open('xb') as f:np.savez_compressed(f,**arrays)
    write_new(directory/(name+'.json'),reduced)


def reference_snapshot(model,extension):
    from n01.response008 import state_digest
    model.assert_frozen_unchanged(deep=True);extension.assert_frozen()
    require(all(x.grad is None and not x.requires_grad for x in model.parameters()),'Frozen inference parameters/grad accumulator changed')
    return dict(baseline=state_digest(model.baseline.state_dict()),noise005=state_digest(extension.model005.state_dict()),
        extension=state_digest(extension.state_dict()),head=state_digest(model.basis_head.state_dict()),stage=model.stage)


def prepare_numeric(paths,p,store,device):
    import numpy as np
    import torch
    from n01.model008 import N01Response008
    from n01.model010 import FrozenDeterministicVoltageShape010
    from n01.response008 import make_extension,statistics_digest,state_digest,loss_tensor_statistics
    from n01.losses008_v2 import N01Loss008
    previous=read_json(paths['mechanism012_complete'])
    require(previous.get('schema')=='n01/mechanism012-result/v1' and previous.get('engineering_checks_passed') is True
        and previous.get('artificial_only') is False and not previous.get('inflight')
        and previous.get('head_parameter_count')==1288 and previous.get('AD_queries_completed')==11
        and previous['inputs_sha256'].get(str(paths['head_final']))==p['inputs']['head_final']['sha256'],'Saved012 gradient/current head provenance mismatch')
    stats,domain,cfg=[read_json(paths[k]) for k in ('statistics','data_statistics','base_config')]
    require(stats['statistics_split']==domain['split']==stats['loss']['fit_split']=='TRAIN'
        and stats['prepared_dataset_sha256']==p['inputs']['dataset']['sha256']
        and stats['data_statistics_sha256']==p['inputs']['data_statistics']['sha256'],'Frozen TRAIN statistics chain changed')
    with np.load(paths['dataset'],allow_pickle=False) as z:data={n:z[n].copy() for n in TRAIN_KEYS+REAL_METADATA}
    shapes=dict(source=(6593,256),source_carrier384=(6593,384),real=(21047,256),real_baseline100=(21047,100),
        source_ids=(6593,),real_ids=(21047,),real_groups=(21047,),source_role=(6593,),real_role=(21047,))
    for n,s in shapes.items():
        x=data[n];require(x.shape==s and x.dtype.kind!='O' and (x.dtype.kind not in 'fc' or np.isfinite(x).all()),'Invalid legal TRAIN member: '+n)
    for n in REAL_METADATA:require(data[n].shape==(21047,) and data[n].dtype.kind in 'biu','REAL saved metadata lineage mismatch')
    data['real_observation_flags']=dict(peak_tied=data['real_peak_tied_samples']>1,peak_at_search_boundary=data['real_peak_search_edge'],
        global_peak_outside_search=data['real_global_peak_outside_search'])
    require(np.all(data['source_role']=='TRAIN') and np.all(data['real_role']=='TRAIN')
        and len(np.unique(data['source_ids']))==6593 and len(np.unique(data['real_ids']))==21047,'Role or identity changed')
    load=lambda key:store.call('checkpoint_'+key,dict(checkpoint_loads=1),lambda:torch.load(paths[key],map_location='cpu',weights_only=False))
    a=load('deterministic')
    require(a.get('schema')=='response008/v1' and a.get('arm')=='deterministic' and a['G_updates']==19000
        and a['stage']=='low_learning_rate' and a['D_updates']=={h:16000 for h in HYPOTHESES}
        and a['noise_updates']==0 and a['population_learning'] is False and a['config']==cfg
        and statistics_digest(a['statistics'])==statistics_digest(stats),'OriginalA metadata changed')
    baseline=N01Response008(stats['model'],'deterministic');baseline.load_state_dict(a['model'],strict=True)
    baseline.set_stage(a['stage']);baseline.eval();baseline.requires_grad_(False);baseline.to(device)
    n=load('noise005');require(n['schema_version']==5 and n['update']==5000 and n['model_type']=='ObservedBaselineNoiseV3'
        and n['engineering'] is False,'Original005 metadata changed')
    extension=make_extension(n,device);require(state_digest(extension.model005.state_dict())==a['noise005_state_sha256'],'005/A bound state mismatch')
    cp=load('head_final');model=FrozenDeterministicVoltageShape010(baseline)
    from wp1_parameter_probe import validated_head_state
    validated_head_state(cp);model.basis_head.load_state_dict(cp['new_head'],strict=True);model.eval();model.requires_grad_(False)
    snapshot=reference_snapshot(model,extension)
    require({k:snapshot[k] for k in ('baseline','noise005','extension')}==cp['frozen_state_sha256'],'Final frozen A/005/extension state mismatch')
    projection=p['reference']['projection'];loss=N01Loss008(loss_tensor_statistics(stats['loss']),seed=projection['training_seed'],evaluation_seed=projection['seed']).to(device)
    context=loss.create_evaluation_context();require(context.metadata()=={k:projection[k] for k in ('kind','seed','update','fingerprint')},'Fixed evaluation projection changed')
    with np.load(paths['gradients'],allow_pickle=False) as z:grad={n:z[n].copy() for n in ('total','wave','QPSD')}
    return dict(model=model,extension=extension,data=data,loss=loss,context=context,head_cp=cp,gradients=grad,
        device=device,snapshot=snapshot)


def information_join(data,information_path):
    import numpy as np
    names=('source_id','role','canonical_Q_Vns','amplitude_384_V','PC_scores_384_Qnorm','PC_valid_384_Qnorm','physical_weight_known')
    with np.load(information_path,allow_pickle=False) as z:info={n:z[n].copy() for n in names}
    lookup={str(x):i for i,x in enumerate(info['source_id'])};require(len(lookup)==len(info['source_id']),'Duplicate information identity')
    indices=np.asarray([lookup[str(x)] for x in data['source_ids']],np.int64)
    require(np.all(info['role'][indices]=='TRAIN') and np.array_equal(info['source_id'][indices],data['source_ids']),'SOURCE ID/TRAIN join mismatch')
    return {n:x[indices] for n,x in info.items()}


def group_real_control(groups,rule):
    import numpy as np
    unique,count=np.unique(groups,return_counts=True);rng=np.random.default_rng(rule['seed']);sides=[[],[]];extra=[]
    for size in sorted(np.unique(count)):
        chosen=rng.permutation(np.sort(unique[count==size]));paired=2*(len(chosen)//2)
        sides[0].extend(chosen[:paired:2]);sides[1].extend(chosen[1:paired:2]);extra.extend(chosen[paired:])
    rows=[np.flatnonzero(np.isin(groups,x)) for x in sides]
    require(not set(sides[0])&set(sides[1]) and all(len(x)==rule['rows_per_side'] for x in rows)
        and all(len(x)==rule['groups_per_side'] for x in sides),'Registered group-isolated RR counts changed')
    histogram=[{str(s):int((np.asarray([int((groups==g).sum()) for g in side])==s).sum()) for s in (48,49,50)} for side in sides]
    require(histogram[0]==histogram[1]==rule['histogram_each_side'],'RR group-size stratification changed')
    return rows,dict(left_groups=sides[0],right_groups=sides[1],unpaired_groups=extra,rows_per_side=[len(x) for x in rows],
        group_size_histograms=histogram,not_irreducible_lower_bound=True,main_population_exclusions=0)


class Runtime:
    def __init__(self,p,ctx,store,evaluation_dir,source_info):
        import numpy as np
        self.p,self.ctx,self.store,self.out,self.info=p,ctx,store,evaluation_dir,source_info
        self.model,self.ext,self.data,self.loss,self.context,self.device=[ctx[k] for k in ('model','extension','data','loss','context','device')]
        self.parents=np.sort(np.random.default_rng(p['parameter_validation']['selection_seed']).choice(len(self.data['source']),p['parameter_validation']['parents'],replace=False))
        self.references=[];self.subsets=[];self.records=[];self.primary_cells=[];self.immutable_start=ctx['snapshot']
        from wp1_evaluation import observable_features
        self.real_features=observable_features(self.data['real'],self.data['real_baseline100'],float(self.loss.q_floor))
        self.reference_features={};self.defined_states=0
    def native_call(self,name,charges,fn,**detail):
        import torch
        def synced():
            result=fn()
            if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
            return result
        return self.store.call(name,charges,synced,**detail)
    def forward(self,x,z,*,theta=None,direction=None,scale=None,capture=False):
        from wp1_parameter_probe import parameter_forward
        from n01.model008 import control_average008
        original=self.model._baseline_once;cache={}
        # Nested A return is durable even when a later head operation fails.
        def baseline_once(values):
            result=self.native_call('frozen_A',dict(old_A_rows=len(values)),lambda:original(values))
            if capture:cache['decoded']=result[1].detach().cpu().clone()
            return result
        self.model._baseline_once=baseline_once
        hook=self.model.basis_head.register_forward_hook(lambda m,args,out:cache.update(logits=control_average008(out).detach().double().cpu())) if capture else None
        try:
            result=self.native_call('actual_head_parameter_forward',dict(head_actual_parameter_forward_rows=len(x),head_total_kernel_rows=len(x)),
                lambda:self.model(x,z) if theta is None else parameter_forward(self.model,x,z,theta,direction,scale))
        finally:
            del self.model._baseline_once
            if hook is not None:hook.remove()
        require(float(result['canonical_charge_identity_residual'].abs().max())<=1e-10 and bool((result['signal_pre'][:,:100]==0).all()),'Frozen physical prefix/Q engineering identity failed')
        return result,cache
    def observe(self,carrier,*,charge_ADC=True,peaks=None):
        from n01.observation008 import observe_generated_carrier
        from mechanism012_losses import gather_observation
        charges=dict(all_path_gather_rows=len(carrier))
        if charge_ADC:charges['hard_ADC_rows']=len(carrier)
        return self.native_call('hard_ADC_dynamic' if charge_ADC else 'same_ADC_fixed_gather',charges,
            lambda:observe_generated_carrier(carrier,ste=False,q_floor=float(self.loss.q_floor),return_carrier=True,return_indices=True)
            if charge_ADC else gather_observation(carrier,peaks,float(self.loss.q_floor)))
    def references_forward(self):
        import numpy as np
        import torch
        from n01.train_shape010 import Streams010
        from n01.response008 import fixed_noises
        p,r,chunk=self.p,self.p['reference']['draws_per_bank'],self.p['reference']['chunk_rows'];fullparents=np.repeat(np.arange(len(self.data['source'])),r)
        for bank,seeds in enumerate(p['reference']['bank_seeds']):
            streams=Streams010(seeds['shape'],seeds['noise005'],seeds['extension']);parts={arm:{h:[] for h in HYPOTHESES} for arm in ('A','theta')}
            sub={name:[] for name in ('source','shape_z','decoded','logits','signal','base')};sub['noise']={h:[] for h in HYPOTHESES};sub['observations']={h:[] for h in HYPOTHESES}
            allsignals=[];allbase=[];shared_baseline=[];latent_parts={'shape_z':[],'extension':[]};noise_latent_parts={}
            for start in range(0,len(fullparents),chunk):
                stop=min(start+chunk,len(fullparents));pr=fullparents[start:stop]
                x=torch.as_tensor(self.data['source_carrier384'][pr],device=self.device,dtype=torch.float64)
                z,nz,epsilon=streams.draw(len(x),self.device)
                noises=self.native_call('shared_frozen005_and_extensions',dict(noise005_rows=len(x)),lambda:fixed_noises(self.ext,nz,epsilon),bank=bank,start=start,stop=stop)
                with torch.no_grad():output,cache=self.forward(x,z,capture=True)
                allsignals.append(arr(output['signal_pre']));allbase.append(arr(output['base_signal_pre_V']))
                chosen=np.isin(pr,self.parents)
                if chosen.any():
                    for key,value in (('source',x),('shape_z',z),('signal',output['signal_pre']),('base',output['base_signal_pre_V'])):sub[key].append(arr(value)[chosen])
                    sub['decoded'].append(cache['decoded'][torch.from_numpy(chosen)].numpy());sub['logits'].append(cache['logits'][torch.from_numpy(chosen)].numpy())
                    for h in HYPOTHESES:sub['noise'][h].append(arr(noises[h])[chosen])
                for arm,key in (('A','base_signal_pre_V'),('theta','signal_pre')):
                    for h in HYPOTHESES:
                        carrier=output[key]+noises[h];obs=self.observe(carrier);rows=observation_rows(obs,carrier,self.loss)
                        parts[arm][h].append(rows)
                        if arm=='theta' and chosen.any():sub['observations'][h].append({n:v[chosen] for n,v in rows.items()})
                        if arm=='theta' and h==HYPOTHESES[0]:shared_baseline.append(arr(obs['baseline100']))
                for name,value in nz.items():noise_latent_parts.setdefault(name,[]).append(arr(value))
                latent_parts['shape_z'].append(arr(z));latent_parts['extension'].append(arr(epsilon))
                if bank==0 and start==0:
                    require(reference_snapshot(self.model,self.ext)==self.immutable_start,'First family mutated frozen tensors')
                    unchanged(self.store.checked)
                    self.store.json('FIRST_WP1_PROOF.json',dict(self.store.base(),first_family_rows=len(x),reference_head_G_updates=2000,
                        all_three_hypotheses=True,A_theta_noise_family_completed=True,first_family_completed=True,frozen_state_unchanged=True,
                        inputs_unchanged=True,extra_pilot_rows=0,optimizer_created=False,
                        actual_science=not self.store.artificial))
            concatenate=lambda seq:{n:np.concatenate([x[n] for x in seq]) for n in seq[0]}
            reference={arm:{h:concatenate(parts[arm][h]) for h in HYPOTHESES} for arm in ('A','theta')}
            reference.update(signal=np.concatenate(allsignals),base=np.concatenate(allbase),parent=fullparents,draw=np.tile(np.arange(r),len(self.data['source'])))
            subset={k:np.concatenate(v) for k,v in sub.items() if k not in ('noise','observations')}
            subset['noise']={h:np.concatenate(v) for h,v in sub['noise'].items()};subset['observations']={h:concatenate(v) for h,v in sub['observations'].items()}
            subset.update(parent=np.repeat(self.parents,r),draw=np.tile(np.arange(r),len(self.parents)))
            self.references.append(reference);self.subsets.append(subset)
            from wp1_evaluation import observable_features
            for o in subset['observations'].values():
                self.reference_features[id(o['window_V'])]=observable_features(o['window_V'],o['baseline100'],float(self.loss.q_floor))
            self.store.npz('BANK_%d.npz'%bank,**{k:np.concatenate(v) for k,v in latent_parts.items()},
                **{k:np.concatenate(v) for k,v in noise_latent_parts.items()},source_parent_rows=fullparents,
                **{'stream_state_'+k:arr(v) for k,v in streams.state_dict().items()})
    def component(self,w,b,ref=None):
        from wp1_evaluation import component_distances
        rw,rb=(self.data['real'],self.data['real_baseline100']) if ref is None else ref
        cached=self.real_features if ref is None else self.reference_features.get(id(rw))
        return component_distances(w,b,rw,rb,self.loss,self.context,cached_reference=cached)
    def complete_evaluations(self):
        import numpy as np
        from wp1_evaluation import evaluate_hard_population,repeated_parent_statistics,observable_features
        summary=[]
        for bank,ref in enumerate(self.references):
            for arm in ('A','theta'):
                signal=ref['base'] if arm=='A' else ref['signal'];grouped=signal.reshape(len(self.data['source']),self.p['reference']['draws_per_bank'],384)
                signal_name='SIGNAL_MOMENTS_%d_%s.npz'%(bank,arm)
                self.store.npz(signal_name,mean_V=grouped.mean(1),sample_variance_V2=grouped.var(1,ddof=1),source_parent_row=np.arange(len(grouped)))
                for h in HYPOTHESES:
                    obs=ref[arm][h];result=evaluate_hard_population(obs['window_V'],obs['baseline100'],real_wave256=self.data['real'],
                        real_baseline100=self.data['real_baseline100'],parent_row=ref['parent'],draw_index=ref['draw'],bank_index=np.full(len(ref['parent']),bank,np.int64),
                        source_information=self.info,loss=self.loss,context=self.context,real_groups=self.data['real_groups'],
                        observation_flags=observation_booleans(obs),real_observation_flags=self.data['real_observation_flags'],real_unknown_flags=REAL_UNKNOWN_FLAGS,
                        cached_real=self.real_features)
                    result['repeated_parent']['preADC_signal_shared_artifact']=dict(file='../artifacts/'+signal_name,
                        identical_for_all_three_hypotheses=True,draws_per_parent=self.p['reference']['draws_per_bank'],
                        mean_MCSE_reconstruction='sqrt(sample_variance_V2/draws_per_parent)',complete_covariance='centered_saved_signal_draws/(draws-1)')
                    result['observation_diagnostics']={k[12:]:v for k,v in obs.items() if k.startswith('diagnostic__')}
                    # Raw per-row fields remain in NPZ; never JSON lists or console.
                    save_report(self.out,'FULL_%d_%s_%s'%(bank,arm,h),result)
                    summary.append(dict(bank=bank,arm=arm,hypothesis=h,metrics=self.component(obs['window_V'],obs['baseline100'])))
            arrays={arm+'_'+h+'__'+key:value for arm in ('A','theta') for h in HYPOTHESES for key,value in lossless_window_codes(ref[arm][h]).items()}
            arrays.update(theta_signal_pre384=ref['signal'],source_parent_row=ref['parent'],draw_index=ref['draw'],baseline100=ref['theta'][HYPOTHESES[0]]['baseline100'])
            for arm in ('A','theta'):
                for h in HYPOTHESES:
                    arrays.update({arm+'_'+h+'__'+k:v for k,v in ref[arm][h].items() if k.startswith(('flag__','diagnostic__'))})
            if bank==0:
                base=ref['base'].reshape(len(self.data['source']),self.p['reference']['draws_per_bank'],384)
                require(np.array_equal(base,np.broadcast_to(base[:,:1],base.shape)),'A repeat pure-signal bytes differ')
                arrays['A_signal_per_parent384']=base[:,0]
            self.store.npz('REFERENCE_%d.npz'%bank,**arrays)
        self.store.npz('REAL_ROW_FLAGS.npz',**{n:self.data[n] for n in REAL_METADATA},**self.data['real_observation_flags'],
            real_ids=self.data['real_ids'],real_groups=self.data['real_groups'])
        self.store.json('REFERENCE_ENCODING.json',dict(encoding='ADC_crop_codes_uint16_plus_actual_baseline_mean_float64_plus_optional_IEEE754_XOR',
            reconstruction='(ADC_crop_codes.astype(float64)/16383-.5-baseline_mean[:,None]).view(uint64) XOR IEEE754_reconstruction_XOR_uint64 then view(float64)',
            missing_XOR_means_all_zero=True,original_dtype='float64',all_rows_byte_reconstruction_checked=True,baseline100_shared_across_arms_and_hypotheses=True,
            A_signal_file='REFERENCE_0.npz:A_signal_per_parent384',A_signal_broadcast_across_banks_and_draws=True,extra_ADC_or_model_calls=0))
        require(np.array_equal(self.references[0]['base'],self.references[1]['base']),'A pure signal cross-bank bytes differ')
        for arm in ('A','theta'):
            pure=np.concatenate([r['base'] if arm=='A' else r['signal'] for r in self.references]);parent=np.concatenate([r['parent'] for r in self.references])
            draws=np.concatenate([r['draw'] for r in self.references]);banks=np.concatenate([np.full(len(r['parent']),b,np.int64) for b,r in enumerate(self.references)])
            for h in HYPOTHESES:
                wave=np.concatenate([r[arm][h]['window_V'] for r in self.references]);baseline=np.concatenate([r[arm][h]['baseline100'] for r in self.references])
                features,_=observable_features(wave,baseline,float(self.loss.q_floor))
                moments=repeated_parent_statistics(wave,features,parent,draws,banks,pure if h==HYPOTHESES[0] else None)
                if h!=HYPOTHESES[0]:moments['preADC_signal_shared_artifact']='REPEATS_8_%s_%s.npz'%(arm,HYPOTHESES[0])
                save_report(self.out,'REPEATS_8_%s_%s'%(arm,h),moments)
        self.store.json('FULL_POPULATION_COMPARISON.json',dict(records=summary,all_SOURCE_parents=len(self.data['source']),all_REAL_rows=len(self.data['real']),
            reference_head_G_updates=2000,model_seed_uncertainty='not_estimated',independent_acquisition='unverified',candidate_acceptance=False))
        left_right,meta=group_real_control(self.data['real_groups'],self.p['RR'])
        rr=self.component(self.data['real'][left_right[0]],self.data['real_baseline100'][left_right[0]],
            ref=(self.data['real'][left_right[1]],self.data['real_baseline100'][left_right[1]]))
        self.store.json('REAL_REAL_CONTROL.json',dict(group_rule=meta,metrics=rr,tolerance_source=False))
        return summary

    def calibrate(self):
        import numpy as np
        import torch
        from wp1_parameter_probe import sensitivity_cases,apply_sensitivity
        cases=sensitivity_cases(self.p['sensitivity']);records=[];allowed=self.p['calibration']['tolerance_cases']
        reference_metrics={(bank,h):self.component(o['window_V'],o['baseline100']) for bank,sub in enumerate(self.subsets) for h,o in sub['observations'].items()}
        def chosen(case):
            key={'gain':'gain_percent_signed','tail':'tail_percent_signed','noise':'noise_percent_signed','width':'width_sigma_software_ns_positive'}[case['kind']]
            return case.get('percent',case.get('sigma')) in allowed[key]
        chunk=self.p['reference']['chunk_rows']
        for index,case in enumerate(cases):
            for bank,sub in enumerate(self.subsets):
                for h in HYPOTHESES:
                    waves=[];baselines=[];peaks=[];q_signal=[];flag_parts={};diagnostic_parts={}
                    for start in range(0,len(sub['source']),chunk):
                        stop=min(start+chunk,len(sub['source']));s=torch.as_tensor(sub['signal'][start:stop],device=self.device)
                        n=torch.as_tensor(sub['noise'][h][start:stop],device=self.device)
                        carrier,details=apply_sensitivity(s,n,case,tail_gate_width=self.p['sensitivity']['tail_gate_width'],gaussian_radius=self.p['sensitivity']['gaussian_radius'])
                        obs=self.observe(carrier);waves.append(arr(obs['window_V']));baselines.append(arr(obs['baseline100']));peaks.append(arr(obs['peak_index']))
                        rows=observation_rows(obs,carrier,self.loss)
                        for key,value in rows.items():
                            if key.startswith('flag__'):flag_parts.setdefault(key[6:],[]).append(value)
                            elif key.startswith('diagnostic__'):diagnostic_parts.setdefault(key[12:],[]).append(value)
                        q_signal.append(arr(details['signal'][:,110:366].sum(1)))
                    w,b=np.concatenate(waves),np.concatenate(baselines);ref=sub['observations'][h]
                    real_metric=self.component(w,b);self_metric=self.component(w,b,ref=(ref['window_V'],ref['baseline100']))
                    record=dict(case_index=index,case=case,bank=bank,hypothesis=h,tolerance_case=chosen(case),
                        distance_to_REAL=real_metric,self_distance=self_metric,
                        signed_REAL_changes=signed_changes(real_metric['values'],reference_metrics[bank,h]['values']),
                        guard_counts={k:dict(count=int(np.concatenate(v).sum()),denominator=len(w)) for k,v in flag_parts.items()},
                        changed_crop_rows=int((np.concatenate(peaks)!=ref['peak_index']).sum()))
                    records.append(record)
                    self.store.npz('CALIBRATION_%02d_%d_%s.npz'%(index,bank,h),Q=w.sum(1),tail=w[:,46:].sum(1),
                        peak_index=np.concatenate(peaks),canonical_signal_Q=np.concatenate(q_signal),source_parent_row=sub['parent'],draw_index=sub['draw'],
                        **{'flag__'+k:np.concatenate(v) for k,v in flag_parts.items()},**{'diagnostic__'+k:np.concatenate(v) for k,v in diagnostic_parts.items()})
        tolerance={};status={}
        for h in HYPOTHESES:
            relevant=[r for r in records if r['hypothesis']==h and r['tolerance_case']]
            require(len(relevant)==14,'Seven tolerance cases across two banks required')
            tolerance[h]={};status[h]={}
            for key in relevant[0]['self_distance']['values']:
                means=[];quality=True
                for index in sorted({r['case_index'] for r in relevant}):
                    pair=[r for r in relevant if r['case_index']==index];values=[r['self_distance']['values'][key] for r in pair]
                    if len(values)!=2 or any(v is None or not np.isfinite(v) for v in values):quality=False;continue
                    if key=='QPSD' and any(not r['self_distance']['coverage']['conditional_coverage_complete'] for r in pair):quality=False
                    means.append(float(np.mean(values)))
                value=max(means) if quality and len(means)==7 else None
                if value is None or value<=0:tolerance[h][key]=None;status[h][key]='not_calibrated'
                else:tolerance[h][key]=value;status[h][key]='calibrated_from_registered_artificial_self_distances'
        self.store.json('CALIBRATION.json',dict(records=records,reference_head_G_updates=2000,synthetic_not_measurement=True,
            application_assumptions=self.p['application_tolerances'],RR_used_to_set_tolerance=False))
        self.store.json('CALIBRATED_TOLERANCES.json',dict(schema='n01/wp1-tolerance-map/v1',T_by_hypothesis=tolerance,status=status,
            rule=self.p['calibration'],before_new_candidate=True,hardware_calibrated=False))
        return tolerance

    def parameter_states(self,tolerance):
        import numpy as np
        import torch
        from wp1_parameter_probe import parameter_directions_from_saved_state,analytic_head_direction_map,nominal_scales,select_method_once
        from n01.model import ADC_MIN,ADC_MAX
        from wp1_evaluation import observable_features,flag_rates,output_q_bins
        p=self.p;theta,directions=parameter_directions_from_saved_state(self.ctx['head_cp'],self.ctx['gradients'],random_seed=p['parameter_validation']['random_direction_seed'])
        require([d['name'] for d in directions]==p['parameter_validation']['directions'],'Registered direction names changed')
        chunk=p['reference']['chunk_rows'];n=len(self.subsets[0]['source']);batch_rows=p['parameter_validation']['parents_per_batch']*p['reference']['draws_per_bank']
        reference_cells={}
        for bank,sub in enumerate(self.subsets):
            for batch in range(p['parameter_validation']['batches']):
                sl=slice(batch*batch_rows,(batch+1)*batch_rows)
                for h in HYPOTHESES:
                    o=sub['observations'][h];reference_cells[bank,batch,h]=self.component(o['window_V'][sl],o['baseline100'][sl])
        primary=[];records=[];undefined={}
        for direction in directions:
            squared=0.;mapping_rows=0
            if direction['direction_defined']:
                for bank,sub in enumerate(self.subsets):
                    for start in range(0,n,chunk):
                        stop=min(start+chunk,n);to=lambda key:torch.as_tensor(sub[key][start:stop],device=self.device)
                        dy=self.native_call('analytic_head_direction_map',dict(head_linear_direction_map_rows=stop-start,head_total_kernel_rows=stop-start),
                            lambda:analytic_head_direction_map(self.model,to('decoded'),to('base'),to('shape_z'),direction['vector'],reference_control_logits=to('logits')),
                            direction=direction['name'],bank=bank,start=start,stop=stop)
                        squared+=float(dy.square().sum());mapping_rows+=stop-start
                scales=nominal_scales(squared,mapping_rows)
            else:scales=dict(direction_defined=False,reason=direction['undefined_reason'],lambda_per_scale=[None]*3)
            if not scales['direction_defined']:
                undefined[direction['name']]=scales['reason'];records.append(dict(direction=direction['name'],defined=False,reason=scales['reason'],
                    skipped_registered_states=[dict(nominal_scale_LSB=t,sign=s) for t in p['parameter_validation']['nominal_carrier_RMS_ADC_LSB'] for s in p['parameter_validation']['signs']],
                    all_primary_12cells_retained_by_method_selector=True,actual_parameter_rows=0));continue
            self.defined_states+=len(p['parameter_validation']['nominal_carrier_RMS_ADC_LSB'])*len(p['parameter_validation']['signs'])
            for target,scale in zip(p['parameter_validation']['nominal_carrier_RMS_ADC_LSB'],scales['lambda_per_scale']):
                for sign in p['parameter_validation']['signs']:
                    for bank,sub in enumerate(self.subsets):
                        parts={path:{h:dict(wave=[],baseline=[],peak=[]) for h in HYPOTHESES} for path in ('S','H_fixed','H_moving')};rms=[];maxima=[];base_errors=[]
                        hard_flags={path:{h:{} for h in HYPOTHESES} for path in ('H_fixed','H_moving')}
                        for start in range(0,n,chunk):
                            stop=min(start+chunk,n);x=torch.as_tensor(sub['source'][start:stop],device=self.device);z=torch.as_tensor(sub['shape_z'][start:stop],device=self.device)
                            output,_=self.forward(x,z,theta=theta,direction=direction['vector'],scale=sign*scale)
                            delta=output['signal_pre']-torch.as_tensor(sub['signal'][start:stop],device=self.device)
                            base_error=(output['base_signal_pre_V']-torch.as_tensor(sub['base'][start:stop],device=self.device)).abs().amax(1)
                            require(float(base_error.max())<=1e-7,'Baseline numerical replay differs beyond original point tolerance')
                            rms.append(arr(delta.square().mean(1).sqrt()));maxima.append(arr(delta.abs().amax(1)));base_errors.append(arr(base_error))
                            for h in HYPOTHESES:
                                old=sub['observations'][h];noise=torch.as_tensor(sub['noise'][h][start:stop],device=self.device)
                                carrier=output['signal_pre']+noise;moving=self.observe(carrier)
                                peaks=torch.as_tensor(old['peak_index'][start:stop],device=self.device,dtype=torch.long)
                                fixed=self.observe(moving['adc_carrier'],charge_ADC=False,peaks=peaks)
                                current=torch.as_tensor(sub['signal'][start:stop],device=self.device)+noise
                                proxy=torch.as_tensor(old['adc_carrier'][start:stop],device=self.device)+(carrier.clamp(ADC_MIN,ADC_MAX)-current.clamp(ADC_MIN,ADC_MAX))
                                continuous=self.observe(proxy,charge_ADC=False,peaks=peaks)
                                for path,o in (('S',continuous),('H_fixed',fixed),('H_moving',moving)):
                                    for key,field in (('wave','window_V'),('baseline','baseline100'),('peak','peak_index')):parts[path][h][key].append(arr(o[field]))
                                    if path!='S':
                                        fields=observation_rows(o,carrier,self.loss)
                                        for key,value in fields.items():
                                            if key.startswith(('flag__','diagnostic__')):hard_flags[path][h].setdefault(key,[]).append(value)
                        arrays=dict(carrier_delta_RMS_V=np.concatenate(rms),carrier_delta_absmax_V=np.concatenate(maxima),baseline_replay_absmax_V=np.concatenate(base_errors),
                            source_parent_row=sub['parent'],draw_index=sub['draw'])
                        joined={path:{h:{key:np.concatenate(v) for key,v in cols.items()} for h,cols in hs.items()} for path,hs in parts.items()}
                        for h in HYPOTHESES:
                            feature_flags={};bins={}
                            for path in joined:
                                o=joined[path][h];arrays[path+'_'+h+'_Q']=o['wave'].sum(1);arrays[path+'_'+h+'_tail']=o['wave'][:,46:].sum(1);arrays[path+'_'+h+'_peak']=o['peak']
                                features,flags=observable_features(o['wave'],o['baseline'],float(self.loss.q_floor));feature_flags[path]=flags
                                for key,value in features.items():arrays[path+'_'+h+'__'+key]=value
                                if path!='S':
                                    for key,values in hard_flags[path][h].items():
                                        value=np.concatenate(values);arrays[path+'_'+h+'__'+key]=value
                                        if key.startswith('flag__'):flags[key[6:]]=value
                                for key,value in flags.items():arrays[path+'_'+h+'__flag__'+key]=value
                                bins[path]=output_q_bins(features['Q_Vns'],self.loss);arrays[path+'_'+h+'__Q_bin']=bins[path]
                            ref_features,_=observable_features(sub['observations'][h]['window_V'],sub['observations'][h]['baseline100'],float(self.loss.q_floor))
                            ref_bins=output_q_bins(ref_features['Q_Vns'],self.loss)
                            for batch in range(p['parameter_validation']['batches']):
                                sl=slice(batch*batch_rows,(batch+1)*batch_rows);metrics={path:self.component(o[h]['wave'][sl],o[h]['baseline'][sl]) for path,o in joined.items()}
                                selfSH=self.component(joined['S'][h]['wave'][sl],joined['S'][h]['baseline'][sl],ref=(joined['H_moving'][h]['wave'][sl],joined['H_moving'][h]['baseline'][sl]))
                                ref=reference_cells[bank,batch,h];row=dict(direction=direction['name'],nominal_scale_LSB=float(target),sign=sign,bank=bank,batch=batch,hypothesis=h,defined=True,
                                    lambda_parameter_scale=sign*scale,paths=metrics,reference=ref,S_H_self_distance=selfSH['values'],
                                    S_delta_objective=metrics['S']['legacy_objective']-ref['legacy_objective'],H_delta_objective=metrics['H_moving']['legacy_objective']-ref['legacy_objective'],
                                    hard_component_delta=signed_changes(metrics['H_moving']['values'],ref['values']),
                                    guard_counts={path:flag_rates({k:v[sl] for k,v in flags.items()}) for path,flags in feature_flags.items()},
                                    changed_crop_rows=int((joined['H_moving'][h]['peak'][sl]!=sub['observations'][h]['peak_index'][sl]).sum()),
                                    output_Q_bin_migration={path:int((value[sl]!=ref_bins[sl]).sum()) for path,value in bins.items()},
                                    actual_carrier_delta_RMS_V=float(np.sqrt(np.mean(np.concatenate(rms)[sl]**2))),actual_carrier_delta_absmax_V=float(np.concatenate(maxima)[sl].max()))
                                records.append(row)
                                if direction['name'] in p['parameter_validation']['primary_directions'] and sign==1 and target==p['parameter_validation']['primary_scale_LSB']:primary.append(row)
                        filename='PARAMETER_%s_%g_%+d_bank%d.npz'%(direction['name'],target,sign,bank);self.store.npz(filename,**arrays)
                        require(reference_snapshot(self.model,self.ext)==self.immutable_start,'Parameter probe mutated original frozen tensors')
        self.store.npz('PARAMETER_DIRECTIONS.npz',**{d['name']:d['vector'] for d in directions},reference_head=theta)
        selection=select_method_once(primary,tolerance,p['parameter_validation']['primary_directions'],undefined_directions=undefined)
        self.store.json('PARAMETER_VALIDATION.json',dict(records=records,directions=[{k:v for k,v in d.items() if k!='vector'} for d in directions],
            reference_head_G_updates=2000,actual_optimizer_or_training_steps=0,actual_new_AD_queries=0,carrier_probe=False,
            inherited_gradient_scope='012_fixed_saved_bank_not_new_bank',extra_parameter_states=0))
        self.store.json('METHOD_SELECTION.json',selection);return selection


def main(argv=None):
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args(argv)
    p,paths,checked=gate(args.config,args.config_sha256);job=args.output
    require(p.get('artificial_only',False) or (os.name=='posix' and job==Path(p['remote_root'])/'runs/primary'),'Only registered isolated native job output is legal')
    require(job.is_dir() and not (job/'artifacts').exists(),'Owned job parent must exist, artifact leaf fresh')
    store=Store(job/'artifacts',p,checked,artificial=bool(p.get('artificial_only',False)))
    try:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from wp1_input_information import run_information
        from wp1_boundary_saved import run_saved_boundary
        store.phase='information_analysis';store.status();run_information(args.config,args.config_sha256,job/'information')
        store.phase='saved_boundary';store.status();run_saved_boundary(args.config,args.config_sha256,job/'boundary')
        import torch
        torch.set_num_threads(p['resource_limits']['CPU_threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        if str(args.device).startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(p['resource_limits']['GPU_GiB']*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ctx=prepare_numeric(paths,p,store,args.device);info_path=job/'information/PARENT_INFORMATION.npz'
        checked[str(info_path)]=sha(info_path);info=information_join(ctx['data'],info_path)
        evaluation=job/'evaluation';require(not evaluation.exists(),'Preserve evaluation leaf');evaluation.mkdir()
        runtime=Runtime(p,ctx,store,evaluation,info)
        store.phase='full_TRAIN_reference';store.status();runtime.references_forward()
        store.phase='unified_hard_evaluation';store.status();runtime.complete_evaluations()
        store.phase='TRAIN_artificial_tolerance_mapping';store.status();tolerance=runtime.calibrate()
        store.phase='actual_parameter_validation';store.status();selection=runtime.parameter_states(tolerance)
        require(reference_snapshot(ctx['model'],ctx['extension'])==ctx['snapshot'],'Final frozen state changed');unchanged(checked)
        store.phase='complete';store.status(dict(engineering_checks_passed=True,inputs_unchanged=True,frozen_state_unchanged=True,method_choice=selection['choice']))
        array_headers={}
        import numpy as np
        for leaf in ('information','boundary','evaluation','artifacts'):
            for q in (job/leaf).glob('*.npz'):
                members={}
                with zipfile.ZipFile(q) as archive:
                    for name in archive.namelist():
                        with archive.open(name) as f:
                            version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
                        members[name.removesuffix('.npy')]=dict(shape=list(shape),dtype=str(dtype),fortran_order=fortran)
                array_headers[q.relative_to(job).as_posix()]=members
        store.json('ARRAY_SCHEMAS.json',array_headers)
        inventory={q.relative_to(job).as_posix():dict(bytes=q.stat().st_size,sha256=sha(q)) for leaf in ('information','boundary','evaluation','artifacts') for q in (job/leaf).rglob('*') if q.is_file()}
        complete=dict(store.base(),engineering_checks_passed=True,inputs_unchanged=True,frozen_state_unchanged=True,artifacts=inventory,
            reference_head_G_updates=2000,all_SOURCE_TRAIN_parents=6593,all_REAL_TRAIN_rows=21047,banks=2,probe_registered_states=30,
            probe_defined_states=runtime.defined_states,calibration_cases=21,method_choice=selection['choice'],actual_training_updates=0,formal_response_acceptance=False)
        complete_bytes=len((json.dumps(clean(complete),indent=2,allow_nan=False)+'\n').encode('utf8'))
        owned_size=sum(q.stat().st_size for q in job.rglob('*') if q.is_file())
        require(owned_size+complete_bytes<=p['resource_limits']['output_MiB']*1024**2,'Final owned-job output cap exceeded')
        require(time.monotonic()-started<=p['resource_limits']['wall_seconds'],'Final driver wall cap exceeded')
        store.json('COMPLETE.json',complete)
        print(json.dumps(dict(status='complete',method_choice=selection['choice'],scientific_training_updates=0)))
    except BaseException as error:
        store.error=repr(error);store.phase='failed';store.status();store.json('FAILED.json',dict(store.base(),error_type=type(error).__name__,error=str(error),
            completed_counts_are_durable=True,unknown_inflight_retained=True,automatic_restart=False))
        raise

if __name__=='__main__':main()
