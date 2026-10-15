"""Bounded M0/M1 pairs; a saved-TRAIN mean gate precedes all CP/NN/training.

The immutable 014 runtime, C1 loss, samplers, streams, and full hard evaluation
are reused. This file never resumes a failed run or reads original DEV arrays.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import run_wp2_014 as old
from wp1_ops import Store, clean, read_json, require, sha, unchanged, write_new

RUN_ID='wp3_015_20261015'
HYPOTHESES=('fast','slow','common_level')
OWNED_FILES=('scripts/run_wp3_015.py','src/n01/model015.py','scripts/wp3_controller.py')
TOLERANCE_SHA015=old.TOLERANCE_SHA014


def seed_set015(initialization):
    return dict(initialization=initialization,source_sampler=initialization+1,real_sampler=initialization+2,
        train_shape=initialization+3,train_noise005=initialization+4,train_extension=initialization+5,
        train_projection=initialization+6,integral_projection=initialization+7)


def settings015():
    return dict(schema='n01/wp3-015-config/v1',run_id=RUN_ID,status='draft',candidates=['M0','M1'],
        seed_sets=[seed_set015(2026101540),seed_set015(2026101640)],G_updates=2000,
        TRAIN_counts=dict(source=6593,real=21047,REAL_logical_groups=422),
        training=dict(source_parents_per_update=512,draws_per_parent=2,real_rows_per_update=1024,
            forward_chunk_rows=256,G_lr=.0002,G_weight_decay=.0001,clip_norm=5.,
            wave_weight=1.,distribution_weight=2.,regular_weight=.01,status_interval=100,
            checkpoint_steps=[1000,2000],fresh_AdamW=True,inherit_014_optimizer=False),
        mean_adapter=dict(parameters=6,zero_initialization=True,control_Q_slice=[110,366],control_tail_slice=[156,366],
            Gaussian_sigma_ns=2.,Gaussian_length=13,causal_tau_ns=16.,causal_length=64,
            coefficient_limit=.1,RMS_trust_fraction=.1,epsilon_V=1e-12,mean_regular_weight=.01,
            M0_trainable_parameters=1288,M1_trainable_parameters=6,stochastic_amplitude_from_original_A=True),
        mean_evidence=dict(REPORT_mean_member='root/repeated_parent/hard_wave_mean_V',
            REPORT_parent_member='root/repeated_parent/parent_rows',minimum_RMS_V=1e-7,
            SE_multiplier=3.,minimum_bank_residual_cosine=.5,ALL_seed_hypothesis_cells=6,
            real_cluster_variance='G/(G-1)*sum_g(sum_i_in_g(y_i-row_mean))^2/N^2',
            group_is_independent_acquisition=False,not_conditional_mean_identification=True),
        evaluation=dict(parents=6593,draws=4,banks=2,chunk_rows=128,bank_seeds=[],projection_seed=None,
            whole_population=True,snapshot_arms=dict(initial=['A','initial'],final=['final'])),
        hypotheses=list(HYPOTHESES),CPU_threads=12,
        run_resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=7200,output_MiB=1536),
        resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=28800,output_MiB=6144,
            poll_seconds=10,TERM_grace_seconds=20),
        one_active_scientific_job=True,serial_runs=True,automatic_restart=False,overwrite_outputs=False,
        DEV_numeric_read=False,scientific_status='response_match_not_established')


def budget_caps015(p):
    runs=2*len(p['seed_sets']);t=p['training'];e=p['evaluation'];h=len(HYPOTHESES)
    train=runs*p['G_updates']*t['source_parents_per_update']*t['draws_per_parent']
    evaluation=2*runs*e['banks']*e['parents']*e['draws']
    signal=train+evaluation
    mean=len(p['seed_sets'])*(p['G_updates']*t['source_parents_per_update']*t['draws_per_parent']
        +2*e['banks']*e['parents']*e['draws'])
    return dict(old_A_rows=signal,head_actual_parameter_forward_rows=signal,head_total_kernel_rows=signal,
        noise005_rows=signal,mean_adapter_rows=mean,
        hard_ADC_rows=h*train+3*runs*e['banks']*e['parents']*e['draws']*h,
        all_path_gather_rows=h*train+3*runs*e['banks']*e['parents']*e['draws']*h,
        checkpoint_loads=2+len(p['seed_sets']),optimizer_created=runs,optimizer_steps=runs*p['G_updates'],
        G_updates=runs*p['G_updates'],training_backward_queries=runs*p['G_updates'],new_AD_queries=runs*p['G_updates'],
        calibration_map_rows=0,integral_statistics_fits=0,probe_AD_queries=0,
        mean_evidence_REPORT_files=12,mean_evidence_REPORT_parent_rows=12*e['parents'],
        mean_evidence_REAL_rows=p['TRAIN_counts']['real'],mean_evidence_tests=6,
        D_updates=0,noise_updates=0,population_updates=0,C2ST_updates=0,DEV_model_rows=0)


def validate_settings015(p,*,frozen=True):
    reference=settings015()
    require(p.get('schema')==reference['schema'] and p.get('run_id')==RUN_ID,'015 identity differs')
    require(not frozen or p.get('status')=='frozen','015 draft cannot run')
    for key in reference:
        if key in ('status','evaluation'):continue
        require(p.get(key)==reference[key],'015 registered setting differs: '+key)
    for key in reference['evaluation']:
        if key in ('bank_seeds','projection_seed'):continue
        require(p['evaluation'].get(key)==reference['evaluation'][key],'015 evaluation differs: '+key)
    require(len(p['evaluation']['bank_seeds'])==2 and type(p['evaluation']['projection_seed']) is int,
        '015 fixed evaluation banks/projection required')
    require(p.get('budget_caps')==budget_caps015(p),'015 exact operation caps differ')


def gate015(config,expected_sha,output):
    cp=Path(config).resolve();require(cp.is_file() and sha(cp)==expected_sha,'015 config SHA differs')
    p=read_json(cp);validate_settings015(p)
    required={'dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information',
        'integral_statistics','C1_head_seed0','C1_head_seed1'}
    required.update(f'mean_seed{s}_bank{b}_{h}' for s in range(2) for b in range(2) for h in HYPOTHESES)
    require(required<=set(p.get('inputs',{})),'015 bound inputs incomplete')
    require({'observation','data_roles','acceptance','tolerance_table'}<=set(p.get('contracts',{})),'015 contracts incomplete')
    require(p['contracts']['tolerance_table']['sha256']==TOLERANCE_SHA015,'015 frozen pre-candidate T changed')
    require(set(OWNED_FILES)<=set(p.get('source_files',{})),'015 own sources not frozen')
    roots=[Path(r).resolve() for r in p['allowed_input_roots']]
    require(cp.parent in roots,'015 launch root is not allowed')
    checked={str(cp):expected_sha};paths={}
    records={**{'source:'+k:v for k,v in p['source_files'].items()},**{'input:'+k:v for k,v in p['inputs'].items()},
        **{'contract:'+k:v for k,v in p['contracts'].items()},**{'protected:'+k:v for k,v in p.get('protection_files',{}).items()}}
    for label,row in records.items():
        path=Path(row['path']);resolved=path.resolve()
        require(path.is_absolute() and resolved.is_file() and not path.is_symlink()
            and any(resolved.is_relative_to(r) for r in roots),'015 input/source escaped roots: '+label)
        require(os.name!='nt' or path.drive.upper() in ('E:','D:'),'015 all files must stay E/D')
        require(sha(path)==row['sha256'] and ('bytes' not in row or path.stat().st_size==row['bytes']),
            '015 frozen bytes changed: '+label)
        if label.startswith('source:'):
            require(resolved==(Path(__file__).resolve().parents[1]/label[7:]).resolve(),'015 imported source path mismatch')
        key=str(resolved);require(key not in checked or checked[key]==row['sha256'],'015 conflicting SHA alias')
        checked[key]=row['sha256']
        if label.startswith('input:'):paths[label[6:]]=resolved
    target=Path(output).resolve()
    require(not target.exists() and target!=cp.parent and target.is_relative_to(cp.parent),'015 fresh output inside launch root required')
    require(os.name!='nt' or target.drive.upper() in ('E:','D:'),'015 output must stay E/D')
    return p,paths,checked


class Store015(old.Store014):
    def base(self):
        row=super().base();row.update(schema='n01/wp3-015-state/v1',scope='paired C1 continuation versus six-parameter mean adapter')
        return row


def population_mean_gate015(real,groups,means,settings):
    """Pure finite working-cluster population evidence; no random sampling."""
    import numpy as np
    real=np.asarray(real,np.float64);groups=np.asarray(groups)
    require(real.ndim==2 and real.shape[1]==256 and len(groups)==len(real) and np.isfinite(real).all(),
        '015 invalid REAL mean evidence')
    unique,inverse=np.unique(groups,return_inverse=True);G=len(unique);N=len(real)
    require(G>1,'015 cluster sandwich needs at least two logical groups')
    mu=real.mean(0);scores=np.zeros((G,256),np.float64)
    np.add.at(scores,inverse,real-mu)
    variance=G/(G-1)*np.square(scores).sum(0)/N**2
    real_se=float(np.sqrt(variance.mean()));rows=[]
    for seed in range(2):
        for h in HYPOTHESES:
            bank=[np.asarray(means[(seed,b,h)],np.float64) for b in range(2)]
            require(all(v.ndim==2 and v.shape[1]==256 and np.isfinite(v).all() for v in bank),'015 nonfinite REPORT mean')
            m=[v.mean(0) for v in bank];residual=[v-mu for v in m]
            center=(m[0]+m[1])*.5-mu
            rms=float(np.sqrt(np.square(center).mean()));bank_se=float(np.sqrt(np.square(m[0]-m[1]).mean())/math.sqrt(2))
            denominator=float(np.linalg.norm(residual[0])*np.linalg.norm(residual[1]))
            cosine=None if denominator==0 else float(residual[0]@residual[1]/denominator)
            bound=max(settings['minimum_RMS_V'],settings['SE_multiplier']*math.sqrt(bank_se**2+real_se**2))
            passed=rms>bound and cosine is not None and cosine>settings['minimum_bank_residual_cosine']
            rows.append(dict(seed=seed,hypothesis=h,residual_RMS_V=rms,bank_MC_SE_RMS_V=bank_se,
                real_cluster_SE_RMS_V=real_se,required_residual_RMS_V=bound,bank_residual_cosine=cosine,
                hard256_residual_DC_V=float(center.mean()),hard256_DC_is_not_electronics_baseline100=True,
                passed=passed,reason='finite_population_mean_evidence' if passed else 'registered_mean_evidence_not_met'))
    return dict(schema='n01/wp3-mean-residual-evidence/v1',all6_passed=all(v['passed'] for v in rows),cells=rows,
        REAL_rows=N,REAL_logical_groups=G,REAL_mean_wave256_V=mu,REAL_cluster_variance_of_mean_V2=variance,
        real_cluster_SE_RMS_V=real_se,group_is_independent_acquisition=False,
        conditional_mean_identified=False,paired_REAL_measurements_available=False,
        baseline100_DC_evidence='reported separately by complete hard evaluation; hard256 DC is not baseline100',
        finite_budget_rationale_only=True,scientific_acceptance=False)


def mean_evidence015(p,paths,store):
    import numpy as np
    allowed=('real','real_groups','real_role','real_ids')
    def read_real():
        with np.load(paths['dataset'],allow_pickle=False) as a:return {n:a[n].copy() for n in allowed}
    data=store.call('REAL_TRAIN_population_mean_evidence_read',dict(mean_evidence_REAL_rows=p['TRAIN_counts']['real']),read_real)
    require(data['real'].shape==(21047,256) and np.unique(data['real_groups']).size==422
        and set(np.unique(data['real_role']))=={'TRAIN'} and np.unique(data['real_ids']).size==21047,'015 REAL mean evidence role/count mismatch')
    means={};e=p['mean_evidence']
    for seed in range(2):
        for bank in range(2):
            for h in HYPOTHESES:
                key=f'mean_seed{seed}_bank{bank}_{h}'
                def read_mean(key=key):
                    with np.load(paths[key],allow_pickle=False) as a:
                        parents=a[e['REPORT_parent_member']].copy();value=a[e['REPORT_mean_member']].copy()
                    require(np.array_equal(parents,np.arange(6593)) and value.shape==(6593,256),
                        '015 saved mean parent identity/order changed')
                    return value
                means[(seed,bank,h)]=store.call('saved014_C1_population_mean_read',
                    dict(mean_evidence_REPORT_files=1,mean_evidence_REPORT_parent_rows=6593),read_mean,input_key=key)
    result=store.call('six_registered_population_mean_tests',dict(mean_evidence_tests=6),
        lambda:population_mean_gate015(data['real'],data['real_groups'],means,e))
    dense={k:result.pop(k) for k in ('REAL_mean_wave256_V','REAL_cluster_variance_of_mean_V2')}
    store.npz('MEAN_EVIDENCE_REAL_SUMMARY.npz',**dense)
    result.update(actual_dataset_members_decoded=list(allowed),REPORT_members_decoded=[e['REPORT_mean_member'],e['REPORT_parent_member']],
        before_checkpoint_load=True,before_NN=True,before_new_random=True,before_optimizer=True,
        source_numeric_decoded=False,DEV_numeric_decoded=False)
    store.json('MEAN_RESIDUAL_GATE.json',result)
    return result


class Runtime015(old.Runtime014):
    def forward(self,model,x,z):
        original=model.mean_adapter
        def adapter(source,base_signal,*,enabled=True):
            if not enabled:return original(source,base_signal,enabled=False)
            return self.call('six_parameter_mean_adapter',dict(mean_adapter_rows=len(source)),
                lambda:original(source,base_signal,enabled=True),arm=model.arm)
        model.mean_adapter=adapter
        try:return super().forward(model,x,z)
        finally:del model.mean_adapter


def prepare_numeric015(p,paths,store,device):
    """The original 014 loader without its single integral-statistics fit."""
    import numpy as np
    import torch
    from n01.model008 import N01Response008
    from n01.response008 import make_extension,statistics_digest,state_digest
    from n01.diagnose_mean_random009 import read_train_members,validate_train
    from run_wp1_013 import information_join,REAL_METADATA
    stats,domain,cfg=[read_json(paths[k]) for k in ('statistics','data_statistics','base_config')]
    require(stats.get('statistics_split')==domain.get('split')==stats['loss']['fit_split']=='TRAIN'
        and stats.get('DEV_waveforms_decoded') is False and domain.get('DEV_waveforms_decoded') is False
        and stats['prepared_dataset_sha256']==p['inputs']['dataset']['sha256']
        and stats['data_statistics_sha256']==p['inputs']['data_statistics']['sha256'],'015 original TRAIN stats provenance mismatch')
    with np.load(paths['dataset'],allow_pickle=False) as a:
        data=read_train_members(a)
        for name in REAL_METADATA:data[name]=a[name].copy()
    validate_train(data)
    data['real_observation_flags']=dict(peak_tied=data['real_peak_tied_samples']>1,
        peak_at_search_boundary=data['real_peak_search_edge'],global_peak_outside_search=data['real_global_peak_outside_search'])
    def load(key):return store.call('checkpoint_'+key,dict(checkpoint_loads=1),lambda:torch.load(paths[key],map_location='cpu',weights_only=False))
    a=load('deterministic')
    require(a.get('schema')=='response008/v1' and a.get('arm')=='deterministic' and a.get('G_updates')==19000
        and a.get('stage')=='low_learning_rate' and a.get('D_updates')=={h:16000 for h in HYPOTHESES}
        and a.get('noise_updates')==0 and a.get('population_learning') is False and a.get('config')==cfg
        and statistics_digest(a['statistics'])==statistics_digest(stats),'015 frozen A metadata mismatch')
    baseline=N01Response008(stats['model'],'deterministic');baseline.load_state_dict(a['model'],strict=True)
    baseline.set_stage(a['stage']);baseline.eval();baseline.requires_grad_(False);baseline.to(device)
    noise=load('noise005')
    require(noise.get('schema_version')==5 and noise.get('update')==5000
        and noise.get('model_type')=='ObservedBaselineNoiseV3' and noise.get('engineering') is False,'015 005 metadata mismatch')
    extension=make_extension(noise,device)
    require(state_digest(extension.model005.state_dict())==a['noise005_state_sha256'],'015 frozen A/005 binding mismatch')
    integral=read_json(paths['integral_statistics']);store.json('INTEGRAL_TRAIN_STATISTICS_REFERENCE.json',integral)
    info=information_join(data,paths['source_information'])
    return data,baseline,extension,stats,integral,info


class Engine015(old.Engine014):
    def __init__(self,runtime,seed,arm,initial_head,reference,directory):
        import torch
        from n01.data import FullCoverageSampler
        from n01.model015 import FrozenC1MeanResponse015
        from n01.losses014 import N01Loss014
        from n01.response008 import loss_tensor_statistics
        from n01.train_shape010 import Streams010
        self.rt,self.seed,self.candidate,self.out=runtime,seed,arm,Path(directory);self.out.mkdir()
        self.started,self.update=time.monotonic(),0;self.reference=reference
        self.model=FrozenC1MeanResponse015(runtime.baseline,initial_head,arm=arm,seed=seed['initialization'],
            q_reference=runtime.statistics['loss']['q_reference'],q_floor=runtime.statistics['loss']['q_floor'])
        self.loss=N01Loss014(loss_tensor_statistics(runtime.statistics['loss']),runtime.integral_statistics,
            seed=seed['train_projection'],evaluation_seed=runtime.p['evaluation']['projection_seed'],integral_seed=seed['integral_projection']).to(runtime.device)
        self.source_sampler=FullCoverageSampler(len(runtime.data['source']),seed['source_sampler'])
        self.real_sampler=FullCoverageSampler(len(runtime.data['real']),seed['real_sampler'])
        self.streams=Streams010(seed['train_shape'],seed['train_noise005'],seed['train_extension'])
        t=runtime.p['training'];self.optimizer=runtime.call('fresh_arm_AdamW_created',dict(optimizer_created=1),
            lambda:torch.optim.AdamW(list(self.model.trainable_parameters()),lr=t['G_lr'],weight_decay=t['G_weight_decay']))
        self.model.assert_optimizer_membership(self.optimizer);self.last_metrics={}
        write_new(self.out/'RUN_INPUTS.json',dict(schema='n01/wp3-run-inputs/v1',arm=arm,seed=seed,reference=reference,
            shape_reference_head_sha256=self.model.reference_C1_head_sha256,mean_theta_initial=[0.]*6,
            trainable_parameter_names=list(self.model.trainable_parameter_names()),trainable_parameters=sum(v.numel() for v in self.model.trainable_parameters()),
            frozen_A_005_sha256=runtime.frozen,fresh_AdamW=True,inherit_014_optimizer=False,DEV_numeric_read=False,
            same_seed_reset_streams=True,reference_is_development_only=True,weights_promoted=False))

    def checkpoint(self,name,purpose):
        from n01.train_shape010 import cpu_tree
        old.atomic_checkpoint(self.out/name,dict(schema='n01/wp3-checkpoint/v1',arm=self.candidate,
            initial_seed=self.seed['initialization'],G_updates=self.update,purpose=purpose,
            new_head=cpu_tree(self.model.basis_head.state_dict()),mean_theta=cpu_tree(self.model.mean_theta),
            trainable_parameter_names=list(self.model.trainable_parameter_names()),optimizer=cpu_tree(self.optimizer.state_dict()),
            source_sampler=self.source_sampler.state_dict(),real_sampler=self.real_sampler.state_dict(),streams=self.streams.state_dict(),
            last_metrics=self.last_metrics,reference=self.reference,frozen_state_sha256=self.rt.frozen,counters=self.rt.store.counts.copy(),
            inflight=copy.deepcopy(self.rt.store.inflight),complete_step_boundary=purpose!='failure',automatic_restart=False))

    def step(self):
        import numpy as np
        import torch
        from n01.response008 import saturation_penalty
        rt,model,t=self.rt,self.model,self.rt.p['training'];rt.guard(self.out,self.started)
        parents=self.source_sampler.draw(t['source_parents_per_update']);si=np.repeat(parents,t['draws_per_parent'])
        ri=self.real_sampler.draw(t['real_rows_per_update']);z,nz,epsilon=self.streams.draw(len(si),rt.device)
        noise=rt.noise(nz,epsilon);self.optimizer.zero_grad(set_to_none=True);model.train();outputs=[]
        for start in range(0,len(si),t['forward_chunk_rows']):
            x=torch.as_tensor(rt.data['source_carrier384'][si[start:start+t['forward_chunk_rows']]],dtype=torch.float64,device=rt.device)
            outputs.append(rt.forward(model,x,z[start:start+len(x)]))
        signal_value=torch.cat([o['signal_pre'] for o in outputs])
        shape_regular=sum(o['stochastic_shape_regular']*len(o['signal_pre'])/len(si) for o in outputs)
        mean_regular=sum(o['mean_adapter_regular']*len(o['signal_pre'])/len(si) for o in outputs)
        context=self.loss.create_projection_context(self.update+1)
        real,rb=[torch.as_tensor(rt.data[n][ri],dtype=torch.float64,device=rt.device) for n in ('real','real_baseline100')]
        waves=[];distributions=[];saturations=[];metrics={}
        for h in HYPOTHESES:
            obs=rt.observe(signal_value+noise[h],ste=True)
            parts=self.loss.training_components(dict(waveform=obs['window_V'],baseline100=obs['baseline100']),real,
                candidate='C1',projection_context=context,real_baseline100=rb)
            waves.append(parts['wave']);distributions.append(parts['distribution']);saturations.append(saturation_penalty(signal_value+noise[h]))
            metrics[h]=dict(wave=float(parts['wave'].detach()),training_distribution=float(parts['distribution'].detach()),
                QPSD_native=float(parts['QPSD_native'].detach()),diagnostics=parts['diagnostics'])
        wave=torch.stack(waves).mean();distribution=torch.stack(distributions).mean()
        regular=shape_regular+mean_regular+torch.stack(saturations).mean()
        objective=wave+2*distribution+t['regular_weight']*regular
        require(bool(torch.isfinite(objective)),'015 nonfinite objective')
        rt.call('one_training_backward',dict(training_backward_queries=1,new_AD_queries=1),lambda:objective.backward())
        params=list(model.trainable_parameters());gradient=torch.nn.utils.clip_grad_norm_(params,t['clip_norm'],error_if_nonfinite=True)
        before=torch.cat([v.detach().reshape(-1).double().clone() for v in params]);model.assert_optimizer_membership(self.optimizer)
        rt.call('one_arm_AdamW_step',dict(optimizer_steps=1,G_updates=1),lambda:self.optimizer.step());self.update+=1
        after=torch.cat([v.detach().reshape(-1).double() for v in params]);delta=after-before
        require(bool(torch.isfinite(after).all()),'015 nonfinite updated parameters')
        self.last_metrics=dict(G_updates=self.update,candidate=self.candidate,initial_seed=self.seed['initialization'],
            objective=float(objective.detach()),wave=float(wave.detach()),weighted_distribution=float((2*distribution).detach()),
            weighted_regular=float((t['regular_weight']*regular).detach()),weighted_stochastic_regular=float((.01*shape_regular).detach()),
            weighted_mean_regular=float((.01*mean_regular).detach()),hypotheses=metrics,
            preclip_total_gradient_norm=float(gradient),actual_parameter_update_norm=float(delta.norm()),
            actual_parameter_update_absmax=float(delta.abs().max()),parameters_changed=bool(delta.ne(0).any()),
            mean_theta=old.arr(model.mean_theta).tolist(),mean_relative_RMS_max=max(float(o['mean_RMS_relative_A'].detach().max()) for o in outputs),
            mean_trust_scale_min=min(float(o['mean_trust_scale'].detach().min()) for o in outputs),
            source_parent_rows=parents.tolist(),real_row_count=len(ri),coverage=self.coverage(),projection_context=context.metadata(),
            trainable_parameters=sum(v.numel() for v in params),all_legal_TRAIN_sampling=True,D_updates=0,noise_updates=0)
        if self.update==1 or self.update%t['status_interval']==0:
            rt.verify(model,files=self.update==1)
            with (self.out/'training.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(clean(self.last_metrics),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
            rt.store.status(dict(current_run=self.out.name,current_run_G_updates=self.update))
        if self.update==1:
            require(float(gradient)>0 and bool(delta.ne(0).any()),'015 first step has no gradient or update')
            proof=dict(rt.store.base(),schema='n01/wp3-first-real-update/v1',actual_G_updates=1,
                batch_completed_G_updates=rt.store.counts['G_updates'],trial=self.out.name,candidate=self.candidate,
                initial_seed=self.seed['initialization'],frozen_A_005_unchanged=True,
                protected_C1_shape_unchanged=self.candidate=='M1',gradient_nonzero=True,actual_parameter_update_nonzero=True,
                trainable_parameters=sum(v.numel() for v in params),trainable_parameter_names=list(model.trainable_parameter_names()),
                mean_gate_all6_passed=True,frozen_state_sha256=rt.frozen,reference=self.reference,metrics=self.last_metrics)
            write_new(self.out/'FIRST_UPDATE_PROOF.json',proof)
            if not (rt.store.root/'FIRST_UPDATE_PROOF.json').exists():rt.store.json('FIRST_UPDATE_PROOF.json',proof)
        context.validate();return self.last_metrics

    def evaluate(self,*,initial):
        result=super().evaluate(initial=initial)
        directory=self.out/('initial_evaluation' if initial else 'final_evaluation');rows=[]
        for file in sorted(directory.glob('*_REPORT.json')):
            report=read_json(file);noise=report['population']['noise']
            rows.append(dict(report=file.name,fake_baseline_mean_V=noise['fake']['baseline_mean_V'],
                REAL_baseline_mean_V=noise['real']['baseline_mean_V'],signal_adapter_prefix100_exact_zero=True,
                electronics_005_frozen=True,mean_adapter_cannot_repair_electronics_baseline=True))
        write_new(directory/'ELECTRONICS_LIMITS.json',dict(schema='n01/wp3-electronics-limit/v1',rows=rows,
            original_baseline100_not_pulse_first100=True,mean_adapter_does_not_modify_005=True))
        return result


def artifact_inventory015(store):
    inventory={str(q.relative_to(store.root)).replace('\\','/'):dict(sha256=sha(q),bytes=q.stat().st_size)
        for q in sorted(store.root.rglob('*')) if q.is_file() and q.name!='status.json'}
    store.json('ARTIFACT_INVENTORY.json',dict(schema='n01/wp3-artifact-inventory/v1',files=inventory,
        raw_private_arrays_remain_on_execution_host=True))


def run015(config,config_sha256,output,device):
    p,paths,checked=gate015(config,config_sha256,output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(p['CPU_threads'])
    store=Store015(output,p,checked);store.status();engine=None
    try:
        store.json('BINDING.json',dict(configuration_sha256=config_sha256,immutable_files=checked,budget_caps=p['budget_caps'],before_numeric_decode=True))
        evidence=mean_evidence015(p,paths,store)
        if not evidence['all6_passed']:
            unchanged(checked)
            artifact_inventory015(store)
            result=dict(store.base(),schema='n01/wp3-batch-complete/v1',native_status='COMPLETE',training_skipped=True,
                reason='registered_population_mean_evidence_not_met',trials=[],all_four_trials_completed=False,
                no_checkpoint_NN_optimizer_AD_G=True,mean_evidence_sha256=sha(store.root/'MEAN_RESIDUAL_GATE.json'),scientific_acceptance=False)
            store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE',training_skipped=True));return result
        import torch
        from n01.evaluate_window008 import _memory_limit
        from n01.train_shape010 import cpu_tree
        from n01.model010 import tensor_state_digest010
        torch.set_num_threads(p['CPU_threads']);torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']),device)
        data,baseline,extension,stats,integral,info=prepare_numeric015(p,paths,store,device)
        runtime=Runtime015(p,data,baseline,extension,stats,integral,store,device,info)
        tolerance=read_json(p['contracts']['tolerance_table']['path']);completed=[]
        for index,seed in enumerate(p['seed_sets']):
            key='C1_head_seed%d'%index
            checkpoint=runtime.call('checkpoint_'+key,dict(checkpoint_loads=1),lambda key=key:torch.load(paths[key],map_location='cpu',weights_only=False))
            require(checkpoint.get('schema')=='n01/wp2-checkpoint/v1' and checkpoint.get('candidate')=='C1'
                and checkpoint.get('G_updates')==2000 and checkpoint.get('initial_seed')==seed['initialization']
                and checkpoint.get('complete_step_boundary') is True and checkpoint.get('purpose')=='final'
                and checkpoint.get('frozen_state_sha256')==runtime.frozen,'015 C1 head reference metadata mismatch')
            initial=cpu_tree(checkpoint['new_head']);reference=dict(input_key=key,sha256=p['inputs'][key]['sha256'],
                WP2_G_updates=2000,optimizer_inherited=False,development_only=True,weights_promoted=False)
            del checkpoint
            for arm in p['candidates']:
                engine=Engine015(runtime,seed,arm,initial,reference,store.root/f'seed{index}_{arm}')
                if not (store.root/'INIT_PROOF.json').exists():
                    store.json('INIT_PROOF.json',dict(store.base(),schema='n01/wp3-init-proof/v1',native_initialized=True,
                        mean_gate_all6_passed=True,first_trial=engine.out.name,shape_reference_head_sha256=engine.model.reference_C1_head_sha256,
                        zero_mean_adapter=True,trainable_parameters=sum(v.numel() for v in engine.model.trainable_parameters()),
                        trainable_parameter_names=list(engine.model.trainable_parameter_names()),paired_initialization_bytes=True,
                        frozen_state_sha256=runtime.frozen,reference=reference,fresh_AdamW=True))
                engine.checkpoint('initial.pt','initial');initial_metrics=engine.evaluate(initial=True)
                while engine.update<p['G_updates']:
                    engine.step()
                    if engine.update in p['training']['checkpoint_steps']:
                        runtime.verify(engine.model,files=True)
                        engine.checkpoint('final.pt' if engine.update==p['G_updates'] else 'step_%04d.pt'%engine.update,
                            'final' if engine.update==p['G_updates'] else 'periodic')
                final_metrics=engine.evaluate(initial=False);comparison=old.compare_metrics014(initial_metrics,final_metrics,tolerance)
                write_new(engine.out/'HARD_COMPARISON.json',dict(schema='n01/wp3-hard-comparison/v1',candidate=arm,
                    initial_seed=seed['initialization'],per_bank_hypothesis=comparison,
                    common_initial_is_frozen014_C1_final=True,frozen_tolerance=p['contracts']['tolerance_table'],scientific_acceptance=False))
                coverage=engine.coverage();require(coverage['source_unique']==6593 and coverage['real_unique']==21047,'015 incomplete TRAIN coverage')
                runtime.verify(engine.model,files=True)
                result=dict(schema='n01/wp3-trial-complete/v1',native_status='COMPLETE',candidate=arm,seed=seed,G_updates=engine.update,
                    coverage=coverage,frozen_A_005_unchanged=True,frozen_state_sha256=runtime.frozen,
                    protected_C1_shape_unchanged=arm=='M1',mean_theta=old.arr(engine.model.mean_theta).tolist(),
                    trainable_parameter_names=list(engine.model.trainable_parameter_names()),last_metrics=engine.last_metrics,
                    per_bank_hypothesis=comparison,resources=runtime.guard(engine.out,engine.started),
                    raw_artifacts_retained_on_execution_host=True,automatic_restart=False,scientific_acceptance=False,
                    scientific_status='response_match_not_established',root_selection_pending=True)
                write_new(engine.out/'COMPLETE.json',result)
                completed.append(dict(directory=engine.out.name,candidate=arm,initial_seed=seed['initialization'],G_updates=engine.update,
                    COMPLETE_sha256=sha(engine.out/'COMPLETE.json')))
                engine=None
                if str(device).startswith('cuda'):torch.cuda.empty_cache()
        require(not store.inflight,'015 cannot complete with unknown in-flight operations');unchanged(checked)
        for key,cap in p['budget_caps'].items():require(store.counts[key]==cap,'015 completed operation count differs: '+key)
        require(time.monotonic()-store.started<=p['resource_limits']['wall_seconds'],'015 final batch wall cap exceeded')
        artifact_inventory015(store)
        result=dict(store.base(),schema='n01/wp3-batch-complete/v1',native_status='COMPLETE',training_skipped=False,
            trials=completed,G_updates_per_trial=p['G_updates'],all_four_trials_completed=True,
            all_frozen_A_005_unchanged=True,mean_gate_all6_passed=True,no_DEV=True,scientific_acceptance=False,
            scientific_status='response_match_not_established',root_choice_pending=True)
        store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=p['resource_limits']['output_MiB']*1024**2,
            '015 final batch output cap exceeded')
        return result
    except BaseException as error:
        failure=dict(store.base(),schema='n01/wp3-batch-failure/v1',native_status='FAILED',error_type=type(error).__name__,error=str(error),
            automatic_restart=False,completed_G_updates=store.counts.get('G_updates',0),current_trial=None if engine is None else engine.out.name,
            current_trial_G_updates=None if engine is None else engine.update,inflight_completion='unknown; never asserted completed')
        store.json('FAILED.json',failure);store.status(failure)
        if engine is not None:
            try:engine.checkpoint('failure.pt','failure');write_new(engine.out/'FAILED.json',failure)
            except BaseException as saving_error:store.json('FAILURE_CHECKPOINT_ERROR.json',dict(error=repr(saving_error),original_error=repr(error),completed_G_updates=store.counts.get('G_updates',0)))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    def terminate(signum,frame):raise InterruptedError('Owned015 received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    result=run015(args.config,args.config_sha256,args.output,args.device)
    print(json.dumps({k:result[k] for k in ('native_status','scientific_training_updates','training_skipped','all_four_trials_completed')},allow_nan=False),flush=True)


if __name__=='__main__':main()
