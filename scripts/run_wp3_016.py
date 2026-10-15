"""Two bounded Q1 trials over frozen015 M0; original014 hard evaluation.

Factories are pure stdlib. Numeric modules load only after the immutable gate.
The driver is never a resume or restart interface.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import run_wp2_014 as old
import run_wp3_015 as prior
from wp1_ops import clean,read_json,require,sha,unchanged,write_new

RUN_ID='wp3_016_20261015'
HYPOTHESES=('fast','slow','common_level')
OWNED_FILES=('scripts/run_wp3_016.py','src/n01/model016.py','scripts/wp3_016_controller.py')


def seed_set016(initialization,charge):
    return dict(initialization=initialization,source_sampler=initialization+1,real_sampler=initialization+2,
        train_shape=initialization+3,train_noise005=initialization+4,train_extension=initialization+5,
        train_projection=initialization+6,integral_projection=initialization+7,train_charge=charge)


def settings016():
    return dict(schema='n01/wp3-016-config/v1',run_id=RUN_ID,status='draft',candidates=['Q1'],
        seed_sets=[seed_set016(2026101540,2026101741),seed_set016(2026101640,2026101751)],G_updates=2000,
        TRAIN_counts=dict(source=6593,real=21047,REAL_logical_groups=422),
        training=dict(source_parents_per_update=512,draws_per_parent=2,real_rows_per_update=1024,
            forward_chunk_rows=256,G_lr=.01,G_weight_decay=0.,clip_norm=5.,wave_weight=1.,distribution_weight=2.,
            regular_weight=.01,status_interval=100,checkpoint_steps=[1000,2000],fresh_AdamW=True,inherit_015_optimizer=False),
        charge=dict(parameters=2,q_reference_Vns=3.7103998046755806,positive_mean_corrected_lognormal=True,
            denominator_floor=False,electronic_constant_variance=0,shape_multiplied_by_random_charge=False,
            charge_disabled_exact_M0_bytes=True,independent_explicit_latent=True),
        calibration=dict(parents=6593,coordinates=384,chunk_rows=256,target_expected_RMS_ADC_LSB=1.,
            zero_prefix100_in_RMS_denominator=True,all_actual_A_Q_positive_required=True,
            A_only=True,finite_MC_retuning=False,shared_across_seeds=True),
        evaluation=dict(parents=6593,draws=4,banks=2,chunk_rows=128,
            bank_seeds=[dict(shape=2026101501,noise005=2026101502,extension=2026101503),
                        dict(shape=2026101511,noise005=2026101512,extension=2026101513)],
            charge_bank_seeds=[2026101721,2026101731],projection_seed=2026101430,
            projection_fingerprint='27ed58d90114de6bfa5bf30ff5ad251516cdc9bcadfb5eb10e2a6ec45ed11482',
            whole_population=True,snapshot_arms=dict(initial=['A','M0','initial'],final=['final']),
            save_charge_semantics_and_preADC_moments=True),
        hypotheses=list(HYPOTHESES),CPU_threads=12,
        run_resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=7200,output_MiB=1536),
        resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=14400,output_MiB=3072,poll_seconds=10,TERM_grace_seconds=20),
        one_active_scientific_job=True,serial_runs=True,automatic_restart=False,overwrite_outputs=False,
        DEV_numeric_read=False,scientific_status='response_match_not_established')


def budget_caps016(p):
    runs=len(p['seed_sets']);t=p['training'];e=p['evaluation'];h=len(HYPOTHESES)
    train=runs*p['G_updates']*t['source_parents_per_update']*t['draws_per_parent']
    evaluation=2*runs*e['banks']*e['parents']*e['draws'];signal=train+evaluation
    observed=h*train+4*runs*e['banks']*e['parents']*e['draws']*h
    return dict(calibration_A_rows=p['calibration']['parents'],old_A_rows=signal+p['calibration']['parents'],
        head_actual_parameter_forward_rows=signal,head_total_kernel_rows=signal,noise005_rows=signal,
        charge_module_rows=signal,charge_latent_scalar_draws=signal,charge_initialization_fits=1,
        hard_ADC_rows=observed,all_path_gather_rows=observed,checkpoint_loads=2+runs,
        optimizer_created=runs,optimizer_steps=runs*p['G_updates'],G_updates=runs*p['G_updates'],
        training_backward_queries=runs*p['G_updates'],new_AD_queries=runs*p['G_updates'],
        probe_AD_queries=0,integral_statistics_fits=0,mean_adapter_rows=0,
        D_updates=0,noise_updates=0,population_updates=0,C2ST_updates=0,DEV_model_rows=0)


def validate_settings016(p,*,frozen=True):
    reference=settings016()
    require(not frozen or p.get('status')=='frozen','016 draft cannot run')
    require(all(p.get(k)==v for k,v in reference.items() if k!='status'),'016 registered settings differ')
    require(p.get('budget_caps')==budget_caps016(p),'016 exact operation caps differ')


def gate016(config,expected_sha,output):
    cp=Path(config).resolve();require(cp.is_file() and sha(cp)==expected_sha,'016 config SHA differs')
    p=read_json(cp);validate_settings016(p)
    required={'dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information',
        'integral_statistics','M0_head_seed0','M0_head_seed1'}
    require(required<=set(p.get('inputs',{})),'016 bound inputs incomplete')
    require({'observation','data_roles','acceptance','tolerance_table'}<=set(p.get('contracts',{})),'016 contracts incomplete')
    require(p['contracts']['tolerance_table']['sha256']==old.TOLERANCE_SHA014,'016 frozen pre-candidate T changed')
    require(set(OWNED_FILES)<=set(p.get('source_files',{})),'016 own sources not frozen')
    roots=[Path(r).resolve() for r in p['allowed_input_roots']]
    require(cp.parent in roots,'016 launch root not allowed')
    checked={str(cp):expected_sha};paths={}
    records={**{'source:'+k:v for k,v in p['source_files'].items()},**{'input:'+k:v for k,v in p['inputs'].items()},
        **{'contract:'+k:v for k,v in p['contracts'].items()},**{'protected:'+k:v for k,v in p.get('protection_files',{}).items()}}
    for label,row in records.items():
        path=Path(row['path']);resolved=path.resolve()
        require(path.is_absolute() and resolved.is_file() and not path.is_symlink()
            and any(resolved.is_relative_to(r) for r in roots),'016 input/source escaped roots: '+label)
        require(os.name!='nt' or path.drive.upper() in ('E:','D:'),'016 all files must stay E/D')
        require(sha(path)==row['sha256'] and ('bytes' not in row or path.stat().st_size==row['bytes']),'016 frozen bytes changed: '+label)
        if label.startswith('source:'):
            require(resolved==(Path(__file__).resolve().parents[1]/label[7:]).resolve(),'016 imported source path mismatch')
        key=str(resolved);require(key not in checked or checked[key]==row['sha256'],'016 conflicting SHA alias')
        checked[key]=row['sha256']
        if label.startswith('input:'):paths[label[6:]]=resolved
    target=Path(output).resolve()
    require(not target.exists() and target!=cp.parent and target.is_relative_to(cp.parent),'016 fresh output inside launch root required')
    require(os.name!='nt' or target.drive.upper() in ('E:','D:'),'016 output must stay E/D')
    return p,paths,checked


class Store016(old.Store014):
    def base(self):
        row=super().base();row.update(schema='n01/wp3-016-state/v1',scope='two charge-only trials over frozen015 M0')
        return row


class ChargeStreams016:
    """Separate charge state never consumes an existing015 prefix."""
    def __init__(self,shape_seed,noise_seed,extension_seed,charge_seed):
        import torch
        from n01.train_shape010 import Streams010
        self.existing=Streams010(shape_seed,noise_seed,extension_seed)
        self.charge_seed=charge_seed
        self.charge=torch.Generator(device='cpu').manual_seed(charge_seed)

    def draw(self,rows,device,operation):
        import torch
        z,nz,epsilon=self.existing.draw(rows,device)
        zq=operation('independent_charge_scalar_draws',dict(charge_latent_scalar_draws=rows),
            lambda:torch.randn(rows,generator=self.charge,dtype=torch.float64).to(device),charge_seed=self.charge_seed)
        return z,nz,epsilon,zq

    def state_dict(self):
        return dict(existing015=self.existing.state_dict(),charge=self.charge.get_state().clone(),charge_seed=self.charge_seed)


class Runtime016(old.Runtime014):
    def forward(self,model,x,z,zq):
        original=model._baseline_once;apply=model.apply_charge
        def baseline_once(values):
            return self.call('original_frozen_A',dict(old_A_rows=len(values)),lambda:original(values))
        def apply_charge(out,latent,*,enabled=True):
            if not enabled:return apply(out,latent,enabled=False)
            return self.call('two_parameter_charge_module',dict(charge_module_rows=len(out['signal_pre'])),
                lambda:apply(out,latent,enabled=True))
        model._baseline_once=baseline_once;model.apply_charge=apply_charge
        try:out=self.call('frozen_M0_head_forward',dict(head_actual_parameter_forward_rows=len(x)),lambda:model(x,z,zq))
        finally:del model._baseline_once;del model.apply_charge
        require(float(out['canonical_charge_identity_residual'].detach().abs().max())<=1e-10
            and bool((out['signal_pre'][:,:100]==0).all()),'016 signal prefix/canonical charge identity failed')
        return out

    def calibrate_charge(self):
        import numpy as np
        import torch
        from n01.losses008 import ADC_STEP
        from n01.model016 import calibration_statistics016
        count=self.p['calibration']['parents']
        require(count==len(self.data['source_carrier384']),'016 calibration must cover all legal TRAIN parents')
        values=[]
        with torch.no_grad():
            for start in range(0,count,self.p['calibration']['chunk_rows']):
                x=torch.as_tensor(self.data['source_carrier384'][start:start+self.p['calibration']['chunk_rows']],dtype=torch.float64,device=self.device)
                zero={name:torch.zeros((len(x),)+shape,dtype=torch.float32,device=self.device) for name,shape in self.baseline.noise_shapes.items()}
                result=self.call('charge_initialization_actual_A_only',dict(old_A_rows=len(x),calibration_A_rows=len(x)),
                    lambda:self.baseline.signal_forward(x,zero))
                values.append(result['signal_pre'].detach())
            A=torch.cat(values)
            result=self.call('analytic_charge_initialization_fit',dict(charge_initialization_fits=1),
                lambda:calibration_statistics016(A,adc_step=ADC_STEP))
        result.update(schema='n01/wp3-016-charge-initialization/v1',source_parent_rows=list(range(count)),
            source_parent_ids_sha256=hashlib.sha256(np.asarray(self.data['source_ids']).tobytes()).hexdigest(),
            no_head_noise_ADC_AD_or_new_random=True,shared_across_two_seeds=True,scientific_acceptance=False)
        self.store.json('CHARGE_INITIALIZATION.json',result)
        self.verify(files=True);return result


class Engine016(old.Engine014):
    def __init__(self,runtime,seed,initial_head,calibration,reference,directory):
        import torch
        from n01.data import FullCoverageSampler
        from n01.model016 import FrozenM0ChargeResponse016
        from n01.losses014 import N01Loss014
        from n01.response008 import loss_tensor_statistics
        self.rt,self.seed,self.candidate,self.out=runtime,seed,'Q1',Path(directory);self.out.mkdir()
        self.started,self.update=time.monotonic(),0;self.reference=reference;self.calibration=calibration
        self.model=FrozenM0ChargeResponse016(runtime.baseline,initial_head,seed=seed['initialization'],calibration=calibration)
        self.loss=N01Loss014(loss_tensor_statistics(runtime.statistics['loss']),runtime.integral_statistics,
            seed=seed['train_projection'],evaluation_seed=runtime.p['evaluation']['projection_seed'],integral_seed=seed['integral_projection']).to(runtime.device)
        self.source_sampler=FullCoverageSampler(len(runtime.data['source']),seed['source_sampler'])
        self.real_sampler=FullCoverageSampler(len(runtime.data['real']),seed['real_sampler'])
        self.streams=ChargeStreams016(seed['train_shape'],seed['train_noise005'],seed['train_extension'],seed['train_charge'])
        self.optimizer=runtime.call('fresh_charge_AdamW_created',dict(optimizer_created=1),
            lambda:torch.optim.AdamW(list(self.model.trainable_parameters()),lr=.01,weight_decay=0.))
        self.model.assert_optimizer_membership(self.optimizer);self.last_metrics={}
        write_new(self.out/'RUN_INPUTS.json',dict(schema='n01/wp3-016-run-inputs/v1',candidate='Q1',seed=seed,reference=reference,
            M0_head_sha256=self.model.reference_M0_head_sha256,trainable_parameter_names=['theta_a','theta_b'],
            trainable_parameters=2,charge_initialization=calibration,frozen_A_005_sha256=runtime.frozen,
            fresh_AdamW=True,inherit_015_optimizer=False,DEV_numeric_read=False,mean_adapter_enabled=False,
            reference_is_development_only=True,weights_promoted=False))

    def checkpoint(self,name,purpose):
        from n01.train_shape010 import cpu_tree
        old.atomic_checkpoint(self.out/name,dict(schema='n01/wp3-016-checkpoint/v1',candidate='Q1',
            initial_seed=self.seed['initialization'],G_updates=self.update,purpose=purpose,
            theta_a=cpu_tree(self.model.theta_a),theta_b=cpu_tree(self.model.theta_b),
            M0_reference=self.reference,M0_head_sha256=self.model.reference_M0_head_sha256,
            charge_initialization=self.calibration,trainable_parameter_names=['theta_a','theta_b'],optimizer=cpu_tree(self.optimizer.state_dict()),
            source_sampler=self.source_sampler.state_dict(),real_sampler=self.real_sampler.state_dict(),streams=self.streams.state_dict(),
            last_metrics=self.last_metrics,frozen_state_sha256=self.rt.frozen,counters=self.rt.store.counts.copy(),
            inflight=copy.deepcopy(self.rt.store.inflight),complete_step_boundary=purpose!='failure',automatic_restart=False))

    def step(self):
        import numpy as np
        import torch
        from n01.response008 import saturation_penalty
        rt,model,t=self.rt,self.model,self.rt.p['training'];rt.guard(self.out,self.started)
        parents=self.source_sampler.draw(t['source_parents_per_update']);si=np.repeat(parents,t['draws_per_parent'])
        ri=self.real_sampler.draw(t['real_rows_per_update']);z,nz,epsilon,zq=self.streams.draw(len(si),rt.device,rt.call)
        noise=rt.noise(nz,epsilon);self.optimizer.zero_grad(set_to_none=True);model.train();outputs=[]
        for start in range(0,len(si),t['forward_chunk_rows']):
            x=torch.as_tensor(rt.data['source_carrier384'][si[start:start+t['forward_chunk_rows']]],dtype=torch.float64,device=rt.device)
            outputs.append(rt.forward(model,x,z[start:start+len(x)],zq[start:start+len(x)]))
        signal_value=torch.cat([o['signal_pre'] for o in outputs])
        shape_regular=sum(o['stochastic_shape_regular']*len(o['signal_pre'])/len(si) for o in outputs)
        charge_regular=sum(o['charge_variance_regular']*len(o['signal_pre'])/len(si) for o in outputs)
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
        wave=torch.stack(waves).mean();distribution=torch.stack(distributions).mean();saturation=torch.stack(saturations).mean()
        regular=shape_regular+charge_regular+saturation;objective=wave+2*distribution+.01*regular
        require(bool(torch.isfinite(objective)),'016 nonfinite objective')
        rt.call('one_training_backward',dict(training_backward_queries=1,new_AD_queries=1),lambda:objective.backward())
        params=list(model.trainable_parameters());gradient=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
        before=torch.stack([v.detach().double().clone() for v in params]);model.assert_optimizer_membership(self.optimizer)
        rt.call('one_charge_AdamW_step',dict(optimizer_steps=1,G_updates=1),lambda:self.optimizer.step());self.update+=1
        after=torch.stack([v.detach().double() for v in params]);delta=after-before
        require(bool(torch.isfinite(after).all()),'016 nonfinite updated charge parameters')
        from n01.model016 import charge_coefficients016
        a,b=charge_coefficients016(model.theta_a,model.theta_b,model.charge_q_reference)
        self.last_metrics=dict(G_updates=self.update,candidate='Q1',initial_seed=self.seed['initialization'],
            objective=float(objective.detach()),wave=float(wave.detach()),weighted_distribution=float((2*distribution).detach()),
            weighted_regular=float((.01*regular).detach()),weighted_frozen_shape_regular=float((.01*shape_regular).detach()),
            weighted_charge_regular=float((.01*charge_regular).detach()),weighted_mean_saturation_penalty=float((.01*saturation).detach()),
            hypotheses=metrics,preclip_total_gradient_norm=float(gradient),actual_parameter_update_norm=float(delta.norm()),
            actual_parameter_update_absmax=float(delta.abs().max()),parameters_changed=bool(delta.ne(0).any()),
            theta_a=float(model.theta_a.detach()),theta_b=float(model.theta_b.detach()),a_Vns=float(a.detach()),b=float(b.detach()),
            source_parent_rows=parents.tolist(),real_row_count=len(ri),coverage=self.coverage(),projection_context=context.metadata(),
            trainable_parameters=2,all_legal_TRAIN_sampling=True,D_updates=0,noise_updates=0,mean_adapter_enabled=False)
        if self.update==1 or self.update%t['status_interval']==0:
            rt.verify(model,files=self.update==1)
            with (self.out/'training.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(clean(self.last_metrics),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            rt.store.status(dict(current_run=self.out.name,current_run_G_updates=self.update))
        if self.update==1:
            require(float(gradient)>0 and bool(delta.ne(0).any()),'016 first step has no gradient or update')
            proof=dict(rt.store.base(),schema='n01/wp3-016-first-real-update/v1',actual_G_updates=1,
                batch_completed_G_updates=rt.store.counts['G_updates'],trial=self.out.name,candidate='Q1',
                initial_seed=self.seed['initialization'],frozen_A_005_unchanged=True,protected_M0_shape_unchanged=True,
                gradient_nonzero=True,actual_parameter_update_nonzero=True,trainable_parameters=2,
                trainable_parameter_names=['theta_a','theta_b'],charge_initialization_complete=True,mean_adapter_enabled=False,
                M0_head_sha256=model.reference_M0_head_sha256,frozen_state_sha256=rt.frozen,reference=self.reference,metrics=self.last_metrics)
            write_new(self.out/'FIRST_UPDATE_PROOF.json',proof)
            if not (rt.store.root/'FIRST_UPDATE_PROOF.json').exists():rt.store.json('FIRST_UPDATE_PROOF.json',proof)
        context.validate();return self.last_metrics

    def evaluate(self,*,initial):
        """Mature014 evaluation/save path, with one extra paired M0 signal arm."""
        import numpy as np
        import torch
        from wp1_evaluation import evaluate_hard_population,component_distances,observable_features
        from run_wp1_013 import observation_rows,observation_booleans,lossless_window_codes,save_report,REAL_UNKNOWN_FLAGS
        rt,e=self.rt,self.rt.p['evaluation'];directory=self.out/('initial_evaluation' if initial else 'final_evaluation');directory.mkdir()
        parent_rows=np.repeat(np.arange(len(rt.data['source'])),e['draws']);draw_index=np.tile(np.arange(e['draws']),len(rt.data['source']))
        arms=e['snapshot_arms']['initial' if initial else 'final'];context=self.loss.create_evaluation_context();metrics={};charge_summaries=[]
        cached_real=observable_features(rt.data['real'],rt.data['real_baseline100'],float(self.loss.q_floor));self.model.eval()
        with torch.no_grad():
            for bank,seeds in enumerate(e['bank_seeds']):
                streams=ChargeStreams016(seeds['shape'],seeds['noise005'],seeds['extension'],e['charge_bank_seeds'][bank]);rng_initial=streams.state_dict()
                pieces={arm:{h:[] for h in HYPOTHESES} for arm in arms};codes={arm:{h:[] for h in HYPOTHESES} for arm in arms}
                baseline_pieces=[];q_values=[];mu_values=[];increment_sum=expected_sum=0.;canonical_error=projection_error=0.;sample_coordinates=0
                for start in range(0,len(parent_rows),e['chunk_rows']):
                    selected=parent_rows[start:start+e['chunk_rows']]
                    x=torch.as_tensor(rt.data['source_carrier384'][selected],dtype=torch.float64,device=rt.device)
                    z,nz,epsilon,zq=streams.draw(len(x),rt.device,rt.call);noise=rt.noise(nz,epsilon);out=rt.forward(self.model,x,z,zq)
                    q_values.append(old.arr(out['qsig']));mu_values.append(old.arr(out['Q_mu_Vns']))
                    increment_sum+=float(out['charge_increment_pre_V'].square().sum());sample_coordinates+=len(x)*384
                    expected_sum+=float(out['charge_expected_increment_RMS_V'].square().sum())*384
                    canonical_error=max(canonical_error,float(out['canonical_charge_identity_residual'].abs().max()))
                    projection_error=max(projection_error,float(out['inherited_M0_canonical_projection_residual_Vns'].abs().max()))
                    shared_baseline=None
                    for arm in arms:
                        signal_value=out['base_signal_pre_V'] if arm=='A' else out['M0_reference_signal_pre_V'] if arm=='M0' else out['signal_pre']
                        for h in HYPOTHESES:
                            carrier=signal_value+noise[h];obs=rt.observe(carrier,ste=False);rows=observation_rows(obs,carrier,self.loss)
                            if shared_baseline is None:shared_baseline=rows['baseline100'];baseline_pieces.append(shared_baseline)
                            else:require(np.array_equal(shared_baseline,rows['baseline100']),'016 unpaired electronics prefix')
                            codes[arm][h].append(lossless_window_codes(rows));pieces[arm][h].append({n:v for n,v in rows.items() if n not in ('adc_carrier','baseline100')})
                    rt.guard(self.out,self.started)
                baseline=np.concatenate(baseline_pieces)
                with (directory/f'bank{bank}_BASELINE100.npz').open('xb') as f:
                    np.savez_compressed(f,baseline100=baseline,source_parent_row=parent_rows,draw_index=draw_index)
                q=np.concatenate(q_values).reshape(-1,e['draws']);mu=np.concatenate(mu_values).reshape(-1,e['draws'])
                require(np.all(mu==mu[:,:1]),'016 per-parent A canonical Q changed across draws')
                a,b=[float(v) for v in (out['charge_a_Vns'],out['charge_b'])];theory_var=a*mu[:,0]+b*mu[:,0]**2
                observed_mean=q.mean(1);variance=q.var(1,ddof=1);mcse=np.sqrt(variance/e['draws']);rng_final=streams.state_dict()
                moments=dict(Q_mu_Vns=mu[:,0],q_sig_draws_Vns=q,sample_mean_Vns=observed_mean,sample_variance_Vns2=variance,
                    conditional_mean_MCSE_Vns=mcse,theory_conditional_variance_Vns2=theory_var,source_parent_row=np.arange(len(q)),
                    charge_rng_initial_state=old.arr(rng_initial['charge']),charge_rng_final_state=old.arr(rng_final['charge']))
                for name,state in rng_initial['existing015'].items():moments['existing015_'+name+'_initial_state']=old.arr(state)
                for name,state in rng_final['existing015'].items():moments['existing015_'+name+'_final_state']=old.arr(state)
                with (directory/f'bank{bank}_CHARGE_PREADC_MOMENTS.npz').open('xb') as f:np.savez_compressed(f,**moments)
                charge_summaries.append(dict(bank=bank,charge_seed=e['charge_bank_seeds'][bank],rows=q.size,parents=len(q),draws=e['draws'],
                    a_Vns=a,b=b,Q_mu_min_Vns=float(mu.min()),Q_mu_max_Vns=float(mu.max()),
                    realized_increment_RMS_V=math.sqrt(increment_sum/sample_coordinates),actual_A_expected_increment_RMS_V=math.sqrt(expected_sum/sample_coordinates),
                    conditional_CV_min=float(np.sqrt(a/mu[:,0]+b).min()),conditional_CV_mean=float(np.sqrt(a/mu[:,0]+b).mean()),conditional_CV_max=float(np.sqrt(a/mu[:,0]+b).max()),
                    finite_MC_mean_error_RMS_Vns=float(np.sqrt(np.square(observed_mean-mu[:,0]).mean())),
                    finite_MC_mean_MCSE_RMS_Vns=float(np.sqrt(np.square(mcse).mean())),canonical_identity_absmax_Vns=canonical_error,
                    inherited_M0_projection_residual_absmax_Vns=projection_error,covariance_denominator=e['draws']-1,
                    finite_MC_recalibration=False,preADC_conditional_mean_protection=True,hard_observation_expectation_guaranteed_unchanged=False,
                    retained_moments_file=f'bank{bank}_CHARGE_PREADC_MOMENTS.npz',retained_moments_sha256=sha(directory/f'bank{bank}_CHARGE_PREADC_MOMENTS.npz')))
                for arm in arms:
                    for h in HYPOTHESES:
                        joined={n:np.concatenate([r[n] for r in pieces[arm][h]]) for n in pieces[arm][h][0]}
                        storage={n:np.concatenate([r[n] for r in codes[arm][h]]) for n in codes[arm][h][0] if n!='IEEE754_reconstruction_XOR_uint64'}
                        if any('IEEE754_reconstruction_XOR_uint64' in r for r in codes[arm][h]):
                            storage['IEEE754_reconstruction_XOR_uint64']=np.concatenate([r.get('IEEE754_reconstruction_XOR_uint64',np.zeros(r['ADC_crop_codes'].shape,np.uint64)) for r in codes[arm][h]])
                        stem=f'bank{bank}_{arm}_{h}';storage.update(source_parent_row=parent_rows,source_parent_ids=rt.data['source_ids'][parent_rows],
                            draw_index=draw_index,bank_index=np.full(len(parent_rows),bank,np.int64));storage.update({n:v for n,v in joined.items() if n!='window_V'})
                        with (directory/(stem+'_HARD_ROWS.npz')).open('xb') as f:np.savez_compressed(f,**storage)
                        report=evaluate_hard_population(joined['window_V'],baseline,real_wave256=rt.data['real'],real_baseline100=rt.data['real_baseline100'],
                            parent_row=parent_rows,draw_index=draw_index,bank_index=np.full(len(parent_rows),bank,np.int64),source_information=rt.info,
                            loss=self.loss,context=context,observation_flags=observation_booleans(joined),real_observation_flags=rt.data.get('real_observation_flags'),
                            real_unknown_flags=REAL_UNKNOWN_FLAGS,real_groups=rt.data['real_groups'],cached_real=cached_real)
                        save_report(directory,stem+'_REPORT',report)
                        metrics[f'bank{bank}_{arm}_{h}']=component_distances(joined['window_V'],baseline,rt.data['real'],rt.data['real_baseline100'],self.loss,context,cached_reference=cached_real)
                        rt.guard(self.out,self.started)
        write_new(directory/'CHARGE_SEMANTICS.json',dict(schema='n01/wp3-016-charge-semantics/v1',banks=charge_summaries,
            actual_canonical_A_Q=True,no_denominator_floor=True,shape_amplitude_from_original_A=True,mean_adapter_enabled=False,
            moments_are_finite_MC_not_exact_analytic_moments=True,full_preADC_carrier_saved=False,scientific_acceptance=False))
        write_new(directory/'METRICS.json',dict(schema='n01/wp3-016-full-hard-metrics/v1',metrics=metrics,source_parents=len(rt.data['source']),
            draws_per_bank=e['draws'],MC_banks=e['banks'],REAL_rows=len(rt.data['real']),statistical_unit='SOURCE_parent_and_original_REAL_group; MC draws are not new physical records',
            no_DEV=True,all_signed_rows_retained=True,tolerance_table=rt.p['contracts']['tolerance_table'],all_metric_classes=True,root_acceptance_pending=True,scientific_acceptance=False))
        rt.verify(self.model,files=True);return metrics


def compare_metrics016(initial,final,tolerance):
    # The M0 replacement reference is explicit; charge-on initialization is separate.
    paired=copy.deepcopy(initial)
    for key in list(paired):
        if '_M0_' in key:paired[key.replace('_M0_','_initial_')]=paired[key]
    rows=old.compare_metrics014(paired,final,tolerance)
    for row in rows:
        row['common_initial_reference']='frozen015_M0_charge_off'
        key=f"{row['bank']}_initial_{row['hypothesis']}"
        row['charge_on_initial_values']=initial[key]['values']
    return rows


def run016(config,config_sha256,output,device):
    p,paths,checked=gate016(config,config_sha256,output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(p['CPU_threads'])
    store=Store016(output,p,checked);store.status();engine=None
    try:
        store.json('BINDING.json',dict(configuration_sha256=config_sha256,immutable_files=checked,budget_caps=p['budget_caps'],before_numeric_decode=True))
        import torch
        from n01.evaluate_window008 import _memory_limit
        from n01.train_shape010 import cpu_tree
        torch.set_num_threads(p['CPU_threads']);torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']),device)
        data,baseline,extension,stats,integral,info=prior.prepare_numeric015(p,paths,store,device)
        runtime=Runtime016(p,data,baseline,extension,stats,integral,store,device,info);calibration=runtime.calibrate_charge()
        tolerance=read_json(p['contracts']['tolerance_table']['path']);completed=[]
        for index,seed in enumerate(p['seed_sets']):
            key=f'M0_head_seed{index}'
            checkpoint=runtime.call('checkpoint_'+key,dict(checkpoint_loads=1),lambda key=key:torch.load(paths[key],map_location='cpu',weights_only=False))
            require(checkpoint.get('schema')=='n01/wp3-checkpoint/v1' and checkpoint.get('arm')=='M0' and checkpoint.get('G_updates')==2000
                and checkpoint.get('initial_seed')==seed['initialization'] and checkpoint.get('complete_step_boundary') is True
                and checkpoint.get('purpose')=='final' and checkpoint.get('frozen_state_sha256')==runtime.frozen
                and checkpoint.get('trainable_parameter_names')==['basis_head.weight','basis_head.bias']
                and bool((checkpoint['mean_theta']==0).all()),'016 frozen015 M0 reference metadata mismatch')
            initial=cpu_tree(checkpoint['new_head']);reference=dict(input_key=key,sha256=p['inputs'][key]['sha256'],WP3_015_G_updates=2000,
                optimizer_inherited=False,development_only=True,weights_promoted=False);del checkpoint
            engine=Engine016(runtime,seed,initial,calibration,reference,store.root/f'seed{index}_Q1')
            if not (store.root/'INIT_PROOF.json').exists():
                store.json('INIT_PROOF.json',dict(store.base(),schema='n01/wp3-016-init-proof/v1',native_initialized=True,
                    charge_initialization_complete=True,calibration_all_Q_positive=True,first_trial=engine.out.name,
                    M0_head_sha256=engine.model.reference_M0_head_sha256,mean_adapter_enabled=False,trainable_parameters=2,
                    trainable_parameter_names=['theta_a','theta_b'],frozen_state_sha256=runtime.frozen,reference=reference,fresh_AdamW=True))
            engine.checkpoint('initial.pt','initial');initial_metrics=engine.evaluate(initial=True)
            while engine.update<p['G_updates']:
                engine.step()
                if engine.update in p['training']['checkpoint_steps']:
                    runtime.verify(engine.model,files=True);engine.checkpoint('final.pt' if engine.update==p['G_updates'] else f'step_{engine.update:04d}.pt',
                        'final' if engine.update==p['G_updates'] else 'periodic')
            final_metrics=engine.evaluate(initial=False);comparison=compare_metrics016(initial_metrics,final_metrics,tolerance)
            write_new(engine.out/'HARD_COMPARISON.json',dict(schema='n01/wp3-016-hard-comparison/v1',candidate='Q1',initial_seed=seed['initialization'],
                per_bank_hypothesis=comparison,replacement_reference='frozen015_M0_charge_off',frozen_tolerance=p['contracts']['tolerance_table'],scientific_acceptance=False))
            coverage=engine.coverage();require(coverage['source_unique']==6593 and coverage['real_unique']==21047,'016 incomplete TRAIN coverage')
            runtime.verify(engine.model,files=True)
            result=dict(schema='n01/wp3-016-trial-complete/v1',native_status='COMPLETE',candidate='Q1',seed=seed,G_updates=engine.update,
                coverage=coverage,frozen_A_005_unchanged=True,frozen_state_sha256=runtime.frozen,protected_M0_shape_unchanged=True,
                M0_head_sha256=engine.model.reference_M0_head_sha256,trainable_parameter_names=['theta_a','theta_b'],last_metrics=engine.last_metrics,
                per_bank_hypothesis=comparison,resources=runtime.guard(engine.out,engine.started),raw_artifacts_retained_on_execution_host=True,
                automatic_restart=False,scientific_acceptance=False,scientific_status='response_match_not_established',root_selection_pending=True)
            write_new(engine.out/'COMPLETE.json',result);completed.append(dict(directory=engine.out.name,candidate='Q1',initial_seed=seed['initialization'],
                G_updates=engine.update,COMPLETE_sha256=sha(engine.out/'COMPLETE.json')));engine=None
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
        require(not store.inflight,'016 cannot complete with unknown in-flight operations');unchanged(checked)
        for key,cap in p['budget_caps'].items():require(store.counts[key]==cap,'016 completed operation count differs: '+key)
        require(time.monotonic()-store.started<=p['resource_limits']['wall_seconds'],'016 final batch wall cap exceeded')
        prior.artifact_inventory015(store)
        result=dict(store.base(),schema='n01/wp3-016-batch-complete/v1',native_status='COMPLETE',training_skipped=False,
            trials=completed,G_updates_per_trial=p['G_updates'],all_two_trials_completed=True,all_frozen_A_005_unchanged=True,
            all_M0_heads_unchanged=True,charge_initialization_complete=True,no_DEV=True,scientific_acceptance=False,
            scientific_status='response_match_not_established',root_choice_pending=True)
        store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=p['resource_limits']['output_MiB']*1024**2,'016 final batch output cap exceeded')
        return result
    except BaseException as error:
        failure=dict(store.base(),schema='n01/wp3-016-batch-failure/v1',native_status='FAILED',error_type=type(error).__name__,error=str(error),
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
    def terminate(signum,frame):raise InterruptedError('Owned016 received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate);result=run016(args.config,args.config_sha256,args.output,args.device)
    print(json.dumps({k:result[k] for k in ('native_status','scientific_training_updates','all_two_trials_completed')},allow_nan=False),flush=True)


if __name__=='__main__':main()
