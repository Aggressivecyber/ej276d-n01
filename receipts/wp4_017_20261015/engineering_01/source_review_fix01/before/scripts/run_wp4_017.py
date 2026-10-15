"""WP4 fixed known-base charge recovery. Pure factories precede numeric IO."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from wp1_ops import Store, clean, require, read_json, sha, unchanged, write_new

RUN_ID='wp4_017_20261015'
REMOTE='/home/rylan/projects/ej276d-n01/wp4_017_20261015'
PROPOSAL_SHA='b2db64dfd3549f6cd22cb5253a1e374738248012e22554308ec26885d2d42e8e'
HYPOTHESES=('fast','slow','common_level')
TRIALS=('T1_paired_seed0','T1_paired_seed1','T1_unpaired_seed0','T1_unpaired_seed1',
        'T2_paired_seed0','T2_paired_seed1','T2_unpaired_seed0','T2_unpaired_seed1')
OWNED_FILES=('scripts/run_wp4_017.py','scripts/wp4_017_controller.py','src/n01/model017.py',
             'src/n01/stats017.py','src/n01/saved_array_kernels017.py')

def settings017():
    return dict(schema='n01/wp4-017-config/v1',run_id=RUN_ID,status='draft',remote_root=REMOTE,
        proposal_sha256=PROPOSAL_SHA,trainable_parameters=2,
        candidates=['T1_paired','T1_unpaired','T2_paired','T2_unpaired'],seed_sets=[0,1],
        trial_order=list(TRIALS),G_updates=2000,CPU_threads=12,
        training=dict(fit_parents=5274,test_parents=1319,all_parents=6593,parents=512,
            paired_draws=2,unpaired_rows_per_side=1024,chunk_rows=256,
            lr=.01,weight_decay=0.,clip=5.,checkpoints=[1000,2000]),
        evaluation=dict(parents=6593,banks=2,draws=32,chunk_rows=128,
            metric_dimension=257,distance_block_rows=64,full_stochastic_preADC_saved=False),
        teachers=dict(T1=dict(a_fraction_qref=.01,b=.0025),T2=dict(m=.9,component=0)),
        initial=dict(a_fraction_qref=.001,b=.00025),qref=3.7103998046755806,
        hypotheses=list(HYPOTHESES),cache_chunk_parents=128,
        run_resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=7200,output_MiB=1536),
        resource_limits=dict(GPU_GiB=10,job_tree_RSS_GiB=8,wall_seconds=64800,output_MiB=17408,poll_seconds=10,TERM_grace_seconds=20),
        shared_output_MiB=5120,stage1_output_MiB=8192,stage1_wall_seconds=21600,
        failure_checkpoint_job_cap=1,selected_collection_MiB=256,
        one_active_scientific_job=True,serial_runs=True,automatic_restart=False,overwrite_outputs=False,
        REAL_numeric_read=False,DEV_numeric_read=False,scientific_status='response_match_not_established')

def budget_caps017(p,stage1=False):
    paired,unpaired,teachers=(2,0,1) if stage1 else (4,4,2)
    runs=paired+unpaired;P=p['evaluation']['parents'];R=p['evaluation']['draws'];B=p['evaluation']['banks']
    H=len(p['hypotheses']);K=p['training']['parents'];G=p['G_updates'];E=P*R*B
    S=runs*G*K*2+(1+runs)*E;T=runs*G*K*2+teachers*E;total=S+T;samplers=paired+4*unpaired
    return dict(trial_slots=runs,G_updates=runs*G,optimizer_steps=runs*G,training_backward_queries=runs*G,
        new_AD_queries=runs*G,optimizers_created=runs,checkpoint_loads=3,A_original_forward_rows=P,
        original_U_Net_decoded_rows=P,fixed_M0_head_actual_kernel_rows=P,
        student_charge_module_rows=S,teacher_charge_module_rows=T,shape_cached_basis_assembly_rows=total,
        noise005_rows=total,all_three_extensions_rows=H*total,hard_ADC_rows=H*total,hard_first_argmax_gather_rows=H*total,
        training_student_rows=runs*G*K*2,training_teacher_rows=runs*G*K*2,
        evaluation_kernel_snapshots=1+runs+teachers,evaluation_signal_rows=(1+runs+teachers)*E,
        normal_shape_scalar_requests=total*8,normal_charge_scalar_requests=total,
        normal_noise005_scalar_requests=total*101,normal_extension284_scalar_requests=total*284,
        total_scalar_normal_requests=total*394,T2_teacher_sign_uniform_requests=0 if stage1 else 4*G*K*2+E,
        coverage_PIT_uniform_requests=teachers*E*H*6,parent_sampler_streams=samplers,
        parent_randint_calls=samplers*G,parent_randint_scalar_requests=samplers*G*K,parent_randperm_calls=0,
        paired_training_energy_pair_distances=paired*G*K*5*H,
        unpaired_training_energy_pair_distances=unpaired*G*(2*K)*(2*K)*2*H,
        unpaired_training_cross_retained_pairs=unpaired*G*(2*K)*(2*K)*H,
        unpaired_training_self_ordered_retained_pairs=unpaired*G*(2*K)*(2*K-1)*H,
        unpaired_training_self_diagonal_computed_then_masked=unpaired*G*(2*K)*H,
        evaluation_energy_candidate_comparisons=runs+teachers,
        evaluation_energy_pair_distances=((runs+teachers)*B*2+teachers*3)*P*H*R*R,
        checkpoint_saves_complete=1+2*runs,REAL_waveform_reads=0,original_DEV_waveform_reads=0,
        A_005_M0_updates=0,mean_adapter_updates=0,shape_head_updates=0,teacher_parameter_updates=0,
        noise_updates=0,population_updates=0,D_updates=0,C2ST_updates=0,probe_AD_queries=0,
        integral_scale_fits=0,charge_initialization_fits=0,teacher_or_student_latent_in_deployment_inputs=0,
        new_hardware_or_REAL_teacher_claims=0)

def validate_settings017(p,*,frozen=True):
    ref=settings017()
    require(not frozen or p.get('status')=='frozen','017 draft cannot run')
    require(all(p.get(k)==v for k,v in ref.items() if k!='status'),'017 settings differ from accepted proposal')
    require(p.get('budget_caps')==budget_caps017(p),'017 additive budget differs')
    require(p.get('stage1_budget_caps')==budget_caps017(p,True),'017 stage1 exact budget differs')

def gate017(config,expected_sha,output):
    cp=Path(config).resolve();require(sha(cp)==expected_sha,'017 exact config SHA')
    p=read_json(cp);validate_settings017(p);require(set(OWNED_FILES)<=set(p['source_files']),'017 source closure missing')
    needed={'dataset','statistics','data_statistics','base_config','deterministic','noise005','M0_head_seed0'}
    require(needed<=set(p['inputs']),'017 fixed source-only inputs missing')
    require(p['contracts']['proposal']['sha256']==PROPOSAL_SHA,'017 accepted proposal pin differs')
    require(p['contracts']['contract_review']['sha256']=='31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7','017 contract PASS pin differs')
    require(not Path(output).exists(),'017 preserve previous output')
    checked={str(cp):expected_sha};paths={};roots=[Path(v).resolve() for v in p['allowed_input_roots']]
    records={**p['source_files'],**{'input:'+k:v for k,v in p['inputs'].items()},
        **{'contract:'+k:v for k,v in p['contracts'].items()},**{'protected:'+k:v for k,v in p.get('protection_files',{}).items()}}
    for label,row in records.items():
        path=Path(row['path']);resolved=path.resolve()
        require(not path.is_symlink() and any(resolved.is_relative_to(r) for r in roots),'017 redirected/outside bound path')
        require(path.stat().st_size==row['bytes'],'017 bound byte size differs')
        if str(path) in checked:require(checked[str(path)]==row['sha256'],'017 alias SHA differs')
        else:require(sha(path)==row['sha256'],'017 immutable SHA differs: '+label);checked[str(path)]=row['sha256']
        if label.startswith('input:'):paths[label[6:]]=path
    return p,paths,checked

class Store017(Store):
    def base(self):
        result=super().base();result.update(schema='n01/wp4-017-state/v1',scientific_training_updates=self.counts.get('G_updates',0))
        return result

def seed017(namespace):
    payload=json.dumps(namespace,sort_keys=True,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(('N01-WP4-017-v1|'+payload).encode()).digest()[:8],'little')&((1<<63)-1)

class Streams017:
    def __init__(self,namespace,call=lambda name,charges,fn,**detail:fn()):
        import torch
        self.call=call;self.namespace=copy.deepcopy(namespace)
        self.generators={k:torch.Generator(device='cpu').manual_seed(seed017(dict(namespace,stream=k)))
            for k in ('shape','charge','noise005','extension284','sign','PIT')}
    def state_dict(self):return {k:g.get_state().clone() for k,g in self.generators.items()}
    def draw(self,count,device,*,mixture=False):
        import torch
        def normals():
            g=self.generators
            return (torch.randn(count,8,generator=g['shape'],dtype=torch.float32),torch.randn(count,generator=g['charge'],dtype=torch.float32),
                {'baseline_gaussian':torch.randn(count,1,100,generator=g['noise005'],dtype=torch.float32),
                 'baseline_scale':torch.randn(count,1,1,generator=g['noise005'],dtype=torch.float32)},
                torch.randn(count,1,284,generator=g['extension284'],dtype=torch.float32))
        z,q,n,e=self.call('explicit_latent_tensors',dict(normal_shape_scalar_requests=count*8,normal_charge_scalar_requests=count,
            normal_noise005_scalar_requests=count*101,normal_extension284_scalar_requests=count*284,total_scalar_normal_requests=count*394),normals)
        sign=None
        if mixture:sign=self.call('explicit_T2_sign',dict(T2_teacher_sign_uniform_requests=count),
            lambda:(torch.rand(count,generator=self.generators['sign'])>=.5).to(torch.int8)*2-1).to(device)
        return z.to(device),q.to(device),{k:v.to(device) for k,v in n.items()},e.to(device),sign
    def PIT(self,parents):
        import torch
        return self.call('explicit_PIT_uniforms',dict(coverage_PIT_uniform_requests=parents*2*32*6*3),
            lambda:torch.rand(parents,2,32,6,3,generator=self.generators['PIT'],dtype=torch.float64).numpy())

class ParentSampler017:
    def __init__(self,fit,namespace,call):
        import numpy as np
        import torch
        self.fit=np.asarray(fit);self.seen=np.zeros(len(fit),bool);self.call=call
        self.generator=torch.Generator(device='cpu').manual_seed(seed017(namespace))
        call('new_IID_parent_stream',dict(parent_sampler_streams=1),lambda:None)
    def draw(self,count):
        import torch
        ix=self.call('IID_parent_indices',dict(parent_randint_calls=1,parent_randint_scalar_requests=count),
            lambda:torch.randint(len(self.fit),(count,),generator=self.generator).numpy())
        self.seen[ix]=True;return self.fit[ix]
    def coverage(self):return dict(fit_parents=len(self.fit),unique=int(self.seen.sum()),all_covered=bool(self.seen.all()))

def prepare_numeric017(p,paths,store,device):
    """Only invoked below full freeze gate. No REAL/DEV numeric member access."""
    import numpy as np
    import torch
    from n01.model008 import N01Response008
    from n01.model014 import FrozenDeterministicVoltageShape014
    from n01.model017 import deterministic_loading017
    from n01.response008 import make_extension,statistics_digest,state_digest
    stats,domain,cfg=[read_json(paths[k]) for k in ('statistics','data_statistics','base_config')]
    require(stats['prepared_dataset_sha256']==p['inputs']['dataset']['sha256']
        and stats['data_statistics_sha256']==p['inputs']['data_statistics']['sha256']
        and stats['statistics_split']==domain['split']=='TRAIN' and stats['DEV_waveforms_decoded'] is False,'017 known statistics provenance')
    mandatory=('source_carrier384','source_ids','source_role');data={}
    with np.load(paths['dataset'],allow_pickle=False) as archive:
        require(set(mandatory)<=set(archive.files),'017 SOURCE member whitelist absent')
        for name in mandatory:data[name]=archive[name].copy()
        if 'original_source_weights' in archive.files:data['original_source_weights']=archive['original_source_weights'].copy()
    x=data['source_carrier384'];N=p['training']['all_parents']
    require(x.shape==(N,384) and x.dtype.kind=='f' and np.isfinite(x).all()
        and data['source_ids'].shape==data['source_role'].shape==(N,)
        and len(np.unique(data['source_ids']))==N and set(np.unique(data['source_role']))=={'TRAIN'},'017 all SOURCE legal unique TRAIN parents required')
    if 'original_source_weights' in data:require(data['original_source_weights'].shape==(N,) and np.isfinite(data['original_source_weights']).all(),'017 original source weights schema')
    def load(key):return store.call('checkpoint_'+key,dict(checkpoint_loads=1),lambda:torch.load(paths[key],map_location='cpu',weights_only=False))
    A=load('deterministic');noise=load('noise005');head=load('M0_head_seed0')
    require(A.get('schema')=='response008/v1' and A.get('arm')=='deterministic' and A.get('G_updates')==19000
        and A.get('stage')=='low_learning_rate' and A.get('noise_updates')==0 and A.get('population_learning') is False
        and A.get('config')==cfg and statistics_digest(A['statistics'])==statistics_digest(stats),'017 exact known A metadata')
    require(noise.get('schema_version')==5 and noise.get('update')==5000 and noise.get('model_type')=='ObservedBaselineNoiseV3'
        and noise.get('engineering') is False,'017 original005 metadata')
    require(head.get('schema')=='n01/wp3-checkpoint/v1' and head.get('arm')=='M0' and head.get('G_updates')==2000
        and head.get('initial_seed')==2026101540 and head.get('purpose')=='final' and head.get('complete_step_boundary') is True
        and head.get('trainable_parameter_names')==['basis_head.weight','basis_head.bias'] and bool((head['mean_theta']==0).all()),'017 fixed seed0 M0 metadata')
    with deterministic_loading017():
        baseline=N01Response008(stats['model'],'deterministic');baseline.load_state_dict(A['model'],strict=True)
        baseline.set_stage(A['stage']);baseline.eval();baseline.requires_grad_(False);baseline.to(device)
        extension=make_extension(noise,device)
        M0=FrozenDeterministicVoltageShape014(baseline,seed=2026101540)
        M0.basis_head.load_state_dict(head['new_head'],strict=True)
    M0.eval();M0.requires_grad_(False);M0.basis_head.eval()
    frozen=dict(baseline=state_digest(baseline.state_dict()),noise005=state_digest(extension.model005.state_dict()),
        extension=state_digest(extension.state_dict()))
    require(frozen['noise005']==A['noise005_state_sha256'] and head['frozen_state_sha256']==frozen,'017 A/005/M0 frozen binding')
    store.json('SOURCE_ONLY_ROLES.json',dict(members_read=list(data),all_parents=N,
        source_dtype=x.dtype.str,source_carrier_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        REAL_DEV_members_materialized=[],original_physical_weights='preserved if available, otherwise unknown',
        known_bases_are_not_unseen_REAL_CV=True))
    return data,M0,extension,stats,frozen

def build_frozen_cache017(model,source,ids,directory,call,*,chunk=128,heldout=1319,original_weights=None):
    import numpy as np
    import torch
    from n01.model008 import canonical_charge008
    from n01.model010 import tensor_state_digest010
    from n01.stats017 import partition017
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    N=len(source);A=np.lib.format.open_memmap(directory/'A.npy',mode='w+',dtype=np.float64,shape=(N,384))
    B=np.lib.format.open_memmap(directory/'B.npy',mode='w+',dtype=np.float64,shape=(N,8,384))
    Q=np.lib.format.open_memmap(directory/'Q.npy',mode='w+',dtype=np.float64,shape=(N,))
    decoded_sha=hashlib.sha256();head_sha=tensor_state_digest010(model.basis_head);source_hashes=[];forward_hashes=[]
    source_features={k:[] for k in ('q_source','source_peak_index','source_baseline_mean_V')}
    device=model.baseline.mean_head.weight.device;original=model._baseline_once;captured=[]
    def capture(value):
        out,decoded=original(value);captured.append(decoded);return out,decoded
    model._baseline_once=capture
    try:
        with torch.no_grad():
            for start in range(0,N,chunk):
                x=torch.as_tensor(source[start:start+chunk],dtype=torch.float64,device=device);n=len(x)
                source_hashes.extend(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in source[start:start+n])
                forward_hashes.extend(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in x.cpu().numpy())
                out=call('one_A_decoded_fixed_head_chunk',dict(A_original_forward_rows=n,original_U_Net_decoded_rows=n,
                    fixed_M0_head_actual_kernel_rows=n),lambda:model(x,torch.zeros(n,8,device=device)))
                require(len(captured)==1 and captured[0].shape==(n,32,384) and captured[0].dtype==torch.float32,'017 actual decoded precision/shape')
                decoded_sha.update(captured.pop().detach().cpu().contiguous().numpy().tobytes())
                a=out['base_signal_pre_V'];b=out['voltage_basis_V'];q=canonical_charge008(a)
                require(set(source_features)<=set(out),'017 one-pass actual SOURCE diagnostic interface changed')
                for k in source_features:source_features[k].append(out[k].detach().cpu().numpy())
                require(a.dtype==b.dtype==q.dtype==torch.float64 and bool((q>0).all() and torch.isfinite(q).all()),'017 actual positive cache Q, no floor')
                require(bool((a[:,:100]==0).all() and (b[:,:,:100]==0).all()),'017 cache prefix')
                A[start:start+n]=a.cpu().numpy();B[start:start+n]=b.cpu().numpy();Q[start:start+n]=q.cpu().numpy()
                del out,a,b,q,x
    finally:del model._baseline_once
    for value in (A,B,Q):value.flush();value.flags.writeable=False
    fit,test=partition017(ids,heldout);save_npz017(directory/'PARENT_SPLIT.npz',fit_rows=fit,test_rows=test,source_ids=ids,
        original_source_row_bytes_sha256=np.asarray(source_hashes,dtype='S64'),forward_float64_row_bytes_sha256=np.asarray(forward_hashes,dtype='S64'),
        original_source_dtype=np.asarray(source.dtype.str),forward_dtype=np.asarray('float64'),
        **{k:np.concatenate(v) for k,v in source_features.items()})
    if original_weights is not None:save_npz017(directory/'ORIGINAL_PHYSICAL_WEIGHTS.npz',weights=original_weights,source_ids=ids)
    require(tensor_state_digest010(model.basis_head)==head_sha,'017 changed fixed M0 during cache')
    contrast=float(np.sqrt(np.mean(A*A*(.01*3.7103998046755806/Q[:,None]+.0025)))*(1-np.sqrt(.1)))
    require(contrast>=3/16383,'017 known T1/init contrast below frozen3LSB; untestable, no retuning')
    manifest=dict(schema='n01/wp4-017-frozen-cache/v1',parents=N,
        A_dtype='float64',B_dtype='float64',decoded_dtype='float32',decoded_shape=[N,32,384],
        decoded_ordered_raw_bytes_sha256=decoded_sha.hexdigest(),M0_head_tensor_sha256=head_sha,
        actual_A_positive_Q_min=float(Q.min()),actual_A_positive_Q_max=float(Q.max()),
        teacher_initial_expected_increment_RMS_difference_V=contrast,heldout_new_parameter_parents=len(test),fit_parents=len(fit),
        cache_files={name:dict(bytes=(directory/name).stat().st_size,sha256=sha(directory/name)) for name in
            (('A.npy','B.npy','Q.npy','PARENT_SPLIT.npz')+('ORIGINAL_PHYSICAL_WEIGHTS.npz',) if original_weights is not None else ('A.npy','B.npy','Q.npy','PARENT_SPLIT.npz'))},
        one_A_pass=True,full_decoded_persisted=False,REAL_CV=False)
    write_new(directory/'CACHE_MANIFEST.json',manifest)
    return dict(A=A,B=B,Q=Q,fit=fit,test=test,ids=ids,directory=directory,manifest=manifest,
        source_features={k:np.concatenate(v) for k,v in source_features.items()})

class Runtime017:
    def __init__(self,p,cache,extension,stats,frozen,store,device):
        import torch
        from types import SimpleNamespace
        self.p,self.cache,self.extension,self.stats,self.frozen,self.store=p,cache,extension,stats,frozen,store
        self.device=torch.device(device);self.loss_flags=SimpleNamespace(q_floor=float(stats['loss']['q_floor']),peak_floor=float(stats['loss']['peak_floor']))
    def call(self,name,charges,fn,**detail):
        def run():
            value=fn()
            if self.device.type=='cuda':
                import torch
                torch.cuda.synchronize(self.device)
            return value
        return self.store.call(name,charges,run,**detail)
    def cache_rows(self,rows):
        import numpy as np
        import torch
        return tuple(torch.as_tensor(np.array(self.cache[k][rows],copy=True),dtype=torch.float64,device=self.device) for k in ('A','B','Q'))
    def noises(self,nz,epsilon):
        from n01.response008 import fixed_noises
        return self.call('frozen005_all3_extensions',dict(noise005_rows=len(epsilon),all_three_extensions_rows=3*len(epsilon)),lambda:fixed_noises(self.extension,nz,epsilon))
    def observe(self,carrier,*,ste):
        from n01.observation008 import observe_generated_carrier
        return self.call('hard_ADC_first_argmax',dict(hard_ADC_rows=len(carrier),hard_first_argmax_gather_rows=len(carrier)),
            lambda:observe_generated_carrier(carrier,ste=ste,q_floor=self.loss_flags.q_floor,return_carrier=True,return_indices=True))
    def signals(self,rows,streams,*,model=None,teacher=None,training=False):
        import torch
        from n01.model017 import assemble_cached017,teacher_shape_coefficients017
        z,zq,n,e,sign=streams.draw(len(rows),self.device,mixture=teacher=='T2')
        if teacher is not None:
            require(teacher in ('T1','T2'),'017 teacher identity')
            z=teacher_shape_coefficients017(z,teacher=teacher,sign=sign)
        else:require(model is not None,'017 student model required')
        chunk=self.p['training']['chunk_rows'] if training else len(rows);pieces=[];noises=[]
        for start in range(0,len(rows),chunk):
            stop=min(start+chunk,len(rows));A,B,Q=self.cache_rows(rows[start:stop]);size=len(A)
            charges=dict(shape_cached_basis_assembly_rows=size)
            if teacher is not None:
                a=A.new_tensor(.01*3.7103998046755806);b=A.new_tensor(.0025);charges['teacher_charge_module_rows']=size
                if training:charges['training_teacher_rows']=size
                out=self.call('known_'+teacher+'_signal',charges,lambda:assemble_cached017(A,B,Q,z[start:stop],zq[start:stop],a,b))
            else:
                charges['student_charge_module_rows']=size
                if training:charges['training_student_rows']=size
                out=self.call('student_cached_signal',charges,lambda:model(A,B,Q,z[start:stop],zq[start:stop]))
            pieces.append(out);noises.append(self.noises({k:v[start:stop] for k,v in n.items()},e[start:stop]))
        combined={k:torch.cat([part[k] for part in pieces],0) if isinstance(value,torch.Tensor) else value for k,value in pieces[0].items()}
        return combined,{h:torch.cat([part[h] for part in noises],0) for h in HYPOTHESES}

    def training_window(self,carrier,*,ste):
        import torch
        chunk=self.p['training']['chunk_rows']
        return torch.cat([self.observe(carrier[start:start+chunk],ste=ste)['window_V'] for start in range(0,len(carrier),chunk)],0)

    def guard(self,directory,started,*,shared=False):
        from run_wp2_014 import current_rss014
        limits=self.p['run_resource_limits'];elapsed=time.monotonic()-started
        size=sum(q.stat().st_size for q in Path(directory).rglob('*') if q.is_file())
        cap=self.p['shared_output_MiB'] if shared else limits['output_MiB']
        require(elapsed<=limits['wall_seconds'] and size<=cap*1024**2,'017 per-unit wall/output cap')
        rss=current_rss014();require(rss is not None and rss<=limits['job_tree_RSS_GiB']*1024**3,'017 current owned worker RSS cap')
        self.extension.assert_frozen()
        return dict(elapsed_seconds=elapsed,output_bytes=size,process_RSS_bytes=rss)

    def verify(self,*,files=False):
        frozen_verify017(self,self.original_model,self.store.checked,files=files)

    def shared_guard(self):
        shared=[q for q in self.store.root.iterdir() if q.name not in TRIALS]
        size=sum(q.stat().st_size if q.is_file() else sum(f.stat().st_size for f in q.rglob('*') if f.is_file()) for q in shared)
        require(size<=self.p['shared_output_MiB']*1024**2 and self.shared_elapsed<=7200,'017 whole shared cache/evaluation cap')
        return dict(shared_output_bytes=size,shared_elapsed_seconds=self.shared_elapsed)

class Engine017:
    def __init__(self,rt,teacher,mode,seed,out):
        import torch
        from n01.model017 import CachedFrozenQ1Charge017
        self.rt,self.teacher,self.mode,self.seed,self.out=rt,teacher,mode,seed,Path(out)
        require(teacher in ('T1','T2') and mode in ('paired','unpaired') and seed in (0,1),'017 trial identity')
        self.out.mkdir(parents=True,exist_ok=False);self.started=time.monotonic();self.update=0
        self.model=CachedFrozenQ1Charge017(device=rt.device)
        self.optimizer=rt.call('fresh_2theta_AdamW',dict(optimizers_created=1,trial_slots=1),
            lambda:torch.optim.AdamW(self.model.trainable_parameters(),lr=.01,weight_decay=0.))
        self.model.assert_optimizer_membership(self.optimizer)
        # No teacher/mode in Gaussian namespace: matched mode/teacher prefixes.
        self.student_stream=Streams017(dict(kind='training',replicate=seed,role='student'),rt.call)
        self.teacher_stream=Streams017(dict(kind='training',replicate=seed,role='teacher'),rt.call)
        sampler_roles=('shared_parent',) if mode=='paired' else ('student0','student1','teacher0','teacher1')
        self.samplers={role:ParentSampler017(rt.cache['fit'],dict(kind='parent',mode=mode,replicate=seed,role=role),rt.call) for role in sampler_roles}
        self.last_metrics={}

    def coverage(self):return {k:v.coverage() for k,v in self.samplers.items()}

    def checkpoint(self,name,*,failure=False):
        from run_wp2_014 import atomic_checkpoint
        state=dict(schema='n01/wp4-017-checkpoint/v1',run_id=RUN_ID,trial=self.out.name,teacher=self.teacher,mode=self.mode,
            seed=self.seed,G_updates=self.update,parameters={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
            optimizer=self.optimizer.state_dict(),student_streams=self.student_stream.state_dict(),teacher_streams=self.teacher_stream.state_dict(),
            parent_sampler_states={k:dict(generator=v.generator.get_state().clone(),seen=v.seen.copy()) for k,v in self.samplers.items()},
            cache_manifest_sha256=sha(self.rt.cache['directory']/'CACHE_MANIFEST.json'),frozen_state_sha256=self.rt.frozen,
            complete_step_boundary=not failure,last_metrics=self.last_metrics,counters=self.rt.store.counts.copy(),
            attempted_counters=self.rt.store.attempted.copy(),inflight=copy.deepcopy(self.rt.store.inflight),automatic_restart=False)
        if failure:
            require(not getattr(self.rt.store,'failure_checkpoint_attempted',False),'017 whole-job unique failure CP already attempted')
            self.rt.store.failure_checkpoint_attempted=True
            return self.rt.call('whole_job_unique_failure_checkpoint',{},lambda:atomic_checkpoint(self.out/name,state))
        return self.rt.call('complete_2theta_checkpoint',dict(checkpoint_saves_complete=1),lambda:atomic_checkpoint(self.out/name,state))

    def step(self):
        import numpy as np
        import torch
        from n01.model017 import energy_score_U017
        rt=self.rt;rt.guard(self.out,self.started);K=rt.p['training']['parents']
        if self.mode=='paired':
            parents=self.samplers['shared_parent'].draw(K);student_rows=teacher_rows=np.repeat(parents,2)
        else:
            student_rows=np.concatenate([self.samplers[k].draw(K) for k in ('student0','student1')])
            teacher_rows=np.concatenate([self.samplers[k].draw(K) for k in ('teacher0','teacher1')])
        self.optimizer.zero_grad(set_to_none=True)
        # Teachers and all their electronics are independent of student streams.
        with torch.no_grad():T,Tnoise=rt.signals(teacher_rows,self.teacher_stream,teacher=self.teacher,training=True)
        S,Snoise=rt.signals(student_rows,self.student_stream,model=self.model,training=True)
        scores=[]
        for h in HYPOTHESES:
            with torch.no_grad():target=rt.training_window(T['signal_pre']+Tnoise[h],ste=False)
            prediction=rt.training_window(S['signal_pre']+Snoise[h],ste=True)
            paired=self.mode=='paired'
            if paired:prediction,target=prediction.reshape(K,2,256),target.reshape(K,2,256)
            charges=dict(paired_training_energy_pair_distances=K*5) if paired else dict(
                unpaired_training_energy_pair_distances=2*(2*K)**2,unpaired_training_cross_retained_pairs=(2*K)**2,
                unpaired_training_self_ordered_retained_pairs=(2*K)*(2*K-1),unpaired_training_self_diagonal_computed_then_masked=2*K)
            scores.append(rt.call('proper_'+self.mode+'_ES_U',charges,
                lambda prediction=prediction,target=target:energy_score_U017(prediction,target,paired=paired,block_rows=64).mean()))
        loss=torch.stack(scores).mean();require(bool(torch.isfinite(loss)),'017 objective must be finite, signed values allowed')
        rt.call('one_training_backward',dict(training_backward_queries=1,new_AD_queries=1),lambda:loss.backward())
        parameters=self.model.trainable_parameters();grads=[float(p.grad.detach()) if p.grad is not None else None for p in parameters]
        require(all(v is not None and np.isfinite(v) for v in grads),'017 missing/nonfinite gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
        before=torch.stack([p.detach().clone() for p in parameters]);self.model.assert_optimizer_membership(self.optimizer)
        rt.call('one_2theta_optimizer_step',dict(G_updates=1,optimizer_steps=1),lambda:self.optimizer.step());self.update+=1
        after=torch.stack([p.detach().clone() for p in parameters]);delta=after-before;a,b=self.model.coefficients()
        require(bool(torch.isfinite(after).all()),'017 nonfinite actual parameter state')
        self.last_metrics=dict(G_updates=self.update,teacher=self.teacher,mode=self.mode,seed=self.seed,
            objective=float(loss.detach()),hypothesis_objectives={h:float(v.detach()) for h,v in zip(HYPOTHESES,scores)},
            preclip_gradients=grads,preclip_gradient_norm=float(norm),actual_parameter_deltas=delta.cpu().tolist(),
            theta_a=float(after[0]),theta_b=float(after[1]),a_Vns=float(a.detach()),b=float(b.detach()),
            parent_coverage=self.coverage(),teacher_student_all_latents_independent=True,proper_U_statistic=True,
            signed_loss_not_clamped=True,REAL_DEV_reads=0)
        with (self.out/'training.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(clean(self.last_metrics),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        if self.update==1 or self.update%100==0:
            rt.store.status(dict(current_trial=self.out.name,current_trial_G_updates=self.update))
        if self.update==1:
            require(all(v!=0 for v in grads) and bool(delta.ne(0).all()),'017 first update requires BOTH true gradients and actual theta changes')
            rt.verify(files=True)
            proof=dict(rt.store.base(),schema='n01/wp4-017-first-real-update/v1',actual_G_updates=1,
                trial=self.out.name,batch_completed_G_updates=rt.store.counts['G_updates'],trainable_parameters=2,
                trainable_parameter_names=['theta_a','theta_b'],gradient_nonzero=True,both_gradients_nonzero=True,
                actual_parameter_update_nonzero=True,both_parameter_updates_nonzero=True,
                frozen_A_005_M0_unchanged=True,frozen_state_sha256=rt.frozen,
                cache_manifest_sha256=sha(rt.cache['directory']/'CACHE_MANIFEST.json'),metrics=self.last_metrics)
            write_new(self.out/'FIRST_UPDATE_PROOF.json',proof)
            if not (rt.store.root/'FIRST_UPDATE_PROOF.json').exists():rt.store.json('FIRST_UPDATE_PROOF.json',proof)
        return self.last_metrics

def save_npz017(path,**arrays):
    import numpy as np
    with Path(path).open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())

def evaluate_kernel017(rt,directory,*,model=None,teacher=None):
    """One registered kernel snapshot. No REAL/CDF/loss refit or second ADC."""
    import numpy as np
    import torch
    from n01.stats017 import scalar_features017
    from n01.saved_array_kernels017 import scalar_moments
    from run_wp1_013 import observation_rows,lossless_window_codes
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    e=rt.p['evaluation'];N=len(rt.cache['Q']);R=e['draws'];rows=np.repeat(np.arange(N),R);draw=np.tile(np.arange(R),N)
    identities=rt.cache['ids'][rows];evaluation_states={}
    with torch.no_grad():
        for bank in range(e['banks']):
            streams=Streams017(dict(kind='evaluation',role='teacher' if teacher else 'student',bank=bank),rt.call)
            evaluation_states[str(bank)]={'initial':streams.state_dict()};prefix=[];charge=[];mu=[];variance=[]
            pieces={h:[] for h in HYPOTHESES}
            for start in range(0,len(rows),e['chunk_rows']):
                selected=rows[start:start+e['chunk_rows']];size=len(selected)
                out,noise=rt.signals(selected,streams,model=model,teacher=teacher)
                rt.call('evaluation_signal_rows',dict(evaluation_signal_rows=size),lambda:None)
                charge.append(out['qsig'].cpu().numpy());mu.append(out['Q_mu_Vns'].cpu().numpy());variance.append(out['theory_variance_Vns2'].cpu().numpy())
                first_prefix=None
                for h in HYPOTHESES:
                    carrier=out['signal_pre']+noise[h];obs=rt.observe(carrier,ste=False)
                    fields=observation_rows(obs,carrier,rt.loss_flags)
                    if first_prefix is None:first_prefix=fields['baseline100'];prefix.append(first_prefix)
                    else:require(np.array_equal(first_prefix.view(np.uint64),fields['baseline100'].view(np.uint64)),'017 all3 actual ADC prefix mismatch')
                    encoded=lossless_window_codes(fields)
                    values,morph=scalar_features017(fields['window_V'],fields['baseline100'],fields['peak_index'])
                    require(size%R==0,'017 parent may not cross evaluation chunk')
                    wave=fields['window_V'].reshape(-1,R,256)
                    flags={k:v for k,v in fields.items() if k.startswith(('flag__','diagnostic__'))}
                    flags.update({'flag__'+k:v for k,v in morph.items() if np.asarray(v).dtype.kind=='b'})
                    pieces[h].append(dict(encoded=encoded,flags=flags,scalar=values,
                        wave_mean=wave.mean(1),wave_variance=wave.var(1,ddof=1)))
                rt.guard(directory,started,shared=teacher is not None or model is not None and directory.name=='initial')
            baseline=np.concatenate(prefix)
            save_npz017(directory/f'bank{bank}_BASELINE100.npz',baseline100=baseline,source_parent_row=rows,source_parent_ids=identities,draw_index=draw)
            save_npz017(directory/f'bank{bank}_CHARGE.npz',q=np.concatenate(charge).reshape(N,R),actual_mu=np.concatenate(mu).reshape(N,R),
                actual_v=np.concatenate(variance).reshape(N,R),source_parent_ids=rt.cache['ids'],source_parent_rows=np.arange(N),
                draw_index=np.arange(R),bank_index=np.asarray(bank),
                cache_manifest_sha256=np.asarray(sha(rt.cache['directory']/'CACHE_MANIFEST.json')))
            for h in HYPOTHESES:
                parts=pieces.pop(h);coded={k:np.concatenate([part['encoded'][k] for part in parts]) for k in ('ADC_crop_codes','baseline_mean_V','peak_index')}
                if any('IEEE754_reconstruction_XOR_uint64' in part['encoded'] for part in parts):
                    coded['IEEE754_reconstruction_XOR_uint64']=np.concatenate([part['encoded'].get('IEEE754_reconstruction_XOR_uint64',np.zeros(part['encoded']['ADC_crop_codes'].shape,np.uint64)) for part in parts])
                flags={k:np.concatenate([part['flags'][k] for part in parts]) for k in parts[0]['flags']}
                coded.update(flags,source_parent_row=rows,source_parent_ids=identities,draw_index=draw,bank_index=np.full(len(rows),bank,np.int64))
                save_npz017(directory/f'bank{bank}_{h}_HARD_ROWS.npz',**coded)
                scalar=np.concatenate([part['scalar'] for part in parts]).reshape(N,R,9);moments=scalar_moments(scalar)
                save_npz017(directory/f'bank{bank}_{h}_REPORT.npz',scalar9=scalar,
                    wave_mean=np.concatenate([part['wave_mean'] for part in parts]),wave_variance=np.concatenate([part['wave_variance'] for part in parts]),
                    **{'scalar_'+k:v for k,v in moments.items()})
                del parts,coded,scalar,flags,moments
            evaluation_states[str(bank)]['final']=streams.state_dict()
            rt.guard(directory,started,shared=teacher is not None or directory.name=='initial')
        # RNG controls are compact metadata arrays, not model reload checkpoints.
        state_arrays={f'{bank}_{endpoint}_{stream}':value.numpy() for bank,record in evaluation_states.items() for endpoint,states in record.items() for stream,value in states.items()}
        save_npz017(directory/'EVALUATION_RANDOM_STATES.npz',**state_arrays)
        if teacher is not None:
            streams=Streams017(dict(kind='coverage',teacher=teacher),rt.call)
            initial_state=streams.generators['PIT'].get_state().numpy().copy();uniforms=streams.PIT(N)
            save_npz017(directory/'PIT_UNIFORMS.npz',uniforms=uniforms,initial_state=initial_state,
                final_state=streams.generators['PIT'].get_state().numpy(),initial_seed=np.asarray(seed017(dict(kind='coverage',teacher=teacher,stream='PIT'))))
        rt.call('one_evaluation_kernel_complete',dict(evaluation_kernel_snapshots=1),lambda:None)
    derive_kernel017(directory,N)
    evaluation_axes017(directory,rt)
    a,b=(.01*3.7103998046755806,.0025) if teacher is not None else [float(v.detach()) for v in model.coefficients()]
    metadata=dict(schema='n01/wp4-017-evaluation/v1',teacher=teacher,student=model is not None,parents=N,banks=2,draws=32,a_Vns=a,b=b,
        cache_manifest_sha256=sha(rt.cache['directory']/'CACHE_MANIFEST.json'),codec='original013 lossless code256 + actual baseline mean + optional IEEE XOR',
        exact_saved_Y_via_codec=True,exact_Y_second_float64_copy=False,full_stochastic_preADC_saved=False,
        arrays={q.name:dict(bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(directory.glob('*.npz'))},
        native32_moments=True,all_original_parent_draw_flags_retained=True,scientific_acceptance=False)
    write_new(directory/'EVALUATION.json',metadata)
    return directory

def decode_hard017(path):
    """Decode our saved output only. The original raw dataset is never reopened."""
    import numpy as np
    with np.load(path,allow_pickle=False) as a:
        codes=a['ADC_crop_codes'];mean=a['baseline_mean_V'];peaks=a['peak_index']
        reconstructed=codes.astype(np.float64)/16383-.5-mean[:,None]
        if 'IEEE754_reconstruction_XOR_uint64' in a.files:
            reconstructed=(reconstructed.view(np.uint64)^a['IEEE754_reconstruction_XOR_uint64']).view(np.float64)
        flags={k:a[k].copy() for k in a.files if k.startswith('flag__')}
        return reconstructed,flags,peaks

def energy_vectors017(rt,student,teacher,*,teacher_reference=False):
    import numpy as np
    import torch
    from n01.model017 import energy_score_U017
    N=rt.p['evaluation']['parents'];R=32;result={}
    with torch.no_grad():
        for h in HYPOTHESES:
            if teacher_reference:
                T0=decode_hard017(Path(teacher)/f'bank0_{h}_HARD_ROWS.npz')[0].reshape(N,R,256)
                T1=decode_hard017(Path(teacher)/f'bank1_{h}_HARD_ROWS.npz')[0].reshape(N,R,256)
                # One cross plus two self matrices. ES_U calls each direction
                # would recompute cross; use explicit same-matrix reference.
                values=[]
                for start in range(0,N,64):
                    from n01.model017 import metric_features017
                    A=metric_features017(torch.as_tensor(T0[start:start+64],device=rt.device))
                    B=metric_features017(torch.as_tensor(T1[start:start+64],device=rt.device))
                    def ref():
                        cross=torch.cdist(A,B,compute_mode='donot_use_mm_for_euclid_dist').mean((-2,-1))
                        mask=~torch.eye(R,dtype=torch.bool,device=rt.device)
                        self0=torch.where(mask,torch.cdist(A,A,compute_mode='donot_use_mm_for_euclid_dist'),0.).sum((-2,-1))/(R*(R-1))
                        self1=torch.where(mask,torch.cdist(B,B,compute_mode='donot_use_mm_for_euclid_dist'),0.).sum((-2,-1))/(R*(R-1))
                        return cross-.25*(self0+self1)
                    n=len(A);values.append(rt.call('teacher32_reverse_reference',dict(evaluation_energy_pair_distances=n*3*R*R),ref).cpu().numpy())
                result[h]=np.concatenate(values);del T0,T1
            else:
                banks=[]
                for bank in range(2):
                    S=decode_hard017(Path(student)/f'bank{bank}_{h}_HARD_ROWS.npz')[0].reshape(N,R,256)
                    T=decode_hard017(Path(teacher)/f'bank{bank}_{h}_HARD_ROWS.npz')[0].reshape(N,R,256);values=[]
                    for start in range(0,N,64):
                        n=min(64,N-start);s=torch.as_tensor(S[start:start+n],device=rt.device);t=torch.as_tensor(T[start:start+n],device=rt.device)
                        values.append(rt.call('conditional32_ES_U',dict(evaluation_energy_pair_distances=n*2*R*R),
                            lambda:energy_score_U017(s,t,paired=True,block_rows=64)).cpu().numpy())
                    banks.append(np.concatenate(values));del S,T
                result[h]=.5*(banks[0]+banks[1])
        if not teacher_reference:rt.call('one_kernel_teacher_score_comparison',dict(evaluation_energy_candidate_comparisons=1),lambda:None)
    return result

def derived_hypothesis017(directory,h,N):
    import numpy as np
    from n01.stats017 import scalar_banks017,pooled_wave017
    banks=[];means=[];variances=[];flags=[]
    for b in range(2):
        with np.load(Path(directory)/f'bank{b}_{h}_REPORT.npz',allow_pickle=False) as ar:
            banks.append(ar['scalar9'].copy());means.append(ar['wave_mean'].copy());variances.append(ar['wave_variance'].copy())
        with np.load(Path(directory)/f'bank{b}_{h}_HARD_ROWS.npz',allow_pickle=False) as ar:
            flags.append({k:ar[k].reshape(N,32).mean(1) for k in ar.files if k.startswith('flag__')})
    scalar=np.stack(banks,axis=1);moments=scalar_banks017(scalar)
    mean,variance=pooled_wave017(means[0],variances[0],means[1],variances[1])
    require(flags[0].keys()==flags[1].keys(),'017 same kernel bank flag schema')
    return scalar,moments,mean,variance,{k:.5*(flags[0][k]+flags[1][k]) for k in flags[0]}

def derive_kernel017(directory,N):
    for h in HYPOTHESES:
        scalar,moments,mean,variance,flags=derived_hypothesis017(directory,h,N)
        save_npz017(Path(directory)/f'DERIVED64_{h}.npz',wave_mean=mean,wave_variance=variance,
            **{'scalar_'+k:v for k,v in moments['combined'].items()},
            feature_support=moments['feature_support'],pair_support=moments['pair_support'],
            full_slot_undefined_fraction=moments['full_slot_undefined_fraction'],**{'fraction_'+k:v for k,v in flags.items()})

def evaluation_axes017(directory,rt):
    """Descriptive four-axis summaries; no extra forward, RNG or gate choice."""
    import numpy as np
    from n01.stats017 import FEATURES017
    result={};N=len(rt.cache['Q'])
    for h in HYPOTHESES:
        scalar,moments,mean,variance,flags=derived_hypothesis017(directory,h,N)
        flat=scalar.reshape(-1,9);q=flat[:,0];floor=rt.loss_flags.q_floor
        def features(rows):
            values=flat[rows];defined=np.isfinite(values);counts=defined.sum(0);averages=np.full(9,np.nan)
            np.divide(np.where(defined,values,0.).sum(0),counts,out=averages,where=counts>0)
            return dict(rows=int(rows.sum()),finite_counts=counts,means=averages,undefined_counts=len(values)-counts)
        conditions={'Q_nonpositive':q<=0,'positive_Q_below_original_q_floor':(q>0)&(q<floor),'positive_Q_at_or_above_original_q_floor':q>=floor}
        trends={}
        sources={'actual_A_Q':rt.cache['Q'],**rt.cache.get('source_features',{})}
        for name,x in sources.items():
            y=moments['combined']['mean'][:,0];valid=np.isfinite(x)&np.isfinite(y)
            covariance=float(np.cov(np.asarray(x)[valid],y[valid],ddof=1)[0,1]) if valid.sum()>1 else None
            trends[name]=dict(parents=int(valid.sum()),covariance_with_parent_mean_observed_Q=covariance,
                source_values_index='cache/PARENT_SPLIT.npz' if name!='actual_A_Q' else 'cache/Q.npy',not_a_causal_identification_claim=True)
        result[h]=dict(whole_population=features(np.ones(len(flat),bool)),output_Q_conditioned={k:features(v) for k,v in conditions.items()},
            SOURCE_A_Q_and_input_trends=trends,same_parent_repeats=dict(native_bank_draws=32,derived_draws=64,
                parent_count=N,native_moments_files=[f'bank{b}_{h}_REPORT.npz' for b in range(2)],derived_moments_file=f'DERIVED64_{h}.npz'),
            full_original_slot_flag_fractions={k:float(v.mean()) for k,v in flags.items()})
    compact_report017(Path(directory)/'EVALUATION_AXES.json',dict(schema='n01/wp4-017-four-axes/v1',hypotheses=result,
        scalar9=list(FEATURES017),physical_parents=N,draws_per_parent=64,MC_draws_are_not_independent_physical_parents=True,
        output_Q_conditions_are_descriptive_not_calibrated_energy=True,original013_T_not_redefined=True,
        original_REAL_absolute_T='not evaluated: REAL reads0; synthetic recovery uses the separately frozen017 gates'))

def compact_report017(path,value):
    """Selected JSON is compact; all full-parent arrays remain immutable remote."""
    import numpy as np
    arrays={}
    def walk(x,key='root'):
        if isinstance(x,np.ndarray):
            arrays[key]=x;return dict(array=key,file=Path(path).stem+'.npz',shape=list(x.shape),dtype=str(x.dtype))
        if isinstance(x,dict):return {str(k):walk(v,key+'/'+str(k)) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [walk(v,key+'/'+str(i)) for i,v in enumerate(x)]
        return clean(x)
    result=walk(value)
    if arrays:save_npz017(Path(path).with_suffix('.npz'),**arrays)
    write_new(path,result);return result

def compare_kernel017(rt,directory,teacher_directory,initial_energy,reference_energy,final_energy,model,trial):
    import numpy as np
    from n01.stats017 import (FEATURES017,TIME017,groups_weights017,variance_curve_cell017,waveform_cells017,
        covariance_cell017,timing_cell017,coverage017,coverage_cells017,missingness_cells017)
    directory=Path(directory);test=rt.cache['test'];Q=rt.cache['Q'];ids=rt.cache['ids'];N=len(Q)
    groups,weights,cuts=groups_weights017(Q[rt.cache['fit']],Q[test],ids[test]);cells=[]
    a,b=[float(x.detach()) for x in model.coefficients()]
    for group,g in groups.items():
        for weight,w in weights.items():cells.append(dict(variance_curve_cell017(Q[test],a,b,g,w),metric='analytic_charge_variance',group=group,weights=weight))
    with np.load(Path(teacher_directory)/'PIT_UNIFORMS.npz',allow_pickle=False) as ar:U=ar['uniforms']
    for hindex,h in enumerate(HYPOTHESES):
        S,sm,Smean,Svar,Sflags=derived_hypothesis017(directory,h,N)
        T,tm,Tmean,Tvar,Tflags=derived_hypothesis017(teacher_directory,h,N)
        coverage=coverage017(S,T,U[:,:,:,:,hindex])
        # All parents retained in the raw arrays, including fit-parent diagnostics.
        save_npz017(directory/f'PARENT_COMPARISON_{h}.npz',parent_ids=ids,test_rows=test,
            energy_final=final_energy[h],energy_initial=initial_energy[h],energy_teacher_reference=reference_energy[h],
            **{'coverage_'+k:v for k,v in coverage.items() if isinstance(v,np.ndarray)})
        selected_S,selected_T=S[test],T[test]
        sub=lambda obj:{k:([sub(v) for v in x] if isinstance(x,list) else x[test] if isinstance(x,np.ndarray) else x) for k,x in obj.items()}
        sm_test,tm_test=sub(sm),sub(tm)
        C={k:(v[test] if isinstance(v,np.ndarray) else v) for k,v in coverage.items()}
        for group,g in groups.items():
            for weight,w in weights.items():
                tag=dict(group=group,weights=weight,hypothesis=h)
                waves=waveform_cells017(Smean[test],Svar[test],Tmean[test],Tvar[test],g,w)
                for name,item in waves.items():cells.append(dict(item,metric='hard_wave_'+name,**tag))
                cov=covariance_cell017(selected_S,selected_T,g,w,moments=(sm_test,tm_test))
                cells.append(dict(cov,metric='conditional_cov9',**tag))
                for f in TIME017:cells.append(dict(timing_cell017(selected_S,selected_T,f,g,w,moments=(sm_test,tm_test)),
                    metric='time_'+FEATURES017[f],**tag))
                for item in coverage_cells017(C,g,w):cells.append(dict(item,metric='PIT_coverage',**tag))
                for item in missingness_cells017(sm_test,tm_test,{k:v[test] for k,v in Sflags.items()},
                    {k:v[test] for k,v in Tflags.items()},g,w):cells.append(dict(item,metric='missingness_or_censor',**tag))
                from n01.stats017 import energy_cell017
                cells.append(dict(energy_cell017(final_energy[h][test],initial_energy[h][test],reference_energy[h][test],g,w),
                    metric='conditional_energy',**tag))
        del S,T,sm,tm,Smean,Svar,Tmean,Tvar,selected_S,selected_T
    # Secondary identifiability is descriptive; cannot reopen the primary gate.
    X=np.column_stack((Q[test],Q[test]**2));X=X/np.linalg.norm(X,axis=0);eigen=np.linalg.eigvalsh(X.T@X)
    conditioning=float(eigen[0]/eigen[-1]);aT=.01*3.7103998046755806;bT=.0025
    report=dict(schema='n01/wp4-017-comparison/v1',trial=trial,cells=cells,
        all_required_cells_pass=all(x['status']=='pass' for x in cells),Q_group_cuts=cuts,
        heldout_parent_ids=ids[test],heldout_source_rows=test,weights=weights,groups=groups,
        original_physical_weights_preserved=True,parameter_recovery_secondary=dict(a=a,b=b,
            a_relative_error=abs(a-aT)/aT,b_relative_error=abs(b-bT)/bT,scaled_Gram_eigenvalue_ratio=conditioning,
            individually_identifiable=conditioning>=1e-4,secondary_25percent_pass=(abs(a-aT)/aT<=.25 and abs(b-bT)/bT<=.25) if conditioning>=1e-4 else None),
        MC_resolution_is_empirical_not_64draw_oracle=True,full_per_parent_arrays_remain_remote=True,
        REAL_hardware_acceptance=False,scientific_status='response_match_not_established')
    compact_report017(directory.parent/'COMPARISON.json',report)
    return report

def frozen_verify017(rt,original_model,checked,*,files=False):
    from n01.response008 import state_digest
    from n01.model010 import tensor_state_digest010
    rt.extension.assert_frozen()
    require(state_digest(original_model.baseline.state_dict())==rt.frozen['baseline']
        and state_digest(rt.extension.state_dict())==rt.frozen['extension']
        and tensor_state_digest010(original_model.basis_head)==rt.cache['manifest']['M0_head_tensor_sha256'],
        '017 original A/005/M0 tensor mutation')
    for name,row in rt.cache['manifest']['cache_files'].items():
        path=rt.cache['directory']/name
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'017 fixed cache bytes changed')
    if files:unchanged(checked)

def artifact_inventory017(store):
    files={q.relative_to(store.root).as_posix():dict(sha256=sha(q),bytes=q.stat().st_size)
        for q in sorted(store.root.rglob('*')) if q.is_file() and q not in
        (store.root/'status.json',store.root/'ARTIFACT_INVENTORY.json',store.root/'COMPLETE.json')}
    store.json('ARTIFACT_INVENTORY.json',dict(schema='n01/wp4-017-artifact-inventory/v1',files=files,
        raw_arrays_retained_remote=True,selected_package_is_not_full_raw_evidence=True))

def run017(config,config_sha256,output,device):
    p,paths,checked=gate017(config,config_sha256,output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(p['CPU_threads'])
    store=Store017(output,p,checked);store.status();engine=None;stage_gate=None
    try:
        store.json('BINDING.json',dict(configuration_sha256=config_sha256,immutable_files=checked,budget_caps=p['budget_caps'],before_numeric_decode=True))
        import torch
        from n01.model017 import CachedFrozenQ1Charge017
        from n01.stats017 import stage_gate017
        from n01.evaluate_window008 import _memory_limit
        from run_wp2_014 import atomic_checkpoint
        torch.set_num_threads(p['CPU_threads']);torch.use_deterministic_algorithms(True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        _memory_limit(dict(gpu_memory_limit_GiB=p['resource_limits']['GPU_GiB']),device)
        shared_start=time.monotonic();data,original,extension,stats,frozen=prepare_numeric017(p,paths,store,device)
        cache=build_frozen_cache017(original,data['source_carrier384'],data['source_ids'],store.root/'cache',store.call,
            chunk=p['cache_chunk_parents'],heldout=p['training']['test_parents'],original_weights=data.get('original_source_weights'))
        del data
        rt=Runtime017(p,cache,extension,stats,frozen,store,device)
        rt.original_model=original;rt.shared_elapsed=time.monotonic()-shared_start;rt.shared_guard()
        initial=CachedFrozenQ1Charge017(device=device)
        rt.call('shared_initial_2theta_checkpoint',dict(checkpoint_saves_complete=1),lambda:atomic_checkpoint(store.root/'initial.pt',dict(
            schema='n01/wp4-017-initial-checkpoint/v1',parameters=initial.state_dict(),trainable_parameters=2,
            cache_manifest_sha256=sha(cache['directory']/'CACHE_MANIFEST.json'),frozen_state_sha256=frozen)))
        completed=[];comparisons=[];teacher_dirs={};initial_energy={};reference_energy={}
        for index,name in enumerate(TRIALS):
            teacher,mode,seedlabel=name.split('_');seed=int(seedlabel[-1])
            if index==2:
                stage_gate=stage_gate017(comparisons[:2]);compact_report017(store.root/'STAGE1_GATE.json',stage_gate)
                if not stage_gate['stage2_allowed']:break
            engine=Engine017(rt,teacher,mode,seed,store.root/name)
            if index==0:
                store.json('INIT_PROOF.json',dict(store.base(),schema='n01/wp4-017-init-proof/v1',native_initialized=True,
                    first_trial=name,cache_manifest_sha256=sha(cache['directory']/'CACHE_MANIFEST.json'),all_6593_Q_positive=True,
                    one_pass_A_decoded_M0_cache_complete=True,known_T1_initial_RMS_contrast_at_least3LSB=True,
                    mean_adapter_enabled=False,trainable_parameters=2,trainable_parameter_names=['theta_a','theta_b'],
                    frozen_state_sha256=frozen,fresh_AdamW=True,REAL_DEV_reads=0,known_bases_not_unseen_REAL_CV=True))
            # First actual update precedes the large full-parent evaluation.
            while engine.update<p['G_updates']:
                engine.step()
                if engine.update in p['training']['checkpoints']:
                    frozen_verify017(rt,original,checked,files=True)
                    engine.checkpoint('final.pt' if engine.update==p['G_updates'] else 'step_1000.pt')
            require(all(x['all_covered'] and x['unique']==p['training']['fit_parents'] for x in engine.coverage().values()),
                '017 incomplete IID fit-parent stream coverage; no repair draws')
            if teacher not in teacher_dirs:
                shared_start=time.monotonic()
                teacher_dirs[teacher]=evaluate_kernel017(rt,store.root/teacher,teacher=teacher)
                if index==0:evaluate_kernel017(rt,store.root/'initial',model=initial)
                initial_energy[teacher]=energy_vectors017(rt,store.root/'initial',teacher_dirs[teacher])
                reference_energy[teacher]=energy_vectors017(rt,None,teacher_dirs[teacher],teacher_reference=True)
                shared_duration=time.monotonic()-shared_start;rt.shared_elapsed+=shared_duration;engine.started+=shared_duration
                rt.shared_guard()
            final=evaluate_kernel017(rt,engine.out/'evaluation',model=engine.model)
            final_energy=energy_vectors017(rt,final,teacher_dirs[teacher])
            comparison=compare_kernel017(rt,final,teacher_dirs[teacher],initial_energy[teacher],reference_energy[teacher],final_energy,engine.model,name)
            frozen_verify017(rt,original,checked,files=True)
            resources=rt.guard(engine.out,engine.started)
            trial_record=dict(schema='n01/wp4-017-trial-complete/v1',native_status='COMPLETE',trial=name,teacher=teacher,mode=mode,seed=seed,
                G_updates=engine.update,coverage=engine.coverage(),frozen_A_005_M0_unchanged=True,resources=resources,
                trainable_parameter_names=['theta_a','theta_b'],last_metrics=engine.last_metrics,
                all_required_cells_pass=comparison['all_required_cells_pass'],COMPARISON_sha256=sha(engine.out/'COMPARISON.json'),
                scientific_acceptance=False,known_response_scope_only=True)
            write_new(engine.out/'COMPLETE.json',trial_record)
            completed.append(dict(directory=name,G_updates=engine.update,teacher=teacher,mode=mode,seed=seed,COMPLETE_sha256=sha(engine.out/'COMPLETE.json')))
            comparisons.append(comparison);engine=None
            if index==1:
                require(time.monotonic()-store.started<=p['stage1_wall_seconds'],'017 stage1 wall cap')
                require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=p['stage1_output_MiB']*1024**2,'017 stage1 output cap')
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
        require(stage_gate is not None,'017 missing two-trial stage1 gate')
        full=stage_gate['stage2_allowed'];expected=p['budget_caps'] if full else p['stage1_budget_caps']
        require([r['directory'] for r in completed]==list(TRIALS if full else TRIALS[:2]),'017 completed actual trial set differs')
        require(not store.inflight and store.counts==store.attempted==expected,'017 exact terminal additive counts differ')
        frozen_verify017(rt,original,checked,files=True)
        artifact_inventory017(store)
        result=dict(store.base(),schema='n01/wp4-017-batch-complete/v1',native_status='COMPLETE',trials=completed,
            G_updates_per_trial=p['G_updates'],stage1_gate=clean(stage_gate),stage2_entered=full,
            terminal_scope='full_eight_trials' if full else 'stage1_closed_normally',
            frozen_A_005_M0_unchanged=True,cache_complete=True,no_REAL_DEV=True,
            scientific_acceptance=False,scientific_status='response_match_not_established',root_interpretation_pending=True)
        store.json('COMPLETE.json',result);store.status(dict(native_status='COMPLETE'))
        require(sum(q.stat().st_size for q in store.root.rglob('*') if q.is_file())<=p['resource_limits']['output_MiB']*1024**2,'017 final batch output cap')
        return result
    except BaseException as error:
        failure=dict(store.base(),schema='n01/wp4-017-batch-failure/v1',native_status='FAILED',error_type=type(error).__name__,error=str(error),
            automatic_restart=False,current_trial=None if engine is None else engine.out.name,
            current_trial_G_updates=None if engine is None else engine.update,inflight_completion='unknown; never assert completed')
        store.json('FAILED.json',failure);store.status(failure)
        if engine is not None:
            try:engine.checkpoint('failure.pt',failure=True);write_new(engine.out/'FAILED.json',failure)
            except BaseException as saving_error:store.json('FAILURE_CHECKPOINT_ERROR.json',dict(error=repr(saving_error),original_error=repr(error)))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    def terminate(signum,frame):raise InterruptedError('Owned017 received signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate);result=run017(args.config,args.config_sha256,args.output,args.device)
    print(json.dumps({k:result[k] for k in ('native_status','scientific_training_updates','terminal_scope')},allow_nan=False),flush=True)

if __name__=='__main__':main()
