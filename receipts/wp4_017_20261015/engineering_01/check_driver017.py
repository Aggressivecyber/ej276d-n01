"""Manufactured CPU cache/Engine/evaluation; never original numeric inputs."""
from pathlib import Path
import copy
import ctypes
import hashlib
import json
import os
import sys
import time
if os.name=='nt':
    import platform
    platform._wmi=None
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
print('artificial driver importing CPU Torch',flush=True)
import numpy as np
import torch
from run_wp4_017 import (settings017,budget_caps017,Store017,Runtime017,Engine017,Streams017,
    build_frozen_cache017,evaluate_kernel017,energy_vectors017,compare_kernel017,save_npz017,sha,write_new,decode_hard017)
from run_wp2_014 import atomic_checkpoint
from n01.model017 import CachedFrozenQ1Charge017,deterministic_loading017
from n01.model008 import canonical_charge008
from n01.response008 import state_digest
from n01.stats017 import stage_gate017
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
checks={};before_global=torch.get_rng_state().clone()
def check(name,value):
    assert bool(value),name
    checks[name]=True;print(name,flush=True)
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError):checks[name]=True
    else:raise AssertionError(name)

class FakeKnown(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with deterministic_loading017():
            self.baseline=torch.nn.Module();self.baseline.mean_head=torch.nn.Conv1d(1,1,1)
            self.basis_head=torch.nn.Conv1d(32,8,5,padding=2)
        self.calls=0
    def _baseline_once(self,x):
        self.calls+=1
        return {'signal_pre':x},x.float()[:,None,:].expand(-1,32,-1).clone()
    def forward(self,x,z):
        out,decoded=self._baseline_once(x)
        A=out['signal_pre'];B=torch.zeros(len(x),8,384,dtype=torch.float64)
        for k in range(8):
            shape=torch.zeros(384,dtype=torch.float64);shape[112:144]=torch.sin(torch.arange(32,dtype=torch.float64)*(k+1)*torch.pi/16)
            shape[112:144]-=shape[112:144].mean();B[:,k,:]=shape*.0001
        return {'base_signal_pre_V':A,'voltage_basis_V':B,'q_source':canonical_charge008(x),
            'source_peak_index':x[:,112:144].argmax(1)+112,'source_baseline_mean_V':x[:,:100].mean(1)}

class FakeExtension(torch.nn.Module):
    def assert_frozen(self):return True

class ManualRuntime(Runtime017):
    def noises(self,nz,epsilon):
        def value():
            prefix=nz['baseline_gaussian'][:,0,:].double()*1e-4
            tail=epsilon[:,0,:].double()*1e-4
            return {h:torch.cat((prefix,tail*scale),1) for h,scale in zip(('fast','slow','common_level'),(1.,1.1,.9))}
        return self.call('artificial_frozen_electronics_formula_no_NN',dict(noise005_rows=len(epsilon),all_three_extensions_rows=3*len(epsilon)),value)

p=settings017();p['training']=dict(p['training'],fit_parents=2,test_parents=4,all_parents=6,parents=8,checkpoints=[1,2]);p['G_updates']=2
p['evaluation']=dict(p['evaluation'],parents=6,chunk_rows=128);p['CPU_threads']=2
p['resource_limits']=dict(p['resource_limits'],wall_seconds=600);p['budget_caps']=budget_caps017(p)
p['stage1_budget_caps']=budget_caps017(p,True)
store=Store017(OUT/'artifacts',p,{},artificial=True);store.status()
store.call('manufactured_CP_load_accounting_only_no_file',{'checkpoint_loads':3},lambda:None)
x=torch.zeros(6,384,dtype=torch.float64)
for i in range(6):
    t=torch.arange(384,dtype=torch.float64);x[i]=(.03+i*.005)*torch.exp(-((t-128)/12)**2);x[i,:100]=0
source=x.numpy().copy();ids=np.arange(6)+100
fake=FakeKnown().eval().requires_grad_(False)
cache=build_frozen_cache017(fake,source,ids,store.root/'cache',store.call,chunk=2,heldout=4)
check('one_pass_chunk_decoded_cache_and_positive_Q',fake.calls==3 and cache['A'].dtype==cache['B'].dtype==np.float64
    and len(cache['fit'])==2 and len(cache['test'])==4 and np.all(cache['Q']>0))
extension=FakeExtension();frozen={'baseline':state_digest(fake.baseline.state_dict()),'noise005':state_digest(extension.state_dict()),'extension':state_digest(extension.state_dict())}
rt=ManualRuntime(p,cache,extension,{'loss':{'q_floor':1e-6,'peak_floor':1e-5}},frozen,store,'cpu')
rt.original_model=fake;rt.shared_elapsed=0
initial=CachedFrozenQ1Charge017();rt.call('shared_initial_artificial_checkpoint',{'checkpoint_saves_complete':1},lambda:atomic_checkpoint(store.root/'initial.pt',initial.state_dict()))
trial_reports=[];trial_dirs=[];teacher_dir=initial_energy=reference=None
for seed in (0,1):
    name=f'T1_paired_seed{seed}';engine=Engine017(rt,'T1','paired',seed,store.root/name)
    while engine.update<2:
        engine.step();engine.checkpoint('step_1000.pt' if engine.update==1 else 'final.pt')
    check(name+'_both_theta_step_and_full_IID_support',engine.last_metrics['actual_parameter_deltas'][0]!=0
        and engine.last_metrics['actual_parameter_deltas'][1]!=0 and all(v['all_covered'] for v in engine.coverage().values()))
    if teacher_dir is None:
        teacher_dir=evaluate_kernel017(rt,store.root/'T1',teacher='T1')
        evaluate_kernel017(rt,store.root/'initial',model=initial)
        initial_energy=energy_vectors017(rt,store.root/'initial',teacher_dir)
        reference=energy_vectors017(rt,None,teacher_dir,teacher_reference=True)
    final=evaluate_kernel017(rt,engine.out/'evaluation',model=engine.model)
    final_energy=energy_vectors017(rt,final,teacher_dir)
    report=compare_kernel017(rt,final,teacher_dir,initial_energy,reference,final_energy,engine.model,name)
    trial_reports.append(report);trial_dirs.append(engine.out)
check('artificial_stage1_exact_all_counters',store.counts==store.attempted==p['stage1_budget_caps'] and not store.inflight)
gate=stage_gate017(trial_reports)
check('tiny_unsupported_groups_close_stage2_with_all_cells',not gate['stage2_allowed'] and gate['status']=='closed'
    and gate['required_cell_count']>1392 and all(c['status']=='pending' for c in gate['nonpass']))
missing=copy.deepcopy(trial_reports);missing[0]['cells'].pop(0);reject('missing_primary_cell_rejected',lambda:stage_gate017(missing))
duplicate=copy.deepcopy(trial_reports);duplicate[0]['cells'].append(duplicate[0]['cells'][0]);reject('duplicate_primary_cell_rejected',lambda:stage_gate017(duplicate))
passed=copy.deepcopy(trial_reports)
for r in passed:
    for c in r['cells']:c['status']='pass'
check('complete_artificial_matrix_opens_gate',stage_gate017(passed)['stage2_allowed'])
passed[1]['cells'][2]['status']='pending';check('single_pending_closes_gate',not stage_gate017(passed)['stage2_allowed'])
passed[1]['cells'][2]['status']='fail';check('single_fail_closes_gate',not stage_gate017(passed)['stage2_allowed'])
# No shared electronics/charge/shape prefixes between score cross samples.
s=Streams017({'kind':'training','replicate':0,'role':'student'});t=Streams017({'kind':'training','replicate':0,'role':'teacher'})
S=s.draw(4,'cpu');T=t.draw(4,'cpu')
check('every_teacher_student_latent_stream_independent',all(not torch.equal(a,b) for a,b in
    ((S[0],T[0]),(S[1],T[1]),(S[2]['baseline_gaussian'],T[2]['baseline_gaussian']),
     (S[2]['baseline_scale'],T[2]['baseline_scale']),(S[3],T[3]))))
s1=Streams017({'kind':'training','replicate':0,'role':'student'}).draw(4,'cpu')
check('matched_student_CRN_prefixes',all(torch.equal(a,b) for a,b in ((S[0],s1[0]),(S[1],s1[1]),(S[3],s1[3]))))
# Strict original013 codec reconstructs a deliberately mismatching reduction.
from run_wp1_013 import lossless_window_codes
from n01.observation008 import observe_generated_carrier
wave=observe_generated_carrier(torch.as_tensor(source[:2]),return_carrier=True)
fields={k:wave[k].numpy() for k in ('window_V','baseline100','peak_index','adc_carrier')}
fields['diagnostic__baseline_mean_V']=wave['baseline_mean_V'].numpy()
code=lossless_window_codes(fields);save_npz017(OUT/'CODEC.npz',**code)
decoded=decode_hard017(OUT/'CODEC.npz')[0]
check('mature_codec_actual_Y_bits',np.array_equal(decoded.view(np.uint64),fields['window_V'].view(np.uint64)))
check('global_CPU_RNG_unchanged_all_explicit',torch.equal(before_global,torch.get_rng_state()))
from wp1_ops import durable_progress
journal=durable_progress(store.root);check('durable_entered_completed_exact_no_failed',not journal['failed_ids'] and not journal['unknown_inflight_ids']
    and all(journal['completed_counts'].get(k,0)==v==journal['entered_counts'].get(k,0) for k,v in store.counts.items()))
sources={n:dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in ('scripts/run_wp4_017.py','src/n01/model017.py','src/n01/stats017.py')}
result=dict(passed=True,checks=checks,sources=sources,artificial_only=True,actual_scientific_CP_NN_AD_G_RNG_SOURCE_REAL_DEV_GPU=0,
    manufactured_engineering_counts=store.counts,operations_entered=store.entered,operations_completed=store.completed,
    engineering_explicit_latent_extra_stream_check_rows=12,engineering_backward_queries=4,engineering_optimizer_steps=4,
    engineering_small_checkpoint_files=5,original_numeric_files_opened=[],stage1_gate=gate,
    output_bytes=sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file()))
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'output_bytes':result['output_bytes'],'counts':store.counts}),flush=True)
