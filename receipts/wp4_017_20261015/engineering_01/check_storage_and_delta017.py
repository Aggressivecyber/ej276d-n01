"""Only changed017 math/serialization on short manufactured CPU arrays."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
if os.name=='nt':
    import platform
    platform._wmi=None
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
import numpy as np
import torch
from run_wp4_017 import (settings017,budget_caps017,Store017,Runtime017,build_frozen_cache017,evaluate_kernel017,
    Streams017,sha,read_json)
from n01.model017 import deterministic_loading017,energy_score_U017
from n01.model008 import canonical_charge008
from n01.response008 import state_digest
from n01.stats017 import variance_curve_cell017,pooled_wave017,waveform_cells017,coverage_cells017,missingness_cells017
torch.set_num_threads(2)
checks={}
def check(name,value):
    assert bool(value),name
    checks[name]=True

P=96;Q=np.linspace(1,5,P);group=np.ones(P,bool);w=np.linspace(1,2,P)
v=variance_curve_cell017(Q,.01*3.7103998046755806,.0025,group,w)
check('analytic_RMS_and_max_formula',v['status']=='pass' and v['relative_RMS']==v['relative_max']==0)
check('analytic_wrong_variance_fails',variance_curve_cell017(Q,.02*3.7103998046755806,.005,group,w)['status']=='fail')
zero=np.zeros((P,256));one=np.ones_like(zero)
m,var=pooled_wave017(zero,one,one,one)
check('native32_to_derived64_pool_formula',np.all(m==.5) and np.all(var==(62+16)/63))
g=waveform_cells017(one*.02,one*1.05,zero,one,group,w)
check('wave_mean_and_conditional_trace_no_between_scatter',g['mean']['status']==g['variance']['status']=='pass'
    and np.isclose(g['mean']['normalized_mean_RMS'],.02) and np.isclose(g['variance']['pooled_conditional_trace_ratio'],1.05))
cov={'supported':np.ones((P,6),bool),'rates':np.tile(np.asarray([.5,.9]),(P,6,1))}
check('coverage_parent_weighted_fixed_targets',all(x['status']=='pass' for x in coverage_cells017(cov,group,w)))
mom={'full_slot_undefined_fraction':np.zeros((P,9))};flag={'flag__retained':np.zeros(P)}
missing=missingness_cells017(mom,mom,{'flag__retained':np.full(P,.02)},flag,group,w)
check('full_original64_flag_difference_fails',missing[-1]['status']=='fail' and all(x['status']=='pass' for x in missing[:-1]))
S=torch.arange(8*2*256,dtype=torch.float64).reshape(8,2,256)/1e7;T=S.flip(0)+.0001
distances={'cross':0,'unique_self':0};original_cdist=torch.cdist;original_norm=torch.linalg.vector_norm
def cdist(*args,**kwargs):
    result=original_cdist(*args,**kwargs);distances['cross']+=result.numel();return result
def norm(*args,**kwargs):
    result=original_norm(*args,**kwargs);distances['unique_self']+=result.numel();return result
torch.cdist=cdist;torch.linalg.vector_norm=norm
try:score=energy_score_U017(S,T,paired=True)
finally:torch.cdist=original_cdist;torch.linalg.vector_norm=original_norm
from n01.model017 import metric_features017
s,t=metric_features017(S),metric_features017(T)
oracle=((s[:,:,None,:]-t[:,None,:,:]).square().sum(-1).sqrt().mean((1,2))
    -.5*(s[:,0]-s[:,1]).square().sum(-1).sqrt())
oracle_bound=128*np.finfo(np.float64).eps*(torch.abs(score)+torch.abs(oracle))
check('paired_exact_five_actual_distances_and_U_oracle',distances=={'cross':32,'unique_self':8} and bool((torch.abs(score-oracle)<=oracle_bound).all()))
z=torch.zeros(3,2,256,dtype=torch.float64,requires_grad=True);loss=energy_score_U017(z,z.detach().clone(),paired=True).sum();loss.backward()
check('paired_overlap_zero_stable_gradient',torch.isfinite(z.grad).all() and not z.grad.any())

# Reuse ONLY the artificial fixture class definitions, not its executable body.
tree=ast.parse((Path(__file__).parent/'check_driver017.py').read_bytes())
classes=[x for x in tree.body if isinstance(x,ast.ClassDef)];assert [x.name for x in classes]==['FakeKnown','FakeExtension','ManualRuntime']
exec(compile(ast.Module(body=classes,type_ignores=[]),'<manufactured fixture classes only>','exec'),globals())
p=settings017();p['evaluation']=dict(p['evaluation'],parents=6);p['budget_caps']=budget_caps017(p)
store=Store017(OUT/'artifacts',p,{},artificial=True)
t=torch.arange(384,dtype=torch.float64);source=np.stack([((.03+i*.005)*torch.exp(-((t-128)/12)**2)).numpy() for i in range(6)]).astype(np.float32);source[:,:100]=0
weights=np.arange(1,7,dtype=np.float64);ids=np.arange(100,106)
fake=FakeKnown();cache=build_frozen_cache017(fake,source,ids,store.root/'cache',store.call,chunk=2,heldout=4,original_weights=weights)
with np.load(cache['directory']/'PARENT_SPLIT.npz',allow_pickle=False) as a:
    expected=np.asarray([hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest().encode() for r in source],dtype='S64')
    expected_forward=np.asarray([hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest().encode() for r in source.astype(np.float64)],dtype='S64')
    check('source_original_float32_and_actual_float64_input_hashes_distinct_preserved',np.array_equal(a['original_source_row_bytes_sha256'],expected)
        and np.array_equal(a['forward_float64_row_bytes_sha256'],expected_forward) and not np.array_equal(expected,expected_forward))
with np.load(cache['directory']/'ORIGINAL_PHYSICAL_WEIGHTS.npz',allow_pickle=False) as a:
    check('original_weights_preserved_and_in_cache_immutable_manifest',np.array_equal(a['weights'],weights)
        and 'ORIGINAL_PHYSICAL_WEIGHTS.npz' in cache['manifest']['cache_files'])
rt=ManualRuntime(p,cache,FakeExtension(),{'loss':{'q_floor':1e-6,'peak_floor':1e-5}},dict(artificial=True),store,'cpu')
directory=evaluate_kernel017(rt,store.root/'T1',teacher='T1')
meta=read_json(directory/'EVALUATION.json')
check('kernel_actual_coefficients_in_JSON',meta['a_Vns']==.01*3.7103998046755806 and meta['b']==.0025)
with np.load(directory/'PIT_UNIFORMS.npz',allow_pickle=False) as a:
    g=torch.Generator(device='cpu');g.set_state(torch.as_tensor(a['initial_state']));u=torch.rand(6,2,32,6,3,generator=g,dtype=torch.float64).numpy()
    check('PIT_actual_initial_final_state_and_exact_budget',np.array_equal(u,a['uniforms']) and np.array_equal(g.get_state().numpy(),a['final_state'])
        and store.counts['coverage_PIT_uniform_requests']==6*2*32*6*3)
with np.load(directory/'bank1_CHARGE.npz',allow_pickle=False) as a:
    check('charge_parent_draw_bank_ids_actual_mu_v',np.array_equal(a['source_parent_ids'],ids) and np.array_equal(a['draw_index'],np.arange(32))
        and int(a['bank_index'])==1 and np.all(a['actual_mu']==cache['Q'][:,None])
        and np.array_equal(a['actual_v'],np.broadcast_to(.01*3.7103998046755806*cache['Q'][:,None]+.0025*cache['Q'][:,None]**2,(6,32))))
result=dict(passed=True,checks=checks,distances=distances,paired_oracle_max_abs_error=float(torch.abs(score-oracle).max()),
    paired_oracle_engineering_bound='128*float64_eps*(abs(actual)+abs(manual broadcast oracle)); exact distance counts and formula, different vector reduction order',
    artificial_only=True,actual_scientific_CP_NN_AD_G_RNG_SOURCE_REAL_DEV_GPU=0,
    engineering_backward_queries=1,engineering_optimizer_steps=0,manufactured_counters=store.counts,
    sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in ('scripts/run_wp4_017.py','src/n01/model017.py','src/n01/stats017.py')})
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'distances':distances}),flush=True)
