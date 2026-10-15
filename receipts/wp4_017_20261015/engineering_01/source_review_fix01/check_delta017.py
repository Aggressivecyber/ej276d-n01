"""Four contract-alignment deltas only; no old suite or full1024 rerun."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import os
import sys
if os.name=='nt':
    import platform
    platform._wmi=None
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
import numpy as np
import torch
from n01.stats017 import (charge_identifiability017,variance_curve_cell017,missingness_cells017,
    stage_gate017,REQUIRED_MISSING017,REQUIRED_MISSING_FLAGS017)
from run_wp4_017 import settings017,budget_caps017,Store017,Runtime017,Engine017,seed017,sha,read_json
torch.set_num_threads(2);checks={};rng=torch.get_rng_state().clone()
def check(name,value):
    assert bool(value),name
    checks[name]=True
def reject(name,fn):
    try:fn()
    except (ValueError,RuntimeError,FileExistsError):checks[name]=True
    else:raise AssertionError(name)
old=HERE/'before'
def node(path,cls,method):
    tree=ast.parse(path.read_bytes())
    parent=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls) if cls else tree
    return ast.dump(next(x for x in parent.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==method),include_attributes=False)
unchanged={}
for cls,method in (('Engine017','step'),('Runtime017','signals'),('Runtime017','training_window'),
    (None,'settings017'),(None,'budget_caps017'),(None,'build_frozen_cache017'),(None,'evaluate_kernel017'),(None,'energy_vectors017')):
    key=(cls+'.' if cls else '')+method
    unchanged[key]=node(old/'scripts/run_wp4_017.py',cls,method)==node(ROOT/'scripts/run_wp4_017.py',cls,method)
check('full1024_and_tiny_pipeline_unchanged_AST_boundaries',all(unchanged.values()))
check('model_energy_controller_kernel_exact_bytes_reused',all((old/n).read_bytes()==(ROOT/n).read_bytes()
    for n in ('src/n01/model017.py','scripts/wp4_017_controller.py','src/n01/saved_array_kernels017.py')))

qref=3.7103998046755806;fit=np.asarray([1.,2.,4.,8.]);test=fit*1.1;a=.01*qref;b=.0025
result=charge_identifiability017(fit,test,a,b)
for role,q in (('fit',fit),('test',test)):
    x=q/qref;expected=np.asarray([[np.mean(x*x),np.mean(x**3)],[np.mean(x**3),np.mean(x**4)]])
    g=result['design_by_role'][role]
    trace=np.trace(expected);disc=np.sqrt((expected[0,0]-expected[1,1])**2+4*expected[0,1]**2)
    ev=np.asarray([(trace-disc)/2,(trace+disc)/2])
    check(role+'_fixed_scaled_Gram_independent_formula',np.allclose(g['Gram'],expected,rtol=128*np.finfo(float).eps,atol=0)
        and np.allclose(g['eigenvalues'],ev,rtol=512*np.finfo(float).eps,atol=0)
        and np.isclose(g['eigenvalue_ratio'],ev[0]/ev[1],rtol=512*np.finfo(float).eps,atol=0)
        and np.isclose(g['condition_number'],ev[1]/ev[0],rtol=512*np.finfo(float).eps,atol=0))
check('both_fit_test_identifiable_secondary_defined',result['individually_identifiable'] and result['secondary_25percent_pass'] is True)
for role,qs,qt in (('test',fit,np.full(4,2.)),('fit',np.full(4,2.),test)):
    value=charge_identifiability017(qs,qt,a,b)
    check(role+'_unresolved_blocks_individual_identifiability',not value['individually_identifiable'] and value['secondary_25percent_pass'] is None)
old_X=np.column_stack((test,test**2));old_X=old_X/np.linalg.norm(old_X,axis=0);e=np.linalg.eigvalsh(old_X.T@old_X)
check('new_design_is_not_old_column_L2_design',abs(result['design_by_role']['test']['eigenvalue_ratio']-e[0]/e[1])>1e-4)

Q=np.linspace(1,5,100);Q[-1]=1e200;group=np.ones(100,bool);w=np.ones(100)
invalid=variance_curve_cell017(Q,a,b,group,w)
check('one_invalid_original_parent_cannot_hide_within95percent',invalid['status']=='pending'
    and invalid['reasons']==['reference_undefined'] and invalid['original_parents']==100
    and np.array_equal(invalid['invalid_parent_indices'],[99]) and invalid['invalid_parents_deleted'] is False)
reject('nonpositive_analytic_coefficient_rejected',lambda:variance_curve_cell017(np.linspace(1,5,100),-a,b,group,w))
reject('NaN_analytic_coefficient_rejected',lambda:variance_curve_cell017(np.linspace(1,5,100),np.nan,b,group,w))
valid=variance_curve_cell017(np.linspace(1,5,100),a,b,group,w)
check('valid_all_original_analytic_variance_still_passes',valid['status']=='pass' and valid['relative_RMS']==valid['relative_max']==0)

required_flags={'flag__PSD_undefined','flag__rise_undefined','flag__FWHM_undefined','flag__temporal_nonpositive_peak',
    'flag__rise_prepeak_censored','flag__rise_other_undefined','flag__FWHM_prepeak_censored','flag__FWHM_postpeak_censored','flag__FWHM_other_undefined'}
check('exact_nine_flags_plus_nine_scalar_mandatory_set',set(REQUIRED_MISSING_FLAGS017)==required_flags and len(REQUIRED_MISSING017)==18)
mom={'full_slot_undefined_fraction':np.zeros((100,9))}
tf={k:np.zeros(100) for k in required_flags|{'flag__ADC_rail_hit','flag__Q_nonpositive','flag__peak_at_search_boundary'}}
sf={k:v.copy() for k,v in tf.items()}
for k in ('flag__ADC_rail_hit','flag__Q_nonpositive','flag__peak_at_search_boundary'):sf[k][:]=1
missing=missingness_cells017(mom,mom,sf,tf,group,w)
check('all_nonmissing_flags_diagnostic_without_threshold',all(x['status']=='diagnostic' and x['scientific_threshold'] is None
    and x['absolute_difference']==1 for x in missing if not x['required_for_gate'])
    and all(x['status']=='pass' for x in missing if x['required_for_gate']))
sf['flag__PSD_undefined'][:]=.02
bad=missingness_cells017(mom,mom,sf,tf,group,w)
check('required_undefined_cause_uses_frozen_onepercent_gate',next(x for x in bad if x['feature']=='flag__PSD_undefined')['status']=='fail')
broken={k:v for k,v in sf.items() if k!='flag__FWHM_other_undefined'}
reject('missing_required_cause_schema_rejected',lambda:missingness_cells017(mom,mom,broken,{k:tf[k] for k in broken},group,w))

# Use only prior manufactured JSON cell identities, no old raw NPZ decode.
fixture=ROOT/'receipts/wp4_017_20261015/engineering_01/artificial_driver_02/artifacts'
reports=[read_json(fixture/f'T1_paired_seed{seed}/COMPARISON.json') for seed in (0,1)]
for r in reports:
    r['cells']=[c for c in r['cells'] if c['metric']!='missingness_or_censor' or c['feature'] in REQUIRED_MISSING017]
    for c in r['cells']:c['status']='pass'
    r['other_retained_flag_diagnostics']=[dict(status='diagnostic',feature='flag__ADC_rail_hit',absolute_difference=1.)]
opened=stage_gate017(reports)
check('complete_exact_mandatory_grid_ignores_other_flag_diagnostics',opened['stage2_allowed'] and opened['required_cell_count']==2688)
removed=copy.deepcopy(reports)
ix=next(i for i,c in enumerate(removed[0]['cells']) if c['metric']=='missingness_or_censor' and c['feature']=='flag__PSD_undefined')
removed[0]['cells'].pop(ix);reject('stage_gate_missing_specific_cause_rejected',lambda:stage_gate017(removed))
substituted=copy.deepcopy(reports);substituted[0]['cells'][ix]['feature']='flag__ADC_rail_hit'
reject('stage_gate_nonmissing_flag_cannot_replace_mandatory_cause',lambda:stage_gate017(substituted))
pending=copy.deepcopy(reports);pending[0]['cells'][ix]['status']='pending'
check('single_mandatory_pending_closes_stage2',not stage_gate017(pending)['stage2_allowed'])

cache_dir=fixture/'cache';cache={k:np.load(cache_dir/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('A','B','Q')}
with np.load(cache_dir/'PARENT_SPLIT.npz',allow_pickle=False) as z:cache.update(fit=z['fit_rows'],test=z['test_rows'],ids=z['source_ids'])
cache.update(directory=cache_dir,manifest=read_json(cache_dir/'CACHE_MANIFEST.json'))
class NoNumericExtension:
    def assert_frozen(self):return True
p=settings017();p['budget_caps']=budget_caps017(p);store=Store017(OUT/'artifacts',p,{},artificial=True)
rt=Runtime017(p,cache,NoNumericExtension(),{'loss':{'q_floor':1e-6,'peak_floor':1e-5}},dict(artificial=True),store,'cpu')
engine=Engine017(rt,'T2','unpaired',0,store.root/'T2_unpaired_seed0')
meta=read_json(engine.out/'TRAINING_INITIAL_STATES.json');initial_digest=sha(engine.out/'TRAINING_INITIAL_STATES.npz')
expected_keys={role+'__'+k for role in ('student','teacher') for k in ('shape','charge','noise005','extension284','sign','PIT')}
expected_keys|={'parent__'+role+'__'+field for role in engine.samplers for field in ('generator','seen','fit_source_rows')}
with np.load(engine.out/'TRAINING_INITIAL_STATES.npz',allow_pickle=False) as states:
    check('all_initial_latent_and_four_parent_full_states_saved_before_draw',set(states.files)==expected_keys
        and all(np.array_equal(states[role+'__'+k],g.get_state().numpy()) for role,stream in (('student',engine.student_stream),('teacher',engine.teacher_stream)) for k,g in stream.generators.items())
        and all(not states['parent__'+role+'__seen'].any() for role in engine.samplers))
check('all_initial_namespaces_seeds_roles_and_seen_zero_recorded',all(v['role']==role and v['namespace']==stream.namespace
    and all(seed==seed017(dict(stream.namespace,stream=k)) for k,seed in v['stream_initial_seeds'].items())
    for role,stream in (('student',engine.student_stream),('teacher',engine.teacher_stream)) for v in [meta['latent_roles'][role]])
    and all(v['seen_count']==0 and v['namespace']==engine.samplers[role].namespace and v['initial_seed']==seed017(v['namespace']) for role,v in meta['parent_roles'].items()))
check('saving_initial_state_added_no_rng_forward_AD_or_CP_budget',all(store.counts[k]==0 for k in
    ('normal_shape_scalar_requests','normal_charge_scalar_requests','normal_noise005_scalar_requests','normal_extension284_scalar_requests',
     'student_charge_module_rows','teacher_charge_module_rows','checkpoint_saves_complete','training_backward_queries','new_AD_queries','G_updates')))
engine.student_stream.draw(3,'cpu');engine.teacher_stream.draw(3,'cpu',mixture=True)
for s in engine.samplers.values():s.draw(3)
before_save=store.counts.copy();engine.save_stream_states('end')
with np.load(engine.out/'TRAINING_END_STATES.npz',allow_pickle=False) as states:
    check('complete_end_states_and_seen_match_actual_streams',all(np.array_equal(states[role+'__'+k],g.get_state().numpy())
        for role,stream in (('student',engine.student_stream),('teacher',engine.teacher_stream)) for k,g in stream.generators.items())
        and all(np.array_equal(states['parent__'+role+'__seen'],s.seen) and np.array_equal(states['parent__'+role+'__generator'],s.generator.get_state().numpy()) for role,s in engine.samplers.items()))
check('end_state_metadata_adds_zero_budget_and_initial_bytes_immutable',store.counts==before_save and sha(engine.out/'TRAINING_INITIAL_STATES.npz')==initial_digest)
engine.checkpoint('engineering.pt');checkpoint=torch.load(engine.out/'engineering.pt',map_location='cpu',weights_only=False)
check('checkpoint_binds_initial_receipt_and_current_all_states',checkpoint['initial_stream_states']==engine.initial_stream_receipt
    and all(torch.equal(checkpoint[role+'_streams'][k],g.get_state()) for role,stream in (('student',engine.student_stream),('teacher',engine.teacher_stream)) for k,g in stream.generators.items())
    and all(torch.equal(checkpoint['parent_sampler_states'][role]['generator'],s.generator.get_state()) for role,s in engine.samplers.items()))
check('all_focused_checks_preserve_global_CPU_rng',torch.equal(rng,torch.get_rng_state()))
source_names=('scripts/run_wp4_017.py','scripts/wp4_017_controller.py','src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py')
receipt=dict(passed=True,checks=checks,unchanged_AST_reuse_boundaries=unchanged,
    artificial_only=True,real_original_CP_SOURCE_REAL_DEV_NN_AD_G_RNG_GPU=0,engineering_actual_NN_AD_G_optimizer_steps=0,
    engineering_fresh_optimizer_created=1,engineering_small_CP_save_and_own_reopen_each=1,
    engineering_explicit_latent_rows_per_role=3,engineering_parent_indices=12,manufactured_counters=store.counts,
    original_specification_proposal_seed_law_budget_unchanged=True,
    current_required_cells_two_seed=2688,primary_cells_two_seed=1392,required_missingness_cells_two_seed=1296,
    sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in source_names})
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'primary_plus_missingness_cells':2688,'scientific_operations':0}),flush=True)
