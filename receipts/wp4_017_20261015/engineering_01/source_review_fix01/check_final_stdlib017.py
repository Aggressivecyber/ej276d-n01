"""Final source/metadata-only binding. No numeric package or native action."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
import run_wp4_017 as d
import wp4_017_controller as c
checks={}
def check(name,value):
    assert bool(value),name
    checks[name]=True
def reject(name,fn):
    try:fn()
    except ValueError:checks[name]=True
    else:raise AssertionError(name)
old=HERE/'before';preflight_before=HERE/'before_preflight_metadata'
source=(ROOT/'scripts/run_wp4_017.py').read_bytes();prior=(old/'scripts/run_wp4_017.py').read_bytes()
ref,budget=c.factory_contract(source);old_ref,_=c.factory_contract(prior)
preflight=dict(minimum_free_disk_GiB=40,minimum_available_RAM_GiB=10,GPU_must_be_empty=True)
check('fixed_preflight40_RAM10_GPUempty_literal',ref['preflight']==preflight)
without=copy.deepcopy(ref);without.pop('preflight')
check('all_other_settings_exact_previous_contract',without==old_ref)
check('full_and_stage1_budget_exact_previous',budget(ref)==d.budget_caps017(old_ref) and budget(ref,True)==d.budget_caps017(old_ref,True))
p=copy.deepcopy(ref);p.update(status='frozen',budget_caps=budget(ref),stage1_budget_caps=budget(ref,True),
    inputs={k:{} for k in ('dataset','statistics','data_statistics','base_config','deterministic','noise005','M0_head_seed0')},
    contracts={k:{} for k in ('proposal','contract_review','configuration','acceptance','observation','data_roles','tolerance_table')},
    allowed_input_roots=[c.REMOTE],source_files={n:{} for n in c.OWNED})
p['contracts']['proposal']['sha256']=d.PROPOSAL_SHA
p['contracts']['contract_review']['sha256']='31a0a106bf033ec7d363abc13811a7f0e6dd87e79d139ea7ad6c867a67927fd7'
p['contracts']['tolerance_table']['sha256']=c.T_SHA
c.validate_config(p,driver_source=source);d.validate_settings017(p)
check('driver_and_controller_validate_same_resource_contract',True)
for label,mutate in (
    ('missing_preflight',lambda z:z.pop('preflight')),
    ('old_default8GiB',lambda z:z['preflight'].update(minimum_free_disk_GiB=8)),
    ('RAM_below10GiB',lambda z:z['preflight'].update(minimum_available_RAM_GiB=9)),
    ('GPU_not_required_empty',lambda z:z['preflight'].update(GPU_must_be_empty=False))):
    bad=copy.deepcopy(p);mutate(bad)
    reject('controller_rejects_'+label,lambda bad=bad:c.validate_config(bad,driver_source=source))
    reject('driver_rejects_'+label,lambda bad=bad:d.validate_settings017(bad))
def extract(data,cls,name):
    tree=ast.parse(data);parent=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls) if cls else tree
    return copy.deepcopy(next(x for x in parent.body if isinstance(x,ast.FunctionDef) and x.name==name))
def same(cls,name):return ast.dump(extract(source,cls,name))==ast.dump(extract(prior,cls,name))
unchanged={}
for cls,name in (('Engine017','step'),('Runtime017','signals'),('Runtime017','training_window'),
    (None,'budget_caps017'),(None,'prepare_numeric017'),(None,'energy_vectors017')):
    key=(cls+'.' if cls else '')+name;unchanged[key]=same(cls,name)
check('prior_full1024_and_training_math_AST_exact',all(unchanged.values()))
for name,allowed_key in (('build_frozen_cache017','T2_fixed_B0_scope'),('evaluate_kernel017','T2_mechanism_scope'),
    ('checkpoint','initial_stream_states')):
    cls='Engine017' if name=='checkpoint' else None
    current=extract(source,cls,name);previous=extract(prior,cls,name)
    for node in ast.walk(current):
        if isinstance(node,ast.ImportFrom) and node.module=='n01.stats017':node.names=[x for x in node.names if x.name!='T2_fixed_basis_scope017']
        if isinstance(node,ast.Call):node.keywords=[x for x in node.keywords if x.arg!=allowed_key]
    check(name+'_unchanged_except_explicit_reporting_or_state_metadata',ast.dump(current)==ast.dump(previous))
for n in ('src/n01/model017.py','scripts/wp4_017_controller.py','src/n01/saved_array_kernels017.py'):
    check(n+'_exact_reused_bytes',(ROOT/n).read_bytes()==(old/n).read_bytes())
scope=extract((ROOT/'src/n01/stats017.py').read_bytes(),None,'T2_fixed_basis_scope017')
calls=[n for n in ast.walk(scope) if isinstance(n,ast.Call)]
check('T2_scope_no_random_forward_distance_or_other_component_selection',not any(
    isinstance(n.func,ast.Name) and n.func.id in ('seed017','energy_score_U017') or
    isinstance(n.func,ast.Attribute) and n.func.attr in ('rand','randn','cdist','forward','step','backward') for n in calls))
dict_calls=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='dict'];assert len(dict_calls)==1
literals={kw.arg:kw.value.value for kw in dict_calls[0].keywords if isinstance(kw.value,ast.Constant)}
check('T2_boundary_literals_frozen_without_new_comparison',literals['fixed_shape_component']==0 and literals['theoretical_eta_fourth_moment']==1.6878
    and literals['separate_T1_T2_observed_contrast_computed'] is False and literals['extra_contrast_distances']==0
    and literals['hard_T1_T2_distinguishability_established'] is False and literals['conditional_charge_gate_pass_implies_mixture_recovery'] is False
    and literals['component_or_mechanism_replacement_allowed'] is False)
protected=json.loads((HERE.parent/'PROTECTED_BEFORE.json').read_bytes())['files']
check('all124_original_controls_exact',len(protected)==124 and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==r['sha256']
    and (ROOT/n).stat().st_size==r['bytes'] for n,r in protected.items()))
for n in d.OWNED_FILES:ast.parse((ROOT/n).read_bytes(),filename=n)
check('no_numpy_torch_import_or_remote_calls','numpy' not in sys.modules and 'torch' not in sys.modules)
sources={n:dict(bytes=(ROOT/n).stat().st_size,sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in d.OWNED_FILES}
result=dict(passed=True,checks=checks,unchanged_full_math_AST=unchanged,sources=sources,
    numeric_package_imports=False,numerical_experiments_added=0,scientific_operations=0,SSH_SCP_or_server_queries=0,
    scope='Final stdlib fields/AST/hash binding after the existing27 focused delta checks; no new numerical suite')
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({'passed':True,'checks':len(checks),'scientific_operations':0,'source_driver_sha256':sources['scripts/run_wp4_017.py']['sha256']}))
