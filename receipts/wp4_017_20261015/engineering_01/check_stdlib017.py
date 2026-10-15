"""No NumPy/Torch import, original numeric inputs or native operations."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True)
names=['scripts/run_wp4_017.py','scripts/wp4_017_controller.py','src/n01/model017.py','src/n01/stats017.py','src/n01/saved_array_kernels017.py']
records={}
for name in names:
    data=(ROOT/name).read_bytes();ast.parse(data,filename=name);records[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
d=load('run_wp4_017');c=load('wp4_017_controller');p=d.settings017()
proposal=json.loads((ROOT/'receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json').read_bytes())
def flat(value):
    value=copy.deepcopy(value);normal=value.pop('scalar_normal_requests');value.update({'normal_'+k+'_scalar_requests':v for k,v in normal.items()});return value
assert d.budget_caps017(p)==flat(proposal['exact_planned_budgets']['full_eight_trial_envelope'])
assert d.budget_caps017(p,True)==flat(proposal['exact_planned_budgets']['stage1_gate'])
ref,b=c.factory_contract((ROOT/'scripts/run_wp4_017.py').read_bytes());assert ref==p and b(p)==d.budget_caps017(p)
assert c.operations().identity_adaptations==dict(launch=1,start=1,child_CPU_binding=1)
assert c.operations().STATE_NAMES[-1]=='runs/primary/artifacts/STAGE1_GATE.json'
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
draft=dict(p,budget_caps=d.budget_caps017(p),stage1_budget_caps=d.budget_caps017(p,True))
(OUT/'SETTINGS_DRAFT.json').write_text(json.dumps(draft,indent=2)+'\n',encoding='utf8')
result=dict(passed=True,checks=['five_source_AST','full_budget_equals_accepted_proposal','stage1_budget_equals_accepted_proposal',
    'controller_factory_equals_driver','mature_launch_identity_adaptations','stage_gate_snapshot_included','no_numeric_import'],sources=records)
(OUT/'ENGINEERING_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
