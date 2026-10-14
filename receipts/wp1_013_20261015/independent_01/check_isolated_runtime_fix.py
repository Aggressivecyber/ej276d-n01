"""One synthetic import/CDF smoke from only the repaired84 source recipe."""
from __future__ import annotations
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path('E:/GAN/research/N01')
OUT=ROOT/'receipts/wp1_013_20261015/independent_01/native_import_fix01'
PACKAGE=OUT/'isolated_source84'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text('utf8'))
old=read(ROOT/'MECHANISM012_SOURCE_MANIFEST.json')
new={'scripts/wp1_ops.py','scripts/run_wp1_013.py','scripts/wp1_evaluation.py','scripts/wp1_parameter_probe.py',
    'scripts/wp1_input_information.py','scripts/wp1_boundary_saved.py','scripts/wp1_controller.py',
    'USER_MASTER_PLAN_20261015.txt','OBSERVATION_CONTRACT.json','DATA_ROLES.json','docs/ACCEPTANCE_CONTRACT.md',
    'scripts/stability011_metrics.py','SHAPE010_SOURCE_MANIFEST.json','STABILITY011_SOURCE_MANIFEST.json','MECHANISM012_SOURCE_MANIFEST.json'}
names=set(old['files'])|new
assert len(names)==84 and not PACKAGE.exists()
OUT.mkdir(parents=True,exist_ok=True)
PACKAGE.mkdir()
pins={}
for name in sorted(names):
    source=ROOT/name;target=PACKAGE/name
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(source.read_bytes())
    pins[name]=sha(target)
    assert pins[name]==sha(source)
assert pins['scripts/stability011_metrics.py']=='fd87e544673f63887e57c8efafeb0327bbe4db430b04c9697c71cd2e244c3f78'

# Resolve every static project import reachable from the actual WP1 entrypoints,
# including lazy function-body imports. No imported project code is executed here.
provided={Path(name).stem:name for name in names if name.startswith('scripts/') and name.endswith('.py')}
for name in names:
    if name.startswith('src/') and name.endswith('.py'):
        module=name[4:-3].replace('/','.')
        if module.endswith('.__init__'):module=module[:-9]
        provided[module]=name
pending=[n for n in provided if n.startswith('wp1_') or n=='run_wp1_013']
checked=set();edges=[]
known_local={p.stem for p in (ROOT/'scripts').glob('*.py')}|{p.stem for p in (ROOT/'src/n01').glob('*.py')}
while pending:
    module=pending.pop()
    if module in checked:continue
    checked.add(module);file=provided[module]
    tree=ast.parse((PACKAGE/file).read_text('utf8'),filename=file)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):targets=[a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom):
            if node.level:
                parts=module.split('.')[:-1]
                if file.endswith('/__init__.py'):parts=module.split('.')
                parts=parts[:len(parts)-node.level+1]
                target='.'.join(parts+([node.module] if node.module else []))
            else:target=node.module or ''
            targets=[target] if target else []
        else:continue
        for target in targets:
            project=target.startswith('n01') or target.split('.')[0] in known_local
            if project:
                assert target in provided,(file,node.lineno,target,'not in isolated84')
                edges.append(dict(source=file,line=node.lineno,module=target,package_member=provided[target]))
                pending.append(target)

# Clear all repository paths: site packages and stdlib remain, but no old/full
# local scripts/src path can hide a missing repaired-package member.
sys.path=[str(PACKAGE/'scripts'),str(PACKAGE/'src')]+[
    p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
assert all(not Path(p).resolve().is_relative_to(ROOT) or Path(p).resolve().is_relative_to(PACKAGE) for p in sys.path)
evaluation=importlib.import_module('wp1_evaluation')
import numpy as np
x=np.array([0.,2.]);y=np.array([1.,3.,4.])
actual=evaluation.raw_distance(x,y)
assert abs(actual['value']-5/3)<1e-14 and actual['fake_rows']==2 and actual['real_rows']==3
cdf=importlib.import_module('stability011_metrics')
assert Path(evaluation.__file__).resolve()==PACKAGE/'scripts/wp1_evaluation.py'
assert Path(cdf.__file__).resolve()==PACKAGE/'scripts/stability011_metrics.py'
assert 'torch' not in sys.modules
receipt=dict(schema='n01/wp1-isolated-import-repair-artificial/v1',status='PASS',passed=True,
    native_exit_code=0,artificial_only=True,source_files=84,source_sha256=pins,
    reachable_project_modules=sorted(checked),static_project_import_edges=edges,
    tests=[dict(name='reachable_lazy_project_import_closure84',passed=True),
           dict(name='import_from_only_isolated_source_package_no_repository_fallback',passed=True),
           dict(name='unequalN_CDF_two_vs_three_manufactured_values_W1_5over3',passed=True,value=actual['value'])],
    saved_or_original_numeric_arrays_decoded=0,checkpoint_loads=0,trained_model_forwards=0,autograd_queries=0,
    optimizer_steps=0,training_updates=0,DEV_values_read=0,SSH_calls=0,Torch_imported=False,
    temporary_and_data_root=str(OUT))
target=OUT/'ISOLATED_DEPENDENCY_SMOKE.json'
with target.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
assert read(target)['status']=='PASS'
print(json.dumps(dict(status='PASS',checks=3,reachable_project_modules=len(checked),receipt=str(target),sha256=sha(target))))
