"""One fixed artificial check of the actual023 cell path, with no repo fallback."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

started=time.perf_counter()
here=Path(__file__).resolve().parent
package=here/'source_package'
sys.path[:0]=[str(package/'scripts'),str(package/'src')]
assert all(Path(p).resolve() not in {Path('E:/GAN/research/N01/scripts'),Path('E:/GAN/research/N01/src')}
    for p in sys.path if p)
import run_wp4_023 as driver
import run_wp4_018
import numpy as np
from n01.stats018 import Z95, DISTANCE_COUNTS018
from run_wp4_017 import compact_report017

assert 'torch' not in sys.modules
assert Z95==1.959963984540054
assert list(sys.modules['n01'].__path__)==[str(package/'src'/'n01')]
names=('run_wp4_023','run_wp4_022','run_wp4_019','run_wp4_018','run_wp4_017','wp1_ops',
    'n01.stats017','n01.stats018','n01.saved_array_kernels017')
modules={name:str(Path(sys.modules[name].__file__).resolve()) for name in names}
assert all(Path(path).is_relative_to(package) for path in modules.values())
pins=json.loads((here/'SOURCE_PACKAGE_PINS.json').read_text('utf8'))
for relative,row in pins['files'].items():
    payload=(package/relative).read_bytes()
    assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
old=ast.parse((package/'scripts/run_wp4_022.py').read_text('utf8'))
new=ast.parse((package/'scripts/run_wp4_023.py').read_text('utf8'))
oldcell=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='contrast_cell022')
newcell=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='contrast_cell023')
assert [ast.dump(n,include_attributes=False) for n in oldcell.body[2:]]==[
    ast.dump(n,include_attributes=False) for n in newcell.body[3:]]
assert [n.module for n in newcell.body if isinstance(n,ast.ImportFrom)]==['n01.stats017','n01.stats018']

DISTANCE_COUNTS018.clear()
n=128
weights=np.ones(n)
group=np.ones(n,bool)
zero=np.zeros(n)
specs=(
    ('positive',np.full(n,.2),zero,zero,group,'supported_at_registered_tolerance'),
    ('negative_U',np.full(n,-.02),zero,zero,group,'not_established_at_registered_tolerance'),
    ('cross_MC',np.tile([0.,1.],n//2),zero,zero,group,'MC_unresolved'),
    ('T1_null',np.full(n,.2),np.full(n,.06),zero,group,'MC_unresolved'),
    ('T2_null',np.full(n,.2),zero,np.full(n,.06),group,'MC_unresolved'),
    ('support',np.full(n,.2),zero,zero,np.arange(n)<32,'MC_unresolved'),
    ('straddling',.1+np.tile([-.01,.01],n//2),zero,zero,group,'tolerance_straddling_unresolved'))
cases={}
for name,cross,null1,null2,selected,expected in specs:
    cases[name]=driver.contrast_cell023(dict(cross_parent=cross,T2_null=null2,full257_excess=np.full(n,-3.)),
        null1,selected,weights)
    assert cases[name]['status']==expected
assert cases['positive']['supported_at_registered_tolerance'] is True
assert abs(cases['negative_U']['cross_parent_cluster']['mean']+.02)<1e-15
assert cases['negative_U']['full257_descriptive']['mean']==-3.
assert cases['cross_MC']['lower95']>.1 and cases['cross_MC']['half_width95']>.05
assert cases['cross_MC']['reasons']==['cross_MC_halfwidth_above005']
assert abs(cases['cross_MC']['half_width95']-Z95/(2*math.sqrt(n-1)))<1e-14
assert cases['T1_null']['reasons']==['T1_teacher32_null_unresolved']
assert cases['T2_null']['reasons']==['T2_teacher32_null_unresolved']
assert 'fewer_than64_common_parents' in cases['support']['reasons']
assert not DISTANCE_COUNTS018

result=here/'DIRECT_CELL023_FIXED_RESULT.json'
compact_report017(result,dict(schema='n01/wp4-023-artificial-cell-cases/v1',run_id=driver.RUN_ID,cases=cases))
saved=json.loads(result.read_text('utf8'))
for name,_,_,_,_,expected in specs:
    assert saved['cases'][name]['status']==expected
with np.load(result.with_suffix('.npz'),allow_pickle=False) as ar:
    assert np.array_equal(ar['root/cases/positive/supported_parent_indices'],np.arange(n))
    assert np.array_equal(ar['root/cases/support/supported_parent_indices'],np.arange(32))
assert 'torch' not in sys.modules and not DISTANCE_COUNTS018
receipt=dict(schema='n01/wp4-023-direct-cell-fixture/v1',run_id=driver.RUN_ID,status='PASS',
    data_scope='fixed artificial parent vectors only; no private input or actual contrast analysis',
    source_package_pins=pins,project_module_paths=modules,no_repository_fallback=True,
    checked_cases={name:case['status'] for name,case in cases.items()},
    old_cell_body_except_Z95_import_AST_identical=True,Z95_provider='n01.stats018',Z95=Z95,
    serialization_JSON_and_parent_index_NPZ_reopened=True,
    NN_rows=0,checkpoint_loads=0,AD_queries=0,G_updates=0,RNG_requests=0,pair_distances=0,
    crossed_bank_codec_005_or_old_suites_repeated=False,
    sys_executable=sys.executable,python_version=sys.version,numpy_version=np.__version__,
    elapsed_seconds=time.perf_counter()-started,
    temporary_roots={key:os.environ[key] for key in ('TEMP','TMP','XDG_CACHE_HOME')},
    fixture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    result_files={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (result,result.with_suffix('.npz'))})
with (here/'DIRECT_CELL_SEMANTICS_RECEIPT.json').open('x',encoding='utf8') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(status='PASS',cases=len(cases),NN_CP_AD_G_RNG_pair_distances=0,
    elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
