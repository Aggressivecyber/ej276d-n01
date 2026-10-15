"""Bind the minimal loader-only fix; preserve all old failure/freeze evidence."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import difflib
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as c
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

old=HERE/'before/wp3_016_controller.py';new=ROOT/c.CONTROLLER
assert sha(old)=='f1d5eceadb145d6a921c41055dd7a7e4d9fb05c36a150e99689e50e21358d02a'
old_ready=HERE/'before/ENGINEERING_READY.json';assert sha(old_ready)=='d66bf5c4bb3215622bc2b6aee09686468452cf409f31e096540421de95c1f3de'
assert sha(ROOT/'receipts/wp3_016_20261015/engineering_01/ENGINEERING_READY.json')==sha(old_ready)
a,b=[ast.parse(path.read_bytes()) for path in [old,new]]
fa={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)};fb={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
dump=lambda x:ast.dump(x,include_attributes=False)
assert set(fa)==set(fb) and all(dump(fa[k])==dump(fb[k]) for k in fa if k!='operations')
assert [dump(n) for n in fa['operations'].body[4:]]==[dump(n) for n in fb['operations'].body[5:]]
delta=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),
    fromfile='before/wp3_016_controller.py',tofile='scripts/wp3_016_controller.py'))
with (HERE/'CONTROLLER_FIX.diff').open('x',encoding='utf-8') as f:f.write(delta)
engineering=read(HERE/'bootstrap_stdlib_01/ENGINEERING_RECEIPT.json')
assert engineering['passed'] and engineering['source_sha256']==sha(new) and len(engineering['checks'])==5
manifest=read(ROOT/'WP3_016_SOURCE_MANIFEST.json');assert len(manifest['files'])==115
assert all(sha(ROOT/n)==v['sha256'] for n,v in manifest['files'].items() if n!=c.CONTROLLER)
prior=read(ROOT/'WP3_SOURCE_MANIFEST.json');assert len(prior['files'])==105
assert all(sha(ROOT/n)==v['sha256'] for n,v in prior['files'].items())
assert sha(ROOT/'wp3_015_config.json')==prior['config_sha256']
freeze=ROOT/'receipts/wp3_016_20261015/runtime_package_01'
assert (ROOT/c.CFG).read_bytes()==(freeze/c.CFG).read_bytes()
assert (ROOT/c.MANIFEST).read_bytes()==(freeze/c.MANIFEST).read_bytes()
settings,budget=c.factory_contract((ROOT/c.DRIVER).read_bytes())
old_draft=read(ROOT/'receipts/wp3_016_20261015/implementation_01/SETTINGS_DRAFT.json')
assert {k:v for k,v in old_draft.items() if k!='budget_caps'}==settings and budget(settings)==old_draft['budget_caps']
assert all(sha(ROOT/n)==read(old_ready)['sources'][n]['sha256'] for n in ['scripts/run_wp3_016.py','src/n01/model016.py'])
result=dict(schema='n01/wp3-016-controller-bootstrap-fix-ready/v1',UTC=datetime.now(timezone.utc).isoformat(),passed=True,
    status='MINIMAL_BOOTSTRAP_LOADER_FIX_READY_PENDING_INDEPENDENT_REVIEW_AND_REFREEZE',
    previous_controller=ref(old),controller=ref(new),previous_ENGINEERING_READY=ref(old_ready),
    test=ref(HERE/'bootstrap_stdlib_01/ENGINEERING_RECEIPT.json'),test_script=ref(HERE/'check_bootstrap016.py'),diff=ref(HERE/'CONTROLLER_FIX.diff'),
    repair='Restore the mature015 candidate order: SELF_ROOT/scripts/wp2_controller.py, then the original014 absolute controller. First existing candidate must still equal BASE_SHA.',
    scope_checks=dict(only_operations_loader_candidate_selection_changed=True,all_other_controller_functions_AST_identical=True,
        remainder_of_operations_AST_identical=True,driver_model_scientific_settings_identical=True,
        original105_sources_and107_controls_unchanged=True,other114_frozen016_sources_unchanged=True,
        original016_config_manifest_package_preserved=True,old_engineering_READY_preserved=True,
        original_deploy_and_native_failure_evidence_preserved=True),
    sources={n:ref(ROOT/n) for n in c.OWNED},budget_caps=budget(settings),
    reused_numeric_engineering=ref(ROOT/'receipts/wp3_016_20261015/engineering_01/artificial_cpu_01/ENGINEERING_RECEIPT.json'),
    reused_health_engineering=ref(ROOT/'receipts/wp3_016_20261015/engineering_01/controller_stdlib_01/ENGINEERING_RECEIPT.json'),
    new_tests=dict(stdlib_bootstrap_cases=5,numeric_test_repeats=0,private_arrays_or_CP_reads=0,NN_AD_training_random_scientific=0,
        native_process_launches=0,SSH_calls=0),
    next_action='Root independent delta review, runtime_package_fix01 exact refreeze, source commit and Dsource_02 backup before any separately authorized native attempt.',
    deploy_restart_or_retry_performed=False,weights_promoted=False,scientific_acceptance=False)
write(HERE/'CONTROLLER_FIX_READY.json',result)
print(json.dumps(dict(passed=True,controller_sha256=sha(new),ready=str(HERE/'CONTROLLER_FIX_READY.json'),ready_sha256=sha(HERE/'CONTROLLER_FIX_READY.json'))))
