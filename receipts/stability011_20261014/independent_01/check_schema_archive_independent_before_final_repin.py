"""Actual Store writer/consumer bridge and safe archive synthetic contract QA.

The500 counters and native-exit document are expressly injected schema fixtures,
not computed statistical replicates or scientific/native-run evidence.
"""
import os
os.environ['TEMP'] = os.environ['TMP'] = os.environ['TMPDIR'] = 'E:/GAN/runtime_tmp'
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path('E:/GAN/research/N01')
OWN = Path(__file__).resolve().parent
OUTPUT = OWN/'schema_archive_cpu_01'
OUTPUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'scripts'))
import run_stability011 as runner
import stability011_ops as ops
sources = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
           ('scripts/run_stability011.py','scripts/stability011_ops.py')}
assert sources['scripts/run_stability011.py'] == '37510bf0a607c04bea3e97ddbeb38c64714ecb461a94a6febefc6e2c43bbeee1'
checks = []
read = lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
policy = read(ROOT/'stability011_policy.json')


def run_test(name, test):
    test()
    checks.append(dict(name=name,passed=True))


jobroot = OUTPUT/'artificial_contract_root'
artifacts = jobroot/'runs/primary/artifacts'
input_file = OUTPUT/'fake_bound_input.artificial'
source_file = OUTPUT/'fake_bound_source.artificial'
input_file.write_bytes(b'EXPLICIT SYNTHETIC METADATA INPUT, NOT DATASET')
source_file.write_bytes(b'EXPLICIT SYNTHETIC SOURCE MARKER, NOT FROZEN SCIENCE CODE')
protect = dict(inputs_sha256={str(input_file):sha(input_file)}, sources_sha256={str(source_file):sha(source_file)})
# false is required solely to exercise the native-consumer schema branch. These
# are manually injected counters in a visibly artificial fixture directory.
store = runner.Store(artifacts,policy,artificial=False)
store.protection = protect
store.point_passed = True
store.phase = 'bootstrap'
store.saved = store.attempted = store.finished = 1
context = dict(kind='evaluation',seed=2026101430,update=None,fingerprint='0'*64)
store.json('FIRST_BOOTSTRAP_PROOF.json',dict(store.base(),projection_context=context,inputs_unchanged=True,
          schema_fixture_only=True,first_replicate_index=0))
store.saved = store.attempted = store.finished = 500
store.phase = 'complete'
store.status(dict(classification='coverage_limited'))
inventory = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts.iterdir() if p.is_file()}
store.json('COMPLETE.json',dict(store.base(),artifacts=inventory,registered_bootstrap_replicates=500,
          classification='coverage_limited',inputs_unchanged=True,automatic_restart=False))
supervisor_pid = 913001
state = {'START_RECEIPT.json':dict(supervisor_pid=supervisor_pid),
    'runs/primary/supervisor/launch.json':dict(supervisor_pid=supervisor_pid,child_pid=os.getpid()),
    'runs/primary/supervisor/exit_receipt.json':dict(supervisor_pid=supervisor_pid,child_pid=os.getpid(),
        exit_code=0,child_native_exit_code=0,reason=None,error=None,registered_bootstrap_budget=500),
    'runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json':read(artifacts/'FIRST_BOOTSTRAP_PROOF.json'),
    'runs/primary/artifacts/status.json':read(artifacts/'status.json'),
    'runs/primary/artifacts/COMPLETE.json':read(artifacts/'COMPLETE.json'),
    'runs/primary/artifacts/FAILED.json':None,
    '_verified_freeze':dict(bound_inputs=protect['inputs_sha256'],new_sources=protect['sources_sha256'])}


def actual_Store_status_native_consumer_bridge():
    result = ops.verify_completed(state)
    assert result['bootstrap_replicates_completed'] == 500
    assert result['classification'] == 'coverage_limited'  # completion is not an acceptance gate
    assert state['runs/primary/artifacts/status.json']['inputs_sha256'] == protect['inputs_sha256']
    assert state['runs/primary/artifacts/status.json']['sources_sha256'] == protect['sources_sha256']


def wrong_partial_failure_native_scope_and_hash_refused():
    mutators = [
        lambda x:x['runs/primary/artifacts/COMPLETE.json'].update(bootstrap_replicates_completed=499),
        lambda x:x['runs/primary/artifacts/COMPLETE.json'].update(bootstrap_replicates_attempted=501),
        lambda x:x.update({'runs/primary/artifacts/FAILED.json':{'schema_fixture_only':True}}),
        lambda x:x['runs/primary/supervisor/exit_receipt.json'].update(child_native_exit_code=7),
        lambda x:x['runs/primary/artifacts/status.json'].update(inputs_sha256={}),
        lambda x:x['runs/primary/artifacts/status.json'].update(sources_sha256={}),
        lambda x:x['runs/primary/artifacts/COMPLETE.json'].update(artificial_only=True),
        lambda x:x['runs/primary/artifacts/FIRST_BOOTSTRAP_PROOF.json']['counters'].update(model_forward_rows=1),
        lambda x:x['_verified_freeze'].update(bound_inputs={str(input_file):'a'*64}),
        lambda x:x['runs/primary/artifacts/COMPLETE.json']['artifacts'].update({'unexpected.pt':{'bytes':1,'sha256':'a'*64}})]
    for mutate in mutators:
        fixture = copy.deepcopy(state)
        mutate(fixture)
        try:
            ops.verify_completed(fixture)
        except ValueError:
            pass
        else:
            raise AssertionError('Bad fixture accepted')


def archive_collection_actual_writer_bytes_and_fresh_extract():
    for name in ops.CONTROLS:
        file = jobroot/name; file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(state.get(name,{'explicitly_synthetic_control':True})),encoding='utf-8')
    for name in ('runs/primary/console.log','runs/primary/supervisor/launch.json',
                 'runs/primary/supervisor/resource_status.json','runs/primary/supervisor/exit_receipt.json'):
        file = jobroot/name; file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(state.get(name,{'explicitly_synthetic_control':True})),encoding='utf-8')
    inventory, verification = ops.collection_inventory(jobroot,state)
    assert verification['bootstrap_replicates_completed'] == 500
    archive = OUTPUT/'artificial_collection.tar'
    with tarfile.open(archive,'x:') as tf:
        for name in sorted(inventory):
            tf.add(jobroot/name,arcname=name,recursive=False)
    assert ops.tar_index(archive,maximum_bytes=1024**2) == inventory
    target = OUTPUT/'extracted_artificial_collection'
    ops.exclusive_extract(archive,target,inventory,maximum_bytes=1024**2)
    assert all(sha(target/name) == row['sha256'] for name,row in inventory.items())
    before = {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
    try:
        ops.exclusive_extract(archive,target,inventory,maximum_bytes=1024**2)
    except ValueError:
        pass
    else:
        raise AssertionError('Existing archive output overwritten')
    assert before == {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
    wrong = copy.deepcopy(inventory)
    first = next(iter(wrong)); wrong[first]['sha256'] = 'b'*64
    rejected = OUTPUT/'must_not_be_created'
    try:
        ops.exclusive_extract(archive,rejected,wrong,maximum_bytes=1024**2)
    except ValueError:
        pass
    else:
        raise AssertionError('Altered archive inventory accepted')
    assert not rejected.exists()


def unsafe_tar_members_refused_without_extraction():
    specifications = [(n,tarfile.REGTYPE) for n in ('../escape','/absolute','a//b','a/./b','C:/bad','a\\b')]
    specifications += [('symbolic',tarfile.SYMTYPE),('hardlink',tarfile.LNKTYPE),('directory',tarfile.DIRTYPE)]
    for number,(name,kind) in enumerate(specifications):
        path = OUTPUT/f'unsafe_{number}.tar'
        with tarfile.open(path,'x:') as tf:
            m = tarfile.TarInfo(name);m.type=kind
            if kind == tarfile.REGTYPE:
                m.size=1;tf.addfile(m,io.BytesIO(b'x'))
            else:
                m.linkname='../not_a_real_file';tf.addfile(m)
        try:
            ops.tar_index(path,maximum_bytes=1024)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe member accepted')


for name,test in [('actual_Store_status_COMPLETE_ops_bridge',actual_Store_status_native_consumer_bridge),
                 ('partial_FAILED_nonzero_scope_and_hash_reject',wrong_partial_failure_native_scope_and_hash_refused),
                 ('actual_writer_inventory_tar_fresh_nooverwrite',archive_collection_actual_writer_bytes_and_fresh_extract),
                 ('unsafe_tar_name_link_directory_reject',unsafe_tar_members_refused_without_extraction)]:
    run_test(name,test)
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name,digest in sources.items())
receipt = dict(schema='n01/stability011-independent-schema-archive-artificial/v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed',checks=checks,passed_checks=len(checks),source_sha256=sources,
 artificial_only=True,counter_values_500_manually_injected_for_schema_contract=True,
 native_exit_is_mock_document=True,not_a_scientific_FIRST_proof=True,
 actual_scientific_bootstrap_replicates=0,scientific_array_reads=0,checkpoint_loads=0,
 model_instances=0,model_forwards=0,gradient_evaluations=0,optimizer_updates=0,DEV_reads=0,
 SSH_calls=0,SCP_calls=0,process_launches=0,real_signal_calls=0,
 reused_numerical_CDF_tests_not_rerun=True,output_root=str(OUTPUT))
with (OUTPUT/'ENGINEERING_RECEIPT.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(receipt))
