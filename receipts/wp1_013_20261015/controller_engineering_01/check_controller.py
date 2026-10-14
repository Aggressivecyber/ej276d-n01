"""Manufactured stdlib WP1 controller checks. No scientific data or SSH."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from unittest import mock

REPO = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
SOURCE = REPO/'scripts/wp1_controller.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def rejects(function, fragment=None):
    try:
        function()
    except (ValueError, FileExistsError) as error:
        assert fragment is None or fragment in str(error), (fragment, str(error))
        return str(error)
    raise AssertionError('Expected rejection')


def main():
    out = HERE/'artificial_stdlib_01'
    assert not out.exists(), 'Preserve existing engineering output'
    out.mkdir()
    controller = load(SOURCE, 'wp1_controller_artificial')
    core, peer = controller.core(), controller.ops()
    results = []
    def case(name, function):
        value = function(); results.append(dict(name=name, passed=True, detail=value))

    original_config = json.loads((REPO/controller.CFG).read_text('utf8'))
    assert controller.sha(REPO/controller.CFG) == '5f29f43cbac524441d44ddad7aaf978176e614f4184ff8bfdfb5dbcbb903e9d5'
    inventories = {n:json.loads((REPO/n).read_text('utf8'))['files'] for n in
        ('SHAPE010_SOURCE_MANIFEST.json','STABILITY011_SOURCE_MANIFEST.json','MECHANISM012_SOURCE_MANIFEST.json')}
    fake = out/'manufactured_repo'; fake.mkdir()
    names = set().union(*(set(v) for v in inventories.values())) | set(peer.NEW_FILES) | {controller.CONTROLLER}
    for name in names:
        path = fake/name; path.parent.mkdir(parents=True, exist_ok=True)
        if name in ('scripts/mechanism012_ops.py',controller.CONTROLLER,'scripts/wp1_ops.py',original_config['authority']['path']):
            path.write_bytes((REPO/name).read_bytes())
        else:
            path.write_text('Manufactured engineering placeholder: '+name+'\n',encoding='utf8')
    config = copy.deepcopy(original_config)
    for key, row in config['inputs'].items():
        path = fake/'fixture_inputs'/key
        path.parent.mkdir(exist_ok=True)
        path.write_text('Artificial opaque bytes for '+key+'; not a numeric payload.\n',encoding='utf8')
        row.update(local_path=str(path),sha256=controller.sha(path),bytes=path.stat().st_size)
    binding_names = ('dataset','statistics','data_statistics','noise005','deterministic','base_config','training_complete')
    fakebinding = dict(source_manifest={},files={k:{'path':config['inputs'][k]['path'],'sha256':config['inputs'][k]['sha256']} for k in binding_names})
    for name, inventory in inventories.items():
        value = dict(files={n:controller.sha(fake/n) for n in inventory})
        if name == 'SHAPE010_SOURCE_MANIFEST.json':
            value['bound_inputs'] = {k:r['sha256'] for k,r in fakebinding['files'].items()}
        write(fake/name,value)
    fakebinding['source_manifest'] = dict(path=controller.OLD010+'/SHAPE010_SOURCE_MANIFEST.json',sha256=controller.sha(fake/'SHAPE010_SOURCE_MANIFEST.json'))
    write(fake/'SHAPE010_BINDINGS.json',fakebinding)
    for key, name in (('shape010_manifest','SHAPE010_SOURCE_MANIFEST.json'),('shape010_bindings','SHAPE010_BINDINGS.json')):
        path=fake/name; config['inputs'][key].update(local_path=str(path),sha256=controller.sha(path),bytes=path.stat().st_size)
    for row in config['contracts'].values():
        relative = Path(row['local_path']).relative_to(REPO)
        path = fake/relative; row.update(local_path=str(path),sha256=controller.sha(path),bytes=path.stat().st_size)
    write(fake/controller.CFG,config)
    frozen = out/'manufactured_freeze'
    receipt = controller.freeze(fake,frozen,controller.sha(fake/controller.CFG),artificial=True)
    pins,bundle = controller.load_freeze(frozen)
    p,m,si,pi = bundle
    case('separate_packages_noncyclic_freeze',lambda: {
        'source_count':len(m['files']),'private_count':len(pi),'protected_count':len(p['protection_files']),
        'assertions':assert_bundle(controller,receipt,p,m,si,pi)})
    rejects(lambda:controller.freeze(fake,frozen,controller.sha(fake/controller.CFG),artificial=True),'Preserve existing directory')
    rejects(lambda:controller.controller_action('deploy',frozen,out/'forbidden_network'),'artificial package cannot contact native host')
    results.append(dict(name='fresh_only_and_artificial_dispatch_refusal',passed=True))

    local = copy.deepcopy(p)
    for name,row in local['source_files'].items(): row['path']=str(fake/name)
    for row in local['inputs'].values(): row['path']=row['local_path']
    for row in local['contracts'].values(): row['path']=row['local_path']
    for row in local['protection_files'].values(): row['path']=row['local_path']
    local_config = out/'LOCAL_GATE_CONFIG.json'; write(local_config,local)
    returned,paths,checked = peer.gate(local_config,controller.sha(local_config))
    assert returned==local and len(paths)==22
    case('actual_ops_gate_artificial_transitive_bytes',lambda:{'input_count':len(paths),'verified_distinct_paths':len(checked)})

    bad = copy.deepcopy(pins); bad['private_package_sha256']='0'*64
    rejects(lambda:controller.read_bundle(frozen/controller.SOURCE_ARCHIVE,frozen/controller.PRIVATE_ARCHIVE,bad),'package SHA')
    leak = out/'source_with_private.tar'
    with tarfile.open(frozen/controller.SOURCE_ARCHIVE,'r:*') as a, tarfile.open(leak,'x:') as b:
        for member in a: b.addfile(member,a.extractfile(member))
        info=tarfile.TarInfo('private_inputs/leaked.npz');info.size=4;b.addfile(info,io.BytesIO(b'fake'))
    bad=copy.deepcopy(pins);bad['source_package_sha256']=controller.sha(leak)
    rejects(lambda:controller.read_bundle(leak,frozen/controller.PRIVATE_ARCHIVE,bad),'source package inventory')
    results.append(dict(name='package_sha_and_private_leak_refusal',passed=True))

    hashes = controller.runtime_hashes(p,pins['config_sha256'])
    job = out/'producer_job'; job.mkdir()
    store = peer.Store(job/'artifacts',p,hashes,artificial=True)
    firstcharges=dict(old_A_rows=128,head_actual_parameter_forward_rows=128,head_total_kernel_rows=128,
        noise005_rows=128,hard_ADC_rows=768,all_path_gather_rows=768,checkpoint_loads=3)
    store.call('manufactured_first_family_metadata',firstcharges,lambda:None)
    first={**store.base(),'artificial_only':False,'actual_science':True,'first_family_rows':128,'reference_head_G_updates':2000,
        'first_family_completed':True,'A_theta_noise_family_completed':True,'all_three_hypotheses':True,'frozen_state_unchanged':True,
        'inputs_unchanged':True,'extra_pilot_rows':0,'optimizer_created':False}
    store.json('FIRST_WP1_PROOF.json',first)
    state=dict(START_RECEIPT_json_placeholder=None)
    state.update({'START_RECEIPT.json':{'run_id':controller.RUN_ID,'supervisor_pid':987654321,'boot_id':'fixture-boot'},
        'runs/primary/supervisor/launch.json':{'supervisor_pid':987654321,'child_pid':first['pid']},
        'runs/primary/artifacts/FIRST_WP1_PROOF.json':first,'current_boot_id':'fixture-boot',
        '_verified_freeze':{'immutable_files_sha256':hashes}})
    case('actual_store_first128_schema_bridge',lambda:controller.proof_valid(state,p))
    wrong=copy.deepcopy(state);wrong['runs/primary/artifacts/FIRST_WP1_PROOF.json']['reference_head_G_updates']=0
    rejects(lambda:controller.proof_valid(wrong,p),'G2000')
    wrong=copy.deepcopy(state);wrong['runs/primary/supervisor/launch.json']['child_pid']=-1
    rejects(lambda:controller.proof_valid(wrong,p),'PID chain')
    wrong=copy.deepcopy(state);wrong['runs/primary/artifacts/FIRST_WP1_PROOF.json']['counters']['new_AD_queries']=1
    rejects(lambda:controller.proof_valid(wrong,p),'counter cap')
    results.append(dict(name='first_positive_G_PID_zero_AD_guards',passed=True))
    final=dict(old_A_rows=52744,head_actual_parameter_forward_rows=52744,head_total_kernel_rows=52744,noise005_rows=52744,
        hard_ADC_rows=832560,all_path_gather_rows=832560,checkpoint_loads=3)
    store.call('manufactured_completion_metadata',{k:v-store.counts[k] for k,v in final.items()},lambda:None)
    store.phase='complete';store.status({'engineering_checks_passed':True,'inputs_unchanged':True,'frozen_state_unchanged':True})
    artifact_records={q.relative_to(job).as_posix():controller.named_record(q) for q in (job/'artifacts').iterdir() if q.is_file()}
    complete=dict(store.base(),artificial_only=False,engineering_checks_passed=True,inputs_unchanged=True,frozen_state_unchanged=True,
        all_SOURCE_TRAIN_parents=6593,all_REAL_TRAIN_rows=21047,banks=2,reference_head_G_updates=2000,probe_registered_states=30,
        probe_defined_states=0,calibration_cases=21,actual_training_updates=0,formal_response_acceptance=False,
        method_choice='synthetic_undefined_direction_exit',artifacts=artifact_records)
    store.json('COMPLETE.json',complete)
    native={'child_native_exit_code':0,'exit_code':0,'reason':None,'error':None,'supervisor_pid':987654321,'child_pid':first['pid']}
    state.update({'runs/primary/artifacts/COMPLETE.json':complete,'runs/primary/artifacts/status.json':peer.read_json(store.root/'status.json'),
        'runs/primary/supervisor/exit_receipt.json':native,'durable_child_progress':peer.durable_progress(store.root),
        'protection_errors':[]})
    case('actual_store_complete_durable_schema_bridge',lambda:controller.complete_valid(state,p))
    wrong=copy.deepcopy(state);wrong['runs/primary/artifacts/COMPLETE.json']['counters']=first['counters']
    wrong['runs/primary/artifacts/COMPLETE.json']['attempted_counters']=first['attempted_counters']
    rejects(lambda:controller.complete_valid(wrong,p),'completion work/counter formula')
    wrong=copy.deepcopy(state);wrong['runs/primary/supervisor/exit_receipt.json']['child_native_exit_code']=1
    rejects(lambda:controller.complete_valid(wrong,p),'nonzero')
    wrong=copy.deepcopy(state);wrong['durable_child_progress']['unknown_inflight_ids']=[2]
    rejects(lambda:controller.complete_valid(wrong,p),'unresolved durable')
    results.append(dict(name='first_only_nonzero_native_inflight_cannot_complete',passed=True))

    native_root = out/'manufactured_native'; native_root.mkdir()
    shutil.copytree(job,native_root/'runs/primary')
    for name in (controller.CFG,controller.MANIFEST):shutil.copyfile(frozen/name,native_root/name)
    for name in ('START_RECEIPT.json','runs/primary/supervisor/launch.json','runs/primary/supervisor/exit_receipt.json'):
        path=native_root/name;path.parent.mkdir(parents=True,exist_ok=True);write(path,state[name])
    real_unredirected=core.unredirected
    def mapped(path):return native_root if str(path)==controller.REMOTE else real_unredirected(path)
    with mock.patch.object(core,'unredirected',side_effect=mapped):
        inventory,verified=controller.archive_inventory(state,p)
    assert verified['scientific_acceptance'] is False
    case('complete_regular_inventory_preserves_scientific_status',lambda:{'regular_files':len(inventory),'producer_artifacts':len(artifact_records)})
    archive=out/'manufactured_complete.tar'
    with tarfile.open(archive,'x:') as tar:
        for name in inventory:controller.add_regular(tar,native_root/name,name)
    archive_receipt={'archive':controller.REMOTE+'/wp1_complete_01.tar','archive_sha256':controller.sha(archive),'archive_bytes':archive.stat().st_size,
        'inventory':inventory,'partial_evidence_only':False,'job':verified}
    def fake_transfer(command,output,name,timeout=600):
        shutil.copyfile(archive,Path(command[-1]));core.write_new(output/name,{'artificial_only':True,'network_called':False})
    actual_receipt={**receipt,'artificial_only':False}
    with mock.patch.object(controller,'load_freeze',return_value=(actual_receipt,bundle)),mock.patch.object(controller,'ssh_action',return_value=archive_receipt),mock.patch.object(controller,'transfer',side_effect=fake_transfer):
        collected=out/'collected_success'
        collected_receipt=controller.controller_action('collect',frozen,out/'collection_success',collected)
    assert collected_receipt['partial_evidence_only'] is False and collected_receipt['scientific_acceptance'] is False
    assert (collected/'runs/primary/artifacts/COMPLETE.json').is_file()
    case('fresh_extract_SHA_and_complete_consumer_roundtrip',lambda:{'collected_root':str(collected),'file_count':collected_receipt['file_count']})

    failure_store=peer.Store(out/'failure_artifacts',p,hashes,artificial=True)
    try:failure_store.call('manufactured_failure',{'old_A_rows':128},lambda:(_ for _ in ()).throw(RuntimeError('artificial interruption')))
    except RuntimeError:pass
    progress=peer.durable_progress(failure_store.root)
    assert progress['entered_counts']['old_A_rows']==128 and progress['completed_counts']['old_A_rows']==0 and progress['unknown_inflight_ids']==[0]
    partialstate=copy.deepcopy(state);partialstate['current_boot_id']='different-fixture-boot'
    partialstate['runs/primary/supervisor/exit_receipt.json']=None;partialstate['durable_child_progress']=progress
    partialstate['runs/primary/artifacts/COMPLETE.json']=None
    with mock.patch.object(core,'unredirected',side_effect=mapped):
        partialinventory,partialverified=controller.archive_inventory(partialstate,p,partial=True)
    assert partialverified['actual_inflight_completion']=='unknown' and partialverified['scientific_acceptance'] is False
    case('failed_partial_journal_preserves_unknown_without_native0',lambda:progress)
    # A corrupt FAILED/status record is preserved as bytes, never promoted to completion.
    (native_root/'runs/primary/artifacts/status.json').write_text('{truncated',encoding='utf8')
    (native_root/'runs/primary/artifacts/operations.jsonl').write_text('{truncated',encoding='utf8')
    (native_root/'runs/primary/artifacts/COMPLETE.json').unlink()
    (native_root/'runs/primary/supervisor/exit_receipt.json').write_text('{truncated',encoding='utf8')
    with mock.patch.object(core,'unredirected',side_effect=mapped):
        partialinventory,partialverified=controller.archive_inventory(partialstate,p,partial=True)
    partialarchive=out/'manufactured_partial.tar'
    with tarfile.open(partialarchive,'x:') as tar:
        for name in partialinventory:controller.add_regular(tar,native_root/name,name)
    partial_receipt=dict(archive=controller.REMOTE+'/wp1_partial_01.tar',archive_sha256=controller.sha(partialarchive),archive_bytes=partialarchive.stat().st_size,
        inventory=partialinventory,partial_evidence_only=True,job=partialverified)
    def fake_partial_transfer(command,output,name,timeout=600):
        shutil.copyfile(partialarchive,Path(command[-1]));core.write_new(output/name,{'artificial_only':True,'network_called':False})
    with mock.patch.object(controller,'load_freeze',return_value=(actual_receipt,bundle)),mock.patch.object(controller,'ssh_action',return_value=partial_receipt),mock.patch.object(controller,'transfer',side_effect=fake_partial_transfer):
        partialresult=controller.controller_action('collect-failed',frozen,out/'collection_partial',out/'collected_partial')
    assert partialresult['partial_evidence_only'] is True and len(partialresult['control_read_errors'])>=2
    case('partial_corrupt_controls_archive_and_error_receipt',lambda:{'read_errors':partialresult['control_read_errors'],'scientific_acceptance':False})
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    write(out/'ENGINEERING_RECEIPT.json',dict(schema='n01/wp1-controller-engineering/v1',status='PASS',artificial_only=True,
        test_count=len(results),tests=results,controller_sha256=controller.sha(SOURCE),ops_sha256=controller.sha(REPO/'scripts/wp1_ops.py'),
        actual_data_array_decodes=0,actual_checkpoint_loads=0,actual_model_forwards=0,actual_AD_queries=0,actual_training_updates=0,
        actual_repo_freezes=0,SSH_calls=0,scientific_dispatches=0,numpy_imported=False,torch_imported=False,
        manufactured_budget_charges_are_metadata_only=True,native_exit_code=0))
    print(json.dumps(dict(status='PASS',checks=len(results),receipt=str(out/'ENGINEERING_RECEIPT.json'),controller_sha256=controller.sha(SOURCE))))


def assert_bundle(controller,receipt,p,m,si,pi):
    assert len(m['files'])==83 and len(pi)==4 and len(p['protection_files'])==185
    assert set(si)==set(m['files'])|{controller.CFG,controller.MANIFEST}
    assert not any(n.startswith('private_inputs/') for n in si)
    assert all(n.startswith('private_inputs/') for n in pi)
    assert controller.CFG not in p['source_files'] and controller.MANIFEST not in p['source_files']
    assert receipt['artificial_only'] and receipt['all_local_original_files_unchanged']
    return True


if __name__=='__main__':main()
