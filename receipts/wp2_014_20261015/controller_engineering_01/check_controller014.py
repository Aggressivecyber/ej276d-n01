"""New 014 controller interfaces only; synthetic counters are not science cost."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def check(condition,message):
    if not condition:raise AssertionError(message)


def expect_error(function,message):
    try:function()
    except (ValueError,AssertionError):return
    raise AssertionError(message)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args();repo=a.repo.resolve();out=a.out.resolve()
    check(out.drive.upper() in ('E:','D:') and not out.exists(),'fresh artificial E/D output required')
    out.mkdir(parents=True);tmp=out/'tmp';tmp.mkdir();os.environ.update(TEMP=str(tmp),TMP=str(tmp),PYTHONDONTWRITEBYTECODE='1')
    spec=importlib.util.spec_from_file_location('controller014_under_test',repo/'scripts/wp2_controller.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    passed=[]
    settings,budget=c.factory_contract((repo/c.DRIVER).read_bytes())
    check(not {'numpy','torch','n01'}&set(sys.modules),'numeric module imported by pure factories')
    p=copy.deepcopy(settings);p.update(remote_root=c.REMOTE,status='frozen',allowed_input_roots=[c.REMOTE,c.OLD013,c.OLD012,
        '/home/rylan/projects/ej276d-n01/window008_resume01_20261013','/home/rylan/projects/ej276d-n01/shape010_20261014','/home/rylan/projects/ej276d-n01/stability011_20261014'])
    p['evaluation'].update(bank_seeds=[dict(shape=1,noise005=2,extension=3),dict(shape=11,noise005=12,extension=13)],projection_seed=2026101430)
    p['budget_caps']=budget(p)
    p['inputs']={n:dict(path=c.OLD012+'/synthetic_'+n,sha256='a'*64) for n in ('dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information')}
    p['contracts']={n:dict(path=c.OLD013+'/synthetic_'+n,sha256=c.T_SHA if n=='tolerance_table' else 'b'*64) for n in ('observation','data_roles','acceptance','tolerance_table')}
    old=json.loads((repo/'WP1_SOURCE_MANIFEST.json').read_text())
    source_names=set(old['files'])|set(c.OWNED)|{'WP1_SOURCE_MANIFEST.json'}
    p['source_files']={n:dict(path=c.REMOTE+'/'+n,**c.named_record(repo/n)) for n in sorted(source_names)}
    c.validate_config(p,driver_source=(repo/c.DRIVER).read_bytes())
    bad=copy.deepcopy(p);bad['budget_caps']['G_updates']-=1
    expect_error(lambda:c.validate_config(bad,driver_source=(repo/c.DRIVER).read_bytes()),'changed G budget accepted')
    bad=copy.deepcopy(p);bad['contracts']['tolerance_table']['sha256']='c'*64
    expect_error(lambda:c.validate_config(bad,driver_source=(repo/c.DRIVER).read_bytes()),'changed T accepted')
    passed.append('pure_actual_factories_exact_budget_and_T_no_numeric_import')

    closure=c.project_import_closure(source_names,lambda n:(repo/n).read_bytes())
    check('scripts/stability011_metrics.py' in closure,'CDF lazy dependency absent')
    expect_error(lambda:c.project_import_closure(source_names-{'scripts/stability011_metrics.py'},lambda n:(repo/n).read_bytes()),'missing lazy dependency accepted')
    passed.append('reachable_lazy_project_imports_include_CDF_missing_dependency_rejected')

    bundle=out/'synthetic_source';bundle.mkdir()
    manifest=dict(schema='n01/wp2-source/v1',run_id=c.RUN_ID,config_sha256='',files={n:{k:r[k] for k in ('bytes','sha256')} for n,r in p['source_files'].items()},project_import_closure=closure)
    c.core().write_new(bundle/c.CFG,p);manifest['config_sha256']=c.sha(bundle/c.CFG)
    c.core().write_new(bundle/c.MANIFEST,manifest)
    with tarfile.open(bundle/c.SOURCE_ARCHIVE,'x:gz') as archive:
        for n in sorted(source_names):c.add_regular(archive,repo/n,n)
        for n in (c.CFG,c.MANIFEST):c.add_regular(archive,bundle/n,n)
    pins=dict(source_package_sha256=c.sha(bundle/c.SOURCE_ARCHIVE),config_sha256=c.sha(bundle/c.CFG),source_manifest_sha256=c.sha(bundle/c.MANIFEST),controller_sha256=c.sha(repo/c.CONTROLLER))
    _,_,inventory=c.read_bundle(bundle/c.SOURCE_ARCHIVE,pins)
    isolated=out/'isolated_runtime';c.core().exclusive_extract(bundle/c.SOURCE_ARCHIVE,isolated,inventory,c.SOURCE_LIMIT)
    code="import sys;sys.path[:]=[x for x in sys.path if not x.startswith(sys.argv[1])];sys.path.insert(0,sys.argv[2]);import run_wp2_014;assert not ({'torch','numpy','n01'}&set(sys.modules));assert run_wp2_014.settings014()['run_id']=='wp2_014_20261015';print('isolated_stdlib_import_ok')"
    result=subprocess.run([sys.executable,'-S','-B','-c',code,str(repo),str(isolated/'scripts')],cwd=isolated,env={**os.environ,'PYTHONPATH':'','PYTHONNOUSERSITE':'1'},capture_output=True,text=True)
    check(result.returncode==0 and 'isolated_stdlib_import_ok' in result.stdout,'isolated packaged entry failed: '+result.stderr)
    c.core().write_new(out/'ISOLATED_ENTRY_RECEIPT.json',dict(artificial_only=True,native_subprocess_exit=result.returncode,stdout=result.stdout,stderr=result.stderr,source_count=len(source_names),closure=closure,
        private_arrays=0,checkpoint_loads=0,model_forwards=0,AD_queries=0,optimizer_steps=0,SSH_calls=0))
    c.core().write_new(bundle/'FREEZE_RECEIPT.json',dict(schema='n01/wp2-freeze/v1',run_id=c.RUN_ID,**pins,artificial_only=True))
    calls=[];original_ssh=c.ssh_action;c.ssh_action=lambda *args,**kwargs:calls.append(args)
    try:expect_error(lambda:c.controller_action('start',bundle,out/'forbidden_dispatch'),'artificial archive allowed dispatch')
    finally:c.ssh_action=original_ssh
    check(not calls,'artificial fixture contacted SSH')
    passed.append('actual_source_bundle_isolated_stdlib_entry_and_artificial_dispatch_rejection')

    # Actual Store014 class and writer; all operations below execute lambda only.
    sys.path.insert(0,str(repo/'scripts'))
    ds=importlib.util.spec_from_file_location('driver014_artificial_schema',repo/c.DRIVER);driver=importlib.util.module_from_spec(ds);ds.loader.exec_module(driver)
    check(not {'numpy','torch','n01'}&set(sys.modules),'actual Store014 import loaded numeric module')
    synthetic_root=out/'synthetic_native';synthetic_root.mkdir();job=synthetic_root/'runs/primary';job.mkdir(parents=True)
    immutable={'/synthetic/read-only/source':'d'*64}
    store=driver.Store014(job/'artifacts',p,immutable,artificial=True)
    store.call('SYNTHETIC_CP_AND_OPTIMIZER_ANNOTATION',dict(checkpoint_loads=2,optimizer_created=1),lambda:None)
    init=dict(store.base(),schema='n01/wp2-init-proof/v1',native_initialized=True,checkpoint_loads=2,G_updates=0,optimizer_created=1,first_trial='seed0_C0',
        paired_initialization_bytes=True,head_parameters=1288,trainable_parameter_names=['basis_head.weight','basis_head.bias'],no_C2=True,frozen_state_sha256={'A':'e'*64,'005':'f'*64})
    store.json('INIT_PROOF.json',init)
    store.call('SYNTHETIC_G1_ANNOTATION',dict(G_updates=1,optimizer_steps=1,training_backward_queries=1,new_AD_queries=1,
        old_A_rows=1024,head_actual_parameter_forward_rows=1024,noise005_rows=1024,hard_ADC_rows=3072,all_path_gather_rows=3072),lambda:None)
    first=dict(store.base(),schema='n01/wp2-first-real-update/v1',actual_G_updates=1,batch_completed_G_updates=1,trial='seed0_C0',candidate='C0',gradient_nonzero=True,
        actual_parameter_update_nonzero=True,frozen_A_005_unchanged=True,head_parameters=1288,trainable_parameter_names=['basis_head.weight','basis_head.bias'],frozen_state_sha256=init['frozen_state_sha256'])
    store.json('FIRST_UPDATE_PROOF.json',first)
    state={'START_RECEIPT.json':dict(supervisor_pid=900000,boot_id='synthetic-boot'),
        'runs/primary/supervisor/launch.json':dict(supervisor_pid=900000,child_pid=os.getpid()),
        'runs/primary/supervisor/exit_receipt.json':None,'runs/primary/artifacts/INIT_PROOF.json':copy.deepcopy(init),
        'runs/primary/artifacts/FIRST_UPDATE_PROOF.json':copy.deepcopy(first),'runs/primary/artifacts/FAILED.json':None,
        'runs/primary/artifacts/status.json':store.base(),'current_boot_id':'synthetic-boot','_verified_freeze':dict(immutable_files_sha256=immutable)}
    expect_error(lambda:c.first_valid(state,p),'artificial proof accepted as science')
    # Toggle only in-memory contract copies to exercise positive native schemas.
    for name in ('runs/primary/artifacts/INIT_PROOF.json','runs/primary/artifacts/FIRST_UPDATE_PROOF.json'):state[name]['artificial_only']=False
    c.first_valid(state,p)
    for failed in ('FAILED_file','status_failed','native_nonzero'):
        bad=copy.deepcopy(state)
        if failed=='FAILED_file':bad['runs/primary/artifacts/FAILED.json']={'error':'manufactured'}
        if failed=='status_failed':bad['runs/primary/artifacts/status.json']['native_status']='FAILED'
        if failed=='native_nonzero':bad['runs/primary/supervisor/exit_receipt.json']={'child_native_exit_code':1,'exit_code':1}
        expect_error(lambda bad=bad:c.first_valid(bad,p),'historical G1 accepted despite same snapshot failure')
    passed.append('actual_Store014_schema_init_G1_bridge_and_same_snapshot_failure_refusal')

    remaining={k:v-store.attempted[k] for k,v in p['budget_caps'].items() if v-store.attempted[k]}
    store.call('SYNTHETIC_REMAINING_COUNTER_ANNOTATION_NOT_EXECUTION',remaining,lambda:None)
    trial_records=[]
    for index,name in enumerate(c.TRIALS):
        d=store.root/name;d.mkdir()
        row=dict(schema='n01/wp2-trial-complete/v1',candidate=p['candidates'][index%2],seed=p['seed_sets'][index//2],G_updates=2000,
            frozen_A_005_unchanged=True,coverage=dict(source_unique=6593,real_unique=21047),resources=dict(output_bytes=100,elapsed_seconds=1,process_RSS_bytes=1024),artificial_only=True)
        driver.write_new(d/'COMPLETE.json',row)
        trial_records.append(dict(directory=name,candidate=row['candidate'],initial_seed=row['seed']['initialization'],G_updates=2000,COMPLETE_sha256=c.sha(d/'COMPLETE.json')))
        (d/'final.pt').write_bytes(b'manufactured-thin-head-checkpoint-no-tensors')
        (d/'RAW_PARENT_ROWS.npz').write_bytes(b'manufactured-raw-not-numpy')
    store.phase='complete'
    complete=dict(store.base(),schema='n01/wp2-batch-complete/v1',native_status='COMPLETE',all_four_trials_completed=True,all_frozen_A_005_unchanged=True,
        no_C2=True,no_DEV=True,G_updates_per_trial=2000,trials=trial_records,scientific_acceptance=False)
    inventory={str(q.relative_to(store.root)).replace('\\','/'):c.named_record(q) for q in store.root.rglob('*') if q.is_file() and q.name!='status.json'}
    store.json('ARTIFACT_INVENTORY.json',dict(schema='n01/wp2-artifact-inventory/v1',files=inventory))
    store.json('COMPLETE.json',complete);store.status(dict(native_status='COMPLETE'))
    state.update({'runs/primary/artifacts/COMPLETE.json':copy.deepcopy(complete),'runs/primary/artifacts/status.json':driver.read_json(store.root/'status.json'),
        'durable_child_progress':c.journal_tools().durable_progress(store.root),
        'runs/primary/supervisor/exit_receipt.json':dict(child_native_exit_code=0,exit_code=0,reason=None,error=None,supervisor_pid=900000,child_pid=os.getpid(),elapsed_seconds=10,peak_output_bytes=1000)})
    state['runs/primary/artifacts/COMPLETE.json']['artificial_only']=False
    c.complete_valid(state,p)
    bad=copy.deepcopy(state);bad['runs/primary/artifacts/COMPLETE.json']['trials'][0]['G_updates']=1999
    expect_error(lambda:c.complete_valid(bad,p),'partial trial accepted')
    bad=copy.deepcopy(state);bad['runs/primary/artifacts/COMPLETE.json']['counters']['old_A_rows']-=128
    expect_error(lambda:c.complete_valid(bad,p),'missing mandatory forwards accepted')
    passed.append('actual_Store014_terminal_schema_four_trial_budget_and_partial_refusal')

    original_remote=c.REMOTE
    try:
        c.REMOTE=str(synthetic_root)
        all_files,selected,omitted,verified=c.archive_inventory(state,p,include_heads=False)
        check(all(n.endswith(('.json','.jsonl','.log','.txt','.md')) for n in selected),'raw leaked into default selective collection')
        check(len([n for n in omitted if n.endswith('.npz')])==4 and len([n for n in omitted if n.endswith('.pt')])==4,'omitted inventory lost raw/head identity')
        with_heads,still_omitted=c.select_inventory(all_files,include_heads=True)
        check(len(with_heads)==len(selected)+4 and all(not n.endswith('.pt') for n in still_omitted),'optional head selection failed')
        archive=out/'synthetic_selected.tar'
        with tarfile.open(archive,'x:') as stream:
            for n in sorted(selected):c.add_regular(stream,synthetic_root/n,n)
        check(c.index(archive,c.COLLECTION_LIMIT)==selected,'selected SHA inventory mismatch')
        target=out/'selected_local';c.core().exclusive_extract(archive,target,selected,c.COLLECTION_LIMIT)
        expect_error(lambda:c.core().exclusive_extract(archive,target,selected,c.COLLECTION_LIMIT),'collection overwrite accepted')
        failed=copy.deepcopy(state);failed['runs/primary/artifacts/FAILED.json']={'error':'manufactured failure','artificial_only':True};failed['START_RECEIPT.json']['boot_id']='different-boot'
        _,partial,_,ver=c.archive_inventory(failed,p,partial=True)
        check(ver['status']=='partial_evidence_only' and ver['actual_inflight_completion']=='unknown' and partial,'failed evidence was not preserved as partial')
    finally:c.REMOTE=original_remote
    passed.append('remote_full_inventory_selected_roundtrip_optional_heads_raw_retained_failed_partial')

    check(not {'numpy','torch','n01'}&set(sys.modules),'numeric module imported by controller fixtures')
    receipt=dict(schema='n01/wp2-controller-engineering/v1',artificial_only=True,passed_checks=passed,passed_count=len(passed),native_test_process_exit=0,
        controller_sha256=c.sha(repo/c.CONTROLLER),driver_source_sha256=c.sha(repo/c.DRIVER),prior_manifest_sha256=c.sha(repo/'WP1_SOURCE_MANIFEST.json'),
        synthetic_counter_annotations_are_not_scientific_execution=True,positive_native_proof_markers_in_memory_only=True,
        actual_science_counters=dict(array_decodes=0,checkpoint_loads=0,model_forwards=0,AD_queries=0,optimizer_created=0,optimizer_steps=0,G_updates=0,DEV_rows=0,SSH_calls=0),
        numerical_packages_imported=False,full_old42_ops_suite_repeated=False,isolated_dependency_closure=closure)
    c.core().write_new(out/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps(dict(status='synthetic_controller_interfaces_passed',checks=len(passed),controller_sha256=receipt['controller_sha256'],receipt_sha256=c.sha(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
