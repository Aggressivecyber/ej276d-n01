"""Small stdlib 015 policy/schema bridge; injected costs are not computation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_controller as c
import run_wp3_015 as d
from wp1_ops import write_new,sha


def reject(fn,text):
    try:fn()
    except ValueError as e:assert text in str(e),str(e);return
    raise AssertionError('expected rejection: '+text)


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    started=time.monotonic();checks=[];b=c.operations();p=d.settings015()
    p.update(remote_root=c.REMOTE,allowed_input_roots=[c.REMOTE],budget_caps=d.budget_caps015(p))
    p['remote_inventory_evidence']=dict(path=c.REMOTE+'/WP2_REMOTE_FULL_INVENTORY.json',sha256=c.REMOTE_INVENTORY_SHA,bytes=129532)
    p['evaluation'].update(bank_seeds=[dict(shape=1,noise005=2,extension=3),dict(shape=4,noise005=5,extension=6)],projection_seed=7)
    names={'dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information','integral_statistics','C1_head_seed0','C1_head_seed1'}
    names.update(f'mean_seed{s}_bank{bank}_{h}' for s in range(2) for bank in range(2) for h in d.HYPOTHESES)
    p['inputs']={n:{'path':c.OLD014+'/synthetic/'+n,'sha256':'a'*64} for n in names}
    p['contracts']={n:{'path':c.REMOTE+'/synthetic/'+n,'sha256':c.T_SHA if n=='tolerance_table' else 'a'*64}
        for n in ('observation','data_roles','acceptance','tolerance_table')}
    c.validate_config(p,frozen=False);bad=copy.deepcopy(p);bad['budget_caps']['checkpoint_loads']=2
    reject(lambda:c.validate_config(bad,frozen=False),'caps')
    assert b.REMOTE==c.REMOTE and b.CFG==c.CFG and b.DRIVER==c.DRIVER and b.CONTROLLER==c.CONTROLLER
    assert b.SOURCE_LIMIT==16*1024**2 and b.COLLECTION_LIMIT==512*1024**2 and b.HEAD_LIMIT==64*1024**2
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    checks.append('pure_factory_exact_015_contract_22_slots_resource_and_collection_caps')

    evidence=json.loads((ROOT/'receipts/wp2_014_20261015/collection_01/REMOTE_FULL_INVENTORY.json').read_text())
    name=next(n for n in evidence['whole_remote_inventory'] if n.endswith('_REPORT.npz'))
    row=dict(path=c.OLD014+'/'+name,**evidence['whole_remote_inventory'][name])
    assert c.remote_input_binding015(row,evidence)==name
    reject(lambda:c.remote_input_binding015(dict(row,sha256='0'*64),evidence),'inventory')
    reject(lambda:c.remote_input_binding015(dict(row,path=c.REMOTE+'/'+name),evidence),'original014')
    checks.append('remote_only_REPORT_bound_to_prior318_SHA_metadata_no_raw_transfer')

    old=json.loads((ROOT/'WP2_SOURCE_MANIFEST.json').read_text())
    files={**old['files'],**{n:b.named_record(ROOT/n) for n in c.OWNED}}
    closure=b.project_import_closure(files,lambda n:(ROOT/n).read_bytes())
    assert 'scripts/stability011_metrics.py' in closure and 'src/n01/model015.py' in closure and 'scripts/run_wp2_014.py' in closure
    bad=dict(files);bad.pop('scripts/stability011_metrics.py')
    reject(lambda:b.project_import_closure(bad,lambda n:(ROOT/n).read_bytes()),'dependency')
    checks.append('new_lazy_import_closure_over_original95_and_new3_missing_CDF_refused')

    expected={'synthetic_immutable_file':'a'*64};formal=p['budget_caps'];zero={k:0 for k in formal}
    first_counts=zero.copy()
    first_counts.update({k:v for k,v in formal.items() if k.startswith('mean_evidence_')})
    first_counts.update(checkpoint_loads=3,optimizer_created=1,G_updates=1,optimizer_steps=1,training_backward_queries=1,new_AD_queries=1,
        old_A_rows=53768,head_actual_parameter_forward_rows=53768,head_total_kernel_rows=53768,noise005_rows=53768,
        hard_ADC_rows=319536,all_path_gather_rows=319536)
    fixture=d.Store015(out/'ARTIFICIAL_FIRST',p,expected,artificial=True)
    fixture.call('INJECTED_COUNTERS_NO_COMPUTATION',first_counts,lambda:None,synthetic_only=True)
    # Actual new producer proof from the tiny CPU test, altered in memory only.
    first=json.loads((Path(__file__).parent/'artificial_cpu_02/ARTIFICIAL_TWO_ARM_BATCH/FIRST_UPDATE_PROOF.json').read_text())
    first.update(fixture.base());first.update(artificial_only=False,pid=2,actual_G_updates=1,batch_completed_G_updates=1,trial=c.TRIALS[0],candidate='M0')
    init=dict(fixture.base(),artificial_only=False,pid=2,native_initialized=True,mean_gate_all6_passed=True,
        zero_mean_adapter=True,first_trial=c.TRIALS[0],paired_initialization_bytes=True,fresh_AdamW=True,
        frozen_state_sha256=first['frozen_state_sha256'])
    init['counters']=dict(zero,checkpoint_loads=3,optimizer_created=1)
    init['attempted_counters']=init['counters'].copy()
    state={'START_RECEIPT.json':{'supervisor_pid':1,'boot_id':'synthetic'},
        'runs/primary/supervisor/launch.json':{'supervisor_pid':1,'child_pid':2},
        'runs/primary/artifacts/FIRST_UPDATE_PROOF.json':first,'runs/primary/artifacts/INIT_PROOF.json':init,
        'runs/primary/artifacts/FAILED.json':None,'runs/primary/artifacts/status.json':dict(fixture.base()),
        'runs/primary/supervisor/exit_receipt.json':None,'current_boot_id':'synthetic',
        '_verified_freeze':{'immutable_files_sha256':expected}}
    assert c.first_valid(state,p)['actual_completed_G_updates']==1
    bad=copy.deepcopy(state);bad['runs/primary/artifacts/FAILED.json']={'synthetic_failure':True}
    reject(lambda:c.first_valid(bad,p),'same snapshot')
    bad=copy.deepcopy(state);bad['runs/primary/artifacts/FIRST_UPDATE_PROOF.json']['mean_gate_all6_passed']=False
    reject(lambda:c.first_valid(bad,p),'family')
    checks.append('actual_new_producer_G1_schema_same_snapshot_FAILURE_and_mean_gate_refusal')

    for skip in (False,True):
        store=d.Store015(out/('ARTIFICIAL_SKIP' if skip else 'ARTIFICIAL_FULL'),p,expected,artificial=True)
        costs={k:(v if (not skip or k.startswith('mean_evidence_')) else 0) for k,v in formal.items()}
        store.call('INJECTED_COUNTERS_NOT_ACTUAL_WORK',costs,lambda:None,synthetic_only=True)
        complete=dict(store.base(),schema='n01/wp3-batch-complete/v1',artificial_only=False,pid=2,native_status='COMPLETE',
            training_skipped=skip,trials=[] if skip else [dict(directory=n,G_updates=2000) for n in c.TRIALS],
            all_four_trials_completed=not skip,G_updates_per_trial=2000,all_frozen_A_005_unchanged=True,
            mean_gate_all6_passed=not skip,no_DEV=True)
        if skip:complete.update(reason='registered_population_mean_evidence_not_met',no_checkpoint_NN_optimizer_AD_G=True)
        native=dict(child_native_exit_code=0,exit_code=0,reason=None,error=None,supervisor_pid=1,child_pid=2,elapsed_seconds=10,peak_output_bytes=100)
        current=copy.deepcopy(state);current.update({'runs/primary/artifacts/COMPLETE.json':complete,
            'runs/primary/artifacts/status.json':dict(complete),'runs/primary/supervisor/exit_receipt.json':native,
            'durable_child_progress':b.journal_tools().durable_progress(store.root)})
        verdict=c.complete_valid(current,p);assert verdict['training_skipped']==skip
        if skip:
            current['runs/primary/artifacts/FIRST_UPDATE_PROOF.json']=None
            current['runs/primary/artifacts/INIT_PROOF.json']=None
            start_verdict=c.first_valid(current,p)
            assert start_verdict['mean_gate_skipped_training'] and not start_verdict['actual_first_G1_verified']
        bad=copy.deepcopy(current);bad['runs/primary/artifacts/COMPLETE.json']['counters']['checkpoint_loads']+=1
        bad['runs/primary/artifacts/COMPLETE.json']['attempted_counters']['checkpoint_loads']+=1
        reject(lambda:c.complete_valid(bad,p),'cap' if not skip else 'training/model')
    checks.append('normal_full_and_legal_gate_SKIP_terminal_actual_Store_journal_bridge')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    receipt=dict(schema='n01/wp3-controller-stdlib-engineering/v1',status='PASS',artificial_only=True,checks=checks,
        elapsed_seconds=time.monotonic()-started,all_scientific_counters_are_injected_mock_values=True,
        actual_array_CP_NN_AD_training_SSH=0,reused_original_operations_sha256=c.BASE_SHA,
        source_SHA256={n:sha(ROOT/n) for n in c.OWNED},lazy_import_closure=closure)
    write_new(out/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps({'status':'PASS','checks':len(checks),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
