"""016 stdlib contract and same-snapshot health negative cases; no science."""
from pathlib import Path
import copy
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import wp3_016_controller as c
import run_wp3_016 as d
from wp1_ops import sha,write_new


def reject(fn,text):
    try:fn()
    except (ValueError,KeyError) as e:
        assert text in str(e),str(e);return
    raise AssertionError('expected refusal '+text)


def main():
    out=Path(sys.argv[1]).resolve();assert out.drive.upper() in ('E:','D:') and not out.exists();out.mkdir(parents=True)
    started=time.monotonic();checks=[];b=c.operations();p,budget=c.factory_contract((ROOT/c.DRIVER).read_bytes())
    design=json.loads((ROOT/'wp3_016_design.json').read_bytes());assert budget(p)==design['budget_caps_design']
    p.update(remote_root=c.REMOTE,allowed_input_roots=[c.REMOTE],budget_caps=budget(p))
    p['inputs']={k:dict(path=c.OLD015+'/synthetic/'+k,sha256='a'*64) for k in
        ['dataset','statistics','data_statistics','base_config','deterministic','noise005','source_information','integral_statistics','M0_head_seed0','M0_head_seed1']}
    p['contracts']={k:dict(path=c.REMOTE+'/synthetic/'+k,sha256=c.T_SHA if k=='tolerance_table' else 'a'*64) for k in
        ['observation','data_roles','acceptance','tolerance_table']}
    c.validate_config(p,frozen=False);bad=copy.deepcopy(p);bad['budget_caps']['checkpoint_loads']=3
    reject(lambda:c.validate_config(bad,frozen=False),'caps')
    bad=copy.deepcopy(p);bad['evaluation']['bank_seeds'][0]['shape']+=1
    reject(lambda:c.validate_config(bad,frozen=False),'setting')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert b.COLLECTION_LIMIT==256*1024**2 and b.identity_adaptations==dict(launch=1,start=1,child_CPU_binding=1)
    checks.append(dict(name='pure_factory_matches_all_design_caps_fixed_streams_and_pinned_identity_adaptations'))

    old=json.loads((ROOT/'WP3_SOURCE_MANIFEST.json').read_bytes());files={**old['files'],**{n:b.named_record(ROOT/n) for n in c.OWNED}}
    closure=b.project_import_closure(files,lambda n:(ROOT/n).read_bytes())
    assert all(n in closure for n in ['src/n01/model016.py','scripts/run_wp3_016.py','scripts/run_wp3_015.py','scripts/run_wp2_014.py','scripts/wp3_016_controller.py'])
    assert not any('readback_engineering' in n for n in closure)
    checks.append(dict(name='new_import_closure_over105_plus3_uses_embedded_health_no_remote_missing_helper',reachable=closure))

    expected={'synthetic_immutable':'a'*64};config_sha='c'*64;zero={k:0 for k in p['budget_caps']}
    first_counts=dict(zero,checkpoint_loads=3,optimizer_created=1,charge_initialization_fits=1,calibration_A_rows=6593,
        old_A_rows=60361,head_actual_parameter_forward_rows=53768,head_total_kernel_rows=53768,noise005_rows=53768,
        charge_module_rows=53768,charge_latent_scalar_draws=53768,hard_ADC_rows=477768,all_path_gather_rows=477768,
        G_updates=1,optimizer_steps=1,new_AD_queries=1,training_backward_queries=1)
    store=d.Store016(out/'ARTIFICIAL_FIRST',p,expected,artificial=True)
    store.call('INJECTED_COUNTERS_NOT_COMPUTATION',first_counts,lambda:None,artificial_only=True)
    first_path=Path(__file__).parent/'artificial_cpu_01/ARTIFICIAL_BATCH/seed0_Q1/FIRST_UPDATE_PROOF.json'
    first=json.loads(first_path.read_bytes());first.update(store.base());first.update(pid=102,artificial_only=False)
    init=dict(store.base(),pid=102,artificial_only=False,native_initialized=True,charge_initialization_complete=True,calibration_all_Q_positive=True,
        first_trial=c.TRIALS[0],fresh_AdamW=True,mean_adapter_enabled=False,trainable_parameters=2,trainable_parameter_names=['theta_a','theta_b'],
        frozen_state_sha256=first['frozen_state_sha256'],M0_head_sha256=first['M0_head_sha256'])
    init['counters']=dict(zero,checkpoint_loads=3,optimizer_created=1,charge_initialization_fits=1,calibration_A_rows=6593,old_A_rows=6593)
    init['attempted_counters']=init['counters'].copy()
    supervisor_command=['python','-B',c.REMOTE+'/'+c.CONTROLLER,'--mode','supervise','--config-sha256',config_sha,'--lock-fd','7']
    worker_command=['python','-B',c.REMOTE+'/'+c.DRIVER,'--config',c.REMOTE+'/'+c.CFG,'--config-sha256',config_sha,'--output',c.REMOTE+'/runs/primary/artifacts','--device','cuda:0']
    state={
        'UTC':'synthetic_same_snapshot','current_boot_id':'synthetic_boot','protection_errors':[],
        'START_RECEIPT.json':dict(supervisor_pid=101,boot_id='synthetic_boot',supervisor_start_ticks=1000,command=supervisor_command),
        'runs/primary/supervisor/launch.json':dict(supervisor_pid=101,child_pid=102,boot_id='synthetic_boot',supervisor_start_ticks=1000,child_start_ticks=1001,command=worker_command,supervisor_started_monotonic=time.monotonic()),
        'runs/primary/artifacts/FIRST_UPDATE_PROOF.json':first,'runs/primary/artifacts/INIT_PROOF.json':init,
        'runs/primary/artifacts/FAILED.json':None,'runs/primary/supervisor/exit_receipt.json':None,
        'runs/primary/artifacts/status.json':dict(store.base(),pid=102),
        '_verified_freeze':dict(immutable_files_sha256=expected,config_sha256=config_sha),
        'durable_child_progress':b.journal_tools().durable_progress(store.root),
        'live_processes':dict(supervisor=dict(alive=True,pid=101,start_ticks=1000,parent_pid=1,cwd=c.REMOTE,command=supervisor_command,CPU_affinity=list(range(16)),VmRSS=1000000),
            worker=dict(alive=True,pid=102,start_ticks=1001,parent_pid=101,cwd=c.REMOTE,command=worker_command,CPU_affinity=list(range(12)),VmRSS=2000000)),
        'current_resource_health':dict(collected_by_native_snapshot=True,snapshot_UTC='synthetic_same_snapshot',resource_limits=p['resource_limits'],
            owned_process_tree=[101,102],elapsed_seconds=10,RSS_bytes=3000000,GPU_bytes=4000000,output_bytes=1000),
    }
    verdict=c.first_valid(state,p);assert verdict['same_snapshot_health']=='current_owned_process_chain_and_resources_verified'
    negative=[]
    def case(name,mutate,message):
        bad=copy.deepcopy(state);mutate(bad);reject(lambda:c.first_valid(bad,p),message);negative.append(name)
    case('worker_dead_without_exit',lambda s:s['live_processes']['worker'].update(alive=False),'current owned process chain')
    case('supervisor_dead_without_exit',lambda s:s['live_processes']['supervisor'].update(alive=False),'current owned process chain')
    case('PID_reused_new_start_ticks',lambda s:s['live_processes']['worker'].update(start_ticks=2222),'current owned process chain')
    case('stale_boot',lambda s:s.update(current_boot_id='otherboot'),'boot')
    case('wrong_parent_PID',lambda s:s['live_processes']['worker'].update(parent_pid=99),'current owned process chain')
    case('wrong_worker_command',lambda s:s['live_processes']['worker'].update(command=['otherjob']),'command/cwd')
    case('wrong_worker_cwd',lambda s:s['live_processes']['worker'].update(cwd='/some/other/job'),'command/cwd')
    case('wrong_worker_affinity',lambda s:s['live_processes']['worker'].update(CPU_affinity=[0,1]),'command/cwd')
    case('missing_same_snapshot_resources',lambda s:s.pop('current_resource_health'),'resource health')
    case('stale_resource_snapshot',lambda s:s['current_resource_health'].update(snapshot_UTC='older'),'resource health')
    case('RSS_over_cap',lambda s:s['current_resource_health'].update(RSS_bytes=9*1024**3),'resource health')
    case('GPU_over_cap',lambda s:s['current_resource_health'].update(GPU_bytes=11*1024**3),'resource health')
    case('durable_failure_after_firstG1',lambda s:s['durable_child_progress'].update(failed_ids=[9]),'durable progress')
    case('FAILED_same_snapshot',lambda s:s.update({'runs/primary/artifacts/FAILED.json':{'failure':True}}),'same snapshot')
    case('native0_without_COMPLETE',lambda s:s.update({'runs/primary/supervisor/exit_receipt.json':dict(child_native_exit_code=0,exit_code=0,reason=None,error=None)}),'full native0')
    checks.append(dict(name='active_firstG1_requires_current_owned_PID_boot_startticks_command_cwd_CPU_resources_and_durable_health',negative_cases=negative))

    # Snapshot itself records live identity and fresh owned-tree resources; mocks
    # replace all operating-system reads and do not launch a process.
    base_snapshot=b._base_native_snapshot;identity=c.process_identity;helpers=b.supervisor_tools
    try:
        b._base_native_snapshot=lambda pins,partial=False:(p,copy.deepcopy(state))
        c.process_identity=lambda pid:copy.deepcopy(state['live_processes']['supervisor' if pid==101 else 'worker'])
        class Helpers:
            def process_tree(self,pid):assert pid==101;return {101,102},3000000
            def gpu_bytes(self,ids):assert ids=={101,102};return 4000000
            def output_bytes(self,path):assert str(path).replace('\\','/')==c.REMOTE+'/runs/primary';return 1000
        b.supervisor_tools=lambda:Helpers()
        _,sample=c.native_snapshot({'config_sha256':config_sha})
        assert c.first_valid(sample,p)['same_snapshot_health']=='current_owned_process_chain_and_resources_verified'
    finally:b._base_native_snapshot=base_snapshot;c.process_identity=identity;b.supervisor_tools=helpers
    checks.append(dict(name='single_snapshot_collects_current_identity_and_resources_without_poll_or_launch'))

    complete_store=d.Store016(out/'ARTIFICIAL_FULL',p,expected,artificial=True)
    complete_store.call('INJECTED_FULL_COSTS_NOT_COMPUTATION',p['budget_caps'],lambda:None,artificial_only=True)
    complete=dict(complete_store.base(),schema='n01/wp3-016-batch-complete/v1',pid=102,artificial_only=False,native_status='COMPLETE',training_skipped=False,
        trials=[dict(directory=n,G_updates=2000) for n in c.TRIALS],G_updates_per_trial=2000,all_two_trials_completed=True,
        all_frozen_A_005_unchanged=True,all_M0_heads_unchanged=True,charge_initialization_complete=True,no_DEV=True)
    terminal=copy.deepcopy(state);terminal.update({'runs/primary/artifacts/COMPLETE.json':complete,'runs/primary/artifacts/status.json':dict(complete),
        'runs/primary/supervisor/exit_receipt.json':dict(child_native_exit_code=0,exit_code=0,reason=None,error=None,supervisor_pid=101,child_pid=102,
            elapsed_seconds=10,peak_RSS_bytes=3000000,peak_GPU_bytes=4000000,peak_output_bytes=1000),
        'durable_child_progress':b.journal_tools().durable_progress(complete_store.root),
        'live_processes':dict(supervisor=dict(alive=False,pid=101),worker=dict(alive=False,pid=102))})
    assert c.first_valid(terminal,p)['same_snapshot_health']=='native_completed_zero'
    assert c.complete_valid(terminal,p)['status']=='native_and_two_trial_producer_complete'
    bad=copy.deepcopy(terminal);bad['runs/primary/supervisor/exit_receipt.json']['exit_code']=1
    reject(lambda:c.complete_valid(bad,p),'same snapshot')
    bad=copy.deepcopy(terminal);bad['durable_child_progress']['truncated_tail']=True
    reject(lambda:c.complete_valid(bad,p),'durable progress')
    bad=copy.deepcopy(terminal);bad['runs/primary/supervisor/exit_receipt.json']['peak_RSS_bytes']=9*1024**3
    reject(lambda:c.first_valid(bad,p),'terminal resources')
    checks.append(dict(name='complete_native0_branch_requires_all_two_trials_exact_caps_journal_PID_and_resources_dead_processes_allowed'))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result=dict(schema='n01/wp3-016-stdlib-controller-engineering/v1',passed=True,artificial_only=True,checks=checks,
        elapsed_seconds=time.monotonic()-started,scientific_CP_NN_AD_random_training=0,private_arrays_read=0,SSH_calls=0,
        injected_costs_are_not_computation=True,source_sha256=sha(ROOT/c.CONTROLLER),pure_factory_design_budget=p['budget_caps'],
        reused_live_identity_method='015 readback_control.py process_identity plus live/terminal same-snapshot separation; embedded to preserve remote import closure')
    write_new(out/'ENGINEERING_RECEIPT.json',result);print(json.dumps(dict(passed=True,checks=len(checks),negative_cases=len(negative),receipt=str(out/'ENGINEERING_RECEIPT.json'))))


if __name__=='__main__':main()
