"""Manufactured controller states only. No native IO, numeric import or training."""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
import wp4_017_controller as c

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(d,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def main():
    out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).parent/'run_01'
    assert out.is_relative_to(Path(__file__).parent.resolve()) and not out.exists();out.mkdir(parents=True)
    started=time.monotonic();source_sha=sha(ROOT/c.CONTROLLER)
    p,budget=c.factory_contract((ROOT/c.DRIVER).read_bytes())
    p.update(budget_caps=budget(p),stage1_budget_caps=budget(p,stage1=True))
    b=c.operations();checks=[];negatives=[]
    def accept(name,fn,expected=None):
        value=fn()
        if expected is not None:assert value==expected,(name,value,expected)
        checks.append(dict(name=name,result='pass'));return value
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError) as error:
            negatives.append(dict(name=name,result='rejected',reason=str(error)));return
        raise AssertionError('accepted invalid manufactured state: '+name)
    zero={k:0 for k in p['budget_caps']}
    first_counts=dict(zero,checkpoint_loads=3,optimizers_created=1,A_original_forward_rows=6593,
        original_U_Net_decoded_rows=6593,fixed_M0_head_actual_kernel_rows=6593,
        G_updates=1,optimizer_steps=1,new_AD_queries=1,training_backward_queries=1,
        student_charge_module_rows=1024,teacher_charge_module_rows=1024,
        training_student_rows=1024,training_teacher_rows=1024,
        noise005_rows=2048,shape_cached_basis_assembly_rows=2048,all_three_extensions_rows=6144,
        hard_ADC_rows=6144,hard_first_argmax_gather_rows=6144,
        normal_shape_scalar_requests=16384,normal_charge_scalar_requests=2048,
        normal_noise005_scalar_requests=206848,normal_extension284_scalar_requests=581632,
        total_scalar_normal_requests=806912,parent_sampler_streams=1,
        parent_randint_calls=1,parent_randint_scalar_requests=512,
        paired_training_energy_pair_distances=7680,checkpoint_saves_complete=1)
    init_counts=dict(zero,checkpoint_loads=3,optimizers_created=1,A_original_forward_rows=6593,
        original_U_Net_decoded_rows=6593,fixed_M0_head_actual_kernel_rows=6593,checkpoint_saves_complete=1)
    expected={'manufactured_immutable':'a'*64};config_sha='c'*64;frozen={'A':'d'*64,'005':'e'*64,'M0':'f'*64}
    proof=dict(run_id=c.RUN_ID,pid=102,artificial_only=False,trial=c.TRIALS[0],actual_G_updates=1,
        batch_completed_G_updates=1,trainable_parameters=2,trainable_parameter_names=['theta_a','theta_b'],
        both_gradients_nonzero=True,both_parameter_updates_nonzero=True,frozen_A_005_M0_unchanged=True,
        frozen_state_sha256=frozen,cache_manifest_sha256='b'*64,immutable_files_sha256=expected,
        counters=first_counts,attempted_counters=first_counts.copy(),inflight=[])
    init=dict(run_id=c.RUN_ID,pid=102,artificial_only=False,native_initialized=True,
        all_6593_Q_positive=True,one_pass_A_decoded_M0_cache_complete=True,
        known_T1_initial_RMS_contrast_at_least3LSB=True,mean_adapter_enabled=False,
        first_trial=c.TRIALS[0],fresh_AdamW=True,trainable_parameters=2,
        trainable_parameter_names=['theta_a','theta_b'],frozen_state_sha256=frozen,
        cache_manifest_sha256='b'*64,immutable_files_sha256=expected,counters=init_counts,
        attempted_counters=init_counts.copy(),inflight=[])
    supervisor_command=['python','-B',c.REMOTE+'/'+c.CONTROLLER,'--mode','supervise','--config-sha256',config_sha,'--lock-fd','7']
    worker_command=['python','-B',c.REMOTE+'/'+c.DRIVER,'--config',c.REMOTE+'/'+c.CFG,'--config-sha256',config_sha,'--output',c.REMOTE+'/runs/primary/artifacts','--device','cuda:0']
    live={
        'UTC':'ARTIFICIAL_same_snapshot','current_boot_id':'ARTIFICIAL_boot','protection_errors':[],
        'START_RECEIPT.json':dict(supervisor_pid=101,boot_id='ARTIFICIAL_boot',supervisor_start_ticks=1000,command=supervisor_command),
        'runs/primary/supervisor/launch.json':dict(supervisor_pid=101,child_pid=102,boot_id='ARTIFICIAL_boot',
            supervisor_start_ticks=1000,child_start_ticks=1001,command=worker_command,supervisor_started_monotonic=time.monotonic()),
        'runs/primary/artifacts/FIRST_UPDATE_PROOF.json':proof,'runs/primary/artifacts/INIT_PROOF.json':init,
        'runs/primary/artifacts/FAILED.json':None,'runs/primary/artifacts/COMPLETE.json':None,
        'runs/primary/supervisor/exit_receipt.json':None,
        'runs/primary/artifacts/status.json':dict(run_id=c.RUN_ID,pid=102,counters=first_counts,native_status='RUNNING'),
        '_verified_freeze':dict(immutable_files_sha256=expected,config_sha256=config_sha),
        'durable_child_progress':dict(entered_counts=first_counts.copy(),completed_counts=first_counts.copy(),
            unknown_inflight_ids=[],failed_ids=[],truncated_tail=False,counts_are_durable_lower_bounds=True),
        'live_processes':dict(supervisor=dict(alive=True,pid=101,start_ticks=1000,parent_pid=1,cwd=c.REMOTE,
            command=supervisor_command,CPU_affinity=list(range(16)),VmRSS=1000000),
            worker=dict(alive=True,pid=102,start_ticks=1001,parent_pid=101,cwd=c.REMOTE,command=worker_command,
                CPU_affinity=list(range(12)),VmRSS=2000000)),
        'current_resource_health':dict(collected_by_native_snapshot=True,snapshot_UTC='ARTIFICIAL_same_snapshot',
            resource_limits=p['resource_limits'],owned_process_tree=[101,102],elapsed_seconds=10,
            RSS_bytes=3000000,GPU_bytes=4000000,output_bytes=1000),
    }
    accept('live_first_update_and_current_health',lambda:c.first_valid(live,p)['same_snapshot_health'],
        'current_owned_process_chain_and_resources_verified')
    def bad_live(name,mutate):
        state=copy.deepcopy(live);mutate(state);reject(name,lambda:c.first_valid(state,p))
    bad_live('dead_worker',lambda s:s['live_processes']['worker'].update(alive=False))
    bad_live('dead_supervisor',lambda s:s['live_processes']['supervisor'].update(alive=False))
    bad_live('reused_worker_PID_new_start_ticks',lambda s:s['live_processes']['worker'].update(start_ticks=2000))
    bad_live('stale_boot',lambda s:s.update(current_boot_id='other_boot'))
    bad_live('wrong_parent_PID',lambda s:s['live_processes']['worker'].update(parent_pid=999))
    bad_live('wrong_worker_cwd',lambda s:s['live_processes']['worker'].update(cwd='/other/job'))
    bad_live('wrong_worker_command',lambda s:s['live_processes']['worker'].update(command=['other_job']))
    bad_live('wrong_CPU_affinity',lambda s:s['live_processes']['worker'].update(CPU_affinity=[0,1]))
    bad_live('missing_current_resource_snapshot',lambda s:s.pop('current_resource_health'))
    bad_live('stale_resource_timestamp',lambda s:s['current_resource_health'].update(snapshot_UTC='older'))
    bad_live('RSS_over_cap',lambda s:s['current_resource_health'].update(RSS_bytes=9*1024**3))
    bad_live('GPU_over_cap',lambda s:s['current_resource_health'].update(GPU_bytes=11*1024**3))
    bad_live('output_over_cap',lambda s:s['current_resource_health'].update(output_bytes=20000*1024**2))
    bad_live('wall_over_cap',lambda s:s['current_resource_health'].update(elapsed_seconds=64801))
    bad_live('durable_failed_operation',lambda s:s['durable_child_progress'].update(failed_ids=[9]))
    bad_live('durable_truncated_tail',lambda s:s['durable_child_progress'].update(truncated_tail=True))
    bad_live('durable_first_update_not_complete',lambda s:s['durable_child_progress']['completed_counts'].update(G_updates=0))
    bad_live('same_snapshot_FAILED',lambda s:s.update({'runs/primary/artifacts/FAILED.json':{'artificial_failure':True}}))
    bad_live('native_nonzero',lambda s:s.update({'runs/primary/supervisor/exit_receipt.json':{'exit_code':1,'child_native_exit_code':1}}))
    bad_live('native_zero_without_COMPLETE',lambda s:s.update({'runs/primary/supervisor/exit_receipt.json':{'exit_code':0,'child_native_exit_code':0}}))
    active_inflight=copy.deepcopy(live)
    active_inflight['durable_child_progress']['entered_counts']['new_AD_queries']=2
    active_inflight['durable_child_progress']['unknown_inflight_ids']=[123]
    accept('active_inflight_allowed_as_unknown_completed_G_stays_one',
        lambda:c.first_valid(active_inflight,p)['current_batch_completed_G_updates'],1)
    assert active_inflight['durable_child_progress']['completed_counts']['new_AD_queries']==1

    def terminal(full,nonpass='fail'):
        state=copy.deepcopy(live);counts=copy.deepcopy(p['budget_caps'] if full else p['stage1_budget_caps'])
        gate=dict(stage2_allowed=full,required_trials=list(c.TRIALS[:2]),required_cell_count=72,
            status='pass' if full else 'closed',nonpass=[] if full else [dict(status=nonpass,reason='ARTIFICIAL_semantic_case')])
        done=dict(schema='n01/wp4-017-batch-complete/v1',run_id=c.RUN_ID,pid=102,artificial_only=False,
            native_status='COMPLETE',counters=counts,attempted_counters=counts.copy(),inflight=[],
            stage1_gate=gate,stage2_entered=full,terminal_scope='full_eight_trials' if full else 'stage1_closed_normally',
            trials=[dict(directory=n,G_updates=2000) for n in (c.TRIALS if full else c.TRIALS[:2])],
            G_updates_per_trial=2000,frozen_A_005_M0_unchanged=True,cache_complete=True,no_REAL_DEV=True,
            immutable_files_sha256=expected)
        state.update({'runs/primary/artifacts/COMPLETE.json':done,'runs/primary/artifacts/status.json':copy.deepcopy(done),
            'runs/primary/supervisor/exit_receipt.json':dict(child_native_exit_code=0,exit_code=0,reason=None,error=None,
                supervisor_pid=101,child_pid=102,elapsed_seconds=10,peak_RSS_bytes=3000000,peak_GPU_bytes=4000000,peak_output_bytes=1000),
            'durable_child_progress':dict(entered_counts=counts.copy(),completed_counts=counts.copy(),unknown_inflight_ids=[],
                failed_ids=[],truncated_tail=False,counts_are_durable_lower_bounds=False),
            'live_processes':dict(supervisor=dict(alive=False,pid=101),worker=dict(alive=False,pid=102))})
        return state
    terminals={}
    for name,full,nonpass in [('stage1_fail',False,'fail'),('stage1_pending',False,'pending'),('full8',True,'fail')]:
        state=terminal(full,nonpass);terminals[name]=state
        accept(name+'_terminal_policy_exact_selected_caps',lambda s=state:c.terminal_policy017(s['runs/primary/artifacts/COMPLETE.json'],p),
            p['budget_caps'] if full else p['stage1_budget_caps'])
        value=accept(name+'_native0_complete_with_dead_processes',lambda s=state:c.complete_valid(s,p))
        assert len(value['trials'])==(8 if full else 2)
        accept(name+'_terminal_first_proof_preserved',lambda s=state:c.first_valid(s,p)['same_snapshot_health'],'native_completed_zero')
    def bad_terminal(name,kind,mutate):
        state=copy.deepcopy(terminals[kind]);mutate(state);reject(name,lambda:c.complete_valid(state,p))
    def complete(s):return s['runs/primary/artifacts/COMPLETE.json']
    bad_terminal('fail_gate_cannot_enter_remaining_six','stage1_fail',lambda s:complete(s).update(stage2_entered=True))
    bad_terminal('pending_gate_cannot_enter_remaining_six','stage1_pending',lambda s:complete(s).update(stage2_entered=True))
    bad_terminal('closed_gate_cannot_claim_full_trial_list','stage1_fail',lambda s:complete(s).update(trials=[dict(directory=n,G_updates=2000) for n in c.TRIALS]))
    bad_terminal('pass_gate_missing_one_trial','full8',lambda s:complete(s)['trials'].pop())
    bad_terminal('pass_gate_cannot_claim_only_two','full8',lambda s:complete(s).update(trials=complete(s)['trials'][:2]))
    bad_terminal('closed_gate_needs_nonpass_reason','stage1_pending',lambda s:complete(s)['stage1_gate'].update(nonpass=[]))
    bad_terminal('false_gate_cannot_have_pass_status','stage1_fail',lambda s:complete(s)['stage1_gate'].update(status='pass'))
    bad_terminal('full_gate_cannot_be_pending','full8',lambda s:complete(s)['stage1_gate'].update(status='pending'))
    bad_terminal('trial_below_2000_G','stage1_fail',lambda s:complete(s)['trials'][1].update(G_updates=1999))
    bad_terminal('entered_completed_mismatch','stage1_fail',lambda s:s['durable_child_progress']['entered_counts'].update(G_updates=4001))
    bad_terminal('completed_count_missing_from_journal','stage1_fail',lambda s:s['durable_child_progress']['completed_counts'].update(G_updates=3999))
    bad_terminal('terminal_inflight_unknown_not_complete','stage1_pending',lambda s:s['durable_child_progress'].update(unknown_inflight_ids=[5]))
    bad_terminal('producer_inflight_not_complete','stage1_pending',lambda s:complete(s).update(inflight=[5]))
    bad_terminal('terminal_failed_operation','full8',lambda s:s['durable_child_progress'].update(failed_ids=[5]))
    bad_terminal('terminal_truncated_journal','full8',lambda s:s['durable_child_progress'].update(truncated_tail=True))
    bad_terminal('terminal_native_nonzero','stage1_fail',lambda s:s['runs/primary/supervisor/exit_receipt.json'].update(exit_code=1))
    bad_terminal('terminal_child_native_nonzero','stage1_pending',lambda s:s['runs/primary/supervisor/exit_receipt.json'].update(child_native_exit_code=1))
    bad_terminal('terminal_resource_reason','full8',lambda s:s['runs/primary/supervisor/exit_receipt.json'].update(reason='resource_limit'))
    bad_terminal('terminal_supervisor_PID_mismatch','stage1_fail',lambda s:s['runs/primary/supervisor/exit_receipt.json'].update(supervisor_pid=999))
    bad_terminal('terminal_peak_RSS_over_cap','stage1_pending',lambda s:s['runs/primary/supervisor/exit_receipt.json'].update(peak_RSS_bytes=9*1024**3))
    bad_terminal('terminal_immutable_control_changed','full8',lambda s:complete(s).update(immutable_files_sha256={'manufactured_immutable':'0'*64}))
    bad_terminal('terminal_2000_only_cannot_claim_stage1_cap','stage1_fail',lambda s:complete(s)['counters'].update(G_updates=2000))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert sha(ROOT/c.CONTROLLER)==source_sha,'controller changed during this check'
    write(out/'ARTIFICIAL_LIVE_STATE.json',live)
    for name,state in terminals.items():write(out/('ARTIFICIAL_'+name+'_TERMINAL.json'),state)
    receipt=dict(schema='n01/wp4-017-root-controller-engineering/v1',UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        passed=True,artificial_only=True,source_sha256=source_sha,checks=checks,negative_cases=negatives,
        positive_checks=len(checks),rejected_negative_cases=len(negatives),elapsed_seconds=time.monotonic()-started,
        manufactured_state_fields_include_native_shaped_artificial_false_only_to_exercise_policy=True,
        actual_native_execution=False,CP_loads=0,NN_rows=0,AD_queries=0,training_updates=0,random_draws=0,
        real_SOURCE_REAL_DEV_array_decodes=0,SSH_calls=0,processes_launched_or_stopped=0,GPU_work=0,
        numerical_imports=False,implementation_sources_modified=False,
        reused_engineering_method='016 manufactured live/terminal identity and resource fixture;017 phase policy and counters adapted')
    write(out/'ENGINEERING_RECEIPT.json',receipt)
    print(json.dumps(dict(passed=True,positive_checks=len(checks),rejected_negative_cases=len(negatives),
        source_sha256=source_sha,receipt=str(out/'ENGINEERING_RECEIPT.json'))))

if __name__=='__main__':main()
