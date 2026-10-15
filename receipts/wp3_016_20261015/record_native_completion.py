"""Account the sole saved native terminal snapshot without remote or tensor calls."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HERE=OUT/'postrun_root_01'
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)


def write(p,value,*,new=False):
    with p.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    state_path=OUT/'wake_status_01/REMOTE_STATE.json'
    state=read(state_path);p=read(ROOT/'wp3_016_config.json')
    source=read(OUT/'SOURCE_COMMIT_READBACK_FIX02.json')
    complete=state['runs/primary/artifacts/COMPLETE.json']
    native=state['runs/primary/supervisor/exit_receipt.json']
    progress=state['durable_child_progress']
    assert source['configuration_sha256']==sha(ROOT/'wp3_016_config.json')
    assert complete['native_status']=='COMPLETE' and not complete['artificial_only']
    assert complete['all_two_trials_completed'] and complete['G_updates_per_trial']==2000
    assert complete['all_frozen_A_005_unchanged'] and complete['all_M0_heads_unchanged'] and complete['no_DEV']
    assert complete['charge_initialization_complete'] and complete['scientific_acceptance'] is False
    assert native['exit_code']==native['child_native_exit_code']==0
    assert state['runs/primary/artifacts/FAILED.json'] is None and not state['protection_errors']
    assert all(v['alive'] is False for v in state['live_processes'].values())
    counts=complete['counters']
    assert counts==complete['attempted_counters']==p['budget_caps']
    assert complete['operations_entered']==complete['operations_completed']==94249
    assert complete['inflight']=={} and not progress['unknown_inflight_ids'] and not progress['failed_ids'] and not progress['truncated_tail']
    for name,value in counts.items():
        assert progress['entered_counts'].get(name,0)==progress['completed_counts'].get(name,0)==value,name
        assert native['entered_counts'].get(name,0)==native['completed_counts'].get(name,0)==value,name
    prior=p['runtime_repair']['prior_actual_cost']
    aggregate={key:value+prior[key] for key,value in counts.items()}
    assert aggregate==p['runtime_repair']['aggregate_budget_caps']
    paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
    master,program,jobs,current=[read(v) for v in [paths[0],paths[1],paths[2],CURRENT]]
    assert program['new_training_updates']==program['cumulative_training_updates']==175804
    assert program['wp3_016_G_updates']==137 and program['short_candidate_runs_completed']==8
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015')
    assert job['status']=='native_running_last_verified' and job['G_updates']==137
    HERE.mkdir(exist_ok=False)
    before=HERE/'metadata_before_native_completion';before.mkdir()
    for path in paths:shutil.copy2(path,before/path.name)
    stamp=dt.datetime.now(dt.timezone.utc).isoformat();added=counts['G_updates']-137;total=175804+added
    assert added==3863 and total==179667
    receipt=dict(schema='n01/wp3-016-native-completion-readback/v1',UTC=stamp,
        status='native0_COMPLETE_two_trials_pending_collection_and_independent_numeric_QA',
        source_commit=source['source_commit'],configuration_sha256=source['configuration_sha256'],
        only_wake_snapshot=ref(state_path),snapshot_UTC=state['UTC'],producer_complete_UTC=complete['UTC'],
        native_exit_UTC=native['UTC'],native_child_exit=0,native_supervisor_exit=0,
        per_trial_G_updates=2000,completed_trials=2,operations_entered=94249,operations_completed=94249,
        entered_counts=complete['attempted_counters'],actual_completed_counts=counts,
        prior_failed_native_actual_cost=prior,aggregate_actual_counts_all_native_attempts=aggregate,
        unknown_inflight_ids=[],failed_operation_ids=[],truncated_tail=False,
        cumulative_G_pre016=175667,previously_accounted016_G=137,newly_added_G=added,
        cumulative_G=total,short_search_trials_completed=10,WP3_short_trials_completed=6,
        resources={k:native[k] for k in ['elapsed_seconds','peak_RSS_bytes','peak_GPU_bytes','peak_output_bytes']},
        active_scientific_jobs=0,collection_pending=True,independent_numeric_acceptance_pending=True,
        scientific_acceptance=False,weights_promoted=False,scientific_status='response_match_not_established')
    terminal_path=HERE/'NATIVE_COMPLETION_READBACK.json';write(terminal_path,receipt,new=True)
    refs=copy.deepcopy(program['wp3_016_execution'])
    refs.update(status='completed_native0_pending_collection_and_independent_readback',
        terminal_readback=ref(terminal_path),native_exit_code=0,supervisor_exit_code=0,
        ended_UTC=native['UTC'],last_verified_UTC=state['UTC'],last_verified_completed_counts=counts,
        aggregate_completed_counts_last_verified=aggregate,unknown_inflight_ids=[],
        actual_costs_are_durable_lower_bounds_while_running=False,new_training_updates=4000,new_AD_queries=4000,
        analysis_pid=None,supervisor_pid=None,actual_progress_after_saved_snapshot='native_terminal',
        collection_complete=False,independent_readback_complete=False)
    for value in [master,program,current]:
        value.update(status='WP3_016_native0_complete_independent_readback_pending',active_scientific_jobs=0,
            active_research=None,pending_run_id=None,current_stage='WP3_016_collection_and_independent_numeric_QA',
            wp3_016_execution=refs,wp3_execution=refs,wp3_016_G_updates=4000,wp3_016_AD_queries=4000,
            new_training_updates=total,cumulative_training_updates=total,current_run_training_updates=4000,
            wp3_G_updates=12000,wp3_AD_queries=12000,wp3_total_completed_G_updates_lower_bound=12000,
            short_candidate_runs_completed=10,wp3_short_candidate_runs_completed=6,
            active_scientific_jobs_status_basis='native0 COMPLETE and both owned processes absent at sole saved wake snapshot',
            training_updates_after_dispatch='terminal 4000 complete; no inflight; 3863 new G added to prior accounted175804',
            current_followup_status='16_45_wake_delivered_cpu_readback_registration_pending',
            next_action='Finish selective collection and JSON selection; register one necessary independent CPU saved-array readback, then continue WP4-WP6.',
            scientific_status='response_match_not_established',updated_UTC=stamp)
    program['training_update_accounting']='175667 before016 + exact terminal016 4000 =179667; only3863 added at this wake. Failed first native had G0 and retained nontraining compute.'
    job.update(status='completed_native0_pending_collection_and_independent_readback',execution=refs,
        actual_counters=counts,actual_counters_last_verified=counts,aggregate_completed_counts_last_verified=aggregate,
        actual_counters_status='terminal_exact_complete_journal_and_COMPLETE_agree',G_updates=4000,new_AD_queries=4000,
        completed_short_candidate_runs=2,native_exit_code=0,supervisor_exit_code=0,analysis_pid=None,supervisor_pid=None,
        native_ended_UTC=native['UTC'],terminal_readback=ref(terminal_path))
    jobs.update(active_scientific_jobs=0,updated_UTC=stamp,active_scientific_jobs_status_basis=program['active_scientific_jobs_status_basis'])
    wp3=next(w for w in master['work_packages'] if w['id']=='WP3')
    wp3.update(status='016_native0_complete_independent_readback_pending',execution=refs,execution016=refs)
    master['budgets']['short_candidate_runs'].update(completed=10,completed_WP3_trials=6,remaining_unregistered_WP3_slots=0)
    master['budgets']['WP3_016']['aggregate_actual_completed_counts']=aggregate
    old=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in old
    prefix='''# N01 当前执行：WP3/016 两次短试完成，独立数值复核待执行

两个seed各2000G，计算与监督均native0。94249项操作进入=完成，无在途；本轮共4000G/AD，扣除已入账137G后仅补记3863G，累计179667。搜索短试已完成10次、WP3完成6/6次，不再为近阈值追加短试。首次失败的非训练费用与原目录继续保留。

当前回收小结果包并按冻结合同做独立JSON选择及必要保存数组CPU复核；研究参考选择尚未最终确认，科学状态仍为response_match_not_established。016训练结束不代表六工作包全部完成。后续施工和独审使用GPT-6.1-Sol Max。

[016原生终点](receipts/wp3_016_20261015/postrun_root_01/NATIVE_COMPLETION_READBACK.json) · [016合同](docs/WP3_016_CONFIGURATION.md) · [015报告](reports/wp3_015_20261015/REPORT.html)

'''
    paths[3].write_text(prefix+old[old.index(marker):],encoding='utf-8',newline='\n')
    write(paths[0],master);program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for path,value in [(paths[1],program),(paths[2],jobs),(CURRENT,current)]:write(path,value)
    print(json.dumps(dict(status=receipt['status'],actual016_G=4000,newly_added_G=added,cumulative_G=total,
        active_scientific_jobs=0,operations=94249,receipt=ref(terminal_path))))


if __name__=='__main__':main()
