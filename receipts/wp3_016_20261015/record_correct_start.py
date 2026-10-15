"""Account only the single saved016 first-step/healthy snapshot, once."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size)


def write(path,value,new=False):
    with path.open('x' if new else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    p=read(ROOT/'wp3_016_config.json')
    revision=p.get('launch_revision')
    suffix={'fix01':'_FIX01','fix02':'_FIX02'}.get(revision,'')
    source_receipt=OUT/('SOURCE_COMMIT_READBACK'+suffix+'.json')
    source=read(source_receipt)
    start_dir,verify_dir=[OUT/name for name in (('start_native_02','start_verification_02') if revision=='fix02' else ('start_native_01','start_verification_01'))]
    launch=read(start_dir/'START_RECEIPT.json')
    state=read(verify_dir/'REMOTE_STATE.json')
    saved_validation=read(verify_dir/'START_VALIDATION.json')
    spec=importlib.util.spec_from_file_location('root016_start_replay',ROOT/'scripts/wp3_016_controller.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    verification=controller.first_valid(state,p,current_boot=True)
    assert verification==saved_validation
    assert verification['same_snapshot_health']=='current_owned_process_chain_and_resources_verified'
    assert state.get('runs/primary/supervisor/exit_receipt.json') is None
    progress=state['durable_child_progress'];counts=progress['completed_counts']
    assert not progress.get('failed_ids') and not progress.get('truncated_tail')
    assert 1<=counts['G_updates']<4000 and counts['new_AD_queries']>=counts['G_updates']
    assert source['configuration_sha256']==sha(ROOT/'wp3_016_config.json')
    paths=[ROOT/n for n in ('MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md')]+[CURRENT]
    master,program,jobs,current=[read(path) for path in (paths[0],paths[1],paths[2],CURRENT)]
    assert program['new_training_updates']==175667
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015')
    assert job['status'] in ('frozen_ready_not_started','start_dispatched_first_update_verification_pending')
    snapshot=OUT/'metadata_before_correct_start';snapshot.mkdir(exist_ok=False)
    for path in paths:shutil.copy2(path,snapshot/path.name)
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    refs=copy.deepcopy(program['wp3_016_execution'])
    refs.update(status='native_running_last_verified',source_commit=source['source_commit'],source_commit_pending_local_readback=False,
        source_commit_readback=ref(source_receipt),start_receipt=ref(start_dir/'START_RECEIPT.json'),
        startup_verification=ref(verify_dir/'START_VALIDATION.json'),startup_snapshot=ref(verify_dir/'REMOTE_STATE.json'),
        analysis_pid=verification['child_pid'],supervisor_pid=launch['supervisor_pid'],
        last_verified_UTC=state['UTC'],last_verified_completed_counts=counts,
        unknown_inflight_ids=progress.get('unknown_inflight_ids',[]),actual_costs_are_durable_lower_bounds_while_running=True,
        correct_start_verified=True,new_training_updates=counts['G_updates'],new_AD_queries=counts['new_AD_queries'],
        native_exit_code=None,supervisor_exit_code=None)
    prior=p.get('runtime_repair',{}).get('prior_actual_cost',{})
    aggregate={k:prior.get(k,0)+counts.get(k,0) for k in set(prior)|set(counts)}
    refs.update(prior_native_actual_cost=prior,aggregate_completed_counts_last_verified=aggregate,
        start_receipt_directory=start_dir.name,startup_verification_directory=verify_dir.name)
    for value in (master,program,current):
        value.update(status='WP3_016_native_running_last_verified',active_scientific_jobs=1,
            active_research=p['run_id'],pending_run_id=None,current_stage='WP3_016_native_charge_trials',
            latest_run_id=p['run_id'],latest_training_run_id=p['run_id'],latest_training_source_commit=source['source_commit'],
            wp3_016_execution=refs,wp3_execution=refs,wp3_016_G_updates=counts['G_updates'],wp3_016_AD_queries=counts['new_AD_queries'],
            cumulative_training_updates=175667+counts['G_updates'],wp3_total_completed_G_updates_lower_bound=8000+counts['G_updates'],
            next_action='Set one future n01 wake, rest, then collect016 and continue WP4-WP6.',
            active_scientific_jobs_status_basis='one first real two-parameter update and current same-snapshot owned-process/resource health',
            latest_subagent_model_preference='GPT-6.1-Sol Max',updated_UTC=stamp)
    program.update(new_training_updates=175667+counts['G_updates'],current_run_training_updates=counts['G_updates'],
        training_update_accounting='175667 before016 plus last saved durable016 G; future reads add only the new difference, never the whole count again.')
    job.update(status='native_running_last_verified',execution=refs,actual_counters_last_verified=counts,
        actual_counters=counts,actual_counters_status='durable_lower_bounds_at_only_saved_live_snapshot',
        G_updates=counts['G_updates'],new_AD_queries=counts['new_AD_queries'],
        analysis_pid=verification['child_pid'],supervisor_pid=launch['supervisor_pid'],
        correct_start_verified=True,verified_start_UTC=state['UTC'],native_exit_code=None,supervisor_exit_code=None)
    job.update(prior_actual_cost=prior,aggregate_completed_counts_last_verified=aggregate,
        native_execution_attempts=2 if revision=='fix02' else 1)
    jobs.update(active_scientific_jobs=1,updated_UTC=stamp,
        active_scientific_jobs_status_basis=program['active_scientific_jobs_status_basis'])
    wp3=next(w for w in master['work_packages'] if w['id']=='WP3')
    wp3.update(status='016_native_running_last_verified',execution=refs,execution016=refs)
    text=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。'
    assert marker in text
    prefix=f'''# N01 当前执行：WP3/016 已正确启动

016已通过首次真实两参数AdamW更新及同一快照的当前进程身份/资源健康验证。原A/005和015 M0形状头冻结，mean关闭；两seed各2000G，最多4000G/AD。实际入账为唯一快照的{counts['G_updates']}G，累计{175667+counts['G_updates']}，在途操作不补满。全部后续工作使用GPT-6.1-Sol Max。首次原生失败的前向/随机/权重加载成本单独保留，修复运行使用独立目录。

当前等待一次未来n01回收唤醒后休息。最终效果和原硬观察门由完整结果判定；科学状态仍为response_match_not_established。015报告、用途6/6和均值适配负结果已闭合保留。

[016配置](wp3_016_config.json) · [016启动验证]({verify_dir.relative_to(ROOT).as_posix()}/START_VALIDATION.json) · [015报告](reports/wp3_015_20261015/REPORT.md)

'''
    paths[3].write_text(prefix+text[text.index(marker):],encoding='utf-8')
    write(paths[0],master)
    program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0])
    current['plan_sha256']=sha(paths[3])
    for path,value in ((paths[1],program),(paths[2],jobs),(CURRENT,current)):write(path,value)
    receipt=dict(schema='n01/wp3-016-correct-start-accounting/v1',UTC=stamp,
        status='actual_first_charge_update_and_current_same_snapshot_health_verified',
        source_commit=source['source_commit'],source_controls=source['source_controls'],
        first_update=ref(verify_dir/'START_VALIDATION.json'),only_snapshot=ref(verify_dir/'REMOTE_STATE.json'),
        actual_counts_last_verified=counts,unknown_inflight_ids=progress.get('unknown_inflight_ids',[]),
        launch_revision=revision,remote_root=p['remote_root'],prior_native_actual_cost=prior,
        aggregate_completed_counts_last_verified=aggregate,
        start_receipt_directory=start_dir.name,startup_verification_directory=verify_dir.name,
        cumulative_training_before016=175667,cumulative_training_updates=175667+counts['G_updates'],
        source_A_005_M0_frozen=True,model_policy='gpt-6.1-sol / max',future_wake_pending=True,
        scientific_acceptance=False,whole_project_complete=False)
    write(OUT/'EXECUTION_STARTED.json',receipt,new=True)
    print(json.dumps(dict(status=receipt['status'],last_saved_G=counts['G_updates'],
        cumulative=receipt['cumulative_training_updates'],future_wake_pending=True)))


if __name__=='__main__':main()
