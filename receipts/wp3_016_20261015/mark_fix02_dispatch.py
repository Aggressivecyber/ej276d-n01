"""Record local start dispatch without treating it as verified healthy compute."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CURRENT=Path('E:/GAN/CURRENT_SIM2REAL_PROGRAM.json')


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    config=read(ROOT/'wp3_016_config.json')
    source=read(OUT/'SOURCE_COMMIT_READBACK_FIX02.json')
    start_path=OUT/'start_native_02/START_RECEIPT.json';launch=read(start_path)
    assert config['launch_revision']=='fix02' and source['configuration_sha256']==sha(ROOT/'wp3_016_config.json')
    assert launch['run_id']==config['run_id'] and type(launch['supervisor_pid']) is int and launch['supervisor_pid']>0
    paths=[ROOT/n for n in ['MASTER_EXECUTION_PLAN.json','PROGRAM.json','jobs.json','README.md']]+[CURRENT]
    master,program,jobs,current=[read(p) for p in [paths[0],paths[1],paths[2],CURRENT]]
    assert program['new_training_updates']==175667 and program['active_scientific_jobs']==0
    job=next(j for j in jobs['jobs'] if j['job_id']=='N01_WP3_016_20261015')
    assert job['status']=='frozen_ready_not_started'
    snapshot=OUT/'metadata_before_dispatch_notice_FIX02';snapshot.mkdir(exist_ok=False)
    for p in paths:shutil.copy2(p,snapshot/p.name)
    refs=copy.deepcopy(program['wp3_016_execution'])
    refs.update(status='start_dispatched_first_update_verification_pending',
        supervisor_pid=launch['supervisor_pid'],correct_start_verified=False,
        start_receipt=dict(path=start_path.relative_to(ROOT).as_posix(),sha256=sha(start_path),bytes=start_path.stat().st_size),
        actual_progress_unknown_after_dispatch=True,new_training_updates=None,new_AD_queries=None)
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    for value in [master,program,current]:
        value.update(status='WP3_016_start_dispatched_first_update_verification_pending',
            active_scientific_jobs=1,active_research=config['run_id'],pending_run_id=None,
            current_stage='WP3_016_first_update_verification_pending',wp3_016_execution=refs,wp3_execution=refs,
            wp3_016_G_updates=None,wp3_016_AD_queries=None,current_run_training_updates=None,
            active_scientific_jobs_status_basis='one local dispatch receipt; first update and same-snapshot health not yet verified',
            training_updates_after_dispatch='unknown_until_the_one_saved_snapshot; 175667 is prior accounted total',updated_UTC=stamp)
    job.update(status=refs['status'],execution=refs,supervisor_pid=launch['supervisor_pid'],
        actual_counters=None,actual_counters_last_verified=None,G_updates=None,new_AD_queries=None,
        actual_counters_status='unknown_until_first_saved_snapshot',correct_start_verified=False,native_execution_attempts=2)
    jobs.update(active_scientific_jobs=1,updated_UTC=stamp,active_scientific_jobs_status_basis=program['active_scientific_jobs_status_basis'])
    next(w for w in master['work_packages'] if w['id']=='WP3').update(status='016_first_update_verification_pending',execution=refs,execution016=refs)
    old=paths[3].read_text(encoding='utf-8-sig');marker='以下保留此前阶段说明。';assert marker in old
    prefix='''# N01 当前执行：WP3/016 修复运行已分派，首步验证中

fix02已在新隔离目录分派，首个真实训练更新和同一快照健康尚待验证。原生启动后的进度当前未知；累计175667为分派前已入账步数。首次失败的非训练计算与E/D证据保留，只允许本次已审查替代运行，原科学设置和两seed各2000G不变。后续施工和独审使用GPT-6.1-Sol Max。

[016配置](wp3_016_config.json) · [修复及累计预算](docs/WP3_016_RUNTIME_FIX02.json) · [015报告](reports/wp3_015_20261015/REPORT.html)

'''
    paths[3].write_text(prefix+old[old.index(marker):],encoding='utf-8',newline='\n')
    paths[0].write_text(json.dumps(master,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    program['master_plan_sha256']=current['master_plan_sha256']=sha(paths[0]);current['plan_sha256']=sha(paths[3])
    for p,v in [(paths[1],program),(paths[2],jobs),(CURRENT,current)]:
        p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=refs['status'],source_commit=source['source_commit'],supervisor_pid=launch['supervisor_pid'],accounted_prior_G=175667,new_progress='unknown')))


if __name__=='__main__':main()
