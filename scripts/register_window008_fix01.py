"""Register the bounded recovery and preserve all earlier status records."""
import datetime as dt
import json
import os
from window008_fix01_runtime import ROOT,read,sha,write_new,validate_inputs

def replace_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(temp,path)

def main():
    cfg,_,_=validate_inputs();stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    rec=ROOT/'receipts/window008_fix01_20261013/registration_before'
    rec.mkdir(exist_ok=False)
    paths=[ROOT/'jobs.json',ROOT/'PROGRAM.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']
    before={}
    for path in paths:
        (rec/path.name).write_bytes(path.read_bytes());before[str(path)]=sha(path)
    jobs=read(ROOT/'jobs.json')
    if any(row['job_id']=='N01_WINDOW008_FIX01_20261013' for row in jobs['jobs']):raise ValueError('Recovery already registered')
    jobs['jobs'].append(dict(job_id='N01_WINDOW008_FIX01_20261013',status='registered_reviewed_recovery_pending_source_freeze_and_dispatch',
        registered_UTC=stamp,host='100.82.231.102',pid=None,supervisor_pid=None,scientific_job=True,
        remote_root=cfg['remote_root'],configuration='window008_fix01_config.json',
        protocol='docs/WINDOW008_STABILITY_FIX01_PROTOCOL.md',source_manifest=cfg['source_manifest'],
        planned_new_G_updates=35000,planned_new_D_updates=96000,observed_new_G_updates=0,observed_new_D_updates=0,
        retained_deterministic_G_updates=19000,inherited_full_mean_G_updates=3000,
        effective_comparison_G_budget=57000,effective_comparison_D_budget=144000,
        historical_actual_G_after_success=57269,historical_actual_D_after_success=144807,
        discarded_legacy_joint_G_updates=269,discarded_legacy_joint_D_updates=807,
        execution_arms=['full','simple'],comparison_arms=['deterministic','full','simple'],
        C2ST_updates=0,noise_updates=0,population_updates=0,wall_limit_s=21600,
        effective_observation_samples=256,internal_carrier_samples=384,
        original_job_native_exit_code=1,original_deterministic_endpoint_reused=True,
        native_CUDA_engineering_G_updates=4,native_CUDA_engineering_D_updates=6,
        scientific_status='response_match_not_established'))
    replace_json(ROOT/'jobs.json',jobs)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(path);value.update(status='window008_fix01_reviewed_pending_source_freeze_and_dispatch',
            active_scientific_jobs=0,current_arm=None,current_stage='fix01_native_QA_complete',
            latest_run_id=cfg['run_id'],active_research=cfg['run_id'],updated_UTC=stamp,
            window008_fix01_configuration='window008_fix01_config.json',
            window008_fix01_protocol='docs/WINDOW008_STABILITY_FIX01_PROTOCOL.md',
            window008_fix01_source_manifest=cfg['source_manifest'],window008_fix01_new_G_updates=0,
            window008_fix01_new_D_updates=0,window008_fix01_training_pid=None,window008_fix01_supervisor_pid=None,
            window008_fix01_planned_new_G_updates=35000,window008_fix01_planned_new_D_updates=96000,
            next_action='Freeze reviewed fix01 sources, verify strict stage3000 import and first scientific update, then schedule estimated-completion wakeup.',
            scientific_status='response_match_not_established')
        if path.name=='PROGRAM.json':
            value['training_update_accounting']='Prior N01 stages 100000 + original Window008 actual22269 + fix01 NEW scientific G; imported and engineering updates excluded. D counted separately.'
            value['current_run_training_updates']=0
        replace_json(path,value)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');first,rest=text.split('\n',1)
    latest='''

## 当前执行状态：Window008 fix01 恢复

2026-10-13 定时检查确认原008已原生退出1：确定性臂19,000 G终点完成，full在3,269 G之后因随机形状基积分使电荷归一化分母非正而停止。原现场、源码和失败回执保留。修复仅给full随机形状基施加零规范电荷投影；独立CPU审查、原生CUDA失败重现、正式人工批量正反向和数值回放已通过。修复还不构成物理响应匹配证明。

恢复配置为`window008_fix01_config.json`。保留原确定性19k终点，full严格导入原3k均值检查点的模型、优化器、采样器与随机流，从3001更新；simple从原共同初始化运行。新增35,000 G / 96,000 D，三臂有效终点仍为57,000 G / 144,000 D；原失败的269 G / 807 D另外保留在实际历史计数中。有效波形256点，内部384点，噪声/人口/C2ST新增优化均0。

长任务正确启动后安排一次预计完成时苏醒，无需持续监控。当前PID、启动证据与实际状态以`PROGRAM.json`、`jobs.json`和`receipts/window008_fix01_20261013/START_RECEIPT.json`为准。原始第9节结果与历史记录继续保留；当前科学状态为`response_match_not_established`。恢复三臂终点且新原生退出0之后，才冻结独立评价、数值回放、报告与E盘私有归档；不发布或推送。
'''
    readme.write_text(first+latest+'\n'+rest,encoding='utf8')
    pointer=read(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json');pointer['plan_sha256']=sha(readme);replace_json(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json',pointer)
    write_new(ROOT/'receipts/window008_fix01_20261013/REGISTRATION_RECEIPT.json',dict(UTC=stamp,status='recovery_registered_before_scientific_dispatch',
        before_files_sha256=before,config_sha256=sha(ROOT/'window008_fix01_config.json'),new_scientific_G_updates=0,new_scientific_D_updates=0))
    print(json.dumps(dict(status='recovery_registered',new_G_budget=35000,new_D_budget=96000)))

if __name__=='__main__':main()
