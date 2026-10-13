"""Register the explicit server-online continuation before scientific dispatch."""
import datetime as dt
import json
import os
from window008_resume01_runtime import ROOT,read,sha,write_new,validate_inputs

def replace_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(temp,path)

def main():
    cfg,_,_=validate_inputs();stamp=dt.datetime.now(dt.timezone.utc).isoformat();rec=ROOT/'receipts/window008_resume01_20261013'
    backup=rec/'registration_before';backup.mkdir(parents=True,exist_ok=False)
    for path in (ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):(backup/path.name).write_bytes(path.read_bytes())
    jobs=read(ROOT/'jobs.json')
    if any(x['job_id']=='N01_WINDOW008_RESUME01_20261013' for x in jobs['jobs']):raise ValueError('Continuation already registered')
    jobs['jobs'].append(dict(job_id='N01_WINDOW008_RESUME01_20261013',status='registered_pending_review_freeze_and_dispatch',registered_UTC=stamp,
        host='100.82.231.102',pid=None,supervisor_pid=None,scientific_job=True,remote_root=cfg['remote_root'],configuration='window008_resume01_config.json',
        source_manifest=cfg['source_manifest'],protocol=cfg['protocol'],planned_new_G_updates=32000,planned_new_D_updates=87000,
        observed_new_G_updates=0,observed_new_D_updates=0,inherited_effective_G_updates=25000,inherited_effective_D_updates=57000,
        historical_actual_G_before_resume=25667,historical_actual_D_before_resume=59001,retained_original_deterministic_G_updates=19000,
        imported_full_G_updates=6000,discarded_user_stop_G_updates=398,discarded_user_stop_D_updates=1194,
        execution_arms=['full','simple'],comparison_arms=['deterministic','full','simple'],C2ST_updates=0,noise_updates=0,population_updates=0,
        effective_observation_samples=256,internal_carrier_samples=384,wall_limit_s=21600,
        user_resume_authorized=True,scientific_status='response_match_not_established'))
    replace_json(ROOT/'jobs.json',jobs)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=read(path);value.update(status='window008_resume01_registered_pending_review_and_dispatch',active_scientific_jobs=0,
            current_arm=None,current_stage='resume01_boundary_validation_ready',latest_run_id=cfg['run_id'],active_research=cfg['run_id'],updated_UTC=stamp,
            window008_resume01_configuration='window008_resume01_config.json',window008_resume01_source_manifest=cfg['source_manifest'],
            window008_resume01_protocol=cfg['protocol'],window008_resume01_new_G_updates=0,window008_resume01_new_D_updates=0,
            window008_resume01_training_pid=None,window008_resume01_supervisor_pid=None,
            next_action='Complete bounded continuation engineering/review, freeze sources, verify first science6001 and schedule one completion wakeup.',
            scientific_status='response_match_not_established')
        if path.name=='PROGRAM.json':value.update(new_training_updates=125667,current_run_training_updates=0)
        replace_json(path,value)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf8');first,rest=text.split('\n',1)
    notice='\n\n当前推进：用户已通知服务器恢复并要求推进。Window008 resume01严格从full完整6000步周期检查点恢复全部状态；保留原det19k终点，simple从原共同初始化运行19k。新增固定32000 G / 87000 D。原两次native1及停机现场保留，禁止直接恢复可能处于微批中途的FAILURE_CHECKPOINT。当前状态和PID以PROGRAM/jobs及新启动回执为准。子智能体按用户新指令统一GPT-6.1-Sol Max。\n'
    readme.write_text(first+notice+rest,encoding='utf8')
    pointer=read(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json');pointer['plan_sha256']=sha(readme);replace_json(ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json',pointer)
    write_new(rec/'REGISTRATION_RECEIPT.json',dict(UTC=stamp,status='resume01_registered_before_scientific_dispatch',
        config_sha256=sha(ROOT/'window008_resume01_config.json'),scientific_G_updates=0,scientific_D_updates=0))
    print(json.dumps(dict(status='resume01_registered',new_G_budget=32000,new_D_budget=87000)))

if __name__=='__main__':main()
