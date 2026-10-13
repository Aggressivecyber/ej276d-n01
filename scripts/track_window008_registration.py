"""Register the already-reviewed256-point replacement before scientific dispatch."""
from pathlib import Path
import datetime as dt
import json
import os

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    os.replace(temporary,path)


def main():
    cfg=json.loads((ROOT/'window008_config.json').read_text(encoding='utf8'))
    jobs_path=ROOT/'jobs.json';jobs=json.loads(jobs_path.read_text(encoding='utf8'))
    job_id='N01_WINDOW008_20261013'
    if any(j['job_id']==job_id for j in jobs['jobs']):raise ValueError('Already registered; preserve existing state')
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    jobs['jobs'].append(dict(job_id=job_id,status='registered_pending_training_freeze_and_dispatch',registered_UTC=stamp,
        host='100.82.231.102',pid=None,supervisor_pid=None,scientific_job=True,
        remote_root=cfg['remote_root'],configuration='window008_config.json',protocol='docs/WINDOW008_PROTOCOL.md',
        source_manifest=cfg['source_manifest'],planned_G_updates=57000,planned_D_updates=144000,
        observed_G_updates=0,observed_D_updates=0,C2ST_updates=0,noise_updates=0,population_updates=0,
        arms=cfg['arms'],G_updates_per_arm=19000,wall_limit_s=cfg['wall_limit_s'],
        effective_observation_samples=256,internal_carrier_samples=384,
        evaluation='separate_frozen_job_after_training_native_exit',scientific_status='response_match_not_established'))
    write(jobs_path,jobs)
    for path in (ROOT/'PROGRAM.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json'):
        value=json.loads(path.read_text(encoding='utf8'))
        value.update(status='window008_registered_pending_freeze',active_scientific_jobs=0,
            latest_run_id=cfg['run_id'],active_research=cfg['run_id'],updated_UTC=stamp,
            current_stage='training_implementation_and_native_QA_complete',current_arm=None,
            window008_configuration='window008_config.json',window008_protocol='docs/WINDOW008_PROTOCOL.md',
            window008_G_updates=0,window008_D_updates=0,window008_C2ST_updates=0,window008_noise_updates=0,
            window008_population_updates=0,window008_training_pid=None,window008_supervisor_pid=None,
            window008_effective_observation_samples=256,window008_internal_carrier_samples=384,
            window008_source_manifest=cfg['source_manifest'],scientific_status='response_match_not_established',
            next_action='Freeze reviewed256-point training sources, dispatch fixed three-arm19000-G comparison, then evaluate the frozen endpoints separately.')
        if path.name=='PROGRAM.json':value['current_run_training_updates']=0
        write(path,value)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf8')
    text='当前工作：用户指定的256点有效观察窗版本008已完成实现与原生CUDA工程检查，正冻结并启动三组固定终点训练。内部384点保留100点基线和寻峰上下文；损失、评价和交付波形为256点。当前进度以 `PROGRAM.json`、`jobs.json` 和 `receipts/window008_20261013/` 为准。\n\n'+text
    readme.write_text(text,encoding='utf8')
    print(json.dumps(dict(job_id=job_id,status='registered',UTC=stamp)))


if __name__=='__main__':main()
