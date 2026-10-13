"""Register the new assumption-explicit bounded response experiment."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ID='response_007_20261013'
REC=ROOT/'receipts'/ID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not (REC/'PREREGISTRATION.json').exists(),'Already registered'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head=='505468fe74bae43e2dd52308d8a73a4628ca5284'
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json')
    assert program['active_scientific_jobs']==jobs['active_scientific_jobs']==0
    for name,digest in read(ROOT/'INTEGRATION006_ID_FIX02_SOURCE_MANIFEST.json')['files'].items():
        assert sha(ROOT/name)==digest,name
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot=ROOT/'provenance/before_response007';snapshot.mkdir(exist_ok=False)
    saved=[]
    for p in [ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']:
        target=snapshot/p.name;shutil.copy2(p,target)
        saved.append(dict(source=str(p),copy=target.relative_to(ROOT).as_posix(),sha256=sha(target)))
    write(snapshot/'SNAPSHOT.json',dict(UTC=now,files=saved))
    prior=read(ROOT/'integration006_config.json')
    refs={k:prior['references'][k] for k in ['noise005','statistics001','complete005','config005','final005','scope005']}
    cfg=dict(run_id=ID,registered_UTC=now,previous_completed_commit=head,
        local_training_container=prior['local_training_container'],remote_training_container=prior['remote_training_container'],
        train_container_sha256=prior['train_container_sha256'],identities_sha256=prior['identities_sha256'],
        remote_root='/home/rylan/projects/ej276d-n01/'+ID,references=refs,
        source_manifest='RESPONSE007_SOURCE_MANIFEST.json',hypotheses=prior['hypotheses'],extension=prior['extension'],
        arms=['deterministic','full'],hypothesis_weights=[1/3]*3,signal_hypothesis_input=False,
        response=dict(mean_response_updates=3000,joint_stochastic_updates=12000,low_learning_rate_updates=4000,
            updates_per_arm=19000,total_G_updates=38000,total_D_updates=96000,warm_start=False,
            population_learning=False,common_initial_state=True,independent_D_per_hypothesis=True),
        loss=dict(version=2,weights=dict(wave=1.,QPSD=2.,noise=1.,adv=.1,regular=.01,trend=.1),
            statistics='statistics001',hypothesis_reduction='mean_of_separate_losses',
            G_lr=2e-4,D_lr=1e-4,G_low_lr=5e-5,D_low_lr=2.5e-5,G_weight_decay=1e-4,clip_norm=5.),
        distribution_microbatches=4,source_parents_per_microbatch=128,responses_per_parent=2,real_rows_per_microbatch=256,
        seeds=dict(initialization=2026101301,discriminator=2026101302,source_sampler=2026101303,real_sampler=2026101304,
            train_response=2026101305,train_noise005=2026101306,train_extension=2026101307,train_projection=2026101308,
            evaluation_projection=2026101309,TRAIN_response=2026101310,TRAIN_noise005=2026101311,TRAIN_extension=2026101312,
            DEV_response=2026101313,DEV_noise005=2026101314,DEV_extension=2026101315,
            selected_TRAIN=2026101316,selected_DEV=2026101317,real_TRAIN=2026101318,real_DEV=2026101319,
            repeat_selection=2026101320,repeat_response=2026101321,repeat_noise005=2026101322,repeat_extension=2026101323,
            intervention_response=2026101324,intervention_noise005=2026101325,intervention_extension=2026101326,
            bootstrap_TRAIN=2026101327,bootstrap_DEV=2026101328,classifier=2026101329,classifier_bootstrap=2026101330),
        evaluation=dict(draws_per_parent=4,chunk_rows=128,sample_waveform_rows=128,
            repeat_parents=128,repeat_draws=64,intervention_pairs=32,intervention_draws=8,
            matched_TRAIN_per_class=2048,matched_DEV_per_class=256,bootstrap_draws=1000,
            minimum_parents_per_fixed_bin=32,minimum_rows_per_fixed_bin=128,
            classifier_views=['full480','prefix100','pulse380'],classifiers=21,updates_per_classifier=200,
            classifier_total_updates=4200,classifier_batch_per_class=64,classifier_lr=.001,
            no_GMM_fits=True,scientific_pass_threshold=None,DEV_selects_parameters=False),
        engineering=dict(signal_prefix_max_V=1/16383,nonpositive_Q_stops=True,guarded_PSD_stops=True,
            waveform_replay_first_steps=2,gradient_audit_interval=1000),
        cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,wall_limit_s=28800,output_limit_MiB=2048,
        checkpoint_interval=1000,status_interval=25,one_active_scientific_job=True,noise_updates=0,population_updates=0,
        independent_acquisition_available=False,validation_previously_exposed=True,
        previous_006_outcome='sensitivity_detected',review_mode='primary_agent_self_review_no_subagents')
    write(ROOT/'response007_config.json',cfg)
    write(REC/'PREREGISTRATION.json',dict(run_id=ID,UTC=now,config_sha256=sha(ROOT/'response007_config.json'),
        protocol_sha256=sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md'),references=refs,previous_completed_commit=head,
        before_new_scientific_array_decode=True,before_new_scientific_forward=True,
        planned_G_updates=38000,planned_D_updates=96000,planned_C2ST_updates=4200,
        noise_updates=0,subagents_used_for_implementation=False,legacy_results_modified=False))
    program.update(status='response007_registered',latest_run_id=ID,active_repair=ID,active_scientific_jobs=0,
        updated_UTC=now,current_stage='response007_implementation',current_run_training_updates=0,
        response007_configuration='response007_config.json',response007_protocol='docs/RESPONSE_007_PROTOCOL.md',
        response007_G_updates=0,response007_D_updates=0,response007_C2ST_updates=0,response007_noise_updates=0,
        next_action='Implement and self-review registered two-arm three-hypothesis response training, then execute fixed budget.')
    write(ROOT/'PROGRAM.json',program)
    jobs['updated_UTC']=now
    jobs['jobs'].append(dict(job_id='N01_RESPONSE007_20261013',status='registered',registered_UTC=now,host='100.82.231.102',
        pid=None,supervisor_pid=None,response_updates_planned=38000,D_updates_planned=96000,C2ST_updates_planned=4200,
        noise_updates_planned=0,configuration='response007_config.json',protocol='docs/RESPONSE_007_PROTOCOL.md',
        preregistration=f'receipts/{ID}/PREREGISTRATION.json',remote_root=cfg['remote_root']))
    write(ROOT/'jobs.json',jobs)
    pointer=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';value=read(pointer)
    value.update(status=program['status'],latest_run_id=ID,active_repair=ID,active_scientific_jobs=0,updated_UTC=now,
        response007_configuration=str(ROOT/'response007_config.json'),response007_protocol=str(ROOT/'docs/RESPONSE_007_PROTOCOL.md'),
        response007_G_updates=0,response007_D_updates=0,response007_C2ST_updates=0)
    write(pointer,value)
    p=ROOT/'README.md';old=p.read_text(encoding='utf8');lines=old.splitlines(keepends=True)
    lines.insert(1,'\n当前推进007：两臂、三种固定长程噪声假设的共同响应训练，已登记、尚未启动。计划38000次G更新；006敏感性与响应匹配未建立的结论保留。见 [007协议](docs/RESPONSE_007_PROTOCOL.md)、PROGRAM.json 和 jobs.json。按用户要求全部由主代理实施及自检。\n')
    p.write_text(''.join(lines),encoding='utf8',newline='\n')
    print(json.dumps(dict(status='registered',run_id=ID,config_sha256=sha(ROOT/'response007_config.json'),protocol_sha256=sha(ROOT/'docs/RESPONSE_007_PROTOCOL.md'))))
if __name__=='__main__':main()
