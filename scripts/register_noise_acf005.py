"""Prospectively bind the user-requested single ACF20 candidate. No array decode."""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ID='noise_acf005_20261012'
REC=ROOT/'receipts'/ID
PROTO=ROOT/'docs/NOISE_ACF005_PROTOCOL.md'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    if (REC/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve frozen registration')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='0f914546564e37bd4bdfdb58302d6becbef1f02f'
    prior=read(ROOT/'noise_improvement_004_config.json')
    program=read(ROOT/'PROGRAM.json');jobs=read(ROOT/'jobs.json')
    assert program['active_scientific_jobs']==jobs['active_scientific_jobs']==0
    for name,digest in read(ROOT/'runs/noise_improvement_004_20261012/runs/science/COMPLETE.json')['source_files'].items():
        assert sha(ROOT/name)==digest,name
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot=ROOT/'provenance/before_noise_acf005';snapshot.mkdir(exist_ok=False)
    copies=[]
    for path in [ROOT/'PROGRAM.json',ROOT/'jobs.json',ROOT/'README.md',ROOT/'noise_improvement_004_config.json',ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json']:
        target=snapshot/path.name;shutil.copy2(path,target)
        copies.append(dict(source=str(path),copy=target.relative_to(ROOT).as_posix(),sha256=sha(target)))
    write(snapshot/'SNAPSHOT.json',dict(UTC=now,files=copies))
    folder=ROOT/'provenance/noise_acf005_reference';folder.mkdir(exist_ok=False)
    oldscience=ROOT/'runs/noise_improvement_004_20261012/runs/science'
    sources={
        'initial004':oldscience/'noise/initial_state.pt',
        'final004':oldscience/'noise/final.pt',
        'gate004':oldscience/'noise_validation/NOISE_GATE.json',
        'config004':ROOT/'noise_improvement_004_config.json',
        'calibration003':ROOT/'runs/noise_origin_audit_003_20261012/continuation_001/runs/science/real_real/RR_CALIBRATION_REPORT.json',
        'origin003':ROOT/'runs/noise_origin_audit_003_20261012/runs/science/origin/ORIGIN_REPORT.json',
    }
    for view in ('raw','demean','detrend'):
        sources['control004_'+view]=oldscience/('noise_validation/classifiers/'+view+'/REAL_REAL/predictions.npz')
    references={}
    for key,path in sources.items():
        target=folder/(key+path.suffix);shutil.copy2(path,target)
        references[key]=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),source=path.relative_to(ROOT).as_posix())
        assert sha(path)==sha(target)
    config=dict(run_id=ID,registered_UTC=now,reviewed_commit='0f914546564e37bd4bdfdb58302d6becbef1f02f',
        noise=copy.deepcopy(prior['noise']),model=copy.deepcopy(prior['model']),
        train_container_sha256=prior['train_container_sha256'],identities_sha256=prior['identities_sha256'],
        local_training_container=prior['local_training_container'],remote_training_container=prior['remote_training_container'],
        remote_root='/home/rylan/projects/ej276d-n01/'+ID,references=references,
        source_manifest='NOISE_ACF005_SOURCE_MANIFEST.json',
        acf20=dict(lags=list(range(1,21)),scale=.015,coefficient=1.,
            objective='.5*(mean(e^2)+max(e^2))',reduction='eventwise_ratios_then_equal_event_mean',
            variance_floor_V2=(1/16383)**2/12,target_role='all_21047_canonical_REAL_TRAIN',target_fixed=True),
        diagnostics=dict(rows=8192,chunk_size=256,seed=prior['noise']['generation_seed']+20000,
            trajectory_updates=list(range(0,5001,250)),fixed_explicit_latents=True,
            save_initial_and_004final_and_005final_pre_post_ADC=True,
            save_full_covariance_every_trajectory=True,log_gradient_components_every_update=True,
            gradient_order='unchanged_named_parameters',angles_degrees=True,
            old_vs_acf_angle=True,old_vs_total_angle=True,zero_gradient_angle=None,
            training_rng_untouched=True),
        controls=dict(reference_permutations_per_refit=255,reference_refits=128,
            current_refits_for_conditional_test=0,conditional_primary_view='raw',
            conditional_permutations=9999,conditional_seed=2026131221,conditional_alpha=.05,
            conditional_statistic='abs(AUC-.5)',tail='>= including ties',p_rule='(tail+1)/(draws+1)',
            status_after_valid_implementation='measurement_uncertain_single_exposed_acquisition',
            legacy_classifiers_refit_once=6,legacy_classifier_updates=1200,
            RR_score_reproduction_required=True),
        uncertainty=dict(bootstrap_resamples=1000,TRAIN_seed=2026131222,development_seed=2026131223,
            resampling='original_group_clusters_for_REAL_and_independent_generated_events',
            outputs=['marginal95','simultaneous95_centered_max'],acceptance_gate=False),
        decision=dict(TRAIN_max_ACF_match=.015,improvement='acf20_loss_strictly_less_and_max_error_no_greater_than_004_on_same_diag_latents',
            numerical_tolerance=0.,candidate_checks='all_ten_unchanged_legacy_checks_except_two_negative_control_checks',
            design_check_requires_TRAIN_match_improvement_and_candidate_screen=True,
            design_check_only=True,scientifically_qualified=False),
        cpu_threads=12,gpu_memory_limit_GiB=10,job_tree_RSS_limit_GiB=8,wall_limit_s=1800,output_limit_MiB=512,
        checkpoint_policy='unique_final_5000_no_validation_selection',response_updates=0,response_integration_allowed=False,
        old_gates_changed=False,validation_previously_used_for_diagnostics=True,independent_acquisition_available=False,
        one_active_scientific_job=True)
    assert config['noise']==prior['noise'] and config['model']==prior['model']
    write(ROOT/'noise_acf005_config.json',config)
    write(REC/'PREREGISTRATION.json',dict(run_id=ID,UTC=now,config_sha256=sha(ROOT/'noise_acf005_config.json'),
        protocol_sha256=sha(PROTO),reviewed_commit=config['reviewed_commit'],
        before_new_scientific_array_decode=True,before_new_initialization=True,before_new_fit=True,
        noise_update_budget=5000,classifier_update_budget=1200,response_update_budget=0,
        old_noise_and_model_config_unchanged=True,old_gates_changed=False,references=references))
    program.update(status='noise_acf005_registered',active_repair=ID,active_scientific_jobs=0,
        updated_UTC=now,current_stage='ACF20_implementation_and_zero_training_diagnostic_pending',
        current_run_training_updates=0,acf005_configuration='noise_acf005_config.json',
        acf005_protocol='docs/NOISE_ACF005_PROTOCOL.md',acf005_noise_updates=0,acf005_response_updates=0,
        next_action='One fixed ACF20 candidate after zero-training comparison; then separated development evaluation; no formal pulse training.')
    write(ROOT/'PROGRAM.json',program)
    jobs.update(updated_UTC=now)
    jobs['jobs'].append(dict(job_id='N01_NOISE_ACF005_20261012',status='registered',registered_UTC=now,
        host='100.82.231.102',pid=None,supervisor_pid=None,noise_updates_planned=5000,
        classifier_updates_planned=1200,response_updates_planned=0,
        configuration='noise_acf005_config.json',protocol='docs/NOISE_ACF005_PROTOCOL.md',
        preregistration=(REC/'PREREGISTRATION.json').relative_to(ROOT).as_posix(),remote_root=config['remote_root']))
    write(ROOT/'jobs.json',jobs)
    readme_path=ROOT/'README.md';readme=readme_path.read_text(encoding='utf8')
    split=readme.split('\n',2)
    paragraph='**当前任务：005 ACF 目标覆盖修复已预登记。先做零训练初始化对照，再从相同初始化训练唯一 5000 次更新候选；旧门、候选检查与对照校准分别报告，正式脉冲训练预算为 0。** 见 [005 协议](docs/NOISE_ACF005_PROTOCOL.md) 与 `noise_acf005_config.json`。\n\n'
    readme_path.write_text(split[0]+'\n\n'+paragraph+split[2],encoding='utf8',newline='\n')
    pointer_path=ROOT.parents[1]/'CURRENT_SIM2REAL_PROGRAM.json';pointer=read(pointer_path)
    pointer.update(status=program['status'],active_repair=ID,active_scientific_jobs=0,updated_UTC=now,
        plan_sha256=sha(readme_path),acf005_configuration=str(ROOT/'noise_acf005_config.json'),acf005_response_updates=0)
    write(pointer_path,pointer)
    print(json.dumps(dict(registered=ID,config_sha256=sha(ROOT/'noise_acf005_config.json'),protocol_sha256=sha(PROTO),scientific_updates=0)))


if __name__=='__main__':main()
