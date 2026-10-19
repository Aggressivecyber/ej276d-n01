from datetime import datetime,timezone
from pathlib import Path
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
now=datetime.now(timezone.utc).isoformat()
before=here/'before_registration'
before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,before/name)
est=root/'receipts/joint_response_01_20261020/c1_estimator_01/terminal_01'
review=json.loads((est/'TERMINAL_REVIEW.json').read_bytes())
remote=json.loads((est/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_ESTIMATOR_AUDIT_NO_SCIENTIFIC_ACCEPTANCE'
assert remote['native_exit']['exit_code']==0 and remote['processes']==[]
ledger=json.loads((root/'jobs.json').read_bytes())
old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==123 and old_jobs[-1]['job_id']=='JOINT_RESPONSE_01_C1_ESTIMATOR_DIAGNOSTIC_20261020'
old_config=json.loads((root/'configs/joint_response_01_c1_estimator_20261020.json').read_bytes())
p=copy.deepcopy(old_config)
for key in ('endpoint_iterations','bank_seed','order_seed','independent_repeats'):
    del p[key]
p.update(schema='n01/joint-response01-c1-matched-control/v1',job_id='JOINT_RESPONSE_01_C1_MATCHED_CONTROL_20261020',
    registered_UTC=now,stage='3_C1_target_family_control',method_contract='docs/JOINT_RESPONSE_01_C1_MATCHED_CONTROL_20261020.md',
    target_family='C1',feature_statistics_target_family='original',iterations=12,population_stress_iterations=0,
    precision_bank_seeds=[[2026102241,2026102242],[2026102243,2026102244]],
    positive_control_only=True,automatic_REAL_promotion=False,original_cross_family_failure_preserved=True)
p['resources'].update(internal_seconds=1800,hard_wall_seconds=2100,output_MiB=192)
p['operation_budget']=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,DEV_waveforms_decoded=0,
    old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
    allocated_banks=1274,shape_standard_normal_values=37789712,noise005_calls=1266,noise005_rows=2626562,extension_rows=7879686,
    joint_response_calls=39023,joint_response_rows=78214562,generated_observation_calls=107757,generated_observation_rows=214982886,
    finite_difference_columns=1296,joint_direction_solves=24,joint_candidate_trials=72,teacher_accepted_joint_updates=24,
    conditional_signal_rows=196608,native_metric_rows=1179648,feature_maps_constructed=18,
    REAL_response_updates=0,AD_queries=0,classifier_fits=0,full_fit_parent_banks=96,fullpool_FIT_chunk_columns=16848,
    stream_equivalence_checks=2)
inputs={}
for name in ('AUDIT_AND_DECOMPOSE.py','TERMINAL_REVIEW.json','SAVED_FEATURE_DECOMPOSITION.json','SAVED_SURROGATE_CHECK.py','SAVED_SURROGATE_CHECK.json'):
    path=est/name;relative=path.relative_to(root).as_posix()
    p['source_pins'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    inputs[relative]=str(path)
assert len(p['source_pins'])==96
config='configs/joint_response_01_c1_matched_control_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
script='scripts/run_joint_response01_c1_matched_control.py'
tree=ast.parse((root/script).read_text())
assert 'from run_joint_response01_c1_fullpool import fit_track_c1' in (root/script).read_text()
assert p['solver']==old_config['solver'] and p['success']==old_config['success']
assert p['batch']==old_config['batch'] and p['feature_seeds']==old_config['feature_seeds']
assert all(p['source_pins'][k]==v for k,v in old_config['source_pins'].items())
static=dict(status='PASS',UTC=now,syntax_pass=True,unchanged_fullpool_fit_import=True,old_pins_preserved=91,
    old_solver_success_batch_feature_roles_unchanged=True,new_files=[config,script,p['method_contract']],
    neural_fit_directions=24,max_updates=24,REAL_updates=0,no_automatic_promotion=True,
    expected_generator_calls=24*115*13+5+2+4*16+4*768,
    expected_generator_rows=24*115*25860+12288+12930+4*131072+4*1572864,
    recursive_collection_required_for_dual_target_outputs=True)
assert static['expected_generator_calls']==p['operation_budget']['joint_response_calls']
assert static['expected_generator_rows']==p['operation_budget']['joint_response_rows']
(here/'STATIC_REVIEW.json').write_text(json.dumps(static,indent=2)+'\n',encoding='utf-8',newline='\n')
ledger['jobs'][-1].update(status='completed_independently_audited_no_updates',native_exit_code=0,
    ended_UTC=remote['native_exit']['ended_utc'],terminal_review=str((est/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),
    execution_counts=review['execution_counts'],source_files_verified=744,scientific_acceptance=False)
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,
    config=config,config_sha256=digest,host='rylan@100.82.231.102',
    remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_matched_control_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],scientific_acceptance=False,
    positive_control_only=True,automatic_REAL_promotion=False))
assert ledger['jobs'][:122]==old_jobs[:122]
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    data=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    data.update(status='JOINT_RESPONSE_01_C1_matched_control_registered',active_scientific_jobs=0,
        current_stage=p['stage'],active_work_package=p['stage'],updated_UTC=now,last_verified_UTC=review['UTC'],
        scientific_status='response_match_not_established',scientific_acceptance=False,
        current_action='Frozen estimator audit complete; registered target-family-only positive control.',
        next_action='Dispatch the bounded matched-family control, verify exact original feature factors, then estimate one wake from measured work.',
        pending_job_id=p['job_id'],active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,
        followup_next_run_local=None,next_wake_local=None)
    joint=data['joint_response_01']
    joint.update(c1_estimator_native_exit_code=0,c1_estimator_independent_audit_pass=True,
        c1_estimator_terminal_review=str((est/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),
        c1_estimator_feature_decomposition=str((est/'SAVED_FEATURE_DECOMPOSITION.json').relative_to(root)).replace('\\','/'),
        c1_estimator_execution_counts=review['execution_counts'],c1_estimator_parameter_updates=0,
        c1_estimator_scientific_acceptance=False,c1_estimator_last_checked_UTC=remote['UTC'],
        c1_matched_control_config=config,c1_matched_control_config_sha256=digest,
        c1_matched_control_receipts=str(here.relative_to(root)).replace('\\','/'),c1_matched_control_started=False,
        c1_matched_control_positive_only=True,c1_matched_control_automatic_REAL_promotion=False)
    (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
registration=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=752,source_pins=96,science_started=False,previous_estimator_closed=True,prior_122_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(registration))
