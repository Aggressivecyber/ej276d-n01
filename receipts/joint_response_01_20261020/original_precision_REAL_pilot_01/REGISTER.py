from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root=Path('E:/GAN/research/N01'); here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/original_draw_precision_fix01'
review_path=previous/'terminal_01/TERMINAL_REVIEW.json'
review=json.loads(review_path.read_bytes()); assert review['two_seed_FRESH_pass'] and review['committed_parameter_updates']==0
ledger=json.loads((root/'jobs.json').read_bytes()); old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==132 and ledger['active_scientific_jobs']==0
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,here/'before_registration'/name)
now=datetime.now(timezone.utc).isoformat()
old_config='configs/joint_response_01_multibank_REAL_pilot_20261020.json'
p=json.loads((root/old_config).read_bytes()); old=copy.deepcopy(p)
p.update(schema='n01/joint-response01-original-precision-real-pilot/v1',
    job_id='JOINT_RESPONSE_01_ORIGINAL_PRECISION_REAL_PILOT_20261020',registered_UTC=now,
    method_contract='docs/JOINT_RESPONSE_01_ORIGINAL_PRECISION_REAL_PILOT_20261020.md',
    multibank_method_review=review_path.relative_to(root).as_posix(),iterations=3,
    training_seed=202660300000,evaluation_seed=202660400000,kernel_seed=202660500000,
    same_generator_and_scientific_gates=True,no_automatic_iteration_extension=True,
    CPU_resident_banks_and_carrier_references=True)
p['batch']['draws_per_parent']=32
p['feature_seeds']={name:2026103101+i for i,name in enumerate(old['feature_seeds'])}
p['native_projection_seeds']={name:2026103111+i for i,name in enumerate(old['native_projection_seeds'])}
p['resources'].update(GPU_GiB=4,internal_seconds=5400,hard_wall_seconds=5700,output_MiB=384,GPU_free_required_MiB=6144)
b=p['operation_budget']
directions=4*p['iterations']; fd=2*(54+46)*p['iterations']; candidates=3*directions
train_calls=8*directions+6*fd+4*candidates
train_rows=train_calls*32768; final_calls=16; final_rows=final_calls*8192
b.update(REAL_fit_feature_rows=directions*3*4096,REAL_CHECK_feature_rows=directions*4096,
    allocated_banks=directions*4+8,shape_standard_normal_values=(directions*4*32768+4*8192+4*128*1024)*8,
    noise005_calls=directions*4+4,noise005_rows=directions*4*32768+4*8192,
    extension_rows=3*(directions*4*32768+4*8192),joint_response_calls=train_calls+final_calls+2048,
    joint_response_rows=train_rows+final_rows+2097152,generated_observation_calls=3*(train_calls+final_calls),
    generated_observation_rows=3*(train_rows+final_rows),native_component_rows=train_calls*3*(32768+4096),
    finite_difference_columns=fd,FIT_bank_finite_difference_columns=3*fd,
    joint_direction_solve_attempts=candidates,joint_direction_solves=candidates,joint_candidate_trials=candidates,
    secant_matrix_corrections=2*directions,REAL_response_updates=directions)
assert fd==600 and b['joint_response_rows']==128057344 and b['shape_standard_normal_values']==17039360
assert p['success']==old['success'] and p['solver']==old['solver'] and p['native_guard']==old['native_guard']
inputs={p['multibank_method_review']:str(review_path)}
script='scripts/run_joint_response01_original_precision_real.py'
for name in (old_config,script,p['method_contract'],'scripts/diagnose_joint01_original_draw_precision_fix01.py',
    'src/n01/joint01_integer_w1.py',p['multibank_method_review']):
    p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
config='configs/joint_response_01_original_precision_REAL_pilot_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
manifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources=len(set(manifest['files'])|set(inputs)|{config,script,p['method_contract']})
ast.parse((root/script).read_text('utf-8'))
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,
    config_sha256=digest,host='rylan@100.82.231.102',
    remote_root='/home/rylan/projects/ej276d-n01/joint_response01_original_precision_REAL_pilot_20261020',
    resources=p['resources'],operation_budget=b,parameter_updates=0,scientific_acceptance=False,
    automatic_restart=False,outer_iterations_per_track=3,matched_physical_controls=True))
assert ledger['jobs'][:132]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_original_precision_REAL_pilot_registered',active_scientific_jobs=0,
        pending_job_id=p['job_id'],active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,
        updated_UTC=now,current_stage='4_original_precision_REAL_pilot',active_work_package='4_original_precision_REAL_pilot',
        scientific_acceptance=False,current_action='Three-round full-coverage 32-draw REAL pilot and matched physical controls registered.',
        next_action='Deploy and launch one bounded pilot; retain original final gates, matched kernel controls and prior failures.')
    d['joint_response_01'].update(original_precision_REAL_pilot_config=config,
        original_precision_REAL_pilot_config_sha256=digest,original_precision_REAL_pilot_status='registered_not_started',
        original_precision_REAL_pilot_started=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,
    config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),
    science_started=False,prior_132_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
