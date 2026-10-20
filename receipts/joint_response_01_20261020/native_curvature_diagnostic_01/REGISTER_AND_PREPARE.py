from datetime import datetime,timezone
from pathlib import Path
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/original_precision_REAL_pilot_01'
old=json.loads((root/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes())
audit=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert audit['status']=='PASS_SAVED_ORIGINAL_PRECISION_REAL_PILOT_AUDIT' and not audit['REAL_calibration_candidate_pass_all']
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==133 and ledger['active_scientific_jobs']==0
before=here/'before_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
script='scripts/diagnose_joint01_native_curvature.py';contract='docs/JOINT_RESPONSE_01_NATIVE_CURVATURE_DIAGNOSTIC_20261020.md'
config='configs/joint_response_01_native_curvature_diagnostic_20261020.json'
now=datetime.now(timezone.utc).isoformat();inputs={}
saved=[]
for seed in range(2):
    target=f'diagnostic_inputs/CURVATURE_seed{seed}_joint_ITER02.npz'
    inputs[target]=str(previous/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz');saved.append(target)
fixed='diagnostic_inputs/CURVATURE_FIXED_PROPOSALS.npz';inputs[fixed]=str(previous/'SAVED_CONSTRAINT_PROPOSALS.npz')
diagnosis=(previous/'SAVED_CONSTRAINT_DIAGNOSIS.json').relative_to(root).as_posix();inputs[diagnosis]=str(previous/'SAVED_CONSTRAINT_DIAGNOSIS.json')
review=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix();inputs[review]=str(previous/'terminal_01/TERMINAL_REVIEW.json')
p={key:copy.deepcopy(old[key]) for key in ('frozen_setup_config','roles_file','batch','solver','native_guard','success')}
p.update(schema='n01/joint-response01-native-curvature-diagnostic/v1',job_id='JOINT_RESPONSE_01_NATIVE_CURVATURE_DIAGNOSTIC_20261020',registered_UTC=now,
    method_contract=contract,previous_terminal_review=review,saved_diagnosis=diagnosis,saved_arrays=saved,fixed_proposals=fixed,
    fresh_seed=202660600000,feature_seeds={**{k:v for k,v in old['feature_seeds'].items() if k!='FINAL'},'FRESH':2026103201},
    native_projection_seeds={**{k:v for k,v in old['native_projection_seeds'].items() if k!='FINAL'},'FRESH':2026103202},
    resources=dict(CPU_threads=4,GPU_GiB=4,RSS_GiB=12,internal_seconds=360,hard_wall_seconds=480,output_MiB=64,GPU_free_required_MiB=6144),
    operation_budget=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
        REAL_observable_replay_calls=1,REAL_observable_replay_rows=21047,feature_maps_constructed=5,new_native_projection_contexts=5,
        REAL_fit_feature_rows=24576,REAL_CHECK_feature_rows=16384,allocated_banks=10,noise005_calls=10,noise005_rows=327680,extension_rows=983040,
        shape_standard_normal_values=2621440,joint_response_calls=40,joint_response_rows=1310720,generated_observation_calls=120,generated_observation_rows=3932160,
        native_component_rows=4423680,saved_center_replays_verified=8,saved_anchor_replays_verified=8,saved_linear_candidate_replays_verified=8,fixed_candidate_checks=20,
        finite_difference_columns=0,joint_direction_solves=0,secant_matrix_corrections=0,REAL_response_updates=0,DEV_waveforms_decoded=0,AD_queries=0,classifier_fits=0),
    automatic_restart=False,automatic_training=False,committed_parameter_updates=0,
    original_final_gates_unchanged=True,source_pins=copy.deepcopy(old['source_pins']))
for name in (script,contract,diagnosis,review,'receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/DIAGNOSE_SAVED_CONSTRAINTS.py'):
    p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if name not in (script,contract):inputs[name]=str(root/name)
for name,path in inputs.items():p['source_pins'][name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
ast.parse((root/script).read_text('utf-8'))
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
previous_manifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());source_count=len(set(previous_manifest['files'])|set(inputs)|{script,contract,config})
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,
    host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_native_curvature_diagnostic_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False))
assert ledger['jobs'][:133]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_native_curvature_diagnostic_registered',active_scientific_jobs=0,pending_job_id=p['job_id'],active_job_id=None,active_run_id=None,
        updated_UTC=now,current_stage='4_native_curvature_diagnostic',active_work_package='4_native_curvature_diagnostic',scientific_acceptance=False,
        current_action='Fixed curvature and initial-anchor candidate diagnostic registered after complete saved-matrix analysis.',
        next_action='Replay original centers and fixed linear controls, test one curvature candidate per head with new FRESH views, and audit; no updates or automatic training.')
    d['joint_response_01'].update(native_curvature_diagnostic_config=config,native_curvature_diagnostic_config_sha256=digest,native_curvature_diagnostic_status='registered_not_started')
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=source_count,source_pins=len(p['source_pins']),science_started=False,prior_133_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for filename in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
    content=(previous/filename).read_text('utf-8')
    if filename=='PREPARE_DEPLOYMENT.py':
        content=content.replace('original_draw_precision_fix01/SOURCE_MANIFEST.json','original_precision_REAL_pilot_01/SOURCE_MANIFEST.json')
        content=content.replace("previous=base/'joint_response01_original_draw_precision_fix01_20261020'","previous=base/'joint_response01_original_precision_REAL_pilot_20261020'")
    content=content.replace('original_precision_REAL_pilot_01','native_curvature_diagnostic_01') if filename!='PREPARE_DEPLOYMENT.py' else content
    content=content.replace('joint_response_01_original_precision_REAL_pilot_20261020','joint_response_01_native_curvature_diagnostic_20261020')
    content=content.replace("root=base/'joint_response01_original_precision_REAL_pilot_20261020'","root=base/'joint_response01_native_curvature_diagnostic_20261020'")
    content=content.replace("root = Path('/home/rylan/projects/ej276d-n01/joint_response01_original_precision_REAL_pilot_20261020')","root = Path('/home/rylan/projects/ej276d-n01/joint_response01_native_curvature_diagnostic_20261020')")
    content=content.replace("root=Path('/home/rylan/projects/ej276d-n01/joint_response01_original_precision_REAL_pilot_20261020')","root=Path('/home/rylan/projects/ej276d-n01/joint_response01_native_curvature_diagnostic_20261020')")
    content=content.replace('run_joint_response01_original_precision_real','diagnose_joint01_native_curvature')
    content=content.replace('3272ef3806a6293d004a21859886d74d0de8f52137a4e0ae7c4907e64eed6445',digest)
    content=content.replace('== 819',f'== {source_count}').replace('==819',f'=={source_count}')
    content=content.replace("'5700s'","'480s'").replace('384*1024**2','64*1024**2').replace('original_precision_REAL_pilot_terminal_01.tar','native_curvature_diagnostic_terminal_01.tar')
    content=content.replace('n01/joint-response01-original-precision-real-pilot','n01/joint-response01-native-curvature-diagnostic')
    if filename=='REMOTE_STARTUP_REQUEST.py':content=content[:content.index("result['tracks']={}")]+"print(json.dumps(result))\n"
    ast.parse(content);(here/filename).write_text(content,encoding='utf-8',newline='\n')
print(json.dumps(record))
