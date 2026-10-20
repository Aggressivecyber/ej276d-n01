from datetime import datetime,timezone
from pathlib import Path
import ast,copy,hashlib,json,shutil
import numpy as np

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'stable_projection_diagnostic_01'
now=datetime.now(timezone.utc).isoformat();rel=HERE.relative_to(ROOT).as_posix()
review=json.loads((PREV/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
screen=json.loads((PREV/'RECENTERED_RESTORATION_SCREEN.json').read_bytes())
assert review['status']=='PASS_SAVED_STABLE_PROJECTION_DIAGNOSTIC_AUDIT' and not review['two_seed_stable_projection_all_local_pass']
assert screen['two_seed_screen_feasible'] and all(r['KKT_pass'] for r in screen['reports'])
for record in screen['reports']:
    assert hashlib.sha256(Path(record['evidence_NAS_path']).read_bytes()).hexdigest()==record['evidence_sha256']
before=HERE/'before_registered';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(ROOT/name,before/name)
manifest=json.loads((PREV/'SOURCE_MANIFEST.json').read_bytes())
inputs={name:str(ROOT/name) for name in manifest['files'] if name.startswith(('src/','scripts/')) or name=='configs/response038_20261017.json'}
driver='scripts/diagnose_joint01_recentered_derivatives.py';contract='docs/JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020.md'
inputs[driver]=str(ROOT/driver);inputs[contract]=str(ROOT/contract)
roles='diagnostic_inputs/REAL_CALIBRATION_ROLES.npz'
inputs[roles]=str(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz')
saved=[]
for seed in range(2):
    name=f'diagnostic_inputs/RECENTERED_seed{seed}_STABLE_PROJECTION.npz'
    inputs[name]=str(PREV/f'terminal_01/collected/run_01/seed{seed}_STABLE_PROJECTION.npz');saved.append(name)
    with np.load(inputs[name],allow_pickle=False) as z:
        assert z['curvature_update_control'].shape==(54,) and np.isfinite(z['curvature_update_control']).all()
        assert np.all(z['curvature_update_control']>=z['lower']) and np.all(z['curvature_update_control']<=z['upper'])
        for label in ('FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3'):
            assert z[label+'_source_parents'].shape==(1024,) and z[label+'_real_rows'].shape==(4096,)
for leaf in ('terminal_01/TERMINAL_REVIEW.json','PAIRED_PROJECTION_DIAGNOSIS.json','RECENTERED_RESTORATION_SCREEN_DESIGN.json','RECENTERED_RESTORATION_SCREEN.json'):
    path=PREV/leaf;inputs[path.relative_to(ROOT).as_posix()]=str(path)
pins={}
for name,path in inputs.items():
    raw=Path(path).read_bytes();pins[name]=hashlib.sha256(raw).hexdigest()
    if name in manifest['files']:assert pins[name]==manifest['files'][name]['sha256']
    if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
base=json.loads((ROOT/'configs/joint_response_01_stable_projection_diagnostic_20261020.json').read_bytes())
budget=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,old_A_calls=52,old_A_rows=6593,
    M0_head_calls=104,M0_head_rows=13186,REAL_observable_replay_calls=1,REAL_observable_replay_rows=21047,
    feature_maps_constructed=3,new_native_projection_contexts=3,REAL_fit_feature_rows=57344,REAL_CHECK_feature_rows=0,
    allocated_banks=14,noise005_calls=14,noise005_rows=458752,extension_rows=1376256,shape_standard_normal_values=3670016,
    joint_response_calls=1512,joint_response_rows=49545216,generated_observation_calls=4536,
    generated_observation_rows=148635648,native_component_rows=167215104,finite_difference_columns=742,
    saved_previous_center_replays_verified=14,saved_new_center_replays_verified=14,saved_initial_anchors_reused=14,
    fixed_control_checks=14,joint_direction_solves=0,new_parameter_candidates=0,REAL_response_updates=0,
    DEV_waveforms_decoded=0,AD_queries=0,classifier_fits=0)
p=dict(schema='n01/joint-response01-recentered-derivative-diagnostic/v1',
    job_id='JOINT_RESPONSE_01_RECENTERED_DERIVATIVE_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,
    frozen_setup_config=base['frozen_setup_config'],roles_file=roles,batch=base['batch'],solver=base['solver'],
    native_guard=base['native_guard'],success=base['success'],saved_arrays=saved,
    feature_seeds={k:base['feature_seeds'][k] for k in ('FIT0','FIT1','FIT2')},
    native_projection_seeds={k:base['native_projection_seeds'][k] for k in ('FIT0','FIT1','FIT2')},
    finite_difference_step=.004,response_coordinates=53,noise_correction=0,
    prospective_restoration=dict(radius=.003125,native_linear_cap_factor=1.005,original_native_execution_factor=1.01,
        objective='minimum Euclidean correction norm',maximum_future_local_solves=2,new_generator_calls_in_saved_matrix_solve=0,
        solver_settings=dict(max_iterations=100,time_limit_seconds=30,tolerance_feasibility=1e-10,tolerance_gap=1e-10)),
    resources=dict(CPU_threads=4,GPU_GiB=4,RSS_GiB=12,internal_seconds=3600,hard_wall_seconds=3900,
        output_MiB=64,GPU_free_required_MiB=6144),operation_budget=budget,source_pins=pins,
    automatic_restart=False,automatic_training=False,committed_parameter_updates=0,
    storage=dict(remote_root='/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020',
        windows_root='Z:/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020',
        large_local_or_compute_disk_backups=False,existing_data_migration_performed=False))
config='configs/joint_response_01_recentered_derivative_diagnostic_20261020.json'
(ROOT/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
digest=hashlib.sha256((ROOT/config).read_bytes()).hexdigest()
inputs[config]=str(ROOT/config)
(HERE/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
records={name:dict(bytes=Path(path).stat().st_size,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for name,path in inputs.items()}
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(schema='n01/recentered-derivative-source/v1',config_sha256=digest,files=records),indent=2)+'\n',encoding='utf-8')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=json.loads((ROOT/name).read_bytes());assert d['active_scientific_jobs']==0 and d['followup']['count']==0
    if name=='jobs.json':
        old=copy.deepcopy(d['jobs']);assert len(old)==140
        d['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,
            config_sha256=digest,remote_root=p['storage']['remote_root'],operation_budget=budget,resources=p['resources'],
            automatic_restart=False,automatic_training=False,parameter_updates=0,scientific_acceptance=False))
        assert d['jobs'][:-1]==old
    d.update(status='JOINT_RESPONSE_01_recentered_derivative_registered',updated_UTC=now,pending_job_id=p['job_id'],
        current_stage='4_recentered_derivative_diagnostic',active_work_package='4_recentered_derivative_diagnostic',
        current_action='Fresh derivatives at the fixed measured control registered; no correction candidate or update.',
        next_action='Verify NAS deployment and start one bounded derivative collection; estimate one wake from measured progress.')
    d['joint_response_01'].update(recentered_derivative_status='registered_not_started',recentered_derivative_config=config,
        recentered_derivative_config_sha256=digest,recentered_derivative_remote_root=p['storage']['remote_root'])
    (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files=len(records),source_pins=len(pins),source_bytes=sum(v['bytes'] for v in records.values()),
    prior_140_jobs_unchanged=True,science_started=False,storage=p['storage'])
(HERE/'REGISTRATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
