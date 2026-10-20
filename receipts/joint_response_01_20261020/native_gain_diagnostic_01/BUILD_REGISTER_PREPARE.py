from datetime import datetime,timezone
from pathlib import Path
import ast,copy,difflib,hashlib,json,shutil
root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent;previous=here.parent/'native_curvature_diagnostic_01'
old_script=(root/'scripts/diagnose_joint01_native_curvature.py').read_text('utf-8')
script=old_script.replace('Fixed saved curvature/anchor candidate versus original linear candidate; no updates.','Fixed common-native-gain candidate versus prior curvature candidate; no updates.')
script=script.replace("candidates=dict(linear_control=torch.as_tensor(saved['trial0_proposal'],device=device),\n            curvature_anchor=torch.as_tensor(proposals[f'seed{seed}_joint_ITER02_curvature1_anchor1'],device=device))", "with np.load(ROOT/p['control_arrays'][seed],allow_pickle=False) as z:control={k:z[k].copy() for k in z.files}\n        candidates=dict(curvature_control=torch.as_tensor(control['curvature_anchor'],device=device),\n            native_gain=torch.as_tensor(proposals[f'seed{seed}_rankone0'],device=device))")
script=script.replace("name=='linear_control'","name=='curvature_control'").replace("saved['trial0_'+label+'_actual_'+suffix]","control['curvature_anchor_'+label+'_actual_'+suffix]")
script=script.replace('saved_linear_candidate_replays_verified','saved_control_candidate_replays_verified').replace('two_seed_curvature_all_local_pass','two_seed_native_gain_all_local_pass').replace("v['curvature_anchor']","v['native_gain']")
script=script.replace('native_curvature_fixed_candidate_checks','native_gain_fixed_candidate_checks').replace('native_curvature_setup_loaded','native_gain_setup_loaded').replace('CURVATURE.npz','NATIVE_GAIN.npz').replace('CURVATURE.json','NATIVE_GAIN.json')
script=script.replace('Registered curvature diagnostic config changed','Registered native-gain diagnostic config changed').replace('Pinned curvature diagnostic input changed','Pinned native-gain diagnostic input changed')
assert script!=old_script and "control['curvature_anchor_'+label+'_actual_'+suffix]" in script
name='scripts/diagnose_joint01_native_gain.py';(root/name).write_text(script,encoding='utf-8',newline='\n');ast.parse(script)
(here/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(old_script.splitlines(True),script.splitlines(True),fromfile='native_curvature',tofile='native_gain')),encoding='utf-8')
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes());assert not review['two_seed_curvature_all_local_pass'] and review['committed_parameter_updates']==0
diagnosis=json.loads((previous/'SAVED_NATIVE_GAIN_DIAGNOSIS.json').read_bytes());assert all(x['success'] for x in diagnosis['reports'] if not x['FIT_positive_rank_one_correction'])
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==134 and ledger['active_scientific_jobs']==0
before=here/'before_registration';before.mkdir()
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/file,before/file)
old_config='configs/joint_response_01_native_curvature_diagnostic_20261020.json';p=json.loads((root/old_config).read_bytes());now=datetime.now(timezone.utc).isoformat()
contract='docs/JOINT_RESPONSE_01_NATIVE_GAIN_DIAGNOSTIC_20261020.md';config='configs/joint_response_01_native_gain_diagnostic_20261020.json'
inputs={};controls=[]
for seed in range(2):
 target=f'diagnostic_inputs/NATIVE_GAIN_CONTROL_seed{seed}.npz';inputs[target]=str(previous/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz');controls.append(target)
fixed='diagnostic_inputs/NATIVE_GAIN_FIXED_PROPOSALS.npz';inputs[fixed]=str(previous/'SAVED_NATIVE_GAIN_PROPOSALS.npz')
review_path=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix();diagnosis_path=(previous/'SAVED_NATIVE_GAIN_DIAGNOSIS.json').relative_to(root).as_posix()
for file in [review_path,diagnosis_path,(previous/'DIAGNOSE_NATIVE_GAIN.py').relative_to(root).as_posix(),old_config]:inputs[file]=str(root/file)
p.update(schema='n01/joint-response01-native-gain-diagnostic/v1',job_id='JOINT_RESPONSE_01_NATIVE_GAIN_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,
    previous_terminal_review=review_path,saved_diagnosis=diagnosis_path,control_arrays=controls,fixed_proposals=fixed,fresh_seed=202660700000)
p['feature_seeds']['FRESH']=2026103301;p['native_projection_seeds']['FRESH']=2026103302
p['operation_budget']['saved_control_candidate_replays_verified']=p['operation_budget'].pop('saved_linear_candidate_replays_verified')
for file in [name,contract,*inputs]:p['source_pins'][file]=hashlib.sha256(Path(inputs[file] if file in inputs else root/file).read_bytes()).hexdigest()
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{name,contract,config})
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_native_gain_diagnostic_20261020',resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False))
assert ledger['jobs'][:-1]==old_jobs
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=ledger if file=='jobs.json' else json.loads((root/file).read_bytes())
 d.update(status='JOINT_RESPONSE_01_native_gain_diagnostic_registered',updated_UTC=now,active_scientific_jobs=0,pending_job_id=p['job_id'],current_stage='4_native_gain_diagnostic',active_work_package='4_native_gain_diagnostic',
    current_action='One common-native-gain candidate per head registered from FIT-only saved matrices.',next_action='Compare fixed gain and curvature candidates on replayed FIT/CHECK and new FRESH; no updates or automatic training.')
 d['joint_response_01'].update(native_gain_diagnostic_status='registered_not_started',native_gain_diagnostic_config=config,native_gain_diagnostic_config_sha256=digest)
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_134_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((previous/'REGISTRATION.json').read_bytes())['config_sha256']
for file in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
 content=(previous/file).read_text('utf-8')
 if file=='PREPARE_DEPLOYMENT.py':
  content=content.replace('original_precision_REAL_pilot_01/SOURCE_MANIFEST.json','native_curvature_diagnostic_01/SOURCE_MANIFEST.json')
  content=content.replace("previous=base/'joint_response01_original_precision_REAL_pilot_20261020'","previous=base/'joint_response01_native_curvature_diagnostic_20261020'")
 content=content.replace('joint_response_01_native_curvature_diagnostic_20261020','joint_response_01_native_gain_diagnostic_20261020')
 content=content.replace("root=base/'joint_response01_native_curvature_diagnostic_20261020'","root=base/'joint_response01_native_gain_diagnostic_20261020'")
 content=content.replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_native_curvature_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_native_gain_diagnostic_20261020')")
 content=content.replace('diagnose_joint01_native_curvature','diagnose_joint01_native_gain').replace(old_digest,digest).replace('== 828',f'== {sources}').replace('==828',f'=={sources}')
 content=content.replace('native_curvature_diagnostic_terminal_01.tar','native_gain_diagnostic_terminal_01.tar').replace('n01/joint-response01-native-curvature-diagnostic','n01/joint-response01-native-gain-diagnostic')
 ast.parse(content);(here/file).write_text(content,encoding='utf-8',newline='\n')
print(json.dumps(record))
