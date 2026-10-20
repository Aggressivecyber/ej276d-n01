from datetime import datetime,timezone
from pathlib import Path
import ast,copy,difflib,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent;previous=here.parent/'curvature_update_diagnostic_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes());terminal=json.loads((previous/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes());diagnosis=json.loads((previous/'SECOND_CURVATURE_DIAGNOSIS.json').read_bytes())
assert review['status']=='PASS_SAVED_CURVATURE_UPDATE_DIAGNOSTIC_AUDIT' and not review['two_seed_curvature_update_all_local_pass'] and not terminal['processes'] and diagnosis['both_candidates_certified']
before=here/'before_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat();old=(root/'scripts/diagnose_joint01_curvature_update.py').read_text('utf-8');script=old
script=script.replace('Fixed original-joint candidate with one mixed-curvature correction; no updates.','Fixed original-joint candidate with two FIT mixed-curvature corrections; no updates.')
script=script.replace('curvature_control','curvature_update_control').replace("control['curvature_anchor']","control['curvature_update']").replace("control['curvature_anchor_'+label","control['curvature_update_'+label")
script=script.replace("curvature_update=torch.as_tensor(proposals[f'seed{seed}_curvature_update']","second_curvature=torch.as_tensor(proposals[f'seed{seed}_second_curvature']").replace("v['curvature_update']","v['second_curvature']")
script=script.replace('two_seed_curvature_update_all_local_pass','two_seed_second_curvature_all_local_pass').replace('CURVATURE_UPDATE.npz','SECOND_CURVATURE.npz').replace('CURVATURE_UPDATE.json','SECOND_CURVATURE.json').replace('curvature_update_fixed_candidate_checks','second_curvature_fixed_candidate_checks').replace('curvature_update_setup_loaded','second_curvature_setup_loaded').replace('curvature-update diagnostic','second-curvature diagnostic')
name='scripts/diagnose_joint01_second_curvature.py';ast.parse(script);(root/name).write_text(script,encoding='utf-8',newline='\n');(here/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),script.splitlines(True))),encoding='utf-8')
old_config='configs/joint_response_01_curvature_update_diagnostic_20261020.json';config='configs/joint_response_01_second_curvature_diagnostic_20261020.json';contract='docs/JOINT_RESPONSE_01_SECOND_CURVATURE_DIAGNOSTIC_20261020.md';p=json.loads((root/old_config).read_bytes());inputs={};controls=[]
for seed in range(2):
 file=f'diagnostic_inputs/SECOND_CURVATURE_CONTROL_seed{seed}.npz';inputs[file]=str(previous/f'terminal_01/collected/run_01/seed{seed}_CURVATURE_UPDATE.npz');controls.append(file)
fixed='diagnostic_inputs/SECOND_CURVATURE_FIXED_PROPOSALS.npz';inputs[fixed]=str(previous/'CERTIFIED_SECOND_CURVATURE_PROPOSALS.npz');assert hashlib.sha256(Path(inputs[fixed]).read_bytes()).hexdigest()==diagnosis['proposals_sha256']
for leaf in ['terminal_01/TERMINAL_REVIEW.json','SECOND_CURVATURE_DIAGNOSIS.json','DIAGNOSE_SECOND_CURVATURE.py','seed0_SECOND_CURVATURE_SOLVER_EVIDENCE.npz','seed1_SECOND_CURVATURE_SOLVER_EVIDENCE.npz']:
 file=(previous/leaf).relative_to(root).as_posix();inputs[file]=str(root/file)
inputs[old_config]=str(root/old_config)
p.update(schema='n01/joint-response01-second-curvature-diagnostic/v1',job_id='JOINT_RESPONSE_01_SECOND_CURVATURE_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,previous_terminal_review=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),saved_diagnosis=(previous/'SECOND_CURVATURE_DIAGNOSIS.json').relative_to(root).as_posix(),fixed_proposals=fixed,control_arrays=controls,fresh_seed=202670300000)
p['feature_seeds']['FRESH']=2026103601;p['native_projection_seeds']['FRESH']=2026103602
old_streams=set()
for label in ('native_curvature','native_gain','scientific_progress','curvature_update'):
 q=json.loads((root/f'configs/joint_response_01_{label}_diagnostic_20261020.json').read_bytes())
 assert p['feature_seeds']['FRESH'] not in q['feature_seeds'].values() and p['native_projection_seeds']['FRESH'] not in q['native_projection_seeds'].values()
 for seed in range(2):old_streams.update(q['fresh_seed']+100000*seed+x for x in (100,200,1000,1001,1002))
new_streams=[p['fresh_seed']+100000*seed+x for seed in range(2) for x in (100,200,1000,1001,1002)];assert len(new_streams)==len(set(new_streams)) and not old_streams.intersection(new_streams)
for file in [name,contract,*inputs]:p['source_pins'][file]=hashlib.sha256(Path(inputs[file] if file in inputs else root/file).read_bytes()).hexdigest()
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest();(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{name,contract,config})
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==137 and ledger['active_scientific_jobs']==0 and old_jobs[-1]['status']=='completed_audited_local_gates_failed'
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_second_curvature_diagnostic_20261020',resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False));assert ledger['jobs'][:-1]==old_jobs
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=ledger if file=='jobs.json' else json.loads((root/file).read_bytes());assert d['active_scientific_jobs']==0
 d.update(status='JOINT_RESPONSE_01_second_curvature_diagnostic_registered',updated_UTC=now,pending_job_id=p['job_id'],current_stage='4_second_curvature_diagnostic',active_work_package='4_second_curvature_diagnostic',current_action='Third and last fixed candidate in the same-center PSD curvature sequence registered.',next_action='Fixed original hard-observation check; no third curvature correction, automatic training, or extension of prior calibration.')
 d['joint_response_01'].update(second_curvature_diagnostic_status='registered_not_started',second_curvature_diagnostic_config=config,second_curvature_diagnostic_config_sha256=digest)
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_137_jobs_unchanged=True,fresh_streams=new_streams,fresh_streams_disjoint_from_four_predecessors=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((previous/'REGISTRATION.json').read_bytes())['config_sha256']
for file in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
 content=(previous/file).read_text('utf-8')
 if file=='PREPARE_DEPLOYMENT.py':content=content.replace('scientific_progress_diagnostic_01/SOURCE_MANIFEST.json','curvature_update_diagnostic_01/SOURCE_MANIFEST.json').replace("previous=base/'joint_response01_scientific_progress_diagnostic_20261020'","previous=base/'joint_response01_curvature_update_diagnostic_20261020'")
 content=content.replace('joint_response_01_curvature_update_diagnostic_20261020','joint_response_01_second_curvature_diagnostic_20261020').replace("root=base/'joint_response01_curvature_update_diagnostic_20261020'","root=base/'joint_response01_second_curvature_diagnostic_20261020'").replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_curvature_update_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_second_curvature_diagnostic_20261020')")
 content=content.replace('diagnose_joint01_curvature_update','diagnose_joint01_second_curvature').replace(old_digest,digest).replace('== 858',f'== {sources}').replace('==858',f'=={sources}').replace('curvature_update_diagnostic_terminal_01.tar','second_curvature_diagnostic_terminal_01.tar').replace('n01/joint-response01-curvature-update-diagnostic','n01/joint-response01-second-curvature-diagnostic');ast.parse(content);(here/file).write_text(content,encoding='utf-8',newline='\n')
plan=root/'JOINT_RESPONSE_01_PLAN.md';plan.write_text(plan.read_text('utf-8')+'\n[第二次FIT曲率修正检查](docs/JOINT_RESPONSE_01_SECOND_CURVATURE_DIAGNOSTIC_20261020.md)已前瞻登记：同一中心的第三个固定曲率候选，两头仅用FIT保存误差得到Solved及数值证书，零更新；本序列不再自动增加第三个修正项。\n',encoding='utf-8',newline='\r\n')
print(json.dumps(record))
