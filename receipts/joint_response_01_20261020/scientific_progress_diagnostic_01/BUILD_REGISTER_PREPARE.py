from datetime import datetime,timezone
from pathlib import Path
import ast,copy,difflib,hashlib,json,shutil
root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent;previous=here.parent/'native_gain_diagnostic_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes());terminal=json.loads((previous/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_NATIVE_GAIN_DIAGNOSTIC_AUDIT' and not review['two_seed_native_gain_all_local_pass'] and not terminal['processes']
before=here/'before_close_and_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat()
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=json.loads((root/file).read_bytes())
 if file=='jobs.json':
  old_jobs=copy.deepcopy(d['jobs']);assert len(old_jobs)==135 and old_jobs[-1]['job_id']=='JOINT_RESPONSE_01_NATIVE_GAIN_DIAGNOSTIC_20261020'
  d['jobs'][-1].update(status='completed_audited_local_gates_failed',native_exit_code=0,ended_UTC=review['native_exit']['ended_utc'],execution_counts=review['execution_counts'],terminal_review=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),two_seed_local_pass=False,parameter_updates=0)
  assert d['jobs'][:-1]==old_jobs[:-1]
 d.update(active_scientific_jobs=0,active_job_id=None,active_run_id=None,pending_job_id=None,training_pid=None,supervisor_pid=None,current_job_native_exit_code=0,current_job_ended_UTC=review['native_exit']['ended_utc'],last_verified_UTC=terminal['UTC'],execution_state_asof_UTC=terminal['UTC'])
 d['joint_response_01'].update(native_gain_diagnostic_status='completed_audited_local_gates_failed',native_gain_diagnostic_completed=True,native_gain_diagnostic_terminal_review=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),native_gain_diagnostic_execution_counts=review['execution_counts'])
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
(previous/'CLOSURE_RECEIPT.json').write_text(json.dumps(dict(status='CLOSED_AUDITED_NEGATIVE',UTC=now,prior_134_jobs_unchanged=True,active_jobs=0,automations=0,scientific_acceptance=False),indent=2)+'\n',encoding='utf-8')
plan=root/'JOINT_RESPONSE_01_PLAN.md'
paragraph='''
## 2026-10-20 原生共同增益诊断闭合

[独立审核](receipts/joint_response_01_20261020/native_gain_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)确认837源/40pins/13输出、exit0、38.57秒、40调用及2个完整求解/20接受检查通过，零更新。seed0新候选五库全过；seed1只有CHECK联合剩余0.9922715，超过0.99，按原门失败。两头FRESH联合剩余0.982307/0.982824，native wave最大1.001429/0.996884、QPSD最大0.992584/0.987276。实际native改善与联合守门的冲突保留，不自动训练。

[保存矩阵共同进度诊断](receipts/joint_response_01_20261020/native_gain_diagnostic_01/SAVED_SCIENTIFIC_PROGRESS_DIAGNOSIS.json)使用两条已测FIT方向的非负秩一曲率，并以原joint/peak/native改善比例0.20/0.10/0.05共同求解，两头在200步内得到正进度。下一项前瞻固定两个新方向，仅进行原硬读出与新FRESH比较；所有历史失败、原科学门和阶段5待验状态保持。
'''
plan.write_text(plan.read_text('utf-8')+paragraph,encoding='utf-8',newline='\r\n')
old=(root/'scripts/diagnose_joint01_native_gain.py').read_text('utf-8');script=old
script=script.replace('Fixed common-native-gain candidate versus prior curvature candidate; no updates.','Fixed goal-scaled joint/peak/native candidate versus native-gain control; no updates.')
script=script.replace('curvature_control','native_gain_control').replace("control['curvature_anchor']","control['native_gain']").replace("control['curvature_anchor_'+label+'_actual_'+suffix]","control['native_gain_'+label+'_actual_'+suffix]")
script=script.replace("native_gain=torch.as_tensor(proposals[f'seed{seed}_rankone0'],device=device)","scientific_progress=torch.as_tensor(proposals[f'seed{seed}_scientific_progress'],device=device)")
script=script.replace("v['native_gain']","v['scientific_progress']").replace('two_seed_native_gain_all_local_pass','two_seed_scientific_progress_all_local_pass').replace('NATIVE_GAIN.npz','SCIENTIFIC_PROGRESS.npz').replace('NATIVE_GAIN.json','SCIENTIFIC_PROGRESS.json')
script=script.replace('native_gain_fixed_candidate_checks','scientific_progress_fixed_candidate_checks').replace('native_gain_setup_loaded','scientific_progress_setup_loaded').replace('native-gain diagnostic','scientific-progress diagnostic')
name='scripts/diagnose_joint01_scientific_progress.py';ast.parse(script);(root/name).write_text(script,encoding='utf-8',newline='\n')
(here/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),script.splitlines(True))),encoding='utf-8')
diagnosis=json.loads((previous/'SAVED_SCIENTIFIC_PROGRESS_DIAGNOSIS.json').read_bytes());assert diagnosis['two_positive_solved_directions']
config='configs/joint_response_01_scientific_progress_diagnostic_20261020.json';contract='docs/JOINT_RESPONSE_01_SCIENTIFIC_PROGRESS_DIAGNOSTIC_20261020.md'
old_config='configs/joint_response_01_native_gain_diagnostic_20261020.json';p=json.loads((root/old_config).read_bytes());inputs={};controls=[]
for seed in range(2):
 target=f'diagnostic_inputs/SCIENTIFIC_PROGRESS_CONTROL_seed{seed}.npz';inputs[target]=str(previous/f'terminal_01/collected/run_01/seed{seed}_NATIVE_GAIN.npz');controls.append(target)
fixed='diagnostic_inputs/SCIENTIFIC_PROGRESS_FIXED_PROPOSALS.npz';inputs[fixed]=str(previous/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz')
rp=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix();dp=(previous/'SAVED_SCIENTIFIC_PROGRESS_DIAGNOSIS.json').relative_to(root).as_posix()
for file in [rp,dp,(previous/'DIAGNOSE_SCIENTIFIC_PROGRESS.py').relative_to(root).as_posix(),old_config]:inputs[file]=str(root/file)
p.update(schema='n01/joint-response01-scientific-progress-diagnostic/v1',job_id='JOINT_RESPONSE_01_SCIENTIFIC_PROGRESS_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,previous_terminal_review=rp,saved_diagnosis=dp,control_arrays=controls,fixed_proposals=fixed,fresh_seed=202660800000)
p['feature_seeds']['FRESH']=2026103401;p['native_projection_seeds']['FRESH']=2026103402
for file in [name,contract,*inputs]:p['source_pins'][file]=hashlib.sha256(Path(inputs[file] if file in inputs else root/file).read_bytes()).hexdigest()
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{name,contract,config})
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==135
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_scientific_progress_diagnostic_20261020',resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False))
assert ledger['jobs'][:-1]==old_jobs
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=ledger if file=='jobs.json' else json.loads((root/file).read_bytes())
 d.update(status='JOINT_RESPONSE_01_scientific_progress_diagnostic_registered',updated_UTC=now,pending_job_id=p['job_id'],current_stage='4_scientific_progress_diagnostic',active_work_package='4_scientific_progress_diagnostic',current_action='FIT-only goal-scaled progress direction with two measured mixed-curvature terms registered.',next_action='Compare fixed candidates under original gates and new FRESH; no updates or automatic training.')
 d['joint_response_01'].update(scientific_progress_diagnostic_status='registered_not_started',scientific_progress_diagnostic_config=config,scientific_progress_diagnostic_config_sha256=digest)
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_135_jobs_unchanged_after_previous_close=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((previous/'REGISTRATION.json').read_bytes())['config_sha256']
for file in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
 content=(previous/file).read_text('utf-8')
 if file=='PREPARE_DEPLOYMENT.py':
  content=content.replace('native_curvature_diagnostic_01/SOURCE_MANIFEST.json','native_gain_diagnostic_01/SOURCE_MANIFEST.json').replace("previous=base/'joint_response01_native_curvature_diagnostic_20261020'","previous=base/'joint_response01_native_gain_diagnostic_20261020'")
 content=content.replace('joint_response_01_native_gain_diagnostic_20261020','joint_response_01_scientific_progress_diagnostic_20261020')
 content=content.replace("root=base/'joint_response01_native_gain_diagnostic_20261020'","root=base/'joint_response01_scientific_progress_diagnostic_20261020'")
 content=content.replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_native_gain_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_scientific_progress_diagnostic_20261020')")
 content=content.replace('diagnose_joint01_native_gain','diagnose_joint01_scientific_progress').replace(old_digest,digest).replace('== 837',f'== {sources}').replace('==837',f'=={sources}').replace('native_gain_diagnostic_terminal_01.tar','scientific_progress_diagnostic_terminal_01.tar').replace('n01/joint-response01-native-gain-diagnostic','n01/joint-response01-scientific-progress-diagnostic')
 ast.parse(content);(here/file).write_text(content,encoding='utf-8',newline='\n')
print(json.dumps(record))
