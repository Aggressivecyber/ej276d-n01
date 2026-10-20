from datetime import datetime,timezone
from pathlib import Path
import ast,copy,difflib,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent;previous=here.parent/'scientific_progress_diagnostic_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes());terminal=json.loads((previous/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_SCIENTIFIC_PROGRESS_DIAGNOSTIC_AUDIT' and not review['two_seed_scientific_progress_all_local_pass'] and not terminal['processes']
certificate=json.loads((previous/'CONIC_CANDIDATE_CERTIFICATE.json').read_bytes())
assert certificate['status']=='PASS_INDEPENDENT_CONVEX_OPTIMALITY_CERTIFICATES' and all(v['independent_certificate_pass'] for v in certificate['entries'])
before=here/'before_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat()
old=(root/'scripts/diagnose_joint01_native_gain.py').read_text('utf-8');script=old
script=script.replace('Fixed common-native-gain candidate versus prior curvature candidate; no updates.','Fixed original-joint candidate with one mixed-curvature correction; no updates.')
script=script.replace("native_gain=torch.as_tensor(proposals[f'seed{seed}_rankone0'],device=device)","curvature_update=torch.as_tensor(proposals[f'seed{seed}_curvature_update'],device=device)")
script=script.replace("v['native_gain']","v['curvature_update']").replace('two_seed_native_gain_all_local_pass','two_seed_curvature_update_all_local_pass').replace('NATIVE_GAIN.npz','CURVATURE_UPDATE.npz').replace('NATIVE_GAIN.json','CURVATURE_UPDATE.json')
script=script.replace('native_gain_fixed_candidate_checks','curvature_update_fixed_candidate_checks').replace('native_gain_setup_loaded','curvature_update_setup_loaded').replace('native-gain diagnostic','curvature-update diagnostic')
name='scripts/diagnose_joint01_curvature_update.py';ast.parse(script);(root/name).write_text(script,encoding='utf-8',newline='\n')
(here/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),script.splitlines(True))),encoding='utf-8')
config='configs/joint_response_01_curvature_update_diagnostic_20261020.json';contract='docs/JOINT_RESPONSE_01_CURVATURE_UPDATE_DIAGNOSTIC_20261020.md'
old_config='configs/joint_response_01_scientific_progress_diagnostic_20261020.json';p=json.loads((root/old_config).read_bytes());inputs={}
fixed='diagnostic_inputs/CURVATURE_UPDATE_FIXED_PROPOSALS.npz';inputs[fixed]=str(previous/'CERTIFIED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz')
assert hashlib.sha256(Path(inputs[fixed]).read_bytes()).hexdigest()==certificate['proposals_sha256']
for leaf in ['terminal_01/TERMINAL_REVIEW.json','CONIC_CANDIDATE_CERTIFICATE.json','CONIC_INDEPENDENT_KKT_CHECK.json','SAVED_REDUCED_CONIC_CURVATURE_UPDATE_DIAGNOSIS.json','DIAGNOSE_REDUCED_CONIC_CURVATURE_UPDATE.py','RNG_CROSS_STUDY_AUDIT.json','seed0_REDUCED_CONIC_SOLVER_EVIDENCE.npz','seed1_REDUCED_CONIC_SOLVER_EVIDENCE.npz']:
 file=(previous/leaf).relative_to(root).as_posix();inputs[file]=str(root/file)
inputs[old_config]=str(root/old_config)
gain=json.loads((root/'configs/joint_response_01_native_gain_diagnostic_20261020.json').read_bytes())
p.update(schema='n01/joint-response01-curvature-update-diagnostic/v1',job_id='JOINT_RESPONSE_01_CURVATURE_UPDATE_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,previous_terminal_review=(previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root).as_posix(),saved_diagnosis=(previous/'CONIC_CANDIDATE_CERTIFICATE.json').relative_to(root).as_posix(),control_arrays=gain['control_arrays'],fixed_proposals=fixed,fresh_seed=202670100000)
p['feature_seeds']['FRESH']=2026103501;p['native_projection_seeds']['FRESH']=2026103502
old_streams=set()
for label in ('native_curvature','native_gain','scientific_progress'):
 q=json.loads((root/f'configs/joint_response_01_{label}_diagnostic_20261020.json').read_bytes())
 for seed in range(2):
  old_streams.update(q['fresh_seed']+100000*seed+x for x in (100,200,1000,1001,1002))
new_streams=[p['fresh_seed']+100000*seed+x for seed in range(2) for x in (100,200,1000,1001,1002)]
assert len(new_streams)==len(set(new_streams)) and not old_streams.intersection(new_streams)
for file in [name,contract,*inputs]:p['source_pins'][file]=hashlib.sha256(Path(inputs[file] if file in inputs else root/file).read_bytes()).hexdigest()
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{name,contract,config})
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==136 and old_jobs[-1]['status']=='completed_audited_local_gates_failed' and ledger['active_scientific_jobs']==0
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_curvature_update_diagnostic_20261020',resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False))
assert ledger['jobs'][:-1]==old_jobs
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=ledger if file=='jobs.json' else json.loads((root/file).read_bytes());assert d['active_scientific_jobs']==0
 d.update(status='JOINT_RESPONSE_01_curvature_update_diagnostic_registered',updated_UTC=now,pending_job_id=p['job_id'],current_stage='4_curvature_update_diagnostic',active_work_package='4_curvature_update_diagnostic',current_action='Original full-joint direction with one FIT mixed-curvature correction and independent conic certificate registered.',next_action='Check two fixed candidates under original gates with disjoint FRESH streams; no updates or automatic training.')
 d['joint_response_01'].update(curvature_update_diagnostic_status='registered_not_started',curvature_update_diagnostic_config=config,curvature_update_diagnostic_config_sha256=digest,cross_study_rng_overlap_record=(previous/'RNG_CROSS_STUDY_AUDIT.json').relative_to(root).as_posix())
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_136_jobs_unchanged=True,fresh_streams=new_streams,fresh_streams_disjoint_from_three_predecessors=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((previous/'REGISTRATION.json').read_bytes())['config_sha256']
for file in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
 content=(previous/file).read_text('utf-8')
 if file=='PREPARE_DEPLOYMENT.py':
  content=content.replace('native_gain_diagnostic_01/SOURCE_MANIFEST.json','scientific_progress_diagnostic_01/SOURCE_MANIFEST.json').replace("previous=base/'joint_response01_native_gain_diagnostic_20261020'","previous=base/'joint_response01_scientific_progress_diagnostic_20261020'")
 content=content.replace('joint_response_01_scientific_progress_diagnostic_20261020','joint_response_01_curvature_update_diagnostic_20261020')
 content=content.replace("root=base/'joint_response01_scientific_progress_diagnostic_20261020'","root=base/'joint_response01_curvature_update_diagnostic_20261020'")
 content=content.replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_scientific_progress_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_curvature_update_diagnostic_20261020')")
 content=content.replace('diagnose_joint01_scientific_progress','diagnose_joint01_curvature_update').replace(old_digest,digest).replace('== 846',f'== {sources}').replace('==846',f'=={sources}').replace('scientific_progress_diagnostic_terminal_01.tar','curvature_update_diagnostic_terminal_01.tar').replace('n01/joint-response01-scientific-progress-diagnostic','n01/joint-response01-curvature-update-diagnostic')
 ast.parse(content);(here/file).write_text(content,encoding='utf-8',newline='\n')
plan=root/'JOINT_RESPONSE_01_PLAN.md'
paragraph='''
## 2026-10-20 共同进度负结果、数值求解与随机流复核

[共同进度独立审核](receipts/joint_response_01_20261020/scientific_progress_diagnostic_01/terminal_01/TERMINAL_REVIEW.json)确认846源/49pins/13输出、exit0、75.18秒、40生成调用、零更新；seed0五库全过，seed1 CHECK joint剩余0.9930517，按原1%门失败。两头FRESH joint剩余0.977546/0.967419；native wave最大1.001134/0.997828，QPSD最大1.007563/0.996013。原科学门尚未通过。

[固定方向新检查](docs/JOINT_RESPONSE_01_CURVATURE_UPDATE_DIAGNOSTIC_20261020.md)回到原联合目标，仅用第一项曲率候选FIT误差加入一次非负秩一修正。原SLSQP seed0未收敛的记录保持。同一凸问题的锥化候选均通过独立原始/对偶与原约束证书；seed1原始AlmostSolved状态保留。只登记固定原生检查，不直接训练。

[随机流复核](receipts/joint_response_01_20261020/scientific_progress_diagnostic_01/RNG_CROSS_STUDY_AUDIT.json)发现前三项相邻诊断跨头重复部分FRESH流，同一头相对自身前次不重复。不能声称跨诊断跨头完全独立；历史负结果和拟合角色范围保持。新检查改用已验证不重叠的202670100000/202670200000流与新投影。
'''
plan.write_text(plan.read_text('utf-8')+paragraph,encoding='utf-8',newline='\r\n')
readme=root/'README.md';text=readme.read_text('utf-8');boundary=text.index('以下保留历史快照；')
text='''# N01 当前阶段：阶段4曲率修正固定方向检查

> 当前执行状态（2026-10-20）：三轮REAL校准及三项后续诊断已闭合审核，最终科学门和两头局部门未全过。新固定方向已前瞻登记，当前0科学作业、0定时；REAL响应匹配尚未建立。

[原三轮校准审核](receipts/joint_response_01_20261020/original_precision_REAL_pilot_01/terminal_01/TERMINAL_REVIEW.json)确认600差分列、20联合解、80候选接受检查和192最终门复算完成，两神经各24/96改善门失败。后续曲率、共同原生增益及共同科学进度检查均保留负结果。

[新固定方向协议](docs/JOINT_RESPONSE_01_CURVATURE_UPDATE_DIAGNOSTIC_20261020.md)保留原联合目标和原门，只使用已保存FIT混合曲率并核验两头数值证书；下一步执行原硬读出固定候选检查，零更新。随机流跨诊断跨头复用已记录，新检查改用无重叠流。教师、物理比较、历史负结果和阶段5待验状态保持。

'''+text[boundary:]
readme.write_text(text,encoding='utf-8',newline='\r\n')
print(json.dumps(record))
