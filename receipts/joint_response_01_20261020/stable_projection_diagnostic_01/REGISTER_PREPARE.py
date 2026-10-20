from datetime import datetime,timezone
from pathlib import Path
import ast,copy,difflib,hashlib,json,shutil

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'random_precision_diagnostic_01';T=HERE.parent/'second_curvature_diagnostic_01'
review=json.loads((PREV/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status']=='PASS_SAVED_RANDOM_PRECISION_DIAGNOSTIC_AUDIT' and not review['two_head_precision_gate_pass']
fixed=json.loads((PREV/'STABLE_SUBSPACE_AUDIT_AND_PROJECTION.json').read_bytes())
assert fixed['checks'][0]['certificate']['status']=='INDEPENDENT_DUAL_INFEASIBILITY_CERTIFICATE'
before=HERE/'before_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(ROOT/name,before/name)
now=datetime.now(timezone.utc).isoformat()
old=(ROOT/'scripts/diagnose_joint01_second_curvature.py').read_text(encoding='utf-8');s=old
s=s.replace('Fixed original-joint candidate with two FIT mixed-curvature corrections; no updates.','Fixed orthogonal projection of the one-rank control step; no optimization or updates.')
s=s.replace("LABELS=('FIT0','FIT1','FIT2','CHECK','FRESH')","MAP_LABELS=('FIT0','FIT1','FIT2','CHECK','FRESH')\nLABELS=('FIT0','FIT1','FIT2','CHECK','IID0','IID1','IID2','IID3','FRESH')")
s=s.replace('for label in LABELS:\n        fmap=','for label in MAP_LABELS:\n        fmap=')
s=s.replace('for label in LABELS},replay_errors={})','for label in MAP_LABELS},replay_errors={})')
s=s.replace("second_curvature=torch.as_tensor(proposals[f'seed{seed}_second_curvature']","stable_projection=torch.as_tensor(proposals[f'seed{seed}_stable_projection']")
s=s.replace("        for label in LABELS:\n            check_budget", "        for label in LABELS:\n            map_label='FIT0' if label.startswith('IID') else label\n            check_budget")
s=s.replace("rs=p['fresh_seed']+100000*seed","rs=p['fresh_seed']+p['fresh_head_stride']*seed")
old_block="""            else:
                parents=torch.as_tensor(saved[label+'_source_parents'],device=device);rows=torch.as_tensor(saved[label+'_real_rows'],device=device)
                bank_seed=int(saved[label+'_random_seed'])
                np.testing.assert_array_equal(maps[label][HYPOTHESES[0]].cholesky.cpu().numpy(),saved[label+'_cholesky'])"""
new_block="""            elif label.startswith('IID'):
                with np.load(ROOT/p['random_precision_replays'][seed][int(label[-1])],allow_pickle=False) as z:replay={k:z[k].copy() for k in z.files}
                parents=torch.as_tensor(replay['source_parents'],device=device);rows=torch.as_tensor(replay['real_rows'],device=device);bank_seed=int(replay['random_seed'])
                np.testing.assert_array_equal(maps[map_label][HYPOTHESES[0]].cholesky.cpu().numpy(),replay['cholesky'])
            else:
                parents=torch.as_tensor(saved[label+'_source_parents'],device=device);rows=torch.as_tensor(saved[label+'_real_rows'],device=device)
                bank_seed=int(saved[label+'_random_seed'])
                np.testing.assert_array_equal(maps[map_label][HYPOTHESES[0]].cholesky.cpu().numpy(),saved[label+'_cholesky'])"""
assert s.count(old_block)==1;s=s.replace(old_block,new_block)
s=s.replace("counts['REAL_fit_feature_rows' if label.startswith('FIT') else 'REAL_CHECK_feature_rows']","counts['REAL_fit_feature_rows' if label.startswith(('FIT','IID')) else 'REAL_CHECK_feature_rows']")
start=s.index('        for label in LABELS:');end=s.index("        report['all_local_gates']",start)
section=s[start:end].replace('maps[label]','maps[map_label]').replace('contexts[label]','contexts[map_label]')
section=section.replace("saved[label+'_'+key]","(replay[key] if label.startswith('IID') else saved[label+'_'+key])")
section=section.replace("control['curvature_update_'+label+'_actual_'+suffix]","(replay['one_rank_actual_'+suffix] if label.startswith('IID') else control['curvature_update_'+label+'_actual_'+suffix])")
s=s[:start]+section+s[end:]
s=s.replace("v['second_curvature']","v['stable_projection']").replace('two_seed_second_curvature_all_local_pass','two_seed_stable_projection_all_local_pass')
s=s.replace('SECOND_CURVATURE.npz','STABLE_PROJECTION.npz').replace('SECOND_CURVATURE.json','STABLE_PROJECTION.json')
s=s.replace('second_curvature_fixed_candidate_checks','stable_projection_fixed_candidate_checks').replace('second_curvature_setup_loaded','stable_projection_setup_loaded').replace('second-curvature diagnostic','stable-projection diagnostic')
s=s.replace('Fixed FIT-derived candidate comparison with original hard observation and new fitting-role FRESH sample and projections.','Fixed orthogonal projection ablation with paired original one-rank control, eight replayed fitting libraries and one new FRESH per head.')
driver='scripts/diagnose_joint01_stable_projection.py';ast.parse(s);(ROOT/driver).write_text(s,encoding='utf-8',newline='\n')
(HERE/'DRIVER_DIFF.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True))),encoding='utf-8')
base_config='configs/joint_response_01_second_curvature_diagnostic_20261020.json'
config='configs/joint_response_01_stable_projection_diagnostic_20261020.json';contract='docs/JOINT_RESPONSE_01_STABLE_PROJECTION_DIAGNOSTIC_20261020.md'
p=json.loads((ROOT/base_config).read_bytes());pins=json.loads((ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json').read_bytes())['source_pins']
inputs={};replicas=[]
for seed in range(2):
    paths=[]
    for rep in range(4):
        target=f'diagnostic_inputs/STABLE_PROJECTION_seed{seed}_rep{rep}_IID.npz'
        inputs[target]=str(PREV/f'terminal_01/collected/run_01/seed{seed}_rep{rep}_IID.npz');paths.append(target)
    replicas.append(paths)
proposal='diagnostic_inputs/STABLE_PROJECTION_FIXED_PROPOSALS.npz';inputs[proposal]=str(PREV/'STABLE_PROJECTION_PROPOSALS.npz')
assert hashlib.sha256(Path(inputs[proposal]).read_bytes()).hexdigest()==fixed['proposal_sha256']
for leaf in ('terminal_01/TERMINAL_REVIEW.json','terminal_01/RNG_NATIVE_REPLAY.json','SAVED_VARIABILITY_AND_BIAS_DIAGNOSIS.json','STABLE_SUBSPACE_DESIGN.json','STABLE_SUBSPACE_CHECK.json','STABLE_SUBSPACE_AUDIT_AND_PROJECTION.json','CHECK_STABLE_SUBSPACE.py','AUDIT_SUBSPACE_AND_FREEZE_PROJECTION.py','seed0_STABLE_SUBSPACE_SOLVER_EVIDENCE.npz','seed1_STABLE_SUBSPACE_SOLVER_EVIDENCE.npz','seed0_STABLE_SUBSPACE_INFEASIBILITY_CERTIFICATE.npz'):
    name=(PREV/leaf).relative_to(ROOT).as_posix();inputs[name]=str(ROOT/name)
inputs['configs/joint_response_01_random_precision_diagnostic_20261020.json']=str(ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json')
p.update(schema='n01/joint-response01-stable-projection-diagnostic/v1',job_id='JOINT_RESPONSE_01_STABLE_PROJECTION_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,previous_terminal_review=(PREV/'terminal_01/TERMINAL_REVIEW.json').relative_to(ROOT).as_posix(),saved_diagnosis=(PREV/'STABLE_SUBSPACE_AUDIT_AND_PROJECTION.json').relative_to(ROOT).as_posix(),fixed_proposals=proposal,random_precision_replays=replicas,fresh_seed=202690100000,fresh_head_stride=1000000,source_pins=pins)
p['feature_seeds']['FRESH']=2026103701;p['native_projection_seeds']['FRESH']=2026103702
p['operation_budget'].update(REAL_fit_feature_rows=57344,allocated_banks=18,noise005_calls=18,noise005_rows=589824,extension_rows=1769472,shape_standard_normal_values=4718592,joint_response_calls=72,joint_response_rows=2359296,generated_observation_calls=216,generated_observation_rows=7077888,native_component_rows=7962624,saved_center_replays_verified=16,saved_anchor_replays_verified=16,fixed_candidate_checks=36,saved_control_candidate_replays_verified=16)
old_streams=set();old_projection_seeds=set()
for tag in ('native_curvature','native_gain','scientific_progress','curvature_update','second_curvature'):
    previous=json.loads((ROOT/f'configs/joint_response_01_{tag}_diagnostic_20261020.json').read_bytes())
    for seed in range(2):old_streams.update(previous['fresh_seed']+100000*seed+x for x in (100,200,1000,1001,1002))
    old_projection_seeds.update(previous['feature_seeds'].values());old_projection_seeds.update(previous['native_projection_seeds'].values())
rp=json.loads((ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json').read_bytes())
for seed in range(2):
    for rep in range(4):old_streams.update(rp['random_seed']+seed*rp['head_seed_stride']+rep*rp['replicate_seed_stride']+x for x in range(3))
new_streams=[p['fresh_seed']+p['fresh_head_stride']*seed+x for seed in range(2) for x in (100,200,1000,1001,1002)]
assert len(new_streams)==len(set(new_streams)) and not old_streams.intersection(new_streams)
assert not {2026103701,2026103702}.intersection(old_projection_seeds)
for name in [driver,contract,*inputs]:p['source_pins'][name]=hashlib.sha256(Path(inputs[name] if name in inputs else ROOT/name).read_bytes()).hexdigest()
(ROOT/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
digest=hashlib.sha256((ROOT/config).read_bytes()).hexdigest();(HERE/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((PREV/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{driver,contract,config})
d=json.loads((ROOT/'jobs.json').read_bytes());oldjobs=copy.deepcopy(d['jobs'])
assert len(oldjobs)==139 and d['active_scientific_jobs']==0 and d['followup']['count']==0
d['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_stable_projection_diagnostic_20261020',resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False))
assert d['jobs'][:-1]==oldjobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    doc=d if name=='jobs.json' else json.loads((ROOT/name).read_bytes())
    doc.update(status='JOINT_RESPONSE_01_stable_projection_diagnostic_registered',updated_UTC=now,pending_job_id=p['job_id'],current_stage='4_stable_projection_diagnostic',active_work_package='4_stable_projection_diagnostic',current_action='Fixed one-rank-step orthogonal projection ablation registered before native execution.',next_action='Check paired control and fixed projected probes on eight replayed libraries and one new FRESH per head; no new fitting or updates.')
    doc['joint_response_01'].update(stable_projection_diagnostic_status='registered_not_started',stable_projection_diagnostic_config=config,stable_projection_diagnostic_config_sha256=digest)
    (ROOT/name).write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
reg=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_139_jobs_unchanged=True,new_FRESH_streams=new_streams,new_FRESH_streams_disjoint=True,old_IID_streams_reused_intentionally_for_paired_ablation=True)
(HERE/'REGISTRATION.json').write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((PREV/'REGISTRATION.json').read_bytes())['config_sha256']
for name in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
    source=(PREV/name).read_text(encoding='utf-8') if name!='COLLECT.py' else (T/name).read_text(encoding='utf-8')
    if name=='PREPARE_DEPLOYMENT.py':
        source=source.replace('second_curvature_diagnostic_01/SOURCE_MANIFEST.json','random_precision_diagnostic_01/SOURCE_MANIFEST.json').replace("previous=base/'joint_response01_second_curvature_diagnostic_20261020'","previous=base/'joint_response01_random_precision_diagnostic_20261020'")
    source=source.replace('joint_response_01_random_precision_diagnostic_20261020','joint_response_01_stable_projection_diagnostic_20261020').replace("root=base/'joint_response01_random_precision_diagnostic_20261020'","root=base/'joint_response01_stable_projection_diagnostic_20261020'").replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_stable_projection_diagnostic_20261020')")
    source=source.replace('diagnose_joint01_random_precision','diagnose_joint01_stable_projection').replace(old_digest,digest).replace('== 879',f'== {sources}').replace('==879',f'=={sources}').replace('random_precision_diagnostic_terminal_01.tar','stable_projection_diagnostic_terminal_01.tar').replace('n01/joint-response01-random-precision-diagnostic','n01/joint-response01-stable-projection-diagnostic')
    if name=='REMOTE_DISPATCH_REQUEST.py':source=source.replace("'5700s'","'480s'")
    if name=='REMOTE_COLLECT.py':source=source.replace('128*1024**2','64*1024**2')
    ast.parse(source);(HERE/name).write_text(source,encoding='utf-8',newline='\n')
print(json.dumps(reg))
