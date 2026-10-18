from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,copy,hashlib,json,tarfile,shutil
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=datetime.now(timezone.utc).isoformat()
prev=read(r/'configs/response070_20261018.json');closed=r/'receipts/response070_20261018/wake_20261018_2043_01';oldrun=closed/'collected/run_01'
assert read(r/'PROGRAM.json')['response070']['stage']=='CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP' and read(w/'LOCAL_ENGINEERING_TEST.json')['status']=='PASS' and read(w/'LOCAL039_TEST.json')['status']=='PASS'
for name,digest in prev['source_pins'].items():assert sha(r/name)==digest,name
replay=dict(reports={str(seed):{case:read(oldrun/f'seed{seed}_{case}.json') for case in ('M0','LEVEL069','SPLITWARP')} for seed in range(2)},source_files={f'seed{seed}_{case}.json':sha(oldrun/f'seed{seed}_{case}.json') for seed in range(2) for case in ('M0','LEVEL069','SPLITWARP')})
(w/'REPLAY070.json').write_text(json.dumps(replay,indent=2)+'\n')
p=copy.deepcopy(prev)
p.update(schema='n01/response071-frozen-native-peak-tail-decomposition/v1',run_id='response071_native_peak_tail_decomposition_20261018',contract_document='docs/RESPONSE071_NATIVE_PEAK_TAIL_DECOMPOSITION_CONTRACT_20261018.md',replay_file=(w/'REPLAY070.json').relative_to(r).as_posix(),scope='Frozen070 all18H replay plus six BASE054 observations and full-population native Q/tail/peak algebraic decomposition; no changed response or new endpoint.',new_candidate_settings=0,event_content_boundary='Frozen original070 input-dependent mean and stochastic operator; diagnostic computations do not alter the response, draw RNG, fit parameters or change readout.',scientific_acceptance='Diagnostic success requires exact07018H records and72 scientific arrays plus six paired decomposition tables. The070 negative scientific result remains. Goal remains unachieved.',diagnostic_numeric_column_limit=52,diagnostic_boolean_column_limit=10)
p['operation_bounds'].pop('candidate_head_evaluations');p['operation_bounds'].update(new_candidate_settings=0,frozen_endpoint_head_replays=2,diagnostic_BASE054_observation_calls=6,diagnostic_BASE054_observation_rows=39558,diagnostic_decomposition_records=6,diagnostic_decomposition_rows=39558)
extras=['scripts/run_response071.py','scripts/response071_diagnostics.py','scripts/test_response071.py','scripts/verify_response071_diagnostics.py',p['contract_document'],p['replay_file'],'configs/response070_20261018.json']
extras.extend('reports/response070_20261018/'+name for name in ('REPORT.md','SUMMARY.json','DIAGNOSTIC.png','PEAK_EXAMPLES.png','TAIL_DIAGNOSTIC.png','NATIVE_TAIL_DIAGNOSTIC.png'))
extras.extend((closed/name).relative_to(r).as_posix() for name in ('TERMINAL_REVIEW.json','CLOSURE_RECEIPT.json','VISUAL_QA.json','ARTIFACT_VERIFICATION.json','COLLECTION_VERIFICATION.json','SPLIT_WARP_ANALYSIS.json','SPLIT_WARP_TOPOLOGY.json','TAIL_AND_CONDITIONAL_PSD_AUDIT.json','NATIVE_TAIL_DECOMPOSITION.json','ANALYZE_PROFILES.py','ANALYZE_TOPOLOGY.py','ANALYZE_TAIL.py','ANALYZE_NATIVE_TAIL.py','VERIFY_ARTIFACTS.py'))
base='receipts/response060_20261018/wake_20261018_1605_01/collected/run_01/'
extras.extend(base+f'seed{seed}_BASE054_{h}_NATIVE_ROWS.npz' for seed in range(2) for h in ('fast','slow','common_level'))
extras.extend(base+f'seed{seed}_BASE054.json' for seed in range(2))
for name in extras:p['source_pins'][name]=sha(r/name)
assert p['contract']==read(r/'configs/response039_20261018.json')['contract']
config=r/'configs/response071_20261018.json';assert not config.exists();config.write_text(json.dumps(p,indent=2)+'\n');digest=sha(config)
estimated=read(closed/'COLLECTION_VERIFICATION.json')['run_bytes']+39558*(52*8+10)+3*1024**2;assert estimated<64*1024**2
for name in ('scripts/run_response071.py','scripts/response071_diagnostics.py','scripts/test_response071.py','scripts/verify_response071_diagnostics.py'):ast.parse((r/name).read_text())
review=dict(UTC=now,root_only=True,prior070_closed_negative=True,original039_contract_exact=True,evidence='070 reference mean/random front/tail integrals preserved. About15% of SOURCE rows with differing native peak from BASE054 contribute about95% of paired absolute tail changes. Only78 saved fast examples support full attribution, requiring all-population replay.',tradeoff='Collect the missing same-window versus crop terms and intermediate peak contributions before selecting a new stochastic-timing correction. No new candidate, no score optimization, unchanged physics and readout.',unique_new_setting=None,stochastic_parameter_learning=False,search_or_AD_or_confirmation=False,predeclared_startup='First head9H/37 original arrays exact070; all3 new diagnostic tables19779 rows match BASE054060 and close algebraically;39 independent fast examples.',local_engineering=read(w/'LOCAL_ENGINEERING_TEST.json'),resources=p['resources'],operation_bounds=p['operation_bounds'],config_sha256=digest,source_pin_count=len(p['source_pins']),output_size_estimate_with_margin_bytes=estimated,output_estimate_is_not_compression_guarantee=True,scientific_goal_achieved=False)
(w/'ROOT_DESIGN_AND_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');before=w/'before_state';before.mkdir()
job=dict(job_id=p['run_id'],run_id=p['run_id'],stage='REGISTERED_NOT_STARTED',active=False,kind='bounded_frozen070_native_peak_tail_decomposition',config=config.relative_to(r).as_posix(),config_sha256=digest,contract=p['contract_document'],script='scripts/run_response071.py',script_sha256=sha(r/'scripts/run_response071.py'),root_design=(w/'ROOT_DESIGN_AND_REVIEW.json').relative_to(r).as_posix(),root_design_sha256=sha(w/'ROOT_DESIGN_AND_REVIEW.json'),remote_host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/'+p['run_id'],registered_UTC=now,resources=p['resources'],operation_bounds=p['operation_bounds'],scientific_status='response_match_not_established',native_started=False,agent_policy='current_root_only_no_subagents')
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 path=r/name;d=read(path);assert d['active_scientific_jobs']==0 and not d['response070']['active'];(before/name).write_bytes(path.read_bytes());d['response071']=copy.deepcopy(job)
 d.update(status='response071_registered_not_started',current_stage='response071_native_peak_tail_decomposition',current_arm='response071_frozen_peak_tail_diagnostic',active_job_id=None,updated_UTC=now,current_action='Diagnose remaining070 native Q/tail failures with frozen all-population replay and paired readout decomposition.',next_action='Verify first9H/37 scientific arrays exact070,three new diagnostic tables19779 rows and39 independent fast examples; save identities and schedule continuous followup.')
 d['response_goal_20261018'].update(stage_status='registered_not_started',current_stage=d['current_stage'],next_action=d['next_action'],updated_UTC=now)
 if name=='jobs.json':assert len(d['jobs'])==89;d['jobs'].append(copy.deepcopy(job))
 path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert read(before/'jobs.json')['jobs']==read(r/'jobs.json')['jobs'][:-1]
dep=w.parent/'deployment_01';dep.mkdir();files=set(p['source_pins'])|{p['setup038']['path'],config.relative_to(r).as_posix(),p['goal']};files.update(f.relative_to(r).as_posix() for f in w.iterdir() if f.is_file())
manifest=dict(schema='n01/response071-source-manifest/v1',UTC=now,config_sha256=digest,files={name:dict(bytes=(r/name).stat().st_size,sha256=sha(r/name)) for name in sorted(files)})
(dep/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');archive=dep/'response071_source.tar.gz';assert shutil.disk_usage(w).free>256*1024**2
with tarfile.open(archive,'w:gz') as tf:
 for name in manifest['files']:tf.add(r/name,arcname=name,recursive=False)
archivehash=sha(archive);olddep=r/'receipts/response070_20261018/deployment_01';oldrootname=PurePosixPath(read(olddep/'DISPATCH_RECEIPT.json')['remote_root']).name
for name in ('REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','INSTALL.py','DISPATCH.py','STARTUP.py'):
 code=(olddep/name).read_text().replace(oldrootname,p['run_id']).replace('response070','response071').replace("len(d['jobs'])==89","len(d['jobs'])==90").replace("'065 ","'071 ")
 if name=='REMOTE_STARTUP_REQUEST.py':
  code=code.replace("['candidate_head_evaluations']","['frozen_endpoint_head_replays']")
  code=code.replace('files={name:json.loads',"names += [f'run_01/seed0_SPLITWARP_{h}_DECOMPOSITION.json' for h in ('fast','slow','common_level')]\nfiles={name:json.loads")
  code=code.replace("binary_files={}","names += [f'run_01/seed0_SPLITWARP_{h}_DECOMPOSITION_ROWS.npz' for h in ('fast','slow','common_level')]\nbinary_files={}")
 ast.parse(code);(dep/name).write_text(code)
oldpkg=read(olddep/'PACKAGE_RECEIPT.json');code=(olddep/'REMOTE_INSTALL_REQUEST.py').read_text().replace(oldrootname,p['run_id']).replace('response070','response071').replace(oldpkg['archive_sha256'],archivehash).replace(oldpkg['archive_sha256'][:12],archivehash[:12])
code=code.replace("('039','069','070')","('039','070','071')").replace("('069','070')","('070','071')").replace("reused069_CUDA=results['069'],new070_CUDA=results['070']","reused070_CUDA=results['070'],new071_CUDA=results['071']").replace("'070 ","'071 ")
ast.parse(code);(dep/'REMOTE_INSTALL_REQUEST.py').write_text(code);(dep/'REGISTER_STARTUP.py').write_bytes((w/'STARTUP_VERIFIER_SOURCE.py').read_bytes())
pkg=dict(archive_sha256=archivehash,archive_bytes=archive.stat().st_size,source_files=len(files),source_pins=len(p['source_pins']),config_sha256=digest,remote_archive='/home/rylan/projects/ej276d-n01/'+f'response071_source_{archivehash[:12]}.tar.gz',remote_manifest='/home/rylan/projects/ej276d-n01/response071_source_manifest_01.json',native_started=False)
(dep/'PACKAGE_RECEIPT.json').write_text(json.dumps(pkg,indent=2)+'\n');print(json.dumps(pkg))
