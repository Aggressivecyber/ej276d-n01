from datetime import datetime,timezone
from pathlib import Path
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent;previous=here.parent/'second_curvature_diagnostic_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes());terminal=json.loads((previous/'terminal_01/REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_SECOND_CURVATURE_DIAGNOSTIC_AUDIT' and not review['two_seed_second_curvature_all_local_pass'] and not terminal['processes']
before=here/'before_registration';before.mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):shutil.copy2(root/name,before/name)
now=datetime.now(timezone.utc).isoformat();name='scripts/diagnose_joint01_random_precision.py';config='configs/joint_response_01_random_precision_diagnostic_20261020.json';contract='docs/JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020.md';old_config='configs/joint_response_01_second_curvature_diagnostic_20261020.json'
old=json.loads((root/old_config).read_bytes());p={key:old[key] for key in ('frozen_setup_config','roles_file','batch','solver','native_guard','success','source_pins')};inputs={};fixed=[]
for seed in range(2):
 file=f'diagnostic_inputs/RANDOM_PRECISION_FIXED_seed{seed}.npz';inputs[file]=str(previous/f'terminal_01/collected/run_01/seed{seed}_SECOND_CURVATURE.npz');fixed.append(file)
for leaf in ['terminal_01/TERMINAL_REVIEW.json','SAVED_ANCHOR_AND_GEOMETRY_DIAGNOSIS.json','DIAGNOSE_SAVED_GEOMETRY.py','STRICT_ANCHOR_GAIN_INFEASIBILITY_CERTIFICATE.json','RESIDUAL_MARGIN_INFEASIBILITY_CERTIFICATE.json']:
 file=(previous/leaf).relative_to(root).as_posix();inputs[file]=str(root/file)
inputs[old_config]=str(root/old_config)
resources=dict(CPU_threads=4,GPU_GiB=4,RSS_GiB=12,internal_seconds=5400,hard_wall_seconds=5700,output_MiB=128,GPU_free_required_MiB=6144)
budget=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,REAL_observable_replay_calls=1,REAL_observable_replay_rows=21047,feature_maps_constructed=1,new_native_projection_contexts=1,REAL_fit_feature_rows=8192,allocated_banks=10,derived_antithetic_banks=8,noise005_calls=10,noise005_rows=327680,extension_rows=983040,shape_standard_normal_values=2621440,joint_response_calls=1800,joint_response_rows=58982400,generated_observation_calls=5400,generated_observation_rows=176947200,native_component_rows=199065600,saved_center_replays_verified=2,saved_anchor_replays_verified=2,saved_control_candidate_replays_verified=4,fixed_candidate_checks=36,finite_difference_columns=864,joint_direction_solves=0,secant_matrix_corrections=0,REAL_response_updates=0,DEV_waveforms_decoded=0,AD_queries=0,classifier_fits=0)
p.update(schema='n01/joint-response01-random-precision-diagnostic/v1',job_id='JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020',registered_UTC=now,method_contract=contract,saved_arrays=old['saved_arrays'],fixed_candidates=fixed,candidate_keys={'one_rank':'curvature_update_control','two_rank':'second_curvature'},finite_difference_step=.004,replicates=4,random_seed=202680100000,head_seed_stride=1000000,replicate_seed_stride=10000,feature_seed=2026103101,native_projection_seed=2026103111,diagnostic_gate=dict(maximum_variance_ratio=.75,mean_J_norm_ratio_low=.8,mean_J_norm_ratio_high=1.25,native_Taylor_error_must_not_increase=True),resources=resources,operation_budget=budget,committed_parameter_updates=0,new_parameter_candidates=0,automatic_restart=False,automatic_training=False,original_final_gates_unchanged=True)
new_streams=[p['random_seed']+seed*p['head_seed_stride']+rep*p['replicate_seed_stride']+offset for seed in range(2) for rep in range(4) for offset in (0,1,2)]
old_streams=set()
for label in ('native_curvature','native_gain','scientific_progress','curvature_update','second_curvature'):
 c=json.loads((root/f'configs/joint_response_01_{label}_diagnostic_20261020.json').read_bytes())
 for seed in range(2):old_streams.update(c['fresh_seed']+100000*seed+x for x in (100,200,1000,1001,1002))
assert len(new_streams)==len(set(new_streams)) and not set(new_streams).intersection(old_streams)
for file in [name,contract,*inputs]:p['source_pins'][file]=hashlib.sha256(Path(inputs[file] if file in inputs else root/file).read_bytes()).hexdigest()
ast.parse((root/name).read_text('utf-8'));(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n');digest=hashlib.sha256((root/config).read_bytes()).hexdigest();(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
oldmanifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(oldmanifest['files'])|set(inputs)|{name,contract,config})
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs']);assert len(old_jobs)==138 and ledger['active_scientific_jobs']==0 and old_jobs[-1]['status']=='completed_audited_local_gates_failed'
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020',resources=resources,operation_budget=budget,parameter_updates=0,new_parameter_candidates=0,scientific_acceptance=False,automatic_restart=False,automatic_training=False));assert ledger['jobs'][:-1]==old_jobs
for file in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
 d=ledger if file=='jobs.json' else json.loads((root/file).read_bytes());assert d['active_scientific_jobs']==0
 d.update(status='JOINT_RESPONSE_01_random_precision_diagnostic_registered',updated_UTC=now,pending_job_id=p['job_id'],current_stage='4_random_precision_diagnostic',active_work_package='4_random_precision_diagnostic',current_action='Four paired IID32 versus antithetic-shape-latent32 libraries per head at frozen FIT0 registered.',next_action='Run fixed-endpoint Jacobian dispersion and prediction-error diagnostic; zero directions or updates. Estimate one wake from measured startup speed.')
 d['joint_response_01'].update(random_precision_diagnostic_status='registered_not_started',random_precision_diagnostic_config=config,random_precision_diagnostic_config_sha256=digest)
 (root/file).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_138_jobs_unchanged=True,new_random_streams=new_streams,new_random_streams_disjoint_from_five_predecessors=True,paired_schemes_share_noise_and_first_16_latents=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
old_digest=json.loads((previous/'REGISTRATION.json').read_bytes())['config_sha256']
for file in ('PREPARE_DEPLOYMENT.py','REMOTE_DISPATCH_REQUEST.py','REMOTE_STARTUP_REQUEST.py','REMOTE_COLLECT.py','COLLECT.py'):
 content=(previous/file).read_text('utf-8')
 if file=='PREPARE_DEPLOYMENT.py':content=content.replace('curvature_update_diagnostic_01/SOURCE_MANIFEST.json','second_curvature_diagnostic_01/SOURCE_MANIFEST.json').replace("previous=base/'joint_response01_curvature_update_diagnostic_20261020'","previous=base/'joint_response01_second_curvature_diagnostic_20261020'")
 content=content.replace('joint_response_01_second_curvature_diagnostic_20261020','joint_response_01_random_precision_diagnostic_20261020').replace("root=base/'joint_response01_second_curvature_diagnostic_20261020'","root=base/'joint_response01_random_precision_diagnostic_20261020'").replace("Path('/home/rylan/projects/ej276d-n01/joint_response01_second_curvature_diagnostic_20261020')","Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')")
 content=content.replace('diagnose_joint01_second_curvature','diagnose_joint01_random_precision').replace(old_digest,digest).replace('== 869',f'== {sources}').replace('==869',f'=={sources}').replace('second_curvature_diagnostic_terminal_01.tar','random_precision_diagnostic_terminal_01.tar').replace('n01/joint-response01-second-curvature-diagnostic','n01/joint-response01-random-precision-diagnostic').replace("'480s'","'5700s'").replace('64*1024**2','128*1024**2');ast.parse(content);(here/file).write_text(content,encoding='utf-8',newline='\n')
plan=root/'JOINT_RESPONSE_01_PLAN.md';text=plan.read_text('utf-8');lines=text.splitlines();lines[2]='> 当前执行状态（2026-10-20）：三轮REAL校准及五项固定方向诊断已闭合负结果；新四组随机精度诊断已前瞻登记，当前0科学作业、0定时。原科学门未通过，阶段5尚未进入。'
paragraph='''
## 2026-10-20 保存局部几何审计及随机估计诊断登记

[保存几何分析](receipts/joint_response_01_20261020/second_curvature_diagnostic_01/SAVED_ANCHOR_AND_GEOMETRY_DIAGNOSIS.json)显示原两头FIT0中心初值QPSD最大1.010624/1.015118，进入本轮时已经超过1.01。seed0后两步在38个低于ridge的方向上分配76%/81%步长平方，方向余弦0.0432而FIT收益几乎相同，追加曲率未稳定解。严格初值不退化+每FIT至少1%joint改善，以及保留已测正预测误差余量的两个局部问题均在seed0不可行，已独立核验对偶证书；没有据此启动新候选，也不推广为真实物理不可行。

[随机精度诊断](docs/JOINT_RESPONSE_01_RANDOM_PRECISION_DIAGNOSTIC_20261020.md)仅比较固定FIT0记录/端点/读出下的IID32与响应潜变量正负成对32，四库×两头，电子学噪声在同组两方案完全共用。864完整差分列、1800生成调用，无新求解/候选/更新；以两头J和native导数离散度至少下降25%且预测误差不增等前瞻门判定。成对方案可能失败，不据通用方差缩减结论直接训练。新内部5400/外部5700秒预算，正确启动后用实测进度估算一次苏醒。
'''
plan.write_text('\n'.join(lines)+'\n'+paragraph,encoding='utf-8',newline='\r\n')
print(json.dumps(record))
