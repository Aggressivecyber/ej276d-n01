from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
old=root/'receipts/joint_response_01_20261020/c1_draw_precision_01';terminal=old/'terminal_01'
now=datetime.now(timezone.utc).isoformat()
review=json.loads((terminal/'TERMINAL_REVIEW.json').read_bytes())
memory=json.loads((terminal/'SAVED_MEMORY_DIAGNOSIS.json').read_bytes())
remote=json.loads((terminal/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_PARTIAL_DRAW_PRECISION_AUDIT_RESOURCE_STOP' and remote['processes']==[]
assert review['reports']['0']['fractions_passing_both']['32']==[1.,.5,.25]
assert memory['unnecessary_two_bank_32_draw_copies_bytes']==3829762560
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==126 and ledger['active_scientific_jobs']==0
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,here/'before_registration'/name)
contract='docs/JOINT_RESPONSE_01_C1_DRAW_COMPLETION_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：仅补全第二头缺失的方向精度检验

2026-10-20前瞻登记。原4/32draw诊断在RSS峰15,096,520,704字节时被原12GiB监督限额终止，exit -15；原作业关闭并保持资源失败判定，不续期。766源、108pins、16输出及162差分列、3个联合方向、第一头12候选的保存数组独立复算通过。第一头32draw完整步长在CHECK/VERIFY联合风险比0.79515/0.78171，三个预设步长均通过两库；4draw完整步长失败，较小步长通过。第二头缺32draw方向与12个独立库候选检查，尚不能作两头方法结论。

原计数最后快照为162列、2求解；保存矩阵证明第三求解完成。没有第二头高精度快照，异步SIGTERM可能发生在第9列计数增加与首次写盘之间，因此未保存完整列保守记0—9，原费用区间不抹去。

内存检查确认三种噪声均为384点float64。每个32draw全父库含1,914,881,280字节张量；原all-row高级索引额外复制两个完整库，占3,829,762,560字节。合成索引检查验证原拷贝与直接复用数值相同，后者没有新增存储。尚未精确归因第二头其余分配器驻留；本次原生RSS仍需核验。

单独运行第二头缺失内容，不重算第一头或增加方向数。保留原6465父、64父分块、全部数据角色、C1目标、固定端点、特征因子、原绝对随机种子、差分／ridge／半径／候选／信赖域规则和12GiB限额。高精度直接复用原32draw库，删除无效完整复制，噪声dtype不变。

先按原绝对流重建第二头两套拟合库和9个因子，逐值核对保存父排列、起点和4draw基线残差；重算预定第0、24、45、53列（均值／共享随机／M0尺度／电子学）并精确核对高低残差及跨度，再用保存完整J重放原4draw联合解。所有重复工作独立计费。随后只计算缺失32draw的54列及联合方向，再对原4/32draw方向各三个预设步长执行原独立CHECK和VERIFY。全部候选报告，零参数更新，不选新端点。

完成标准：原生0退出、恢复检查精确、资源与计数符合登记，保存全部剩余方向和12候选；独立复算并与第一头冻结结果合并报告。两头高精度收益在两个库重复，才有依据登记后续有界校准；仍不替代原条件核门、跨家族失败或自动开放REAL。

本次预算：4CPU，GPU分配2.75GiB，RSS12GiB；内部600秒，外部780秒加20秒，输出128MiB。6checkpoint、52A0调用／6593行、104M0调用／13186行（共享加载器），613随机分块库、6完整库、9特征因子；58差分列含4列恢复重放，2联合求解含1个旧解重放，12候选，0更新。生成13873调用／26331945行，观察41619调用／78995835行。原任务未保存费用仍另存区间。一次运行，不自动重启或延长；实测启动后按剩余工作安排单次苏醒。
''',encoding='utf-8',newline='\n')
old_config='configs/joint_response_01_c1_draw_precision_20261020.json'
p=json.loads((root/old_config).read_bytes())
p.update(schema='n01/joint-response01-c1-draw-completion/v1',job_id='JOINT_RESPONSE_01_C1_DRAW_COMPLETION_20261020',
    registered_UTC=now,stage='3_C1_draw_precision_missing_head_completion',method_contract=contract,
    completion_of='JOINT_RESPONSE_01_C1_DRAW_PRECISION_20261020',completion_seed=1,replay_columns=[0,24,45,53],
    low_direction='c1_draw_precision_inputs/seed1_draw4_DIRECTION.npz',saved_banks='c1_draw_precision_inputs/seed1_BANKS.npz',
    old_resource_failure_preserved=True,old_unknown_execution_bounds=review['unobserved_execution_bounds'])
p['resources'].update(internal_seconds=600,hard_wall_seconds=780,output_MiB=128)
p['operation_budget']=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,DEV_waveforms_decoded=0,
    old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
    allocated_banks=1+6*102,shape_standard_normal_values=8*(6465+6*6465*32),noise005_calls=1+6*102,
    noise005_rows=6465+6*6465*32,extension_rows=3*(6465+6*6465*32),
    joint_response_calls=1+(110+10+16)*102,joint_response_rows=6465+(110+16)*6465*32+10*6465*4,
    generated_observation_calls=3*(1+(110+10+16)*102),generated_observation_rows=3*(6465+(110+16)*6465*32+10*6465*4),
    finite_difference_columns=58,fullpool_FIT_chunk_columns=58*102,replayed_FD_columns=4,joint_direction_solves=2,joint_candidate_trials=12,
    teacher_accepted_joint_updates=0,feature_maps_constructed=9,full_fit_parent_banks=6,
    conditional_signal_rows=0,native_metric_rows=0,REAL_response_updates=0,AD_queries=0,classifier_fits=0)
inputs={}
for name in ('AUDIT_PARTIAL.py','TERMINAL_REVIEW.json','SAVED_MEMORY_DIAGNOSIS.py','SAVED_MEMORY_DIAGNOSIS.json'):
    path=terminal/name;inputs[path.relative_to(root).as_posix()]=str(path)
for key in ('low_direction','saved_banks'):
    inputs[p[key]]=str(terminal/'collected/run_01'/Path(p[key]).name)
for name in (old_config,'scripts/diagnose_joint01_c1_draw_precision.py'):
    p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
for name,path in inputs.items():p['source_pins'][name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
script='scripts/complete_joint01_c1_draw_precision.py';source=(root/script).read_text();ast.parse(source)
assert 'fit_view,target_view=fit,target' in source and 'for seed in (1,)' in source
assert 'theta=proposal' not in source
config='configs/joint_response_01_c1_draw_completion_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
static=dict(status='PASS',UTC=now,syntax_pass=True,only_seed1_missing_work=True,original_seed0_not_rerun=True,
    full_32_draw_bank_alias_no_copy=True,replay_columns=p['replay_columns'],original_solver_seeds_endpoint_roles_unchanged=True,
    operation_budget=p['operation_budget'],parameter_updates=0,native_memory_fix_verified=False)
(here/'STATIC_REVIEW.json').write_text(json.dumps(static,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=json.loads((old/'SOURCE_MANIFEST.json').read_bytes());sources=len(set(manifest['files'])|set(inputs)|{config,script,contract})
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,
    host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_draw_completion_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],completion_of=p['completion_of'],old_resource_failure_preserved=True,
    old_unknown_execution_bounds=p['old_unknown_execution_bounds'],parameter_updates=0,scientific_acceptance=False))
assert ledger['jobs'][:126]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_C1_draw_completion_registered',active_scientific_jobs=0,pending_job_id=p['job_id'],
        active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,updated_UTC=now,
        current_stage=p['stage'],active_work_package=p['stage'],scientific_acceptance=False,
        current_action='Registered only missing seed1 diagnostic; direct high-draw banks remove redundant copies.',
        next_action='Dispatch bounded completion, require exact saved low-draw replay and measure native RSS.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(c1_draw_completion_config=config,c1_draw_completion_config_sha256=digest,c1_draw_completion_started=False,
        c1_draw_completion_scientific_acceptance=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,new_model_updates=0,prior_126_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
