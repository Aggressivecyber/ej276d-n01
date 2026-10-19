from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,shutil

root=Path('E:/GAN/research/N01');here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/c1_draw_completion_01'
review=json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status']=='PASS_SAVED_DRAW_COMPLETION_AUDIT'
assert all(.5 in review['combined_reports'][str(seed)]['fractions_passing_both']['32'] for seed in (0,1))
ledger=json.loads((root/'jobs.json').read_bytes());old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==127 and ledger['active_scientific_jobs']==0
assert old_jobs[-1]['status']=='completed_independently_audited_zero_updates'
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,here/'before_registration'/name)
now=datetime.now(timezone.utc).isoformat()
contract='docs/JOINT_RESPONSE_01_C1_PRECISION_PILOT_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：32 draw有界联合校准

2026-10-20前瞻登记。固定端点精度对照与第二头补全均已独立审计。两头32 draw方向在两套独立库均有同一步长通过原联合接受规则；半步的CHECK/VERIFY风险比为seed0的0.87008/0.83989、seed1的0.94718/0.90202。4 draw在seed1没有两库共同通过的步长。原4/32方向余弦仅0.1733/0.1475。这支持检验更高抽样精度能否形成可重复的联合校准，而不是直接认定条件核已恢复。

从原同家族对照的两个冻结ITER11端点出发，每头固定4次新联合求解，共8方向、最多24候选／8更新。4次用于检验一次有用方向之后的迭代行为，全部完成后才评价；不按已暴露的Q方差阈值选择中间端点或追加迭代。新方法与预算单独登记；原4 draw失败和资源停止不改写。

每次FIT和CHECK均使用全部6465拟合父、32 draw，64父一块共102块；128报告父仍不参与拟合。目标保持C1同家族教师的无配对观测分布，已知参数仅进入教师生成器，配对条件标签不进入求解。18个原特征因子逐值复现，54坐标一起求解，差分0.04、ridge 0.02、半径0.4、候选1/0.5/0.25、原宏块与信赖域函数完全保持。CHECK按原顺序接受首个通过候选。

拟合父排列种子2026102501+1000*头+100*迭代+角色；随机库202610250000+1000000*头+10000*迭代+1000*角色+3*块。角色0/1为FIT源／目标，2/3为CHECK源／目标，9792个子流互斥。最终硬读出两库另用202620250000+1000000*头+10000*库+1000*源目标角色，24个子流与拟合分离。2048 draw条件核两库用2026102541至2026102544。两教师家族的末端报告故意复用同一报告随机库以作配对比较；统计因子仍用原种子复现。

所有4次更新完成后，M0、候选、教师在新的两套硬读出库上按原规则评价；分别报告matched_C1和original教师。两家族均用新2048 draw两库复算实际条件均值、16余弦投影协方差、Q／波形方差及MC差异。原36硬门和18精度门（每个家族、两头合计）全部保持；仅同家族通过仍不能替代原跨家族恢复试验，任何结果都不自动开放REAL。A0/M0与128报告父已有暴露，只称冻结系统后验报告，不称全流程未见验证。

内存安排：按目标、FIT、检查目标、CHECK顺序，仅保留当前一个完整32 draw随机库；每个阶段释放后调用Linux分配器malloc_trim归还空闲页。随机库使用显式局部种子，改变分配次序不改变库内容，汇总仍调用原pooled_mean/pooled_candidate并保持原顺序和dtype。该安排降低同时存活张量量，原生RSS仍受12GiB监督限额控制。预算／资源停止后不自动重启或延长。

上限：4CPU、GPU分配2.75GiB、RSS12GiB，内部1500秒、外部1740秒加20秒，输出192MiB。432差分列、44064分块列、8联合求解、24候选／8更新；3290随机库分配、3282电子学调用、6764162噪声行、20292486扩展行、70890512标准正态数；生成96978调用／197158274行，观察281622调用／571814022行。最终报告条件信号196608行、原生指标1179648行；32全父库、18因子。完整执行计数必须根据实际候选数复算，不能把最大预算写成实耗。

成功层次分开：工程成功需要原生0退出、哈希／计数／角色／资源通过、保存矩阵与接受规则可复算；本次方法成功需要两个头原条件核与硬观察门重复通过。项目科学接受、REAL联合改善与阶段5仍待后续证据。按实测每列与已完成迭代成本估算剩余时长，仅在实际长等待时安排本聊天一次苏醒。
''',encoding='utf-8',newline='\n')
p=json.loads((root/'configs/joint_response_01_c1_matched_control_20261020.json').read_bytes())
prior=json.loads((root/'configs/joint_response_01_c1_draw_completion_20261020.json').read_bytes())
p.update(schema='n01/joint-response01-c1-precision-pilot/v1',job_id='JOINT_RESPONSE_01_C1_PRECISION_PILOT_20261020',
    registered_UTC=now,stage='3_C1_precision_pilot',method_contract=contract,iterations=4,parents_per_chunk=64,
    batch=dict(source_parents=6465,draws_per_parent=32,target_rows=6465*32),order_seed=2026102501,
    bank_seed=202610250000,evaluation_bank_seed=202620250000,
    precision_bank_seeds=[[2026102541,2026102542],[2026102543,2026102544]],
    endpoint_iterations=prior['endpoint_iterations'],source_pins=prior['source_pins'],
    fixed_iteration_count_no_report_based_endpoint_selection=True,full_bank_storage='one bank at a time with allocator release')
p['resources'].update(internal_seconds=1500,hard_wall_seconds=1740,output_MiB=192)
fit_passes=8*(110+2+3);fit_rows=fit_passes*6465*32;fit_calls=fit_passes*102
noise_rows=8*4*6465*32+2*6465+16*8192
p['operation_budget']=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,DEV_waveforms_decoded=0,
    old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
    allocated_banks=8*4*102+2+16+8,shape_standard_normal_values=noise_rows*8+8*128*2048*8,
    noise005_calls=8*4*102+2+16,noise005_rows=noise_rows,extension_rows=3*noise_rows,
    joint_response_calls=fit_calls+2+8*8+8*128*3,
    joint_response_rows=fit_rows+2*6465+8*8*8192+8*128*3*2048,
    generated_observation_calls=3*fit_calls+6+8*12,
    generated_observation_rows=3*fit_rows+6*6465+8*12*8192,
    finite_difference_columns=8*54,fullpool_FIT_chunk_columns=8*54*102,joint_direction_solves=8,joint_candidate_trials=24,
    teacher_accepted_joint_updates=8,conditional_signal_rows=8*3*8192,native_metric_rows=8*3*3*2*8192,
    feature_maps_constructed=18,full_fit_parent_banks=32,REAL_response_updates=0,AD_queries=0,classifier_fits=0)
inputs={}
for name in ('AUDIT_COMPLETE.py','terminal_01/TERMINAL_REVIEW.json'):
    path=previous/name;inputs[path.relative_to(root).as_posix()]=str(path)
for name,path in inputs.items():p['source_pins'][name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
script='scripts/run_joint_response01_c1_precision_pilot.py'
for name in (script,contract):p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
training=[p['bank_seed']+1000000*h+10000*i+1000*r+3*c+o for h in range(2) for i in range(4) for r in range(4) for c in range(102) for o in range(3)]
evaluation=[p['evaluation_bank_seed']+1000000*h+10000*l+1000*r+o for h in range(2) for l in range(2) for r in range(2) for o in range(3)]
precision=sum(p['precision_bank_seeds'],[])
assert len(training)==len(set(training))==9792 and len(set(training+evaluation+precision))==9820
old_streams=[prior['bank_seed']+1000000*h+1000*r+3*c+o for h in range(2) for r in range(6) for c in range(102) for o in range(3)]
assert not set(training+evaluation+precision)&set(old_streams)
source=(root/script).read_text();ast.parse(source)
old_eval=(root/'scripts/run_joint_response01_teacher.py').read_text()
old_eval=old_eval[old_eval.index('def evaluate_seed('):old_eval.index('\n\ndef recovery_decision(')]
old_eval=old_eval.replace('def evaluate_seed(','def evaluate_fresh_seed(').replace('1920000 + seed_index * 10000 + library * 100',"p['evaluation_bank_seed'] + seed_index * 1000000 + library * 10000").replace('1930000 + seed_index * 10000 + library * 100',"p['evaluation_bank_seed'] + seed_index * 1000000 + library * 10000 + 1000")
new_eval=source[source.index('def evaluate_fresh_seed('):source.index('\n\ndef run(')]
assert ast.dump(ast.parse(new_eval))==ast.dump(ast.parse(old_eval))
assert p['solver']==prior['solver'] and p['success']==json.loads((root/'configs/joint_response_01_c1_matched_control_20261020.json').read_bytes())['success']
config='configs/joint_response_01_c1_precision_pilot_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
(here/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='PASS',UTC=now,syntax_pass=True,source_functions_reviewed=True,
    hard_evaluator_AST_same_except_fresh_seeds=True,unchanged_solver_and_success_rules=True,
    single_full_bank_live=True,training_substreams_unique=9792,evaluation_substreams_unique=24,precision_streams_unique=4,
    disjoint_training_evaluation_precision_and_previous_direction_banks=True,original_pooled_reducers_and_local_seed_generators_unchanged=True,
    operation_budget=p['operation_budget'],scientific_acceptance=False),indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources=len(set(manifest['files'])|set(inputs)|{config,script,contract})
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,
    host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_precision_pilot_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],fixed_iterations_per_head=4,known_teacher_labels_not_used_for_fitting=True,
    positive_control_not_stage3_replacement=True,automatic_REAL_promotion=False,scientific_acceptance=False))
assert ledger['jobs'][:127]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_C1_precision_pilot_registered',active_scientific_jobs=0,pending_job_id=p['job_id'],
        active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,updated_UTC=now,
        current_stage=p['stage'],active_work_package=p['stage'],scientific_acceptance=False,
        current_action='Registered 4 fresh 32-draw joint updates per head, preserving original acceptance and conditional-kernel gates.',
        next_action='Verify deployment, dispatch once, verify startup and estimate remaining wall time for one wake.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(c1_precision_pilot_config=config,c1_precision_pilot_config_sha256=digest,c1_precision_pilot_started=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_127_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
