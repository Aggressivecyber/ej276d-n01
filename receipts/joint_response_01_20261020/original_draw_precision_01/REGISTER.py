from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
previous=root/'receipts/joint_response_01_20261020/c1_objective_alignment_01'
low=root/'receipts/joint_response_01_20261020/multibank_m0_01/terminal_01'
ledger=json.loads((root/'jobs.json').read_bytes())
old_jobs=copy.deepcopy(ledger['jobs'])
assert len(old_jobs)==130 and ledger['active_scientific_jobs']==0
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,here/'before_registration'/name)
now=datetime.now(timezone.utc).isoformat()
contract='docs/JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：原生成器的随机估计精度诊断

2026-10-20前瞻登记。原JointResponse01教师恢复已通过2048draw精度确认，原36硬门及18精度门保留。C1直接原教师恢复为36/36硬门、16/18精度门；冻结同库比较的24个视图均显示原目标偏好条件响应较差的C1端点，matched-C1目标的排序全部反转。已有C1导数替代原生成器导数的检查也未降低误差。本次保持原JointResponse01，检验随机估计精度对原REAL联合方向的影响。

固定已审计的M0三FIT诊断的全部SOURCE父、REAL行、五组joint/native投影、零增量初值及求解规则。每头每角色1024父，REAL4096行，五角色为FIT0/FIT1/FIT2/CHECK/FRESH。原4draw逐项重建并复算五组中心残差与native值，源/REAL索引及Cholesky必须精确一致。每父追加独立28draw，拼成32draw；原四个z及三个噪声假设的噪声序列必须逐值保留。两头共10组嵌套库。诊断抽样仍来自原6465/18948拟合池，128/2099报告角色排除；本次固定批不声称遍历全部拟合父。

全部54参数在同一M0点、同一32draw库作0.004中央差分。三个FIT的816维残差/J除sqrt(3)堆叠，27个native约束分列保留，完整跨参数二次项不变。原SLSQP、ridge0.02、半径0.0125、最多3次候选及拒绝后FIT割线修正/半径减半均不变。各FIT及CHECK原联合至少1%收益、宏块1.15+2/4096、信赖域均值0.08/p99 0.25、flag增量max(4,0.005n)、wave/QPSD≤1.01及noise≤1.10必须同时通过。4096是固定REAL样本量，宏块MC项沿用原值。首个通过候选只作一次FRESH检查；失败不重选。所有候选均零提交更新。

旧4draw的已保存J/N及未提交候选作配对参照，重新报告J/N相对差异和夹角；旧候选只在32draw CHECK/FRESH评价，不参与新方向或选择。两头32draw候选均通过原FIT/CHECK/FRESH局部门，才支持登记后续精度方法试验；最终REAL 20%联合、10%峰码、5%native改善门及阶段5仍须另行验证。条件响应、原人口计数、005与三种长程假设保持，C1失败及旧REAL失败均保留。

原4draw种子由旧role_seed_bases+100000*头+1000恢复；新增28draw种子202650280000+1000000*头+10000*角色，z/noise子流占基值、+1、+2。30条新增子流互斥并与原流及前项C1流分离。各32draw库保存在CPU；每次只传当前库至GPU，基准carrier及flag计数保存在CPU。生成器、原native函数和硬读出源码不改。新GPU分配上限4GiB，启动需至少6GiB空闲；4CPU、RSS12GiB、内部5400秒、外部5700秒加20秒、输出128MiB。超过原12GiB RSS或任一预算即停止，无自动重启或延长。

最大108参数差分列/324 FIT批列、6求解及候选、4割线修正、2 FRESH新候选检查、4旧候选检查。6检查点、TRAIN/PCA各1、A0 52调用/6593行、M0 106调用/13442行；REAL原观察1次21047行，5joint图/5native投影，FIT特征24576行、CHECK/FRESH特征16384行。20随机库分配/噪声调用、327680噪声行、983040扩展行、2623488标准正态数。最多700生成调用/22585600行，其中2次256行零增量检查；观察2094调用/67756032行；native比较76333056行。未用候选按实际计数扣除。DEV、AD、classifier、训练更新均0。

保存差分向量、参数、分母、中心/候选原生风险与门、父/REAL排列及种子。独立审计复算完整矩阵、所有实际割线与联合求解、接受门及预算；保存向量审计和原始波形重放分别报告。先完成准备与部署，启动后按本次实际列速度、剩余列和评价费用估时；只有长等待才安排本聊天单次苏醒。
''',encoding='utf-8',newline='\n')
prior_name='configs/joint_response_01_multibank_m0_20261020.json'
p=json.loads((root/prior_name).read_bytes())
p.update(schema='n01/joint-response01-original-draw-precision/v1',job_id='JOINT_RESPONSE_01_ORIGINAL_DRAW_PRECISION_20261020',
    registered_UTC=now,stage='4_original_generator_draw_precision',method_contract=contract,
    parent_failed_job='JOINT_RESPONSE_01_MULTIBANK_REAL_PILOT_20261020',extension_bank_seed=202650280000,
    method_scope='Nested 4-to-32-draw controlled numerical diagnostic on the original generator.',
    FRESH_role='Fixed historical fitting-role rows and projections, original four draws plus independent 28-draw extension.',
    data_roles_and_source_population_unchanged=True,scientific_acceptance=False)
p['batch']['draws_per_parent']=32
p['resources'].update(GPU_GiB=4,internal_seconds=5400,hard_wall_seconds=5700,output_MiB=128,GPU_free_required_MiB=6144)
high_calls=2*(5+54*2*3+3*4+1+2)
low_calls=10
generated_rows=high_calls*32768+low_calls*4096
p['operation_budget'].update(allocated_banks=20,shape_standard_normal_values=327680*8+256*8,
    noise005_calls=20,noise005_rows=327680,extension_rows=983040,joint_response_calls=high_calls+low_calls+2,
    joint_response_rows=generated_rows+256,generated_observation_calls=3*(high_calls+low_calls),
    generated_observation_rows=3*generated_rows,native_component_rows=3*(high_calls*(32768+4096)+low_calls*8192),
    low_draw_center_replays_verified=10,nested_banks_verified=10,frozen_low_candidate_checks=4)
assert p['operation_budget']['joint_response_rows']==22585600
assert p['operation_budget']['generated_observation_rows']==67756032
assert p['operation_budget']['native_component_rows']==76333056
inputs={}
p['low_draw_arrays']=[]
for seed in range(2):
    name=f'original_precision_inputs/seed{seed}_MULTIBANK_M0.npz'
    inputs[name]=str(low/f'run_01/seed{seed}_MULTIBANK_M0.npz')
    p['low_draw_arrays'].append(name)
for path in (low/'TERMINAL_REVIEW.json',
    root/'receipts/joint_response_01_20261019/kernel_precision_confirmation_01/TERMINAL_REVIEW.json',
    previous/'terminal_01/TERMINAL_REVIEW.json',previous/'terminal_01/SAVED_OBJECTIVE_DECOMPOSITION.json',
    root/'receipts/joint_response_01_20261020/c1_estimator_01/terminal_01/SAVED_SURROGATE_CHECK.json'):
    inputs[path.relative_to(root).as_posix()]=str(path)
script='scripts/diagnose_joint01_original_draw_precision.py'
ast.parse((root/script).read_text())
for name in (prior_name,script,contract):p['source_pins'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
for name,local in inputs.items():p['source_pins'][name]=hashlib.sha256(Path(local).read_bytes()).hexdigest()
streams=[p['extension_bank_seed']+1000000*h+10000*r+o for h in range(2) for r in range(5) for o in range(3)]
assert len(streams)==len(set(streams))==30
old_streams=[base+100000*h+1000+o for h in range(2) for base in p['role_seed_bases'].values() for o in range(3)]
assert not set(streams)&set(old_streams)
config='configs/joint_response_01_original_draw_precision_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources=len(set(manifest['files'])|set(inputs)|{config,script,contract})
(here/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='PASS',UTC=now,syntax_pass=True,
    original_generator_retained=True,original_native_functions_retained=True,old_center_numerical_replay_required=True,
    nested_z_and_noise_exact_replay_required=True,source_and_target_indices_frozen=True,
    GPU_residency_one_bank=True,baseline_carriers_retained_on_CPU=True,new_streams=30,
    full_cross_module_solve=True,hard_gates_unchanged=True,parameter_updates=0,
    operation_budget=p['operation_budget'],scientific_acceptance=False),indent=2)+'\n',encoding='utf-8',newline='\n')
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,
    config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_original_draw_precision_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,automatic_training=False,scientific_acceptance=False))
assert ledger['jobs'][:130]==old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_original_draw_precision_registered',active_scientific_jobs=0,
        pending_job_id=p['job_id'],active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,
        updated_UTC=now,current_stage=p['stage'],active_work_package=p['stage'],scientific_acceptance=False,
        current_action='Registered nested original-generator 4-to-32-draw numerical diagnostic; zero updates.',
        next_action='Verify deployment and dispatch bounded original-generator direction diagnostic.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,
        followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(original_draw_precision_config=config,original_draw_precision_config_sha256=digest,original_draw_precision_started=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,
    config_sha256=digest,source_files_expected=sources,source_pins=len(p['source_pins']),science_started=False,prior_130_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
