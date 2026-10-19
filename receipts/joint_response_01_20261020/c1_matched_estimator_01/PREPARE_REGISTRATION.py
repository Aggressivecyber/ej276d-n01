from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root = Path('E:/GAN/research/N01')
here = Path(__file__).resolve().parent
previous = root/'receipts/joint_response_01_20261020/c1_matched_control_01'
terminal = previous/'terminal_01'
now = datetime.now(timezone.utc).isoformat()
review = json.loads((terminal/'TERMINAL_REVIEW.json').read_bytes())
remote = json.loads((terminal/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status'] == 'PASS_DUAL_TARGET_SAVED_AUDIT_MATCHED_RECOVERY_FAILED'
assert remote['native_exit']['exit_code'] == 0 and remote['processes'] == []
ledger = json.loads((root/'jobs.json').read_bytes())
assert len(ledger['jobs']) == 124 and ledger['active_scientific_jobs'] == 0
old_jobs = copy.deepcopy(ledger['jobs'])
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name, here/'before_registration'/name)

script = 'scripts/diagnose_joint01_c1_matched_estimator.py'
source = (root/'scripts/diagnose_joint01_c1_estimator.py').read_text(encoding='utf-8')
source = source.replace('Frozen full-parent teacher residuals versus independent-bank noise; no fitting.',
    'Frozen matched-family residuals versus independent-bank noise; no fitting.')
source = source.replace("LABELS = ('original_oracle', 'C1_known_truth', 'C1_before_last', 'C1_endpoint')",
    "LABELS = ('C1_known_truth', 'C1_before_last', 'C1_endpoint')")
start = source.index('def pooled_three_maps(')
end = source.index('def run(', start)
source = source[:start] + 'from diagnose_joint01_c1_estimator import pooled_three_maps, summarize_residuals\n\n\n' + source[end:]
source = source.replace('target = pooled_three_maps(p,teacher,truth,target_bank',
    'target = pooled_three_maps(p,student,truth,target_bank')
source = source.replace('zip(LABELS,(teacher,student,student,student),(truth,truth,before,endpoint),strict=True)',
    'zip(LABELS,(student,student,student),(truth,before,endpoint),strict=True)')
source = source.replace('C1_estimator_frozen_repeats', 'C1_matched_estimator_frozen_repeats')
source = source.replace("return dict(status='COMPLETE',reports=reports", "return dict(status='COMPLETE',target_family='C1',reports=reports")
ast.parse(source)
assert 'stats_obs = prediction(teacher,truth,stats_bank' in source
assert 'target = pooled_three_maps(p,student,truth,target_bank' in source
(root/script).write_text(source, encoding='utf-8', newline='\n')

contract = 'docs/JOINT_RESPONSE_01_C1_MATCHED_ESTIMATOR_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：匹配家族的冻结估计器检验

2026-10-20，登记后执行，零参数更新。两头同家族教师对照已完成：seed0两个2048 draw精度库的条件Q方差比约0.494，低于原0.50门。原家族失败保留。此前原教师估计器同时含家族误差，不能直接作为此次同家族端点的噪声证据。

冻结两头的最后一步之前和最终参数；另设C1已知教师参数作为零差异诊断对照。目标和三个SOURCE点均使用同一C1生成器；各SOURCE点共享随机数，目标独立。已知参数仅进入诊断对照，不生成训练方向、标签或候选。固定原6465拟合父集，128报告父完全排除。

每头8对独立全父随机库，每库每父4 draw、512父分块。预登记order seed 2026102301及bank base 202610230000，头/重复/两侧偏移沿用原诊断规则。18个FIT/CHECK/FINAL投影及白化因子由原教师相同统计库精确重现；不换特征、门限、模型或人口。

保存原始特征均值、白化因子、父排列、种子和残差。独立复算每库风险、两组16 draw及32 draw聚合风险、跨独立库内积U、MC噪声项、8库留一jackknife标准误及最后一步配对U差。U允许为负，8库误差只作描述。三个投影均已暴露，仅用于方法诊断。

成功标准是身份、随机流、计数和上述代数复算一致，并量化端点残差是否高于本次MC噪声。若差异主要淹没于噪声，不能仅凭单库风险扩训练；若仍有清楚的结构残差，结合最后一步差异选择一项有依据的后续检验。无论结果如何均不替代原条件核门、不自动开放REAL。

资源：4 CPU，GPU分配上限2.75 GiB，RSS 12 GiB，内部240秒、外部300秒加20秒终止余量，输出32 MiB。只执行一次；无自动重启或预算续期。预期约16–20秒，短步骤在本次直接完成，不安排定时等待。
''', encoding='utf-8', newline='\n')

p = json.loads((root/'configs/joint_response_01_c1_estimator_20261020.json').read_bytes())
matched = json.loads((root/'configs/joint_response_01_c1_matched_control_20261020.json').read_bytes())
p['source_pins'] = copy.deepcopy(matched['source_pins'])
p.update(schema='n01/joint-response01-c1-matched-estimator/v1',job_id='JOINT_RESPONSE_01_C1_MATCHED_ESTIMATOR_20261020',
    registered_UTC=now,stage='3_C1_matched_estimator_resolution',method_contract=contract,
    target_family='C1',feature_statistics_target_family='original',order_seed=2026102301,bank_seed=202610230000,
    endpoint_iterations=[f'c1_matched_inputs/seed{s}_joint_ITER11.npz' for s in range(2)],
    matched_control_review=str((terminal/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'))
inputs = {}
for name in ('AUDIT_COMPLETE.py','TERMINAL_REVIEW.json','SAVED_FAILURE_DIAGNOSIS.json'):
    inputs[(terminal/name).relative_to(root).as_posix()] = str(terminal/name)
for seed in range(2):
    inputs[p['endpoint_iterations'][seed]] = str(terminal/f'collected/run_01/seed{seed}_joint_ITER11.npz')
for name in ('scripts/run_joint_response01_c1_matched_control.py','configs/joint_response_01_c1_matched_control_20261020.json'):
    p['source_pins'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
for name,path in inputs.items():
    p['source_pins'][name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
budget = p['operation_budget']
budget.update(joint_response_calls=2+16*4*13,joint_response_rows=2*6465+16*4*6465*4,
    generated_observation_calls=3*(2+16*4*13),generated_observation_rows=3*(2*6465+16*4*6465*4),
    fixed_endpoint_evaluations=48)
config = 'configs/joint_response_01_c1_matched_estimator_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest = hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
static = dict(status='PASS',UTC=now,syntax_pass=True,target_C1=True,statistics_original_unchanged=True,
    pooled_estimation_and_summary_functions_imported_unchanged=True,new_fixed_endpoints=2,labels=3,independent_pairs=16,
    expected_counts=budget,parameter_updates=0,REAL_updates=0,source_pins=len(p['source_pins']))
(here/'STATIC_REVIEW.json').write_text(json.dumps(static,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest = json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
expected_sources = len(set(manifest['files']) | set(inputs) | {config,script,contract})
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,
    config_sha256=digest,host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_matched_estimator_20261020',
    resources=p['resources'],operation_budget=budget,parameter_updates=0,scientific_acceptance=False))
assert ledger['jobs'][:124] == old_jobs
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    data = ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    data.update(status='JOINT_RESPONSE_01_C1_matched_estimator_registered',current_stage=p['stage'],active_work_package=p['stage'],
        active_scientific_jobs=0,pending_job_id=p['job_id'],active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,
        updated_UTC=now,scientific_acceptance=False,current_action='Registered frozen matched-family endpoint/noise diagnostic.',
        next_action='Run the short zero-update diagnostic now; audit saved independent-bank estimates before changing the method.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
    data['joint_response_01'].update(c1_matched_estimator_config=config,c1_matched_estimator_config_sha256=digest,
        c1_matched_estimator_started=False,c1_matched_estimator_scientific_acceptance=False)
    (root/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=expected_sources,source_pins=len(p['source_pins']),science_started=False,new_model_updates=0,prior_124_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
