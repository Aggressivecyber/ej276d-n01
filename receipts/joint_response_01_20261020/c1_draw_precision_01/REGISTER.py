from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
old=root/'receipts/joint_response_01_20261020/c1_matched_estimator_01'
terminal=old/'terminal_01'
now=datetime.now(timezone.utc).isoformat()
review=json.loads((terminal/'TERMINAL_REVIEW.json').read_bytes())
remote=json.loads((terminal/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
assert review['status']=='PASS_SAVED_MATCHED_ESTIMATOR_AUDIT_NO_SCIENTIFIC_ACCEPTANCE'
assert remote['native_exit']['exit_code']==0 and remote['processes']==[]
ledger=json.loads((root/'jobs.json').read_bytes())
assert len(ledger['jobs'])==125 and ledger['jobs'][-1]['job_id']=='JOINT_RESPONSE_01_C1_MATCHED_ESTIMATOR_20261020'
prior_jobs=copy.deepcopy(ledger['jobs'])
(here/'before_registration').mkdir()
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json','README.md','JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name,here/'before_registration'/name)

basis={'UTC':now,'native_zero_update_diagnostic_elapsed_seconds':json.loads((terminal/'collected/run_01/RESULT.json').read_bytes())['elapsed_seconds'],
    'source_review':str((terminal/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),'reports':review['reports'],
    'interpretation':'Endpoint cross-bank discrepancy remains resolved in these eight banks, while four-draw MC variance is several times larger; last accepted seed1 step increases new FIT/CHECK U. This supports a paired precision-direction diagnostic, not an iteration extension or scientific acceptance.',
    'choice_32_draws':'Eight-fold averaging lowers the measured four-draw MC contribution below the smallest endpoint U point estimate; sixteen draws would still exceed it. No claim of a significance test from eight-bank jackknife.',
    'parameter_updates':0,'scientific_acceptance':False}
(here/'SAVED_EVIDENCE_BASIS.json').write_text(json.dumps(basis,indent=2)+'\n',encoding='utf-8',newline='\n')
contract='docs/JOINT_RESPONSE_01_C1_DRAW_PRECISION_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：4与32次生成的联合方向对照

2026-10-20，前瞻登记，当前端点不更新。匹配家族冻结诊断通过760源文件、103输入与12输出复核：6个头/投影视图的端点跨库U为5.01e-6至1.25e-5，32 draw聚合MC项为4.33e-6至4.70e-6；单4 draw对应项约为其8倍。C1教师自身U接近零。seed1最后一步在新FIT/CHECK库的U增加约2.3e-6。因此，当前单库收益可能受MC噪声影响，不能直接加训练步数。

本项只比较估计精度：固定两头同家族对照的最终参数、完整6465拟合父、C1目标、原统计库与18个特征因子。54参数联合Jacobian、差分步0.04、ridge 0.02、方向半径0.4及原候选规则全部保持。每头各算4 draw与32 draw两个方向；4 draw取32 draw每父前4个，保留共同随机数。所有54参数同点联合求解，真值仅用于无配对目标分布生成，不进入方向标签。

块改为64父，使32 draw每块仍至多2048行。全6465父共102块。新bank base为202610240000，头间偏移1000000、FIT/目标/CHECK/目标/VERIFY/目标间偏移1000、块间偏移3；每库306种子范围互斥。不得沿用旧100的角色间隔，因为102块会重叠。新父排列seed 2026102401，头间偏移100、角色偏移0至5。

两个方向的1、1/2、1/4三个预设比例全部报告，用独立32 draw CHECK库和另一独立32 draw VERIFY库检查；分别使用原CHECK和FINAL投影。两次都满足原联合收益、宏块非劣与输出信赖域才记为可重复的方向证据。不选择或保存新的训练端点，不读取128报告父，不计算条件真值作训练或挑选。此诊断不替代条件核恢复门或原跨家族失败，不自动开放REAL。

保留四套完整残差、差分高低点、Jacobian、联合方向，以及24个候选的实际硬读出残差、信赖域向量、标志计数。独立复算身份、种子间隔、差分与联合求解、全部候选规则后，比较4/32方向的实际收益和复现情况。只有更高精度方向的收益重复，才有依据登记后续有界校准；否则继续定位目标信息或联合方向问题，不盲目放大运行。

资源上限4 CPU、GPU分配2.75 GiB、RSS12 GiB，内部1200秒、外部1500秒加20秒终止余量，192 MiB输出。两头共216差分列、4联合求解、24候选、0参数更新。一次执行，无自动重启或预算续期。启动后按实际差分吞吐与剩余4/32工作量安排单次苏醒。
''',encoding='utf-8',newline='\n')
script='scripts/diagnose_joint01_c1_draw_precision.py'
text=(root/script).read_text();ast.parse(text)
p=json.loads((root/'configs/joint_response_01_c1_matched_estimator_20261020.json').read_bytes())
p.update(schema='n01/joint-response01-c1-draw-precision/v1',job_id='JOINT_RESPONSE_01_C1_DRAW_PRECISION_20261020',
    registered_UTC=now,stage='3_C1_draw_precision_joint_direction',method_contract=contract,
    parents_per_chunk=64,order_seed=2026102401,bank_seed=202610240000,direction_draws=[4,32],validation_draws=32)
p['batch'].update(draws_per_parent=32,target_rows=6465*32)
del p['independent_repeats']
p['resources'].update(internal_seconds=1200,hard_wall_seconds=1500,output_MiB=192)
p['operation_budget']=dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,DEV_waveforms_decoded=0,
    old_A_calls=52,old_A_rows=6593,M0_head_calls=104,M0_head_rows=13186,
    allocated_banks=2+12*102,shape_standard_normal_values=8*(2*6465+12*6465*32),
    noise005_calls=2+12*102,noise005_rows=2*6465+12*6465*32,extension_rows=3*(2*6465+12*6465*32),
    joint_response_calls=2+2*2*110*102+2*2*8*102,
    joint_response_rows=2*6465+2*110*6465*(4+32)+2*2*8*6465*32,
    generated_observation_calls=3*(2+2*2*110*102+2*2*8*102),
    generated_observation_rows=3*(2*6465+2*110*6465*(4+32)+2*2*8*6465*32),
    finite_difference_columns=216,fullpool_FIT_chunk_columns=216*102,joint_direction_solves=4,joint_candidate_trials=24,
    teacher_accepted_joint_updates=0,feature_maps_constructed=18,full_fit_parent_banks=12,
    conditional_signal_rows=0,native_metric_rows=0,REAL_response_updates=0,AD_queries=0,classifier_fits=0)
inputs={}
for path in (terminal/'AUDIT_SAVED.py',terminal/'TERMINAL_REVIEW.json',here/'SAVED_EVIDENCE_BASIS.json',
    root/'scripts/diagnose_joint01_c1_matched_estimator.py',root/'configs/joint_response_01_c1_matched_estimator_20261020.json'):
    name=path.relative_to(root).as_posix();p['source_pins'][name]=hashlib.sha256(path.read_bytes()).hexdigest();inputs[name]=str(path)
config='configs/joint_response_01_c1_draw_precision_20261020.json'
(root/config).write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8',newline='\n')
streams=[p['bank_seed']+1000000*s+1000*r+3*c+o for s in range(2) for r in range(6) for c in range(102) for o in range(3)]
assert len(streams)==len(set(streams))==3672
assert p['solver']==json.loads((root/'configs/joint_response_01_c1_matched_control_20261020.json').read_bytes())['solver']
assert 'theta=proposal' not in text and 'teacher_accepted_joint_updates' not in text
static=dict(status='PASS',UTC=now,syntax_pass=True,source_stream_seeds_unique=3672,full_pool_parents=6465,
    chunks=102,max_rows_per_chunk=2048,unchanged_solver_and_candidate_rules=True,parameter_updates=0,
    nested_draws_verified_by_registered_repeat_interleave_order=True,operation_budget=p['operation_budget'])
(here/'STATIC_REVIEW.json').write_text(json.dumps(static,indent=2)+'\n',encoding='utf-8',newline='\n')
manifest=json.loads((old/'SOURCE_MANIFEST.json').read_bytes())
source_count=len(set(manifest['files'])|set(inputs)|{config,script,contract})
ledger['jobs'][-1].update(status='completed_independently_audited_no_updates',ended_UTC=remote['native_exit']['ended_utc'],native_exit_code=0,
    terminal_review=str((terminal/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),
    execution_counts=review['execution_counts'],source_files_verified=760,scientific_acceptance=False)
ledger['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,config=config,config_sha256=digest,
    host='rylan@100.82.231.102',remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_draw_precision_20261020',
    resources=p['resources'],operation_budget=p['operation_budget'],parameter_updates=0,scientific_acceptance=False))
assert ledger['jobs'][:124]==prior_jobs[:124]
for name in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    d=ledger if name=='jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_C1_draw_precision_registered',active_scientific_jobs=0,pending_job_id=p['job_id'],
        active_job_id=None,active_run_id=None,training_pid=None,supervisor_pid=None,updated_UTC=now,last_verified_UTC=remote['UTC'],
        current_stage=p['stage'],active_work_package=p['stage'],scientific_acceptance=False,
        current_action='Matched estimator audited; registered paired 4/32-draw direction diagnostic at fixed endpoints.',
        next_action='Dispatch, verify startup and measure progress before estimating one wake.',
        followup_status='NO_PENDING_WAIT',followup_automation_id=None,followup_scheduled_local=None,followup_next_run_local=None,next_wake_local=None)
    d['joint_response_01'].update(c1_matched_estimator_native_exit_code=0,c1_matched_estimator_independent_audit_pass=True,
        c1_matched_estimator_terminal_review=str((terminal/'TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),
        c1_matched_estimator_execution_counts=review['execution_counts'],c1_matched_estimator_parameter_updates=0,
        c1_draw_precision_config=config,c1_draw_precision_config_sha256=digest,c1_draw_precision_started=False,
        c1_draw_precision_scientific_acceptance=False)
    (root/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\r\n')
record=dict(status='REGISTERED_BEFORE_EXECUTION',UTC=now,job_id=p['job_id'],config=config,config_sha256=digest,
    source_files_expected=source_count,source_pins=len(p['source_pins']),science_started=False,new_model_updates=0,prior_124_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
