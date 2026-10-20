from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root = Path('E:/GAN/research/N01')
here = Path(__file__).resolve().parent
previous = root/'receipts/joint_response_01_20261020/c1_original_precision_01'
positive = root/'receipts/joint_response_01_20261020/c1_precision_pilot_01'
review = json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status'] == 'PASS_SAVED_ORIGINAL_PRECISION_AUDIT' and not review['original_recovery_pass']
ledger = json.loads((root/'jobs.json').read_bytes())
old_jobs = copy.deepcopy(ledger['jobs'])
assert len(old_jobs) == 129 and ledger['active_scientific_jobs'] == 0
assert old_jobs[-1]['status'] == 'completed_audited_scientific_failure'
(here/'before_registration').mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'README.md', 'JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name, here/'before_registration'/name)
now = datetime.now(timezone.utc).isoformat()
diagnosis = json.loads((previous/'terminal_01/SAVED_FIT_DIAGNOSIS.json').read_bytes())
basis = dict(UTC=now, source_review=str((previous/'terminal_01/TERMINAL_REVIEW.json').relative_to(root)).replace('\\','/'),
    original_recovery_pass=False, hard_pass=36, precision_pass=16, precision_total=18,
    failed_gates=review['by_target']['original']['failed_gates'],
    seed1_accepted_update=diagnosis['tracks']['seed1_joint']['iterations'][1],
    observed_mean_error_increased_across_different_reporting_banks=True,
    same_bank_endpoint_objective_ranking_not_yet_measured=True,
    diagnostic_question='Do frozen conditionally accurate and inaccurate endpoints receive different rankings under the same full-parent hard-observation objective?',
    parameter_updates=0, scientific_acceptance=False)
(here/'SAVED_EVIDENCE_BASIS.json').write_text(json.dumps(basis, indent=2)+'\n', encoding='utf-8', newline='\n')
contract = 'docs/JOINT_RESPONSE_01_C1_OBJECTIVE_ALIGNMENT_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：冻结端点的观测目标排序检验

2026-10-20前瞻登记。直接原教师高精度恢复已独立复算：36/36硬门、16/18条件精度门；seed1两库实际条件均值误差5.7628%/5.7221%超过原5%门。该头唯一接受的联合更新使当次CHECK风险比降至0.91524，原宏块与信赖域通过。此前正对照端点对原教师的条件均值误差约1.1%，但两项试验的评价随机库不同；尚不能仅凭分数判断目标排序还是求解路径的问题。

本次零参数更新，固定五个源生成器／向量：original/oracle、C1/known-truth、原跨家族ITER11初值、已通过条件核的正对照末端、本次直接原教师末端。两个目标是original和matched_C1教师；各家族使用相同的目标随机库，五源共用同一源随机库，源与目标库独立。已知真值仅作诊断控制，不进入候选构造或参数校正。

两头各4套独立库，每套遍历全部6465拟合父×32 draw，64父一块102块；128报告父排除。原FIT/CHECK/FINAL三个映射均使用，并逐值复现18个原特征因子。保留三种观察假设和原完整硬观察，只比较固定端点的联合目标及分块贡献，不修改特征、尺度、模型或验收门。

父排列2027202701+1000*头+100*重复+源目标角色；随机库202650270000+1000000*头+10000*重复+1000*角色+3*块，4896子流互斥。目标两家族有意配对，源五标签有意配对；重复之间、源与目标之间独立。每次仅保留一个完整32draw库，先目标后源，按原分块顺序汇总并释放空闲页。新随机流与前项校准分离。

保存每套原特征均值、残差、映射因子、参数与父排列，计算4库交叉乘积U风险、聚合128draw的MC项及配对端点风险差；四库jackknife仅作描述，不据其给出强显著性结论。独立复算保留完整白化协方差、分块交叉项及目标家族差异。若条件较差端点稳定获得更低风险，证据支持目标排序与条件恢复之间有张力；若条件较好端点风险更低，支持检查求解路径；混合或接近的结果保持不确定。任何结果都不自动选择新端点、调整参数、改变门或开放REAL。

上限4CPU、GPU分配2.75GiB、RSS12GiB、内部300秒、外部420秒加20秒，输出32MiB。6检查点、1 TRAIN档案、1 PCA档案；A0 52调用/6593行，M0 104调用/13186行；1634库分配、1634噪声调用/3323010行、9969030扩展行、26584080标准正态数；生成5714调用/11598210行、观察17142调用/34794630行。18特征映射、16完整父库、8独立库对、40固定端点评价、16目标家族评价；FD、求解、候选、模型更新、REAL、DEV和分类器均0。预算停止不自动重启或延长；当前短步骤立即完成，只有实际等待才安排估时后单次苏醒。
''', encoding='utf-8', newline='\n')
prior_name = 'configs/joint_response_01_c1_original_precision_20261020.json'
prior = json.loads((root/prior_name).read_bytes())
p = copy.deepcopy(prior)
p.update(schema='n01/joint-response01-c1-objective-alignment/v1', job_id='JOINT_RESPONSE_01_C1_OBJECTIVE_ALIGNMENT_20261020',
    registered_UTC=now, stage='3_C1_frozen_objective_alignment', method_contract=contract, iterations=0,
    target_family='both_frozen', direct_original_target_fitted=False, independent_repeats=4,
    order_seed=2027202701, bank_seed=202650270000)
for key in ('precision_draws_per_parent', 'precision_bank_seeds', 'evaluation_bank_seed', 'fixed_iteration_count_no_report_based_endpoint_selection'):
    del p[key]
p['resources'].update(internal_seconds=300, hard_wall_seconds=420, output_MiB=32)
noise_rows = 2*6465+8*2*6465*32
calls, rows = 2+8*7*102, 2*6465+8*7*6465*32
p['operation_budget'] = dict(checkpoint_loads=6, TRAIN_archives_decoded=1, PCA_archives_decoded=1, DEV_waveforms_decoded=0,
    old_A_calls=52, old_A_rows=6593, M0_head_calls=104, M0_head_rows=13186,
    allocated_banks=1634, shape_standard_normal_values=noise_rows*8, noise005_calls=1634, noise005_rows=noise_rows,
    extension_rows=3*noise_rows, joint_response_calls=calls, joint_response_rows=rows,
    generated_observation_calls=3*calls, generated_observation_rows=3*rows,
    finite_difference_columns=0, fullpool_FIT_chunk_columns=0, joint_direction_solves=0, joint_candidate_trials=0,
    teacher_accepted_joint_updates=0, conditional_signal_rows=0, native_metric_rows=0, feature_maps_constructed=18,
    full_fit_parent_banks=16, fixed_endpoint_evaluations=40, target_family_evaluations=16, independent_bank_pairs=8,
    REAL_response_updates=0, AD_queries=0, classifier_fits=0)
inputs = {str((previous/name).relative_to(root).as_posix()): str(previous/name)
          for name in ('AUDIT_COMPLETE.py', 'terminal_01/TERMINAL_REVIEW.json', 'terminal_01/SAVED_FIT_DIAGNOSIS.json')}
for label, folder in (('positive', positive), ('original', previous)):
    p[label+'_endpoints'] = []
    for seed in range(2):
        name = f'c1_objective_inputs/seed{seed}_{label}_STATE.npz'
        inputs[name] = str(folder/f'terminal_01/collected/run_01/seed{seed}_joint_STATE.npz')
        p[label+'_endpoints'].append(name)
basis_name = (here/'SAVED_EVIDENCE_BASIS.json').relative_to(root).as_posix()
inputs[basis_name] = str(here/'SAVED_EVIDENCE_BASIS.json')
script = 'scripts/diagnose_joint01_c1_objective_alignment.py'
for name in (prior_name, script, contract):
    p['source_pins'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
for name, local in inputs.items():
    p['source_pins'][name] = hashlib.sha256(Path(local).read_bytes()).hexdigest()
streams = [p['bank_seed']+1000000*h+10000*r+1000*s+3*c+o for h in range(2) for r in range(4) for s in range(2) for c in range(102) for o in range(3)]
old_streams = [prior['bank_seed']+1000000*h+10000*r+1000*s+3*c+o for h in range(2) for r in range(4) for s in range(4) for c in range(102) for o in range(3)]
old_streams += [prior['evaluation_bank_seed']+1000000*h+10000*l+1000*s+o for h in range(2) for l in range(2) for s in range(2) for o in range(3)]
assert len(streams) == len(set(streams)) == 4896 and not set(streams)&set(old_streams)
ast.parse((root/script).read_text())
config = 'configs/joint_response_01_c1_objective_alignment_20261020.json'
(root/config).write_text(json.dumps(p, indent=2)+'\n', encoding='utf-8', newline='\n')
digest = hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs, indent=2)+'\n', encoding='utf-8', newline='\n')
(here/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='PASS', UTC=now, syntax_pass=True,
    frozen_endpoints_only=True, original_pooled_three_map_reducer_unchanged=True,
    source_target_independent=True, labels_paired_within_source_bank=True, target_families_paired=True,
    unique_streams=4896, one_full_bank_live=True, parameter_updates=0, operation_budget=p['operation_budget'], scientific_acceptance=False), indent=2)+'\n', encoding='utf-8', newline='\n')
manifest = json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources = len(set(manifest['files']) | set(inputs) | {config, script, contract})
ledger['jobs'].append(dict(job_id=p['job_id'], status='registered_not_started', registered_UTC=now, config=config,
    config_sha256=digest, host='rylan@100.82.231.102', remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_objective_alignment_20261020',
    resources=p['resources'], operation_budget=p['operation_budget'], parameter_updates=0,
    original_cross_family_failure_preserved=True, automatic_REAL_promotion=False, scientific_acceptance=False))
assert ledger['jobs'][:129] == old_jobs
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = ledger if name == 'jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_C1_objective_alignment_registered', active_scientific_jobs=0,
        pending_job_id=p['job_id'], active_job_id=None, active_run_id=None, training_pid=None, supervisor_pid=None,
        updated_UTC=now, current_stage=p['stage'], active_work_package=p['stage'], scientific_acceptance=False,
        current_action='Registered frozen endpoint rankings under original and matched teacher objectives; zero updates.',
        next_action='Verify deployment and execute bounded diagnostic, audit saved residuals and endpoint rankings.',
        followup_status='NO_PENDING_WAIT', followup_automation_id=None, followup_scheduled_local=None,
        followup_next_run_local=None, next_wake_local=None)
    d['joint_response_01'].update(c1_objective_alignment_config=config, c1_objective_alignment_config_sha256=digest, c1_objective_alignment_started=False)
    (root/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\r\n')
record = dict(status='REGISTERED_BEFORE_EXECUTION', UTC=now, job_id=p['job_id'], config=config,
    config_sha256=digest, source_files_expected=sources, source_pins=len(p['source_pins']), science_started=False, prior_129_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
