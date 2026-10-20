from pathlib import Path
from datetime import datetime, timezone
import ast, copy, hashlib, json, shutil

root = Path('E:/GAN/research/N01')
here = Path(__file__).resolve().parent
previous = root/'receipts/joint_response_01_20261020/c1_precision_pilot_01'
review = json.loads((previous/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status'] == 'PASS_SAVED_PRECISION_PILOT_AUDIT'
assert review['matched_recovery_pass'] and review['original_target_report_pass']
ledger = json.loads((root/'jobs.json').read_bytes())
old_jobs = copy.deepcopy(ledger['jobs'])
assert len(old_jobs) == 128 and ledger['active_scientific_jobs'] == 0
assert old_jobs[-1]['status'] == 'completed_audited_positive_control_all_gates_pass'
(here/'before_registration').mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'README.md', 'JOINT_RESPONSE_01_PLAN.md'):
    shutil.copy2(root/name, here/'before_registration'/name)
now = datetime.now(timezone.utc).isoformat()
contract = 'docs/JOINT_RESPONSE_01_C1_ORIGINAL_PRECISION_20261020.md'
(root/contract).write_text('''# JOINT_RESPONSE_01：原教师32 draw直接恢复确认

2026-10-20前瞻登记。32 draw同家族正对照已独立复算：8联合方向、11候选、7接受更新；同家族C1和原教师报告各36/36硬门、18/18条件精度门通过。原生exit 0，耗时1012.74秒，RSS峰6.22GiB。该端点拟合的是C1教师，尚未验证直接拟合原教师时同一方法的恢复能力。

本次从原跨家族完整父集试验的两头ITER11冻结端点出发，直接拟合原JointResponse01教师。两端点取c1_fullpool_inputs/seed0_joint_ITER11.npz和seed1_joint_ITER11.npz，不使用同家族试验末端。学生仍为C1；已知参数只供教师生成，学生仅接收无配对观测分布。两头各固定4轮，沿用已验证的fit_track_precision函数；不按报告门选择中间端点、增加轮数或直接指定随机参数。

每轮FIT与CHECK覆盖全部6465拟合父×32 draw；64父分块、102块，128报告父排除。原18特征因子精确重现，全部54坐标联合求解。有限差分0.04、ridge 0.02、半径0.4、1/0.5/0.25回溯及原接受／信赖域／宏块门完全不变。继续每次仅保留一个完整随机库，顺序释放并调用malloc_trim。原精度试验与历史4 draw失败分别保留。

父排列种子2026202601+1000*头+100*迭代+角色；FIT/CHECK库202630250000+1000000*头+10000*迭代+1000*角色+3*块。最终硬观察库202640250000+1000000*头+10000*库+1000*源目标角色；2048 draw精度库2026202641至2026202644。9792拟合子流、24硬评价子流及4精度流互斥，也与已执行的正对照／方向诊断分离；统计因子仍用原种子复现。

每头4轮完成后，M0、联合候选和原教师在两套新64 draw硬库及两套2048 draw条件核库评价。成功要求两头合计原36硬门及18条件精度门全部通过，门限不变；不以低draw条件方差替代高精度报告。既有物理基线与固定响应人口压力试验保留。已暴露A0/M0和128报告父只支持冻结系统后验评价；项目科学接受、REAL联合改善和阶段5另行验证，本次不自动开放REAL。

上限：4CPU、GPU分配2.75GiB、RSS12GiB，内部1500秒、外部1740秒加20秒，输出192MiB；不自动重启或延长。432差分列、44064分块列、8联合方向、最多24候选／8更新；3278库分配、3274电子学调用、6698626噪声行、20095878扩展行、61977616标准正态数；生成95410调用／193750402行，观察281574调用／571420806行；条件信号98304行、原生指标589824行。实际计数按候选数复算。

工程验收要求原生0退出、输入输出哈希、预算／角色覆盖、保存差分和完整联合解、候选风险／信赖域及参数链复算。科学门依据实际完整硬观察和条件核汇总，保存矩阵复算不替代波形再生成。仅在正确启动后的实际长等待安排一次苏醒，按当次实测速度、剩余轮数和原实测报告成本估时并加余量。
''', encoding='utf-8', newline='\n')
prior_name = 'configs/joint_response_01_c1_precision_pilot_20261020.json'
prior = json.loads((root/prior_name).read_bytes())
original = json.loads((root/'configs/joint_response_01_c1_estimator_20261020.json').read_bytes())
p = copy.deepcopy(prior)
p.update(schema='n01/joint-response01-c1-original-precision/v1', job_id='JOINT_RESPONSE_01_C1_ORIGINAL_PRECISION_20261020',
    registered_UTC=now, stage='3_C1_original_precision_confirmation', method_contract=contract,
    target_family='original', positive_control_only=False, direct_original_target_fitted=True,
    order_seed=2026202601, bank_seed=202630250000, evaluation_bank_seed=202640250000,
    precision_bank_seeds=[[2026202641, 2026202642], [2026202643, 2026202644]],
    endpoint_iterations=original['endpoint_iterations'])
fit_passes = 8*(110+2+3)
fit_rows, fit_calls = fit_passes*6465*32, fit_passes*102
noise_rows = 8*4*6465*32+2*6465+8*8192
p['operation_budget'] = dict(checkpoint_loads=6, TRAIN_archives_decoded=1, PCA_archives_decoded=1, DEV_waveforms_decoded=0,
    old_A_calls=52, old_A_rows=6593, M0_head_calls=104, M0_head_rows=13186,
    allocated_banks=8*4*102+2+8+4, shape_standard_normal_values=noise_rows*8+4*128*2048*8,
    noise005_calls=8*4*102+2+8, noise005_rows=noise_rows, extension_rows=3*noise_rows,
    joint_response_calls=fit_calls+2+4*8+4*128*3,
    joint_response_rows=fit_rows+2*6465+4*8*8192+4*128*3*2048,
    generated_observation_calls=3*fit_calls+6+4*12,
    generated_observation_rows=3*fit_rows+6*6465+4*12*8192,
    finite_difference_columns=432, fullpool_FIT_chunk_columns=44064, joint_direction_solves=8,
    joint_candidate_trials=24, teacher_accepted_joint_updates=8,
    conditional_signal_rows=4*3*8192, native_metric_rows=4*3*3*2*8192,
    feature_maps_constructed=18, full_fit_parent_banks=32, REAL_response_updates=0, AD_queries=0, classifier_fits=0)
inputs = {str((previous/name).relative_to(root).as_posix()): str(previous/name)
          for name in ('AUDIT_COMPLETE.py', 'terminal_01/TERMINAL_REVIEW.json')}
script = 'scripts/run_joint_response01_c1_original_precision.py'
for name in (prior_name, script, contract, *inputs):
    p['source_pins'][name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
training = [p['bank_seed']+1000000*h+10000*i+1000*r+3*c+o for h in range(2) for i in range(4) for r in range(4) for c in range(102) for o in range(3)]
evaluation = [p['evaluation_bank_seed']+1000000*h+10000*l+1000*r+o for h in range(2) for l in range(2) for r in range(2) for o in range(3)]
precision = sum(p['precision_bank_seeds'], [])
assert len(training) == len(set(training)) == 9792 and len(set(training+evaluation+precision)) == 9820
old_streams = [prior['bank_seed']+1000000*h+10000*i+1000*r+3*c+o for h in range(2) for i in range(4) for r in range(4) for c in range(102) for o in range(3)]
old_streams += [prior['evaluation_bank_seed']+1000000*h+10000*l+1000*r+o for h in range(2) for l in range(2) for r in range(2) for o in range(3)]
old_streams += sum(prior['precision_bank_seeds'], [])
direction = json.loads((root/'configs/joint_response_01_c1_draw_precision_20261020.json').read_bytes())
old_streams += [direction['bank_seed']+1000000*h+1000*r+3*c+o for h in range(2) for r in range(6) for c in range(102) for o in range(3)]
assert not set(training+evaluation+precision) & set(old_streams)
assert p['solver'] == prior['solver'] and p['success'] == prior['success'] and p['iterations'] == prior['iterations'] == 4
assert p['feature_factors'] == original['feature_factors']
endpoint_map = json.loads((root/'receipts/joint_response_01_20261020/c1_estimator_01/INPUT_MAP.json').read_bytes())
for name in p['endpoint_iterations']:
    assert hashlib.sha256(Path(endpoint_map[name]).read_bytes()).hexdigest() == p['source_pins'][name]
source = (root/script).read_text()
ast.parse(source)
assert 'from run_joint_response01_c1_precision_pilot import fit_track_precision, evaluate_fresh_seed' in source
assert 'fit_track_precision(p, setup, teacher, truth, student, initial' in source
config = 'configs/joint_response_01_c1_original_precision_20261020.json'
(root/config).write_text(json.dumps(p, indent=2)+'\n', encoding='utf-8', newline='\n')
digest = hashlib.sha256((root/config).read_bytes()).hexdigest()
(here/'INPUT_MAP.json').write_text(json.dumps(inputs, indent=2)+'\n', encoding='utf-8', newline='\n')
(here/'STATIC_REVIEW.json').write_text(json.dumps(dict(status='PASS', UTC=now, syntax_pass=True,
    frozen_fitter_and_hard_evaluator_imported_unchanged=True, original_teacher_and_original_endpoints_bound=True,
    unchanged_solver_success_and_iteration_count=True, original_endpoint_hashes_verified=True,
    training_substreams_unique=9792, evaluation_substreams_unique=24, precision_streams_unique=4,
    streams_disjoint_from_previous_pilot_and_direction_diagnostic=True,
    operation_budget=p['operation_budget'], scientific_acceptance=False), indent=2)+'\n', encoding='utf-8', newline='\n')
manifest = json.loads((previous/'SOURCE_MANIFEST.json').read_bytes())
sources = len(set(manifest['files']) | set(inputs) | {config, script, contract})
ledger['jobs'].append(dict(job_id=p['job_id'], status='registered_not_started', registered_UTC=now, config=config,
    config_sha256=digest, host='rylan@100.82.231.102',
    remote_root='/home/rylan/projects/ej276d-n01/joint_response01_c1_original_precision_20261020',
    resources=p['resources'], operation_budget=p['operation_budget'], fixed_iterations_per_head=4,
    known_teacher_labels_not_used_for_fitting=True, original_cross_family_failure_preserved=True,
    automatic_REAL_promotion=False, scientific_acceptance=False))
assert ledger['jobs'][:128] == old_jobs
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = ledger if name == 'jobs.json' else json.loads((root/name).read_bytes())
    d.update(status='JOINT_RESPONSE_01_C1_original_precision_registered', active_scientific_jobs=0,
        pending_job_id=p['job_id'], active_job_id=None, active_run_id=None, training_pid=None, supervisor_pid=None,
        updated_UTC=now, current_stage=p['stage'], active_work_package=p['stage'], scientific_acceptance=False,
        current_action='Registered direct original-teacher recovery with frozen 32-draw fitter and four iterations per head.',
        next_action='Verify deployment, dispatch once, verify native startup and estimate remaining wall time for one wake.',
        followup_status='NO_PENDING_WAIT', followup_automation_id=None, followup_scheduled_local=None,
        followup_next_run_local=None, next_wake_local=None)
    d['joint_response_01'].update(c1_original_precision_config=config,
        c1_original_precision_config_sha256=digest, c1_original_precision_started=False)
    (root/name).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\r\n')
record = dict(status='REGISTERED_BEFORE_EXECUTION', UTC=now, job_id=p['job_id'], config=config,
    config_sha256=digest, source_files_expected=sources, source_pins=len(p['source_pins']),
    science_started=False, prior_128_jobs_unchanged=True)
(here/'REGISTRATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
