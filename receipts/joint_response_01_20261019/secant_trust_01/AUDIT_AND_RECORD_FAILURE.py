from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np

ROOT = Path('E:/GAN/research/N01')
HERE = Path(__file__).resolve().parent
p = json.loads((ROOT / 'configs/joint_response_01_secant_trust_20261019.json').read_text())
review = json.loads((HERE / 'FAILURE_COLLECTION_REVIEW.json').read_text())
report = json.loads((HERE / 'failed_collected/run_01/seed0_SECANT_TRUST.json').read_text())
with np.load(HERE / 'failed_collected/run_01/seed0_SECANT_TRUST.npz', allow_pickle=False) as archive:
    arrays = {key: archive[key].copy() for key in archive.files}
for key, value in arrays.items():
    assert np.isfinite(value).all(), key
roles_path = ROOT / 'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(roles_path.read_bytes()).hexdigest() == p['source_pins'][p['roles_file']]
with np.load(roles_path, allow_pickle=False) as roles:
    for label in ('FIT', 'CHECK', 'FRESH'):
        assert np.isin(arrays[label + '_source_parents'], roles['source_train_rows']).all()
        assert np.isin(arrays[label + '_real_rows'], roles['real_train_rows']).all()
theta = arrays['theta']
prefix = 'raw_radius_control_trial0_'
step = arrays[prefix + 'proposal'] - theta
np.testing.assert_allclose(arrays[prefix + 'predicted_residual'],
    arrays['FIT_baseline_residual'] + arrays[prefix + 'jacobian'] @ step, rtol=1e-12, atol=1e-13)
np.testing.assert_allclose(arrays[prefix + 'predicted_native'],
    arrays['FIT_baseline_native'] + arrays[prefix + 'native_jacobian'] @ step, rtol=1e-12, atol=1e-13)
trial = report['arms']['raw_radius_control']['trials'][0]
for label in ('FIT', 'CHECK'):
    check = trial['checks'][label]
    native = arrays[prefix + label + '_actual_native']
    ratio = (native / arrays[label + '_baseline_native']).reshape(3, 3)
    anchor = (native / arrays[label + '_M0_anchor_native']).reshape(3, 3)
    np.testing.assert_allclose(check['native_ratios'], ratio, rtol=1e-13)
    np.testing.assert_allclose(check['initial_anchor_ratios'], anchor, rtol=1e-13)
    native_pass = all((v[:, :2] <= 1.01).all() and (v[:, 2] <= 1.10).all() for v in (ratio, anchor))
    assert bool(native_pass) == check['native_pass'] == False
    L = arrays[label + '_cholesky']
    for kind, vector, recorded in (
            ('baseline', arrays[label + '_baseline_residual'], report['baseline_risks'][label]),
            ('candidate', arrays[prefix + label + '_actual_residual'], check['risks'])):
        for ih, h in enumerate(('fast', 'slow', 'common_level')):
            white = vector[ih * len(L):(ih + 1) * len(L)] * np.sqrt(3)
            raw = L @ white
            np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-12)
            for ib, block in enumerate(('wave', 'charge_morphology', 'peak_codes', 'baseline', 'joint')):
                s = slice(ib * 56, (ib + 1) * 56) if ib < 4 else slice(224, 272)
                value = np.mean(raw[s]**2 / np.square(L).sum(1)[s])
                np.testing.assert_allclose(recorded[h][block], value, rtol=1e-11, atol=1e-13)
    old = report['baseline_risks'][label]
    new = check['risks']
    mean_old = np.mean([v['joint_whitened'] for v in old.values()])
    mean_new = np.mean([v['joint_whitened'] for v in new.values()])
    blocks_pass = all(new[h][b] <= 1.15 * old[h][b] + 2. / 4096 for h in old
        for b in ('wave', 'charge_morphology', 'peak_codes', 'baseline'))
    trust = check['trust']
    trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(
        n for flags in trust['flag_count_changes'].values() for n in flags.values()) <= max(4, .005 * 4096)
    assert trust_pass == trust['pass']
    assert bool(trust_pass and mean_new <= .99 * mean_old and blocks_pass) == check['original_joint_accept']
    assert not check['accepted']
review.update(status='FAILED_SOLVER_SAVED_CONTROL_NUMERIC_AUDIT_PASSED',
    finite_arrays_verified=len(arrays), fitting_role_rows_verified=3 * (1024 + 4096),
    raw_control_predictions_recomputed=2, raw_control_risk_views_recomputed=12,
    raw_control_native_and_acceptance_recomputed=True,
    raw_control_FIT_wave_vs_current=trial['checks']['FIT']['native_ratios'],
    raw_control_CHECK_wave_vs_current=trial['checks']['CHECK']['native_ratios'],
    corrected_trials_numeric_audit_possible=False,
    output_trust_waveforms_not_saved=True,
    solver_scale_review='SOLVER_SCALE_REVIEW.json', scientific_acceptance=False)
(HERE / 'TERMINAL_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
now = datetime.now(timezone.utc).isoformat()
backup = HERE / 'before_failure_update'
backup.mkdir()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json', 'JOINT_RESPONSE_01_PLAN.md', 'README.md'):
    shutil.copy2(ROOT / name, backup / name)
relative = HERE.relative_to(ROOT).as_posix()
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    path = ROOT / name
    obj = json.loads(path.read_text())
    assert obj['pending_next_run_id'] == p['job_id']
    obj.update(status='JOINT_RESPONSE_01_secant_trust_failed_solver', active_run_id=None,
        latest_run_id=p['job_id'], pending_next_run_id=None, active_scientific_jobs=0, updated_UTC=now,
        next_action='Register a bounded equivalent-variable solver and per-trial persistence repair, preserving this failed attempt and all recomputation costs.')
    joint = obj['joint_response_01']
    joint.update(status='secant_trust_failed_solver', secant_trust_started=True,
        secant_trust_terminal_review=relative + '/TERMINAL_REVIEW.json',
        secant_trust_native_exit_code=1, secant_trust_scientific_acceptance=False,
        secant_trust_committed_updates=0, secant_trust_last_checked_UTC=now)
    joint['stages']['4'] = 'native_REAL_failed_both_seeds_secant_trust_solver_repair_pending'
    if name == 'jobs.json':
        job = obj['jobs'][-1]
        assert job['job_id'] == p['job_id']
        job.update(status='failed_solver_partial_control_audit_passed', scientific_process_started=True,
            deployment_verified=True, native_import_passed=True, source_files_verified=645,
            native_exit_code=1, correct_startup_verified=False,
            started_UTC=review['native_exit']['started_utc'], ended_UTC=review['native_exit']['ended_utc'],
            last_checked_UTC=now, terminal_review=relative + '/TERMINAL_REVIEW.json',
            execution_counts=review['failure']['execution_counts'], direction_solve_attempts=4,
            complete_successful_direction_solves=3, failed_direction_solve_attempts=1,
            elapsed_seconds=review['failure']['elapsed_seconds'], peak_RSS_bytes=review['native_exit']['peak_rss_bytes'],
            dispatch_receipt=relative + '/DISPATCH_RECEIPT.json', preflight_attempts=1)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert json.loads(path.read_text())['active_scientific_jobs'] == 0
paragraph = ('\n联合割线试验首次执行于10:45:43 UTC派发，6.41秒后因第4次SLSQP求解的线搜索错误退出。'
    '645源码/9输出哈希及已保存首步对照的数值审核通过，3次候选实际检查、零提交更新；'
    '中途两次校正候选明细因仅在支路结束保存而缺失，计算次数及报错保留。'
    '[失败审核](receipts/joint_response_01_20261019/secant_trust_01/TERMINAL_REVIEW.json)。'
    '独立16次CPU求解检查中，原变量有1次同类失败，按半径归一化的8个等价问题全部可行；'
    '原第4次矩阵未保存，不能称其精确故障重放。拟单独登记变量尺度和逐试探持久化修复，原门不变。\n')
for name in ('JOINT_RESPONSE_01_PLAN.md', 'README.md'):
    path = ROOT / name
    path.write_text(path.read_text(encoding='utf-8').rstrip() + '\n' + paragraph, encoding='utf-8')
print(json.dumps({key: review[key] for key in ('status', 'finite_arrays_verified', 'fitting_role_rows_verified', 'scientific_acceptance')}))
