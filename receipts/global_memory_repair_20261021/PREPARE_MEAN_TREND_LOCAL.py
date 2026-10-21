from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
prior = here / 'mean_trend_fix02'
out = here / 'mean_trend_local_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
native = json.loads((prior / 'RESULT.json').read_bytes())
exit_record = json.loads((prior / 'exit_receipt.json').read_bytes())
endpoint = json.loads((prior / 'TRAINING_RESULT.json').read_bytes())
analysis = json.loads((prior / 'DIRECTION_ANALYSIS.json').read_bytes())
assert exit_record['exit_code'] == 0 and native['committed_optimizer_updates'] == 0
assert analysis['endpoint_equals_inherited'] and native['execution_counts']['optimizer_directions'] == 3
closure = dict(status='COMPLETE_THREE_NATIVE_DIRECTIONS_REJECTED', UTC=stamp, native_exit_code=0,
    elapsed_seconds=native['elapsed_seconds'], actual_execution_counts=native['execution_counts'],
    committed_optimizer_updates=0, inherited_checkpoint_updates=10, endpoint_equals_inherited=True,
    scientific_acceptance=False, failed_checks=[k for k,v in endpoint['endpoint']['B']['pilot_checks'].items() if not v],
    cause='Every measured direction worsens at least one common high-Q conditional-wave cell, so each native cone contains only zero.',
    blocking_duals=analysis['finite_direction_geometry'],
    next_step='One newly registered local direction, adding only3 actually blocking cell gradients and restoring the previously used0.05 native probe radius. No extension of closed job166.',
    followup_deleted_no_waiting_work=True, figures_pending=True)
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
old_path = root / 'configs/mean_trend_repair_fix02_20261021.json'
p = json.loads(old_path.read_bytes())
old_remote = p['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/mean_trend_local_20261021'
p.pop('engineering_recovery')
p.update(job_id='N01_MEAN_TREND_LOCAL_20261021', registered_UTC=stamp,
    directions=1, stop_consecutive_rejections=1, secant_radius=.05,
    active_cells=p['active_cells'] + ['Q_wave_L16_13', 'Q_wave_L32_24', 'Q_wave_L16_12'],
    endpoint_seed=2166102100,
    evidence=dict(parent_job=166, failed_direction_count=3, accepted_updates=0,
        same_TRAIN_direction_bank=True, native_linear_cone_zero=True,
        missing_directions='Positive native changes in L16_13, L32_24 and L16_12 certify the finite cones were blocked.',
        local_radius_reason='At unit probe radius even Q-wave descent increased its own hard risk and PSD rose8x-22x for several directions; use previously implemented0.05 local radius once, without a radius sweep.',
        budget_rule='Exactly1 new direction and at most11 probes/3 hard trials; no automatic additional directions regardless of result.'))
p['optimizer_contract']['method'] = 'Adam plus wave/Q-wave/trend,5 evidence-derived conditional cells, and2 ADC gradients; at most11 native probes.'
p['optimizer_contract']['stop'] = 'Exactly1 direction, at most11 native probes at radius0.05 and3 hard combination scales; no sweep, no extension.'
p['operation_budget'] = dict(response_rows=2999760, conditioned_mean_rows=150000,
    optimizer_directions=1, committed_optimizer_updates=1, native_secant_probes=11,
    joint_hard_trials=3, mean_adjoint_rows=64650,
    derivation='Initial1 pair + gradient2 pairs +11 probes +3 trials =17*103440;3 independent endpoint models add1241280.')
p['resources'].update(internal_seconds=900, hard_wall_seconds=960, output_MiB=128)
p['storage']['remote_root'] = remote
p['measured_budget_basis'] = dict(prior_gradient_seconds=45.7, prior_adjoint_directions=7,
    this_adjoint_directions=10, estimated_gradient_seconds=66, mean_native_probe_seconds=13,
    max_probes=11, endpoint_seconds=152, expected_seconds=425,
    wake_rule='Confirm actual native startup, use measured first-stage timing plus remaining workload and margin.')
path = root / 'configs/mean_trend_local_20261021.json'
path.write_bytes((json.dumps(p, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
snapshot = out / 'before_registration'
snapshot.mkdir()
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    raw = (root / name).read_bytes()
    (snapshot / name).write_bytes(raw)
    d = json.loads(raw)
    d.update(status='N01_MEAN_TREND_local_direction_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=p['job_id'], pending_registered_job_id=p['job_id'],
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0,
        current_job_ended_UTC=exit_record['ended_utc'], updated_UTC=stamp, execution_state_asof_UTC=stamp,
        followup_automation_id=None, followup_automation_status='DELETED', followup_status='DELETED_NO_WAITING_WORK',
        followup_scheduled_local=None, next_wake_local=None,
        current_action='Job166 closed at0 updates; one local correction registered from actual high-Q blocking cells.',
        next_action='Run one native local step with all original gates; collect and plot its endpoint.')
    d['global_memory_repair'].update(status='mean_trend_local_registered', active_scientific_jobs=0,
        config=path.relative_to(root).as_posix(), config_sha256=sha,
        mean_trend_closure=(prior / 'CLOSURE_RECEIPT.json').relative_to(root).as_posix(),
        current_native_execution_counts=native['execution_counts'], counts_asof_UTC=exit_record['ended_utc'],
        next_step_design=path.relative_to(root).as_posix())
    if name == 'jobs.json':
        assert len(d['jobs']) == 166
        d['jobs'][-1].update(status='completed_three_native_directions_rejected', native_exit_code=0,
            actual_execution_counts=native['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=(prior / 'CLOSURE_RECEIPT.json').relative_to(root).as_posix())
        d['jobs'].append(dict(job_id=p['job_id'], status='registered', registered_UTC=stamp,
            config=path.relative_to(root).as_posix(), config_sha256=sha,
            resources=p['resources'], operation_budget=p['operation_budget'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write('\njob166原生exit0，609.04秒，3方向/22探针全部拒绝，0更新，独立终点与继承状态完全一致。'
        '三个固定训练库的有限方向锥分别被L16_13、L32_24、L16_12阻塞。'
        'job167仅登记一次新局部方向：补这三单元梯度、使用此前0.05原生探针尺度，原终点门槛不变，不追加job166步数或扫描参数。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ['REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py']:
    code = (prior / name).read_text('utf-8').replace(old_remote, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    code = code.replace("'5460s'", "'960s'")
    (out / name).write_bytes(code.encode())
collector = (here / 'COLLECT_MEAN_TREND_FIX02.py').read_text('utf-8').replace("'mean_trend_fix02'", "'mean_trend_local_01'").replace(old_remote, remote)
(here / 'COLLECT_MEAN_TREND_LOCAL.py').write_bytes(collector.encode())
with tarfile.open(out / 'runtime.tar', 'w') as archive:
    for member in list((root / 'src/n01').glob('*.py')) + list((root / 'scripts').glob('*.py')) + [path]:
        archive.add(member, arcname=member.relative_to(root).as_posix())
deploy = """import sys,tarfile
from pathlib import Path
root=Path(REMOTE)
root.mkdir(parents=True,exist_ok=False)
(root/'cache').mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive: archive.extractall(root,filter='data')
print('RUNTIME_DEPLOYED')
""".replace('REMOTE', repr(remote))
with (out / 'runtime.tar').open('rb') as stream:
    subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deploy)],
        stdin=stream, capture_output=True, check=True, timeout=60)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(dict(status='REGISTERED_AND_DEPLOYED',
    UTC=stamp, job_number=167, config_sha256=sha, prior_job_updates=0, max_new_updates=1), indent=2) + '\n').encode())
print(json.dumps(dict(status='REGISTERED_AND_DEPLOYED', job_number=167, config_sha256=sha)))
