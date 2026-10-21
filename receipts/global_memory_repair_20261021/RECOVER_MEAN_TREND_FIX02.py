from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
prior = here / 'mean_trend_fix01'
out = here / 'mean_trend_fix02'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
failure = json.loads((prior / 'FAILED.json').read_bytes())
native_exit = json.loads((prior / 'exit_receipt.json').read_bytes())
assert native_exit['exit_code'] == 1 and failure['error_type'] == 'AssertionError'
assert failure['execution_counts'].get('optimizer_directions', 0) == 0
closure = dict(status='ENGINEERING_FAILURE_AT_INITIAL_IDENTITY_BEFORE_OPTIMIZER', UTC=stamp,
    native_exit_code=1, actual_execution_counts=failure['execution_counts'], optimizer_updates=0,
    cause='Recomputed conditioned U-Net differs numerically from frozen cached decoding despite identical inputs;64-row max discrepancy0.000586622V and128-row0.000177189V.',
    fix='Zero-start function difference M_inherited+qbar*(U_train-U_initial), evaluated with the same parent microbatches. GPU synthetic identity0 and adjoint error8.94e-8; original baseline unchanged.',
    scientific_parameters_changed=False, scientific_acceptance=False)
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
old_path = root / 'configs/mean_trend_repair_fix01_20261021.json'
p = json.loads(old_path.read_bytes())
old_remote = p['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/mean_trend_repair_fix02_20261021'
p.update(job_id='N01_MEAN_TREND_REPAIR_FIX02_20261021', registered_UTC=stamp,
    engineering_recovery=dict(parent_job='N01_MEAN_TREND_REPAIR_FIX01_20261021',
        cause=closure['cause'], fix=closure['fix'], previous_optimizer_updates=0,
        repeated_setup_costs_recorded_separately=True, scientific_budget_and_thresholds_unchanged=True))
p['storage']['remote_root'] = remote
p['architecture']['zero_start_function'] = closure['fix']
closure['identity_diagnostic'] = json.loads((prior / 'IDENTITY_DIAGNOSTIC_STDOUT.json').read_bytes())
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
(out / 'GPU_SYNTHETIC_CHECKS.json').write_bytes((prior / 'GPU_DELTA_SYNTHETIC_CHECKS.json').read_bytes())
path = root / 'configs/mean_trend_repair_fix02_20261021.json'
path.write_bytes((json.dumps(p, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
snapshot = out / 'before_registration'
snapshot.mkdir()
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    raw = (root / name).read_bytes()
    (snapshot / name).write_bytes(raw)
    d = json.loads(raw)
    d.update(status='N01_MEAN_TREND_engineering_recovery_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=p['job_id'], pending_registered_job_id=p['job_id'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='Preserved job165 initial-identity failure; zero-start functional difference registered at unchanged scientific budget.')
    d['global_memory_repair'].update(status='mean_trend_engineering_recovery_registered',
        config=path.relative_to(root).as_posix(), config_sha256=sha,
        mean_trend_engineering_failure=(prior / 'CLOSURE_RECEIPT.json').relative_to(root).as_posix(),
        next_step_design=path.relative_to(root).as_posix())
    if name == 'jobs.json':
        assert len(d['jobs']) == 165
        d['jobs'][-1].update(status='engineering_failed_before_optimizer', native_exit_code=1,
            actual_execution_counts=failure['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=native_exit['ended_utc'], closure=(prior / 'CLOSURE_RECEIPT.json').relative_to(root).as_posix())
        d['jobs'].append(dict(job_id=p['job_id'], status='registered', registered_UTC=stamp,
            config=path.relative_to(root).as_posix(), config_sha256=sha, resources=p['resources'],
            operation_budget=p['operation_budget'], kind='conditioned_mean_trend_engineering_recovery', scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write('\njob165原生起点重现检查失败，0更新；追加有界固定权重诊断确认输入一致而U-Net重算存在分块相关数值差异。job166采用M_inherited+qbar*(U_train-U_initial)精确零起点函数差值，原门槛和预算不变；GPU合成起点误差0、同分块梯度误差8.94e-8。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ['REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py']:
    code = (prior / name).read_text('utf-8').replace(old_remote, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    (out / name).write_bytes(code.encode())
collector = (here / 'COLLECT_MEAN_TREND_FIX01.py').read_text('utf-8').replace("'mean_trend_fix01'", "'mean_trend_fix02'").replace(old_remote, remote)
(here / 'COLLECT_MEAN_TREND_FIX02.py').write_bytes(collector.encode())
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
    subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deploy)],
        stdin=stream, capture_output=True, check=True, timeout=60)
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=166, config_sha256=sha,
    previous_setup_counts=failure['execution_counts'], previous_optimizer_updates=0,
    scientific_settings_and_budget_unchanged=True)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
