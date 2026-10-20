from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import shlex
import subprocess
import tarfile

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
prior = here / 'joint_secant_01'
out = here / 'joint_local_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
native = json.loads((prior / 'RESULT.json').read_bytes())
exit_record = json.loads((prior / 'exit_receipt.json').read_bytes())
step = json.loads((prior / 'TRAINING_STEPS.jsonl').read_text())
assert native['committed_optimizer_updates'] == 0 and exit_record['exit_code'] == 0
assert not step['accepted'] and len(step['trials']) == 3
closure = dict(status='COMPLETE_FIRST_JOINT_STEP_REJECTED', UTC=stamp,
    native_exit_code=0, execution_counts=native['execution_counts'], new_optimizer_updates=0,
    scientific_acceptance=False, original_weights_restored=True,
    reason='Full-scale secants predicted improvement, but every actual combined step increased PSD and continuous ADC risk. Waveform improved. No model accepted.',
    next_change='Use local secants and introduce independent continuous ADC-descent directions; keep original hard acceptance and12-direction cap.')
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
old_path = root / 'configs/global_memory_joint_secant_20261021.json'
config = json.loads(old_path.read_bytes())
remote_old = config['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/global_memory_joint_local_20261021'
config.update(schema='n01/global-memory-joint-local-pilot/v1',
    job_id='GLOBAL_MEMORY_JOINT_LOCAL_PILOT_20261021', registered_UTC=stamp,
    initial_secant_radius=.05, adaptive_secant_radius=True, ADC_descent_basis=True,
    endpoint_seed=config['bank_seed'] + 70000000,
    endpoint_archive=remote + '/run_01/ENDPOINT_SCALARS.npz')
config['storage']['remote_root'] = remote
config['operation_budget'] = dict(response_rows=13240320, optimizer_directions=12,
    committed_optimizer_updates=12, native_secant_probes=48, joint_hard_trials=36,
    continuous_ADC_gradient_pairs=12, new_global_gradient_pairs=11, first_step_saved_gradient_replays=1,
    derivation='103440 normalization + 9*103440 first direction + 11*10*103440 later directions + 8*103440 endpoint =13240320.')
config['joint_rule'].update(
    basis=['aligned_positive_tangent', 'positive_wave_descent', 'ADC_bank0_descent', 'ADC_bank1_descent'],
    ADC_directions='Exact gradient of continuous pre-ADC margin risk for each TRAIN bank separately, mapped through original diagonal lr-squared metric and normalized to wave-direction metric norm. Omit exactly-zero hinge gradients.',
    secants='Native probes at the SAME local radius for all directions; clipping occurs before measured secants. First radius0.05 rounds down the smallest rejected wave coefficient0.05163 from job157.',
    solver='Linear program in convex hull of zero and the up to4 local actual probe displacements, with unchanged simultaneous constraints.',
    trust_region='After a hard-accepted step, double its effective local radius only when actual/predicted total-objective reduction>=0.75; cap1. Otherwise retain effective radius. Rejection shrinks to smallest tested scale; same3-consecutive-rejection stop.',
    why='job157 continuous ADC risk increased even at the smallest combined step; adding a continuous ADC degree of freedom directly addresses that active conflict. Local secants address full-step prediction sign failures.')
config['resources'].update(internal_seconds=1200, hard_wall_seconds=1320)
config['measured_budget_basis'] = dict(prior_rows=1551600, prior_actual_seconds=native['elapsed_seconds'],
    currently_idle_GPU=True, expected_seconds=330, conservative_seconds=500,
    basis='Measured job157 on idle GPU took30.94s for1.55M rows; max13.24M rows plus continuous and global backward work. Reestimate after first accepted step.')
path = root / 'configs/global_memory_joint_local_20261021.json'
path.write_bytes((json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['scripts/run_global_memory01.py', 'src/n01/global_memory01.py']:
    ast.parse((root / name).read_text())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_joint_local_training_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0,
        current_job_ended_UTC=exit_record['ended_utc'], updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='job157 rejected before training; local native secants plus continuous ADC directions registered from observed failure.',
        next_action='Start local joint feasibility then bounded continuation; preserve endpoint thresholds.')
    d['global_memory_repair'].update(status='joint_local_training_registered',
        config=str(path.relative_to(root)), config_sha256=sha, current_committed_training_updates=0,
        joint_secant_closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)), next_step_design=str(path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 157
        d['jobs'][-1].update(status='completed_joint_feasibility_rejected_no_training',
            native_exit_code=0, actual_execution_counts=native['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)))
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='local_native_secant_ADC_constrained_training',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob157在首步拒绝后停止，新增更新0：大尺度割线在三个实际组合步上均预测失准，'
        '波形改善但PSD和连续ADC风险增大。据此登记job158，补两库各自的连续ADC下降方向，'
        '用0.05局部半径测量真实割线；通过后按实际/预测下降比调整局部半径。'
        '保留12方向上限与原终点验收，不把当前局部失败解释为表示能力不足。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    text = (prior / name).read_text().replace(remote_old, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    (out / name).write_bytes(text.encode())
collector = (here / 'COLLECT_JOINT_SECANT.py').read_text().replace("'joint_secant_01'", "'joint_local_01'").replace(remote_old, remote)
(here / 'COLLECT_JOINT_LOCAL.py').write_bytes(collector.encode())
payload = out / 'runtime.tar'
with tarfile.open(payload, 'w') as archive:
    for member in list((root / 'src/n01').glob('*.py')) + list((root / 'scripts').glob('*.py')) + [path]:
        archive.add(member, arcname=member.relative_to(root).as_posix())
deployment = """import sys,tarfile
from pathlib import Path
root=Path(REMOTE)
root.mkdir(parents=True,exist_ok=False)
(root/'cache').mkdir()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive: archive.extractall(root,filter='data')
print('RUNTIME_DEPLOYED')
""".replace('REMOTE', repr(remote))
with payload.open('rb') as stream:
    r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deployment)],
        stdin=stream, capture_output=True, check=True, timeout=60)
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=158,
    config_sha256=sha, prior_failure_preserved=True, new_optimizer_updates=0, scientific_acceptance=False)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
