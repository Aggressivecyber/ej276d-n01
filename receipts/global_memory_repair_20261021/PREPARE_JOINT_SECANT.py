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
out = here / 'joint_secant_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
old_path = root / 'configs/global_memory_selection_gradient_20261021.json'
old = json.loads(old_path.read_bytes())
remote = '/home/rylan/projects/ej276d-n01/global_memory_joint_secant_20261021'
keys = ['frozen_setup_config', 'roles_file', 'candidate_arrays', 'parent_chunk', 'micro_rows',
        'hypothesis', 'primary_seed_head', 'draws_per_parent', 'bank_seed', 'identity_tolerance_V',
        'memory', 'learning_rates', 'optimizer', 'replay_checkpoint', 'replay_memory', 'replay_steps',
        'replay_accepted_updates', 'replay_direction_iteration', 'peak_selection_gradient',
        'gradient_rule', 'temperature_codes', 'temperature_search', 'endpoint_draws', 'endpoint_decision']
config = {k: old[k] for k in keys}
config.update(schema='n01/global-memory-joint-secant-pilot/v1',
    job_id='GLOBAL_MEMORY_JOINT_SECANT_PILOT_20261021', mode='joint_secant_pilot', registered_UTC=stamp,
    authorization='2026-10-21 user: 启动下一步 推进新训练',
    direction_file=old['storage']['remote_root'] + '/run_01/DIRECTION_DIAGNOSTIC.pt',
    endpoint_seed=old['bank_seed'] + 60000000,
    endpoint_archive=remote + '/run_01/ENDPOINT_SCALARS.npz',
    operation_budget=dict(response_rows=9516480, optimizer_directions=12,
        committed_optimizer_updates=12, native_secant_probes=24, joint_hard_trials=36,
        new_gradient_pairs=11, first_step_saved_gradient_replays=1,
        derivation='103440 normalization + 6*103440 first direction + 11*7*103440 subsequent directions + 8*103440 endpoint =9516480.'),
    resources=dict(CPU_threads=4, GPU_GiB=4, RSS_GiB=12, internal_seconds=1200,
                   hard_wall_seconds=1320, GPU_free_required_MiB=6144),
    storage=dict(remote_root=remote, archival_policy='Active compute checkpoint and endpoint scalars needed for immediate analysis; no NAS probes or duplicate large backups.'),
    initialize_from='One previously committed B update. First direction reuses job156 saved gradients and two direction tensors; later directions recompute gradients at current weights.',
    joint_rule=dict(
        basis=['aligned_positive_tangent', 'positive_wave_descent'],
        secants='Evaluate each full basis displacement on the same complete two TRAIN random banks, with native ADC and argmax. Domain clipping included before secants.',
        solver='Linear program in convex hull of zero and both actual probe displacements: weights>=0, sum<=1. Minimize native secant of original global cross-bank objective.',
        constraints='Q_mass <=+0.5%, waveform <=0%, peak_codes <=+1%; PSD_CDF and each-bank continuous ADC margin risk nonincreasing.',
        ADC_margin='Mean per-row squared positive rail proximity in ADC codes; margin starts one code inside both physical rails, using actual pre-ADC min/max over384 samples.',
        validation='Hard evaluate combination at scales1,.5,.25; commit only if original guards, all flag counts, continuous rail risk, strict PSD improvement and strict total objective decrease pass.',
        optimizer='Adam moment update is retained only when the jointly selected step commits. Rejected weights and moments are restored.',
        limitation='Native secants are local approximations, not guarantees across peak jumps; hard evaluation is decisive.'),
    stop_rule='If first joint feasibility step fails, stop training immediately. Otherwise at most12 total directions, or stop after3 consecutive rejections. No budget extension or threshold relaxation.',
    endpoint_streams_used_for_selection=False, endpoint_thresholds_unchanged=True,
    report_role_rows_used_for_training=0, representation_changes=False,
    scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False,
    measured_budget_basis=dict(prior_724080_rows_seconds=94.07759003600007,
        prior_gradient_seconds=28.1345016690002, expected_seconds=850, conservative_seconds=1100,
        basis='Maximum12 directions, two native secant probes and up to3 validations each, plus fresh32-draw endpoint; revise wake from measured first-direction time.'))
config['optimizer']['acceptance'] = 'Original Q/wave/peak and all hard flags pass; additionally PSD and global objective decrease and continuous ADC margin risk does not increase.'
path = root / 'configs/global_memory_joint_secant_20261021.json'
path.write_bytes((json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
engineering = json.loads((here / 'joint_secant_engineering_01/SYNTHETIC_CHECKS.json').read_bytes())
assert engineering['status'] == 'PASS_SYNTHETIC_SEMANTICS'
for name in ['scripts/run_global_memory01.py', 'src/n01/global_memory01.py', 'scripts/check_global_memory01.py']:
    ast.parse((root / name).read_text())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    assert d['active_scientific_jobs'] == 0
    d.update(status='GLOBAL_MEMORY_REPAIR_joint_secant_training_registered',
        active_scientific_jobs=0, active_job_id=None, active_run_id=None,
        pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='User authorized new training; joint native secant feasibility and bounded continuation implemented and registered.',
        next_action='Launch first joint hard feasibility step, continue within12 directions only after acceptance, then evaluate fresh endpoint.')
    d['global_memory_repair'].update(status='joint_secant_training_registered',
        current_committed_training_updates=0, config=str(path.relative_to(root)), config_sha256=sha,
        active_scientific_jobs=0, automatic_training_continuation=True,
        continuation_rule=config['stop_rule'], next_step_design=str(path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 156
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='joint_native_secant_bounded_training',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\n2026-10-21 用户明确要求启动下一步和新训练。登记job157：联合原生割线约束训练，'
        '复用job156两个方向，加入ADC量化前一代码余量，先求原生电荷/波形/峰码/PSD/ADC共同可行组合。'
        '第一步通过才继续，总上限12方向、连续3次拒绝停止；硬前向和终点阈值不改。'
        '之后方向在当前权重上重算，最终用全新两库各32抽样评价，原1次已接受更新与新增更新单列。\n')
template = here / 'selection_gradient_01'
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    text = (template / name).read_text().replace(old['storage']['remote_root'], remote).replace(old_sha, sha)
    text = text.replace(old_path.name, path.name).replace("'360s'", "'1320s'")
    (out / name).write_bytes(text.encode())
members = list((root / 'src/n01').glob('*.py')) + list((root / 'scripts').glob('*.py')) + [path]
payload = out / 'runtime.tar'
with tarfile.open(payload, 'w') as archive:
    for member in members:
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
    response = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deployment)],
        stdin=stream, capture_output=True, timeout=60)
response.check_returncode()
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=157,
    config_sha256=sha, engineering_checks=str((here / 'joint_secant_engineering_01/SYNTHETIC_CHECKS.json').relative_to(root)),
    no_scientific_data_loaded_before_registration=True, new_optimizer_updates=0, scientific_acceptance=False)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
