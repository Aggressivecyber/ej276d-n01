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
out = here / 'conditional_wave_01'
assert json.loads((out / 'SYNTHETIC_CHECKS.json').read_bytes())['status'] == 'PASS_SYNTHETIC_SEMANTICS'
assert not (out / 'REGISTRATION_RECEIPT.json').exists()
stamp = datetime.now(timezone.utc).isoformat()
prior = here / 'joint_local_01'
old_path = root / 'configs/global_memory_joint_local_20261021.json'
config = json.loads(old_path.read_bytes())
remote_old = config['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/global_memory_conditional_wave_20261021'
config.update(schema='n01/global-memory-conditional-wave-probe/v1',
    job_id='GLOBAL_MEMORY_CONDITIONAL_WAVE_PROBE_20261021', registered_UTC=stamp,
    authorization='2026-10-21 user accepted the explained conditional waveform memory and one bounded direction check.',
    replay_checkpoint=remote_old + '/run_01/B_CURRENT.pt',
    replay_memory=remote_old + '/run_01/TRAIN_DISTRIBUTION_MEMORY.pt',
    replay_steps=remote_old + '/run_01/TRAINING_STEPS.jsonl',
    replay_accepted_updates=10, replay_direction_iteration=12,
    direction_seed=config['bank_seed'] + 100000000,
    endpoint_seed=config['bank_seed'] + 110000000,
    endpoint_archive=remote + '/run_01/ENDPOINT_SCALARS.npz',
    adaptive_secant_radius=False,
    initialize_from='Frozen job158 B with10 cumulative accepted updates; new Adam moments and new conditional objective gradients.',
    stop_rule='Exactly one direction attempt, at most5 native basis probes and3 combined hard trials; at most1 committed update. Stop regardless of result; no appended steps or sweep.')
del config['direction_file']
config['memory']['conditional_wave_min_count'] = 56
config['storage']['remote_root'] = remote
config['optimizer'].update(direction_attempts_per_arm=1, stop_consecutive_rejections=1,
    acceptance='Existing Q/wave/peak/PSD/ADC guards plus nonincrease of every supported conditional-wave cell and strict total Q-wave risk improvement.')
config['conditional_wave_contract'] = dict(
    features='Existing56 waveform features from raw256, amplitude-normalized256 and normalized first differences; standardized by fixed REAL FIT global feature mean and variance.',
    target='m_k = mean_REAL[a_k(Q) * standardized_wave_features]. Old16/32/64 soft-Q partitions and all18948 REAL FIT rows.',
    objective='Sum over levels and all cells of ||m_generated-m_REAL||^2/(3*56*max(REAL_mass,56/18948)); independent-bank cross product for training.',
    normalization='Append one block normalized once from original A0 under the original TRAIN normalization banks. Old target/weights prefix must remain bitwise identical.',
    cell_guard='Mean squared joint moment residual divided by max(REAL_mass,56/18948)^2; every supported cell nonincreasing within the fixed direction banks.',
    supported='Both REAL soft mass*18948 and effective sample count (sum weights)^2/sum(weights^2) >=56, the feature count. This is a declared engineering support rule, not a statistical confidence bound.',
    sparse='All cells and all rows remain in objective; mass denominator capped below at56/18948. Unsupported cells reported without individual hard constraints.',
    final_check='On independent two-bank32: unchanged original pilot gates; Q-wave strictly improves vs inherited B and every supported cell <=1.02 times inherited risk. At least one committed update required.',
    limitations='Joint distribution control in observed Q, not eventwise SOURCE-REAL pairing, source-population invariance, calibrated energy, temporal memory, or future-acquisition validation.',
    report_rows_used=0)
config['joint_rule'].update(
    basis=['aligned_positive_tangent', 'positive_wave_descent', 'conditional_wave_descent', 'ADC_bank0_descent', 'ADC_bank1_descent'],
    secants='Single radius0.05 from existing local method; measured native probes. No radius sweep.',
    solver='Same linear program in convex hull of zero and at most5 native directions; append supported-cell nonincrease constraints.',
    optimizer='Reset Adam moments because objective changed; retain moments only if the single combined update commits.',
    trust_region='No adaptation: one registered direction.',
    why='Saved population audit found opposing global wave contributions. This probe tests explicit local-Q waveform supervision and guards.')
config['operation_budget'] = dict(response_rows=2482560, optimizer_directions=1,
    committed_optimizer_updates=1, native_secant_probes=5, joint_hard_trials=3,
    continuous_ADC_gradient_pairs=1, new_global_gradient_pairs=1,
    derivation='103440 rows per two-bank8 pair:1 original normalization +2 gradient passes +1 ADC-gradient pass +5 probes +3 trials +12 endpoint pairs(original,inherited,new; two-bank32) =24*103440.')
config['resources'].update(internal_seconds=600, hard_wall_seconds=660)
config['measured_budget_basis'] = dict(prior_job=158, prior_rows=9930240,
    prior_seconds=256.295, expected_seconds=90, conservative_seconds=150,
    basis='At most2.483M rows with one extra waveform backward and conditional moments; reestimate from actual launch progress.')
path = root / 'configs/global_memory_conditional_wave_20261021.json'
assert not path.exists()
path.write_bytes((json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
for name in ('scripts/run_global_memory01.py', 'src/n01/global_memory01.py', 'scripts/check_global_memory01.py'):
    ast.parse((root / name).read_text())
snapshot = out / 'before_registration'
snapshot.mkdir(exist_ok=False)
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    raw = (root / name).read_bytes()
    (snapshot / name).write_bytes(raw)
    d = json.loads(raw)
    assert d['active_scientific_jobs'] == 0
    d.update(status='GLOBAL_MEMORY_REPAIR_conditional_wave_probe_registered',
        active_job_id=None, active_run_id=None, pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='TRAIN-only multiscale conditional wave memory implemented and synthetic semantics passed.',
        next_action='Execute one registered conditional-wave-aware joint direction; no further training automatically appended.')
    d['global_memory_repair'].update(status='conditional_wave_probe_registered', config=str(path.relative_to(root)),
        config_sha256=sha, current_committed_training_updates=0, inherited_checkpoint_updates=10,
        next_step_design=str(path.relative_to(root)), automatic_training_continuation=False)
    if name == 'jobs.json':
        assert len(d['jobs']) == 161
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='conditional_Q_wave_single_direction_probe',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\n2026-10-21 用户同意条件波形记忆方案。登记job162：在现有16/32/64软Q分区追加56维波形联合矩；'
        '全18948 REAL FIT固定监督，全6465源父保留。软计数与有效计数均至少56的单元逐个约束不恶化，'
        '稀疏单元仍参与总体目标；该支撑数是工程规则。继承job158的10次累计更新，重置新目标的Adam状态，'
        '仅检验1个方向、最多5个原生基方向与3次组合硬检查、至多1次提交。'
        '独立终点比较原始、继承和新状态，原验收阈值不变；REPORT不进训练或规则选择。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    code = (prior / name).read_text().replace(remote_old, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    code = code.replace("'1320s'", "'660s'")
    code = code.replace("'run_','diagnose_'", "'run_','evaluate_','diagnose_'")
    (out / name).write_bytes(code.encode())
collector = (here / 'COLLECT_JOINT_LOCAL.py').read_text().replace("'joint_local_01'", "'conditional_wave_01'").replace(remote_old, remote)
collector = collector.replace("'run_01/BINDINGS.json'", "'run_01/CONDITIONAL_WAVE_MEMORY.json','run_01/BINDINGS.json'")
(here / 'COLLECT_CONDITIONAL_WAVE.py').write_bytes(collector.encode())
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
    r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(deploy)],
        stdin=stream, capture_output=True, check=True, timeout=60)
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=162, config_sha256=sha,
    new_optimizer_updates=0, inherited_checkpoint_updates=10, scientific_acceptance=False,
    old_jobs_preserved=True, report_rows_used=0)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
