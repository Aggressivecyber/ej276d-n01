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
out = here / 'mean_trend_01'
assert json.loads((out / 'SYNTHETIC_CHECKS.json').read_bytes())['status'] == 'PASS_SYNTHETIC_SEMANTICS'
stamp = datetime.now(timezone.utc).isoformat()
template_path = root / 'configs/global_memory_conditional_wave_20261021.json'
old = json.loads(template_path.read_bytes())
remote = '/home/rylan/projects/ej276d-n01/mean_trend_repair_20261021'
p = {k: old[k] for k in ['frozen_setup_config', 'roles_file', 'candidate_arrays', 'memory',
    'replay_checkpoint', 'replay_memory', 'replay_accepted_updates', 'bank_seed', 'hypothesis',
    'parent_chunk', 'micro_rows', 'identity_tolerance_V']}
p.update(schema='n01/conditioned-mean-trend-repair/v1', job_id='N01_MEAN_TREND_REPAIR_20261021',
    registered_UTC=stamp, authorization='2026-10-21 user: 去修复',
    mean_chunk=64, draws_per_parent=8, endpoint_draws=32,
    direction_seed=2146102100, endpoint_seed=2156102100,
    directions=24, stop_consecutive_rejections=3, step_scales=[1., .5, .25], trend_points=128,
    active_cells=['Q_wave_L16_04', 'Q_wave_L64_26'],
    learning_rates=dict(response=.002, mean=.0002, conditioning=.0001),
    architecture=dict(unfrozen='Copied U-Net condition MLP and all FiLM affine layers plus copied terminal mean convolution.',
        frozen='Original models; copied nonconditioning convolutions, group norms, mean time map, charge map, stochastic M0 bases and electronics.',
        parameters=273174, original_mean_identity_required=True,
        source_input='Full normalized waveform, source logQ and log amplitude, original detector condition. No event IDs as inputs.'),
    trend_contract=dict(grid='Two FIT shape prototypes from prior declared source-PSD groups;128 evenly spaced logQ coordinates within each group support.',
        features='Mean normalized SOURCE features and filtered shapes; log amplitude follows same physical amplitude rescaling as logQ.',
        loss='Mean squared second derivative of canonical-charge-normalized full384 mean waveform in logQ, each group normalized once by its inherited value; equal group weights.',
        objective_weight=1., coverage='All6465 FIT parents and all18948 REAL FIT rows remain in distribution supervision. Prototype trend covers6301 dominant-template rows; other164 retained.',
        purpose='Suppress diagnosed cross-Q oscillation while conditional waveform memory protects actual shape; no monotonicity law imposed.'),
    optimizer_contract=dict(method='Adam objective proposal plus normalized wave, Q-wave, trend, two previously active cell, and two ADC gradients; at most8 measured native secant directions.',
        exact_adjoint='Cache one mean per FIT parent; sum all draw adjoints, then replay U-Net per parent chunk. Synthetic equality tested.',
        constraints='Existing supported92 conditional-wave cells and Q-wave nonincrease, global wave nonincrease, Q+0.5%, peak+1%, PSD nonincrease, both trend risks nonincrease and both ADC risks nonincrease.',
        commit='Hard original ADC/readout; every constraint and flag count passes, with strict PSD, Q-wave, objective and both trend reductions.',
        stop='At most24 directions, stop after3 consecutive rejects; no threshold relaxation, added steps or parameter sweep.'),
    endpoint_contract=dict(existing='Fresh two-bank32: >=10% PSD improvement vs original, Q/wave/peak <=+2%, flags not increased; Q-wave improves vs inherited and all supported cells <=+2%.',
        added='Both full-wave curvature risks and128 equal-count fixed-window median-PSD excess variations <=50% of inherited values; at least1 actual update.',
        scope='FIT pilot only. Fixed REPORT evaluation only after pilot gates; REPORT never trains or selects settings. No physical response acceptance from FIT alone.'),
    operation_budget=dict(response_rows=33618000, conditioned_mean_rows=2200000,
        optimizer_directions=24, committed_optimizer_updates=24, native_secant_probes=192,
        joint_hard_trials=72, mean_adjoint_rows=1086120,
        derivation='Initial103440 +24*(two gradient passes+8 probes+3 trials)*103440 +3 models*two banks*32*6465. ADC shares gradient observation pass.'),
    resources=dict(CPU_threads=4, GPU_GiB=4, RSS_GiB=12, internal_seconds=5400,
        hard_wall_seconds=5460, GPU_free_required_MiB=6144, output_MiB=1024),
    storage=dict(remote_root=remote, policy='Active training weights/arrays on compute storage; no duplicate large backups or NAS checks.'),
    scientific_acceptance=False, report_rows_used_for_training=0,
    NAS_file_IO_checks=False, NAS_file_hash_verification=False,
    measured_budget_basis=dict(reference='job163 frozen component diagnostic17.059s; native parent adjoint semantics verified locally, new GPU step duration not yet measured.',
        wake_rule='Use first complete direction timing and remaining directions plus endpoint workload and margin; no fixed polling interval.'))
path = root / 'configs/mean_trend_repair_20261021.json'
assert not path.exists()
path.write_bytes((json.dumps(p, ensure_ascii=False, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['src/n01/mean_trend01.py', 'scripts/run_mean_trend01.py', 'scripts/check_mean_trend01.py']:
    ast.parse((root / name).read_text('utf-8'))
snapshot = out / 'before_registration'
snapshot.mkdir()
for name in ['PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json']:
    raw = (root / name).read_bytes()
    (snapshot / name).write_bytes(raw)
    d = json.loads(raw)
    assert d['active_scientific_jobs'] == 0
    d.update(status='N01_MEAN_TREND_repair_registered', active_job_id=None, active_run_id=None,
        pending_job_id=p['job_id'], pending_registered_job_id=p['job_id'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='User authorized root-cause repair; copied conditioning path unfrozen with real full-wave trend gradients.',
        next_action='Run bounded conditioning-path training, then native hard endpoint comparison and full-Q figures.')
    d['global_memory_repair'].update(status='mean_trend_repair_registered', config=path.relative_to(root).as_posix(),
        config_sha256=sha, current_committed_training_updates=0, automatic_training_continuation=False,
        next_step_design=path.relative_to(root).as_posix())
    if name == 'jobs.json':
        assert len(d['jobs']) == 163
        d['jobs'].append(dict(job_id=p['job_id'], status='registered', registered_UTC=stamp,
            config=path.relative_to(root).as_posix(), config_sha256=sha, resources=p['resources'],
            operation_budget=p['operation_budget'], kind='conditioned_mean_trend_repair', scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf-8', newline='\n') as stream:
    stream.write('\n用户要求修复已定位波浪。job164解冻复制的Q/幅度条件MLP与FiLM及均值末端，'
        '冻结原模型、时间/电荷图、其他卷积与电子学。使用全部FIT分布记忆和真正的跨logQ全波形曲率梯度，'
        '最多24方向、连续3拒绝停止；原门槛不变，新增两主要分支波浪及曲率均减半的终点门槛。'
        '先通过原生FIT门槛再固定REPORT评估，失败不扩步或扫参。\n')
old_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
for name in ['REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py']:
    code = (here / 'conditional_wave_01' / name).read_text('utf-8')
    code = code.replace(old['storage']['remote_root'], remote).replace(old_sha, sha)
    code = code.replace(template_path.name, path.name).replace('run_global_memory01.py', 'run_mean_trend01.py').replace("'660s'", "'5460s'")
    (out / name).write_bytes(code.encode())
collector = (here / 'COLLECT_COMB_COMPONENTS.py').read_text('utf-8')
collector = collector.replace("'comb_components_01'", "'mean_trend_01'").replace('/comb_components_20261021', '/mean_trend_repair_20261021')
collector = collector.replace("'run_01/FAILED.json','run_01/COMPLETE.json'", "'run_01/FAILED.json','run_01/COMPLETE.json','run_01/TRAINING_RESULT.json','run_01/TRAINING_STEPS.jsonl'")
(here / 'COLLECT_MEAN_TREND.py').write_bytes(collector.encode())
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
receipt = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=164, config_sha256=sha,
    planned_trainable_parameters=273174, optimizer_updates=0, inherited_updates=10,
    source_FIT=6465, real_FIT=18948, REPORT_training_rows=0)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps(receipt))
