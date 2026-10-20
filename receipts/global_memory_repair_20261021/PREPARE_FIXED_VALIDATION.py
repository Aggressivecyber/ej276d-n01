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
prior = here / 'joint_local_01'
out = here / 'fixed_validation_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
result = json.loads((prior / 'TRAINING_RESULT.json').read_bytes())
native = json.loads((prior / 'RESULT.json').read_bytes())
exit_record = json.loads((prior / 'exit_receipt.json').read_bytes())
steps = [json.loads(s) for s in (prior / 'TRAINING_STEPS.jsonl').read_text().splitlines()]
assert exit_record['exit_code'] == 0 and native['committed_optimizer_updates'] == 9
assert sum(s['accepted'] for s in steps) == 9 and len(steps) == 12
assert result['accepted_updates']['B'] == 10 and result['new_committed_optimizer_updates'] == 9
assert all(all(s['trials'][-1]['checks'].values()) for s in steps if s['accepted'])
assert result['endpoint']['B']['passes_bounded_pilot']
initial, final = [result['endpoint'][n]['positive_metrics'] for n in ('initial', 'B')]
changes = {k: 100 * (final[k] / v - 1) for k, v in initial.items() if v != 0.}
closure = dict(status='COMPLETE_BOUNDED_PILOT_PASSED_RESPONSE_MATCH_NOT_ESTABLISHED', UTC=stamp,
    native_exit_code=0, elapsed_seconds=native['elapsed_seconds'], execution_counts=native['execution_counts'],
    direction_attempts=12, inherited_updates=1, new_optimizer_updates=9, candidate_total_updates=10,
    endpoint_changes_percent=changes, endpoint_checks=result['endpoint']['B']['pilot_checks'],
    ADC_rail_counts=dict(initial=[r['ADC_rail_hit'] for r in result['endpoint']['initial']['flags']],
                         candidate=[r['ADC_rail_hit'] for r in result['endpoint']['B']['flags']]),
    response_match_established=False, scientific_acceptance=False,
    visual_conclusion='Comb-like Q/PSD structure remains conspicuous; passing relative10% pilot improvement does not establish distribution repair.',
    next_action='Frozen candidate, two fresh banks, all three noise hypotheses and existing FIT/REPORT roles; no further optimizer updates.')
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
(prior / 'AUTOMATION_CLOSED.json').write_bytes((json.dumps(dict(status='DELETED_CONFIRMED', automation_id='n01-psd',
    UTC=stamp, reason='job158 completed; no outstanding wait before the next registered evaluation.'), indent=2) + '\n').encode())
figure = root / 'deliverables/global_memory_repair_20261021/joint_local_01'
qa = dict(status='VISUALLY_INSPECTED', UTC=stamp, images=['GLOBAL_MEMORY_PSD_LINEAR.png',
    'GLOBAL_MEMORY_PSD_FULL_RANGE.png', 'CONDITIONAL_PSD_64_COMPARISON.png'],
    observations=['Legible titles and shared population normalization.', 'All Q values retained; linear PSD clipping separately counted and full-range view contains all rows.',
                  'Strong comb structure persists in candidate; REAL bands remain visibly different.',
                  '64-bin conditional curves show modest local changes; sparse and missing cells remain explicit.'],
    physical_energy_calibration=False, scientific_acceptance=False)
(figure / 'VISUAL_QA.json').write_bytes((json.dumps(qa, indent=2) + '\n').encode())
conditional = json.loads((figure / 'CONDITIONAL_PSD_REPORT.json').read_bytes())['models']
cells0, cells1 = [conditional[n]['64']['cells'] for n in ('initial', 'B')]
common = [(a, b) for a, b in zip(cells0, cells1) if a['empirical_PSD_W1'] is not None and b['empirical_PSD_W1'] is not None]
weight = sum(a['real_rows'] for a, b in common)
w0 = sum(a['real_rows'] * a['empirical_PSD_W1'] for a, b in common) / weight
w1 = sum(a['real_rows'] * b['empirical_PSD_W1'] for a, b in common) / weight
slice_report = dict(common_scored_bins=len(common), REAL_rows_in_common_bins=weight, total_REAL_rows=18948,
    initial_REAL_weighted_conditional_W1=w0, candidate_REAL_weighted_conditional_W1=w1,
    relative_change_percent=100 * (w1 / w0 - 1),
    improved_bins=sum(b['empirical_PSD_W1'] < a['empirical_PSD_W1'] for a, b in common),
    denominator='Same current endpoint bank0; REAL-row weights over shared scored bins only; sparse cells included and not independent experimental units.',
    scientific_acceptance=False)
(figure / 'CONDITIONAL_EFFECT_SIZE.json').write_bytes((json.dumps(slice_report, indent=2) + '\n').encode())
temporary = (prior / 'ENDPOINT_SCALARS_ANALYSIS.npz').resolve()
assert temporary.parent == prior.resolve() and temporary.drive.upper() == 'E:'
temporary.unlink()
(prior / 'ANALYSIS_STORAGE_RECEIPT.json').write_bytes((json.dumps(dict(UTC=stamp, E_temporary_scalars_removed=True,
    primary_active_scalars=result['endpoint_arrays'], NAS_file_checks=False, NAS_file_hash_verification=False), indent=2) + '\n').encode())
old_path = root / 'configs/global_memory_joint_local_20261021.json'
old = json.loads(old_path.read_bytes())
remote_old = old['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/global_memory_fixed_validation_20261021'
config = {k: old[k] for k in ['frozen_setup_config', 'roles_file', 'candidate_arrays', 'parent_chunk', 'micro_rows', 'memory']}
config.update(schema='n01/global-memory-fixed-validation/v1', job_id='GLOBAL_MEMORY_FIXED_VALIDATION_20261021',
    registered_UTC=stamp, authorization='Continue user-authorized PSD repair; validate frozen successful pilot before any promotion.',
    frozen_candidate=remote_old + '/run_01/B_CURRENT.pt', frozen_memory=remote_old + '/run_01/TRAIN_DISTRIBUTION_MEMORY.pt',
    draws_per_parent=32, evaluation_seed=2126102100, hypotheses=['fast', 'slow', 'common_level'],
    roles=dict(FIT=dict(source_parents=6465, REAL_rows=18948), REPORT=dict(source_parents=128, REAL_rows=2099)),
    memory_rule='Only original REAL FIT constructs the feature map; reuse saved FIT objective weights. REPORT contributes an evaluation target only, with memory target and candidate asserted unchanged.',
    evidence_scope='Same-acquisition historically designated report role, not a new acquisition or claim of future prediction. Original fixed row membership; no resplit or reweighting.',
    success_rule='Report unchanged10% PSD/2% Q-wave-peak/no-flag-increase checks separately for every role and hypothesis; no choosing favorable hypothesis, no fits or model changes.',
    operation_budget=dict(response_rows=843904, hard_observation_rows=2531712,
        committed_optimizer_updates=0, optimizer_directions=0,
        derivation='(6465+128)*32 draws*2 banks*2 models=843904 signal rows;3 readout hypotheses=2531712 observed rows.'),
    resources=dict(CPU_threads=4, GPU_GiB=4, RSS_GiB=12, internal_seconds=300, hard_wall_seconds=360, GPU_free_required_MiB=6144),
    storage=dict(remote_root=remote, outputs='Small role/hypothesis metric JSONs and fixed feature means on active compute storage; no waveform archives.'),
    measured_budget_basis=dict(expected_seconds=75, conservative_seconds=150,
        basis='job158 fresh827520-response-row endpoint took about15s; this run reuses each signal for3 observed hypotheses plus setup, projected under75s on idle GPU.'),
    stop_rule='One fixed candidate evaluation on two banks per role. No optimizer, parameter selection, extra banks or retries except an explicitly recorded operational correction.',
    original_endpoint_thresholds_unchanged=True, report_training_rows=0, seed_head=0,
    scientific_acceptance=False, NAS_file_IO_checks=False, NAS_file_hash_verification=False)
path = root / 'configs/global_memory_fixed_validation_20261021.json'
path.write_bytes((json.dumps(config, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
ast.parse((root / 'scripts/evaluate_global_memory01.py').read_text())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_fixed_validation_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0, current_job_ended_UTC=exit_record['ended_utc'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='job158 completed with9 new accepted updates and all pilot endpoint gates passing; visible PSD comb remains.',
        next_action='Evaluate frozen candidate across fixed roles and all3 noise hypotheses; no new training.',
        followup_automation_id=None, followup_automation_status='DELETED', followup_status='DELETED_NO_WAITING_WORK',
        followup_next_run_local=None, followup_scheduled_local=None, next_wake_local=None)
    d['followup'] = dict(status='DELETED', automation_id='n01-psd', next_wake_local=None,
        receipt=str((prior / 'AUTOMATION_CLOSED.json').relative_to(root)))
    d['global_memory_repair'].update(status='fixed_validation_registered', active_scientific_jobs=0,
        current_committed_training_updates=0, most_recent_training_updates=9, inherited_checkpoint_updates=1,
        total_accepted_updates_across_pilots=17, current_native_execution_counts=native['execution_counts'],
        config=str(path.relative_to(root)), config_sha256=sha, joint_local_closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)),
        joint_local_endpoint_changes_percent=changes, bounded_pilot_passed=True, science_accepted=False,
        automatic_training_continuation=False, next_step_design=str(path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 158
        d['jobs'][-1].update(status='completed_bounded_pilot_passed_not_scientific_acceptance', native_exit_code=0,
            actual_execution_counts=native['execution_counts'], committed_optimizer_updates=9,
            ended_UTC=exit_record['ended_utc'], closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)))
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='frozen_candidate_role_and_noise_sensitivity',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob158完成12方向、9次新更新，原生exit0；独立随机库PSD误差下降11.14%，'
        '波形、电荷和峰码误差均下降，ADC计数4/5变为3/5，首次通过既定小试验终点门。'
        '全Q图仍有明显梳齿，不能称总体分布修复成功。登记job159固定模型评价：'
        '全部既有FIT/REPORT、三噪声假设、两库各32抽样，0训练，报告数据不进记忆。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    text = (prior / name).read_text().replace(remote_old, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    text = text.replace('run_global_memory01.py', 'evaluate_global_memory01.py').replace("'1320s'", "'360s'")
    text = text.replace("('run_','diagnose_'", "('run_','evaluate_','diagnose_'")
    (out / name).write_bytes(text.encode())
collector = (here / 'COLLECT_JOINT_LOCAL.py').read_text().replace("'joint_local_01'", "'fixed_validation_01'").replace(remote_old, remote)
collector = collector.replace("'run_01/TRAINING_RESULT.json'", "'run_01/EVALUATION_RESULT.json','run_01/TRAINING_RESULT.json'")
(here / 'COLLECT_FIXED_VALIDATION.py').write_bytes(collector.encode())
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
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=159,
    config_sha256=sha, new_optimizer_updates=0, scientific_acceptance=False)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(dict(registration=record, endpoint_changes=changes, conditional_effect=slice_report)))
