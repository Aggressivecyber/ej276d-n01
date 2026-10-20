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
prior = here / 'fixed_validation_fix01'
out = here / 'parent_audit_01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
evaluation = json.loads((prior / 'EVALUATION_RESULT.json').read_bytes())
native = json.loads((prior / 'RESULT.json').read_bytes())
exit_record = json.loads((prior / 'exit_receipt.json').read_bytes())
assert exit_record['exit_code'] == 0 and native['committed_optimizer_updates'] == 0
assert all(r['passes_consistency_check'] for r in evaluation['results']['FIT'].values())
assert not any(r['passes_consistency_check'] for r in evaluation['results']['REPORT'].values())
closure = dict(status='COMPLETE_REPORT_ROLE_FAILED', UTC=stamp, native_exit_code=0,
    execution_counts=native['execution_counts'], new_optimizer_updates=0,
    candidate_unchanged=True, report_rows_added_to_memory=0, scientific_acceptance=False,
    reason='All3 FIT hypotheses pass; all3 REPORT hypotheses fail PSD10% and waveform2% gates.',
    next='One fixed-candidate parent-level feature replay and FIT-only equal-size population subsampling; no training.')
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
old_path = root / 'configs/global_memory_fixed_validation_fix01_20261021.json'
config = json.loads(old_path.read_bytes())
remote_old = config['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/global_memory_parent_audit_20261021'
config.update(schema='n01/global-memory-parent-population-audit/v1', job_id='GLOBAL_MEMORY_PARENT_AUDIT_20261021',
    registered_UTC=stamp, parent_wave_audit=True, hypotheses=['fast'],
    purpose='Separate finite128-parent population variability from systematic source-dependent waveform correction effects, using frozen weights.',
    success_rule='Reconstruct exact per-parent56-dimensional wave feature means, reproduce prior fast role aggregates; then characterize128-parent FIT subsets. No model promotion or new hypothesis testing claim.',
    analysis=dict(seed=2026102161, subset_parents=128, finite_population_subsets=2000,
        sampling='Uniform without replacement within each subset of the6465 FIT parents; subsets overlap. Empirical reference distribution, not formal p-value for the fixed REPORT selection.',
        reference='Frozen FIT map and fixed REAL REPORT wave reference; also compare original FIXED REPORT128 parents.',
        Q_groups='Eight FIT-only quantile groups of SOURCE Q; diagnose macro-vs-conditional cancellation without physical truth pairing.',
        draws='Same two32-draw banks as job160; this is a per-parent replay, not independent statistical evidence.'),
    stop_rule='One registered replay and2000 CPU subset summaries, no optimizer, no new architectures or additional training steps.')
config['operation_budget'].update(hard_observation_rows=843904,
    derivation='(6465+128)*32*2 banks*2 models=843904 signal and fast-observation rows. Save parent means, not waveforms.')
config['storage'] = dict(remote_root=remote, outputs='Active per-parent wave feature means and small analysis reports; no raw waveform outputs or backup copies.')
config['measured_budget_basis'] = dict(prior_three_hypothesis_seconds=native['elapsed_seconds'], expected_seconds=25,
    conservative_seconds=60, basis='Same843904 signals as34.48s job160, one readout hypothesis instead of3 plus per-parent accumulation.')
path = root / 'configs/global_memory_parent_audit_20261021.json'
path.write_bytes((json.dumps(config, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
ast.parse((root / 'scripts/evaluate_global_memory01.py').read_text())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_parent_population_audit_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=0, current_job_ended_UTC=exit_record['ended_utc'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='Fixed candidate fails REPORT waveform consistency across3 noise hypotheses; saved-feature decomposition localizes sign reversal to source population.',
        next_action='Quantify128-parent FIT population variability using one fixed per-parent replay; no training or gate changes.')
    d['global_memory_repair'].update(status='parent_population_audit_registered', config=str(path.relative_to(root)), config_sha256=sha,
        fixed_validation_closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)), next_step_design=str(path.relative_to(root)),
        fixed_validation_all_roles_pass=False, science_accepted=False)
    if name == 'jobs.json':
        assert len(d['jobs']) == 160
        d['jobs'][-1].update(status='completed_FIXED_REPORT_consistency_failed', native_exit_code=0,
            actual_execution_counts=native['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)))
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='frozen_parent_population_diagnostic',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob160原生exit0，三噪声FIT均通过，REPORT均未通过：PSD改善约7%，波形退化5.76%—6.50%。'
        '交换REAL参考的已存特征分析表明反向变化随SOURCE人口改变，而非仅REAL参考改变。'
        '登记job161，只复现fast逐父事件56维波形特征，在FIT内取2000个128父事件子集量化人口敏感性，新增训练0。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    text = (prior / name).read_text().replace(remote_old, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    (out / name).write_bytes(text.encode())
collector = (here / 'COLLECT_FIXED_VALIDATION_FIX01.py').read_text().replace("'fixed_validation_fix01'", "'parent_audit_01'").replace(remote_old, remote)
(here / 'COLLECT_PARENT_AUDIT.py').write_bytes(collector.encode())
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
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=161,
    config_sha256=sha, new_optimizer_updates=0, scientific_acceptance=False)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
