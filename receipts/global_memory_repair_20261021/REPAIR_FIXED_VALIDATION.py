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
prior = here / 'fixed_validation_01'
out = here / 'fixed_validation_fix01'
out.mkdir(exist_ok=False)
stamp = datetime.now(timezone.utc).isoformat()
failed = json.loads((prior / 'FAILED.json').read_bytes())
exit_record = json.loads((prior / 'exit_receipt.json').read_bytes())
assert exit_record['exit_code'] == 1 and failed['error'] == 'Inference tensors do not track version counter.'
assert failed['execution_counts'].get('response_rows', 0) == 0
closure = dict(status='LOADER_FAILED_BEFORE_RESPONSE_FORWARD', UTC=stamp, native_exit_code=1,
    reason='FrozenObservedNoise008 constructor records tensor version counters, unavailable inside inference_mode.',
    correction='Use the existing native no_grad loading/evaluation context. Candidate weights and all metric/random/row settings unchanged.',
    execution_counts=failed['execution_counts'], response_rows=0, optimizer_updates=0,
    checkpoint_read_may_have_started=True, scientific_acceptance=False)
(prior / 'CLOSURE_RECEIPT.json').write_bytes((json.dumps(closure, indent=2) + '\n').encode())
old_path = root / 'configs/global_memory_fixed_validation_20261021.json'
config = json.loads(old_path.read_bytes())
remote_old = config['storage']['remote_root']
remote = '/home/rylan/projects/ej276d-n01/global_memory_fixed_validation_fix01_20261021'
config.update(job_id='GLOBAL_MEMORY_FIXED_VALIDATION_FIX01_20261021', registered_UTC=stamp,
    operational_correction=closure['correction'], prior_failed_job='GLOBAL_MEMORY_FIXED_VALIDATION_20261021',
    prior_failed_response_rows=0)
config['storage']['remote_root'] = remote
path = root / 'configs/global_memory_fixed_validation_fix01_20261021.json'
path.write_bytes((json.dumps(config, indent=2) + '\n').encode())
sha = hashlib.sha256(path.read_bytes()).hexdigest()
ast.parse((root / 'scripts/evaluate_global_memory01.py').read_text())
for name in ('PROGRAM.json', 'MASTER_EXECUTION_PLAN.json', 'jobs.json'):
    d = json.loads((root / name).read_bytes())
    d.update(status='GLOBAL_MEMORY_REPAIR_fixed_validation_loader_fix_registered', active_scientific_jobs=0,
        active_job_id=None, active_run_id=None, pending_job_id=config['job_id'], pending_registered_job_id=config['job_id'],
        training_pid=None, supervisor_pid=None, current_job_native_exit_code=1, current_job_ended_UTC=exit_record['ended_utc'],
        updated_UTC=stamp, execution_state_asof_UTC=stamp,
        current_action='Fixed-candidate validation loader failed before response forwards; original no_grad context restored.',
        next_action='Re-run exactly the registered fixed candidate evaluation in new runtime; no changed scientific settings.')
    d['global_memory_repair'].update(status='fixed_validation_loader_fix_registered', config=str(path.relative_to(root)),
        config_sha256=sha, fixed_validation_loader_failure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)),
        next_step_design=str(path.relative_to(root)))
    if name == 'jobs.json':
        assert len(d['jobs']) == 159
        d['jobs'][-1].update(status='failed_loader_before_response_forward', native_exit_code=1,
            actual_execution_counts=failed['execution_counts'], committed_optimizer_updates=0,
            ended_UTC=exit_record['ended_utc'], closure=str((prior / 'CLOSURE_RECEIPT.json').relative_to(root)))
        d['jobs'].append(dict(job_id=config['job_id'], status='registered', registered_UTC=stamp,
            config=str(path.relative_to(root)), config_sha256=sha, kind='fixed_candidate_loader_correction',
            operation_budget=config['operation_budget'], resources=config['resources'], scientific_acceptance=False))
    (root / name).write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + '\n').encode())
with (root / 'JOINT_RESPONSE_01_PLAN.md').open('a', encoding='utf8', newline='\n') as stream:
    stream.write('\njob159加载失败于冻结噪声的张量版本检查，新增响应行0；'
        '改回原流程no_grad，作为job160重新登记相同固定候选、角色、噪声、随机流与预算，不覆盖失败记录。\n')
old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
for name in ('REMOTE_DISPATCH.py', 'RUN_REMOTE_DISPATCH.py'):
    text = (prior / name).read_text().replace(remote_old, remote).replace(old_sha, sha).replace(old_path.name, path.name)
    (out / name).write_bytes(text.encode())
collector = (here / 'COLLECT_FIXED_VALIDATION.py').read_text().replace("'fixed_validation_01'", "'fixed_validation_fix01'").replace(remote_old, remote)
(here / 'COLLECT_FIXED_VALIDATION_FIX01.py').write_bytes(collector.encode())
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
record = dict(status='REGISTERED_AND_DEPLOYED', UTC=stamp, job_number=160,
    config_sha256=sha, scientific_settings_unchanged=True, new_optimizer_updates=0, scientific_acceptance=False)
(out / 'REGISTRATION_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
