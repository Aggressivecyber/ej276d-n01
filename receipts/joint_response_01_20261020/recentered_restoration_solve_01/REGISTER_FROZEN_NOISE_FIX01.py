from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
F=HERE.parent/'recentered_restoration_solve_fix01';F.mkdir(exist_ok=False)
old_cp=ROOT/'configs/joint_response_01_recentered_restoration_solves_20261020.json'
p=json.loads(old_cp.read_bytes());old_digest=hashlib.sha256(old_cp.read_bytes()).hexdigest()
exit_record=json.loads((HERE/'EXIT_RECEIPT.json').read_bytes())
assert exit_record['exit_code']==1 and not Path(p['report']).exists() and not Path(p['progress']).exists()
assert list(Path(p['NAS_output']).iterdir())==[]
changes=[('solve_joint01_recentered_restoration.py','assert center.shape==(54,)','assert center.shape==(54,) and center[53]==0'),
    ('audit_joint01_recentered_restoration.py',"assert saved['candidate'][53]==center[53] and float(saved['radius'])==radius", "assert saved['candidate'][53]==center[53]==0 and float(saved['radius'])==radius")]
originals={}
for name,after,before in changes:
    source=ROOT/'scripts'/name;original=HERE/'engineering_fix01'/name
    raw=source.read_bytes();old=original.read_bytes()
    assert hashlib.sha256(old).hexdigest()==p['source_pins'][str(source)]['sha256']
    text=raw.decode('utf-8');assert text.count(after)==1
    assert text.replace(after,before)==old.decode('utf-8')
    ast.parse(text)
    originals[str(original)]=dict(sha256=hashlib.sha256(old).hexdigest(),bytes=len(old))
now=datetime.now(timezone.utc).isoformat()
failure=dict(status='AUDITED_PREFLIGHT_FAILURE_ZERO_SOLVES',UTC=now,exit_receipt=exit_record,
    cause='Incorrect absolute-zero assertion for the frozen noise coordinate. Saved center has tiny nonzero values; protocol freezes its correction, not its value.',
    observed_center_noise={'0':1.586863287121728e-21,'1':7.420802711335723e-20},
    original_sources=originals,source_changes='Only remove two absolute-zero assertions; preserve saved noise coordinate byte-for-byte.',
    solver_invocations=0,generator_calls=0,actual_candidate_evaluations=0,parameter_updates=0,
    NAS_output_directory_empty=True,old_configuration_retained=True,scientific_acceptance=False)
(HERE/'PREFLIGHT_FAILURE_REVIEW.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8',newline='\n')
old_id=p['job_id'];p['job_id']='JOINT_RESPONSE_01_RECENTERED_RESTORATION_SOLVES_FIX01_20261020'
p.update(registered_UTC=now,previous_config=dict(path=str(old_cp),sha256=old_digest),
    engineering_correction='Frozen noise correction is zero; candidate coordinate equals its exact saved center value.',
    previous_solver_invocations=0,NAS_output='Z:/GAN/research/N01/server_runs/joint_response01_recentered_restoration_solve_fix01_20261020/run_01',
    report=str(F/'SOLVE_RESULT.json'),progress=str(F/'PROGRESS.json'),audit_report=str(F/'INDEPENDENT_AUDIT.json'))
for name in p['source_pins']:
    raw=Path(name).read_bytes();p['source_pins'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
cp=ROOT/'configs/joint_response_01_recentered_restoration_solves_fix01_20261020.json'
assert not cp.exists();cp.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8',newline='\n')
digest=hashlib.sha256(cp.read_bytes()).hexdigest()
wrapper=(HERE/'RUN_SOLVES.py').read_text(encoding='utf-8')
assert wrapper.count('joint_response_01_recentered_restoration_solves_20261020.json')==1
wrapper=wrapper.replace('joint_response_01_recentered_restoration_solves_20261020.json',cp.name)
ast.parse(wrapper);(F/'RUN_SOLVES.py').write_text(wrapper,encoding='utf-8',newline='\n')
rel=HERE.relative_to(ROOT).as_posix();frel=F.relative_to(ROOT).as_posix()
for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
    path=ROOT/filename;d=json.loads(path.read_bytes());assert d['active_job_id']==old_id
    d.update(status='JOINT_RESPONSE_01_fresh_restoration_fix01_registered',updated_UTC=now,
        active_scientific_jobs=0,active_job_id=None,pending_registered_job_id=p['job_id'],
        current_action='Closed the preflight assertion failure before any solve. Registered minimal fix preserving the exact frozen noise value; original two-solve budget remains unconsumed.',
        next_action='Execute the same fixed two restoration programs with frozen noise correction, then independent matrix/KKT audit.',
        active_scientific_jobs_status_basis='Local preflight process exited1; no solver invocation, output payload or current scientific process.')
    d['joint_response_01'].update(recentered_restoration_status='fix01_registered_not_started',
        recentered_restoration_preflight_failure=rel+'/PREFLIGHT_FAILURE_REVIEW.json',
        recentered_restoration_config=cp.relative_to(ROOT).as_posix(),recentered_restoration_config_sha256=digest)
    if filename=='jobs.json':
        assert len(d['jobs'])==143 and d['jobs'][-1]['job_id']==old_id
        first142=hashlib.sha256(json.dumps(d['jobs'][:142],sort_keys=True).encode()).hexdigest()
        d['jobs'][-1].update(status='failed_preflight_zero_solves',native_exit_code=1,verified_UTC=now,
            failure_review=rel+'/PREFLIGHT_FAILURE_REVIEW.json',solver_invocations=0,generator_calls=0,
            actual_candidate_evaluations=0,parameter_updates=0)
        d['jobs'].append(dict(job_id=p['job_id'],status='registered_not_started',registered_UTC=now,
            config=cp.relative_to(ROOT).as_posix(),config_sha256=digest,operation_budget=p['operation_budget'],
            resources=p['resources'],source_files=len(p['source_pins']),matrix_inputs=14,
            previous_preflight_failure=old_id,previous_solver_invocations=0,
            automatic_restart=False,automatic_training=False,scientific_acceptance=False))
        assert hashlib.sha256(json.dumps(d['jobs'][:142],sort_keys=True).encode()).hexdigest()==first142
    path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
receipt=dict(status='PASS_MINIMAL_FROZEN_NOISE_CORRECTION_REGISTERED',UTC=now,config_sha256=digest,
    source_changes=2,prior_solver_invocations=0,new_solver_limit=2,problem_bounds_thresholds_and_radius_unchanged=True,
    original_failed_job_preserved=True,scientific_acceptance=False)
(F/'REGISTRATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
