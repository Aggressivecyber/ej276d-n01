from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent
cp=ROOT/'configs/joint_response_01_recentered_restoration_solves_fix01_20261020.json'
p=json.loads(cp.read_bytes());digest=hashlib.sha256(cp.read_bytes()).hexdigest()
assert not (HERE/'LAUNCH.json').exists()
assert p['resources']['CPU_threads']==4 and p['resources']['GPU_GiB']==0
assert p['radius']==.003125 and p['native_linear_cap_factor']==1.005
assert 106+7*(817+12*57)+42+54==10709
for name,entry in p['source_pins'].items():
    raw=Path(name).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    if name.endswith('.py'):ast.parse(raw.decode('utf-8'))
environment=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
command=[sys.executable,'-B',str(ROOT/'scripts/solve_joint01_recentered_restoration.py'),'--config',str(cp),'--config-sha256',digest]
started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
with (HERE/'SOLVE_STDOUT.json').open('xb') as stdout,(HERE/'SOLVE_STDERR.txt').open('xb') as stderr:
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=environment)
    (HERE/'LAUNCH.json').write_text(json.dumps(dict(UTC=started,pid=process.pid,command=command,config_sha256=digest,
        source_pins_verified=len(p['source_pins']),process='Windows local CPU; existing NAS files in process memory only'),indent=2)+'\n',encoding='utf-8')
    for filename in ('PROGRAM.json','MASTER_EXECUTION_PLAN.json','jobs.json'):
        path=ROOT/filename;d=json.loads(path.read_bytes())
        assert d['active_scientific_jobs']==0 and d['pending_registered_job_id']==p['job_id']
        d.update(status='JOINT_RESPONSE_01_fresh_restoration_running',updated_UTC=started,
            active_scientific_jobs=1,active_job_id=p['job_id'],pending_registered_job_id=None,
            current_action='Execute the two predeclared fresh-center convex restoration solves; no generator or physical candidate execution.',
            next_action='Independently audit the saved constraint matrices and certificates before any actual candidate registration.')
        d['joint_response_01']['recentered_restoration_status']='running'
        if filename=='jobs.json':
            job=d['jobs'][-1];assert job['job_id']==p['job_id']
            job.update(status='running',started_UTC=started,pid=process.pid,launch=HERE.relative_to(ROOT).as_posix()+'/LAUNCH.json')
        path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    try:
        code=process.wait(timeout=p['resources']['wall_seconds'])
    except subprocess.TimeoutExpired:
        process.kill();process.wait()
        (HERE/'TIMEOUT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),pid=process.pid,
            wall_seconds=p['resources']['wall_seconds'],scientific_restart=False),indent=2)+'\n',encoding='utf-8')
        raise
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=code,elapsed_seconds=time.perf_counter()-timer,
    pid=process.pid,config_sha256=digest)
(HERE/'EXIT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
assert code==0,(HERE/'SOLVE_STDERR.txt').read_text()
print(json.dumps(receipt))
