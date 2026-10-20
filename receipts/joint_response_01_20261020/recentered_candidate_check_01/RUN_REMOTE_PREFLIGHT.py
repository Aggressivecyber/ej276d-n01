from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess

HERE=Path(__file__).resolve().parent
script=(HERE/'REMOTE_PREFLIGHT.py').read_bytes()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
with (HERE/'NATIVE_PREFLIGHT_STDOUT.jsonl').open('xb') as stdout,(HERE/'NATIVE_PREFLIGHT_STDERR.txt').open('xb') as stderr:
    try:
        r=subprocess.run(['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10','rylan@100.82.231.102',
            'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
            input=script,stdout=stdout,stderr=stderr,timeout=240,env=env)
    except subprocess.TimeoutExpired:
        (HERE/'PREFLIGHT_TIMEOUT.json').write_text(json.dumps(dict(status='EXISTING_SCIENTIFIC_PATH_PREFLIGHT_TIMEOUT',
            UTC=datetime.now(timezone.utc).isoformat(),seconds=240,scientific_dispatch_attempted=False,
            storage_configuration_changes=False,automatic_retry=False),indent=2)+'\n',encoding='utf-8')
        raise
(HERE/'PREFLIGHT_EXIT.json').write_text(json.dumps(dict(exit_code=r.returncode,UTC=datetime.now(timezone.utc).isoformat()),indent=2)+'\n',encoding='utf-8')
r.check_returncode()
record=json.loads((HERE/'NATIVE_PREFLIGHT_STDOUT.jsonl').read_bytes().splitlines()[-1])
assert record['status']=='PASS_NATIVE_EXISTING_PATH_SCIENTIFIC_PREFLIGHT'
(HERE/'NATIVE_PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
