"""One bounded presence-only SSH wait; no status/metrics or original arrays."""
from pathlib import Path
from datetime import datetime,timezone
import json
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
CODE='''from pathlib import Path
from datetime import datetime,timezone
import json,time,sys
root=Path("/home/rylan/projects/ej276d-n01/wp3_016_saved_readback_20261015/artifacts")
names=["FAILED.json","COMPLETE.json","FIRST_PAIR_PROOF.json"]
started=time.monotonic();deadline=started+300
while True:
    exists={name:(root/name).is_file() for name in names}
    if any(exists.values()) or time.monotonic()>=deadline:
        print(json.dumps(dict(root=str(root),exists=exists,observed_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,maximum_wait_seconds=300,content_reads=0,status_snapshots=0)),flush=True)
        sys.exit(0 if any(exists.values()) else 124)
    time.sleep(min(2,max(0,deadline-time.monotonic())))
'''
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
command=['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10','rylan@100.82.231.102',shlex.join([PYTHON,'-B','-c',CODE])]
begin=datetime.now(timezone.utc).isoformat();clock=time.monotonic()
try:
    r=subprocess.run(command,capture_output=True,timeout=330);stdout=r.stdout.decode('utf8',errors='replace');stderr=r.stderr.decode('utf8',errors='replace')
    result=dict(schema='n01/wp3-016-readback-first-pair-presence-wait/v1',started_UTC=begin,finished_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-clock,command=command,SSH_exit_code=r.returncode,stdout=stdout,stderr=stderr,
        remote_wait_result=json.loads(stdout) if r.returncode in (0,124) and stdout else None,unknown_remote_result=r.returncode not in (0,124),
        SSH_calls=1,maximum_remote_wait_seconds=300,progress_content_reads=0,status_snapshots=0,automatic_retry=False,CP_NN_AD_RNG_training_GPU=0)
except subprocess.TimeoutExpired as error:
    result=dict(schema='n01/wp3-016-readback-first-pair-presence-wait/v1',started_UTC=begin,finished_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-clock,command=command,SSH_exit_code=None,unknown_remote_result=True,
        stdout=(error.stdout or b'').decode('utf8',errors='replace'),stderr=(error.stderr or b'').decode('utf8',errors='replace'),
        SSH_calls=1,maximum_remote_wait_seconds=300,progress_content_reads=0,status_snapshots=0,automatic_retry=False,CP_NN_AD_RNG_training_GPU=0)
with (HERE/'FIRST_PAIR_WAIT_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result.get(k) for k in ['SSH_exit_code','remote_wait_result','unknown_remote_result','elapsed_seconds']},ensure_ascii=False))
