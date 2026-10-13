"""Start only one zero-update replay after the owned science job has exited."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert str(ROOT)=='/home/rylan/projects/ej276d-n01/response_007_20261013_deadline01'
    binding=json.loads((ROOT/'RESPONSE007_DEADLINE01_QA_BINDING.json').read_text())
    for name,digest in binding['remote_files'].items():assert sha(ROOT/name)==digest,name
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime(2026,10,13,3,30,tzinfo=dt.timezone.utc)
    native=json.loads((ROOT/'runs/supervisor/exit_receipt.json').read_text())
    complete=json.loads((ROOT/'runs/science/COMPLETE.json').read_text())
    assert native['exit_code']==0 and complete['G_updates']==8000 and complete['D_updates']==6000
    assert not (ROOT/'runs/verification_deadline01').exists() and not (ROOT/'runs/verification_supervisor').exists()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not apps,'GPU has an active process; replay not started'
    with (ROOT/'runs/verification_launch.log').open('ab') as log:
        process=subprocess.Popen(['bash',str(ROOT/'scripts/launch_response007_deadline01_replay.sh')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),supervisor_pid=process.pid,
        status='engineering_replay_dispatched_not_yet_passed',new_training_updates=0,new_unique_random_draws=0,
        binding_sha256=sha(ROOT/'RESPONSE007_DEADLINE01_QA_BINDING.json'),scientific_complete_sha256=sha(ROOT/'runs/science/COMPLETE.json'))
    (ROOT/'runs/REPLAY_REMOTE_START.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
