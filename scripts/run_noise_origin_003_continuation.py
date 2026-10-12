"""Continue only unstarted REAL-REAL fits after a pre-fit CUDA API guard repair."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

from run_noise_origin_003 import owned_bytes,stage_completion,write,utc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/science'

def main():
    configp=ROOT/'noise_origin_003_config.json'
    config=json.loads(configp.read_text(encoding='utf8'))
    if str(ROOT)!=config['remote_root']+'/continuation_001':
        raise RuntimeError('Bound continuation root mismatch')
    manifest=json.loads((ROOT/'CONTINUATION_BINDING.json').read_text(encoding='utf8'))
    for item in manifest['files']:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Continuation source binding mismatch: '+item['path'])
    prior=json.loads((ROOT.parent/'runs/science/origin/COMPLETE.json').read_text(encoding='utf8'))
    originp=ROOT.parent/'runs/science/origin/ORIGIN_REPORT.json'
    if prior['status']!='complete' or prior['report_sha256']!=hashlib.sha256(originp.read_bytes()).hexdigest():
        raise RuntimeError('Parent origin completion/hash mismatch')
    if json.loads(originp.read_text())['config_sha256']!=hashlib.sha256(configp.read_bytes()).hexdigest():
        raise RuntimeError('Parent origin and continuation config mismatch')
    if prior['rows_completed']!=21047 or not prior['bitwise_target_equal']:
        raise RuntimeError('Original successful origin receipt absent')
    if (OUT/'status.json').exists():
        raise FileExistsError('Preserve existing continuation')
    status={'experiment_id':config['experiment_id'],'status':'running','phase':'real_real_only',
            'started_UTC':utc(),'pid':os.getpid(),'prior_classifier_updates':0,'noise_training_updates':0,
            'response_training_updates':0,'origin_recomputed':False,'compatibility_fix':'explicit_same_logical_cuda_index_0'}
    write(OUT/'status.json',status)
    cmd=[sys.executable,'-B',str(ROOT/'scripts/calibrate_real_real_null.py'),'--config',str(configp),
         '--out',str(OUT/'real_real'),'--device','cuda','--container',config['remote_training_container'],
         '--identities',str(ROOT/'data/PARENT_IDENTITIES.npz')]
    proc=None;code=0;start=time.monotonic();peak_bytes=0
    try:
        proc=subprocess.Popen(cmd,cwd=ROOT)
        while proc.poll() is None:
            size=owned_bytes();peak_bytes=max(size,peak_bytes)
            if size>1024**3 or time.monotonic()-start>1800:
                raise RuntimeError('Registered output or wall budget exceeded')
            time.sleep(1)
        status['native_classifier_exit_code']=proc.wait()
        if proc.returncode!=0:
            raise RuntimeError('Fixed REAL-REAL classifier stage failed')
        status.update(stage_completion('real_real'))
        size=owned_bytes();peak_bytes=max(size,peak_bytes)
        if size>1024**3:
            raise RuntimeError('Final registered output budget exceeded')
        status.update(status='completed',evaluation_classifier_updates=25600)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        code=1;status.update(status='failed',error=repr(exc))
    status.update(ended_UTC=utc(),exit_code=code,elapsed_s=time.monotonic()-start,
                  peak_observed_owned_root_bytes=peak_bytes,output_monitor_scope='entire_dedicated_continuation_root')
    write(OUT/'status.json',status);write(OUT/'CONTINUATION_EXIT.json',status)
    return code

if __name__=='__main__':
    raise SystemExit(main())
