"""Fresh resource/source preflight and one owned Window008 training dispatch."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import time

from freeze_window008 import RUN_ID,bound,check_hashes,read,require,validate_evidence

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
def command(args):return subprocess.check_output(args,text=True).strip()


def main():
    cfg=read(ROOT/'window008_config.json')
    require(os.name=='posix' and str(ROOT)==cfg['remote_root'],'Run only on the bound Linux root')
    import fcntl
    runs=ROOT/'runs';runs.mkdir(exist_ok=True)
    with (runs/'.start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return dispatch(cfg,runs)


def dispatch(cfg,runs):
    for name in ('science','REMOTE_START.json','START_ATTEMPT.json','PREFLIGHT.json','LAUNCH_FAILURE.json','supervisor/launch.json'):
        require(not (runs/name).exists(),'Preserve prior dispatch attempt: '+name)
    manifest=read(bound(ROOT,cfg['source_manifest']))
    require(manifest['schema']=='n01/window008/source_manifest/v1' and manifest['run_id']==cfg['run_id']==RUN_ID,'Manifest version/run differs')
    require(manifest['before_scientific_model_forward'] is True,'Manifest was not frozen before science')
    check_hashes(ROOT,manifest['files'],'Frozen manifest',('src/n01/train_window008.py','src/n01/response008.py',
        'scripts/start_window008_remote.py','scripts/freeze_window008.py','scripts/launch_window008.sh',
        'scripts/supervise.py','data/PARENT_IDENTITIES.npz',cfg['dataset'],cfg['statistics']))
    require(sha(ROOT/'window008_config.json')==manifest['config_sha256'],'Frozen config differs')
    require(sha(Path(cfg['remote_training_container']))==cfg['train_container_sha256'],'Original remote data differs')
    evidence=validate_evidence(ROOT,cfg)
    for key,name in (('engineering_cuda_receipt','qa'),('engineering_cuda_native_exit','native'),
                     ('full_batch_receipt','full'),('full_batch_native_exit','full_native'),
                     ('steady_full_batch_receipt','steady'),('steady_full_batch_native_exit','steady_native')):
        require(bound(ROOT,manifest[key])==evidence[name], 'Manifest evidence path differs: '+key)
    apps=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    require(not apps,'Another GPU compute job is active')
    active=[line for line in command(['ps','-eo','pid,args']).splitlines() if '-m n01.' in line or 'scripts/supervise.py' in line]
    require(not active,'Another project/supervised job is active')
    gpu=command(['nvidia-smi','--id=0','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
    total,used=map(int,command(['nvidia-smi','--id=0','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits']).split(','))
    require((total-used)*1024**2>=cfg['gpu_memory_limit_GiB']*1024**3,'Insufficient free GPU0 memory')
    disk=shutil.disk_usage(ROOT)
    require(disk.free>=8*1024**3,'Insufficient output disk space')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    require(available>=10*1024**3,'Insufficient available RAM')
    preflight=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='passed',GPU=gpu,GPU_compute_processes=apps,
        active_project_jobs=active,disk_free_bytes=disk.free,RAM_available_bytes=available,
        source_manifest_sha256=sha(ROOT/cfg['source_manifest']),source_files_verified=len(manifest['files']),
        config_sha256=sha(ROOT/'window008_config.json'),raw_array_decode=False,scientific_updates=0)
    # The claim remains even if the dispatcher dies between Popen and receipt.
    write(runs/'START_ATTEMPT.json',dict(UTC=preflight['UTC'],dispatcher_pid=os.getpid(),
        source_manifest_sha256=preflight['source_manifest_sha256'],config_sha256=preflight['config_sha256']))
    process=None
    try:
        write(runs/'PREFLIGHT.json',preflight)
        with (runs/'launch.log').open('xb') as log:
            process=subprocess.Popen(['bash',str(ROOT/'scripts/launch_window008.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(1.)
        code=process.poll()
        require(code is None,'Launcher exited before native training confirmation: '+str(code))
        receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='dispatched_pending_native_confirmation',
            supervisor_pid=process.pid,remote_root=str(ROOT),source_commit=manifest['source_commit'],
            source_manifest_sha256=preflight['source_manifest_sha256'],config_sha256=preflight['config_sha256'],
            preflight_sha256=sha(runs/'PREFLIGHT.json'),G_budget=57000,D_budget=144000,
            scientific_update_observed=False,native_exit_observed=False)
        write(runs/'REMOTE_START.json',receipt);print(json.dumps(receipt))
    except BaseException as error:
        write(runs/'LAUNCH_FAILURE.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            status='dispatch_error_preserved_process_state_requires_inspection',error=repr(error),
            supervisor_pid=None if process is None else process.pid,native_exit=None if process is None else process.poll(),
            automatic_retry_allowed=False,scientific_update_observed=False))
        raise


if __name__=='__main__':main()
