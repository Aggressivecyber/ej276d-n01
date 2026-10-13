"""Fresh source/resource preflight and one exclusive fix01 dispatch."""
from pathlib import Path
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from window008_resume01_runtime import ROOT,RUN_ID,bound,read,sha,write_new,require,validate_evidence


def main():
    cfg,base,legacy=validate_evidence()
    require(os.name=='posix' and str(ROOT)==cfg['remote_root'],'Run only on the bound Linux root')
    import fcntl
    runs=ROOT/'runs';runs.mkdir(exist_ok=True)
    with (runs/'.start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ('science','REMOTE_START.json','START_ATTEMPT.json','PREFLIGHT.json','LAUNCH_FAILURE.json','supervisor/launch.json'):
            require(not (runs/name).exists(),'Preserve prior dispatch: '+name)
        manifest=read(ROOT/cfg['source_manifest'])
        require(manifest['schema']=='n01/window008-resume01-source/v1' and manifest['run_id']==RUN_ID and manifest['before_scientific_model_forward'] is True,'Wrong source manifest')
        require(manifest['config_sha256']==sha(ROOT/'window008_resume01_config.json') and manifest['base_manifest_sha256']==cfg['legacy_source_manifest_sha256'],'Source/config identity differs')
        require(manifest['fix01_manifest_sha256']==cfg['fix01_source_manifest_sha256'],'Parent fix01 source identity differs')
        for name,digest in manifest['files'].items():require(sha(bound(ROOT,name))==digest,'Frozen runtime differs: '+name)
        require(sha(Path(base['remote_training_container']))==base['train_container_sha256'],'Original container differs')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        require(not apps,'Another GPU compute job is active')
        active=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:argv=path.read_bytes().decode().strip('\0').split('\0')
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if '-m' in argv and argv[argv.index('-m')+1].startswith('n01.train_'):active.append(dict(pid=int(path.parent.name),argv=argv))
        require(not active,'Another N01 training module is active')
        total,used=map(int,subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
        require((total-used)*1024**2>=cfg['gpu_memory_limit_GiB']*1024**3,'Insufficient free GPU0 memory')
        disk=shutil.disk_usage(ROOT).free
        memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        require(disk>=8*1024**3 and memory>=10*1024**3,'Insufficient free disk/RAM')
        stamp=dt.datetime.now(dt.timezone.utc).isoformat()
        preflight=dict(UTC=stamp,status='passed',source_manifest_sha256=sha(ROOT/cfg['source_manifest']),
            config_sha256=sha(ROOT/'window008_resume01_config.json'),source_commit=manifest['source_commit'],
            source_files_verified=len(manifest['files']),GPU_compute_processes=apps,active_N01_jobs=active,
            disk_free_bytes=disk,RAM_available_bytes=memory,new_G_budget=32000,new_D_budget=87000,
            prior_execution_native_exit_code=1,DEV_numeric_decodes=0)
        write_new(runs/'START_ATTEMPT.json',dict(UTC=stamp,dispatcher_pid=os.getpid(),source_manifest_sha256=preflight['source_manifest_sha256']))
        process=None
        try:
            write_new(runs/'PREFLIGHT.json',preflight)
            with (runs/'launch.log').open('xb') as stream:
                process=subprocess.Popen(['bash',str(ROOT/'scripts/launch_window008_resume01.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            time.sleep(1)
            require(process.poll() is None,'Launcher exited before native confirmation')
            result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='dispatched_pending_first_observed_update',
                supervisor_pid=process.pid,remote_root=str(ROOT),source_commit=manifest['source_commit'],
                source_manifest_sha256=preflight['source_manifest_sha256'],recovery_config_sha256=preflight['config_sha256'],
                new_G_budget=32000,new_D_budget=87000,scientific_update_observed=False)
            write_new(runs/'REMOTE_START.json',result);print(json.dumps(result))
        except BaseException as error:
            write_new(runs/'LAUNCH_FAILURE.json',dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),error=repr(error),
                supervisor_pid=None if process is None else process.pid,automatic_retry_allowed=False))
            raise


if __name__=='__main__':main()
