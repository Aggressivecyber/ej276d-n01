"""Read-only live preflight followed by one owned isolated background job."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def command(args):return subprocess.check_output(args,text=True).strip()
def main():
    assert os.name=='posix' and str(ROOT)=='/home/rylan/projects/ej276d-n01/response_007_20261013_statusfix01'
    cfg=json.loads((ROOT/'response007_config.json').read_text(encoding='utf8'))
    manifest=json.loads((ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json').read_text(encoding='utf8'))
    assert not (ROOT/'runs/science').exists() and not (ROOT/'runs/REMOTE_START.json').exists()
    for relative,digest in manifest['files'].items():assert sha(ROOT/relative)==digest,relative
    raw=Path(cfg['remote_training_container']);assert sha(raw)==cfg['train_container_sha256']
    apps=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    assert not apps,'Another GPU compute process is active; no start'
    process_lines=command(['ps','-eo','pid,args']).splitlines()
    active=[line for line in process_lines if '-m n01.' in line or 'scripts/supervise.py' in line]
    assert not active,'Another N01/supervised scientific job is active; no start'
    gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'])
    memory=command(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits']).split(',')
    assert (int(memory[0])-int(memory[1]))*1024**2>=cfg['gpu_memory_limit_GiB']*1024**3
    usage=shutil.disk_usage(ROOT);assert usage.free>=8*1024**3
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available>=10*1024**3
    runs=ROOT/'runs';runs.mkdir(exist_ok=True)
    preflight=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='passed',GPU=gpu,GPU_compute_processes=apps,
        project_scientific_processes=active,disk_free_bytes=usage.free,RAM_available_bytes=available,
        source_files_verified=len(manifest['files']),source_manifest_sha256=sha(ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json'),
        container_sha256=sha(raw),raw_array_decode=False)
    write(runs/'PREFLIGHT.json',preflight)
    with (runs/'launch.log').open('ab') as log:
        process=subprocess.Popen(['bash',str(ROOT/'scripts/launch_response007_statusfix01.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),supervisor_pid=process.pid,remote_root=str(ROOT),
        config_sha256=sha(ROOT/'response007_config.json'),source_manifest_sha256=sha(ROOT/'RESPONSE007_STATUSFIX01_SOURCE_MANIFEST.json'),
        preflight_sha256=sha(runs/'PREFLIGHT.json'),status='dispatched_not_yet_verified_training')
    write(runs/'REMOTE_START.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
