"""Retain the stopped Window008 job, all checkpoints, logs and native receipts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/rylan/projects/ej276d-n01/window008_20261013'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
HOST='rylan@100.82.231.102'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def main():
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=ROOT/'runs/window008_20261013';raw=target/('failure_collection_'+stamp)
    raw.mkdir(parents=True,exist_ok=False)
    code='''from pathlib import Path
import datetime as dt,hashlib,json,subprocess,tarfile
r=Path(ROOT_VALUE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
native=json.loads((r/'runs/supervisor/exit_receipt.json').read_text())
failure=json.loads((r/'runs/science/FAILURE.json').read_text())
if native['exit_code']!=1 or failure['G_updates']!=22269 or failure['D_updates']!=48807:raise ValueError('Unexpected failed endpoint')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
owned=[]
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=proc.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if '-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008':owned.append(dict(pid=int(proc.parent.name),argv=argv))
if owned:raise ValueError('The owned training module is still active')
manifest=json.loads((r/'WINDOW008_SOURCE_MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
 if sha(r/name)!=digest:raise ValueError('Frozen source changed: '+name)
files=[]
for folder in ('runs/science','runs/supervisor'):
 files.extend(p for p in (r/folder).rglob('*') if p.is_file())
files.extend(r/name for name in ('runs/REMOTE_START.json','runs/PREFLIGHT.json','runs/START_ATTEMPT.json'))
inventory={p.relative_to(r).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
archive=r/'incoming'/ARCHIVE_VALUE
if archive.exists():raise FileExistsError(archive)
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files):t.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
result=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='stopped_failed_job_archived',native=native,failure=failure,
 files=inventory,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
 frozen_training_manifest_sha256=sha(r/'WINDOW008_SOURCE_MANIFEST.json'),GPU_compute_processes=apps,
 owned_training_processes=owned,DEV_waveforms_decoded=False,new_training_updates=0)
print(json.dumps(result))
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('ARCHIVE_VALUE',repr('failed_job_'+stamp+'.tar.gz'))
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no',HOST,command],text=True,encoding='utf8'))
    archive=raw/Path(result['archive']).name
    subprocess.run(['scp','-q',HOST+':'+result['archive'],str(archive)],check=True)
    if sha(archive)!=result['archive_sha256']:raise ValueError('Failure archive transfer hash differs')
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if set(member.name for member in members)!=set(result['files']):raise ValueError('Archive inventory differs')
        for member in members:
            destination=(target/member.name).resolve()
            if not member.isfile() or not destination.is_relative_to(target.resolve()):raise ValueError('Unsafe failure artifact path')
            if destination.exists():raise FileExistsError('Preserve earlier local artifact: '+str(destination))
            destination.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,destination.open('xb') as sink:
                for part in iter(lambda:source.read(1024**2),b''):sink.write(part)
            if sha(destination)!=result['files'][member.name]['sha256']:raise ValueError('Reopened artifact hash differs: '+member.name)
    result.update(local_archive=str(archive),local_collection_root=str(target),all_transferred_member_hashes_verified=True)
    receipt=ROOT/'receipts/window008_20261013/FAILURE_COLLECTION.json'
    with receipt.open('x',encoding='utf8') as stream:stream.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='failed_job_collected_and_hash_verified',files=len(result['files']),bytes=result['archive_bytes'],receipt=str(receipt))))


if __name__=='__main__':main()
