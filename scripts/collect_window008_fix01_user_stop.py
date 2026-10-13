"""Recover the intentionally stopped job before the server is shut down."""
from pathlib import Path
import datetime as dt
import json
import shlex
import subprocess
import tarfile
from window008_fix01_runtime import ROOT,read,sha,write_new

HOST='rylan@100.82.231.102'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'

def main():
    cfg=read(ROOT/'window008_fix01_config.json');stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=ROOT/'runs/window008_fix01_20261013';raw=target/('user_stop_collection_'+stamp);raw.mkdir(parents=True,exist_ok=False)
    code='''from pathlib import Path
import datetime as dt,hashlib,json,subprocess,tarfile
r=Path(ROOT_VALUE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
stop=json.loads((r/'runs/USER_STOP_VERIFICATION.json').read_text())
native=json.loads((r/'runs/supervisor/exit_receipt.json').read_text())
failure=json.loads((r/'runs/science/FAILURE.json').read_text())
if stop['status']!='user_requested_stop_verified' or stop['files']['supervisor/exit_receipt.json']!=native or stop['files']['science/FAILURE.json']!=failure:raise ValueError('Stop/native/artifact binding differs')
owned=[]
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=proc.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if '-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008_fix01' and str(r/'runs/science') in argv:owned.append(int(proc.parent.name))
if owned:raise ValueError('Owned trainer still active')
manifest=json.loads((r/'WINDOW008_FIX01_SOURCE_MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
 if sha(r/name)!=digest:raise ValueError('Frozen source differs: '+name)
files=[]
for folder in ('runs/science','runs/supervisor'):files.extend(p for p in (r/folder).rglob('*') if p.is_file())
files.extend(p for p in (r/'runs').iterdir() if p.is_file())
inventory={p.relative_to(r).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
archive=r/'incoming'/ARCHIVE_VALUE
if archive.exists():raise FileExistsError(archive)
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files):t.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='user_stopped_job_archived',
 native=native,failure=failure,files=inventory,archive=str(archive),archive_bytes=archive.stat().st_size,
 archive_sha256=sha(archive),frozen_manifest_sha256=sha(r/'WINDOW008_FIX01_SOURCE_MANIFEST.json'),
 owned_training_processes=owned,new_training_updates_during_collection=0,DEV_numeric_decodes=0)))
'''.replace('ROOT_VALUE',repr(cfg['remote_root'])).replace('ARCHIVE_VALUE',repr('user_stop_'+stamp+'.tar.gz'))
    command=' '.join([shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','-o','ConnectTimeout=8',HOST,command],text=True,encoding='utf8'))
    archive=raw/Path(result['archive']).name
    subprocess.run(['scp','-q','-o','ConnectTimeout=8',HOST+':'+result['archive'],str(archive)],check=True)
    if sha(archive)!=result['archive_sha256']:raise ValueError('Transferred archive hash differs')
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if set(member.name for member in members)!=set(result['files']):raise ValueError('Archive inventory differs')
        for member in members:
            path=(target/member.name).resolve()
            if not member.isfile() or not path.is_relative_to(target.resolve()):raise ValueError('Unsafe archive member')
            if path.exists():raise FileExistsError('Preserve prior local artifact: '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,path.open('xb') as sink:
                for block in iter(lambda:source.read(1024**2),b''):sink.write(block)
            if sha(path)!=result['files'][member.name]['sha256']:raise ValueError('Reopened artifact differs: '+member.name)
    result.update(local_archive=str(archive),local_collection_root=str(target),all_transferred_member_hashes_verified=True,
        resume_requires_coherent_boundary_checkpoint_validation=True)
    write_new(ROOT/'receipts/window008_fix01_20261013/USER_STOP_COLLECTION.json',result)
    print(json.dumps(dict(status='user_stopped_job_collected_and_hash_verified',files=len(result['files']),archive_bytes=result['archive_bytes'])))

if __name__=='__main__':main()
