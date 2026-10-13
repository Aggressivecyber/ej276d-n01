"""Collect the fixed completed resume job without altering scientific artifacts."""
from pathlib import Path, PurePosixPath
import datetime as dt
import json
import shlex
import subprocess
import tarfile
from window008_resume01_runtime import ROOT, read, sha, write_new

HOST = 'rylan@100.82.231.102'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def main():
    cfg = read(ROOT / 'window008_resume01_config.json')
    receipt = ROOT / 'receipts/window008_resume01_20261013/COMPLETE_COLLECTION.json'
    if receipt.exists():
        raise FileExistsError('Preserve existing completed collection')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = ROOT / 'runs/window008_resume01_20261013'
    raw = target / ('complete_collection_' + stamp)
    raw.mkdir(parents=True, exist_ok=False)
    code = '''from pathlib import Path
import datetime as dt,hashlib,json,tarfile
r=Path(ROOT_VALUE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
native=read(r/'runs/supervisor/exit_receipt.json')
launch=read(r/'runs/supervisor/launch.json')
complete=read(r/'runs/science/TRAINING_COMPLETE.json')
if native['exit_code']!=0 or native.get('reason') is not None:raise ValueError('No native success')
if any(native[k]!=launch[k] for k in ('supervisor_pid','child_pid')):raise ValueError('Native launch chain differs')
if complete['schema']!='n01/window008-resume01-training-complete/v1':raise ValueError('Wrong completion schema')
if (complete['new_execution_G_updates'],complete['new_execution_D_updates'])!=(32000,87000):raise ValueError('Incomplete fixed budget')
if (r/'runs/science/FAILURE.json').exists():raise ValueError('Failure exists')
owned=[]
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=proc.read_bytes().decode().strip('\\0').split('\\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if '-m' in argv and argv[argv.index('-m')+1]=='n01.train_window008_resume01' and str(r/'runs/science') in argv:owned.append(int(proc.parent.name))
if owned:raise ValueError('Owned trainer still active')
manifest=read(r/'WINDOW008_RESUME01_SOURCE_MANIFEST.json')
for name,digest in manifest['files'].items():
 if sha(r/name)!=digest:raise ValueError('Frozen source differs: '+name)
for name,digest in complete['files'].items():
 if sha(r/'runs/science'/name)!=digest:raise ValueError('Completion file differs: '+name)
for arm,entry in complete['endpoint_map'].items():
 if sha(r/entry['path'])!=entry['sha256']:raise ValueError('Endpoint differs: '+arm)
files=[]
for folder in ('runs/science','runs/supervisor'):files.extend(p for p in (r/folder).rglob('*') if p.is_file())
files.extend(p for p in (r/'runs').iterdir() if p.is_file())
if any(p.is_symlink() for p in files):raise ValueError('Symlink in outputs')
inventory={p.relative_to(r).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
archive=r/'incoming'/ARCHIVE_VALUE
with tarfile.open(archive,'x:gz') as t:
 for p in sorted(files):t.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='completed_resume01_job_archived',native=native,
 training_complete_sha256=sha(r/'runs/science/TRAINING_COMPLETE.json'),files=inventory,archive=str(archive),
 archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
 frozen_manifest_sha256=sha(r/'WINDOW008_RESUME01_SOURCE_MANIFEST.json'),frozen_files_verified=len(manifest['files']),
 endpoint_map=complete['endpoint_map'],owned_training_processes=owned,optimizer_updates_during_collection=0,DEV_numeric_decodes=0)))
'''.replace('ROOT_VALUE', repr(cfg['remote_root'])).replace('ARCHIVE_VALUE', repr('complete_' + stamp + '.tar.gz'))
    command = ' '.join([shlex.quote(PYTHON), '-B', '-c', shlex.quote(code)])
    result = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ForwardX11=no',
        '-o', 'ConnectTimeout=8', HOST, command], text=True, encoding='utf8'))
    write_new(raw / 'REMOTE_COLLECTION.json', result)
    archive = raw / Path(result['archive']).name
    subprocess.run(['scp', '-q', '-o', 'ConnectTimeout=8', HOST + ':' + result['archive'], str(archive)], check=True)
    if sha(archive) != result['archive_sha256']:
        raise ValueError('Transferred archive hash differs')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        if len(members) != len(result['files']) or {m.name for m in members} != set(result['files']):
            raise ValueError('Archive inventory differs')
        for member in members:
            relative = PurePosixPath(member.name)
            path = (target / member.name).resolve()
            if (not member.isfile() or member.issym() or member.islnk() or relative.is_absolute()
                    or '..' in relative.parts or '\\' in member.name or ':' in member.name
                    or not path.is_relative_to(target.resolve())):
                raise ValueError('Unsafe archive member')
            if path.exists():
                raise FileExistsError('Preserve prior local artifact: ' + str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, path.open('xb') as sink:
                for block in iter(lambda: source.read(1024 ** 2), b''):
                    sink.write(block)
            if path.stat().st_size != result['files'][member.name]['bytes'] or sha(path) != result['files'][member.name]['sha256']:
                raise ValueError('Reopened artifact differs: ' + member.name)
    result.update(local_archive=str(archive), local_collection_root=str(target),
        all_transferred_member_hashes_verified=True, scientific_status='response_match_not_established')
    write_new(receipt, result)
    print(json.dumps(dict(status='completed_resume01_collected_and_hash_verified', files=len(result['files']),
        archive_bytes=result['archive_bytes'], native_exit_code=result['native']['exit_code'])))


if __name__ == '__main__':
    main()
