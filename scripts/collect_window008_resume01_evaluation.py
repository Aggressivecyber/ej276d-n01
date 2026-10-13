"""Collect one completed evaluation job and reopen every transferred file hash."""
from pathlib import Path, PurePosixPath
import argparse
import datetime as dt
import json
import shlex
import subprocess
import tarfile
from window008_resume01_runtime import ROOT, read, sha, write_new

HOST = 'rylan@100.82.231.102'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def collect(kind):
    cfg = read(ROOT / 'window008_resume01_config.json')
    target = ROOT / 'runs' / cfg['run_id']
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw = target / (kind + '_collection_' + stamp)
    raw.mkdir(parents=True, exist_ok=False)
    receipt = ROOT / 'receipts' / cfg['run_id'] / (kind.upper() + '_COLLECTION.json')
    if receipt.exists():
        raise FileExistsError('Preserve completed collection: ' + str(receipt))
    code = '''from pathlib import Path
import datetime as dt,hashlib,json,sys,tarfile
r=Path(ROOT_VALUE);kind=KIND_VALUE
sys.path.insert(0,str(r/'scripts'));sys.path.insert(0,str(r/'src'))
from window008_resume01_evaluation_common import completed_job,sha,read
result=completed_job(kind,r)
job=result['paths']['job']
files=sorted(p for p in job.rglob('*') if p.is_file())
if any(p.is_symlink() for p in files):raise ValueError('Symlink in job outputs')
inventory={p.relative_to(r).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
archive=r/'incoming'/ARCHIVE_VALUE
with tarfile.open(archive,'x:gz') as bundle:
 for p in files:bundle.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
print(json.dumps(dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),kind=kind,status='complete_job_archived',
 native=result['native'],core_native=result['core']['exit_code'],summary_sha256=sha(result['paths']['artifacts']/'SUMMARY.json'),
 files=inventory,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
 manifest_sha256=sha(r/'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'),
 optimizer_updates_during_collection=0,new_model_forwards=0)))
'''.replace('ROOT_VALUE', repr(cfg['remote_root'])).replace('KIND_VALUE', repr(kind)).replace('ARCHIVE_VALUE', repr(kind+'_'+stamp+'.tar.gz'))
    command = ' '.join(['env', 'PYTHONDONTWRITEBYTECODE=1', shlex.quote(PYTHON), '-B', '-c', shlex.quote(code)])
    result = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ForwardX11=no',
        '-o', 'ConnectTimeout=8', HOST, command], text=True, encoding='utf8'))
    write_new(raw/'REMOTE_COLLECTION.json', result)
    archive = raw / Path(result['archive']).name
    subprocess.run(['scp', '-q', '-o', 'ConnectTimeout=8', HOST+':'+result['archive'], str(archive)], check=True)
    if sha(archive) != result['archive_sha256']:
        raise ValueError('Transferred archive differs')
    with tarfile.open(archive,'r:gz') as bundle:
        members = bundle.getmembers()
        if len(members) != len(result['files']) or {m.name for m in members} != set(result['files']):
            raise ValueError('Archive inventory differs')
        for member in members:
            relative = PurePosixPath(member.name)
            path = (target/member.name).resolve()
            if (not member.isfile() or member.issym() or member.islnk() or relative.is_absolute() or '..' in relative.parts
                    or '\\' in member.name or ':' in member.name or not path.is_relative_to(target.resolve())):
                raise ValueError('Unsafe archive member')
            if path.exists():
                raise FileExistsError('Preserve local prior file: '+str(path))
            path.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source, path.open('xb') as sink:
                for block in iter(lambda:source.read(1024**2), b''): sink.write(block)
            if path.stat().st_size != result['files'][member.name]['bytes'] or sha(path) != result['files'][member.name]['sha256']:
                raise ValueError('Reopened transferred file differs: '+member.name)
    result.update(local_archive=str(archive),local_collection_root=str(target),all_transferred_member_hashes_verified=True)
    write_new(receipt,result)
    return dict(kind=kind,status='collected_and_hash_verified',files=len(result['files']),archive_bytes=result['archive_bytes'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('primary','replay01'),required=True)
    print(json.dumps(collect(parser.parse_args().kind)))
