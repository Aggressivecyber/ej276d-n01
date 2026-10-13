"""Stage an owned Window008 runtime and optionally run its synthetic CUDA check."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import datetime as dt

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/rylan/projects/ej276d-n01/window008_20261013'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'
HOST='rylan@100.82.231.102'
SSH=['ssh','-o','BatchMode=yes','-o','ForwardX11=no',HOST]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda:f.read(1024**2),b''):h.update(piece)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('engineering','science'),required=True)
    parser.add_argument('--run-engineering',action='store_true');args=parser.parse_args()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=ROOT/'receipts/window008_20261013/runtime_packages';folder.mkdir(parents=True,exist_ok=True)
    package=folder/f'{args.kind}_{stamp}.tar.gz'
    if package.exists():raise FileExistsError(package)
    excluded={'evaluate_window008.py','train_response007_continuation01.py'}
    paths=[p for p in (ROOT/'src/n01').glob('*.py') if p.name not in excluded]
    paths += [ROOT/name for name in ('window008_config.json','provenance/integration006_reference/noise005.pt',
        'scripts/check_window008_engine.py','scripts/supervise.py','scripts/launch_window008.sh','data/PARENT_IDENTITIES.npz')]
    if args.kind=='science':
        manifest=json.loads((ROOT/'WINDOW008_SOURCE_MANIFEST.json').read_text(encoding='utf8'))
        paths=[ROOT/name for name in manifest['files']]+[ROOT/'WINDOW008_SOURCE_MANIFEST.json',ROOT/'scripts/start_window008_remote.py']
    paths=sorted(set(paths))
    with tarfile.open(package,'w:gz') as archive:
        for path in paths:archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(package,'r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():raise ValueError('Runtime package must contain regular files only')
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=sha(ROOT/member.name):raise ValueError(member.name)
    subprocess.run(SSH+['mkdir -p '+shlex.quote(REMOTE+'/incoming')],check=True)
    destination=REMOTE+'/incoming/'+package.name
    subprocess.run(['scp','-q',str(package),HOST+':'+destination],check=True)
    code='''from pathlib import Path
import hashlib,tarfile,shutil,json
root=Path(ROOT_VALUE);package=Path(PACKAGE_VALUE)
assert not (root/'runs/science').exists(),'Scientific run already exists; no code replacement'
assert hashlib.sha256(package.read_bytes()).hexdigest()==SHA_VALUE
with tarfile.open(package,'r:gz') as archive:
 members=archive.getmembers()
 for member in members:
  target=(root/member.name).resolve()
  assert member.isfile() and target.is_relative_to(root)
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
print(json.dumps(dict(status='runtime_staged',files=len(members),package_sha256=SHA_VALUE)))
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('PACKAGE_VALUE',repr(destination)).replace('SHA_VALUE',repr(sha(package)))
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    subprocess.run(SSH+[command],check=True)
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),kind=args.kind,package=str(package),
        package_bytes=package.stat().st_size,package_sha256=sha(package),members=len(paths),remote_root=REMOTE)
    (folder/(package.stem+'.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt),flush=True)
    if args.run_engineering:
        if args.kind!='engineering':raise ValueError('Engineering flag requires engineering package')
        code='''from pathlib import Path
import os,subprocess,json
r=Path(ROOT_VALUE);py=PYTHON_VALUE
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip(),'GPU busy'
assert not (r/'runs/engineering_cuda_01').exists(),'Preserve existing engineering result'
for name in ('tmp','cache','runs'): (r/name).mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'src'),TMPDIR=str(r/'tmp'),XDG_CACHE_HOME=str(r/'cache'),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12')
args=[py,'-B','scripts/supervise.py','--output',str(r/'runs/engineering_supervisor_01'),'--rss-limit-gib','8','--','timeout','--signal=TERM','--kill-after=15s','300s',py,'-B','scripts/check_window008_engine.py','--device','cuda:0','--out',str(r/'runs/engineering_cuda_01')]
result=subprocess.run(args,cwd=r,env=env)
print(json.dumps(dict(engineering_native_exit_code=result.returncode)))
raise SystemExit(result.returncode)
'''.replace('ROOT_VALUE',repr(REMOTE)).replace('PYTHON_VALUE',repr(PYTHON))
        command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
        subprocess.run(SSH+[command],check=True)


if __name__=='__main__':main()
