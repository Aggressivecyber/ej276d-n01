"""Stage a fresh owned root without replacing an existing scientific execution."""
from pathlib import Path
import argparse
import datetime as dt
import json
import shlex
import subprocess
import tarfile
from window008_resume01_runtime import ROOT,read,sha,runtime_files,write_new

HOST='rylan@100.82.231.102'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('engineering','science'),required=True);args=parser.parse_args()
    cfg=read(ROOT/'window008_resume01_config.json');remote=cfg['remote_root']
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=ROOT/'receipts/window008_resume01_20261013/runtime_packages';folder.mkdir(parents=True,exist_ok=True)
    package=folder/(args.kind+'_'+stamp+'.tar.gz')
    names=runtime_files()
    if args.kind=='science':
        manifest=read(ROOT/cfg['source_manifest']);names=sorted(set(manifest['files'])|{cfg['source_manifest']})
        for name,digest in manifest['files'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    if package.exists():raise FileExistsError(package)
    with tarfile.open(package,'w:gz') as bundle:
        for name in names:bundle.add(ROOT/name,arcname=name,recursive=False)
    with tarfile.open(package,'r:gz') as bundle:
        for member in bundle.getmembers():
            import hashlib
            if not member.isfile() or hashlib.sha256(bundle.extractfile(member).read()).hexdigest()!=sha(ROOT/member.name):raise ValueError('Package source differs')
    ssh=['ssh','-o','BatchMode=yes','-o','ForwardX11=no',HOST]
    subprocess.run(ssh+['mkdir -p '+shlex.quote(remote+'/incoming')],check=True)
    destination=remote+'/incoming/'+package.name
    subprocess.run(['scp','-q',str(package),HOST+':'+destination],check=True)
    code='''from pathlib import Path
import hashlib,shutil,tarfile,json
r=Path(ROOT_VALUE);p=Path(PACKAGE_VALUE)
if (r/'runs/science').exists() or (r/'runs/START_ATTEMPT.json').exists():raise ValueError('Preserve existing scientific dispatch')
if hashlib.sha256(p.read_bytes()).hexdigest()!=SHA_VALUE:raise ValueError('Transferred package differs')
saved=r/'incoming'/BACKUP_VALUE
with tarfile.open(p,'r:gz') as bundle:
 members=bundle.getmembers()
 for member in members:
  target=(r/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(r):raise ValueError('Unsafe stage path')
  data=bundle.extractfile(member).read()
  if target.exists() and hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(data).digest():
   prior=saved/member.name;prior.parent.mkdir(parents=True,exist_ok=True)
   if prior.exists():raise FileExistsError(prior)
   shutil.copy2(target,prior)
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
print(json.dumps(dict(status='versioned_runtime_staged',files=len(members),package_sha256=SHA_VALUE)))
'''.replace('ROOT_VALUE',repr(remote)).replace('PACKAGE_VALUE',repr(destination)).replace('SHA_VALUE',repr(sha(package))).replace('BACKUP_VALUE',repr('stage_before_'+stamp))
    subprocess.run(ssh+[' '.join(['env','PYTHONDONTWRITEBYTECODE=1',shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])],check=True)
    receipt=dict(UTC=dt.datetime.now(dt.timezone.utc).isoformat(),kind=args.kind,package=str(package),
        package_sha256=sha(package),package_bytes=package.stat().st_size,members=len(names),remote_root=remote)
    write_new(folder/(package.stem+'.json'),receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
