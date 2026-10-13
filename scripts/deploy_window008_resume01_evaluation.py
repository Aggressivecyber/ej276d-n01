"""Transfer a verified additive source bundle and invoke its guarded installer."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shlex
import subprocess
from window008_resume01_runtime import ROOT,read,sha,write_new

HOST='rylan@100.82.231.102'
PYTHON='/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def deploy(bundle_path):
    metadata=read(bundle_path);archive=Path(metadata['archive'])
    if sha(archive)!=metadata['archive_sha256']:raise ValueError('Local bundle changed')
    cfg=read(ROOT/'window008_resume01_config.json');remote=cfg['remote_root']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    remote_archive=remote+'/incoming/evaluation_sources_'+stamp+'.tar.gz'
    subprocess.run(['scp','-q','-o','ConnectTimeout=8',str(archive),HOST+':'+remote_archive],check=True)
    code='''from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
r=Path(ROOT_VALUE);archive=Path(ARCHIVE_VALUE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
if sha(archive)!=ARCHIVE_SHA:raise ValueError('Transferred bundle differs')
boot=r/'incoming'/BOOT_VALUE
(boot/'scripts').mkdir(parents=True,exist_ok=False)
names=['scripts/window008_resume01_evaluation_common.py','scripts/window008_resume01_diagnostics_common.py',
 'scripts/stage_window008_resume01_evaluation.py']
with tarfile.open(archive,'r:gz') as t:
 m=t.extractfile('WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json').read()
 if hashlib.sha256(m).hexdigest()!=MANIFEST_SHA:raise ValueError('Wrong main evaluation freeze')
 manifest=json.loads(m)
 for name in names:
  member=t.getmember(name)
  if not member.isfile() or member.issym() or member.islnk():raise ValueError('Unsafe bootstrap')
  content=t.extractfile(member).read()
  if hashlib.sha256(content).hexdigest()!=manifest['files'][name]:raise ValueError('Bootstrap source changed')
  with (boot/name).open('xb') as f:f.write(content)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(r/'tmp'),XDG_CACHE_HOME=str(r/'cache'))
result=subprocess.run([sys.executable,'-B',str(boot/'scripts/stage_window008_resume01_evaluation.py'),
 '--install',str(archive),'--root',str(r)],cwd=r,env=env,text=True,capture_output=True)
print(json.dumps(dict(status='installed' if result.returncode==0 else 'installation_failed_preserved',
 native_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,archive_sha256=sha(archive),
 UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())))
if result.returncode:sys.exit(result.returncode)
'''.replace('ROOT_VALUE',repr(remote)).replace('ARCHIVE_VALUE',repr(remote_archive)).replace('ARCHIVE_SHA',repr(metadata['archive_sha256'])).replace('BOOT_VALUE',repr('evaluation_installer_'+stamp)).replace('MANIFEST_SHA',repr(metadata['evaluation_source_manifest_sha256']))
    command=' '.join([shlex.quote(PYTHON),'-B','-c',shlex.quote(code)])
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ForwardX11=no','-o','ConnectTimeout=8',HOST,command],text=True,encoding='utf8',capture_output=True)
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),ssh_exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
        local_bundle=str(Path(bundle_path).resolve()),remote_archive=remote_archive,automatic_retry_allowed=False)
    write_new(ROOT/'receipts/window008_resume01_20261013'/('EVALUATION_DEPLOY_'+stamp+'.json'),result)
    if proc.returncode:raise RuntimeError('Source deployment failed/unknown; preserved receipt: '+proc.stdout+proc.stderr)
    value=json.loads(proc.stdout);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);deploy(p.parse_args().bundle)
