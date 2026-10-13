"""Run one bounded artificial CUDA fixture in the registered target environment."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shlex
import subprocess
from window008_resume01_runtime import ROOT, read, sha, write_new

HOST = 'rylan@100.82.231.102'
PYTHON = '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python'


def main():
    cfg = read(ROOT / 'window008_resume01_config.json')
    remote = cfg['remote_root']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    relative = 'receipts/window008_resume01_20261013/diagnostics_cuda_' + stamp
    local = ROOT / relative
    local.mkdir(parents=True, exist_ok=False)
    code = '''from pathlib import Path
import json,os,subprocess,sys
r=Path(ROOT_VALUE)
os.chdir(r)
sys.path[:0]=[str(r/'scripts'),str(r/'src')]
from window008_resume01_evaluation_common import evaluation_binding,sha,write_new,require
from window008_resume01_diagnostics_common import evaluation_binding as addon_binding
evaluation_binding(r);addon_binding(r)
require(sha(r/'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json')==MAIN_SHA,'Main freeze differs')
require(sha(r/'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json')==ADDON_SHA,'Addon freeze differs')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
require(not apps,'Another GPU compute job is active')
out=r/REL_VALUE
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r/'src'),TMPDIR=str(r/'tmp'),
 XDG_CACHE_HOME=str(r/'cache'),OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12',
 CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
command=[sys.executable,'-B','scripts/supervise.py','--output',str(out/'supervisor'),'--rss-limit-gib','8',
 '--','timeout','--signal=TERM','--kill-after=20s','600s',sys.executable,'-B',
 'scripts/check_window008_resume01_diagnostics.py','--out',str(out/'check'),'--device','cuda:0','--full-shape','--architecture-128']
write_new(out/'ATTEMPT.json',dict(command=command,source_manifest_sha256=MAIN_SHA,addon_source_manifest_sha256=ADDON_SHA,
 scientific_signal_forward_rows=0,actual_SOURCE_REAL_DEV_numeric_decodes=0,automatic_retry_allowed=False))
proc=subprocess.run(command,cwd=r,env=env)
result=dict(native_exit_code=proc.returncode,relative=REL_VALUE,files={})
for name in ['ATTEMPT.json','check/RECEIPT.json','check/FAILED.json','supervisor/launch.json','supervisor/exit_receipt.json']:
 p=out/name
 if p.is_file():result['files'][name]=dict(sha256=sha(p),text=p.read_text(),value=json.loads(p.read_text()))
log=out/'supervisor/console.log'
result['console_tail']=log.read_text()[-12000:] if log.exists() else None
print(json.dumps(result))
sys.exit(proc.returncode)
'''.replace('ROOT_VALUE', repr(remote)).replace('REL_VALUE', repr(relative)).replace('MAIN_SHA', repr(sha(ROOT / 'WINDOW008_RESUME01_EVALUATION_SOURCE_MANIFEST.json'))).replace('ADDON_SHA', repr(sha(ROOT / 'WINDOW008_RESUME01_DIAGNOSTICS_SOURCE_MANIFEST.json')))
    (local / 'REMOTE_COMMAND.py').write_text(code, encoding='utf8')
    command = ' '.join([shlex.quote(PYTHON), '-B', '-c', shlex.quote(code)])
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ForwardX11=no', '-o', 'ConnectTimeout=8',
        HOST, command], text=True, encoding='utf8', capture_output=True)
    write_new(local / 'SSH_RECEIPT.json', dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr, automatic_retry_allowed=False))
    if proc.returncode:
        raise RuntimeError('Artificial target check failed or unknown; preserve and inspect: ' + str(local))
    result = json.loads(proc.stdout)
    for name, item in result['files'].items():
        target = local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf8', newline='') as stream:
            stream.write(item['text'])
        if sha(target) != item['sha256']:
            raise ValueError('Transferred engineering receipt hash differs: ' + name)
    write_new(local / 'REMOTE_RESULT.json', result)
    receipt = result['files']['check/RECEIPT.json']['value']
    native = result['files']['supervisor/exit_receipt.json']['value']
    if not (receipt['status'] == 'passed' and native['exit_code'] == 0 and native['reason'] is None):
        raise ValueError('Target artificial check did not pass natively')
    print(json.dumps(dict(status='artificial_target_check_passed_native0', receipt=str(local),
        wall_seconds=receipt['wall_seconds'], peak_RSS_bytes=native['peak_rss_bytes'], GPU=receipt['GPU'],
        output_GiB=receipt['output_GiB'], architecture128=receipt['architecture128'],
        scientific_signal_forward_rows=0, actual_SOURCE_REAL_DEV_numeric_decodes=0)))


if __name__ == '__main__':
    main()
