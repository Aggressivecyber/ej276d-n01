from pathlib import Path
import base64
import json
import subprocess
import sys

root = Path('E:/GAN/research/N01')
out = Path(__file__).parent / 'mean_trend_local_01'
payload = {name: base64.b64encode((root / name).read_bytes()).decode() for name in
           ['scripts/run_mean_trend01.py', 'configs/mean_trend_local_20261021.json']}
code = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shutil
root=Path('/home/rylan/projects/ej276d-n01/mean_trend_local_20261021')
prior=Path('/home/rylan/projects/ej276d-n01/mean_trend_repair_fix02_20261021')
if (root/'run_01').exists():raise RuntimeError('Run already exists; collect its state instead of redeploying')
root.mkdir(parents=True,exist_ok=True)
(root/'cache').mkdir(exist_ok=True)
for name in ['src','scripts']:
    shutil.copytree(prior/name,root/name,dirs_exist_ok=True)
(root/'configs').mkdir(exist_ok=True)
payload=json.loads(PAYLOAD)
for name,value in payload.items():(root/name).write_bytes(base64.b64decode(value))
record=dict(status='SERVER_LOCAL_CODE_REUSED_DELTA_DEPLOYED',UTC=datetime.now(timezone.utc).isoformat(),
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in payload},
    no_scientific_dispatch_performed=True,NAS_file_IO_checks=False,NAS_file_hash_verification=False)
(root/'DEPLOYMENT_COMPLETE.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''.replace('PAYLOAD', repr(json.dumps(payload)))
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', 'rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
    input=code.encode(), capture_output=True, timeout=45)
(out / (sys.argv[1] + '_DEPLOY_STDOUT.json')).write_bytes(r.stdout)
(out / (sys.argv[1] + '_DEPLOY_STDERR.txt')).write_bytes(r.stderr)
r.check_returncode()
print(r.stdout.decode())
