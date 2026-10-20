from datetime import datetime,timezone
from pathlib import Path
import json,subprocess

HERE=Path(__file__).resolve().parent
config_sha256='980a5bc41e8da93cd90649839517b97339cd47cc7c016af6dbf299cd445367f7'
assert not (HERE/'DISPATCH_RECEIPT.json').exists()
script=(HERE/'REMOTE_DISPATCH.py').read_bytes()
r=subprocess.run(['ssh','-x','-o','BatchMode=yes','-o','ConnectTimeout=10','rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
    input=script,capture_output=True,timeout=90)
(HERE/'DISPATCH_STDOUT.json').write_bytes(r.stdout);(HERE/'DISPATCH_STDERR.txt').write_bytes(r.stderr)
(HERE/'DISPATCH_EXIT.json').write_text(json.dumps(dict(exit_code=r.returncode,UTC=datetime.now(timezone.utc).isoformat()),indent=2)+'\n',encoding='utf-8')
r.check_returncode();record=json.loads(r.stdout)
assert record['status']=='DISPATCHED' and record['config_sha256']==config_sha256
(HERE/'DISPATCH_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(record))
