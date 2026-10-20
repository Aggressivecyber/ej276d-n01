from datetime import datetime, timezone
from pathlib import Path
import ast
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
assert json.loads((HERE/'ISOLATED_CONNECTION_PROBE.json').read_bytes())['read_probe']['status'] == 'PASS_PRIVATE_NAS_SESSION_READ_PROBE'
bootstrap = (HERE/'PRIVATE_NAS_SESSION_BOOTSTRAP.py').read_text(encoding='utf-8')
remote = (HERE/'VERIFY_WINDOWS_DEPLOYED_TEXT_REMOTE.py').read_text(encoding='utf-8')
request = ('payload='+repr(dict(manifest_raw=(HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8')))+'\n'+bootstrap+'\n'+remote).encode('utf-8')
ast.parse(request.decode('utf-8'))
started = time.monotonic()
with (HERE/'PRIVATE_NATIVE_DEPLOYMENT_STDOUT.jsonl').open('xb') as stdout, (HERE/'PRIVATE_NATIVE_DEPLOYMENT_STDERR.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', 'sudo -n unshare --mount --propagation private /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=180)
exit_record = dict(UTC=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
    elapsed_seconds=time.monotonic()-started, native_verification_seconds=150, new_scientific_calls=0)
(HERE/'PRIVATE_NATIVE_DEPLOYMENT_EXIT.json').write_text(json.dumps(exit_record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(exit_record), flush=True)
result.check_returncode()
rows = [json.loads(line) for line in (HERE/'PRIVATE_NATIVE_DEPLOYMENT_STDOUT.jsonl').read_text(encoding='utf-8').splitlines()]
record = rows[-1]
assert record['status'] == 'PASS_NAS_DEPLOYMENT'
(HERE/'PRIVATE_DEPLOYMENT_SESSION.json').write_text(json.dumps(rows[0], indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
