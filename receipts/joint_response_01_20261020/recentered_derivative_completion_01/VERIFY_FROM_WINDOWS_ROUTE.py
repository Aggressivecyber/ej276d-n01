from datetime import datetime, timezone
from pathlib import Path
import ast
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
assert json.loads((HERE/'WINDOWS_TEXT_DEPLOYMENT.json').read_bytes())['status'] == 'PASS_WINDOWS_NAS_SMALL_TEXT_DEPLOYMENT_ONLY'
remote = (HERE/'VERIFY_WINDOWS_DEPLOYED_TEXT_REMOTE.py').read_text(encoding='utf-8').replace('signal.alarm(60)', 'signal.alarm(150)')
ast.parse(remote)
(HERE/'VERIFY_WINDOWS_DEPLOYED_TEXT_REMOTE.py').write_text(remote, encoding='utf-8', newline='\n')
request = ('payload='+repr(dict(manifest_raw=(HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8')))+'\n'+remote).encode('utf-8')
started = time.monotonic()
with (HERE/'NATIVE_VERIFY_AFTER_WINDOWS_STDOUT.jsonl').open('xb') as stdout, (HERE/'NATIVE_VERIFY_AFTER_WINDOWS_STDERR.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=180)
exit_record = dict(UTC=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
    elapsed_seconds=time.monotonic()-started, maximum_remote_seconds=150, new_scientific_calls=0)
(HERE/'NATIVE_VERIFY_AFTER_WINDOWS_EXIT.json').write_text(json.dumps(exit_record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(exit_record), flush=True)
result.check_returncode()
rows = [json.loads(line) for line in (HERE/'NATIVE_VERIFY_AFTER_WINDOWS_STDOUT.jsonl').read_text(encoding='utf-8').splitlines()]
record = rows[-1]
assert record['status'] == 'PASS_NAS_DEPLOYMENT'
(HERE/'DEPLOYMENT_VERIFICATION.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(record))
