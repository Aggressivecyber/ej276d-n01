from pathlib import Path
import ast
import json
import subprocess

HERE = Path(__file__).resolve().parent
T = HERE/'terminal_01'
T.mkdir(exist_ok=True)
target = T/'REMOTE_TERMINAL_VERIFICATION.json'
assert not target.exists()
request = (HERE/'REMOTE_STATUS_REQUEST.py').read_bytes()
ast.parse(request.decode('utf-8'))
with (T/'EXISTING_PATH_STATUS_STDOUT.json').open('xb') as stdout, (T/'EXISTING_PATH_STATUS_STDERR.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=240)
(T/'EXISTING_PATH_STATUS_EXIT.json').write_text(json.dumps(dict(returncode=result.returncode), indent=2)+'\n', encoding='utf-8', newline='\n')
result.check_returncode()
d = json.loads((T/'EXISTING_PATH_STATUS_STDOUT.json').read_bytes())
target.write_text(json.dumps(d, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status='EXISTING_PATH_TERMINAL_STATUS_COLLECTED', snapshot_UTC=d['UTC'],
    source_files_verified=d['source_files_verified'], process_count=len(d['processes']),
    exit=d['native_exit'], complete=d['complete_present'], failed=d['failed_present'])))
