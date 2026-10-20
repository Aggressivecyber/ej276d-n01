from pathlib import Path
import argparse
import ast
import json
import subprocess

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
assert not args.out.exists()
bootstrap = (HERE/'PRIVATE_NAS_SESSION_BOOTSTRAP.py').read_text(encoding='utf-8')
remote = (HERE/'REMOTE_STATUS_REQUEST.py').read_text(encoding='utf-8')
request = (bootstrap+'\n'+remote).encode('utf-8')
ast.parse(request.decode('utf-8'))
with args.out.with_suffix('.jsonl').open('xb') as stdout, args.out.with_suffix('.stderr.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', 'sudo -n unshare --mount --propagation private /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=180)
args.out.with_suffix('.exit.json').write_text(json.dumps(dict(returncode=result.returncode), indent=2)+'\n', encoding='utf-8', newline='\n')
result.check_returncode()
rows = [json.loads(line) for line in args.out.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
assert rows[0]['stage'] == 'N01_PRIVATE_NAS_SESSION_READY'
args.out.with_suffix('.session.json').write_text(json.dumps(rows[0], indent=2)+'\n', encoding='utf-8', newline='\n')
args.out.write_text(json.dumps(rows[-1], indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status='PRIVATE_NAS_STATUS_COLLECTED', receipt=str(args.out),
    source_files_verified=rows[-1]['source_files_verified'], snapshot_UTC=rows[-1]['UTC'],
    complete='run_01/COMPLETE.json' in rows[-1]['files'], failed='run_01/FAILED.json' in rows[-1]['files'])))
