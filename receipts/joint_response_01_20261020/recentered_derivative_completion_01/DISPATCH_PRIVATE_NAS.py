from pathlib import Path
import ast
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
review_raw = (HERE/'PRE_DISPATCH_REVIEW.json').read_text(encoding='utf-8')
assert json.loads(review_raw)['status'] == 'PASS_STATIC_PRE_DISPATCH_REVIEW'
assert json.loads((HERE/'DEPLOYMENT_VERIFICATION.json').read_bytes())['private_NAS_session']
bootstrap = (HERE/'PRIVATE_NAS_SESSION_BOOTSTRAP.py').read_text(encoding='utf-8')
policy = json.loads((HERE/'PRIVATE_NAS_SESSION_POLICY.json').read_bytes())
assert policy['bootstrap_sha256'] == hashlib.sha256(bootstrap.encode('utf-8')).hexdigest()
remote = (HERE/'REMOTE_DISPATCH_REQUEST.py').read_text(encoding='utf-8')
prefix = "Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020/PRE_DISPATCH_REVIEW.json').write_text(review_raw,encoding='utf-8')\n"
request = ('review_raw='+repr(review_raw)+'\n'+bootstrap+'\n'+prefix+remote).encode('utf-8')
ast.parse(request.decode('utf-8'))
with (HERE/'PRIVATE_DISPATCH_STDOUT.jsonl').open('xb') as stdout, (HERE/'PRIVATE_DISPATCH_STDERR.txt').open('xb') as stderr:
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        'rylan@100.82.231.102', 'sudo -n unshare --mount --propagation private /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'],
        input=request, stdout=stdout, stderr=stderr, timeout=180)
(HERE/'PRIVATE_DISPATCH_EXIT.json').write_text(json.dumps(dict(returncode=result.returncode), indent=2)+'\n', encoding='utf-8', newline='\n')
result.check_returncode()
rows = [json.loads(line) for line in (HERE/'PRIVATE_DISPATCH_STDOUT.jsonl').read_text(encoding='utf-8').splitlines()]
assert rows[-1]['status'] == 'DISPATCHED'
(HERE/'DISPATCH_SESSION.json').write_text(json.dumps(rows[0], indent=2)+'\n', encoding='utf-8', newline='\n')
(HERE/'DISPATCH_RECEIPT.json').write_text(json.dumps(rows[-1], indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(rows[-1]))
