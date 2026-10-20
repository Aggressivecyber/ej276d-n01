"""Run the prepared independent auditor on NAS; return only small review evidence."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
T = HERE/'terminal_01'
terminal_raw = (T/'REMOTE_TERMINAL_VERIFICATION.json').read_text(encoding='utf-8')
terminal = json.loads(terminal_raw)
assert terminal['native_exit']['exit_code'] == 0 and not terminal['processes']
assert terminal['complete_present'] and not terminal['failed_present']
payload = dict(
    audit=(HERE/'AUDIT_COMPLETE.py').read_text(encoding='utf-8'),
    manifest=(HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'),
    terminal=terminal_raw)
remote_script = """
from pathlib import Path
import hashlib,json,os,subprocess
source=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_completion_thin_20261020')
receipt=Path('/mnt/MiNas/GAN/research/N01/receipts/joint_response_01_20261020/recentered_derivative_completion_01')
terminal_dir=receipt/'terminal_01'
terminal_dir.mkdir(parents=True,exist_ok=True)
for name,text in [('AUDIT_COMPLETE.py',payload['audit']),('SOURCE_MANIFEST.json',payload['manifest']),('terminal_01/REMOTE_TERMINAL_VERIFICATION.json',payload['terminal'])]:
    (receipt/name).write_text(text,encoding='utf-8')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TMPDIR=str(source/'cache'),XDG_CACHE_HOME=str(source/'cache'))
command=['/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python','-B',str(receipt/'AUDIT_COMPLETE.py'),'--root',str(source),'--receipt-root',str(receipt),'--run-root',str(source)]
r=subprocess.run(command,capture_output=True,env=env,timeout=240)
(terminal_dir/'AUDIT_STDOUT.txt').write_bytes(r.stdout)
(terminal_dir/'AUDIT_STDERR.txt').write_bytes(r.stderr)
if r.returncode!=0:
    raise RuntimeError(r.stderr.decode('utf-8'))
raw=(terminal_dir/'TERMINAL_REVIEW.json').read_bytes()
print(json.dumps(dict(status='REMOTE_SAVED_ARRAY_AUDIT_RETURNED',review_raw=raw.decode('utf-8'),review_sha256=hashlib.sha256(raw).hexdigest(),stdout=r.stdout.decode('utf-8'),large_payloads_copied=False,remote_receipt=str(receipt))))
"""
bootstrap=(HERE/'PRIVATE_NAS_SESSION_BOOTSTRAP.py').read_text(encoding='utf-8')
request = 'payload='+repr(payload)+'\n'+bootstrap+'\n'+remote_script
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    'rylan@100.82.231.102', 'sudo -n unshare --mount --propagation private /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -'], input=request.encode('utf-8'), capture_output=True, timeout=300)
(T/'REMOTE_AUDIT_STDERR.txt').write_bytes(r.stderr)
r.check_returncode()
rows=[json.loads(line) for line in r.stdout.decode('utf-8').splitlines()]
(T/'REMOTE_AUDIT_SESSION.json').write_text(json.dumps(rows[0],indent=2)+'\n',encoding='utf-8')
returned=rows[-1]
raw = returned.pop('review_raw').encode('utf-8')
assert hashlib.sha256(raw).hexdigest() == returned['review_sha256']
review = json.loads(raw)
assert review['status'] == 'PASS_SAVED_COMPLETED_RECENTERED_DERIVATIVE_AUDIT'
(T/'TERMINAL_REVIEW.json').write_bytes(raw)
(T/'REMOTE_AUDIT_RETURN.json').write_text(json.dumps(returned, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=review['status'], audit_counts=review['audit_counts'],
    review_sha256=returned['review_sha256'], scientific_acceptance=False, large_payloads_copied=False)))
