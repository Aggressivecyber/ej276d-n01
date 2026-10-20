from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T = HERE/'terminal_01'
config_path = ROOT/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json'
raw_config = config_path.read_bytes()
assert hashlib.sha256(raw_config).hexdigest() == '1bb8ae27f1450350b393af374f638e10006f057ff158790e57b98e8943df5505'
config = json.loads(raw_config)
manifest = json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
failed = json.loads((HERE/'WAKE1402_FAILED_STATUS.json').read_bytes())['files']['run_01/FAILED.json']
source = (HERE/'PREFIX_REMOTE_AUDIT.py').read_text(encoding='utf-8')
ast.parse(source)
inputs = dict(config=config, manifest=manifest, failed=failed)
design = dict(status='READ_ONLY_PREFIX_AUDIT_REGISTERED', UTC=datetime.now(timezone.utc).isoformat(),
    script_sha256=hashlib.sha256(source.encode()).hexdigest(), input_config_sha256=hashlib.sha256(raw_config).hexdigest(),
    input_manifest_sha256=hashlib.sha256((HERE/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    source_files=185, maximum_seconds=900, IO_workers=4, CPU_threads=4,
    new_generator_calls=0, new_solves=0, parameter_updates=0, large_payloads_copied=False,
    remote_root=config['storage']['remote_root'], source_and_outputs_read_only=True)
(HERE/'PREFIX_AUDIT_DESIGN.json').write_text(json.dumps(design, indent=2)+'\n', encoding='utf-8', newline='\n')
request = ('INPUT='+repr(inputs)+'\n'+source).encode('utf-8')
command = ['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
    'rylan@100.82.231.102', '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -']
with (T/'PREFIX_AUDIT_PROGRESS.jsonl').open('w', encoding='utf-8', newline='\n') as progress, (T/'PREFIX_AUDIT_STDERR.txt').open('wb') as errors:
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
    process.stdin.write(request)
    process.stdin.close()
    for line in process.stdout:
        text = line.decode('utf-8')
        progress.write(text)
        progress.flush()
        record = json.loads(text)
        if record['stage'] == 'PREFIX_AUDIT_COMPLETE':
            (T/'PREFIX_REVIEW.json').write_text(json.dumps(record['result'], indent=2)+'\n', encoding='utf-8', newline='\n')
            print(json.dumps(dict(stage=record['stage'], audit_counts=record['result']['audit_counts'], payload_errors=record['result']['payload_errors'], elapsed_seconds=record['result']['elapsed_seconds'])), flush=True)
        else:
            print(text, end='', flush=True)
    code = process.wait()
(T/'PREFIX_AUDIT_EXIT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=code, local_ssh_pid=process.pid), indent=2)+'\n', encoding='utf-8', newline='\n')
if code != 0:
    raise RuntimeError((T/'PREFIX_AUDIT_STDERR.txt').read_text(encoding='utf-8'))
assert record['stage'] == 'PREFIX_AUDIT_COMPLETE'
