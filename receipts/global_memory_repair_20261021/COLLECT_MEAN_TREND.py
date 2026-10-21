from pathlib import Path
from datetime import datetime, timezone
import json
import shlex
import subprocess
import sys

out = Path(__file__).parent / 'mean_trend_01'
remote = '/home/rylan/projects/ej276d-n01/mean_trend_repair_20261021'
code = """from pathlib import Path
import json
from datetime import datetime,timezone
root=Path(REMOTE)
record={'UTC':datetime.now(timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
for name in ['control_01/launch.json','control_01/resource_status.json','control_01/exit_receipt.json','control_01/console.log',
             'run_01/BINDINGS.json','run_01/PROGRESS.json','run_01/RESULT.json',
             'run_01/FAILED.json','run_01/COMPLETE.json','run_01/TRAINING_RESULT.json','run_01/TRAINING_STEPS.jsonl']:
    try: raw=(root/name).read_text()
    except FileNotFoundError: record[name]=None; continue
    record[name]=json.loads(raw) if name.endswith('.json') else raw
print(json.dumps(record))
""".replace('REMOTE', repr(remote))
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(code)],
    capture_output=True, check=True, timeout=30)
record = json.loads(r.stdout)
(out / (sys.argv[1] + '.json')).write_bytes((json.dumps(record, indent=2) + '\n').encode())
for name, value in record.items():
    if '/' in name and value is not None:
        raw = json.dumps(value, indent=2) + '\n' if name.endswith('.json') else value
        (out / Path(name).name).write_bytes(raw.encode())
print(json.dumps(dict(UTC=record['UTC'], progress=record['run_01/PROGRESS.json'],
    result=record['run_01/RESULT.json'], failure=record['run_01/FAILED.json'],
    native_exit=record['control_01/exit_receipt.json'])))
