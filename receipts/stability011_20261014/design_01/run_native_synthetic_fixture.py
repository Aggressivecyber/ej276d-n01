"""One SSH stdin synthetic CPU benchmark, no remote persistent output."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

own = Path(__file__).resolve().parent
source = (own/'synthetic_feasibility.py').read_bytes()
command = ['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=yes', 'rylan@100.82.231.102',
    'env PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B - benchmark --workers 12']
result = subprocess.run(command, input=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=55)
receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), native_ssh_exit_code=result.returncode,
    command=command, stdin_source_sha256=hashlib.sha256(source).hexdigest(),
    artificial_only=True, remote_project_array_reads=0, checkpoint_decodes=0,
    scientific_bootstrap_replicates=0, new_model_forwards=0, optimizer_updates=0,
    remote_persistent_output=False, stderr=result.stderr.decode('utf8', errors='replace'))
try:
    receipt['benchmark'] = json.loads(result.stdout.decode('utf8'))
except ValueError:
    receipt['stdout_unparsed'] = result.stdout.decode('utf8', errors='replace')
with (own/'NATIVE_SYNTHETIC_BENCHMARK.json').open('x', encoding='utf8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(receipt, allow_nan=False))
sys.exit(result.returncode)
