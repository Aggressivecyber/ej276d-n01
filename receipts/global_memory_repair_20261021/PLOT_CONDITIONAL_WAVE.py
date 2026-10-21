from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shlex
import subprocess

root = Path('E:/GAN/research/N01')
here = Path(__file__).parent
out = here / 'conditional_wave_01'
result = json.loads((out / 'TRAINING_RESULT.json').read_bytes())
assert json.loads((out / 'exit_receipt.json').read_bytes())['exit_code'] == 0
arrays = out / 'ENDPOINT_SCALARS_ANALYSIS.npz'
remote_code = 'import sys; from pathlib import Path; sys.stdout.buffer.write(Path(' + repr(result['endpoint_arrays']) + ').read_bytes())'
with arrays.open('xb') as stream:
    subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
        '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(remote_code)],
        stdout=stream, stderr=subprocess.PIPE, check=True, timeout=60)
python = 'C:/Users/10629/AppData/Local/Programs/Python/Python313/python.exe'
figure = root / 'deliverables/global_memory_repair_20261021/conditional_wave_01'
env = os.environ.copy()
env.update(MPLCONFIGDIR=str(root / 'cache/matplotlib'), PYTHONDONTWRITEBYTECODE='1',
           TMP=str(root / 'cache'), TEMP=str(root / 'cache'))
subprocess.run([python, '-B', 'scripts/plot_global_memory01.py', '--arrays', str(arrays),
    '--result', str(out / 'TRAINING_RESULT.json'), '--out', str(figure)], cwd=root, env=env, check=True)
subprocess.run([python, '-B', 'scripts/plot_conditional_wave01.py', '--receipt', str(out), '--out', str(figure)],
    cwd=root, env=env, check=True)
subprocess.run([python, '-B', 'scripts/plot_global_memory_comparison01.py',
    '--first', str(root / 'deliverables/global_memory_repair_20261021/CONDITIONAL_PSD_REPORT.json'),
    '--second', str(root / 'deliverables/global_memory_repair_20261021/projected_01/CONDITIONAL_PSD_REPORT.json'),
    '--latest', str(figure / 'CONDITIONAL_PSD_REPORT.json'), '--out', str(figure)], cwd=root, env=env, check=True)
arrays.unlink()
record = dict(UTC=datetime.now(timezone.utc).isoformat(), local_temporary_scalars_removed=True,
    server_primary_scalars=result['endpoint_arrays'], NAS_file_IO_checks=False,
    NAS_file_hash_verification=False, visual_QA_pending=True)
(out / 'ANALYSIS_STORAGE_RECEIPT.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
