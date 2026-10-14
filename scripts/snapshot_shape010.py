"""Take one remote metadata snapshot; never start or poll a scientific job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOST = 'rylan@100.82.231.102'
REMOTE = '/home/rylan/projects/ej276d-n01/shape010_20261014'
SCRIPT = '''from pathlib import Path
import json,subprocess
r=Path('/home/rylan/projects/ej276d-n01/shape010_20261014')
names=['START_RECEIPT.json','runs/primary/supervisor/launch.json','runs/primary/supervisor/resource_status.json','runs/primary/supervisor/exit_receipt.json','runs/primary/artifacts/FIRST_UPDATE_PROOF.json','runs/primary/artifacts/status.json','runs/primary/artifacts/COMPLETE.json','runs/primary/artifacts/FAILED.json']
data={n:json.loads((r/n).read_text()) if (r/n).is_file() else None for n in names}
data['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
p=r/'runs/primary/console.log'
if p.is_file():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-8192));data['console_tail']=f.read().decode('utf8','replace')
print(json.dumps(data))
'''


def main():
    result = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, 'python3', '-'],
                            input=SCRIPT, text=True, encoding='utf8', capture_output=True, timeout=40)
    if result.returncode != 0:
        raise RuntimeError('Remote snapshot unavailable: ' + result.stderr)
    state = json.loads(result.stdout)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = ROOT / 'receipts/shape010_20261014' / ('snapshot_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'REMOTE_STATE.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    launch = state['START_RECEIPT.json']
    native = state['runs/primary/supervisor/exit_receipt.json']
    proof = state['runs/primary/artifacts/FIRST_UPDATE_PROOF.json']
    print(json.dumps(dict(snapshot=str(out), supervisor_pid=None if launch is None else launch['supervisor_pid'],
        actual_first_batch_proof=proof, native_exit=native, gpu_processes=state['gpu_processes'],
        status=state['runs/primary/artifacts/status.json'], console_tail=state.get('console_tail') if proof is None else None)))


if __name__ == '__main__':
    main()
