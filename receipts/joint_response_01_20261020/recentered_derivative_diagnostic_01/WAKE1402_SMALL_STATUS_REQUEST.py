from pathlib import Path
from datetime import datetime,timezone
import json
root=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
files={}
for name in ['control_01/exit_receipt.json','run_01/COMPLETE.json','run_01/FAILED.json','run_01/RESULT.json','run_01/PROGRESS.json']:
 print(json.dumps(dict(stage='before_read',name=name,UTC=datetime.now(timezone.utc).isoformat())),flush=True)
 if (root/name).exists():files[name]=json.loads((root/name).read_bytes())
 print(json.dumps(dict(stage='read_done',name=name,UTC=datetime.now(timezone.utc).isoformat())),flush=True)
print(json.dumps(dict(stage='terminal_status',UTC=datetime.now(timezone.utc).isoformat(),files=files)),flush=True)
