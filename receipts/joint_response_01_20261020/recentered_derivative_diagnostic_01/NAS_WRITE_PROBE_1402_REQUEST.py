from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
root=Path('/mnt/MiNas/GAN/research/N01/receipts/joint_response_01_20261020/recentered_derivative_diagnostic_01')
root.mkdir(parents=True,exist_ok=True)
tmp=root/'NAS_WRITE_PROBE_1402.tmp';final=root/'NAS_WRITE_PROBE_1402.bin';raw=os.urandom(65536);started=time.monotonic()
with tmp.open('xb') as handle:
 handle.write(raw);handle.flush();os.fsync(handle.fileno())
tmp.replace(final)
readback=final.read_bytes();assert readback==raw
job=Path('/mnt/MiNas/GAN/research/N01/server_runs/joint_response01_recentered_derivative_diagnostic_20261020')
logs={n:(job/n).read_text()[-6000:] for n in ('dispatcher.log','control_01/console.log')}
print(json.dumps(dict(status='PASS_NAS_WRITE_FLUSH_RENAME_READBACK',UTC=datetime.now(timezone.utc).isoformat(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),elapsed_seconds=time.monotonic()-started,path=str(final),logs=logs)))
