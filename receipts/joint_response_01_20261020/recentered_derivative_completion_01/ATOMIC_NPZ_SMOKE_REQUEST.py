from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
import numpy as np
root=Path('/mnt/MiNas/GAN/research/N01/receipts/joint_response_01_20261020/recentered_derivative_completion_01');tmp=root/'ATOMIC_NPZ_SMOKE.tmp.npz';target=root/'ATOMIC_NPZ_SMOKE.npz';expected=np.arange(16,dtype=np.float64);started=time.monotonic()
np.savez(tmp,probe=expected)
with tmp.open('rb') as stream:os.fsync(stream.fileno())
tmp.replace(target)
with np.load(target,allow_pickle=False) as saved:np.testing.assert_array_equal(saved['probe'],expected)
print(json.dumps(dict(status='PASS_NATIVE_NAS_NPZ_FLUSH_REPLACE_READBACK',UTC=datetime.now(timezone.utc).isoformat(),path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,new_scientific_calls=0)))
