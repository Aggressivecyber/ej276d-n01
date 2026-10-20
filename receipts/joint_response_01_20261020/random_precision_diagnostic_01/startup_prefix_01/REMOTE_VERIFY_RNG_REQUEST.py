from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import torch
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')
path=root/'run_01/seed0_rep0_BANK.npz'
raw=path.read_bytes()
with np.load(path,allow_pickle=False) as z:
 saved=z['IID_z'].copy();seed=int(z['bank_seed'])
assert seed==202680100000
g=torch.Generator().manual_seed(seed)
actual=torch.randn(32768,8,generator=g,dtype=torch.float32).numpy()
np.testing.assert_array_equal(actual,saved)
uniform=torch.rand(32768,8,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy()
print(json.dumps(dict(status='PASS_SAME_RUNTIME_EXACT_RANDOM_REPLAY',UTC=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability(),executable=sys.executable,seed=seed,shape=list(actual.shape),dtype=str(actual.dtype),values_verified=actual.size,bank_file_sha256=hashlib.sha256(raw).hexdigest(),normal_array_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),post_normal_generator_state_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),uniform_array_sha256=hashlib.sha256(uniform.tobytes()).hexdigest(),model_calls=0,science_process_queried=False)))
