from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import torch

root=Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')
config=root/'configs/joint_response_01_random_precision_diagnostic_20261020.json'
raw=config.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='0e576b39f0209aa3fb1bcc732cf5bffccb5af194848f12a60a1cc6ba75f45907'
p=json.loads(raw)
records={}
for head in range(2):
    for replicate in range(4):
        name=f'run_01/seed{head}_rep{replicate}_BANK.npz'
        raw=(root/name).read_bytes()
        with np.load(root/name,allow_pickle=False) as z:
            saved=z['IID_z'].copy();seed=int(z['bank_seed'])
        assert seed==p['random_seed']+head*p['head_seed_stride']+replicate*p['replicate_seed_stride']
        assert saved.dtype==np.float32 and saved.shape==(32768,8)
        g=torch.Generator().manual_seed(seed)
        replay=torch.randn(32768,8,generator=g,dtype=torch.float32).numpy()
        np.testing.assert_array_equal(replay,saved)
        records[name]=dict(seed=seed,values_verified=replay.size,bytes=len(raw),bank_file_sha256=hashlib.sha256(raw).hexdigest(),normal_array_sha256=hashlib.sha256(replay.tobytes()).hexdigest(),post_normal_generator_state_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest())
print(json.dumps(dict(status='PASS_SAME_RUNTIME_EXACT_RANDOM_REPLAY',UTC=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability(),executable=sys.executable,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),banks=records,values_verified=sum(r['values_verified'] for r in records.values()),model_calls=0,science_process_queried=False)))
