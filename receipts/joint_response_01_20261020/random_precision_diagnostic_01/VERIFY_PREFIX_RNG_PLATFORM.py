from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, platform, shutil, subprocess, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE/'startup_prefix_01'
BEFORE=HERE/'before_rng_audit_fix_01'
BEFORE.mkdir()
for name in ('CHECK_STARTUP_PREFIX.py','AUDIT_COMPLETE.py'):
    shutil.copy2(HERE/name,BEFORE/name)
manifest=json.loads((OUT/'REMOTE_MANIFEST.json').read_bytes())
name='run_01/seed0_rep0_BANK.npz'
raw=(OUT/'collected'/name).read_bytes()
assert hashlib.sha256(raw).hexdigest()==manifest['outputs'][name]['sha256']
with np.load(OUT/'collected'/name,allow_pickle=False) as z:
    saved=z['IID_z'].copy()
    seed=int(z['bank_seed'])
g=torch.Generator().manual_seed(seed)
local=torch.randn(32768,8,generator=g,dtype=torch.float32).numpy()
state_hash=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()
uniform=torch.rand(32768,8,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy()
bits_delta=np.abs(local.view(np.uint32).astype(np.int64)-saved.view(np.uint32).astype(np.int64))
values,counts=np.unique(bits_delta,return_counts=True)
local_record=dict(UTC=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability(),seed=seed,shape=list(local.shape),dtype=str(local.dtype),total_values=local.size,differing_values=int(np.count_nonzero(local!=saved)),all_signbits_equal=bool(np.array_equal(np.signbit(local),np.signbit(saved))),maximum_absolute_difference=float(np.abs(local-saved).max()),maximum_relative_difference=float(np.max(np.abs(local-saved)/np.abs(saved))),ULP_histogram={str(int(k)):int(v) for k,v in zip(values,counts)},maximum_ULP_difference=int(bits_delta.max()),local_normal_array_sha256=hashlib.sha256(local.tobytes()).hexdigest(),saved_normal_array_sha256=hashlib.sha256(saved.tobytes()).hexdigest(),post_normal_generator_state_sha256=state_hash,uniform_array_sha256=hashlib.sha256(uniform.tobytes()).hexdigest(),bank_file_sha256=hashlib.sha256(raw).hexdigest(),first_failure=dict(script='CHECK_STARTUP_PREFIX.py',assertion='np.testing.assert_array_equal(local_torch_randn, saved_IID_z)',exit_code=1,differing_values=2591,total_values=262144,maximum_absolute_difference=2.3841858e-7,maximum_relative_difference=2.1274e-7))
(HERE/'AUDITOR_RNG_LOCAL_DIFFERENCE.json').write_text(json.dumps(local_record,indent=2)+'\n',encoding='utf-8')
request='''from datetime import datetime,timezone
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
'''
(OUT/'REMOTE_VERIFY_RNG_REQUEST.py').write_text(request,encoding='utf-8')
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rylan@100.82.231.102','env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 3 timeout 45s /home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -']
v=subprocess.run(command,input=request.encode(),capture_output=True,timeout=60)
(OUT/'REMOTE_VERIFY_RNG_STDERR.txt').write_bytes(v.stderr)
v.check_returncode()
remote=json.loads(v.stdout)
(HERE/'AUDITOR_RNG_NATIVE_REPLAY.json').write_text(json.dumps(remote,indent=2)+'\n',encoding='utf-8')
assert remote['bank_file_sha256']==local_record['bank_file_sha256']
assert remote['normal_array_sha256']==local_record['saved_normal_array_sha256']
assert remote['post_normal_generator_state_sha256']==local_record['post_normal_generator_state_sha256']
assert remote['uniform_array_sha256']==local_record['uniform_array_sha256']
assert local_record['all_signbits_equal']
review=dict(status='RNG_STREAM_VERIFIED_EXACT_IN_NATIVE_RUNTIME',UTC=datetime.now(timezone.utc).isoformat(),local=local_record,remote=remote,conclusion='Same saved bank exactly reproduced in original Linux Torch runtime; identical CPU uniform values and post-normal generator state across platforms. Windows and Linux normal values differ by the recorded float32 ULP histogram. Saved native values remain authoritative.',science_inputs_changed=False,science_gates_changed=False,new_model_calls=0,local_normal_bit_exact=False)
(HERE/'AUDITOR_RNG_PLATFORM_DIFFERENCE.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],local=local_record,remote=remote)))
