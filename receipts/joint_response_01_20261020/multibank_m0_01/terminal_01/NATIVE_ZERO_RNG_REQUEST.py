from pathlib import Path
import json,hashlib,torch,numpy as np
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_multibank_m0_20261020')
torch.set_num_threads(4)
p=json.loads((root/'configs/joint_response_01_multibank_m0_20261020.json').read_text())
z=np.load(root/'run_01/M0_ZERO_IDENTITY.npz',allow_pickle=False)
for seed in range(2):
 expected=torch.randn(128,8,generator=torch.Generator().manual_seed(p['zero_identity_seed']+seed*100000),dtype=torch.float32).numpy()
 assert expected.tobytes()==z[f'seed{seed}_z'].tobytes()
print(json.dumps(dict(status='PASS_NATIVE_RANDOM_STREAM_EXACT',torch_version=torch.__version__,streams=2,values=2048,zero_archive_sha256=hashlib.sha256((root/'run_01/M0_ZERO_IDENTITY.npz').read_bytes()).hexdigest(),scientific_solves=0,model_updates=0)))
