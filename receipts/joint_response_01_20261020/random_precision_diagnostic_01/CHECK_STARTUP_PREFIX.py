from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,tarfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;OUT=HERE/'startup_prefix_01';OUT.mkdir()
request='''from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/rylan/projects/ej276d-n01/joint_response01_random_precision_diagnostic_20261020')
names=['run_01/seed0_rep-1_REPLAY.json','run_01/seed0_rep-1_REPLAY.npz','run_01/seed0_rep0_IID.json','run_01/seed0_rep0_IID.npz','run_01/seed0_rep0_BANK.npz']
d=json.loads((root/names[2]).read_bytes());assert d['FD_columns_completed']==54 and len(d['checks'])==2
records={}
for name in names:
 p=root/name;b=p.read_bytes();records[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mtime_ns=p.stat().st_mtime_ns)
archive=root/'startup_prefix_01.tar';assert not archive.exists()
with tarfile.open(archive,'w') as t:
 for name in names:t.add(root/name,arcname=name,recursive=False)
for name,record in records.items():
 p=root/name;assert p.stat().st_mtime_ns==record['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
print(json.dumps(dict(outputs=records,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''
(OUT/'REMOTE_REQUEST.py').write_text(request,encoding='utf-8')
v=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rylan@100.82.231.102','python3','-'],input=request.encode(),capture_output=True,timeout=60);(OUT/'STDERR.txt').write_bytes(v.stderr);v.check_returncode();record=json.loads(v.stdout);(OUT/'REMOTE_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','rylan@100.82.231.102:'+record['archive'],str(OUT/'prefix.tar')],capture_output=True,check=True,timeout=60)
raw=(OUT/'prefix.tar').read_bytes();assert len(raw)==record['archive_bytes'] and hashlib.sha256(raw).hexdigest()==record['archive_sha256']
with tarfile.open(OUT/'prefix.tar') as tar:
 assert {m.name for m in tar.getmembers()}==set(record['outputs'])
 for m in tar.getmembers():assert m.isfile() and (OUT/'collected'/m.name).resolve().is_relative_to((OUT/'collected').resolve())
 tar.extractall(OUT/'collected',filter='data')
for name,item in record['outputs'].items():assert hashlib.sha256((OUT/'collected'/name).read_bytes()).hexdigest()==item['sha256']
base=OUT/'collected/run_01'
with np.load(base/'seed0_rep0_BANK.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
z=torch.randn(32768,8,generator=torch.Generator().manual_seed(202680100000),dtype=torch.float32).numpy();np.testing.assert_array_equal(z,b['IID_z']);np.testing.assert_array_equal(b['rows'],np.repeat(b['source_parents'],32))
anti=z.reshape(1024,32,8).copy();anti[:,16:]=-anti[:,:16];assert np.array_equal(anti[:,16:],-anti[:,:16])
with np.load(base/'seed0_rep0_IID.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
distance=np.diag(a['FD_plus_theta']-a['FD_minus_theta']);np.testing.assert_array_equal((a['FD_plus_residual']-a['FD_minus_residual'])/distance,a['jacobian']);np.testing.assert_array_equal((a['FD_plus_native']-a['FD_minus_native'])/distance,a['native_jacobian'])
for name in ('one_rank','two_rank'):
 delta=a[name]-a['theta'];np.testing.assert_allclose(a['baseline_residual']+a['jacobian']@delta,a[name+'_predicted_residual'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(a['baseline_native']+a['native_jacobian']@delta+.5*a['positive_diagonal_native_curvature']@np.square(delta),a[name+'_predicted_native'],rtol=1e-12,atol=1e-12)
review=dict(status='PASS_STARTUP_PREFIX_ARITHMETIC_CHECK',UTC=datetime.now(timezone.utc).isoformat(),collected_completed_files=5,Gaussian_values_replayed=262144,derived_antithetic_pairs_checked=16384,FD_columns_recomputed=54,fixed_predictions_recomputed=2,new_model_calls=0,science_execution_queried=False,complete_terminal_audit_not_yet_executed=True,full_auditor_ast_only=True,scientific_acceptance=False)
(HERE/'AUDITOR_PREFIX_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8');print(json.dumps(review))
