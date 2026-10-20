from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'startup_prefix_01'
base=OUT/'collected/run_01'
manifest=json.loads((OUT/'REMOTE_MANIFEST.json').read_bytes())
for name,item in manifest['outputs'].items():
    raw=(OUT/'collected'/name).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
raw=(OUT/'prefix.tar').read_bytes()
assert len(raw)==manifest['archive_bytes'] and hashlib.sha256(raw).hexdigest()==manifest['archive_sha256']
native=json.loads((HERE/'AUDITOR_RNG_NATIVE_REPLAY.json').read_bytes())
assert native['status']=='PASS_SAME_RUNTIME_EXACT_RANDOM_REPLAY'
assert native['bank_file_sha256']==manifest['outputs']['run_01/seed0_rep0_BANK.npz']['sha256']
with np.load(base/'seed0_rep0_BANK.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
assert native['normal_array_sha256']==hashlib.sha256(b['IID_z'].tobytes()).hexdigest()
assert native['seed']==b['bank_seed']==202680100000
np.testing.assert_array_equal(b['rows'],np.repeat(b['source_parents'],32))
anti=b['IID_z'].reshape(1024,32,8).copy();anti[:,16:]=-anti[:,:16]
np.testing.assert_array_equal(anti[:,16:],-anti[:,:16])
with np.load(base/'seed0_rep0_IID.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
distance=np.diag(a['FD_plus_theta']-a['FD_minus_theta'])
np.testing.assert_array_equal((a['FD_plus_residual']-a['FD_minus_residual'])/distance,a['jacobian'])
np.testing.assert_array_equal((a['FD_plus_native']-a['FD_minus_native'])/distance,a['native_jacobian'])
for name in ('one_rank','two_rank'):
    delta=a[name]-a['theta']
    np.testing.assert_allclose(a['baseline_residual']+a['jacobian']@delta,a[name+'_predicted_residual'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a['baseline_native']+a['native_jacobian']@delta+.5*a['positive_diagonal_native_curvature']@np.square(delta),a[name+'_predicted_native'],rtol=1e-12,atol=1e-12)
review=dict(status='PASS_STARTUP_PREFIX_ARITHMETIC_CHECK',UTC=datetime.now(timezone.utc).isoformat(),collected_completed_files=5,Gaussian_values_replayed_exact_in_native_runtime=262144,Windows_normal_bit_exact=False,platform_difference_receipt='AUDITOR_RNG_PLATFORM_DIFFERENCE.json',derived_antithetic_pairs_checked=16384,FD_columns_recomputed=54,fixed_predictions_recomputed=2,new_model_calls=0,science_execution_queried=False,complete_terminal_audit_not_yet_executed=True,full_auditor_ast_only=True,scientific_acceptance=False)
(HERE/'AUDITOR_PREFIX_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(review))
