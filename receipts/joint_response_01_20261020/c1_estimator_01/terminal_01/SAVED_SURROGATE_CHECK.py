"""Check existing C1 signal secants against the original generator's movements."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
old=root/'receipts/joint_response_01_20261020/c1_engineering_01'
manifest=json.loads((old/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
rows=[]
for seed in range(2):
    name=f'run_01/seed{seed}_C1_ENGINEERING.npz'
    raw=(old/'collected'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['outputs'][name]['sha256']
    a=np.load(old/'collected'/name,allow_pickle=False)
    np.testing.assert_array_equal(a['original_baseline_signal'],a['C1_baseline_signal'])
    baseline=a['original_baseline_signal']
    for index,point in enumerate(('trial0','trial1')):
        movement=a[f'original_{point}_actual_signal']-baseline
        original=a['original_signal_tangent'][index]
        alternative=a['C1_signal_tangent'][index]
        error_original=np.linalg.norm(movement-original)
        error_surrogate=np.linalg.norm(movement-alternative)
        rows.append(dict(seed=seed,point=point,rows=len(baseline),
            original_error_relative_to_baseline=float(error_original/np.linalg.norm(baseline)),
            C1_surrogate_error_relative_to_baseline=float(error_surrogate/np.linalg.norm(baseline)),
            C1_surrogate_error_relative_to_movement=float(error_surrogate/np.linalg.norm(movement)),
            error_ratio_surrogate_over_original=float(error_surrogate/error_original),
            surrogate_movement_cosine=float(np.sum(movement*alternative)/(np.linalg.norm(movement)*np.linalg.norm(alternative)))))
result=dict(status='COMPLETE_SAVED_ONLY_CROSS_GENERATOR_SECANT_CHECK',UTC=datetime.now(timezone.utc).isoformat(),
    input_hashes_verified=True,checks=rows,parameter_updates=0,new_generator_evaluations=0,
    interpretation='Posthoc fixed-direction signal diagnostics only. No target fitting, new scientific gate or permission to replace the original hard generator.',scientific_acceptance=False)
(here/'SAVED_SURROGATE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))
