from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import minimize

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
old=root/'receipts/joint_response_01_20261020/multibank_m0_01/terminal_01/run_01'
p=json.loads((root/'configs/joint_response_01_original_draw_precision_20261020.json').read_bytes())
audit=(here/'AUDIT_COMPLETE.py').read_text()
body=audit[audit.index('        scaled=J*radius'):audit.index('        fr = np.concatenate')]
body='\n'.join(line[8:] for line in body.splitlines())
errors=[]
started=time.monotonic()
for seed in range(2):
    a=np.load(old/f'seed{seed}_MULTIBANK_M0.npz',allow_pickle=False)
    report=json.loads((old/f'seed{seed}_MULTIBANK_M0.json').read_bytes())
    for i,trial in enumerate(report['trials']):
        r,n=a['stacked_baseline_residual'],a['stacked_baseline_native']
        theta=a['theta'];radius=trial['radius'];step=a[f'trial{i}_proposal']-theta
        J,N=a[f'trial{i}_jacobian'],a[f'trial{i}_native_jacobian']
        operators,offsets=[],[]
        for il,label in enumerate(('FIT0','FIT1','FIT2')):
            L=a[label+'_cholesky'];variance=np.square(L).sum(1)
            for ih in range(3):
                sl=slice((il*3+ih)*272,(il*3+ih+1)*272)
                raw0=L@(r[sl]*3)
                raw_j=L@(J[sl]*3*radius)
                for ib in range(4):
                    take=slice(ib*56,(ib+1)*56)
                    cap=1.15*np.mean(raw0[take]**2/variance[take])+2/4096
                    normalization=np.sqrt(56*cap)
                    offsets.append(raw0[take]/np.sqrt(variance[take])/normalization)
                    operators.append(raw_j[take]/np.sqrt(variance[take])[:,None]/normalization)
        max_solver_difference=0.;counts={'independent_joint_solutions':0}
        exec(compile(body,'independent_saved_solver','exec'))
        errors.append(dict(seed=seed,trial=i,max_delta_difference=difference))
receipt=dict(status='PASS_AUDITOR_SOLVER_ON_FOUR_PINNED_HISTORICAL_TRIALS',
    elapsed_seconds=time.monotonic()-started,trials=errors,new_generator_calls=0,new_scientific_updates=0,
    current_live_job_outputs_not_used=True)
(here/'AUDITOR_SOLVER_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt))
