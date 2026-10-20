from pathlib import Path
import json
import shlex
import subprocess

out = Path(__file__).parent / 'fixed_validation_fix01'
code = r'''from pathlib import Path
import json
import numpy as np
import torch
base=Path('/home/rylan/projects/ej276d-n01')
memory=torch.load(base/'global_memory_joint_local_20261021/run_01/TRAIN_DISTRIBUTION_MEMORY.pt',map_location='cpu',weights_only=False)
lower=memory['joint_cholesky'].numpy()
variance=(lower*lower).sum(1)[:56]
weights=memory['weights'].numpy()
with np.load(base/'global_memory_fixed_validation_fix01_20261021/run_01/EVALUATION_FEATURE_MEANS.npz',allow_pickle=False) as archive:
    values={k:archive[k].copy() for k in archive.files}
result=[]
for h in ('fast','slow','common_level'):
    for source in ('FIT','REPORT'):
        initial=sum(values[f'{source}_initial_{h}_bank{b}_mean'] for b in range(2))/2
        candidate=sum(values[f'{source}_B_{h}_bank{b}_mean'] for b in range(2))/2
        for real in ('FIT','REPORT'):
            target=values[f'{real}_evaluation_target']
            r0=(lower@(initial-target)[-272:])[:56]
            delta=(lower@(candidate-initial)[-272:])[:56]
            old=np.mean(r0*r0/variance)
            linear=np.mean(2*r0*delta/variance)
            quadratic=np.mean(delta*delta/variance)
            new=np.mean((r0+delta)**2/variance)
            qn=sum(memory['levels'])
            end=qn+qn*len(memory['thresholds'][0])
            old_psd=np.sum(weights[qn:end]*(initial-target)[qn:end]**2)
            new_psd=np.sum(weights[qn:end]*(candidate-target)[qn:end]**2)
            result.append(dict(hypothesis=h,SOURCE_population=source,REAL_reference=real,
                wave_initial=float(old),wave_candidate=float(new),wave_change_percent=float(100*(new/old-1)),
                wave_linear_term=float(linear),wave_quadratic_term=float(quadratic),
                wave_identity_error=float(abs(new-old-linear-quadratic)),
                PSD_change_percent=float(100*(new_psd/old_psd-1))))
print(json.dumps(dict(status='SAVED_FEATURE_ANALYSIS_COMPLETE',new_model_forwards=0,new_optimizer_updates=0,
    scope='Post-hoc reference/population decomposition in frozen FIT feature map; cross-role cells are diagnostic, not new acceptance tests or physical pairings.',
    results=result)))
'''
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(code)],
    capture_output=True, check=True, timeout=30)
record = json.loads(r.stdout)
assert all(v['wave_identity_error'] < 1e-12 for v in record['results'])
(out / 'POPULATION_REFERENCE_DECOMPOSITION.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
