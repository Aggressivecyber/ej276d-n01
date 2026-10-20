from pathlib import Path
import json
import shlex
import subprocess

out = Path(__file__).parent / 'parent_audit_01'
code = r'''from pathlib import Path
import json
import numpy as np
import torch
base=Path('/home/rylan/projects/ej276d-n01')
memory=torch.load(base/'global_memory_joint_local_20261021/run_01/TRAIN_DISTRIBUTION_MEMORY.pt',map_location='cpu',weights_only=False)
with np.load(base/'global_memory_parent_audit_20261021/run_01/EVALUATION_FEATURE_MEANS.npz',allow_pickle=False) as z:
    data={k:z[k].copy() for k in z.files}
prior=json.loads((base/'global_memory_fixed_validation_fix01_20261021/run_01/EVALUATION_RESULT.json').read_text())
lower=memory['joint_cholesky'].numpy()
variance=(lower*lower).sum(1)[:56]
target=(lower@data['REPORT_evaluation_target'][-272:])[:56]
fit0,fit1=data['FIT_initial_parent_wave'],data['FIT_B_parent_wave']
report0,report1=data['REPORT_initial_parent_wave'],data['REPORT_B_parent_wave']
old=np.mean((report0.mean(0)-target)**2/variance)
new=np.mean((report1.mean(0)-target)**2/variance)
change=100*(new/old-1)
assert abs(change-prior['results']['REPORT']['fast']['changes_percent']['block_wave'])<1e-8
rng=np.random.default_rng(2026102161)
changes=[]
absolute=[]
for i in range(2000):
    rows=rng.choice(len(fit0),128,replace=False)
    first=np.mean((fit0[rows].mean(0)-target)**2/variance)
    last=np.mean((fit1[rows].mean(0)-target)**2/variance)
    changes.append(100*(last/first-1))
    absolute.append(last-first)
changes=np.asarray(changes)
fit_target=(lower@data['FIT_evaluation_target'][-272:])[:56]
r0=fit0.mean(0)-fit_target
delta=fit1-fit0
d=delta.mean(0)
contribution=np.mean(delta*(2*r0+d)/variance,axis=1)
full_change=np.mean((r0+d)**2/variance)-np.mean(r0*r0/variance)
assert abs(contribution.mean()-full_change)<1e-12
q=data['FIT_source_Q'].reshape(-1)
edges=np.quantile(q,np.linspace(0,1,9))
groups=np.searchsorted(edges[1:-1],q,side='right')
strata=[]
for group in range(8):
    selected=groups==group
    strata.append(dict(Q_left=float(edges[group]),Q_right=float(edges[group+1]),parents=int(selected.sum()),
        mean_signed_global_wave_contribution=float(contribution[selected].mean()),
        population_weighted_global_wave_contribution=float(contribution[selected].sum()/len(q))))
record=dict(status='COMPLETE_FIXED_PARENT_POPULATION_AUDIT',new_optimizer_updates=0,
    report_wave_change_percent=float(change),subset_parents=128,subsets=2000,
    FIT_subset_change_percent_quantiles={str(p):float(np.quantile(changes,p)) for p in [.025,.16,.5,.84,.975]},
    FIT_subset_fraction_worsening=float(np.mean(changes>0)),FIT_subset_fraction_above2pct=float(np.mean(changes>2)),
    REPORT_position_in_FIT_subset_reference=float(np.mean(changes<=change)),
    FIT_macro_absolute_wave_change=float(full_change),FIT_Q_strata_signed_contributions=strata,
    aggregate_reproduced=True,paired_common_random_numbers=True,
    interpretation_scope='Conditional on frozen model and two saved random banks. FIT subsets are overlapping finite-population reference draws, not independent experiments or a formal REPORT p-value. Source-Q strata decompose global loss; they do not pair SOURCE and REAL physically.',
    scientific_acceptance=False)
print(json.dumps(record))
'''
r = subprocess.run(['ssh', '-x', '-o', 'BatchMode=yes', 'rylan@100.82.231.102',
    '/home/rylan/projects/hpge-cpunet-validation-20261003/venv/bin/python -B -c ' + shlex.quote(code)],
    capture_output=True, check=True, timeout=30)
record = json.loads(r.stdout)
(out / 'PARENT_POPULATION_ANALYSIS.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
