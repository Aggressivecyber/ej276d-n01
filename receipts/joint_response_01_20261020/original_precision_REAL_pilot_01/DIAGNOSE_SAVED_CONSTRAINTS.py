from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,numpy as np
from scipy.optimize import minimize

ROOT=Path('E:/GAN/research/N01'); HERE=Path(__file__).resolve().parent
OUT=HERE/'terminal_01/collected/run_01'
p=json.loads((ROOT/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes())
audit=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert audit['status']=='PASS_SAVED_ORIGINAL_PRECISION_REAL_PILOT_AUDIT'
labels=('FIT0','FIT1','FIT2'); metrics=('wave','QPSD','noise'); hypotheses=('fast','slow','common_level')
records=[]; inputs={}; proposals={}; models=[]
for seed in range(2):
 for kind,dim in [('joint',54),('physical',46)]:
  for iteration in range(3):
   stem=f'seed{seed}_{kind}_ITER{iteration:02d}'
   path=OUT/(stem+'.npz'); inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
   with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   report=json.loads((OUT/(stem+'.json')).read_bytes())
   theta=a['theta']; n=a['stacked_baseline_native']; r=a['stacked_baseline_residual']
   J=a['original_jacobian']; N=a['original_native_jacobian']; h=p['finite_difference_step']
   # Electronics is at its one-sided lower bound, so no central curvature is available there.
   symmetric=np.isclose(np.diag(a['FD_plus_theta'])-theta,h,rtol=0,atol=1e-14)&np.isclose(theta-np.diag(a['FD_minus_theta']),h,rtol=0,atol=1e-14)
   H=np.zeros_like(N); H[:,symmetric]=(a['FD_plus_native'][:,symmetric]+a['FD_minus_native'][:,symmetric]-2*n[:,None])/h**2
   Hpositive=np.maximum(H,0)
   groups=[slice(0,24),slice(24,45),slice(45,dim-1),slice(dim-1,dim)]
   gram=J.T@J; within=np.zeros_like(gram)
   for g in groups:within[g,g]=gram[g,g]
   models.append(dict(stem=stem,symmetric_curvature_columns=np.flatnonzero(symmetric).tolist(),
       cross_group_gram_Frobenius_fraction=float(np.linalg.norm(gram-within)/np.linalg.norm(gram)),
       center_anchor_ratios=(n/np.concatenate([a[l+'_initial_native'] for l in labels])).reshape(3,3,3).tolist()))
   for k,trial in enumerate(report['trials']):
    step=a[f'trial{k}_proposal']-theta
    actual=np.concatenate([a[f'trial{k}_{l}_actual_native'] for l in labels])
    original=n+N@step; quadratic=original+.5*Hpositive@np.square(step)
    used=a[f'trial{k}_predicted_native']; linear_error=(actual-original)/n; quadratic_error=(actual-quadratic)/n
    by_metric={m:dict(original_linear_error_range=[float(linear_error.reshape(3,3,3)[:,:,i].min()),float(linear_error.reshape(3,3,3)[:,:,i].max())],
       positive_diagonal_error_range=[float(quadratic_error.reshape(3,3,3)[:,:,i].min()),float(quadratic_error.reshape(3,3,3)[:,:,i].max())],
       original_linear_mean_ratio=float((original/n).reshape(3,3,3)[:,:,i].mean()),actual_mean_ratio=float((actual/n).reshape(3,3,3)[:,:,i].mean()),
       positive_diagonal_mean_ratio=float((quadratic/n).reshape(3,3,3)[:,:,i].mean()),
       used_secant_error_range=[float(((actual-used)/n).reshape(3,3,3)[:,:,i].min()),float(((actual-used)/n).reshape(3,3,3)[:,:,i].max())]) for i,m in enumerate(metrics)}
    failures=[]
    for label,c in trial['checks'].items():
     old=report['baseline_risks'][label]
     ratio=np.mean([c['risks'][hh]['joint_whitened'] for hh in hypotheses])/np.mean([old[hh]['joint_whitened'] for hh in hypotheses])
     if ratio>.99:failures.append(label+'/joint_gain')
     for source,key in [('current','native_ratios'),('initial','initial_anchor_ratios')]:
      values=np.array(c[key])
      for ih,hh in enumerate(hypotheses):
       for im,m in enumerate(metrics):
        if values[ih,im]>(1.10 if m=='noise' else 1.01):failures.append(label+'/'+source+'/'+hh+'/'+m)
    records.append(dict(stem=stem,trial=k,radius=trial['radius'],accepted=trial['accepted'],
        predicted_joint_gain_fraction=1-trial['geometry']['predicted_joint_remaining_fraction'],
        actual_to_predicted_joint_gain_ratio=trial['reduction_ratio'],group_step_norms=[float(np.linalg.norm(step[g])) for g in groups],
        actual_failure_reasons=failures,native_prediction=by_metric))
   if kind!='joint' or iteration!=2:continue
   radius=p['initial_radius']; operators=[]; offsets=[]
   for il,label in enumerate(labels):
    L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
    for ih in range(3):
     sl=slice((3*il+ih)*len(L),(3*il+ih+1)*len(L));center=L@(r[sl]*3)/scale;derivative=(L@(J[sl]*3*radius))/scale[:,None]
     for ib in range(4):
      take=slice(56*ib,56*(ib+1));cap=1.15*np.square(center[take]).mean()+2/4096
      offsets.append(center[take]/np.sqrt(56*cap));operators.append(derivative[take]/np.sqrt(56*cap))
   aa,bb=np.stack(operators),np.stack(offsets);scaled=J*radius; gram=scaled.T@scaled
   hh=gram+p['solver']['ridge_fraction']*np.diag(gram).mean()*np.eye(dim);gg=scaled.T@r
   initial=np.concatenate([a[l+'_initial_native'] for l in labels]);hard_cap=initial*np.tile([1.01,1.01,1.10],9)
   for curved,anchored in [(False,False),(False,True),(True,False),(True,True)]:
    name=f'{stem}_curvature{int(curved)}_anchor{int(anchored)}'; C=(Hpositive if curved else np.zeros_like(H))*radius**2/n[:,None]
    D=N*radius/n[:,None]; cap=(np.minimum(n,hard_cap) if anchored else n)/n-1
    def objective(u):return float((.5*u@hh@u+gg@u)/(r@r))
    def gradient(u):return (hh@u+gg)/(r@r)
    def constraints(u):return np.r_[1-np.square(bb+aa@u).sum(1),cap-D@u-.5*C@np.square(u),1-u@u]
    def jacobian(u):return np.vstack((-2*np.einsum('mni,mn->mi',aa,bb+aa@u),-D-C*u,-2*u))
    solve=minimize(objective,np.zeros(dim),jac=gradient,method='SLSQP',bounds=list(zip((a['lower']-theta)/radius,(a['upper']-theta)/radius)),constraints={'type':'ineq','fun':constraints,'jac':jacobian},options={'maxiter':200,'ftol':1e-9,'disp':False})
    record=dict(name=name,success=bool(solve.success),message=solve.message,iterations=int(solve.nit),minimum_slack=float(constraints(solve.x).min()))
    if solve.success:
     assert constraints(solve.x).min()>=-1e-6
     step=solve.x*radius;pn=n+N@step+.5*(Hpositive if curved else np.zeros_like(H))@np.square(step);pr=r+J@step
     if not curved and not anchored:np.testing.assert_allclose(step,a['trial0_proposal']-theta,rtol=1e-8,atol=2e-7)
     proposals[name]=theta+step
     record.update(parameter_step_norm=float(np.linalg.norm(step)),predicted_joint_remaining=float(pr@pr/(r@r)),
       predicted_native_ratios=(pn/n).reshape(3,3,3).tolist(),predicted_anchor_ratios=(pn/initial).reshape(3,3,3).tolist(),
       active_native_constraints=[f'{labels[i//9]}/{hypotheses[(i%9)//3]}/{metrics[i%3]}' for i,x in enumerate(constraints(solve.x)[36:63]) if x<1e-7],
       group_step_norms=[float(np.linalg.norm(step[g])) for g in groups])
    models.append(record)
np.savez(HERE/'SAVED_CONSTRAINT_PROPOSALS.npz',**proposals)
neural=[x for x in records if '_joint_' in x['stem']]
first=[x for x in neural if x['trial']==0]
result=dict(status='SAVED_MATRIX_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,
    original_audit_sha256=hashlib.sha256((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes()).hexdigest(),
    original_scientific_runs_unchanged=True,new_generator_calls=0,parameter_updates=0,
    trials=records,models=models,proposals_sha256=hashlib.sha256((HERE/'SAVED_CONSTRAINT_PROPOSALS.npz').read_bytes()).hexdigest(),
    summary=dict(neural_trials=len(neural),physical_trials=len(records)-len(neural),first_neural_trials=len(first),
      all_first_neural_trials_actual_wave_mean_increase=all(x['native_prediction']['wave']['actual_mean_ratio']>1 for x in first),
      all_first_neural_trials_linear_wave_mean_decrease=all(x['native_prediction']['wave']['original_linear_mean_ratio']<1 for x in first),
      native_failure_counts=dict(Counter(reason.split('/')[-1] for x in neural for reason in x['actual_failure_reasons'] if reason.split('/')[-1] in metrics))),
    evidence_limit='Exploratory arithmetic on saved FIT matrices. Positive diagonal curvature omits mixed terms and is not a proven upper bound; no candidate waveform has been generated. One-sided electronics curvature is not inferred. No CHECK/final observations were used in alternative solves.')
(HERE/'SAVED_CONSTRAINT_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(summary=result['summary'],alternative_solves=[{k:v for k,v in x.items() if k not in ['predicted_native_ratios','predicted_anchor_ratios']} for x in models if 'name' in x]),indent=2))
