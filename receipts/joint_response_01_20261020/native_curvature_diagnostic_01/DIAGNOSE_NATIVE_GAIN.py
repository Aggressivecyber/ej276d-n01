from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'original_precision_REAL_pilot_01'
review=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes());assert not review['two_seed_curvature_all_local_pass']
reports=[];arrays={};fits=('FIT0','FIT1','FIT2');radius=.0125
for seed in range(2):
 with np.load(PREVIOUS/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(HERE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',allow_pickle=False) as z:new={k:z[k] for k in z.files}
 r=a['stacked_baseline_residual'];n=a['stacked_baseline_native'];J=a['original_jacobian'];N=a['original_native_jacobian']
 H=np.maximum((a['FD_plus_native']+a['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 old_step=new['curvature_anchor']-a['theta'];old_u=old_step/radius;direction=old_u/(old_u@old_u)
 observed=np.concatenate([new['curvature_anchor_'+l+'_actual_native'] for l in fits])
 discrepancy=observed-(n+N@old_step+.5*H@np.square(old_step));positive=np.maximum(discrepancy,0)/n
 native_rows=np.array([i for i in range(27) if i%3!=2]);D=N*radius/n[:,None];C=H*radius**2/n[:,None]
 anchor=np.concatenate([a[l+'_initial_native'] for l in fits]);cap=np.minimum(n,anchor*np.tile([1.01,1.01,1.10],9))/n-1
 operators=[];offsets=[]
 for il,label in enumerate(fits):
  L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
  for ih in range(3):
   sl=slice((il*3+ih)*272,(il*3+ih+1)*272);b=L@(3*r[sl])/scale;A=(L@(3*J[sl]*radius))/scale[:,None]
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));normal=np.sqrt(56*(1.15*np.mean(b[take]**2)+2/4096));operators.append(A[take]/normal);offsets.append(b[take]/normal)
 AA=np.stack(operators);bb=np.stack(offsets)
 JR=(J*radius).reshape(3,816,54);rr=r.reshape(3,816);den=np.square(rr).sum(1)
 for corrected in (False,True):
  q=positive if corrected else np.zeros(27)
  def constraints(x):
   u=x[:-1];t=x[-1];native=D@u+.5*C@np.square(u)+q*(direction@u)**2
   return np.r_[1-u@u,1-np.square(AA@u+bb).sum(1),.99-np.square(rr+JR@u).sum(1)/den,cap-native,-native[native_rows]-t]
  def jacobian(x):
   u=x[:-1];nv=D+C*u+2*q[:,None]*(direction@u)*direction
   rows=np.vstack((-2*u,-2*np.einsum('kij,ki->kj',AA,AA@u+bb),-2*np.einsum('kij,ki->kj',JR,rr+JR@u)/den[:,None],-nv,-nv[native_rows]))
   last=np.r_[np.zeros(len(rows)-len(native_rows)),-np.ones(len(native_rows))]
   return np.column_stack((rows,last))
  x0=np.r_[old_u,-.1]
  numerical=np.column_stack([(constraints(x0+np.eye(55)[i]*1e-6)-constraints(x0-np.eye(55)[i]*1e-6))/(2e-6) for i in range(55)])
  grad_error=float(np.max(np.abs(numerical-jacobian(x0))));assert grad_error<1e-7,grad_error
  solution=minimize(lambda x:-x[-1],x0,jac=lambda x:np.r_[np.zeros(54),-1.],method='SLSQP',bounds=[*zip((a['lower']-a['theta'])/radius,(a['upper']-a['theta'])/radius),(None,None)],constraints={'type':'ineq','fun':constraints,'jac':jacobian},options={'maxiter':200,'ftol':1e-9,'disp':False})
  row=dict(seed=seed,FIT_positive_rank_one_correction=corrected,success=bool(solution.success),message=solution.message,iterations=int(solution.nit),analytic_constraint_jacobian_error=grad_error,minimum_slack=float(constraints(solution.x).min()),
      measured_positive_curvature_underprediction_range=[float(positive[native_rows].min()),float(positive[native_rows].max())])
  if solution.success:
   assert constraints(solution.x).min()>-1e-6
   u=solution.x[:-1];delta=u*radius;native=D@u+.5*C@np.square(u)+q*(direction@u)**2;predicted=r+J@delta
   row.update(maximum_common_native_reduction=float(solution.x[-1]),predicted_joint_ratios=(np.square(rr+JR@u).sum(1)/den).tolist(),
       predicted_native_ratios=(1+native).reshape(3,3,3).tolist(),predicted_anchor_ratios=((1+native)*n/anchor).reshape(3,3,3).tolist(),
       step_norm=float(np.linalg.norm(delta)),full_joint_remaining=float(predicted@predicted/(r@r)))
   arrays[f'seed{seed}_rankone{int(corrected)}']=a['theta']+delta
  reports.append(row)
np.savez(HERE/'SAVED_NATIVE_GAIN_PROPOSALS.npz',**arrays)
result=dict(status='SAVED_NATIVE_GAIN_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),reports=reports,
    terminal_review_sha256=hashlib.sha256((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes()).hexdigest(),
    proposals_sha256=hashlib.sha256((HERE/'SAVED_NATIVE_GAIN_PROPOSALS.npz').read_bytes()).hexdigest(),
    new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,
    evidence_limit='Optimistic local model feasibility, not actual waveform results. Only saved FIT matrices and FIT values enter the solves. Positive rank-one terms retain measured mixed-parameter directional effects but do not certify unseen directions.')
(HERE/'SAVED_NATIVE_GAIN_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in x.items() if k not in ['predicted_native_ratios','predicted_anchor_ratios']} for x in reports],indent=2))
