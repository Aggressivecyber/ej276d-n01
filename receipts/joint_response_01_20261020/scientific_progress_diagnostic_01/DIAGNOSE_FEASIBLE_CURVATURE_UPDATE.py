from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_curvature_diagnostic_01'
reports=[];proposals={};inputs={};radius=.0125;fits=('FIT0','FIT1','FIT2')
for seed in range(2):
 data=[]
 for path in [BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz']:
  inputs[str(path.relative_to(HERE.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:data.append({k:z[k] for k in z.files})
 a,c=data;theta=a['theta'];r=a['stacked_baseline_residual'];n=a['stacked_baseline_native'];J=a['original_jacobian'];N=a['original_native_jacobian']
 H=np.maximum((a['FD_plus_native']+a['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 measured=c['curvature_anchor']-theta;u0=measured/radius;v=u0/(u0@u0);actual=np.concatenate([c['curvature_anchor_'+l+'_actual_native'] for l in fits])
 correction=np.maximum(actual-(n+N@measured+.5*H@np.square(measured)),0)/n
 D=N*radius/n[:,None];C=H*radius**2/n[:,None];anchor=np.concatenate([a[l+'_initial_native'] for l in fits]);cap=np.minimum(n,anchor*np.tile([1.01,1.01,1.10],9))/n-1
 A=[];b=[]
 for il,label in enumerate(fits):
  L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
  for ih in range(3):
   sl=slice((3*il+ih)*272,(3*il+ih+1)*272);raw=L@(3*r[sl])/scale;der=(L@(3*J[sl]*radius))/scale[:,None]
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));normal=np.sqrt(56*(1.15*np.mean(raw[take]**2)+2/4096));A.append(der[take]/normal);b.append(raw[take]/normal)
 AA,bb=np.stack(A),np.stack(b);gram=(J*radius).T@(J*radius);G=gram+.02*np.trace(gram)/54*np.eye(54);g=(J*radius).T@r
 def cons(u):return np.r_[1-np.square(AA@u+bb).sum(1),cap-D@u-.5*C@np.square(u)-correction*(v@u)**2,1-u@u]
 def jac(u):return np.vstack((-2*np.einsum('kij,ki->kj',AA,AA@u+bb),-D-C*u-2*correction[:,None]*(v@u)*v,-2*u))
 numerical=np.column_stack([(cons(u0+np.eye(54)[i]*1e-6)-cons(u0-np.eye(54)[i]*1e-6))/(2e-6) for i in range(54)])
 grad_error=float(np.max(np.abs(numerical-jac(u0))));assert grad_error<1e-7
 phase_initial=np.r_[np.zeros(54),max(0.,-float(cons(np.zeros(54))[:-1].min()))]
 phase=minimize(lambda x:x[-1],phase_initial,jac=lambda x:np.r_[np.zeros(54),1.],method='SLSQP',bounds=[*zip((a['lower']-theta)/radius,(a['upper']-theta)/radius),(0,None)],constraints={'type':'ineq','fun':lambda x:np.r_[cons(x[:-1])[:-1]+x[-1],cons(x[:-1])[-1]],'jac':lambda x:np.column_stack((jac(x[:-1]),np.r_[np.ones(63),0.]))},options={'maxiter':200,'ftol':1e-10,'disp':False})
 phase_record=dict(seed=seed,success=bool(phase.success),message=phase.message,iterations=int(phase.nit),relaxation=float(phase.x[-1]),unrelaxed_minimum_slack=float(cons(phase.x[:-1]).min()))
 (HERE/f'seed{seed}_FEASIBILITY_PHASE.json').write_text(json.dumps(phase_record,indent=2)+'\n',encoding='utf-8')
 assert phase.success and phase.x[-1]<1e-8 and cons(phase.x[:-1]).min()>-1e-8,phase_record
 solved=minimize(lambda u:(.5*u@G@u+g@u)/(r@r),phase.x[:-1],jac=lambda u:(G@u+g)/(r@r),method='SLSQP',bounds=list(zip((a['lower']-theta)/radius,(a['upper']-theta)/radius)),constraints={'type':'ineq','fun':cons,'jac':jac},options={'maxiter':200,'ftol':1e-9,'disp':False})
 row=dict(seed=seed,feasibility_phase=phase_record,success=bool(solved.success),message=solved.message,iterations=int(solved.nit),minimum_slack=float(cons(solved.x).min()),analytic_constraint_jacobian_error=grad_error,positive_rank_one_correction_range=[float(correction.min()),float(correction.max())])
 if solved.success:
  assert cons(solved.x).min()>-1e-6;u=solved.x;step=u*radius;native=1+D@u+.5*C@np.square(u)+correction*(v@u)**2;rr=(r+J@step).reshape(3,816);oldr=r.reshape(3,816)
  row.update(predicted_joint_remaining=float(np.sum(rr**2)/np.sum(oldr**2)),predicted_joint_ratios=(np.sum(rr**2,1)/np.sum(oldr**2,1)).tolist(),predicted_native_ratios=native.reshape(3,3,3).tolist(),predicted_anchor_ratios=(native*n/anchor).reshape(3,3,3).tolist(),step_norm=float(np.linalg.norm(step)))
  proposals[f'seed{seed}_curvature_update']=theta+step
 reports.append(row)
np.savez(HERE/'SAVED_FEASIBLE_CURVATURE_UPDATE_PROPOSALS.npz',**proposals)
result=dict(status='SAVED_FEASIBLE_CURVATURE_UPDATE_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,reports=reports,two_solutions=all(x['success'] for x in reports),
    proposals_sha256=hashlib.sha256((HERE/'SAVED_FEASIBLE_CURVATURE_UPDATE_PROPOSALS.npz').read_bytes()).hexdigest(),original_joint_objective_retained=True,
    new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,evidence_limit='Only original FIT matrices and the first measured curvature candidate FIT vector are used; no native-gain or goal-scaled candidate values, CHECK or FRESH enter this solve. Positive rank-one curvature is not a global upper bound.')
(HERE/'SAVED_FEASIBLE_CURVATURE_UPDATE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in row.items() if k not in ['predicted_native_ratios','predicted_anchor_ratios']} for row in reports],indent=2))
