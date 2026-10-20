from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np
from scipy import sparse
import sys
sys.path.insert(0,"E:/GAN/tools/clarabel_0_11_1")
import clarabel

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
 matrices=[np.vstack((np.eye(54),-np.eye(54)))];vectors=[np.r_[(a['upper']-theta)/radius,-(a['lower']-theta)/radius]];cones=[clarabel.NonnegativeConeT(108)]
 for Am,bm in zip(AA,bb):
  matrices.append(np.vstack((np.zeros(54),-Am)));vectors.append(np.r_[1.,bm]);cones.append(clarabel.SecondOrderConeT(57))
 native_blocks=[]
 for i in range(27):
  B=np.vstack((np.diag(np.sqrt(.5*C[i])),np.sqrt(correction[i])*v))
  Ai=np.vstack((D[i],-2*B,D[i]));bi=np.r_[cap[i]+1.,np.zeros(len(B)),cap[i]-1.]
  matrices.append(Ai);vectors.append(bi);cones.append(clarabel.SecondOrderConeT(len(bi)));native_blocks.append((Ai,bi))
 matrices.append(np.vstack((np.zeros(54),-np.eye(54))));vectors.append(np.r_[1.,np.zeros(54)]);cones.append(clarabel.SecondOrderConeT(55))
 conic_A=sparse.csc_matrix(np.vstack(matrices));conic_b=np.concatenate(vectors);P=G/(r@r);linear=g/(r@r)
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.time_limit=30.;settings.max_threads=4
 settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
 solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(P)).tocsc(),linear,conic_A,conic_b,cones,settings);solution=solver.solve()
 success=str(solution.status)=='Solved';message=str(solution.status);nit=solution.iterations;x=np.array(solution.x)
 translation_errors=[]
 for test in (u0,x):
  for i,(Ai,bi) in enumerate(native_blocks):
   ss=bi-Ai@test;formula=(ss[0]**2-ss[-1]**2-ss[1:-1]@ss[1:-1])/4
   original=cap[i]-D[i]@test-.5*C[i]@np.square(test)-correction[i]*(v@test)**2
   translation_errors.append(abs(formula-original))
 assert max(translation_errors)<1e-12
 slack=np.array(solution.s);dual=np.array(solution.z)
 kkt=dict(primal_equation_error=float(np.max(np.abs(conic_A@x+slack-conic_b))),stationarity_error=float(np.max(np.abs(P@x+linear+conic_A.T@dual))),complementarity=float(abs(slack@dual)),maximum_cone_translation_error=max(translation_errors),solve_seconds=solution.solve_time)
 evidence=dict(unit_step=x,conic_A=conic_A.toarray(),conic_b=conic_b,P=P,q=linear,s=slack,z=dual,theta=theta,original_residual=r,original_jacobian=J,original_native=n,original_native_jacobian=N,positive_diagonal_curvature=H,rank_one_correction=correction,rank_one_unit_direction=v)
 np.savez(HERE/f'seed{seed}_CONIC_SOLVER_EVIDENCE.npz',**evidence)

 row=dict(seed=seed,solver='Clarabel_0.11.1',full_54_parameters_and_27_native_constraints=True,kkt=kkt,success=bool(success),message=message,iterations=int(nit),minimum_slack=float(cons(x).min()),analytic_constraint_jacobian_error=grad_error,positive_rank_one_correction_range=[float(correction.min()),float(correction.max())])
 if success:
  assert cons(x).min()>-1e-6;u=x;step=u*radius;native=1+D@u+.5*C@np.square(u)+correction*(v@u)**2;rr=(r+J@step).reshape(3,816);oldr=r.reshape(3,816)
  row.update(predicted_joint_remaining=float(np.sum(rr**2)/np.sum(oldr**2)),predicted_joint_ratios=(np.sum(rr**2,1)/np.sum(oldr**2,1)).tolist(),predicted_native_ratios=native.reshape(3,3,3).tolist(),predicted_anchor_ratios=(native*n/anchor).reshape(3,3,3).tolist(),step_norm=float(np.linalg.norm(step)))
  proposals[f'seed{seed}_curvature_update']=theta+step
 reports.append(row)
np.savez(HERE/'SAVED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz',**proposals)
result=dict(status='SAVED_CONIC_CURVATURE_UPDATE_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,reports=reports,two_solutions=all(x['success'] for x in reports),
    proposals_sha256=hashlib.sha256((HERE/'SAVED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz').read_bytes()).hexdigest(),original_joint_objective_retained=True,
    new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,evidence_limit='Only original FIT matrices and the first measured curvature candidate FIT vector are used; no native-gain or goal-scaled candidate values, CHECK or FRESH enter this solve. Positive rank-one curvature is not a global upper bound.')
(HERE/'SAVED_CONIC_CURVATURE_UPDATE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in row.items() if k not in ['predicted_native_ratios','predicted_anchor_ratios']} for row in reports],indent=2))
