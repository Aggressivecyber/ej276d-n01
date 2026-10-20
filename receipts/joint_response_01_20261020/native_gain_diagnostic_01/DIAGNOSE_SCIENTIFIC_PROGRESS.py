from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_curvature_diagnostic_01'
review=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes());assert not review['two_seed_native_gain_all_local_pass']
rows=[];proposals={};inputs={};fits=('FIT0','FIT1','FIT2');radius=.0125
for seed in range(2):
 paths=[BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',HERE/f'terminal_01/collected/run_01/seed{seed}_NATIVE_GAIN.npz']
 data=[]
 for path in paths:
  inputs[str(path.relative_to(HERE.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:data.append({k:z[k] for k in z.files})
 a,c,g=data;r=a['stacked_baseline_residual'];n=a['stacked_baseline_native'];J=a['original_jacobian'];N=a['original_native_jacobian']
 H=np.maximum((a['FD_plus_native']+a['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0;D=N*radius/n[:,None];C=H*radius**2/n[:,None]
 rank_directions=[];rank_values=[];replay=[]
 for record,name in [(c,'curvature_anchor'),(g,'native_gain')]:
  delta=record[name]-a['theta'];u=delta/radius;actual=np.concatenate([record[name+'_'+l+'_actual_native'] for l in fits])/n-1
  pred=D@u+.5*C@np.square(u)
  for direction,value in zip(rank_directions,rank_values):pred+=value*(direction@u)**2
  value=np.maximum(actual-pred,0);direction=u/(u@u);rank_directions.append(direction);rank_values.append(value)
  replay.append(dict(name=name,added_positive_native_error_range=[float(value.min()),float(value.max())]))
 V=np.stack(rank_directions);Q=np.stack(rank_values,axis=1)
 for record,name in [(c,'curvature_anchor'),(g,'native_gain')]:
  u=(record[name]-a['theta'])/radius;actual=np.concatenate([record[name+'_'+l+'_actual_native'] for l in fits])/n-1
  assert (D@u+.5*C@np.square(u)+Q@np.square(V@u)>=actual-1e-12).all()
 macroA=[];macrob=[];peakA=[];peakb=[]
 for il,label in enumerate(fits):
  L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
  for ih in range(3):
   sl=slice((il*3+ih)*272,(il*3+ih+1)*272);b=L@(3*r[sl])/scale;A=(L@(3*J[sl]*radius))/scale[:,None]
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));normal=np.sqrt(56*(1.15*np.mean(b[take]**2)+2/4096));macroA.append(A[take]/normal);macrob.append(b[take]/normal)
   take=slice(112,168);normal=np.linalg.norm(b[take]);peakA.append(A[take]/normal);peakb.append(b[take]/normal)
 MA,mb,PA,pb=np.stack(macroA),np.stack(macrob),np.stack(peakA),np.stack(peakb)
 jr=r.reshape(9,272);jscale=np.linalg.norm(jr,axis=1);JA=(J*radius).reshape(9,272,54)/jscale[:,None,None];jb=jr/jscale[:,None]
 FR=(J*radius).reshape(3,816,54);fr=r.reshape(3,816);fd=np.square(fr).sum(1)
 cap=np.minimum(n,np.concatenate([a[l+'_initial_native'] for l in fits])*np.tile([1.01,1.01,1.10],9))/n-1
 idx=np.array([i for i in range(27) if i%3!=2])
 def cons(x):
  u,t=x[:-1],x[-1];native=D@u+.5*C@np.square(u)+Q@np.square(V@u)
  return np.r_[1-u@u,1-np.square(MA@u+mb).sum(1),.99-np.square(FR@u+fr).sum(1)/fd,
      cap-native,1-.20*t-np.square(JA@u+jb).sum(1),1-.10*t-np.square(PA@u+pb).sum(1),-native[idx]-.05*t]
 def jac(x):
  u=x[:-1];nv=D+C*u+2*(Q*(V@u)[None,:])@V
  mat=np.vstack((-2*u,-2*np.einsum('kij,ki->kj',MA,MA@u+mb),-2*np.einsum('kij,ki->kj',FR,FR@u+fr)/fd[:,None],-nv,
      -2*np.einsum('kij,ki->kj',JA,JA@u+jb),-2*np.einsum('kij,ki->kj',PA,PA@u+pb),-nv[idx]))
  return np.column_stack((mat,np.r_[np.zeros(1+36+3+27),np.full(9,-.20),np.full(9,-.10),np.full(18,-.05)]))
 x0=np.r_[(g['native_gain']-a['theta'])/radius,0.]
 numeric=np.column_stack([(cons(x0+np.eye(55)[i]*1e-6)-cons(x0-np.eye(55)[i]*1e-6))/(2e-6) for i in range(55)])
 error=float(np.max(np.abs(numeric-jac(x0))));assert error<1e-7
 solved=minimize(lambda x:-x[-1],x0,jac=lambda x:np.r_[np.zeros(54),-1.],method='SLSQP',bounds=[*zip((a['lower']-a['theta'])/radius,(a['upper']-a['theta'])/radius),(None,None)],constraints={'type':'ineq','fun':cons,'jac':jac},options={'maxiter':200,'ftol':1e-9,'disp':False})
 row=dict(seed=seed,success=bool(solved.success),message=solved.message,iterations=int(solved.nit),minimum_slack=float(cons(solved.x).min()),analytic_constraint_jacobian_error=error,rank_one_corrections=replay)
 if solved.success:
  assert cons(solved.x).min()>-1e-6;u=solved.x[:-1];native=D@u+.5*C@np.square(u)+Q@np.square(V@u)
  row.update(common_fraction_of_original_scientific_reductions=float(solved.x[-1]),joint_ratios=np.square(JA@u+jb).sum(1).reshape(3,3).tolist(),
      peak_ratios=np.square(PA@u+pb).sum(1).reshape(3,3).tolist(),native_ratios=(1+native).reshape(3,3,3).tolist(),step_norm=float(np.linalg.norm(u*radius)))
  proposals[f'seed{seed}_scientific_progress']=a['theta']+u*radius
 rows.append(row)
np.savez(HERE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz',**proposals)
result=dict(status='SAVED_SCIENTIFIC_PROGRESS_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,reports=rows,
    original_reduction_fractions=dict(joint=.20,peak=.10,native=.05),two_positive_solved_directions=all(x['success'] and x['common_fraction_of_original_scientific_reductions']>0 for x in rows),
    proposals_sha256=hashlib.sha256((HERE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz').read_bytes()).hexdigest(),new_generator_calls=0,parameter_updates=0,
    scientific_acceptance=False,evidence_limit='FIT-only local model. Two positive semidefinite rank-one terms majorize the two observed FIT directions only; other directions and new records still require native checks. Scientific goal fractions scale the objective, not the unchanged acceptance gates.')
(HERE/'SAVED_SCIENTIFIC_PROGRESS_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
