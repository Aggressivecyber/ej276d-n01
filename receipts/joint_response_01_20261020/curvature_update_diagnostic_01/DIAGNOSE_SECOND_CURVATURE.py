from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy import sparse
sys.path.insert(0,'E:/GAN/tools/clarabel_0_11_1')
import clarabel

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_curvature_diagnostic_01'
reports=[];proposals={};inputs={};fits=('FIT0','FIT1','FIT2');radius=.0125
for seed in range(2):
 data=[]
 for path in [BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',HERE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE_UPDATE.npz']:
  inputs[path.relative_to(HERE.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:data.append({k:z[k] for k in z.files})
 a,first,second=data;r=a['stacked_baseline_residual'];n=a['stacked_baseline_native'];J=a['original_jacobian'];N=a['original_native_jacobian'];theta=a['theta']
 H=np.maximum((a['FD_plus_native']+a['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 D=N*radius/n[:,None];C=H*radius**2/n[:,None];anchor=np.concatenate([a[l+'_initial_native'] for l in fits]);cap=np.minimum(n,anchor*np.tile([1.01,1.01,1.10],9))/n-1
 directions=[];corrections=[];measured_errors=[]
 for measured,name in [(first,'curvature_anchor'),(second,'curvature_update')]:
  u=(measured[name]-theta)/radius;actual=np.concatenate([measured[name+'_'+label+'_actual_native'] for label in fits])/n
  predicted=1+D@u+.5*C@np.square(u)
  for v,e in zip(directions,corrections):predicted+=e*(v@u)**2
  e=np.maximum(actual-predicted,0);v=u/(u@u);directions.append(v);corrections.append(e)
  measured_errors.append(dict(name=name,maximum_positive_normalized_error=float(e.max()),native_errors=(actual-predicted).reshape(3,3,3).tolist(),predicted_anchor_ratios=(predicted*n/anchor).reshape(3,3,3).tolist(),actual_anchor_ratios=(actual*n/anchor).reshape(3,3,3).tolist()))
 assert (N[2::3,:53]==0).all() and (H[2::3,:53]==0).all() and all((e[2::3]==0).all() for e in corrections) and N[2::3,53].min()<0<N[2::3,53].max()
 matrices=[np.vstack([np.eye(53),-np.eye(53)])];vectors=[np.r_[(a['upper'][:53]-theta[:53])/radius,(theta[:53]-a['lower'][:53])/radius]];sizes=[106];cones=[clarabel.NonnegativeConeT(106)];macros=[]
 for il,label in enumerate(fits):
  L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
  for ih in range(3):
   sl=slice((3*il+ih)*272,(3*il+ih+1)*272);raw=L@(3*r[sl])/scale;der=(L@(3*J[sl]*radius))/scale[:,None]
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));norm=np.sqrt(56*(1.15*np.mean(raw[take]**2)+2/4096));Am=der[take]/norm;bm=raw[take]/norm;macros.append((Am,bm))
    matrices.append(np.vstack([np.zeros(53),-Am[:,:53]]));vectors.append(np.r_[1.,bm]);sizes.append(57);cones.append(clarabel.SecondOrderConeT(57))
 for k in range(27):
  if k%3==2:continue
  B=np.vstack([np.diag(np.sqrt(.5*C[k,:53])),*[np.sqrt(e[k])*v[:53] for v,e in zip(directions,corrections)]])
  matrices.append(np.vstack([D[k,:53],-2*B,D[k,:53]]));vectors.append(np.r_[cap[k]+1.,np.zeros(len(B)),cap[k]-1.]);sizes.append(len(B)+2);cones.append(clarabel.SecondOrderConeT(len(B)+2))
 matrices.append(np.vstack([np.zeros(53),-np.eye(53)]));vectors.append(np.r_[1.,np.zeros(53)]);sizes.append(54);cones.append(clarabel.SecondOrderConeT(54))
 A=np.vstack(matrices);b=np.concatenate(vectors);gram=(J*radius).T@(J*radius);G=(gram+.02*np.trace(gram)/54*np.eye(54))/(r@r);g=(J*radius).T@r/(r@r)
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.time_limit=30.;settings.max_threads=4;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
 solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(G[:53,:53])).tocsc(),g[:53],sparse.csc_matrix(A),b,cones,settings);solution=solver.solve();u=np.r_[np.array(solution.x),0.];dual=np.array(solution.z);slack=b-A@u[:53]
 primal=[];dual_margin=[];offset=0
 for index,size in enumerate(sizes):
  ss=slack[offset:offset+size];zz=dual[offset:offset+size];offset+=size
  primal.append(float(ss.min()) if index==0 else float(ss[0]-np.linalg.norm(ss[1:])))
  dual_margin.append(float(zz.min()) if index==0 else float(zz[0]-np.linalg.norm(zz[1:])))
 stationarity=float(np.max(np.abs(G[:53,:53]@u[:53]+g[:53]+A.T@dual)));complementarity=float(abs(slack@dual))
 native=1+D@u+.5*C@np.square(u)
 for v,e in zip(directions,corrections):native+=e*(v@u)**2
 original=np.r_[[1-np.square(Am@u+bm).sum() for Am,bm in macros],1+cap-native,1-u@u]
 certified=min(primal)>=-1e-8 and min(dual_margin)>=-1e-8 and stationarity<=1e-8 and complementarity<=1e-8 and original.min()>=-1e-6
 objective=float(.5*u@G@u+g@u);predicted=r+J@(u*radius)
 report=dict(seed=seed,original_solver_status=str(solution.status),iterations=solution.iterations,solve_seconds=solution.solve_time,certificate_pass=bool(certified),minimum_primal_cone_margin=min(primal),minimum_dual_cone_margin=min(dual_margin),stationarity=stationarity,complementarity=complementarity,minimum_original_constraint_slack=float(original.min()),objective=objective,predicted_joint_ratio=float(predicted@predicted/(r@r)),predicted_joint_ratios=(np.square(predicted.reshape(3,816)).sum(1)/np.square(r.reshape(3,816)).sum(1)).tolist(),predicted_native_ratios=native.reshape(3,3,3).tolist(),predicted_anchor_ratios=(native*n/anchor).reshape(3,3,3).tolist(),measured_direction_errors=measured_errors)
 np.savez(HERE/f'seed{seed}_SECOND_CURVATURE_SOLVER_EVIDENCE.npz',unit_step=u,conic_A=A,conic_b=b,P=G[:53,:53],q=g[:53],z=dual,s=np.array(solution.s),cone_sizes=np.array(sizes),theta=theta,positive_diagonal_curvature=H,rank_one_corrections=np.stack(corrections),rank_one_directions=np.stack(directions))
 if certified:proposals[f'seed{seed}_second_curvature']=theta+radius*u
 reports.append(report)
np.savez(HERE/'CERTIFIED_SECOND_CURVATURE_PROPOSALS.npz',**proposals)
record=dict(status='SAVED_SECOND_CURVATURE_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),reports=reports,both_candidates_certified=all(x['certificate_pass'] for x in reports),inputs=inputs,proposals_sha256=hashlib.sha256((HERE/'CERTIFIED_SECOND_CURVATURE_PROPOSALS.npz').read_bytes()).hexdigest(),new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,original_joint_objective_retained=True,uses_only_first_curvature_and_curvature_update_FIT_vectors=True,evidence_limit='Only numerical candidates; two PSD corrections bound the measured FIT directions, not unmeasured directions. Original failed status and original gates remain.')
(HERE/'SECOND_CURVATURE_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in x.items() if k not in ['predicted_native_ratios','predicted_anchor_ratios','measured_direction_errors']} for x in reports],indent=2))
