from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
sys.path.insert(0,'E:/GAN/tools/clarabel_0_11_1')
import clarabel

HERE=Path(__file__).resolve().parent;OUT=HERE/'terminal_01/collected/run_01'
BASE=HERE.parent/'original_precision_REAL_pilot_01';Q=HERE.parent/'curvature_update_diagnostic_01'
design=dict(status='PROSPECTIVE_SAVED_MATRIX_SUBSPACE_CHECK',UTC=datetime.now(timezone.utc).isoformat(),schemes_used_for_fit=['IID'],replicates=4,threshold=4.,definition='Generalized mean-J signal versus estimated variance of the four-bank mean plus original ridge; keep eigenvalues >=4. This is a descriptive twofold signal-to-root-variance-plus-ridge rule, not a confidence bound.',noise_coordinate='Exactly zero as already implied by opposite native-noise slopes and unchanged noise constraints; use first 53 coordinates.',subspace_orthonormalization='Euclidean QR of retained generalized eigenvectors.',objective='Original three-FIT complete joint objective including all cross-coordinate terms and original ridge, restricted to retained subspace.',constraints='Original box, macro, radius .0125, current native nonincrease and fixed initial 1.01/1.10 cap with the same two existing PSD curvature corrections; additionally require each FIT predicted joint <=.99.',new_curvature_corrections=0,new_generator_calls=0,new_candidates_at_most=2,solver_calls=2,solver_seconds_each=30,max_iterations_each=200,threshold_retuning_or_fallback=False,execution_rule='Only independently certified candidates may be considered for a separately registered native diagnostic. No model calls or training in this check.')
assert not (HERE/'STABLE_SUBSPACE_DESIGN.json').exists()
(HERE/'STABLE_SUBSPACE_DESIGN.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf-8')
reports=[];inputs={};proposals={}
for seed in range(2):
    banks=[]
    for rep in range(4):
        path=OUT/f'seed{seed}_rep{rep}_IID.npz';inputs[path.relative_to(HERE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:banks.append({k:z[k] for k in z.files})
    rscale=np.sqrt(np.mean([a['baseline_residual']@a['baseline_residual'] for a in banks]))
    Js=np.stack([a['jacobian'] for a in banks])*.0125/rscale
    mean=Js.mean(0);B=mean.T@mean;ridge=.02*np.trace(B)/54
    dif=Js-mean;K=np.einsum('rpi,rpj->ij',dif,dif)/3
    vals,V=eigh(B[:53,:53],K[:53,:53]/4+ridge*np.eye(53))
    keep=vals>=4.;assert keep.any()
    U,_=np.linalg.qr(V[:,keep]);assert np.max(np.abs(U.T@U-np.eye(U.shape[1])))<1e-12
    arrays=[]
    for path in (BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',Q/f'seed{seed}_SECOND_CURVATURE_SOLVER_EVIDENCE.npz'):
        inputs[path.relative_to(HERE.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
    original,conic=arrays
    P=conic['P'];q=conic['q'];A0=conic['conic_A'];b0=conic['conic_b'];sizes0=conic['cone_sizes']
    r=original['stacked_baseline_residual'];J=original['original_jacobian'];theta=original['theta']
    k=U.shape[1];matrices=[A0@U];vectors=[b0];sizes=sizes0.tolist()
    for fit in range(3):
        sl=slice(fit*816,(fit+1)*816);normal=np.sqrt(.99*(r[sl]@r[sl]))
        matrices.append(np.vstack([np.zeros(k),-J[sl,:53]@U*.0125/normal]));vectors.append(np.r_[1.,r[sl]/normal]);sizes.append(817)
    A=np.vstack(matrices);b=np.concatenate(vectors);Pc=U.T@P@U;qc=U.T@q
    cones=[clarabel.NonnegativeConeT(sizes[0])]+[clarabel.SecondOrderConeT(n) for n in sizes[1:]]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.time_limit=30.;settings.max_threads=4
    settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9;settings.tol_feas=1e-9
    solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(Pc)).tocsc(),qc,sparse.csc_matrix(A),b,cones,settings)
    sol=solver.solve();y=np.array(sol.x);dual=np.array(sol.z);slack=b-A@y
    p_margins=[];d_margins=[];offset=0
    for i,size in enumerate(sizes):
        ss=slack[offset:offset+size];zz=dual[offset:offset+size];offset+=size
        p_margins.append(float(ss.min()) if i==0 else float(ss[0]-np.linalg.norm(ss[1:])))
        d_margins.append(float(zz.min()) if i==0 else float(zz[0]-np.linalg.norm(zz[1:])))
    stationarity=float(np.max(np.abs(Pc@y+qc+A.T@dual)));complementarity=float(abs(slack@dual))
    infeasibility=dict(minimum_dual_cone_margin=min(d_margins),ATz_max=float(np.max(np.abs(A.T@dual))),b_dot_z=float(b@dual))
    feasible=min(p_margins)>=-1e-8 and min(d_margins)>=-1e-8 and stationarity<=1e-8 and complementarity<=1e-8
    infeasible=min(d_margins)>=-1e-8 and infeasibility['ATz_max']<=1e-8 and infeasibility['b_dot_z']<-1e-8
    u=np.r_[U@y,0.];pred=r+J@(.0125*u)
    assert not (feasible and infeasible)
    if feasible:proposals[f'seed{seed}_stable_subspace']=theta+.0125*u
    report=dict(seed=seed,retained_dimension=k,generalized_eigenvalues=vals.tolist(),original_ridge=float(ridge),solver_status=str(sol.status),iterations=sol.iterations,solve_seconds=sol.solve_time,feasibility_certificate_pass=bool(feasible),infeasibility_certificate_pass=bool(infeasible),minimum_primal_cone_margin=min(p_margins),stationarity=stationarity,complementarity=complementarity,infeasibility_certificate=infeasibility,predicted_joint_ratio=float(pred@pred/(r@r)),predicted_per_FIT_joint_ratio=(np.square(pred.reshape(3,816)).sum(1)/np.square(r.reshape(3,816)).sum(1)).tolist(),step_norm=float(np.linalg.norm(.0125*u)))
    np.savez(HERE/f'seed{seed}_STABLE_SUBSPACE_SOLVER_EVIDENCE.npz',U=U,eigenvalues=vals,mean_J_gram=B,sample_noise_gram=K,conic_A=A,conic_b=b,P=Pc,q=qc,y=y,z=dual,s=np.array(sol.s),cone_sizes=np.array(sizes),unit_step=u,theta=theta)
    reports.append(report)
np.savez(HERE/'CERTIFIED_STABLE_SUBSPACE_PROPOSALS.npz',**proposals)
result=dict(status='SAVED_STABLE_SUBSPACE_CHECK_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),reports=reports,inputs=inputs,design_sha256=hashlib.sha256((HERE/'STABLE_SUBSPACE_DESIGN.json').read_bytes()).hexdigest(),both_candidates_certified=all(r['feasibility_certificate_pass'] for r in reports),new_generator_calls=0,parameter_updates=0,scientific_acceptance=False)
(HERE/'STABLE_SUBSPACE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k!='generalized_eigenvalues'} for r in reports],indent=2))
