from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.linalg import eigh

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'original_precision_REAL_pilot_01';Q=HERE.parent/'curvature_update_diagnostic_01'
checks=[];proposals={};sources={}
for seed in range(2):
    path=HERE/f'seed{seed}_STABLE_SUBSPACE_SOLVER_EVIDENCE.npz'
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    path=Q/f'seed{seed}_SECOND_CURVATURE_SOLVER_EVIDENCE.npz'
    with np.load(path,allow_pickle=False) as z:old={k:z[k] for k in z.files}
    path=BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz'
    with np.load(path,allow_pickle=False) as z:o={k:z[k] for k in z.files}
    banks=[]
    for rep in range(4):
        path=HERE/f'terminal_01/collected/run_01/seed{seed}_rep{rep}_IID.npz'
        sources[path.relative_to(HERE.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:banks.append({k:z[k] for k in z.files})
    magnitude=np.sqrt(sum(v['baseline_residual']@v['baseline_residual'] for v in banks)/4)
    j=np.stack([v['jacobian'] for v in banks])*.0125/magnitude
    mean=j.mean(0);B=mean.T@mean
    K=sum((v-mean).T@(v-mean) for v in j)/3
    ridge=.02*np.trace(B)/54
    eigen,V=eigh(B[:53,:53],K[:53,:53]/4+ridge*np.eye(53));P,_=np.linalg.qr(V[:,eigen>=4.])
    np.testing.assert_allclose(B,a['mean_J_gram'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(K,a['sample_noise_gram'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(P@P.T,a['U']@a['U'].T,rtol=1e-10,atol=1e-12)
    U=a['U'];k=U.shape[1];nr=len(old['conic_b'])
    np.testing.assert_allclose(a['conic_A'][:nr],old['conic_A']@U,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(a['conic_b'][:nr],old['conic_b'])
    r=o['stacked_baseline_residual'];J=o['original_jacobian'];offset=nr
    for fit in range(3):
        sl=slice(fit*816,(fit+1)*816);scale=np.sqrt(.99*(r[sl]@r[sl]))
        np.testing.assert_allclose(a['conic_A'][offset:offset+817],np.concatenate((np.zeros((1,k)),-.0125/scale*J[sl,:53]@U)),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a['conic_b'][offset:offset+817],np.r_[1.,r[sl]/scale],rtol=1e-12,atol=1e-12);offset+=817
    assert offset==len(a['conic_b'])
    A=a['conic_A'];b=a['conic_b'];z=a['z'];y=a['y'];slack=b-A@y
    if seed==0:
        corrected=z-A@np.linalg.solve(A.T@A,A.T@z)
        margins=[];pos=0
        for i,n in enumerate(a['cone_sizes']):
            block=corrected[pos:pos+n];pos+=n
            margins.append(float(block.min()) if i==0 else float(block[0]-np.linalg.norm(block[1:])))
        atz=float(np.max(np.abs(A.T@corrected)));bz=float(b@corrected)
        assert min(margins)>=0 and atz<=1e-11 and bz<-.01
        certificate=dict(status='INDEPENDENT_DUAL_INFEASIBILITY_CERTIFICATE',raw_ATz_max=float(np.max(np.abs(A.T@z))),correction_norm=float(np.linalg.norm(corrected-z)),minimum_cone_margin=min(margins),ATz_max=atz,b_dot_z=bz,changes_to_problem=0,raw_solver_record_preserved=True)
        np.savez(HERE/'seed0_STABLE_SUBSPACE_INFEASIBILITY_CERTIFICATE.npz',dual=corrected)
    else:
        margins=[];dual_margins=[];pos=0
        for i,n in enumerate(a['cone_sizes']):
            ss=slack[pos:pos+n];zz=z[pos:pos+n];pos+=n
            margins.append(float(ss.min()) if i==0 else float(ss[0]-np.linalg.norm(ss[1:])))
            dual_margins.append(float(zz.min()) if i==0 else float(zz[0]-np.linalg.norm(zz[1:])))
        stationarity=float(np.max(np.abs(a['P']@y+a['q']+A.T@z)));gap=float(abs(slack@z))
        assert min(margins)>=-1e-8 and min(dual_margins)>=-1e-8 and stationarity<1e-8 and gap<1e-8
        certificate=dict(status='INDEPENDENT_FEASIBLE_KKT_CERTIFICATE',minimum_primal_cone_margin=min(margins),minimum_dual_cone_margin=min(dual_margins),stationarity=stationarity,complementarity=gap)
    delta=banks[0]['one_rank']-banks[0]['theta'];projection=np.r_[U@(U.T@delta[:53]),0.]
    theta=o['theta'];candidate=theta+projection
    assert np.all(candidate>=o['lower']) and np.all(candidate<=o['upper']) and np.linalg.norm(projection)<=np.linalg.norm(delta)+1e-14
    assert delta[-1]==0
    proposals[f'seed{seed}_stable_projection']=candidate
    checks.append(dict(seed=seed,retained_dimension=k,certificate=certificate,projection_definition='Orthogonal projection of the fixed one-rank control step into the frozen stable subspace; no renormalization or new optimization.',full_step_norm=float(np.linalg.norm(delta)),projected_step_norm=float(np.linalg.norm(projection)),removed_step_norm=float(np.linalg.norm(delta-projection)),retained_step_squared_fraction=float(projection@projection/(delta@delta)),parameter_bounds_pass=True,proposed_as_fixed_ablation_only=True))
np.savez(HERE/'STABLE_PROJECTION_PROPOSALS.npz',**proposals)
record=dict(status='SAVED_SUBSPACE_AUDIT_AND_FIXED_PROJECTION_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),checks=checks,inputs=sources,proposal_sha256=hashlib.sha256((HERE/'STABLE_PROJECTION_PROPOSALS.npz').read_bytes()).hexdigest(),new_generator_calls=0,parameter_updates=0,scientific_acceptance=False,scope='Seed0 infeasibility is confined to the frozen projected convex approximation. Both orthogonal projections are separate fixed ablation probes, not solver-feasible proposals or accepted response endpoints. Native execution requires a new prospective registered job.')
(HERE/'STABLE_SUBSPACE_AUDIT_AND_PROJECTION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
