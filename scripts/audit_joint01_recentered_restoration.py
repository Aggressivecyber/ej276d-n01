"""Rebuild the fixed fresh-derivative programs and independently check certificates."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--config-sha256',required=True)
args=parser.parse_args();raw=args.config.read_bytes()
assert hashlib.sha256(raw).hexdigest()==args.config_sha256
p=json.loads(raw);solves=json.loads(Path(p['report']).read_bytes());reports=[]
assert solves['config_sha256']==args.config_sha256 and solves['execution_counts']==p['operation_budget']
assert solves['scientific_acceptance'] is False and not Path(p['audit_report']).exists()
for record in solves['reports']:
    seed=record['seed'];raw=Path(record['payload']).read_bytes()
    assert len(raw)==record['payload_bytes'] and hashlib.sha256(raw).hexdigest()==record['payload_sha256']
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:saved=dict(z)
    for value in saved.values():assert np.isfinite(value).all()
    radius=p['radius'];step=radius*saved['u']
    expected_A=[np.concatenate((np.eye(53),-np.eye(53)),axis=0)]
    expected_b=[];sizes=[106];native_A=[];native_b=[];maximum_joint=maximum_macro=maximum_native=0.
    for index,label in enumerate(p['libraries']):
        source=p['inputs'][str(seed)][label];raw=Path(source['path']).read_bytes()
        assert len(raw)==source['bytes'] and hashlib.sha256(raw).hexdigest()==source['sha256']
        assert record['source_hashes'][str(Path(source['path']))]==source['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:a=dict(z)
        center=a['theta'];J=a['jacobian'];N=a['native_jacobian']
        np.testing.assert_array_equal(saved['center'],center)
        if index==0:
            expected_b.append(np.concatenate((a['upper'][:53]-center[:53],center[:53]-a['lower'][:53]))/radius)
            lower=a['lower'];upper=a['upper']
        r0=a['previous_residual'];r=a['baseline_residual'];n0=a['previous_native']
        n=a['baseline_native'];anchor=a['initial_native'];prediction=r+J@step
        cap=p['joint_cap']*np.square(r0).sum();scale=np.sqrt(cap)
        expected_A.append(np.vstack((np.zeros((1,53)),-radius*J/scale)))
        expected_b.append(np.r_[1.,r/scale]);sizes.append(817)
        maximum_joint=max(maximum_joint,float(prediction@prediction/cap))
        L=a['cholesky'];variance=(L*L).sum(axis=1)
        for h in range(3):
            rows=slice(272*h,272*(h+1))
            previous=np.sqrt(3)*(L@r0[rows]);current=np.sqrt(3)*(L@r[rows]);derivative=np.sqrt(3)*(L@J[rows])
            for block in range(4):
                take=slice(56*block,56*(block+1))
                cap=p['macro_cap']*np.mean(previous[take]**2/variance[take])+p['macro_sampling_allowance']
                scale=np.sqrt(56*variance[take]*cap)
                expected_A.append(np.vstack((np.zeros((1,53)),-radius*derivative[take]/scale[:,None])))
                expected_b.append(np.r_[1.,current[take]/scale]);sizes.append(57)
                maximum_macro=max(maximum_macro,float(np.mean((current[take]+derivative[take]@step)**2/variance[take])/cap))
        keep=[0,1,3,4,6,7];cap=p['native_linear_cap_factor']*np.minimum(n0,anchor)
        native_A.append(radius*N[keep]/n0[keep,None]);native_b.append((cap[keep]-n[keep])/n0[keep])
        maximum_native=max(maximum_native,float(np.max((n+N@step)[keep]/cap[keep])))
        assert not N[2::3].any()
        np.testing.assert_array_equal((n+N@step)[2::3],n[2::3])
    expected_A.extend((np.vstack(native_A),np.vstack((np.zeros((1,53)),-np.eye(53)))))
    expected_b.extend((np.concatenate(native_b),np.r_[1.,np.zeros(53)]));sizes.extend((42,54))
    A=np.vstack(expected_A);b=np.concatenate(expected_b)
    assert A.shape==(10709,53) and len(b)==10709
    np.testing.assert_allclose(saved['A'],A,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(saved['b'],b,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(saved['cone_sizes'],sizes)
    np.testing.assert_array_equal(saved['native_linear_A'],np.vstack(native_A))
    np.testing.assert_array_equal(saved['native_linear_b'],np.concatenate(native_b))
    expected_candidate=center.copy();expected_candidate[:53]+=step
    np.testing.assert_array_equal(saved['candidate'],expected_candidate)
    assert saved['candidate'][53]==center[53] and float(saved['radius'])==radius
    u=saved['u'];dual=saved['dual'];slack=b-A@u
    np.testing.assert_allclose(saved['slack'],slack,rtol=1e-12,atol=1e-12)
    certificate=dual.copy()
    if record['solver_status']=='PrimalInfeasible':
        assert b@dual<0
        certificate/=-(b@dual)
    primal_margins=[];dual_margins=[];cursor=0
    for i,size in enumerate(sizes):
        s=slack[cursor:cursor+size];z=certificate[cursor:cursor+size];cursor+=size
        linear=i in (0,len(sizes)-2)
        primal_margins.append(float(s.min()) if linear else float(s[0]-np.linalg.norm(s[1:])))
        dual_margins.append(float(z.min()) if linear else float(z[0]-np.linalg.norm(z[1:])))
    assert cursor==10709
    stationarity=float(np.linalg.norm(u+A.T@dual,ord=np.inf));complementarity=float(abs(slack@dual))
    feasible=False;decision='UNRESOLVED_NUMERICAL_SOLVE_NO_CANDIDATE'
    extra={}
    if record['solver_status']=='Solved':
        assert min(primal_margins)>=-1e-8 and min(dual_margins)>=-1e-8
        assert stationarity<1e-8 and complementarity<1e-8
        assert max(maximum_joint,maximum_macro,maximum_native)<=1+1e-8
        assert np.linalg.norm(step)<=radius+1e-10
        assert (expected_candidate>=lower-1e-10).all() and (expected_candidate<=upper+1e-10).all()
        feasible=True;decision='FEASIBLE_KKT_CERTIFIED_LOCAL_MODEL_ONLY'
    elif record['solver_status']=='PrimalInfeasible':
        residual=float(np.linalg.norm(A.T@certificate,ord=np.inf))
        assert min(dual_margins)>=-1e-8 and residual<1e-8
        np.testing.assert_allclose(b@certificate,-1.,rtol=1e-10,atol=1e-10)
        decision='INFEASIBILITY_CERTIFIED_NO_CANDIDATE'
        extra=dict(infeasibility_stationarity=residual,certificate_b_dot=float(b@certificate))
    np.testing.assert_allclose(record['correction_norm'],np.linalg.norm(step),rtol=1e-12)
    reports.append(dict(seed=seed,solver_status=record['solver_status'],decision=decision,local_feasible=feasible,
        rows=cursor,libraries=7,joint_constraints=7,macro_constraints=84,native_linear_constraints=42,
        minimum_primal_margin=min(primal_margins),minimum_dual_margin=min(dual_margins),
        stationarity=stationarity,complementarity=complementarity,correction_norm=float(np.linalg.norm(step)),
        maximum_joint_fraction_of_cap=maximum_joint,maximum_macro_fraction_of_cap=maximum_macro,
        maximum_native_fraction_of_cap=maximum_native,payload_sha256=record['payload_sha256'],**extra))
assert len(reports)==2 and {r['seed'] for r in reports}=={0,1}
result=dict(status='PASS_INDEPENDENT_FRESH_RESTORATION_MATRIX_AND_CERTIFICATE_AUDIT',
    UTC=datetime.now(timezone.utc).isoformat(),reports=reports,two_seed_local_feasible=all(r['local_feasible'] for r in reports),
    config_sha256=args.config_sha256,new_solves=0,generator_calls=0,actual_candidate_evaluations=0,parameter_updates=0,
    scientific_acceptance=False,
    evidence_limit='Fresh-center saved linearized models only. Actual hard-readout, trust-region and original native gates require a separately registered fixed-candidate experiment.')
Path(p['audit_report']).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result))
