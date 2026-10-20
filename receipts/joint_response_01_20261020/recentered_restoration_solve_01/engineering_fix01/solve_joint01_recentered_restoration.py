"""One fixed fresh-derivative restoration solve per saved response head."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
sys.path.insert(0, 'E:/GAN/tools/clarabel_0_11_1')
import clarabel
import numpy as np
from scipy.sparse import csc_matrix, eye

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
parser.add_argument('--config-sha256',required=True)
args=parser.parse_args()
config_raw=args.config.read_bytes()
assert hashlib.sha256(config_raw).hexdigest()==args.config_sha256
p=json.loads(config_raw)
assert p['operation_budget']['local_solves']==2 and not Path(p['report']).exists()
for name,entry in p['source_pins'].items():
    raw=Path(name).read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
audit_raw=Path(p['derivative_audit']['path']).read_bytes()
assert hashlib.sha256(audit_raw).hexdigest()==p['derivative_audit']['sha256']
assert json.loads(audit_raw)['status']=='PASS_SAVED_COMPLETED_RECENTERED_DERIVATIVE_AUDIT'
out=Path(p['NAS_output']);out.mkdir(parents=True,exist_ok=False)
started=time.perf_counter();reports=[]
for seed in range(2):
    matrices=[np.r_[np.eye(53),-np.eye(53)]]
    limits=[];sizes=[106];cones=[clarabel.NonnegativeConeT(106)]
    native_matrices=[];native_limits=[];sources={}
    for index,label in enumerate(p['libraries']):
        source=p['inputs'][str(seed)][label]
        path=Path(source['path']);raw=path.read_bytes()
        assert len(raw)==source['bytes'] and hashlib.sha256(raw).hexdigest()==source['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:a=dict(z)
        sources[str(path)]=source['sha256']
        if index==0:
            center=a['theta'];lower=a['lower'];upper=a['upper'];radius=p['radius']
            assert center.shape==(54,) and center[53]==0
            limits.append(np.r_[(upper[:53]-center[:53])/radius,(center[:53]-lower[:53])/radius])
        np.testing.assert_array_equal(a['theta'],center)
        np.testing.assert_array_equal(a['lower'],lower);np.testing.assert_array_equal(a['upper'],upper)
        J=a['jacobian'];N=a['native_jacobian']
        assert J.shape==(816,53) and N.shape==(9,53) and not N[2::3].any()
        r=a['baseline_residual'];r0=a['previous_residual']
        n=a['baseline_native'];n0=a['previous_native'];anchor=a['initial_native']
        denominator=np.sqrt(p['joint_cap']*(r0@r0))
        matrices.append(np.vstack([np.zeros(53),-J*radius/denominator]))
        limits.append(np.r_[1.,r/denominator]);sizes.append(817);cones.append(clarabel.SecondOrderConeT(817))
        L=a['cholesky'];variance=np.square(L).sum(1)
        for ih in range(3):
            section=slice(ih*272,(ih+1)*272)
            old_raw=L@(r0[section]*np.sqrt(3));current_raw=L@(r[section]*np.sqrt(3))
            derivative_raw=L@(J[section]*np.sqrt(3)*radius)
            for block in range(4):
                take=slice(56*block,56*(block+1))
                old_risk=np.mean(old_raw[take]**2/variance[take])
                scale=np.sqrt(56*(p['macro_cap']*old_risk+p['macro_sampling_allowance'])*variance[take])
                matrices.append(np.vstack([np.zeros(53),-derivative_raw[take]/scale[:,None]]))
                limits.append(np.r_[1.,current_raw[take]/scale]);sizes.append(57);cones.append(clarabel.SecondOrderConeT(57))
        keep=[0,1,3,4,6,7]
        cap=p['native_linear_cap_factor']*np.minimum(n0,anchor)
        native_matrices.append(N[keep]*radius/n0[keep,None])
        native_limits.append((cap[keep]-n[keep])/n0[keep])
    matrices.extend([np.vstack(native_matrices),np.vstack([np.zeros(53),-np.eye(53)])])
    limits.extend([np.concatenate(native_limits),np.r_[1.,np.zeros(53)]])
    sizes.extend([42,54]);cones.extend([clarabel.NonnegativeConeT(42),clarabel.SecondOrderConeT(54)])
    A=np.vstack(matrices);b=np.concatenate(limits)
    assert A.shape==(10709,53) and b.shape==(10709,)
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.max_iter=p['solver']['max_iterations'];settings.time_limit=p['solver']['time_limit_seconds']
    settings.tol_gap_abs=settings.tol_gap_rel=p['solver']['tolerance_gap']
    settings.tol_feas=p['solver']['tolerance_feasibility']
    assert time.perf_counter()-started<p['resources']['wall_seconds']
    answer=clarabel.DefaultSolver(eye(53,format='csc'),np.zeros(53),csc_matrix(A),b,cones,settings).solve()
    u=np.array(answer.x);dual=np.array(answer.z);slack=b-A@u
    candidate=center.copy();candidate[:53]+=radius*u
    status=str(answer.status)
    payload=out/f'seed{seed}_RECENTERED_RESTORATION.npz'
    np.savez(payload,A=A,b=b,u=u,dual=dual,slack=slack,center=center,candidate=candidate,
        cone_sizes=np.array(sizes),radius=np.array(radius),native_linear_A=np.vstack(native_matrices),
        native_linear_b=np.concatenate(native_limits))
    payload_raw=payload.read_bytes()
    record=dict(seed=seed,solver_status=status,iterations=answer.iterations,solve_seconds=answer.solve_time,
        objective=float(answer.obj_val) if status=='Solved' else None,correction_norm=float(np.linalg.norm(radius*u)),
        payload=str(payload),payload_bytes=len(payload_raw),payload_sha256=hashlib.sha256(payload_raw).hexdigest(),
        source_hashes=sources,requires_independent_audit=True,actual_candidate_executed=False)
    reports.append(record)
    Path(p['progress']).write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),
        completed_local_solves=len(reports),reports=reports),indent=2)+'\n',encoding='utf-8',newline='\n')
peak_rss=int(subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',
    f'(Get-Process -Id {os.getpid()}).PeakWorkingSet64'],text=True).strip())
elapsed=time.perf_counter()-started
assert elapsed<=p['resources']['wall_seconds'] and peak_rss<=p['resources']['RSS_GiB']*1024**3
assert sum(r['payload_bytes'] for r in reports)<=p['resources']['output_MiB']*1024**2
result=dict(status='FIXED_FRESH_DERIVATIVE_RESTORATION_SOLVES_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),
    config_sha256=args.config_sha256,reports=reports,elapsed_seconds=elapsed,peak_rss_bytes=peak_rss,
    clarabel_version=clarabel.__version__,numpy_version=np.__version__,
    execution_counts=dict(local_solves=2,matrix_libraries=14,response_coordinates=53,
        joint_constraints=14,macro_constraints=168,native_linear_constraints=84,
        generator_calls=0,actual_candidate_evaluations=0,parameter_updates=0,REAL_CHECK_rows=0,DEV_rows=0),
    scientific_acceptance=False,automatic_training=False,
    evidence_limit='Fixed convex local models only; independent constraint and KKT audit is required before any actual candidate registration.')
Path(p['report']).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result))
