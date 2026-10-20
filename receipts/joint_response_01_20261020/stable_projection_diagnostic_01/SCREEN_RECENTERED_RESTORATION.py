"""Screen a new-center restoration using saved intercepts and explicitly stale derivatives."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, sys
sys.path.insert(0,'E:/GAN/tools/clarabel_0_11_1')
import clarabel
import numpy as np
from scipy.sparse import csc_matrix, eye

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'original_precision_REAL_pilot_01/terminal_01/collected/run_01'
RANDOM=HERE.parent/'random_precision_diagnostic_01/terminal_01/collected/run_01'
OUT=HERE/'terminal_01/collected/run_01'
PAYLOAD=Path('Z:/GAN/research/N01/receipts/joint_response_01_20261020/stable_projection_diagnostic_01')
PAYLOAD.mkdir(parents=True,exist_ok=True)
design=dict(UTC=datetime.now(timezone.utc).isoformat(),center='fixed unprojected one-rank control',
    derivative_center='previous ITER02 theta; stale, screening only',libraries=['FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3'],
    radius=.0125/4,native_restoration_factor=1.005,original_native_gate=1.01,
    joint_cap=.99,macro_cap=1.15,macro_sampling_allowance=2/4096,
    objective='minimum Euclidean correction norm',response_coordinates=53,noise_correction=0,
    generator_calls=0,solver_limit=2,threshold_search=False,candidates_not_for_native_execution=True)
(HERE/'RECENTERED_RESTORATION_SCREEN_DESIGN.json').write_text(json.dumps(design,indent=2)+'\n',encoding='utf-8')
reports=[]
for seed in range(2):
    paths=[BASE/f'seed{seed}_joint_ITER02.npz',OUT/f'seed{seed}_STABLE_PROJECTION.npz']
    with np.load(paths[0],allow_pickle=False) as z:old=dict(z)
    with np.load(paths[1],allow_pickle=False) as z:measured=dict(z)
    center=measured['curvature_update_control'];radius=design['radius']
    matrices=[np.vstack([np.eye(53),-np.eye(53)])]
    limits=[np.r_[(old['upper'][:53]-center[:53])/radius,(center[:53]-old['lower'][:53])/radius]]
    native_matrices=[];native_limits=[];cones=[clarabel.NonnegativeConeT(106)]
    library_evidence=[]
    for label in design['libraries']:
        if label.startswith('IID'):
            path=RANDOM/f'seed{seed}_rep{label[-1]}_IID.npz';paths.append(path)
            with np.load(path,allow_pickle=False) as z:b=dict(z)
            J=b['jacobian'];N=b['native_jacobian']
        else:
            index=int(label[-1])
            J=old['original_jacobian'][index*816:(index+1)*816]*np.sqrt(3)
            N=old['original_native_jacobian'][index*9:(index+1)*9]
        rc=measured['curvature_update_control_'+label+'_actual_residual']
        nc=measured['curvature_update_control_'+label+'_actual_native']
        rb=measured[label+'_baseline_residual'];nb=measured[label+'_baseline_native'];anchor=measured[label+'_initial_native']
        keep=np.arange(9)%3!=2
        cap=1.005*np.minimum(nb,anchor)
        native_matrices.append(N[keep,:53]*radius/nb[keep,None])
        native_limits.append((cap[keep]-nc[keep])/nb[keep])
        denominator=np.sqrt(.99*(rb@rb))
        matrices.append(np.vstack([np.zeros(53),-J[:,:53]*radius/denominator]))
        limits.append(np.r_[1.,rc/denominator]);cones.append(clarabel.SecondOrderConeT(817))
        L=measured[label+'_cholesky'];variance=np.square(L).sum(1)
        for ih in range(3):
            sl=slice(ih*272,(ih+1)*272)
            raw_old=L@(rb[sl]*np.sqrt(3));raw_current=L@(rc[sl]*np.sqrt(3))
            raw_J=L@(J[sl,:53]*np.sqrt(3)*radius)
            for block in range(4):
                take=slice(56*block,56*(block+1))
                old_risk=np.mean(raw_old[take]**2/variance[take])
                denom=np.sqrt(56*(1.15*old_risk+2/4096)*variance[take])
                matrices.append(np.vstack([np.zeros(53),-raw_J[take]/denom[:,None]]))
                limits.append(np.r_[1.,raw_current[take]/denom]);cones.append(clarabel.SecondOrderConeT(57))
        library_evidence.append(dict(label=label,center_native_max_anchor_ratio=float((nc[keep]/anchor[keep]).max()),
            center_joint_ratio=float(rc@rc/(rb@rb))))
    matrices.append(np.vstack(native_matrices));limits.append(np.concatenate(native_limits));cones.append(clarabel.NonnegativeConeT(42))
    matrices.append(np.vstack([np.zeros(53),-np.eye(53)]));limits.append(np.r_[1.,np.zeros(53)]);cones.append(clarabel.SecondOrderConeT(54))
    A=np.vstack(matrices);bb=np.concatenate(limits)
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
    settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
    answer=clarabel.DefaultSolver(eye(53,format='csc'),np.zeros(53),csc_matrix(A),bb,cones,settings).solve()
    u=np.array(answer.x);dual=np.array(answer.z);slack=bb-A@u
    cone_sizes=[106]+[size for _ in design['libraries'] for size in [817]+[57]*12]+[42,54]
    margins=[];dual_margins=[];offset=0
    for i,size in enumerate(cone_sizes):
        ss=slack[offset:offset+size];zz=dual[offset:offset+size];offset+=size
        nonnegative=i in (0,len(cone_sizes)-2)
        margins.append(float(ss.min()) if nonnegative else float(ss[0]-np.linalg.norm(ss[1:])))
        dual_margins.append(float(zz.min()) if nonnegative else float(zz[0]-np.linalg.norm(zz[1:])))
    assert offset==len(bb)
    stationarity=float(np.max(np.abs(u+A.T@dual)));complementarity=float(abs(slack@dual))
    status=str(answer.status)
    kkt=status=='Solved' and min(margins)>=-1e-8 and min(dual_margins)>=-1e-8 and stationarity<1e-8 and complementarity<1e-8
    record=dict(seed=seed,solver_status=status,iterations=answer.iterations,KKT_pass=kkt,
        minimum_primal_margin=min(margins),minimum_dual_margin=min(dual_margins),stationarity=stationarity,complementarity=complementarity,
        correction_norm=float(np.linalg.norm(u)*radius),radius=radius,rows=len(bb),
        libraries=library_evidence,source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    payload_path=PAYLOAD/f'seed{seed}_RECENTERED_RESTORATION_SCREEN.npz'
    np.savez(payload_path,A=A,b=bb,u=u,dual=dual,cone_sizes=cone_sizes,center=center,
        stale_derivative_screen_only=np.array(True),native_linear_A=np.vstack(native_matrices),native_linear_b=np.concatenate(native_limits))
    record['evidence_NAS_path']=str(payload_path)
    record['evidence_sha256']=hashlib.sha256(payload_path.read_bytes()).hexdigest()
    if kkt:
        record['next_requirement']='Fresh full derivatives at this measured center before any proposed correction is executed.'
    else:
        record['next_requirement']='Do not dispatch a fresh-derivative restoration experiment based on this screen.'
    reports.append(record)
result=dict(status='SAVED_RECENTERED_RESTORATION_SCREEN_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),
    two_seed_screen_feasible=all(r['KKT_pass'] for r in reports),reports=reports,generator_calls=0,parameter_updates=0,
    scientific_acceptance=False,evidence_limit='Old-center derivatives with new measured intercepts only. KKT proves this local linearized screen, not the fresh-center derivative model or physical feasibility. No native candidate executed.')
(HERE/'RECENTERED_RESTORATION_SCREEN.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
