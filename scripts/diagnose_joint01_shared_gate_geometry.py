"""Two frozen-matrix convex progress diagnostics; no waveform generation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform, time
import numpy as np
import scipy
from scipy.optimize import minimize

args=argparse.ArgumentParser()
args.add_argument('--config',required=True)
args.add_argument('--config-sha256',required=True)
args.add_argument('--out',required=True)
args=args.parse_args()
root=Path(__file__).resolve().parents[1]; config_path=Path(args.config)
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==args.config_sha256
p=json.loads(config_path.read_bytes());out=Path(args.out);out.mkdir()
for name,digest in p['input_pins'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
started=time.monotonic(); counts=dict(direction_solve_attempts=0,direction_solves=0,constraint_gradient_probe_evaluations=0,model_calls=0,parameter_updates=0)
results={}
for case in p['cases']:
    assert time.monotonic()-started<p['resources']['internal_seconds']
    source=root/case['arrays'];report=json.loads((root/case['report']).read_bytes())
    with np.load(source,allow_pickle=False) as z: saved={k:z[k].copy() for k in z.files}
    prefix='trial1_';assert report['selected_trial']==1 and report['trials'][1]['accepted'] and report['iteration']==0
    theta=saved['theta'];assert np.array_equal(theta,np.zeros(54))
    radius=report['trials'][1]['radius'];assert radius==p['radius']==.00625
    r=saved['stacked_baseline_residual'];j=saved[prefix+'jacobian']*radius
    n=saved['stacked_baseline_native'];native=saved[prefix+'native_jacobian']*radius/n[:,None]
    joint_A=[];joint_b=[];peak_A=[];peak_b=[];macro_A=[];macro_b=[];native_goal=[]
    for bank,label in enumerate(['FIT0','FIT1','FIT2']):
        L=saved[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
        for h in range(3):
            sl=slice((bank*3+h)*272,(bank*3+h+1)*272)
            rr,jj=r[sl],j[sl];norm=np.linalg.norm(rr)
            joint_b.append(rr/norm);joint_A.append(jj/norm)
            raw_r=L@(rr*3)/scale;raw_j=(L@(jj*3))/scale[:,None]
            peak=slice(112,168);pnorm=np.linalg.norm(raw_r[peak])
            peak_b.append(raw_r[peak]/pnorm);peak_A.append(raw_j[peak]/pnorm)
            for block in range(4):
                take=slice(block*56,(block+1)*56)
                cap=1.15*np.mean(raw_r[take]**2)+2/4096
                macro_b.append(raw_r[take]/np.sqrt(56*cap));macro_A.append(raw_j[take]/np.sqrt(56*cap))
            native_goal.extend([p['target_improvement']['native'],p['target_improvement']['native'],0.])
    joint_A,joint_b,peak_A,peak_b,macro_A,macro_b,native_goal=map(np.asarray,[joint_A,joint_b,peak_A,peak_b,macro_A,macro_b,native_goal])
    def constraints(x):
        u,g=x[:-1],x[-1]
        return np.r_[1-p['target_improvement']['joint']*g-np.square(joint_b+joint_A@u).sum(1),
                     1-p['target_improvement']['peak']*g-np.square(peak_b+peak_A@u).sum(1),
                     -native@u-native_goal*g,1-np.square(macro_b+macro_A@u).sum(1),1-u@u]
    def constraint_jacobian(x):
        u=x[:-1]
        return np.vstack((np.c_[-2*np.einsum('mni,mn->mi',joint_A,joint_b+joint_A@u),np.full(9,-p['target_improvement']['joint'])],
                          np.c_[-2*np.einsum('mni,mn->mi',peak_A,peak_b+peak_A@u),np.full(9,-p['target_improvement']['peak'])],
                          np.c_[-native,-native_goal],np.c_[-2*np.einsum('mni,mn->mi',macro_A,macro_b+macro_A@u),np.zeros(36)],np.r_[-2*u,0.][None]))
    x0=np.zeros(55);analytic=constraint_jacobian(x0);fd=[]
    for column in range(55):
        step=np.zeros(55);step[column]=1e-6
        fd.append((constraints(x0+step)-constraints(x0-step))/(2e-6));counts['constraint_gradient_probe_evaluations']+=2
    fd=np.stack(fd,1);np.testing.assert_allclose(fd,analytic,rtol=1e-6,atol=1e-8)
    assert constraints(x0).min()>=-1e-12
    bounds=list(zip((saved['lower']-theta)/radius,(saved['upper']-theta)/radius))+[(0.,1.)]
    counts['direction_solve_attempts']+=1
    solution=minimize(lambda x:-x[-1],x0,jac=lambda x:np.r_[np.zeros(54),-1.],method='SLSQP',bounds=bounds,
        constraints={'type':'ineq','fun':constraints,'jac':constraint_jacobian},options=p['solver'])
    slack=constraints(solution.x);proposal=theta+solution.x[:-1]*radius
    multipliers=solution.multipliers
    stationarity=np.r_[np.zeros(54),-1.]-constraint_jacobian(solution.x).T@multipliers
    bounds_array=np.asarray(bounds);at_lower=np.abs(solution.x-bounds_array[:,0])<=1e-8;at_upper=np.abs(solution.x-bounds_array[:,1])<=1e-8
    violations=np.abs(stationarity);violations[at_lower]=np.maximum(-stationarity[at_lower],0.);violations[at_upper]=np.maximum(stationarity[at_upper],0.)
    arrays=dict(theta=theta,lower=saved['lower'],upper=saved['upper'],unit_solution=solution.x,proposal=proposal,
        baseline_residual=r,jacobian=j/radius,baseline_native=n,native_jacobian=native*n[:,None]/radius,
        joint_A=joint_A,joint_b=joint_b,peak_A=peak_A,peak_b=peak_b,macro_A=macro_A,macro_b=macro_b,native_operator=native,native_goal=native_goal,
        slack=slack,constraint_jacobian_at_zero=analytic,finite_difference_constraint_jacobian_at_zero=fd,
        multipliers=multipliers,constraint_jacobian_at_solution=constraint_jacobian(solution.x),stationarity=stationarity,bounds=bounds_array)
    np.savez(out/(case['name']+'_GEOMETRY.npz'),**arrays)
    result=dict(seed=case['seed'],solver_success=bool(solution.success),solver_status=int(solution.status),solver_message=str(solution.message),iterations=int(solution.nit),objective_evaluations=int(solution.nfev),
        common_original_gate_progress=float(solution.x[-1]),minimum_slack=float(slack.min()),unit_step_norm=float(np.linalg.norm(solution.x[:-1])),parameter_step_norm=float(np.linalg.norm(proposal-theta)),constraint_gradient_max_error=float(np.abs(fd-analytic).max()),
        KKT_stationarity_max_violation=float(violations.max()),minimum_multiplier=float(multipliers.min()),maximum_complementarity_residual=float(np.abs(multipliers*slack).max()),
        predicted_joint_remaining=np.square(joint_b+joint_A@solution.x[:-1]).sum(1).tolist(),predicted_peak_remaining=np.square(peak_b+peak_A@solution.x[:-1]).sum(1).tolist(),predicted_native_remaining=(1+native@solution.x[:-1]).reshape(3,3,3).tolist(),
        enough_predicted_progress_for_native_confirmation=bool(solution.success and slack.min()>=-1e-6 and solution.x[-1]>=p['minimum_progress_for_native_confirmation']))
    (out/(case['name']+'_GEOMETRY.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not solution.success:raise RuntimeError('Frozen geometry solver failed: '+str(solution.message))
    if slack.min() < -1e-6:raise RuntimeError('Frozen geometry solution violates constraints')
    counts['direction_solves']+=1;results[case['name']]=result
    (out/'PROGRESS.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),execution_counts=counts,completed_cases=list(results)),indent=2)+'\n',encoding='utf-8')
assert counts['direction_solves']==2 and counts['constraint_gradient_probe_evaluations']==220
result=dict(status='COMPLETE',cases=results,execution_counts=counts,elapsed_seconds=time.monotonic()-started,config_sha256=args.config_sha256,
    runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
    native_confirmation_supported=all(x['enough_predicted_progress_for_native_confirmation'] for x in results.values()),scientific_acceptance=False,model_calls=0,parameter_updates=0,
    scope='Frozen FIT matrix geometry at the two original M0 first-iteration centers, using the same accepted-trial secant matrices and radius. No new physical waveforms, endpoint updates, CHECK/FRESH/FINAL selection, or response-match claim.')
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',result_sha256=hashlib.sha256((out/'RESULT.json').read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
