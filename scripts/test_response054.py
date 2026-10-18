"""Analytic constrained optima and nonlinear prediction errors; no dataset or CP."""
import argparse,copy
from pathlib import Path
import numpy as np
from n01.data import write_json
from run_response054 import constraint_vector054,solve054
from run_response039 import HYPOTHESES,PROTECTED,MORPH,objective039

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--device',default='cpu');args=parser.parse_args()
contract=dict(fit_primary_relative_worsening_max=.01,protected_allowance_T=.05,morphology_individual_allowance_T=.05)
T={h:dict.fromkeys(PROTECTED+MORPH,1.) for h in HYPOTHESES}
ref={h:dict(values=dict.fromkeys(('wave','joint_morphology')+PROTECTED+MORPH+('negative_lobe_area_Vns',),10.),
    fake_flags=dict(undefined=dict(count=0,denominator=17)),observation_flags=dict(ADC=dict(count=0,denominator=17)),
    conditional_coverage=dict(conditional_coverage_complete=True)) for h in HYPOTHESES}
base,upper,labels=constraint_vector054(ref,ref,T,contract)
assert np.array_equal(base,np.zeros(len(base))) and (base<=upper).all()
violation_checks=0
for h in HYPOTHESES:
    for key in ('wave','joint_morphology')+PROTECTED+MORPH+('negative_lobe_area_Vns',):
        bad=copy.deepcopy(ref);bad[h]['values'][key]+=.11 if key in ('wave','joint_morphology') else .051 if key in PROTECTED+MORPH else .001
        v,u,_=constraint_vector054(bad,ref,T,contract)
        assert not objective039(bad,ref,T,contract)['feasible'] and int((v>u).sum())==1
        violation_checks+=1
    for family in ('fake_flags','observation_flags'):
        bad=copy.deepcopy(ref);bad[h][family][next(iter(bad[h][family]))]['count']=1
        v,u,_=constraint_vector054(bad,ref,T,contract)
        assert not objective039(bad,ref,T,contract)['feasible'] and int((v>u).sum())==1
        violation_checks+=1
    bad=copy.deepcopy(ref);bad[h]['conditional_coverage']['conditional_coverage_complete']=False
    v,u,_=constraint_vector054(bad,ref,T,contract)
    assert not objective039(bad,ref,T,contract)['feasible'] and int((v>u).sum())==1
    violation_checks+=1

# Four independent upper bounds make the analytic optimum unique.
G=np.stack((np.vstack((-np.arange(1,5),np.eye(4))),np.vstack((-np.arange(1,5)/2,np.eye(4)))))
b=np.zeros((2,5));u=np.tile([10.,.2,.3,.4,.5],(2,1));probes=np.stack((-G.transpose(0,2,1),G.transpose(0,2,1)),axis=2)
sol=solve054(b,probes,u,[0]);known=np.array([.2,.3,.4,.5])
assert np.allclose(sol['normalized_coefficients'],known,rtol=0,atol=1e-12)
assert abs(sol['predicted_common_score_gain']-2.)<1e-12
assert np.max(np.abs(sol['central_second_difference']))==0
perm=[4,2,0,3,1];reordered=solve054(b[:,perm],probes[:,:,:,perm],u[:,perm],[2])
assert np.allclose(reordered['normalized_coefficients'],known,rtol=0,atol=1e-12)
assert abs(reordered['predicted_common_score_gain']-2.)<1e-12
blocked=probes.copy();blocked[:,:,:,0]=0
zero=solve054(b,blocked,u,[0]);assert zero['predicted_common_score_gain']==0

# Even quadratic terms leave the central slope unchanged, but cause a measured
# endpoint discrepancy that must not be described as predicted physical gain.
curved=probes.copy();curved[:,:,:,0]+=np.array([.1,.2,.3,.4])[None,:,None]
nonlinear=solve054(b,curved,u,[0]);assert np.allclose(nonlinear['normalized_coefficients'],known,atol=1e-12,rtol=0)
assert np.allclose(nonlinear['central_second_difference'][:,0],[[.2,.4,.6,.8]]*2,atol=1e-12,rtol=0)
actual=np.einsum('smj,j->sm',G,known);actual[:,0]+=np.dot([.1,.2,.3,.4],known**2)
residual=actual-nonlinear['predicted_scaled_endpoint']
assert np.allclose(residual[:,0],.17,atol=1e-12,rtol=0)
write_json(args.out,dict(status='PASS',solver_device='cpu',requested_operator_device=args.device,
    checkpoint_loads=0,dataset_loads=0,AD_queries=0,original039_constraint_violation_checks=violation_checks,
    analytic_optimum_coefficients=sol['normalized_coefficients'],analytic_common_gain=sol['predicted_common_score_gain'],
    active_protected_constraints_exact=True,row_permutation_invariant=True,zero_common_gain_detected=True,
    known_nonlinear_prediction_residual=residual[:,0],curvature_record_correct=True,
    primal_violation=sol['maximum_primal_violation'],stationarity_residual=sol['maximum_stationarity_residual'],
    no_physical_acceptance_inferred=True))
print('response054 analytic LP, original guard encoding, and nonlinear prediction audit: PASS')
