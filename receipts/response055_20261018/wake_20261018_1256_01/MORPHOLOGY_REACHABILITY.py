from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
from scipy.optimize import linprog
w=Path(__file__).parent;r=w.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=r/'receipts/response054_20261018/wake_20261018_1233_01/collected/run_01/LINEARIZATION_AND_SOLUTION.json';s=read(src)
J=np.array(s['jacobian']);b=np.array(s['scaled_base']);upper=np.array(s['scaled_upper']);labels=s['constraint_row_labels_by_seed']
for seed in (0,1):
 for i,label in enumerate(labels[seed]):
  if label['family']=='values' and label['metric'] in ('wave','joint_morphology'):upper[seed,i]=-.02
A=np.column_stack((J.reshape(-1,4),np.zeros(J.shape[0]*J.shape[1])));rhs=(upper-b).ravel();objective_labels=[]
for seed in (0,1):
 for h in ('fast','slow','common_level'):
  indices=[i for i,z in enumerate(labels[seed]) if z['hypothesis']==h and z['family']=='values' and z['metric'] in ('peak_voltage_V','rise10_90_ns','FWHM_ns')]
  assert len(indices)==3
  A=np.vstack((A,np.r_[J[seed,indices].mean(0),1.]));rhs=np.r_[rhs,-b[seed,indices].mean()];objective_labels.append(dict(seed=seed,hypothesis=h,row_indices=indices))
c=np.array([0.,0.,0.,0.,-1.]);z=linprog(c,A_ub=A,b_ub=rhs,bounds=[(-1,1)]*4+[(0,None)],method='highs',options=dict(time_limit=10.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
if not z.success:raise RuntimeError('Stored-model reachability solve failed: '+str(z.status)+' '+z.message)
x=z.x;y=z.ineqlin.marginals;lo=z.lower.marginals;hi=z.upper.marginals;slack=rhs-A@x
primal=max(0.,float(-slack.min()),float(np.max(np.abs(x[:4])-1)),-x[4]);stationarity=float(np.max(np.abs(c-A.T@y-lo-hi)))
complementarity=max(float(np.max(np.abs(slack*y))),float(np.max(np.abs(lo*np.r_[x[:4]+1,x[4]]))),float(np.max(np.abs(hi[:4]*(1-x[:4])))))
dual=float(rhs@y-lo[:4].sum()+hi[:4].sum());gap=float(c@x-dual)
assert y.max()<=1e-9 and lo.min()>=-1e-9 and hi.max()<=1e-9 and hi[4]==0 and max(primal,stationarity,complementarity,abs(gap))<1e-8
pred=b+np.einsum('smj,j->sm',J,x[:4]);target=[]
for o in objective_labels:target.append(dict(**o,predicted_mean_morphology_gain_T=float(-pred[o['seed'],o['row_indices']].mean())))
active=[]
for i in range(J.shape[0]*J.shape[1]):
 if abs(slack[i])<1e-9 and abs(y[i])>1e-10:active.append(dict(seed=i//J.shape[1],**labels[i//J.shape[1]][i%J.shape[1]],dual_marginal=float(y[i])))
audit=dict(UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sha(src),plan_sha256=sha(w/'MORPHOLOGY_REACHABILITY_PLAN.md'),
 status='VERIFIED_STORED_LINEAR_MODEL_OPTIMUM',scope='Posthoc existing FIT scalar diagnostic; no new scientific population or actual endpoint evaluation',
 physical_parameter_count=4,normalized_box=[-1,1],common_morphology_gain_upper_bound_T=float(x[4]),target_T=1.,target_reachable_in_this_linear_model=bool(x[4]>=1.),
 x=x.tolist(),predicted_by_head_H=target,active_weighted_guards=active,
 KKT=dict(primal_residual=primal,stationarity_residual=stationarity,complementarity_residual=complementarity,duality_gap=gap),
 A_ub=A.tolist(),b_ub=rhs.tolist(),c=c.tolist(),inequality_marginals=y.tolist(),lower_marginals=lo.tolist(),upper_marginals=hi.tolist(),slack=slack.tolist(),
 LP_solves=1,checkpoint_loads=0,TRAIN_decodes=0,new_random_banks=0,new_native_evaluations=0,confirmation_used_to_fit=False,
 nonlinear_global_impossibility_claim=False,scientific_goal_achieved=False)
(w/'MORPHOLOGY_REACHABILITY.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ('status','common_morphology_gain_upper_bound_T','target_reachable_in_this_linear_model','x','predicted_by_head_H','active_weighted_guards','KKT')},indent=2))
