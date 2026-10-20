from pathlib import Path
import ast
here=Path(__file__).resolve().parent;old=here.parent/'native_curvature_diagnostic_01/AUDIT_COMPLETE.py'
s=old.read_text('utf-8').replace("PREVIOUS=HERE.parent/'original_precision_REAL_pilot_01'","PREVIOUS=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_curvature_diagnostic_01'")
s=s.replace('joint_response_01_native_curvature_diagnostic_20261020','joint_response_01_native_gain_diagnostic_20261020').replace('==828','==837').replace('source_files_verified=828','source_files_verified=837')
s=s.replace("PREVIOUS/'SAVED_CONSTRAINT_PROPOSALS.npz'","CURVE/'SAVED_NATIVE_GAIN_PROPOSALS.npz'")
s=s.replace("OUT/f'seed{seed}_CURVATURE", "OUT/f'seed{seed}_NATIVE_GAIN")
s=s.replace("report=json.loads((OUT/f'seed{seed}_NATIVE_GAIN.json').read_bytes())", "with np.load(CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',allow_pickle=False) as z:control={k:z[k] for k in z.files}\n report=json.loads((OUT/f'seed{seed}_NATIVE_GAIN.json').read_bytes())")
s=s.replace("np.testing.assert_array_equal(a['linear_control'],old['trial0_proposal']);np.testing.assert_array_equal(a['curvature_anchor'],proposals[f'seed{seed}_joint_ITER02_curvature1_anchor1'])", "np.testing.assert_array_equal(a['curvature_control'],control['curvature_anchor']);np.testing.assert_array_equal(a['native_gain'],proposals[f'seed{seed}_rankone0'])")
start=s.index(' # Reconstruct the complete convex local problem');end=s.index(' risk_records=[]',start)
block=''' # Reconstruct all macro, per-FIT joint and common-native constraints independently.
 r=old['stacked_baseline_residual'];n=old['stacked_baseline_native'];J=old['original_jacobian'];N=old['original_native_jacobian'];radius=.0125
 H=np.maximum((old['FD_plus_native']+old['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 upper=np.minimum(n,np.concatenate([old[l+'_initial_native'] for l in fits])*np.tile([1.01,1.01,1.10],9))
 operators=[];centers=[]
 for il,label in enumerate(fits):
  L=old[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
  for ih in range(3):
   sl=slice((il*3+ih)*272,(il*3+ih+1)*272);center=L@(3*r[sl])/scale;operator=(L@(3*J[sl]*radius))/scale[:,None]
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));scale_block=np.sqrt(56*(1.15*np.mean(center[take]**2)+2/4096));centers.append(center[take]/scale_block);operators.append(operator[take]/scale_block)
 aa=np.stack(operators);bb=np.stack(centers);JR=(J*radius).reshape(3,816,54);rr=r.reshape(3,816);den=np.square(rr).sum(1)
 D=N*radius/n[:,None];C=H*radius**2/n[:,None];b=upper/n-1;take=np.array([i for i in range(27) if i%3!=2])
 def cons(x):
  u=x[:-1];native=D@u+.5*C@np.square(u)
  return np.r_[1-u@u,1-np.square(aa@u+bb).sum(1),.99-np.square(rr+JR@u).sum(1)/den,b-native,-native[take]-x[-1]]
 def cjac(x):
  u=x[:-1];native=D+C*u
  rows=np.vstack((-2*u,-2*np.einsum('kij,ki->kj',aa,aa@u+bb),-2*np.einsum('kij,ki->kj',JR,rr+JR@u)/den[:,None],-native,-native[take]))
  return np.column_stack((rows,np.r_[np.zeros(len(rows)-len(take)),-np.ones(len(take))]))
 x0=np.r_[(control['curvature_anchor']-old['theta'])/radius,-.1]
 solve=minimize(lambda x:-x[-1],x0,jac=lambda x:np.r_[np.zeros(54),-1.],method='SLSQP',bounds=[*zip((old['lower']-old['theta'])/radius,(old['upper']-old['theta'])/radius),(None,None)],constraints={'type':'ineq','fun':cons,'jac':cjac},options={'maxiter':200,'ftol':1e-9,'disp':False})
 assert solve.success,solve.message
 step=a['native_gain']-a['theta'];error=float(np.max(np.abs(solve.x[:-1]*radius-step)));assert error<2e-7
 slack=float(cons(np.r_[step/radius,solve.x[-1]]).min());assert slack>=-1e-6 and abs(step[-1])<1e-15
 solves[str(seed)]=dict(maximum_parameter_difference=error,minimum_constraint_slack=slack,maximum_common_native_reduction=float(solve.x[-1]),predicted_joint_remaining=float(np.sum((r+J@step)**2)/np.sum(r*r)),predicted_native_ratios=((n+N@step+.5*H@np.square(step))/n).reshape(3,3,3).tolist())
 counts['independent_native_gain_solves']+=1
'''
s=s[:start]+block+s[end:]
s=s.replace("name=='linear_control'","name=='curvature_control'").replace("old['trial0_'+label+'_actual_'+key]","control['curvature_anchor_'+label+'_actual_'+key]")
s=s.replace("v['all_pass']['curvature_anchor']","v['all_pass']['native_gain']").replace('two_seed_curvature_all_local_pass','two_seed_native_gain_all_local_pass').replace('PASS_SAVED_NATIVE_CURVATURE_DIAGNOSTIC_AUDIT','PASS_SAVED_NATIVE_GAIN_DIAGNOSTIC_AUDIT')
s=s.replace('linear candidates','curvature control candidates')
ast.parse(s);(here/'AUDIT_COMPLETE.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared complete fixed native-gain audit')
