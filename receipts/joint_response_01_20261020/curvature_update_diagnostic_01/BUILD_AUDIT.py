from pathlib import Path
import ast

here=Path(__file__).resolve().parent
code=(here.parent/'scientific_progress_diagnostic_01/AUDIT_COMPLETE.py').read_text('utf-8')
code=code.replace("CURVE=HERE.parent/'native_gain_diagnostic_01'","CURVE=HERE.parent/'native_curvature_diagnostic_01';CERT=HERE.parent/'scientific_progress_diagnostic_01'")
code=code.replace('scientific_progress_diagnostic_20261020','curvature_update_diagnostic_20261020').replace('==846','==858').replace('verified=846','verified=858')
code=code.replace("CURVE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz'","CERT/'CERTIFIED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz'")
start=code.index('reference_path=');end=code.index('for seed in range(2):',start)
code=code[:start]+code[end:]
code=code.replace('SCIENTIFIC_PROGRESS.npz','CURVATURE_UPDATE.npz').replace('SCIENTIFIC_PROGRESS.json','CURVATURE_UPDATE.json').replace('NATIVE_GAIN.npz','CURVATURE.npz')
code=code.replace("a['native_gain_control'],control['native_gain']","a['curvature_control'],control['curvature_anchor']")
code=code.replace('scientific_progress','curvature_update').replace("CERT=HERE.parent/'curvature_update_diagnostic_01'","CERT=HERE.parent/'scientific_progress_diagnostic_01'").replace("name=='native_gain_control'","name=='curvature_control'").replace("control['native_gain_'+label","control['curvature_anchor_'+label")
start=code.index(' # Replay the fixed reference solve');end=code.index(' risk_records=[]',start)
independent=''' # Independently check the original convex problem and its saved primal/dual certificate.
 with np.load(CERT/f'seed{seed}_REDUCED_CONIC_SOLVER_EVIDENCE.npz',allow_pickle=False) as z:e={k:z[k] for k in z.files}
 r=old['stacked_baseline_residual'];n=old['stacked_baseline_native'];J=old['original_jacobian'];N=old['original_native_jacobian'];delta=a['curvature_update']-a['theta'];radius=.0125;u=delta/radius
 np.testing.assert_allclose(u,e['unit_step'],rtol=1e-11,atol=1e-11)
 assert abs(u[-1])<1e-14
 H=np.maximum((old['FD_plus_native']+old['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 d=control['curvature_anchor']-old['theta'];actual=np.concatenate([control['curvature_anchor_'+l+'_actual_native'] for l in fits]);extra=np.maximum(actual-(n+N@d+.5*H@np.square(d)),0)
 assert (N[2::3,:53]==0).all() and (H[2::3,:53]==0).all() and (extra[2::3]==0).all() and N[2::3,53].min()<0<N[2::3,53].max()
 pn=n+N@delta+.5*H@np.square(delta)+extra*(d@delta/(d@d))**2
 anchor=np.concatenate([old[l+'_initial_native'] for l in fits]);cap=np.minimum(n,anchor*np.tile([1.01,1.01,1.10],9))
 native_slack=(cap-pn)/n;assert native_slack.min()>=-1e-6
 assert np.linalg.norm(delta)<=radius*(1+1e-6) and (a['curvature_update']>=old['lower']-1e-12).all() and (a['curvature_update']<=old['upper']+1e-12).all()
 predicted=r+J@delta;macro_slacks=[]
 expected_A=[np.vstack([np.eye(53),-np.eye(53)])];expected_b=[np.r_[(old['upper'][:53]-old['theta'][:53])/radius,(old['theta'][:53]-old['lower'][:53])/radius]]
 for il,label in enumerate(fits):
  L=old[label+'_cholesky'];variance=np.square(L).sum(1)
  for ih in range(3):
   sl=slice((3*il+ih)*272,(3*il+ih+1)*272);rb=L@(r[sl]*3);rc=L@(predicted[sl]*3);raw_derivative=L@(J[sl,:53]*3*radius)
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));b=np.mean(rb[take]**2/variance[take]);v=np.mean(rc[take]**2/variance[take]);limit=1.15*b+2/4096;macro_slacks.append(1-v/limit)
    divisor=np.sqrt(56*limit*variance[take]);expected_A.append(np.vstack([np.zeros(53),-raw_derivative[take]/divisor[:,None]]));expected_b.append(np.r_[1.,rb[take]/divisor])
 assert len(macro_slacks)==36 and min(macro_slacks)>=-1e-6
 for k in range(27):
  if k%3==2:continue
  linear=N[k,:53]*radius/n[k];B=np.vstack([np.diag(np.sqrt(.5*H[k,:53]*radius**2/n[k])),np.sqrt(extra[k]/n[k])*d[:53]*radius/(d@d)])
  c=(cap[k]-n[k])/n[k];expected_A.append(np.vstack([linear,-2*B,linear]));expected_b.append(np.r_[c+1,np.zeros(54),c-1])
 expected_A.append(np.vstack([np.zeros(53),-np.eye(53)]));expected_b.append(np.r_[1.,np.zeros(53)])
 np.testing.assert_allclose(e['conic_A'],np.vstack(expected_A),rtol=1e-11,atol=1e-12);np.testing.assert_allclose(e['conic_b'],np.concatenate(expected_b),rtol=1e-11,atol=1e-12)
 # Verify the quadratic objective from the full 54-coordinate definition, then cone feasibility.
 gram=(J*radius).T@(J*radius);P=(gram+.02*np.trace(gram)/54*np.eye(54))/(r@r);q=(J*radius).T@r/(r@r)
 np.testing.assert_allclose(e['P'],P[:53,:53],rtol=1e-12,atol=1e-14);np.testing.assert_allclose(e['q'],q[:53],rtol=1e-12,atol=1e-14)
 slack=e['conic_b']-e['conic_A']@u[:53];dual=e['z'];assert len(slack)==3220
 cone_margins=[];dual_margins=[];offset=0
 for index,size in enumerate([106]+[57]*36+[56]*18+[54]):
  ss=slack[offset:offset+size];zz=dual[offset:offset+size]
  cone_margins.append(float(ss.min()) if index==0 else float(ss[0]-np.linalg.norm(ss[1:])))
  dual_margins.append(float(zz.min()) if index==0 else float(zz[0]-np.linalg.norm(zz[1:])))
  offset+=size
 assert offset==len(slack) and min(cone_margins)>=-1e-8 and min(dual_margins)>=-1e-8
 stationarity=float(np.max(np.abs(P[:53,:53]@u[:53]+q[:53]+e['conic_A'].T@dual)));complementarity=float(abs(slack@dual))
 assert stationarity<=1e-8 and complementarity<=1e-8
 objective=float(.5*u[:53]@P[:53,:53]@u[:53]+q[:53]@u[:53]);cert=json.loads((CERT/'CONIC_CANDIDATE_CERTIFICATE.json').read_bytes())['entries'][seed]
 assert abs(objective-cert['objective'])<1e-10
 solves[str(seed)]=dict(original_solver_status=cert['original_solver_status'],independent_macro_constraints=36,independent_native_constraints=27,minimum_native_slack=float(native_slack.min()),minimum_macro_slack=min(macro_slacks),minimum_primal_cone_slack=min(cone_margins),minimum_dual_cone_slack=min(dual_margins),stationarity=stationarity,complementarity=complementarity,objective=objective,predicted_joint_ratio=float(predicted@predicted/(r@r)))
 counts['independent_conic_KKT_checks']+=1;counts['independent_complete_constraint_checks']+=1
'''
code=code[:start]+independent+code[end:]
code=code.replace('PASS_SAVED_SCIENTIFIC_PROGRESS_DIAGNOSTIC_AUDIT','PASS_SAVED_CURVATURE_UPDATE_DIAGNOSTIC_AUDIT')
code=code.replace('candidate reference solves replayed and constraints plus gate summaries independently audited','fixed candidate original convex constraints and primal/dual certificates plus gate summaries independently audited').replace('native-gain control candidates','original curvature control candidates')
ast.parse(code);(here/'AUDIT_COMPLETE.py').write_text(code,encoding='utf-8',newline='\n')
print('Audit source constructed; no scientific generation or solver execution.')
