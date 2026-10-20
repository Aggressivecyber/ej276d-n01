from pathlib import Path
import ast

here=Path(__file__).resolve().parent
code=(here.parent/'curvature_update_diagnostic_01/AUDIT_COMPLETE.py').read_text('utf-8')
code=code.replace("CURVE=HERE.parent/'native_curvature_diagnostic_01';CERT=HERE.parent/'scientific_progress_diagnostic_01'","CURVE=HERE.parent/'curvature_update_diagnostic_01';CERT=CURVE")
code=code.replace('curvature_update_diagnostic_20261020','second_curvature_diagnostic_20261020').replace('==858','==869').replace('verified=858','verified=869')
code=code.replace('CERTIFIED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz','CERTIFIED_SECOND_CURVATURE_PROPOSALS.npz').replace('seed{seed}_REDUCED_CONIC_SOLVER_EVIDENCE.npz','seed{seed}_SECOND_CURVATURE_SOLVER_EVIDENCE.npz')
code=code.replace("OUT/f'seed{seed}_CURVATURE_UPDATE.npz'","OUT/f'seed{seed}_SECOND_CURVATURE.npz'").replace("OUT/f'seed{seed}_CURVATURE_UPDATE.json'","OUT/f'seed{seed}_SECOND_CURVATURE.json'").replace("CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz'","CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE_UPDATE.npz'")
code=code.replace("a['curvature_control'],control['curvature_anchor']","a['curvature_update_control'],control['curvature_update']").replace("a['curvature_update']","a['second_curvature']").replace("proposals[f'seed{seed}_curvature_update']","proposals[f'seed{seed}_second_curvature']")
start=code.index(" d=control['curvature_anchor']");end=code.index(" anchor=np.concatenate",start)
code=code[:start]+''' with np.load(HERE.parent/f'native_curvature_diagnostic_01/terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',allow_pickle=False) as z:first={k:z[k] for k in z.files}
 terms=[]
 for measured,name in [(first,'curvature_anchor'),(control,'curvature_update')]:
  d=measured[name]-old['theta'];actual=np.concatenate([measured[name+'_'+l+'_actual_native'] for l in fits]);prediction=n+N@d+.5*H@np.square(d)
  for direction,extra in terms:prediction+=extra*(direction@d)**2
  extra=np.maximum(actual-prediction,0);direction=d/(d@d);terms.append((direction,extra))
 assert (N[2::3,:53]==0).all() and (H[2::3,:53]==0).all() and all((e[2::3]==0).all() for d,e in terms) and N[2::3,53].min()<0<N[2::3,53].max()
 pn=n+N@delta+.5*H@np.square(delta)
 for direction,extra in terms:pn+=extra*(direction@delta)**2
'''+code[end:]
code=code.replace("B=np.vstack([np.diag(np.sqrt(.5*H[k,:53]*radius**2/n[k])),np.sqrt(extra[k]/n[k])*d[:53]*radius/(d@d)])","B=np.vstack([np.diag(np.sqrt(.5*H[k,:53]*radius**2/n[k])),*[np.sqrt(extra[k]/n[k])*direction[:53]*radius for direction,extra in terms]])")
code=code.replace('np.zeros(54),c-1','np.zeros(55),c-1').replace('len(slack)==3220','len(slack)==3238').replace('[56]*18','[57]*18')
code=code.replace("CERT/'CONIC_CANDIDATE_CERTIFICATE.json'","CERT/'SECOND_CURVATURE_DIAGNOSIS.json'").replace("read_bytes())['entries'][seed]","read_bytes())['reports'][seed]")
code=code.replace("name=='curvature_control'","name=='curvature_update_control'").replace("control['curvature_anchor_'+label","control['curvature_update_'+label").replace("['all_pass']['curvature_update']","['all_pass']['second_curvature']").replace('two_seed_curvature_update_all_local_pass','two_seed_second_curvature_all_local_pass').replace('PASS_SAVED_CURVATURE_UPDATE_DIAGNOSTIC_AUDIT','PASS_SAVED_SECOND_CURVATURE_DIAGNOSTIC_AUDIT').replace('original curvature control candidates','previous once-corrected curvature control candidates')
ast.parse(code);(here/'AUDIT_COMPLETE.py').write_text(code,encoding='utf-8',newline='\n')
print('Second-curvature independent matrix, primal/dual certificate and saved-vector audit prepared.')
