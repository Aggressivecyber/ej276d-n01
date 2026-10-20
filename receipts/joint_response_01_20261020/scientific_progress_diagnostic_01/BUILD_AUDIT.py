from pathlib import Path
import ast
here=Path(__file__).resolve().parent;s=(here.parent/'native_gain_diagnostic_01/AUDIT_COMPLETE.py').read_text('utf-8')
s=s.replace("CURVE=HERE.parent/'native_curvature_diagnostic_01'","CURVE=HERE.parent/'native_gain_diagnostic_01'")
s=s.replace('joint_response_01_native_gain_diagnostic_20261020','joint_response_01_scientific_progress_diagnostic_20261020').replace('==837','==846').replace('source_files_verified=837','source_files_verified=846')
s=s.replace("CURVE/'SAVED_NATIVE_GAIN_PROPOSALS.npz'","CURVE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz'").replace("OUT/f'seed{seed}_NATIVE_GAIN","OUT/f'seed{seed}_SCIENTIFIC_PROGRESS")
s=s.replace("CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz'","CURVE/f'terminal_01/collected/run_01/seed{seed}_NATIVE_GAIN.npz'")
s=s.replace("a['curvature_control'],control['curvature_anchor']","a['native_gain_control'],control['native_gain']").replace("a['native_gain'],proposals[f'seed{seed}_rankone0']","a['scientific_progress'],proposals[f'seed{seed}_scientific_progress']")
s=s.replace("counts=Counter();summary={};solves={};checks=[]",'''counts=Counter();summary={};solves={};checks=[]
reference_path=CURVE/'DIAGNOSE_SCIENTIFIC_PROGRESS.py'
reference_source=reference_path.read_text('utf-8').split("np.savez(HERE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz'")[0]
reference={'__file__':str(reference_path)}
exec(compile(reference_source,str(reference_path),'exec'),reference)
''')
start=s.index(' # Reconstruct all macro, per-FIT joint');end=s.index(' risk_records=[]',start)
block=''' # Replay the fixed reference solve, then independently recompute all constraints below.
 reconstructed=reference['proposals'][f'seed{seed}_scientific_progress'];error=float(np.max(np.abs(reconstructed-a['scientific_progress'])));assert error<2e-7
 ref=reference['rows'][seed];assert ref['success'];t=ref['common_fraction_of_original_scientific_reductions']
 r=old['stacked_baseline_residual'];n=old['stacked_baseline_native'];J=old['original_jacobian'];N=old['original_native_jacobian'];delta=a['scientific_progress']-a['theta'];radius=.0125
 H=np.maximum((old['FD_plus_native']+old['FD_minus_native']-2*n[:,None])/.004**2,0);H[:,-1]=0
 with np.load(HERE.parent/f'native_curvature_diagnostic_01/terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',allow_pickle=False) as z:curvature={k:z[k] for k in z.files}
 terms=[]
 for data,name in [(curvature,'curvature_anchor'),(control,'native_gain')]:
  d=data[name]-old['theta'];values=np.concatenate([data[name+'_'+l+'_actual_native'] for l in fits]);pred=n+N@d+.5*H@np.square(d)
  for direction,extra in terms:pred+=extra*(direction@d)**2
  extra=np.maximum(values-pred,0);direction=d/(d@d);terms.append((direction,extra))
 pn=n+N@delta+.5*H@np.square(delta)
 for direction,extra in terms:pn+=extra*(direction@delta)**2
 anchor=np.concatenate([old[l+'_initial_native'] for l in fits]);cap=np.minimum(n,anchor*np.tile([1.01,1.01,1.10],9))
 assert ((pn-cap)/n<=1e-6).all() and ((pn/n).reshape(9,3)[:,:2]<=1-.05*t+1e-6).all()
 assert np.linalg.norm(delta)<=radius*(1+1e-6) and (a['scientific_progress']>=old['lower']-1e-12).all() and (a['scientific_progress']<=old['upper']+1e-12).all()
 predicted=r+J@delta;ratios=[];peak_ratios=[];macro_count=0
 for il,label in enumerate(fits):
  L=old[label+'_cholesky'];variance=np.square(L).sum(1)
  block=slice(il*816,(il+1)*816);assert predicted[block]@predicted[block]/(r[block]@r[block])<=.99+1e-6
  for ih in range(3):
   sl=slice((3*il+ih)*272,(3*il+ih+1)*272);ratio=predicted[sl]@predicted[sl]/(r[sl]@r[sl]);assert ratio<=1-.20*t+1e-6;ratios.append(float(ratio))
   rb=L@(r[sl]*3);rc=L@(predicted[sl]*3)
   for ib in range(4):
    take=slice(56*ib,56*(ib+1));b=np.mean(rb[take]**2/variance[take]);v=np.mean(rc[take]**2/variance[take]);assert v<=1.15*b+2/4096+1e-9;macro_count+=1
    if ib==2:assert v/b<=1-.10*t+1e-6;peak_ratios.append(float(v/b))
 assert macro_count==36
 solves[str(seed)]=dict(reference_solution_maximum_difference=error,reference_solver_iterations=ref['iterations'],independent_macro_constraints=macro_count,
    common_fraction_of_original_scientific_reductions=t,independent_predicted_joint_ratios=ratios,independent_predicted_peak_ratios=peak_ratios,independent_predicted_native_ratios=(pn/n).reshape(3,3,3).tolist())
 counts['reference_solver_replays']+=1;counts['independent_complete_constraint_checks']+=1
'''
s=s[:start]+block+s[end:]
s=s.replace("name=='curvature_control'","name=='native_gain_control'").replace("control['curvature_anchor_'+label+'_actual_'+key]","control['native_gain_'+label+'_actual_'+key]")
s=s.replace("v['all_pass']['native_gain']","v['all_pass']['scientific_progress']").replace('two_seed_native_gain_all_local_pass','two_seed_scientific_progress_all_local_pass').replace('PASS_SAVED_NATIVE_GAIN_DIAGNOSTIC_AUDIT','PASS_SAVED_SCIENTIFIC_PROGRESS_DIAGNOSTIC_AUDIT')
s=s.replace('independent_solutions=solves','solution_and_constraint_audit=solves').replace('full candidate constrained solves and gate summaries audited','candidate reference solves replayed and constraints plus gate summaries independently audited').replace('curvature control candidates','native-gain control candidates')
ast.parse(s);(here/'AUDIT_COMPLETE.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared fixed scientific-progress candidate audit')
