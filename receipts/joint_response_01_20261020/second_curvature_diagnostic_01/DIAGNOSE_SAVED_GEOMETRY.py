from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BASE=HERE.parent/'original_precision_REAL_pilot_01';Q=HERE.parent/'curvature_update_diagnostic_01'
studies=[('diagonal','native_curvature_diagnostic_01','CURVATURE','curvature_anchor'),('native_gain','native_gain_diagnostic_01','NATIVE_GAIN','native_gain'),('scientific_progress','scientific_progress_diagnostic_01','SCIENTIFIC_PROGRESS','scientific_progress'),('one_rank','curvature_update_diagnostic_01','CURVATURE_UPDATE','curvature_update'),('two_rank','second_curvature_diagnostic_01','SECOND_CURVATURE','second_curvature')]
entries=[];inputs={};fits=('FIT0','FIT1','FIT2');hypotheses=('fast','slow','common_level');metrics=('wave','QPSD','noise')
for seed in range(2):
 path=BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz';inputs[path.relative_to(HERE.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 r=a['stacked_baseline_residual'];J=a['original_jacobian'];n=a['stacked_baseline_native'];N=a['original_native_jacobian'];theta=a['theta'];radius=.0125
 gram=(J*radius).T@(J*radius)/(r@r);ridge=.02*np.trace(gram)/54;eig,U=np.linalg.eigh(gram);values={};reports={};rows=[];center=[]
 for role in (*fits,'CHECK'):
  ratios=(a[role+'_baseline_native']/a[role+'_initial_native']).reshape(3,3)
  for ih,h in enumerate(hypotheses):
   for im,m in enumerate(metrics):
    if ratios[ih,im]>[1.01,1.01,1.10][im]:center.append(dict(role=role,hypothesis=h,metric=m,center_to_initial_ratio=float(ratios[ih,im]),limit=[1.01,1.01,1.10][im]))
 for label,folder,stem,candidate in studies:
  path=HERE.parent/folder/f'terminal_01/collected/run_01/seed{seed}_{stem}.npz';inputs[path.relative_to(HERE.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path,allow_pickle=False) as z:b={k:z[k] for k in z.files}
  rep=json.loads(path.with_suffix('.json').read_bytes());reports[label]=rep;values[label]=b;step=b[candidate]-theta;u=step/radius;pc=U.T@u
  rows.append(dict(candidate=label,step_norm=float(np.linalg.norm(step)),mean_norm=float(np.linalg.norm(step[:24])),shared_random_norm=float(np.linalg.norm(step[24:45])),M0_scale_norm=float(np.linalg.norm(step[45:53])),noise_delta=float(step[-1]),unit_norm_fraction_below_ridge=float(np.square(pc[eig<ridge]).sum()/np.square(pc).sum()),predicted_joint_ratio=float(np.square(r+J@step).sum()/(r@r)),actual_FIT_joint_ratio=float(sum(np.square(b[candidate+'_'+role+'_actual_residual']).sum() for role in fits)/(3*(r@r))),roles=[dict(role=role,accepted=rep['checks'][candidate][role]['accepted'],baseline_to_initial_max=(b[role+'_baseline_native']/b[role+'_initial_native']).reshape(3,3).max(0).tolist(),candidate_to_initial_max=(b[candidate+'_'+role+'_actual_native']/b[role+'_initial_native']).reshape(3,3).max(0).tolist(),candidate_to_current_max=(b[candidate+'_'+role+'_actual_native']/b[role+'_baseline_native']).reshape(3,3).max(0).tolist()) for role in (*fits,'CHECK','FRESH')]))
 d1=values['one_rank']['curvature_update']-theta;d2=values['two_rank']['second_curvature']-theta;change=d2-d1;order=np.argsort(-np.abs(change))
 native_difference=np.concatenate([values['two_rank']['second_curvature_'+l+'_actual_native']-values['one_rank']['curvature_update_'+l+'_actual_native'] for l in fits])/n
 comparison=dict(parameter_difference_norm=float(np.linalg.norm(change)),direction_cosine=float(d1@d2/(np.linalg.norm(d1)*np.linalg.norm(d2))),largest_parameter_changes=[dict(coordinate=int(k),difference=float(change[k])) for k in order[:8]],FIT_native_relative_change=native_difference.reshape(3,3,3).tolist(),linear_joint_change_norm_over_current=float(np.linalg.norm(J@change)/np.linalg.norm(r)))
 entries.append(dict(seed=seed,center_already_outside_initial_anchor=center,normalized_gram_eigenvalues=eig.tolist(),ridge=float(ridge),directions_below_ridge=int((eig<ridge).sum()),candidate_results=rows,one_to_two_rank_comparison=comparison))
record=dict(status='SAVED_ANCHOR_AND_GEOMETRY_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),entries=entries,inputs=inputs,new_generator_calls=0,new_solver_calls=0,parameter_updates=0,scientific_acceptance=False,same_fit_CHECK_roles_reused=True,FRESH_rows_and_projections_differ_between_studies=True,conclusions=['Both saved centers already exceed the fixed initial QPSD guard on some FIT0 hypotheses; a new candidate must repair that pre-existing excursion on the new round bank.','Adding a PSD FIT correction does not guarantee a stable parameter direction or native improvement in unmeasured directions.','FRESH comparisons between studies confound random records, noise and projection; within-study control comparisons are paired.','The three-candidate same-center curvature sequence remains closed negative.'])
(HERE/'SAVED_ANCHOR_AND_GEOMETRY_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(seed=x['seed'],center_exceedances=x['center_already_outside_initial_anchor'],directions_below_ridge=x['directions_below_ridge'],ridge=x['ridge'],candidate_geometry=[{k:r[k] for k in ('candidate','mean_norm','shared_random_norm','M0_scale_norm','unit_norm_fraction_below_ridge','actual_FIT_joint_ratio')} for r in x['candidate_results']],comparison=x['one_to_two_rank_comparison']) for x in entries],indent=2))
