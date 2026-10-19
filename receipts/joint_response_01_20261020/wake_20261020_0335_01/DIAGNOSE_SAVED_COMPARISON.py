from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, numpy as np

ROOT=Path('E:/GAN/research/N01'); HERE=Path(__file__).resolve().parent
OLD=ROOT/'receipts/joint_response_01_20261020/wake_20261020_0254_01/collected/run_01'
NEW=HERE/'collected/run_01'
review=json.loads((HERE/'TERMINAL_REVIEW.json').read_bytes())
review0=json.loads((ROOT/'receipts/joint_response_01_20261020/wake_20261020_0218_01/SEED0_REVIEW.json').read_bytes())
checks=review0['acceptance_checks']+review['seed1_acceptance_checks']
tracks={}; quantiles=[0,.25,.5,.75,1]
for seed in [0,1]:
 for kind in ['joint','physical']:
  track=f'seed{seed}_{kind}'; records=[x for x in checks if x['track']==track]
  current=np.asarray([x['native_ratios'] for x in records]).reshape(-1,3)
  anchored=np.asarray([x['initial_anchor_ratios'] for x in records]).reshape(-1,3)
  native_prediction=[];native_actual=[];native_error=[];gains=[];vector_errors=[];binding=[];accepted_iterations=[];trial_outcomes=[];grad_cosines=[];curvature=[];joint_singular_values=[]
  for iteration in range(12):
   source=OLD if seed==0 or (seed==1 and kind=='joint' and iteration<11) else NEW
   stem=f'{track}_ITER{iteration:02d}'; report=json.loads((source/(stem+'.json')).read_bytes())
   with np.load(source/(stem+'.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
   n=a['stacked_baseline_native']; N=a['original_native_jacobian'].reshape(3,3,3,-1)
   for metric in [0,1]:
    for h in range(3):
     for left,right in [(0,1),(0,2),(1,2)]:
      u,v=N[left,h,metric],N[right,h,metric]
      assert np.linalg.norm(u)>0 and np.linalg.norm(v)>0
      grad_cosines.append(dict(metric=['wave','QPSD'][metric],cosine=float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))))
   theta=a['theta']; central=np.isclose(a['FD_plus_theta'].diagonal()+a['FD_minus_theta'].diagonal(),2*theta,rtol=0,atol=1e-14)
   plus=a['FD_plus_native'][:,central]-n[:,None];minus=a['FD_minus_native'][:,central]-n[:,None]
   for metric in [0,1]:
    rows=np.arange(metric,27,3);numer=np.linalg.norm((plus+minus)[rows],axis=0);denom=np.linalg.norm((plus-minus)[rows],axis=0)
    assert (denom>0).all()
    curvature.append(dict(iteration=iteration,metric=['wave','QPSD'][metric],column_count=int(central.sum()),median_symmetric_to_antisymmetric_ratio=float(np.median(numer/denom)),max_symmetric_to_antisymmetric_ratio=float((numer/denom).max())))
   sv=np.linalg.svd(a['original_jacobian'],compute_uv=False); joint_singular_values.append(dict(iteration=iteration,largest=float(sv[0]),smallest=float(sv[-1]),rank_at_relative_1e_8=int((sv>sv[0]*1e-8).sum())))
   if report['selected_trial'] is not None:accepted_iterations.append(dict(iteration=iteration,trial=report['selected_trial']))
   for ti,trial in enumerate(report['trials']):
    prefix=f'trial{ti}_'; pred=(a[prefix+'predicted_native']/n-1).reshape(-1,3)
    actual=(np.concatenate([a[prefix+label+'_actual_native'] for label in ['FIT0','FIT1','FIT2']])/n-1).reshape(-1,3)
    native_prediction.extend(pred.tolist());native_actual.extend(actual.tolist());native_error.extend((actual-pred).tolist());binding.extend((np.abs(pred)<=1e-6).tolist())
    gains.append(trial['reduction_ratio']);vector_errors.append(trial['FIT_joint_vector_relative_error'])
    groups=Counter()
    for check in trial['checks'].values():
     groups['joint_pass']+=int(check['original_joint_accept']);groups['native_pass']+=int(check['native_pass'])
    trial_outcomes.append(dict(iteration=iteration,trial=ti,accepted=trial['accepted'],all_four_joint_pass=groups['joint_pass']==4,all_four_native_pass=groups['native_pass']==4,radius=trial['radius']))
   endpoint=a['endpoint']
  pred=np.array(native_prediction);actual=np.array(native_actual);error=np.array(native_error)
  outcomes=Counter((('joint' if x['all_four_joint_pass'] else 'no_joint')+'_'+('native' if x['all_four_native_pass'] else 'no_native')) for x in trial_outcomes)
  endpoint_groups=dict(mean=float(np.linalg.norm(endpoint[:24])),stochastic=float(np.linalg.norm(endpoint[24:45])),M0_modes=float(np.linalg.norm(endpoint[45:53])) if kind=='joint' else 0.,noise=float(endpoint[-1]))
  tracks[track]=dict(native_metric_order=['wave','QPSD','noise'],native_comparison_denominator_per_metric=len(current),current_exceedances=(current>np.array([1.01,1.01,1.1])).sum(0).tolist(),initial_anchor_exceedances=(anchored>np.array([1.01,1.01,1.1])).sum(0).tolist(),maximum_ratio_current=current.max(0).tolist(),maximum_ratio_initial_anchor=anchored.max(0).tolist(),predicted_constraint_denominator_per_metric=len(pred),predicted_nonincrease_binding_counts=np.array(binding).sum(0).tolist(),FIT_predicted_relative_change_quantiles=np.quantile(pred,quantiles,axis=0).tolist(),FIT_actual_relative_change_quantiles=np.quantile(actual,quantiles,axis=0).tolist(),FIT_realization_minus_prediction_quantiles=np.quantile(error,quantiles,axis=0).tolist(),FIT_actual_current_exceedances=(actual>np.array([.01,.01,.10])).sum(0).tolist(),FIT_joint_prediction_relative_vector_error_range=[min(vector_errors),max(vector_errors)],FIT_joint_prediction_relative_vector_error_median=float(np.median(vector_errors)),actual_to_predicted_gain_range=[min(gains),max(gains)],actual_to_predicted_gain_median=float(np.median(gains)),candidate_gate_groups=dict(outcomes),accepted_iterations=accepted_iterations,trials=trial_outcomes,endpoint_group_norms=endpoint_groups,native_gradient_between_FIT_cosines={metric:dict(denominator=sum(x['metric']==metric for x in grad_cosines),quantiles=np.quantile([x['cosine'] for x in grad_cosines if x['metric']==metric],quantiles).tolist()) for metric in ['wave','QPSD']},FD_symmetric_to_antisymmetric_native_change=curvature,joint_Jacobian_singular_values=joint_singular_values)
receipt=dict(status='COMPLETED_TWO_HEAD_SAVED_EVIDENCE_DIAGNOSIS',UTC=datetime.now(timezone.utc).isoformat(),scope='Original 48 completed directions and 95 candidates; saved matrix/vector arithmetic only. No new model calls, optimizations, endpoints, updates, evaluation selection or changed gates.',terminal_review_sha256=hashlib.sha256((HERE/'TERMINAL_REVIEW.json').read_bytes()).hexdigest(),quantile_probabilities=quantiles,tracks=tracks,final_evidence=review['final_evidence'],scientific_acceptance=False)
(HERE/'SAVED_COMPARISON_DIAGNOSIS.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({track:{k:v for k,v in d.items() if k not in ['trials','FD_symmetric_to_antisymmetric_native_change','joint_Jacobian_singular_values','final_evidence']} for track,d in tracks.items()},indent=2))
