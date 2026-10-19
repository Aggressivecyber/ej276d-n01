from pathlib import Path
from collections import Counter
import json,hashlib,numpy as np
root=Path('E:/GAN/research/N01');r=root/'receipts/joint_response_01_20261019/wake_20261019_0602_01';out=r/'collected/run_01';old=root/'receipts/joint_response_01_20261019/wake_20261019_0539_01/collected/run_01'
remote=json.loads((r/'REMOTE_TERMINAL_REVIEW.json').read_text(encoding='utf-8'))
for name,item in remote['outputs'].items():
 raw=(r/'collected'/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
result=json.loads((out/'RESULT.json').read_text(encoding='utf-8'));p=json.loads((root/'configs/joint_response_01_native_REAL_completion_20261019.json').read_text(encoding='utf-8'));prior=json.loads((old/'FAILED.json').read_text(encoding='utf-8'))
assert result['config_sha256']==hashlib.sha256((root/'configs/joint_response_01_native_REAL_completion_20261019.json').read_bytes()).hexdigest()
assert all(v<=p['operation_budget'][k] for k,v in result['execution_counts'].items())
assert result['prior_execution_counts']==prior['execution_counts']
assert result['combined_execution_counts']==dict(Counter(prior['execution_counts'])+Counter(result['execution_counts']))
assert result['combined_execution_counts']['finite_difference_columns']==2403 and result['combined_execution_counts']['joint_direction_solves']==48
assert result['new_neural_updates']==0 and result['new_physical_updates']==9
with np.load(out/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as a: roles={k:a[k] for k in a.files}
with np.load(old/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as a:
 for k in roles:assert np.array_equal(roles[k],a[k])
tracks={};geometry={};summaries={};kernels={};flags={}
for seed in [0,1]:
 for kind in ['joint','physical']:
  label=f'seed{seed}_{kind}';folder=out if label=='seed1_physical' else old
  trace=json.loads((folder/(label+'_TRACE.json')).read_text(encoding='utf-8'))['iterations'];assert len(trace)==12
  with np.load(folder/(label+'_STATE.npz'),allow_pickle=False) as a:
   for prefix,total in [('source',6593),('real',21047)]:
    cov=a[prefix+'_coverage'];assert len(cov)==total and set(np.flatnonzero(cov))==set(roles[prefix+'_train_rows']) and np.all(cov[roles[prefix+'_heldout_rows']]==0)
   theta_norm=float(np.linalg.norm(a['theta']))
  accepted=sum(x['accepted'] for x in trace);assert accepted==result['tracks'][label]['accepted_updates']
  tracks[label]={'iterations':len(trace),'accepted_updates':accepted,'trials':sum(len(x['attempts']) for x in trace),'theta_norm':theta_norm,'source_visited':6465,'real_visited':18948,'heldout_excluded':True}
  ratio=np.array([np.array(row['geometry']['predicted_native_values'])/np.array(row['geometry']['original_native_values']) for row in trace]).reshape(12,3,3)
  geometry[label]={metric:{'ratio_min':float(ratio[:,:,i].min()),'ratio_max':float(ratio[:,:,i].max()),'binding_count_at_1_within_1e-6':int((np.abs(ratio[:,:,i]-1)<1e-6).sum()),'denominator':36} for i,metric in enumerate(['wave','QPSD','noise'])}
 folder=old if seed==0 else out;evaluation=json.loads((folder/f'seed{seed}_EVALUATION.json').read_text(encoding='utf-8'));decision=json.loads((folder/f'seed{seed}_DECISION.json').read_text(encoding='utf-8'))
 expected=[]
 for lib,reports in evaluation.items():
  b,c=reports['M0'],reports['joint']
  for h in b['risks']:
   prefix=f'library{lib}/{h}/'
   for metric,factor in [('joint_whitened',p['success']['joint_remaining_fraction']),('peak_codes',p['success']['peak_remaining_fraction'])]:
    expected.append((prefix+metric,b['risks'][h][metric],c['risks'][h][metric],factor*b['risks'][h][metric]))
   for metric in ['wave','QPSD','noise']:
    factor=p['success']['noise_max_factor'] if metric=='noise' else p['success']['native_remaining_fraction'];expected.append((prefix+'native_'+metric,b['native'][h]['values'][metric],c['native'][h]['values'][metric],factor*b['native'][h]['values'][metric]))
   for metric in ['wave','charge_morphology','peak_codes','baseline']:
    limit=p['success']['macro_max_factor']*b['risks'][h][metric]+2/(p['final']['parents']*p['final']['draws_per_parent']);expected.append((prefix+'macro_'+metric,b['risks'][h][metric],c['risks'][h][metric],limit))
   for flag,entry in c['flags'][h].items():
    oldflag=b['flags'][h][flag];limit=oldflag['count']+max(4,p['success']['flag_increase_fraction']*entry['denominator']);expected.append((prefix+'flag_'+flag,oldflag['count'],entry['count'],limit))
 expected={row[0]:row[1:] for row in expected};assert len(expected)==len(decision['gates'])==96
 for g in decision['gates']:
  eb,ec,eu=expected[g['name']];assert (g['baseline'],g['candidate'],g['upper'])==(eb,ec,eu) and g['pass_gate']==(ec<=eu)
 assert decision==result['decisions'][str(seed)] and decision['pass_all']==all(g['pass_gate'] for g in decision['gates'])
 summaries[str(seed)]={'failed_gates':[g['name'] for g in decision['gates'] if not g['pass_gate']],'ratios_by_model_vs_M0':{}}
 for name in ['joint','physical_initial','physical']:
  rr={}
  for reports in evaluation.values():
   for h in reports['M0']['risks']:
    for metric in ['joint_whitened','wave','charge_morphology','peak_codes','baseline']:rr.setdefault('joint_'+metric,[]).append(reports[name]['risks'][h][metric]/reports['M0']['risks'][h][metric])
    for metric in ['wave','QPSD','noise']:rr.setdefault('native_'+metric,[]).append(reports[name]['native'][h]['values'][metric]/reports['M0']['native'][h]['values'][metric])
  summaries[str(seed)]['ratios_by_model_vs_M0'][name]={k:[min(v),max(v)] for k,v in rr.items()}
 reported=json.loads((folder/f'seed{seed}_KERNEL.json').read_text(encoding='utf-8'));assert reported==result['kernel_controls'][str(seed)];matrices={};kernels[str(seed)]={'libraries':{},'precision':{}}
 for lib in [0,1]:
  with np.load(folder/f'seed{seed}_library{lib}_KERNEL.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
  assert np.array_equal(a['parents'],roles['source_heldout_rows']) and len(a['parents'])==128
  current={}
  for name in ['M0','joint','physical_initial','physical']:
   assert a[name+'_mean'].shape==(128,384) and a[name+'_covariance'].shape==(128,16,16)
   assert all(np.isfinite(a[name+'_'+k]).all() for k in ['mean','covariance','Q_variance','wave_variance'])
   matrices[lib,name]=a[name+'_covariance'];qmean=a[name+'_mean'][:,110:366].sum(1);cv=np.sqrt(a[name+'_Q_variance'])/np.abs(qmean)
   current[name]={'Q_variance_mean':float(a[name+'_Q_variance'].mean()),'Q_CV_median':float(np.median(cv)),'Q_CV_range':[float(cv.min()),float(cv.max())]}
  for name,ref in [('joint','M0'),('physical','physical_initial')]:
   cov,cov0=a[name+'_covariance'],a[ref+'_covariance'];q,q0=a[name+'_Q_variance'],a[ref+'_Q_variance'];w,w0=a[name+'_wave_variance'],a[ref+'_wave_variance']
   metrics=dict(actual_MC_mean_relative_change=float(np.linalg.norm(a[name+'_mean']-a[ref+'_mean'])/np.linalg.norm(a[ref+'_mean'])),covariance_relative_change=float(np.linalg.norm(cov-cov0)/np.linalg.norm(cov0)),covariance_trace_ratio=float(np.trace(cov,axis1=1,axis2=2).sum()/np.trace(cov0,axis1=1,axis2=2).sum()),Q_variance_ratio=float(q.mean()/q0.mean()),wave_variance_ratio=float(w.mean()/w0.mean()),covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),Q_variance_zero_parent_count=int((q==0).sum()))
   for k,v in metrics.items():assert np.isclose(v,reported['comparisons'][str(lib)][name][k],rtol=1e-10,atol=1e-12),(seed,lib,name,k)
   current[name].update(metrics)
  kernels[str(seed)]['libraries'][str(lib)]=current
 for name in ['M0','joint','physical_initial','physical']:
  a,b=matrices[0,name],matrices[1,name];value=float(np.linalg.norm(a-b)/(.5*(np.linalg.norm(a)+np.linalg.norm(b))));assert np.isclose(value,reported['between_library_covariance_relative_difference'][name],rtol=1e-10);kernels[str(seed)]['precision'][name]=value
assert not result['REAL_calibration_candidate_pass_all'] and not result['scientific_acceptance']
review=dict(status='COMPLETE_EVIDENCE_VERIFIED_SCIENTIFIC_GATES_FAILED',UTC=remote['UTC'],source_files_verified=617,outputs_verified=len(remote['outputs']),native_exit=remote['native_exit'],result_sha256=remote['result_sha256'],execution_counts=result['execution_counts'],prior_execution_counts=prior['execution_counts'],combined_execution_counts=result['combined_execution_counts'],elapsed_seconds=result['elapsed_seconds'],all_counters_reconciled=True,all_four_tracks_complete=tracks,all_192_final_gates_recomputed=True,final_evidence=summaries,kernel_evidence_recomputed=kernels,linear_prediction_binding_summary=geometry,new_neural_updates=0,all_neural_updates=6,all_physical_updates=20,scientific_acceptance=False,next_action='Diagnose coupled native/joint local direction geometry without endpoint updates')
(r/'TERMINAL_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'elapsed_seconds':review['elapsed_seconds'],'tracks':tracks,'seed1':summaries['1'],'binding':geometry,'seed1kernel':kernels['1']}))