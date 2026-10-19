from pathlib import Path
from collections import Counter
import json,hashlib,numpy as np
root=Path('E:/GAN/research/N01'); receipt=root/'receipts/joint_response_01_20261019/wake_20261019_0539_01'; out=receipt/'collected/run_01'
r=json.loads((receipt/'REMOTE_TERMINAL_REVIEW.json').read_text(encoding='utf-8'))
for name,item in r['outputs'].items():
 raw=(receipt/'collected'/name).read_bytes(); assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
p=json.loads((root/'configs/joint_response_01_native_REAL_pilot_20261019.json').read_text(encoding='utf-8'))
counts=r['failure']['execution_counts']; assert all(v<=p['operation_budget'][k] for k,v in counts.items())
with np.load(out/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z: roles={k:z[k] for k in z.files}
print(json.dumps({'role_shapes':{k:list(v.shape) for k,v in roles.items()}}))
tracks={}
for label in ['seed0_joint','seed0_physical','seed1_joint']:
 trace=json.loads((out/(label+'_TRACE.json')).read_text(encoding='utf-8')); rows=trace['iterations']; assert len(rows)==12
 with np.load(out/(label+'_STATE.npz'),allow_pickle=False) as z: state={k:z[k] for k in z.files}
 for name,role,total in [('source','source',6593),('real','real',21047)]:
  cov=state[name+'_coverage']; allowed=roles[role+'_train_rows']; held=roles[role+'_heldout_rows']
  assert len(cov)==total and set(np.flatnonzero(cov))==set(allowed) and np.all(cov[held]==0)
  assert set(allowed).isdisjoint(held) and len(set(allowed)|set(held))==total
 reasons=Counter(); attempted=0
 for row in rows:
  for attempt in row['attempts']:
   attempted+=1
   for part,check in attempt['checks'].items():
    if not check['native_pass']:reasons[part+'_native']+=1
    if not check['original_joint_accept']:reasons[part+'_joint']+=1
    if not check['trust']['pass']:reasons[part+'_trust']+=1
 tracks[label]=dict(iterations=len(rows),accepted_updates=sum(x['accepted'] for x in rows),attempts=attempted,rejection_counts=dict(reasons),source_parents_visited=int((state['source_coverage']>0).sum()),real_records_visited=int((state['real_coverage']>0).sum()),source_visit_sum=int(state['source_coverage'].sum()),real_visit_sum=int(state['real_coverage'].sum()),parameter_norm=float(np.linalg.norm(state['theta'])),theta=state['theta'].tolist())
assert sum(t['accepted_updates'] for t in tracks.values())==counts['REAL_response_updates']
decision=json.loads((out/'seed0_DECISION.json').read_text(encoding='utf-8')); gates=decision['gates']; assert len(gates)==96
for g in gates:assert g['pass_gate']==(g['candidate']<=g['upper'])
assert decision['pass_all']==all(g['pass_gate'] for g in gates)
ratios={}; failed=[]
for g in gates:
 if not g['pass_gate']:failed.append(g['name'])
 if g['baseline']!=0 and '/flag_' not in g['name']:
  key=g['name'].split('/')[-1];ratios.setdefault(key,[]).append(g['candidate']/g['baseline'])
ratio_ranges={k:[min(v),max(v)] for k,v in ratios.items()}
reported_kernel=json.loads((out/'seed0_KERNEL.json').read_text(encoding='utf-8')); matrices={};kernel={}
for lib in range(2):
 with np.load(out/f'seed0_library{lib}_KERNEL.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
 assert np.array_equal(a['parents'],roles['source_heldout_rows']) and len(a['parents'])==128
 for name in ['joint','M0','physical','physical_initial']:
  assert a[name+'_mean'].shape==(128,384) and a[name+'_covariance'].shape==(128,16,16)
  assert all(np.isfinite(a[name+'_'+k]).all() for k in ['mean','covariance','Q_variance','wave_variance'])
  matrices[lib,name]=a[name+'_covariance']
 kernel[str(lib)]={}
 for name,ref in [('joint','M0'),('physical','physical_initial')]:
  cov,cov0=a[name+'_covariance'],a[ref+'_covariance']; q,q0=a[name+'_Q_variance'],a[ref+'_Q_variance']; w,w0=a[name+'_wave_variance'],a[ref+'_wave_variance']
  metrics=dict(actual_MC_mean_relative_change=float(np.linalg.norm(a[name+'_mean']-a[ref+'_mean'])/np.linalg.norm(a[ref+'_mean'])),covariance_relative_change=float(np.linalg.norm(cov-cov0)/np.linalg.norm(cov0)),covariance_trace_ratio=float(np.trace(cov,axis1=1,axis2=2).sum()/np.trace(cov0,axis1=1,axis2=2).sum()),Q_variance_ratio=float(q.mean()/q0.mean()),wave_variance_ratio=float(w.mean()/w0.mean()),covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),Q_variance_zero_parent_count=int((q==0).sum()))
  for k,v in metrics.items():assert np.isclose(v,reported_kernel['comparisons'][str(lib)][name][k],rtol=1e-10,atol=1e-12),(name,k)
  metrics['per_parent_Q_variance_ratio_range']=[float((q/q0).min()),float((q/q0).max())]
  kernel[str(lib)][name]=metrics
precision={}
for name in ['joint','M0','physical','physical_initial']:
 a,b=matrices[0,name],matrices[1,name];v=float(np.linalg.norm(a-b)/(.5*(np.linalg.norm(a)+np.linalg.norm(b))))
 assert np.isclose(v,reported_kernel['between_library_covariance_relative_difference'][name],rtol=1e-10)
 precision[name]=v
summary=dict(status='FAILED_INTERNAL_WALL_PARTIAL_SCIENTIFIC_EVIDENCE_VERIFIED',UTC=r['UTC'],source_files_verified=r['source_files_verified'],collected_outputs_verified=len(r['outputs']),native_exit=r['native_exit'],failure=r['failure'],operation_counts_within_budget=True,output_bytes=sum(v['bytes'] for k,v in r['outputs'].items() if k.startswith('run_01/')),completed_tracks=tracks,incomplete_track='seed1_physical',seed1_physical_committed_updates=0,seed1_final_evaluation_available=False,seed1_kernel_available=False,seed0_pass_all=decision['pass_all'],seed0_failed_gates=failed,seed0_candidate_M0_ratio_ranges=ratio_ranges,seed0_actual_kernel_recomputed=kernel,seed0_kernel_between_library_precision=precision,all_role_denominators_retained=True,source_heldout_excluded_from_completed_fit_tracks=True,real_heldout_excluded_from_completed_fit_tracks=True,scientific_acceptance=False,automatic_restart=False)
(receipt/'TERMINAL_REVIEW.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['failure','completed_tracks']}))
print(json.dumps({k:{a:b for a,b in v.items() if a!='theta'} for k,v in tracks.items()}))