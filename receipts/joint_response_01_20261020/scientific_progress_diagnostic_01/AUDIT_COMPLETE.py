from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np,torch
from scipy.optimize import minimize

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;T=HERE/'terminal_01';OUT=T/'collected/run_01'
PREVIOUS=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_gain_diagnostic_01'
p=json.loads((ROOT/'configs/joint_response_01_scientific_progress_diagnostic_20261020.json').read_bytes())
m=json.loads((T/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes());s=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
result=json.loads((OUT/'RESULT.json').read_bytes());complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert m['native_exit']['exit_code']==0 and not m['processes'] and m['complete_present'] and not m['failed_present']
assert m['source_files_verified']==len(s['files'])==846
assert hashlib.sha256((ROOT/'configs/joint_response_01_scientific_progress_diagnostic_20261020.json').read_bytes()).hexdigest()==s['config_sha256']==result['config_sha256']
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==complete['result_sha256']
local_hashes=0
for name,entry in s['files'].items():
 if (ROOT/name).is_file():
  raw=(ROOT/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'];local_hashes+=1
for name,digest in p['source_pins'].items():assert s['files'][name]['sha256']==digest
for name,path in json.loads((HERE/'INPUT_MAP.json').read_bytes()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==s['files'][name]['sha256']
for name,entry in m['outputs'].items():
 raw=(T/'collected'/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
raw=(T/'terminal.tar').read_bytes();assert len(raw)==m['archive_bytes'] and hashlib.sha256(raw).hexdigest()==m['archive_sha256']
assert m['native_exit']['peak_rss_bytes']<=12*1024**3 and result['peak_GPU_reserved_bytes']<=4*1024**3 and result['elapsed_seconds']<=360
assert sum(v['bytes'] for v in m['outputs'].values())<=64*1024**2
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:roles={k:z[k] for k in z.files}
with np.load(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:
 for k in z.files:np.testing.assert_array_equal(roles[k],z[k])
with np.load(CURVE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz',allow_pickle=False) as z:proposals={k:z[k] for k in z.files}
labels=('FIT0','FIT1','FIT2','CHECK','FRESH');fits=labels[:3];hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint')
counts=Counter();summary={};solves={};checks=[]
reference_path=CURVE/'DIAGNOSE_SCIENTIFIC_PROGRESS.py'
reference_source=reference_path.read_text('utf-8').split("np.savez(HERE/'SAVED_SCIENTIFIC_PROGRESS_PROPOSALS.npz'")[0]
reference={'__file__':str(reference_path)}
exec(compile(reference_source,str(reference_path),'exec'),reference)

for seed in range(2):
 with np.load(OUT/f'seed{seed}_SCIENTIFIC_PROGRESS.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(PREVIOUS/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
 with np.load(CURVE/f'terminal_01/collected/run_01/seed{seed}_NATIVE_GAIN.npz',allow_pickle=False) as z:control={k:z[k] for k in z.files}
 report=json.loads((OUT/f'seed{seed}_SCIENTIFIC_PROGRESS.json').read_bytes())
 for name,value in a.items():assert np.isfinite(value).all();counts['finite_arrays']+=1
 np.testing.assert_array_equal(a['theta'],old['theta']);np.testing.assert_array_equal(a['initial'],np.zeros(54))
 np.testing.assert_array_equal(a['native_gain_control'],control['native_gain']);np.testing.assert_array_equal(a['scientific_progress'],proposals[f'seed{seed}_scientific_progress'])
 # Replay the fixed reference solve, then independently recompute all constraints below.
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
 risk_records=[];by_model={name:{} for name in report['checks']}
 for label in labels:
  for name,number,offset,key in [('source',1024,100,'source_parents'),('real',4096,200,'real_rows')]:
   values=a[label+'_'+key]
   if label=='FRESH':
    pool=roles[name+'_train_rows'];expected=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['fresh_seed']+100000*seed+offset)).numpy()[:number]]
   else:expected=old[label+'_'+key]
   np.testing.assert_array_equal(values,expected);assert np.isin(values,roles[name+'_train_rows']).all();counts['role_entries_replayed']+=len(values)
  expected_seed=p['fresh_seed']+100000*seed+1000 if label=='FRESH' else old[label+'_random_seed']
  assert a[label+'_random_seed']==expected_seed;counts['random_bank_seeds_verified']+=1
  assert report['projection_contexts'][label]['seed']==p['native_projection_seeds'][label]
  for key in ('baseline_residual','baseline_native','initial_native'):
   if label!='FRESH':np.testing.assert_allclose(a[label+'_'+key],old[label+'_'+key],rtol=1e-12,atol=1e-12)
  np.testing.assert_array_equal(a[label+'_baseline_native'],report['baseline_native'][label]);np.testing.assert_array_equal(a[label+'_initial_native'],report['initial_native'][label])
  risk_records.append((label,a[label+'_baseline_residual'],report['baseline_risks'][label]))
  for name in report['checks']:
   check=report['checks'][name][label];native=a[name+'_'+label+'_actual_native'];risk_records.append((label,a[name+'_'+label+'_actual_residual'],check['risks']))
   np.testing.assert_array_equal(native,check['native']);ratio=(native/a[label+'_baseline_native']).reshape(3,3);anchor=(native/a[label+'_initial_native']).reshape(3,3)
   np.testing.assert_allclose(ratio,check['native_ratios'],rtol=1e-12);np.testing.assert_allclose(anchor,check['initial_anchor_ratios'],rtol=1e-12)
   if label!='FRESH' and name=='native_gain_control':
    for key in ('residual','native'):np.testing.assert_allclose(a[name+'_'+label+'_actual_'+key],control['native_gain_'+label+'_actual_'+key],rtol=1e-12,atol=1e-12)
   base=report['baseline_risks'][label];risk=check['risks'];trust=check['trust']
   joint=np.mean([risk[h]['joint_whitened'] for h in hypotheses])/np.mean([base[h]['joint_whitened'] for h in hypotheses])
   macro=all(risk[h][block]<=1.15*base[h][block]+2/4096 for h in hypotheses for block in blocks[:4])
   assert trust['denominator']==32768
   tr=trust['mean_relative_RMS']<=.08 and trust['p99_relative_RMS']<=.25 and max(v for hh in trust['flag_count_changes'].values() for v in hh.values())<=.005*32768
   ng=bool((ratio<=np.array([1.01,1.01,1.10])).all() and (anchor<=np.array([1.01,1.01,1.10])).all());jg=bool(joint<=.99 and macro and tr)
   assert tr==trust['pass'] and ng==check['native_pass'] and jg==check['original_joint_accept'] and (ng and jg)==check['accepted']
   entry=dict(seed=seed,model=name,label=label,accepted=check['accepted'],joint_ratio=float(joint),macro_pass=macro,trust_pass=tr,native_pass=ng,
       native_ratio_min=ratio.min(0).tolist(),native_ratio_max=ratio.max(0).tolist(),initial_anchor_ratio_max=anchor.max(0).tolist())
   checks.append(entry);by_model[name][label]=entry;counts['acceptance_checks_recomputed']+=1
 for label,vector,recorded in risk_records:
  L=a[label+'_cholesky'];variance=np.square(L).sum(1)
  for ih,h in enumerate(hypotheses):
   white=vector[ih*272:(ih+1)*272]*np.sqrt(3);raw=L@white
   np.testing.assert_allclose(recorded[h]['joint_whitened'],np.mean(white**2),rtol=1e-11)
   for ib,block in enumerate(blocks):
    take=slice(56*ib,56*(ib+1)) if ib<4 else slice(224,272)
    np.testing.assert_allclose(recorded[h][block],np.mean(raw[take]**2/variance[take]),rtol=1e-10,atol=1e-12)
   counts['risk_views_recomputed']+=1
 all_pass={name:all(x['accepted'] for x in rows.values()) for name,rows in by_model.items()}
 assert all_pass==report['all_local_gates']==result['summaries'][str(seed)]
 summary[str(seed)]=dict(all_pass=all_pass,by_model=by_model)
for name,maximum in p['operation_budget'].items():assert result['execution_counts'].get(name,0)==maximum,(name,result['execution_counts'].get(name,0),maximum)
assert set(result['execution_counts'])<=set(p['operation_budget']) and result['native_timing']['calls']==40
passed=all(v['all_pass']['scientific_progress'] for v in summary.values());assert passed==result['two_seed_scientific_progress_all_local_pass']
assert not result['scientific_acceptance'] and result['committed_parameter_updates']==0 and not result['automatic_training']
review=dict(status='PASS_SAVED_SCIENTIFIC_PROGRESS_DIAGNOSTIC_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),source_files_verified=846,source_pins=len(p['source_pins']),local_source_hashes=local_hashes,
    output_files_rehashed=len(m['outputs']),output_bytes=sum(v['bytes'] for v in m['outputs'].values()),native_exit=m['native_exit'],elapsed_seconds=result['elapsed_seconds'],
    config_sha256=result['config_sha256'],result_sha256=complete['result_sha256'],audit_counts=dict(counts),execution_counts=result['execution_counts'],
    solution_and_constraint_audit=solves,summaries=summary,two_seed_scientific_progress_all_local_pass=passed,scientific_acceptance=False,committed_parameter_updates=0,
    evidence_limit='Saved vector arithmetic, candidate reference solves replayed and constraints plus gate summaries independently audited. Native execution replayed original centers, anchors and native-gain control candidates; complete candidate waveforms and trust extraction were not independently regenerated. FRESH is newly sampled from historically exposed fitting records.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),summary=summary,solution_and_constraint_audit=solves),indent=2))
