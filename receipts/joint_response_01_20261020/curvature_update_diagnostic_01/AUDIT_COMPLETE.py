from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,numpy as np,torch
from scipy.optimize import minimize

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;T=HERE/'terminal_01';OUT=T/'collected/run_01'
PREVIOUS=HERE.parent/'original_precision_REAL_pilot_01';CURVE=HERE.parent/'native_curvature_diagnostic_01';CERT=HERE.parent/'scientific_progress_diagnostic_01'
p=json.loads((ROOT/'configs/joint_response_01_curvature_update_diagnostic_20261020.json').read_bytes())
m=json.loads((T/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes());s=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
result=json.loads((OUT/'RESULT.json').read_bytes());complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert m['native_exit']['exit_code']==0 and not m['processes'] and m['complete_present'] and not m['failed_present']
assert m['source_files_verified']==len(s['files'])==858
assert hashlib.sha256((ROOT/'configs/joint_response_01_curvature_update_diagnostic_20261020.json').read_bytes()).hexdigest()==s['config_sha256']==result['config_sha256']
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
with np.load(CERT/'CERTIFIED_CONIC_CURVATURE_UPDATE_PROPOSALS.npz',allow_pickle=False) as z:proposals={k:z[k] for k in z.files}
labels=('FIT0','FIT1','FIT2','CHECK','FRESH');fits=labels[:3];hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint')
counts=Counter();summary={};solves={};checks=[]
for seed in range(2):
 with np.load(OUT/f'seed{seed}_CURVATURE_UPDATE.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(PREVIOUS/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
 with np.load(CURVE/f'terminal_01/collected/run_01/seed{seed}_CURVATURE.npz',allow_pickle=False) as z:control={k:z[k] for k in z.files}
 report=json.loads((OUT/f'seed{seed}_CURVATURE_UPDATE.json').read_bytes())
 for name,value in a.items():assert np.isfinite(value).all();counts['finite_arrays']+=1
 np.testing.assert_array_equal(a['theta'],old['theta']);np.testing.assert_array_equal(a['initial'],np.zeros(54))
 np.testing.assert_array_equal(a['curvature_control'],control['curvature_anchor']);np.testing.assert_array_equal(a['curvature_update'],proposals[f'seed{seed}_curvature_update'])
 # Independently check the original convex problem and its saved primal/dual certificate.
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
   if label!='FRESH' and name=='curvature_control':
    for key in ('residual','native'):np.testing.assert_allclose(a[name+'_'+label+'_actual_'+key],control['curvature_anchor_'+label+'_actual_'+key],rtol=1e-12,atol=1e-12)
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
passed=all(v['all_pass']['curvature_update'] for v in summary.values());assert passed==result['two_seed_curvature_update_all_local_pass']
assert not result['scientific_acceptance'] and result['committed_parameter_updates']==0 and not result['automatic_training']
review=dict(status='PASS_SAVED_CURVATURE_UPDATE_DIAGNOSTIC_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),source_files_verified=858,source_pins=len(p['source_pins']),local_source_hashes=local_hashes,
    output_files_rehashed=len(m['outputs']),output_bytes=sum(v['bytes'] for v in m['outputs'].values()),native_exit=m['native_exit'],elapsed_seconds=result['elapsed_seconds'],
    config_sha256=result['config_sha256'],result_sha256=complete['result_sha256'],audit_counts=dict(counts),execution_counts=result['execution_counts'],
    solution_and_constraint_audit=solves,summaries=summary,two_seed_curvature_update_all_local_pass=passed,scientific_acceptance=False,committed_parameter_updates=0,
    evidence_limit='Saved vector arithmetic, fixed candidate original convex constraints and primal/dual certificates plus gate summaries independently audited. Native execution replayed original centers, anchors and original curvature control candidates; complete candidate waveforms and trust extraction were not independently regenerated. FRESH is newly sampled from historically exposed fitting records.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),summary=summary,solution_and_constraint_audit=solves),indent=2))
