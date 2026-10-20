from collections import Counter
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;T=HERE/'terminal_01';OUT=T/'collected/run_01';BASE=HERE.parent/'original_precision_REAL_pilot_01';FIXED=HERE.parent/'second_curvature_diagnostic_01'
p=json.loads((ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json').read_bytes());manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes());terminal=json.loads((T/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes());result=json.loads((OUT/'RESULT.json').read_bytes());complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert terminal['native_exit']['exit_code']==0 and not terminal['processes'] and terminal['complete_present'] and not terminal['failed_present']
assert terminal['source_files_verified']==len(manifest['files'])==879
assert hashlib.sha256((ROOT/'configs/joint_response_01_random_precision_diagnostic_20261020.json').read_bytes()).hexdigest()==manifest['config_sha256']==result['config_sha256']
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==complete['result_sha256']
rng=json.loads((T/'RNG_NATIVE_REPLAY.json').read_bytes())
assert rng['status']=='PASS_SAME_RUNTIME_EXACT_RANDOM_REPLAY' and len(rng['banks'])==8 and rng['values_verified']==8*32768*8
assert rng['config_sha256']==manifest['config_sha256'] and rng['model_calls']==0
local_hashes=0
for name,entry in manifest['files'].items():
 if (ROOT/name).is_file():
  raw=(ROOT/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'];local_hashes+=1
for name,digest in p['source_pins'].items():assert manifest['files'][name]['sha256']==digest
for name,path in json.loads((HERE/'INPUT_MAP.json').read_bytes()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['files'][name]['sha256']
for name,entry in terminal['outputs'].items():
 raw=(T/'collected'/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
raw=(T/'terminal.tar').read_bytes();assert len(raw)==terminal['archive_bytes'] and hashlib.sha256(raw).hexdigest()==terminal['archive_sha256']
assert terminal['native_exit']['peak_rss_bytes']<=12*1024**3 and result['peak_GPU_reserved_bytes']<=4*1024**3 and result['elapsed_seconds']<=5400
assert sum(v['bytes'] for v in terminal['outputs'].values())<=128*1024**2
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:roles={k:z[k] for k in z.files}
with np.load(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:
 for k in z.files:np.testing.assert_array_equal(roles[k],z[k])
hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint');schemes=('IID','ANTITHETIC_Z')
counts=Counter();summary={};checks=[];dispersion_detail={}
for seed in range(2):
 with np.load(BASE/f'terminal_01/collected/run_01/seed{seed}_joint_ITER02.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
 with np.load(FIXED/f'terminal_01/collected/run_01/seed{seed}_SECOND_CURVATURE.npz',allow_pickle=False) as z:fixed={k:z[k] for k in z.files}
 geometry={scheme:[] for scheme in schemes};errors={scheme:[] for scheme in schemes};paired_noise=[]
 for rep in range(-1,4):
  expected_seed=int(old['FIT0_random_seed']) if rep==-1 else p['random_seed']+seed*p['head_seed_stride']+rep*p['replicate_seed_stride']
  if rep>=0:
   with np.load(OUT/f'seed{seed}_rep{rep}_BANK.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
   assert bank['bank_seed']==expected_seed
   np.testing.assert_array_equal(bank['source_parents'],old['FIT0_source_parents']);np.testing.assert_array_equal(bank['real_rows'],old['FIT0_real_rows']);np.testing.assert_array_equal(bank['rows'],np.repeat(bank['source_parents'],32))
   native_replay=rng['banks'][f'run_01/seed{seed}_rep{rep}_BANK.npz']
   assert native_replay['seed']==expected_seed and native_replay['values_verified']==32768*8
   assert native_replay['bank_file_sha256']==terminal['outputs'][f'run_01/seed{seed}_rep{rep}_BANK.npz']['sha256']
   iid_z=bank['IID_z'];assert hashlib.sha256(iid_z.tobytes()).hexdigest()==native_replay['normal_array_sha256'];counts['random_latent_values_replayed']+=iid_z.size;counts['new_random_libraries_replayed']+=1
  reports={};native_noise=[]
  for scheme in (('REPLAY',) if rep==-1 else schemes):
   stem=f'seed{seed}_rep{rep}_{scheme}'
   with np.load(OUT/(stem+'.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
   report=json.loads((OUT/(stem+'.json')).read_bytes());reports[scheme]=report
   for name,value in a.items():assert np.isfinite(value).all();counts['finite_arrays']+=1
   assert report['seed']==seed and report['replicate']==rep and report['scheme']==scheme and report['projection_context']['seed']==p['native_projection_seed'] and a['random_seed']==expected_seed
   np.testing.assert_array_equal(a['theta'],old['theta']);np.testing.assert_array_equal(a['initial'],np.zeros(54));np.testing.assert_array_equal(a['lower'],old['lower']);np.testing.assert_array_equal(a['upper'],old['upper']);np.testing.assert_array_equal(a['cholesky'],old['FIT0_cholesky'])
   for label,number in [('source',1024),('real',4096)]:
    key='source_parents' if label=='source' else 'real_rows';np.testing.assert_array_equal(a[key],old['FIT0_'+key]);assert len(a[key])==number and np.isin(a[key],roles[label+'_train_rows']).all();counts['role_entries_verified']+=number
   if rep==-1:
    for key in ('baseline_residual','baseline_native','initial_native'):np.testing.assert_allclose(a[key],old['FIT0_'+key],rtol=1e-12,atol=1e-12)
    assert report['FD_columns_completed']==0;counts['saved_center_and_anchor_replays_verified']+=1
   else:
    z=iid_z.reshape(1024,32,8).copy()
    if scheme=='ANTITHETIC_Z':z[:,16:]=-z[:,:16];np.testing.assert_array_equal(z[:,16:],-z[:,:16]);counts['antithetic_pairs_verified']+=1024*16
    means=z.astype(np.float64).mean(1)
    np.testing.assert_allclose(means.mean(0),report['shape_latent_mean'],atol=1e-15,rtol=1e-12);assert abs(float(np.abs(means).max())-report['maximum_parent_latent_mean_absolute'])<1e-15
    assert report['FD_columns_completed']==54
    plus=np.repeat(a['theta'][:,None],54,axis=1);minus=plus.copy();idx=np.arange(54);plus[idx,idx]=np.minimum(a['upper'],a['theta']+.004);minus[idx,idx]=np.maximum(a['lower'],a['theta']-.004)
    np.testing.assert_array_equal(plus,a['FD_plus_theta']);np.testing.assert_array_equal(minus,a['FD_minus_theta']);distance=np.diag(plus-minus)
    J=(a['FD_plus_residual']-a['FD_minus_residual'])/distance;N=(a['FD_plus_native']-a['FD_minus_native'])/distance;H=np.maximum((a['FD_plus_native']+a['FD_minus_native']-2*a['baseline_native'][:,None])/.004**2,0);H[:,-1]=0
    np.testing.assert_array_equal(J,a['jacobian']);np.testing.assert_array_equal(N,a['native_jacobian']);np.testing.assert_array_equal(H,a['positive_diagonal_native_curvature']);counts['FD_columns_recomputed']+=54
    geometry[scheme].append(dict(J=J,N=N,n=a['baseline_native']));native_noise.append(a['baseline_native'][2::3])
   np.testing.assert_array_equal(a['baseline_native'],report['baseline_native']);np.testing.assert_array_equal(a['initial_native'],report['initial_native'])
   risk_records=[(a['baseline_residual'],report['baseline_risks'])]
   for name,key in p['candidate_keys'].items():
    np.testing.assert_array_equal(a[name],fixed[key]);c=report['checks'][name];native=a[name+'_actual_native'];risk_records.append((a[name+'_actual_residual'],c['risks']))
    np.testing.assert_array_equal(native,c['native']);ratio=(native/a['baseline_native']).reshape(3,3);anchor=(native/a['initial_native']).reshape(3,3)
    np.testing.assert_allclose(ratio,c['native_ratios'],rtol=1e-12);np.testing.assert_allclose(anchor,c['initial_anchor_ratios'],rtol=1e-12)
    if rep==-1:
     for suffix in ('residual','native'):np.testing.assert_allclose(a[name+'_actual_'+suffix],fixed[key+'_FIT0_actual_'+suffix],rtol=1e-12,atol=1e-12)
     counts['saved_fixed_candidate_replays_verified']+=1
    else:
     delta=a[name]-a['theta'];pr=a['baseline_residual']+J@delta;pn=a['baseline_native']+N@delta+.5*H@np.square(delta)
     np.testing.assert_allclose(pr,a[name+'_predicted_residual'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(pn,a[name+'_predicted_native'],rtol=1e-12,atol=1e-12)
     error=(native-pn)/a['baseline_native'];errors[scheme].append(error.reshape(3,3)[:,:2]);np.testing.assert_allclose(error.reshape(3,3),c['diagonal_native_relative_error'],rtol=1e-12,atol=1e-12)
     er=float(np.linalg.norm(a[name+'_actual_residual']-pr)/np.linalg.norm(a['baseline_residual']));assert abs(er-c['linear_joint_vector_relative_error'])<1e-12;counts['fixed_candidate_prediction_errors_recomputed']+=1
    base=report['baseline_risks'];risk=c['risks'];trust=c['trust'];joint=sum(risk[h]['joint_whitened'] for h in hypotheses)/sum(base[h]['joint_whitened'] for h in hypotheses)
    macro=all(risk[h][block]<=1.15*base[h][block]+2/4096 for h in hypotheses for block in blocks[:4]);assert trust['denominator']==32768
    tr=trust['mean_relative_RMS']<=.08 and trust['p99_relative_RMS']<=.25 and max(v for hh in trust['flag_count_changes'].values() for v in hh.values())<=max(4,.005*32768)
    ng=bool((ratio<=np.array([1.01,1.01,1.10])).all() and (anchor<=np.array([1.01,1.01,1.10])).all());jg=bool(joint<=.99 and macro and tr)
    assert tr==trust['pass'] and ng==c['native_pass'] and jg==c['original_joint_accept'] and (ng and jg)==c['accepted'];counts['acceptance_checks_recomputed']+=1
    checks.append(dict(seed=seed,replicate=rep,scheme=scheme,candidate=name,accepted=c['accepted'],joint_ratio=float(joint),native_ratio_max=ratio.max(0).tolist(),initial_anchor_ratio_max=anchor.max(0).tolist()))
   L=a['cholesky'];variance=np.square(L).sum(1)
   for vector,record in risk_records:
    for ih,h in enumerate(hypotheses):
     white=vector[ih*272:(ih+1)*272]*np.sqrt(3);raw=L@white;np.testing.assert_allclose(record[h]['joint_whitened'],np.mean(white**2),rtol=1e-11)
     for ib,block in enumerate(blocks):
      take=slice(56*ib,56*(ib+1)) if ib<4 else slice(224,272);np.testing.assert_allclose(record[h][block],np.mean(raw[take]**2/variance[take]),rtol=1e-10,atol=1e-12)
     counts['risk_views_recomputed']+=1
  if rep>=0:
   assert reports['IID']['noise_hashes']==reports['ANTITHETIC_Z']['noise_hashes'] and reports['IID']['noise_shared_with_paired_scheme'] and reports['ANTITHETIC_Z']['noise_shared_with_paired_scheme'];counts['paired_noise_hash_sets_verified']+=1
   paired_noise.append(dict(replicate=rep,baseline_noise_values_equal=bool(np.array_equal(*native_noise))))
 scale=np.mean([v['n'] for g in geometry.values() for v in g],axis=0);by_scheme={};detail={}
 for scheme,g in geometry.items():
  js=np.stack([v['J'] for v in g]);ns=np.stack([v['N']/scale[:,None] for v in g]).reshape(4,3,3,54);native=ns[:,:,:2].reshape(4,6,54)
  by_scheme[scheme]=dict(joint_derivative_dispersion=float(np.var(js,axis=0,ddof=1).sum()),native_wave_QPSD_derivative_dispersion=float(np.var(native,axis=0,ddof=1).sum()),mean_joint_J_norm=float(np.linalg.norm(js.mean(0))),mean_native_N_norm=float(np.linalg.norm(native.mean(0))),mean_absolute_native_Taylor_error=float(np.abs(errors[scheme]).mean()))
  detail[scheme]=dict(parameter_blocks={label:dict(J_dispersion=float(np.var(js[:,:,lo:hi],axis=0,ddof=1).sum()),N_wave_QPSD_dispersion=float(np.var(native[:,:,lo:hi],axis=0,ddof=1).sum()),mean_J_norm=float(np.linalg.norm(js.mean(0)[:,lo:hi]))) for label,lo,hi in [('mean',0,24),('shared_random',24,45),('M0_scale',45,53),('noise_scale',53,54)]},native_component_dispersion={name:float(np.var(ns[:,:,k],axis=0,ddof=1).sum()) for k,name in enumerate(('wave','QPSD','noise'))},pairwise_J_cosines=[dict(a=i,b=j,cosine=float(np.vdot(js[i],js[j])/(np.linalg.norm(js[i])*np.linalg.norm(js[j])))) for i,j in combinations(range(4),2)],mean_J_gram_eigenvalues=np.linalg.eigvalsh(js.mean(0).T@js.mean(0)).tolist())
  for key,value in by_scheme[scheme].items():np.testing.assert_allclose(value,result['summaries'][str(seed)]['by_scheme'][scheme][key],rtol=1e-11,atol=1e-12)
 iid,anti=by_scheme['IID'],by_scheme['ANTITHETIC_Z'];ratios={key:anti[key]/iid[key] for key in ('joint_derivative_dispersion','native_wave_QPSD_derivative_dispersion')};norm=anti['mean_joint_J_norm']/iid['mean_joint_J_norm']
 passed=all(v<=.75 for v in ratios.values()) and .8<=norm<=1.25 and anti['mean_absolute_native_Taylor_error']<=iid['mean_absolute_native_Taylor_error'];assert bool(passed)==result['summaries'][str(seed)]['diagnostic_gate_pass']
 summary[str(seed)]=dict(by_scheme=by_scheme,variance_ratios=ratios,mean_J_norm_ratio=norm,diagnostic_gate_pass=bool(passed));dispersion_detail[str(seed)]=dict(details=detail,paired_noise=paired_noise)
assert counts['FD_columns_recomputed']==864 and counts['acceptance_checks_recomputed']==36 and counts['new_random_libraries_replayed']==8 and counts['antithetic_pairs_verified']==131072
for name,maximum in p['operation_budget'].items():assert result['execution_counts'].get(name,0)==maximum,(name,result['execution_counts'].get(name,0),maximum)
assert set(result['execution_counts'])<=set(p['operation_budget']) and result['native_timing']['calls']==1800
passed=all(v['diagnostic_gate_pass'] for v in summary.values());assert passed==result['two_head_precision_gate_pass'] and not result['scientific_acceptance'] and result['committed_parameter_updates']==0 and result['new_parameter_candidates']==0 and not result['automatic_training']
review=dict(status='PASS_SAVED_RANDOM_PRECISION_DIAGNOSTIC_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),source_files_verified=879,source_pins=len(p['source_pins']),local_source_hashes=local_hashes,output_files_rehashed=len(terminal['outputs']),output_bytes=sum(v['bytes'] for v in terminal['outputs'].values()),native_exit=terminal['native_exit'],elapsed_seconds=result['elapsed_seconds'],config_sha256=result['config_sha256'],result_sha256=complete['result_sha256'],audit_counts=dict(counts),execution_counts=result['execution_counts'],summaries=summary,dispersion_details=dispersion_detail,checks=checks,two_head_precision_gate_pass=passed,scientific_acceptance=False,committed_parameter_updates=0,evidence_limit='Saved J/N, curvature, risks, gate summaries and Gaussian inputs independently recomputed. Paired electronic noise verified by saved hashes; native generation replayed original endpoints. Complete waveforms and RMS extraction were not independently regenerated. Four replicates at one historically exposed fitting batch do not establish a population variance guarantee or scientific response acceptance.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),summaries=summary),indent=2))
