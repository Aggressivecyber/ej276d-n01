from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('E:/GAN/research/N01'))
parser.add_argument('--receipt-root',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--run-root',type=Path)
args=parser.parse_args()
ROOT=args.root;HERE=args.receipt_root;T=HERE/'terminal_01'
p=json.loads((ROOT/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json').read_bytes())
NAS=args.run_root if args.run_root is not None else Path(p['storage']['windows_root']);OUT=NAS/'run_01'
manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_bytes())
terminal=json.loads((T/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
result=json.loads((OUT/'RESULT.json').read_bytes());complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert terminal['native_exit']['exit_code']==0 and not terminal['processes'] and terminal['complete_present'] and not terminal['failed_present']
assert terminal['source_files_verified']==len(manifest['files'])==185
assert result['config_sha256']==manifest['config_sha256']==hashlib.sha256((ROOT/'configs/joint_response_01_recentered_derivative_diagnostic_20261020.json').read_bytes()).hexdigest()
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==complete['result_sha256']
for name,entry in manifest['files'].items():
    raw=(NAS/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
for name,digest in p['source_pins'].items():assert manifest['files'][name]['sha256']==digest
for name,entry in terminal['outputs'].items():
    raw=(NAS/name).read_bytes();assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
assert terminal['native_exit']['peak_rss_bytes']<=12*1024**3
assert result['peak_GPU_reserved_bytes']<=4*1024**3 and result['elapsed_seconds']<=3600
assert sum(v['bytes'] for v in terminal['outputs'].values())<=64*1024**2
with np.load(NAS/p['roles_file'],allow_pickle=False) as z:roles=dict(z)
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:
    for key in z.files:np.testing.assert_array_equal(z[key],roles[key])
labels=('FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3');hypotheses=('fast','slow','common_level')
blocks=('wave','charge_morphology','peak_codes','baseline','joint');counts=Counter();summary={}
previous_review=json.loads((NAS/'receipts/joint_response_01_20261020/stable_projection_diagnostic_01/terminal_01/TERMINAL_REVIEW.json').read_bytes())
for seed in range(2):
    with np.load(NAS/p['saved_arrays'][seed],allow_pickle=False) as z:saved=dict(z)
    summary[str(seed)]={}
    for label in labels:
        stem=f'seed{seed}_{label}_RECENTERED'
        with np.load(OUT/(stem+'.npz'),allow_pickle=False) as z:a=dict(z)
        report=json.loads((OUT/(stem+'.json')).read_bytes())
        for value in a.values():assert np.isfinite(value).all();counts['finite_arrays']+=1
        assert report['seed']==seed and report['label']==label and report['FD_columns_completed']==53
        map_label='FIT0' if label.startswith('IID') else label
        assert report['projection_context']['seed']==p['native_projection_seeds'][map_label]
        for key in ('initial','lower','upper'):np.testing.assert_array_equal(a[key],saved[key])
        np.testing.assert_array_equal(a['previous_theta'],saved['theta'])
        np.testing.assert_array_equal(a['theta'],saved['curvature_update_control'])
        for key,number,pool in [('source_parents',1024,'source_train_rows'),('real_rows',4096,'real_train_rows')]:
            np.testing.assert_array_equal(a[key],saved[label+'_'+key]);assert len(a[key])==number and np.isin(a[key],roles[pool]).all()
            counts['role_entries_replayed']+=number
        np.testing.assert_array_equal(a['cholesky'],saved[label+'_cholesky'])
        assert a['random_seed']==saved[label+'_random_seed'];counts['random_seeds_verified']+=1
        for key,source_key in [('previous_residual',label+'_baseline_residual'),('previous_native',label+'_baseline_native'),
            ('initial_native',label+'_initial_native'),('baseline_residual','curvature_update_control_'+label+'_actual_residual'),
            ('baseline_native','curvature_update_control_'+label+'_actual_native')]:
            np.testing.assert_allclose(a[key],saved[source_key],rtol=1e-12,atol=1e-12)
        counts['previous_centers_replayed']+=1;counts['new_centers_replayed']+=1;counts['initial_anchors_reuse_verified']+=1
        plus=np.repeat(a['theta'][:,None],53,axis=1);minus=plus.copy()
        for j in range(53):
            plus[j,j]=min(a['upper'][j],a['theta'][j]+.004);minus[j,j]=max(a['lower'][j],a['theta'][j]-.004)
        np.testing.assert_array_equal(a['FD_plus_theta'],plus);np.testing.assert_array_equal(a['FD_minus_theta'],minus)
        width=np.diag(plus[:53])-np.diag(minus[:53])
        J=(a['FD_plus_residual']-a['FD_minus_residual'])/width
        N=(a['FD_plus_native']-a['FD_minus_native'])/width
        assert J.shape==(816,53) and N.shape==(9,53)
        np.testing.assert_allclose(a['jacobian'],J,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a['native_jacobian'],N,rtol=1e-12,atol=1e-12)
        for key in ('FD_plus_native','FD_minus_native'):
            np.testing.assert_array_equal(a[key][2::3],np.repeat(a['baseline_native'][2::3,None],53,axis=1))
        assert not N[2::3].any();counts['finite_difference_columns_recomputed']+=53
        L=a['cholesky'];variance=np.square(L).sum(1)
        for vector,recorded in [(a['previous_residual'],report['original_risks']),(a['baseline_residual'],report['center_check']['risks'])]:
            for ih,h in enumerate(hypotheses):
                white=vector[ih*272:(ih+1)*272]*np.sqrt(3);raw=L@white
                np.testing.assert_allclose(recorded[h]['joint_whitened'],np.mean(white**2),rtol=1e-11)
                for ib,block in enumerate(blocks):
                    take=slice(56*ib,56*(ib+1)) if ib<4 else slice(224,272)
                    np.testing.assert_allclose(recorded[h][block],np.mean(raw[take]**2/variance[take]),rtol=1e-10,atol=1e-12)
                counts['risk_views_recomputed']+=1
        check=report['center_check'];risk=check['risks'];old=report['original_risks'];trust=check['trust']
        native=a['baseline_native'];ratio=(native/a['previous_native']).reshape(3,3);anchored=(native/a['initial_native']).reshape(3,3)
        np.testing.assert_array_equal(native,check['native'])
        np.testing.assert_allclose(ratio,check['native_ratios'],rtol=1e-12)
        np.testing.assert_allclose(anchored,check['initial_anchor_ratios'],rtol=1e-12)
        joint=np.mean([risk[h]['joint_whitened'] for h in hypotheses])/np.mean([old[h]['joint_whitened'] for h in hypotheses])
        macro=all(risk[h][b]<=1.15*old[h][b]+2/4096 for h in hypotheses for b in blocks[:4])
        tr=trust['mean_relative_RMS']<=.08 and trust['p99_relative_RMS']<=.25 and max(v for d in trust['flag_count_changes'].values() for v in d.values())<=.005*32768
        assert trust['denominator']==32768 and tr==trust['pass']
        ng=bool((ratio<=np.array([1.01,1.01,1.10])).all() and (anchored<=np.array([1.01,1.01,1.10])).all())
        jg=bool(joint<=.99 and macro and tr)
        assert ng==check['native_pass'] and jg==check['original_joint_accept'] and (ng and jg)==check['accepted']
        assert check['accepted']==previous_review['summaries'][str(seed)]['by_model']['curvature_update_control'][label]['accepted']
        expected=result['summaries'][str(seed)][label]
        assert expected['columns']==53 and expected['center_original_gate_pass']==check['accepted']
        np.testing.assert_allclose(expected['joint_J_norm'],np.linalg.norm(J),rtol=1e-12)
        np.testing.assert_allclose(expected['native_N_norm'],np.linalg.norm(N),rtol=1e-12)
        counts['control_gate_checks_recomputed']+=1
        summary[str(seed)][label]=dict(columns=53,joint_J_norm=float(np.linalg.norm(J)),native_N_norm=float(np.linalg.norm(N)),
            center_original_gate_pass=check['accepted'],joint_ratio=float(joint),noise_derivative_exact_zero=True)
for name,maximum in p['operation_budget'].items():assert result['execution_counts'].get(name,0)==maximum,(name,result['execution_counts'].get(name,0),maximum)
assert set(result['execution_counts'])<=set(p['operation_budget']) and result['native_timing']['calls']==1512
assert result['new_parameter_candidates']==result['committed_parameter_updates']==0
assert not result['scientific_acceptance'] and not result['automatic_training']
review=dict(status='PASS_SAVED_RECENTERED_DERIVATIVE_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),source_files_verified=185,
    source_pins=184,output_files_rehashed=len(terminal['outputs']),output_bytes=sum(v['bytes'] for v in terminal['outputs'].values()),
    native_exit=terminal['native_exit'],elapsed_seconds=result['elapsed_seconds'],audit_counts=dict(counts),summaries=summary,
    execution_counts=result['execution_counts'],config_sha256=result['config_sha256'],result_sha256=complete['result_sha256'],
    storage=p['storage'],no_local_or_compute_disk_output_copy=True,scientific_acceptance=False,committed_parameter_updates=0,
    evidence_limit='Saved finite differences, centers, roles and gate arithmetic independently audited. Anchors are pinned reused outputs. Complete waveforms and RMS extraction were not independently regenerated. No restoration solve or candidate execution occurred.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),elapsed_seconds=result['elapsed_seconds'])))
