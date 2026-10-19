from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,numpy as np,torch

torch.set_num_threads(4)
ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;OUT=HERE/'collected/run_01'
p=json.loads((ROOT/'configs/joint_response_01_multibank_REAL_pilot_20261020.json').read_text())
manifest=json.loads((HERE/'REMOTE_TERMINAL_VERIFICATION.json').read_text())
OLD_ROOT=ROOT/'receipts/joint_response_01_20261020/wake_20261020_0254_01'
OLD=OLD_ROOT/'collected/run_01'
old_manifest=json.loads((OLD_ROOT/'REMOTE_TERMINAL_VERIFICATION.json').read_text())
for name,item in old_manifest['outputs'].items():
    raw=(OLD_ROOT/'collected'/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
assert hashlib.sha256((ROOT/'configs/joint_response_01_multibank_REAL_pilot_20261020.json').read_bytes()).hexdigest()=='c1beef05173cb47a16723b535b0214dd44914957f91e8f4d089ecd9656b17f6f'
for name,item in manifest['outputs'].items():
    raw=(HERE/'collected'/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
for name,digest in p['source_pins'].items():
    if not name.startswith('diagnostic_inputs/'):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:roles={k:z[k] for k in z.files}
with np.load(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:
    for k in z.files:np.testing.assert_array_equal(roles[k],z[k])
counts=Counter();comparisons=[];tracks={};details={}
hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint');fit_labels=('FIT0','FIT1','FIT2')
limits=np.array([1.01,1.01,1.10])[None,:];seed=1
for kind,dim in [('joint',54),('physical',46)]:
    track=f'seed1_{kind}'
    trace=json.loads((OUT/(track+'_TRACE.json')).read_text())
    assert trace['track']==track and len(trace['iterations'])==12 and not trace['paired_supervision'] and not trace['population_weights_fitted']
    initial=np.zeros(dim)
    if kind=='physical':initial[24:45][np.tril_indices(6)[0]==np.tril_indices(6)[1]]=.08
    endpoint=initial.copy();orders={};coverage={};track_rows=[]
    for name,number,offset,total in [('source',1024,100,6593),('real',4096,200,21047)]:
        pool=roles[name+'_train_rows'];visits=12*3*number;epochs=(visits+len(pool)-1)//len(pool)
        orders[name]=np.concatenate([pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000+offset+epoch)).numpy()] for epoch in range(epochs)])
        coverage[name]=np.zeros(total,dtype=np.int64)
    for iteration,row in enumerate(trace['iterations']):
        stem=f'{track}_ITER{iteration:02d}';source=OLD if kind=='joint' and iteration<11 else OUT;report=json.loads((source/(stem+'.json')).read_text())
        with np.load(source/(stem+'.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        for key,value in a.items():assert np.isfinite(value).all(),(stem,key)
        counts['finite_arrays']+=len(a);theta=a['theta']
        np.testing.assert_array_equal(theta,endpoint);np.testing.assert_array_equal(a['initial'],initial)
        assert theta.shape==(dim,) and (a['lower'][:-1]==-1).all() and a['lower'][-1]==0 and (a['upper']==1).all()
        assert report['track']==track and report['iteration']==iteration and report['FD_columns_completed']==dim and report['current_solve'] is None
        risk_records=[];checks=[]
        for il,label in enumerate((*fit_labels,'CHECK')):
            for name,number,offset in [('source',1024,50000),('real',4096,60000)]:
                if label in fit_labels:
                    pos=iteration*3+il;expected=orders[name][pos*number:(pos+1)*number]
                    np.add.at(coverage[name],expected,1)
                else:
                    pool=roles[name+'_train_rows'];expected=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000+offset+iteration)).numpy()[:number]]
                suffix='source_parents' if name=='source' else 'real_rows'
                np.testing.assert_array_equal(a[label+'_'+suffix],expected)
                counts['role_entries_replayed']+=number;counts['selection_streams_replayed']+=1
            assert report['native_projection_contexts'][label]['seed']==p['native_projection_seeds'][label]
            np.testing.assert_array_equal(report['baseline_native'][label],a[label+'_baseline_native'])
            np.testing.assert_array_equal(report['initial_native'][label],a[label+'_initial_native'])
            risk_records.append((label,a[label+'_baseline_residual'],report['baseline_risks'][label]))
        r=np.concatenate([a[k+'_baseline_residual'] for k in fit_labels])/np.sqrt(3)
        n=np.concatenate([a[k+'_baseline_native'] for k in fit_labels])
        np.testing.assert_allclose(r,a['stacked_baseline_residual'],rtol=1e-14,atol=1e-14);np.testing.assert_array_equal(n,a['stacked_baseline_native'])
        plus=np.minimum(a['upper'][:,None],theta[:,None]+p['finite_difference_step']*np.eye(dim))
        minus=np.maximum(a['lower'][:,None],theta[:,None]-p['finite_difference_step']*np.eye(dim))
        np.testing.assert_array_equal(a['FD_plus_theta'],plus);np.testing.assert_array_equal(a['FD_minus_theta'],minus)
        difference=np.diag(plus)-np.diag(minus)
        J=(a['FD_plus_residual']-a['FD_minus_residual'])/difference;N=(a['FD_plus_native']-a['FD_minus_native'])/difference
        np.testing.assert_array_equal(J,a['original_jacobian']);np.testing.assert_array_equal(N,a['original_native_jacobian'])
        assert J.shape==(2448,dim) and N.shape==(27,dim);counts['FD_columns_recomputed']+=dim
        accepted=[]
        for i, trial in enumerate(report['trials']):
            prefix = f'trial{i}_'
            radius = p['initial_radius'] * p['rejection_radius_factor']**i
            assert trial['radius'] == radius
            np.testing.assert_allclose(a[prefix + 'jacobian'], J, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(a[prefix + 'native_jacobian'], N, rtol=1e-10, atol=1e-11)
            proposal = a[prefix + 'proposal']; step = proposal - theta
            assert (proposal >= a['lower'] - 1e-12).all() and (proposal <= a['upper'] + 1e-12).all()
            assert np.linalg.norm(step) <= radius * np.sqrt(1 + 1e-6)
            pr, pn = r + J @ step, n + N @ step
            np.testing.assert_allclose(a[prefix + 'predicted_residual'], pr, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(a[prefix + 'predicted_native'], pn, rtol=1e-10, atol=1e-11)
            g = trial['geometry']; assert g['variable_scale'] == radius
            ridge = p['solver']['ridge_fraction'] * np.diag(J.T @ J).mean()
            np.testing.assert_allclose(g['ridge'], ridge, rtol=1e-10)
            np.testing.assert_allclose(g['unit_ridge'] / radius**2, ridge, rtol=1e-10)
            np.testing.assert_allclose(g['unit_step_norm'] * radius, np.linalg.norm(step), rtol=1e-10)
            np.testing.assert_allclose(g['predicted_joint_remaining_fraction'], pr @ pr / (r @ r), rtol=1e-10)
            assert ((pn - n) / n <= 1e-6).all()
            macro_slacks = []
            for il, label in enumerate(fit_labels):
                L = a[label + '_cholesky']; variance = np.square(L).sum(1)
                for ih, h in enumerate(hypotheses):
                    sl = slice((il * 3 + ih) * len(L), (il * 3 + ih + 1) * len(L))
                    raw0, raw1 = L @ (r[sl] * 3), L @ (pr[sl] * 3)
                    for ib, block in enumerate(blocks[:4]):
                        take = slice(ib * 56, (ib + 1) * 56)
                        cap = 1.15 * np.mean(raw0[take]**2 / variance[take]) + 2 / 4096
                        risk = np.mean(raw1[take]**2 / variance[take]); key = label + '/' + h + '/' + block
                        np.testing.assert_allclose(g['macro_limits'][key], cap, rtol=1e-10)
                        np.testing.assert_allclose(g['predicted_macro_risks'][key], risk, rtol=1e-10)
                        macro_slacks.append(1 - risk / cap)
            slack = min(*macro_slacks, *(-(pn - n) / n), 1 - step @ step / radius**2)
            np.testing.assert_allclose(g['normalized_minimum_slack'], slack, atol=1e-12, rtol=1e-5)
            assert slack >= -1e-6
            fr = np.concatenate([a[prefix + label + '_actual_residual'] for label in fit_labels]) / np.sqrt(3)
            fn = np.concatenate([a[prefix + label + '_actual_native'] for label in fit_labels])
            predicted_gain, actual_gain = r @ r - pr @ pr, r @ r - fr @ fr
            np.testing.assert_allclose(trial['predicted_joint_reduction'], predicted_gain, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(trial['actual_joint_reduction'], actual_gain, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(trial['FIT_joint_vector_relative_error'], np.linalg.norm(fr - pr) / np.linalg.norm(r), rtol=1e-10)
            np.testing.assert_allclose(trial['FIT_native_relative_error'], ((fn - pn) / n).reshape(3, 3, 3), rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(trial['reduction_ratio'], actual_gain / predicted_gain, rtol=1e-9, atol=1e-11)
            for label, check in trial['checks'].items():
                risk_records.append((label, a[prefix + label + '_actual_residual'], check['risks']))
                checks.append((i, label, a[prefix + label + '_actual_native'], check))
            assert trial['accepted'] == all(c['accepted'] for c in trial['checks'].values())
            if trial['accepted']:
                accepted.append(i)
            elif i + 1 < len(report['trials']):
                J += np.outer(fr - r - J @ step, step) / (step @ step)
                N += np.outer(fn - n - N @ step, step) / (step @ step)
                np.testing.assert_allclose(J @ step, fr - r, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(N @ step, fn - n, rtol=1e-10, atol=1e-12)
                counts['secant_corrections_recomputed'] += 1
            counts['joint_candidates'] += 1
        if accepted:
            assert accepted==[len(report['trials'])-1] and report['selected_trial']==accepted[0] and report['stop_reason']=='first_all_FIT_CHECK_pass'
            endpoint=a[f'trial{accepted[0]}_proposal'].copy()
        else:
            assert report['selected_trial'] is None and report['stop_reason']=='trial_budget' and len(report['trials'])==3
        np.testing.assert_array_equal(a['endpoint'],endpoint)
        assert row['accepted']==bool(accepted) and row['selected_trial']==report['selected_trial'] and row['stop_reason']==report['stop_reason'] and row['evidence']==stem and row['trials']==len(report['trials'])
        np.testing.assert_allclose(row['parameter_norm'],np.linalg.norm(endpoint),rtol=1e-12)
        track_rows.append(dict(iteration=iteration,accepted=bool(accepted),trials=len(report['trials']),selected_trial=report['selected_trial'],parameter_norm=float(np.linalg.norm(endpoint)),predicted_gains=[x['predicted_joint_reduction'] for x in report['trials']],actual_gains=[x['actual_joint_reduction'] for x in report['trials']],vector_errors=[x['FIT_joint_vector_relative_error'] for x in report['trials']]))
        for label, vector, recorded in risk_records:
            L = a[label + '_cholesky']; variance = np.square(L).sum(1)
            for ih, h in enumerate(hypotheses):
                white = vector[ih * len(L):(ih + 1) * len(L)] * np.sqrt(3); raw = L @ white
                np.testing.assert_allclose(recorded[h]['joint_whitened'], np.mean(white**2), rtol=1e-11)
                for ib, block in enumerate(blocks):
                    take = slice(ib * 56, (ib + 1) * 56) if ib < 4 else slice(224, 272)
                    np.testing.assert_allclose(recorded[h][block], np.mean(raw[take]**2 / variance[take]), rtol=1e-10, atol=1e-12)
                counts['risk_views_recomputed'] += 1
        for i, label, native, check in checks:
            np.testing.assert_array_equal(native, check['native'])
            ratio = (native / a[label + '_baseline_native']).reshape(3, 3)
            np.testing.assert_allclose(ratio, check['native_ratios'], rtol=1e-12)
            anchored = (native / a[label + '_initial_native']).reshape(3, 3)
            np.testing.assert_allclose(anchored, check['initial_anchor_ratios'], rtol=1e-12)
            native_pass = bool((ratio <= limits).all() and (anchored <= limits).all()); assert native_pass == check['native_pass']
            old, new, trust = report['baseline_risks'][label], check['risks'], check['trust']
            old_mean, new_mean = (np.mean([risk[h]['joint_whitened'] for h in hypotheses]) for risk in (old, new))
            block_pass = all(new[h][block] <= 1.15 * old[h][block] + 2 / 4096 for h in hypotheses for block in blocks[:4])
            assert trust['denominator'] == 4096
            trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(v for flags in trust['flag_count_changes'].values() for v in flags.values()) <= max(4, .005 * 4096)
            assert trust['pass'] == trust_pass
            joint_pass = bool(trust_pass and new_mean <= .99 * old_mean and block_pass)
            assert joint_pass == check['original_joint_accept'] and (joint_pass and native_pass) == check['accepted']
            comparisons.append(dict(track=track, iteration=iteration, trial=i, label=label, original_joint_accept=joint_pass, native_pass=native_pass, trust_pass=trust_pass, macro_pass=block_pass, initial_anchor_ratios=anchored.tolist(), joint_ratio=new_mean/old_mean, native_ratios=ratio.tolist(), accepted=check['accepted']))
            counts['acceptance_checks_recomputed'] += 1
    with np.load(OUT/(track+'_STATE.npz'),allow_pickle=False) as z:
        np.testing.assert_array_equal(z['theta'],endpoint)
        for name in ('source','real'):
            np.testing.assert_array_equal(z[name+'_coverage'],coverage[name]);assert set(np.flatnonzero(coverage[name]))==set(roles[name+'_train_rows']) and (coverage[name][roles[name+'_heldout_rows']]==0).all()
    tracks[track]=dict(iterations=12,accepted_updates=sum(x['accepted'] for x in track_rows),trials=sum(x['trials'] for x in track_rows),parameter_norm=float(np.linalg.norm(endpoint)),coverage={name:dict(unique=int((v>0).sum()),visits=int(v.sum()),min_visits=int(v[v>0].min()),max_visits=int(v.max())) for name,v in coverage.items()},heldout_excluded=True)
    details[track]=track_rows
assert counts['FD_columns_recomputed']==1200 and counts['joint_candidates']==48
assert counts['secant_corrections_recomputed']==24
folder=OUT;evaluation=json.loads((folder/'seed1_EVALUATION.json').read_text());decision=json.loads((folder/'seed1_DECISION.json').read_text());summaries={};kernels={}
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
assert decision['pass_all']==all(g['pass_gate'] for g in decision['gates'])
summaries[str(seed)]={'failed_gates':[g['name'] for g in decision['gates'] if not g['pass_gate']],'ratios_by_model_vs_M0':{}}
for name in ['joint','physical_initial','physical']:
 rr={}
 for reports in evaluation.values():
  for h in reports['M0']['risks']:
   for metric in ['joint_whitened','wave','charge_morphology','peak_codes','baseline']:rr.setdefault('joint_'+metric,[]).append(reports[name]['risks'][h][metric]/reports['M0']['risks'][h][metric])
   for metric in ['wave','QPSD','noise']:rr.setdefault('native_'+metric,[]).append(reports[name]['native'][h]['values'][metric]/reports['M0']['native'][h]['values'][metric])
 summaries[str(seed)]['ratios_by_model_vs_M0'][name]={k:[min(v),max(v)] for k,v in rr.items()}
reported=json.loads((folder/f'seed{seed}_KERNEL.json').read_text(encoding='utf-8'));matrices={};kernels[str(seed)]={'libraries':{},'precision':{}}
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
assert not decision['pass_all']

config_path=ROOT/'configs/joint_response_01_multibank_REAL_completion_20261020.json'
config=json.loads(config_path.read_bytes());assert hashlib.sha256(config_path.read_bytes()).hexdigest()=='e103f28429fb0dc5bb5b88387583a98ed106d63b81d171a22abd1eb99dec6b1a'
completion=ROOT/'receipts/joint_response_01_20261020/multibank_completion_01'
inputs=json.loads((ROOT/'receipts/joint_response_01_20261020/multibank_REAL_pilot_01/INPUT_MAP.json').read_bytes());inputs.update(json.loads((completion/'INPUT_MAP.json').read_bytes()));inputs['diagnostic_inputs/REAL_CALIBRATION_ROLES.npz']=str(ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz')
for name,digest in config['source_pins'].items():
    path=Path(inputs[name]) if name in inputs else ROOT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
prior0=ROOT/'receipts/joint_response_01_20261020/wake_20261020_0218_01'
prior0_review=json.loads((prior0/'SEED0_REVIEW.json').read_bytes());prior_review=json.loads((OLD_ROOT/'TERMINAL_REVIEW.json').read_bytes())
assert prior0_review['status']=='PASS_COMPLETED_SEED0_SAVED_NUMERIC_AUDIT_SCIENTIFIC_GATES_FAILED' and prior_review['status']=='PASS_TERMINAL_BUDGET_STOP_AND_SAVED_STATE_AUDIT'
for name in ['seed0_EVALUATION.json','seed0_DECISION.json','seed0_KERNEL.json']:
    assert (OUT/name).read_bytes()==(OLD/name).read_bytes()==(prior0/'collected/run_01'/name).read_bytes(),name
old_trace=json.loads((OLD/'seed1_joint_TRACE.json').read_bytes());trace=json.loads((OUT/'seed1_joint_TRACE.json').read_bytes());assert trace['iterations'][:11]==old_trace['iterations'] and len(trace['iterations'])==12
with np.load(OLD/'seed1_joint_ITER11.npz',allow_pickle=False) as z:old_partial={k:z[k].copy() for k in z.files}
with np.load(OUT/'seed1_joint_ITER11.npz',allow_pickle=False) as z:completed_partial={k:z[k].copy() for k in z.files}
for key,old_value in old_partial.items():
    new_value=completed_partial[key][:,:9] if key.startswith('FD_') else completed_partial[key]
    assert new_value.dtype==old_value.dtype and new_value.shape==old_value.shape and new_value.tobytes()==old_value.tobytes(),key
replay=json.loads((OUT/'RESTART_REPLAY.json').read_bytes());assert replay['status']=='PASS_NATIVE_BASELINE_AND_SAVED_FD_BYTE_REPLAY' and replay['saved_arrays_replayed']==len(old_partial)==36 and replay['saved_FD_columns_replayed']==9
result=json.loads((OUT/'RESULT.json').read_bytes());complete=json.loads((OUT/'COMPLETE.json').read_bytes());c=result['execution_counts'];ex=manifest['native_exit']
assert ex['exit_code']==0 and complete['result_sha256']==hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==manifest['result_sha256']
assert not (OUT/'FAILED.json').exists() and result['elapsed_seconds']<3600 and ex['peak_rss_bytes']<12*1024**3 and result['peak_GPU_allocated_bytes']<=2.75*1024**3 and result['peak_GPU_reserved_bytes']<=2.75*1024**3
assert sum(x['bytes'] for x in manifest['outputs'].values())<768*1024**2
assert all(v<=config['operation_budget'][k] for k,v in c.items())
assert c['finite_difference_columns']==54+12*46==606 and c['FIT_bank_finite_difference_columns']==1818
new_candidates=tracks['seed1_joint']['trials']-prior_review['seed1_track']['trials']+tracks['seed1_physical']['trials']
assert new_candidates==c['joint_direction_solves']==c['joint_direction_solve_attempts']==c['joint_candidate_trials']==19
assert c['secant_matrix_corrections']==counts['secant_corrections_recomputed']-prior_review['new_seed1_completed_prefix_audit_counts']['secant_corrections_recomputed']==6
assert c['REAL_response_updates']==tracks['seed1_joint']['accepted_updates']-prior_review['seed1_track']['accepted_updates']+tracks['seed1_physical']['accepted_updates']==10
train_calls=13*4*2+c['finite_difference_columns']*6+c['joint_candidate_trials']*4
assert c['joint_response_calls']==train_calls+8+1024 and c['joint_response_rows']==train_calls*4096+8*8192+1024*1024
assert c['generated_observation_calls']==(train_calls+8)*3 and c['generated_observation_rows']==3*(train_calls*4096+8*8192)
assert c['native_component_rows']==2*3*train_calls*4096 and c['native_metric_rows']==8*3*(8192+2099)
assert c['kernel_signal_calls']==1024 and c['kernel_signal_rows']==1024*1024
assert c['allocated_banks']==13*4+2+2 and c['noise005_calls']==13*4+2 and c['noise005_rows']==13*4*4096+2*8192 and c['extension_rows']==3*c['noise005_rows']
assert c['shape_standard_normal_values']==13*4*4096*8+2*8192*8+2*128*1024*8
assert c['REAL_fit_feature_rows']==13*3*4096 and c['REAL_CHECK_feature_rows']==13*4096 and c['REAL_evaluation_feature_rows']==2099
assert c['checkpoint_loads']==6 and c['TRAIN_archives_decoded']==c['PCA_archives_decoded']==1 and c['old_A_calls']==52 and c['old_A_rows']==6593 and c['M0_head_calls']==104 and c['M0_head_rows']==13186
assert c['REAL_observable_replay_calls']==1 and c['REAL_observable_replay_rows']==21047 and c['feature_maps_constructed']==c['new_native_projection_contexts']==5
old_counts=prior_review['failure']['execution_counts'];assert result['prior_execution_counts']==old_counts
combined={k:old_counts.get(k,0)+c.get(k,0) for k in set(old_counts)|set(c)};assert combined==result['combined_execution_counts'] and combined['finite_difference_columns']==2412 and combined['REAL_response_updates']==29
assert result['new_committed_parameter_updates']==10 and result['prior_committed_parameter_updates']==19 and result['combined_committed_parameter_updates']==29 and result['duplicate_partial_FD_columns']==12 and result['original_iteration_limit_per_track']==12
assert result['full_four_track_comparison_complete'] and not result['scientific_acceptance'] and not result['REAL_calibration_candidate_pass_all'] and result['stage5_validation_pending'] and result['final_all_TRAIN_refit_pending']
for seed_key in ['0','1']:
    dec=json.loads((OUT/f'seed{seed_key}_DECISION.json').read_bytes());assert result['decisions'][seed_key]==dec
    ker=json.loads((OUT/f'seed{seed_key}_KERNEL.json').read_bytes());assert result['kernel_controls'][seed_key]==ker
for track,summary in {**prior0_review['tracks'],**tracks}.items():
    item=result['tracks'][track];assert item['accepted_updates']==summary['accepted_updates'] and item['source_parents_visited']==summary['coverage']['source']['unique']==6465 and item['real_records_visited']==summary['coverage']['real']['unique']==18948
    assert item['source_visit_sum']==summary['coverage']['source']['visits']==36864 and item['real_visit_sum']==summary['coverage']['real']['visits']==147456
summaries={'0':prior0_review['final_evidence']['0'],**summaries};kernels={'0':prior0_review['kernel_evidence_recomputed']['0'],**kernels}
rejects={track:dict(Counter((x['label']+'/'+reason) for x in comparisons if x['track']==track and not x['accepted'] for reason,failed in [('native',not x['native_pass']),('joint',not x['original_joint_accept']),('trust',not x['trust_pass']),('macro',not x['macro_pass'])] if failed)) for track in tracks}
review=dict(status='PASS_FOUR_TRACK_COMPARISON_AUDIT_SCIENTIFIC_GATES_FAILED',UTC=datetime.now(timezone.utc).isoformat(),scientific_job_terminal=True,native_exit=ex,elapsed_seconds=result['elapsed_seconds'],peak_GPU_allocated_bytes=result['peak_GPU_allocated_bytes'],peak_GPU_reserved_bytes=result['peak_GPU_reserved_bytes'],source_files_verified=manifest['source_files_verified'],source_pins_verified=30,outputs_verified=len(manifest['outputs']),output_bytes=sum(x['bytes'] for x in manifest['outputs'].values()),audit_counts_seed1=dict(counts),tracks={**prior0_review['tracks'],**tracks},seed1_iteration_details=details,all_192_final_gates_recomputed=True,seed0_prior_review_reused_sha256=hashlib.sha256((prior0/'SEED0_REVIEW.json').read_bytes()).hexdigest(),seed0_final_evaluation_decision_kernel_bytes_unchanged=True,seed1_saved_partial_arrays_replayed=len(old_partial),prior_seed1_TRACE_prefix_unchanged=True,final_evidence=summaries,kernel_evidence_recomputed=kernels,seed1_acceptance_checks=comparisons,rejection_summary_seed1=rejects,new_execution_counts=c,combined_execution_counts=combined,new_committed_parameter_updates=10,combined_committed_parameter_updates=29,scientific_acceptance=False,full_four_track_comparison_complete=True,stage5_validation_pending=True,scope='Seed1 saved matrix/vector arithmetic, all 96 final gate summaries and actual-MC kernel aggregates audited. Seed0 completed audit reused after identical final evidence hashes. Trial waveforms and individual kernel draws were not retained; raw waveform extraction and per-draw moment accumulation are not replayed.',next_action='Diagnose the completed two-head saved evidence before registering further scientific computation; preserve all original gates and negative results.')
(HERE/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),tracks=review['tracks'],final=summaries,kernels=kernels,new_execution_counts=c,rejects=rejects)))
