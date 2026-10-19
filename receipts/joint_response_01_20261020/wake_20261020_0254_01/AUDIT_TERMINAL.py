from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,numpy as np,torch

torch.set_num_threads(4)
ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;OUT=HERE/'collected/run_01'
p=json.loads((ROOT/'configs/joint_response_01_multibank_REAL_pilot_20261020.json').read_text())
manifest=json.loads((HERE/'REMOTE_TERMINAL_VERIFICATION.json').read_text())
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
for kind,dim in [('joint',54)]:
    track=f'seed1_{kind}'
    trace=json.loads((OUT/(track+'_TRACE.json')).read_text())
    assert trace['track']==track and len(trace['iterations'])==11 and not trace['paired_supervision'] and not trace['population_weights_fitted']
    initial=np.zeros(dim)
    if kind=='physical':initial[24:45][np.tril_indices(6)[0]==np.tril_indices(6)[1]]=.08
    endpoint=initial.copy();orders={};coverage={};track_rows=[]
    for name,number,offset,total in [('source',1024,100,6593),('real',4096,200,21047)]:
        pool=roles[name+'_train_rows'];visits=12*3*number;epochs=(visits+len(pool)-1)//len(pool)
        orders[name]=np.concatenate([pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000+offset+epoch)).numpy()] for epoch in range(epochs)])
        coverage[name]=np.zeros(total,dtype=np.int64)
    for iteration,row in enumerate(trace['iterations']):
        stem=f'{track}_ITER{iteration:02d}';report=json.loads((OUT/(stem+'.json')).read_text())
        with np.load(OUT/(stem+'.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
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
    tracks[track]=dict(iterations=11,accepted_updates=sum(x['accepted'] for x in track_rows),trials=sum(x['trials'] for x in track_rows),parameter_norm=float(np.linalg.norm(endpoint)),coverage={name:dict(unique=int((v>0).sum()),visits=int(v.sum()),min_visits=int(v[v>0].min()),max_visits=int(v.max())) for name,v in coverage.items()},heldout_excluded=True)
    details[track]=track_rows
assert counts['FD_columns_recomputed']==594 and counts['joint_candidates']==29 and counts['secant_corrections_recomputed']==18
prior=ROOT/'receipts/joint_response_01_20261020/wake_20261020_0218_01'
prior_manifest=json.loads((prior/'SEED0_REMOTE_MANIFEST.json').read_text());prior_review=json.loads((prior/'SEED0_REVIEW.json').read_text())
for name,item in prior_manifest['outputs'].items():assert manifest['outputs'][name]['sha256']==item['sha256'],name
assert prior_review['status']=='PASS_COMPLETED_SEED0_SAVED_NUMERIC_AUDIT_SCIENTIFIC_GATES_FAILED'
report=json.loads((OUT/'seed1_joint_ITER11.json').read_text())
with np.load(OUT/'seed1_joint_ITER11.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
assert report['iteration']==11 and report['FD_columns_completed']==9 and not report['trials'] and report['current_solve'] is None and report['selected_trial'] is None
assert 'endpoint' not in a and 'original_jacobian' not in a
np.testing.assert_array_equal(a['theta'],endpoint);np.testing.assert_array_equal(a['initial'],np.zeros(54))
for key,value in a.items():assert np.isfinite(value).all(),key
for il,label in enumerate((*fit_labels,'CHECK')):
    for name,number,offset in [('source',1024,50000),('real',4096,60000)]:
        if label in fit_labels:
            pos=11*3+il;expected=orders[name][pos*number:(pos+1)*number]
        else:
            pool=roles[name+'_train_rows'];expected=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000+offset+11)).numpy()[:number]]
        suffix='source_parents' if name=='source' else 'real_rows';np.testing.assert_array_equal(a[label+'_'+suffix],expected)
    np.testing.assert_array_equal(a[label+'_baseline_native'],report['baseline_native'][label]);np.testing.assert_array_equal(a[label+'_initial_native'],report['initial_native'][label])
    assert report['native_projection_contexts'][label]['seed']==p['native_projection_seeds'][label]
    L=a[label+'_cholesky'];variance=np.square(L).sum(1);vector=a[label+'_baseline_residual']
    for ih,h in enumerate(hypotheses):
        white=vector[ih*len(L):(ih+1)*len(L)]*np.sqrt(3);raw=L@white
        np.testing.assert_allclose(report['baseline_risks'][label][h]['joint_whitened'],np.mean(white**2),rtol=1e-11)
        for ib,block in enumerate(blocks):
            take=slice(ib*56,(ib+1)*56) if ib<4 else slice(224,272)
            np.testing.assert_allclose(report['baseline_risks'][label][h][block],np.mean(raw[take]**2/variance[take]),rtol=1e-10,atol=1e-12)
rvec=np.concatenate([a[k+'_baseline_residual'] for k in fit_labels])/np.sqrt(3);nvec=np.concatenate([a[k+'_baseline_native'] for k in fit_labels])
np.testing.assert_allclose(rvec,a['stacked_baseline_residual'],rtol=1e-14,atol=1e-14);np.testing.assert_array_equal(nvec,a['stacked_baseline_native'])
plus=np.minimum(a['upper'][:,None],endpoint[:,None]+.004*np.eye(54))[:,:9];minus=np.maximum(a['lower'][:,None],endpoint[:,None]-.004*np.eye(54))[:,:9]
np.testing.assert_array_equal(a['FD_plus_theta'],plus);np.testing.assert_array_equal(a['FD_minus_theta'],minus)
assert all(a[k].shape==(2448,9) for k in ['FD_plus_residual','FD_minus_residual']) and all(a[k].shape==(27,9) for k in ['FD_plus_native','FD_minus_native'])
failed=json.loads((OUT/'FAILED.json').read_text());c=failed['execution_counts'];ex=manifest['native_exit']
assert ex['exit_code']==1 and failed['error_type']=='TeacherBudget01' and failed['error']=='Teacher internal wall bound reached' and failed['elapsed_seconds']>7200 and failed['elapsed_seconds']<7500
assert ex['peak_rss_bytes']<12*1024**3 and sum(v['bytes'] for v in manifest['outputs'].values())<768*1024**2
assert all(v<=p['operation_budget'][k] for k,v in c.items())
assert c['REAL_response_updates']==5+10+tracks['seed1_joint']['accepted_updates']==19
assert c['finite_difference_columns']==1200+594+12 and c['FIT_bank_finite_difference_columns']==3*c['finite_difference_columns']
assert c['joint_direction_solves']==c['joint_direction_solve_attempts']==c['joint_candidate_trials']==47+29==76
assert c['secant_matrix_corrections']==23+18==41
assert c['REAL_fit_feature_rows']==36*3*4096 and c['REAL_CHECK_feature_rows']==36*4096 and c['REAL_evaluation_feature_rows']==2099
train_calls=36*4*2+c['finite_difference_columns']*6+c['joint_candidate_trials']*4
assert c['joint_response_calls']==train_calls+8+1024 and c['joint_response_rows']==train_calls*4096+8*8192+1024*1024
assert c['generated_observation_calls']==(train_calls+8)*3 and c['generated_observation_rows']==3*(train_calls*4096+8*8192)
assert c['native_component_rows']==2*3*train_calls*4096 and c['native_metric_rows']==8*3*(8192+2099)
assert c['kernel_signal_calls']==1024 and c['kernel_signal_rows']==1024*1024
assert c['allocated_banks']==36*4+2+2 and c['noise005_calls']==36*4+2 and c['noise005_rows']==36*4*4096+2*8192 and c['extension_rows']==3*c['noise005_rows']
assert c['shape_standard_normal_values']==36*4*4096*8+2*8192*8+2*128*1024*8
assert not (OUT/'COMPLETE.json').exists() and not (OUT/'seed1_EVALUATION.json').exists() and not (OUT/'seed1_physical_TRACE.json').exists()
receipt=dict(status='PASS_TERMINAL_BUDGET_STOP_AND_SAVED_STATE_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),native_exit=ex,failure=failed,source_files_verified=manifest['source_files_verified'],outputs_verified=len(manifest['outputs']),output_bytes=sum(v['bytes'] for v in manifest['outputs'].values()),prior_seed0_files_unchanged=len(prior_manifest['outputs']),prior_seed0_review_sha256=hashlib.sha256((prior/'SEED0_REVIEW.json').read_bytes()).hexdigest(),prior_seed0_final_failed_gates=24,new_seed1_completed_prefix_audit_counts=dict(counts),seed1_track=tracks['seed1_joint'],seed1_iteration_details=details,seed1_acceptance_checks=comparisons,last_partial_iteration=dict(zero_based_index=11,saved_FD_columns=9,consumed_FD_columns=12,consumed_but_unsaved_FD_columns=3,saved_arrays=len(a),role_entries_verified=20480,baseline_risk_views_recomputed=12,new_solves=0,new_updates=0,theta_matches_last_committed_state=True),all_execution_counts_reconciled=True,committed_parameter_updates=19,neural_updates=9,physical_updates=10,full_four_track_comparison_complete=False,scientific_acceptance=False,completion_scope=dict(remaining_seed1_neural_iterations=1,remaining_seed1_physical_iterations=12,new_FD_columns_if_recompute_partial_iteration=606,duplicated_FD_columns_including_unsaved=12,remaining_final_evaluation_heads=[1],remaining_kernel_heads=[1]),evidence_limit='Saved matrix/vector arithmetic and reported gates audited; trial waveforms and individual kernel draws were not retained. Seed1 final evidence does not yet exist. Lifetime GPU peak was not written to FAILED; allocator cap remains the unchanged configured 2.75GiB.',next_action='Prospectively register fixed-method completion of the remaining original direction, physical comparison, final evaluation and same-parent kernels; separately account repeated setup and partial-direction work.')
(HERE/'TERMINAL_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k not in ['seed1_acceptance_checks','seed1_iteration_details']}))
