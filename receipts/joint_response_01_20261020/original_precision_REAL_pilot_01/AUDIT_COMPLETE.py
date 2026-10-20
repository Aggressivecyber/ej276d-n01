from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,numpy as np,torch
from scipy.optimize import minimize

torch.set_num_threads(4)
ROOT=Path('E:/GAN/research/N01');BASE=Path(__file__).resolve().parent
HERE=BASE/'terminal_01';OUT=HERE/'collected/run_01'
p=json.loads((ROOT/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes())
manifest=json.loads((HERE/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
source=json.loads((BASE/'SOURCE_MANIFEST.json').read_bytes())
result=json.loads((OUT/'RESULT.json').read_bytes())
complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert hashlib.sha256((ROOT/'configs/joint_response_01_original_precision_REAL_pilot_20261020.json').read_bytes()).hexdigest()==source['config_sha256']==result['config_sha256']
assert hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()==complete['result_sha256']
assert manifest['source_files_verified']==len(source['files'])==819
assert manifest['native_exit']['exit_code']==0 and not manifest['processes']
assert manifest['complete_present'] and not manifest['failed_present']
local_hashes=0
for name,item in source['files'].items():
    if (ROOT/name).is_file():
        raw=(ROOT/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
        local_hashes+=1
for name,digest in p['source_pins'].items():assert source['files'][name]['sha256']==digest,name
for name,item in manifest['outputs'].items():
    raw=(HERE/'collected'/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
raw=(HERE/'terminal.tar').read_bytes();assert len(raw)==manifest['archive_bytes'] and hashlib.sha256(raw).hexdigest()==manifest['archive_sha256']
assert manifest['native_exit']['peak_rss_bytes']<=p['resources']['RSS_GiB']*1024**3
assert result['elapsed_seconds']<=p['resources']['internal_seconds']
assert result['peak_GPU_reserved_bytes']<=p['resources']['GPU_GiB']*1024**3
assert sum(v['bytes'] for v in manifest['outputs'].values())<=p['resources']['output_MiB']*1024**2
with np.load(OUT/'REAL_CALIBRATION_ROLES.npz',allow_pickle=False) as z:roles={k:z[k].copy() for k in z.files}
original_roles=ROOT/'receipts/joint_response_01_20261019/REAL_calibration_deployment_01/REAL_CALIBRATION_ROLES.npz'
assert hashlib.sha256(original_roles.read_bytes()).hexdigest()==p['source_pins'][p['roles_file']]
with np.load(original_roles,allow_pickle=False) as z:
    for k in z.files:np.testing.assert_array_equal(roles[k],z[k])
counts=Counter();comparisons=[];tracks={};details={};max_solver_error=0.
hypotheses=('fast','slow','common_level');blocks=('wave','charge_morphology','peak_codes','baseline','joint');fit_labels=('FIT0','FIT1','FIT2')
limits=np.array([1.01,1.01,1.10])[None,:]

def independent_solve(r,J,N,a,p,radius,step):
    operators,offsets=[],[]
    for il,label in enumerate(fit_labels):
        L=a[label+'_cholesky'];scale=np.sqrt(np.square(L).sum(1))
        for ih in range(3):
            sl=slice((3*il+ih)*len(L),(3*il+ih+1)*len(L))
            center=L@(r[sl]*3)/scale
            derivative=(L@(J[sl]*3*radius))/scale[:,None]
            for ib in range(4):
                take=slice(56*ib,56*(ib+1));cap=1.15*np.square(center[take]).mean()+2/4096
                offsets.append(center[take]/np.sqrt(56*cap));operators.append(derivative[take]/np.sqrt(56*cap))
    aa,bb=np.stack(operators),np.stack(offsets)
    scaled=J*radius;gram=scaled.T@scaled
    hh=gram+p['solver']['ridge_fraction']*np.diag(gram).mean()*np.eye(len(step))
    gg=scaled.T@r; nn=N*radius/np.concatenate([a[label+'_baseline_native'] for label in fit_labels])[:,None]
    def objective(u):return float((.5*u@hh@u+gg@u)/(r@r))
    def gradient(u):return (hh@u+gg)/(r@r)
    def constraints(u):return np.r_[1-np.square(bb+aa@u).sum(1),-nn@u,1-u@u]
    def constraint_jacobian(u):return np.vstack((-2*np.einsum('mni,mn->mi',aa,bb+aa@u),-nn,-2*u))
    solved=minimize(objective,np.zeros(len(step)),jac=gradient,method='SLSQP',
        bounds=list(zip((a['lower']-a['theta'])/radius,(a['upper']-a['theta'])/radius)),
        constraints={'type':'ineq','fun':constraints,'jac':constraint_jacobian},
        options={'maxiter':p['constrained_solver']['max_iterations'],'ftol':p['constrained_solver']['ftol'],'disp':False})
    assert solved.success,solved.message
    difference=float(np.max(np.abs(solved.x*radius-step)))
    assert difference<2e-7,difference
    assert constraints(solved.x).min()>=-1e-6
    return difference

for seed in range(2):
    for kind,dim in [('joint',54),('physical',46)]:
        track=f'seed{seed}_{kind}'
        trace=json.loads((OUT/(track+'_TRACE.json')).read_text())
        assert trace['track']==track and len(trace['iterations'])==p['iterations'] and not trace['paired_supervision'] and not trace['population_weights_fitted']
        initial=np.zeros(dim)
        if kind=='physical':initial[24:45][np.tril_indices(6)[0]==np.tril_indices(6)[1]]=.08
        endpoint=initial.copy();orders={};coverage={};track_rows=[]
        for name,number,offset,total in [('source',1024,100,6593),('real',4096,200,21047)]:
            pool=roles[name+'_train_rows'];visits=p['iterations']*3*number;epochs=(visits+len(pool)-1)//len(pool)
            orders[name]=np.concatenate([pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000*seed+offset+epoch)).numpy()] for epoch in range(epochs)])
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
                        pool=roles[name+'_train_rows'];expected=pool[torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['training_seed']+100000*seed+offset+iteration)).numpy()[:number]]
                    suffix='source_parents' if name=='source' else 'real_rows'
                    np.testing.assert_array_equal(a[label+'_'+suffix],expected)
                    counts['role_entries_replayed']+=number;counts['selection_streams_replayed']+=1
                bank_seed=p['training_seed']+100000*seed+(1000+100*iteration+10*il if label in fit_labels else 70000+100*iteration)
                assert int(a[label+'_random_seed'])==bank_seed
                counts['random_bank_seeds_verified']+=1
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
                max_solver_error=max(max_solver_error,independent_solve(r,J,N,a,p,radius,step))
                counts['independent_joint_solves']+=1
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
                elif i + 1 < len(report['trials']) or report['stop_reason']=='zero_joint_direction':
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
                assert report['selected_trial'] is None
                if report['stop_reason']=='zero_joint_direction':
                    prefix=f"trial{len(report['trials'])}_";radius=p['initial_radius']*p['rejection_radius_factor']**len(report['trials'])
                    np.testing.assert_allclose(a[prefix+'jacobian'],J,rtol=1e-10,atol=1e-11)
                    np.testing.assert_allclose(a[prefix+'native_jacobian'],N,rtol=1e-10,atol=1e-11)
                    max_solver_error=max(max_solver_error,independent_solve(r,J,N,a,p,radius,np.zeros(dim)))
                    g=report['zero_direction_geometry'];assert g['step_norm']==0 and g['unit_step_norm']==0
                    np.testing.assert_array_equal(g['original_native_values'],n)
                    np.testing.assert_array_equal(g['predicted_native_values'],n)
                    assert g['predicted_joint_remaining_fraction']==1 and g['normalized_minimum_slack']>=-1e-6
                    counts['independent_joint_solves']+=1;counts['zero_directions_recomputed']+=1
                else:
                    assert report['stop_reason']=='trial_budget' and len(report['trials'])==3
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
                assert trust['denominator'] == 32768
                trust_pass = trust['mean_relative_RMS'] <= .08 and trust['p99_relative_RMS'] <= .25 and max(v for flags in trust['flag_count_changes'].values() for v in flags.values()) <= max(4, .005 * 32768)
                assert trust['pass'] == trust_pass
                joint_pass = bool(trust_pass and new_mean <= .99 * old_mean and block_pass)
                assert joint_pass == check['original_joint_accept'] and (joint_pass and native_pass) == check['accepted']
                comparisons.append(dict(track=track, iteration=iteration, trial=i, label=label, original_joint_accept=joint_pass, native_pass=native_pass, trust_pass=trust_pass, macro_pass=block_pass, initial_anchor_ratios=anchored.tolist(), joint_ratio=new_mean/old_mean, native_ratios=ratio.tolist(), accepted=check['accepted']))
                counts['acceptance_checks_recomputed'] += 1
        with np.load(OUT/(track+'_STATE.npz'),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['theta'],endpoint)
            for name in ('source','real'):
                np.testing.assert_array_equal(z[name+'_coverage'],coverage[name]);assert set(np.flatnonzero(coverage[name]))==set(roles[name+'_train_rows']) and (coverage[name][roles[name+'_heldout_rows']]==0).all()
        tracks[track]=dict(iterations=p['iterations'],accepted_updates=sum(x['accepted'] for x in track_rows),trials=sum(x['trials'] for x in track_rows),parameter_norm=float(np.linalg.norm(endpoint)),coverage={name:dict(unique=int((v>0).sum()),visits=int(v.sum()),min_visits=int(v[v>0].min()),max_visits=int(v.max())) for name,v in coverage.items()},heldout_excluded=True)
        details[track]=track_rows
        for name in ('source','real'):
            assert tracks[track]['coverage'][name]['unique']==len(roles[name+'_train_rows'])
            assert tracks[track]['coverage'][name]['visits']==p['iterations']*3*(1024 if name=='source' else 4096)
        original_summary=result['tracks'][track]
        assert original_summary['accepted_updates']==tracks[track]['accepted_updates']
        assert original_summary['source_parents_visited']==6465 and original_summary['real_records_visited']==18948

summaries={};kernels={}
for seed in range(2):
    folder=OUT;evaluation=json.loads((folder/f'seed{seed}_EVALUATION.json').read_text());decision=json.loads((folder/f'seed{seed}_DECISION.json').read_text())
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
    assert decision==result['decisions'][str(seed)]

c=result['execution_counts']
for name,value in c.items():assert value<=p['operation_budget'][name],name
assert c['finite_difference_columns']==counts['FD_columns_recomputed']==600
assert c['FIT_bank_finite_difference_columns']==1800
assert c['joint_candidate_trials']==counts['joint_candidates']
assert c['joint_direction_solves']==c['joint_direction_solve_attempts']==counts['independent_joint_solves']
assert c['secant_matrix_corrections']==counts['secant_corrections_recomputed']
assert c['REAL_response_updates']==sum(v['accepted_updates'] for v in tracks.values())
train_calls=12*8+600*6+4*c['joint_candidate_trials']
assert c['joint_response_calls']==train_calls+16+2048
assert c['joint_response_rows']==train_calls*32768+16*8192+2048*1024
assert c['generated_observation_calls']==3*(train_calls+16)
assert c['generated_observation_rows']==3*(train_calls*32768+16*8192)
assert c['native_component_rows']==3*train_calls*(32768+4096)
assert c['native_metric_rows']==16*3*(8192+2099)
assert c['allocated_banks']==56 and c['noise005_calls']==52
assert c['noise005_rows']==1605632 and c['extension_rows']==4816896
assert c['shape_standard_normal_values']==17039360
assert c['kernel_signal_calls']==2048 and c['kernel_signal_rows']==2097152
assert c['REAL_fit_feature_rows']==147456 and c['REAL_CHECK_feature_rows']==49152 and c['REAL_evaluation_feature_rows']==4198
assert c['checkpoint_loads']==6 and c['TRAIN_archives_decoded']==c['PCA_archives_decoded']==1
assert c['old_A_calls']==52 and c['old_A_rows']==6593 and c['M0_head_calls']==104 and c['M0_head_rows']==13186
assert c['REAL_observable_replay_calls']==1 and c['REAL_observable_replay_rows']==21047
assert c['feature_maps_constructed']==c['new_native_projection_contexts']==5
assert result['native_timing']['calls']==train_calls
passed=all(not v['failed_gates'] for v in summaries.values())
assert result['REAL_calibration_candidate_pass_all']==passed and not result['scientific_acceptance']
assert result['stage5_validation_pending'] and result['final_all_TRAIN_refit_pending']
rejects={track:dict(Counter((x['label']+'/'+reason) for x in comparisons if x['track']==track and not x['accepted'] for reason,failed in [('native',not x['native_pass']),('joint',not x['original_joint_accept']),('trust',not x['trust_pass']),('macro',not x['macro_pass'])] if failed)) for track in tracks}
review=dict(status='PASS_SAVED_ORIGINAL_PRECISION_REAL_PILOT_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=819,local_source_hashes=local_hashes,source_pins=len(p['source_pins']),
    outputs_rehashed=len(manifest['outputs']),output_bytes=sum(x['bytes'] for x in manifest['outputs'].values()),
    native_exit=manifest['native_exit'],elapsed_seconds=result['elapsed_seconds'],config_sha256=result['config_sha256'],
    result_sha256=complete['result_sha256'],execution_counts=c,audit_counts=dict(counts),
    tracks=tracks,iteration_details=details,max_independent_solver_delta_error=max_solver_error,
    final_gates_recomputed=192,final_evidence=summaries,kernel_evidence_recomputed=kernels,
    acceptance_checks=comparisons,rejection_summary=rejects,REAL_calibration_candidate_pass_all=passed,
    scientific_acceptance=False,stage5_pending=True,
    evidence_limit='Saved matrix and vector arithmetic, complete solves including any zero directions, reported gate summaries and kernel moments audited. Original trial waveforms, trust RMS/flag extraction and per-draw kernel accumulation are not regenerated. Reporting roles are historically exposed frozen-system evaluation.')
(HERE/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),tracks=tracks,final=summaries,execution_counts=c,max_solver_error=max_solver_error)))
