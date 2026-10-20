from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import io,json
import numpy as np

ROOT=Path('E:/GAN/research/N01');HERE=Path(__file__).resolve().parent;T=HERE/'terminal_01'
cp=ROOT/'configs/joint_response_01_fixed_candidate_remaining8_20261021.json';p=json.loads(cp.read_bytes())
NAS=Path(p['storage']['windows_root']);NEWOUT=NAS/'run_01'
OLDOUT=Path('Z:/GAN/research/N01/server_runs/joint_response01_recentered_candidate_check_20261020/run_01')
terminal=json.loads((T/'TERMINAL_VERIFICATION.json').read_bytes())
assert terminal['native_exit']['exit_code']==0 and not terminal['processes']
assert terminal['complete_present'] and not terminal['failed_present']
result=json.loads((NEWOUT/'RESULT.json').read_bytes());complete=json.loads((NEWOUT/'COMPLETE.json').read_bytes())
assert result['config_sha256']=='072e8e94c632ee85c9195c9b0169f5da49bf49341924df654d59951529572f4e'
assert terminal['native_exit']['peak_rss_bytes']<=12*1024**3
assert result['peak_GPU_reserved_bytes']<=4*1024**3 and result['elapsed_seconds']<=600
with np.load(io.BytesIO(Path(p['roles_file'].replace('/mnt/MiNas/','Z:/')).read_bytes()),allow_pickle=False) as z:roles=dict(z)
with np.load(io.BytesIO((NEWOUT/'REAL_CALIBRATION_ROLES.npz').read_bytes()),allow_pickle=False) as z:
    for key in z.files:np.testing.assert_array_equal(z[key],roles[key])
labels=('FIT0','FIT1','FIT2','IID0','IID1','IID2','IID3');hypotheses=('fast','slow','common_level')
blocks=('wave','charge_morphology','peak_codes','baseline','joint');counts=Counter();summary={}
for seed in range(2):
    with np.load(io.BytesIO(Path(p['saved_arrays'][seed].replace('/mnt/MiNas/','Z:/')).read_bytes()),allow_pickle=False) as z:previous=dict(z)
    with np.load(io.BytesIO(Path(p['candidate_arrays'][seed].replace('/mnt/MiNas/','Z:/')).read_bytes()),allow_pickle=False) as z:proposal=z['candidate']
    summary[str(seed)]={}
    for label in labels:
        OUT=OLDOUT if label in p['completed_prefix_labels'][str(seed)] else NEWOUT
        stem=f'seed{seed}_{label}_FIXED_CANDIDATE'
        with np.load(io.BytesIO((OUT/(stem+'.npz')).read_bytes()),allow_pickle=False) as z:a=dict(z)
        report=json.loads((OUT/(stem+'.json')).read_bytes())
        for value in a.values():assert np.isfinite(value).all();counts['finite_arrays']+=1
        assert report['seed']==seed and report['label']==label and report['fixed_candidate']
        map_label='FIT0' if label.startswith('IID') else label
        assert report['projection_context']['seed']==p['native_projection_seeds'][map_label]
        for key in ('initial','lower','upper'):np.testing.assert_array_equal(a[key],previous[key])
        np.testing.assert_array_equal(a['previous_theta'],previous['theta'])
        np.testing.assert_array_equal(a['theta'],previous['curvature_update_control'])
        np.testing.assert_array_equal(a['candidate_theta'],proposal)
        assert a['candidate_theta'][53]==a['theta'][53]
        assert (proposal>=a['lower']).all() and (proposal<=a['upper']).all()
        assert np.linalg.norm(proposal[:53]-a['theta'][:53])<=.003125+1e-10
        for key,number,pool in [('source_parents',1024,'source_train_rows'),('real_rows',4096,'real_train_rows')]:
            np.testing.assert_array_equal(a[key],previous[label+'_'+key])
            assert len(a[key])==number and np.isin(a[key],roles[pool]).all();counts['role_entries_replayed']+=number
        np.testing.assert_array_equal(a['cholesky'],previous[label+'_cholesky'])
        assert a['random_seed']==previous[label+'_random_seed'];counts['random_seeds_verified']+=1
        for key,source in [('previous_residual',label+'_baseline_residual'),('previous_native',label+'_baseline_native'),
            ('initial_native',label+'_initial_native'),('baseline_residual','curvature_update_control_'+label+'_actual_residual'),
            ('baseline_native','curvature_update_control_'+label+'_actual_native')]:
            np.testing.assert_allclose(a[key],previous[source],rtol=1e-12,atol=1e-12)
        dsource=p['derivative_inputs'][str(seed)][label]
        raw=Path(dsource['path']).read_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:derivatives=dict(z)
        L=a['cholesky'];variance=np.square(L).sum(1)
        for key,recorded in [('previous_residual',report['original_risks']),('baseline_residual',report['center_check']['risks']),('candidate_residual',report['candidate_check']['risks'])]:
            for ih,h in enumerate(hypotheses):
                white=a[key][ih*272:(ih+1)*272]*np.sqrt(3);raw=L@white
                np.testing.assert_allclose(recorded[h]['joint_whitened'],np.mean(white**2),rtol=1e-11)
                for ib,block in enumerate(blocks):
                    take=slice(56*ib,56*(ib+1)) if ib<4 else slice(224,272)
                    np.testing.assert_allclose(recorded[h][block],np.mean(raw[take]**2/variance[take]),rtol=1e-10,atol=1e-12)
                counts['risk_views_recomputed']+=1
        decisions={}
        for name,key in [('center_check','baseline_native'),('candidate_check','candidate_native')]:
            check=report[name];risk=check['risks'];old=report['original_risks'];trust=check['trust'];native=a[key]
            ratio=(native/a['previous_native']).reshape(3,3);anchored=(native/a['initial_native']).reshape(3,3)
            np.testing.assert_array_equal(native,check['native'])
            np.testing.assert_allclose(ratio,check['native_ratios'],rtol=1e-12);np.testing.assert_allclose(anchored,check['initial_anchor_ratios'],rtol=1e-12)
            joint=np.mean([risk[h]['joint_whitened'] for h in hypotheses])/np.mean([old[h]['joint_whitened'] for h in hypotheses])
            macro=all(risk[h][b]<=1.15*old[h][b]+2/4096 for h in hypotheses for b in blocks[:4])
            tr=trust['mean_relative_RMS']<=.08 and trust['p99_relative_RMS']<=.25 and max(v for d in trust['flag_count_changes'].values() for v in d.values())<=.005*32768
            assert trust['denominator']==32768 and tr==trust['pass']
            ng=bool((ratio<=np.array([1.01,1.01,1.10])).all() and (anchored<=np.array([1.01,1.01,1.10])).all())
            jg=bool(joint<=.99 and macro and tr)
            assert ng==check['native_pass'] and jg==check['original_joint_accept'] and (ng and jg)==check['accepted']
            decisions[name]=dict(accepted=ng and jg,joint_ratio=float(joint),macro_pass=macro,native_pass=ng,trust_pass=tr,
                max_wave_QPSD_previous_ratio=float(ratio[:,:2].max()),max_wave_QPSD_initial_ratio=float(anchored[:,:2].max()))
            counts['original_gate_checks_recomputed']+=1
        expected=result['summaries'][str(seed)][label]
        assert expected['candidate_accepted']==decisions['candidate_check']['accepted']
        assert expected['control_accepted']==decisions['center_check']['accepted']
        step=proposal[:53]-a['theta'][:53]
        predicted_r=a['baseline_residual']+derivatives['jacobian']@step
        predicted_n=a['baseline_native']+derivatives['native_jacobian']@step
        summary[str(seed)][label]=dict(**decisions,candidate_residual_prediction_error=float(np.linalg.norm(a['candidate_residual']-predicted_r)),
            maximum_native_prediction_error_over_previous=float(np.max(np.abs(a['candidate_native']-predicted_n)/a['previous_native'])))
        counts['fixed_candidate_library_checks_recomputed']+=1
for key,maximum in p['operation_budget'].items():assert result['execution_counts'].get(key,0)==maximum,(key,result['execution_counts'].get(key,0),maximum)
assert set(result['execution_counts'])<=set(p['operation_budget']) and result['native_timing']['calls']==24
assert result['committed_parameter_updates']==0 and result['new_parameter_candidates']==0 and result['existing_fixed_candidates_evaluated']==2
assert not result['scientific_acceptance'] and not result['automatic_training']
passed=all(v['candidate_check']['accepted'] for by_seed in summary.values() for v in by_seed.values())
assert result['all_fixed_candidate_gates_pass']==passed and counts['fixed_candidate_library_checks_recomputed']==14
review=dict(status='PASS_INDEPENDENT_COMBINED14_FIXED_CANDIDATE_RESULT_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    all_fixed_candidate_gates_pass=passed,NAS_file_IO_checks=False,NAS_file_hash_verification=False,
    native_exit=terminal['native_exit'],elapsed_seconds=result['elapsed_seconds'],config_sha256=result['config_sha256'],
    audit_counts=dict(counts),summaries=summary,execution_counts=result['execution_counts'],scientific_acceptance=False,
    committed_parameter_updates=0,new_parameter_candidates=0,unique_fixed_candidates=2,old_completed_libraries_reused=6,new_libraries=8,new_generator_calls=24,unique_generator_calls=42,old_unsaved_generator_calls_unknown_range=[0,3],cumulative_actual_generator_call_range=[42,45],restart_setup_charged_separately=True,old_native_exit_code=None,evidence_limit='Existing exposed libraries only. Saved centers, roles, risks/native and gate arithmetic are independently audited; complete waveforms/native distances and RMS extraction are not independently regenerated. No stage4 scientific acceptance or stage5 execution.')
(T/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(status=review['status'],all_fixed_candidate_gates_pass=passed,audit_counts=dict(counts))))
