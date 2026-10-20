from datetime import datetime,timezone
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.linalg import eigh

HERE=Path(__file__).resolve().parent;OUT=HERE/'terminal_01/collected/run_01'
review=json.loads((HERE/'terminal_01/TERMINAL_REVIEW.json').read_bytes())
assert review['status']=='PASS_SAVED_RANDOM_PRECISION_DIAGNOSTIC_AUDIT'
entries=[];inputs={}
for seed in range(2):
    for scheme in ('IID','ANTITHETIC_Z'):
        banks=[];reports=[]
        for rep in range(4):
            path=OUT/f'seed{seed}_rep{rep}_{scheme}.npz';inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as z:banks.append({k:z[k] for k in z.files})
            reports.append(json.loads(path.with_suffix('.json').read_bytes()))
        J=np.stack([a['jacobian'] for a in banks]);N=np.stack([a['native_jacobian'] for a in banks])
        H=np.stack([a['positive_diagonal_native_curvature'] for a in banks])
        n=np.stack([a['baseline_native'] for a in banks]);anchor=np.stack([a['initial_native'] for a in banks])
        r=np.stack([a['baseline_residual'] for a in banks]);rscale=np.sqrt(np.mean(np.square(r).sum(1)))
        Jscale=J*.0125/rscale;meanJ=Jscale.mean(0);B=meanJ.T@meanJ
        noiseJ=Jscale-meanJ;K=np.einsum('rpi,rpj->ij',noiseJ,noiseJ)/3
        ridge=.02*np.trace(B)/54
        vals,vectors=eigh(B,K/4+ridge*np.eye(54))
        candidates=[]
        for name in ('one_rank','two_rank'):
            delta=banks[0][name]-banks[0]['theta']
            actual=np.stack([a[name+'_actual_native'] for a in banks]);pred=np.stack([a[name+'_predicted_native'] for a in banks])
            linear=np.einsum('rij,j->ri',N,delta)/n
            quadratic=.5*np.einsum('rij,j->ri',H,delta**2)/n
            change=(actual-n)/n;error=(actual-pred)/n
            selected=error.reshape(4,3,3)[:,:,:2]
            jerror=np.stack([a[name+'_actual_residual']-a[name+'_predicted_residual'] for a in banks])/rscale
            Jdelta=np.einsum('rij,j->ri',J,delta)/rscale
            candidates.append(dict(candidate=name,accepted_repeats=sum(v['checks'][name]['accepted'] for v in reports),joint_ratio=[sum(v['checks'][name]['risks'][h]['joint_whitened'] for h in ('fast','slow','common_level'))/sum(v['baseline_risks'][h]['joint_whitened'] for h in ('fast','slow','common_level')) for v in reports],initial_anchor_max_each_repeat=(actual/anchor).reshape(4,3,3).max(1).tolist(),native_actual_change_mean=change.reshape(4,3,3).mean(0).tolist(),native_actual_change_sd=change.reshape(4,3,3).std(0,ddof=1).tolist(),native_linear_mean=linear.reshape(4,3,3).mean(0).tolist(),native_quadratic_mean=quadratic.reshape(4,3,3).mean(0).tolist(),native_prediction_error_mean=error.reshape(4,3,3).mean(0).tolist(),native_prediction_error_sd=error.reshape(4,3,3).std(0,ddof=1).tolist(),native_bias_MSE_fraction=float(np.square(selected.mean(0)).sum()/np.square(selected).mean(0).sum()),native_mean_abs_error=float(np.abs(selected).mean()),joint_vector_bias_MSE_fraction=float(np.square(jerror.mean(0)).sum()/np.square(jerror).mean(0).sum()),joint_linear_change_relative_SD=float(np.sqrt(np.var(Jdelta,axis=0,ddof=1).sum())/np.linalg.norm(Jdelta.mean(0)))))
        difference=banks[0]['two_rank']-banks[0]['one_rank']
        Jd=np.einsum('rij,j->ri',J,difference)/rscale
        actual_difference=np.stack([a['two_rank_actual_residual']-a['one_rank_actual_residual'] for a in banks])/rscale
        entry=dict(seed=seed,scheme=scheme,replicates=4,center_initial_anchor_max_each_repeat=(n/anchor).reshape(4,3,3).max(1).tolist(),joint_J_rms_fluctuation_over_mean_norm=float(np.sqrt(np.var(J,axis=0,ddof=1).sum())/np.linalg.norm(J.mean(0))),normalized_gram_eigenvalues=np.linalg.eigvalsh(B).tolist(),normalized_noise_gram_eigenvalues=np.linalg.eigvalsh(K).tolist(),original_ridge=float(ridge),generalized_mean_signal_to_mean_sampling_variance_plus_original_ridge_eigenvalues=vals.tolist(),generalized_eigenvalue_counts={str(limit):int((vals>=limit).sum()) for limit in (1,4,9)},candidates=candidates,one_to_two_rank=dict(parameter_difference_norm=float(np.linalg.norm(difference)),joint_mean_linear_difference_norm=float(np.linalg.norm(Jd.mean(0))),joint_linear_difference_rms_SD=float(np.sqrt(np.var(Jd,axis=0,ddof=1).sum())),actual_joint_difference_mean_norm=float(np.linalg.norm(actual_difference.mean(0))),actual_joint_difference_rms_SD=float(np.sqrt(np.var(actual_difference,axis=0,ddof=1).sum()))))
        entries.append(entry)
record=dict(status='SAVED_VARIABILITY_AND_BIAS_DIAGNOSIS_COMPLETE',UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,entries=entries,new_generator_calls=0,new_solver_calls=0,new_candidates=0,parameter_updates=0,scientific_acceptance=False,interpretation_scope='Descriptive variance/bias decomposition of four fixed-record replicates, not confidence intervals or independent model acceptance. Generalized eigenvalues compare mean-J signal to the estimated variance of that four-bank mean plus the original isotropic ridge; no subspace has been selected for execution.')
(HERE/'SAVED_VARIABILITY_AND_BIAS_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(seed=e['seed'],scheme=e['scheme'],J_relative_SD=e['joint_J_rms_fluctuation_over_mean_norm'],generalized_counts=e['generalized_eigenvalue_counts'],candidates=[{k:c[k] for k in ('candidate','accepted_repeats','native_bias_MSE_fraction','native_mean_abs_error','joint_vector_bias_MSE_fraction','joint_linear_change_relative_SD','native_actual_change_mean','native_prediction_error_mean','native_prediction_error_sd')} for c in e['candidates']],candidate_difference=e['one_to_two_rank']) for e in entries],indent=2))
