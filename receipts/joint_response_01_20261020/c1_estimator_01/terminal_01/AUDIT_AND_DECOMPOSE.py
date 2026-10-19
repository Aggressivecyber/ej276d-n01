"""Independent saved-array audit; exact bilinear attribution retains cross terms."""
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations
import hashlib
import json
import time

import numpy as np
from scipy.linalg import solve_triangular
import torch

started = time.monotonic()
root = Path('E:/GAN/research/N01')
out = Path(__file__).resolve().parent
run = out/'collected/run_01'
p = json.loads((root/'configs/joint_response_01_c1_estimator_20261020.json').read_bytes())
manifest = json.loads((out.parent/'SOURCE_MANIFEST.json').read_bytes())
remote = json.loads((out/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
result = json.loads((run/'RESULT.json').read_bytes())
complete = json.loads((run/'COMPLETE.json').read_bytes())
assert remote['native_exit']['exit_code'] == 0 and remote['processes'] == []
assert remote['complete_present'] and remote['result_present'] and not remote['failed_present']
assert complete['result_sha256'] == hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest()
assert result['config_sha256'] == hashlib.sha256((root/'configs/joint_response_01_c1_estimator_20261020.json').read_bytes()).hexdigest()
assert (out/'terminal.tar').stat().st_size == remote['archive_bytes']
assert hashlib.sha256((out/'terminal.tar').read_bytes()).hexdigest() == remote['archive_sha256']
local_sources = remote_sources = 0
for name, record in manifest['files'].items():
    if (root/name).is_file():
        raw = (root/name).read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], name
        local_sources += 1
    else:
        remote_sources += 1
assert len(manifest['files']) == remote['source_files_verified'] == 744
for name, digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256'] == digest, name
for name, record in remote['outputs'].items():
    raw = (out/'collected'/name).read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], name
assert result['elapsed_seconds'] < p['resources']['internal_seconds']
assert remote['native_exit']['peak_rss_bytes'] <= p['resources']['RSS_GiB']*1024**3
assert result['peak_GPU_allocated_bytes'] <= p['resources']['GPU_GiB']*1024**3
assert sum(v['bytes'] for k,v in remote['outputs'].items() if k.startswith('run_01/')) < p['resources']['output_MiB']*1024**2

old = root/'receipts/joint_response_01_20261020/c1_fullpool_teacher_01/terminal_01/collected/run_01'
roles = np.load(old/'TEACHER_ROLES.npz', allow_pickle=False)
pool = roles['train_rows']
assert len(pool) == 6465 and len(roles['heldout_rows']) == 128
assert np.intersect1d(pool,roles['heldout_rows']).size == 0
map_roles = ('FIT','CHECK','FINAL')
hypotheses = ('fast','slow','common_level')
labels = ('original_oracle','C1_known_truth','C1_before_last','C1_endpoint')
assert result['labels'] == list(labels) and result['map_roles'] == list(map_roles)
block_slices = dict(wave=(0,56),charge_morphology=(56,112),peak_codes=(112,168),baseline=(168,224),joint=(224,272))
denominator, n = 272., 8
pairs = list(combinations(range(n),2))
reports, decomposition = {}, {}
max_residual_error = max_summary_error = max_identity_error = 0.
all_streams = []

for seed in range(2):
    a = np.load(run/f'seed{seed}_SAVED.npz',allow_pickle=False)
    endpoint = np.load(old/f'seed{seed}_joint_ITER11.npz',allow_pickle=False)
    factors = np.load(old/f'seed{seed}_FEATURE_MAPS.npz',allow_pickle=False)
    assert all(np.isfinite(a[key]).all() for key in a.files)
    np.testing.assert_array_equal(a['pool'],pool)
    np.testing.assert_array_equal(a['before'],endpoint['theta'])
    np.testing.assert_array_equal(a['endpoint'],endpoint['endpoint'])
    truth = np.zeros(54)
    truth[:24].reshape(6,4)[:,:3] = np.array([[.18,.06,-.04],[.22,-.04,.05],[-.12,.05,.03],[.25,.06,-.05],[.18,-.03,.05],[-.12,.04,.02]])
    truth[24:45] = np.array([[.18,0,0,0,0,0],[.09,.13,0,0,0,0],[-.05,.06,.14,0,0,0],[-.10,.04,.06,.18,0,0],[.05,-.04,.03,.08,.14,0],[.03,.02,-.04,.03,.02,.12]])[np.tril_indices(6)]
    truth[45:53] = np.linspace(-.15,.15,8)
    truth[-1] = .60
    np.testing.assert_allclose(a['truth'],truth,atol=3e-17,rtol=0)
    assert a['source_feature_means'].shape == (8,4,3,3,272)
    assert a['target_feature_means'].shape == (8,3,3,272)
    assert a['residuals'].shape == (8,4,3,816)
    assert a['parent_orders'].shape == (8,2,6465)
    for repeat in range(8):
        for side in range(2):
            order = torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['order_seed']+100*seed+2*repeat+side)).numpy()
            np.testing.assert_array_equal(a['parent_orders'][repeat,side],pool[order])
            base = p['bank_seed']+seed*100000+repeat*1000+side*100
            assert a['bank_base_seeds'][repeat,side] == base
            all_streams.extend(base+3*chunk+offset for chunk in range(13) for offset in range(3))
    reports[str(seed)], decomposition[str(seed)] = {}, {}
    for ri, role in enumerate(map_roles):
        white = np.empty((8,4,3,272))
        for hi,h in enumerate(hypotheses):
            factor = a[role+'_'+h+'_cholesky']
            np.testing.assert_array_equal(factor,factors[role+'_'+h+'_cholesky'])
            raw = a['source_feature_means'][:,:,ri,hi]-a['target_feature_means'][:,None,ri,hi]
            white[:,:,hi] = solve_triangular(factor,raw.reshape(-1,272).T,lower=True).T.reshape(8,4,272)
        reconstructed = white.reshape(8,4,816)/np.sqrt(3.)
        max_residual_error = max(max_residual_error,float(np.max(np.abs(reconstructed-a['residuals'][:,:,ri]))))
        np.testing.assert_allclose(reconstructed,a['residuals'][:,:,ri],atol=2e-14,rtol=2e-11)
        reports[str(seed)][role] = {}
        all_loo = []
        for li,label in enumerate(labels):
            v = reconstructed[:,li]
            pair_values = np.array([np.dot(v[i],v[j])/denominator for i,j in pairs])
            loo = np.array([np.mean([value for (i,j),value in zip(pairs,pair_values,strict=True) if i != k and j != k]) for k in range(8)])
            u = float(pair_values.mean())
            noise = sum(np.square(v[i]-v[j]).sum() for i,j in pairs)/(n*n*(n-1)*denominator)
            recomputed = dict(single_four_draw_risks=np.square(v).sum(1)/denominator,
                pooled_sixteen_draw_risks=[np.square(v[:4].mean(0)).sum()/denominator,np.square(v[4:].mean(0)).sum()/denominator],
                pooled_thirtytwo_draw_risk=np.square(v.mean(0)).sum()/denominator,
                estimated_pooled_MC_noise=noise,independent_cross_product_risk=u,
                jackknife_standard_error=np.sqrt((n-1)/n*np.square(loo-loo.mean()).sum()),
                leave_one_bank_out_cross_product_risks=loo)
            for key,value in recomputed.items():
                observed = result['reports'][str(seed)][role][label][key]
                max_summary_error = max(max_summary_error,float(np.max(np.abs(np.asarray(value)-observed))))
                np.testing.assert_allclose(value,observed,atol=2e-14,rtol=2e-10)
            np.testing.assert_allclose(recomputed['pooled_thirtytwo_draw_risk'],u+noise,atol=1e-17,rtol=1e-12)
            reports[str(seed)][role][label] = dict(U_risk=u,pooled_MC=noise,jackknife_SE=float(recomputed['jackknife_standard_error']))
            all_loo.append(loo)
        step_loo = all_loo[3]-all_loo[2]
        step_change = reports[str(seed)][role]['C1_endpoint']['U_risk']-reports[str(seed)][role]['C1_before_last']['U_risk']
        reports[str(seed)][role]['last_step'] = dict(U_change=step_change,paired_jackknife_SE=float(np.sqrt((n-1)/n*np.square(step_loo-step_loo.mean()).sum())))
        origin = reconstructed[:,0]
        family = reconstructed[:,1]-origin
        shifts = [reconstructed[:,2]-reconstructed[:,1],reconstructed[:,3]-reconstructed[:,1]]
        attribution = {}
        for label,shift,li in zip(('before_last','endpoint'),shifts,(2,3),strict=True):
            terms = np.stack((origin,family,shift),1)
            np.testing.assert_allclose(terms.sum(1),reconstructed[:,li],atol=2e-17,rtol=1e-12)
            gram = sum(terms[i]@terms[j].T+terms[j]@terms[i].T for i,j in pairs)/(n*(n-1)*denominator)
            total = reports[str(seed)][role][labels[li]]['U_risk']
            max_identity_error = max(max_identity_error,abs(float(gram.sum())-total))
            np.testing.assert_allclose(gram.sum(),total,atol=2e-17,rtol=1e-12)
            attribution[label] = dict(term_order=['original_oracle','family_at_truth','parameter_change_from_C1_truth'],
                full_bilinear_matrix=gram.tolist(),sum_all_entries=float(gram.sum()),
                same_family_parameter_discrepancy=float(gram[2,2]),family_parameter_interaction=float(2*gram[1,2]),
                cancellation_cosine=float(gram[1,2]/np.sqrt(gram[1,1]*gram[2,2])))
        block_terms = np.zeros((8,5,3,272))
        marginal = {}
        for hi,h in enumerate(hypotheses):
            factor = a[role+'_'+h+'_cholesky']
            raw_family = a['source_feature_means'][:,1,ri,hi]-a['source_feature_means'][:,0,ri,hi]
            variance = np.square(factor).sum(1)
            marginal[h] = {}
            for bi,(block,(lo,hi_index)) in enumerate(block_slices.items()):
                only_block = np.zeros_like(raw_family)
                only_block[:,lo:hi_index] = raw_family[:,lo:hi_index]
                block_terms[:,bi,hi] = solve_triangular(factor,only_block.T,lower=True).T/np.sqrt(3.)
                scaled = raw_family[:,lo:hi_index]/np.sqrt(variance[lo:hi_index])
                marginal[h][block] = float(np.mean([np.dot(scaled[i],scaled[j])/(hi_index-lo) for i,j in pairs]))
        block_terms = block_terms.reshape(8,5,816)
        np.testing.assert_allclose(block_terms.sum(1),family,atol=2e-14,rtol=2e-11)
        block_gram = sum(block_terms[i]@block_terms[j].T+block_terms[j]@block_terms[i].T for i,j in pairs)/(n*(n-1)*denominator)
        family_U = attribution['endpoint']['full_bilinear_matrix'][1][1]
        np.testing.assert_allclose(block_gram.sum(),family_U,atol=2e-17,rtol=1e-11)
        decomposition[str(seed)][role] = dict(parameter_attribution=attribution,
            family_block_order=list(block_slices),family_full_bilinear_matrix=block_gram.tolist(),
            family_block_signed_contributions=block_gram.sum(1).tolist(),
            family_total_U=float(block_gram.sum()),diagonal_normalized_block_diagnostics=marginal)

assert len(all_streams) == len(set(all_streams)) == 1248
expected = dict(checkpoint_loads=6,TRAIN_archives_decoded=1,PCA_archives_decoded=1,old_A_calls=52,old_A_rows=6593,
    M0_head_calls=104,M0_head_rows=13186,allocated_banks=2+16*2*13,shape_standard_normal_values=8*(2*6465+16*2*6465*4),
    noise005_calls=418,noise005_rows=2*6465+16*2*6465*4,extension_rows=3*(2*6465+16*2*6465*4),
    joint_response_calls=2+16*5*13,joint_response_rows=2*6465+16*5*6465*4,
    generated_observation_calls=3*(2+16*5*13),generated_observation_rows=3*(2*6465+16*5*6465*4),
    feature_maps_constructed=18,full_fit_parent_banks=32,fixed_endpoint_evaluations=64,independent_bank_pairs=16)
assert result['execution_counts'] == expected
for key,value in p['operation_budget'].items():
    assert expected.get(key,0) == value, key
assert result['parameter_updates'] == result['REAL_response_updates'] == 0
assert not result['scientific_acceptance'] and not result['report_parent_rows_used'] and not result['paired_conditional_labels_used_for_fitting']
review = dict(status='PASS_SAVED_ESTIMATOR_AUDIT_NO_SCIENTIFIC_ACCEPTANCE',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=744,local_source_files_rehashed=local_sources,remote_only_source_files_from_native_manifest=remote_sources,
    pinned_inputs=91,output_files_rehashed=len(remote['outputs']),archive_verified=True,native_exit_code=0,
    independent_parent_permutation_replays=32,feature_factors_exact=18,independent_stream_seeds=1248,
    residual_vectors_recomputed=192,summary_views_recomputed=24,max_residual_error=max_residual_error,
    max_summary_error=max_summary_error,max_decomposition_identity_error=max_identity_error,execution_counts=expected,
    reports=reports,raw_generator_replay=False,negative_U_estimates_retained=True,eight_bank_jackknife_descriptive_only=True,
    all_maps_exposed_diagnostic_only=True,parameter_updates=0,scientific_acceptance=False,elapsed_seconds=time.monotonic()-started)
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
diagnosis = dict(status='SAVED_ALGEBRAIC_DIAGNOSIS_COMPLETE',UTC=review['UTC'],source_review='TERMINAL_REVIEW.json',
    definition='Symmetric cross-repeat bilinear products; sum of all matrix entries equals the full whitened discrepancy. Signed block allocations include cross terms and are descriptive, not independent causal effects.',
    reports=decomposition,source_common_random_numbers_across_labels=True,target_independent_of_source=True,
    cross_repeat_independence_required=True,known_truth_is_diagnostic_only=True,parameter_updates=0,scientific_acceptance=False)
(out/'SAVED_FEATURE_DECOMPOSITION.json').write_text(json.dumps(diagnosis,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:review[k] for k in ('status','UTC','max_residual_error','max_summary_error','max_decomposition_identity_error','elapsed_seconds')}))
for seed,views in decomposition.items():
    for role,view in views.items():
        before,after=view['parameter_attribution']['before_last'],view['parameter_attribution']['endpoint']
        print(json.dumps(dict(seed=seed,role=role,family_U=view['family_total_U'],same_family_before=before['same_family_parameter_discrepancy'],
            same_family_after=after['same_family_parameter_discrepancy'],interaction=after['family_parameter_interaction'],
            cosine=after['cancellation_cosine'],blocks=dict(zip(view['family_block_order'],view['family_block_signed_contributions'],strict=True)),last_step=reports[seed][role]['last_step'])))
