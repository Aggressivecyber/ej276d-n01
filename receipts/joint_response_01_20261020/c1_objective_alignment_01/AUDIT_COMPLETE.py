"""Independent saved feature audit with paired rankings and full covariance terms."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy.linalg import solve_triangular
import torch

started = time.monotonic()
root = Path('E:/GAN/research/N01')
here = Path(__file__).resolve().parent
out = here/'terminal_01'
run = out/'collected/run_01'
config = 'configs/joint_response_01_c1_objective_alignment_20261020.json'
p = json.loads((root/config).read_bytes())
manifest = json.loads((here/'SOURCE_MANIFEST.json').read_bytes())
remote = json.loads((out/'REMOTE_TERMINAL_VERIFICATION.json').read_bytes())
result = json.loads((run/'RESULT.json').read_bytes())
complete = json.loads((run/'COMPLETE.json').read_bytes())
assert remote['native_exit']['exit_code'] == 0 and remote['processes'] == []
assert remote['complete_present'] and remote['result_present'] and not remote['failed_present']
assert complete['result_sha256'] == hashlib.sha256((run/'RESULT.json').read_bytes()).hexdigest()
assert result['config_sha256'] == hashlib.sha256((root/config).read_bytes()).hexdigest()
assert (out/'terminal.tar').stat().st_size == remote['archive_bytes']
assert hashlib.sha256((out/'terminal.tar').read_bytes()).hexdigest() == remote['archive_sha256']
local_sources = remote_only_sources = 0
for name, record in manifest['files'].items():
    if (root/name).is_file():
        raw = (root/name).read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], name
        local_sources += 1
    else:
        remote_only_sources += 1
assert len(manifest['files']) == remote['source_files_verified'] == 796
assert len(p['source_pins']) == 136
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
assert len(pool) == 6465 and len(roles['heldout_rows']) == 128 and not np.intersect1d(pool,roles['heldout_rows']).size
labels = ('original_oracle','C1_known_truth','C1_original_initial','C1_positive_endpoint','C1_original_endpoint')
targets = ('original','matched_C1')
map_roles = ('FIT','CHECK','FINAL')
hypotheses = ('fast','slow','common_level')
assert result['labels'] == list(labels) and result['targets'] == list(targets) and result['map_roles'] == list(map_roles)
blocks = dict(wave=(0,56),charge_morphology=(56,112),peak_codes=(112,168),baseline=(168,224),joint=(224,272))
input_map = json.loads((here/'INPUT_MAP.json').read_bytes())
n, denominator = 4, 272.
pairs = list(combinations(range(n),2))
reports, rankings, block_reports = {}, {}, {}
max_residual_error = max_summary_error = max_identity_error = 0.
streams = []
for seed in range(2):
    a = np.load(run/f'seed{seed}_SAVED.npz', allow_pickle=False)
    factors = np.load(old/f'seed{seed}_FEATURE_MAPS.npz', allow_pickle=False)
    assert all(np.isfinite(a[key]).all() for key in a.files)
    np.testing.assert_array_equal(a['pool'],pool)
    np.testing.assert_array_equal(a['initial'],np.load(old/f'seed{seed}_joint_ITER11.npz')['endpoint'])
    for label in ('positive','original'):
        observed = a['positive' if label=='positive' else 'endpoint']
        np.testing.assert_array_equal(observed,np.load(input_map[p[label+'_endpoints'][seed]])['theta'])
    truth = np.zeros(54)
    truth[:24].reshape(6,4)[:,:3] = np.array([[.18,.06,-.04],[.22,-.04,.05],[-.12,.05,.03],[.25,.06,-.05],[.18,-.03,.05],[-.12,.04,.02]])
    truth[24:45] = np.array([[.18,0,0,0,0,0],[.09,.13,0,0,0,0],[-.05,.06,.14,0,0,0],[-.10,.04,.06,.18,0,0],[.05,-.04,.03,.08,.14,0],[.03,.02,-.04,.03,.02,.12]])[np.tril_indices(6)]
    truth[45:53] = np.linspace(-.15,.15,8)
    truth[-1] = .60
    np.testing.assert_allclose(a['truth'],truth,atol=3e-17,rtol=0)
    assert a['source_feature_means'].shape == (4,5,3,3,272)
    assert a['target_feature_means'].shape == (4,2,3,3,272)
    assert a['residuals'].shape == (4,2,5,3,816)
    for repeat in range(4):
        for side in range(2):
            order = torch.randperm(len(pool),generator=torch.Generator().manual_seed(p['order_seed']+1000*seed+100*repeat+side)).numpy()
            np.testing.assert_array_equal(a['parent_orders'][repeat,side],pool[order])
            base = p['bank_seed']+1000000*seed+10000*repeat+1000*side
            assert a['bank_base_seeds'][repeat,side] == base
            streams.extend(base+3*chunk+offset for chunk in range(102) for offset in range(3))
    for ri, role in enumerate(map_roles):
        inverses = []
        for h in hypotheses:
            factor = a[role+'_'+h+'_cholesky']
            np.testing.assert_array_equal(factor,factors[role+'_'+h+'_cholesky'])
            inverses.append(solve_triangular(factor,np.eye(272),lower=True))
        for ti, target in enumerate(targets):
            raw = a['source_feature_means'][:,:,ri]-a['target_feature_means'][:,ti,None,ri]
            white = np.stack([raw[:,:,hi] @ inverse.T for hi,inverse in enumerate(inverses)],axis=2)
            reconstructed = white.reshape(4,5,816)/np.sqrt(3.)
            max_residual_error = max(max_residual_error,float(np.max(np.abs(reconstructed-a['residuals'][:,ti,:,ri]))))
            np.testing.assert_allclose(reconstructed,a['residuals'][:,ti,:,ri],atol=2e-14,rtol=2e-11)
            key = f'seed{seed}/{target}/{role}'
            reports[key], block_reports[key] = {}, {}
            all_loo, all_single, all_U = [], [], []
            for li, label in enumerate(labels):
                v = reconstructed[:,li]
                pair_values = np.array([np.dot(v[i],v[j])/denominator for i,j in pairs])
                loo = np.array([np.mean([value for (i,j),value in zip(pairs,pair_values,strict=True) if i!=k and j!=k]) for k in range(4)])
                u = float(pair_values.mean())
                single = np.square(v).sum(1)/denominator
                noise = sum(np.square(v[i]-v[j]).sum() for i,j in pairs)/(n*n*(n-1)*denominator)
                summary = dict(single_32draw_risks=single,pooled_128draw_risk=np.square(v.mean(0)).sum()/denominator,
                    independent_cross_product_risk=u,estimated_pooled_MC_noise=noise,
                    jackknife_standard_error=np.sqrt((n-1)/n*np.square(loo-loo.mean()).sum()),leave_one_bank_out_cross_product_risks=loo)
                for name,value in summary.items():
                    saved = result['reports'][str(seed)][target][role][label][name]
                    max_summary_error = max(max_summary_error,float(np.max(np.abs(np.asarray(value)-saved))))
                    np.testing.assert_allclose(value,saved,atol=2e-14,rtol=2e-10)
                np.testing.assert_allclose(summary['pooled_128draw_risk'],u+noise,atol=1e-17,rtol=1e-12)
                reports[key][label] = dict(U_risk=u,pooled_MC=float(noise),jackknife_SE=float(summary['jackknife_standard_error']))
                all_U.append(u);all_loo.append(loo);all_single.append(single)
                components = []
                for left,right in blocks.values():
                    parts = [raw[:,li,hi,left:right] @ inverse[:,left:right].T for hi,inverse in enumerate(inverses)]
                    components.append(np.concatenate(parts,axis=1)/np.sqrt(3.))
                component = np.stack(components,axis=1)
                np.testing.assert_allclose(component.sum(1),v,atol=2e-14,rtol=2e-11)
                total = component.sum(0)
                cross = (total @ total.T-np.einsum('nid,njd->ij',component,component))/(n*(n-1)*denominator)
                max_identity_error = max(max_identity_error,abs(float(cross.sum())-u))
                np.testing.assert_allclose(cross.sum(),u,atol=2e-14,rtol=2e-10)
                block_reports[key][label] = dict(block_order=list(blocks),full_cross_block_U_matrix=cross.tolist(),
                    signed_row_contributions=cross.sum(1).tolist(),total_U=float(cross.sum()))
            rankings[key] = {}
            for title,left,right in (('original_minus_positive',4,3),('original_minus_initial',4,2),('known_C1_minus_oracle',1,0)):
                loo_gap = all_loo[left]-all_loo[right]
                gaps = all_single[left]-all_single[right]
                rankings[key][title] = dict(U_difference=all_U[left]-all_U[right],
                    paired_jackknife_SE=float(np.sqrt((n-1)/n*np.square(loo_gap-loo_gap.mean()).sum())),
                    same_bank_risk_differences=gaps.tolist(),all_four_banks_negative=bool((gaps<0).all()),all_four_banks_positive=bool((gaps>0).all()))

assert len(streams)==len(set(streams))==4896
expected = {k:v for k,v in p['operation_budget'].items() if v}
assert result['execution_counts']==expected
assert result['parameter_updates']==result['REAL_response_updates']==0
assert not result['scientific_acceptance'] and not result['report_parent_rows_used']
assert result['known_truth_is_diagnostic_only'] and result['all_endpoints_frozen']
assert result['paired_source_randomness_across_labels'] and result['paired_target_randomness_across_families'] and result['source_target_independent']
review = dict(status='PASS_SAVED_OBJECTIVE_ALIGNMENT_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=796,local_source_files_rehashed=local_sources,remote_only_source_files=remote_only_sources,pinned_inputs=136,
    output_files_rehashed=len(remote['outputs']),archive_verified=True,native_exit=remote['native_exit'],
    independent_parent_permutation_replays=16,feature_factors_exact=18,independent_stream_seeds=4896,
    residual_vectors_recomputed=240,summary_views_recomputed=60,max_residual_error=max_residual_error,max_summary_error=max_summary_error,
    max_cross_block_identity_error=max_identity_error,execution_counts=expected,reports=reports,paired_rankings=rankings,
    raw_generator_replay=False,four_bank_jackknife_descriptive_only=True,negative_U_estimates_retained=True,
    parameter_updates=0,scientific_acceptance=False,elapsed_seconds=time.monotonic()-started)
(out/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'SAVED_OBJECTIVE_DECOMPOSITION.json').write_text(json.dumps(dict(block_reports=block_reports,paired_rankings=rankings,
    full_covariance_cross_terms_retained=True,parameter_updates=0,scientific_acceptance=False),indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:review[k] for k in ('status','UTC','output_files_rehashed','max_residual_error','max_summary_error','max_cross_block_identity_error','elapsed_seconds')}))
for key,value in rankings.items():
    print(json.dumps(dict(view=key,ranking=value)))
