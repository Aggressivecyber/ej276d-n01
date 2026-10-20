from pathlib import Path
import ast

root = Path('E:/GAN/research/N01')
source = (root/'scripts/diagnose_joint01_multibank_m0.py').read_text(encoding='utf-8')
source = source.replace('"""Three-bank joint directions from exact M0; bounded checks and no updates."""',
    '"""Nested 4-to-32-draw original-generator direction check; no updates."""')
source = source.replace('from diagnose_joint01_secant_trust import hard_check\n', '')
source = source.replace('load_joint_setup, progress_file, random_bank, prediction,', 'load_joint_setup, progress_file, random_bank, prediction as raw_prediction,')
source = source.replace('from run_joint_response01_teacher import check_budget, draw_parents',
    'from run_joint_response01_teacher import check_budget, draw_parents, accept_joint')
source = source.replace("FIT_LABELS = ('FIT0', 'FIT1', 'FIT2')", '''FIT_LABELS = ('FIT0', 'FIT1', 'FIT2')


def prediction(model, theta, bank, q_floor, counts):
    active = dict(rows=bank['rows'].to(theta.device), z=bank['z'].to(theta.device),
        noise={h: bank['noise'][h].to(theta.device) for h in HYPOTHESES})
    return raw_prediction(model, theta, active, q_floor, counts)


def hard_check(p, model, proposal, bank, target, features, maps, baseline,
               old_risk, old_native, anchor, setup, counts):
    obs = prediction(model, proposal, bank, setup['loss'].q_floor, counts)
    means = feature_means(obs, maps)
    residual, risk = residual_vector(means, features, maps), risks(means, features, maps)
    native = native_values(obs, target, setup, counts)
    scale = model.cache['A'][bank['rows'].to(proposal.device)].abs().amax(1)
    relative = torch.cat([((obs[h]['carrier_input'] - baseline[h]['carrier'].to(proposal.device)).square().mean(1).sqrt()/scale)
        for h in HYPOTHESES])
    changes = {h: {key: int(value.sum())-baseline[h]['flags'][key] for key,value in obs[h]['flags'].items()} for h in HYPOTHESES}
    trust = dict(mean_relative_RMS=float(relative.mean()), p99_relative_RMS=float(torch.quantile(relative,.99)),
        max_relative_RMS=float(relative.max()), flag_count_changes=changes, denominator=len(bank['rows']))
    trust['pass'] = (trust['mean_relative_RMS'] <= p['solver']['output_TR_mean_RMS_fraction']
        and trust['p99_relative_RMS'] <= p['solver']['output_TR_p99_RMS_fraction']
        and max(v for d in changes.values() for v in d.values()) <= max(4,p['solver']['flag_increase_fraction']*len(bank['rows'])))
    ratios, anchored = (native/old_native).reshape(3,3), (native/anchor).reshape(3,3)
    native_pass = all(bool((value[:,:2] <= p['native_guard']['wave_QPSD_factor']).all()
        and (value[:,2] <= p['native_guard']['noise_factor']).all()) for value in (ratios,anchored))
    joint_pass = accept_joint(old_risk,risk,trust,p['batch']['target_rows'],p)
    return residual,native,dict(risks=risk,native=native.tolist(),native_ratios=ratios.tolist(),
        initial_anchor_ratios=anchored.tolist(),native_pass=native_pass,trust=trust,
        original_joint_accept=joint_pass,accepted=native_pass and joint_pass)
''')
source = source.replace("theta = model.initial()\n        banks, targets", "theta = model.initial()\n        saved = np.load(ROOT/p['low_draw_arrays'][seed_index],allow_pickle=False)\n        banks, targets")
old = """            banks[label] = random_bank(setup, parents, 4, seed + 1000, counts)
            targets[label] = select_real(real, real_rows)
            features[label] = feature_means({h: targets[label] for h in HYPOTHESES}, maps[label])
            baseline[label] = prediction(model, theta, banks[label], setup['loss'].q_floor, counts)
            means = feature_means(baseline[label], maps[label])
            old_risks[label] = risks(means, features[label], maps[label])
            residuals[label] = residual_vector(means, features[label], maps[label])
            old_native[label] = native_values(baseline[label], targets[label], role_setup[label], counts)
"""
new = """            np.testing.assert_array_equal(arrays[label+'_source_parents'],saved[label+'_source_parents'])
            np.testing.assert_array_equal(arrays[label+'_real_rows'],saved[label+'_real_rows'])
            np.testing.assert_array_equal(arrays[label+'_cholesky'],saved[label+'_cholesky'])
            targets[label] = select_real(real,real_rows)
            features[label] = feature_means({h:targets[label] for h in HYPOTHESES},maps[label])
            low = random_bank(setup,parents,4,seed+1000,counts)
            low_obs = prediction(model,theta,low,setup['loss'].q_floor,counts)
            low_residual = residual_vector(feature_means(low_obs,maps[label]),features[label],maps[label])
            low_native = native_values(low_obs,targets[label],role_setup[label],counts)
            np.testing.assert_allclose(low_residual.cpu().numpy(),saved[label+'_baseline_residual'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(low_native.cpu().numpy(),saved[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
            arrays[label+'_replayed_low_residual'],arrays[label+'_replayed_low_native'] = low_residual.cpu().numpy(),low_native.cpu().numpy()
            counts['low_draw_center_replays_verified'] += 1
            del low_obs
            extension_seed = p['extension_bank_seed']+1000000*seed_index+10000*list(p['role_seed_bases']).index(label)
            extra = random_bank(setup,parents,p['batch']['draws_per_parent']-4,extension_seed,counts)
            draws = p['batch']['draws_per_parent']
            banks[label] = dict(rows=parents.repeat_interleave(draws).cpu(),
                z=torch.cat((low['z'].reshape(1024,4,8),extra['z'].reshape(1024,draws-4,8)),1).reshape(-1,8).cpu(),
                noise={h:torch.cat((low['noise'][h].reshape(1024,4,384),extra['noise'][h].reshape(1024,draws-4,384)),1).reshape(-1,384).cpu() for h in HYPOTHESES})
            np.testing.assert_array_equal(banks[label]['z'].reshape(1024,draws,8)[:,:4].numpy(),low['z'].cpu().numpy().reshape(1024,4,8))
            for h in HYPOTHESES:
                np.testing.assert_array_equal(banks[label]['noise'][h].reshape(1024,draws,384)[:,:4].numpy(),low['noise'][h].cpu().numpy().reshape(1024,4,384))
            arrays[label+'_random_seeds'] = np.array([seed+1000,extension_seed],dtype=np.int64)
            counts['nested_banks_verified'] += 1
            del low,extra
            current = prediction(model,theta,banks[label],setup['loss'].q_floor,counts)
            means = feature_means(current,maps[label])
            old_risks[label] = risks(means,features[label],maps[label])
            residuals[label] = residual_vector(means,features[label],maps[label])
            old_native[label] = native_values(current,targets[label],role_setup[label],counts)
            baseline[label] = {h:dict(carrier=current[h]['carrier_input'].cpu(),
                flags={key:int(value.sum()) for key,value in current[h]['flags'].items()}) for h in HYPOTHESES}
            del current
"""
assert source.count(old) == 1
source = source.replace(old,new)
source = source.replace("arrays['original_jacobian'], arrays['original_native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()",
"""arrays['original_jacobian'], arrays['original_native_jacobian'] = J.cpu().numpy().copy(), N.cpu().numpy().copy()
        report['draws_per_parent'] = p['batch']['draws_per_parent']
        report['nested_low_draw_replays'] = 5
        report['low_to_high_jacobian'] = {}
        for label,key,value in (('joint','original_jacobian',J),('native','original_native_jacobian',N)):
            low_j = torch.as_tensor(saved[key],device=device)
            report['low_to_high_jacobian'][label] = dict(relative_difference=float((value-low_j).norm()/low_j.norm()),
                cosine=float((value.flatten()@low_j.flatten())/(value.norm()*low_j.norm())))
        report['frozen_low_candidate_checks'] = {}
        low_candidate = torch.as_tensor(saved['uncommitted_candidate'],device=device)
        arrays['frozen_low_candidate'] = low_candidate.cpu().numpy()
        for label in ('CHECK','FRESH'):
            ar,an,ck = hard_check(p,model,low_candidate,banks[label],targets[label],features[label],maps[label],baseline[label],
                old_risks[label],old_native[label],old_native[label],role_setup[label],counts)
            arrays['low_candidate_'+label+'_actual_residual'],arrays['low_candidate_'+label+'_actual_native'] = ar.cpu().numpy(),an.cpu().numpy()
            report['frozen_low_candidate_checks'][label] = ck
            counts['frozen_low_candidate_checks'] += 1
        saved.close()
""")
source = source.replace("'multibank_M0_joint_Jacobian'", "'original_precision_joint_Jacobian'")
source = source.replace("                    bank_n.append(native_values(obs, targets[label], role_setup[label], counts))",
    "                    bank_n.append(native_values(obs, targets[label], role_setup[label], counts))\n                    del obs")
source = source.replace("            arrays[label + '_baseline_native'] = old_native[label].cpu().numpy()",
    "            arrays[label + '_baseline_native'] = old_native[label].cpu().numpy()\n            progress_file(out,counts,'original_precision_bank_ready',seed=seed_index,role=label)")
source = source.replace("scope='Combined M0-start and three-bank method check, not an isolated causal ablation of either change.'",
    "scope='Nested 4-to-32-draw comparison at frozen M0 with unchanged source/REAL rows, joint/native projections, generator and solver.'")
source = source.replace("FRESH_role='New fitting-role records/random library and projections; not full-pipeline unseen validation.'",
    "FRESH_role='Original fitting-role records/projections with 28 new draws per parent plus four replayed draws; no full-pipeline unseen claim.'")
ast.parse(source)
(root/'scripts/diagnose_joint01_original_draw_precision.py').write_text(source,encoding='utf-8',newline='\n')
print('Original 32-draw diagnostic built; AST passed.')
