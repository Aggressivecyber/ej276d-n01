from pathlib import Path
import ast

root=Path('E:/GAN/research/N01')
here=Path(__file__).resolve().parent
source=(root/'receipts/joint_response_01_20261020/multibank_m0_01/terminal_01/AUDIT_MULTIBANK.py').read_text(encoding='utf-8')
source=source.replace('import torch','import torch\nfrom scipy.optimize import minimize')
source=source.replace("HERE = Path(__file__).resolve().parent\nOUT = HERE / 'run_01'", "BASE = Path(__file__).resolve().parent\nHERE = BASE/'terminal_01'\nCOLLECTED=HERE/'collected'\nOUT = COLLECTED/'run_01'")
source=source.replace('joint_response_01_multibank_m0_20261020.json','joint_response_01_original_draw_precision_20261020.json')
source=source.replace("raw = (HERE / name).read_bytes()", "raw = (COLLECTED / name).read_bytes()")
start=source.index('for name, digest in p[\'source_pins\'].items():')
end=source.index("assert result['elapsed_seconds']",start)
source=source[:start]+'''manifest=json.loads((BASE/'SOURCE_MANIFEST.json').read_bytes())
inputs=json.loads((BASE/'INPUT_MAP.json').read_bytes())
local_count=0
for name,record in manifest['files'].items():
    if (ROOT/name).is_file():
        raw=(ROOT/name).read_bytes()
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],name
        local_count+=1
assert terminal['source_files_verified']==len(manifest['files'])==805
for name,digest in p['source_pins'].items():
    assert manifest['files'][name]['sha256']==digest,name
    if name in inputs:assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()==digest,name
complete=json.loads((OUT/'COMPLETE.json').read_bytes())
assert complete['result_sha256']==hashlib.sha256((OUT/'RESULT.json').read_bytes()).hexdigest()
assert terminal['processes']==[] and terminal['complete_present'] and not terminal['failed_present']
raw=(HERE/'terminal.tar').read_bytes()
assert len(raw)==terminal['archive_bytes'] and hashlib.sha256(raw).hexdigest()==terminal['archive_sha256']
'''+source[end:]
source=source.replace("native_rng = json.loads((HERE / 'NATIVE_ZERO_RNG_REVIEW.json').read_text())", "old_out=ROOT/'receipts/joint_response_01_20261020/multibank_m0_01/terminal_01'\nnative_rng = json.loads((old_out/'NATIVE_ZERO_RNG_REVIEW.json').read_text())")
source=source.replace("assert native_rng['zero_archive_sha256'] == hashlib.sha256((OUT / 'M0_ZERO_IDENTITY.npz').read_bytes()).hexdigest()", "assert native_rng['zero_archive_sha256'] == hashlib.sha256((old_out/'run_01/M0_ZERO_IDENTITY.npz').read_bytes()).hexdigest()\nold_zero=np.load(old_out/'run_01/M0_ZERO_IDENTITY.npz',allow_pickle=False)\nassert set(old_zero.files)==set(zero)\nfor key in old_zero.files:np.testing.assert_array_equal(zero[key],old_zero[key])")
source=source.replace('comparisons = []','comparisons = []\ndraw_comparisons=[]\nmax_solver_difference=0.')
source=source.replace("    theta = a['theta']", "    low=np.load(Path(inputs[p['low_draw_arrays'][seed]]),allow_pickle=False)\n    theta = a['theta']")
source=source.replace("        assert report['native_projection_contexts'][label]['seed']", """        for suffix in ('source_parents','real_rows','cholesky'):
            np.testing.assert_array_equal(a[label+'_'+suffix],low[label+'_'+suffix])
        np.testing.assert_allclose(a[label+'_replayed_low_residual'],low[label+'_baseline_residual'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a[label+'_replayed_low_native'],low[label+'_baseline_native'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(a[label+'_random_seeds'],[seed_base+seed*100000+1000,
            p['extension_bank_seed']+1000000*seed+10000*list(p['role_seed_bases']).index(label)])
        counts['low_draw_center_replays_recomputed']+=1
        assert report['native_projection_contexts'][label]['seed']""")
source=source.replace("    accepted = []", """    assert report['draws_per_parent']==32 and report['nested_low_draw_replays']==5
    for label,key,value in (('joint','original_jacobian',J),('native','original_native_jacobian',N)):
        prior=low[key]
        values=dict(relative_difference=float(np.linalg.norm(value-prior)/np.linalg.norm(prior)),
            cosine=float(value.flatten()@prior.flatten()/(np.linalg.norm(value)*np.linalg.norm(prior))))
        for key2,value2 in values.items():np.testing.assert_allclose(report['low_to_high_jacobian'][label][key2],value2,rtol=1e-11,atol=1e-12)
    np.testing.assert_array_equal(a['frozen_low_candidate'],low['uncommitted_candidate'])
    for label,check in report['frozen_low_candidate_checks'].items():
        risk_records.append((label,a['low_candidate_'+label+'_actual_residual'],check['risks']))
        checks.append(('frozen_low_candidate',label,a['low_candidate_'+label+'_actual_native'],check))
    draw_comparisons.append(dict(seed=seed,jacobians=report['low_to_high_jacobian']))
    accepted = []""")
source=source.replace("        macro_slacks = []", "        macro_slacks = []\n        operators,offsets=[],[]")
source=source.replace("                    macro_slacks.append(1 - risk / cap)", """                    macro_slacks.append(1-risk/cap)
                    normalization=np.sqrt(56*cap)
                    offsets.append(raw0[take]/np.sqrt(variance[take])/normalization)
                    raw_j=L@(J[sl]*3*radius)
                    operators.append(raw_j[take]/np.sqrt(variance[take])[:,None]/normalization)
""")
source=source.replace("        assert slack >= -1e-6", """        assert slack >= -1e-6
        scaled=J*radius
        gram=scaled.T@scaled
        hh=gram+p['solver']['ridge_fraction']*np.diag(gram).mean()*np.eye(54)
        gradient=scaled.T@r
        aa,bb=np.stack(operators),np.stack(offsets)
        nn=N*radius/n[:,None]
        def objective(u):return float((.5*u@hh@u+gradient@u)/(r@r))
        def gradient_at(u):return (hh@u+gradient)/(r@r)
        def constraints(u):return np.r_[1-np.square(bb+aa@u).sum(1),-nn@u,1-u@u]
        def constraint_jacobian(u):return np.vstack((-2*np.einsum('mni,mn->mi',aa,bb+aa@u),-nn,-2*u))
        solution=minimize(objective,np.zeros(54),jac=gradient_at,method='SLSQP',
            bounds=list(zip((a['lower']-theta)/radius,(a['upper']-theta)/radius)),
            constraints={'type':'ineq','fun':constraints,'jac':constraint_jacobian},
            options={'maxiter':p['constrained_solver']['max_iterations'],'ftol':p['constrained_solver']['ftol'],'disp':False})
        assert solution.success,solution.message
        difference=float(np.max(np.abs(solution.x*radius-step)))
        assert difference < 2e-7,(seed,i,difference)
        max_solver_difference=max(max_solver_difference,difference)
        counts['independent_joint_solutions']+=1
""")
start=source.index("    assert accepted == [len(report['trials']) - 1]")
end=source.index('    for label, vector, recorded in risk_records:',start)
source=source[:start]+'''    if accepted:
        assert accepted==[len(report['trials'])-1] and report['selected_trial']==accepted[0]
        assert report['stop_reason']=='first_all_FIT_CHECK_pass'
        np.testing.assert_array_equal(a['uncommitted_candidate'],a[f'trial{accepted[0]}_proposal'])
        check=report['fresh_confirmation'];assert report['fresh_pass']==check['accepted']
        risk_records.append(('FRESH',a['FRESH_actual_residual'],check['risks']))
        checks.append((accepted[0],'FRESH',a['FRESH_actual_native'],check))
        counts['FRESH_checks']+=1
    else:
        assert report['selected_trial'] is None and not report['fresh_pass']
        assert report['stop_reason'] in ('trial_budget','zero_joint_direction')
    assert result['summaries'][str(seed)]['fresh_pass']==report['fresh_pass']
'''+source[end:]
source=source.replace("assert trust['denominator'] == 4096", "assert trust['denominator'] == 32768")
source=source.replace('max(4, .005 * 4096)','max(4, .005 * 32768)')
start=source.index("c = result['execution_counts']")
source=source[:start]+'''c=result['execution_counts']
assert counts['joint_candidates']==c['joint_candidate_trials']
assert counts['independent_joint_solutions']<=c['joint_direction_solves']==c['joint_direction_solve_attempts']<=6
assert counts['secant_corrections_recomputed']<=c['secant_matrix_corrections']<=4
assert counts['FRESH_checks']==c.get('FRESH_candidate_checks',0)
assert counts['FD_columns_recomputed']==c['finite_difference_columns']==108 and c['FIT_bank_finite_difference_columns']==324
high_calls=10+2*3*108+4*c['joint_candidate_trials']+c.get('FRESH_candidate_checks',0)+4
assert c['joint_response_calls']==2+10+high_calls
assert c['joint_response_rows']==256+10*4096+high_calls*32768
assert c['generated_observation_calls']==3*(10+high_calls)
assert c['generated_observation_rows']==3*(10*4096+high_calls*32768)
assert c['native_component_rows']==3*(10*8192+high_calls*36864)
assert c['allocated_banks']==c['noise005_calls']==20
assert c['noise005_rows']==327680 and c['extension_rows']==983040 and c['shape_standard_normal_values']==2623488
assert c['low_draw_center_replays_verified']==c['nested_banks_verified']==counts['low_draw_center_replays_recomputed']==10
assert c['frozen_low_candidate_checks']==4
assert result['two_seed_FRESH_pass']==all(v['fresh_pass'] for v in result['summaries'].values())
assert not result['scientific_acceptance'] and result['committed_parameter_updates']==0
review=dict(status='PASS_SAVED_ORIGINAL_DRAW_PRECISION_AUDIT',UTC=datetime.now(timezone.utc).isoformat(),
    source_files_verified=terminal['source_files_verified'],local_source_hashes=local_count,
    source_pins=len(p['source_pins']),outputs_rehashed=len(terminal['outputs']),output_bytes=sum(v['bytes'] for v in terminal['outputs'].values()),
    config_sha256=result['config_sha256'],result_sha256=complete['result_sha256'],native_exit=terminal['native_exit'],
    elapsed_seconds=result['elapsed_seconds'],execution_counts=c,audit_counts=dict(counts),
    comparisons=comparisons,jacobian_comparisons=draw_comparisons,max_independent_solver_delta_error=max_solver_difference,
    two_seed_FRESH_pass=result['two_seed_FRESH_pass'],committed_parameter_updates=0,scientific_acceptance=False,
    evidence_limit='Saved arrays independently verify directions, predictions, risks, gates and counters. The native generator verified nested z/noise arrays; this audit does not regenerate noise or independently recompute unsaved waveform, trust RMS and flag statistics. Historical fitting-role FRESH rows/projections are reused with 28 added draws.')
(HERE/'TERMINAL_REVIEW.json').write_text(json.dumps(review,indent=2)+'\\n',encoding='utf-8',newline='\\n')
print(json.dumps(dict(status=review['status'],audit_counts=dict(counts),two_seed_FRESH_pass=review['two_seed_FRESH_pass'],
    max_independent_solver_delta_error=max_solver_difference,fresh=[x for x in comparisons if x['label']=='FRESH'])))
'''
ast.parse(source)
(here/'AUDIT_COMPLETE.py').write_text(source,encoding='utf-8',newline='\n')
print('Saved-array audit prepared; execution awaits terminal outputs.')
