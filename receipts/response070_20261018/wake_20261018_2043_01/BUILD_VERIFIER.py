from pathlib import Path
import ast
r=Path('E:/GAN/research/N01');w=Path(__file__).parent;s=(r/'receipts/response070_20261018/deployment_01/REGISTER_STARTUP.py').read_text()
head=s[:s.index("v=read(w/'STARTUP_RAW.json')")].replace('import base64,copy,','import ')
head+='''v=read(w/'REMOTE_STATUS.json');p=read(r/'configs/response070_20261018.json');run=w/'collected/run_01';files={'run_01/'+x.name:read(x) for x in run.glob('*.json')}
collection=read(w/'COLLECTION_VERIFICATION.json');exit_receipt=read(w/'collected/control_01/exit_receipt.json');result=files['run_01/RESULT.json'];complete=files['run_01/COMPLETE.json']
assert exit_receipt['exit_code']==0 and exit_receipt['reason'] is None and not (run/'FAILED.json').exists()
assert complete['result_sha256']==sha(run/'RESULT.json') and result['run_id']==p['run_id'] and result['execution_counts']==p['operation_bounds']
assert result['frozen_A0_M0_005_verified'] and result['all_TRAIN_SOURCE_parents']==6593 and result['all_TRAIN_REAL_rows']==21047 and result['all_parent_count_weights_preserved']
assert result['confirmation_banks_opened']==0 and not result['DEV_opened'] and not result['mean_stage_accepted'] and not result['full_random_response_acceptance']
assert not any(x['present'] for x in v['identities'].values())
assert result['elapsed_seconds']<780 and exit_receipt['peak_rss_bytes']<12*1024**3 and collection['run_bytes']<64*1024**2
assert p['contract']==read(r/'configs/response039_20261018.json')['contract']
code=(r/'scripts/run_response070.py').read_text();assert code.index("if dict(progress)!=p['operation_bounds']:")<code.index('model.assert_frozen_unchanged(deep=True)')<code.index("r.write_json(out/'RESULT.json'")
arrays={x.name:dict(np.load(x,allow_pickle=False)) for x in run.glob('*.npz')};assert len(arrays)==72
expected=read(r/p['replay_file'])['reports']
'''
a=s.index("prior=r/'receipts/response069");b=s.index('inputs=dict(np.load',a);body=s[a:b]
init=next(line for line in body.splitlines() if line.startswith('coupling_formula_error='));body=body.replace(init+'\n','')
queryinit='query_error=0.;amplitude_scale_error=0.;knot_error=0.;split_mean_error=0.;split_random_error=0.;seam_error=0.';assert queryinit in body;body=body.replace(queryinit+'\n','')
body=body.replace("f'seed0_","f'seed{seed}_").replace("'seed0_","f'seed{seed}_").replace("p['coupling_example_parent_rows']['0']","p['coupling_example_parent_rows'][str(seed)]")
head+=init+'\n'+queryinit+'\nfor seed in range(2):\n    m0=files[f"run_01/seed{seed}_M0.json"];assert m0==expected[str(seed)]["M0"]\n    assert files[f"run_01/seed{seed}_LEVEL069.json"]["metrics"]==expected[str(seed)]["LEVELWARP"]["metrics"]\n'
head+=''.join('    '+line+'\n' for line in body.splitlines())
c=s.index('assert len(profile_audit)',b);profile=s[b:c]
profile=profile.replace("labels=['REAL','seed0_LEVEL069_fast','seed0_LEVEL069_intrinsic','seed0_SPLITWARP_fast','seed0_SPLITWARP_intrinsic']","labels=['REAL']+[f'seed{seed}_{case}_{kind}' for seed in range(2) for case in ('LEVEL069','SPLITWARP') for kind in ('fast','intrinsic')]")
profile=profile.replace("arrays['seed0_LEVEL069_fast_NATIVE_ROWS.npz']","arrays[label[:5]+'_LEVEL069_fast_NATIVE_ROWS.npz']")
head+=profile+'assert len(profile_audit)==9 and sum(x["rows"] for x in profile_audit)==73791 and sum(x["group_view_cells"] for x in profile_audit)==272\nassert max(max_moment_error,max_linear_error,max_cov_error,max_parity_error)<1e-10\nassert coupling_examples_verified==156 and max(coupling_formula_error,coupling_scalar_error,coupling_feature_error,amplitude_scale_error)<1e-12 and query_error<1e-10 and knot_error<1e-12\n'
a=s.index('from scipy.stats import wasserstein_distance');b=s.index('for name,digest in p[',a);cdf=s[a:b]
c=cdf.index("for case in ('LEVEL069','SPLITWARP'):");tail=cdf.index('assert len(W1audit)==54',c)
head+=cdf[:c]+'for seed in range(2):\n'+''.join('    '+line+'\n' for line in cdf[c:tail].replace('seed0_','seed{seed}_').splitlines())+'assert len(W1audit)==108\n'
head+='''for name,digest in p['source_pins'].items():assert sha(r/name)==digest,name
manifest=read(w/'collected/SOURCE_MANIFEST.json');assert manifest['config_sha256']==sha(r/'configs/response070_20261018.json')
for name,meta in manifest['files'].items():assert (r/name).stat().st_size==meta['bytes'] and sha(r/name)==meta['sha256'],name
assert v['terminal_source_verification']['all_hashes_verified'] and v['terminal_source_verification']['source_files']==len(manifest['files'])==490
points=all(x['decision']['feasible'] and all(u['passes_point_targets'] for u in x['point_usefulness'].values()) for x in result['results'])
assert points==result['original_point_targets_passed'] and result['stage_outcome']==('FIT_USEFUL_MEAN_PENDING_CONFIRMATION' if points else 'CLOSED_NEGATIVE_SPLIT_AREA_CROSSING_LEVEL_WARP')
record=dict(status='PASS',UTC=datetime.now(timezone.utc).isoformat(),job_id=p['run_id'],source_pins_verified=len(p['source_pins']),deployed_files_verified=len(manifest['files']),execution_counts=result['execution_counts'],original_native_H_JSON_exact069=12,full_native_H_verified=18,original069_baseline_arrays_exact=True,
    geometry_rows_verified=26372,native_feature_rows_verified=79116,intrinsic_rows=26372,mean_examples_independently_verified=156,original_mean_EX_consistency_rows=128,native_examples_verified=384,fast_carrier_examples_verified=128,coupling_rows_verified=26372,coupling_examples_independently_verified=156,candidate_query_rows_independently_reconstructed=13186,
    split_mean_integral_max_error_Vns=split_mean_error,split_random_integral_max_error_Vns=split_random_error,split_seam_metadata_max_error_V=seam_error,queries_and_knots_exact069=True,formula_max_error_V=formula_error,query_max_error_samples=query_error,knot_max_error_samples=knot_error,amplitude_scale_max_error=amplitude_scale_error,observation_flag_tables_verified=12,observation_flag_rows_verified=79116,ADC_integer_codes_exact=True,ADC_voltage_max_error_V=adc_voltage_error,window_max_error_V=window_error,example_feature_max_error=feature_error,example_crossing_max_error_ns=crossing_error,
    coupling_formula_max_error_V=coupling_formula_error,coupling_scalar_max_error=coupling_scalar_error,coupling_native_feature_max_error=coupling_feature_error,W1_CDF_checked=len(W1audit),W1_max_native_error=max(x['absolute_native_error'] for x in W1audit),W1_audit=W1audit,
    profiles=profile_audit,profile_moment_max_error=max_moment_error,profile_scalar_linear_max_error=max_linear_error,profile_covariance_reconstruction_max_error=max_cov_error,profile_parity_reconstruction_max_error=max_parity_error,
    original_point_targets_passed=points,stage_outcome=result['stage_outcome'],full_random_response_acceptance=False,frozen_A0_M0_005_verified=True,uncertainty_scope='All saved rows,query maps andaggregate moments,156 mean/random examples plus128 fast-carrier and384 native windows checked; unsaved full-population waves not reread.')
(w/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps({k:v for k,v in record.items() if k not in ('W1_audit','profiles','execution_counts')}))
'''
ast.parse(head);(w/'VERIFY_ARTIFACTS.py').write_text(head);print('070 full verifier prepared')
