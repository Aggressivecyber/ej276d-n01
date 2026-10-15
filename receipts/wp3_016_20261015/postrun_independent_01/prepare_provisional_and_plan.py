"""Local JSON-only independent016 decision and bounded saved-array QA scope."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
COLLECTED=ROOT/'runs/wp3_016_20261015/selected_collected_01'
ART=COLLECTED/'runs/primary/artifacts'
OLD=ROOT/'receipts/wp3_015_20261015'
H=('fast','slow','common_level')
P,R,N=6593,4,26372
checked={}
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):digest.update(b)
    return digest.hexdigest()
def rec(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
def load(p):
    assert p.suffix=='.json'
    checked[p.relative_to(ROOT).as_posix()]=rec(p)
    return json.loads(p.read_text(encoding='utf-8'))
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def save(name,obj):
    p=HERE/name
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    assert json.loads(p.read_text(encoding='utf-8'))==obj
    return rec(p)

cfg=load(ROOT/'wp3_016_config.json');manifest=load(ROOT/'WP3_016_SOURCE_MANIFEST.json')
assert rec(ROOT/'wp3_016_config.json')['sha256']=='5279597aed74feae87448bfc71aa1387bd711609591eabde775142fa6d1303ab'
assert rec(ROOT/'WP3_016_SOURCE_MANIFEST.json')['sha256']=='824780d297c440d0f1bc51b3ea3344e488ec42ccb46af959d9d0412d8240fc14'
freeze_review=load(STAGE/'launch_fix02/FROZEN_PACKAGE_REVIEW.json')
assert freeze_review['passed'] and not freeze_review['blockers'] and len(manifest['files'])==116
for name,row in freeze_review['sources'].items():assert {k:v for k,v in rec(ROOT/name).items() if k!='path'}==row==manifest['files'][name]
collection=load(STAGE/'collection_01/COLLECTION_RECEIPT.json');inventory=load(STAGE/'collection_01/REMOTE_FULL_INVENTORY.json')
assert inventory['config_sha256']==manifest['config_sha256']==rec(ROOT/'wp3_016_config.json')['sha256']
assert inventory['source_manifest_sha256']==rec(ROOT/'WP3_016_SOURCE_MANIFEST.json')['sha256']
selected,whole,raw=[inventory[k] for k in ('inventory','whole_remote_inventory','omitted_remote_inventory')]
assert len(selected)==99 and len(whole)==211 and len(raw)==112
assert set(selected).isdisjoint(raw) and {**selected,**raw}==whole
assert collection['selected_file_count']==99 and collection['whole_remote_file_count']==211 and collection['omitted_file_count']==112
assert collection['whole_remote_inventory_verified'] and not collection['independent_numerical_readback_complete']
assert collection['raw_artifacts_retained_remote'] and collection['all_raw_artifacts_collected'] is False
assert collection['archive_sha256']==inventory['archive_sha256']=='8862c6127b12037e11a1e24e4cda4bddc530e5a83d69437273da397749cae36f'
nonCP_verified=0;CP_metadata_only=0;journal_name='runs/primary/artifacts/operations.jsonl'
for name,row in selected.items():
    path=COLLECTED/name;assert path.is_file() and path.stat().st_size==row['bytes'] and not name.endswith('.npz')
    if name.endswith('.pt'):
        CP_metadata_only+=1;continue
    if name==journal_name:continue
    assert sha(path)==row['sha256'];nonCP_verified+=1
assert CP_metadata_only==6
assert (COLLECTED/'wp3_016_config.json').read_bytes()==(ROOT/'wp3_016_config.json').read_bytes()
assert (COLLECTED/'WP3_016_SOURCE_MANIFEST.json').read_bytes()==(ROOT/'WP3_016_SOURCE_MANIFEST.json').read_bytes()

complete=load(ART/'COMPLETE.json');exit_receipt=load(COLLECTED/'runs/primary/supervisor/exit_receipt.json')
native=load(STAGE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json')
assert complete['native_status']=='COMPLETE' and complete['all_two_trials_completed'] and not complete['training_skipped']
assert complete['all_frozen_A_005_unchanged'] and complete['all_M0_heads_unchanged'] and complete['no_DEV']
assert complete['counters']==complete['attempted_counters']==cfg['budget_caps']
assert complete['operations_entered']==complete['operations_completed']==94249 and not complete['inflight']
assert exit_receipt['child_native_exit_code']==exit_receipt['exit_code']==0 and exit_receipt['reason'] is None and exit_receipt['error'] is None
assert native['source_commit']=='cc68f75d7e17f675a5ad0605167320cc3e9c93a3' and native['cumulative_G']==179667
assert native['active_scientific_jobs']==0 and native['actual_completed_counts']==cfg['budget_caps']
entered=completed=records=0;active={};counts={k:0 for k in cfg['budget_caps']};journal_digest=hashlib.sha256()
with (COLLECTED/journal_name).open('rb') as f:
    for line in f:
        journal_digest.update(line);records+=1;assert line.endswith(b'\n')
        event=json.loads(line);key=event['operation_id']
        if event['event']=='entered':
            assert key==entered and key not in active;entered+=1;active[key]=event['charges']
        elif event['event']=='completed':
            assert key in active
            for k,v in active.pop(key).items():counts[k]+=v
            completed+=1
        else:raise AssertionError('failed/unexpected durable event')
assert not active and entered==completed==94249 and records==188498 and counts==cfg['budget_caps']
assert journal_digest.hexdigest()==selected[journal_name]['sha256'];nonCP_verified+=1
assert nonCP_verified==93

expected_raw=set();snapshots=[];pair_specs=[]
for seed in (0,1):
    trial=f'seed{seed}_Q1'
    for snapshot,arms in [('initial_evaluation',('A','M0','initial')),('final_evaluation',('final',))]:
        directory=f'runs/primary/artifacts/{trial}/{snapshot}/';snapshots.append(dict(trial=trial,snapshot=snapshot,arms=list(arms)))
        for bank in (0,1):
            expected_raw|={directory+f'bank{bank}_BASELINE100.npz',directory+f'bank{bank}_CHARGE_PREADC_MOMENTS.npz'}
            expected_raw|={directory+f'bank{bank}_{arm}_{hyp}_{kind}.npz' for arm in arms for hyp in H for kind in ('HARD_ROWS','REPORT')}
        pair_specs.extend(dict(trial=trial,snapshot=snapshot,arm=arm,hypothesis=hyp) for arm in arms for hyp in H)
assert expected_raw==set(raw) and len(pair_specs)==24
groups={}
for suffix,count in [('HARD_ROWS',48),('REPORT',48),('BASELINE100',8),('CHARGE_PREADC_MOMENTS',8)]:
    entries={n:r for n,r in raw.items() if n.endswith('_'+suffix+'.npz')}
    assert len(entries)==count;groups[suffix]=dict(files=count,compressed_bytes=sum(v['bytes'] for v in entries.values()))
raw_bytes=sum(v['bytes'] for v in raw.values());assert raw_bytes==2169655337
assert groups=={'HARD_ROWS':{'files':48,'compressed_bytes':540810096},'REPORT':{'files':48,'compressed_bytes':1467413227},
    'BASELINE100':{'files':8,'compressed_bytes':26575048},'CHARGE_PREADC_MOMENTS':{'files':8,'compressed_bytes':134856966}}

tolerance=load(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json');tmap=tolerance['T_by_hypothesis']
assert rec(ROOT/'CALIBRATED_TOLERANCES_WP1_013.json')['sha256']==cfg['contracts']['tolerance_table']['sha256']=='f33217e07b1872dfd51f56e86ed30cda665a705568f2f9ada937865d8a33607b'
cells=[];wave_values={arm:[] for arm in ['A','M0','initial','final']};charge_summaries=[];metric_refs={}
for seed in (0,1):
    trial=f'seed{seed}_Q1';trial_complete=load(ART/trial/'COMPLETE.json')
    assert trial_complete['G_updates']==2000 and trial_complete['native_status']=='COMPLETE'
    assert trial_complete['coverage']['source_unique']==6593 and trial_complete['coverage']['real_unique']==21047
    assert trial_complete['frozen_A_005_unchanged'] and trial_complete['protected_M0_shape_unchanged']
    batch_trial=next(t for t in complete['trials'] if t['directory']==trial)
    assert sha(ART/trial/'COMPLETE.json')==batch_trial['COMPLETE_sha256']
    initial=load(ART/trial/'initial_evaluation/METRICS.json');final=load(ART/trial/'final_evaluation/METRICS.json')
    producer=load(ART/trial/'HARD_COMPARISON.json')
    for obj,arms in [(initial,('A','M0','initial')),(final,('final',))]:
        assert obj['source_parents']==P and obj['draws_per_bank']==R and obj['MC_banks']==2 and obj['REAL_rows']==21047
        assert obj['no_DEV'] and obj['all_signed_rows_retained']
        assert set(obj['metrics'])=={f'bank{b}_{a}_{h}' for b in (0,1) for a in arms for h in H}
        for metric in obj['metrics'].values():
            assert metric['all_rows_retained'] and metric['fake_rows']==N and metric['reference_rows']==21047
            assert metric['coverage']['conditional_coverage_complete'] and metric['coverage']['conditional_bins_used']==8
    for row in producer['per_bank_hypothesis']:
        bank=row['bank'];hyp=row['hypothesis']
        for feature,check in row['checks'].items():
            assert check['final']==final['metrics'][f'{bank}_final_{hyp}']['values'][feature]
            assert check['A']==initial['metrics'][f'{bank}_A_{hyp}']['values'][feature]
            assert check['common_initial']==initial['metrics'][f'{bank}_M0_{hyp}']['values'][feature]
            assert check['T']==tmap[hyp][feature]
        assert row['common_initial_reference']=='frozen015_M0_charge_off'
        assert row['charge_on_initial_values']==initial['metrics'][f'{bank}_initial_{hyp}']['values']
    metric_refs[trial]=dict(initial=rec(ART/trial/'initial_evaluation/METRICS.json'),final=rec(ART/trial/'final_evaluation/METRICS.json'))
    for hyp in H:
        components={};pending=[];failures=[];uncalibrated=[];absolute_failures=[]
        for feature,T in tmap[hyp].items():
            banks={arm:[(initial if arm!='final' else final)['metrics'][f'bank{b}_{arm}_{hyp}']['values'].get(feature) for b in (0,1)] for arm in wave_values}
            values={arm:sum(v)/2 if all(finite(x) for x in v) else None for arm,v in banks.items()}
            calibrated=finite(T) and T>0;required_defined=all(finite(values[a]) for a in ('A','M0','final'))
            da=values['final']-values['A'] if required_defined else None;dm=values['final']-values['M0'] if required_defined else None
            if not calibrated:outcome='pending_not_calibrated';uncalibrated.append(feature)
            elif not required_defined:outcome='pending_required_component_missing_or_undefined';pending.append(feature)
            elif da<=T and dm<=T:outcome='noninferior_to_both_references'
            else:outcome='failed_dual_reference_noninferiority';failures.append(feature)
            within=values['final']<=T if calibrated and required_defined else None
            if within is False:absolute_failures.append(feature)
            components[feature]={**values,'bank_values':banks,'T':T,'status':outcome,'delta_final_minus_A':da,'delta_final_minus_M0':dm,
                'absolute_distance_over_T':values['final']/T if calibrated and required_defined else None,'absolute_within_T':within}
        wave=components['wave'];gain_A=(wave['A']-wave['final'])/wave['A'];gain_M0=(wave['M0']-wave['final'])/wave['M0']
        cells.append(dict(candidate='Q1',seed=seed,hypothesis=hyp,eligible_cell=not pending and not failures,
            required_pending=pending,noninferiority_failures=failures,pending_uncalibrated=uncalibrated,
            wave_relative_improvement_vs_A=gain_A,wave_relative_improvement_vs_M0=gain_M0,wave_usefulness_2percent=gain_A>=.02,
            absolute_T_all_positive_defined=not absolute_failures and not pending,absolute_T_failures=absolute_failures,components=components))
        for arm in wave_values:wave_values[arm].extend(components['wave']['bank_values'][arm])
    for snapshot in ['initial_evaluation','final_evaluation']:
        sem=load(ART/trial/snapshot/'CHARGE_SEMANTICS.json')
        assert sem['actual_canonical_A_Q'] and sem['no_denominator_floor'] and sem['shape_amplitude_from_original_A'] and sem['mean_adapter_enabled'] is False
        assert sem['actual_mu_and_theory_variance_per_draw_saved'] and sem['full_stochastic_preADC_carrier_saved'] is False
        for bank,entry in enumerate(sem['banks']):
            path=f'runs/primary/artifacts/{trial}/{snapshot}/bank{bank}_CHARGE_PREADC_MOMENTS.npz'
            assert entry['retained_moments_sha256']==raw[path]['sha256']
            diagnostic=load(ART/trial/snapshot/f'bank{bank}_A_NUMERIC_REPEATS.json')
            assert diagnostic['saved_arrays_sha256']==raw[path]['sha256'] and diagnostic['within_numerical_tolerance']
            assert diagnostic['numerical_tolerance_V']==1e-7 and diagnostic['Q_mu_CPU_GPU_within_derived_bound']
            charge_summaries.append(dict(trial=trial,snapshot=snapshot,bank=bank,producer_only_pending_array_QA=True,
                a_Vns=entry['a_Vns'],b=entry['b'],A_repeat_absmax_V=entry['A_numeric_repeat_absmax_V'],
                mu_repeat_absmax_Vns=entry['Q_mu_numeric_repeat_absmax_Vns'],theory_mean_MCSE_RMS_Vns=entry['theory_sample_mean_MCSE_RMS_Vns'],
                moments_record=dict(path=path,**raw[path])))
scores={arm:sum(v)/12 for arm,v in wave_values.items()};assert all(len(v)==12 for v in wave_values.values())
eligible=all(c['eligible_cell'] for c in cells);gain=scores['M0']-scores['final'];replace=eligible and gain>1e-12
candidate=dict(eligible=eligible,dual_reference_noninferior_cells=sum(c['eligible_cell'] for c in cells),
    final_wave_mean12=scores['final'],paired_M0_wave_mean12=scores['M0'],A_wave_mean12=scores['A'],charge_on_initial_wave_mean12=scores['initial'],
    mean12_absolute_improvement_vs_M0=gain,mean12_relative_improvement_vs_M0=gain/scores['M0'],
    improvement_epsilon=1e-12,exceeds_M0_improvement_epsilon=gain>1e-12,relative_wave_usefulness_cells=sum(c['wave_usefulness_2percent'] for c in cells),
    relative_wave_usefulness_ALL6=all(c['wave_usefulness_2percent'] for c in cells),absolute_T_ALL6=all(c['absolute_T_all_positive_defined'] for c in cells),
    calibrated_components_per_hypothesis={h:sum(finite(v) and v>0 for v in tmap[h].values()) for h in H})
selection=dict(schema='n01/wp3-016-provisional-independent-development-selection/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='PROVISIONAL_JSON_DECISION_PENDING_BOUNDED_SAVED_ARRAY_QA',run_id='wp3_016_20261015',assigned_reviewer=dict(model='gpt-6.1-sol',effort='max',assignment='parent/user policy; runtime telemetry not separately exposed'),
    frozen_configuration=rec(ROOT/'wp3_016_config.json'),frozen_manifest=rec(ROOT/'WP3_016_SOURCE_MANIFEST.json'),
    collection_receipt=rec(STAGE/'collection_01/COLLECTION_RECEIPT.json'),inventory=rec(STAGE/'collection_01/REMOTE_FULL_INVENTORY.json'),
    native_completion=rec(STAGE/'postrun_root_01/NATIVE_COMPLETION_READBACK.json'),tolerance_table_sha256=cfg['contracts']['tolerance_table']['sha256'],
    candidates={'Q1':candidate},cells=cells,reference_scores_mean12=scores,metric_sources=metric_refs,charge_semantics_producer_summaries=charge_summaries,
    eligible_candidates=['Q1'] if eligible else [],selected_research_reference='Q1' if replace else 'M0',provisional_selected_research_reference='Q1' if replace else 'M0',
    selected_mean_adapter_enabled=False,selected_charge_enabled=replace,current_confirmed_research_reference='015_M0_mean_off_charge_off',
    development_selection_final=False,independent_raw_saved_array_readback_complete=False,
    frozen_rule=cfg['development_selection'],decision_scope='Arithmetic aggregation of frozen producer JSON; full empirical distances not independently replayed.',
    independent_JSON_checks=dict(selected_noncheckpoint_file_hashes_verified=nonCP_verified,checkpoint_inventory_entries_only=CP_metadata_only,
        checkpoint_payload_reads=0,journal_records=records,operations_entered=entered,operations_completed=completed,unknown_inflight_ids=[],counts=counts,
        all_rows_and_conditional_bins_retained=True,two_trial_coverage_complete=True),
    expected_followup='Finalize this development reference only after the single registered saved-array CPU QA; no new candidate or training.',
    cumulative_scientific_G=179667,WP3_short_trials_used=6,WP3_short_trial_limit=6,total_short_trials_used=10,
    six_work_packages_complete=False,weights_promoted=False,scientific_acceptance=False,scientific_status='response_match_not_established')
selection_ref=save('PROVISIONAL_DEVELOPMENT_SELECTION.json',selection)

scope=dict(schema='n01/wp3-016-independent-saved-readback-scope/v1',UTC=datetime.now(timezone.utc).isoformat(),
    independently_recomputed_planned=[
        'Saved ADC uint16/16383-.5-double baseline plus optional IEEE XOR inverse; exact code roundtrip; all rows retained.',
        'Parent/seed/trial/bank/draw/ID lineage; Q/front/tail/PSD/peak/rise/FWHM/baseline; reconstructible flags and preserved full-carrier flag counts.',
        'Native4draw wave/scalar moments and empirical MCSE,9x9 pairwise-finite covariance/counts; derived two-bank8draw denominator7 and4+4 identity.',
        'All8 actual frozen-A384 float64 banks, actual positive mu_d/q_d, per-draw v_d=a*mu_d+b*mu_d^2, existing1e-7V repeat gate and predeclared float64 canonical reduction bound.',
        'Saved per-draw charge means/variances, E(samplevar q)=mean(v)+samplevar(mu,ddof1), theory meanMCSE=sqrt(sum v)/4, empirical q MCSE and q-mu ddof0/1 moments separately.',
        'Saved original-input and float64-forward hash arrays: row schema, per-parent repetition and cross-bank/source-ID consistency only; original input waveforms not reopened.'
    ],
    producer_JSON_only=['Complete wave/QPSD/noise/other empirical distances; independent work checks saved-JSON decision algebra, not CDF/loss numerics.',
        'Actual original SOURCE input bytes against stored input hash; no original SOURCE payload read in this task.',
        'Realized charge increment RMS and full stochastic signal/canonical assembly identity: log multiplier, M0 stochastic preADC signal and carrier were not saved.'],
    not_reconstructed=['charge Z or any RNG draws from saved states','full stochastic preADC384 signal or carrier','original Y second floating copy',
        'checkpoint tensor states or new model forwards','original C1 median fit','full producer CDF/loss replay','bootstrap or new MC confidence intervals'],
    native_charge_r8_report_exists=False,charge_r8_added_to_this_scope=False,hard_r8_is_new_derived_statistic=True,
    native_016_combined8_hard_report_exists=False,hard_empirical_wave_rank_bound=7,
    empirical_hard_rank_is_preADC_theoretical_rank=False,pairwise_missing_covariance_automatic_PSD_claim=False,
    full_frozen_A384_is_saved=True,full_stochastic_preADC_carrier_is_saved=False,
    original_failed_run_repeat_magnitude='unknown_not_saved; this complete replacement cannot retroactively measure it',
    old015_readback_reused_as_mature_source_not_rerun=True,scientific_acceptance=False,scientific_status='response_match_not_established')
scope_ref=save('SCOPE_NOTE.json',scope)

counters=dict(raw_files_SHA_checked=112,raw_bytes_SHA_read=raw_bytes,hard_rows_files_decoded=48,report_npz_files_decoded=48,
    baseline_npz_files_decoded=8,probe_raw_npz_files_decoded=0,probe_gradient_npz_files_decoded=0,waveform_rows_decoded=48*N,
    bank_pairs_verified=24,probe_snapshots_verified=0,charge_npz_files_decoded=8,charge_A_rows_verified=8*N,
    charge_A_float64_cells_verified=8*N*384,charge_mu_draws_verified=8*N,charge_q_draws_verified=8*N,charge_parent_moment_records_verified=8*P)
assert sum([112,8,8,48,24])==200 and counters['waveform_rows_decoded']==1265856
plan=dict(schema='n01/wp3-016-independent-readback-plan/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='EXACT_INVENTORY_SCOPE_READY_FOR_ENGINEERING_NOT_EXECUTION',run_id='wp3_016_saved_readback_20261015',
    original_root=cfg['remote_root'],sidecar_root='/home/rylan/projects/ej276d-n01/wp3_016_saved_readback_20261015',original_files_read_only=True,
    assigned_reviewer=dict(model='gpt-6.1-sol',effort='max'),frozen_configuration=rec(ROOT/'wp3_016_config.json'),
    frozen_manifest=rec(ROOT/'WP3_016_SOURCE_MANIFEST.json'),frozen_source_file_count=116,
    inventory=rec(STAGE/'collection_01/REMOTE_FULL_INVENTORY.json'),collection_receipt=rec(STAGE/'collection_01/COLLECTION_RECEIPT.json'),
    provisional_selection=selection_ref,scope_note=scope_ref,
    intended_worker='receipts/wp3_016_20261015/postrun_independent_01/readback_saved_arrays.py',
    intended_kernel='receipts/wp3_016_20261015/postrun_independent_01/saved_array_kernels.py',
    intended_controller='receipts/wp3_016_20261015/readback_engineering_01/readback_control.py',
    implementation_owner='engineering Sol6.1Max agent; independent reviewer checks delta before root registers execution',
    raw_inventory=raw,raw_groups=groups,expected_raw_files=dict(total_npz_SHA=112,total_npz_decode=112,hard_rows_npz=48,report_npz=48,
        baseline100_npz=8,charge_moments_npz=8,additional_mean_summary_npz=0,probe_or_gradient_npz=0,bank_pairs=24,
        SOURCE_parents=P,draws_per_bank=R,derived_hard_draws=8,response_rows_per_hard_file=N,total_hard_rows=48*N,
        baseline_rows=8*N,actual_A384_rows=8*N,actual_A_float64_cells=8*N*384,actual_A_uncompressed_bytes=8*N*384*8,
        charge_mu_draws=8*N,charge_q_draws=8*N,charge_parent_moment_records=8*P,raw_compressed_bytes=raw_bytes),
    expected_complete_counters=counters,expected_operations=200,
    operation_derivation='112 raw SHA +8 baseline decode +8 charge-bank decode/verify +48 hard-bank statistics(each1HARD+1REPORT) +24 completed two-bank hard pairs =200 durable entered/completed operations.',
    snapshot_order=snapshots,pair_specs=pair_specs,
    finite_resource_budget=dict(CPU_threads=2,process_RSS_MiB=1536,virtual_memory_backstop_MiB=3072,wall_seconds=1800,output_MiB=64,
        expected_output_MiB_upper_estimate=8,raw_file_SHA_passes=1,raw_decode_passes=1,fixed_input_compressed_envelope_MiB=3072,
        input_IO_upper_envelope_MiB=6144,exact_SHA_content_bytes=raw_bytes,logical_SHA_plus_one_decode_compressed_content_bytes=2*raw_bytes,
        maximum_simultaneous_bank_pairs=1,maximum_simultaneous_charge_banks=1,release_actual_A_before_hard_pairs=True,
        covariance_parent_block=64,dense256x256_covariance_matrices=0,automatic_retry=False,
        empty_or_failed_readback='Preserve FAILED/status/journal and stop; no tolerance/scope expansion or automatic second worker.'),
    scientific_budget=dict(CP_loads=0,original_SOURCE_REAL_TRAIN_waveform_reads=0,original_DEV_waveform_reads=0,model_forwards=0,
        AD_queries=0,optimizer_or_sampler_created=0,training_updates=0,random_library_draws=0,new_CDF_or_loss_replays=0,bootstrap_replicates=0,GPU_work=0),
    saved_charge_checks=dict(actual_A_shape_per_bank=[N,384],actual_A_dtype='float64',mu_q_v_shape_per_bank=[P,R],
        exact_parent_draw_order=True,strict_positive_finite_mu_and_q=True,coefficients_finite_strict_positive=True,
        no_actual_A_mu_q_substitution=True,A_repeat_point_tolerance_V=1e-7,A_gate_source='scripts/run_wp1_013.py:A_numerical_residual frozen in source116',
        canonical_window=[110,366],canonical_dt_software_ns=1,
        CPU_GPU_sum_bound='2*gamma255/(1-gamma255)*sum(abs(A[110:366]))+512*smallest_subnormal; nextafter +inf',
        bound_assumptions='finite same actual float64 inputs, IEEE nearest additions, at most255 additions, no overflow; tiny subnormal allowance',
        v_d='a*mu_d+b*mu_d**2',expected_sample_variance='mean(v_d)+samplevar(mu_d,ddof1)',theory_mean_MCSE='sqrt(sum(v_d))/4',
        empirical_q_mean_MCSE='sqrt(samplevar(q,ddof1)/4)',centered_second_moment='mean((q-mu)**2), ddof0',
        centered_sample_variance='samplevar(q-mu,ddof1)',expected_A_increment_RMS='sqrt(mean(A**2*(a/mu+b)))',
        original_source_byte_hash_replay=False,saved_input_hash_repeat_and_crossbank_consistency=True,
        full_charge_Z_or_stochastic_carrier_replay=False,charge_r8_added=False),
    first_proof='Only after first actual complete formal two-bank hard pair and saved-charge checks for that snapshot; current same-snapshot PID/boot/resource/durable health or native0 COMPLETE terminal proof.',
    mature_reuse=dict(worker=rec(OLD/'postrun_independent_01/readback_saved_arrays.py'),kernel=rec(OLD/'postrun_independent_01/saved_array_kernels.py'),
        controller=rec(OLD/'readback_engineering_01/readback_control.py'),kernel_should_remain_byte_exact=True,
        closed015_receipt=rec(ROOT/'runs/wp3_015_20261015/independent_readback_collected_01/artifacts/ARRAY_READBACK.json'),
        old015_execution_or_numeric_tests_rerun=False,allowed_delta='016 paths/source bindings/actual inventory/caps/initial A+M0+initial arms/two trials plus charge-array verification; no new research protocol'),
    actual_preparation=dict(stdlib_JSON_journal_only=True,actual_saved_NPZ_reads=0,original_SOURCE_REAL_DEV_waveform_reads=0,
        checkpoint_tensor_loads=0,NN_rows=0,AD_queries=0,optimizer_or_sampler_created=0,training_updates=0,new_random_draws=0,SSH_calls=0,numerical_fixtures_rerun=0),
    execution_authority='Root must register exact final worker/kernel/controller/plan/inventory pins and explicitly authorize one CPU task. This plan starts no worker and makes no SSH calls.',
    limitations=scope,scientific_acceptance=False,scientific_status='response_match_not_established')
plan_ref=save('READBACK_PLAN.json',plan)
receipt=dict(schema='n01/wp3-016-provisional-and-readback-scope-preparation/v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='JSON_SELECTION_PROVISIONAL_EXACT_CPU_SCOPE_PREPARED_NOT_STARTED',outputs=dict(selection=selection_ref,plan=plan_ref,scope=scope_ref),
    native_operations=94249,raw_files=112,raw_bytes=raw_bytes,readback_expected_operations=200,candidates={'Q1':candidate},
    source_JSON_evidence=checked,actual_NPZ_or_CP_tensor_reads=0,scientific_operations=0,SSH_calls=0,main_metadata_changes=0)
receipt_ref=save('PREPARATION_RECEIPT.json',receipt)
print(json.dumps(dict(outputs=receipt['outputs'],receipt=receipt_ref,candidate=candidate,raw_groups=groups,expected_operations=200,
    raw_bytes=raw_bytes,expected_complete_counters=counters),ensure_ascii=False,indent=2))
