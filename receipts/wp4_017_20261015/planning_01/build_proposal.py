"""Stdlib-only WP4 proposal and exact planned cost arithmetic; no data/model IO."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
read = lambda name: json.loads((ROOT / name).read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def ref(name, **extra):
    p = ROOT / name
    return dict(path=name, bytes=p.stat().st_size, sha256=sha(p), **extra)

def write_new(name, value):
    with (OUT / name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

master = read('MASTER_EXECUTION_PLAN.json')
teacher_budget = master['budgets']['teacher_runs']
assert teacher_budget['maximum'] == 8 and teacher_budget['completed'] == 0
assert teacher_budget['G_steps_per_run_maximum'] == 2000
old = read('wp3_016_config.json')
final = read('receipts/wp3_016_20261015/postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json')
plan_lines = (ROOT / 'USER_MASTER_PLAN_20261015.txt').read_text(encoding='utf8').splitlines()
begin = next(i for i, row in enumerate(plan_lines) if row.startswith('6. WP4'))
end = next(i for i, row in enumerate(plan_lines[begin+1:], begin+1) if row.startswith('7. WP5'))

P, Pfit, Ptest, K, H, steps = 6593, 5274, 1319, 512, 3, 2000
banks, R = 2, 32
qref = 3.7103998046755806
delta = 1 / 16383
per_eval = P * banks * R
per_train_side = steps * K * 2
normal_dimensions = dict(shape=8, charge=1, noise005=101, extension284=284)

def costs(paired_runs, unpaired_runs, teachers_evaluated):
    runs = paired_runs + unpaired_runs
    student_eval_systems = 1 + runs  # one common initial plus every final
    eval_systems = student_eval_systems + teachers_evaluated
    student_rows = runs * per_train_side + student_eval_systems * per_eval
    teacher_rows = runs * per_train_side + teachers_evaluated * per_eval
    all_rows = student_rows + teacher_rows
    # IID parent indices keep the marginal student self U-statistic unbiased.
    # End-of-trial full fit-parent coverage is verified, never presumed.
    sampler_streams = paired_runs + 4 * unpaired_runs
    sign_rows = (4 * per_train_side + per_eval) if teachers_evaluated == 2 else 0
    return dict(
        trial_slots=runs, G_updates=runs*steps, optimizer_steps=runs*steps,
        training_backward_queries=runs*steps, new_AD_queries=runs*steps,
        optimizers_created=runs,
        checkpoint_loads=3, A_original_forward_rows=P,
        original_U_Net_decoded_rows=P, fixed_M0_head_actual_kernel_rows=P,
        student_charge_module_rows=student_rows, teacher_charge_module_rows=teacher_rows,
        shape_cached_basis_assembly_rows=all_rows, noise005_rows=all_rows,
        all_three_extensions_rows=H*all_rows,
        hard_ADC_rows=H*all_rows, hard_first_argmax_gather_rows=H*all_rows,
        training_student_rows=runs*per_train_side,
        training_teacher_rows=runs*per_train_side,
        evaluation_kernel_snapshots=eval_systems, evaluation_signal_rows=eval_systems*per_eval,
        scalar_normal_requests={k:all_rows*v for k,v in normal_dimensions.items()},
        total_scalar_normal_requests=all_rows*sum(normal_dimensions.values()),
        T2_teacher_sign_uniform_requests=sign_rows,
        coverage_PIT_uniform_requests=teachers_evaluated*per_eval*H*6,
        parent_sampler_streams=sampler_streams, parent_randint_calls=sampler_streams*steps,
        parent_randint_scalar_requests=(paired_runs+4*unpaired_runs)*steps*K,
        parent_randperm_calls=0,
        paired_training_energy_pair_distances=paired_runs*steps*K*5*H,
        unpaired_training_energy_pair_distances=unpaired_runs*steps*(2*K)*(2*K)*2*H,
        unpaired_training_cross_retained_pairs=unpaired_runs*steps*(2*K)*(2*K)*H,
        unpaired_training_self_ordered_retained_pairs=unpaired_runs*steps*(2*K)*(2*K-1)*H,
        unpaired_training_self_diagonal_computed_then_masked=unpaired_runs*steps*(2*K)*H,
        evaluation_energy_candidate_comparisons=runs+teachers_evaluated,
        evaluation_energy_pair_distances=((runs+teachers_evaluated)*banks*2+teachers_evaluated*3)*P*H*R*R,
        checkpoint_saves_complete=1+2*runs,
        REAL_waveform_reads=0, original_DEV_waveform_reads=0,
        A_005_M0_updates=0, mean_adapter_updates=0, shape_head_updates=0,
        teacher_parameter_updates=0, noise_updates=0, population_updates=0,
        D_updates=0, C2ST_updates=0, probe_AD_queries=0,
        integral_scale_fits=0, charge_initialization_fits=0,
        teacher_or_student_latent_in_deployment_inputs=0,
        new_hardware_or_REAL_teacher_claims=0)

stage1 = costs(2, 0, 1)
full = costs(4, 4, 2)
cache = dict(
    parents=P, carrier_samples=384, actual_A_dtype='float64',
    actual_A_bytes=P*384*8, canonical_Q_bytes=P*8,
    decoded_dtype='original float32, no precision change', decoded_shape=[P,32,384],
    decoded_total_bytes_produced=P*32*384*4,
    fixed_B_dtype='float64', fixed_B_shape=[P,8,384], fixed_B_bytes=P*8*384*8,
    original_prepared_SOURCE384_float64_max_resident_bytes=P*384*8,
    chunk_parents=128, decoded_chunk_bytes=128*32*384*4,
    fixed_B_chunk_bytes=128*8*384*8,
    persisted_A_B_Q_numerical_bytes=P*384*8+P*8*384*8+P*8,
    SOURCE384_plus_A_B_Q_max_numerical_bytes=P*384*8*2+P*8*384*8+P*8,
    full_decoded_persistent_cache_bytes=0,
    full_decoded_if_materialized_numerical_peak_bytes=P*384*8*2+P*8*384*8+P*8+P*32*384*4,
    allocation_policy='One formal A pass in128-parent chunks captures A and decoded. In the same chunk the one fixed M0 head call produces B. Hash decoded by ordered shape/dtype/row bytes; release decoded chunk after B. Never call A again for training/evaluation or population changes.',
    cache_binding='Actual original SOURCE member hash/dtype, stable parent IDs, exact A/005/M0 CP digests and original expansion/gate/projection digests. 016 saved A alone cannot create this cache.',
    cache_all_parent_features_are_not_fitted_on_heldout_targets=True)

proposal = dict(
    schema='n01/wp4-017-engineering-proposal/v1', run_id='wp4_017_20261015',
    recorded_UTC=datetime.now(timezone.utc).isoformat(),
    status='LOCAL_PROPOSAL_ONLY_NOT_REGISTERED_NOT_IMPLEMENTED_NOT_STARTED',
    actual_scientific_work_in_this_preparation=dict(CP=0, SOURCE_REAL_DEV_reads=0,
        NN=0, AD=0, G=0, RNG=0, remote_calls=0),
    title_zh='固定已知A/005/M0形状下的电荷条件核恢复',
    purpose_zh='检验最终Q1两参数电荷族在已知教师、独立潜变量和未参与新参数拟合的父事件上能否恢复条件分布；再量化无配对信息与单一族外机制的局限。',
    authority=dict(
        user_plan=ref('USER_MASTER_PLAN_20261015.txt'), chapter6_lines=[begin+1,end],
        chapter6_text='\n'.join(plan_lines[begin:end]),
        master_historical_read_only_snapshot=ref('MASTER_EXECUTION_PLAN.json',
            mutable_live_metadata_not_an_immutable_source_pin=True),
        teacher_budget_snapshot=teacher_budget,
        accepted_root_decision='2theta student; frozen mean/A/005/M0 shape. Root plus independent reviewer perform one contract review before implementation; no normal-difference audit expansion.',
        teacher_budget_separate_from_short_candidates=True),
    contracts={name:ref(name) for name in ('OBSERVATION_CONTRACT.json','DATA_ROLES.json','docs/ACCEPTANCE_CONTRACT.md',
        'docs/WP3_016_CONFIGURATION.md','WP3_016_DATA_ROLES.json')},
    final_Q1_context=dict(selection=ref('receipts/wp3_016_20261015/postrun_independent_01/FINAL_DEVELOPMENT_SELECTION.json'),
        final_array_review=ref('receipts/wp3_016_20261015/postrun_independent_01/FINAL_ARRAY_AND_SELECTION_REVIEW.json'),
        same_module_family=True, fitted_016_charge_checkpoint_loaded=False,
        mean_adapter=False, physical_scientific_acceptance=False),
    fixed_bases={name:old['inputs'][name] for name in ('deterministic','noise005','M0_head_seed0')},
    bases_explanation_zh='固定015 seed0 M0头供全部八trial使用，不按教师结果挑seed；A和005是先前全部原TRAIN拟合过的已知教师基座。学生seed只改变采样/随机流，不切换教师shape。此测试不是未见REAL交叉验证，也不是所有数据依赖组件重拟合的CV。',
    student=dict(
        parameters=['theta_a','theta_b'], parameter_count=2,
        law='Q=canonical_charge008(actual_A)>0; a=qref*softplus(theta_a); b=softplus(theta_b); s2=log1p(a/Q+b); q=Q*exp(sqrt(s2)*zq-s2/2)',
        assembly='signal=A+B*eta_shape+A*expm1(sqrt(s2)*zq-s2/2); B is controlled only by deterministic A and decoded features, never q.',
        q_reference_Vns=qref, a_initial_Vns=.001*qref, b_initial=.00025,
        initial_theta='stable inverse-softplus of a_initial/qref and b_initial; no calibration fit or head initialization RNG',
        initial_variance_fraction_of_T1_at_every_Q=.1,
        fixed_mean='A; expected preADC mean fixed, no learned mean/time/gain',
        protected=dict(A=True,noise005=True,M0_1288_shape_head=True,mean_adapter_disabled=True,
            original_gate_expansion_projection=True),
        charge_off='Exact original frozen M0 bytes, no charge randomness or mean correction.',
        deployment_inputs='Complete clean source waveform and registered detector configuration only; no parentID, teacher q, z, mixture sign or teacher sample.',
        learned_scope_zh='只恢复两参数有效电荷方差律；固定B及其形状协方差、均值和硬观察时间量必须实测预测核对，不以构造自动PASS；不宣称重新学得1288形状头、平均响应或新时间机制。'),
    teachers=dict(
        T1=dict(family='Exactly the two-parameter Q1 family over the common frozen A/005/M0 base',
            a_known_Vns=.01*qref,b_known=.0025, known_qref_Vns=qref,
            CV_at_Qref=math.sqrt(.0125), variance_vs_initial=10.,
            signal_conditional_mean='A', signal_conditional_covariance='B B^T + A A^T*(a/Q+b)',
            charge_mean='Q',charge_variance='a*Q+b*Q^2',
            known_shape='Actual frozen B(x) from original decoded32x384 features and M0 head, not A alone.',
            covariance_recovery='Predicted total covariance/charge-dependent increment checked against independent teacher draws; fixed BB^T is a known nuisance, not a learned claim.'),
        T2=dict(base='Same A/B/005, a,b, observation and independence as T1',
            only_out_of_family_mechanism='Replace eta_shape[0] by m*Rademacher + sqrt(1-m^2)*independent_N01; eta_shape[1:8] remain independent N01.',
            m=.9, Rademacher_probabilities=[.5,.5], mean=0.,variance=1.,
            fourth_moment=3-2*.9**4, T1_fourth_moment=3.,
            teacher_student_latents_independent=True,
            same_first_second_moments_as_T1_preADC=True,
            first_basis_fixed_without_result_selection=True,
            zero_first_basis_or_unresolvable_observed_contrast='Record mechanism untestable/inconclusive; do not change component, m, mechanism, student family or add trials.',
            added_time_or_local_noise_or_population_mechanism=False)),
    preparation_cache=cache,
    deterministic_materialization=dict(
        original_CP_loads=3, additional_016_CP_loads=0,
        no_random_placeholder_initialization=True,
        loader_plan='Construct the unchanged original classes using deterministic zero placeholders for new parameter initialization in a scoped local loader; strictly load every registered original tensor, restore init functions in finally, then prove original state digest and forward algebra in the manufactured fixture. Keep the forward classes unchanged.',
        cached_student='New2theta class holds fixed A/B/Q buffers and uses pure charge_coefficients016/inverse_softplus016; it must not call the014/016 constructor that would randomly reinitialize a head before copying M0.',
        constructor_global_CPU_GPU_RNG_state_must_be_unchanged=True,
        if_zero_placeholder_materialization_not_semantically_supported='Resolve once in local manufactured engineering before source freeze; do not silently draw native placeholder randomness outside the declared budget.'),
    data_roles=dict(
        SOURCE_legal_original_TRAIN_parents=P, original_source_universe_retained=True,
        fit_parents=Pfit, heldout_new_parameter_parents=Ptest,
        split='Sort stable original parent IDs by SHA256("N01-WP4-017-parent-split-v1|"+canonical_ID); first1319 heldout, remaining5274 fit; ties by original unique parent ID. Freeze once for every teacher/mode/seed.',
        required_ID_uniqueness=True, no_parent_crosses_partition=True,
        heldout_targets_never_used_for_fit_initialization_optimization_scale_or_setting_selection=True,
        all6593_evaluated_train_and_test_separately=True,
        base_A_has_seen_original_TRAIN='Known fixed system; only the new2theta fit sees5274 parents. Holdout means unseen to this synthetic target fitting, not unseen to historical A/005 or real CV.',
        SOURCE_input_member_only='Formal execution binds and loads only SOURCE carrier/lineage/available original physical weight members. It must not materialize REAL/DEV members of a combined archive.',
        REAL_reads=0, DEV_reads=0, fits_from_REAL_or_DEV=0,
        original_count_and_physical_weight_fields='Preserve verbatim. Absent physical weights remain unknown; uniform synthetic count weights are separately named.',
        teacher_latent_excluded_from_student_input=True),
    objective=dict(
        type='proper conditional energy score in the full hard observed256 output',
        feature_map='f(y)=concat(y/(sqrt(256)*ADC_LSB),sum(y)/(256*ADC_LSB)); fixed injective257-dimensional map; no sample-fitted normalization.',
        general_unbiased_U_statistic='For m>=2 independent student draws and n>=1 independent teacher draws: ES_U= sum_ij d(S_i,T_j)/(m*n) - sum_{i!=j} d(S_i,S_j)/(2*m*(m-1)); student self excludes its exact sample-index diagonal. No teacher latent cross-coupling.',
        paired='For each of512 IID fit parents generate two teacher draws and two student draws independently conditional on x. m=n=2: .25*sum_{r,l}d(S_r,T_l)-.5*d(S_1,S_2); average parents and all3 noise hypotheses.',
        paired_independence='Teacher vs student shape, charge, 005, extension streams independent. Student draw1 vs2 and teacher draw1 vs2 independent. Same latent is not used in the training score.',
        unpaired='Draw1024 IID student parent indices and1024 IID teacher parent indices from the5274 fit parents, using four independently seeded512-index banks. Draw independent output latents per row. Use ES_U with m=n=1024; all self diagonal terms excluded, denominator1024*1023. No target-parent match used by the objective.',
        unpaired_target='Marginal population energy score; conditional identifiability is evaluated, not assumed. Even when IID parent indices coincide, independent rows are distinct samples and retained; only the same sample-index self diagonal is excluded.',
        sampler_qualifier='Uniform IID randint parent streams, not dependent no-replacement batches. At2000G verify actual union coverage of all5274 fit parents in every required stream; a missing parent makes the trial incomplete with no extra draw/step/retry. Original parent identities never enter neural inputs.',
        pairwise_distance_block_rows=64, pairwise_distance_columns_unpaired=1024, dimension=257,
        matched_cost='Both modes generate1024 teacher and1024 student rows per update; same2theta, init,2000G and latent tensor budgets. Unpaired pairwise arithmetic is larger and explicitly counted.',
        hard_forward_with_STE=True, final_evaluation_STE=False,
        single_draw_MSE=False, extra_variance_CV_regularizer=False,
        biased_V_statistic=False, clamp_negative_score_to_zero=False,
        first_real_update_requires_positive_loss=False,
        score_check='Finite only; signed unbiased energy-distance or teacher-self-adjusted diagnostics may be negative. First proof checks each finite nonzero gradient and actual parameter update, never loss>0.',
        inherited_real_CDF_or_integral_loss=False, extra_saturation_loss=False,
        reason_no_regularizer='A variance shrink penalty would bias recovery of the known teacher; score alone is optimized and flags are evaluated.',
        teacher_same_latent_use='Only optional manufactured engineering equality examples; never scientific fitting, recovery evidence or an unregistered native forward.'),
    training=dict(
        teachers=['T1','T2'], modes=['paired','unpaired'], seeds=[0,1], maximum_trials=8,
        updates_per_trial=steps, optimizer='fresh AdamW exactly theta_a/theta_b',
        lr=.01, weight_decay=0., clip_norm=5., one_backward_per_update=True,
        output_chunks=256, checkpoint_steps=[1000,2000], initial_shared_checkpoint=True,
        inherited_optimizer=False, resumes_or_automatic_retry=False,
        parameter_log='Save each theta, a,b, preclip gradients and actual delta; first real step both parameters require finite nonzero gradient and actual update.',
        scientific_trial_order=['T1_paired_seed0','T1_paired_seed1','GATE_STOP_OR_CONTINUE',
            'T1_unpaired_seed0','T1_unpaired_seed1','T2_paired_seed0','T2_paired_seed1','T2_unpaired_seed0','T2_unpaired_seed1'],
        stage1_G_cap=4000, full_G_cap=16000,
        stage2_only_after_fixed_T1_paired_both_seed_gate=True,
        failure='Preserve actual operations, partial CP, FAILED and nativeexit. No replacement native or extra candidate is authorized by this proposal.'),
    randomness=dict(
        explicit_CPU_generators=True, generated_normal_dtype='float32 then same prescribed float64 law arithmetic',
        normal_scalars_per_signal_row=normal_dimensions,
        seed_rule='low63bits(little_endian(SHA256("N01-WP4-017-v1|"+canonical_json(namespace))[:8])); namespace includes replicate, role and bank. No implicit global RNG.',
        training_namespaces='Teacher/student roles separate. Matched paired/unpaired replicate may share latent prefixes; modes differ only in parent association and objective. Source sampler role-specific. T1/T2 share Gaussian prefixes; only T2 adds its independent sign generator.',
        evaluation_namespaces='One fixed initial/student eval library reused by every final kernel for paired comparisons; teacher eval library independent of all students.2 banks mutually independent. Teacher coverage tie uniforms independent of wave generation and reused across student comparisons.',
        generator_initial_end_and_checkpoint_states_saved=True,
        rng_count_unit='Requested normal/uniform/randint tensor scalar outputs. Internal PRNG words/algorithm consumption are not inferred from scalar count.',
        extra_projection_or_bootstrap_randomness=0),
    evaluation=dict(
        all_parents=P, banks=banks, draws_per_parent_per_bank=R, hypotheses=['fast','slow','common_level'],
        snapshots='T1 teacher once + common student initial once + each final trial once. T2 teacher generated once only after stage1 passes. No intermediate1000G model evaluation.',
        total_kernel_snapshots_full=11, total_eval_signal_rows_full=11*per_eval,
        observations='Original384 carrier, prefix100,14bitADC[-.5,.5]nearest-even,firstargmax[112:144],crop[p-16:p+240]256,1software ns,tail46 unchanged.',
        per_bank_and_combined='Keep native32-draw bank moments, bankwise energy scores and mean2bank score.64draw moments/PIT explicitly derived from the two32 banks, not native r64.',
        axes=['whole synthetic population','output_Q_conditioned','SOURCE_A_Q_and_input_trends','same_parent_repeats'],
        conditional_moments='Per-parent waveform means/diagonal variances;9scalar finite-pair covariance with actual joint counts; no dense256cov required. Compare covariance discrepancy on normalized9scalar entries; missing-pair covariance is not automatically PSD.',
        scalar9=['Q','tail','PSD_raw','baseline_mean','baseline_RMS','peak_voltage','rise10_90','FWHM','peak_index'],
        timing='Peak index plus rise/FWHM and all censor/undefined causes; no new random time parameter is inferred.',
        coverage_features=['Q','tail','peak_voltage','peak_index','rise10_90','FWHM'],
        coverage='For each defined independent teacher target, use the actual finite student sample count nS_finite: PIT=(L+U*(E+1))/(nS_finite+1). L counts finite student values strictly below the target; E counts finite ties. nS=0 or insufficient fixed support is pending, never imputed. See finite_support/coverage contracts.',
        finite_samples='64draws are MC repeats, not64 physical parents. Store parent counts/draw counts/valid counts and report empirical parent-cluster SE separately from model-seed variation.',
        original013_T='Continue reporting original physical application gates where meaningful, but WP4 synthetic recovery gates below have separate names/units and do not alter013T.',
        two_theta_identifiability='Using known actual cached Q on fit/test, report scaled Gram for [Q/qref,(Q/qref)^2], eigenvalue ratio/condition number and analytic variance-curve error. Separate coefficient errors only called identifiable if lambda_min/lambda_max>=1e-4; otherwise individual a/b recovery remains unestablished.'),
    gates=dict(
        frozen_before_any_real_CP_SOURCE_forward_or_new_randomness=True,
        cache_precondition=dict(all_Q_finite_strict_positive=True, no_floor=True,
            SOURCE_ID_unique_and_fixed_partition=True, exact_original_input_dtype_and_bytes_preserved=True,
            actual_frozen_A_and_B_finite_prefix100_zero=True,
            T1_vs_initial_expected_charge_increment_RMS_difference_min_ADC_LSB=3.,
            failed_contrast='Stop as untestable before trial; no coefficient or threshold adjustment.'),
        all_seed_hypothesis_required='Both T1paired seeds must pass all3 hypotheses; every required metric must be defined. Missing/undefined is pending and does not open stage2.',
        primary_variance_curve=dict(heldout_relative_RMS_error_max=.10, heldout_max_relative_error_max=.20,
            definition='v_model=a_model*Q+b_model*Q^2 vs known v_teacher, no MC estimator or inferred theta distance substitutes for this functional recovery.'),
        hard_mean=dict(heldout_wave_standardized_RMSE_max=.30,
            definition='sqrt(mean_parent,time((student_mean64-teacher_mean64)^2))/sqrt(mean_parent,time(teacher_sample_variance64)); MC-inclusive discrepancy, all rows retained.'),
        hard_variance=dict(heldout_pooled_conditional_trace_relative_error_max=.10,
            repeat_on=['all heldout','3 input-Q tertiles fixed using fit-parent Q before target generation']),
        hard_covariance=dict(normalized9scalar_Frobenius_relative_error_max=.15,
            definition='See conditional_covariance_contract: per-parent within covariance normalized by teacher64 SD/fixed floor, then group weighted mean, then Frobenius relative discrepancy; no between-parent mean scatter.',
            teacher32_split_empirical_resolution_max=.075,
            unresolved='If fixed support/reference norm/teacher-only empirical resolution fails, pending, not expression failure; no extra samples or relaxed gate.'),
        timing=dict(heldout_absolute_aggregate_mean_error_max_software_ns=1.,
            heldout_per_parent_mean_RMSE_max_software_ns=2.,
            required=['peak_index','rise10_90','FWHM'], undefined_or_censor_fraction_difference_max=.01),
        coverage=dict(level50_absolute_error_max=.04,level90_absolute_error_max=.03,
            apply_to='Per-parent defined-target coverage, then fixed parent-weight mean; actual nS+1 and support rules. Stage1 matrix explicitly requires allheldout+3fit-Q groups x nominal/low/high x3hyp x2seeds.',
            undefined_or_censor_fraction_difference_max=.01),
        conditional_energy=dict(final_minus_initial_parent_cluster_upper95_required_lt=0.,
            final_excess_over_teacher_two_bank_empirical_reference_max='max(0.02*abs(teacher_reference_score),3*parent_cluster_empirical_SE)',
            reference='Use independent saved T1teacher bank0 vsbank1 in both directions as the finite-sample teacher reference; no third oracle bank or extra forward.',
            SE='See energy/weight contracts: same-parent final-initial or final-reference differences and fixed weighted HC1 parent-cluster SE; no independent-error addition or negative-value clipping.'),
        individual_a_b=dict(relative_error_reported=True, if_design_identifiable_recovery_precision_target=.25,
            use='Secondary parameter identifiability result; cannot replace the required conditional functional and hard observed gates.'),
        no_flags_silently_dropped=True,
        T2='Apply the same observed recovery metrics and scope limits, report family failure or observed indistinguishability without adding a component or trial.',
        exit_interpretation=dict(
            T1_paired_fail='Stop after2trials; diagnose expression/optimization/observation in this fixed2theta task, no real long training.',
            paired_pass_unpaired_fail='Current unpaired marginal score/population information insufficient for this known conditional relationship; not a universal impossibility claim.',
            both_pass_REAL_fail='Check input physics/population/selection/noise assumptions; no automatic assertion that REAL is wrong.',
            T2_fail='Boundary for this predeclared two-parameter family; no immediate expansion or extra native.')),
    population_change=dict(
        SOURCE_population_support='Same heldout1319 original parents; no new artificial waveform or relabelled metadata.',
        nominal='Separate uniform synthetic parent-count weights.',
        alternatives=['w_low=exp(-log(2)*u)','w_high=exp(log(2)*u)'],
        u='2*(rank(actual_A_Q)+0.5)/Ntest-1 using stable parent-ID tie order; weights independent of teacher/student outputs.',
        bounds=[.5,2.], minimum_ESS_fraction=.5,
        response_kernel_frozen=True, refitting_or_retraining=False,
        compute='Reaggregate existing heldout conditional metrics/energy/coverage with both weightsets; additional CP/NN/AD/RNG/hardADC/G all0.',
        original_counts_and_physical_weights_retained=True,
        comparison_weights_separate_from_original_physical_weights=True,
        altered_population_gates='Same primary conditional variance curve/mean/variance/covariance/timing/coverage tolerances; do not use a marginal-distribution score against the old population as conditional failure.',
        generalization_scope='Known-kernel finite-support population reweighting only; not independent source acquisition or physical extrapolation.'),
    exact_planned_budgets=dict(stage1_gate=stage1, full_eight_trial_envelope=full,
        counts_are_maximum_planned_not_actual=True,
        unexecuted_after_gate_failure='Do not enter remaining6slots or count them completed.',
        full_gated_increment_after_stage1={k:({j:full[k][j]-stage1[k][j] for j in full[k]} if isinstance(full[k],dict) else full[k]-stage1[k]) for k in full},
        invariant_metadata=dict(trainable_parameters_per_optimizer=2, stage1=2, stage2=2,
            unpaired_score_samples_per_side=1024, paired_score_draws_per_parent_per_side=2),
        whole_job_nonadditive_allowances=dict(failure_checkpoint_saves_max=1,
            stage1_available=1, stage2_available_after_successful_stage1=1,
            rule='At most one failure checkpoint over the whole job. Stage1 success consumes none; stage2 still has one available. Do not difference1-1 into0 or reset the allowance per phase.')),
    resources=dict(
        serial_trials=True, CPU_threads=12, GPU_GiB=10, job_tree_RSS_GiB=8,
        wall_seconds_per_trial=7200, shared_cache_and_teacher_evaluation_wall_seconds=7200,
        full_job_wall_seconds=64800, stage1_wall_seconds=21600,
        output_MiB_per_trial=1536, shared_output_MiB=5120,
        full_output_MiB=17408, stage1_output_MiB=8192,
        selected_small_collection_MiB=256, required_remote_free_disk_GiB_before_start=40,
        distance_matrix='Stream64 rows against1024 rows, mask self sample-index diagonals explicitly; no P6593 xR64 xR64 x256 resident tensor.',
        rollout='Only owned N01 tree. Native supervisor caps RSS/VM as supported/GPU/output/wall, original exclusive lock, durable entered/completed, atomic CP and FAILED retained. First update plus same-snapshot current PID/boot/startticks/parent/cmd/cwd/resource health or full native0 terminal, then one future wake and rest.',
        preflight='Actual host idle GPU/CPU/RAM/disk checked only on later explicit native authorization, not queried for this proposal.',
        cached_tensors_release='Release decoded immediately after each fixed head chunk; mmap A/B/Q read-only, transfer needed parent chunk only, release eval rawbank after archive/moments. No arbitrary cache of REAL or old016 incomplete features.'),
    storage=dict(
        full_kernel_snapshots=11, hard_rows_npz_files=11*banks*H,
        baseline100_npz_files=11*banks, charge_moment_npz_files=11*banks,
        hard256_int16_payload_bytes=11*per_eval*H*256*2,
        baseline100_int16_payload_bytes=11*per_eval*100*2,
        exact_A_B_Q_cache_bytes=cache['persisted_A_B_Q_numerical_bytes'],
        report='Scalar features/finite9x9moments and waveform mean/diagonal variance only. No duplicate float64 waveform in REPORT, no dense256cov. Full hard code256 + original baseline100 permit savedADC roundtrip.',
        charge='Per-draw actual q/mu/v, parent/draw IDs and fixed A/B cache digest, no broadcast to substitute actual values. Actual A is one formal fixed cache value per parent; no repeatedA NN calls.',
        teacher_latent_storage='RNG state and exact requested shapes/digests; teacher latent not deployment input or same-latent acceptance evidence.',
        raw_files='Retain remote, select small JSON/CSV/logs/thin checkpoints only; later saved-array audit if registered once, never silently rerun scientific forward.'),
    implementation_paths=dict(
        model='src/n01/model017.py', driver='scripts/run_wp4_017.py', controller='scripts/wp4_017_controller.py',
        controls='receipts/wp4_017_20261015/prepare_controls.py; register_execution.py',
        contract='docs/WP4_017_CONFIGURATION.md; WP4_017_DATA_ROLES.json; wp4_017_config.json',
        engineering='receipts/wp4_017_20261015/engineering_01/check_wp4_017.py; check_controller017.py; ENGINEERING_READY.json',
        expected_remote_root='/home/rylan/projects/ej276d-n01/wp4_017_20261015',
        helper_policy='Prefer cached-basis methods and energy score in model017/driver; explicitly list every import in the source pack. No hidden local helper.',
        mature_reuse=['wp3_016_controller current-health/boot/startticks/durable validation','wp1_ops exclusive lock, operation journal, atomic output/CP, resource supervisor',
            'model016 stable inverse-softplus/charge law and exact-off semantics','model010/014 actual decoded and fixed basis arithmetic',
            'observation008 hard truth operator','015/016 independent saved-array morphology/finite-pair covariance kernels']),
    proposed_API=dict(
        driver_factories=['settings017()','budget_caps017(config)','validate_settings017(config, frozen=True)'],
        driver_execution=['build_frozen_cache017(runtime)','generate_teacher017(cache, independent_streams, teacher)','Engine017.step()','evaluate_kernel017()','stage1_gate017()','worker017()'],
        model=['CachedFrozenQ1Charge017','teacher_shape_coefficients017()','energy_score_U017()'],
        controller_CLI='--mode deploy/start/verify-start/status/collect/collect-failed/collect-inspect --freeze <exact-package> --out <fresh-directory>; controller state/health must name017 only.',
        registrar_contract='Pure stdlib settings/budget factories exported for root source/config binding. Every counter and single-pass cache load is derived, not manually copied from016.'),
    focused_engineering_plan=[
        'Artificial exact known two-point distributions enumerate independent teacher/student states: energy score minimized at correct dispersion; single-target MSE collapse contrast is labelled engineering only.',
        'Artificial cached A/decoded/B fixture:1288head frozen, exactly2theta optimizer membership/grad/delta; original A/005/M0 bytes unchanged; cached vs mature one-forward algebra under existing numerical gates.',
        'Teacher/student streams demonstrably distinct; same prefixes across paired/unpaired only for matching replicate; explicit shapes/counters and checkpoint state roundtrip; no ID/latent input.',
        'T2 centered mixture exact enumerated mean/variance/fourth moment; firstcomponent fixed; student remains Gaussian shape and2theta only; illegal mu/NaN/noise identity negatives.',
        'Parent partition/coverage and randomized tie-PIT on manufactured discrete data; variance/covariance/flags/undefined denominators and fixed population weight/ESS examples.',
        'Two tiny artificial trials check stage1 gate open/closed, no unentered stage2 cost,2000G caps factories, durable failures, firstproof current/terminal health and no automatic retry. Reuse mature checks; only changed loss/cache/gate deltas get new semantic tests.'],
    root_decisions_once=[
        dict(item='Freeze the fully specified numerical contract as one batch', recommended='Accept T1 a=.01qref,b=.0025 versus initial .001qref,.00025; T2 m=.9 firstshape coefficient; energy score/64draw recovery gates,5274/1319 split,3fit-Q strata,8trial envelope and17408MiB cap as written.',
             consequence='No result-dependent threshold, mechanism, amplitude, split, seed or model-family revision; failure ends the bounded question.'),
        dict(item='Bind exact SOURCE-only members/IDs/available original physical weights and existing CP digests', recommended='Use existing metadata bindings without decoding anything now; no REAL/DEV materialization. Historical A/005 known-base boundary stays explicit.',
             consequence='Root source/contract registration after016report and one independent contract review; only then local implementation or native authorizations as separately dispatched.')],
    next_authorized_action='Root reviews this proposal with the existing independent SolMax reviewer once. This document does not register or authorize a native run.',
    scientific_acceptance=False, hardware_calibration=False,
    changes_to_root_metadata_or_original_sources=False)

# One concentrated contract clarification, using only existing two teacher banks.
proposal['contract_revision'] = dict(
    id='focused_semantics_01', prior=ref('receipts/wp4_017_20261015/planning_01/before_contract_clarifications_01/ENGINEERING_PROPOSAL.json'),
    accepted_T1_T2_law_parameters_trial_slots_and_2theta_unchanged=True,
    frozen_before_scientific_data=True, added_scientific_CP_NN_AD_G_RNG_or_banks=0)

proposal['finite_support_contract'] = dict(
    parent_draw_population='Every original parent has32 original draw slots per bank,64 total per kernel/hypothesis. All slots and invalid/censor causes remain in raw output and denominators.',
    finite_definition='A feature is defined iff its computed scalar is finite under the unchanged observation/morphology rule. Q<=0/lowQ/guarded flags do not delete finite values. NaN/undefined values are not replaced by0.',
    feature_minimum=dict(student_per_bank=24, teacher_per_bank=24,
        student_combined=48, teacher_combined=48),
    covariance_pair_minimum=dict(student_joint_per_bank=24, teacher_joint_per_bank=24,
        student_joint_combined=48, teacher_joint_combined=48,
        all_unique_symmetric_pairs=45, full_matrix_entries=81),
    per_parent_timing_and_coverage='A feature parent is supported only if that same parent meets all student/teacher bank and combined minima. nS=0 is an explicit reason, not a zero PIT or zero-error result.',
    per_parent_covariance='Common parent support must satisfy the feature minima for all9 features and joint minima for every45pair in BOTH teacher and student. The same complete-parent set is used for all81 aggregated entries and the matched teacher-split resolution screen.',
    original_group='Heldout all or one of3 fixed fit-Q strata BEFORE any valid-value restriction. Keep original parent count and original comparison-weight sum.',
    group_minimum=dict(common_supported_parents=64, common_parent_fraction_of_original_group=.95,
        common_weight_mass_fraction_of_original_group=.95,
        ESS_fraction_of_original_group_count=.5),
    minimum_applies_to='Every required feature or complete covariance matrix, group and weightset; do not pool across strata, hypotheses or model seeds to manufacture support.',
    missingness='On ALL original group parents/draw slots, record finite/undefined/censor counts by cause separately for teacher/student, every feature and joint pair, each bank and combined. Weighted missing fraction = sum_i(w_i*invalid_slots_i)/(64*sum_i w_i), with the original group denominator; absolute teacher-student difference <=.01 for each required undefined/censor flag.',
    unsupported_output='Record parent IDs, feature/pair, nS/nT per bank and combined, missing causes, excluded weight mass, original/supported counts and ESS. Undefined per-parent metrics remain null with a reason. Retain their raw rows; no metric or parent silently disappears.',
    status='Support failure => pending/support_insufficient. Stage1 cannot open on pending. A supported but excessive discrepancy is failed only when its required resolution screen is also resolved.',
    waveform_exception='Full256 hard wave must be finite in every original slot. Nonfinite voltage is engineering FAILED, not a feature-support exclusion.')

proposal['parent_weight_contract'] = dict(
    members='Original heldout all group plus3 strata using fit-parent actualA-Q tertiles with fixed NumPy linear quantile rule; group membership frozen before targets. Parents remain in exactly one stratum and in allheldout.',
    weightsets=['nominal','low','high'],
    raw_weights='nominal w_i=1; low/high use the existing globally ranked heldout-Q exp(-/+log2*u_i) weights. Do not rerank within a stratum or renormalize original physical weights.',
    original_denominator='For each original group G, W_G=sum_{i in G}w_i and N_G=|G|. Supported set I is metric-specific only as allowed above.',
    normalized_weights='alpha_i=w_i/sum_{j in I}w_j for i in common support I; sum(alpha)=1. Record W_I/W_G and |I|/N_G.',
    mean='mu_w(x)=sum_i alpha_i*x_i',
    ESS='ESS=1/sum_i(alpha_i^2)=(sum_i w_i)^2/sum_i(w_i^2); require ESS>=.5*N_G and the other fixed support minima.',
    same_parent_difference_SE='For a parent-vector d_i, mu=sum alpha_i*d_i and SE_HC1=sqrt(N_I/(N_I-1)*sum_i alpha_i^2*(d_i-mu)^2). N_I>=64. This reduces exactly to sampleSD/sqrt(N) at equal weights.',
    scope='A prespecified descriptive parent-cluster dispersion estimate, pointwise per fixed cell on this finite synthetic cohort; not an exact64draw MC oracle, familywise confidence guarantee, model-retrain uncertainty or hardware interval.',
    non_scalar_metrics='Ratios/RMSE/Frobenius below use the same alpha and explicit fixed formulas; do not average per-parent ratios unless explicitly named.')

proposal['conditional_metric_contract'] = dict(
    primary_variance_curve_relative_RMS='sqrt(sum_i alpha_i*(vS_i-vT_i)^2 / sum_i alpha_i*vT_i^2). The denominator must be finite positive. This is relative RMS of the variance curve, not RMS of per-point relative errors.',
    primary_variance_curve_max_relative='max_i abs(vS_i-vT_i)/vT_i on ALL original group parents; vT>0 required. No unsupported-parent deletion is allowed for this analytic charge-law gate.',
    hard_wave_mean='sqrt(sum_i alpha_i*sum_t((meanS_i,t-meanT_i,t)^2) / sum_i alpha_i*sum_t(varT_i,t)); each mean/variance from the same parent64draws, variance ddof1; no between-parent scatter.',
    hard_pooled_conditional_trace='abs(sum_i alpha_i*tr(CS_i) / sum_i alpha_i*tr(CT_i) -1). CS_i/CT_i are within-parent64draw waveform covariance traces from diagonal variances, not covariance of pooled parents.',
    invalid_ratio_denominator='Zero/nonfinite teacher norm or variance denominator => pending/reference_undefined, never epsilon-adjusted or PASS.',
    scalar_time_errors='For each timing feature on common support: d_i=meanS64_i-meanT64_i. Absolute aggregate error=abs(sum alpha_i*d_i); per-parent mean RMSE=sqrt(sum alpha_i*d_i^2).')

proposal['conditional_covariance_contract'] = dict(
    sample_estimator='For each parent i and feature pair j,k, use only that parent jointly finite draws, pair-specific centered sample covariance with denominator n_joint-1. Keep n_joint and means for each pair; no covariance of all parents pooled together.',
    teacher_SD_scale='From finite teacher64 feature variance vT_i,j, d_i,j=max(sqrt(vT_i,j),fixed_floor_j). Floors are frozen before scientific inputs; record every active floor.',
    fixed_floors=dict(Q_Vns=256*delta,tail_Vns=256*delta,PSD_raw=1e-6,
        baseline_mean_V=delta,baseline_RMS_V=delta,peak_voltage_V=delta,
        rise10_90_software_ns=1.,FWHM_software_ns=1.,peak_index_software_ns=1.),
    normalized_parent='Ctilde_M_i[j,k]=C_M_i[j,k]/(d_i,j*d_i,k), for M=student64 or teacher64; both sides use the SAME actual teacher scales.',
    group_aggregate='Cbar_M=sum_{i in complete_common_support} alpha_i*Ctilde_M_i. No outer product of parent mean differences is added.',
    group_discrepancy='Frob(Cbar_S-Cbar_T)/Frob(Cbar_T), with teacher Frob norm finite positive. Required<=.15 after support and empirical resolution pass.',
    per_parent_error='Also store Frob(Ctilde_S_i-Ctilde_T_i)/Frob(Ctilde_T_i) for each defined parent plus weighted summaries; null/reason otherwise. The .15 gate applies to group aggregate, not to every noisy parent.',
    teacher_split='Using the same common parent set/weights and teacher64 scale d: normalize each within-parent teacher bank32 covariance C_T0_i,C_T1_i, aggregate separately to Cbar_T0,Cbar_T1.',
    teacher_only_empirical_resolution='resolution_cov=Frob(Cbar_T0-Cbar_T1)/Frob(.5*(Cbar_T0+Cbar_T1)). The denominator must be finite positive. Require resolution_cov<=.075, exactly half the unchanged .15 gate.',
    unresolved='If teacher split support/reference fails or resolution_cov>.075, this covariance cell is pending/MC_unresolved independent of the student discrepancy; do not count it as expression failure or widen. No new banks/NN/RNG.',
    interpretation='The direct32-bank difference is a conservative empirical screen. It is not a theoretical64draw oracle error or confidence guarantee. Pairwise missing-data covariance may be non-PSD; do not clip eigenvalues or call it automatically PSD.')

proposal['conditional_time_contract'] = dict(
    required=['peak_index','rise10_90','FWHM'],
    supported_parent='Same parent must meet the fixed actual finite counts in student and teacher, both32banks and combined64; all missing/censor causes retained on original64 slots.',
    student_comparison='Use combined finite student64 and teacher64 per-parent means with their actual defined-count denominators. abs(weighted mean d)<=1software_ns and sqrt(weighted mean d^2)<=2software_ns.',
    teacher_only_split='On the SAME support/weights define r_i=mean(Tbank0 finite draws)-mean(Tbank1 finite draws). Resolution aggregate=abs(sum alpha*r); resolution_RMSE=sqrt(sum alpha*r^2).',
    teacher_only_resolution_gate=dict(aggregate_abs_max_software_ns=.5,per_parent_RMSE_max_software_ns=1.),
    unresolved='If either teacher-only split resolution exceeds half of its unchanged1ns/2ns student gate, or finite support fails, pending/MC_unresolved; do not call student expression failure. Missing/censor fraction gate remains separate.',
    empirical_reference_only='Reuse existing teacher32+32. No extra timing mechanism, fitted floor, sample bank, random number or observation is introduced.')

proposal['conditional_coverage_contract'] = dict(
    features=proposal['evaluation']['coverage_features'],
    prediction='For each parent/feature use all finite values among that student64 draw vector; actual nS_finite may be48..64 when supported.',
    target='Only that parent defined teacher targets count in the coverage numerator and target denominator. Every undefined teacher original slot remains in its separate full64 missingness count.',
    randomized_PIT='For each defined teacher y, L=sum_{finite student}(S<y), E=sum_{finite student}(S==y); U is the independent predeclared uniform for this original target/feature/hypothesis. PIT=(L+U*(E+1))/(nS_finite+1).',
    missing='nS=0, nT=0, either bank<24finite or combined<48finite => unsupported parent/pending reason; never use65 as a substitute denominator, pad predictions or label undefined target uncovered.',
    per_parent='c_i,50=hits_[.25,.75]/nT_finite; c_i,90=hits_[.05,.95]/nT_finite. Save hit counts, nT_finite and nS_finite per original parent, bank valid counts and missing/censor reasons.',
    group='C_level=sum_i alpha_i*c_i,level on common supported parents. Do NOT pool hits across parents or implicitly weight a parent by its valid teacher draw count. Require abs(C50-.5)<=.04 and abs(C90-.9)<=.03, plus support and full-slot missingness-difference gates.',
    interpretation='Coverage conditional on a defined feature; undefined/censor probability is evaluated separately. ADC/discrete timing ties are handled by independent U, not deterministic inclusive intervals.',
    RNG_budget='Draw the already budgeted uniform for EVERY original teacher row/6feature/3hyp slot even if the feature is undefined; an unused slot does not add or remove RNG requests. Same uniforms may be reused for student initial/final comparisons, never for wave generation.')

proposal['conditional_energy_contract'] = dict(
    bank_formula='For parent i,h,bank b use m=n=32 independent student/teacher hard waves in fixed257metric: e_i,b=mean_{r,l}d(S_i,b,r,T_i,b,l)-sum_{r!=s}d(S_i,b,r,S_i,b,s)/(2*32*31). Self diagonal excluded; no teacher-student latent or noise sharing.',
    mean_two_banks='e_i^M=.5*(e_i,0^M+e_i,1^M), M=initial or final. No pooled64 energy distance substitution.',
    teacher_reference='r_i=.5*(ES_U(Tbank0 forecast,Tbank1 targets)+ES_U(Tbank1 forecast,Tbank0 targets)), each m=n=32 and self denominator32*31. Independent opposite teacher banks; cross distance matrix may be transposed/reused for the reverse direction.',
    improvement_difference='d_i=e_i^final-e_i^initial on the SAME original parent, hypothesis, score-bank and common random student initial/final library. Teacher library independent of every student.',
    upper95='U95=mu_w(d)+1.959963984540054*SE_HC1(d), fixed z before data. Require U95<0; signed means/SE/upper retained. No loss positivity gate or clamp.',
    excess_reference='z_i=e_i^final-r_i, using matched parent i. Compute mu_w(z) and SE_HC1(z) directly. Require mu_w(z)<=max(.02*abs(mu_w(r)),3*SE_HC1(z)). Do not add separate independent SEs of final/reference.',
    support='All32 waves per score bank must be finite. Parent/group support and ESS use original group membership; no scalar-feature missingness exclusion affects energy.',
    teacher_T1_T2_optional_descriptive_contrast='If reporting conditional T1-vs-T2 energy distance, compare T1bank0 withT2bank1 and T1bank1 withT2bank0 using unbiased cross/self terms. Corresponding same-bank Gaussian prefixes are coupled and may NOT be treated as independent cross samples. This contrast is descriptive only, no additional gate/forward/AD/RNG.',
    separate_cross_teacher_contrast_computed_in_this_plan=False,
    signed_values='Unbiased teacher-self-adjusted/energy-distance estimates may be negative; preserve them exactly.',
    inference_scope='Pointwise descriptive parent-cluster bound on fixed synthetic parents and random libraries, not a simultaneous95% guarantee across the gate matrix.')

proposal['gate_matrix'] = dict(
    required_stage1_trials=['T1_paired_seed0','T1_paired_seed1'],
    original_parent_groups=['all_heldout','fit_Q_tertile0_heldout','fit_Q_tertile1_heldout','fit_Q_tertile2_heldout'],
    comparison_weightsets=['nominal','low','high'], hypotheses=['fast','slow','common_level'],
    required_rows=[
        dict(metric='analytic_charge_variance_curve_RMS_and_max',groups='all4',weights='all3',hypotheses='not applicable, once per kernel',cells_stage1=24),
        dict(metric='hard_wave_mean',groups='all4',weights='all3',hypotheses='all3',cells_stage1=72),
        dict(metric='hard_pooled_conditional_variance_trace',groups='all4',weights='all3',hypotheses='all3',cells_stage1=72),
        dict(metric='conditional_cov9_Frobenius_and_teacher_MC_resolution',groups='all4',weights='all3',hypotheses='all3',cells_stage1=72),
        dict(metric='time_mean_RMSE_and_teacher_MC_resolution',features=3,groups='all4',weights='all3',hypotheses='all3',cells_stage1=216),
        dict(metric='finite_feature_PIT_50_and90',features=6,levels=2,groups='all4',weights='all3',hypotheses='all3',cells_stage1=864),
        dict(metric='conditional_energy_improvement_upper95_and_excess_reference',groups='all4',weights='all3',hypotheses='all3',cells_stage1=72),
        dict(metric='finite_support_missingness_censor_and_ESS',groups='all4',weights='all3',hypotheses='all3',applies_to='Every required feature/pair/matrix cell; counts on original group')],
    diagnostics_only=['Fit-parent results','output-Q conditions','secondary individual a,b errors/design conditioning','descriptive T1-vs-T2 contrast'],
    stage1_gate='PASS only if every required cell for BOTH seeds is resolved and passes. Any pending or failed cell closes stage2; save all reasons. A pending MC screen is not relabelled expression failure.',
    remaining_trials='Same matrix reported for each later trial, with no additional model selection, retry or family. T2 failure or observed indistinguishability is a result, not authorization for an extension.',
    multiple_cells='Shared parents/noise prefixes/weights are correlated; do not multiply the apparent sample size or call these simultaneous95% tests.')

proposal['distance_accounting'] = dict(
    paired='m=n=2: compute4 cross distances and1 unique student off-diagonal distance per parent/h/update. The student self ordered sum contains2 equal terms;5 actual computed distances enter budget.',
    unpaired='m=n=1024: actual implementation computes one1024x1024 cross matrix and one1024x1024 student self matrix in64-row blocks. The1024 diagonal distances of self are computed then masked BEFORE summation/backward; they are excluded from ES_U and denominator m(m-1).',
    unpaired_full_actual_computed=full['unpaired_training_energy_pair_distances'],
    unpaired_full_cross_retained=full['unpaired_training_cross_retained_pairs'],
    unpaired_full_self_ordered_retained=full['unpaired_training_self_ordered_retained_pairs'],
    unpaired_full_self_diagonal_masked=full['unpaired_training_self_diagonal_computed_then_masked'],
    evaluation='Each initial-vs-teacher or final-vs-teacher comparison computes2 banks x(cross+self)32x32 per parent/h; shared initial is compared with both teachers but generated once. Each teacher two-direction reference computes1 cross+2 self32x32 matrices per parent/h. Teacher-reference reverse cross uses transpose, not a new computation.',
    teacher_contrast='No separate T1-vs-T2 energy contrast is computed in this plan, so its distance count is0. The stated opposite-bank rule prevents any future descriptive interpretation of coupled same-bank teacher samples as independent cross samples; it is not an optional result-dependent operation or gate.')

proposal['storage'].update(
    remote_arrays='Complete per-parent native32 and derived64 waveform means/diagonal variances, cov9 matrices, every feature/pair valid count, timing/coverage denominators and support reasons remain as remote arrays with SHA/inventory. Parent error arrays remain remote; selected JSON is an aggregate view, not a replacement.',
    selected_small_package='<=256MiB including controls/logs: all aggregate gate cells, original/finite support denominators, missing/censor summaries, weights/ESS, split/index and raw-array SHA inventory plus necessary small2theta checkpoints. Do not try to select full per-parent waveform/cov arrays.',
    cap_failure='17GiB total and per-trial/shared caps are hard. If output or selected-package cap would be exceeded, preserve partial evidence and fail; do not silently delete a required metric, parent, gate or denominator to fit the cap.')

proposal['gates']['timing']['teacher_only_empirical_resolution'] = proposal['conditional_time_contract']['teacher_only_resolution_gate']
proposal['population_change']['gate_matrix'] = 'The two shifted weightsets and all4heldout groups are stage1 requirements exactly as gate_matrix, not optional favourable-result plots.'

# Small literal arithmetic examples clarify formulas only, not a model/NN test.
focused = {}
def check(name, condition):
    assert condition, name
    focused[name] = True

def weighted(values, weights):
    total = sum(weights)
    alpha = [w/total for w in weights]
    mean = sum(a*v for a,v in zip(alpha,values))
    N = len(values)
    se = math.sqrt(N/(N-1)*sum(a*a*(v-mean)**2 for a,v in zip(alpha,values)))
    return mean, se, 1/sum(a*a for a in alpha)

def pit_literal(student, target, uniform):
    finite = [x for x in student if x is not None and math.isfinite(x)]
    if not finite or target is None or not math.isfinite(target):
        return None
    lower = sum(x < target for x in finite)
    equal = sum(x == target for x in finite)
    return (lower + uniform*(equal+1))/(len(finite)+1)

def support_literal(s0,s1,t0,t1):
    return min(s0,s1,t0,t1)>=24 and s0+s1>=48 and t0+t1>=48

def group_support_literal(original,supported,mass_fraction,ESS):
    return supported>=64 and supported/original>=.95 and mass_fraction>=.95 and ESS>=.5*original

def matrix_norm(M):
    return math.sqrt(sum(x*x for row in M for x in row))

def matrix_error(S,T):
    denom = matrix_norm(T)
    if denom == 0:
        return None
    return matrix_norm([[s-t for s,t in zip(sr,tr)] for sr,tr in zip(S,T)])/denom

def covariance_group_literal(matrices,scales,weights):
    total=sum(weights)
    return [[sum(w*M[j][k]/(d[j]*d[k]) for M,d,w in zip(matrices,scales,weights))/total
        for k in range(len(scales[0]))] for j in range(len(scales[0]))]

def energy_literal(S,T):
    m,n=len(S),len(T)
    return sum(abs(s-t) for s in S for t in T)/(m*n)-sum(abs(S[i]-S[j]) for i in range(m) for j in range(m) if i!=j)/(2*m*(m-1))

check('PIT_actual_finite_n_plus1',pit_literal([1.,None,2.,2.],2.,.5)==.625)
check('PIT_zero_student_or_undefined_teacher_pending',pit_literal([None],2.,.5) is None and pit_literal([1.],None,.5) is None)
check('draw_support_24_24_48_positive',support_literal(24,24,24,24))
check('draw_support_one_bank23_rejected',not support_literal(23,25,24,24))
check('group_support_original_fraction_mass_ESS',group_support_literal(100,95,.95,80))
check('group_support_below95_count_mass_or64_rejected',not group_support_literal(100,94,.99,80) and not group_support_literal(100,95,.94,80) and not group_support_literal(66,63,.99,60))
mean,se,ess=weighted([1.,3.],[1.,3.])
check('weighted_mean_HC1_ESS_literal',mean==2.5 and se==.75 and ess==1.6)
check('same_parent_constant_delta_SE_zero',weighted([3.-2.,7.-6.],[1.,1.])[1]==0)
check('coverage_average_parent_not_hits',weighted([24/48,56/64],[1.,1.])[0]==.6875 and .6875 != (24+56)/(48+64))
check('variance_curve_RMS_fixed_ratio',math.isclose(math.sqrt((2-1)**2/((1**2)+(10**2))),math.sqrt(1/101),rel_tol=1e-15))
check('pooled_trace_ratio_not_parent_ratio_average',math.isclose(abs((2+10)/(1+10)-1),1/11,rel_tol=1e-14))
matrices=[[[1.,0.],[0.,4.]],[[4.,0.],[0.,1.]]]
scales=[[1.,2.],[2.,1.]]
cov=covariance_group_literal(matrices,scales,[1.,1.])
check('cov_parent_normalize_then_group_no_between_scatter',cov==[[1.,0.],[0.,1.]] and matrix_error(cov,cov)==0.)
check('cov_double_discrepancy_and_zero_reference_pending',matrix_error([[2.,0.],[0.,2.]],cov)==1. and matrix_error(cov,[[0.,0.],[0.,0.]]) is None)
check('cov_teacher_empirical_half_gate',.074<=proposal['gates']['hard_covariance']['teacher32_split_empirical_resolution_max']<.076)
check('time_teacher_MC_unresolved_cases',abs(weighted([.6,.6],[1.,1.])[0])>.5 and math.sqrt((1.1**2+(-1.1)**2)/2)>1.)
check('energy_U_self_diagonal_denominator',energy_literal([1.,2.],[-1.,3.])==1.5)
signed_ED=2*(sum(abs(s-t) for s in [-1.,1.] for t in [-1.,1.])/4)-2.-2.
check('negative_unbiased_energy_distance_retained',signed_ED==-2.)
check('bankwise32_energy_and_fixed_z',proposal['conditional_energy_contract']['bank_formula'].find('2*32*31')>=0 and proposal['conditional_energy_contract']['upper95'].find('1.959963984540054')>=0)
check('stage1_matrix_all_groups_weights_hypotheses',proposal['gate_matrix']['required_rows'][0]['cells_stage1']==2*4*3 and proposal['gate_matrix']['required_rows'][1]['cells_stage1']==2*4*3*3 and proposal['gate_matrix']['required_rows'][5]['cells_stage1']==2*4*3*3*6*2)
check('budget_2theta_is_invariant_not_zero_increment',proposal['exact_planned_budgets']['invariant_metadata']['stage2']==2 and 'trainable_parameters_per_optimizer' not in proposal['exact_planned_budgets']['full_gated_increment_after_stage1'])
check('failure_checkpoint_whole_job_stage2_available1',proposal['exact_planned_budgets']['whole_job_nonadditive_allowances']['failure_checkpoint_saves_max']==proposal['exact_planned_budgets']['whole_job_nonadditive_allowances']['stage2_available_after_successful_stage1']==1)
check('unpaired_computed_vs_retained_plus_masked',full['unpaired_training_energy_pair_distances']==full['unpaired_training_cross_retained_pairs']+full['unpaired_training_self_ordered_retained_pairs']+full['unpaired_training_self_diagonal_computed_then_masked'])
increment=proposal['exact_planned_budgets']['full_gated_increment_after_stage1']
check('all_additive_budget_stage_reconstruction',all(
    all(stage1[k][j]+increment[k][j]==full[k][j] for j in full[k]) if isinstance(full[k],dict)
    else stage1[k]+increment[k]==full[k] for k in full))
check('same_T1_T2_parameters_law_and_two_theta',proposal['student']['parameter_count']==2 and proposal['teachers']['T1']['a_known_Vns']==.01*qref and proposal['teachers']['T1']['b_known']==.0025 and proposal['teachers']['T2']['m']==.9 and proposal['training']['maximum_trials']==8)

# Pure arithmetic and scope validation: no numeric fixture, arrays or random draws.
assert Pfit+Ptest == P
assert stage1['G_updates'] == 4000 and full['G_updates'] == 16000
assert full['training_student_rows'] == full['training_teacher_rows'] == 16384000
assert full['evaluation_signal_rows'] == 4641472
assert full['noise005_rows'] == full['shape_cached_basis_assembly_rows'] == 37409472
assert full['hard_ADC_rows'] == 112228416
assert full['checkpoint_loads'] == 3 and full['fixed_M0_head_actual_kernel_rows'] == P
assert all(proposal['actual_scientific_work_in_this_preparation'][k] == 0 for k in proposal['actual_scientific_work_in_this_preparation'])
assert full['REAL_waveform_reads'] == full['original_DEV_waveform_reads'] == full['probe_AD_queries'] == 0
assert proposal['student']['parameter_count'] == 2 and proposal['training']['maximum_trials'] == teacher_budget['maximum']
assert len(proposal['teachers']) == 2 and full['evaluation_kernel_snapshots'] == 11
assert proposal['resources']['full_output_MiB'] == proposal['resources']['shared_output_MiB'] + 8*proposal['resources']['output_MiB_per_trial']
write_new('ENGINEERING_PROPOSAL.json', proposal)
receipt = dict(schema='n01/wp4-017-proposal-validation/v1', passed=True,
    recorded_UTC=datetime.now(timezone.utc).isoformat(), checks='Scope/count arithmetic assertions plus focused literal standard-library statistical definition examples; no real data/CP/native/NN/AD/RNG.',
    focused_arithmetic_examples=focused, focused_examples_passed=len(focused),
    proposal=ref('receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json'),
    builder=ref('receipts/wp4_017_20261015/planning_01/build_proposal.py'),
    actual_scientific_work=proposal['actual_scientific_work_in_this_preparation'])
write_new('PROPOSAL_VALIDATION.json', receipt)
print(json.dumps(dict(passed=True, proposal=receipt['proposal'], stage1=stage1, full=full, cache=cache), ensure_ascii=False))
