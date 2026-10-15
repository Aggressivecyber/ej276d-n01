"""Write the one final LOCAL ONLY WP4-017 source review from accepted evidence."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path('E:/GAN/research/N01')
HERE=Path(__file__).resolve().parent
ENG=ROOT/'receipts/wp4_017_20261015/engineering_01'


def record(path):
    assert path.suffix.lower() in ('.py','.json','.md','.txt')
    value=path.read_bytes()
    return dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())


def read(path):
    return json.loads(path.read_bytes())


def reference(path):
    return dict(path=path.relative_to(ROOT).as_posix(),**record(path))


def main():
    ready=read(ENG/'ENGINEERING_READY.json')
    binding=read(HERE/'SOURCE_BINDING_CHECKS.json')
    delta=read(HERE/'FINAL_SEMANTIC_DELTA_CHECKS.json')
    prep=read(HERE/'PREPARED_SEMANTIC_CHECKS.json')
    sources={k:dict(sha256=v['sha256'],bytes=v['bytes']) for k,v in ready['sources'].items()}
    assert len(sources)==5
    assert all(record(ROOT/k)==v for k,v in sources.items())
    assert ready['passed'] and binding['passed'] and delta['passed']
    assert len(prep['NumPy_artificial']['checks'])==25 and len(prep['torch_artificial']['checks'])==10
    assert all(c['passed'] for c in delta['checks']) and all(binding['checks'].values())
    assert ready['unresolved_engineering_blockers']==[]
    evidence={key:dict(path=row['relative_path'],sha256=row['sha256'],bytes=row['bytes'])
              for key,row in ready['engineering_receipts'].items()}
    for value in evidence.values():
        assert record(ROOT/value['path'])=={k:value[k] for k in ('bytes','sha256')}
    fixture=read(ENG/'artificial_driver_02/ENGINEERING_RECEIPT.json')
    full=read(ENG/'artificial_full1024_02/ENGINEERING_RECEIPT.json')
    controller=read(ENG/'artificial_controller_root_01/run_01/ENGINEERING_RECEIPT.json')
    result=dict(
        schema='n01/wp4-017-independent-source-review/v1',
        recorded_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        passed=True,blockers=[],reviewer_model='gpt-6.1-sol',reasoning_effort='max',
        review_round='single requested source review with concentrated local corrections',
        status='LOCAL_ONLY_SOURCE_AND_ARTIFICIAL_ENGINEERING_ACCEPTED',
        latest_user_boundary='做好就行 不用推送服务器 服务器马上要关闭了',
        sources=sources,
        accepted_proposal=reference(ROOT/'receipts/wp4_017_20261015/planning_01/ENGINEERING_PROPOSAL.json'),
        accepted_contract_review=reference(ROOT/'receipts/wp4_017_20261015/planning_01/INDEPENDENT_CONTRACT_REVIEW.json'),
        engineering_receipt=reference(ENG/'ENGINEERING_READY.json'),
        configuration_document=reference(ROOT/'docs/WP4_017_CONFIGURATION.md'),
        data_roles=reference(ROOT/'WP4_017_DATA_ROLES.json'),
        settings_draft=reference(ENG/'SETTINGS_DRAFT.json'),
        local_delivery=dict(source_freeze_permitted=True,raw_Git_blob_readback_and_D_backup_permitted=True,
            ready_for_native_dispatch=False,server_access_authorized=False,additional_review_layer_required=False,
            actual_scientific_execution='NOT_RUN',known_teacher_recovery='NOT_TESTED',
            source_or_main_metadata_modified_by_reviewer=False),
        conclusion_zh='本地源码及人工工程证据通过。固定已知A/005/M0条件下的两参数电荷方差恢复实现与已接受合同一致；第二阶段仅在两次T1 paired全部必需门resolved且通过后开放。当前没有真实教师恢复结果，不授予硬件或REAL响应的科学接受。',
        scope=dict(trainable_parameters=2,frozen_A_005_M0=True,mean_adapter_enabled=False,
            shape_head_1288_recovery_claim=False,mean_or_timing_kernel_learning_claim=False,
            REAL_or_DEV_numeric_input=False,physical_weight_preservation_separate_from_comparison_weights=True,
            new_macro_protocol_or_scientific_gate=False),
        budget_caps=ready['exact_full_additive_budget'],
        stage1_budget_caps=ready['exact_stage1_additive_budget'],
        budget_invariants=dict(trainable_parameters_per_optimizer=2,teacher_parameters_updated=0,
            original_checkpoint_loads_planned=3,whole_job_failure_checkpoint_maximum=1,
            failure_checkpoint_allowance_after_successful_stage1=1,
            stage2_caps_are_full_minus_stage1_for_additive_counts_only=True,
            two_theta_dimension_and_failure_allowance_are_not_subtracted=True,
            actual_scientific_G_updates_in_this_work=0,cumulative_preexisting_scientific_G_updates=179667),
        independent_checks=dict(
            prepared_core=dict(reference=reference(HERE/'PREPARED_SEMANTIC_CHECKS.json'),
                NumPy_cases=25,Torch_CPU_cases=10,reused_not_rerun=True,
                scope=['independent scalar HC1/ESS and finite PIT denominators',
                    'parent-first coverage and conditional covariance excluding between-parent scatter',
                    'teacher32 split MC_unresolved despite identical student/teacher',
                    'proper U energy, zero/duplicate distances and finite gradients',
                    'T2 exact fourth moment 1.6878 from deterministic quadrature',
                    'actual two-theta ES gradients versus finite differences and real AdamW update',
                    'A/B immutability, positive charge, exact disabled branch and shape not multiplied by q']),
            final_delta=dict(reference=reference(HERE/'FINAL_SEMANTIC_DELTA_CHECKS.json'),
                cases=29,passed=True,torch_imported=False,actual_stage_gate_executed=True,
                scope=['1392 primary +1296 mandatory missingness cells =2688 total over two stage1 trials',
                    'one pending/fail closes stage2; omitted mandatory cause/duplicate primary rejected',
                    'nonmissing ADC/search/Q-sign diagnostic differences do not create gates',
                    'fit AND test fixed [Q/qref,(Q/qref)^2] Gram from scalar summation oracle',
                    'invalid analytic parent cannot hide inside95percent empirical support',
                    '40GiB disk/10GiB RAM/GPU-empty exact preflight; missing/old8GiB/weakRAM/GPU-false rejected',
                    'existing B0 zero/nonzero and unestablished T2 observed mechanism metadata']),
            source_binding=dict(reference=reference(HERE/'SOURCE_BINDING_CHECKS.json'),
                cases=18,passed=True,original_controls_verified=124,bound_small_references=23,
                unchanged_heavy_AST=binding['unchanged_heavy_AST'],
                unchanged_statistical_AST=binding['unchanged_statistical_AST'],
                exact_reused_bytes=binding['unchanged_bytes'],
                metadata_only_AST=binding['metadata_only_AST'])),
        engineering_evidence=evidence,
        evidence_assessment=dict(
            two_tiny_trials=dict(passed=True,manufactured_only=True,checks=len(fixture['checks']),
                operations_entered=fixture['operations_entered'],operations_completed=fixture['operations_completed'],
                full_stage1_counter_equalities_verified=True,
                native_scientific_execution=False),
            full1024_unpaired=dict(passed=True,manufactured_only=True,three_hypotheses=True,
                actual_backward_queries=1,actual_optimizer_steps=1,theta_gradients=full['metrics']['preclip_gradients'],
                actual_theta_deltas=full['metrics']['actual_parameter_deltas'],
                elapsed_seconds=full['elapsed_seconds'],CPU_peak_RSS_bytes=full['memory']['peak_RSS_bytes'],
                unique_saved_tensor_storage_bytes=full['unique_saved_storage_bytes'],
                no_full1024x1024x257_difference_tensor=True,GPU_memory_measured=False,
                reuse_justification='Exact unchanged Engine.step/Runtime observation and energy AST; model bytes unchanged'),
            actual_original_classes=dict(manufactured_class_cache_and_comparator_checks=10,
                original_scientific_CP_loaded=False,manufactured_original_class_forward_rows=4,
                actual_A_B_Q_decoded_cache_interface_verified=True),
            serialization=dict(manufactured_reopen_checks=12,lossless_saved_Y_and_baseline_verified=True,
                own_small_CP_state_reopened=True,original_CP_reopened=False),
            controller=dict(positive_cases=len(controller['checks']),negative_cases=len(controller['negative_cases']),
                reused_exact_controller_bytes=True,same_snapshot_identity_resources_and_durable_chain=True,
                stage1_closed_native0_two_slot_branch=True,stage1_pass_full8_slot_branch=True,
                no_native_or_server_operations=True),
            focused_corrections=dict(artificial_cases=27,final_stdlib_binding_cases=23,
                full_generator_initial_and_end_states_verified=True,
                complete_initial_namespace_seed_role_and_parent_seen_zero=True,
                checkpoint_binds_initial_receipt=True,
                state_metadata_does_not_add_RNG_forward_AD_CP_budget=True),
            earlier_failed_test_attempts=dict(reference=reference(ENG/'TEST_ATTEMPTS.json'),preserved=True,
                resolved_fixture_issues=['Windows WMI import stall handled only within private artificial fixture process',
                    'strict cdist versus manual norm bit equality replaced by128epsilon engineering oracle after5.329070518200751e-15 discrepancy'],
                scientific_thresholds_changed=False,production_environment_or_library_patched=False)),
        resolved_findings=[
            dict(topic='secondary identifiability',resolution='Both fit and test use fixed dimensionless columns [Q/qref,(Q/qref)^2], Gram X.T@X/N, eigenvalues/ratio/condition retained; no separate column L2 normalization. Secondary25percent a/b criterion is undefined unless both ratios>=1e-4.'),
            dict(topic='initial provenance',resolution='All latent and parent generators save complete initial states, namespace/seed/role and seen0 before training draw; normal end and1000/2000 checkpoint states retained. Checkpoints bind immutable initial receipt. No additional draws, NN or checkpoint-count operation.'),
            dict(topic='analytic variance max',resolution='Analytic comparison covers ALL original group parents. Invalid model/reference or denominator gives pending/rejection, never deletion under empirical95percent support.'),
            dict(topic='mandatory missingness scope',resolution='Exactly9 scalar undefined fractions plus9 registered undefined/censor flags use1percent gate. Other flags retain64-slot fractions/differences as diagnostics, never added to required cells or used to delete finite values.'),
            dict(topic='preflight binding',resolution='Factory, draft, driver and controller validators bind free disk40GiB, available RAM10GiB and GPU empty; old014 default8GiB cannot silently replace017 disk contract. Underlying pinned native preflight already enforces RAM10GiB, CPU12 and empty GPU; no host check was performed.'),
            dict(topic='T2 interpretation',resolution='Only existing B0 cache finite/nonzero/max/RMS metadata added. Zero/invalid is untestable; nonzero remains inconclusive for observed mixture distinguishability. Eta fourth moment1.6878 retained; no independent T1-T2 observed energy comparison computed, no mechanism recovery claim or replacement, no new gate/probe/bank/distance/random draw.')],
        source_semantics=dict(
            charge_law='a=qref*softplus(theta_a), b=softplus(theta_b); s2=log1p(a/Q+b); q=Q*exp(sqrt(s2)*Z-.5*s2). Actual canonical A Q must be strictly positive, no floor.',
            assembly='M0 + A*expm1(log_multiplier); fixed B shape term never multiplied by q. Disabled charge returns exact M0 bytes and original zero prefix.',
            teacher_variance='T1 a=.01*qref,b=.0025; initial a=.001*qref,b=.00025 yields exactly one tenth conditional q variance for every positive Q.',
            T2='Only shape component0 changes to .9*independent_Rademacher+sqrt(.19)*independent_Normal; same mean and variance, fourth moment1.6878.',
            conditional_proper_score='Paired2+2 uses4 cross +1 unique student self distance. Unpaired1024+1024 uses independent role streams and removes only student sample-index diagonals, retaining duplicate parent IDs with independent latents. Full257-dimensional injective hard256 feature; signed score not clamped.',
            gradient='Actual STE hard-forward observation truth,256-row signal/ADC chunks,64-row cdist blocks, all3 hypotheses averaged into one backward. No loss detach; optimizer whitelist exactly two theta, AdamW .01/wd0/clip5.',
            parent_split='Deterministic SHA-isolated5274 fit/1319 heldout parents; IID with replacement. Every required fit stream must cover all5274 by2000G or incomplete, no repair draws. Original physical weights preserved separately from comparison reweights.',
            random_provenance='Teacher/student role streams independent for all shape/charge/noise005/extension/sign components. Matched mode/teacher Gaussian prefixes intentional; two evaluation banks independent; student initial/final share evaluation CRN. PIT uniforms reserved for all original teacher slots.',
            finite_support='At least24 finite values/joint pairs in EACH native32 bank and48 combined64. Common group>=64 parents,>=95percent original count and mass, ESS>=half original N. Insufficient is pending, zero reference pending, no zero imputation.',
            PIT='Actual n_student_finite+1 denominator; only defined teacher targets counted; parent finite-target rate first, then parent-weighted aggregation. All undefined/censor original64-slot fractions retained; zero-finite cannot become uniform PASS.',
            covariance='Each parent finite-pair within covariance first, normalized by actual teacher64 SD or fixed feature floor, then group-weighted aggregate and Frobenius ratio. No between-parent mean scatter. Per-parent errors separate; teacher32 split<=.075 required for resolution.',
            timing='Original1ns aggregate and2ns parent-RMSE gates; teacher32 split<=.5ns aggregate and1ns RMSE required for resolution, otherwise pending. This is empirical MC screening, not guaranteed64-draw oracle accuracy.',
            energy_gate='Native32 bankwise ES_U averaged over2 banks. Opposite teacher-bank reference uses both directions. Same-parent final-initial mean+1.959963984540054*HC1SE<0 and same-parent excess reference criterion; no arbitrary unpaired SE.',
            population='Nominal plus low/high fixed rank weights .5..2, original physical weights unchanged; full group/denominator/ESS retained. Pointwise HC1 is descriptive, not simultaneous familywise or physical model uncertainty.',
            staged_control='Two T1 paired seeds first. ALL2688 required cells must pass, with1392 primary and1296 mandatory missing/censor cells. Fail OR pending closes stage2 normally after2 slots, native0 possible. Only PASS opens6 additional slots; full success requires8 actual trials.',
            accounting='Exact stage1/full factories, durable entered/completed and no unknown inflight at terminal, original exclusive lock and partial failure preservation. First actual theta change plus same snapshot current identity/resource/durable chain or valid complete native0 endpoint.',
            saved_arrays='Actual q/mu/v, IDs/parent/draw/bank, lossless code256 baseline mean and optional IEEE XOR, actual baseline100, all flags, native32 and derived64 moments. Full stochastic preADC carrier/latent replay not saved or claimed.'),
        resource_contract=dict(CPU_threads=12,GPU_GiB=10,job_tree_RSS_GiB=8,
            per_trial_wall_seconds=7200,per_trial_output_MiB=1536,shared_wall_seconds=7200,shared_output_MiB=5120,
            full_job_wall_seconds=64800,full_job_output_MiB=17408,stage1_wall_seconds=21600,stage1_output_MiB=8192,
            selected_small_collection_MiB=256,preflight=dict(minimum_free_disk_GiB=40,
                minimum_available_RAM_GiB=10,GPU_must_be_empty=True),
            observed_artificial_CPU_peak_RSS_bytes=442200064,observed_artificial_saved_storage_bytes=103156848,
            GPU_peak_measured=False,server_available_capacity_checked=False,
            output_or_compression_estimate_is_guarantee=False,
            cap_exceeded_policy='stop and preserve partial; never delete required metrics, silently shrink scope, restart or create replacement candidate'),
        provenance_preservation=dict(original116_sources_plus2_training_controls_plus6_CPU_controls=124,
            all124_actual_SHA_and_bytes_verified=True,original_native_arrays_or_CP_read=False,
            previous_READY_and_all5_previous_sources_preserved=True,
            intermediate27_fixture_source_bytes_preserved_and_bound=True,
            source_review_artifacts_only_in_E=True,root_metadata_or_original_sources_changed=False),
        actual_scientific_operations=dict(original_CP_loads=0,original_SOURCE_REAL_DEV_numeric_reads=0,
            scientific_NN_forward_rows=0,scientific_AD=0,scientific_G=0,scientific_new_RNG_requests=0,GPU=0),
        actual_server_operations=dict(SSH=0,SCP=0,status=0,deploy=0,start=0,server_queries=0),
        scientific_acceptance=False,weights_promoted=False,known_teacher_recovery_established=False,
        hardware_or_REAL_response_match_established=False,
        scientific_status='response_match_not_established',
        remaining_boundary='Local engineering acceptance permits the requested local frozen source delivery. A future scientific run needs a later explicit user authorization; no current native job or recovery outcome exists.',
        reviewer_scripts={p.name:reference(p) for p in (HERE/'prepare_semantic_checks.py',HERE/'final_semantic_delta.py',
            HERE/'audit_source_binding.py',Path(__file__))})
    assert record(ENG/'ENGINEERING_READY.json')['sha256']=='eaa0bba9a7f6c4be9c721bf7dac36b3fe67b722a18e989527b58cd2d64934885'
    assert all(record(ROOT/k)==v for k,v in sources.items())
    path=HERE/'SOURCE_REVIEW.json'
    assert not path.exists(),'Preserve previous source review'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    reopened=read(path)
    assert reopened['passed'] and reopened['blockers']==[] and reopened['sources']==sources
    print(json.dumps(dict(path=str(path),passed=True,blockers=[],**record(path)),ensure_ascii=False))


if __name__=='__main__':
    main()
